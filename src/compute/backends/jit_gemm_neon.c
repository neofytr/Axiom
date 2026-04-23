/* jit_gemm_neon.c — emits the NEON 8x12 GEMM micro-kernel.

   register allocation:
     v0 .. v23  : 24 c accumulators (8 rows × 3 cols)
                  row r col c → v[3r + c]
     v24, v25   : a_lo, a_hi (8 floats, two 4S regs)
     v26, v27, v28 : b0, b1, b2

   AArch64 ABI (AAPCS64):
     x0 = kc, x1 = ap, x2 = bp, x3 = c, x4 = ldc_bytes
     LR = x30 (return address; preserved by epilogue with RET)

   K-loop body uses post-indexed LD1 to advance pointers; K is decremented
   by SUBS, looped via B.NE (rel offset to label). */

#include "jit_gemm_neon.h"
#include "jit_arm64.h"

#include <pthread.h>

#define MR 8
#define NR 12

enum { X0=0, X1=1, X2=2, X3=3, X4=4, X5=5, X6=6, X7=7,
       X8=8, X9=9, X10=10, X11=11, X12=12, X13=13, X14=14, X15=15,
       X16=16, X17=17, X18=18, X19=19, X20=20, X21=21, X22=22, X23=23,
       X24=24, X25=25, X26=26, X27=27, X28=28, X29=29, X30=30 };

static ax_jit_arm64_buf_t          *g_jit_buf_neon = NULL;
static ax_jit_gemm_neon_kernel_fn   g_kernel_neon  = NULL;
static pthread_once_t               g_jit_once_neon = PTHREAD_ONCE_INIT;

/* AAPCS64 register-31 alias used as base for SP-relative loads/stores.
   the ld/str/sub/add immediate encodings interpret Rn=31 as SP (rather
   than XZR like the data-processing forms). */
#define X31_SP 31

static void emit_8x12_kernel(ax_jit_arm64_buf_t *b) {
    /* prologue: callee-save NEON regs v8..v15. AAPCS64 says the lower
       64 bits of these must be preserved across calls. our accumulators
       v0..v23 include v8..v15 — without this prologue the kernel
       silently corrupts caller state, which manifested as test_sdpa_causal
       and test_large_mlp_trains failing on arm64 (CI run 24803112568)
       while non-causal SDPA + scalar tests passed.
       we save the full 128 bits via STR Q for code simplicity (slightly
       more stack than strictly required). 128 bytes keeps SP 16-aligned. */
    ax_jit_arm64_emit_sub_imm(b, X31_SP, X31_SP, 128);
    for (int i = 0; i < 8; i++) {
        /* str q(8+i), [sp, #i*16] — imm12 field is scaled by 16, so pass i */
        ax_jit_arm64_emit_str_q_imm(b, 8 + i, X31_SP, i);
    }

    /* zero v0..v23 (24 accumulators) */
    for (int i = 0; i < 24; i++) ax_jit_arm64_emit_movi_zero(b, i);

    /* loop label */
    ax_jit_arm64_label_t L_loop = ax_jit_arm64_here(b);

    /* load b0..b2 (post-indexed, advance bp by 3*16 = 48 bytes per K iter) */
    ax_jit_arm64_emit_ld1_post(b, 26, X2, 16);
    ax_jit_arm64_emit_ld1_post(b, 27, X2, 16);
    ax_jit_arm64_emit_ld1_post(b, 28, X2, 16);

    /* load a_lo, a_hi (post-indexed, advance ap by 2*16 = 32 bytes per K iter) */
    ax_jit_arm64_emit_ld1_post(b, 24, X1, 16);
    ax_jit_arm64_emit_ld1_post(b, 25, X1, 16);

    /* 24 FMLAs: c[r,col] += b_col * a[lane], where lane comes from a_lo or a_hi.
       row r ∈ [0..3]: source = v24 (a_lo), lane = r
       row r ∈ [4..7]: source = v25 (a_hi), lane = r - 4 */
    for (int r = 0; r < MR; r++) {
        int a_reg = (r < 4) ? 24 : 25;
        int lane  = (r < 4) ? r  : (r - 4);
        for (int c = 0; c < 3; c++) {
            int b_reg = 26 + c;
            int acc   = 3*r + c;
            ax_jit_arm64_emit_fmla_lane(b, acc, b_reg, a_reg, lane);
        }
    }

    /* SUBS x0, x0, #1 ; B.NE loop */
    ax_jit_arm64_emit_subs_imm(b, X0, X0, 1);
    ax_jit_arm64_emit_bne_to(b, L_loop);

    /* writeback: load existing C row, FADD with accumulator, store back.
       row r address = c + r * ldc_bytes. use x5 as roving row pointer:
         mov x5, x3
         <write 3 vectors, [x5], #16 each>
         add x5, x5, ldc_bytes (subtract the 48 we advanced)
       simpler: compute fresh addr each row via ADD reg.
       actually we use plain LDR/STR with #imm post not allowed for FP →
       just use LD1 / ST1 with post-imm 16, then advance with x5 += ldc - 48. */
    ax_jit_arm64_emit_mov_reg(b, X5, X3);  /* x5 = c base */
    /* x6 = -48 + ldc_bytes is hard. simpler: do plain LDR/STR with #0 imm
       and bump x5 by ldc_bytes per row. but LD1 post-imm advances x5 by 48
       across 3 vectors. so after writing row r, x5 = c + r*48 + 48. we
       want x5 = c + (r+1)*ldc_bytes. trick: use LDR Q + STR Q with
       explicit immediate offset (no auto-advance) and bump x5 by ldc_bytes.

       LDR Q with imm: imm12 scaled by 16. for offsets 0, 16, 32 we use
       imm 0, 1, 2. */

    for (int r = 0; r < MR; r++) {
        for (int c = 0; c < 3; c++) {
            int acc = 3*r + c;
            /* load existing c[r, c*4 .. c*4+3] into v29 */
            ax_jit_arm64_emit_ldr_q_imm(b, 29, X5, c);  /* offset = c*16 */
            /* fadd v_acc, v_acc, v29 */
            ax_jit_arm64_emit_fadd(b, acc, acc, 29);
            /* store back */
            ax_jit_arm64_emit_str_q_imm(b, acc, X5, c);
        }
        /* advance row pointer: x5 += ldc_bytes  (skip on last row) */
        if (r < MR - 1) {
            ax_jit_arm64_emit_add_reg(b, X5, X5, X4);
        }
    }

    /* epilogue: restore v8..v15, deallocate stack, RET */
    for (int i = 0; i < 8; i++) {
        ax_jit_arm64_emit_ldr_q_imm(b, 8 + i, X31_SP, i);
    }
    ax_jit_arm64_emit_add_imm(b, X31_SP, X31_SP, 128);
    ax_jit_arm64_emit_ret(b);
}

static void jit_init_once_neon(void) {
    g_jit_buf_neon = ax_jit_arm64_buf_create(8192);
    if (!g_jit_buf_neon) return;
    emit_8x12_kernel(g_jit_buf_neon);
    if (!ax_jit_arm64_buf_make_executable(g_jit_buf_neon)) {
        ax_jit_arm64_buf_destroy(g_jit_buf_neon);
        g_jit_buf_neon = NULL;
        return;
    }
    g_kernel_neon = (ax_jit_gemm_neon_kernel_fn)ax_jit_arm64_buf_entry(g_jit_buf_neon);
}

ax_jit_gemm_neon_kernel_fn ax_jit_gemm_neon_get_8x12(void) {
    pthread_once(&g_jit_once_neon, jit_init_once_neon);
    return g_kernel_neon;
}

bool ax_jit_gemm_neon_available(void) {
    return ax_jit_gemm_neon_get_8x12() != NULL;
}
