/* jit_gemm_avx512.c — emits the AVX-512 14x32 GEMM micro-kernel.

   layout:
     zmm0..zmm27  : c accumulators (14 rows × 2 cols × 16 floats)
                    row r col 0 → zmm[2r], col 1 → zmm[2r+1]
     zmm28        : a broadcast (reused per row)
     zmm29, zmm30 : b0, b1
     zmm31        : scratch for vaddps writeback

   gpr usage:
     rdi : K loop counter
     rsi : pack_a (advances by MR=14 floats per K iter)
     rdx : pack_b (advances by NR=32 floats per K iter)
     rcx : C base (preserved)
     r8  : ldc_bytes
     rax : roving row pointer (computed during writeback)
     callee-saved: none used (kernel fits in caller-saved gprs).

   K-loop body is large (~50 instructions). uses jnz_rel32 (6 bytes).
   total emitted size ~600 bytes, fits in one page. */

#include "jit_gemm_avx512.h"
#include "jit_x64.h"

#include <pthread.h>

#define MR 14
#define NR 32

enum { GPR_RAX=0, GPR_RCX=1, GPR_RDX=2, GPR_RBX=3,
       GPR_RSP=4, GPR_RBP=5, GPR_RSI=6, GPR_RDI=7,
       GPR_R8=8, GPR_R9=9, GPR_R10=10, GPR_R11=11,
       GPR_R12=12, GPR_R13=13, GPR_R14=14, GPR_R15=15 };

static ax_jit_buf_t              *g_jit_buf512 = NULL;
static ax_jit_gemm_zmm_kernel_fn  g_kernel512  = NULL;
static pthread_once_t             g_jit_once_512 = PTHREAD_ONCE_INIT;

static void emit_14x32_kernel(ax_jit_buf_t *b) {
    /* ----- prologue: zero 28 zmm accumulators ----- */
    for (int i = 0; i < 28; i++) ax_jit_emit_vxorps_zero_zmm(b, i);

    /* ----- K loop ----- */
    ax_jit_label_t L_loop = ax_jit_here(b);

    /* load b0=zmm29, b1=zmm30 from [rdx], [rdx+64] */
    ax_jit_emit_vmovaps_load_zmm(b, /*dst*/29, /*base*/GPR_RDX,  0);
    ax_jit_emit_vmovaps_load_zmm(b, /*dst*/30, /*base*/GPR_RDX, 64);

    /* 14 rows: a = bcast [rsi+r*4]; c[r,0] += a*b0; c[r,1] += a*b1 */
    for (int r = 0; r < MR; r++) {
        ax_jit_emit_vbroadcastss_zmm(b, /*dst*/28, /*base*/GPR_RSI, r * 4);
        ax_jit_emit_vfmadd231ps_zmm (b, /*dst*/2*r,     /*a=*/28, /*b=*/29);
        ax_jit_emit_vfmadd231ps_zmm (b, /*dst*/2*r + 1, /*a=*/28, /*b=*/30);
    }

    /* increment ap and bp; decrement K; loop */
    ax_jit_emit_add_r64_imm32(b, GPR_RSI, MR * 4);   /* +56 */
    ax_jit_emit_add_r64_imm32(b, GPR_RDX, NR * 4);   /* +128 */
    ax_jit_emit_dec_r64(b, GPR_RDI);
    ax_jit_emit_jnz_to(b, L_loop);   /* will use rel32 due to body size */

    /* ----- writeback: rax = c base; for each row, accumulate into [rax]
       and [rax+64], then add ldc_bytes to rax ----- */
    ax_jit_emit_mov_r64_r64(b, GPR_RAX, GPR_RCX);

    for (int r = 0; r < MR; r++) {
        /* lo: load c[r, 0..15], add accumulator, store back */
        ax_jit_emit_vmovups_load_zmm (b, /*dst*/31, /*base*/GPR_RAX,  0);
        ax_jit_emit_vaddps_zmm       (b, /*dst*/2*r,   /*a=*/2*r,   /*b=*/31);
        ax_jit_emit_vmovups_store_zmm(b, /*base*/GPR_RAX,  0, /*src*/2*r);
        /* hi */
        ax_jit_emit_vmovups_load_zmm (b, /*dst*/31, /*base*/GPR_RAX, 64);
        ax_jit_emit_vaddps_zmm       (b, /*dst*/2*r+1, /*a=*/2*r+1, /*b=*/31);
        ax_jit_emit_vmovups_store_zmm(b, /*base*/GPR_RAX, 64, /*src*/2*r+1);
        /* advance row pointer (skip on the last row) */
        if (r < MR - 1) {
            ax_jit_emit_add_r64_r64(b, GPR_RAX, GPR_R8);
        }
    }

    ax_jit_emit_ret(b);
}

static void jit_init_once_512(void) {
    g_jit_buf512 = ax_jit_buf_create(8192);
    if (!g_jit_buf512) return;
    emit_14x32_kernel(g_jit_buf512);
    if (!ax_jit_buf_make_executable(g_jit_buf512)) {
        ax_jit_buf_destroy(g_jit_buf512);
        g_jit_buf512 = NULL;
        return;
    }
    g_kernel512 = (ax_jit_gemm_zmm_kernel_fn)ax_jit_buf_entry(g_jit_buf512);
}

ax_jit_gemm_zmm_kernel_fn ax_jit_gemm_avx512_get_14x32(void) {
    pthread_once(&g_jit_once_512, jit_init_once_512);
    return g_kernel512;
}

bool ax_jit_gemm_avx512_available(void) {
    return ax_jit_gemm_avx512_get_14x32() != NULL;
}
