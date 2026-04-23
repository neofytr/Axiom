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

/* ====================================================================
   strided-A 14x32 variant — see header. per-kc unrolled, lazy-init,
   cached. abi:
     rdi = kc       (unrolled, consumed at emit time — runtime arg ignored)
     rsi = ap       (advances by lda_bytes per K iter)
     rdx = lda_bytes (runtime — used to bump rsi between K iters)
     rcx = bp       (B base; per-K-iter offsets baked as immediates)
     r8  = c base
     r9  = ldc_bytes
   register layout matches the packed kernel (zmm0..zmm27 = 28 acc,
   zmm28 = a bcast, zmm29/30 = b loads, zmm31 = wb scratch).
   no callee-saved gprs touched — rax used as roving writeback ptr.
   ==================================================================== */

#define KC_MIN 1
#define KC_MAX 256

static ax_jit_gemm_zmm_stridedA_kernel_fn
g_kernel_kc_fns_stridedA_512[KC_MAX + 1] = {0};
static ax_jit_buf_t *g_jit_kc_buf_stridedA_512[KC_MAX + 1] = {0};
static pthread_mutex_t g_jit_kc_mu_stridedA_512 = PTHREAD_MUTEX_INITIALIZER;

static void emit_one_kernel_14x32_stridedA_kc(ax_jit_buf_t *b, int64_t kc) {
    /* zero accumulators zmm0..zmm27 */
    for (int i = 0; i < 28; i++) ax_jit_emit_vxorps_zero_zmm(b, i);

    /* fully unrolled K loop. b offset per K iter is k*NR*4 bytes (baked
       as an immediate disp32). a is read from rsi which gets bumped by
       rdx (lda_bytes) at the end of each iter. broadcasts use disp8
       (+0..+52) so they stay short. */
    for (int64_t k = 0; k < kc; k++) {
        int b_off = (int)(k * NR * 4);
        ax_jit_emit_vmovaps_load_zmm(b, /*dst*/29, /*base*/GPR_RCX, b_off);
        ax_jit_emit_vmovaps_load_zmm(b, /*dst*/30, /*base*/GPR_RCX, b_off + 64);
        for (int r = 0; r < MR; r++) {
            ax_jit_emit_vbroadcastss_zmm(b, /*dst*/28, /*base*/GPR_RSI, r * 4);
            ax_jit_emit_vfmadd231ps_zmm (b, /*dst*/2*r,     /*a=*/28, /*b=*/29);
            ax_jit_emit_vfmadd231ps_zmm (b, /*dst*/2*r + 1, /*a=*/28, /*b=*/30);
        }
        /* advance rsi by lda_bytes (rdx) to point at next A row */
        ax_jit_emit_add_r64_r64(b, GPR_RSI, GPR_RDX);
    }

    /* writeback: rax = c base, then for each row accumulate [rax]+[rax+64]
       with the two accumulator vectors and bump rax by ldc_bytes (r9). */
    ax_jit_emit_mov_r64_r64(b, GPR_RAX, GPR_R8);

    for (int r = 0; r < MR; r++) {
        ax_jit_emit_vmovups_load_zmm (b, /*dst*/31, /*base*/GPR_RAX,  0);
        ax_jit_emit_vaddps_zmm       (b, /*dst*/2*r,   /*a=*/2*r,   /*b=*/31);
        ax_jit_emit_vmovups_store_zmm(b, /*base*/GPR_RAX,  0, /*src*/2*r);
        ax_jit_emit_vmovups_load_zmm (b, /*dst*/31, /*base*/GPR_RAX, 64);
        ax_jit_emit_vaddps_zmm       (b, /*dst*/2*r+1, /*a=*/2*r+1, /*b=*/31);
        ax_jit_emit_vmovups_store_zmm(b, /*base*/GPR_RAX, 64, /*src*/2*r+1);
        if (r < MR - 1) {
            ax_jit_emit_add_r64_r64(b, GPR_RAX, GPR_R9);
        }
    }

    ax_jit_emit_ret(b);
}

ax_jit_gemm_zmm_stridedA_kernel_fn
ax_jit_gemm_avx512_get_14x32_stridedA_kc(int64_t kc) {
    if (kc < KC_MIN || kc > KC_MAX) return NULL;

    /* lock-free hot path: aligned pointer load is atomic on x86. mutex
       on slow path provides the release fence. */
    ax_jit_gemm_zmm_stridedA_kernel_fn cached =
        g_kernel_kc_fns_stridedA_512[kc];
    if (cached) return cached;

    pthread_mutex_lock(&g_jit_kc_mu_stridedA_512);
    cached = g_kernel_kc_fns_stridedA_512[kc];
    if (cached) {
        pthread_mutex_unlock(&g_jit_kc_mu_stridedA_512);
        return cached;
    }

    /* byte budget per K iter (worst case, b offsets in disp32 range):
         2 vmovaps load_zmm:   2 × 10 = 20
         14 vbroadcastss_zmm:  14 × 9 = 126
         28 vfmadd231ps_zmm:   28 × 7 = 196
         1 add r64,r64:                  3
         subtotal:                     345
         safety pad:                    35
         ─── ~380 B/iter
       fixed cost (zero 28 acc + writeback ~14 rows × 4 ops + ret):
         28 vxorps_zero:        ~120 B
         14 wb rows × ~32 B:    ~450 B
         + mov rax,r8 + ret:     ~10 B
         total fixed:           ~600 B
       safety pad:               412 B */
    size_t per_kc_need = (size_t)kc * 380 + 1024;
    ax_jit_buf_t *buf = ax_jit_buf_create(per_kc_need);
    if (!buf) {
        pthread_mutex_unlock(&g_jit_kc_mu_stridedA_512);
        return NULL;
    }

    size_t before_pos = buf->pos;
    emit_one_kernel_14x32_stridedA_kc(buf, kc);
    /* defensive: kernel must end with RET (0xC3); zero-pad fall-through
       would SEGV. */
    if (buf->pos == before_pos
        || buf->base[buf->pos - 1] != 0xC3
        || buf->pos > buf->capacity) {
        ax_jit_buf_destroy(buf);
        pthread_mutex_unlock(&g_jit_kc_mu_stridedA_512);
        return NULL;
    }
    if (!ax_jit_buf_make_executable(buf)) {
        ax_jit_buf_destroy(buf);
        pthread_mutex_unlock(&g_jit_kc_mu_stridedA_512);
        return NULL;
    }

    g_jit_kc_buf_stridedA_512[kc]  = buf;
    cached = (ax_jit_gemm_zmm_stridedA_kernel_fn)ax_jit_buf_entry(buf);
    /* publish — release fence via the mutex unlock below */
    g_kernel_kc_fns_stridedA_512[kc] = cached;
    pthread_mutex_unlock(&g_jit_kc_mu_stridedA_512);
    return cached;
}
