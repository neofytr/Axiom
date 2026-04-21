/* jit_gemm_avx2.c — emits the AVX2 6x16 GEMM micro-kernel at runtime.

   matches the C intrinsics kernel in cpu_opt.c (the AVX2 6x16 case):
     - 12 ymm accumulators (rows 0..5 × cols 0/1)
     - per K iter: 2 b loads + 6 a broadcasts + 12 fmas
     - writeback accumulates into existing C (load + vaddps + store)

   register allocation:
     ymm0..ymm11  : c00,c01, c10,c11, c20,c21, c30,c31, c40,c41, c50,c51
     ymm12        : a broadcast (reused per row)
     ymm13, ymm14 : b0, b1
     ymm15        : free (used by writeback to load existing C)

   gpr usage during kernel:
     rdi : K loop counter
     rsi : pack_a pointer (advances by mr=6 floats per iter)
     rdx : pack_b pointer (advances by nr=16 floats per iter)
     rcx : C base (row 0)
     r8  : ldc_bytes
     r9..r14 : row base pointers (precomputed: c, c+ldc, c+2ldc, ..., c+5ldc)
     r15 : scratch (unused)

   we save/restore r12..r15 (callee-saved). */

#include "jit_gemm_avx2.h"
#include "jit_x64.h"

#include <pthread.h>
#include <string.h>

/* must mirror cpu_opt.c's AVX2 GEMM_MR / GEMM_NR */
#define MR 6
#define NR 16   /* in floats; 2 ymm registers (8 floats each) per row */

enum { GPR_RAX=0, GPR_RCX=1, GPR_RDX=2, GPR_RBX=3,
       GPR_RSP=4, GPR_RBP=5, GPR_RSI=6, GPR_RDI=7,
       GPR_R8=8, GPR_R9=9, GPR_R10=10, GPR_R11=11,
       GPR_R12=12, GPR_R13=13, GPR_R14=14, GPR_R15=15 };

static ax_jit_buf_t       *g_jit_buf    = NULL;
static ax_jit_gemm_kernel_fn g_kernel_fn = NULL;
static pthread_once_t      g_jit_once   = PTHREAD_ONCE_INIT;

static void emit_6x16_kernel(ax_jit_buf_t *b) {
    /* ----- prologue: save callee-saved gprs we'll clobber ----- */
    ax_jit_emit_push_r64(b, GPR_R12);
    ax_jit_emit_push_r64(b, GPR_R13);
    ax_jit_emit_push_r64(b, GPR_R14);
    ax_jit_emit_push_r64(b, GPR_R15);

    /* ----- compute row base pointers r9..r14 ----- */
    /* r9  = rcx (row 0) */
    ax_jit_emit_mov_r64_r64(b, GPR_R9, GPR_RCX);
    /* r10 = r9  + r8 (row 1 = base + ldc_bytes) */
    ax_jit_emit_mov_r64_r64(b, GPR_R10, GPR_R9);
    ax_jit_emit_add_r64_r64(b, GPR_R10, GPR_R8);
    /* r11 = r10 + r8 (row 2) */
    ax_jit_emit_mov_r64_r64(b, GPR_R11, GPR_R10);
    ax_jit_emit_add_r64_r64(b, GPR_R11, GPR_R8);
    /* r12 = r11 + r8 (row 3) */
    ax_jit_emit_mov_r64_r64(b, GPR_R12, GPR_R11);
    ax_jit_emit_add_r64_r64(b, GPR_R12, GPR_R8);
    /* r13 = r12 + r8 (row 4) */
    ax_jit_emit_mov_r64_r64(b, GPR_R13, GPR_R12);
    ax_jit_emit_add_r64_r64(b, GPR_R13, GPR_R8);
    /* r14 = r13 + r8 (row 5) */
    ax_jit_emit_mov_r64_r64(b, GPR_R14, GPR_R13);
    ax_jit_emit_add_r64_r64(b, GPR_R14, GPR_R8);

    /* ----- zero accumulators ymm0..ymm11 ----- */
    for (int i = 0; i < 12; i++) ax_jit_emit_vxorps_zero(b, i);

    /* ----- K loop ----- */
    ax_jit_label_t L_loop = ax_jit_here(b);

    /* note: prefetch was tried (~8 K iters ahead) and measurably hurt
       (-4 to -7%). the JIT kernel's compact body lets ooo + hardware
       prefetcher stream ahead naturally; explicit prefetches just
       compete for issue slots. */

    /* load b0=ymm13, b1=ymm14 from [rdx], [rdx+32] */
    ax_jit_emit_vmovaps_load(b, /*dst*/13, /*base*/GPR_RDX,  0);
    ax_jit_emit_vmovaps_load(b, /*dst*/14, /*base*/GPR_RDX, 32);

    /* row 0: a = bcast [rsi+0]; c00 += a*b0; c01 += a*b1 */
    ax_jit_emit_vbroadcastss(b, /*dst*/12, /*base*/GPR_RSI,  0);
    ax_jit_emit_vfmadd231ps (b, /*dst*/0,  /*a=*/12, /*b=*/13);
    ax_jit_emit_vfmadd231ps (b, /*dst*/1,  /*a=*/12, /*b=*/14);
    /* row 1 */
    ax_jit_emit_vbroadcastss(b, /*dst*/12, /*base*/GPR_RSI,  4);
    ax_jit_emit_vfmadd231ps (b, /*dst*/2,  /*a=*/12, /*b=*/13);
    ax_jit_emit_vfmadd231ps (b, /*dst*/3,  /*a=*/12, /*b=*/14);
    /* row 2 */
    ax_jit_emit_vbroadcastss(b, /*dst*/12, /*base*/GPR_RSI,  8);
    ax_jit_emit_vfmadd231ps (b, /*dst*/4,  /*a=*/12, /*b=*/13);
    ax_jit_emit_vfmadd231ps (b, /*dst*/5,  /*a=*/12, /*b=*/14);
    /* row 3 */
    ax_jit_emit_vbroadcastss(b, /*dst*/12, /*base*/GPR_RSI, 12);
    ax_jit_emit_vfmadd231ps (b, /*dst*/6,  /*a=*/12, /*b=*/13);
    ax_jit_emit_vfmadd231ps (b, /*dst*/7,  /*a=*/12, /*b=*/14);
    /* row 4 */
    ax_jit_emit_vbroadcastss(b, /*dst*/12, /*base*/GPR_RSI, 16);
    ax_jit_emit_vfmadd231ps (b, /*dst*/8,  /*a=*/12, /*b=*/13);
    ax_jit_emit_vfmadd231ps (b, /*dst*/9,  /*a=*/12, /*b=*/14);
    /* row 5 */
    ax_jit_emit_vbroadcastss(b, /*dst*/12, /*base*/GPR_RSI, 20);
    ax_jit_emit_vfmadd231ps (b, /*dst*/10, /*a=*/12, /*b=*/13);
    ax_jit_emit_vfmadd231ps (b, /*dst*/11, /*a=*/12, /*b=*/14);

    /* increment pointers: rsi += MR*4 = 24 ; rdx += NR*4 = 64 */
    ax_jit_emit_add_r64_imm32(b, GPR_RSI, MR * 4);
    ax_jit_emit_add_r64_imm32(b, GPR_RDX, NR * 4);
    /* --rdi; jnz loop */
    ax_jit_emit_dec_r64(b, GPR_RDI);
    ax_jit_emit_jnz_to(b, L_loop);

    /* ----- writeback: C[row, 0..15] += accumulators -----
       use vmovups (C may not be 32-byte aligned at tile boundaries) */
    #define WB_ROW(rowbase, lo, hi) do {                          \
        /* lo: load c[0..7], add accumulator, store back */       \
        ax_jit_emit_vmovups_load (b, 15, (rowbase),  0);          \
        ax_jit_emit_vaddps       (b, (lo), (lo), 15);             \
        ax_jit_emit_vmovups_store(b, (rowbase),  0, (lo));        \
        /* hi: load c[8..15], add, store */                       \
        ax_jit_emit_vmovups_load (b, 15, (rowbase), 32);          \
        ax_jit_emit_vaddps       (b, (hi), (hi), 15);             \
        ax_jit_emit_vmovups_store(b, (rowbase), 32, (hi));        \
    } while (0)

    WB_ROW(GPR_R9,   0,  1);
    WB_ROW(GPR_R10,  2,  3);
    WB_ROW(GPR_R11,  4,  5);
    WB_ROW(GPR_R12,  6,  7);
    WB_ROW(GPR_R13,  8,  9);
    WB_ROW(GPR_R14, 10, 11);
    #undef WB_ROW

    /* ----- epilogue ----- */
    ax_jit_emit_pop_r64(b, GPR_R15);
    ax_jit_emit_pop_r64(b, GPR_R14);
    ax_jit_emit_pop_r64(b, GPR_R13);
    ax_jit_emit_pop_r64(b, GPR_R12);
    ax_jit_emit_ret(b);
}

static void jit_init_once(void) {
    /* allocate one page (4 KB) — kernel is ~400 bytes, fits easily. */
    g_jit_buf = ax_jit_buf_create(4096);
    if (!g_jit_buf) return;
    emit_6x16_kernel(g_jit_buf);
    if (!ax_jit_buf_make_executable(g_jit_buf)) {
        ax_jit_buf_destroy(g_jit_buf);
        g_jit_buf = NULL;
        return;
    }
    g_kernel_fn = (ax_jit_gemm_kernel_fn)ax_jit_buf_entry(g_jit_buf);
}

ax_jit_gemm_kernel_fn ax_jit_gemm_avx2_get_6x16(void) {
    pthread_once(&g_jit_once, jit_init_once);
    return g_kernel_fn;
}

bool ax_jit_gemm_avx2_available(void) {
    return ax_jit_gemm_avx2_get_6x16() != NULL;
}

/* ===== Batch C: per-kc fully-unrolled kernel ============================
   each (kc) entry maps to a separately-emitted kernel with the K loop
   fully unrolled. emitter walks the K iters at compile time, emitting
   the load+broadcast+fma sequence inline. the K count argument is
   ignored at runtime (caller still passes kc but emitter trusted it
   matches). saves ~6 bytes per K iter (no add/dec/branch).

   range constraints: must fit in one mmap'd page. each K iter emits
   ~70 bytes (vmovaps×2, vbroadcastss×6 with disp8, vfmadd231ps×12).
   epilogue is ~150 bytes. for kc=256, 256*70 + 200 = ~18KB. budget
   one large buffer for all variants combined; per-kernel offset cached. */

#define KC_MIN 1
#define KC_MAX 256

/* per-kc cached function pointers. NULL = not emitted yet. once a slot
   transitions NULL → non-NULL it stays set; readers can race the lookup
   without a lock because pointer-sized writes on x86 are atomic.
   memory ordering: writers use a release fence (mutex unlock) before
   publishing; readers acquire via the read of the slot (x86 has acq
   semantics on aligned loads by default). */
static ax_jit_gemm_kernel_fn g_kernel_kc_fns[KC_MAX + 1] = {0};
/* one mmap'd page per kc. holds the emitted kernel; flipped RW→RX once
   then never touched. NULL until first emission for that kc. */
static ax_jit_buf_t         *g_jit_kc_buf[KC_MAX + 1]  = {0};
static pthread_mutex_t       g_jit_kc_mu = PTHREAD_MUTEX_INITIALIZER;

/* emit one specialized kernel for the given kc. assumes buffer is RW
   and we hold the lock. returns the emitted entry point, or NULL on
   buffer overflow. note: cannot flip page to RX if we want to add more
   kernels later, so we keep the page RW until first call AND we accept
   that we'll need a different buffer per emission round. simpler: emit
   ALL kernels at once on first call using a worker function below. */
static void emit_one_kernel_kc(ax_jit_buf_t *b, int64_t kc) {
    /* prologue: save callee-saved gprs we'll clobber */
    ax_jit_emit_push_r64(b, GPR_R12);
    ax_jit_emit_push_r64(b, GPR_R13);
    ax_jit_emit_push_r64(b, GPR_R14);
    ax_jit_emit_push_r64(b, GPR_R15);

    /* row base pointers r9..r14 (same as runtime-K kernel) */
    ax_jit_emit_mov_r64_r64(b, GPR_R9,  GPR_RCX);
    ax_jit_emit_mov_r64_r64(b, GPR_R10, GPR_R9 ); ax_jit_emit_add_r64_r64(b, GPR_R10, GPR_R8);
    ax_jit_emit_mov_r64_r64(b, GPR_R11, GPR_R10); ax_jit_emit_add_r64_r64(b, GPR_R11, GPR_R8);
    ax_jit_emit_mov_r64_r64(b, GPR_R12, GPR_R11); ax_jit_emit_add_r64_r64(b, GPR_R12, GPR_R8);
    ax_jit_emit_mov_r64_r64(b, GPR_R13, GPR_R12); ax_jit_emit_add_r64_r64(b, GPR_R13, GPR_R8);
    ax_jit_emit_mov_r64_r64(b, GPR_R14, GPR_R13); ax_jit_emit_add_r64_r64(b, GPR_R14, GPR_R8);

    /* zero accumulators ymm0..ymm11 */
    for (int i = 0; i < 12; i++) ax_jit_emit_vxorps_zero(b, i);

    /* fully unrolled K loop. addresses use immediate displacements off
       rsi/rdx — reach is ~±2GB so any kc fits. we walk pointers without
       bumping rsi/rdx; final addresses are baked in. for kc up to ~96
       the displacements stay in disp8 range (±127); larger kc switches
       to disp32 automatically via emit_modrm_disp. */
    for (int64_t k = 0; k < kc; k++) {
        int b_off = (int)(k * NR * 4);
        int a_off = (int)(k * MR * 4);
        ax_jit_emit_vmovaps_load(b, /*dst*/13, /*base*/GPR_RDX, b_off);
        ax_jit_emit_vmovaps_load(b, /*dst*/14, /*base*/GPR_RDX, b_off + 32);
        ax_jit_emit_vbroadcastss(b, /*dst*/12, /*base*/GPR_RSI, a_off + 0);
        ax_jit_emit_vfmadd231ps (b, /*dst*/0,  /*a=*/12, /*b=*/13);
        ax_jit_emit_vfmadd231ps (b, /*dst*/1,  /*a=*/12, /*b=*/14);
        ax_jit_emit_vbroadcastss(b, /*dst*/12, /*base*/GPR_RSI, a_off + 4);
        ax_jit_emit_vfmadd231ps (b, /*dst*/2,  /*a=*/12, /*b=*/13);
        ax_jit_emit_vfmadd231ps (b, /*dst*/3,  /*a=*/12, /*b=*/14);
        ax_jit_emit_vbroadcastss(b, /*dst*/12, /*base*/GPR_RSI, a_off + 8);
        ax_jit_emit_vfmadd231ps (b, /*dst*/4,  /*a=*/12, /*b=*/13);
        ax_jit_emit_vfmadd231ps (b, /*dst*/5,  /*a=*/12, /*b=*/14);
        ax_jit_emit_vbroadcastss(b, /*dst*/12, /*base*/GPR_RSI, a_off + 12);
        ax_jit_emit_vfmadd231ps (b, /*dst*/6,  /*a=*/12, /*b=*/13);
        ax_jit_emit_vfmadd231ps (b, /*dst*/7,  /*a=*/12, /*b=*/14);
        ax_jit_emit_vbroadcastss(b, /*dst*/12, /*base*/GPR_RSI, a_off + 16);
        ax_jit_emit_vfmadd231ps (b, /*dst*/8,  /*a=*/12, /*b=*/13);
        ax_jit_emit_vfmadd231ps (b, /*dst*/9,  /*a=*/12, /*b=*/14);
        ax_jit_emit_vbroadcastss(b, /*dst*/12, /*base*/GPR_RSI, a_off + 20);
        ax_jit_emit_vfmadd231ps (b, /*dst*/10, /*a=*/12, /*b=*/13);
        ax_jit_emit_vfmadd231ps (b, /*dst*/11, /*a=*/12, /*b=*/14);
    }

    /* writeback (identical to runtime-K) */
    #define WB_ROW(rowbase, lo, hi) do {                          \
        ax_jit_emit_vmovups_load (b, 15, (rowbase),  0);          \
        ax_jit_emit_vaddps       (b, (lo), (lo), 15);             \
        ax_jit_emit_vmovups_store(b, (rowbase),  0, (lo));        \
        ax_jit_emit_vmovups_load (b, 15, (rowbase), 32);          \
        ax_jit_emit_vaddps       (b, (hi), (hi), 15);             \
        ax_jit_emit_vmovups_store(b, (rowbase), 32, (hi));        \
    } while (0)
    WB_ROW(GPR_R9,   0,  1); WB_ROW(GPR_R10,  2,  3);
    WB_ROW(GPR_R11,  4,  5); WB_ROW(GPR_R12,  6,  7);
    WB_ROW(GPR_R13,  8,  9); WB_ROW(GPR_R14, 10, 11);
    #undef WB_ROW

    ax_jit_emit_pop_r64(b, GPR_R15);
    ax_jit_emit_pop_r64(b, GPR_R14);
    ax_jit_emit_pop_r64(b, GPR_R13);
    ax_jit_emit_pop_r64(b, GPR_R12);
    ax_jit_emit_ret(b);
}

ax_jit_gemm_kernel_fn ax_jit_gemm_avx2_get_6x16_kc(int64_t kc) {
    if (kc < KC_MIN || kc > KC_MAX) return NULL;

    /* lock-free hot path: aligned pointer load is atomic on x86; the
       mutex on the slow path provides the release fence so once a
       reader sees a non-NULL slot, the bytes it points to are visible. */
    ax_jit_gemm_kernel_fn cached = g_kernel_kc_fns[kc];
    if (cached) return cached;

    pthread_mutex_lock(&g_jit_kc_mu);
    /* re-check after acquiring the lock */
    cached = g_kernel_kc_fns[kc];
    if (cached) {
        pthread_mutex_unlock(&g_jit_kc_mu);
        return cached;
    }

    /* allocate a dedicated buffer for THIS kc. byte budget:
         per K iter (worst case, disp32 forms when k*MR*4 or k*NR*4 > 127):
           - 2 vmovaps load:        2 × 9 = 18  (mod=10, disp32)
           - 6 vbroadcastss:        6 × 9 = 54
           - 12 vfmadd231ps:       12 × 5 = 60
           subtotal:                       132
           safety pad:                      28
           ─── total ~160 B/iter, was estimating 100 B/iter (which was the
           disp8 case only) — that off-by-2x silently truncated kernels
           for kc≳21 once displacements switched to disp32, causing the
           kernel to fall off the end of the page → SEGV in the zero pad.
         prologue (push×4 + 6 mov + 5 add + 12 vxorps): ~120 B
         writeback (6 rows × 6 ops): ~280 B
         epilogue (pop×4 + ret): ~10 B
         total fixed: ~410 B */
    size_t per_kc_need = (size_t)kc * 160 + 512;
    ax_jit_buf_t *buf = ax_jit_buf_create(per_kc_need);
    if (!buf) {
        pthread_mutex_unlock(&g_jit_kc_mu);
        return NULL;
    }

    size_t before_pos = buf->pos;
    emit_one_kernel_kc(buf, kc);
    /* defensive overflow check: if our budget was wrong the buffer would
       silently truncate (ax_jit_emit_u8 no-ops past capacity). detect
       by requiring the kernel to end with a RET (0xC3). otherwise the
       final flow would fall through into zero-padded bytes → SEGV. */
    if (buf->pos == before_pos
        || buf->base[buf->pos - 1] != 0xC3
        || buf->pos > buf->capacity) {
        ax_jit_buf_destroy(buf);
        pthread_mutex_unlock(&g_jit_kc_mu);
        return NULL;
    }
    if (!ax_jit_buf_make_executable(buf)) {
        ax_jit_buf_destroy(buf);
        pthread_mutex_unlock(&g_jit_kc_mu);
        return NULL;
    }

    /* publish: write the buf pointer first, then the fn pointer.
       readers see fn != NULL only after the buffer flip is complete. */
    g_jit_kc_buf[kc] = buf;
    g_kernel_kc_fns[kc] = (ax_jit_gemm_kernel_fn)ax_jit_buf_entry(buf);

    pthread_mutex_unlock(&g_jit_kc_mu);
    return g_kernel_kc_fns[kc];
}
