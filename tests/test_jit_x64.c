/* test_jit_x64.c — correctness tests for the AVX2 jit encoder.

   each test emits a tiny function, makes the buffer executable, calls
   it, and verifies the result. validates the byte encoding of every
   instruction the GEMM micro-kernel needs. */

#include "../src/compute/backends/jit_x64.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#endif

#define ASSERT_EQ_F(a, b, eps) do { \
    if (fabsf((a) - (b)) > (eps)) { \
        fprintf(stderr, "FAIL %s:%d: %.6f != %.6f\n", __FILE__, __LINE__, (double)(a), (double)(b)); \
        return 1; \
    } \
} while (0)

#include <math.h>

/* GPR ids — these live in jit_x64.h's comment but redeclared for tests */
enum { GPR_RAX = 0, GPR_RCX = 1, GPR_RDX = 2, GPR_RBX = 3,
       GPR_RSP = 4, GPR_RBP = 5, GPR_RSI = 6, GPR_RDI = 7,
       GPR_R8  = 8, GPR_R9  = 9, GPR_R10 = 10, GPR_R11 = 11,
       GPR_R12 = 12, GPR_R13 = 13, GPR_R14 = 14, GPR_R15 = 15 };

/* test 1: emit a function that just returns. call it. */
static int test_ret(void) {
    ax_jit_buf_t *b = ax_jit_buf_create(64);
    if (!b) { fprintf(stderr, "buf create failed\n"); return 1; }
    ax_jit_emit_ret(b);
    if (!ax_jit_buf_make_executable(b)) { fprintf(stderr, "make_exec failed\n"); ax_jit_buf_destroy(b); return 1; }
    void (*fn)(void) = (void (*)(void))ax_jit_buf_entry(b);
    fn();
    ax_jit_buf_destroy(b);
    printf("  test_ret: OK\n");
    return 0;
}

#if defined(__x86_64__) || defined(_M_X64)

/* test 2: emit a function that does ONE vfmadd231ps and writes the
   result back. signature: void f(const float *a, const float *b, float *c)
   - a, b, c are aligned __m256 buffers (8 floats each)
   - on entry: rdi=a, rsi=b, rdx=c
   - we'll do c[0..7] = c[0..7] + a[0..7] * b[0..7]
     (i.e. c += a * b)
   logic:
     ymm0 = vmovaps [rdi]
     ymm1 = vmovaps [rsi]
     ymm2 = vmovaps [rdx]   ; load existing c
     vfmadd231ps ymm2, ymm0, ymm1
     vmovaps [rdx], ymm2
     ret */
static int test_one_fma(void) {
    ax_jit_buf_t *b = ax_jit_buf_create(256);
    if (!b) return 1;

    ax_jit_emit_vmovaps_load(b, /*dst*/0, /*base*/GPR_RDI, 0);
    ax_jit_emit_vmovaps_load(b, /*dst*/1, /*base*/GPR_RSI, 0);
    ax_jit_emit_vmovaps_load(b, /*dst*/2, /*base*/GPR_RDX, 0);
    ax_jit_emit_vfmadd231ps(b, /*dst*/2, /*a=*/0, /*b=*/1);
    ax_jit_emit_vmovaps_store(b, /*base*/GPR_RDX, 0, /*src*/2);
    ax_jit_emit_ret(b);

    if (!ax_jit_buf_make_executable(b)) { ax_jit_buf_destroy(b); return 1; }
    void (*fn)(const float*, const float*, float*) =
        (void (*)(const float*, const float*, float*))ax_jit_buf_entry(b);

    float A[8] __attribute__((aligned(32))) = {1, 2, 3, 4, 5, 6, 7, 8};
    float B[8] __attribute__((aligned(32))) = {2, 2, 2, 2, 2, 2, 2, 2};
    float C[8] __attribute__((aligned(32))) = {10, 10, 10, 10, 10, 10, 10, 10};
    fn(A, B, C);
    /* expect: c[i] = 10 + a[i]*2 = 10, 12, 14, 16, 18, 20, 22, 24, 26 */
    static const float want[8] = {12, 14, 16, 18, 20, 22, 24, 26};
    for (int i = 0; i < 8; i++) ASSERT_EQ_F(C[i], want[i], 1e-5f);
    ax_jit_buf_destroy(b);
    printf("  test_one_fma: OK\n");
    return 0;
}

/* test 3: vbroadcastss + fma — emulate one row-step of GEMM micro-kernel.
   c[i] += a_scalar * b[i] for 8 floats.
   sig: void f(const float *a_ptr, const float *b, float *c)
   - rdi=a_ptr (single float), rsi=b (8 floats), rdx=c (8 floats) */
static int test_broadcast_fma(void) {
    ax_jit_buf_t *b = ax_jit_buf_create(256);
    if (!b) return 1;
    ax_jit_emit_vbroadcastss(b, /*dst*/0, /*base*/GPR_RDI, 0);
    ax_jit_emit_vmovaps_load (b, /*dst*/1, /*base*/GPR_RSI, 0);
    ax_jit_emit_vmovaps_load (b, /*dst*/2, /*base*/GPR_RDX, 0);
    ax_jit_emit_vfmadd231ps  (b, /*dst*/2, /*a=*/0, /*b=*/1);
    ax_jit_emit_vmovaps_store(b, /*base*/GPR_RDX, 0, /*src*/2);
    ax_jit_emit_ret(b);
    if (!ax_jit_buf_make_executable(b)) { ax_jit_buf_destroy(b); return 1; }
    void (*fn)(const float*, const float*, float*) =
        (void (*)(const float*, const float*, float*))ax_jit_buf_entry(b);

    float a = 3.0f;
    float B[8] __attribute__((aligned(32))) = {1, 2, 3, 4, 5, 6, 7, 8};
    float C[8] __attribute__((aligned(32))) = {0, 0, 0, 0, 0, 0, 0, 0};
    fn(&a, B, C);
    static const float want[8] = {3, 6, 9, 12, 15, 18, 21, 24};
    for (int i = 0; i < 8; i++) ASSERT_EQ_F(C[i], want[i], 1e-5f);
    ax_jit_buf_destroy(b);
    printf("  test_broadcast_fma: OK\n");
    return 0;
}

/* test 4: zero a register via vxorps, store. verifies vxorps encoding.
   sig: void f(float *c) — sets c[0..7] = 0. */
static int test_xor_zero(void) {
    ax_jit_buf_t *b = ax_jit_buf_create(256);
    if (!b) return 1;
    ax_jit_emit_vxorps_zero  (b, /*dst*/5);  /* zero ymm5 */
    ax_jit_emit_vmovaps_store(b, /*base*/GPR_RDI, 0, /*src*/5);
    ax_jit_emit_ret(b);
    if (!ax_jit_buf_make_executable(b)) { ax_jit_buf_destroy(b); return 1; }
    void (*fn)(float*) = (void (*)(float*))ax_jit_buf_entry(b);

    float C[8] __attribute__((aligned(32))) = {1,1,1,1,1,1,1,1};
    fn(C);
    for (int i = 0; i < 8; i++) ASSERT_EQ_F(C[i], 0.0f, 1e-7f);
    ax_jit_buf_destroy(b);
    printf("  test_xor_zero: OK\n");
    return 0;
}

/* test 5: K-loop with dec + jnz. emulates a tight scalar dot-product
   kernel:  for (i=0; i<N; i++) sum += a[i] * b[i]
   N is passed in rcx. a in rdi, b in rsi, out (8-float) in rdx.
   we'll do: ymm2 = 0; loop: ymm0 = bcast a[0]; ymm1 = b[0..7];
     fma ymm2 += ymm0*ymm1; a += 4; b += 32; --rcx; jnz loop;
     store ymm2 to [rdx]; ret. */
static int test_kloop(void) {
    ax_jit_buf_t *b = ax_jit_buf_create(512);
    if (!b) return 1;

    /* prologue: zero accumulator */
    ax_jit_emit_vxorps_zero(b, /*dst*/2);

    /* loop start (label) */
    ax_jit_label_t L_loop = ax_jit_here(b);

    ax_jit_emit_vbroadcastss(b, /*dst*/0, /*base*/GPR_RDI, 0);
    ax_jit_emit_vmovaps_load (b, /*dst*/1, /*base*/GPR_RSI, 0);
    ax_jit_emit_vfmadd231ps  (b, /*dst*/2, /*a=*/0, /*b=*/1);
    /* a += 4 (one float = 4 bytes) */
    ax_jit_emit_add_r64_imm32(b, GPR_RDI, 4);
    /* b += 32 (8 floats) */
    ax_jit_emit_add_r64_imm32(b, GPR_RSI, 32);
    /* --rcx; jnz loop */
    ax_jit_emit_dec_r64(b, GPR_RCX);
    ax_jit_emit_jnz_to(b, L_loop);

    /* store and return */
    ax_jit_emit_vmovaps_store(b, /*base*/GPR_RDX, 0, /*src*/2);
    ax_jit_emit_ret(b);

    if (!ax_jit_buf_make_executable(b)) { ax_jit_buf_destroy(b); return 1; }

    /* sig: void f(const float *a, const float *b, float *c, int64_t N)
       SysV: rdi=a, rsi=b, rdx=c, rcx=N */
    void (*fn)(const float*, const float*, float*, int64_t) =
        (void (*)(const float*, const float*, float*, int64_t))ax_jit_buf_entry(b);

    const int N = 8;
    float A[8]  __attribute__((aligned(32))) = {1, 2, 3, 4, 5, 6, 7, 8};
    float Bd[64] __attribute__((aligned(32)));
    /* fill: for k-th iteration we read 8 floats starting at Bd + k*8.
       set Bd[k*8 + j] = (j+1).  so each iter accumulates a[k] * (j+1). */
    for (int k = 0; k < N; k++) for (int j = 0; j < 8; j++) Bd[k*8+j] = (float)(j+1);

    float C[8]  __attribute__((aligned(32))) = {0,0,0,0,0,0,0,0};
    fn(A, Bd, C, N);
    /* C[j] = sum_k a[k] * (j+1) = (j+1) * (1+2+...+8) = (j+1) * 36 */
    for (int j = 0; j < 8; j++) ASSERT_EQ_F(C[j], (float)((j+1)*36), 1e-4f);

    ax_jit_buf_destroy(b);
    printf("  test_kloop: OK\n");
    return 0;
}

/* test 6: 2x2 micro-kernel (mr=2, nr=8) with K-loop. mimics our real
   GEMM micro-kernel structure but tiny enough to verify by hand.
     for k in [0, K):
       a0 = bcast a[k*2]; a1 = bcast a[k*2+1]
       b0 = b[k*8 .. k*8+7]
       c00 += a0 * b0
       c10 += a1 * b0
     store ymm0=c00 → c[0..7]
     store ymm1=c10 → c[ldc .. ldc+7]
   sig: void f(const float *a, const float *b, float *c, int64_t ldc, int64_t K)
   SysV: rdi=a, rsi=b, rdx=c, rcx=ldc (in elements, so multiply by 4 for byte stride),
         r8=K. */
static int test_micro_2x8(void) {
    ax_jit_buf_t *b = ax_jit_buf_create(1024);
    if (!b) return 1;

    /* Convert ldc (elements) to ldc_bytes by shifting: shl rcx, 2.
       To keep the encoder small, generate it as: rcx += rcx; rcx += rcx;
       i.e. add rcx, rcx twice (each is shift left by 1).
       Actually we don't have ADD r,r in the encoder. Use: sub rcx, -8 won't
       work. Quick fix: caller passes ldc in BYTES, we don't shift. */

    /* zero c00, c10 */
    ax_jit_emit_vxorps_zero(b, /*dst*/0);
    ax_jit_emit_vxorps_zero(b, /*dst*/1);

    ax_jit_label_t L_loop = ax_jit_here(b);
    /* a0 = bcast [rdi];  a1 = bcast [rdi+4] */
    ax_jit_emit_vbroadcastss(b, /*dst*/4, /*base*/GPR_RDI, 0);
    ax_jit_emit_vbroadcastss(b, /*dst*/5, /*base*/GPR_RDI, 4);
    /* b0 = vmovaps [rsi] */
    ax_jit_emit_vmovaps_load(b, /*dst*/6, /*base*/GPR_RSI, 0);
    /* c00 += a0 * b0  ; c10 += a1 * b0 */
    ax_jit_emit_vfmadd231ps(b, /*dst*/0, /*a=*/4, /*b=*/6);
    ax_jit_emit_vfmadd231ps(b, /*dst*/1, /*a=*/5, /*b=*/6);
    /* a += 8 (2 floats), b += 32 (8 floats) */
    ax_jit_emit_add_r64_imm32(b, GPR_RDI, 8);
    ax_jit_emit_add_r64_imm32(b, GPR_RSI, 32);
    /* --r8; jnz loop */
    ax_jit_emit_dec_r64(b, GPR_R8);
    ax_jit_emit_jnz_to(b, L_loop);

    /* store: c[0..7] += c00; c[ldc .. ldc+7] += c10
       (we'll just store, not accumulate, since C starts at 0 in test) */
    ax_jit_emit_vmovaps_store(b, /*base*/GPR_RDX, 0, /*src*/0);
    /* second row: rdx + rcx (rcx is byte offset). need movups via SIB.
       since our emit_vmov_mem doesn't support [base+index], we can:
       add rdx, rcx  ; vmovaps [rdx], ymm1  ; sub rdx, rcx
       (ldc is small; this is fine for the test) */
    ax_jit_emit_mov_r64_r64(b, GPR_R10, GPR_RDX);  /* save rdx */
    /* add rdx, rcx — encode manually */
    /* use add r64, imm with NULL? no. write add r64, r64.
       not in our encoder. fall back: emit two raw bytes for "add rdx, rcx":
       48 01 CA  — REX.W=1, 01 (ADD r/m64, r64), ModR/M=11 001 010 = CA */
    ax_jit_emit_u8(b, 0x48);
    ax_jit_emit_u8(b, 0x01);
    ax_jit_emit_u8(b, 0xCA);
    ax_jit_emit_vmovaps_store(b, /*base*/GPR_RDX, 0, /*src*/1);
    ax_jit_emit_mov_r64_r64(b, GPR_RDX, GPR_R10);  /* restore rdx */

    ax_jit_emit_ret(b);

    if (!ax_jit_buf_make_executable(b)) { ax_jit_buf_destroy(b); return 1; }
    void (*fn)(const float*, const float*, float*, int64_t, int64_t) =
        (void (*)(const float*, const float*, float*, int64_t, int64_t))ax_jit_buf_entry(b);

    /* test data: K=4 */
    const int K = 4;
    float A[8]  __attribute__((aligned(32))) = {1,2, 3,4, 5,6, 7,8};
    float Bd[32] __attribute__((aligned(32)));
    for (int k = 0; k < K; k++) for (int j = 0; j < 8; j++) Bd[k*8+j] = (float)(k+1)*(float)(j+1);

    float C[16] __attribute__((aligned(32))) = {0};
    int64_t ldc_bytes = 8 * sizeof(float);
    fn(A, Bd, C, ldc_bytes, K);

    /* expected:
       row0: sum_k A[k*2+0] * (k+1)*(j+1)
              = (1*1 + 3*2 + 5*3 + 7*4) * (j+1) = (1+6+15+28)*(j+1) = 50*(j+1)
       row1: A[k*2+1] = 2,4,6,8 → (2*1 + 4*2 + 6*3 + 8*4) = 2+8+18+32 = 60
              row1[j] = 60*(j+1) */
    for (int j = 0; j < 8; j++) ASSERT_EQ_F(C[j],     50.0f * (j+1), 1e-4f);
    for (int j = 0; j < 8; j++) ASSERT_EQ_F(C[8 + j], 60.0f * (j+1), 1e-4f);
    ax_jit_buf_destroy(b);
    printf("  test_micro_2x8: OK\n");
    return 0;
}

#endif /* x86_64 */

int main(void) {
#if !defined(__x86_64__) && !defined(_M_X64)
    printf("test_jit_x64: SKIP (not x86_64)\n");
    return 0;
#else
    printf("test_jit_x64\n");
    if (test_ret())            return 1;
    if (test_one_fma())        return 1;
    if (test_broadcast_fma())  return 1;
    if (test_xor_zero())       return 1;
    if (test_kloop())          return 1;
    if (test_micro_2x8())      return 1;
    printf("all tests passed\n");
    return 0;
#endif
}
