/* test_jit_gemm.c — verify the JIT-emitted 6x16 micro-kernel produces
   identical output to a reference C implementation across a range of K. */

#include "../src/compute/backends/jit_gemm_avx2.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#if defined(__x86_64__) || defined(_M_X64)

#define MR 6
#define NR 16

/* reference: for each K iter, accumulate 6 broadcasts × (b0, b1) into 12
   accumulators, then write back into C as accumulation. matches the JIT
   layout: pack_a[k*MR + i] is row i's contribution at K iter k.
   pack_b[k*NR + j] is the j-th column at K iter k. */
static void ref_kernel_6x16(int64_t kc, const float *ap, const float *bp,
                             float *c, int64_t ldc) {
    float acc[MR][NR] = {{0}};
    for (int64_t k = 0; k < kc; k++) {
        for (int i = 0; i < MR; i++) {
            float a = ap[k * MR + i];
            for (int j = 0; j < NR; j++) {
                acc[i][j] += a * bp[k * NR + j];
            }
        }
    }
    for (int i = 0; i < MR; i++)
        for (int j = 0; j < NR; j++)
            c[i * ldc + j] += acc[i][j];
}

static int test_one(int64_t kc, int64_t ldc) {
    if (!ax_jit_gemm_avx2_available()) {
        fprintf(stderr, "JIT not available, skipping\n");
        return 0;
    }
    ax_jit_gemm_kernel_fn fn = ax_jit_gemm_avx2_get_6x16();

    /* allocate aligned buffers — pack_a/pack_b are produced 64-byte aligned.
       posix aligned_alloc requires size to be a multiple of alignment, so
       round each request up to the next 64-byte boundary. matters for the
       small-kc cases (kc=1 → 24 bytes for ap, not a 64 multiple). */
    #define ROUND64(n) (((n) + 63u) & ~(size_t)63u)
    float *ap = (float *)aligned_alloc(64, ROUND64((size_t)kc * MR * sizeof(float)));
    float *bp = (float *)aligned_alloc(64, ROUND64((size_t)kc * NR * sizeof(float)));
    float *c1 = (float *)aligned_alloc(64, ROUND64((size_t)MR * (size_t)ldc * sizeof(float)));
    float *c2 = (float *)aligned_alloc(64, ROUND64((size_t)MR * (size_t)ldc * sizeof(float)));
    #undef ROUND64
    if (!ap || !bp || !c1 || !c2) { fprintf(stderr, "alloc failed\n"); return 1; }

    /* deterministic random data */
    uint32_t s = 12345u;
    for (int64_t i = 0; i < kc * MR; i++) {
        s ^= s << 13; s ^= s >> 17; s ^= s << 5;
        ap[i] = (float)(s & 0xFFFF) * (1.0f / 65536.0f) - 0.5f;
    }
    for (int64_t i = 0; i < kc * NR; i++) {
        s ^= s << 13; s ^= s >> 17; s ^= s << 5;
        bp[i] = (float)(s & 0xFFFF) * (1.0f / 65536.0f) - 0.5f;
    }
    /* pre-fill C with non-zero so we test accumulation, not overwrite */
    for (int64_t i = 0; i < MR * ldc; i++) {
        s ^= s << 13; s ^= s >> 17; s ^= s << 5;
        c1[i] = (float)(s & 0xFFFF) * (1.0f / 65536.0f);
        c2[i] = c1[i];
    }

    ref_kernel_6x16(kc, ap, bp, c1, ldc);
    fn(kc, ap, bp, c2, ldc * (int64_t)sizeof(float));

    int fails = 0;
    float max_err = 0;
    for (int i = 0; i < MR; i++) {
        for (int j = 0; j < NR; j++) {
            float a = c1[i * ldc + j], b = c2[i * ldc + j];
            float e = fabsf(a - b);
            if (e > max_err) max_err = e;
            float tol = 1e-3f * (1.0f + fabsf(a));
            if (e > tol) {
                if (fails < 5)
                    fprintf(stderr, "    [%d,%d] ref=%.6f jit=%.6f diff=%.6f tol=%.6f\n",
                            i, j, (double)a, (double)b, (double)e, (double)tol);
                fails++;
            }
        }
    }
    free(ap); free(bp); free(c1); free(c2);
    if (fails) {
        fprintf(stderr, "  test kc=%ld ldc=%ld: %d mismatches (max err %.6f)\n",
                (long)kc, (long)ldc, fails, (double)max_err);
        return 1;
    }
    printf("  test kc=%ld ldc=%ld: OK (max err %.2e)\n",
           (long)kc, (long)ldc, (double)max_err);
    return 0;
}

int main(void) {
    printf("test_jit_gemm (6x16 AVX2)\n");
    int rc = 0;
    /* sweep kc and ldc; ldc≥NR; non-aligned ldc tests vmovups path */
    int64_t kcs[]  = {1, 2, 8, 32, 128, 256, 512};
    int64_t ldcs[] = {16, 17, 32, 100, 128};
    for (size_t i = 0; i < sizeof(kcs)/sizeof(kcs[0]); i++)
        for (size_t j = 0; j < sizeof(ldcs)/sizeof(ldcs[0]); j++)
            rc |= test_one(kcs[i], ldcs[j]);
    if (rc) fprintf(stderr, "FAIL\n");
    else    printf("all tests passed\n");
    return rc;
}

#else
int main(void) { printf("test_jit_gemm: SKIP (not x86_64)\n"); return 0; }
#endif
