/* test_jit_avx512_bytes.c — verify AVX-512 EVEX-prefixed encodings byte
   for byte against reference output from `as`/`objdump`. independent of
   execution: runs even on non-AVX-512 hardware (the bytes never execute). */

#include "../src/compute/backends/jit_x64.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

#if defined(__x86_64__) || defined(_M_X64)

static int compare_bytes(const char *name, const uint8_t *got, size_t got_len,
                          const uint8_t *want, size_t want_len) {
    if (got_len != want_len) {
        fprintf(stderr, "FAIL %s: got %zu bytes, want %zu\n", name, got_len, want_len);
        for (size_t i = 0; i < got_len; i++) fprintf(stderr, "%02x ", got[i]);
        fprintf(stderr, "\n");
        return 1;
    }
    for (size_t i = 0; i < got_len; i++) {
        if (got[i] != want[i]) {
            fprintf(stderr, "FAIL %s at byte %zu: got 0x%02x want 0x%02x\n",
                    name, i, got[i], want[i]);
            fprintf(stderr, "  got:  ");
            for (size_t j = 0; j < got_len; j++) fprintf(stderr, "%02x ", got[j]);
            fprintf(stderr, "\n  want: ");
            for (size_t j = 0; j < want_len; j++) fprintf(stderr, "%02x ", want[j]);
            fprintf(stderr, "\n");
            return 1;
        }
    }
    return 0;
}

enum { GPR_RAX=0, GPR_RCX=1, GPR_RDX=2, GPR_RBX=3,
       GPR_RSP=4, GPR_RBP=5, GPR_RSI=6, GPR_RDI=7 };

#define CASE(name, want_array, emit_call) do {                      \
    size_t before = b->pos;                                          \
    emit_call;                                                       \
    size_t got_len = b->pos - before;                                \
    if (compare_bytes(name, b->base + before, got_len,               \
                      want_array, sizeof(want_array))) {             \
        fails++;                                                     \
    } else {                                                         \
        printf("  %s: OK (%zu bytes)\n", name, got_len);             \
    }                                                                \
} while (0)

int main(void) {
    printf("test_jit_avx512_bytes\n");
    ax_jit_buf_t *b = ax_jit_buf_create(4096);
    if (!b) { fprintf(stderr, "buf create failed\n"); return 1; }
    int fails = 0;

    /* reference encodings from `as` + `objdump -d` (verified offline):
       these are independent of CPU support — we just compare bytes. */

    static const uint8_t e_fma_0_1_2[]    = {0x62,0xf2,0x75,0x48,0xb8,0xc2};
    static const uint8_t e_fma_15_14_13[] = {0x62,0x52,0x0d,0x48,0xb8,0xfd};
    static const uint8_t e_fma_16_17_18[] = {0x62,0xa2,0x75,0x40,0xb8,0xc2};
    static const uint8_t e_fma_31_30_29[] = {0x62,0x02,0x0d,0x40,0xb8,0xfd};
    static const uint8_t e_xor_0[]        = {0x62,0xf1,0x7c,0x48,0x57,0xc0};
    static const uint8_t e_xor_20[]       = {0x62,0xa1,0x5c,0x40,0x57,0xe4};
    static const uint8_t e_bcast_rdi[]    = {0x62,0xf2,0x7d,0x48,0x18,0x07};
    static const uint8_t e_bcast_rdi4[]   = {0x62,0xf2,0x7d,0x48,0x18,0x47,0x01};
    static const uint8_t e_movaps_rsi[]   = {0x62,0xf1,0x7c,0x48,0x28,0x06};
    static const uint8_t e_movaps_rsi64[] = {0x62,0xf1,0x7c,0x48,0x28,0x46,0x01};
    static const uint8_t e_movaps_rsi128[]= {0x62,0xf1,0x7c,0x48,0x28,0x46,0x02};
    static const uint8_t e_movups_rdx[]   = {0x62,0xf1,0x7c,0x48,0x11,0x02};
    static const uint8_t e_movups_rdx64[] = {0x62,0xf1,0x7c,0x48,0x11,0x42,0x01};
    static const uint8_t e_addps_0_1_2[]  = {0x62,0xf1,0x74,0x48,0x58,0xc2};

    CASE("vfmadd231ps zmm0, zmm1, zmm2",     e_fma_0_1_2,
         ax_jit_emit_vfmadd231ps_zmm(b, 0, 1, 2));
    CASE("vfmadd231ps zmm15, zmm14, zmm13",  e_fma_15_14_13,
         ax_jit_emit_vfmadd231ps_zmm(b, 15, 14, 13));
    CASE("vfmadd231ps zmm16, zmm17, zmm18",  e_fma_16_17_18,
         ax_jit_emit_vfmadd231ps_zmm(b, 16, 17, 18));
    CASE("vfmadd231ps zmm31, zmm30, zmm29",  e_fma_31_30_29,
         ax_jit_emit_vfmadd231ps_zmm(b, 31, 30, 29));
    CASE("vxorps zmm0, zmm0, zmm0",          e_xor_0,
         ax_jit_emit_vxorps_zero_zmm(b, 0));
    CASE("vxorps zmm20, zmm20, zmm20",       e_xor_20,
         ax_jit_emit_vxorps_zero_zmm(b, 20));
    CASE("vbroadcastss zmm0, [rdi]",         e_bcast_rdi,
         ax_jit_emit_vbroadcastss_zmm(b, 0, GPR_RDI, 0));
    CASE("vbroadcastss zmm0, [rdi+4]",       e_bcast_rdi4,
         ax_jit_emit_vbroadcastss_zmm(b, 0, GPR_RDI, 4));
    CASE("vmovaps zmm0, [rsi]",              e_movaps_rsi,
         ax_jit_emit_vmovaps_load_zmm(b, 0, GPR_RSI, 0));
    CASE("vmovaps zmm0, [rsi+64]",           e_movaps_rsi64,
         ax_jit_emit_vmovaps_load_zmm(b, 0, GPR_RSI, 64));
    CASE("vmovaps zmm0, [rsi+128]",          e_movaps_rsi128,
         ax_jit_emit_vmovaps_load_zmm(b, 0, GPR_RSI, 128));
    CASE("vmovups [rdx], zmm0",              e_movups_rdx,
         ax_jit_emit_vmovups_store_zmm(b, GPR_RDX, 0, 0));
    CASE("vmovups [rdx+64], zmm0",           e_movups_rdx64,
         ax_jit_emit_vmovups_store_zmm(b, GPR_RDX, 64, 0));
    CASE("vaddps zmm0, zmm1, zmm2",          e_addps_0_1_2,
         ax_jit_emit_vaddps_zmm(b, 0, 1, 2));

    ax_jit_buf_destroy(b);

    if (fails) {
        fprintf(stderr, "%d test(s) failed\n", fails);
        return 1;
    }
    printf("all 14 byte-level tests passed\n");
    return 0;
}

#else
int main(void) { printf("test_jit_avx512_bytes: SKIP (not x86_64)\n"); return 0; }
#endif
