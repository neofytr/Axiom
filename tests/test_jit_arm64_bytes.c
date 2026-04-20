/* test_jit_arm64_bytes.c — byte-level encoding tests for the ARM A64
   JIT encoder. validates each emitter against reference encodings
   derived from the ARM Architecture Reference Manual. independent of
   execution: runs on x86 too (the bytes never run). */

#include "../src/compute/backends/jit_arm64.h"
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

static int compare_insn(const char *name, uint32_t got, uint32_t want) {
    if (got == want) return 0;
    fprintf(stderr, "FAIL %s: got 0x%08x want 0x%08x\n", name, got, want);
    return 1;
}

#define CASE_INSN(name, want, emit) do {                                  \
    size_t before = b->pos;                                                \
    emit;                                                                  \
    uint32_t got = (uint32_t)b->base[before]                               \
                 | ((uint32_t)b->base[before+1] << 8)                      \
                 | ((uint32_t)b->base[before+2] << 16)                     \
                 | ((uint32_t)b->base[before+3] << 24);                    \
    if (compare_insn(name, got, want)) fails++;                            \
    else printf("  %s: OK (0x%08x)\n", name, got);                         \
} while (0)

int main(void) {
    printf("test_jit_arm64_bytes\n");
    ax_jit_arm64_buf_t *b = ax_jit_arm64_buf_create(4096);
    if (!b) { fprintf(stderr, "buf create failed\n"); return 1; }
    int fails = 0;

    /* references derived from ARM ARM:
       FMLA Vd.4S, Vn.4S, Vm.S[lane]
         encoding 0Q001111 sz L M Rm 0001 H 0 Rn Rd
         for Q=1, sz=0, lane=0 (L=0,H=0), Vd=V0, Vn=V0, Vm=V0:
           bits: 0 1 0 0 1111 1 0 0 0 0000 0001 0 0 00000 00000
                 = 0100 1111 1000 0000 0001 0000 0000 0000  = 0x4F801000 */
    CASE_INSN("fmla v0.4s, v0.4s, v0.s[0]",   0x4F801000u,
              ax_jit_arm64_emit_fmla_lane(b, 0, 0, 0, 0));
    /* lane=1: H=0 L=1 → bit21 set → +0x00200000 */
    CASE_INSN("fmla v0.4s, v0.4s, v0.s[1]",   0x4FA01000u,
              ax_jit_arm64_emit_fmla_lane(b, 0, 0, 0, 1));
    /* lane=2: H=1 L=0 → bit11 set → +0x00000800 */
    CASE_INSN("fmla v0.4s, v0.4s, v0.s[2]",   0x4F801800u,
              ax_jit_arm64_emit_fmla_lane(b, 0, 0, 0, 2));
    /* lane=3: H=1 L=1 → +0x00200800 */
    CASE_INSN("fmla v0.4s, v0.4s, v0.s[3]",   0x4FA01800u,
              ax_jit_arm64_emit_fmla_lane(b, 0, 0, 0, 3));
    /* Vd=V1 → Rd field +1 */
    CASE_INSN("fmla v1.4s, v0.4s, v0.s[0]",   0x4F801001u,
              ax_jit_arm64_emit_fmla_lane(b, 1, 0, 0, 0));
    /* Vn=V2 → Rn field shifted by 5: +0x40 */
    CASE_INSN("fmla v0.4s, v2.4s, v0.s[0]",   0x4F801040u,
              ax_jit_arm64_emit_fmla_lane(b, 0, 2, 0, 0));
    /* Vm=V3 → Rm field shifted by 16: +0x30000 */
    CASE_INSN("fmla v0.4s, v0.4s, v3.s[0]",   0x4F831000u,
              ax_jit_arm64_emit_fmla_lane(b, 0, 0, 3, 0));
    /* mixed regs + lane=2 — verified via aarch64-linux-gnu-objdump */
    CASE_INSN("fmla v15.4s, v14.4s, v13.s[2]", 0x4F8D19CFu,
              ax_jit_arm64_emit_fmla_lane(b, 15, 14, 13, 2));

    /* MOVI Vd.4S, #0 — verified pattern 0x4F000400 + Rd */
    CASE_INSN("movi v0.4s, #0",   0x4F000400u,
              ax_jit_arm64_emit_movi_zero(b, 0));
    CASE_INSN("movi v5.4s, #0",   0x4F000405u,
              ax_jit_arm64_emit_movi_zero(b, 5));
    CASE_INSN("movi v31.4s, #0",  0x4F00041Fu,
              ax_jit_arm64_emit_movi_zero(b, 31));

    /* LD1 {Vt.4S}, [Xn], #16 — base 0x4CDF7800 + (Rn<<5) + Rt */
    CASE_INSN("ld1 {v0.4s}, [x0], #16",   0x4CDF7800u,
              ax_jit_arm64_emit_ld1_post(b, 0, 0, 16));
    CASE_INSN("ld1 {v3.4s}, [x5], #16",   0x4CDF78A3u,
              ax_jit_arm64_emit_ld1_post(b, 3, 5, 16));

    /* ST1 {Vt.4S}, [Xn], #16 — base 0x4C9F7800 */
    CASE_INSN("st1 {v0.4s}, [x0], #16",   0x4C9F7800u,
              ax_jit_arm64_emit_st1_post(b, 0, 0, 16));

    /* LDR Q-vd, [Xn, #imm] — base 0x3DC00000 + (imm12<<10) + (Rn<<5) + Rt
       imm = bytes/16 */
    CASE_INSN("ldr q0, [x0]",       0x3DC00000u,
              ax_jit_arm64_emit_ldr_q_imm(b, 0, 0, 0));
    CASE_INSN("ldr q0, [x0, #16]",  0x3DC00400u,
              ax_jit_arm64_emit_ldr_q_imm(b, 0, 0, 1));
    CASE_INSN("ldr q1, [x2, #32]",  0x3DC00841u,
              ax_jit_arm64_emit_ldr_q_imm(b, 1, 2, 2));

    /* STR Q — base 0x3D800000 */
    CASE_INSN("str q0, [x0]",       0x3D800000u,
              ax_jit_arm64_emit_str_q_imm(b, 0, 0, 0));
    CASE_INSN("str q5, [x4, #48]",  0x3D800C85u,
              ax_jit_arm64_emit_str_q_imm(b, 5, 4, 3));

    /* FADD Vd.4S, Vn.4S, Vm.4S — base 0x4E20D400 */
    CASE_INSN("fadd v0.4s, v0.4s, v0.4s",   0x4E20D400u,
              ax_jit_arm64_emit_fadd(b, 0, 0, 0));
    CASE_INSN("fadd v1.4s, v2.4s, v3.4s",   0x4E23D441u,
              ax_jit_arm64_emit_fadd(b, 1, 2, 3));

    /* ADD imm — base 0x91000000 + (imm12<<10) + (Rn<<5) + Rd */
    CASE_INSN("add x0, x0, #0",     0x91000000u,
              ax_jit_arm64_emit_add_imm(b, 0, 0, 0));
    CASE_INSN("add x0, x0, #16",    0x91004000u,
              ax_jit_arm64_emit_add_imm(b, 0, 0, 16));
    CASE_INSN("add x1, x2, #56",    0x9100E041u,
              ax_jit_arm64_emit_add_imm(b, 1, 2, 56));

    /* SUB imm — base 0xD1000000 */
    CASE_INSN("sub x0, x0, #4",     0xD1001000u,
              ax_jit_arm64_emit_sub_imm(b, 0, 0, 4));

    /* ADD reg — base 0x8B000000 + (Rm<<16) + (Rn<<5) + Rd */
    CASE_INSN("add x0, x0, x0",     0x8B000000u,
              ax_jit_arm64_emit_add_reg(b, 0, 0, 0));
    CASE_INSN("add x1, x2, x3",     0x8B030041u,
              ax_jit_arm64_emit_add_reg(b, 1, 2, 3));

    /* MOV reg — base 0xAA0003E0 + (Rm<<16) + Rd */
    CASE_INSN("mov x0, x1",         0xAA0103E0u,
              ax_jit_arm64_emit_mov_reg(b, 0, 1));

    /* SUBS imm — base 0xF1000000 */
    CASE_INSN("subs x0, x0, #1",    0xF1000400u,
              ax_jit_arm64_emit_subs_imm(b, 0, 0, 1));

    /* B.NE rel imm19 — base 0x54000001 (cond=NE), imm19 in bits 23..5 */
    CASE_INSN("b.ne #+4",   (uint32_t)(0x54000000u | (1 << 5) | 1),
              ax_jit_arm64_emit_bne_rel(b, 1));
    CASE_INSN("b.ne #-4",   (uint32_t)(0x54000000u | ((-1 & 0x7FFFF) << 5) | 1),
              ax_jit_arm64_emit_bne_rel(b, -1));

    /* RET — fixed 0xD65F03C0 */
    CASE_INSN("ret",   0xD65F03C0u,
              ax_jit_arm64_emit_ret(b));

    ax_jit_arm64_buf_destroy(b);

    if (fails) {
        fprintf(stderr, "%d test(s) failed\n", fails);
        return 1;
    }
    printf("all byte-level tests passed\n");
    return 0;
}
