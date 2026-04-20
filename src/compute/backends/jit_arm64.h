/* jit_arm64.h — minimal ARM A64 (AArch64) instruction encoder.

   A64 instructions are uniformly 32-bit and stored little-endian.
   we encode the few NEON FMLA-by-lane ops, plus the GPR ops needed
   for a K-loop GEMM micro-kernel. macOS/linux mmap PROT_EXEC + the
   __builtin___clear_cache barrier for self-modified instruction
   memory (required on aarch64 unlike x86).

   register naming used in the API:
     - GPR ids 0..30 (X0..X30); SP is special, not exposed.
     - V (NEON) reg ids 0..31.
     - all FP ops are 4×float32 ("4S" arrangement). */

#ifndef AX_JIT_ARM64_H
#define AX_JIT_ARM64_H

#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct ax_jit_arm64_buf {
    uint8_t *base;
    size_t   capacity;
    size_t   pos;
    bool     executable;
} ax_jit_arm64_buf_t;

ax_jit_arm64_buf_t *ax_jit_arm64_buf_create(size_t min_capacity);
void                ax_jit_arm64_buf_destroy(ax_jit_arm64_buf_t *b);
bool                ax_jit_arm64_buf_make_executable(ax_jit_arm64_buf_t *b);
void               *ax_jit_arm64_buf_entry(ax_jit_arm64_buf_t *b);

/* emit one A64 instruction (4 bytes, little-endian). bumps pos by 4. */
void ax_jit_arm64_emit32(ax_jit_arm64_buf_t *b, uint32_t insn);

/* ----- NEON / SIMD ops (4S = four 32-bit floats) ----- */

/* fmla vd.4s, vn.4s, vm.s[lane]   (lane: 0..3)
   vd accumulates += vn * broadcast(vm[lane]). */
void ax_jit_arm64_emit_fmla_lane(ax_jit_arm64_buf_t *b, int vd, int vn, int vm, int lane);

/* movi vd.4s, #0   — zero the register */
void ax_jit_arm64_emit_movi_zero(ax_jit_arm64_buf_t *b, int vd);

/* ld1 { vd.4s }, [xn]            — load 16 contiguous bytes into vd */
void ax_jit_arm64_emit_ld1_post(ax_jit_arm64_buf_t *b, int vd, int xn, int post_imm);
/* st1 { vd.4s }, [xn]            — store vd to mem at xn */
void ax_jit_arm64_emit_st1_post(ax_jit_arm64_buf_t *b, int vd, int xn, int post_imm);
/* ldr q-vd, [xn, #imm]           — load 16 bytes (signed12 imm12 scaled by 16) */
void ax_jit_arm64_emit_ldr_q_imm(ax_jit_arm64_buf_t *b, int vd, int xn, int imm12_scaled16);
/* str q-vd, [xn, #imm] */
void ax_jit_arm64_emit_str_q_imm(ax_jit_arm64_buf_t *b, int vd, int xn, int imm12_scaled16);

/* fadd vd.4s, vn.4s, vm.4s */
void ax_jit_arm64_emit_fadd(ax_jit_arm64_buf_t *b, int vd, int vn, int vm);

/* ----- GPR ops ----- */

/* add xd, xn, #imm12 (no shift) */
void ax_jit_arm64_emit_add_imm(ax_jit_arm64_buf_t *b, int xd, int xn, int imm12);
/* sub xd, xn, #imm12 */
void ax_jit_arm64_emit_sub_imm(ax_jit_arm64_buf_t *b, int xd, int xn, int imm12);
/* add xd, xn, xm */
void ax_jit_arm64_emit_add_reg(ax_jit_arm64_buf_t *b, int xd, int xn, int xm);
/* mov xd, xn (alias for ORR Xd, XZR, Xn) */
void ax_jit_arm64_emit_mov_reg(ax_jit_arm64_buf_t *b, int xd, int xn);
/* subs xd, xn, #imm — flags-setting subtract */
void ax_jit_arm64_emit_subs_imm(ax_jit_arm64_buf_t *b, int xd, int xn, int imm12);

/* B.cond branches (rel imm19 in instruction units). cond=0001 = NE */
void ax_jit_arm64_emit_bne_rel(ax_jit_arm64_buf_t *b, int rel_insn19);

/* labels: position in BYTES from buffer start */
typedef size_t ax_jit_arm64_label_t;
static inline ax_jit_arm64_label_t ax_jit_arm64_here(const ax_jit_arm64_buf_t *b) { return b->pos; }
/* emit B.NE that targets a previously captured label */
void ax_jit_arm64_emit_bne_to(ax_jit_arm64_buf_t *b, ax_jit_arm64_label_t target);

/* ret — branches to LR */
void ax_jit_arm64_emit_ret(ax_jit_arm64_buf_t *b);

#ifdef __cplusplus
}
#endif

#endif
