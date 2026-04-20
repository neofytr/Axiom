/* jit_arm64.c — ARM A64 instruction encoder.

   A64 encoding refs:
     - FMLA (vector, by element): C7.2.86  (FP & Adv SIMD, scalar/vector)
     - LD1/ST1 single struct (no offset): C4.3.7
     - LDR/STR (immediate, SIMD&FP, 128-bit Q): C4.3
     - ADD/SUB/SUBS immediate: C6.2
     - ORR (shifted register, "mov" alias): C6.2
     - B.cond: C6.2.34
     - RET: C6.2.219

   reference: ARM Architecture Reference Manual, ARMv8-A. */

#include "jit_arm64.h"

#include <stdlib.h>
#include <string.h>

#if defined(__linux__) || defined(__APPLE__)
  #include <sys/mman.h>
  #include <unistd.h>
  #define AX_JIT_HAS_MMAP 1
#else
  #define AX_JIT_HAS_MMAP 0
#endif

ax_jit_arm64_buf_t *ax_jit_arm64_buf_create(size_t min_capacity) {
#if !AX_JIT_HAS_MMAP
    (void)min_capacity;
    return NULL;
#else
    long page = sysconf(_SC_PAGESIZE);
    if (page <= 0) page = 4096;
    size_t cap = ((min_capacity + (size_t)page - 1) / (size_t)page) * (size_t)page;
    if (cap == 0) cap = (size_t)page;

    /* on apple silicon (arm64-darwin) we'd need MAP_JIT + pthread_jit_write_protect.
       linux + plain MAP_ANONYMOUS works directly for RW→RX flips. */
    void *p = mmap(NULL, cap, PROT_READ | PROT_WRITE,
                   MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (p == MAP_FAILED) return NULL;

    ax_jit_arm64_buf_t *b = (ax_jit_arm64_buf_t *)calloc(1, sizeof(*b));
    if (!b) { munmap(p, cap); return NULL; }
    b->base = (uint8_t *)p;
    b->capacity = cap;
    b->pos = 0;
    b->executable = false;
    return b;
#endif
}

void ax_jit_arm64_buf_destroy(ax_jit_arm64_buf_t *b) {
    if (!b) return;
#if AX_JIT_HAS_MMAP
    if (b->base) munmap(b->base, b->capacity);
#endif
    free(b);
}

bool ax_jit_arm64_buf_make_executable(ax_jit_arm64_buf_t *b) {
#if !AX_JIT_HAS_MMAP
    (void)b;
    return false;
#else
    if (!b || !b->base) return false;
    if (mprotect(b->base, b->capacity, PROT_READ | PROT_EXEC) != 0) return false;
    /* mandatory on aarch64: invalidate i-cache for the just-written region.
       gcc/clang builtin emits the right cache-maintenance ops. */
#if defined(__aarch64__) || defined(__arm__)
    __builtin___clear_cache((char *)b->base, (char *)(b->base + b->pos));
#endif
    b->executable = true;
    return true;
#endif
}

void *ax_jit_arm64_buf_entry(ax_jit_arm64_buf_t *b) {
    return (b && b->executable) ? (void *)b->base : NULL;
}

void ax_jit_arm64_emit32(ax_jit_arm64_buf_t *b, uint32_t insn) {
    if (!b || b->executable || b->pos + 4 > b->capacity) return;
    /* little-endian (arm64 default) */
    b->base[b->pos + 0] = (uint8_t)(insn       & 0xFF);
    b->base[b->pos + 1] = (uint8_t)((insn >>  8) & 0xFF);
    b->base[b->pos + 2] = (uint8_t)((insn >> 16) & 0xFF);
    b->base[b->pos + 3] = (uint8_t)((insn >> 24) & 0xFF);
    b->pos += 4;
}

/* ===== NEON ============================================================ */

/* FMLA Vd.4S, Vn.4S, Vm.S[lane]   — by-element vector form
   encoding (Q=1, sz=0 single-precision): 0Q00111100 LMm Rm 0001 H 0 Rn Rd
   for 4S: bits[31..0] =
     0 1 0 0 1 1 1 1 1 0 L M Rm[3:0] 0 0 0 1 H 0 Rn Rd
   where lane = (H << 1) | L, M is the high bit of Rm (Rm allowed 0..15
   for sz=0; the 5th bit is encoded in M, but for Vm in 0..15 M=0).
   for our purposes Vm is the broadcast source register and lane ∈ {0..3}. */
void ax_jit_arm64_emit_fmla_lane(ax_jit_arm64_buf_t *b, int vd, int vn, int vm, int lane) {
    /* Vm in 0..31; only low 4 bits go into Rm field, top bit into M (bit 20).
       sz=0 (single precision): the M bit encodes Rm[4]. */
    uint32_t Q = 1;
    uint32_t sz = 0;
    uint32_t L = (uint32_t)(lane & 1);
    uint32_t H = (uint32_t)((lane >> 1) & 1);
    uint32_t M = (uint32_t)((vm >> 4) & 1);
    uint32_t Rm = (uint32_t)(vm & 0xF);
    uint32_t Rn = (uint32_t)(vn & 0x1F);
    uint32_t Rd = (uint32_t)(vd & 0x1F);
    /* bit layout (31..0):
         31: 0
         30: Q
         29..28: 00
         27..24: 1111
         23: 1   (per FMLA-by-element encoding for sz=0)
         22: sz
         21: L
         20: M
         19..16: Rm
         15..12: 0001 (opcode=FMLA)
         11: H
         10: 0
         9..5: Rn
         4..0: Rd
    */
    uint32_t insn =
        (0u << 31)
      | (Q  << 30)
      | (0u << 28)            /* bits 29..28 already 0 */
      | (0xFu << 24)          /* bits 27..24 = 1111 */
      | (1u  << 23)
      | (sz  << 22)
      | (L   << 21)
      | (M   << 20)
      | (Rm  << 16)
      | (1u  << 12)
      | (H   << 11)
      | (Rn  <<  5)
      | (Rd  <<  0);
    ax_jit_arm64_emit32(b, insn);
}

/* MOVI Vd.4S, #0  — encoded as MOVI vector, immediate (cmode=0000, op=0)
   bits: 0 Q 0 0 1111 00000 abc cmode 01 defgh Rd
   for #0 with 4S: Q=1, op=0, abc=000, cmode=0000, defgh=00000.
   simplest encoding: 0x4F000400 + Rd  (this is "movi v0.4s, #0" base). */
void ax_jit_arm64_emit_movi_zero(ax_jit_arm64_buf_t *b, int vd) {
    uint32_t Rd = (uint32_t)(vd & 0x1F);
    /* 0Q001111 00000abc cmode01 defghRd
       Q=1, op=0, abc=000, cmode=0000, defgh=00000 → 0x4F000400 | Rd
       reference: `movi v0.4s, #0` → 0x4f000400 (verified objdump). */
    uint32_t insn = 0x4F000400u | Rd;
    ax_jit_arm64_emit32(b, insn);
}

/* LD1 { Vt.4S }, [Xn], #16  — post-indexed, single 4S register
   bits[31..0]: 0Q001100 110_imm5 _opcode_ size Rn Rt
   for "LD1 single-reg, post-index, register=#imm" the syntactic form:
     LD1 {Vt.4S}, [Xn], #16   — encoded as 0x4CDF7800 | (Rn<<5) | Rt
   verified: as → 0x4CDF7800 base. */
void ax_jit_arm64_emit_ld1_post(ax_jit_arm64_buf_t *b, int vd, int xn, int post_imm) {
    /* we currently only support #16 post-imm (one 4S = 16 bytes).
       generic post-index immediate would need different opcode bits. */
    (void)post_imm;
    uint32_t Rn = (uint32_t)(xn & 0x1F);
    uint32_t Rt = (uint32_t)(vd & 0x1F);
    uint32_t insn = 0x4CDF7800u | (Rn << 5) | Rt;
    ax_jit_arm64_emit32(b, insn);
}

/* ST1 { Vt.4S }, [Xn], #16   — post-indexed
   base encoding: 0x4C9F7800 | (Rn<<5) | Rt */
void ax_jit_arm64_emit_st1_post(ax_jit_arm64_buf_t *b, int vd, int xn, int post_imm) {
    (void)post_imm;
    uint32_t Rn = (uint32_t)(xn & 0x1F);
    uint32_t Rt = (uint32_t)(vd & 0x1F);
    uint32_t insn = 0x4C9F7800u | (Rn << 5) | Rt;
    ax_jit_arm64_emit32(b, insn);
}

/* LDR Qt, [Xn, #imm]   — Q (128-bit SIMD) load with 12-bit unsigned imm,
   scaled by 16. encoding: 00111101 11 imm12 Rn Rt → 0x3DC00000 base.
   imm12 holds (imm/16). */
void ax_jit_arm64_emit_ldr_q_imm(ax_jit_arm64_buf_t *b, int vd, int xn, int imm12_scaled16) {
    uint32_t Rn = (uint32_t)(xn & 0x1F);
    uint32_t Rt = (uint32_t)(vd & 0x1F);
    uint32_t imm = (uint32_t)(imm12_scaled16 & 0xFFF);
    uint32_t insn = 0x3DC00000u | (imm << 10) | (Rn << 5) | Rt;
    ax_jit_arm64_emit32(b, insn);
}

/* STR Qt, [Xn, #imm] — base 0x3D800000, same imm format. */
void ax_jit_arm64_emit_str_q_imm(ax_jit_arm64_buf_t *b, int vd, int xn, int imm12_scaled16) {
    uint32_t Rn = (uint32_t)(xn & 0x1F);
    uint32_t Rt = (uint32_t)(vd & 0x1F);
    uint32_t imm = (uint32_t)(imm12_scaled16 & 0xFFF);
    uint32_t insn = 0x3D800000u | (imm << 10) | (Rn << 5) | Rt;
    ax_jit_arm64_emit32(b, insn);
}

/* FADD Vd.4S, Vn.4S, Vm.4S — 0 Q 0 01110 sz 1 Rm 11010 1 Rn Rd
   for sz=0 (S), Q=1: 0x4E20D400 base. */
void ax_jit_arm64_emit_fadd(ax_jit_arm64_buf_t *b, int vd, int vn, int vm) {
    uint32_t Rd = (uint32_t)(vd & 0x1F);
    uint32_t Rn = (uint32_t)(vn & 0x1F);
    uint32_t Rm = (uint32_t)(vm & 0x1F);
    uint32_t insn = 0x4E20D400u | (Rm << 16) | (Rn << 5) | Rd;
    ax_jit_arm64_emit32(b, insn);
}

/* ===== GPR ============================================================= */

/* ADD (immediate, 64-bit): sf=1 op=0 S=0 100010 sh imm12 Rn Rd → 0x91000000 base.
   imm must fit in 12 bits (no shift). */
void ax_jit_arm64_emit_add_imm(ax_jit_arm64_buf_t *b, int xd, int xn, int imm12) {
    uint32_t Rd = (uint32_t)(xd & 0x1F);
    uint32_t Rn = (uint32_t)(xn & 0x1F);
    uint32_t imm = (uint32_t)(imm12 & 0xFFF);
    uint32_t insn = 0x91000000u | (imm << 10) | (Rn << 5) | Rd;
    ax_jit_arm64_emit32(b, insn);
}

/* SUB (immediate, 64-bit): 0xD1000000 base */
void ax_jit_arm64_emit_sub_imm(ax_jit_arm64_buf_t *b, int xd, int xn, int imm12) {
    uint32_t Rd = (uint32_t)(xd & 0x1F);
    uint32_t Rn = (uint32_t)(xn & 0x1F);
    uint32_t imm = (uint32_t)(imm12 & 0xFFF);
    uint32_t insn = 0xD1000000u | (imm << 10) | (Rn << 5) | Rd;
    ax_jit_arm64_emit32(b, insn);
}

/* ADD (shifted register, 64-bit, no shift): 1 0 0 01011 00 0 Rm 000000 Rn Rd → 0x8B000000 */
void ax_jit_arm64_emit_add_reg(ax_jit_arm64_buf_t *b, int xd, int xn, int xm) {
    uint32_t Rd = (uint32_t)(xd & 0x1F);
    uint32_t Rn = (uint32_t)(xn & 0x1F);
    uint32_t Rm = (uint32_t)(xm & 0x1F);
    uint32_t insn = 0x8B000000u | (Rm << 16) | (Rn << 5) | Rd;
    ax_jit_arm64_emit32(b, insn);
}

/* MOV Xd, Xn — alias for ORR Xd, XZR, Xn (no shift). 1 0 1 01010 00 0 Rm 000000 11111 Rd → 0xAA0003E0 base. */
void ax_jit_arm64_emit_mov_reg(ax_jit_arm64_buf_t *b, int xd, int xn) {
    uint32_t Rd = (uint32_t)(xd & 0x1F);
    uint32_t Rm = (uint32_t)(xn & 0x1F);
    uint32_t insn = 0xAA0003E0u | (Rm << 16) | Rd;
    ax_jit_arm64_emit32(b, insn);
}

/* SUBS (immediate, 64-bit): 1 1 1 100010 sh imm12 Rn Rd → 0xF1000000 base */
void ax_jit_arm64_emit_subs_imm(ax_jit_arm64_buf_t *b, int xd, int xn, int imm12) {
    uint32_t Rd = (uint32_t)(xd & 0x1F);
    uint32_t Rn = (uint32_t)(xn & 0x1F);
    uint32_t imm = (uint32_t)(imm12 & 0xFFF);
    uint32_t insn = 0xF1000000u | (imm << 10) | (Rn << 5) | Rd;
    ax_jit_arm64_emit32(b, insn);
}

/* B.cond — 0101010 0 imm19 0 cond → 0x54000000 base; cond NE = 0001. */
void ax_jit_arm64_emit_bne_rel(ax_jit_arm64_buf_t *b, int rel_insn19) {
    uint32_t imm = (uint32_t)(rel_insn19 & 0x7FFFF);
    uint32_t insn = 0x54000000u | (imm << 5) | 0x1u;   /* cond=NE */
    ax_jit_arm64_emit32(b, insn);
}

/* B.NE to label: rel = (target_pos - this_pos) / 4. range ±1MB. */
void ax_jit_arm64_emit_bne_to(ax_jit_arm64_buf_t *b, ax_jit_arm64_label_t target) {
    if (!b) return;
    long this_pos = (long)b->pos;
    long off_bytes = (long)target - this_pos;
    /* must be 4-aligned */
    if ((off_bytes & 3) != 0) return;
    long off_insn = off_bytes / 4;
    /* imm19 is signed 19-bit: range [-2^18, 2^18-1] in instructions = ±1 MiB bytes */
    if (off_insn < -262144 || off_insn > 262143) return;
    ax_jit_arm64_emit_bne_rel(b, (int)off_insn);
}

/* RET — branches to LR (X30). encoding: 1101011001011111000000 Rn 00000.
   for Rn=X30: 0xD65F03C0. */
void ax_jit_arm64_emit_ret(ax_jit_arm64_buf_t *b) {
    ax_jit_arm64_emit32(b, 0xD65F03C0u);
}
