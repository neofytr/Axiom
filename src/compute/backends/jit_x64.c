/* jit_x64.c — implementation of the AVX2 instruction encoder. */

#include "jit_x64.h"

#include <stdlib.h>
#include <string.h>

#if defined(__linux__) || defined(__APPLE__)
  #include <sys/mman.h>
  #include <unistd.h>
  #define AX_JIT_HAS_MMAP 1
#else
  #define AX_JIT_HAS_MMAP 0
#endif

/* ----- buffer management ------------------------------------------------ */

ax_jit_buf_t *ax_jit_buf_create(size_t min_capacity) {
#if !AX_JIT_HAS_MMAP
    (void)min_capacity;
    return NULL;
#else
    long page = sysconf(_SC_PAGESIZE);
    if (page <= 0) page = 4096;
    size_t cap = ((min_capacity + (size_t)page - 1) / (size_t)page) * (size_t)page;
    if (cap == 0) cap = (size_t)page;

    void *p = mmap(NULL, cap, PROT_READ | PROT_WRITE,
                   MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (p == MAP_FAILED) return NULL;

    ax_jit_buf_t *b = (ax_jit_buf_t *)calloc(1, sizeof(*b));
    if (!b) { munmap(p, cap); return NULL; }
    b->base = (uint8_t *)p;
    b->capacity = cap;
    b->pos = 0;
    b->executable = false;
    return b;
#endif
}

void ax_jit_buf_destroy(ax_jit_buf_t *b) {
    if (!b) return;
#if AX_JIT_HAS_MMAP
    if (b->base) munmap(b->base, b->capacity);
#endif
    free(b);
}

bool ax_jit_buf_make_executable(ax_jit_buf_t *b) {
#if !AX_JIT_HAS_MMAP
    (void)b;
    return false;
#else
    if (!b || !b->base) return false;
    /* drop write, add execute. on systems with W^X enforcement (modern
       linux + selinux) the page must transition through this step; we
       cannot leave it RWX. */
    if (mprotect(b->base, b->capacity, PROT_READ | PROT_EXEC) != 0) return false;
    b->executable = true;

    /* x86 has coherent i/d caches for self-modifying code so no clear_cache
       call is needed here. on aarch64 we'd call __builtin___clear_cache. */
    return true;
#endif
}

void *ax_jit_buf_entry(ax_jit_buf_t *b) {
    return (b && b->executable) ? (void *)b->base : NULL;
}

/* ----- low-level emit -------------------------------------------------- */

void ax_jit_emit_u8(ax_jit_buf_t *b, uint8_t v) {
    if (!b || b->executable || b->pos >= b->capacity) return;
    b->base[b->pos++] = v;
}

void ax_jit_emit_u32(ax_jit_buf_t *b, uint32_t v) {
    ax_jit_emit_u8(b, (uint8_t)(v       & 0xFF));
    ax_jit_emit_u8(b, (uint8_t)((v >> 8 ) & 0xFF));
    ax_jit_emit_u8(b, (uint8_t)((v >> 16) & 0xFF));
    ax_jit_emit_u8(b, (uint8_t)((v >> 24) & 0xFF));
}

/* ----- VEX prefix helpers --------------------------------------------- */

/* 3-byte VEX:
     byte 0: 0xC4
     byte 1: ~R(7) ~X(6) ~B(5) mmmmm(4:0)
     byte 2:  W(7) ~vvvv(6:3) L(2) pp(1:0)
   2-byte VEX:
     byte 0: 0xC5
     byte 1: ~R(7) ~vvvv(6:3) L(2) pp(1:0)
   we only emit 3-byte VEX (always works, and our instructions need either
   the 0F38 escape or non-default W/B bits anyway). */
static void emit_vex3(ax_jit_buf_t *b, int R, int X, int B,
                       int mmmmm, int W, int vvvv, int L, int pp) {
    ax_jit_emit_u8(b, 0xC4);
    uint8_t b1 = (uint8_t)((((~R) & 1) << 7)
                          | (((~X) & 1) << 6)
                          | (((~B) & 1) << 5)
                          | (mmmmm & 0x1F));
    uint8_t b2 = (uint8_t)(((W & 1) << 7)
                          | (((~vvvv) & 0xF) << 3)
                          | ((L & 1) << 2)
                          | (pp & 3));
    ax_jit_emit_u8(b, b1);
    ax_jit_emit_u8(b, b2);
}

/* emit ModR/M + optional SIB + optional displacement for an instruction
   that addresses memory as [base + disp] (no index, scale=0). reg is the
   logical reg of the other operand (low 3 bits go in the reg field;
   the high bit was already encoded as VEX.R). base is the gpr id (low
   3 bits go in the rm field; high bit was VEX.B).
   handles RSP/R12 (need SIB), RBP/R13 (need explicit disp). */
static void emit_modrm_disp(ax_jit_buf_t *b, int reg, int base, int32_t disp) {
    int reg_low  = reg  & 7;
    int base_low = base & 7;
    int needs_sib = (base_low == 4);  /* RSP or R12: must use SIB */
    int needs_disp = (base_low == 5); /* RBP or R13: mod=00 means RIP-rel, force disp8 */

    int mod;
    if (disp == 0 && !needs_disp) {
        mod = 0;
    } else if (disp >= -128 && disp <= 127) {
        mod = 1;
    } else {
        mod = 2;
    }

    uint8_t modrm = (uint8_t)((mod << 6) | (reg_low << 3) | base_low);
    ax_jit_emit_u8(b, modrm);

    if (needs_sib) {
        /* SIB: scale=00 (×1), index=100 (none), base=base_low. */
        uint8_t sib = (uint8_t)((0 << 6) | (4 << 3) | base_low);
        ax_jit_emit_u8(b, sib);
    }

    if (mod == 1) {
        ax_jit_emit_u8(b, (uint8_t)(int8_t)disp);
    } else if (mod == 2) {
        ax_jit_emit_u32(b, (uint32_t)disp);
    }
}

/* ----- AVX2 instructions ---------------------------------------------- */

/* VFMADD231PS ymm_dst, ymm_a, ymm_b/m
   VEX.DDS.256.66.0F38.W0  B8 /r
   ymm_dst = dest (ModR/M reg), ymm_a = src1 (vvvv), ymm_b = src2 (ModR/M r/m) */
void ax_jit_emit_vfmadd231ps(ax_jit_buf_t *b, int ymm_dst, int ymm_a, int ymm_b) {
    int R = (ymm_dst >> 3) & 1;
    int B = (ymm_b   >> 3) & 1;
    emit_vex3(b, R, 0, B, /*mmmmm=*/0x02, /*W=*/0, /*vvvv=*/ymm_a, /*L=*/1, /*pp=*/1);
    ax_jit_emit_u8(b, 0xB8);
    /* reg-reg ModR/M: mod=11 */
    uint8_t modrm = (uint8_t)(0xC0 | ((ymm_dst & 7) << 3) | (ymm_b & 7));
    ax_jit_emit_u8(b, modrm);
}

/* VXORPS ymm_dst, ymm_dst, ymm_dst — zero a register
   VEX.NDS.256.0F.WIG  57 /r */
void ax_jit_emit_vxorps_zero(ax_jit_buf_t *b, int ymm_dst) {
    int R = (ymm_dst >> 3) & 1;
    int B = (ymm_dst >> 3) & 1;  /* same reg used as src2 */
    emit_vex3(b, R, 0, B, /*mmmmm=*/0x01, /*W=*/0, /*vvvv=*/ymm_dst, /*L=*/1, /*pp=*/0);
    ax_jit_emit_u8(b, 0x57);
    uint8_t modrm = (uint8_t)(0xC0 | ((ymm_dst & 7) << 3) | (ymm_dst & 7));
    ax_jit_emit_u8(b, modrm);
}

/* VADDPS ymm_dst, ymm_a, ymm_b
   VEX.NDS.256.0F.WIG  58 /r */
void ax_jit_emit_vaddps(ax_jit_buf_t *b, int ymm_dst, int ymm_a, int ymm_b) {
    int R = (ymm_dst >> 3) & 1;
    int B = (ymm_b   >> 3) & 1;
    emit_vex3(b, R, 0, B, /*mmmmm=*/0x01, /*W=*/0, /*vvvv=*/ymm_a, /*L=*/1, /*pp=*/0);
    ax_jit_emit_u8(b, 0x58);
    uint8_t modrm = (uint8_t)(0xC0 | ((ymm_dst & 7) << 3) | (ymm_b & 7));
    ax_jit_emit_u8(b, modrm);
}

/* VBROADCASTSS ymm_dst, m32
   VEX.256.66.0F38.W0  18 /r */
void ax_jit_emit_vbroadcastss(ax_jit_buf_t *b, int ymm_dst, int gpr_base, int32_t disp) {
    int R = (ymm_dst  >> 3) & 1;
    int B = (gpr_base >> 3) & 1;
    emit_vex3(b, R, 0, B, /*mmmmm=*/0x02, /*W=*/0, /*vvvv=*/0, /*L=*/1, /*pp=*/1);
    ax_jit_emit_u8(b, 0x18);
    emit_modrm_disp(b, ymm_dst, gpr_base, disp);
}

/* VMOVAPS ymm_dst, m256       VEX.256.0F.WIG 28 /r
   VMOVAPS m256, ymm_src       VEX.256.0F.WIG 29 /r
   VMOVUPS ymm_dst, m256       VEX.256.0F.WIG 10 /r
   VMOVUPS m256, ymm_src       VEX.256.0F.WIG 11 /r */
static void emit_vmov_mem(ax_jit_buf_t *b, int ymm, int gpr_base, int32_t disp,
                          uint8_t opcode) {
    int R = (ymm      >> 3) & 1;
    int B = (gpr_base >> 3) & 1;
    emit_vex3(b, R, 0, B, /*mmmmm=*/0x01, /*W=*/0, /*vvvv=*/0, /*L=*/1, /*pp=*/0);
    ax_jit_emit_u8(b, opcode);
    emit_modrm_disp(b, ymm, gpr_base, disp);
}

void ax_jit_emit_vmovaps_load (ax_jit_buf_t *b, int ymm_dst, int gpr_base, int32_t disp) {
    emit_vmov_mem(b, ymm_dst, gpr_base, disp, 0x28);
}
void ax_jit_emit_vmovups_load (ax_jit_buf_t *b, int ymm_dst, int gpr_base, int32_t disp) {
    emit_vmov_mem(b, ymm_dst, gpr_base, disp, 0x10);
}
void ax_jit_emit_vmovaps_store(ax_jit_buf_t *b, int gpr_base, int32_t disp, int ymm_src) {
    emit_vmov_mem(b, ymm_src, gpr_base, disp, 0x29);
}
void ax_jit_emit_vmovups_store(ax_jit_buf_t *b, int gpr_base, int32_t disp, int ymm_src) {
    emit_vmov_mem(b, ymm_src, gpr_base, disp, 0x11);
}

/* ----- GPR ops --------------------------------------------------------- */

/* REX prefix: 0100 W R X B
   W=1 for 64-bit operand. R/X/B extend reg/index/base into r8..r15. */
static void emit_rex(ax_jit_buf_t *b, int W, int R, int X, int B) {
    uint8_t rex = (uint8_t)(0x40 | ((W & 1) << 3) | ((R & 1) << 2)
                                | ((X & 1) << 1) |  (B & 1));
    if (rex != 0x40) ax_jit_emit_u8(b, rex);  /* skip if all zeros (no extension and W=0) */
    else if (W) ax_jit_emit_u8(b, 0x48);
}

/* add r64, imm32 (sign-extended).
   REX.W + 81 /0 id   for non-rax
   REX.W + 05 id      for rax  (but we use the longer form universally) */
void ax_jit_emit_add_r64_imm32(ax_jit_buf_t *b, int gpr, int32_t imm) {
    int B = (gpr >> 3) & 1;
    emit_rex(b, /*W=*/1, /*R=*/0, /*X=*/0, /*B=*/B);
    ax_jit_emit_u8(b, 0x81);
    /* ModR/M: mod=11, reg=000 (extension /0), r/m=gpr_low3 */
    ax_jit_emit_u8(b, (uint8_t)(0xC0 | (0 << 3) | (gpr & 7)));
    ax_jit_emit_u32(b, (uint32_t)imm);
}

/* sub r64, imm32 — REX.W + 81 /5 id */
void ax_jit_emit_sub_r64_imm32(ax_jit_buf_t *b, int gpr, int32_t imm) {
    int B = (gpr >> 3) & 1;
    emit_rex(b, 1, 0, 0, B);
    ax_jit_emit_u8(b, 0x81);
    ax_jit_emit_u8(b, (uint8_t)(0xC0 | (5 << 3) | (gpr & 7)));
    ax_jit_emit_u32(b, (uint32_t)imm);
}

/* dec r64 — REX.W + FF /1 */
void ax_jit_emit_dec_r64(ax_jit_buf_t *b, int gpr) {
    int B = (gpr >> 3) & 1;
    emit_rex(b, 1, 0, 0, B);
    ax_jit_emit_u8(b, 0xFF);
    ax_jit_emit_u8(b, (uint8_t)(0xC0 | (1 << 3) | (gpr & 7)));
}

/* mov r64, imm64 — REX.W + B8+rd io */
void ax_jit_emit_mov_r64_imm64(ax_jit_buf_t *b, int gpr, uint64_t imm) {
    int B = (gpr >> 3) & 1;
    emit_rex(b, 1, 0, 0, B);
    ax_jit_emit_u8(b, (uint8_t)(0xB8 + (gpr & 7)));
    ax_jit_emit_u32(b, (uint32_t)(imm & 0xFFFFFFFFu));
    ax_jit_emit_u32(b, (uint32_t)(imm >> 32));
}

/* mov r64, r64 — REX.W + 89 /r (mr form: dest is r/m) */
void ax_jit_emit_mov_r64_r64(ax_jit_buf_t *b, int gpr_dst, int gpr_src) {
    int R = (gpr_src >> 3) & 1;
    int B = (gpr_dst >> 3) & 1;
    emit_rex(b, 1, R, 0, B);
    ax_jit_emit_u8(b, 0x89);
    ax_jit_emit_u8(b, (uint8_t)(0xC0 | ((gpr_src & 7) << 3) | (gpr_dst & 7)));
}

/* add r64, r64 — REX.W + 01 /r (mr form: dest is r/m) */
void ax_jit_emit_add_r64_r64(ax_jit_buf_t *b, int gpr_dst, int gpr_src) {
    int R = (gpr_src >> 3) & 1;
    int B = (gpr_dst >> 3) & 1;
    emit_rex(b, 1, R, 0, B);
    ax_jit_emit_u8(b, 0x01);
    ax_jit_emit_u8(b, (uint8_t)(0xC0 | ((gpr_src & 7) << 3) | (gpr_dst & 7)));
}

/* push r64 — 50+rd (with optional REX for r8..r15) */
void ax_jit_emit_push_r64(ax_jit_buf_t *b, int gpr) {
    if (gpr >= 8) ax_jit_emit_u8(b, 0x41);  /* REX.B */
    ax_jit_emit_u8(b, (uint8_t)(0x50 + (gpr & 7)));
}

/* pop r64 — 58+rd */
void ax_jit_emit_pop_r64(ax_jit_buf_t *b, int gpr) {
    if (gpr >= 8) ax_jit_emit_u8(b, 0x41);
    ax_jit_emit_u8(b, (uint8_t)(0x58 + (gpr & 7)));
}

/* jnz rel8 — 75 cb */
void ax_jit_emit_jnz_rel8(ax_jit_buf_t *b, int8_t rel) {
    ax_jit_emit_u8(b, 0x75);
    ax_jit_emit_u8(b, (uint8_t)rel);
}

/* jnz rel32 — 0F 85 cd cd cd cd */
void ax_jit_emit_jnz_rel32(ax_jit_buf_t *b, int32_t rel) {
    ax_jit_emit_u8(b, 0x0F);
    ax_jit_emit_u8(b, 0x85);
    ax_jit_emit_u32(b, (uint32_t)rel);
}

/* jnz to a previously captured label. uses rel8 if it fits, else rel32. */
void ax_jit_emit_jnz_to(ax_jit_buf_t *b, ax_jit_label_t target) {
    if (!b) return;
    /* short form: 2 bytes. compute as if we'll emit short. */
    long after_short = (long)b->pos + 2;
    long off_short = (long)target - after_short;
    if (off_short >= -128 && off_short <= 127) {
        ax_jit_emit_jnz_rel8(b, (int8_t)off_short);
        return;
    }
    /* near form: 6 bytes. rel32 from end of instruction. */
    long after_near = (long)b->pos + 6;
    long off_near = (long)target - after_near;
    ax_jit_emit_jnz_rel32(b, (int32_t)off_near);
}

/* ret — C3 */
void ax_jit_emit_ret(ax_jit_buf_t *b) {
    ax_jit_emit_u8(b, 0xC3);
}

/* prefetcht0 m8 — 0F 18 /1
   reg field of ModR/M is the prefetch hint (1 = T0 = into L1). */
void ax_jit_emit_prefetcht0(ax_jit_buf_t *b, int gpr_base, int32_t disp) {
    int B = (gpr_base >> 3) & 1;
    /* REX with B if accessing r8..r15 */
    if (B) ax_jit_emit_u8(b, 0x41);
    ax_jit_emit_u8(b, 0x0F);
    ax_jit_emit_u8(b, 0x18);
    /* reg=001 (T0 hint), r/m=base */
    emit_modrm_disp(b, 1, gpr_base, disp);
}

/* ===== AVX-512 EVEX-prefixed instructions ===========================
   EVEX prefix layout (4 bytes total, byte 0 = 0x62):
     P0 (byte 1): ~R(7) ~X(6) ~B(5) ~R'(4) 0(3) 0(2) mm(1:0)
       mm: 01=0F escape, 10=0F38, 11=0F3A
       R/X/B/R' are stored INVERTED (logical 1 = encoded 0).
     P1 (byte 2):  W(7) ~vvvv(6:3) 1(2) pp(1:0)
       vvvv: 4-bit register encoding, also inverted.
       pp: 00=none, 01=66, 10=F3, 11=F2.
     P2 (byte 3):  z(7) L'(6) L(5) b(4) ~V'(3) aaa(2:0)
       L'L: 00=128, 01=256, 10=512.
       z: zero-mask. b: broadcast/sae/rc. V': extends vvvv to zmm16-31.
       aaa: opmask register.
   for our purpose (single-prec, 512-bit, no mask, no broadcast):
       z=0, L'L=10, b=0, aaa=000.
*/

static void emit_evex(ax_jit_buf_t *b, int R, int X, int B, int Rp,
                       int mm, int W, int vvvv, int Vp, int pp) {
    ax_jit_emit_u8(b, 0x62);
    /* P0: ~R ~X ~B ~R'  0 0  mm */
    uint8_t p0 = (uint8_t)((((~R) & 1) << 7) | (((~X) & 1) << 6)
                         | (((~B) & 1) << 5) | (((~Rp) & 1) << 4)
                         | (mm & 0x3));
    ax_jit_emit_u8(b, p0);
    /* P1: W  ~vvvv  1  pp */
    uint8_t p1 = (uint8_t)(((W & 1) << 7) | (((~vvvv) & 0xF) << 3)
                         | (1 << 2) | (pp & 0x3));
    ax_jit_emit_u8(b, p1);
    /* P2: 0 1 0 0  ~V' 000 — fixed bits z=0, L'L=10, b=0, aaa=0 */
    uint8_t p2 = (uint8_t)((1 << 6) | (0 << 5) | (((~Vp) & 1) << 3));
    ax_jit_emit_u8(b, p2);
}

/* common ZMM reg-reg encoder. opcode is the 1-byte opcode after EVEX.
   pp: 0=none (0F escape map only) or 1 (66 prefix, for 0F38). mm: 1=0F, 2=0F38. */
static void emit_evex_rr_zmm(ax_jit_buf_t *b, int z_dst, int z_a, int z_b,
                              int opcode, int mm, int pp) {
    int R  = (z_dst >> 3) & 1;   /* bit 3 of dest reg */
    int Rp = (z_dst >> 4) & 1;   /* bit 4 of dest (zmm16-31) */
    int B  = (z_b   >> 3) & 1;   /* bit 3 of src2 (in r/m field) */
    int Bx = (z_b   >> 4) & 1;   /* bit 4 of src2 — encoded via X bit */
    /* per Intel: when ZMM16-31 is in the r/m field, the X bit (bit 6 of P0)
       acts as the upper bit of B. so we set X = bit4 of src2. */
    int X  = Bx;
    int vvvv = z_a & 0xF;
    int Vp   = (z_a >> 4) & 1;
    emit_evex(b, R, X, B, Rp, mm, /*W=*/0, vvvv, Vp, pp);
    ax_jit_emit_u8(b, (uint8_t)opcode);
    /* ModR/M: mod=11 reg=z_dst[2:0] r/m=z_b[2:0] */
    uint8_t modrm = (uint8_t)(0xC0 | ((z_dst & 7) << 3) | (z_b & 7));
    ax_jit_emit_u8(b, modrm);
}

/* EVEX-specific ModR/M+disp emitter. unlike VEX/legacy, EVEX uses a
   COMPRESSED disp8: encoded byte = actual_disp / N, where N is the
   per-instruction operand size hint (e.g. N=64 for full ZMM load,
   N=4 for vbroadcastss). when actual_disp is divisible by N and the
   quotient fits in int8, we save 3 bytes (disp8 vs disp32). */
static void emit_evex_modrm_disp(ax_jit_buf_t *b, int reg, int base,
                                  int32_t disp, int N) {
    int reg_low  = reg  & 7;
    int base_low = base & 7;
    int needs_sib  = (base_low == 4);
    int needs_disp = (base_low == 5);

    int mod;
    int8_t disp8 = 0;
    if (disp == 0 && !needs_disp) {
        mod = 0;
    } else if (N > 0 && (disp % N) == 0) {
        int32_t q = disp / N;
        if (q >= -128 && q <= 127) {
            mod = 1;
            disp8 = (int8_t)q;
        } else {
            mod = 2;
        }
    } else {
        mod = 2;
    }

    uint8_t modrm = (uint8_t)((mod << 6) | (reg_low << 3) | base_low);
    ax_jit_emit_u8(b, modrm);
    if (needs_sib) {
        uint8_t sib = (uint8_t)((0 << 6) | (4 << 3) | base_low);
        ax_jit_emit_u8(b, sib);
    }
    if (mod == 1) {
        ax_jit_emit_u8(b, (uint8_t)disp8);
    } else if (mod == 2) {
        ax_jit_emit_u32(b, (uint32_t)disp);
    }
}

/* common ZMM mem-form (load/store) encoder.
   N is the disp8 scale factor: 64 for full ZMM load/store, 4 for
   vbroadcastss (broadcasts a 32-bit scalar). */
static void emit_evex_mem_zmm(ax_jit_buf_t *b, int zmm, int gpr_base,
                               int32_t disp, int opcode, int mm, int pp, int N) {
    int R  = (zmm >> 3) & 1;
    int Rp = (zmm >> 4) & 1;
    int B  = (gpr_base >> 3) & 1;
    int X  = 0;
    emit_evex(b, R, X, B, Rp, mm, /*W=*/0, /*vvvv=*/0, /*Vp=*/0, pp);
    ax_jit_emit_u8(b, (uint8_t)opcode);
    emit_evex_modrm_disp(b, zmm, gpr_base, disp, N);
}

/* vfmadd231ps zmm, zmm, zmm  — EVEX.DDS.512.66.0F38.W0 B8 /r */
void ax_jit_emit_vfmadd231ps_zmm(ax_jit_buf_t *b, int z_dst, int z_a, int z_b) {
    emit_evex_rr_zmm(b, z_dst, z_a, z_b, /*opc=*/0xB8, /*mm=*/2, /*pp=*/1);
}

/* vxorps zmm_dst, zmm_dst, zmm_dst  — EVEX.NDS.512.0F.W0 57 /r */
void ax_jit_emit_vxorps_zero_zmm(ax_jit_buf_t *b, int z_dst) {
    emit_evex_rr_zmm(b, z_dst, z_dst, z_dst, /*opc=*/0x57, /*mm=*/1, /*pp=*/0);
}

/* vaddps zmm, zmm, zmm  — EVEX.NDS.512.0F.W0 58 /r */
void ax_jit_emit_vaddps_zmm(ax_jit_buf_t *b, int z_dst, int z_a, int z_b) {
    emit_evex_rr_zmm(b, z_dst, z_a, z_b, /*opc=*/0x58, /*mm=*/1, /*pp=*/0);
}

/* vbroadcastss zmm, m32  — EVEX.512.66.0F38.W0 18 /r ; N=4 (32-bit scalar) */
void ax_jit_emit_vbroadcastss_zmm(ax_jit_buf_t *b, int z_dst, int gpr_base, int32_t disp) {
    emit_evex_mem_zmm(b, z_dst, gpr_base, disp, /*opc=*/0x18, /*mm=*/2, /*pp=*/1, /*N=*/4);
}

/* vmovaps zmm, m512    EVEX.512.0F.W0 28 /r ; N=64 (full vector)
   vmovaps m512, zmm    EVEX.512.0F.W0 29 /r ; N=64
   vmovups zmm, m512    EVEX.512.0F.W0 10 /r ; N=64
   vmovups m512, zmm    EVEX.512.0F.W0 11 /r ; N=64 */
void ax_jit_emit_vmovaps_load_zmm(ax_jit_buf_t *b, int z_dst, int gpr_base, int32_t disp) {
    emit_evex_mem_zmm(b, z_dst, gpr_base, disp, /*opc=*/0x28, /*mm=*/1, /*pp=*/0, /*N=*/64);
}
void ax_jit_emit_vmovups_load_zmm(ax_jit_buf_t *b, int z_dst, int gpr_base, int32_t disp) {
    emit_evex_mem_zmm(b, z_dst, gpr_base, disp, /*opc=*/0x10, /*mm=*/1, /*pp=*/0, /*N=*/64);
}
void ax_jit_emit_vmovaps_store_zmm(ax_jit_buf_t *b, int gpr_base, int32_t disp, int z_src) {
    emit_evex_mem_zmm(b, z_src, gpr_base, disp, /*opc=*/0x29, /*mm=*/1, /*pp=*/0, /*N=*/64);
}
void ax_jit_emit_vmovups_store_zmm(ax_jit_buf_t *b, int gpr_base, int32_t disp, int z_src) {
    emit_evex_mem_zmm(b, z_src, gpr_base, disp, /*opc=*/0x11, /*mm=*/1, /*pp=*/0, /*N=*/64);
}
