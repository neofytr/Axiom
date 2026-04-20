/* jit_x64.h — minimal AVX2 instruction encoder for runtime-emitted GEMM
   micro-kernels. linux-only (mmap/mprotect). zero deps.

   the encoder writes raw machine bytes into an mmap'd page that's later
   flipped to PROT_EXEC. an emitted block is callable as a function pointer
   matching the C calling convention assumed by the consumer.

   we encode a small set: vmovaps, vmovups, vbroadcastss, vfmadd231ps,
   vxorps, vaddps, plus the gpr arithmetic and control-flow needed for a
   K-loop micro-kernel. enough to JIT a GEMM 6x16 inner kernel for AVX2.
   AVX-512 (EVEX prefix) lives in a sibling header; this one is VEX only. */

#ifndef AX_JIT_X64_H
#define AX_JIT_X64_H

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct ax_jit_buf {
    uint8_t *base;       /* mmap'd region, RW initially, RX after make_executable */
    size_t   capacity;   /* total bytes available */
    size_t   pos;        /* current write offset */
    bool     executable; /* true after ax_jit_buf_make_executable */
} ax_jit_buf_t;

/* allocate an mmap'd RW buffer of (at least) `min_capacity` bytes,
   rounded up to page size. returns NULL on failure. */
ax_jit_buf_t *ax_jit_buf_create(size_t min_capacity);

/* free the buffer (works whether RW or RX). */
void ax_jit_buf_destroy(ax_jit_buf_t *b);

/* flip the page from RW to RX. after this the buffer is callable but
   no longer writable; further emit_* calls will fail. */
bool ax_jit_buf_make_executable(ax_jit_buf_t *b);

/* return a function pointer to the start of the buffer.
   only valid after make_executable. */
void *ax_jit_buf_entry(ax_jit_buf_t *b);

/* low-level emit: write a single byte / 4 bytes (LE). bumps b->pos. */
void ax_jit_emit_u8 (ax_jit_buf_t *b, uint8_t  v);
void ax_jit_emit_u32(ax_jit_buf_t *b, uint32_t v);

/* x86_64 GPR ids: RAX=0, RCX=1, RDX=2, RBX=3, RSP=4, RBP=5, RSI=6, RDI=7,
   R8..R15=8..15. ymm regs use 0..15 with the same numbering. */

/* AVX2 instruction emitters. all use 256-bit (L=1) ymm operands. */

/* c += a*b; vfmadd231ps ymm_dst, ymm_a, ymm_b */
void ax_jit_emit_vfmadd231ps(ax_jit_buf_t *b, int ymm_dst, int ymm_a, int ymm_b);

/* dst = src ^ src (zero); vxorps ymm_dst, ymm_dst, ymm_dst */
void ax_jit_emit_vxorps_zero(ax_jit_buf_t *b, int ymm_dst);

/* dst = a + b; vaddps ymm_dst, ymm_a, ymm_b */
void ax_jit_emit_vaddps(ax_jit_buf_t *b, int ymm_dst, int ymm_a, int ymm_b);

/* ymm_dst = broadcast(*(float *)(gpr_base + disp)) */
void ax_jit_emit_vbroadcastss(ax_jit_buf_t *b, int ymm_dst, int gpr_base, int32_t disp);

/* ymm_dst = *(__m256 *)(gpr_base + disp), aligned */
void ax_jit_emit_vmovaps_load (ax_jit_buf_t *b, int ymm_dst, int gpr_base, int32_t disp);
/* ymm_dst = *(__m256 *)(gpr_base + disp), unaligned */
void ax_jit_emit_vmovups_load (ax_jit_buf_t *b, int ymm_dst, int gpr_base, int32_t disp);
/* *(__m256 *)(gpr_base + disp) = ymm_src, aligned */
void ax_jit_emit_vmovaps_store(ax_jit_buf_t *b, int gpr_base, int32_t disp, int ymm_src);
/* *(__m256 *)(gpr_base + disp) = ymm_src, unaligned */
void ax_jit_emit_vmovups_store(ax_jit_buf_t *b, int gpr_base, int32_t disp, int ymm_src);

/* GPR arithmetic and control flow for K-loops. */

/* gpr += imm32 (sign-extended); add r64, imm32 */
void ax_jit_emit_add_r64_imm32(ax_jit_buf_t *b, int gpr, int32_t imm);
/* gpr -= imm32; sub r64, imm32 */
void ax_jit_emit_sub_r64_imm32(ax_jit_buf_t *b, int gpr, int32_t imm);
/* gpr -= 1, sets ZF; dec r64 */
void ax_jit_emit_dec_r64(ax_jit_buf_t *b, int gpr);
/* mov r64, imm64 */
void ax_jit_emit_mov_r64_imm64(ax_jit_buf_t *b, int gpr, uint64_t imm);
/* mov gpr_dst, gpr_src */
void ax_jit_emit_mov_r64_r64(ax_jit_buf_t *b, int gpr_dst, int gpr_src);

/* gpr_dst += gpr_src; add r64, r64 */
void ax_jit_emit_add_r64_r64(ax_jit_buf_t *b, int gpr_dst, int gpr_src);

/* push/pop preserve callee-saved gprs across the kernel. */
void ax_jit_emit_push_r64(ax_jit_buf_t *b, int gpr);
void ax_jit_emit_pop_r64 (ax_jit_buf_t *b, int gpr);

/* short backward jump if not zero. rel8 must reach back from current pos. */
void ax_jit_emit_jnz_rel8(ax_jit_buf_t *b, int8_t rel);

/* near backward jump if not zero. rel32 reaches anywhere within ±2 GiB. */
void ax_jit_emit_jnz_rel32(ax_jit_buf_t *b, int32_t rel);

/* return; ret near */
void ax_jit_emit_ret(ax_jit_buf_t *b);

/* prefetcht0 [gpr_base + disp] — read into L1 (closest cache).
   non-faulting, hint to bring memory closer ahead of use. */
void ax_jit_emit_prefetcht0(ax_jit_buf_t *b, int gpr_base, int32_t disp);

/* ----- AVX-512 instructions (EVEX prefix, ZMM registers 0..31) -----
   single-precision float kernel set: enough to JIT a 14x32 zmm GEMM
   micro-kernel. zmm0..15 don't need V'/R' extension; zmm16..31 do.
   the encoder handles both transparently from the register id (0..31).
   all forms are 512-bit (L'L=10), W=0 (single-precision). */

/* vfmadd231ps zmm_dst, zmm_a, zmm_b   — c += a * b (b is zmm reg) */
void ax_jit_emit_vfmadd231ps_zmm(ax_jit_buf_t *b, int z_dst, int z_a, int z_b);

/* vxorps zmm_dst, zmm_dst, zmm_dst — zero the register */
void ax_jit_emit_vxorps_zero_zmm(ax_jit_buf_t *b, int z_dst);

/* vaddps zmm_dst, zmm_a, zmm_b */
void ax_jit_emit_vaddps_zmm(ax_jit_buf_t *b, int z_dst, int z_a, int z_b);

/* vbroadcastss zmm_dst, [gpr_base + disp] */
void ax_jit_emit_vbroadcastss_zmm(ax_jit_buf_t *b, int z_dst, int gpr_base, int32_t disp);

/* aligned and unaligned 512-bit load/store of zmm regs */
void ax_jit_emit_vmovaps_load_zmm (ax_jit_buf_t *b, int z_dst, int gpr_base, int32_t disp);
void ax_jit_emit_vmovups_load_zmm (ax_jit_buf_t *b, int z_dst, int gpr_base, int32_t disp);
void ax_jit_emit_vmovaps_store_zmm(ax_jit_buf_t *b, int gpr_base, int32_t disp, int z_src);
void ax_jit_emit_vmovups_store_zmm(ax_jit_buf_t *b, int gpr_base, int32_t disp, int z_src);

/* labels for backwards jumps */
typedef size_t ax_jit_label_t;
static inline ax_jit_label_t ax_jit_here(const ax_jit_buf_t *b) { return b->pos; }

/* emit a jnz that targets a previously captured label (must be backwards
   reachable in 8-bit displacement, i.e. ≤ 128 bytes back). */
void ax_jit_emit_jnz_to(ax_jit_buf_t *b, ax_jit_label_t target);

#ifdef __cplusplus
}
#endif

#endif
