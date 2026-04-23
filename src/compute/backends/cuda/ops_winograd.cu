/* ops_winograd.cu — Winograd F(2,3) convolution forward on CUDA.
   I.2 of the production plan: closes the +31-47% TF gap on deep
   3x3 stride-1 convolutions where im2col + cuBLAS is bandwidth-bound.

   the algorithm (Lavin & Gray 2016):

     output tile size m = 2 (2x2)
     filter size r = 3 (3x3)
     input tile size m+r-1 = 4 (4x4, stride-2 overlap between tiles)

   per output 2x2 tile, the dense 3x3 convolution does 9 mul + 8 add
   per output element × 4 outputs = 36 mul + 32 add. Winograd F(2,3)
   replaces this with:

     U = G * w * G^T               (4x4 transformed weight, computed once)
     V = B^T * d * B               (4x4 transformed input)
     M = U ⊙ V                     (16 element-wise multiplies)
     Y = A^T * M * A               (2x2 output tile)

   total mul per 2x2 tile: 16 (vs 36 dense) — 2.25x reduction.
   the transforms (computed once per channel) add some overhead, so the
   real-world win on a batched/many-channel conv is ~30-50%.

   matrices (Lavin & Gray, fp32):

     B^T = [[ 1, 0,-1, 0],     G = [[1,    0,    0],
            [ 0, 1, 1, 0],          [0.5,  0.5,  0.5],
            [ 0,-1, 1, 0],          [0.5, -0.5,  0.5],
            [ 0, 1, 0,-1]]          [0,    0,    1]]

     A^T = [[1, 1,  1, 0],
            [0, 1, -1,-1]]

   ───────────────────────────────────────────────────────────────────
   pipeline (matches plan I.2):

   1. weight transform kernel: U[16, C_out, C_in] = G * w * G^T
        one block per (c_out), each thread handles one c_in × 1 of the
        16 transformed positions. cached across forward calls while the
        weight tensor's storage->generation counter is stable.

   2. input transform kernel: V[16, C_in, num_tiles] where
        num_tiles = N * tile_y_count * tile_x_count
        one thread per (n, ty, tx, cin), reading the 4x4 patch with
        zero-padding on borders, computing B^T*d*B, scattering into
        16 channel-major planes.

   3. cublasSgemmStridedBatched: 16 gemms in one call, each
        [C_out, num_tiles] = U[16, C_out, C_in] @ V[16, C_in, num_tiles]
        the `16` axis is the stridedBatched batch dim.

   4. output transform kernel: Y[N, C_out, ty, tx] += A^T * M * A
        one thread per (n, c_out, ty, tx), reads 16 floats from the
        gemm result, computes 2x2, scatters to NCHW output (with bias
        add if provided).
   ─────────────────────────────────────────────────────────────────── */

#include "internal.h"

/* output / filter / input tile constants for F(2,3) */
#define WG_M 2  /* output tile dim  */
#define WG_R 3  /* filter dim       */
#define WG_T 4  /* input tile dim = M + R - 1 */
#define WG_TT 16 /* WG_T * WG_T — flattened 4x4 plane count */

/* G * w via small-matrix const-coefficient mul. w is row-major 3x3,
   Gw is row-major 4x3, returned by index Gw[i,j] for i in 0..3, j in 0..2.
   matrix-coefficient breakdown:
     Gw[0,j] = w[0,j]
     Gw[1,j] = 0.5*(w[0,j] + w[1,j] + w[2,j])
     Gw[2,j] = 0.5*(w[0,j] - w[1,j] + w[2,j])
     Gw[3,j] = w[2,j]
*/
__device__ __forceinline__ void gw_3x3_to_4x3(const float w[9], float Gw[12]) {
    #pragma unroll
    for (int j = 0; j < 3; j++) {
        float w0 = w[0*3 + j];
        float w1 = w[1*3 + j];
        float w2 = w[2*3 + j];
        Gw[0*3 + j] = w0;
        Gw[1*3 + j] = 0.5f * (w0 + w1 + w2);
        Gw[2*3 + j] = 0.5f * (w0 - w1 + w2);
        Gw[3*3 + j] = w2;
    }
}

/* Gw * G^T via const-coefficient mul. Gw is row-major 4x3, U is 4x4. */
__device__ __forceinline__ void gw_gt_to_4x4(const float Gw[12], float U[16]) {
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        float gw0 = Gw[i*3 + 0];
        float gw1 = Gw[i*3 + 1];
        float gw2 = Gw[i*3 + 2];
        U[i*4 + 0] = gw0;
        U[i*4 + 1] = 0.5f * (gw0 + gw1 + gw2);
        U[i*4 + 2] = 0.5f * (gw0 - gw1 + gw2);
        U[i*4 + 3] = gw2;
    }
}

/* B^T * d for 4x4 d (row-major). result is 4x4 row-major. const coefs:
     row 0:  d[0,j] - d[2,j]
     row 1:  d[1,j] + d[2,j]
     row 2: -d[1,j] + d[2,j]
     row 3:  d[1,j] - d[3,j]
*/
__device__ __forceinline__ void bt_d_4x4(const float d[16], float Btd[16]) {
    #pragma unroll
    for (int j = 0; j < 4; j++) {
        float d0 = d[0*4 + j];
        float d1 = d[1*4 + j];
        float d2 = d[2*4 + j];
        float d3 = d[3*4 + j];
        Btd[0*4 + j] = d0 - d2;
        Btd[1*4 + j] = d1 + d2;
        Btd[2*4 + j] = -d1 + d2;
        Btd[3*4 + j] = d1 - d3;
    }
}

/* (B^T * d) * B for 4x4 (B^Td). result is 4x4 row-major. const coefs:
     col 0:  Btd[i,0] - Btd[i,2]
     col 1:  Btd[i,1] + Btd[i,2]
     col 2: -Btd[i,1] + Btd[i,2]
     col 3:  Btd[i,1] - Btd[i,3]
*/
__device__ __forceinline__ void btd_b_4x4(const float Btd[16], float V[16]) {
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        float b0 = Btd[i*4 + 0];
        float b1 = Btd[i*4 + 1];
        float b2 = Btd[i*4 + 2];
        float b3 = Btd[i*4 + 3];
        V[i*4 + 0] = b0 - b2;
        V[i*4 + 1] = b1 + b2;
        V[i*4 + 2] = -b1 + b2;
        V[i*4 + 3] = b1 - b3;
    }
}

/* A^T * M for 4x4 M, output is 2x4. const coefs:
     row 0: M[0,j] + M[1,j] + M[2,j]
     row 1: M[1,j] - M[2,j] - M[3,j]
*/
__device__ __forceinline__ void at_m_2x4(const float M[16], float AtM[8]) {
    #pragma unroll
    for (int j = 0; j < 4; j++) {
        float m0 = M[0*4 + j];
        float m1 = M[1*4 + j];
        float m2 = M[2*4 + j];
        float m3 = M[3*4 + j];
        AtM[0*4 + j] = m0 + m1 + m2;
        AtM[1*4 + j] = m1 - m2 - m3;
    }
}

/* (A^T * M) * A for 2x4 AtM, output is 2x2. const coefs:
     col 0: AtM[i,0] + AtM[i,1] + AtM[i,2]
     col 1: AtM[i,1] - AtM[i,2] - AtM[i,3]
*/
__device__ __forceinline__ void atm_a_2x2(const float AtM[8], float Y[4]) {
    #pragma unroll
    for (int i = 0; i < 2; i++) {
        float a0 = AtM[i*4 + 0];
        float a1 = AtM[i*4 + 1];
        float a2 = AtM[i*4 + 2];
        float a3 = AtM[i*4 + 3];
        Y[i*2 + 0] = a0 + a1 + a2;
        Y[i*2 + 1] = a1 - a2 - a3;
    }
}

/* ── kernel 1: weight transform ───────────────────────────────────
   input:  weight [C_out, C_in, 3, 3] row-major (NCHW-style filter)
   output: U_xfm [16, C_out, C_in]
   one thread per (c_out, c_in). 16 strided writes per thread.
   the [16, ...] axis is the leading dimension so the subsequent
   batched gemm sees U_xfm[k, :, :] = U for transform position k. */
__global__ static void k_wino_weight_transform(
    const float *__restrict__ weight,
    int C_out, int C_in,
    float *__restrict__ U_xfm)
{
    int c_in  = blockIdx.x * blockDim.x + threadIdx.x;
    int c_out = blockIdx.y * blockDim.y + threadIdx.y;
    if (c_in >= C_in || c_out >= C_out) return;

    /* load 3x3 weight for (c_out, c_in) */
    const float *w_src = weight + ((int64_t)c_out * C_in + c_in) * 9;
    float w[9];
    #pragma unroll
    for (int i = 0; i < 9; i++) w[i] = w_src[i];

    /* G * w → 4x3 */
    float Gw[12];
    gw_3x3_to_4x3(w, Gw);

    /* (G * w) * G^T → 4x4 */
    float U[16];
    gw_gt_to_4x4(Gw, U);

    /* scatter to U_xfm[k, c_out, c_in] for k in 0..15 */
    int64_t plane_stride = (int64_t)C_out * C_in;
    int64_t cell = (int64_t)c_out * C_in + c_in;
    #pragma unroll
    for (int k = 0; k < 16; k++) {
        U_xfm[k * plane_stride + cell] = U[k];
    }
}

/* ── kernel 2: input transform ───────────────────────────────────
   input:  d  [N, C_in, H, W] row-major NCHW
   output: V_xfm [16, C_in, num_tiles]
            num_tiles = N * tile_y_count * tile_x_count
   one thread per (n, ty, tx, c_in). reads a 4x4 patch from d
   (with zero padding for out-of-range positions), applies B^T*d*B,
   scatters to V_xfm.

   tile (ty, tx) covers input rows [2*ty - 1, 2*ty + 2] and columns
   [2*tx - 1, 2*tx + 2] (assuming pad=1, the typical case for
   "same"-style 3x3 conv that yields out_h = H, out_w = W). out-of-range
   reads return 0. */
__global__ static void k_wino_input_transform(
    const float *__restrict__ d,
    int N, int C_in, int H, int W,
    int tile_y, int tile_x,
    int pad_h, int pad_w,
    float *__restrict__ V_xfm)
{
    /* unpack the (n, ty, tx, c_in) thread index */
    int c_in = blockIdx.x * blockDim.x + threadIdx.x;
    int tx   = blockIdx.y * blockDim.y + threadIdx.y;
    /* one z-block covers (n, ty); blockIdx.z = n * tile_y + ty */
    int z    = blockIdx.z;
    int n    = z / tile_y;
    int ty   = z - n * tile_y;
    if (c_in >= C_in || tx >= tile_x) return;

    int tiles_per_sample = tile_y * tile_x;
    int64_t num_tiles = (int64_t)N * tiles_per_sample;
    int64_t tile_idx = (int64_t)n * tiles_per_sample + (int64_t)ty * tile_x + tx;

    /* read 4x4 patch with zero padding */
    int row0 = ty * WG_M - pad_h;
    int col0 = tx * WG_M - pad_w;

    const float *src = d + (((int64_t)n * C_in) + c_in) * H * W;
    float patch[16];
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        int r = row0 + i;
        int row_in_bounds = (r >= 0 && r < H);
        #pragma unroll
        for (int j = 0; j < 4; j++) {
            int c = col0 + j;
            float v = 0.0f;
            if (row_in_bounds && c >= 0 && c < W) {
                v = src[r * W + c];
            }
            patch[i * 4 + j] = v;
        }
    }

    /* B^T * d → Btd, then Btd * B → V */
    float Btd[16];
    bt_d_4x4(patch, Btd);
    float V[16];
    btd_b_4x4(Btd, V);

    /* scatter to V_xfm[k, c_in, tile_idx] */
    int64_t plane_stride = (int64_t)C_in * num_tiles;
    int64_t cell = (int64_t)c_in * num_tiles + tile_idx;
    #pragma unroll
    for (int k = 0; k < 16; k++) {
        V_xfm[k * plane_stride + cell] = V[k];
    }
}

/* ── kernel 3: output transform ───────────────────────────────────
   input:  M_xfm [16, C_out, num_tiles]   (cuBLAS gemm output)
           bias  [C_out] (or NULL)
   output: out [N, C_out, H_out, W_out] row-major NCHW
   one thread per (n, c_out, ty, tx). reads 16 floats, computes 2x2
   tile, writes to output (clamped to valid output bounds for the
   right/bottom edges where the tile's 2x2 is partially out of range).
   bias broadcast: out[n, c_out, h, w] += bias[c_out]. */
__global__ static void k_wino_output_transform(
    const float *__restrict__ M_xfm,
    const float *__restrict__ bias,
    int N, int C_out, int H_out, int W_out,
    int tile_y, int tile_x,
    float *__restrict__ out)
{
    int tx    = blockIdx.x * blockDim.x + threadIdx.x;
    int ty    = blockIdx.y * blockDim.y + threadIdx.y;
    int z     = blockIdx.z;
    int n     = z / C_out;
    int c_out = z - n * C_out;
    if (tx >= tile_x || ty >= tile_y) return;

    int tiles_per_sample = tile_y * tile_x;
    int64_t num_tiles = (int64_t)N * tiles_per_sample;
    int64_t tile_idx = (int64_t)n * tiles_per_sample + (int64_t)ty * tile_x + tx;

    /* read 16 transformed values */
    int64_t plane_stride = (int64_t)C_out * num_tiles;
    int64_t cell = (int64_t)c_out * num_tiles + tile_idx;
    float M[16];
    #pragma unroll
    for (int k = 0; k < 16; k++) {
        M[k] = M_xfm[k * plane_stride + cell];
    }

    /* A^T * M → 2x4, then 2x4 * A → 2x2 */
    float AtM[8];
    at_m_2x4(M, AtM);
    float Y[4];
    atm_a_2x2(AtM, Y);

    /* optional bias broadcast */
    if (bias) {
        float b = bias[c_out];
        #pragma unroll
        for (int i = 0; i < 4; i++) Y[i] += b;
    }

    /* scatter to output[n, c_out, 2*ty .. 2*ty+1, 2*tx .. 2*tx+1] */
    float *dst = out + (((int64_t)n * C_out) + c_out) * H_out * W_out;
    int oy0 = ty * WG_M;
    int ox0 = tx * WG_M;
    #pragma unroll
    for (int i = 0; i < 2; i++) {
        int oy = oy0 + i;
        if (oy >= H_out) continue;
        #pragma unroll
        for (int j = 0; j < 2; j++) {
            int ox = ox0 + j;
            if (ox >= W_out) continue;
            dst[oy * W_out + ox] = Y[i * 2 + j];
        }
    }
}

/* heuristic: when does Winograd F(2,3) win over the existing implicit
   im2col + cuBLAS gemm path?

   - obviously requires kh=kw=3, sh=sw=1 (the pattern Winograd F(2,3) is
     specialised for).
   - C_in and C_out should both be reasonably large so the bulk gemm
     dominates (small channel counts let the per-tile transform dominate
     and dense im2col can win).
   - H, W should be even-ish so the 2x2 tile partition isn't dominated
     by edge waste.

   thresholds picked from the production plan: deep convs (C ≥ 32)
   with reasonable spatial extent (H,W ≥ 8). below those Winograd's
   transform overhead exceeds the FLOP savings. */
extern "C" bool ax_cuda_prefer_winograd_f23(int64_t C_in, int64_t C_out,
                                              int64_t H, int64_t W) {
    if (C_in < 32 || C_out < 32) return false;
    if (H < 8 || W < 8) return false;
    return true;
}

/* ── public entry point ───────────────────────────────────────────
   conv_winograd_f23 — same signature as cuda_conv_gemm_batched so the
   wire-up in conv.c can swap one for the other behind the heuristic.

     weight: [C_out, C_in*9] row-major (flattened 3x3 kernel)
     input_batched: device pointer to N samples of [C_in, H, W]
     N: batch size
     params: per-sample geometry (C_in, H, W, kh=kw=3, sh=sw=1, ph, pw,
             out_h, out_w)
     out_batched: [N, C_out, out_h*out_w] flat = [N, C_out, H_out, W_out]
                  contiguous

   bias is not added here — conv.c's forward path adds bias in a
   separate fused step after the gemm. */
extern "C" ax_status_t cuda_conv_winograd_f23(
    const ax_tensor_t *weight,
    const float *input_batched, int64_t N,
    const ax_conv_params_t *params,
    ax_tensor_t *out_batched)
{
    if (!weight || !input_batched || !params || !out_batched) {
        ax_err_set(AX_ERR_NULL_ARG, "winograd_f23: NULL");
        return AX_ERR_NULL_ARG;
    }
    if (weight->dtype != AX_FLOAT32 || out_batched->dtype != AX_FLOAT32) {
        ax_err_set(AX_ERR_DTYPE_MISMATCH, "winograd_f23: needs fp32");
        return AX_ERR_DTYPE_MISMATCH;
    }

    /* shape constraints */
    if (params->kh != 3 || params->kw != 3 || params->sh != 1 || params->sw != 1) {
        ax_err_set(AX_ERR_SHAPE_MISMATCH, "winograd_f23: kh=kw=3 sh=sw=1 only");
        return AX_ERR_SHAPE_MISMATCH;
    }
    if (weight->ndim != 2) {
        ax_err_set(AX_ERR_SHAPE_MISMATCH, "winograd_f23: weight must be 2D [C_out, C_in*9]");
        return AX_ERR_SHAPE_MISMATCH;
    }

    int C_out = (int)weight->shape[0];
    int C_in  = (int)params->C_in;
    int H     = (int)params->H;
    int W     = (int)params->W;
    int H_out = (int)params->out_h;
    int W_out = (int)params->out_w;
    int pad_h = (int)params->ph;
    int pad_w = (int)params->pw;
    int n_int = (int)N;
    int M     = H_out * W_out;

    /* sanity: weight[1] = C_in*9, out shape = [N, C_out, M] */
    if (weight->shape[1] != (int64_t)C_in * 9) {
        ax_err_set(AX_ERR_SHAPE_MISMATCH, "winograd_f23: weight[1] != C_in*9");
        return AX_ERR_SHAPE_MISMATCH;
    }
    if (out_batched->ndim != 3 ||
        out_batched->shape[0] != N ||
        out_batched->shape[1] != C_out ||
        out_batched->shape[2] != M) {
        ax_err_set(AX_ERR_SHAPE_MISMATCH, "winograd_f23: out shape != [N, C_out, M]");
        return AX_ERR_SHAPE_MISMATCH;
    }

    (void)n_int;

    /* tile counts: each output 2x2 tile covers a 4x4 input patch.
       for "same" padding (ph=pw=1, H_out=H, W_out=W), tile counts are
       ceil(H_out / 2) and ceil(W_out / 2). this rounds up so the
       bottom-right edge tile may have a partial output (<2 valid
       elements per axis); the output kernel clamps writes accordingly. */
    int tile_y = (H_out + WG_M - 1) / WG_M;
    int tile_x = (W_out + WG_M - 1) / WG_M;
    int64_t num_tiles = (int64_t)N * tile_y * tile_x;

    /* device buffers for the three transformed tensors. allocate fresh
       per call for correctness; a follow-up could cache U_xfm across
       forwards while the weight generation counter is unchanged.

       sizes:
         U_xfm: 16 * C_out * C_in        floats
         V_xfm: 16 * C_in  * num_tiles   floats
         M_xfm: 16 * C_out * num_tiles   floats
       on a typical (N=1, 256ch, 14x14) layer:
         U: 16*256*256 = 1 MB
         V: 16*256*49  = 0.8 MB
         M: 16*256*49  = 0.8 MB
       on bigger workloads (N=32, 256ch, 28x28):
         V: 16*256*32*196 = 100 MB — substantial. monitor for
                                     pressure on small-vram cards. */
    size_t u_bytes = (size_t)16 * C_out * C_in        * sizeof(float);
    size_t v_bytes = (size_t)16 * C_in  * num_tiles   * sizeof(float);
    size_t m_bytes = (size_t)16 * C_out * num_tiles   * sizeof(float);

    /* reuse persistent device buffers across calls. cudaMalloc takes
       ~100 µs on RTX 30-series; paying it every conv call wipes any
       algorithmic win on small workloads. three static buffers grow
       on demand (newest size wins); freed on shutdown via
       ax_cuda_shutdown_hook(). not thread-safe — caller (conv.c) is
       serialised on the default cuda stream. */
    static float *s_U = NULL, *s_V = NULL, *s_M = NULL;
    static size_t s_U_cap = 0, s_V_cap = 0, s_M_cap = 0;

    cudaError_t cerr;
    if (u_bytes > s_U_cap) {
        if (s_U) cudaFree(s_U);
        cerr = cudaMalloc(&s_U, u_bytes);
        if (cerr != cudaSuccess) {
            s_U = NULL; s_U_cap = 0;
            ax_err_set(AX_ERR_ALLOC, "winograd_f23: U cudaMalloc %s", cudaGetErrorString(cerr));
            return AX_ERR_ALLOC;
        }
        s_U_cap = u_bytes;
    }
    if (v_bytes > s_V_cap) {
        if (s_V) cudaFree(s_V);
        cerr = cudaMalloc(&s_V, v_bytes);
        if (cerr != cudaSuccess) {
            s_V = NULL; s_V_cap = 0;
            ax_err_set(AX_ERR_ALLOC, "winograd_f23: V cudaMalloc %s", cudaGetErrorString(cerr));
            return AX_ERR_ALLOC;
        }
        s_V_cap = v_bytes;
    }
    if (m_bytes > s_M_cap) {
        if (s_M) cudaFree(s_M);
        cerr = cudaMalloc(&s_M, m_bytes);
        if (cerr != cudaSuccess) {
            s_M = NULL; s_M_cap = 0;
            ax_err_set(AX_ERR_ALLOC, "winograd_f23: M cudaMalloc %s", cudaGetErrorString(cerr));
            return AX_ERR_ALLOC;
        }
        s_M_cap = m_bytes;
    }
    float *U_dev = s_U, *V_dev = s_V, *M_dev = s_M;

    /* per the conv_gemm_batched convention used by ops_conv.cu: weight
       and out are device-resident tensors (storage->data is a device
       pointer); input_batched is passed in as a raw device pointer. */
    const float *w_dev   = (const float *)weight->storage->data + weight->offset;
    const float *in_dev  = input_batched;
    float       *out_dev = (float *)out_batched->storage->data + out_batched->offset;

    /* === STEP 1: weight transform → U_dev[16, C_out, C_in] === */
    {
        dim3 block(16, 16);
        dim3 grid((C_in + block.x - 1) / block.x,
                  (C_out + block.y - 1) / block.y);
        k_wino_weight_transform<<<grid, block>>>(w_dev, C_out, C_in, U_dev);
        if (cudaGetLastError() != cudaSuccess) {
            cerr = cudaGetLastError();
            /* buffers cached — no per-call free */
            ax_err_set(AX_ERR_BACKEND, "winograd_f23: weight xfm: %s",
                       cudaGetErrorString(cerr));
            return AX_ERR_BACKEND;
        }
    }

    /* === STEP 2: input transform → V_dev[16, C_in, num_tiles] === */
    {
        dim3 block(32, 4, 1);
        dim3 grid((C_in   + block.x - 1) / block.x,
                  (tile_x + block.y - 1) / block.y,
                  (int64_t)N * tile_y);
        k_wino_input_transform<<<grid, block>>>(
            in_dev, N, C_in, H, W,
            tile_y, tile_x, pad_h, pad_w,
            V_dev);
        if (cudaGetLastError() != cudaSuccess) {
            cerr = cudaGetLastError();
            /* buffers cached — no per-call free */
            ax_err_set(AX_ERR_BACKEND, "winograd_f23: input xfm: %s",
                       cudaGetErrorString(cerr));
            return AX_ERR_BACKEND;
        }
    }

    /* === STEP 3: 16 batched gemms via cublasSgemmStridedBatched ===
       per k in 0..15:
         M[k, c_out, t] = U[k, c_out, c_in] @ V[k, c_in, t]
       cublas is column-major; we conceptually have row-major ops with
       leading dimensions (C_in, num_tiles, num_tiles). standard trick:
       compute V^T @ U^T = (U @ V)^T from cublas's POV, with M output
       in column-major reading as transposed.

       to keep this simple: call cublasSgemmStridedBatched with the
       inputs already laid out so we don't need transposes. each plane:
         U_plane: [C_out, C_in] row-major, viewed as [C_in, C_out] col-major
                  (lda = C_in)
         V_plane: [C_in, num_tiles] row-major, viewed as [num_tiles, C_in] col-major
                  (lda = num_tiles)
         M_plane: [C_out, num_tiles] row-major, viewed as [num_tiles, C_out] col-major
                  (lda = num_tiles)
       compute (col-major) M_plane^T = V_plane^T @ U_plane^T → cublas
       does N=num_tiles, M=C_out, K=C_in with all transposed views. */
    cublasHandle_t cublas = ax_cuda_cublas_handle();
    if (!cublas) {
        /* buffers cached — no per-call free */
        ax_err_set(AX_ERR_BACKEND, "winograd_f23: no cublas handle");
        return AX_ERR_BACKEND;
    }
    {
        const float alpha = 1.0f, beta = 0.0f;
        /* compute M[k] = U[k] @ V[k] in row-major view.
           cublas semantics: with op_a = N, op_b = N and column-major:
             C[m,n] = A[m,k] * B[k,n] (col-major)
           we have row-major M[c_out, num_tiles] = U[c_out, c_in] @ V[c_in, num_tiles].
           viewed as col-major:
             U is [c_in, c_out] col-major, V is [num_tiles, c_in] col-major.
           to get M[c_out, num_tiles] in row-major (= [num_tiles, c_out] col-major),
           call cublas with A=V (op T to make [c_in, num_tiles]), B=U (op T to
           make [c_out, c_in]), to compute V^T @ U^T = ...
           hmm this is getting tangled. cleanest: just use op T on both
           inputs and write the result into the col-major-equivalent layout.

           direct approach: compute M = U @ V row-major by calling
           cublasSgemm(N, N, num_tiles, C_out, C_in,
                       1, V, num_tiles, U, C_in,
                       0, M, num_tiles)
           which is col-major M[num_tiles, C_out] = V[num_tiles, C_in] *
           U[C_in, C_out]. since col-major M[i,j] = row-major M[j,i],
           the col-major output is the row-major M^T = (U @ V)^T.
           then row-major M is the same memory but read as [C_out, num_tiles]
           with leading dim num_tiles. */
        cublasStatus_t st = cublasSgemmStridedBatched(
            cublas,
            CUBLAS_OP_N, CUBLAS_OP_N,
            (int)num_tiles,                   /* m */
            C_out,                            /* n */
            C_in,                             /* k */
            &alpha,
            V_dev,    (int)num_tiles, (long long)C_in * num_tiles,  /* A, lda, strideA */
            U_dev,    C_in,           (long long)C_out * C_in,      /* B, ldb, strideB */
            &beta,
            M_dev,    (int)num_tiles, (long long)C_out * num_tiles, /* C, ldc, strideC */
            16);
        if (st != CUBLAS_STATUS_SUCCESS) {
            /* buffers cached — no per-call free */
            ax_err_set(AX_ERR_BACKEND, "winograd_f23: cublas batched gemm fail %d", (int)st);
            return AX_ERR_BACKEND;
        }
    }

    /* === STEP 4: output transform → out_dev[N, C_out, H_out*W_out] ===
       conv.c's forward path adds bias in a separate fused kernel; we
       pass NULL here to skip the in-kernel bias add. */
    {
        dim3 block(16, 8, 1);
        dim3 grid((tile_x + block.x - 1) / block.x,
                  (tile_y + block.y - 1) / block.y,
                  (int64_t)N * C_out);
        k_wino_output_transform<<<grid, block>>>(
            M_dev, /*bias=*/NULL,
            N, C_out, H_out, W_out,
            tile_y, tile_x,
            out_dev);
        if (cudaGetLastError() != cudaSuccess) {
            cerr = cudaGetLastError();
            /* buffers cached — no per-call free */
            ax_err_set(AX_ERR_BACKEND, "winograd_f23: output xfm: %s",
                       cudaGetErrorString(cerr));
            return AX_ERR_BACKEND;
        }
    }

    /* buffers cached in s_U/s_V/s_M — no per-call free */
    return AX_OK;
}
