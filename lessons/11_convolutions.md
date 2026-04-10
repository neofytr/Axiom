# Unit 11: Convolutions

## Why This Matters

Dense layers connect every input to every output. For images, this is wasteful and
impractical — a 224x224 RGB image has 150,528 inputs. A dense layer to just 1000
outputs would need 150 million parameters. Convolutions exploit spatial structure:
they apply small local filters, sharing weights across all positions. This unit
covers the geometry, the math, the im2col trick, and how Axiom implements it all.


## 11.1 The Geometric Intuition

Imagine sliding a small window (the **kernel** or **filter**) across an image.
At each position, you compute the dot product between the kernel and the patch
of the image it overlaps. The result is a single number in the output.

```
Input (5x5):          Kernel (3x3):       Output (3x3):
1 2 3 0 1            1 0 1                ?  ?  ?
0 1 2 3 0            0 1 0                ?  ?  ?
1 0 1 2 1            1 0 1                ?  ?  ?
2 1 0 1 0
0 1 2 0 1

Output[0,0] = 1*1 + 2*0 + 3*1 + 0*0 + 1*1 + 2*0 + 1*1 + 0*0 + 1*1 = 8
```

The kernel is a **feature detector**. A horizontal edge kernel might be:

    -1 -1 -1
     0  0  0
     1  1  1

It responds strongly wherever the image transitions from dark (top) to light (bottom).

Key properties:
- **Weight sharing**: the same kernel is used at every spatial position.
- **Local connectivity**: each output depends on only a small region of the input.
- **Translation equivariance**: shifting the input shifts the output by the same amount.


## 11.2 Multi-Channel Convolutions

Real images have multiple channels (RGB = 3 channels). Real conv layers produce
multiple output channels (feature maps). The full picture:

- **Input**: `[C_in, H, W]` (e.g., 3 channels, 32x32 pixels)
- **Kernel**: `[C_out, C_in, kH, kW]` (e.g., 16 filters, each 3x3x3)
- **Output**: `[C_out, H_out, W_out]` (e.g., 16 feature maps)

Each output channel uses a different 3D filter (C_in x kH x kW). The filter
slides across the spatial dimensions, computing a 3D dot product at each position.


## 11.3 Output Dimension Formula

For one spatial dimension:

    out_dim = floor((in_dim + 2 * padding - kernel_size) / stride) + 1

For the full 2D case with non-square kernels:

    H_out = floor((H + 2 * pad_h - kernel_h) / stride_h) + 1
    W_out = floor((W + 2 * pad_w - kernel_w) / stride_w) + 1

Parameters:
- **Padding**: adds zeros around the input border. `padding = kernel_size / 2`
  (integer division) keeps the output size equal to the input size (for stride 1).
- **Stride**: step size when sliding the kernel. Stride 2 halves the spatial dimensions.

Axiom implements this as:

```c
static inline int64_t conv_out_dim(int64_t in_dim, int kernel, int stride, int pad) {
    if (stride <= 0) return -1;
    int64_t out = (in_dim + 2 * pad - kernel) / stride + 1;
    if (out <= 0) return -1;
    return out;
}
```

Examples (all with kernel=3):

| Input | Padding | Stride | Output | Notes                        |
|-------|---------|--------|--------|------------------------------|
| 32    | 0       | 1      | 30     | Shrinks by kernel-1          |
| 32    | 1       | 1      | 32     | "Same" padding               |
| 32    | 1       | 2      | 16     | Downsamples by 2x            |
| 7     | 0       | 1      | 5      | Small input                  |
| 224   | 3       | 2      | 112    | Typical first conv in ResNet |


## 11.4 The NCHW Data Layout

Axiom uses **NCHW** (batch-first, channels-first) layout throughout:

    N = batch size
    C = channels
    H = height
    W = width

Memory layout for a `[2, 3, 4, 4]` tensor (2 images, 3 channels, 4x4):

```
Image 0, Channel 0:  [...16 floats...]
Image 0, Channel 1:  [...16 floats...]
Image 0, Channel 2:  [...16 floats...]
Image 1, Channel 0:  [...16 floats...]
Image 1, Channel 1:  [...16 floats...]
Image 1, Channel 2:  [...16 floats...]
```

Element at `[n, c, h, w]` is at flat index:

    index = ((n * C + c) * H + h) * W + w

NCHW is standard in most frameworks (PyTorch, Caffe) and is more cache-friendly
for convolution than NHWC (TensorFlow's default), because accessing all spatial
positions of one channel is a contiguous memory read.


## 11.5 The im2col Trick

The naive convolution is a 7-nested loop (batch, C_out, C_in, H_out, W_out, kH, kW).
This is slow because it has poor memory access patterns.

The **im2col trick** transforms convolution into a single matrix multiplication:

1. **im2col**: extract every patch the kernel would overlap and lay them out as
   columns of a matrix. For a single image `[C, H, W]` with kernel `[kH, kW]`:

   - Output matrix shape: `[C * kH * kW, H_out * W_out]`
   - Each column is one flattened patch
   - Each row corresponds to one element of the flattened kernel

2. **Matrix multiply**: `output = weight_matrix @ column_matrix`

   - Weight matrix: `[C_out, C_in * kH * kW]` (kernel reshaped to 2D)
   - Column matrix: `[C_in * kH * kW, H_out * W_out]` (from im2col)
   - Result: `[C_out, H_out * W_out]` (reshaped to `[C_out, H_out, W_out]`)

Why this works: each column of the im2col matrix contains exactly the input
elements that the kernel would see at one spatial position. The dot product of
a kernel row with a column is exactly the convolution output at that position.

Axiom's im2col (`conv.c`):

```c
ax_tensor_t *ax_im2col(ax_tensor_t *input, int kh, int kw,
                        int stride_h, int stride_w,
                        int pad_h, int pad_w)
{
    int64_t C = input->shape[0], H = input->shape[1], W = input->shape[2];
    int64_t out_h = conv_out_dim(H, kh, stride_h, pad_h);
    int64_t out_w = conv_out_dim(W, kw, stride_w, pad_w);

    // Output: [C*kh*kw, out_h*out_w]
    int64_t col_shape[] = {C * kh * kw, out_h * out_w};
    ax_tensor_t *cols = ax_tensor_zeros(col_shape, 2, AX_FLOAT32);

    // For each kernel position (c, ky, kx), fill one row of the matrix
    for (int64_t c = 0; c < C; c++)
        for (int ky = 0; ky < kh; ky++)
            for (int kx = 0; kx < kw; kx++)
                for (int64_t oh = 0; oh < out_h; oh++)
                    for (int64_t ow = 0; ow < out_w; ow++) {
                        int64_t ih = oh * stride_h - pad_h + ky;
                        int64_t iw = ow * stride_w - pad_w + kx;
                        // Padding: out-of-bounds positions stay as 0
                        if (ih >= 0 && ih < H && iw >= 0 && iw < W)
                            cols_data[...] = input_data[...];
                    }
    return cols;
}
```

The tradeoff: im2col uses more memory (the column matrix is much larger than the
input), but the resulting matmul is highly optimized on every platform (BLAS, SIMD,
GPU). This is how Caffe, PyTorch CPU, and most other frameworks implement convolution.


## 11.6 col2im: The Inverse

For the backward pass, we need the inverse operation: scatter column data back
into image format. `col2im` does this, accumulating (adding) values at overlapping
positions:

```c
ax_tensor_t *ax_col2im(ax_tensor_t *cols, int64_t channels,
                        int64_t height, int64_t width, ...);
```

The accumulation is necessary because adjacent patches overlap — each input pixel
contributes to multiple output positions, so during backprop the gradients from
all those positions must be summed.


## 11.7 Conv2d Forward Pass

Axiom's conv2d forward (`conv.c`):

```c
for each image n in the batch:
    1. Extract image n from input: [C_in, H, W]
    2. im2col: [C_in*kh*kw, out_h*out_w]
    3. matmul: weight[C_out, C_in*kh*kw] @ cols[C_in*kh*kw, out_h*out_w]
       = output[C_out, out_h*out_w]
    4. Add bias (broadcast across spatial positions)
    5. Reshape and store in output[n, :, :, :]
```

The weight tensor has shape `[C_out, C_in, kH, kW]` and is treated as
`[C_out, C_in*kH*kW]` for the matrix multiply.


## 11.8 Conv2d Backward Pass

Three gradients to compute:

**Weight gradient** (`dL/dW`):

    dW += grad_out_matrix @ col^T

Where `grad_out_matrix` is `[C_out, out_h*out_w]` and `col` is from im2col.
Result shape: `[C_out, C_in*kH*kW]`, which is the weight shape.

**Input gradient** (`dL/dx`):

    dcol = W^T @ grad_out_matrix
    dx = col2im(dcol)

The weight transpose maps the output gradient back to the column space, then
col2im scatters it back to image space.

**Bias gradient** (`dL/db`):

    db[c] = sum of grad_out over all batch, height, width positions for channel c

Simply sum the output gradient along the N, H, W dimensions.


## 11.9 Pooling

Pooling reduces spatial dimensions by summarizing local regions.

**Max Pooling**: takes the maximum value in each window.

```c
for each window position (y, x):
    output[n, c, y, x] = max over (ky, kx) of input[n, c, y*s-p+ky, x*s-p+kx]
```

Properties:
- Provides translation invariance (small shifts in input don't change output)
- No learnable parameters
- Common: kernel=2, stride=2 (halves spatial dimensions)

**Average Pooling**: takes the mean value in each window.

```c
output[n, c, y, x] = mean over (ky, kx) of input[n, c, y*s-p+ky, x*s-p+kx]
```

**Global Average Pooling**: averages each channel down to a single value.

    input:  [N, C, H, W]  ->  output: [N, C]

```c
for each (n, c):
    output[n, c] = mean over all (h, w) of input[n, c, h, w]
```

This replaces the need for a flatten + dense layer at the end of the network.
Used in modern architectures (ResNet, EfficientNet).

Axiom provides all three: `ax_maxpool2d_create`, `ax_avgpool2d_create`,
`ax_global_avgpool2d_create`.


## 11.10 Flatten

Connects convolutional layers to dense layers:

    input:  [N, C, H, W]  ->  output: [N, C*H*W]

```c
static ax_tensor_t *flatten_forward(ax_layer_t *self, ax_tensor_t *input) {
    int64_t N = input->shape[0];
    int64_t flat = 1;
    for (int d = 1; d < input->ndim; d++)
        flat *= input->shape[d];

    int64_t out_shape[] = {N, flat};

    if (ax_tensor_is_contiguous(input))
        return ax_tensor_reshape(input, out_shape, 2);  // zero-copy!

    ax_tensor_t *c = ax_tensor_contiguous(input);
    ax_tensor_t *r = ax_tensor_reshape(c, out_shape, 2);
    ax_tensor_destroy(c);
    return r;
}
```

If the tensor is already contiguous (the common case), flatten is a zero-copy
reshape — it just changes the shape and strides metadata.


## 11.11 A Typical CNN Architecture

Putting it all together:

```c
ax_layer_t *net = ax_sequential_create();

// Conv block 1: 3 -> 32 channels, 3x3 kernel, same padding
ax_sequential_add(net, ax_conv2d_create(3, 32, 3, 1, 1, true));
ax_sequential_add(net, ax_relu_layer_create());
ax_sequential_add(net, ax_maxpool2d_create(2, 2, 0));  // 32x32 -> 16x16

// Conv block 2: 32 -> 64 channels
ax_sequential_add(net, ax_conv2d_create(32, 64, 3, 1, 1, true));
ax_sequential_add(net, ax_relu_layer_create());
ax_sequential_add(net, ax_maxpool2d_create(2, 2, 0));  // 16x16 -> 8x8

// Classifier
ax_sequential_add(net, ax_flatten_create());            // [N, 64, 8, 8] -> [N, 4096]
ax_sequential_add(net, ax_dense_create(4096, 10, true));
```

Parameter count:
- Conv1: 32 * 3 * 3 * 3 + 32 = 896
- Conv2: 64 * 32 * 3 * 3 + 64 = 18,496
- Dense: 4096 * 10 + 10 = 40,970
- Total: ~60,000 (vs. 30 million for a dense-only approach on 32x32 images)


## 11.12 The Non-Square Kernel API

Axiom supports non-square kernels via `ax_conv2d_create_ex`:

```c
ax_layer_t *ax_conv2d_create_ex(int in_ch, int out_ch,
                                 int kh, int kw,       // kernel height, width
                                 int sh, int sw,       // stride height, width
                                 int ph, int pw,       // padding height, width
                                 bool use_bias);
```

The simplified `ax_conv2d_create` is a wrapper that uses equal values for both
dimensions:

```c
ax_layer_t *ax_conv2d_create(int in_ch, int out_ch, int kernel_size,
                              int stride, int padding, bool use_bias) {
    return ax_conv2d_create_ex(in_ch, out_ch, kernel_size, kernel_size,
                                stride, stride, padding, padding, use_bias);
}
```


## Key Takeaways

1. Convolutions exploit spatial structure with local, weight-shared filters.
2. Output size: `floor((input + 2*padding - kernel) / stride) + 1`.
3. NCHW layout: batch, channels, height, width — standard and cache-friendly.
4. im2col transforms convolution into matrix multiplication (trades memory for speed).
5. Backward: weight grad via `grad_out @ col^T`, input grad via `W^T @ grad_out` then col2im.
6. Pooling reduces spatial dimensions; flatten bridges conv and dense layers.
