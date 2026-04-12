import { createFileRoute } from '@tanstack/react-router'

export const Route = createFileRoute('/docs/api/layers')({
  component: LayersApiPage,
})

function LayersApiPage() {
  return (
    <>
      <h1>Layers API</h1>
      <p>
        Layers are the building blocks of neural networks. Each layer has a forward function,
        knows its parameters, and can switch between train/eval mode. The design uses C
        polymorphism: concrete layers embed <code>ax_layer_t</code> as their first field, so you
        can cast between them.
      </p>
      <p>
        Headers: <code>axiom/layer.h</code>, <code>axiom/conv.h</code>, <code>axiom/norm.h</code>
      </p>

      <h2>Dense (fully connected)</h2>
      <pre><code className="language-c">{`ax_layer_t *ax_dense_create(int64_t in_features, int64_t out_features, bool use_bias);`}</code></pre>
      <p>
        Creates a dense layer: <code>out = input @ weight + bias</code>. Weight shape is
        [in_features, out_features]. Bias shape is [out_features] (or NULL if{' '}
        <code>use_bias = false</code>). Weights are initialized with Kaiming uniform by default.
      </p>
      <pre><code className="language-c">{`ax_layer_t *net = ax_sequential_create();
ax_sequential_add(net, ax_dense_create(784, 256, true));
ax_sequential_add(net, ax_relu_layer_create());
ax_sequential_add(net, ax_dense_create(256, 10, true));`}</code></pre>

      <h2>Conv2D</h2>
      <pre><code className="language-c">{`ax_layer_t *ax_conv2d_create(int in_channels, int out_channels,
                             int kernel_size, int stride, int padding);

ax_layer_t *ax_conv2d_create_ex(int in_channels, int out_channels,
                                 int kh, int kw, int sh, int sw,
                                 int ph, int pw, bool use_bias);`}</code></pre>
      <p>
        2D convolution layer. Input shape: [batch, C_in, H, W]. The basic version uses square
        kernels with equal stride and padding. The <code>_ex</code> variant allows rectangular
        kernels with separate height/width parameters.
      </p>
      <p>
        Internally, convolution is reduced to GEMM via im2col. When the backend provides{' '}
        <code>conv_gemm</code>, the im2col is done implicitly during the GEMM pack phase,
        halving memory usage.
      </p>

      <h2>Pooling</h2>

      <h3>MaxPool2D</h3>
      <pre><code className="language-c">{`ax_layer_t *ax_maxpool2d_create(int kernel_size, int stride, int padding);`}</code></pre>
      <p>Max pooling over spatial dimensions. No trainable parameters.</p>

      <h3>AvgPool2D</h3>
      <pre><code className="language-c">{`ax_layer_t *ax_avgpool2d_create(int kernel_size, int stride, int padding);`}</code></pre>
      <p>Average pooling over spatial dimensions.</p>

      <h3>GlobalAvgPool2D</h3>
      <pre><code className="language-c">{`ax_layer_t *ax_global_avgpool2d_create(void);`}</code></pre>
      <p>
        Averages over the entire spatial extent. Input [batch, C, H, W] becomes [batch, C, 1, 1].
        Common before the final dense layer in CNNs.
      </p>

      <h2>Flatten</h2>
      <pre><code className="language-c">{`ax_layer_t *ax_flatten_create(void);`}</code></pre>
      <p>
        Flattens all non-batch dimensions. Input [batch, C, H, W] becomes [batch, C*H*W].
        Used between convolutional and dense layers.
      </p>

      <h2>Normalization</h2>

      <h3>BatchNorm</h3>
      <pre><code className="language-c">{`ax_layer_t *ax_batchnorm_create(int64_t num_features, float eps, float momentum);`}</code></pre>
      <p>
        Batch normalization. Normalizes across the batch dimension. Has trainable gamma/beta
        parameters and non-trainable running mean/variance buffers. In train mode, uses batch
        statistics and updates the running stats. In eval mode, uses the running stats.
      </p>

      <h3>LayerNorm</h3>
      <pre><code className="language-c">{`ax_layer_t *ax_layernorm_create(int64_t num_features, float eps);`}</code></pre>
      <p>
        Layer normalization. Normalizes across the feature dimension (per-sample). Has trainable
        gamma/beta parameters. No running statistics since it operates per-sample.
      </p>

      <h2>Dropout</h2>
      <pre><code className="language-c">{`ax_layer_t *ax_dropout_create(float p);`}</code></pre>
      <p>
        Randomly zeros elements with probability p during training. Inactive during eval mode.
        The remaining elements are scaled by 1/(1-p) to maintain expected values (inverted dropout).
      </p>

      <h2>Activation layers</h2>
      <p>
        Activation functions wrapped as layers so they can be placed in a sequential model.
        These are stateless with no parameters.
      </p>
      <pre><code className="language-c">{`ax_layer_t *ax_relu_layer_create(void);
ax_layer_t *ax_sigmoid_layer_create(void);
ax_layer_t *ax_tanh_layer_create(void);
ax_layer_t *ax_leaky_relu_layer_create(float alpha);
ax_layer_t *ax_elu_layer_create(float alpha);
ax_layer_t *ax_gelu_layer_create(void);
ax_layer_t *ax_swish_layer_create(void);
ax_layer_t *ax_softmax_layer_create(int axis);`}</code></pre>

      <h2>Sequential</h2>
      <pre><code className="language-c">{`ax_layer_t *ax_sequential_create(void);
ax_layer_t *ax_sequential_add(ax_layer_t *seq, ax_layer_t *layer);`}</code></pre>
      <p>
        A container that chains layers together. The forward pass runs input through each layer
        in order. <code>ax_sequential_add</code> returns the sequential layer for chaining.
        Maximum <code>AX_SEQ_MAX_LAYERS</code> (64) layers.
      </p>
      <pre><code className="language-c">{`ax_layer_t *net = ax_sequential_create();
ax_sequential_add(net, ax_conv2d_create(1, 32, 3, 1, 1));
ax_sequential_add(net, ax_batchnorm_create(32, 1e-5f, 0.1f));
ax_sequential_add(net, ax_relu_layer_create());
ax_sequential_add(net, ax_maxpool2d_create(2, 2, 0));
ax_sequential_add(net, ax_flatten_create());
ax_sequential_add(net, ax_dense_create(32 * 14 * 14, 10, true));`}</code></pre>

      <h2>Fused layers</h2>
      <pre><code className="language-c">{`// AX_LAYER_CONV_BN_RELU — fused conv2d + batchnorm + relu`}</code></pre>
      <p>
        Available as a layer type for maximum performance. Combines three operations
        into a single layer to minimize memory traffic between them.
      </p>

      <h2>Common operations</h2>
      <pre><code className="language-c">{`// run forward pass
ax_tensor_t *ax_layer_forward(ax_layer_t *layer, ax_tensor_t *input);

// get trainable parameters (recursive for sequential)
int ax_layer_get_params(ax_layer_t *layer, ax_tensor_t **params, int max_params);

// get non-trainable buffers (batchnorm running stats, etc)
int ax_layer_get_buffers(ax_layer_t *layer, ax_tensor_t **buffers, int max_buffers);

// switch between train and eval mode (recursive)
void ax_layer_train(ax_layer_t *layer);
void ax_layer_eval(ax_layer_t *layer);

// diagnostics
void ax_layer_summary(ax_layer_t *layer);
int64_t ax_layer_param_count(ax_layer_t *layer);

// cleanup
void ax_layer_destroy(ax_layer_t *layer);`}</code></pre>
    </>
  )
}
