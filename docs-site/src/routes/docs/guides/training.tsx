import { createFileRoute } from '@tanstack/react-router'

export const Route = createFileRoute('/docs/guides/training')({
  component: TrainingGuidePage,
})

function TrainingGuidePage() {
  return (
    <>
      <h1>Training a Model</h1>
      <p>
        This guide walks through a complete training setup: building a network, loading data,
        batched training with a dataloader, learning rate scheduling, evaluation, and saving
        the trained model.
      </p>

      <h2>1. Define the network</h2>
      <pre><code className="language-c">{`ax_layer_t *net = ax_sequential_create();

// conv block
ax_sequential_add(net, ax_conv2d_create(1, 32, 3, 1, 1));
ax_sequential_add(net, ax_batchnorm_create(32, 1e-5f, 0.1f));
ax_sequential_add(net, ax_relu_layer_create());
ax_sequential_add(net, ax_maxpool2d_create(2, 2, 0));

// conv block 2
ax_sequential_add(net, ax_conv2d_create(32, 64, 3, 1, 1));
ax_sequential_add(net, ax_batchnorm_create(64, 1e-5f, 0.1f));
ax_sequential_add(net, ax_relu_layer_create());
ax_sequential_add(net, ax_maxpool2d_create(2, 2, 0));

// classifier head
ax_sequential_add(net, ax_flatten_create());
ax_sequential_add(net, ax_dropout_create(0.25f));
ax_sequential_add(net, ax_dense_create(64 * 7 * 7, 128, true));
ax_sequential_add(net, ax_relu_layer_create());
ax_sequential_add(net, ax_dropout_create(0.5f));
ax_sequential_add(net, ax_dense_create(128, 10, true));`}</code></pre>

      <h2>2. Create model and optimizer</h2>
      <pre><code className="language-c">{`ax_model_t *model = ax_model_create(net);
ax_model_summary(model);  // print architecture

ax_optimizer_t *opt = ax_adamw_create(
    model&gt;params, model&gt;n_params,
    1e-3f,    // lr
    0.9f,     // beta1
    0.999f,   // beta2
    1e-8f,    // eps
    0.01f);   // weight decay

ax_model_compile(model, opt, ax_cross_entropy_loss);`}</code></pre>

      <h2>3. Set up data loading</h2>
      <pre><code className="language-c">{`// assume train_x is [60000, 1, 28, 28] and train_y is [60000, 10] (one-hot)
ax_dataset_t *train_ds = ax_tensor_dataset_create(train_x, train_y);
ax_dataloader_t *train_dl = ax_dataloader_create(train_ds, 256, true);

ax_dataset_t *test_ds = ax_tensor_dataset_create(test_x, test_y);
ax_dataloader_t *test_dl = ax_dataloader_create(test_ds, 256, false);`}</code></pre>

      <h2>4. Learning rate schedule</h2>
      <pre><code className="language-c">{`int epochs = 20;
int steps_per_epoch = (int)ax_dataloader_num_batches(train_dl);
int total_steps = epochs * steps_per_epoch;

ax_lr_scheduler_t *sched = ax_sched_warmup_cosine(
    opt,
    steps_per_epoch,    // warmup for 1 epoch
    total_steps,
    1e-5f);             // min lr`}</code></pre>

      <h2>5. Training loop</h2>
      <pre><code className="language-c">{`for (int epoch = 0; epoch < epochs; epoch++) {
    ax_dataloader_reset(train_dl);
    ax_batch_t batch;
    float epoch_loss = 0;
    int n_batches = 0;

    while (ax_dataloader_next(train_dl, &batch)) {
        float loss = ax_model_train_step(model, batch.input, batch.target);
        epoch_loss += loss;
        n_batches++;

        ax_sched_step(sched);  // step the LR schedule

        ax_tensor_destroy(batch.input);
        ax_tensor_destroy(batch.target);
    }

    printf("epoch %2d  loss %.4f  lr %.2e\\n",
           epoch, epoch_loss / n_batches, ax_sched_get_lr(sched));
}`}</code></pre>

      <h2>6. Evaluation</h2>
      <pre><code className="language-c">{`ax_dataloader_reset(test_dl);
ax_batch_t batch;
int correct = 0, total = 0;

while (ax_dataloader_next(test_dl, &batch)) {
    ax_tensor_t *pred = ax_model_predict(model, batch.input);

    // compare argmax of predictions vs targets
    for (int i = 0; i < batch.batch_size; i++) {
        int pred_class = 0, true_class = 0;
        float pred_max = -1e30f, true_max = -1e30f;
        for (int c = 0; c < 10; c++) {
            float pv = ax_tensor_get_f32(pred, (int64_t[]){i, c});
            float tv = ax_tensor_get_f32(batch.target, (int64_t[]){i, c});
            if (pv > pred_max) { pred_max = pv; pred_class = c; }
            if (tv > true_max) { true_max = tv; true_class = c; }
        }
        if (pred_class == true_class) correct++;
        total++;
    }

    ax_tensor_destroy(pred);
    ax_tensor_destroy(batch.input);
    ax_tensor_destroy(batch.target);
}

printf("accuracy: %d/%d (%.1f%%)\\n", correct, total,
       100.0f * correct / total);`}</code></pre>

      <h2>7. Save the model</h2>
      <pre><code className="language-c">{`ax_model_save(model, "mnist_cnn.axm");`}</code></pre>

      <h2>8. Cleanup</h2>
      <pre><code className="language-c">{`ax_sched_destroy(sched);
ax_dataloader_destroy(train_dl);
ax_dataloader_destroy(test_dl);
ax_dataset_destroy(train_ds);
ax_dataset_destroy(test_ds);
ax_model_destroy(model);  // frees net, optimizer, all params
ax_tensor_destroy(train_x);
ax_tensor_destroy(train_y);
ax_tensor_destroy(test_x);
ax_tensor_destroy(test_y);`}</code></pre>

      <h2>Tips</h2>
      <ul>
        <li>
          <strong>Batch size matters.</strong> Larger batches give more stable gradients and
          better GPU utilization, but may need a larger learning rate. 64-256 is a good range
          for most problems.
        </li>
        <li>
          <strong>Use warmup+cosine scheduling.</strong> Linear warmup prevents early training
          instability, and cosine decay provides a smooth LR reduction. This is the standard
          schedule for transformer training and works well for CNNs too.
        </li>
        <li>
          <strong>Weight decay in AdamW.</strong> Use 0.01-0.1 for AdamW. Don't confuse it with
          L2 regularization in vanilla Adam (they behave differently due to the adaptive scaling).
        </li>
        <li>
          <strong>Dropout rate.</strong> 0.1-0.3 for early layers, 0.5 for the final dense layers.
          Dropout is automatically disabled during <code>ax_model_predict</code> (eval mode).
        </li>
        <li>
          <strong>Seed for reproducibility.</strong> Call <code>ax_set_seed(42)</code> before
          creating layers to get reproducible weight initialization and data shuffling.
        </li>
      </ul>

      <h2>Manual training loop</h2>
      <p>
        If you need more control (gradient clipping, multiple losses, custom logging), skip{' '}
        <code>ax_model_train_step</code> and write the loop yourself:
      </p>
      <pre><code className="language-c">{`ax_optimizer_zero_grad(opt);

ax_tensor_t *pred = ax_layer_forward(net, input);
ax_tensor_t *loss = ax_cross_entropy_loss(pred, target);

ax_backward(loss);

// optional: gradient clipping
for (int i = 0; i < model&gt;n_params; i++) {
    // clip gradients to [-1, 1] or whatever you need
}

ax_optimizer_step(opt);
ax_graph_cleanup(loss);
ax_tensor_destroy(loss);`}</code></pre>
    </>
  )
}
