import time

import tensorflow as tf


def transformer_loss_function(target, pred):
    mask = tf.math.logical_not(tf.math.equal(target, 0))
    loss_ = loss_object(target, pred)

    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask

    return tf.reduce_mean(loss_)


class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, d_model, warmup_steps=4000):
        super(CustomSchedule, self).__init__()
        self.d_model = tf.cast(d_model, tf.float32)
        self.warmup_steps = warmup_steps

    def __call__(self, step):
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)
        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)


def main_train(dataset, transformer, n_epochs, print_every=50):
    """Train the transformer model for n_epochs using the data generator dataset"""

    losses = []
    accuracies = []

    # In every epoch
    for epoch in range(n_epochs):
        print("Inicio del epoch {}".format(epoch + 1))
        start = time.time()

        # Reset the losss and accuracy calculations
        train_loss.reset_states()
        train_accuracy.reset_states()

        # Get a batch of inputs and targets
        for (batch, (enc_inputs, targets)) in enumerate(dataset):
            # Set the decoder inputs
            dec_inputs = targets[:, :-1]

            # Set the target outputs, right shifted
            dec_outputs_real = targets[:, 1:]

            with tf.GradientTape() as tape:
                # Call the transformer and get the predicted output
                predictions = transformer(enc_inputs, dec_inputs, True)

                # Calculate the loss
                loss = transformer_loss_function(dec_outputs_real, predictions)

            # Update the weights and optimizer
            gradients = tape.gradient(loss, transformer.trainable_variables)
            optimizer.apply_gradients(zip(gradients, transformer.trainable_variables))

            # Save and store the metrics
            train_loss(loss)
            train_accuracy(dec_outputs_real, predictions)

            if batch % print_every == 0:
                losses.append(train_loss.result())
                accuracies.append(train_accuracy.result())
                print("Epoch {} Lote {} Pérdida {:.4f} Precisión {:.4f}".format(
                    epoch + 1, batch, train_loss.result(), train_accuracy.result()))

        # Checkpoint the model on every epoch
        ckpt_save_path = ckpt_manager.save()
        print("Saving checkpoint for epoch {} in {}".format(epoch + 1,
                                                            ckpt_save_path))
        print("Time for 1 epoch: {} secs\n".format(time.time() - start))

    return losses, accuracies

# Clean the session
tf.keras.backend.clear_session()
# Create the Transformer model
transformer = Transformer(vocab_size_enc=num_words_inputs,
                          vocab_size_dec=num_words_output,
                          d_model=D_MODEL,
                          n_layers=N_LAYERS,
                          FFN_units=FFN_UNITS,
                          n_heads=N_HEADS,
                          dropout_rate=DROPOUT_RATE)

# Define a categorical cross entropy loss
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True,
                                                            reduction="none")
# Define a metric to store the mean loss of every epoch
train_loss = tf.keras.metrics.Mean(name="train_loss")
# Define a matric to save the accuracy in every epoch
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name="train_accuracy")
# Create the scheduler for learning rate decay
leaning_rate = CustomSchedule(D_MODEL)
# Create the Adam optimizer
optimizer = tf.keras.optimizers.Adam(leaning_rate,
                                     beta_1=0.9,
                                     beta_2=0.98,
                                     epsilon=1e-9)
#Create the Checkpoint
ckpt = tf.train.Checkpoint(transformer=transformer,
                           optimizer=optimizer)

ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)

if ckpt_manager.latest_checkpoint:
    ckpt.restore(ckpt_manager.latest_checkpoint)
    print("Las checkpoint restored.")

# Train the model
losses, accuracies = main_train(dataset, transformer, EPOCHS)