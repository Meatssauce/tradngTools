import tensorflow as tf


def make_vanilla(kernel_size):
    vanilla = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(512, 5)),
        tf.keras.layers.Conv1D(
            filters=32, kernel_size=kernel_size, padding="same", strides=2, activation="relu"
        ),
        tf.keras.layers.Dropout(rate=0.2),
        tf.keras.layers.Conv1D(
            filters=16, kernel_size=kernel_size, padding="same", strides=2, activation="relu"
        ),

        tf.keras.layers.Conv1DTranspose(
            filters=16, kernel_size=kernel_size, padding="same", strides=2, activation="relu"
        ),
        tf.keras.layers.Dropout(rate=0.2),
        tf.keras.layers.Conv1DTranspose(
            filters=32, kernel_size=kernel_size, padding="same", strides=2, activation="relu"
        ),
        tf.keras.layers.Conv1DTranspose(filters=5, kernel_size=kernel_size, padding="same"),
    ])
    return vanilla


# class VanillaAutoEncoderLayer(tf.keras.layers.Layer):
#     def __init__(self, kernel_size: int):
#         super().__init__(name='vanilla_autoencoder')
#         self.kernel_size = kernel_size
#
#     def build(self, input_shape):
#         self.encoder = tf.keras.models.Sequential([
#             tf.keras.layers.Input(shape=input_shape),
#             tf.keras.layers.Conv1D(
#                 filters=32, kernel_size=self.kernel_size, padding="same", strides=2, activation="relu"
#             ),
#             tf.keras.layers.Dropout(rate=0.2),
#             tf.keras.layers.Conv1D(
#                 filters=16, kernel_size=self.kernel_size, padding="same", strides=2, activation="relu"
#             ),
#         ])
#         self.decoder = tf.keras.models.Sequential([
#             tf.keras.layers.Input(shape=self.encoder.output_shape),
#             tf.keras.layers.Conv1DTranspose(
#                 filters=16, kernel_size=self.kernel_size, padding="same", strides=2, activation="relu"
#             ),
#             tf.keras.layers.Dropout(rate=0.2),
#             tf.keras.layers.Conv1DTranspose(
#                 filters=32, kernel_size=self.kernel_size, padding="same", strides=2, activation="relu"
#             ),
#             tf.keras.layers.Conv1DTranspose(filters=input_shape[-1], kernel_size=self.kernel_size, padding="same"),
#         ])
#
#     def call(self, inputs, training=None, mask=None):
#         x = self.encoder(inputs, training=training, mask=mask)
#         x = self.decoder(x, training=training, mask=mask)
#         return x


def make_vanilla_v2(kernel_size, input_size: tuple[int, int]):
    vanilla = tf.keras.Sequential([
        tf.keras.layers.Input(shape=input_size),
        tf.keras.models.Sequential([
            tf.keras.layers.Conv1D(
                filters=32, kernel_size=kernel_size, padding="same", strides=2, activation="relu"
            ),
            tf.keras.layers.Dropout(rate=0.1),
            tf.keras.layers.Conv1D(
                filters=16, kernel_size=kernel_size, padding="same", strides=2, activation="relu"
            ),
            tf.keras.layers.Dropout(rate=0.1),
            tf.keras.layers.Dense(1024)
        ]),
        tf.keras.models.Sequential([
            tf.keras.layers.Dense(2048),
            tf.keras.layers.Dropout(rate=0.1),
            tf.keras.layers.Conv1DTranspose(
                filters=16, kernel_size=kernel_size, padding="same", strides=2, activation="relu"
            ),
            tf.keras.layers.Dropout(rate=0.1),
            tf.keras.layers.Conv1DTranspose(
                filters=32, kernel_size=kernel_size, padding="same", strides=2, activation="relu"
            ),
            tf.keras.layers.Conv1DTranspose(filters=5, kernel_size=kernel_size, padding="same"),
        ])
    ])
    return vanilla
