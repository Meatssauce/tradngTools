import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.layers import (
    Dense,
    # MultiHeadAttention,
    InputLayer,
    LayerNormalization,
    TimeDistributed,
    Layer,
    Dropout,
    Embedding,
    Conv1D,
)
from keras.models import Model, Sequential

from Transformer import PositionalEncoding, EncoderLayer, MultiHeadAttention


class SignalEncoder(Layer):
    def __init__(self,
                 n_layers,
                 FFN_units,
                 n_heads,
                 dropout_rate,
                 d_model,
                 name="signal_encoder"):
        super(SignalEncoder, self).__init__(name=name)
        self.n_layers = n_layers
        self.d_model = d_model

        self.conv1d = Conv1D(1, kernel_size=3, activation='gelu')
        self.conv1d2 = Conv1D(1, kernel_size=3, strides=2, activation='gelu')

        self.pos_encoding = PositionalEncoding()
        self.dropout = Dropout(rate=dropout_rate)

        # Stack of n layers of multi-head attention and FC
        self.enc_layers = [EncoderLayer(FFN_units,
                                        n_heads,
                                        dropout_rate)
                           for _ in range(n_layers)]

    def call(self, inputs, mask, training):
        outputs = self.conv1d(inputs)
        outputs = self.conv1d2(outputs)

        outputs *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))

        outputs = self.pos_encoding(outputs)
        outputs = self.dropout(outputs, training)

        # Call the stacked layers
        for i in range(self.n_layers):
            outputs = self.enc_layers[i](outputs, mask, training)

        return outputs


class CompressiveEncoder(Layer):
    def __init__(self,
                 d_model,
                 reduced_d_model,
                 n_layers,
                 FFN_units,
                 n_heads,
                 dropout_rate):
        super().__init__()
        self.encoder = SignalEncoder(n_layers, FFN_units, n_heads, dropout_rate, d_model)
        self.ffn1 = Dense(units=FFN_units, activation="relu")
        self.ffn2 = Dense(units=reduced_d_model, activation="relu")

    def create_padding_mask(self, seq, *args):  # seq: (batch_size, seq_length)
        # Create the mask for padding
        mask = tf.cast(tf.math.equal(seq, 0), tf.float32)
        return mask[:, tf.newaxis, tf.newaxis, :]

    def call(self, enc_inputs, training):
        # Create the padding mask for the encoder
        enc_mask = self.create_padding_mask(enc_inputs)

        # Call the encoder
        enc_outputs = self.encoder(enc_inputs, enc_mask, training)

        # Call the ffn compressor
        outputs = self.ffn1(enc_outputs)
        outputs = self.ffn2(outputs)

        return outputs


class AttentionEncoder(Model):
    def __init__(self,
                 d_model,
                 reduced_d_model,
                 n_layers,
                 FFN_units,
                 n_heads,
                 dropout_rate,
                 name="transformer"):
        super().__init__(name=name)
        # Build the encoder
        self.encoder = CompressiveEncoder(d_model,
                                          reduced_d_model,
                                          n_layers,
                                          FFN_units,
                                          n_heads,
                                          dropout_rate)
        self.decoder = Sequential([
            Dense(reduced_d_model, activation="relu"),
            Dense(d_model, activation="relu")])

    def call(self, enc_inputs, training):
        encoded = self.encoder(enc_inputs, training)
        decoded = self.decoder(encoded)
        return decoded
