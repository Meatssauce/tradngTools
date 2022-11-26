import json
import os
import pickle

import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

from .models import make_vanilla, make_vanilla_v2
from .mydata import load_data
from .data_processing import ts_train_test_split, resample_ohlcv, make_standardised_segments


raw_data = load_data('btc')
raw_data['Timestamp'] = pd.to_datetime(raw_data['Timestamp'], unit='s')
raw_data = raw_data.set_index('Timestamp')
raw_data = raw_data.reindex(pd.date_range(raw_data.index.min(), raw_data.index.max(), freq='min'), fill_value=np.nan)

# data['Missing'] = data.isna().any(axis=1).astype(int)
data = raw_data.interpolate(method='index')
# todo: show how much is interpolated
data = data[['Open', 'High', 'Low', 'Close', 'Volume_(Currency)']].astype(float)

# window_length = int(5 * 30 * 24 / 4)
window_length = 512
stride = 16
window_range = (-1, 1)

train_data, val_data = ts_train_test_split(data, test_size=0.3, gap_size=stride)
val_data, test_data = ts_train_test_split(val_data, test_size=0.5, gap_size=stride)

freq = '15min'
train_data = resample_ohlcv(train_data, freq, open='Open', high='High', low='Low', close='Close', volume='Volume_(Currency)')
val_data = resample_ohlcv(val_data, freq, open='Open', high='High', low='Low', close='Close', volume='Volume_(Currency)')
test_data = resample_ohlcv(test_data, freq, open='Open', high='High', low='Low', close='Close', volume='Volume_(Currency)')
# train_data = add_features(train_data)

# curriculum = []
# for i in range(window_size):
#     curriculum.append(tokenize(quantise(normalise(segment(data, offset=i))), mask))

train_time_period = train_data.index
val_time_period = val_data.index
test_time_period = test_data.index

train_data = np.stack(make_standardised_segments(train_data, window_length, window_range, stride))
val_data = np.stack(make_standardised_segments(val_data, window_length, window_range, stride))
test_data = np.stack(make_standardised_segments(test_data, window_length, window_range, stride))

input_shape = train_data[0].shape
model = make_vanilla_v2(kernel_size=5, input_size=input_shape)
print(f'{input_shape=}')

# lr_schedule = tf.keras.optimizers.schedules.PolynomialDecay(
#     initial_learning_rate=0.01,
#     decay_steps=10000,
#     end_learning_rate=0.00001,
#     power=1.0,
#     cycle=False,
#     name=None
# )
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005), loss="mse")
model.summary()

history = model.fit(
    train_data,
    train_data,
    batch_size=128,
    epochs=50,
    validation_data=(val_data, val_data),
    callbacks=[
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=20, mode="min")
    ],
    verbose=1
)

# Save training logs

output_dir = '../trainingLog'
os.makedirs(output_dir, exist_ok=True)

with open(os.path.join(output_dir, 'trainHistoryDict.json'), 'w') as f:
    json.dump(history.history, f)

# Save model
model.save('vanilla_autoencoder.model')

print(model.evaluate(test_data, test_data))

test_pred = model.predict(test_data)

for i, title in enumerate(['Open', 'High', 'Low', 'Close', 'Volume']):
    plt.title(title)
    plt.plot(test_data[1000, :, i], label='original')
    plt.plot(test_pred[1000, :, i], label='reconstructed')
    plt.legend()
    plt.savefig(os.path.join(output_dir, title + '.png'))
    plt.show()

# Save encoder and decoder
model.layers[0].save('vanilla_encoder.model')
model.layers[1].save('vanilla_decoder.model')
print(model.layers)

# print(tf.keras.models.load_model('vanilla_encoder.model').predict(test_data).shape)
