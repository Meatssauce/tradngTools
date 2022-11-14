import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.preprocessing import MinMaxScaler


def vectorized_stride(array, length, stride):
    start = 0
    max_time = len(array) - length - 1

    sub_windows = (
            start +
            np.expand_dims(np.arange(length), 0) +
            # Create a rightmost vector as [0, V, 2V, ...].
            np.expand_dims(np.arange(max_time + 1, step=stride), 0).T
    )

    return array[sub_windows]


# arr = np.array([[i] * 10 for i in range(10)])
# vectorized_stride(arr, length=4, stride=4)


# def add_features(df):
#     quotes_list = [
#         Quote(d, o, h, l, c, v)
#         for d, o, h, l, c, v
#         in zip(df['Timestamp'], df['Open'], df['High'], df['Low'], df['Close'], df['Volume_(Currency)'])
#     ]
#     ...
#     return ...


def make_standardised_segments(df, segment_len, segment_amp_range, stride):
    segments = vectorized_stride(df.to_numpy(), length=segment_len, stride=stride)
    scaler = MinMaxScaler(feature_range=segment_amp_range)
    return [pd.DataFrame(scaler.fit_transform(segment), columns=df.columns) for segment in segments]


# def tokenize(df, amplitude_range, resolution):
#     # quantities = np.linspace(0, window_height, resolution)
#     interval = amplitude_range / (resolution - 1)
#
#     df = interval * np.round(df / interval)  # nan ok?
#     # df = df.fillna('<NULL>')
#     # df[df.isna().any(axis=1)] = '<NULL>'
#     df = df.astype(str)  # verify this works
#
#     return df

def make_curriculum(df, window_length, window_range, stride):
    return [make_standardised_segments(df[i:], segment_len=window_length, segment_amp_range=window_range, stride=stride)
            for i in range(window_length)]


def ts_train_test_split(df, test_size, gap_size):
    gap_size = int(gap_size)
    train_end = int((1 - test_size) * (len(df) - gap_size))
    return df[:train_end], df[train_end + gap_size:]


def resample(df, freq):
    return df.resample(freq).agg({
        'Open': 'first',
        'High': 'max',
        'Low': 'min',
        'Close': 'last',
        # 'Volume_(BTC)': 'sum',
        'Volume_(Currency)': 'sum',
        # 'Weighted_Price': 'mean',
        # 'Missing': 'sum',
    })


raw_data = pd.read_csv('bitstampUSD_1-min_data_2012-01-01_to_2021-03-31.csv', index_col='Timestamp')  # min by min
raw_data.index = pd.to_datetime(raw_data.index, unit='s')


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
train_data = resample(train_data, freq)
val_data = resample(val_data, freq)
test_data = resample(test_data, freq)
# train_data = add_features(train_data)

# curriculum = []
# for i in range(window_size):
#     curriculum.append(tokenize(quantise(normalise(segment(data, offset=i))), mask))
train_data = np.stack(make_standardised_segments(train_data, window_length, window_range, stride))
val_data = np.stack(make_standardised_segments(val_data, window_length, window_range, stride))
test_data = np.stack(make_standardised_segments(test_data, window_length, window_range, stride))


model = tf.keras.Sequential(
    [
        tf.keras.layers.Input(shape=(512, 5)),
        tf.keras.layers.Conv1D(
            filters=32, kernel_size=7, padding="same", strides=2, activation="relu"
        ),
        tf.keras.layers.Dropout(rate=0.2),
        tf.keras.layers.Conv1D(
            filters=16, kernel_size=7, padding="same", strides=2, activation="relu"
        ),
        tf.keras.layers.Conv1DTranspose(
            filters=16, kernel_size=7, padding="same", strides=2, activation="relu"
        ),
        tf.keras.layers.Dropout(rate=0.2),
        tf.keras.layers.Conv1DTranspose(
            filters=32, kernel_size=7, padding="same", strides=2, activation="relu"
        ),
        tf.keras.layers.Conv1DTranspose(filters=5, kernel_size=7, padding="same"),
    ]
)
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss="mse")
model.summary()

history = model.fit(
    train_data,
    train_data,
    batch_size=128,
    epochs=50,
    validation_data=(val_data, val_data),
    callbacks=[
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, mode="min")
    ],
    verbose=1
)
