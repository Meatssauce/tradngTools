import numpy as np
import pandas as pd

from sklearn.preprocessing import MinMaxScaler


def fast_rolling_window(array, length, stride):
    """
    Produce np array subsets via rolling window over the original array. Vectorised.

    :param array: n-dim array from which to produce subsets
    :param length: length of the window
    :param stride: offsets between the start of successive windows
    :return: n+1 dimensional array with each element being a subset
    """
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
    segments = fast_rolling_window(df.to_numpy(), length=segment_len, stride=stride)
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


def resample_ohlcv(data: pd.DataFrame, freq: str, open: str = 'open', high: str = 'high', low: str = 'low',
                   close: str = 'close', volume: str = 'volume'):
    """
    Resample temporal resolution of OHLCV data

    :param data: dataframe to resample
    :param freq: new temporal resolution -- see pandas time frequency symbol
    :param open: key to open prices in data
    :param high: key to high prices in data
    :param low: key to low prices in data
    :param close: key to close prices in data
    :param volume: key to volumes in data
    :return: resampled dataframe
    """
    return data.resample(freq).agg({
        open: 'first',
        high: 'max',
        low: 'min',
        close: 'last',
        # 'Volume_(BTC)': 'sum',
        volume: 'sum',
        # 'Weighted_Price': 'mean',
        # 'Missing': 'sum',
    })
