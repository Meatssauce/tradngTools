import glob
import os

import numpy as np
import pandas as pd
from tqdm import tqdm


def _load_btc_bitstamp():
    return pd.read_csv('datasets/bitstampUSD_1-min_data_2012-01-01_to_2021-03-31.csv', index_col=False)


def _load_securities_us():
    malformed_files = []
    all_dfs = []
    total_file_count = 0

    for asset_class in ['Stocks', 'ETFs']:
        dfs = []
        filepaths = glob.glob(f'datasets/Data/{asset_class}/*.txt')
        total_file_count += len(filepaths)

        for filepath in tqdm(filepaths):
            try:
                df = pd.read_csv(filepath, index_col=False)
            except pd.errors.EmptyDataError:
                malformed_files.append(filepath)
            else:
                _, asset_name = os.path.split(filepath)
                df['asset_name'] = asset_name[:-4]
                dfs.append(df)

        df = pd.concat(dfs)
        df['asset_class'] = asset_class
        all_dfs.append(df)

    print(f'{len(malformed_files)} out of {total_file_count} not loaded due to empty data.')

    df.to_csv('datasets/us-stocks-ETF_USD-daily_data.csv')

    return pd.concat(all_dfs)


_loaders = {
    'btc': _load_btc_bitstamp,
    'securities': _load_securities_us,
}
_filenames = {
    'btc': 'datasets/bitstampUSD_1-min_data_2012-01-01_to_2021-03-31.csv',
    'securities': 'datasets/us-stocks-ETF_USD-daily_data.csv',
}


def load_data(dataset: str, reload: bool = False) -> pd.DataFrame:
    """Load data with fixed index

    :param dataset: `btc` or `securitiess`
    :param reload: set to True to force reloading
    :return: a pandas Dataframe object
    """

    filepath = _filenames[dataset]
    if reload or not os.path.exists(filepath):
        return _loaders[dataset]()
    return pd.read_csv(filepath)
