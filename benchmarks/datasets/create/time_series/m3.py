"""
Ref: https://github.com/Nixtla/statsforecast/tree/main/experiments/m3/src
"""

from typing import Tuple
import pandas as pd
from datasetsforecast.m3 import M3, M3Info

dict_datasets = {
    "M3": (M3, M3Info),
}


def get_data(
    directory: str, dataset: str, group: str, train: bool = True
) -> Tuple[pd.DataFrame, int, str, int]:
    """Downloads and loads M3 data.

    Parameters
    ----------
    directory : str
        Directory where data will be downloaded.
    dataset : str
        'M3' only
    group : str
        Group name.
        Allowed groups: 'Yearly', 'Quarterly', 'Monthly', 'Other'.
    train : bool, optional
        Returns the training dataset if True, else the Test dataset,
        by default True

    Returns
    -------
    Tuple[pd.DataFrame, int, str, int]
        dataset (pd.DataFrame):
            Target time series with columns ['unique_id', 'ds', 'y'].
            Train or Test dataset depending on the value of `train`.
        horizon (int):
            Forecast horizon. Should match the length of the Test dataset.
        freq (str):
            Frequency of the time series.
        seasonality (int):
            Seasonality of the time series.

    Raises
    ------
    Exception
        (1) dataset is not equal to 'M3'
        (2) group is not one of the allowed types
    """
    if dataset not in dict_datasets.keys():
        raise Exception(f"dataset {dataset} not found")

    dataclass, data_info = dict_datasets[dataset]
    if group not in data_info.groups:
        raise Exception(f"group {group} not found for {dataset}")

    Y_df, *_ = dataclass.load(directory, group)

    horizon = data_info[group].horizon
    freq = data_info[group].freq
    seasonality = data_info[group].seasonality
    Y_df_test = Y_df.groupby("unique_id").tail(horizon)
    Y_df = Y_df.drop(Y_df_test.index)

    if train:
        return Y_df, horizon, freq, seasonality

    return Y_df_test, horizon, freq, seasonality


def save_data(dataset: str, group: str, train: bool = True):
    df, *_ = get_data("data", dataset, group, train)
    if train:
        df.to_csv(f"data/{dataset}-{group}.csv", index=False)
    else:
        df.to_csv(f"data/{dataset}-{group}-test.csv", index=False)
