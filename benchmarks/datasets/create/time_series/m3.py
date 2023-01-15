"""Ref: https://github.com/Nixtla/statsforecast/tree/main/experiments/m3/src"""

import logging
from typing import Tuple

import numpy as np
import pandas as pd
from datasetsforecast.losses import mape, smape
from datasetsforecast.m3 import M3, M3Info

from benchmarks.utils import return_dirs

logging.basicConfig(level=logging.INFO, format="%(asctime)s: %(message)s")

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
        Time Series Category name.
        Allowed values: 'Yearly', 'Quarterly', 'Monthly', 'Other'.
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
    """Save the dataset to a csv file.

    Parameters
    ----------
    dataset : str
        'M3', M4, etc
    group : str
        Time Series Category name.
        Allowed values: 'Yearly', 'Quarterly', 'Monthly', 'Other'.
    train : bool, optional
        Save the training dataset if True, else the Test dataset,
        by default True
    """
    dataset_ = dataset.lower()
    BASE_DIR = f"data/{dataset_}"
    df, *_ = get_data(BASE_DIR, dataset, group, train)
    if train:
        df.to_csv(f"{BASE_DIR}/{dataset}-{group}.csv", index=False)
    else:
        df.to_csv(f"{BASE_DIR}/{dataset}-{group}-test.csv", index=False)


def evaluate(
    library: str,
    dataset: str,
    group: str,
    model: str,
    model_engine: str,
    execution_engine: str,
    execution_mode: str,
) -> pd.DataFrame:
    """Evaluate the results of a model across a dataset.

    Parameters
    ----------
    library : str
        The library to evaluate, e.g. 'pycaret'
    dataset : str
        'M3' only
    group : str
        Time Series Category name.
        Allowed values: 'Yearly', 'Quarterly', 'Monthly', 'Other'.
    model : str
        Name of the model to evaluate.
    model_engine : str
        Name of the model engine to evaluate.
        e.g. for model = "auto_arima", model_engine can be "pmdarima"
    execution_engine : str, optional
        Evaluate the model based on which engine
        Options: "local", "ray", "spark", by default "ray"
    execution_mode : str, optional
        Evaluate the model based on which execution mode
        Options: "native", "fugue"

    Returns
    -------
    pd.DataFrame
        Dataframe showing the evaluation metrics along with execution times.
    """
    BASE_DIR, FORECAST_DIR, TIME_DIR = return_dirs(dataset=dataset)

    suffix = (
        f"{library}-{dataset}-{group}-"
        f"{model}-{model_engine}-"
        f"{execution_engine}-{execution_mode}"
    )
    logging.info(suffix)
    y_test, horizon, _, _ = get_data(BASE_DIR, dataset, group, False)
    count_ts = len(y_test) / horizon

    primary_model_per = np.nan
    backup_model_per = np.nan
    no_model_per = np.nan
    try:
        forecast = pd.read_csv(f"{FORECAST_DIR}/forecasts-{suffix}.csv")
        no_model_per = forecast["model_name"].isna().sum() / len(forecast) * 100
        primary_model_per = (
            len(forecast.query("model_name == @model")) / len(forecast) * 100
        )
        backup_model_per = 100 - primary_model_per - no_model_per
    except FileNotFoundError:
        forecast = pd.DataFrame(
            columns=["unique_id", "ds", "y_pred", "model_name", "model"]
        )

    all_models = forecast["model_name"].unique().tolist()
    backup_model = [
        candidate_bk_model
        for candidate_bk_model in all_models
        if candidate_bk_model != model
    ] or [""]

    stats = pd.DataFrame(
        {
            "count_ts": [count_ts],
            "primary_model_per": primary_model_per,
            "backup_model_per": backup_model_per,
            "no_model_per": no_model_per,
            "backup_model": backup_model,
        }
    )

    # Just to be double sure
    selected_cols = ["unique_id", "ds", "y_pred"]
    forecast = forecast.query("model_name == @model")[selected_cols]
    forecast["ds"] = pd.to_datetime(forecast["ds"])

    if group == "Monthly":
        # Remove day since one can have Month Start and one can have Month End
        # This is due to internal coercing in PyCaret
        forecast["ds"] = forecast["ds"].dt.strftime("%Y-%m")
        y_test["ds"] = y_test["ds"].dt.strftime("%Y-%m")
    elif group == "Quarterly":
        # Remove day-month since one can have Quarter Start and one can have Quarter End
        # This is due to internal coercing in PyCaret
        forecast["ds"] = (
            forecast["ds"].dt.year.astype(str)
            + "-"
            + forecast["ds"].dt.quarter.astype(str)
        )
        y_test["ds"] = (
            y_test["ds"].dt.year.astype(str) + "-" + y_test["ds"].dt.quarter.astype(str)
        )
    if group == "Yearly":
        # Remove day-month since one can have Year Start and one can have Year End
        # This is due to internal coercing in PyCaret
        forecast["ds"] = forecast["ds"].dt.strftime("%Y")
        y_test["ds"] = y_test["ds"].dt.strftime("%Y")

    combined = pd.merge(y_test, forecast, on=["unique_id", "ds"], how="left")
    y_test = combined["y"].values.reshape(-1, horizon)
    y_hat = combined["y_pred"].values.reshape(-1, horizon)

    evaluations = {}
    for metric in (mape, smape):
        metric_name = metric.__name__
        loss = metric(y_test, y_hat)
        evaluations[metric_name] = loss

    evaluations = pd.DataFrame(evaluations, index=[0])

    try:
        times = pd.read_csv(f"{TIME_DIR}/time-{suffix}.csv")
    except FileNotFoundError:
        times = pd.DataFrame(
            {
                "os": [None],
                "python_version": [None],
                "library": [library],
                "library_version": [None],
                "dataset": dataset,
                "group": group,
                "model": model,
                "model_engine": model_engine,
                "engine": execution_engine,
                "execution_mode": execution_mode,
                "num_cpus": [None],
                "time": [0],
            }
        )

    evaluations = pd.concat([evaluations, times, stats], axis=1)

    return evaluations
