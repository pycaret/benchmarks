"""Utility functions"""
from enum import Enum, auto
from typing import Tuple

from pycaret.containers.models import get_all_ts_model_containers
from pycaret.datasets import get_data
from pycaret.time_series import TSForecastingExperiment


class Engine(Enum):
    """The execution engine to use."""

    local = auto()
    ray = auto()
    spark = auto()


class ExecutionMode(Enum):
    """The execution mode to use."""

    native = auto()
    fugue = auto()


def check_allowed_types(some_string: str, enum_class: Enum) -> bool:
    """Checks if a string is in an Enumeration Class

    Parameters
    ----------
    some_string : str
        String that needs to be checked
    enum_class : Enum
        Enumeration Class that needs to be searched

    Returns
    -------
    bool
        True if the string is defined in the enumeration class, else False
    """
    return any([some_string in enum.name for enum in list(enum_class)])


def return_pycaret_model_names():
    """Return all model names in PyCaret."""
    data = get_data("airline", verbose=False)
    exp = TSForecastingExperiment()
    exp.setup(data=data, session_id=42, verbose=False)
    model_containers = get_all_ts_model_containers(exp)
    return list(model_containers.keys())


def return_dirs(dataset: str) -> Tuple[str, str, str]:
    """Return the directories to use for the dataset.

    Parameters
    ----------
    dataset : str
        Dataset to benchmark, e.g. "M3"

    Returns
    -------
    Tuple[str, str, str, str]
        (1) Base Directory: location of the dataset and results
        (2) Forecasts Directory: location of forecasts
        (3) Time Directory: location of time metrics
    """
    dataset_ = dataset.lower()
    BASE_DIR = "data"
    RESULTS_DIR = f"{BASE_DIR}/{dataset_}/results"
    FORECAST_DIR = f"{RESULTS_DIR}/forecasts"
    TIME_DIR = f"{RESULTS_DIR}/time"
    return BASE_DIR, FORECAST_DIR, TIME_DIR
