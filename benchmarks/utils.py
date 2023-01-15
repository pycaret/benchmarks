"""Utility functions"""
import logging
import re
from enum import Enum, auto
from itertools import product
from typing import List, Optional, Tuple

import pycaret
from pycaret.containers.models import get_all_ts_model_containers
from pycaret.containers.models.time_series import ALL_ALLOWED_ENGINES
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


def return_pycaret_model_engine_names() -> List[Tuple[str, str]]:
    """Return all model names in PyCaret along with their supported engines.

    Returns
    -------
    List[Tuple[str, str]]
        List of tuples of the form (model_name, engine_name)
        If a model supports multiple engines, then there will be multiple tuples
        for that model.
    """
    data = get_data("airline", verbose=False)
    exp = TSForecastingExperiment()
    exp.setup(data=data, session_id=42, verbose=False)
    model_containers = get_all_ts_model_containers(exp)
    all_models = model_containers.keys()

    all_model_engines = ALL_ALLOWED_ENGINES
    return_list = []
    for model in all_models:
        model_engines = all_model_engines.get(model, ["None"])
        for m, e in product([model], model_engines):
            return_list.append((m, e))
    return return_list


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


def _return_pycaret_version_or_hash() -> str:
    """Returns the pycaret version

    If pycaret is installed using pip, else returns the git hash if it is
    installed from git.

    Returns
    -------
    str
        Pycaret version or git hash
    """
    try:
        from pip._internal.operations import freeze
    except ImportError:
        from pip.operations import freeze

    pkgs = " ".join(freeze.freeze())
    match = re.search(r"git\+https://github\.com/pycaret/pycaret@([^#]*)", pkgs)
    if match is None:
        logging.info("Pycaret has been installed using pip. Returning version.")
        PYCARET_VERSION = pycaret.__version__
    else:
        logging.info("Pycaret has been installed from git. Returning git hash.")
        PYCARET_VERSION = match.group(1).split()[0]

    return PYCARET_VERSION


def _get_qualified_model_engine(model: str, model_engine: Optional[str]) -> str:
    """Returns the model engine for the model based on the setup kwargs.

    When the model engine is not specified in the setup kwargs, the model engine
    defaults to None. In such cases, this function returns the model engine name
    based on the internal logic of PyCaret.

    Parameters
    ----------
    model : str
        The model whose engine has to be returned.
    model_engine : Optional[str]
        The model engine to use

    Returns
    -------
    str
        The model engine name
    """
    dummy = get_data("airline", verbose=False)
    exp = TSForecastingExperiment()
    setup_kwargs = {"engine": {model: model_engine}} if model_engine else {}
    exp.setup(data=dummy, verbose=False, **setup_kwargs)
    model_engine = exp.get_engine(model)
    return model_engine
