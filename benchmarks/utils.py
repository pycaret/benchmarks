"""Utility functions"""
from enum import Enum, auto

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
