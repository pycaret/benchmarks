"""Module to assist with the execution of parallel functions."""

import time
from typing import Callable, Optional, Tuple

import pandas as pd
import ray
from fugue import transform

from benchmarks.utils import Engine, ExecutionMode, check_allowed_types


def execute(
    all_groups: pd.DataFrame,
    keys: str,
    function_single_group: Callable,
    execution_mode: str,
    engine: str,
    num_cpus: int,
    schema: Optional[str] = None,
    **apply_kwargs,
) -> Tuple[pd.DataFrame, float]:
    """Applies a function over each group of a dataframe using specified engine.

    Parameters
    ----------
    all_groups : pd.DataFrame
        Dataframe containing multiple groups.
    keys : str
        The column name indicating the unique identifier for each group.
    function_single_group : Callable
        The func to apply to each group.
    execution_mode : str, optional
        Should the execution be done natively or using the Fugue wrapper
        Options: "native", "fugue"
    engine : str
        What engine should be used.
        Options: "local", "ray", "spark", by default "ray"
            - "local" will execute serially using pandas
            - "ray" will execute in parallel using Ray
            - "spark" will execute in parallel using Spark
        NOTE: Currently only "local" and "ray" are supported
    num_cpus : int
        Number of CPUs to use to execute in parallel. In local mode, this is
        ignored and only 1 CPU is used .
    schema : Optional[str], optional
        The schema to use when execution_mode = "fugue", by default None

    Returns
    -------
    Tuple[pd.DataFrame, float]
        (1) The results of the function applied to each group
        (2) The time taken to execute the function across all groups

    Raises
    ------
    ValueError
        (1) engine is set to "spark"
    """
    # Set variables based on the execution mode and engine ----
    if engine == "local":
        if execution_mode == "fugue":
            fugue_engine = None
            as_local = True
    elif engine == "ray":
        if execution_mode == "fugue":
            fugue_engine = "ray"
            as_local = True
    elif engine == "spark":
        raise ValueError("Spark is not supported right now.")

    start = time.time()
    if execution_mode == "native":
        grouped_data = all_groups.groupby(keys)
        if engine == "local":
            all_results = grouped_data.progress_apply(
                function_single_group, **apply_kwargs
            )
        elif engine == "ray":
            all_results = []
            function_remote = ray.remote(function_single_group)
            for single_group in grouped_data.groups.keys():
                result_single_group = function_remote.remote(
                    data=grouped_data.get_group(single_group), **apply_kwargs
                )
                all_results.append(result_single_group)
            all_results = ray.get(all_results)
            # Combine all results into 1 dataframe
            all_results = pd.concat(all_results)
    elif execution_mode == "fugue":
        all_results = transform(
            all_groups,
            function_single_group,
            params=apply_kwargs,
            schema=schema,
            # TODO: In future releases of Fugue, we may not have to pass num_cpus.
            partition={"by": keys, "num": num_cpus},
            engine=fugue_engine,
            as_local=as_local,
        )
    end = time.time()
    time_taken = round(end - start)

    return all_results, time_taken


def run_checks(execution_mode: str, engine: str):
    """Checks that the execution model and engine are of the allowed types.

    Parameters
    ----------
    execution_mode : str
        Should the execution be done natively or using the Fugue wrapper
        Options: "native", "fugue"
    engine : str
        What engine should be used.
        Options: "local", "ray", "spark"

    Raises
    ------
    ValueError
        (1) execution_mode is not one of "native" or "fugue"
        (2) engine is not one of "local", "ray", or "spark"
    """
    # Check that the execution mode and engine are supported ----
    if not check_allowed_types(execution_mode, ExecutionMode):
        raise ValueError(
            f"Execution Mode '{execution_mode}' not supported. "
            f"Please choose from {list(ExecutionMode.__members__.keys())}."
        )
    if not check_allowed_types(engine, Engine):
        raise ValueError(
            f"Engine '{engine}' not supported. "
            f"Please choose from {list(Engine.__members__.keys())}."
        )


def initialize_engine(engine: str, num_cpus: int):
    """Initialized the parallel engine.

    Parameters
    ----------
    engine : str
        The engine to initialize
    num_cpus : int
        Number of CPUs to use if engine supports parallelization.
    """
    # Initialize parallel engines ----
    if engine == "ray":
        ray.init(num_cpus=num_cpus)


def shutdown_engine(engine: str):
    """Shutdown the engine.

    Parameters
    ----------
    engine : str
        The engine to shutdown.
    """
    # Shutdown parallel engines ----
    if engine == "ray":
        ray.shutdown()