"""
Execution command (examples):
>>> python scripts/mseries/extract_properties.py --help
>>> python scripts/mseries/extract_properties.py --dataset=M3 --group=Other --execution_mode=native --execution_engine=ray
"""

import logging
import multiprocessing as mp
import os
import sys
from datetime import date
from typing import Optional

import fire
import numpy as np
import pandas as pd

from benchmarks.datasets.create.time_series.mseries import get_data
from benchmarks.parallel.execution import (
    execute,
    initialize_engine,
    run_execution_checks,
    shutdown_engine,
)
from benchmarks.parallel.time_series.properties import extract_properties
from benchmarks.utils import _return_dirs, _return_pycaret_version_or_hash

logging.basicConfig(level=logging.INFO, format="%(asctime)s: %(message)s")


def main(
    dataset: str = "M3",
    group: str = "Other",
    execution_mode: str = "native",
    execution_engine: str = "ray",
    num_cpus: Optional[int] = None,
) -> None:
    """Benchmark the performance of a single model across multiple individual
    time series.

    Parameters
    ----------
    dataset : str, optional
        Dataset to benchmark, by default "M3"
        NOTE: Currently only M3 is supported
    group : str, optional
        Options: "Yearly", "Quarterly", "Monthly", "Other", by default "Other"
    execution_mode : str, optional
        Should the execution be done natively or using the Fugue wrapper
        Options: "native", "fugue", by default "native"
    execution_engine : str, optional
        What engine should be used for executing the code.
        Options: "local", "ray", "spark", by default "ray"
            - "local" will execute serially using pandas
            - "ray" will execute in parallel using Ray
            - "spark" will execute in parallel using Spark
        NOTE: Currently only "local" and "ray" are supported
    num_cpus : Optional[int], optional
        Number of CPUs to use to execute in parallel, by default None which uses
        up all available CPUs. In local mode, this is ignored and only 1 CPU is
        used .

    Raises
    ------
    ValueError
        (1) execution_mode is not one of "native" or "fugue"
        (2) engine is not one of "local", "ray", or "spark"
        (3) engine is not implemented
    """

    EXEC_MODE_VERSION, EXEC_ENGINE_VERSION = run_execution_checks(
        execution_mode, execution_engine
    )

    logging.info("\n\n")
    LIBRARY = "pycaret"
    LIBRARY_VERSION = _return_pycaret_version_or_hash()
    OS = sys.platform
    PYTHON_VERSION = sys.version.split()[0]
    RUN_DATE = date.today().strftime("%Y-%m-%d")
    num_cpus = num_cpus or mp.cpu_count()
    logging.info(
        f"\nRun Date: {RUN_DATE}"
        f"\nExtracting features for Dataset: '{dataset}' Group: '{group}' using ..."
        f"\n  - OS: '{OS}'"
        f"\n  - Python Version: '{PYTHON_VERSION}'"
        f"\n  - Library: '{LIBRARY}' Version: '{LIBRARY_VERSION}'"
        f"\n  - Execution Engine: '{execution_engine}' Version: '{EXEC_ENGINE_VERSION}'"
        f"\n  - Execution Mode: '{execution_mode}' Version: '{EXEC_MODE_VERSION}'"
        f"\n  - CPUs: {num_cpus}"
    )

    keys = [
        dataset,
        group,
        LIBRARY,
        LIBRARY_VERSION,
        execution_engine,
        EXEC_ENGINE_VERSION,
        execution_mode,
        EXEC_MODE_VERSION,
        num_cpus,
        PYTHON_VERSION,
        OS,
    ]

    prefix = "-".join([str(key) for key in keys])

    # -------------------------------------------------------------------------#
    # Get the data ----
    # -------------------------------------------------------------------------#
    BASE_DIR, _, _, PROPERTIES_DIR = _return_dirs(dataset=dataset)

    # Check if the directory exists. If not, create it.
    for dir in [PROPERTIES_DIR]:
        if not os.path.exists(dir):
            os.makedirs(dir)

    train, fh, _, _ = get_data(directory=BASE_DIR, dataset=dataset, group=group)
    test, _, _, _ = get_data(
        directory=BASE_DIR, dataset=dataset, group=group, train=False
    )

    # We only need the y_test time points. y_test values should be unknown to
    # avoid any chance of leakage
    test["y"] = np.nan

    combined = pd.concat([train, test], axis=0)
    combined["ds"] = pd.to_datetime(combined["ds"])

    # # For local testing on a small subset ----
    # all_ts = combined["unique_id"].unique()
    # combined = combined[combined["unique_id"].isin(all_ts[:2])]

    # -------------------------------------------------------------------------#
    # Experiment Settings ----
    # NOTE:
    # (1) Test only has time points. No values. This is to avoid leakage.
    # (2) Only use the train portion to extract features since test will be imputed.
    # -------------------------------------------------------------------------#

    verbose = False
    setup_kwargs = {
        "fh": fh,
        "target": "y",
        "index": "ds",
        "fold": 1,
        "numeric_imputation_target": "ffill",
        "hyperparameter_split": "train",
        "ignore_features": ["unique_id"],
        "n_jobs": 1,
        "session_id": 42,
        "verbose": verbose,
    }
    apply_kwargs = dict(prefix=prefix, setup_kwargs=setup_kwargs)

    # Fugue required the schema of the returned dataframe ----
    if execution_mode == "fugue":
        schema = (
            "len_total : int"
            ",len_train : int"
            ",len_test : int"
            ",strictly_positive : bool"
            ",white_noise : str"
            ",lowercase_d : int"
            ",uppercase_d : int"
            ",seasonality_present : bool"
            ",seasonality_type: str"
            ",candidate_sps : list"
            ",significant_sps : list"
            ",all_sps : list"
            ",primary_sp : int"
            ",significant_sps_no_harmonics : list"
            ",all_sps_no_harmonics : list"
            ",primary_sp_no_harmonics : int"
            ",time_taken : float"
        )
    else:
        schema = None

    # -------------------------------------------------------------------------#
    # Run Benchmarking ----
    # -------------------------------------------------------------------------#
    initialize_engine(execution_engine, num_cpus)
    properties, time_taken = execute(
        all_groups=combined,
        keys="unique_id",
        function_single_group=extract_properties,
        function_kwargs=apply_kwargs,
        execution_mode=execution_mode,
        execution_engine=execution_engine,
        num_cpus=num_cpus,
        schema=schema,
    )
    shutdown_engine(execution_engine)

    # -------------------------------------------------------------------------#
    # Write Results ----
    # -------------------------------------------------------------------------#

    # Order columns (as different execution engines may have different orders) ----
    cols = [
        "unique_id",
        "len_total",
        "len_train",
        "len_test",
        "strictly_positive",
        "white_noise",
        "lowercase_d",
        "uppercase_d",
        "seasonality_present",
        "seasonality_type",
        "candidate_sps",
        "significant_sps",
        "all_sps",
        "primary_sp",
        "significant_sps_no_harmonics",
        "all_sps_no_harmonics",
        "primary_sp_no_harmonics",
        "time_taken",
    ]
    properties = properties[cols]

    # Write results ----
    properties_file_name = f"{PROPERTIES_DIR}/properties-{prefix}.csv"
    logging.info(f"Writing properties to {properties_file_name}")
    properties.to_csv(properties_file_name, index=False)

    # Add all KEY_COLS along with Non Static columns like time and run date ----
    time_df = pd.DataFrame(
        {
            "dataset": [dataset],
            "group": [group],
            "library": [LIBRARY],
            "library_version": [LIBRARY_VERSION],
            "execution_engine": [execution_engine],
            "execution_engine_version": [EXEC_ENGINE_VERSION],
            "execution_mode": [execution_mode],
            "execution_mode_version": [EXEC_MODE_VERSION],
            "num_cpus": [num_cpus],
            "python_version": [PYTHON_VERSION],
            "os": [OS],
            "time": [time_taken],
            "run_date": [RUN_DATE],
        }
    )
    time_file_name = f"{PROPERTIES_DIR}/time-{prefix}.csv"
    logging.info(f"Writing time to {time_file_name}")
    time_df.to_csv(time_file_name, index=False)

    logging.info("Property Extraction Complete!")


if __name__ == "__main__":
    fire.Fire(main)
