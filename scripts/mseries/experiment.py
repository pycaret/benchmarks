"""
Execution command (examples):
>>> python scripts/mseries/experiment.py --help
>>> python scripts/mseries/experiment.py --model=auto_arima --model_engine=pmdarima --execution_mode=native --execution_engine=local --group=Other
>>> python scripts/mseries/experiment.py --model=auto_arima --model_engine=pmdarima --execution_mode=native --execution_engine=ray --group=Other
>>> python scripts/mseries/experiment.py --model=auto_arima --model_engine=pmdarima --execution_mode=fugue --execution_engine=local --group=Other
>>> python scripts/mseries/experiment.py --model=auto_arima --model_engine=pmdarima --execution_mode=fugue --execution_engine=ray --group=Other
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
from benchmarks.parallel.time_series.forecast_single_ts import forecast_create_model
from benchmarks.utils import (
    _get_qualified_model_engine,
    _impute_time_series_model_engine,
    _return_dirs,
    _return_pycaret_version_or_hash,
    _try_import_and_get_module_version,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s: %(message)s")


def main(
    dataset: str = "M3",
    group: str = "Other",
    model: str = "ets",
    model_engine: Optional[str] = None,
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
    model : str, optional
        The model name to pass to pycaret's create_model in order to benchmark,
        by default "ets". Refer to pycaret's documentation for more information.
    model_engine : Optional[str]
        Name of the model engine to evaluate.
        e.g. for model = "auto_arima", model_engine can be "pmdarima",
        by default None which picks the default engine in pycaret.
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
        f"\nRunning benchmark for Dataset: '{dataset}' Group: '{group}' "
        f"Model: '{model}', Model Engine: '{model_engine}' using ..."
        f"\n  - OS: '{OS}'"
        f"\n  - Python Version: '{PYTHON_VERSION}'"
        f"\n  - Library: '{LIBRARY}' Version: '{LIBRARY_VERSION}'"
        f"\n  - Execution Engine: '{execution_engine}' Version: '{EXEC_ENGINE_VERSION}'"
        f"\n  - Execution Mode: '{execution_mode}' Version: '{EXEC_MODE_VERSION}'"
        f"\n  - CPUs: {num_cpus}"
    )

    model_engine = _get_qualified_model_engine(model=model, model_engine=model_engine)
    # Default model engine is assumed to be sktime
    model_engine_imputed = _impute_time_series_model_engine(engine=model_engine)
    model_engine_version = _try_import_and_get_module_version(model_engine_imputed)
    logging.info(
        f"Passed model engine corresponds to '{model_engine_imputed}'"
        f"- Installed version '{model_engine_version}'"
    )

    backup_model = "naive"

    keys = [
        dataset,
        group,
        LIBRARY,
        LIBRARY_VERSION,
        model,
        model_engine_imputed,
        model_engine_version,
        execution_engine,
        EXEC_ENGINE_VERSION,
        execution_mode,
        EXEC_MODE_VERSION,
        num_cpus,
        backup_model,
        PYTHON_VERSION,
        OS,
    ]

    prefix = "-".join([str(key) for key in keys])

    # -------------------------------------------------------------------------#
    # Get the data ----
    # -------------------------------------------------------------------------#
    BASE_DIR, FORECAST_DIR, TIME_DIR, _ = _return_dirs(dataset=dataset)

    # Check if the directory exists. If not, create it.
    for dir in [FORECAST_DIR, TIME_DIR]:
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
    # combined["ds"] = pd.to_datetime(combined["ds"])
    try:
        # If index is of int type, it may be read in as string.
        # Try to convert it back to int.
        if combined["ds"].dtype == "object":
            combined["ds"] = combined["ds"].astype(int)
    except TypeError:
        logging.warn(
            "Tried converting 'ds' string column to int but failed. It will "
            "remain a string and be assumed to be coercible to datetime for "
            "further processing in pycaret."
        )

    # # For local testing on a small subset ----
    # all_ts = combined["unique_id"].unique()
    # combined = combined[combined["unique_id"].isin(all_ts[:2])]

    # -------------------------------------------------------------------------#
    # Experiment Settings ----
    # NOTE:
    # (1) Disable Cross validation to get fastest results for benchmarking.
    # (2) Test only has time points. No values. This is to avoid leakage.
    # (3) Only use the train portion to extract features since test will be imputed.
    # -------------------------------------------------------------------------#

    verbose = False
    cross_validate = False
    setup_kwargs = {
        "fh": fh,
        "target": "y",
        "index": "ds",
        "fold": 1,
        "numeric_imputation_target": "ffill",
        "hyperparameter_split": "train",
        "ignore_features": ["unique_id"],
        "engine": {model: model_engine},  # non-imputed version of model_engine
        "n_jobs": 1,
        "session_id": 42,
        "verbose": verbose,
    }
    create_model_kwargs = {
        "estimator": model,
        "cross_validation": cross_validate,
        "verbose": verbose,
    }
    backup_model_kwargs = {
        "estimator": backup_model,
        "cross_validation": cross_validate,
        "verbose": verbose,
    }
    apply_kwargs = dict(
        prefix=prefix,
        setup_kwargs=setup_kwargs,
        create_model_kwargs=create_model_kwargs,
        backup_model_kwargs=backup_model_kwargs,
    )

    # Fugue required the schema of the returned dataframe ----
    if execution_mode == "fugue":
        schema = "unique_id:str, ds:date, y_pred:float, model_name:str, model_engine:str, model:str"
    else:
        schema = None

    # -------------------------------------------------------------------------#
    # Run Benchmarking ----
    # -------------------------------------------------------------------------#
    initialize_engine(execution_engine, num_cpus)
    test_results, time_taken = execute(
        all_groups=combined,
        keys="unique_id",
        function_single_group=forecast_create_model,
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
    cols = ["unique_id", "ds", "y_pred", "model_name", "model_engine", "model"]
    test_results = test_results[cols]

    # Write results ----
    result_file_name = f"{FORECAST_DIR}/forecasts-{prefix}.csv"
    logging.info(f"Writing results to {result_file_name}")
    test_results.to_csv(result_file_name, index=False)

    # Add all KEY_COLS along with Non Static columns like time and run date ----
    time_df = pd.DataFrame(
        {
            "dataset": [dataset],
            "group": [group],
            "library": [LIBRARY],
            "library_version": [LIBRARY_VERSION],
            "model": [model],
            "model_engine": [model_engine_imputed],
            "model_engine_version": [model_engine_version],
            "execution_engine": [execution_engine],
            "execution_engine_version": [EXEC_ENGINE_VERSION],
            "execution_mode": [execution_mode],
            "execution_mode_version": [EXEC_MODE_VERSION],
            "num_cpus": [num_cpus],
            "backup_model": [backup_model],
            "python_version": [PYTHON_VERSION],
            "os": [OS],
            "time": [time_taken],
            "run_date": [RUN_DATE],
        }
    )
    time_file_name = f"{TIME_DIR}/time-{prefix}.csv"
    logging.info(f"Writing time to {time_file_name}")
    time_df.to_csv(time_file_name, index=False)

    logging.info("Benchmark Complete!")


if __name__ == "__main__":
    fire.Fire(main)
