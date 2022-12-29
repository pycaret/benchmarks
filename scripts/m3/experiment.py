"""
Execution command (examples):
>>> python scripts/m3/experiment.py --help
>>> python scripts/m3/experiment.py --execution_mode=native --engine=local --ts_category=Other
>>> python scripts/m3/experiment.py --execution_mode=native --engine=ray --ts_category=Other
>>> python scripts/m3/experiment.py --execution_mode=fugue --engine=local --ts_category=Other
>>> python scripts/m3/experiment.py --execution_mode=fugue --engine=ray --ts_category=Other
"""
import multiprocessing as mp
import os
from datetime import date
from typing import Optional

import fire

# import numpy as np
import pandas as pd
import pycaret
from tqdm import tqdm

from benchmarks.datasets.create.time_series.m3 import get_data
from benchmarks.parallel.execution import (
    execute,
    initialize_engine,
    run_checks,
    shutdown_engine,
)
from benchmarks.parallel.time_series.single_ts import forecast_create_model
from benchmarks.utils import return_dirs

# Register `pandas.progress_apply` and `pandas.Series.map_apply` with `tqdm`
tqdm.pandas()


def main(
    dataset: str = "M3",
    ts_category: str = "Other",
    model: str = "ets",
    execution_mode: str = "native",
    engine: str = "ray",
    num_cpus: Optional[int] = None,
) -> None:
    """Benchmark the performance of a single model across multiple individual
    time series.

    Parameters
    ----------
    dataset : str, optional
        Dataset to benchmark, by default "M3"
        NOTE: Currently only M3 is supported
    ts_category : str, optional
        Options: "Yearly", "Quarterly", "Monthly", "Other", by default "Other"
    model : str, optional
        The model name to pass to pycaret's create_model in order to benchmark,
        by default "ets". Refer to pycaret's documentation for more information.
    execution_mode : str, optional
        Should the execution be done natively or using the Fugue wrapper
        Options: "native", "fugue", by default "native"
    engine : str, optional
        What engine should be used.
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

    run_checks(execution_mode, engine)

    num_cpus = num_cpus or mp.cpu_count()
    print(
        f"Running benchmark for Dataset: '{dataset}' Category: '{ts_category}' "
        f"Model: '{model}' using ..."
        f"\n  - Engine: '{engine}'"
        f"\n  - Execution Mode: '{execution_mode}'"
        f"\n  - CPUs: {num_cpus}"
    )

    initialize_engine(engine, num_cpus)

    # Get the data ----
    BASE_DIR, FORECAST_DIR, TIME_DIR = return_dirs(dataset=dataset)

    # Check if the directory exists
    for dir in [FORECAST_DIR, TIME_DIR]:
        if not os.path.exists(dir):
            # Create the directory
            os.makedirs(dir)

    train, fh, _, _ = get_data(directory=BASE_DIR, dataset=dataset, group=ts_category)
    test, _, _, _ = get_data(
        directory=BASE_DIR, dataset=dataset, group=ts_category, train=False
    )

    # Uncomment after this is implementedhttps://github.com/pycaret/pycaret/issues/3202
    # We only need the y_test time points. y_test values should be unknown to
    # avoid any chance of leakage
    # test["y"] = np.nan

    combined = pd.concat([train, test], axis=0)
    combined["ds"] = pd.to_datetime(combined["ds"])

    prefix = f"{dataset}-{ts_category}-{model}-{engine}-{execution_mode}"

    # # For local testing on a small subset ----
    # all_ts = combined["unique_id"].unique()
    # combined = combined[combined["unique_id"].isin(all_ts[:2])]

    RUN_DATE = date.today().strftime("%Y-%m-%d")
    PYCARET_VERSION = pycaret.__version__

    verbose = False
    cross_validate = False
    apply_kwargs = dict(
        prefix=prefix,
        setup_kwargs={
            "fh": fh,
            "target": "y",
            "index": "ds",
            "fold": 1,
            # "numeric_imputation_target": "ffill",
            "ignore_features": ["unique_id"],
            "n_jobs": 1,
            "session_id": 42,
            "verbose": verbose,
        },
        create_model_kwargs={
            "estimator": model,
            "cross_validation": cross_validate,
            "verbose": verbose,
        },
        backup_model_kwargs={
            "estimator": "naive",
            "cross_validation": cross_validate,
            "verbose": verbose,
        },
    )

    # Fugue required the schema of the returned dataframe ----
    if execution_mode == "fugue":
        schema = "unique_id:str, ds:date, y_pred:float, model_name:str, model:str"
    else:
        schema = None

    test_results, time_taken = execute(
        all_groups=combined,
        keys="unique_id",
        function_single_group=forecast_create_model,
        function_kwargs=apply_kwargs,
        execution_mode=execution_mode,
        engine=engine,
        num_cpus=num_cpus,
        schema=schema,
    )

    # Order columns (as different engines may have different orders) ----
    cols = ["unique_id", "ds", "y_pred", "model_name", "model"]
    test_results = test_results[cols]

    # Write results ----
    test_results.to_csv(f"{FORECAST_DIR}/forecasts-{prefix}.csv", index=False)
    time_df = pd.DataFrame(
        {
            "time": [time_taken],
            "dataset": [dataset],
            "group": [ts_category],
            "model": [model],
            "engine": [engine],
            "execution_mode": [execution_mode],
            "run_date": [RUN_DATE],
            "pycaret_version": [PYCARET_VERSION],
        }
    )
    time_df.to_csv(f"{TIME_DIR}/time-{prefix}.csv", index=False)

    shutdown_engine(engine)

    print("\nBenchmark Complete!")


if __name__ == "__main__":
    fire.Fire(main)
