"""
Execution command (examples):
>>> python scripts/experiment_m3.py --help
>>> python scripts/experiment_m3.py --execution_mode=native --engine=local --ts_category=Other
>>> python scripts/experiment_m3.py --execution_mode=native --engine=ray --ts_category=Other
>>> python scripts/experiment_m3.py --execution_mode=fugue --engine=ray --ts_category=Other
>>> python scripts/experiment_m3.py --execution_mode=fugue --engine=ray --ts_category=Other
"""

import multiprocessing as mp
import time

import fire
import pandas as pd
import ray
from fugue import transform
from tqdm import tqdm

from benchmarks.datasets.create.time_series.m3 import get_data
from benchmarks.parallel.time_series.single_ts import forecast_create_model
from benchmarks.utils import Engine, ExecutionMode, check_allowed_types

# Register `pandas.progress_apply` and `pandas.Series.map_apply` with `tqdm`
tqdm.pandas()


def main(
    dataset: str = "M3",
    ts_category: str = "Other",
    model: str = "ets",
    execution_mode: str = "native",
    engine: str = "ray",
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

    Raises
    ------
    ValueError
        (1) execution_mode is not one of "native" or "fugue"
        (2) engine is not one of "local", "ray", or "spark"
        (3) engine is not implemented
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

    # Set variables based on the execution mode and engine ----
    if engine == "local":
        if execution_mode == "fugue":
            fugue_engine = None
            as_local = True
    elif engine == "ray":
        if execution_mode == "fugue":
            fugue_engine = "ray"
            as_local = True
        num_cpus = mp.cpu_count()
        ray.init(num_cpus=num_cpus)
    elif engine == "spark":
        raise ValueError("Spark is not supported right now.")

    # Directory where the data is stored ----
    directory = "data/"

    # # "Other" group since it has the least number of time series
    # ts_categories = ts_categories[-1:]
    # for ts_category in tqdm(ts_categories):
    train, fh, _, _ = get_data(directory=directory, dataset=dataset, group=ts_category)
    test, _, _, _ = get_data(
        directory=directory, dataset=dataset, group=ts_category, train=False
    )

    combined = pd.concat([train, test], axis=0)
    combined["ds"] = pd.to_datetime(combined["ds"])

    prefix = f"{dataset}-{ts_category}-{model}-{engine}-{execution_mode}"

    # # For local testing on a small subset ----
    # all_ts = combined["unique_id"].unique()
    # combined = combined[combined["unique_id"].isin(all_ts[:2])]

    apply_kwargs = dict(
        fh=fh,
        target="y",
        index="ds",
        prefix=prefix,
        fold=1,
        ignore_features=["unique_id"],
        n_jobs=1,
        session_id=42,
        verbose=False,
        create_model_kwargs={"estimator": model, "cross_validation": False},
        backup_model_kwargs={"estimator": "naive", "cross_validation": False},
    )

    start = time.time()
    if execution_mode == "native":
        grouped_data = combined.groupby("unique_id")
        if engine == "local":
            test_results = grouped_data.progress_apply(
                forecast_create_model, **apply_kwargs
            )
        elif engine == "ray":
            test_results = []
            function_remote = ray.remote(forecast_create_model)
            for single_group in grouped_data.groups.keys():
                result_single_group = function_remote.remote(
                    data=grouped_data.get_group(single_group), **apply_kwargs
                )
                test_results.append(result_single_group)
            test_results = ray.get(test_results)
            # Combine all results into 1 dataframe
            test_results = pd.concat(test_results)
    elif execution_mode == "fugue":
        schema = "unique_id:str, ds:date, y_pred:float, model_name:str, model:str"
        test_results = transform(
            combined,
            forecast_create_model,
            params=apply_kwargs,
            schema=schema,
            partition={"by": "unique_id"},
            engine=fugue_engine,
            as_local=as_local,
        )
    end = time.time()
    time_taken = round(end - start)
    print(f"Total time taken for category '{ts_category}': {time_taken}s")

    # Order columns (as different engines may have different orders) ----
    cols = ["unique_id", "ds", "y_pred", "model_name", "model"]
    test_results = test_results[cols]

    # Write results ----
    test_results.to_csv(f"data/forecasts-{prefix}.csv", index=False)
    time_df = pd.DataFrame(
        {
            "time": [time_taken],
            "dataset": [dataset],
            "group": [ts_category],
            "model": [model],
            "engine": [engine],
            "execution_mode": [execution_mode],
        }
    )
    time_df.to_csv(f"data/time-{prefix}.csv", index=False)

    if engine == "ray":
        ray.shutdown()


if __name__ == "__main__":
    fire.Fire(main)
