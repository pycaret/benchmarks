"""
To Run
>>> python scripts\m3\evaluation.py
"""
import logging
from itertools import product

import numpy as np
import pandas as pd

from benchmarks.datasets.create.time_series.m3 import evaluate
from benchmarks.utils import return_pycaret_model_engine_names

logging.basicConfig(level=logging.INFO, format="%(asctime)s: %(message)s")

if __name__ == "__main__":
    library = "pycaret"
    groups = ["Yearly", "Quarterly", "Monthly", "Other"]
    model_and_engines = return_pycaret_model_engine_names()
    datasets = ["M3"]
    # All dataset evaluations are stored in the same BASE_DIR
    BASE_DIR = "data"

    evaluation = [
        evaluate(
            library=library,
            dataset=dataset,
            group=group,
            model=model,
            model_engine=model_engine,
            execution_engine="ray",
            execution_mode="native",
        )
        for (model, model_engine), group in product(model_and_engines, groups)
        for dataset in datasets
    ]
    evaluation = [eval_ for eval_ in evaluation if eval_ is not None]
    evaluation = pd.concat(evaluation)
    evaluation = evaluation[
        [
            "os",
            "python_version",
            "library",
            "library_version",
            "dataset",
            "group",
            "model",
            "model_engine",
            "execution_engine",
            "execution_engine_version",
            "execution_mode",
            "execution_mode_version",
            "run_date",
            "count_ts",
            "primary_model_per",
            "backup_model_per",
            "no_model_per",
            "backup_model",
            "mape",
            "smape",
            "num_cpus",
            "time",
        ]
    ]
    evaluation["time"] /= 60  # minutes
    evaluation = (
        evaluation.set_index(
            [
                "os",
                "python_version",
                "library",
                "library_version",
                "dataset",
                "group",
                "model",
                "model_engine",
                "execution_engine",
                "execution_engine_version",
                "execution_mode",
                "execution_mode_version",
                "run_date",
                "count_ts",
                "primary_model_per",
                "backup_model_per",
                "no_model_per",
                "backup_model",
            ]
        )
        .stack()
        .reset_index()
    )
    evaluation.columns = [
        "os",
        "python_version",
        "library",
        "library_version",
        "dataset",
        "group",
        "model",
        "model_engine",
        "execution_engine",
        "execution_engine_version",
        "execution_mode",
        "execution_mode_version",
        "run_date",
        "count_ts",
        "primary_model_per",
        "backup_model_per",
        "no_model_per",
        "backup_model",
        "metric",
        "val",
    ]
    evaluation = (
        evaluation.set_index(
            [
                "os",
                "python_version",
                "library",
                "library_version",
                "dataset",
                "group",
                "model",
                "model_engine",
                "execution_engine",
                "execution_engine_version",
                "execution_mode",
                "execution_mode_version",
                "run_date",
                "count_ts",
                "primary_model_per",
                "backup_model_per",
                "no_model_per",
                "backup_model",
                "metric",
            ]
        )
        .unstack()
        .round(3)
    )
    evaluation = evaluation.droplevel(0, 1).reset_index()
    cols_to_clean = ["mape", "smape", "time"]
    evaluation[cols_to_clean] = evaluation[cols_to_clean].replace(0, np.nan)

    cols_to_round = ["primary_model_per", "backup_model_per", "no_model_per"]
    evaluation[cols_to_round] = evaluation[cols_to_round].round(4)

    eval_file_name = f"{BASE_DIR}/evaluation_full.csv"
    logging.info(f"\nWriting full evaluation results to {eval_file_name}")
    evaluation.to_csv(eval_file_name, index=False)

    cols_to_drop = ["run_date", "num_cpus", "time"]
    eval_file_name = f"{BASE_DIR}/evaluation_static.csv"
    logging.info(f"\nWriting static evaluation results to {eval_file_name}")
    evaluation.drop(columns=cols_to_drop).to_csv(eval_file_name, index=False)

    logging.info("\nEvaluation Complete!")
