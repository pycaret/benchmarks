"""
To Run
>>> python scripts/m3/evaluation.py --dataset="M3"
"""
import logging
import os
from typing import List

import fire
import pandas as pd

from benchmarks.datasets.create.time_series.mseries import evaluate
from benchmarks.utils import KEY_COLS, _return_dirs

logging.basicConfig(level=logging.INFO, format="%(asctime)s: %(message)s")


def _coerce_columns(evaluation: pd.DataFrame, key_cols: List[str]) -> pd.DataFrame:
    """Coerces the columns in the evaluation dataframe

    Key Columns are coerced to object type, while metric columns are coerced to
    float type. If a column cannot be coerced to the desired type, the original
    data type is preserved.

    Parameters
    ----------
    evaluation : pd.DataFrame
        Evaluation data
    key_cols : List[str]
        Key columns

    Returns
    -------
    pd.DataFrame
        Evaluation data with coerced columns
    """
    logging.info("\n")
    for key in key_cols:
        try:
            evaluation[key] = evaluation[key].astype(object)
        except ValueError:
            logging.warning(f"Could not convert key column '{key}' to object.")

    non_keys = [col for col in evaluation.columns if col not in key_cols]
    for non_key in non_keys:
        try:
            evaluation[non_key] = evaluation[non_key].astype(float)
        except ValueError:
            logging.warning(f"Could not convert metric column '{non_key}' to float.")
    logging.info("\n")
    return evaluation


def main(dataset: str = "M3") -> None:
    """Evaluates the results for a particular dataset

    Parameters
    ----------
    dataset : str, optional
        Dataset for which the evaluation needs to be performed, by default "M3"
    """
    BASE_DIR, FORECAST_DIR, TIME_DIR = _return_dirs(dataset=dataset)

    logging.info("\n\n")
    logging.info(
        f"\nRunning evaluation for dataset: '{dataset}'"
        f"\n    Data Directory: '{BASE_DIR}'"
        f"\n    Forecast Directory: '{FORECAST_DIR}'"
        f"\n    Run Time Directory: '{TIME_DIR}'"
    )

    # -------------------------------------------------------------------------#
    # START: Read all available combinations
    # -------------------------------------------------------------------------#
    # Read combinations from both forecasts and time files
    all_combinations = []
    for filename in os.listdir(FORECAST_DIR):
        if filename.endswith(".csv"):
            name_no_ext = os.path.splitext(filename)[0]
            combination = name_no_ext.split("forecasts-")[1]
            all_combinations.append(combination)

    for filename in os.listdir(TIME_DIR):
        if filename.endswith(".csv"):
            name_no_ext = os.path.splitext(filename)[0]
            combination = name_no_ext.split("time-")[1]
            all_combinations.append(combination)

    # Remove duplicates
    all_combinations = list(set(all_combinations))
    logging.info(f"\nTotal number of combinations found: {len(all_combinations)}")
    logging.info("\n")

    # -------------------------------------------------------------------------#
    # END: Read all available combinations
    # -------------------------------------------------------------------------#

    evaluation = []
    for combination in all_combinations:
        keys = combination.split("-")
        evaluation.append(
            evaluate(
                dataset=keys[0],
                group=keys[1],
                library=keys[2],
                library_version=keys[3],
                model=keys[4],
                model_engine=keys[5],
                model_engine_version=keys[6],
                execution_engine=keys[7],
                execution_engine_version=keys[8],
                execution_mode=keys[9],
                execution_mode_version=keys[10],
                num_cpus=keys[11],
                backup_model=keys[12],
                python_version=keys[13],
                os=keys[14],
            )
        )

    evaluation = [eval_ for eval_ in evaluation if eval_ is not None]
    evaluation = pd.concat(evaluation)
    evaluation["time"] /= 60  # minutes
    evaluation = evaluation.set_index(KEY_COLS).stack().reset_index()
    evaluation.columns = KEY_COLS + ["metric", "val"]
    evaluation = (
        evaluation.set_index(KEY_COLS + ["metric"])
        .unstack()
        .droplevel(0, 1)
        .reset_index()
    )

    evaluation = _coerce_columns(evaluation=evaluation, key_cols=KEY_COLS)
    evaluation = evaluation.round(4)
    evaluation.sort_values(KEY_COLS, inplace=True)

    # -------------------------------------------------------------------------#
    # START: Writing Current Evaluations
    # -------------------------------------------------------------------------#
    # NOTE: Since the full file contains information about run date and run
    # times, it will change with each run. Hence to see if any metric has really
    # we should use the static file instead.
    eval_file_name = f"{BASE_DIR}/{dataset}/current_evaluation_full.csv"
    logging.info(f"Writing full evaluation results to {eval_file_name}")
    evaluation.to_csv(eval_file_name, index=False)

    # NOTE: Static File Use Cases
    # (1) If there is no change in the environment and the experiment code, the
    #     static file should not change.
    # (2) This static file can be used to evaluate the impact of making changes
    #     to the experiment code e.g.
    #     (A) Setting used to create the models (in setup, create_model, etc.)
    #     (B) Flow used to get the final forecasts (e.g. just create_model or
    #         create_model followed by tune_model).
    cols_to_drop = ["run_date", "time"]
    eval_file_name = f"{BASE_DIR}/{dataset}/current_evaluation_static.csv"
    logging.info(f"Writing static evaluation results to {eval_file_name}")
    evaluation.drop(columns=cols_to_drop).to_csv(eval_file_name, index=False)

    # -------------------------------------------------------------------------#
    # END: Writing Current Evaluations
    # -------------------------------------------------------------------------#

    logging.info("\n\nEvaluation Complete!")


if __name__ == "__main__":
    fire.Fire(main)
