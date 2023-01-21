"""
To Run
>>> python scripts/mseries/plot.py
>>> python scripts/mseries/plot.py --dataset=M3 --group=Other
>>> python scripts/mseries/plot.py --dataset=M3 --group=Yearly
>>> python scripts/mseries/plot.py --dataset=M3 --group=Quarterly
>>> python scripts/mseries/plot.py --dataset=M3 --group=Monthly
# TODO: See how to accept list of values for each parameter from command line
"""

import logging
from typing import List, Optional

import fire
import pandas as pd

from benchmarks.plots.metrics import plot_metrics, plot_metrics_vs_time
from benchmarks.utils import KEY_COLS, _return_dirs


def main(
    dataset: str = "M3",
    group: str = "Other",
    library: Optional[List[str]] = None,
    library_version: Optional[List[str]] = None,
    model: Optional[List[str]] = None,  # ["auto_arima"],
    model_engine: Optional[List[str]] = None,
    model_engine_version: Optional[List[str]] = None,
    execution_engine: Optional[List[str]] = None,
    execution_engine_version: Optional[List[str]] = None,
    execution_mode: Optional[List[str]] = None,
    execution_mode_version: Optional[List[str]] = None,
    num_cpus: Optional[List[str]] = None,
    backup_model: Optional[List[str]] = None,
    python_version: Optional[List[str]] = None,
    os: Optional[List[str]] = None,
    metric: str = "smape",
) -> None:
    """Evaluates the results for a particular dataset

    Parameters
    ----------
    dataset : str, optional
        Dataset for which the evaluation needs to be performed, by default "M3"
    """
    BASE_DIR, _, _, _ = _return_dirs(dataset=dataset)
    EVAL_DIR = f"{BASE_DIR}/{dataset}"

    logging.info("\n\n")
    logging.info(
        f"\nPlotting results for dataset: '{dataset}'"
        f"\n    from Directory: '{EVAL_DIR}'"
    )

    # -------------------------------------------------------------------------#
    # START: Read evaluations
    # -------------------------------------------------------------------------#
    running_evals = pd.read_csv(f"{EVAL_DIR}/running_evaluations.csv")
    ext_benchmarks = pd.read_csv(f"{EVAL_DIR}/external_benchmarks.csv")

    running_evals["source"] = "Internal"
    ext_benchmarks["source"] = "External"

    combined = pd.concat([running_evals, ext_benchmarks])
    combined = combined.query("group == @group")

    # -------------------------------------------------------------------------#
    # START: Filter based on user inputs
    # -------------------------------------------------------------------------#

    if library:
        combined = combined.query("library in @library")
    if library_version:
        combined = combined.query("library_version in @library_version")
    if model:
        combined = combined.query("model in @model")
    if model_engine:
        combined = combined.query("model_engine in @model_engine")
    if model_engine_version:
        combined = combined.query("model_engine_version in @model_engine_version")
    if execution_engine:
        combined = combined.query("execution_engine in @execution_engine")
    if execution_engine_version:
        combined = combined.query(
            "execution_engine_version in @execution_engine_version"
        )
    if execution_mode:
        combined = combined.query("execution_mode in @execution_mode")
    if execution_mode_version:
        combined = combined.query("execution_mode_version in @execution_mode_version")
    if num_cpus:
        combined = combined.query("num_cpus in @num_cpus")
    if backup_model:
        combined = combined.query("backup_model in @backup_model")
    if python_version:
        combined = combined.query("python_version in @python_version")
    if os:
        combined = combined.query("os in @os")

    combined["norm_time_cpu"] = combined["time"] * combined["num_cpus"]
    combined["norm_time_cpu_model"] = combined["norm_time_cpu"] / combined["count_ts"]
    combined["key"] = combined[KEY_COLS].apply(
        lambda row: "-".join(row.values.astype(str)), axis=1
    )
    combined["name"] = combined[["library", "model", "model_engine"]].apply(
        lambda row: "-".join(row.values.astype(str)), axis=1
    )
    combined.sort_values(by=metric, inplace=True)
    combined[KEY_COLS] = combined[KEY_COLS].astype(str)

    # -------------------------------------------------------------------------#
    # START: Plot Results
    # -------------------------------------------------------------------------#

    plot_metrics(combined, metric, dataset, group)
    plot_metrics_vs_time(combined, metric, dataset, group)

    logging.info("\nPlotting Complete!")


if __name__ == "__main__":
    fire.Fire(main)
