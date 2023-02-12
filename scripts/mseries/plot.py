"""
To Run
>>> python scripts/mseries/plot.py
>>> python scripts/mseries/plot.py --dataset=M3 --group=Other
>>> python scripts/mseries/plot.py --dataset=M3 --group=Yearly
>>> python scripts/mseries/plot.py --dataset=M3 --group=Quarterly
>>> python scripts/mseries/plot.py --dataset=M3 --group=Monthly

NOTE: Passing multiple values for an arguments
(1) Can not have spaces between commas in the list (see 'model' argument below)
(2) Single values must also be provided as a list (see 'os' argument below)
>>> python scripts/mseries/plot.py --dataset=M3 --group=Monthly --model=["ets","naive"] --os=["win32"]
"""

import ast
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
        f"\nPlotting results for dataset: '{dataset}' group: '{group}' metric: '{metric}'"
        f"\n    from Directory: '{EVAL_DIR}'"
    )

    # -------------------------------------------------------------------------#
    # START: Read evaluations
    # -------------------------------------------------------------------------#
    running_evals = pd.read_csv(f"{EVAL_DIR}/running_evaluations.csv")
    try:
        ext_benchmarks = pd.read_csv(f"{EVAL_DIR}/external_benchmarks.csv")
    except FileNotFoundError:
        ext_benchmarks = pd.DataFrame({}, columns=running_evals.columns)

    running_evals["source"] = "Internal"
    ext_benchmarks["source"] = "External"

    combined = pd.concat([running_evals, ext_benchmarks])
    combined = combined.query("group == @group")

    if metric in combined.columns:

        # ---------------------------------------------------------------------#
        # START: Exclude
        # ---------------------------------------------------------------------#
        # Models that have large errors and distort plots
        exclude_models = ["lar_cds_dt", "par_cds_dt"]
        combined = combined.query("model not in @exclude_models")

        # Intermediate versions
        exclude_versions = [
            "38f01856b5ca14ca79aa800677b489c381d5aa71",
            "6a4914f1552547fe73dd7be13f3d1843e72d1f61",
        ]
        combined = combined.query("library_version not in @exclude_versions")

        # ---------------------------------------------------------------------#
        # START: Filter based on user inputs
        # ---------------------------------------------------------------------#

        def _print_all_available(key: str, values: Optional[str] = None):
            values = values or "All Available"
            print(f"    {key} included: '{values}'")

        key = "Libraries"
        if library:
            values = ast.literal_eval(library)
            _print_all_available(key=key, values=values)
            combined = combined.query("library in @values")
        else:
            _print_all_available(key=key)

        key = "Library Versions"
        if library_version:
            values = ast.literal_eval(library_version)
            _print_all_available(key=key, values=values)
            combined = combined.query("library_version in @values")
        else:
            _print_all_available(key=key)

        key = "Models"
        if model:
            values = ast.literal_eval(str(model))
            _print_all_available(key=key, values=values)
            combined = combined.query("model in @values")
        else:
            _print_all_available(key=key)

        key = "Model Engines"
        if model_engine:
            values = ast.literal_eval(str(model_engine))
            _print_all_available(key=key, values=values)
            combined = combined.query("model_engine in @values")
        else:
            _print_all_available(key=key)

        key = "Model Engine Versions"
        if model_engine_version:
            values = ast.literal_eval(str(model_engine_version))
            _print_all_available(key=key, values=values)
            combined = combined.query("model_engine_version in @values")
        else:
            _print_all_available(key=key)

        key = "Execution Engines"
        if execution_engine:
            values = ast.literal_eval(str(execution_engine))
            _print_all_available(key=key, values=values)
            combined = combined.query("execution_engine in @values")
        else:
            _print_all_available(key=key)

        key = "Execution Engine Versions"
        if execution_engine_version:
            values = ast.literal_eval(str(execution_engine_version))
            _print_all_available(key=key, values=values)
            combined = combined.query("execution_engine_version in @values")
        else:
            _print_all_available(key=key)

        key = "Execution Modes"
        if execution_mode:
            values = ast.literal_eval(str(execution_mode))
            _print_all_available(key=key, values=values)
            combined = combined.query("execution_mode in @values")
        else:
            _print_all_available(key=key)

        key = "Execution Mode Versions"
        if execution_mode_version:
            values = ast.literal_eval(str(execution_mode_version))
            _print_all_available(key=key, values=values)
            combined = combined.query("execution_mode_version in @values")
        else:
            _print_all_available(key=key)

        key = "Number of CPUs"
        if num_cpus:
            values = ast.literal_eval(str(num_cpus))
            _print_all_available(key=key, values=values)
            combined = combined.query("num_cpus in @values")
        else:
            _print_all_available(key=key)

        key = "Backup Models"
        if backup_model:
            values = ast.literal_eval(str(backup_model))
            _print_all_available(key=key, values=values)
            combined = combined.query("backup_model in @values")
        else:
            _print_all_available(key=key)

        key = "Python Versions"
        if python_version:
            values = ast.literal_eval(str(python_version))
            _print_all_available(key=key, values=values)
            combined = combined.query("python_version in @values")
        else:
            _print_all_available(key=key)

        key = "OS"
        if os:
            values = ast.literal_eval(str(os))
            _print_all_available(key=key, values=values)
            combined = combined.query("os in @values")
        else:
            _print_all_available(key=key)

        # Convert mins to seconds ----
        combined["time"] = (combined["time"] * 60).round()

        combined["norm_time_cpu"] = combined["time"] * combined["num_cpus"]
        combined["norm_time_cpu_model"] = (
            combined["norm_time_cpu"] / combined["count_ts"]
        )
        combined["key"] = combined[KEY_COLS].apply(
            lambda row: "-".join(row.values.astype(str)), axis=1
        )
        combined["name"] = combined[["library", "model", "model_engine"]].apply(
            lambda row: "-".join(row.values.astype(str)), axis=1
        )
        combined.sort_values(by=metric, inplace=True)
        combined[KEY_COLS] = combined[KEY_COLS].astype(str)

        # ---------------------------------------------------------------------#
        # START: Plot Results
        # ---------------------------------------------------------------------#

        plot_prefix = f"{dataset}_{group}_{metric}"

        fig1 = plot_metrics(combined, metric, dataset, group, name_col="key")
        # fig1.show()
        fig1.write_html(f"{EVAL_DIR}/{plot_prefix}.html")

        fig2 = plot_metrics_vs_time(combined, metric, dataset, group, name_col="key")
        # fig2.show()
        fig2.write_html(f"{EVAL_DIR}/{plot_prefix}_vs_time.html")

        logging.info("\nPlotting Complete!")
    else:
        logging.warning(f"Metric '{metric}' not found in data. Skipping plotting.")


if __name__ == "__main__":
    fire.Fire(main)
