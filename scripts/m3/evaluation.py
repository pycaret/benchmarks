from itertools import product

import numpy as np
import pandas as pd

from benchmarks.datasets.create.time_series.m3 import evaluate
from benchmarks.utils import return_pycaret_model_names

if __name__ == "__main__":
    groups = ["Yearly", "Quarterly", "Monthly", "Other"]

    models = return_pycaret_model_names()
    datasets = ["M3"]
    evaluation = [
        evaluate(
            dataset=dataset,
            group=group,
            model=model,
            engine="ray",
            execution_mode="native",
        )
        for model, group in product(models, groups)
        for dataset in datasets
    ]
    evaluation = [eval_ for eval_ in evaluation if eval_ is not None]
    evaluation = pd.concat(evaluation)
    evaluation = evaluation[
        [
            "dataset",
            "group",
            "model",
            "engine",
            "execution_mode",
            "mape",
            "smape",
            "time",
        ]
    ]
    evaluation["time"] /= 60  # minutes
    evaluation = (
        evaluation.set_index(["dataset", "group", "model", "engine", "execution_mode"])
        .stack()
        .reset_index()
    )
    evaluation.columns = [
        "dataset",
        "group",
        "model",
        "engine",
        "execution_mode",
        "metric",
        "val",
    ]
    evaluation = (
        evaluation.set_index(
            ["dataset", "group", "model", "engine", "execution_mode", "metric"]
        )
        .unstack()
        .round(3)
    )
    evaluation = evaluation.droplevel(0, 1).reset_index()
    evaluation.replace(0, np.nan, inplace=True)
    evaluation.to_csv("data/evaluation.csv", index=False)
    print(evaluation)
