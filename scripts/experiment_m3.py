import time
from typing import Optional
from tqdm import tqdm
import pandas as pd
from pycaret.time_series import TSForecastingExperiment
from benchmarks.datasets.create.time_series.m3 import get_data

# Register `pandas.progress_apply` and `pandas.Series.map_apply` with `tqdm`
tqdm.pandas()

directory = "data/"
dataset = "M3"
groups = ["Yearly", "Quarterly", "Monthly", "Other"]


def forecast_single(
    data: pd.DataFrame,
    fh: int,
    target: str,
    prefix: str,
    create_model_kwargs: dict,
    backup_model_kwargs: Optional[dict] = None,
    **kwargs,
) -> pd.DataFrame:
    id = data["unique_id"].unique()[0]
    prefix = prefix or "dataset"
    test_preds = pd.DataFrame()
    setup_passed = False
    model_name = None
    model = None
    try:
        exp = TSForecastingExperiment()
        exp.setup(
            data=data, fh=fh, target=target, experiment_name=f"{prefix}_{id}", **kwargs
        )
        setup_passed = True
        try:
            model_name = create_model_kwargs.get("estimator")
            model = exp.create_model(**create_model_kwargs)
            test_preds = exp.predict_model(model)
        except Exception as e:
            print(f"Error occurred for ID: {id} when trying main model: {e}")
            if backup_model_kwargs is not None:
                try:
                    print(f"Trying backup model for ID: {id}")
                    model_name = backup_model_kwargs.get("estimator")
                    model = exp.create_model(**backup_model_kwargs)
                    test_preds = exp.predict_model(model)
                except Exception as e:
                    print(f"Error occurred for ID: {id} when trying backup model: {e}")
    except Exception as e:
        if not setup_passed:
            print(
                f"Error occurred for ID: {id} during experiment setup. No model created: {e}"
            )

    # Add model name and model hyperparameters used ----
    test_preds["model_name"] = model_name
    test_preds["model"] = model
    return test_preds


# "Other" group since it has the least number of time series
groups = groups[-1:]
for group in tqdm(groups):
    train, fh, freq, seasonality = get_data(
        directory=directory, dataset=dataset, group=group
    )
    test, _, _, _ = get_data(
        directory=directory, dataset=dataset, group=group, train=False
    )

    combined = pd.concat([train, test], axis=0)
    combined["ds"] = pd.to_datetime(combined["ds"])
    combined.set_index("ds", inplace=True)

    model = "ets"
    prefix = f"{dataset}-{group}-{model}"
    start = time.time()
    test_results = combined.groupby("unique_id").progress_apply(
        forecast_single,
        fh=fh,
        target="y",
        prefix=prefix,
        fold=1,
        ignore_features=["unique_id"],
        session_id=42,
        verbose=False,
        create_model_kwargs={"estimator": model, "cross_validation": False},
        backup_model_kwargs={"estimator": "naive", "cross_validation": False},
    )
    end = time.time()
    time_taken = round(end - start)
    print(f"Total time taken for group {group}: {time_taken}s")

    test_results.reset_index(inplace=True)
    test_results.rename(columns={"level_1": "ds"}, inplace=True)
    test_results.to_csv(f"data/forecasts-{prefix}.csv", index=False)

    time_df = pd.DataFrame(
        {"time": [time_taken], "dataset": [dataset], "group": [group], "model": [model]}
    )
    time_df.to_csv(f"data/time-{prefix}.csv", index=False)

print("DONE")
