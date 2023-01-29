"""Module to forecast a single time series using various pycaret flows."""
import logging
from typing import Optional

import pandas as pd
from pycaret.time_series import TSForecastingExperiment

from benchmarks.utils import _impute_time_series_model_engine

logging.basicConfig(level=logging.INFO, format="%(asctime)s: %(message)s")


def forecast_create_model(
    data: pd.DataFrame,
    setup_kwargs: dict,
    create_model_kwargs: dict,
    backup_model_kwargs: Optional[dict] = None,
    prefix: Optional[str] = None,
) -> pd.DataFrame:
    """Produces forecasts for a single time series using create_model.

    i.e. Single Time Series, Single Model

    Parameters
    ----------
    data : pd.DataFrame
        The data for a single intersection or group (i.e. unique_id)
    setup_kwargs : dict
        The arguments to pass to pycaret's setup function. The following arguments
        are auto inferred and do not need to be included:
        (1) data: taken from the data argument of this function
        (2) experiment_name: derived from the unique_id in the data and the prefix.
    create_model_kwargs : dict
        The arguments to pass to pycaret's create_model. This must include a key
        called "estimator" which is the name of the model to use. Other arguments
        are optional.
    backup_model_kwargs : Optional[dict], optional
        The model to use to create the forecasts in case the main model fails to
        produce a forecast. If provided, this must include a key called "estimator"
        which is the name of the model to use. Other arguments are optional,
        by default None
    prefix : str
        Prefix to use for the experiment name, by default None

    Returns
    -------
    pd.DataFrame
        Predictions for the time series
    """
    unique_id = data["unique_id"].unique()[0]
    prefix = prefix or "dataset"
    test_preds = pd.DataFrame()
    setup_passed = False
    model_name = None
    model_engine = None
    model = None
    try:
        exp = TSForecastingExperiment()
        exp.setup(data=data, experiment_name=f"{prefix}_{unique_id}", **setup_kwargs)
        setup_passed = True
        try:
            model_name = create_model_kwargs.get("estimator")
            model_engine = exp.get_engine(model_name)
            model = exp.create_model(**create_model_kwargs)
            test_preds = exp.predict_model(model, verbose=False)
        except Exception as e:
            logging.warn(
                f"Error occurred for ID: {unique_id} when trying main model: " f"{e}"
            )
            if backup_model_kwargs is not None:
                try:
                    logging.info(f"Trying backup model for ID: {unique_id}")
                    model_name = backup_model_kwargs.get("estimator")
                    model_engine = exp.get_engine(model_name)
                    model = exp.create_model(**backup_model_kwargs)
                    test_preds = exp.predict_model(model, verbose=False)
                except Exception as e:
                    logging.warn(
                        f"Error occurred for ID: {unique_id} when trying backup model: "
                        f"{e}"
                    )
    except Exception as e:
        if not setup_passed:
            logging.warn(
                f"Error occurred for ID: {unique_id} during experiment setup. "
                f"No model created: {e}"
            )

    # Add model name and model hyperparameters used ----
    test_preds["model_name"] = model_name
    test_preds["model_engine"] = _impute_time_series_model_engine(engine=model_engine)
    test_preds["model"] = model.__repr__()

    # Fugue does not return back the group by column (like Pandas)
    # Hence, add it as a column
    test_preds["unique_id"] = unique_id

    # PyArrow does not support PeriodIndex (returned by PyCaret depending on
    # input data index type). Hence converting to datetime if applicable.
    try:
        if isinstance(test_preds.index, pd.PeriodIndex):
            test_preds = test_preds.to_timestamp()
    except TypeError as e:
        logging.info(f"Index for {unique_id} is not coercible to timestamp: {e}")

    # Fugue may have issues handling index.
    # Hence it is better to reset it before returning.
    test_preds.reset_index(inplace=True)

    test_preds.rename(columns={"index": "ds"}, inplace=True)

    return test_preds
