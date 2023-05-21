"""Module to forecast a single time series using various pycaret flows."""
import logging
from typing import Optional

import pandas as pd
from pycaret.time_series import TSForecastingExperiment
import time

from benchmarks.utils import _impute_time_series_model_engine

logging.basicConfig(level=logging.INFO, format="%(asctime)s: %(message)s")


def forecast_single_ts_ens_freq(
    data: pd.DataFrame,
    setup_kwargs: dict,
    create_model_kwargs: dict,
    backup_model_kwargs: Optional[dict] = None,
    prefix: Optional[str] = None,
    max_models: int = 1,
    weighted: bool = True,
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
    extended_model_name = None
    model_engine = None
    model = None
    total_start, total_end = 0, -1
    setup_start, setup_end = 0, -1
    train_start, train_end = 0, -1
    predict_start, predict_end = 0, -1
    try:
        total_start = time.time()
        setup_start = time.time()
        exp = TSForecastingExperiment()
        cross_validation = True if weighted else False
        exp.setup(data=data, experiment_name=f"{prefix}_{unique_id}", **setup_kwargs)
        setup_end = time.time()
        setup_passed = True
        try:
            model_name = create_model_kwargs.get("estimator")
            # We sue same name for both conditions so that each is counted against
            # the primary model % in the evauation later.
            extended_model_name = _get_extended_model_name(
                model=model_name,
                max_models=max_models,
                weighted=weighted,
                fold=exp.fold,
                remove_harmonics=exp.remove_harmonics,
                harmonic_order_method=exp.harmonic_order_method,
            )
            model_engine = exp.get_engine(model_name)
            train_start = time.time()
            models = []
            weights = [] if weighted else None
            N = min(max_models, len(exp.significant_sps))
            sps_for_ensemble = [sp for sp in exp.candidate_sps[:N] if sp <= 52]

            if len(sps_for_ensemble) > 1:
                for sp in sps_for_ensemble:
                    model_kwargs = _get_model_kwargs(model_name, sp)
                    model = exp.create_model(
                        **create_model_kwargs,
                        cross_validation=cross_validation,
                        **model_kwargs,
                    )
                    models.append(model)
                    if weighted:
                        metrics = exp.pull()
                        weight = 1 / metrics.loc["Mean"]["MAPE"]
                        weights.append(weight)
                # Can not disable cross-validation in blend_models
                combined = exp.blend_models(models, weights=weights, verbose=False)
            else:
                logging.info(
                    f"ID: {unique_id}: "
                    "Number of Models = 1. Hence no ensemble will be created. "
                    "Will use a single model without cross-validation instead."
                )
                combined = exp.create_model(
                    **create_model_kwargs,
                    cross_validation=False,
                )
            train_end = time.time()
            predict_start = time.time()
            test_preds = exp.predict_model(combined, verbose=False)
            predict_end = time.time()
        except Exception as e:
            logging.warn(
                f"Error occurred for ID: {unique_id} when trying main model: " f"{e}"
            )
            if backup_model_kwargs is not None:
                try:
                    logging.info(f"Trying backup model for ID: {unique_id}")
                    model_name = backup_model_kwargs.get("estimator")
                    extended_model_name = model_name
                    model_engine = exp.get_engine(model_name)
                    train_start = time.time()
                    combined = exp.create_model(**backup_model_kwargs)
                    train_end = time.time()
                    predict_start = time.time()
                    test_preds = exp.predict_model(combined, verbose=False)
                    predict_end = time.time()
                except Exception as e:
                    logging.warn(
                        f"Error occurred for ID: {unique_id} when trying backup model: "
                        f"{e}"
                    )
        total_end = time.time()
    except Exception as e:
        if not setup_passed:
            logging.warn(
                f"Error occurred for ID: {unique_id} during experiment setup. "
                f"No model created: {e}"
            )

    # Add timing information ----
    test_preds["total_time"] = round(total_end - total_start, 4)
    test_preds["setup_time"] = round(setup_end - setup_start, 4)
    test_preds["train_time"] = round(train_end - train_start, 4)
    test_preds["predict_time"] = round(predict_end - predict_start, 4)

    # Add model name and model hyperparameters used ----
    test_preds["model_name"] = extended_model_name
    test_preds["model_engine"] = _impute_time_series_model_engine(engine=model_engine)
    test_preds["model"] = combined.__repr__()

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


def _get_model_kwargs(model_name: str, sp: int) -> dict:
    """Returns the model kwargs to use for the models.

    Parameters
    ----------
    model_name : str
        Name of the model
    sp : int
        Seasonal period to use in the model

    Returns
    -------
    dict
        The kwargs to use for the model
    """
    if model_name == "arima":
        model_kwargs = {"seasonal_order": (0, 1, 0, sp)}
    else:
        raise ValueError(
            f"Model kwargs have not been implemented for model: {model_name}."
        )
    return model_kwargs


def _get_extended_model_name(
    model: str,
    max_models: int,
    weighted: bool,
    fold: int,
    remove_harmonics: bool,
    harmonic_order_method: str,
) -> str:
    """Returns the model name with the number of models, weighted, and fold.

    Parameters
    ----------
    model : str
        Name of the model
    max_models : int
        Maximum number of models to use in the ensemble
    weighted : bool
        Whether to use weighted averaging in the ensemble
    fold : int
        Fold number to use for the ensemble
    remove_harmonics : bool
        Whether to remove harmonics from the seasonal periods
    harmonic_order_method : str
        The method to use to order the harmonics

    Returns
    -------
    str
        The extended model name
    """
    if weighted is False:
        fold = None
    if remove_harmonics is False:
        harmonic_order_method = None
    return f"{model}_ens_freq_{max_models}_{weighted}_{fold}_{remove_harmonics}_{harmonic_order_method}"
