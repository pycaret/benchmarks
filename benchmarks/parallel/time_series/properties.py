"""Module to extract the time series properties."""
import logging
import time
from typing import Optional

import pandas as pd
from pycaret.time_series import TSForecastingExperiment

logging.basicConfig(level=logging.INFO, format="%(asctime)s: %(message)s")


def extract_properties(
    data: pd.DataFrame,
    setup_kwargs: dict,
    prefix: Optional[str] = None,
) -> pd.DataFrame:
    """Extracts the properties of a single time series using the setup settings.

    Parameters
    ----------
    data : pd.DataFrame
        The data for a single intersection or group (i.e. unique_id)
    setup_kwargs : dict
        The arguments to pass to pycaret's setup function. The following arguments
        are auto inferred and do not need to be included:
        (1) data: taken from the data argument of this function
        (2) experiment_name: derived from the unique_id in the data and the prefix.
    prefix : str
        Prefix to use for the experiment name, by default None

    Returns
    -------
    pd.DataFrame
        Properties of the time series
    """
    unique_id = data["unique_id"].unique()[0]
    prefix = prefix or "dataset"
    properties = pd.DataFrame()
    setup_passed = False
    try:
        start = time.time()
        exp = TSForecastingExperiment()
        exp.setup(data=data, experiment_name=f"{prefix}_{unique_id}", **setup_kwargs)
        end = time.time()
        setup_passed = True
        try:
            properties = pd.DataFrame(
                {
                    "len_total": [len(exp.y)],
                    "len_train": [len(exp.y_train)],
                    "len_test": [len(exp.y_test)],
                    "strictly_positive": [exp.strictly_positive],
                    "white_noise": [exp.white_noise],
                    "lowercase_d": [exp.lowercase_d],
                    "uppercase_d": [exp.uppercase_d],
                    "seasonality_present": [exp.seasonality_present],
                    "seasonality_type": [exp.seasonality_type],
                    "candidate_sps": [exp.candidate_sps],
                    "significant_sps": [exp.significant_sps],
                    "all_sps": [exp.all_sps_to_use],
                    "primary_sp": [exp.primary_sp_to_use],
                    "significant_sps_no_harmonics": [exp.significant_sps_no_harmonics],
                    "all_sps_no_harmonics": [
                        exp.significant_sps_no_harmonics[0 : len(exp.all_sps_to_use)]
                    ],
                    "primary_sp_no_harmonics": [exp.significant_sps_no_harmonics[0]],
                    "time_taken": [round(end - start, 4)],
                }
            )
        except Exception as e:
            logging.warn(
                f"Error occurred for ID: {unique_id} when extracting properties: "
                f"{e}"
            )
    except Exception as e:
        if not setup_passed:
            logging.warn(
                f"Error occurred for ID: {unique_id} during experiment setup. "
                f"No properties extracted created: {e}"
            )

    # Fugue does not return back the group by column (like Pandas)
    # Hence, add it as a column
    properties["unique_id"] = unique_id

    # Fugue may have issues handling index.
    # Hence it is better to reset it before returning.
    properties.reset_index(inplace=True)

    return properties
