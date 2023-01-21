"""
To Run
>>> python scripts/mseries/update.py --dataset=M3
"""

import os
import logging

import fire
import pandas as pd

from benchmarks.utils import KEY_COLS, NON_STATIC_COLS, _return_dirs


def main(dataset: str = "M3") -> None:
    """Evaluates the results for a particular dataset

    Parameters
    ----------
    dataset : str, optional
        Dataset for which the evaluation needs to be performed, by default "M3"
    """
    BASE_DIR, _, _ = _return_dirs(dataset=dataset)

    # -------------------------------------------------------------------------#
    # START: Read the evaluation results
    # -------------------------------------------------------------------------#
    running_eval_path = f"{BASE_DIR}/{dataset}/running_evaluations.csv"
    current_eval_path = f"{BASE_DIR}/{dataset}/current_evaluation_full.csv"

    current_eval = pd.read_csv(current_eval_path)
    # When running for the first time, running evaluations will not exist
    if os.path.isfile(running_eval_path):
        running_evals = pd.read_csv(running_eval_path)
    else:
        running_evals = current_eval.copy()

    # -------------------------------------------------------------------------#
    # END: Read the evaluation results
    # -------------------------------------------------------------------------#

    running_evals.set_index(KEY_COLS, inplace=True)
    current_eval.set_index(KEY_COLS, inplace=True)

    STATIC_COLS = [col for col in running_evals.columns if col not in NON_STATIC_COLS]

    # -------------------------------------------------------------------------#
    # START: Updating existing evaluations
    # -------------------------------------------------------------------------#

    logging.info("\n")
    logging.info("Updating existing evaluations ...")
    # Update existing keys based on current run. This will only happen if keys
    # that were run earlier are rerun.
    orig_running_evals = running_evals.copy()
    running_evals.update(current_eval)  # inplace operation
    diff = orig_running_evals[STATIC_COLS].compare(running_evals[STATIC_COLS])
    if len(diff) > 0:
        logging.info(
            f"Updated {len(diff)} existing key(s) in running evaluations based on "
            "metrics in current evaluation."
            "\nNon Static metrics like date and run times are not used for this "
            "comparison."
            "\nReviewer should make sure that all models belonging to the updated "
            "keys have been updated"
        )
    else:
        logging.info(
            "No differences detected to existing keys based on static metrics in "
            "the current eval."
        )

    # -------------------------------------------------------------------------#
    # END: Updating existing evaluations
    # -------------------------------------------------------------------------#

    # -------------------------------------------------------------------------#
    # START: Adding new evaluations
    # -------------------------------------------------------------------------#

    # Add any new keys that have been run in the current evaluation
    logging.info("\n")
    logging.info("Adding new evaluations ...")
    new_keys = ~current_eval.index.isin(running_evals.index)
    if sum(new_keys) > 0:
        logging.info(
            f"Adding {sum(new_keys)} new keys to running evaluations based on "
            "current evaluation."
        )
        running_evals = running_evals.append(current_eval[new_keys])
    else:
        logging.info("No new keys detected in current evaluation.")

    # -------------------------------------------------------------------------#
    # END: Adding new evaluations
    # -------------------------------------------------------------------------#

    # -------------------------------------------------------------------------#
    # START: Write new evaluation results
    # -------------------------------------------------------------------------#

    running_evals.reset_index(inplace=True)
    running_evals.to_csv(running_eval_path, index=False)

    # -------------------------------------------------------------------------#
    # END: Write new evaluation results
    # -------------------------------------------------------------------------#

    logging.info("\n\nUpdate Complete!")


if __name__ == "__main__":
    fire.Fire(main)
