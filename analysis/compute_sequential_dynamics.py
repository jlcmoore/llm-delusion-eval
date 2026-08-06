"""Compute within-window sequential dynamics of code prevalence.

For each model, analyzes how binary scores evolve across turns within
a window.  Computes:
- Turn-position prevalence curves
- Onset analysis: at which turn does a code first trigger?
- Persistence: once a code fires, how often does it fire again?

Usage::

    python -m analysis.compute_sequential_dynamics

Outputs to ``analysis/data/``.
"""

import argparse
import logging

import numpy as np
import pandas as pd

from analysis.artifact_paths import DATA_DIR, ensure_output_dirs
from analysis.load_eval_data import load_all_eval_data

logger = logging.getLogger(__name__)

DATA_OUTPUT_DIR = DATA_DIR


def compute_turn_prevalence(df: pd.DataFrame) -> pd.DataFrame:
    """Compute prevalence at each turn position, per model and category.

    Parameters
    ----------
    df:
        Tidy eval DataFrame.

    Returns
    -------
    pd.DataFrame
        Columns: model_label, category, turn_index, prevalence, n.
    """
    df_turn = df[
        df["score"].notna()
        & df["turn_index"].notna()
        & (df["model"] != "original_transcript")
    ].copy()

    result = (
        df_turn.groupby(["model_label", "category", "turn_index"])
        .agg(prevalence=("score", "mean"), n=("score", "count"))
        .reset_index()
    )
    result["prevalence"] = (result["prevalence"] * 100).round(2)
    return result


def compute_onset_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Compute first-trigger turn index statistics per model and code.

    For each (model, code, window), finds the earliest turn where the
    code fires (score == 1).  Reports the distribution of onset turns.

    Parameters
    ----------
    df:
        Tidy eval DataFrame.

    Returns
    -------
    pd.DataFrame
        Columns: model_label, code_short, mean_onset, median_onset,
        std_onset, n_windows_with_onset, n_windows_total.
    """
    df_turn = df[
        df["score"].notna()
        & df["turn_index"].notna()
        & (df["model"] != "original_transcript")
    ].copy()

    rows = []
    for (model, code), group in df_turn.groupby(["model_label", "code_short"]):
        n_windows = group["window_id"].nunique()
        # For each window, find the first turn where score == 1
        positive = group[group["score"] == 1]
        if positive.empty:
            rows.append(
                {
                    "model_label": model,
                    "code_short": code,
                    "mean_onset": np.nan,
                    "median_onset": np.nan,
                    "std_onset": np.nan,
                    "n_windows_with_onset": 0,
                    "n_windows_total": n_windows,
                }
            )
            continue

        first_onset = positive.groupby("window_id")["turn_index"].min()
        rows.append(
            {
                "model_label": model,
                "code_short": code,
                "mean_onset": round(float(first_onset.mean()), 2),
                "median_onset": float(first_onset.median()),
                "std_onset": round(float(first_onset.std()), 2),
                "n_windows_with_onset": len(first_onset),
                "n_windows_total": n_windows,
            }
        )

    return pd.DataFrame(rows)


def compute_persistence(df: pd.DataFrame) -> pd.DataFrame:
    """Compute persistence: P(score=1 at turn t+1 | score=1 at turn t).

    For each model and code, measures how likely the code is to persist
    once it fires.

    Parameters
    ----------
    df:
        Tidy eval DataFrame.

    Returns
    -------
    pd.DataFrame
        Columns: model_label, code_short, persistence_rate,
        n_transitions.
    """
    df_turn = df[
        df["score"].notna()
        & df["turn_index"].notna()
        & (df["model"] != "original_transcript")
    ].copy()
    df_turn = df_turn.sort_values(
        ["model_label", "code_short", "window_id", "turn_index"]
    )

    rows = []
    for (model, code), group in df_turn.groupby(["model_label", "code_short"]):
        n_transitions = 0
        n_persist = 0

        for _wid, wgroup in group.groupby("window_id"):
            scores = wgroup.sort_values("turn_index")["score"].values
            for idx in range(len(scores) - 1):
                if scores[idx] == 1:
                    n_transitions += 1
                    if scores[idx + 1] == 1:
                        n_persist += 1

        persistence = n_persist / n_transitions if n_transitions > 0 else np.nan
        rows.append(
            {
                "model_label": model,
                "code_short": code,
                "persistence_rate": (
                    round(persistence, 3) if not np.isnan(persistence) else np.nan
                ),
                "n_transitions": n_transitions,
            }
        )

    return pd.DataFrame(rows)


def main() -> None:
    """Entry point for sequential dynamics computation."""
    parser = argparse.ArgumentParser(
        description="Compute within-window sequential dynamics."
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose logging.")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    ensure_output_dirs(DATA_OUTPUT_DIR)

    logger.info("Loading eval data...")
    df = load_all_eval_data()

    logger.info("Computing turn-position prevalence...")
    turn_prev = compute_turn_prevalence(df)
    path = DATA_OUTPUT_DIR / "turn_position_prevalence.csv"
    turn_prev.to_csv(path, index=False)
    logger.info("Wrote %s (%d rows)", path, len(turn_prev))

    logger.info("Computing onset statistics...")
    onset = compute_onset_stats(df)
    path = DATA_OUTPUT_DIR / "onset_statistics.csv"
    onset.to_csv(path, index=False)
    logger.info("Wrote %s (%d rows)", path, len(onset))

    logger.info("Computing persistence rates...")
    persist = compute_persistence(df)
    path = DATA_OUTPUT_DIR / "persistence_rates.csv"
    persist.to_csv(path, index=False)
    logger.info("Wrote %s (%d rows)", path, len(persist))

    logger.info("Done.")


if __name__ == "__main__":
    main()
