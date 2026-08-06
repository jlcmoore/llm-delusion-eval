"""Compute per-participant prevalence profiles across models and codes.

Produces a CSV of prevalence rates broken down by participant, model,
and code/category.  Useful for checking whether results are driven by
a small number of participants.

Usage::

    python -m analysis.compute_participant_profiles

Outputs to ``analysis/data/``.
"""

import argparse
import logging

import pandas as pd

from analysis.artifact_paths import DATA_DIR, ensure_output_dirs
from analysis.load_eval_data import CATEGORY_ORDER, CODE_CATEGORIES, load_all_eval_data
from analysis.participant_mapping import add_participant_column

logger = logging.getLogger(__name__)

DATA_OUTPUT_DIR = DATA_DIR


def compute_participant_profiles(df: pd.DataFrame) -> None:
    """Compute and save per-participant prevalence profiles.

    Parameters
    ----------
    df:
        Tidy eval DataFrame with a ``participant`` column.
    """
    df_scored = df[df["score"].notna() & df["participant"].notna()].copy()
    if df_scored.empty:
        logger.warning("No scored data with participant info")
        return

    # -- Per-participant, per-model, per-category prevalence --
    ppt_cat = (
        df_scored.groupby(["participant", "model_label", "category"])
        .agg(
            prevalence=("score", "mean"),
            n=("score", "count"),
        )
        .reset_index()
    )
    ppt_cat["prevalence"] = (ppt_cat["prevalence"] * 100).round(1)
    path = DATA_OUTPUT_DIR / "participant_category_prevalence.csv"
    ppt_cat.to_csv(path, index=False)
    logger.info("Wrote %s", path)

    # -- Per-participant, per-model, per-code prevalence --
    ppt_code = (
        df_scored.groupby(["participant", "model_label", "code_short"])
        .agg(
            prevalence=("score", "mean"),
            n=("score", "count"),
        )
        .reset_index()
    )
    ppt_code["prevalence"] = (ppt_code["prevalence"] * 100).round(1)
    path = DATA_OUTPUT_DIR / "participant_code_prevalence.csv"
    ppt_code.to_csv(path, index=False)
    logger.info("Wrote %s", path)

    # -- Participant contribution summary --
    # How many windows per participant, per code
    ppt_windows = (
        df_scored.groupby(["participant", "code_short"])["window_id"]
        .nunique()
        .reset_index()
        .rename(columns={"window_id": "n_windows"})
    )
    path = DATA_OUTPUT_DIR / "participant_window_counts.csv"
    ppt_windows.to_csv(path, index=False)
    logger.info("Wrote %s", path)

    # -- Wide heatmap: participants x codes (averaged across models) --
    models_only = df_scored[df_scored["model"] != "original_transcript"]
    wide = (
        models_only.groupby(["participant", "code_short"])["score"]
        .mean()
        .unstack("code_short")
        * 100
    )

    # Sort codes by category
    code_order = sorted(
        wide.columns,
        key=lambda code: (
            (
                CATEGORY_ORDER.index(CODE_CATEGORIES.get(code, "unknown"))
                if CODE_CATEGORIES.get(code, "unknown") in CATEGORY_ORDER
                else 99
            ),
            code,
        ),
    )
    wide = wide[[c for c in code_order if c in wide.columns]]
    wide = wide.round(1)
    path = DATA_OUTPUT_DIR / "participant_code_heatmap_data.csv"
    wide.to_csv(path)
    logger.info("Wrote %s", path)

    # -- Participant-level variance: how much does prevalence vary
    # across participants for each model? --
    ppt_overall = (
        models_only.groupby(["participant", "model_label"])["score"]
        .mean()
        .reset_index()
    )
    variance = (
        ppt_overall.groupby("model_label")["score"]
        .agg(["mean", "std", "min", "max", "count"])
        .round(3)
    )
    variance.columns = [
        "mean_prevalence",
        "std_prevalence",
        "min_prevalence",
        "max_prevalence",
        "n_participants",
    ]
    path = DATA_OUTPUT_DIR / "participant_variance_by_model.csv"
    variance.to_csv(path)
    logger.info("Wrote %s", path)


def main() -> None:
    """Entry point for participant profile computation."""
    parser = argparse.ArgumentParser(
        description="Compute per-participant prevalence profiles."
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
    df = add_participant_column(df)
    logger.info("Computing participant profiles...")
    compute_participant_profiles(df)
    logger.info("Done.")


if __name__ == "__main__":
    main()
