"""Compute code co-occurrence and correlation statistics.

For each model, computes:
- Per-window code co-occurrence (does code X and code Y both fire in
  the same window?)
- Phi correlation coefficients between codes
- Tetrachoric correlation approximation for binary data

This analysis requires that the eval dataset includes multiple codes
scored on the same window.  Because our eval design scores each window
with a single code, we approximate co-occurrence using the original
transcript data (where all codes are scored on every message).

Usage::

    python -m analysis.compute_code_correlations

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

# Minimum sample size for valid correlation calculations
MIN_SAMPLE_SIZE = 2


def _phi_coefficient(x: np.ndarray, y: np.ndarray) -> float:
    """Compute the phi coefficient between two binary arrays.

    Parameters
    ----------
    x, y:
        Binary (0/1) arrays of the same length.

    Returns
    -------
    float
        Phi coefficient in [-1, 1], or NaN if undefined.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = ~(np.isnan(x) | np.isnan(y))
    x, y = x[mask], y[mask]
    if len(x) < MIN_SAMPLE_SIZE:
        return np.nan

    n11 = np.sum((x == 1) & (y == 1))
    n10 = np.sum((x == 1) & (y == 0))
    n01 = np.sum((x == 0) & (y == 1))
    n00 = np.sum((x == 0) & (y == 0))

    denom = np.sqrt((n11 + n10) * (n01 + n00) * (n11 + n01) * (n10 + n00))
    if denom == 0:
        return np.nan
    return float((n11 * n00 - n10 * n01) / denom)


def compute_original_transcript_correlations(df: pd.DataFrame) -> None:
    """Compute code correlations from original transcript data.

    The original transcript has all 18 codes scored on every message,
    making it possible to compute genuine co-occurrence statistics.

    Parameters
    ----------
    df:
        Tidy eval DataFrame.
    """
    orig = df[(df["model"] == "original_transcript") & df["score"].notna()].copy()
    if orig.empty:
        logger.warning("No original transcript data for correlation analysis")
        return

    # Pivot: rows = (window_id, turn_index), columns = code_short, values = score
    # But original transcript data has one row per (window, turn, code) since
    # each window is associated with a single code in the eval design.
    # We need to aggregate at the window level: max score per code per window.
    window_code = (
        orig.groupby(["window_id", "code_short"])["score"].max().unstack(fill_value=0)
    )

    codes = sorted(window_code.columns)
    n_codes = len(codes)

    # Phi correlation matrix
    phi_matrix = pd.DataFrame(np.nan, index=codes, columns=codes)
    for i_idx in range(n_codes):
        for j_idx in range(i_idx, n_codes):
            code_i, code_j = codes[i_idx], codes[j_idx]
            if code_i == code_j:
                phi_matrix.loc[code_i, code_j] = 1.0
            else:
                phi = _phi_coefficient(
                    window_code[code_i].values,
                    window_code[code_j].values,
                )
                phi_matrix.loc[code_i, code_j] = phi
                phi_matrix.loc[code_j, code_i] = phi

    path = DATA_OUTPUT_DIR / "code_phi_correlation_original.csv"
    phi_matrix.round(3).to_csv(path)
    logger.info("Wrote %s", path)

    # Co-occurrence counts
    cooccurrence = pd.DataFrame(0, index=codes, columns=codes, dtype=int)
    for i_idx in range(n_codes):
        for j_idx in range(i_idx, n_codes):
            code_i, code_j = codes[i_idx], codes[j_idx]
            count = int(((window_code[code_i] > 0) & (window_code[code_j] > 0)).sum())
            cooccurrence.loc[code_i, code_j] = count
            cooccurrence.loc[code_j, code_i] = count

    path = DATA_OUTPUT_DIR / "code_cooccurrence_original.csv"
    cooccurrence.to_csv(path)
    logger.info("Wrote %s", path)


def compute_cross_model_code_correlation(df: pd.DataFrame) -> None:
    """Compute cross-model correlation of code-level prevalences.

    For each pair of models, compute the Pearson correlation of their
    per-code prevalence vectors.  This measures how similarly two models
    rank the codes.

    Parameters
    ----------
    df:
        Tidy eval DataFrame.
    """
    df_scored = df[df["score"].notna() & (df["model"] != "original_transcript")].copy()

    model_code_prev = (
        df_scored.groupby(["model_label", "code_short"])["score"]
        .mean()
        .unstack("code_short")
    )

    # Drop models with too few codes scored
    model_code_prev = model_code_prev.dropna(axis=0, thresh=10)

    if len(model_code_prev) < MIN_SAMPLE_SIZE:
        logger.warning("Too few models for cross-model correlation")
        return

    corr = model_code_prev.T.corr().round(3)
    path = DATA_OUTPUT_DIR / "model_prevalence_correlation.csv"
    corr.to_csv(path)
    logger.info("Wrote %s", path)


def main() -> None:
    """Entry point for code correlation computation."""
    parser = argparse.ArgumentParser(
        description="Compute code co-occurrence and correlation statistics."
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

    logger.info("Computing original transcript correlations...")
    compute_original_transcript_correlations(df)

    logger.info("Computing cross-model code correlation...")
    compute_cross_model_code_correlation(df)

    logger.info("Done.")


if __name__ == "__main__":
    main()
