"""Per-model code-level residuals against each model's category mean.

For every (model, category) pair, subtract the model's mean prevalence across
the codes in that category from each individual code's prevalence. Positive
residuals identify codes the model triggers more than its category-typical
rate; negative residuals identify codes it suppresses relative to its own
category baseline. This isolates code-level heterogeneity from overall
model-level prevalence differences and supports the item-level analyses in
the paper's Results section.

Reads ``analysis/data/prevalence_by_model_code.csv`` (produced by
``analysis/generate_figures.py``) and writes:

* ``analysis/data/code_residuals_by_model.csv`` -- long-form residual table
  with columns ``code_short, category, model, prevalence, category_mean,
  residual``.
* ``analysis/data/code_residuals_top_by_model.csv`` -- per model, the
  ``TOP_N`` strongest positive and negative residuals.

Usage::

    python -m analysis.compute_code_residuals
"""

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

INPUT_PATH = Path(__file__).parent / "data" / "prevalence_by_model_code.csv"
OUTPUT_LONG = Path(__file__).parent / "data" / "code_residuals_by_model.csv"
OUTPUT_TOP = Path(__file__).parent / "data" / "code_residuals_top_by_model.csv"

TOP_N = 3
NON_MODEL_COLUMNS = ("code_short", "category")


def load_code_prevalence(path: Path = INPUT_PATH) -> pd.DataFrame:
    """Load the wide per-model code prevalence table.

    Returns the dataframe as produced by ``generate_figures.py``: one row per
    code, columns ``code_short``, ``category``, then one column per model
    label with prevalence in percent.
    """
    return pd.read_csv(path)


def compute_residuals(df_wide: pd.DataFrame) -> pd.DataFrame:
    """Return long-form residuals for every (model, code) pair.

    Each residual is ``prevalence - category_mean`` where ``category_mean``
    is the model's mean across all codes in that code's category. Output
    columns: ``code_short, category, model, prevalence, category_mean,
    residual``.
    """
    model_columns = [c for c in df_wide.columns if c not in NON_MODEL_COLUMNS]
    long_df = df_wide.melt(
        id_vars=list(NON_MODEL_COLUMNS),
        value_vars=model_columns,
        var_name="model",
        value_name="prevalence",
    )
    long_df["category_mean"] = long_df.groupby(["model", "category"])[
        "prevalence"
    ].transform("mean")
    long_df["residual"] = long_df["prevalence"] - long_df["category_mean"]
    long_df = long_df.round({"prevalence": 2, "category_mean": 2, "residual": 2})
    return long_df[
        [
            "code_short",
            "category",
            "model",
            "prevalence",
            "category_mean",
            "residual",
        ]
    ]


def top_residuals_per_model(
    residuals: pd.DataFrame, top_n: int = TOP_N
) -> pd.DataFrame:
    """Return the ``top_n`` strongest positive and negative residuals per model.

    The ``rank`` column orders rows within each model from most positive to
    most negative residual; the ``side`` column flags whether a row is in the
    positive or negative top-N.
    """
    rows: list[pd.DataFrame] = []
    for model, group in residuals.groupby("model"):
        sorted_group = group.sort_values("residual", ascending=False)
        positive = sorted_group.head(top_n).assign(side="positive")
        negative = sorted_group.tail(top_n).assign(side="negative")
        rows.append(pd.concat([positive, negative], ignore_index=True))
    out = pd.concat(rows, ignore_index=True)
    out["rank"] = out.groupby(["model", "side"]).cumcount() + 1
    return out[
        [
            "model",
            "side",
            "rank",
            "code_short",
            "category",
            "prevalence",
            "category_mean",
            "residual",
        ]
    ]


def print_top_summary(top: pd.DataFrame) -> None:
    """Print a concise per-model summary of strongest residuals."""
    for model, group in top.groupby("model"):
        print(f"\n=== {model} ===")
        for _, row in group.iterrows():
            sign = "+" if row["residual"] >= 0 else ""
            print(
                f"  [{row['side']:>8} #{int(row['rank'])}] "
                f"{row['code_short']:<32} ({row['category']}): "
                f"{sign}{row['residual']:.2f}pp "
                f"(prev {row['prevalence']:.2f}, cat mean "
                f"{row['category_mean']:.2f})"
            )


def main() -> None:
    """Entry point for command-line execution."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    df_wide = load_code_prevalence()
    residuals = compute_residuals(df_wide)
    top = top_residuals_per_model(residuals)
    OUTPUT_LONG.parent.mkdir(parents=True, exist_ok=True)
    residuals.to_csv(OUTPUT_LONG, index=False)
    top.to_csv(OUTPUT_TOP, index=False)
    logger.info("Wrote %s, %s", OUTPUT_LONG, OUTPUT_TOP)
    print_top_summary(top)


if __name__ == "__main__":
    main()
