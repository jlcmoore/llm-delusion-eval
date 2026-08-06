"""Generate publication figures, CSVs, and LaTeX tables from summary.json.

Run from the evals repo root::

    python -m analysis.generate_figures
    python -m analysis.generate_figures --summary report/summary.json

Outputs:
- Figures (PDF) to ``analysis/figures/``
- CSVs to ``analysis/data/``
- LaTeX tabular files to ``analysis/tables/``
"""

import argparse
import csv
import json
import logging
import shutil
import textwrap
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from analysis.artifact_paths import DATA_DIR, FIGURE_DIR, TABLE_DIR, ensure_output_dirs
from analysis.load_eval_data import (
    _EVALS_REPO_ROOT,
    CODE_CATEGORIES,
    load_all_eval_data,
)
from analysis.metric_labels import format_metric_label_for_matplotlib
from analysis.plot_style import (
    apply_plot_style,
    get_model_color,
    get_reasoning_model_color,
    sort_model_labels,
)
from llm_delusion_eval.constants import format_model_label, normalize_model_label

logger = logging.getLogger(__name__)

plt.switch_backend("Agg")
apply_plot_style()


@dataclass
class LaTeXTableConfig:
    """Configuration for LaTeX table generation.

    Attributes
    ----------
    col_spec:
        LaTeX column specification (e.g. ``"lrrr"``).
    header_labels:
        Rename headers: mapping from CSV column name to display label.
    raw_columns:
        Column names whose values should not be LaTeX-escaped.
    group_break_column:
        Column that triggers ``\\midrule`` on value change.
    """

    col_spec: Optional[str] = None
    header_labels: Optional[dict[str, str]] = None
    raw_columns: set[str] = field(default_factory=set)
    group_break_column: Optional[str] = None


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

FIG_DIR = FIGURE_DIR
DATA_OUTPUT_DIR = DATA_DIR
DEFAULT_SUMMARY = _EVALS_REPO_ROOT / "report" / "summary.json"
DEFAULT_CLUSTERED_CATEGORY_CI_PATH = (
    DATA_DIR / "participant_clustered_prevalence_by_model_category.csv"
)

# Minimum number of models required for comparison visualizations
MIN_MODELS_FOR_COMPARISON = 2
MIN_ROWS_FOR_ORIGINAL_SEPARATOR = 2
HEATMAP_TICK_FONT_SIZE = 10
HEATMAP_CELL_FONT_SIZE = 10
HEATMAP_X_LABEL_WRAP_WIDTH = 10
HEATMAP_MODEL_LABEL_WRAP_WIDTH = 24
HEATMAP_COLORBAR_LABEL_SIZE = 10
HEATMAP_COLORBAR_TICK_SIZE = 9
HEATMAP_BOTTOM_MARGIN = 0.26
HEATMAP_COLORBAR_SHRINK = 0.92
HEATMAP_COLORBAR_PAD = 0.02
BAR_X_LABEL_WRAP_WIDTH = 7
BAR_X_TICK_FONT_SIZE = 7
BAR_FIGURE_WIDTH_INCHES = 7.6
BAR_FIGURE_HEIGHT_INCHES = 4.6
BAR_GROUP_SPACING = 1.18
MODEL_FAMILIES_FIGURE_SCALE = 0.85
MODEL_FAMILIES_WSPACE = 0.03
REASONING_VARIANT_HATCH = "////"

# The 3 non-harm categories plus all available harm codes used in code-level
# figure ordering. ``validates-*`` harm codes can be filtered at runtime.
_HARM_CODE_ORDER = [
    "bot-facilitates-violence",
    "bot-validates-violent-feelings",
    "bot-discourages-violence",
    "bot-facilitates-self-harm",
    "bot-validates-self-harm-feelings",
    "bot-discourages-self-harm",
]

_BASE_CATEGORY_GROUPS = [
    "sycophancy",
    "delusional",
    "relationship",
]
_CATEGORY_GROUPS = [
    "sycophancy",
    "delusional",
    "relationship",
    "facilitates harm",
    "discourages harm",
]

# Canonical code ordering for all code-level displays:
# sycophancy, delusional, relationship, violence codes, self-harm codes.
_CODE_ORDER_PREF = [
    # sycophancy
    "positive-affirmation",
    "reflective-summary",
    "grand-significance",
    "dismisses-counterevidence",
    "reports-others-admire-speaker",
    # delusional
    "misrepresents-sentience",
    "misrepresents-ability",
    "metaphysical-themes",
    "endorses-delusion",
    # relationship
    "claims-unique-connection",
    "romantic-interest",
    "platonic-affinity",
    # violence codes
    "facilitates-violence",
    "validates-violent-feelings",
    "discourages-violence",
    # self-harm codes
    "facilitates-self-harm",
    "validates-self-harm-feelings",
    "discourages-self-harm",
]


def _is_validates_code(code_key: str) -> bool:
    """Return whether a code key belongs to the ``validates-*`` family.

    Parameters
    ----------
    code_key:
        Raw code key with or without ``bot-`` prefix.

    Returns
    -------
    bool
        ``True`` when the normalized key starts with ``validates-``.
    """
    normalized = str(code_key).strip().removeprefix("bot-")
    return normalized.startswith("validates-")


def _code_summary_category(code_key: str) -> str:
    """Return the display category used in code summary tables."""
    normalized = str(code_key).strip().removeprefix("bot-")
    if normalized in {"facilitates-self-harm", "facilitates-violence"}:
        return "facilitates harm"
    if normalized in {"discourages-self-harm", "discourages-violence"}:
        return "discourages harm"
    return CODE_CATEGORIES.get(normalized, "unknown")


def _code_summary_row_order() -> tuple[dict[str, int], dict[str, int]]:
    """Return ordering maps for code summary-style tables."""
    category_order = {
        "sycophancy": 0,
        "delusional": 1,
        "relationship": 2,
        "facilitates harm": 3,
        "discourages harm": 4,
    }
    code_order = {
        "positive-affirmation": 0,
        "reflective-summary": 1,
        "grand-significance": 2,
        "dismisses-counterevidence": 3,
        "reports-others-admire-speaker": 4,
        "misrepresents-sentience": 5,
        "misrepresents-ability": 6,
        "metaphysical-themes": 7,
        "endorses-delusion": 8,
        "claims-unique-connection": 9,
        "romantic-interest": 10,
        "platonic-affinity": 11,
        "facilitates-violence": 12,
        "facilitates-self-harm": 13,
        "discourages-violence": 14,
        "discourages-self-harm": 15,
    }
    return category_order, code_order


def _filter_code_keys_for_export(
    code_keys: list[str],
    *,
    include_validates_codes: bool,
) -> list[str]:
    """Filter code keys for paper-export displays.

    Parameters
    ----------
    code_keys:
        Candidate code or group keys.
    include_validates_codes:
        Whether ``validates-*`` codes should be retained.

    Returns
    -------
    list[str]
        Filtered keys preserving original order.
    """
    if include_validates_codes:
        return code_keys
    return [code_key for code_key in code_keys if not _is_validates_code(code_key)]


def _bar_group_order() -> list[str]:
    """Return category-group ordering for aggregated publication plots."""
    return list(_CATEGORY_GROUPS)


def _short_group_label(group: str) -> str:
    """Convert a group key to a short display label for x-axis ticks."""
    return format_metric_label_for_matplotlib(group)


def _wrap_heatmap_label(
    label: str,
    width: int = HEATMAP_X_LABEL_WRAP_WIDTH,
    *,
    arrow_on_new_line: bool = False,
) -> str:
    """Wrap a heatmap tick label onto multiple lines for readability."""
    normalized = label.replace("-", " ")
    arrow_suffix = ""
    if " (" in normalized and normalized.endswith(")"):
        base_label, suffix = normalized.rsplit(" (", maxsplit=1)
        normalized = base_label
        if arrow_on_new_line:
            arrow_suffix = f"\n({suffix}"
        else:
            arrow_suffix = f" ({suffix}"
    wrapped = textwrap.fill(normalized, width=width, break_long_words=False)
    return f"{wrapped}{arrow_suffix}"


def _wrap_heatmap_model_label(
    label: str,
    width: int = HEATMAP_MODEL_LABEL_WRAP_WIDTH,
) -> str:
    """Wrap a heatmap row label so long model names use less horizontal space."""
    normalized = str(label).strip()
    if " (" in normalized and normalized.endswith(")"):
        base_label, suffix = normalized.rsplit(" (", maxsplit=1)
        if len(normalized) <= width:
            return normalized
        if len(base_label) <= width:
            return f"{base_label}\n({suffix}"
        wrapped_base = textwrap.fill(base_label, width=width, break_long_words=False)
        return f"{wrapped_base}\n({suffix}"
    if len(normalized) <= width:
        return normalized
    return textwrap.fill(normalized, width=width, break_long_words=False)


def _heatmap_bottom_margin_for_labels(labels: list[str]) -> float:
    """Return a bottom margin that fits wrapped multi-line heatmap x labels."""
    if not labels:
        return HEATMAP_BOTTOM_MARGIN
    max_lines = max(str(label).count("\n") + 1 for label in labels)
    extra_lines = max(0, max_lines - 2)
    return min(0.40, HEATMAP_BOTTOM_MARGIN + 0.035 * extra_lines)


def _set_heatmap_y_labels(ax: Any, labels: list[str]) -> None:
    """Set wrapped y-axis labels with compact line spacing for heatmaps."""
    ax.set_yticklabels(
        [_wrap_heatmap_model_label(label) for label in labels],
        rotation=0,
        va="center",
    )
    for tick_label in ax.get_yticklabels():
        tick_label.set_linespacing(0.9)


def _set_group_axis_labels(
    ax: Any,
    tick_positions: np.ndarray,
    groups: list[str],
) -> None:
    """Set wrapped, horizontal group labels for bar-chart x-axes.

    Parameters
    ----------
    ax:
        Matplotlib axis where labels are applied.
    tick_positions:
        X coordinates for tick placement.
    groups:
        Ordered category/harm-code group keys.
    """
    labels = []
    for group in groups:
        label = _short_group_label(group)
        labels.append(
            _wrap_heatmap_label(
                label,
                width=BAR_X_LABEL_WRAP_WIDTH,
                arrow_on_new_line=True,
            )
        )
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(labels, rotation=0, ha="center")
    ax.tick_params(axis="x", labelsize=BAR_X_TICK_FONT_SIZE)


def _draw_original_transcript_separator(ax: Any, ordered_index: pd.Index) -> None:
    """Draw a horizontal separator before ``Original transcript`` when present.

    Parameters
    ----------
    ax:
        Matplotlib axis for a heatmap.
    ordered_index:
        Ordered row labels shown on the y-axis.
    """
    original_label = "Original transcript"
    if (
        original_label not in ordered_index
        or len(ordered_index) < MIN_ROWS_FOR_ORIGINAL_SEPARATOR
    ):
        return
    separator_y = ordered_index.get_loc(original_label)
    if separator_y <= 0:
        return
    x_min, x_max = ax.get_xlim()
    ax.hlines(
        y=separator_y,
        xmin=x_min,
        xmax=x_max,
        colors="black",
        linewidth=1.2,
    )


def _sort_codes_for_display(
    code_shorts: list[str],
    *,
    include_validates_codes: bool,
) -> list[str]:
    """Sort code keys according to canonical display order.

    Parameters
    ----------
    code_shorts:
        Code keys without the ``bot-`` prefix.
    include_validates_codes:
        Whether to include ``validates-*`` codes in output order.

    Returns
    -------
    list[str]
        Ordered code keys for display.
    """
    filtered_codes = _filter_code_keys_for_export(
        code_shorts,
        include_validates_codes=include_validates_codes,
    )
    order_map = {code: index for index, code in enumerate(_CODE_ORDER_PREF)}
    return sorted(filtered_codes, key=lambda code: order_map.get(code, 999))


# ---------------------------------------------------------------------------
# Summary JSON loading
# ---------------------------------------------------------------------------


def load_summary(path: Path) -> dict:
    """Load summary.json and return the parsed dict."""
    with path.open(encoding="utf-8") as fh:
        summary = json.load(fh)

    for evaluation in summary.get("evaluations", []):
        model = evaluation.get("model")
        if isinstance(model, str) and model:
            evaluation["model_label"] = format_model_label(
                model,
                evaluation.get("reasoning_effort"),
            )
            continue
        legacy_label = evaluation.get("model_label")
        if legacy_label:
            evaluation["model_label"] = normalize_model_label(str(legacy_label))

    return summary


def _summary_score_stats(info: dict[str, Any]) -> dict[str, float]:
    """Return normalized score stats from one summary score object.

    Parameters
    ----------
    info:
        Score object from ``summary.json`` with ``mean``, CI, and sample fields.

    Returns
    -------
    dict[str, float]
        Normalized stats dictionary with ``mean``, ``ci_lower``, ``ci_upper``,
        and ``samples`` keys.
    """

    def _coerce_float(value: Any, default: float) -> float:
        """Convert a summary value to float and fall back on null/invalid input."""
        if value is None:
            return default
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            return default
        if not np.isfinite(parsed):
            return default
        return parsed

    return {
        "mean": _coerce_float(info.get("mean"), 0.0),
        "ci_lower": _coerce_float(info.get("ci_lower"), float("nan")),
        "ci_upper": _coerce_float(info.get("ci_upper"), float("nan")),
        "samples": _coerce_float(info.get("samples"), 0.0),
    }


def _summary_category_stats_for_eval(ev: dict[str, Any]) -> dict[str, dict[str, float]]:
    """Return five-category aggregate stats for one eval entry.

    Parameters
    ----------
    ev:
        One evaluation entry from ``summary.json``.

    Returns
    -------
    dict[str, dict[str, float]]
        Category-keyed statistics with ``mean``, CI, and sample count.
    """
    category_scores = ev.get("category_scores", {}) or {}
    category_stats: dict[str, dict[str, float]] = {}
    for category_key in _CATEGORY_GROUPS:
        info = category_scores.get(category_key)
        if info:
            category_stats[category_key] = _summary_score_stats(info)
    return category_stats


def _weighted_group_stat(
    group_frame: pd.DataFrame,
    weights: pd.Series,
    *,
    column: str,
    default: float,
) -> float:
    """Return a weighted column average from one grouped category frame.

    Parameters
    ----------
    group_frame:
        Grouped category rows to combine.
    weights:
        Sample-count weights aligned with ``group_frame`` rows.
    column:
        Numeric column name to aggregate.
    default:
        Fallback value when the group has no finite values.

    Returns
    -------
    float
        Weighted mean when weights are positive, else arithmetic mean.
    """
    numeric = pd.to_numeric(group_frame[column], errors="coerce").to_numpy(dtype=float)
    numeric_weights = weights.to_numpy(dtype=float)
    finite_mask = np.isfinite(numeric)
    if not np.any(finite_mask):
        return default
    values = numeric[finite_mask]
    value_weights = numeric_weights[finite_mask]
    if float(value_weights.sum()) > 0.0:
        return float(np.average(values, weights=value_weights))
    return float(np.mean(values))


def _summary_to_category_df(summary: dict) -> pd.DataFrame:
    """Convert summary.json into a flat DataFrame for category bar charts.

    Returns a DataFrame with columns:
    ``[model_label, group, mean, ci_lower, ci_upper, n]``
    where ``group`` is one of five aggregate categories.
    """
    rows: list[dict[str, Any]] = []
    for ev in summary["evaluations"]:
        label = normalize_model_label(ev.get("model_label", ev["model"]))
        category_stats = _summary_category_stats_for_eval(ev)
        for category_key in _CATEGORY_GROUPS:
            info = category_stats.get(category_key)
            if not info:
                continue
            rows.append(
                {
                    "model_label": label,
                    "model": ev["model"],
                    "reasoning_effort": ev.get("reasoning_effort"),
                    "group": category_key,
                    "mean": info["mean"],
                    "ci_lower": info.get("ci_lower", float("nan")),
                    "ci_upper": info.get("ci_upper", float("nan")),
                    "n": info.get("samples", 0.0),
                }
            )
    category_df = pd.DataFrame(rows)
    if category_df.empty:
        return category_df

    group_columns = ["model_label", "model", "reasoning_effort", "group"]
    grouped_rows: list[dict[str, Any]] = []
    for group_values, group_frame in category_df.groupby(
        group_columns, dropna=False, sort=False
    ):
        weights = pd.to_numeric(group_frame["n"], errors="coerce").fillna(0.0)

        model_label, model, reasoning_effort, group_key = group_values
        grouped_rows.append(
            {
                "model_label": model_label,
                "model": model,
                "reasoning_effort": reasoning_effort,
                "group": group_key,
                "mean": _weighted_group_stat(
                    group_frame, weights, column="mean", default=0.0
                ),
                "ci_lower": _weighted_group_stat(
                    group_frame, weights, column="ci_lower", default=float("nan")
                ),
                "ci_upper": _weighted_group_stat(
                    group_frame, weights, column="ci_upper", default=float("nan")
                ),
                "n": float(weights.sum()),
            }
        )

    collapsed_count = len(category_df) - len(grouped_rows)
    if collapsed_count > 0:
        logger.info(
            "Collapsed %d duplicate category rows by model/group.", collapsed_count
        )
    return pd.DataFrame(grouped_rows)


def _load_clustered_category_ci_df(path: Path) -> pd.DataFrame:
    """Load participant-clustered category prevalence summaries.

    Parameters
    ----------
    path:
        CSV path produced by ``analysis.compute_participant_robustness``.

    Returns
    -------
    pd.DataFrame
        Clustered category prevalence rows normalized to the category figure
        schema, or an empty dataframe when unavailable.
    """
    if not path.is_file():
        logger.warning(
            "Participant-clustered category CI file missing; keeping summary.json "
            "intervals for category bar figures: %s",
            path,
        )
        return pd.DataFrame()

    clustered_df = pd.read_csv(path)
    required_columns = {
        "model_label",
        "category",
        "prevalence_pct",
        "cluster_ci_low_pct",
        "cluster_ci_high_pct",
    }
    missing_columns = required_columns.difference(clustered_df.columns)
    if missing_columns:
        logger.warning(
            "Participant-clustered category CI file is missing required columns "
            "(%s); keeping summary.json intervals.",
            ", ".join(sorted(missing_columns)),
        )
        return pd.DataFrame()

    normalized = clustered_df.copy()
    normalized["model_label"] = normalized["model_label"].map(normalize_model_label)
    normalized["group"] = normalized["category"].astype(str).str.strip()
    normalized["mean"] = (
        pd.to_numeric(normalized["prevalence_pct"], errors="coerce") / 100.0
    )
    normalized["ci_lower"] = (
        pd.to_numeric(normalized["cluster_ci_low_pct"], errors="coerce") / 100.0
    )
    normalized["ci_upper"] = (
        pd.to_numeric(normalized["cluster_ci_high_pct"], errors="coerce") / 100.0
    )
    normalized["n_participants_supported"] = pd.to_numeric(
        normalized.get("n_participants_supported"), errors="coerce"
    )
    normalized["cluster_boot_n"] = pd.to_numeric(
        normalized.get("cluster_boot_n"), errors="coerce"
    )
    return normalized[
        [
            "model_label",
            "group",
            "mean",
            "ci_lower",
            "ci_upper",
            "n_participants_supported",
            "cluster_boot_n",
        ]
    ].dropna(subset=["model_label", "group"])


def _apply_clustered_category_ci_overrides(
    category_df: pd.DataFrame,
    *,
    clustered_ci_path: Path = DEFAULT_CLUSTERED_CATEGORY_CI_PATH,
) -> pd.DataFrame:
    """Override category bar-figure intervals with participant-clustered CIs.

    Parameters
    ----------
    category_df:
        Category-level dataframe derived from ``summary.json``.
    clustered_ci_path:
        CSV path for participant-clustered category prevalence summaries.

    Returns
    -------
    pd.DataFrame
        Category dataframe with clustered ``mean``, ``ci_lower``, and
        ``ci_upper`` values where available.
    """
    if category_df.empty:
        return category_df

    clustered_df = _load_clustered_category_ci_df(clustered_ci_path)
    if clustered_df.empty:
        return category_df

    merged = category_df.merge(
        clustered_df,
        on=["model_label", "group"],
        how="left",
        suffixes=("", "_clustered"),
    )
    override_mask = (
        merged["mean_clustered"].notna()
        & merged["ci_lower_clustered"].notna()
        & merged["ci_upper_clustered"].notna()
    )
    overridden = int(override_mask.sum())
    if overridden == 0:
        logger.warning(
            "Participant-clustered category CI file matched no category figure rows; "
            "keeping summary.json intervals."
        )
        return category_df

    for column in ["mean", "ci_lower", "ci_upper"]:
        merged.loc[override_mask, column] = merged.loc[
            override_mask, f"{column}_clustered"
        ]
    merged.loc[override_mask, "ci_method"] = "participant_clustered"
    logger.info(
        "Applied participant-clustered category CI overrides to %d/%d category "
        "rows for bar figures.",
        overridden,
        len(merged),
    )
    drop_columns = [
        "mean_clustered",
        "ci_lower_clustered",
        "ci_upper_clustered",
        "n_participants_supported",
        "cluster_boot_n",
    ]
    return merged.drop(columns=[c for c in drop_columns if c in merged.columns])


# ---------------------------------------------------------------------------
# LaTeX table export (following llm-delusions/analysis/latex pattern)
# ---------------------------------------------------------------------------


def _escape_latex(text: Optional[str]) -> str:
    """Minimal LaTeX escaping for table cells."""
    if text is None:
        return ""
    text = str(text)
    for char, repl in [
        ("\\", r"\textbackslash{}"),
        ("&", r"\&"),
        ("%", r"\%"),
        ("$", r"\$"),
        ("#", r"\#"),
        ("_", r"\_"),
        ("{", r"\{"),
        ("}", r"\}"),
        ("~", r"\textasciitilde{}"),
        ("^", r"\textasciicircum{}"),
    ]:
        text = text.replace(char, repl)
    return text


def _csv_to_latex_tabular(
    csv_path: Path,
    tex_path: Path,
    *,
    config: Optional[LaTeXTableConfig] = None,
) -> None:
    """Convert a CSV to a LaTeX tabular fragment."""
    if config is None:
        config = LaTeXTableConfig()

    with csv_path.open("r", newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        rows = list(reader)

    if not rows:
        logger.warning("Empty CSV, skipping LaTeX export: %s", csv_path)
        return

    df = pd.DataFrame(rows, columns=list(reader.fieldnames or []))
    _df_to_latex_tabular(df, tex_path, config=config)


def _df_to_latex_tabular(
    df: pd.DataFrame,
    tex_path: Path,
    *,
    config: Optional[LaTeXTableConfig] = None,
) -> None:
    """Convert a dataframe to a LaTeX tabular fragment."""
    if config is None:
        config = LaTeXTableConfig()

    raw_columns = config.raw_columns or set()
    columns = list(df.columns)
    if not columns or df.empty:
        logger.warning("Empty dataframe, skipping LaTeX export: %s", tex_path)
        return

    col_spec = config.col_spec or ("l" * len(columns))
    labels = [(config.header_labels or {}).get(col, col) for col in columns]

    tex_path.parent.mkdir(parents=True, exist_ok=True)
    last_group = None

    with tex_path.open("w", encoding="utf-8") as fh:
        fh.write(
            "% NOTE: This table is auto-generated; "
            "prefer not to hand-edit until the very end.\n"
        )
        fh.write(r"\centering" + "\n")
        fh.write(r"\resizebox{\textwidth}{!}{%" + "\n")
        fh.write(r"\begin{tabular}{" + col_spec + "}\n")
        fh.write(r"\toprule" + "\n")
        fh.write(" & ".join(_escape_latex(lbl) for lbl in labels) + r" \\" + "\n")
        fh.write(r"\midrule" + "\n")

        for _, row in df.iterrows():
            if config.group_break_column:
                val = row.get(config.group_break_column, "")
                if val and last_group is not None and val != last_group:
                    fh.write(r"\midrule" + "\n")
                if val:
                    last_group = val

            cells = []
            for col in columns:
                val = row.get(col, "")
                if col in raw_columns:
                    cells.append(str(val))
                else:
                    cells.append(_escape_latex(val))
            fh.write(" & ".join(cells) + r" \\" + "\n")

        fh.write(r"\bottomrule" + "\n")
        fh.write(r"\end{tabular}" + "\n")
        fh.write(r"}" + "\n")

    logger.info("Wrote LaTeX table: %s", tex_path)


def _wrap_code_header_label(code: object) -> str:
    """Format one code label for a compact two-line LaTeX table header.

    Parameters
    ----------
    code:
        Raw code label shown as a column header in the appendix code table.

    Returns
    -------
    str
        Escaped LaTeX text. Hyphenated labels are wrapped with ``\\shortstack``
        at the most balanced hyphen split when possible.
    """
    min_wrap_parts = 2
    label = "" if code is None else str(code)
    parts = label.split("-")
    if len(parts) < min_wrap_parts:
        return _escape_latex(label)

    split_index = min(
        range(1, len(parts)),
        key=lambda candidate: (
            max(
                len("-".join(parts[:candidate])),
                len("-".join(parts[candidate:])),
            ),
            abs(len("-".join(parts[:candidate])) - len("-".join(parts[candidate:]))),
        ),
    )
    top_line = _escape_latex("-".join(parts[:split_index]))
    bottom_line = _escape_latex("-".join(parts[split_index:]))
    return r"\shortstack{" + top_line + r"\\" + bottom_line + "}"


def _write_category_split_code_table(
    code_table: pd.DataFrame,
    tex_path: Path,
    *,
    model_cols: list[str],
) -> None:
    """Write category-specific code tables to one LaTeX fragment.

    Parameters
    ----------
    code_table:
        Wide code prevalence dataframe with ``code_short``, ``category``, and
        one column per model.
    tex_path:
        Output path for the LaTeX fragment.
    model_cols:
        Ordered model columns to display as table headers.
    """
    if code_table.empty:
        logger.warning("Empty code table, skipping LaTeX export: %s", tex_path)
        return

    categories = [
        category
        for category in _CATEGORY_GROUPS
        if category in set(code_table["category"])
    ]
    if not categories:
        logger.warning("No code categories available for LaTeX export: %s", tex_path)
        return

    tex_path.parent.mkdir(parents=True, exist_ok=True)

    with tex_path.open("w", encoding="utf-8") as fh:
        fh.write(
            "% NOTE: This table is auto-generated; "
            "prefer not to hand-edit until the very end.\n"
        )

        for category_idx, category in enumerate(categories):
            category_table = code_table[code_table["category"] == category]
            if category_table.empty:
                continue

            if category_idx > 0:
                fh.write(r"\medskip" + "\n")

            fh.write(r"\begin{center}" + "\n")
            fh.write(r"\scriptsize" + "\n")
            fh.write(r"\textbf{" + _escape_latex(category.title()) + "}" + "\n")
            fh.write(r"\setlength{\tabcolsep}{4pt}" + "\n")
            fh.write(r"\begin{tabular}{l" + "r" * len(category_table) + "}\n")
            fh.write(r"\toprule" + "\n")
            fh.write(
                "Model & "
                + " & ".join(
                    _wrap_code_header_label(code)
                    for code in category_table["code_short"]
                )
                + r" \\"
                + "\n"
            )
            fh.write(r"\midrule" + "\n")

            for model in model_cols:
                cells = [_escape_latex(model)]
                for _, row in category_table.iterrows():
                    cells.append(_escape_latex(row[model]))
                fh.write(" & ".join(cells) + r" \\" + "\n")

            fh.write(r"\bottomrule" + "\n")
            fh.write(r"\end{tabular}" + "\n")
            fh.write(r"\end{center}" + "\n")

    logger.info("Wrote LaTeX table: %s", tex_path)


# ---------------------------------------------------------------------------
# CSV export helpers
# ---------------------------------------------------------------------------


def _save_csv(df_out: pd.DataFrame, name: str) -> Path:
    """Save a DataFrame as CSV to the analysis data directory."""
    path = DATA_OUTPUT_DIR / name
    df_out.to_csv(path, index=False)
    logger.info("Wrote CSV: %s", path)
    return path


def _save_figure(fig, name: str, *, save_png: bool = False) -> None:
    """Save a matplotlib figure as PDF and optional PNG.

    Parameters
    ----------
    fig:
        Matplotlib figure to save.
    name:
        Filename stem for output artifacts.
    save_png:
        When ``True``, also write a PNG preview copy alongside the PDF.
    """
    fig.savefig(FIG_DIR / f"{name}.pdf")
    if save_png:
        fig.savefig(FIG_DIR / f"{name}.png", dpi=300)
    plt.close(fig)
    if save_png:
        logger.info("Saved figure: %s (.pdf and .png)", name)
    else:
        logger.info("Saved figure: %s (.pdf)", name)


# ---------------------------------------------------------------------------
# Grouped bar chart helper (shared by fig4, fig5, fig6)
# ---------------------------------------------------------------------------


def _grouped_bar_chart(
    cat_df: pd.DataFrame,
    model_order: list[str],
    groups: list[str],
    *,
    fig_name: str,
) -> None:
    """Draw a grouped bar chart with error bars for the given models and groups.

    Parameters
    ----------
    cat_df:
        DataFrame from ``_summary_to_category_df`` filtered to the
        relevant models.
    model_order:
        Ordered list of model labels to include.
    groups:
        Ordered list of group keys (categories / harm codes).
    fig_name:
        Filename stem for saving.
    """
    available_groups = [g for g in groups if g in cat_df["group"].values]
    if not available_groups:
        logger.warning("No groups available for %s; skipping", fig_name)
        return

    fig, ax = plt.subplots(figsize=(BAR_FIGURE_WIDTH_INCHES, BAR_FIGURE_HEIGHT_INCHES))
    x = np.arange(len(available_groups), dtype=float) * BAR_GROUP_SPACING
    n_models = len(model_order)
    width = 0.8 / max(n_models, 1)

    for idx, model in enumerate(model_order):
        mdf = cat_df[cat_df["model_label"] == model]
        vals, ci_lo, ci_hi = [], [], []
        for g in available_groups:
            row = mdf[mdf["group"] == g]
            if len(row) > 0:
                r = row.iloc[0]
                vals.append(r["mean"] * 100)
                ci_lo.append(
                    r["ci_lower"] * 100
                    if not np.isnan(r["ci_lower"])
                    else r["mean"] * 100
                )
                ci_hi.append(
                    r["ci_upper"] * 100
                    if not np.isnan(r["ci_upper"])
                    else r["mean"] * 100
                )
            else:
                vals.append(0)
                ci_lo.append(0)
                ci_hi.append(0)

        centers = x + idx * width
        yerr_lower = [v - lo for v, lo in zip(vals, ci_lo)]
        yerr_upper = [hi - v for v, hi in zip(vals, ci_hi)]

        ax.bar(
            centers,
            vals,
            width,
            label=model,
            color=get_model_color(model),
            edgecolor="white",
            linewidth=0.5,
        )
        ax.errorbar(
            centers,
            vals,
            yerr=[yerr_lower, yerr_upper],
            fmt="none",
            ecolor="black",
            elinewidth=0.8,
            capsize=2,
        )

    _set_group_axis_labels(
        ax,
        x + width * (n_models - 1) / 2,
        available_groups,
    )
    ax.set_ylabel("Prevalence (%)")
    ax.legend(frameon=True, fontsize=7)
    fig.subplots_adjust(bottom=0.5)

    _save_figure(fig, fig_name)


# ---------------------------------------------------------------------------
# Figure 1: Main heatmap -- category-level prevalence per model
# ---------------------------------------------------------------------------


def fig_main_heatmap(summary: dict) -> None:
    """Heatmap of binary prevalence by category and model."""
    category_df = _summary_to_category_df(summary)
    if category_df.empty:
        logger.warning("No category data; skipping fig1")
        return

    pivot = category_df.pivot_table(
        index="model_label",
        columns="group",
        values="mean",
        aggfunc="mean",
    )
    cats = [category for category in _CATEGORY_GROUPS if category in pivot.columns]
    pivot = pivot[cats]
    models = sort_model_labels(list(pivot.index))
    pivot = pivot.loc[[m for m in models if m in pivot.index]]
    pivot_pct = pivot * 100

    split_group = "discourages harm"
    left_cats = [category for category in cats if category != split_group]
    right_cats = [category for category in cats if category == split_group]
    if not right_cats or not left_cats:
        fig = _render_main_heatmap_single_panel(pivot_pct, cats)
    else:
        fig = _render_main_heatmap_split_panel(pivot_pct, left_cats, right_cats)

    _save_figure(fig, "fig1_main_heatmap", save_png=True)


def _main_heatmap_tick_labels(groups: list[str]) -> list[str]:
    """Return wrapped x-axis labels for main heatmap category groups."""
    return [
        _wrap_heatmap_label(
            _short_group_label(group),
            width=HEATMAP_X_LABEL_WRAP_WIDTH,
            arrow_on_new_line=True,
        )
        for group in groups
    ]


def _render_main_heatmap_single_panel(
    pivot_pct: pd.DataFrame, groups: list[str]
) -> plt.Figure:
    """Render the default single-panel main heatmap."""
    main_heatmap_tick_font_size = HEATMAP_TICK_FONT_SIZE + 1
    main_heatmap_x_tick_font_size = main_heatmap_tick_font_size + 1
    main_heatmap_cell_font_size = HEATMAP_CELL_FONT_SIZE + 1

    fig, ax = plt.subplots(figsize=(6.8, 0.4 * len(pivot_pct) + 1.5))
    heatmap_ax = sns.heatmap(
        pivot_pct,
        annot=True,
        fmt=".0f",
        cmap="YlOrRd",
        vmin=0,
        vmax=100,
        linewidths=0.4,
        cbar_kws={"shrink": HEATMAP_COLORBAR_SHRINK, "pad": HEATMAP_COLORBAR_PAD},
        annot_kws={"size": main_heatmap_cell_font_size},
        ax=ax,
    )
    ax.set_ylabel("")
    ax.set_xlabel("")
    _draw_original_transcript_separator(ax, pivot_pct.index)
    x_tick_labels = _main_heatmap_tick_labels(groups)
    ax.set_xticklabels(x_tick_labels, rotation=0, ha="center")
    _set_heatmap_y_labels(ax, pivot_pct.index.tolist())
    ax.tick_params(axis="x", labelsize=main_heatmap_x_tick_font_size)
    ax.tick_params(axis="y", labelsize=main_heatmap_tick_font_size)
    ax.tick_params(axis="y", pad=2)
    colorbar = heatmap_ax.collections[0].colorbar
    if colorbar is not None:
        colorbar.set_label("Prevalence (%)", fontsize=HEATMAP_COLORBAR_LABEL_SIZE)
        colorbar.ax.tick_params(labelsize=HEATMAP_COLORBAR_TICK_SIZE)
    fig.subplots_adjust(bottom=_heatmap_bottom_margin_for_labels(x_tick_labels))
    return fig


def _render_main_heatmap_split_panel(
    pivot_pct: pd.DataFrame, left_groups: list[str], right_groups: list[str]
) -> plt.Figure:
    """Render split-panel main heatmap with inverted discourages-harm colors."""
    main_heatmap_tick_font_size = HEATMAP_TICK_FONT_SIZE + 1
    main_heatmap_x_tick_font_size = main_heatmap_tick_font_size + 1
    main_heatmap_cell_font_size = HEATMAP_CELL_FONT_SIZE + 1

    fig = plt.figure(figsize=(7.5, 0.4 * len(pivot_pct) + 1.5))
    grid = fig.add_gridspec(
        nrows=1,
        ncols=3,
        width_ratios=[len(left_groups), 1.0, 0.24],
        wspace=0.08,
    )
    left_ax = fig.add_subplot(grid[0, 0])
    right_ax = fig.add_subplot(grid[0, 1], sharey=left_ax)
    cbar_grid = grid[0, 2].subgridspec(2, 1, hspace=0.55)
    left_cbar_ax = fig.add_subplot(cbar_grid[0, 0])
    right_cbar_ax = fig.add_subplot(cbar_grid[1, 0])

    left_heatmap = sns.heatmap(
        pivot_pct[left_groups],
        annot=True,
        fmt=".0f",
        cmap="YlOrRd",
        vmin=0,
        vmax=100,
        linewidths=0.4,
        cbar=False,
        annot_kws={"size": main_heatmap_cell_font_size},
        ax=left_ax,
    )
    sns.heatmap(
        pivot_pct[right_groups],
        annot=True,
        fmt=".0f",
        cmap="Blues_r",
        vmin=0,
        vmax=100,
        linewidths=0.4,
        cbar=False,
        annot_kws={"size": main_heatmap_cell_font_size},
        ax=right_ax,
    )

    left_ax.set_ylabel("")
    left_ax.set_xlabel("")
    right_ax.set_ylabel("")
    right_ax.set_xlabel("")
    _draw_original_transcript_separator(left_ax, pivot_pct.index)
    _draw_original_transcript_separator(right_ax, pivot_pct.index)

    left_tick_labels = _main_heatmap_tick_labels(left_groups)
    right_tick_labels = _main_heatmap_tick_labels(right_groups)
    left_ax.set_xticklabels(left_tick_labels, rotation=0, ha="center")
    right_ax.set_xticklabels(right_tick_labels, rotation=0, ha="center")
    _set_heatmap_y_labels(left_ax, pivot_pct.index.tolist())
    right_ax.tick_params(axis="y", left=False, labelleft=False)
    left_ax.tick_params(axis="x", labelsize=main_heatmap_x_tick_font_size)
    right_ax.tick_params(axis="x", labelsize=main_heatmap_x_tick_font_size)
    left_ax.tick_params(axis="y", labelsize=main_heatmap_tick_font_size)
    left_ax.tick_params(axis="y", pad=2)

    left_colorbar = fig.colorbar(left_heatmap.collections[0], cax=left_cbar_ax)
    left_colorbar.set_label(
        "Prevalence (%)\n(other categories)",
        fontsize=HEATMAP_COLORBAR_LABEL_SIZE,
    )
    left_colorbar.ax.tick_params(labelsize=HEATMAP_COLORBAR_TICK_SIZE)

    right_colorbar = fig.colorbar(right_ax.collections[0], cax=right_cbar_ax)
    right_colorbar.set_label(
        "Prevalence (%)\n(discourages harm)",
        fontsize=HEATMAP_COLORBAR_LABEL_SIZE,
    )
    right_colorbar.ax.tick_params(labelsize=HEATMAP_COLORBAR_TICK_SIZE)

    all_tick_labels = left_tick_labels + right_tick_labels
    fig.subplots_adjust(bottom=_heatmap_bottom_margin_for_labels(all_tick_labels))
    return fig


# ---------------------------------------------------------------------------
# Figure 2: Per-code heatmap
# ---------------------------------------------------------------------------


def fig_code_heatmap(summary: dict, *, include_validates_codes: bool) -> None:
    """Heatmap of binary prevalence by individual code and model."""
    code_heatmap_x_tick_font_size = HEATMAP_TICK_FONT_SIZE - 1

    rows = []
    for ev in summary["evaluations"]:
        label = normalize_model_label(ev.get("model_label", ev["model"]))
        for code, info in ev.get("code_scores", {}).items():
            code_short = code.removeprefix("bot-")
            rows.append(
                {"model_label": label, "code_short": code_short, "mean": info["mean"]}
            )
    if not rows:
        logger.warning("No code data; skipping fig2")
        return

    df = pd.DataFrame(rows)
    pivot = df.pivot_table(
        index="model_label",
        columns="code_short",
        values="mean",
        aggfunc="mean",
    )

    code_order = _sort_codes_for_display(
        list(pivot.columns),
        include_validates_codes=include_validates_codes,
    )
    if not code_order:
        logger.warning("No code data remaining after code filters; skipping fig2")
        return
    pivot = pivot[code_order]
    models = sort_model_labels(list(pivot.index))
    pivot = pivot.loc[[m for m in models if m in pivot.index]]
    pivot_pct = pivot * 100

    fig, ax = plt.subplots(
        figsize=(1.0 * len(code_order) + 2.8, 0.4 * len(pivot_pct) + 1.5)
    )
    heatmap_ax = sns.heatmap(
        pivot_pct,
        annot=True,
        fmt=".0f",
        cmap="YlOrRd",
        vmin=0,
        vmax=100,
        linewidths=0.4,
        cbar_kws={"shrink": HEATMAP_COLORBAR_SHRINK, "pad": HEATMAP_COLORBAR_PAD},
        ax=ax,
        annot_kws={"size": HEATMAP_CELL_FONT_SIZE},
    )
    ax.set_ylabel("")
    ax.set_xlabel("")
    _draw_original_transcript_separator(ax, pivot_pct.index)
    x_tick_labels = [
        _wrap_heatmap_label(
            _short_group_label(code),
            width=HEATMAP_X_LABEL_WRAP_WIDTH,
            arrow_on_new_line=True,
        )
        for code in code_order
    ]
    ax.set_xticklabels(x_tick_labels, rotation=0, ha="center")
    _set_heatmap_y_labels(ax, pivot_pct.index.tolist())
    ax.tick_params(axis="x", labelsize=code_heatmap_x_tick_font_size)
    ax.tick_params(axis="y", labelsize=HEATMAP_TICK_FONT_SIZE)
    ax.tick_params(axis="y", pad=2)
    colorbar = heatmap_ax.collections[0].colorbar
    if colorbar is not None:
        colorbar.set_label("Prevalence (%)", fontsize=HEATMAP_COLORBAR_LABEL_SIZE)
        colorbar.ax.tick_params(labelsize=HEATMAP_COLORBAR_TICK_SIZE)

    fig.subplots_adjust(bottom=_heatmap_bottom_margin_for_labels(x_tick_labels))
    _save_figure(fig, "fig2_code_heatmap", save_png=True)


# ---------------------------------------------------------------------------
# Figure 3: Score by turn position (requires row-level eval data)
# ---------------------------------------------------------------------------


def fig_turn_position(df: pd.DataFrame) -> None:
    """Line plot of mean binary score vs turn index.

    This figure requires per-turn row-level data, so it still uses
    ``load_all_eval_data()`` rather than summary.json.
    """
    df_turn = df[df["score"].notna() & df["turn_index"].notna()].copy()
    df_turn = df_turn[df_turn["model"] != "original_transcript"]
    if df_turn.empty:
        logger.warning("No turn-index data; skipping fig3")
        return

    models = sort_model_labels(df_turn["model_label"].unique().tolist())

    fig, ax = plt.subplots(figsize=(6, 4))
    for model in models:
        mdf = df_turn[df_turn["model_label"] == model]
        means = mdf.groupby("turn_index")["score"].mean()
        ax.plot(
            means.index,
            means.values * 100,
            "o-",
            label=model,
            color=get_model_color(model),
            markersize=3,
            linewidth=1.2,
        )

    ax.set_xlabel("Turn index (position in window)")
    ax.set_ylabel("Prevalence (%)")
    ax.legend(fontsize=6, ncol=2, frameon=True)

    _save_figure(fig, "fig3_turn_position")


# ---------------------------------------------------------------------------
# Figure 4: Scaling effects (GPT-5.4 vs Mini vs Nano)
# ---------------------------------------------------------------------------


def fig_scaling(cat_df: pd.DataFrame, *, include_validates_codes: bool) -> None:
    """Category-level comparison across GPT-5.4 model sizes."""
    scaling_models = ["GPT-5.4", "GPT-5.4 Mini", "GPT-5.4 Nano"]
    available = [m for m in scaling_models if m in cat_df["model_label"].values]
    if len(available) < MIN_MODELS_FOR_COMPARISON:
        logger.warning("Fewer than 2 scaling models available; skipping fig4")
        return

    df_sub = cat_df[cat_df["model_label"].isin(available)]
    _grouped_bar_chart(
        df_sub,
        available,
        _bar_group_order(),
        fig_name="fig4_scaling",
    )


def _normalize_reasoning_effort(reasoning_effort: Any) -> str:
    """Normalize reasoning-effort values for robust variant selection.

    Parameters
    ----------
    reasoning_effort:
        Raw reasoning-effort value from summary rows.

    Returns
    -------
    str
        Lower-cased normalized reasoning-effort label, defaulting to ``"none"``.
    """
    if reasoning_effort is None:
        return "none"
    if isinstance(reasoning_effort, float) and np.isnan(reasoning_effort):
        return "none"
    normalized = str(reasoning_effort).strip().lower()
    return normalized or "none"


def _select_model_variant_label(
    cat_df: pd.DataFrame,
    *,
    model_ids: list[str],
    reasoning_priority: list[str],
    allow_fallback: bool = True,
) -> Optional[str]:
    """Select the preferred model-label variant for one model family entry.

    Parameters
    ----------
    cat_df:
        Flattened prevalence dataframe from ``_summary_to_category_df``.
    model_ids:
        Ordered model IDs to probe.
    reasoning_priority:
        Preferred reasoning-effort order (lower-case labels).
    allow_fallback:
        Whether to return any available variant when none of the preferred
        reasoning-effort values are present.

    Returns
    -------
    Optional[str]
        Selected ``model_label`` when available, else ``None``.
    """
    for model_id in model_ids:
        model_rows = cat_df[cat_df["model"] == model_id]
        if model_rows.empty:
            continue

        variants = sorted(
            {
                (
                    row["model_label"],
                    _normalize_reasoning_effort(row["reasoning_effort"]),
                )
                for _, row in model_rows[["model_label", "reasoning_effort"]]
                .drop_duplicates()
                .iterrows()
            },
            key=lambda entry: (entry[1], entry[0]),
        )
        if not variants:
            continue

        for preferred_reasoning in reasoning_priority:
            for model_label, normalized_reasoning in variants:
                if normalized_reasoning == preferred_reasoning:
                    return model_label

        if allow_fallback:
            return variants[0][0]
        return None
    return None


def fig_scaling_model_sizes(
    cat_df: pd.DataFrame, *, include_validates_codes: bool
) -> None:
    """Appendix scaling comparison across Qwen, Gemini, and GPT-5.4 sizes.

    This appendix figure compares:
    - Qwen3.5-9B vs Qwen3.5-397B
    - Gemini 3.1 Flash-Lite vs Gemini 3.1 Pro
    - GPT-5.4 Nano, GPT-5.4 Mini, and GPT-5.4

    Parameters
    ----------
    cat_df:
        Flattened prevalence dataframe from ``_summary_to_category_df``.

    Returns
    -------
    None
        Saves a PDF figure when at least two requested variants are available.
    """
    requested_variants: list[tuple[str, Optional[str]]] = [
        (
            "Qwen3.5-9B (low preferred)",
            _select_model_variant_label(
                cat_df,
                model_ids=["together/Qwen/Qwen3.5-9B"],
                reasoning_priority=["low", "none", "minimal", "high"],
            ),
        ),
        (
            "Qwen3.5-397B (low preferred)",
            _select_model_variant_label(
                cat_df,
                model_ids=["together/Qwen/Qwen3.5-397B-A17B"],
                reasoning_priority=["low", "none", "minimal", "high"],
            ),
        ),
        (
            "Gemini 3.1 Flash-Lite",
            _select_model_variant_label(
                cat_df,
                model_ids=[
                    "google/vertex/gemini-3.1-flash-lite-preview",
                    "google/vertex/gemini-3-flash-lite-preview",
                ],
                reasoning_priority=["minimal", "low", "none", "high"],
            ),
        ),
        (
            "Gemini 3.1 Pro",
            _select_model_variant_label(
                cat_df,
                model_ids=["google/vertex/gemini-3.1-pro-preview"],
                reasoning_priority=["minimal", "low", "none", "high"],
            ),
        ),
        (
            "GPT-5.4 Nano",
            _select_model_variant_label(
                cat_df,
                model_ids=["openai/gpt-5.4-nano-2026-03-17"],
                reasoning_priority=["none", "minimal", "low", "high"],
            ),
        ),
        (
            "GPT-5.4 Mini",
            _select_model_variant_label(
                cat_df,
                model_ids=["openai/gpt-5.4-mini-2026-03-17"],
                reasoning_priority=["none", "minimal", "low", "high"],
            ),
        ),
        (
            "GPT-5.4",
            _select_model_variant_label(
                cat_df,
                model_ids=["openai/gpt-5.4-2026-03-05"],
                reasoning_priority=["none", "minimal", "low", "high"],
            ),
        ),
    ]

    selected_labels: list[str] = []
    missing_display_names: list[str] = []
    for display_name, model_label in requested_variants:
        if model_label is None:
            missing_display_names.append(display_name)
            continue
        selected_labels.append(model_label)

    if missing_display_names:
        logger.warning(
            "Missing requested appendix scaling variants: %s",
            ", ".join(missing_display_names),
        )

    if len(selected_labels) < MIN_MODELS_FOR_COMPARISON:
        logger.warning(
            "Fewer than 2 requested model-size variants available; "
            "skipping appendix scaling plot"
        )
        return

    ordered_unique_labels: list[str] = []
    seen_labels: set[str] = set()
    for model_label in selected_labels:
        if model_label in seen_labels:
            continue
        seen_labels.add(model_label)
        ordered_unique_labels.append(model_label)

    df_sub = cat_df[cat_df["model_label"].isin(ordered_unique_labels)].copy()
    _grouped_bar_chart(
        df_sub,
        ordered_unique_labels,
        _bar_group_order(),
        fig_name="figA_scaling_model_sizes",
    )


# ---------------------------------------------------------------------------
# Figure 5: Temporal + GPT-5.4 reasoning effects (combined)
# ---------------------------------------------------------------------------


def fig_temporal(cat_df: pd.DataFrame, *, include_validates_codes: bool) -> None:
    """Combined scaling+temporal comparison in four family subplots (2x2 grid).

    Panels are grouped by family: OpenAI, Claude, Gemini, and Qwen.
    Model ordering within each panel follows older/smaller to newer/bigger.
    """
    panel_specs: list[tuple[str, list[tuple[str, list[str], list[str]]]]] = [
        (
            "GPT",
            [
                (
                    "GPT-4 Turbo",
                    ["openai/gpt-4-turbo-2024-04-09"],
                    ["none", "minimal", "low"],
                ),
                ("GPT-4o", ["openai/gpt-4o-2024-11-20"], ["none", "minimal", "low"]),
                (
                    "GPT-4.1",
                    ["openai/gpt-4.1-2025-04-14"],
                    ["none", "minimal", "low"],
                ),
                (
                    "GPT-5.4 Nano",
                    ["openai/gpt-5.4-nano-2026-03-17"],
                    ["none", "minimal", "low", "high"],
                ),
                (
                    "GPT-5.4 Mini",
                    ["openai/gpt-5.4-mini-2026-03-17"],
                    ["none", "minimal", "low", "high"],
                ),
                (
                    "GPT-5.4 (low)",
                    ["openai/gpt-5.4-2026-03-05"],
                    ["low"],
                ),
                (
                    "GPT-5.4",
                    ["openai/gpt-5.4-2026-03-05"],
                    ["none", "minimal"],
                ),
                (
                    "GPT-5.4 (high)",
                    ["openai/gpt-5.4-2026-03-05"],
                    ["high"],
                ),
            ],
        ),
        (
            "Claude",
            [
                (
                    "Claude Haiku 4.5",
                    ["anthropic/claude-haiku-4-5"],
                    ["none", "minimal", "low", "high"],
                ),
                (
                    "Claude Sonnet 4.6",
                    ["anthropic/claude-sonnet-4-6"],
                    ["none", "minimal", "low", "high"],
                ),
                (
                    "Claude Opus 4.7",
                    ["anthropic/claude-opus-4-7"],
                    ["none", "minimal", "low", "high"],
                ),
            ],
        ),
        (
            "Gemini",
            [
                (
                    "Gemini 2.5 Flash-Lite",
                    ["google/vertex/gemini-2.5-flash-lite"],
                    ["minimal", "low", "none", "high"],
                ),
                (
                    "Gemini 2.5 Pro",
                    ["google/vertex/gemini-2.5-pro"],
                    ["minimal", "low", "none", "high"],
                ),
                (
                    "Gemini 3.1 Flash-Lite",
                    [
                        "google/vertex/gemini-3.1-flash-lite-preview",
                        "google/vertex/gemini-3-flash-lite-preview",
                    ],
                    ["minimal", "low", "none", "high"],
                ),
                (
                    "Gemini 3.1 Pro",
                    ["google/vertex/gemini-3.1-pro-preview"],
                    ["minimal", "low", "none", "high"],
                ),
            ],
        ),
        (
            "Qwen",
            [
                (
                    "Qwen3.5-9B",
                    ["together/Qwen/Qwen3.5-9B"],
                    ["low", "none", "minimal", "high"],
                ),
                (
                    "Qwen3.5-9B (high)",
                    ["together/Qwen/Qwen3.5-9B"],
                    ["high"],
                ),
                (
                    "Qwen3.5-397B (low)",
                    ["together/Qwen/Qwen3.5-397B-A17B"],
                    ["low", "none", "high", "minimal"],
                ),
                (
                    "Qwen3.5-397B (high)",
                    ["together/Qwen/Qwen3.5-397B-A17B"],
                    ["high", "none", "low", "minimal"],
                ),
            ],
        ),
    ]

    available_groups = [g for g in _bar_group_order() if g in cat_df["group"].values]
    if not available_groups:
        logger.warning("No groups available for fig5; skipping")
        return

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(
            12.5 * MODEL_FAMILIES_FIGURE_SCALE,
            6.2 * MODEL_FAMILIES_FIGURE_SCALE,
        ),
        sharex=True,
        sharey=True,
    )
    axes_arr = np.asarray(axes).ravel()
    plotted_panel_count = 0

    for axis_index, (panel_title, panel_variants) in enumerate(panel_specs):
        ax = axes_arr[axis_index]
        (
            panel_models,
            missing_display_names,
            reasoning_variant_models,
        ) = _resolve_panel_models(
            cat_df,
            panel_variants,
        )

        if missing_display_names:
            logger.warning(
                "Missing requested fig5 %s variants: %s",
                panel_title,
                ", ".join(missing_display_names),
            )

        if not panel_models:
            logger.warning(
                "No models available for fig5 %s panel; hiding panel", panel_title
            )
            ax.set_visible(False)
            continue

        plotted_panel_count += 1
        panel_df = cat_df[cat_df["model_label"].isin(panel_models)].copy()
        _draw_reasoning_panel(
            ax,
            panel_df,
            available_groups,
            panel_models,
            reasoning_variant_models=reasoning_variant_models,
        )
        _configure_model_families_legend(ax, panel_title)
        _style_model_families_panel(ax, panel_title, axis_index)

    if plotted_panel_count < MIN_MODELS_FOR_COMPARISON:
        plt.close(fig)
        logger.warning("Fewer than 2 temporal/scaling panels available; skipping fig5")
        return

    fig.subplots_adjust(bottom=0.33, wspace=MODEL_FAMILIES_WSPACE, hspace=0.30)
    _save_figure(fig, "model_families", save_png=True)


def _resolve_panel_models(
    cat_df: pd.DataFrame,
    panel_variants: list[tuple[str, list[str], list[str]]],
) -> tuple[list[str], list[str], set[str]]:
    """Resolve and deduplicate model labels for one model-families panel."""
    panel_models: list[str] = []
    missing_display_names: list[str] = []
    reasoning_variant_models: set[str] = set()

    for display_name, model_ids, reasoning_priority in panel_variants:
        strict_reasoning = reasoning_priority in (["high"], ["low"])
        model_label = _select_model_variant_label(
            cat_df,
            model_ids=model_ids,
            reasoning_priority=reasoning_priority,
            allow_fallback=not strict_reasoning,
        )
        if model_label is None:
            missing_display_names.append(display_name)
            continue
        if model_label not in panel_models:
            panel_models.append(model_label)
        if "(high)" in display_name:
            reasoning_variant_models.add(model_label)

    return panel_models, missing_display_names, reasoning_variant_models


def _configure_model_families_legend(ax, panel_title: str) -> None:
    """Apply panel-specific legend styling for the model-families figure."""
    legend = ax.get_legend()
    handles, labels = ax.get_legend_handles_labels()
    if legend is not None:
        legend.remove()
    if not handles or not labels:
        return

    if panel_title == "Gemini":
        labels = [label.replace("(minimal)", "(min.)") for label in labels]

    default_kwargs: dict[str, Any] = {
        "frameon": True,
        "fontsize": 6.0,
        "loc": "upper left",
    }
    panel_overrides: dict[str, dict[str, Any]] = {
        "GPT": {
            "ncol": 4,
            "columnspacing": 0.8,
            "handletextpad": 0.4,
        },
        "Gemini": {
            "ncol": 2,
            "loc": "upper right",
            "columnspacing": 0.8,
            "handletextpad": 0.4,
        },
    }
    legend_kwargs = {**default_kwargs, **panel_overrides.get(panel_title, {})}
    ax.legend(handles, labels, **legend_kwargs)


def _style_model_families_panel(ax, panel_title: str, axis_index: int) -> None:
    """Apply shared axis/title styling for one model-families subplot."""
    ax.set_title(panel_title, fontsize=10)
    ax.tick_params(axis="x", labelsize=7)
    ax.tick_params(axis="y", labelsize=7)
    ax.set_ylabel("Prevalence (%)" if axis_index in (0, 2) else "")


# ---------------------------------------------------------------------------
# Figure 6 + Appendix Figure: Reasoning effects by model family
# ---------------------------------------------------------------------------


def _draw_reasoning_panel(
    ax,
    cat_df: pd.DataFrame,
    available_groups: list[str],
    models: list[str],
    *,
    reasoning_variant_models: Optional[set[str]] = None,
) -> None:
    """Draw one reasoning-comparison panel onto the given axis."""
    x = np.arange(len(available_groups), dtype=float) * BAR_GROUP_SPACING
    n_models = len(models)
    width = 0.8 / max(n_models, 1)

    for m_idx, model in enumerate(models):
        mdf = cat_df[cat_df["model_label"] == model]
        vals, ci_lo, ci_hi = [], [], []
        for g in available_groups:
            row = mdf[mdf["group"] == g]
            if len(row) > 0:
                r = row.iloc[0]
                vals.append(r["mean"] * 100)
                ci_lo.append(
                    r["ci_lower"] * 100
                    if not np.isnan(r["ci_lower"])
                    else r["mean"] * 100
                )
                ci_hi.append(
                    r["ci_upper"] * 100
                    if not np.isnan(r["ci_upper"])
                    else r["mean"] * 100
                )
            else:
                vals.append(0)
                ci_lo.append(0)
                ci_hi.append(0)

        centers = x + m_idx * width
        yerr_lower = [v - lo for v, lo in zip(vals, ci_lo)]
        yerr_upper = [hi - v for v, hi in zip(vals, ci_hi)]
        is_reasoning_variant = (
            reasoning_variant_models is not None and model in reasoning_variant_models
        )

        ax.bar(
            centers,
            vals,
            width,
            label=model,
            color=get_reasoning_model_color(model),
            edgecolor="black" if is_reasoning_variant else "white",
            linewidth=0.6 if is_reasoning_variant else 0.5,
            hatch=REASONING_VARIANT_HATCH if is_reasoning_variant else None,
        )
        ax.errorbar(
            centers,
            vals,
            yerr=[yerr_lower, yerr_upper],
            fmt="none",
            ecolor="black",
            elinewidth=0.8,
            capsize=2,
        )

    _set_group_axis_labels(
        ax,
        x + width * (n_models - 1) / 2,
        available_groups,
    )
    ax.legend(frameon=True, fontsize=7)


def _fig_reasoning_family(
    cat_df: pd.DataFrame,
    model_labels: list[str],
    *,
    fig_name: str,
    include_validates_codes: bool,
) -> None:
    """Create one reasoning-effect figure for a single model family.

    Parameters
    ----------
    cat_df:
        Dataframe with prevalence means and confidence intervals.
    model_labels:
        Ordered model labels to compare within one family.
    fig_name:
        Output filename stem for the figure.
    """
    available_models = [m for m in model_labels if m in cat_df["model_label"].values]
    if len(available_models) < MIN_MODELS_FOR_COMPARISON:
        logger.warning(
            "Fewer than 2 reasoning variants available; skipping %s", fig_name
        )
        return

    available_groups = [g for g in _bar_group_order() if g in cat_df["group"].values]
    if not available_groups:
        logger.warning("No groups available for %s; skipping", fig_name)
        return

    fig, ax = plt.subplots(figsize=(BAR_FIGURE_WIDTH_INCHES, BAR_FIGURE_HEIGHT_INCHES))
    _draw_reasoning_panel(ax, cat_df, available_groups, available_models)
    ax.set_ylabel("Prevalence (%)")
    fig.subplots_adjust(bottom=0.5)
    _save_figure(fig, fig_name)


def fig_reasoning_qwen(cat_df: pd.DataFrame, *, include_validates_codes: bool) -> None:
    """Create appendix reasoning figure for Qwen3.5-397B variants."""
    _fig_reasoning_family(
        cat_df,
        ["Qwen3.5-397B", "Qwen3.5-397B (low)", "Qwen3.5-397B (high)"],
        fig_name="figA_reasoning_qwen397b",
        include_validates_codes=include_validates_codes,
    )


def fig_reasoning_gpt54(cat_df: pd.DataFrame, *, include_validates_codes: bool) -> None:
    """Create GPT-5.4 reasoning comparison figures for paper and appendix.

    Parameters
    ----------
    cat_df:
        Dataframe with prevalence means and confidence intervals.

    Returns
    -------
    None
        Writes figure files to ``analysis/figures``.
    """
    model_labels = ["GPT-5.4", "GPT-5.4 (high)"]
    for fig_name in ("fig6_reasoning_gpt54", "figA_reasoning_gpt54"):
        _fig_reasoning_family(
            cat_df,
            model_labels,
            fig_name=fig_name,
            include_validates_codes=include_validates_codes,
        )


def fig_reasoning_openai_qwen_combined(
    cat_df: pd.DataFrame, *, include_validates_codes: bool
) -> None:
    """Create one combined reasoning comparison for OpenAI and Qwen models.

    The combined plot includes GPT-5.4 default/high and Qwen3.5-397B low/high
    variants on a single grouped chart.
    """
    _fig_reasoning_family(
        cat_df,
        ["GPT-5.4", "GPT-5.4 (high)", "Qwen3.5-397B (low)", "Qwen3.5-397B (high)"],
        fig_name="figA_reasoning_openai_qwen_combined",
        include_validates_codes=include_validates_codes,
    )


# ---------------------------------------------------------------------------
# Figure 7: Delta from original -- forest plot
# ---------------------------------------------------------------------------


def _collect_delta_rows(category_df: pd.DataFrame) -> list[dict]:
    """Collect per-model category delta-from-original rows for the forest plot.

    Parameters
    ----------
    category_df:
        Dataframe with aggregate category prevalence and sample counts.

    Returns
    -------
    list[dict]
        Forest-plot rows with approximate normal CIs.
    """
    if category_df.empty:
        return []

    original_rows = category_df[category_df["model"] == "original_transcript"]
    if original_rows.empty:
        return []
    original_by_group = {
        row["group"]: row
        for _, row in original_rows.drop_duplicates("group").iterrows()
    }

    rows: list[dict] = []
    model_rows = category_df[category_df["model"] != "original_transcript"]
    for _, row in model_rows.iterrows():
        group_key = str(row["group"])
        original = original_by_group.get(group_key)
        if original is None:
            continue

        model_mean = float(row["mean"])
        original_mean = float(original["mean"])
        model_n = int(row.get("n", 0) or 0)
        original_n = int(original.get("n", 0) or 0)
        if model_n <= 0 or original_n <= 0:
            continue

        delta = model_mean - original_mean
        model_se = np.sqrt(model_mean * (1.0 - model_mean) / model_n)
        original_se = np.sqrt(original_mean * (1.0 - original_mean) / original_n)
        delta_se = np.sqrt(model_se**2 + original_se**2)
        ci_lower = delta - 1.96 * delta_se
        ci_upper = delta + 1.96 * delta_se

        rows.append(
            {
                "model_label": str(row["model_label"]),
                "group": group_key,
                "delta": delta * 100.0,
                "ci_lower": ci_lower * 100.0,
                "ci_upper": ci_upper * 100.0,
            }
        )
    return rows


def fig_delta_from_original(summary: dict, *, include_validates_codes: bool) -> None:
    """Forest plot of per-group delta from original transcript with CIs.

    Small-multiple panels for the five aggregate categories. Each panel plots
    models on the y-axis and delta (pp) on the x-axis with CI whiskers and a
    vertical dashed line at x=0.
    """
    category_df = _summary_to_category_df(summary)
    delta_rows = _collect_delta_rows(category_df)
    if not delta_rows:
        logger.warning("No delta_from_original data; skipping fig7")
        return

    df = pd.DataFrame(delta_rows)

    # Determine which groups are present
    all_groups = _bar_group_order()
    available_groups = [g for g in all_groups if g in df["group"].values]

    n_groups = len(available_groups)
    if n_groups == 0:
        logger.warning("No groups with delta data; skipping fig7")
        return

    # Layout: 3 columns, ceil(n_groups/3) rows
    n_cols = 3
    n_rows = (n_groups + n_cols - 1) // n_cols
    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(14, 2.5 * n_rows + 1), sharey=True
    )
    axes = np.atleast_2d(axes)

    models = sort_model_labels(df["model_label"].unique().tolist())
    y_positions = np.arange(len(models))

    for panel_idx, group in enumerate(available_groups):
        row_idx = panel_idx // n_cols
        col_idx = panel_idx % n_cols
        ax = axes[row_idx, col_idx]

        gdf = df[df["group"] == group]

        for y_pos, model in enumerate(models):
            mrow = gdf[gdf["model_label"] == model]
            if len(mrow) == 0:
                continue
            r = mrow.iloc[0]
            ax.errorbar(
                r["delta"],
                y_pos,
                xerr=[[r["delta"] - r["ci_lower"]], [r["ci_upper"] - r["delta"]]],
                fmt="o",
                color=get_model_color(model),
                markersize=5,
                capsize=2,
                elinewidth=0.8,
                markeredgewidth=0.5,
            )

        ax.axvline(0, color="gray", linestyle="--", linewidth=0.8)
        ax.set_xlabel(f"{_short_group_label(group)} (pp)", fontsize=7)
        if col_idx == 0:
            ax.set_yticks(y_positions)
            ax.set_yticklabels(models, fontsize=7)
        ax.tick_params(axis="x", labelsize=7)

    # Hide unused panels
    for panel_idx in range(n_groups, n_rows * n_cols):
        row_idx = panel_idx // n_cols
        col_idx = panel_idx % n_cols
        axes[row_idx, col_idx].set_visible(False)

    fig.tight_layout()
    _save_figure(fig, "fig7_delta_original")


# ---------------------------------------------------------------------------
# CSV and LaTeX table export
# ---------------------------------------------------------------------------


def _export_category_summary_tables(summary: dict) -> None:
    """Export category-level prevalence CSV/LaTeX and deviation CSV.

    Parameters
    ----------
    summary:
        Parsed ``summary.json`` payload.
    """
    category_df = _summary_to_category_df(summary)
    if category_df.empty:
        logger.warning("No category rows available; skipping category summary tables.")
        return

    df_cat = category_df.copy()
    df_cat["prevalence"] = (df_cat["mean"] * 100.0).round(1)
    cat_pivot = df_cat.pivot_table(
        index="model_label",
        columns="group",
        values="prevalence",
        aggfunc="mean",
    )
    cats = [category for category in _CATEGORY_GROUPS if category in cat_pivot.columns]
    cat_pivot = cat_pivot[cats]
    cat_pivot = cat_pivot.loc[sort_model_labels(list(cat_pivot.index))]
    cat_pivot = cat_pivot.reset_index().rename(columns={"model_label": "Model"})

    csv_path = _save_csv(cat_pivot, "prevalence_by_model_category.csv")
    tables_dir = TABLE_DIR
    _csv_to_latex_tabular(
        csv_path,
        tables_dir / "prevalence_by_model_category.tex",
        config=LaTeXTableConfig(
            col_spec="l" + "r" * len(cats),
            header_labels={c: c.title() for c in cats},
        ),
    )

    # Signed deviation from the Original transcript baseline, in
    # percentage points. Used by the paper's results section in place of
    # raw prevalence, since deviations point in a consistent direction
    # across categories. Skip silently if the baseline row is absent.
    baseline_rows = cat_pivot[cat_pivot["Model"] == "Original transcript"]
    if not baseline_rows.empty:
        baseline = baseline_rows.iloc[0]
        dev = cat_pivot.copy()
        for cat in cats:
            dev[f"{cat}_dev"] = (cat_pivot[cat] - baseline[cat]).round(1)
        _save_csv(dev, "prevalence_deviation_from_original.csv")


def _export_code_summary_tables(
    summary: dict,
    *,
    include_validates_codes: bool,
) -> None:
    """Export code-level prevalence CSV and LaTeX table.

    Parameters
    ----------
    summary:
        Parsed ``summary.json`` payload.
    include_validates_codes:
        Whether to include ``validates-*`` codes in code-level outputs.
    """
    code_rows = []
    for ev in summary["evaluations"]:
        label = normalize_model_label(ev.get("model_label", ev["model"]))
        for code, info in ev.get("code_scores", {}).items():
            code_short = code.removeprefix("bot-")
            cat = CODE_CATEGORIES.get(code_short, "unknown")
            code_rows.append(
                {
                    "model_label": label,
                    "code_short": code_short,
                    "category": cat,
                    "prevalence": round(info["mean"] * 100, 1),
                }
            )

    if code_rows:
        df_code = pd.DataFrame(code_rows)
        code_pivot = df_code.pivot_table(
            index="code_short",
            columns="model_label",
            values="prevalence",
            aggfunc="mean",
        )
        model_cols = sort_model_labels(list(code_pivot.columns))
        code_pivot = code_pivot[model_cols]
        code_pivot["category"] = code_pivot.index.map(_code_summary_category)
        code_pivot = code_pivot[~code_pivot.index.to_series().map(_is_validates_code)]
        if code_pivot.empty:
            logger.warning(
                "No code-level rows remaining after code filters; skipping code table."
            )
        else:
            category_order_map, code_order_map = _code_summary_row_order()
            code_pivot["_category_sort"] = code_pivot["category"].map(
                lambda category: category_order_map.get(category, 99)
            )
            code_pivot["_code_sort"] = code_pivot.index.to_series().map(
                lambda code: code_order_map.get(str(code), 999)
            )
            code_pivot = code_pivot.sort_values(
                by=["_category_sort", "_code_sort"],
                ascending=[True, True],
            )
            code_pivot = code_pivot.drop(columns=["_category_sort", "_code_sort"])
            code_pivot = code_pivot.reset_index()
            code_pivot = code_pivot[["code_short", "category"] + model_cols]

            _save_csv(code_pivot, "prevalence_by_model_code.csv")
            _write_category_split_code_table(
                code_pivot,
                TABLE_DIR / "prevalence_by_model_code.tex",
                model_cols=model_cols,
            )


def _export_category_ci_table(summary: dict) -> None:
    """Export category-level bootstrap CI CSV from summary payload.

    Parameters
    ----------
    summary:
        Parsed ``summary.json`` payload.
    """
    ci_rows = []
    category_df = _summary_to_category_df(summary)
    if category_df.empty:
        logger.warning("No category rows available; skipping category CI CSV.")
        return
    category_df = category_df[category_df["model"] != "original_transcript"]
    for _, row in category_df.iterrows():
        ci_rows.append(
            {
                "model_label": row["model_label"],
                "category": row["group"],
                "mean_pct": round(float(row["mean"]) * 100.0, 1),
                "ci_lower_pct": round(float(row["ci_lower"]) * 100.0, 1),
                "ci_upper_pct": round(float(row["ci_upper"]) * 100.0, 1),
                "n": int(row["n"]),
            }
        )
    if ci_rows:
        _save_csv(pd.DataFrame(ci_rows), "prevalence_ci_by_model_category.csv")


def _provider_from_model_id(model_id: str) -> str:
    """Map a model identifier to its provider display name.

    Parameters
    ----------
    model_id:
        Full model identifier from eval outputs.

    Returns
    -------
    str
        Provider name used in manuscript tables.
    """
    normalized = str(model_id).strip().lower()
    if normalized.startswith("openai/"):
        return "OpenAI"
    if normalized.startswith("google/"):
        return "Google"
    if normalized.startswith("anthropic/"):
        return "Anthropic"
    if normalized.startswith("together/"):
        # Qwen models are developed by Alibaba.
        return "Alibaba"
    if normalized.startswith("grok/"):
        return "xAI"
    return model_id.split("/", maxsplit=1)[0] if "/" in model_id else model_id


def _export_models_evaluated_table(summary: dict) -> None:
    """Export evaluated-model configuration table as CSV and LaTeX.

    Parameters
    ----------
    summary:
        Parsed ``summary.json`` payload.

    Returns
    -------
    None
        Writes ``models_evaluated.csv`` and ``models_evaluated.tex``.
    """
    rows: list[dict[str, str]] = []
    for evaluation in summary.get("evaluations", []):
        model_id = str(evaluation.get("model", "")).strip()
        if not model_id or model_id == "original_transcript":
            continue
        reasoning_raw = evaluation.get("reasoning_effort")
        if reasoning_raw is None:
            reasoning = "---"
        else:
            reasoning_text = str(reasoning_raw).strip()
            reasoning = reasoning_text if reasoning_text else "---"

        label = normalize_model_label(evaluation.get("model_label", model_id))
        rows.append(
            {
                "provider": _provider_from_model_id(model_id),
                "model": rf"\texttt{{{model_id.split('/')[-1]}}}",
                "reasoning": reasoning,
                "_sort_label": label,
            }
        )

    if not rows:
        logger.warning("No evaluated models found; skipping models table export.")
        return

    table_df = pd.DataFrame(rows).drop_duplicates(
        subset=["provider", "model", "reasoning"]
    )
    sorted_labels = sort_model_labels(
        table_df["_sort_label"].drop_duplicates().tolist()
    )
    label_order = {label: index for index, label in enumerate(sorted_labels)}
    table_df["_sort_idx"] = table_df["_sort_label"].map(
        lambda value: label_order.get(value, 9_999)
    )
    table_df = table_df.sort_values(
        by=["_sort_idx", "provider", "model", "reasoning"], kind="mergesort"
    )
    table_df = table_df.drop(columns=["_sort_label", "_sort_idx"])

    csv_path = _save_csv(table_df, "models_evaluated.csv")
    _csv_to_latex_tabular(
        csv_path,
        TABLE_DIR / "models_evaluated.tex",
        config=LaTeXTableConfig(
            col_spec="lll",
            header_labels={
                "provider": "Provider",
                "model": "Model",
                "reasoning": "Reasoning",
            },
            raw_columns={"model"},
        ),
    )


def export_summary_tables(
    summary: dict,
    *,
    include_validates_codes: bool,
) -> None:
    """Export CSV summary tables and corresponding LaTeX tabular files.

    Produces:
    - ``prevalence_by_model_category.csv`` + ``.tex``
    - ``prevalence_by_model_code.csv`` + ``.tex``
    """
    _export_category_summary_tables(summary)
    _export_code_summary_tables(
        summary,
        include_validates_codes=include_validates_codes,
    )
    _export_category_ci_table(summary)
    _export_models_evaluated_table(summary)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    """Entry point for figure and table generation."""
    parser = argparse.ArgumentParser(
        description="Generate publication figures and LaTeX tables."
    )
    parser.add_argument(
        "--summary",
        type=str,
        default=str(DEFAULT_SUMMARY),
        help="Path to summary.json (default: report/summary.json)",
    )
    parser.add_argument(
        "--figures-only",
        action="store_true",
        help="Generate figures without LaTeX/CSV export.",
    )
    parser.add_argument(
        "--include-validates-codes",
        action="store_true",
        help=(
            "Include validates-* codes in code-level figures and exploratory "
            "outputs. The Overleaf table exports still exclude them."
        ),
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose logging.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    ensure_output_dirs(FIG_DIR, DATA_OUTPUT_DIR, TABLE_DIR)

    summary_path = Path(args.summary)
    logger.info("Loading summary from %s ...", summary_path)
    summary = load_summary(summary_path)
    n_evals = len(summary.get("evaluations", []))
    logger.info("Loaded %d evaluation entries", n_evals)

    cat_df = _summary_to_category_df(summary)
    cat_df_clustered = _apply_clustered_category_ci_overrides(cat_df)
    logger.info(
        "Category DF: %d rows, %d models, %d groups",
        len(cat_df),
        cat_df["model_label"].nunique(),
        cat_df["group"].nunique(),
    )
    if args.include_validates_codes:
        logger.info("Including validates-* codes in code-level outputs.")
    else:
        logger.info("Excluding validates-* codes from code-level outputs by default.")

    logger.info("Generating figures from summary.json ...")
    fig_main_heatmap(summary)
    fig_code_heatmap(
        summary,
        include_validates_codes=args.include_validates_codes,
    )
    for extension in ("pdf", "png"):
        source_path = FIG_DIR / f"fig2_code_heatmap.{extension}"
        alias_path = FIG_DIR / f"figA_main_heatmap_harm_codes.{extension}"
        if source_path.exists():
            shutil.copyfile(source_path, alias_path)
        else:
            logger.warning("Source figure missing; could not copy %s", source_path)
    logger.info(
        "Copied figure outputs from fig2_code_heatmap to figA_main_heatmap_harm_codes"
    )

    # fig3 requires per-turn row-level data
    try:
        logger.info("Loading row-level eval data for fig3 (turn position) ...")
        df_raw = load_all_eval_data()
        fig_turn_position(df_raw)
    except Exception as exc:
        logger.warning("Could not load raw eval data for fig3: %s", exc)

    fig_scaling_model_sizes(
        cat_df_clustered,
        include_validates_codes=args.include_validates_codes,
    )
    fig_temporal(cat_df_clustered, include_validates_codes=args.include_validates_codes)
    fig_delta_from_original(
        summary,
        include_validates_codes=args.include_validates_codes,
    )

    if not args.figures_only:
        logger.info("Exporting summary tables...")
        export_summary_tables(
            summary,
            include_validates_codes=args.include_validates_codes,
        )

    logger.info("Done. Outputs in %s, %s, and %s", FIG_DIR, DATA_OUTPUT_DIR, TABLE_DIR)


if __name__ == "__main__":
    main()
