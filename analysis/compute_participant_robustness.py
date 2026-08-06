"""Compute appendix-ready participant robustness summaries.

This script produces two anonymized participant-level robustness tables for the
paper appendix:

- ``analysis/data/participant_spread_summary.csv``
- ``analysis/data/participant_clustered_main_comparisons.csv``
- ``analysis/data/participant_clustered_prevalence_by_model_category.csv``
- ``analysis/data/participant_leave_one_out_main_comparisons.csv``
- ``analysis/tables/participant_spread_summary.tex``
- ``analysis/tables/participant_clustered_main_comparisons.tex``
- ``analysis/tables/participant_leave_one_out_main_comparisons.tex``

Usage::

    uv run python -m analysis.compute_participant_robustness
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from analysis.artifact_paths import DATA_DIR, TABLE_DIR, ensure_output_dirs
from analysis.bootstrap import (
    BootstrapConfig,
    hierarchical_weighted_delta_ci,
    hierarchical_weighted_mean_ci,
)
from analysis.load_eval_data import load_all_eval_data
from analysis.participant_clustered import (
    aggregate_participant_conversation_value_sums,
    aggregate_participant_value_sums,
    attach_participant_ids,
    build_aligned_value_lookup,
    build_conversation_value_lookup,
    resolve_participant_windows_path,
)
from llm_delusion_eval.participant_exclusions import resolve_excluded_participants
from llm_delusion_eval.paths import DEFAULT_WINDOWS_PATH, resolve_path

logger = logging.getLogger(__name__)

DATA_OUTPUT_DIR = DATA_DIR
TABLE_OUTPUT_DIR = TABLE_DIR

_CATEGORY_GROUPS = [
    "sycophancy",
    "delusional",
    "relationship",
    "facilitates harm",
    "discourages harm",
]
_FACILITATES_HARM_CODES = {
    "facilitates-self-harm",
    "facilitates-violence",
}
_DISCOURAGES_HARM_CODES = {
    "discourages-self-harm",
    "discourages-violence",
}
_SUMMARY_REQUIRED_COLUMNS = {
    "participant",
    "conversation_id",
    "start_message_index",
    "label",
    "messages",
    "meets_code",
    "selected_for_eval",
}
_SUMMARY_OUTPUT_COLUMNS = [
    "participant",
    "n_windows",
    "n_unique_histories",
    "n_messages",
    "n_codes",
]
_SPREAD_OUTPUT_COLUMNS = ["metric", "min", "median", "mean", "max"]
_SPREAD_METRICS = (("Windows per participant", "n_windows"),)
_CLUSTER_BOOT_CONFIG = BootstrapConfig()


@dataclass(frozen=True)
class ComparisonSpec:
    """One appendix comparison for leave-one-participant-out reruns.

    Parameters
    ----------
    section:
        High-level results subsection where the claim appears.
    comparison:
        Human-readable comparison label shown in the appendix table. Positive
        margins mean the first-listed model has higher prevalence.
    higher_model_label:
        First model in the comparison, expected to have higher prevalence.
    lower_model_label:
        Second model in the comparison, expected to have lower prevalence.
    category:
        Paper-level category to compare.
    """

    section: str
    comparison: str
    higher_model_label: str
    lower_model_label: str
    category: str


@dataclass(frozen=True)
class LatexTableSpec:
    """Configuration for one exported LaTeX table fragment.

    Parameters
    ----------
    header_labels:
        Mapping from dataframe column names to display labels.
    column_order:
        Ordered column list for output.
    col_spec:
        LaTeX column specification.
    notes:
        Leading LaTeX comments describing the table semantics.
    preamble_lines:
        Optional LaTeX lines inserted after ``\\centering`` and before the
        ``tabular`` environment.
    """

    header_labels: dict[str, str]
    column_order: list[str]
    col_spec: str
    notes: list[str]
    preamble_lines: tuple[str, ...] = ()


_MAIN_COMPARISONS = (
    ComparisonSpec(
        section="Temporal",
        comparison="GPT-4o vs. GPT-4 Turbo",
        higher_model_label="GPT-4o",
        lower_model_label="GPT-4 Turbo",
        category="delusional",
    ),
    ComparisonSpec(
        section="Temporal",
        comparison="GPT-4o vs. GPT-4 Turbo",
        higher_model_label="GPT-4o",
        lower_model_label="GPT-4 Turbo",
        category="relationship",
    ),
    ComparisonSpec(
        section="Temporal",
        comparison="GPT-4.1 vs. GPT-4 Turbo",
        higher_model_label="GPT-4.1",
        lower_model_label="GPT-4 Turbo",
        category="delusional",
    ),
    ComparisonSpec(
        section="Temporal",
        comparison="GPT-4.1 vs. GPT-4 Turbo",
        higher_model_label="GPT-4.1",
        lower_model_label="GPT-4 Turbo",
        category="relationship",
    ),
    ComparisonSpec(
        section="Temporal",
        comparison="GPT-4o vs. GPT-5.4",
        higher_model_label="GPT-4o",
        lower_model_label="GPT-5.4",
        category="delusional",
    ),
    ComparisonSpec(
        section="Temporal",
        comparison="GPT-4o vs. GPT-5.4",
        higher_model_label="GPT-4o",
        lower_model_label="GPT-5.4",
        category="relationship",
    ),
    ComparisonSpec(
        section="Temporal",
        comparison="GPT-4o vs. GPT-5.4",
        higher_model_label="GPT-4o",
        lower_model_label="GPT-5.4",
        category="facilitates harm",
    ),
    ComparisonSpec(
        section="Temporal",
        comparison="GPT-5.4 vs. GPT-4o",
        higher_model_label="GPT-5.4",
        lower_model_label="GPT-4o",
        category="discourages harm",
    ),
    ComparisonSpec(
        section="Scaling",
        comparison="GPT-5.4 vs. GPT-5.4 Mini",
        higher_model_label="GPT-5.4",
        lower_model_label="GPT-5.4 Mini",
        category="delusional",
    ),
    ComparisonSpec(
        section="Scaling",
        comparison="GPT-5.4 vs. GPT-5.4 Mini",
        higher_model_label="GPT-5.4",
        lower_model_label="GPT-5.4 Mini",
        category="relationship",
    ),
    ComparisonSpec(
        section="Scaling",
        comparison="GPT-5.4 Nano vs. GPT-5.4 Mini",
        higher_model_label="GPT-5.4 Nano",
        lower_model_label="GPT-5.4 Mini",
        category="delusional",
    ),
    ComparisonSpec(
        section="Scaling",
        comparison="GPT-5.4 Nano vs. GPT-5.4 Mini",
        higher_model_label="GPT-5.4 Nano",
        lower_model_label="GPT-5.4 Mini",
        category="relationship",
    ),
    ComparisonSpec(
        section="Scaling",
        comparison="Gemini 2.5 Pro vs. Gemini 2.5 Flash-Lite",
        higher_model_label="Gemini 2.5 Pro (minimal)",
        lower_model_label="Gemini 2.5 Flash-Lite",
        category="delusional",
    ),
    ComparisonSpec(
        section="Scaling",
        comparison="Gemini 2.5 Pro vs. Gemini 2.5 Flash-Lite",
        higher_model_label="Gemini 2.5 Pro (minimal)",
        lower_model_label="Gemini 2.5 Flash-Lite",
        category="facilitates harm",
    ),
    ComparisonSpec(
        section="Scaling",
        comparison="Qwen3.5-397B vs. Qwen3.5-9B",
        higher_model_label="Qwen3.5-397B (low)",
        lower_model_label="Qwen3.5-9B (low)",
        category="sycophancy",
    ),
    ComparisonSpec(
        section="Scaling",
        comparison="Qwen3.5-397B vs. Qwen3.5-9B",
        higher_model_label="Qwen3.5-397B (low)",
        lower_model_label="Qwen3.5-9B (low)",
        category="delusional",
    ),
    ComparisonSpec(
        section="Scaling",
        comparison="Qwen3.5-397B vs. Qwen3.5-9B",
        higher_model_label="Qwen3.5-397B (low)",
        lower_model_label="Qwen3.5-9B (low)",
        category="relationship",
    ),
)


def _message_count(messages: object) -> int:
    """Return the message count for one selected window.

    Parameters
    ----------
    messages:
        Serialized list-like messages payload from ``items.parquet``.

    Returns
    -------
    int
        Number of messages in the window, or ``0`` when unavailable.
    """
    if isinstance(messages, (str, bytes, dict)):
        return 0
    if hasattr(messages, "__len__"):
        try:
            return len(messages)
        except TypeError:
            return 0
    return 0


def _report_category_group(row: pd.Series) -> str:
    """Map one eval row to the paper's five-category grouping.

    Parameters
    ----------
    row:
        One row from the row-level evaluation dataframe.

    Returns
    -------
    str
        Paper-level category label, or ``""`` when the row should be excluded
        from paper-level aggregation.
    """
    code_short = str(row.get("code_short", "")).strip()
    if code_short in _FACILITATES_HARM_CODES:
        return "facilitates harm"
    if code_short in _DISCOURAGES_HARM_CODES:
        return "discourages harm"
    if code_short.startswith("validates-"):
        return ""
    return str(row.get("category", "")).strip()


def _load_participant_summary(
    windows_path: str | Path,
    *,
    excluded_participants: set[str],
) -> pd.DataFrame:
    """Load per-participant benchmark contribution statistics.

    Parameters
    ----------
    windows_path:
        Active windows parquet path or URI.
    excluded_participants:
        Participant IDs to remove from the summary.

    Returns
    -------
    pd.DataFrame
        One row per participant with contribution counts.
    """
    local_windows_path = Path(
        resolve_path(
            "LLM_DELUSIONS_WINDOWS_PATH",
            DEFAULT_WINDOWS_PATH,
            explicit=str(windows_path),
            require_local=True,
        )
    )
    candidate_paths = [local_windows_path]
    if local_windows_path.name != "items.parquet":
        candidate_paths.insert(0, local_windows_path.with_name("items.parquet"))

    unique_candidate_paths: list[Path] = []
    seen_paths: set[str] = set()
    for candidate_path in candidate_paths:
        path_key = str(candidate_path)
        if path_key in seen_paths:
            continue
        seen_paths.add(path_key)
        unique_candidate_paths.append(candidate_path)

    items = pd.DataFrame()
    matched_path: Path | None = None
    for candidate_path in unique_candidate_paths:
        if not candidate_path.exists():
            continue
        candidate_items = pd.read_parquet(candidate_path)
        if not _SUMMARY_REQUIRED_COLUMNS.issubset(candidate_items.columns):
            logger.warning(
                "Skipping participant summary source %s because it is missing "
                "required columns (%s).",
                candidate_path,
                ", ".join(sorted(_SUMMARY_REQUIRED_COLUMNS)),
            )
            continue
        items = candidate_items
        matched_path = candidate_path
        break

    if matched_path is None:
        logger.warning(
            "Could not load participant summary inputs. Checked: %s.",
            ", ".join(str(path) for path in unique_candidate_paths),
        )
        return pd.DataFrame(columns=_SUMMARY_OUTPUT_COLUMNS)

    selected = items[
        items["meets_code"].eq(True) & items["selected_for_eval"].eq(True)
    ].copy()
    selected["participant"] = selected["participant"].astype(str).str.strip()
    selected = selected[selected["participant"] != ""].copy()
    if excluded_participants:
        selected = selected[~selected["participant"].isin(excluded_participants)].copy()
    if selected.empty:
        logger.warning(
            "No selected items remain after participant filtering in %s.",
            matched_path,
        )
        return pd.DataFrame(columns=_SUMMARY_OUTPUT_COLUMNS)

    selected["history_key"] = (
        selected["participant"].astype(str)
        + "::"
        + selected["conversation_id"].astype(str)
        + "::"
        + selected["start_message_index"].astype(str)
    )
    selected["message_count"] = selected["messages"].map(_message_count)
    selected["code_short"] = selected["label"].astype(str).str.removeprefix("bot-")

    summary = (
        selected.groupby("participant", sort=True)
        .agg(
            n_windows=("label", "size"),
            n_unique_histories=("history_key", "nunique"),
            n_messages=("message_count", "sum"),
            n_codes=("code_short", "nunique"),
        )
        .reset_index()
    )
    return summary.sort_values("participant", kind="mergesort").reset_index(drop=True)


def _aggregate_participant_counts(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate binary scores to participant-conversation/model/category counts.

    Parameters
    ----------
    df:
        Eval rows with ``participant`` and ``score`` columns.

    Returns
    -------
    pd.DataFrame
        Aggregated positive/total counts by participant, conversation, model,
        and category.
    """
    scored = df[df["score"].notna()].copy()
    if scored.empty:
        return pd.DataFrame()

    scored["report_category"] = scored.apply(_report_category_group, axis=1)
    scored = scored[scored["report_category"].isin(_CATEGORY_GROUPS)].copy()
    scored["category"] = scored["report_category"]
    grouped = aggregate_participant_conversation_value_sums(
        scored,
        ["model_label", "category"],
        value_col="score",
    ).rename(columns={"value_sum": "positive", "value_count": "total"})
    grouped["positive"] = grouped["positive"].astype(float)
    grouped["total"] = grouped["total"].astype(float)
    return grouped


def _full_prevalence(positives: np.ndarray, totals: np.ndarray) -> float:
    """Return full-sample prevalence for one model-category cell.

    Parameters
    ----------
    positives:
        Participant-aligned positive counts.
    totals:
        Participant-aligned total counts.

    Returns
    -------
    float
        Full-sample prevalence, or ``nan`` when no rows are present.
    """
    total_rows = float(totals.sum())
    if total_rows == 0.0:
        return float("nan")
    return float(positives.sum() / total_rows)


def _leave_one_out_prevalence_values(
    positives: np.ndarray,
    totals: np.ndarray,
) -> tuple[float, np.ndarray]:
    """Return full prevalence and aligned leave-one-out prevalences.

    Parameters
    ----------
    positives:
        Participant-aligned positive counts.
    totals:
        Participant-aligned total counts.

    Returns
    -------
    tuple[float, np.ndarray]
        Full-sample prevalence and one leave-one-out prevalence per
        participant position.
    """
    full_prevalence = _full_prevalence(positives, totals)
    full_total = float(totals.sum())
    leave_one_out = np.full(shape=totals.shape, fill_value=np.nan, dtype=float)
    if np.isnan(full_prevalence) or full_total == 0.0:
        return (float("nan"), leave_one_out)

    positive_sum = float(positives.sum())
    for index, total_value in enumerate(totals):
        total_without = full_total - float(total_value)
        if total_without <= 0.0:
            continue
        leave_one_out[index] = (positive_sum - float(positives[index])) / total_without
    return (full_prevalence, leave_one_out)


def _build_participant_spread_summary(
    participant_summary: pd.DataFrame,
) -> pd.DataFrame:
    """Summarize participant contribution spread without exposing IDs.

    Parameters
    ----------
    participant_summary:
        Per-participant contribution counts.

    Returns
    -------
    pd.DataFrame
        One row per metric with min, median, mean, and max.
    """
    if participant_summary.empty:
        return pd.DataFrame(columns=_SPREAD_OUTPUT_COLUMNS)

    rows: list[dict[str, float | str]] = []
    for metric_label, column_name in _SPREAD_METRICS:
        values = participant_summary[column_name].to_numpy(dtype=float)
        rows.append(
            {
                "metric": metric_label,
                "min": int(np.min(values)),
                "median": round(float(np.median(values)), 1),
                "mean": round(float(np.mean(values)), 1),
                "max": int(np.max(values)),
            }
        )
    return pd.DataFrame(rows)


def _require_model_category_arrays(
    arrays: dict[tuple[str, str], tuple[np.ndarray, np.ndarray]],
    *,
    model_label: str,
    category: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Return count arrays for one model/category pair.

    Parameters
    ----------
    arrays:
        Mapping from ``(model_label, category)`` to aligned counts.
    model_label:
        Model display label.
    category:
        Paper-level category.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Positive counts and total counts for the requested cell.
    """
    key = (model_label, category)
    if key not in arrays:
        raise KeyError(f"Missing model/category cell for appendix table: {key}")
    return arrays[key]


def _build_clustered_main_comparison_summary(
    lookup: dict[tuple[str, str], pd.DataFrame],
    *,
    config: BootstrapConfig,
) -> pd.DataFrame:
    """Summarize hierarchical CIs for the main comparison margins.

    Parameters
    ----------
    lookup:
        Mapping from ``(model_label, category)`` to participant-conversation
        count summaries.
    config:
        Cluster bootstrap configuration.

    Returns
    -------
    pd.DataFrame
        One row per configured main-paper comparison.
    """
    rows: list[dict[str, float | int | str]] = []
    for spec in _MAIN_COMPARISONS:
        higher_key = (spec.higher_model_label, spec.category)
        lower_key = (spec.lower_model_label, spec.category)
        if higher_key not in lookup or lower_key not in lookup:
            raise KeyError(
                "Missing model/category cell for appendix table: "
                f"{higher_key if higher_key not in lookup else lower_key}"
            )
        clustered = hierarchical_weighted_delta_ci(
            lookup[higher_key],
            lookup[lower_key],
            config=config,
            sum_col="positive",
            count_col="total",
        )
        if np.isnan(float(clustered["estimate"])):
            continue
        rows.append(
            {
                "section": spec.section,
                "comparison": spec.comparison,
                "category": spec.category,
                "n_participants_supported": int(clustered["n_participants_supported"]),
                "full_margin_pp": round(float(clustered["estimate"]) * 100.0, 1),
                "cluster_ci_low_pp": round(float(clustered["ci_low"]) * 100.0, 1),
                "cluster_ci_high_pp": round(float(clustered["ci_high"]) * 100.0, 1),
                "cluster_boot_n": int(clustered["cluster_boot_n"]),
            }
        )
    return pd.DataFrame(rows)


def _build_clustered_prevalence_summary(
    lookup: dict[tuple[str, str], pd.DataFrame],
    *,
    config: BootstrapConfig,
) -> pd.DataFrame:
    """Summarize hierarchical model/category prevalences.

    Parameters
    ----------
    lookup:
        Mapping from ``(model_label, category)`` to participant-conversation
        count summaries.
    config:
        Cluster bootstrap configuration.

    Returns
    -------
    pd.DataFrame
        One row per available ``(model_label, category)`` cell.
    """
    rows: list[dict[str, float | int | str]] = []
    for (model_label, category), aggregated in lookup.items():
        clustered = hierarchical_weighted_mean_ci(
            aggregated,
            config=config,
            sum_col="positive",
            count_col="total",
        )
        if np.isnan(float(clustered["estimate"])):
            continue
        rows.append(
            {
                "model_label": model_label,
                "category": category,
                "n_participants_supported": int(clustered["n_participants_supported"]),
                "prevalence_pct": round(float(clustered["estimate"]) * 100.0, 1),
                "cluster_ci_low_pct": round(float(clustered["ci_low"]) * 100.0, 1),
                "cluster_ci_high_pct": round(float(clustered["ci_high"]) * 100.0, 1),
                "cluster_boot_n": int(clustered["cluster_boot_n"]),
            }
        )

    return (
        pd.DataFrame(rows)
        .sort_values(["category", "model_label"], kind="mergesort")
        .reset_index(drop=True)
    )


def _build_leave_one_out_comparison_summary(
    participants: np.ndarray,
    arrays: dict[tuple[str, str], tuple[np.ndarray, np.ndarray]],
) -> pd.DataFrame:
    """Summarize claim-level leave-one-participant-out robustness.

    Parameters
    ----------
    participants:
        Participant IDs aligned to each array entry.
    arrays:
        Mapping from ``(model_label, category)`` to aligned counts.

    Returns
    -------
    pd.DataFrame
        One row per main-paper comparison with full-sample and leave-one-out
        margin summaries.
    """
    if participants.size == 0:
        return pd.DataFrame()

    rows: list[dict[str, float | int | str]] = []
    for spec in _MAIN_COMPARISONS:
        higher_positives, higher_totals = _require_model_category_arrays(
            arrays,
            model_label=spec.higher_model_label,
            category=spec.category,
        )
        lower_positives, lower_totals = _require_model_category_arrays(
            arrays,
            model_label=spec.lower_model_label,
            category=spec.category,
        )

        higher_full, higher_loo = _leave_one_out_prevalence_values(
            higher_positives,
            higher_totals,
        )
        lower_full, lower_loo = _leave_one_out_prevalence_values(
            lower_positives,
            lower_totals,
        )
        if np.isnan(higher_full) or np.isnan(lower_full):
            continue

        loo_margin = (higher_loo - lower_loo) * 100.0
        valid_margin = loo_margin[~np.isnan(loo_margin)]
        if valid_margin.size == 0:
            continue

        rows.append(
            {
                "section": spec.section,
                "comparison": spec.comparison,
                "category": spec.category,
                "full_margin_pp": round((higher_full - lower_full) * 100.0, 1),
                "loo_min_margin_pp": round(float(np.min(valid_margin)), 1),
                "loo_max_margin_pp": round(float(np.max(valid_margin)), 1),
                "unsupported_reruns": int(np.count_nonzero(valid_margin <= 0.0)),
                "n_reruns": int(valid_margin.size),
            }
        )

    return pd.DataFrame(rows)


def _escape_latex(text: object) -> str:
    """Escape a value for safe use in a LaTeX table cell.

    Parameters
    ----------
    text:
        Scalar cell value to escape.

    Returns
    -------
    str
        Escaped string for LaTeX tabular output.
    """
    value = "" if text is None else str(text)
    for char, replacement in (
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
    ):
        value = value.replace(char, replacement)
    return value


def _latex_number(value: float | int) -> str:
    """Format one numeric value for appendix table output.

    Parameters
    ----------
    value:
        Numeric value to format.

    Returns
    -------
    str
        Integer text when exact, otherwise one decimal place.
    """
    numeric = float(value)
    if numeric.is_integer():
        return str(int(numeric))
    return f"{numeric:.1f}"


def _write_latex_table(
    df: pd.DataFrame,
    path: Path,
    spec: LatexTableSpec,
) -> None:
    """Write a LaTeX tabular fragment for one appendix table.

    Parameters
    ----------
    df:
        Table data to export.
    path:
        Output ``.tex`` path.
    spec:
        Table rendering configuration.
    """
    if df.empty:
        logger.warning("Skipping empty LaTeX table: %s", path)
        return

    path.parent.mkdir(parents=True, exist_ok=True)
    display = df[spec.column_order].copy()

    with path.open("w", encoding="utf-8") as file_obj:
        file_obj.write(
            "% NOTE: This table is auto-generated; "
            "prefer not to hand-edit until the very end.\n"
        )
        for note in spec.notes:
            file_obj.write(f"% {note}\n")
        file_obj.write(r"\centering" + "\n")
        for line in spec.preamble_lines:
            file_obj.write(line + "\n")
        file_obj.write(r"\begin{tabular}{" + spec.col_spec + "}\n")
        file_obj.write(r"\toprule" + "\n")
        labels = [
            _escape_latex(spec.header_labels[column]) for column in spec.column_order
        ]
        file_obj.write(" & ".join(labels) + r" \\" + "\n")
        file_obj.write(r"\midrule" + "\n")

        last_section = None
        for row in display.itertuples(index=False):
            row_dict = row._asdict()
            current_section = row_dict.get("section")
            if current_section is not None and last_section is not None:
                if current_section != last_section:
                    file_obj.write(r"\midrule" + "\n")
            if current_section is not None:
                last_section = current_section

            cells = [_escape_latex(row_dict[column]) for column in spec.column_order]
            file_obj.write(" & ".join(cells) + r" \\" + "\n")

        file_obj.write(r"\bottomrule" + "\n")
        file_obj.write(r"\end{tabular}" + "\n")

    logger.info("Wrote LaTeX table: %s", path)


def _write_leave_one_out_latex_table(
    df: pd.DataFrame,
    path: Path,
    *,
    participant_count: int,
) -> None:
    """Write the leave-one-participant-out table without text scaling.

    Parameters
    ----------
    df:
        Claim-level leave-one-participant-out robustness summary.
    path:
        Output ``.tex`` path.
    participant_count:
        Number of included participants.
    """
    if df.empty:
        logger.warning("Skipping empty LaTeX table: %s", path)
        return

    display = df.copy()
    display["category"] = display["category"].replace(
        {
            "facilitates harm": "Facilitates harm",
            "discourages harm": "Discourages harm",
            "sycophancy": "Sycophancy",
            "delusional": "Delusional",
            "relationship": "Relationship",
        }
    )
    for column in [
        "full_margin_pp",
        "loo_min_margin_pp",
        "loo_max_margin_pp",
        "unsupported_reruns",
    ]:
        display[column] = display[column].map(_latex_number)

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file_obj:
        file_obj.write(
            "% NOTE: This table is auto-generated; "
            "prefer not to hand-edit until the very end.\n"
        )
        file_obj.write(
            "% Positive margins mean the first-listed model has higher prevalence.\n"
        )
        file_obj.write(
            "% Unsupported reruns counts leave-one-participant-out reruns "
            "where the expected ordering is not strictly positive.\n"
        )
        file_obj.write(f"% Participants included: {participant_count}.\n")
        file_obj.write(r"\centering" + "\n")
        file_obj.write(r"\setlength{\tabcolsep}{4pt}" + "\n")
        file_obj.write(r"\begin{tabular}{@{}p{2.1in}p{1.1in}rrrr@{}}" + "\n")
        file_obj.write(r"\toprule" + "\n")
        file_obj.write(
            r"Comparison & Category & Full & LOO min & LOO max & Unsup. \\" + "\n"
        )
        file_obj.write(r"\midrule" + "\n")

        last_section = None
        for row in display.itertuples(index=False):
            row_dict = row._asdict()
            current_section = str(row_dict["section"])
            if current_section != last_section:
                if last_section is not None:
                    file_obj.write(r"\midrule" + "\n")
                file_obj.write(
                    r"\multicolumn{6}{@{}l}{\textbf{"
                    + _escape_latex(current_section)
                    + r"}} \\"
                    + "\n"
                )
                last_section = current_section

            file_obj.write(
                " & ".join(
                    [
                        _escape_latex(row_dict["comparison"]),
                        _escape_latex(row_dict["category"]),
                        _escape_latex(row_dict["full_margin_pp"]),
                        _escape_latex(row_dict["loo_min_margin_pp"]),
                        _escape_latex(row_dict["loo_max_margin_pp"]),
                        _escape_latex(row_dict["unsupported_reruns"]),
                    ]
                )
                + r" \\"
                + "\n"
            )

        file_obj.write(r"\bottomrule" + "\n")
        file_obj.write(r"\end{tabular}" + "\n")

    logger.info("Wrote LaTeX table: %s", path)


def _write_csv(df: pd.DataFrame, filename: str) -> None:
    """Write one CSV artifact into ``analysis/data``.

    Parameters
    ----------
    df:
        Dataframe to write.
    filename:
        Output filename within ``analysis/data``.
    """
    if df.empty:
        logger.warning("Skipping empty output: %s", filename)
        return
    path = DATA_OUTPUT_DIR / filename
    df.to_csv(path, index=False)
    logger.info("Wrote %s", path)


def _write_spread_outputs(
    spread_summary: pd.DataFrame,
    *,
    participant_count: int,
) -> None:
    """Write participant spread CSV and LaTeX outputs.

    Parameters
    ----------
    spread_summary:
        Anonymized participant spread summary.
    participant_count:
        Number of included participants.
    """
    _write_csv(spread_summary, "participant_spread_summary.csv")

    latex_frame = spread_summary.copy()
    for column in ["min", "median", "mean", "max"]:
        latex_frame[column] = latex_frame[column].map(_latex_number)
    _write_latex_table(
        latex_frame,
        TABLE_OUTPUT_DIR / "participant_spread_summary.tex",
        LatexTableSpec(
            header_labels={
                "metric": "Metric",
                "min": "Min",
                "median": "Median",
                "mean": "Mean",
                "max": "Max",
            },
            column_order=["metric", "min", "median", "mean", "max"],
            col_spec="lrrrr",
            notes=[f"Participants included: {participant_count}."],
        ),
    )


def _write_clustered_main_comparison_outputs(
    comparison_summary: pd.DataFrame,
    *,
    participant_count: int,
) -> None:
    """Write clustered main-comparison CSV and LaTeX outputs.

    Parameters
    ----------
    comparison_summary:
        Claim-level participant-clustered margin summary.
    participant_count:
        Number of participants included in the analysis pool.
    """
    _write_csv(
        comparison_summary,
        "participant_clustered_main_comparisons.csv",
    )

    latex_frame = comparison_summary.copy()
    latex_frame["category"] = latex_frame["category"].replace(
        {
            "facilitates harm": "Facilitates harm",
            "discourages harm": "Discourages harm",
            "sycophancy": "Sycophancy",
            "delusional": "Delusional",
            "relationship": "Relationship",
        }
    )
    for column in [
        "n_participants_supported",
        "full_margin_pp",
        "cluster_ci_low_pp",
        "cluster_ci_high_pp",
    ]:
        latex_frame[column] = latex_frame[column].map(_latex_number)

    _write_latex_table(
        latex_frame,
        TABLE_OUTPUT_DIR / "participant_clustered_main_comparisons.tex",
        LatexTableSpec(
            header_labels={
                "section": "Section",
                "comparison": "Comparison",
                "category": "Category",
                "n_participants_supported": "N",
                "full_margin_pp": "Full",
                "cluster_ci_low_pp": "CI low",
                "cluster_ci_high_pp": "CI high",
            },
            column_order=[
                "section",
                "comparison",
                "category",
                "n_participants_supported",
                "full_margin_pp",
                "cluster_ci_low_pp",
                "cluster_ci_high_pp",
            ],
            notes=[
                "Positive margins mean the first-listed model has higher prevalence.",
                f"Participants available for resampling: {participant_count}.",
                f"Cluster bootstrap draws per row: {_CLUSTER_BOOT_CONFIG.n_boot}.",
            ],
            preamble_lines=(
                r"\scriptsize",
                r"\setlength{\tabcolsep}{4pt}",
            ),
            col_spec="@{}p{0.6in}p{1.45in}p{0.82in}rlll@{}",
        ),
    )


def _write_leave_one_out_outputs(
    comparison_summary: pd.DataFrame,
    *,
    participant_count: int,
) -> None:
    """Write leave-one-participant-out CSV and LaTeX outputs.

    Parameters
    ----------
    comparison_summary:
        Claim-level leave-one-out robustness summary.
    participant_count:
        Number of included participants.
    """
    _write_csv(
        comparison_summary,
        "participant_leave_one_out_main_comparisons.csv",
    )
    _write_leave_one_out_latex_table(
        comparison_summary,
        TABLE_OUTPUT_DIR / "participant_leave_one_out_main_comparisons.tex",
        participant_count=participant_count,
    )


def _write_clustered_prevalence_outputs(prevalence_summary: pd.DataFrame) -> None:
    """Write clustered model/category prevalence CSV output.

    Parameters
    ----------
    prevalence_summary:
        Participant-clustered model/category prevalence summary.
    """
    _write_csv(
        prevalence_summary,
        "participant_clustered_prevalence_by_model_category.csv",
    )


def main() -> None:
    """Compute participant-clustered and leave-one-out robustness tables."""
    parser = argparse.ArgumentParser(
        description=(
            "Compute anonymized participant spread and leave-one-out "
            "robustness summaries."
        )
    )
    parser.add_argument(
        "--windows-path",
        default="",
        help=(
            "Optional windows parquet override. Defaults to "
            "LLM_DELUSIONS_WINDOWS_PATH or the repo default."
        ),
    )
    parser.add_argument(
        "--excluded-participants",
        default=None,
        help=(
            "Comma-separated participant IDs to exclude. If omitted, uses "
            "LLM_DELUSIONS_EXCLUDED_PARTICIPANTS with default fallback "
            '(default: include all). Pass "" to disable exclusions.'
        ),
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose logging.")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )
    ensure_output_dirs(DATA_OUTPUT_DIR, TABLE_OUTPUT_DIR)

    windows_path = resolve_participant_windows_path(args.windows_path)
    excluded_participants = resolve_excluded_participants(args.excluded_participants)
    logger.info("Using windows path: %s", windows_path)
    logger.info("Excluded participants: %s", sorted(excluded_participants))

    participant_summary = _load_participant_summary(
        windows_path,
        excluded_participants=excluded_participants,
    )
    spread_summary = _build_participant_spread_summary(participant_summary)
    _write_spread_outputs(
        spread_summary,
        participant_count=int(participant_summary["participant"].nunique()),
    )

    eval_rows = load_all_eval_data(include_original_transcript=True)
    if eval_rows.empty:
        logger.warning(
            "No row-level eval data available from report/eval_rows.parquet."
        )
        return
    eval_rows = attach_participant_ids(
        eval_rows,
        windows_path=windows_path,
        excluded_participants=excluded_participants,
    )
    participant_counts = aggregate_participant_value_sums(
        eval_rows.assign(
            report_category=eval_rows.apply(_report_category_group, axis=1)
        )
        .query("report_category in @_CATEGORY_GROUPS")
        .assign(category=lambda frame: frame["report_category"]),
        ["model_label", "category"],
        value_col="score",
    ).rename(columns={"value_sum": "positive", "value_count": "total"})
    counts = _aggregate_participant_counts(eval_rows)
    if counts.empty or participant_counts.empty:
        logger.warning(
            "No participant-mapped scored rows available for leave-one-out "
            "robustness outputs."
        )
        return

    participants, arrays = build_aligned_value_lookup(
        participant_counts,
        ["model_label", "category"],
        sum_col="positive",
        count_col="total",
    )
    conversation_lookup = build_conversation_value_lookup(
        counts,
        ["model_label", "category"],
        sum_col="positive",
        count_col="total",
    )
    clustered_summary = _build_clustered_main_comparison_summary(
        conversation_lookup,
        config=_CLUSTER_BOOT_CONFIG,
    )
    prevalence_summary = _build_clustered_prevalence_summary(
        conversation_lookup,
        config=_CLUSTER_BOOT_CONFIG,
    )
    comparison_summary = _build_leave_one_out_comparison_summary(participants, arrays)
    _write_clustered_main_comparison_outputs(
        clustered_summary,
        participant_count=int(participants.size),
    )
    _write_clustered_prevalence_outputs(prevalence_summary)
    _write_leave_one_out_outputs(
        comparison_summary,
        participant_count=int(participants.size),
    )
    logger.info("Done.")


if __name__ == "__main__":
    main()
