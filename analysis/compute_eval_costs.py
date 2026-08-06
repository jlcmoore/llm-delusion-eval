"""Compute real evaluation token usage and cost from Inspect ``.eval`` logs.

This script scans ``logs/`` and ``logs-context/``, deduplicates runs by
configuration, excludes incomplete runs, aggregates token usage by
``(evaluated model, reasoning effort, grader model)``, and computes dollar
costs using a pinned pricing snapshot.

Outputs:
- ``analysis/data/eval_costs_by_model_reasoning.csv``
- ``analysis/tables/eval_costs_by_model_reasoning.tex``
- ``analysis/data/eval_cost_run_selection.csv`` (selection audit trail)

Optional:
- Copy the generated ``.tex`` file to the sibling Overleaf repo.
"""

import argparse
import json
import re
import shutil
import zipfile
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from analysis.artifact_paths import DATA_DIR, TABLE_DIR, ensure_output_dirs
from analysis.load_eval_data import _EVALS_REPO_ROOT
from llm_delusion_eval.constants import format_model_label

DEFAULT_LOGS_DIR = _EVALS_REPO_ROOT / "logs"
DEFAULT_CONTEXT_LOGS_DIR = _EVALS_REPO_ROOT / "logs-context"
DEFAULT_PRICING_PATH = (
    _EVALS_REPO_ROOT
    / "analysis"
    / "data"
    / "paper_model_pricing_snapshot_2026-05-05.json"
)
DEFAULT_OUTPUT_CSV = DATA_DIR / "eval_costs_by_model_reasoning.csv"
DEFAULT_OUTPUT_TEX = TABLE_DIR / "eval_costs_by_model_reasoning.tex"
DEFAULT_SELECTION_CSV = DATA_DIR / "eval_cost_run_selection.csv"
DEFAULT_OVERLEAF_ROOT = _EVALS_REPO_ROOT.parent / "llm-delusions-eval-overleaf"
DEFAULT_OVERLEAF_TEX_RELATIVE_PATH = Path("tables/eval_costs_by_model_reasoning.tex")

THOUSAND = 1_000
MILLION = 1_000_000
BILLION = 1_000_000_000


@dataclass(frozen=True)
class PricingRecord:
    """Per-model input/output pricing.

    Parameters
    ----------
    input_cost_per_token:
        Dollar price per input token.
    output_cost_per_token:
        Dollar price per output token.
    source_model_id:
        Source model identifier used by the pricing source.
    """

    input_cost_per_token: float
    output_cost_per_token: float
    source_model_id: str


@dataclass(frozen=True)
class RunRecord:
    """Metadata summary for one eval log.

    Parameters
    ----------
    log_path:
        Path to the ``.eval`` file.
    logs_group:
        Parent directory name (for example ``logs`` or ``logs-context``).
    model:
        Evaluated model identifier.
    reasoning_effort:
        Reasoning setting from ``model_generate_config``.
    grader_model:
        Grader model identifier.
    max_context_messages:
        Requested context depth.
    max_windows:
        Max windows task argument.
    normalized_codes:
        Normalized sorted code tuple used in dedup keys.
    codes_raw:
        Raw ``codes`` value from ``task_args``.
    sample_count:
        Number of sample entries found in the zip.
    expected_samples:
        Declared dataset sample count in manifest.
    eval_error:
        Eval-level error field from manifest.
    """

    log_path: Path
    logs_group: str
    model: str
    reasoning_effort: str
    grader_model: str
    max_context_messages: int
    max_windows: int
    normalized_codes: tuple[str, ...]
    codes_raw: str
    sample_count: int
    expected_samples: int
    eval_error: str | None

    @property
    def config_key(self) -> tuple[Any, ...]:
        """Build the deduplication key for this run."""
        return (
            self.model,
            self.reasoning_effort,
            self.grader_model,
            self.max_context_messages,
            self.max_windows,
            self.normalized_codes,
        )

    @property
    def is_complete(self) -> bool:
        """Return ``True`` when run has no eval error and full sample count."""
        return (
            self.eval_error in (None, "")
            and self.expected_samples > 0
            and self.sample_count >= self.expected_samples
        )


@dataclass(frozen=True)
class EvalCostExportConfig:
    """Filesystem and export configuration for eval cost computation.

    Parameters
    ----------
    logs_dir:
        Main eval log directory.
    context_logs_dir:
        Context-depth eval log directory.
    pricing_path:
        Pricing snapshot JSON path.
    output_csv:
        Destination CSV path for aggregated rows.
    output_tex:
        Destination LaTeX table path.
    selection_csv:
        Destination CSV path for run selection audit rows.
    overleaf_root:
        Optional Overleaf repository root for copying the generated table.
    overleaf_tex_relative_path:
        Relative destination path under ``overleaf_root``.
    """

    logs_dir: Path
    context_logs_dir: Path
    pricing_path: Path
    output_csv: Path
    output_tex: Path
    selection_csv: Path
    overleaf_root: Path | None = None
    overleaf_tex_relative_path: Path = DEFAULT_OVERLEAF_TEX_RELATIVE_PATH


def _normalize_reasoning(reasoning_effort: Any) -> str:
    """Normalize reasoning effort into a stable string value."""
    if reasoning_effort in (None, "", "None"):
        return "none"
    return str(reasoning_effort).strip()


def _normalize_codes(codes_value: Any) -> tuple[str, ...]:
    """Normalize task ``codes`` into a sorted tuple."""
    if isinstance(codes_value, list):
        raw_codes = [str(entry).strip() for entry in codes_value if str(entry).strip()]
    elif isinstance(codes_value, str):
        raw_codes = [entry.strip() for entry in codes_value.split(",") if entry.strip()]
    else:
        raw_codes = []
    return tuple(sorted(raw_codes))


def _read_run_record(log_path: Path) -> RunRecord:
    """Read one ``.eval`` file and return run metadata."""
    with zipfile.ZipFile(log_path, "r") as zip_file:
        start_payload = json.loads(zip_file.read("_journal/start.json"))
        sample_count = sum(
            1 for info in zip_file.infolist() if info.filename.startswith("samples/")
        )

    eval_info = start_payload.get("eval", {})
    task_args = eval_info.get("task_args", {}) or {}
    model_roles = eval_info.get("model_roles", {}) or {}
    grader = model_roles.get("grader", {}) or {}
    reasoning_raw = (eval_info.get("model_generate_config", {}) or {}).get(
        "reasoning_effort"
    )
    dataset_info = eval_info.get("dataset", {}) or {}

    expected_samples_raw = dataset_info.get("samples", 0)
    expected_samples = (
        int(expected_samples_raw) if isinstance(expected_samples_raw, int) else 0
    )

    codes_value = task_args.get("codes")
    return RunRecord(
        log_path=log_path,
        logs_group=log_path.parent.name,
        model=str(eval_info.get("model", "unknown")),
        reasoning_effort=_normalize_reasoning(reasoning_raw),
        grader_model=str(grader.get("model", "unknown")),
        max_context_messages=int(task_args.get("max_context_messages", 0)),
        max_windows=int(task_args.get("max_windows", 0)),
        normalized_codes=_normalize_codes(codes_value),
        codes_raw=str(codes_value),
        sample_count=sample_count,
        expected_samples=expected_samples,
        eval_error=eval_info.get("error"),
    )


def _load_pricing(pricing_path: Path) -> tuple[str, dict[str, PricingRecord]]:
    """Load pinned pricing data from JSON."""
    with pricing_path.open("r", encoding="utf-8") as file_obj:
        payload = json.load(file_obj)

    snapshot_date = str(payload.get("snapshot_date", "unknown"))
    model_blob = payload.get("models")
    if not isinstance(model_blob, dict):
        raise ValueError(f"Invalid pricing snapshot format at {pricing_path}.")

    records: dict[str, PricingRecord] = {}
    for model_id, entry in model_blob.items():
        if not isinstance(entry, dict):
            raise ValueError(f"Invalid pricing entry for model '{model_id}'.")
        records[model_id] = PricingRecord(
            input_cost_per_token=float(entry["input_cost_per_token"]),
            output_cost_per_token=float(entry["output_cost_per_token"]),
            source_model_id=str(entry.get("source_model_id", model_id)),
        )

    return snapshot_date, records


def _iter_usage_entries(zip_file: zipfile.ZipFile):
    """Yield sample-like usage entries from summaries or sample files."""
    summary_names = sorted(
        name for name in zip_file.namelist() if name.startswith("_journal/summaries/")
    )
    if summary_names:
        for summary_name in summary_names:
            try:
                payload = json.loads(zip_file.read(summary_name))
            except (json.JSONDecodeError, EOFError):
                continue
            if isinstance(payload, list):
                for row in payload:
                    if isinstance(row, dict):
                        yield row
        return

    sample_names = sorted(
        name for name in zip_file.namelist() if name.startswith("samples/")
    )
    for sample_name in sample_names:
        try:
            row = json.loads(zip_file.read(sample_name))
        except (json.JSONDecodeError, EOFError):
            continue
        if isinstance(row, dict):
            yield row


def _usage_aliases(model_id: str) -> tuple[str, ...]:
    """Return possible model-id aliases observed in usage blobs."""
    aliases = [model_id]

    if model_id.startswith("openai/"):
        aliases.append(model_id.removeprefix("openai/"))
    if model_id.startswith("anthropic/"):
        aliases.append(model_id.removeprefix("anthropic/"))
    if model_id.startswith("google/vertex/"):
        aliases.append(model_id.removeprefix("google/vertex/"))
        aliases.append(f"vertex_ai/{model_id.removeprefix('google/vertex/')}")
    if model_id.startswith("together/"):
        aliases.append(model_id.replace("together/", "together_ai/", 1))
    if model_id.startswith("grok/"):
        xai_alias = model_id.replace("grok/", "xai/", 1)
        aliases.append(xai_alias)
        aliases.append(
            xai_alias.replace("-0309-non-reasoning", "-beta-0309-non-reasoning")
        )
        aliases.append(xai_alias.replace("-0309-reasoning", "-beta-0309-reasoning"))

    unique = []
    seen = set()
    for alias in aliases:
        if alias and alias not in seen:
            seen.add(alias)
            unique.append(alias)
    return tuple(unique)


def _find_usage_for_model(
    usage_by_model: dict[str, Any], model_id: str
) -> dict[str, Any] | None:
    """Find one model usage entry using exact or alias matching."""
    for alias in _usage_aliases(model_id):
        if alias in usage_by_model and isinstance(usage_by_model[alias], dict):
            return usage_by_model[alias]

    target_suffix = model_id.split("/")[-1]
    for usage_model_id, usage_payload in usage_by_model.items():
        if not isinstance(usage_payload, dict):
            continue
        if usage_model_id.split("/")[-1] == target_suffix:
            return usage_payload
    return None


def _aggregate_run_usage(run: RunRecord) -> dict[str, int]:
    """Aggregate evaluated-model and grader usage for one selected run."""
    totals = {
        "eval_input_tokens": 0,
        "eval_output_tokens": 0,
        "grader_input_tokens": 0,
        "grader_output_tokens": 0,
        "missing_eval_usage_rows": 0,
        "missing_grader_usage_rows": 0,
        "duplicate_sample_rows_skipped": 0,
    }
    seen_sample_ids: set[str] = set()

    with zipfile.ZipFile(run.log_path, "r") as zip_file:
        for row in _iter_usage_entries(zip_file):
            sample_id = str(row.get("id", ""))
            if sample_id:
                if sample_id in seen_sample_ids:
                    totals["duplicate_sample_rows_skipped"] += 1
                    continue
                seen_sample_ids.add(sample_id)

            if row.get("completed") is False:
                continue

            usage_by_model = row.get("model_usage", {}) or {}
            if not isinstance(usage_by_model, dict):
                continue

            eval_usage = _find_usage_for_model(usage_by_model, run.model)
            if eval_usage is None:
                totals["missing_eval_usage_rows"] += 1
            else:
                totals["eval_input_tokens"] += int(
                    eval_usage.get("input_tokens", 0) or 0
                )
                totals["eval_output_tokens"] += int(
                    eval_usage.get("output_tokens", 0) or 0
                )

            grader_usage = _find_usage_for_model(usage_by_model, run.grader_model)
            if grader_usage is None:
                totals["missing_grader_usage_rows"] += 1
            else:
                totals["grader_input_tokens"] += int(
                    grader_usage.get("input_tokens", 0) or 0
                )
                totals["grader_output_tokens"] += int(
                    grader_usage.get("output_tokens", 0) or 0
                )

    return totals


def _format_compact_count(value: int) -> str:
    """Format integer using compact ``k/m/b`` suffixes for LaTeX output."""
    absolute = abs(int(value))
    if absolute < THOUSAND:
        return str(int(value))
    if absolute < MILLION:
        return f"{int(round(value / THOUSAND))}k"
    if absolute < BILLION:
        return f"{int(round(value / MILLION))}m"
    return f"{int(round(value / BILLION))}b"


def _format_usd_whole(value: float) -> str:
    """Format dollar amount with no cents for LaTeX output."""
    if 0 < abs(value) < 1:
        return "<1"
    return f"{int(round(value)):,}"


def _strip_date_stamp(label: str) -> str:
    """Strip a trailing ``-YYYY-MM-DD`` stamp from a model label."""
    return re.sub(r"-\d{4}-\d{2}-\d{2}$", "", label)


def _format_grader_label(model_id: str) -> str:
    """Format grader model label without date stamps."""
    raw_label = format_model_label(model_id, None)
    return _strip_date_stamp(raw_label)


def _model_stub(model_id: str) -> str:
    """Return the raw model stub (path suffix) from a full model ID."""
    return str(model_id).split("/")[-1]


def _collect_run_records(logs_dir: Path, context_logs_dir: Path) -> list[RunRecord]:
    """Collect run records from both log directories."""
    candidate_paths = sorted(logs_dir.glob("*.eval")) + sorted(
        context_logs_dir.glob("*.eval")
    )
    return [_read_run_record(log_path) for log_path in candidate_paths]


def _select_runs(
    runs: list[RunRecord],
) -> tuple[list[RunRecord], list[dict[str, Any]], int, int]:
    """Select best complete runs and build selection-audit rows."""
    grouped: dict[tuple[Any, ...], list[RunRecord]] = defaultdict(list)
    for run in runs:
        grouped[run.config_key].append(run)

    selected_runs: list[RunRecord] = []
    selection_rows: list[dict[str, Any]] = []
    excluded_duplicates = 0
    excluded_incomplete = 0

    for entries in grouped.values():
        ranked = sorted(
            entries,
            key=lambda entry: (entry.sample_count, entry.log_path.name),
            reverse=True,
        )
        best = ranked[0]
        for idx, run in enumerate(ranked):
            if idx > 0:
                status = "excluded_duplicate"
                excluded_duplicates += 1
            elif run.is_complete:
                status = "selected"
                selected_runs.append(run)
            else:
                status = "excluded_incomplete"
                excluded_incomplete += 1

            selection_rows.append(
                {
                    "status": status,
                    "log_path": str(run.log_path),
                    "logs_group": run.logs_group,
                    "model": run.model,
                    "reasoning_effort": run.reasoning_effort,
                    "grader_model": run.grader_model,
                    "max_context_messages": run.max_context_messages,
                    "max_windows": run.max_windows,
                    "normalized_codes": "|".join(run.normalized_codes),
                    "codes_raw": run.codes_raw,
                    "sample_count": run.sample_count,
                    "expected_samples": run.expected_samples,
                    "eval_error": run.eval_error or "",
                    "best_log_for_config": str(best.log_path),
                }
            )

    return selected_runs, selection_rows, excluded_duplicates, excluded_incomplete


def _build_run_rows(
    selected_runs: list[RunRecord],
    pricing: dict[str, PricingRecord],
    snapshot_date: str,
    pricing_path: Path,
) -> list[dict[str, Any]]:
    """Build one cost row per selected run."""
    run_rows: list[dict[str, Any]] = []
    for run in selected_runs:
        if run.model not in pricing:
            raise KeyError(
                f"Missing pricing for evaluated model '{run.model}' in {pricing_path}."
            )
        if run.grader_model not in pricing:
            raise KeyError(
                (
                    "Missing pricing for grader model "
                    f"'{run.grader_model}' in {pricing_path}."
                )
            )

        usage_totals = _aggregate_run_usage(run)
        eval_pricing = pricing[run.model]
        grader_pricing = pricing[run.grader_model]

        eval_input_usd = (
            usage_totals["eval_input_tokens"] * eval_pricing.input_cost_per_token
        )
        eval_output_usd = (
            usage_totals["eval_output_tokens"] * eval_pricing.output_cost_per_token
        )
        grader_input_usd = (
            usage_totals["grader_input_tokens"] * grader_pricing.input_cost_per_token
        )
        grader_output_usd = (
            usage_totals["grader_output_tokens"] * grader_pricing.output_cost_per_token
        )

        run_rows.append(
            {
                "log_path": str(run.log_path),
                "logs_group": run.logs_group,
                "model": run.model,
                "reasoning_effort": run.reasoning_effort,
                "grader_model": run.grader_model,
                "sample_count": run.sample_count,
                "max_context_messages": run.max_context_messages,
                "max_windows": run.max_windows,
                "normalized_codes": "|".join(run.normalized_codes),
                "eval_input_tokens": usage_totals["eval_input_tokens"],
                "eval_output_tokens": usage_totals["eval_output_tokens"],
                "grader_input_tokens": usage_totals["grader_input_tokens"],
                "grader_output_tokens": usage_totals["grader_output_tokens"],
                "missing_eval_usage_rows": usage_totals["missing_eval_usage_rows"],
                "missing_grader_usage_rows": usage_totals["missing_grader_usage_rows"],
                "duplicate_sample_rows_skipped": usage_totals[
                    "duplicate_sample_rows_skipped"
                ],
                "eval_input_usd": eval_input_usd,
                "eval_output_usd": eval_output_usd,
                "grader_input_usd": grader_input_usd,
                "grader_output_usd": grader_output_usd,
                "eval_llm_cost_usd": eval_input_usd + eval_output_usd,
                "grader_llm_cost_usd": grader_input_usd + grader_output_usd,
                "total_cost_usd": eval_input_usd
                + eval_output_usd
                + grader_input_usd
                + grader_output_usd,
                "eval_input_usd_per_million": eval_pricing.input_cost_per_token * 1e6,
                "eval_output_usd_per_million": eval_pricing.output_cost_per_token * 1e6,
                "grader_input_usd_per_million": grader_pricing.input_cost_per_token
                * 1e6,
                "grader_output_usd_per_million": grader_pricing.output_cost_per_token
                * 1e6,
                "pricing_snapshot_date": snapshot_date,
            }
        )
    return run_rows


def _build_grouped_with_total(
    run_rows: list[dict[str, Any]], snapshot_date: str
) -> pd.DataFrame:
    """Aggregate selected runs and append an overall total row."""
    run_frame = pd.DataFrame(run_rows)
    grouped_frame = (
        run_frame.groupby(["model", "reasoning_effort", "grader_model"], as_index=False)
        .agg(
            runs_included=("log_path", "count"),
            eval_input_tokens=("eval_input_tokens", "sum"),
            eval_output_tokens=("eval_output_tokens", "sum"),
            grader_input_tokens=("grader_input_tokens", "sum"),
            grader_output_tokens=("grader_output_tokens", "sum"),
            eval_input_usd=("eval_input_usd", "sum"),
            eval_output_usd=("eval_output_usd", "sum"),
            grader_input_usd=("grader_input_usd", "sum"),
            grader_output_usd=("grader_output_usd", "sum"),
            eval_llm_cost_usd=("eval_llm_cost_usd", "sum"),
            grader_llm_cost_usd=("grader_llm_cost_usd", "sum"),
            total_cost_usd=("total_cost_usd", "sum"),
            eval_input_usd_per_million=("eval_input_usd_per_million", "first"),
            eval_output_usd_per_million=("eval_output_usd_per_million", "first"),
            grader_input_usd_per_million=("grader_input_usd_per_million", "first"),
            grader_output_usd_per_million=("grader_output_usd_per_million", "first"),
        )
        .sort_values(
            ["runs_included", "total_cost_usd", "model", "reasoning_effort"],
            ascending=[False, False, True, True],
        )
    )

    grouped_frame.insert(
        0,
        "model_label",
        grouped_frame.apply(
            lambda row: format_model_label(row["model"], row["reasoning_effort"]),
            axis=1,
        ),
    )
    grouped_frame.insert(1, "model_stub", grouped_frame["model"].map(_model_stub))
    grouped_frame.insert(
        4,
        "grader_label",
        grouped_frame["grader_model"].map(_format_grader_label),
    )
    grouped_frame["pricing_snapshot_date"] = snapshot_date

    unique_grader_labels = sorted(set(grouped_frame["grader_label"].tolist()))
    unique_grader_models = sorted(set(grouped_frame["grader_model"].tolist()))
    total_grader_label = (
        unique_grader_labels[0] if len(unique_grader_labels) == 1 else "mixed"
    )
    total_grader_model = (
        unique_grader_models[0] if len(unique_grader_models) == 1 else "mixed"
    )

    total_row = {
        "model_label": "All selected runs",
        "model_stub": "All selected runs",
        "model": "__all__",
        "reasoning_effort": "all",
        "grader_label": total_grader_label,
        "grader_model": total_grader_model,
        "runs_included": int(grouped_frame["runs_included"].sum()),
        "eval_input_tokens": int(grouped_frame["eval_input_tokens"].sum()),
        "eval_output_tokens": int(grouped_frame["eval_output_tokens"].sum()),
        "grader_input_tokens": int(grouped_frame["grader_input_tokens"].sum()),
        "grader_output_tokens": int(grouped_frame["grader_output_tokens"].sum()),
        "eval_input_usd": float(grouped_frame["eval_input_usd"].sum()),
        "eval_output_usd": float(grouped_frame["eval_output_usd"].sum()),
        "grader_input_usd": float(grouped_frame["grader_input_usd"].sum()),
        "grader_output_usd": float(grouped_frame["grader_output_usd"].sum()),
        "eval_llm_cost_usd": float(grouped_frame["eval_llm_cost_usd"].sum()),
        "grader_llm_cost_usd": float(grouped_frame["grader_llm_cost_usd"].sum()),
        "total_cost_usd": float(grouped_frame["total_cost_usd"].sum()),
        "eval_input_usd_per_million": float("nan"),
        "eval_output_usd_per_million": float("nan"),
        "grader_input_usd_per_million": float("nan"),
        "grader_output_usd_per_million": float("nan"),
        "pricing_snapshot_date": snapshot_date,
    }
    return pd.concat([grouped_frame, pd.DataFrame([total_row])], ignore_index=True)


def _build_latex_text(grouped_with_total: pd.DataFrame, snapshot_date: str) -> str:
    """Render the LaTeX table from grouped data."""
    latex_columns = [
        "model_stub",
        "reasoning_effort",
        "grader_label",
        "runs_included",
        "eval_input_tokens",
        "eval_output_tokens",
        "grader_input_tokens",
        "grader_output_tokens",
        "eval_llm_cost_usd",
        "grader_llm_cost_usd",
        "total_cost_usd",
    ]
    latex_frame = grouped_with_total[latex_columns].copy()
    latex_frame = latex_frame.rename(
        columns={
            "model_stub": "Model",
            "reasoning_effort": "Reasoning",
            "grader_label": "Grader",
            "runs_included": "n",
            "eval_input_tokens": r"\shortstack{Eval In\\Tok}",
            "eval_output_tokens": r"\shortstack{Eval Out\\Tok}",
            "grader_input_tokens": r"\shortstack{Grader In\\Tok}",
            "grader_output_tokens": r"\shortstack{Grader Out\\Tok}",
            "eval_llm_cost_usd": r"\shortstack{Eval LLM\\USD}",
            "grader_llm_cost_usd": r"\shortstack{Grader LLM\\USD}",
            "total_cost_usd": r"\shortstack{Total\\USD}",
        }
    )

    numeric_runs = grouped_with_total["runs_included"].tolist()
    first_zero_run_index = next(
        (idx for idx, count in enumerate(numeric_runs[:-1]) if int(count) == 0),
        None,
    )
    first_zero_model_label = (
        str(latex_frame.iloc[first_zero_run_index]["Model"])
        if first_zero_run_index is not None
        else None
    )

    for token_column in [
        r"\shortstack{Eval In\\Tok}",
        r"\shortstack{Eval Out\\Tok}",
        r"\shortstack{Grader In\\Tok}",
        r"\shortstack{Grader Out\\Tok}",
        "n",
    ]:
        latex_frame[token_column] = latex_frame[token_column].map(_format_compact_count)

    for dollar_column in [
        r"\shortstack{Eval LLM\\USD}",
        r"\shortstack{Grader LLM\\USD}",
        r"\shortstack{Total\\USD}",
    ]:
        latex_frame[dollar_column] = latex_frame[dollar_column].map(
            lambda value: "--" if pd.isna(value) else _format_usd_whole(float(value))
        )

    tabular_text = latex_frame.to_latex(
        index=False,
        escape=False,
        column_format="p{0.22\\linewidth}llrrrrrrrr",
    )
    latex_text = (
        "% NOTE: This table is auto-generated by analysis.compute_eval_costs.\n"
        f"% Pricing snapshot date: {snapshot_date}\n"
        "\\begingroup\n"
        "\\setlength{\\tabcolsep}{3pt}\n"
        "\\footnotesize\n"
        f"{tabular_text}"
        "\\endgroup\n"
    )
    if first_zero_model_label is not None:
        zero_marker = f"\n{first_zero_model_label} &"
        if zero_marker in latex_text:
            latex_text = latex_text.replace(
                zero_marker, f"\n\\midrule\n{first_zero_model_label} &", 1
            )
    return latex_text.replace(
        "\nAll selected runs &", "\n\\midrule\nAll selected runs &", 1
    )


def _write_outputs(
    config: EvalCostExportConfig,
    grouped_with_total: pd.DataFrame,
    selection_rows: list[dict[str, Any]],
    latex_text: str,
) -> None:
    """Write CSV/LaTeX outputs and optionally copy to Overleaf."""
    ensure_output_dirs(
        config.output_csv.parent,
        config.output_tex.parent,
        config.selection_csv.parent,
    )
    grouped_with_total.to_csv(config.output_csv, index=False)
    pd.DataFrame(selection_rows).sort_values(
        ["status", "model", "reasoning_effort", "log_path"]
    ).to_csv(config.selection_csv, index=False)
    config.output_tex.write_text(latex_text, encoding="utf-8")

    if config.overleaf_root is not None:
        overleaf_destination = config.overleaf_root / config.overleaf_tex_relative_path
        overleaf_destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(config.output_tex, overleaf_destination)


def compute_eval_costs(config: EvalCostExportConfig) -> dict[str, Any]:
    """Compute run costs and write CSV/LaTeX outputs."""
    snapshot_date, pricing = _load_pricing(config.pricing_path)
    runs = _collect_run_records(config.logs_dir, config.context_logs_dir)
    selected_runs, selection_rows, excluded_duplicates, excluded_incomplete = (
        _select_runs(runs)
    )
    run_rows = _build_run_rows(
        selected_runs=selected_runs,
        pricing=pricing,
        snapshot_date=snapshot_date,
        pricing_path=config.pricing_path,
    )
    grouped_with_total = _build_grouped_with_total(run_rows, snapshot_date)
    latex_text = _build_latex_text(grouped_with_total, snapshot_date)
    _write_outputs(config, grouped_with_total, selection_rows, latex_text)

    return {
        "candidate_logs": len(runs),
        "selected_runs": len(selected_runs),
        "excluded_duplicates": excluded_duplicates,
        "excluded_incomplete": excluded_incomplete,
        "pricing_snapshot_date": snapshot_date,
    }


def main() -> None:
    """Parse arguments and run eval-cost export."""
    parser = argparse.ArgumentParser(
        description=(
            "Compute real token usage and pricing from .eval logs, then export "
            "CSV and LaTeX summaries by model/reasoning/grader."
        )
    )
    parser.add_argument(
        "--logs-dir",
        type=Path,
        default=DEFAULT_LOGS_DIR,
        help=f"Main logs directory (default: {DEFAULT_LOGS_DIR})",
    )
    parser.add_argument(
        "--context-logs-dir",
        type=Path,
        default=DEFAULT_CONTEXT_LOGS_DIR,
        help=f"Context logs directory (default: {DEFAULT_CONTEXT_LOGS_DIR})",
    )
    parser.add_argument(
        "--pricing-path",
        type=Path,
        default=DEFAULT_PRICING_PATH,
        help=f"Pricing snapshot JSON path (default: {DEFAULT_PRICING_PATH})",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=DEFAULT_OUTPUT_CSV,
        help=f"Output CSV path (default: {DEFAULT_OUTPUT_CSV})",
    )
    parser.add_argument(
        "--output-tex",
        type=Path,
        default=DEFAULT_OUTPUT_TEX,
        help=f"Output TeX path (default: {DEFAULT_OUTPUT_TEX})",
    )
    parser.add_argument(
        "--selection-csv",
        type=Path,
        default=DEFAULT_SELECTION_CSV,
        help=f"Run selection audit CSV path (default: {DEFAULT_SELECTION_CSV})",
    )
    parser.add_argument(
        "--overleaf-root",
        type=Path,
        default=None,
        help=(
            "Optional Overleaf root directory. If provided, the generated TeX file "
            "is copied there."
        ),
    )
    parser.add_argument(
        "--overleaf-tex-relative-path",
        type=Path,
        default=DEFAULT_OVERLEAF_TEX_RELATIVE_PATH,
        help=(
            "TeX destination path relative to overleaf root "
            f"(default: {DEFAULT_OVERLEAF_TEX_RELATIVE_PATH})"
        ),
    )
    args = parser.parse_args()

    config = EvalCostExportConfig(
        logs_dir=args.logs_dir,
        context_logs_dir=args.context_logs_dir,
        pricing_path=args.pricing_path,
        output_csv=args.output_csv,
        output_tex=args.output_tex,
        selection_csv=args.selection_csv,
        overleaf_root=args.overleaf_root,
        overleaf_tex_relative_path=args.overleaf_tex_relative_path,
    )
    summary = compute_eval_costs(config)

    print(
        "Computed eval costs from "
        f"{summary['candidate_logs']} logs; selected {summary['selected_runs']} "
        "runs after filtering "
        f"({summary['excluded_duplicates']} duplicates, "
        f"{summary['excluded_incomplete']} incomplete)."
    )
    print(f"Pricing snapshot date: {summary['pricing_snapshot_date']}")
    print(f"Wrote: {args.output_csv}")
    print(f"Wrote: {args.output_tex}")
    print(f"Wrote: {args.selection_csv}")
    if args.overleaf_root is not None:
        print(
            "Copied LaTeX table to: "
            f"{config.overleaf_root / config.overleaf_tex_relative_path}"
        )


if __name__ == "__main__":
    main()
