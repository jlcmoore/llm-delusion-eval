# ruff: noqa: PLR0912, PLR0915
"""
Consumes the output of the inspect runner and generates a JSON report
summarizing the code-level and category-level performance for each model.
"""

import argparse
import csv
import glob
import gzip
import hashlib
import json
import math
import os
import pickle
import shutil
import textwrap
import time
import zipfile
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from inspect_ai.log import EvalLog
from llm_delusions_annotations.annotation_metadata import (
    load_annotation_metadata_with_role_splits,
)
from llm_delusions_annotations.annotation_prompts import ANNOTATIONS_FILE
from plotly.subplots import make_subplots
from scipy.stats import bootstrap

from llm_delusion_eval.constants import format_model_label, get_source_id, normalize_id
from llm_delusion_eval.participant_exclusions import resolve_excluded_window_ids
from llm_delusion_eval.paths import DEFAULT_WINDOWS_PATH, resolve_path
from llm_delusion_eval.scorers.annotation_scorer import _CUTOFFS
from llm_delusion_eval.window_ids import build_eval_subset_id_from_row, build_window_id

_FACILITATES_HARM_CODES = [
    "bot-facilitates-violence",
    "bot-facilitates-self-harm",
]
_DISCOURAGES_HARM_CODES = [
    "bot-discourages-violence",
    "bot-discourages-self-harm",
]
_HARM_CATEGORY_CODE_GROUPS = {
    "facilitates harm": _FACILITATES_HARM_CODES,
    "discourages harm": _DISCOURAGES_HARM_CODES,
}

_CACHE_VERSION = 1


def _model_label(model: str, reasoning: Optional[str]) -> str:
    """Build a display label from model ID and reasoning effort."""
    return format_model_label(model, reasoning)


def _bootstrap_binary_ci(
    scores: list[float], *, n_boot: int = 10_000, seed: int = 42
) -> tuple[float, float]:
    """Return (ci_lower, ci_upper) for a list of binary scores.

    Fully vectorized: generates all resamples in one call.
    """
    arr = np.array([s for s in scores if s is not None and not math.isnan(s)])
    if len(arr) == 0:
        return (float("nan"), float("nan"))
    if len(arr) == 1:
        value = float(arr[0])
        return (value, value)

    rng = np.random.default_rng(seed)
    result = bootstrap(
        (arr,),
        np.mean,
        n_resamples=n_boot,
        confidence_level=0.95,
        method="percentile",
        vectorized=True,
        rng=rng,
    )
    return (
        float(result.confidence_interval.low),
        float(result.confidence_interval.high),
    )


def _bootstrap_delta_ci(
    scores_a: list[float],
    scores_b: list[float],
    *,
    n_boot: int = 10_000,
    seed: int = 42,
) -> tuple[float, float, float]:
    """Return (delta, ci_lower, ci_upper) for mean(a) - mean(b).

    Fully vectorized: generates all resamples in one call.
    """
    a = np.array([s for s in scores_a if s is not None and not math.isnan(s)])
    b = np.array([s for s in scores_b if s is not None and not math.isnan(s)])
    if len(a) == 0 or len(b) == 0:
        return (float("nan"), float("nan"), float("nan"))
    delta = float(a.mean() - b.mean())
    if len(a) == 1 and len(b) == 1:
        return (delta, delta, delta)

    rng = np.random.default_rng(seed)
    result = bootstrap(
        (a, b),
        _mean_delta_statistic,
        n_resamples=n_boot,
        confidence_level=0.95,
        method="percentile",
        vectorized=True,
        paired=False,
        rng=rng,
    )
    return (
        delta,
        float(result.confidence_interval.low),
        float(result.confidence_interval.high),
    )


def _mean_delta_statistic(
    values_a: np.ndarray, values_b: np.ndarray, axis: int
) -> np.ndarray:
    """Return the difference in means along the provided axis."""
    return np.mean(values_a, axis=axis) - np.mean(values_b, axis=axis)


def _replace_non_finite_floats(value: Any) -> Any:
    """Recursively convert non-finite floats to ``None`` for strict JSON dumps.

    Parameters
    ----------
    value:
        Arbitrary nested value that may include lists, tuples, and dictionaries.

    Returns
    -------
    Any
        A JSON-safe value where any ``NaN``/``Infinity`` float is replaced with
        ``None``.
    """
    if isinstance(value, dict):
        return {key: _replace_non_finite_floats(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_replace_non_finite_floats(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_replace_non_finite_floats(item) for item in value)
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    return value


@dataclass
class ReportFilter:
    """Filter criteria for the report generation."""

    max_context: int | None = None
    models: list[str] | None = None
    annotation_ids: list[str] | None = None
    grader_models: list[str] | None = None


def _iter_raw_samples(log_path: str):
    """Yield raw sample dicts from an eval log zip (skips Pydantic parsing)."""
    with zipfile.ZipFile(log_path, "r") as zf:
        for name in sorted(zf.namelist()):
            if not name.startswith("samples/"):
                continue
            try:
                yield json.loads(zf.read(name))
            except (json.JSONDecodeError, EOFError):
                continue


def _read_log_metadata(log_path: str) -> tuple[dict, int]:
    """Read manifest and sample count from an eval log zip in one pass."""
    with zipfile.ZipFile(log_path, "r") as zf:
        manifest = json.loads(zf.read("_journal/start.json"))
        sample_count = sum(
            1 for info in zf.infolist() if info.filename.startswith("samples/")
        )
    return manifest, sample_count


def _cache_paths(cache_dir: str) -> tuple[Path, Path]:
    """Return metadata cache path and sample cache directory."""
    cache_root = Path(cache_dir)
    metadata_path = cache_root / "metadata.json"
    samples_dir = cache_root / "samples"
    return metadata_path, samples_dir


def _load_metadata_cache(metadata_path: Path) -> dict[str, Any]:
    """Load metadata cache from disk."""
    if not metadata_path.exists():
        return {}
    try:
        with open(metadata_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return {}


def _save_metadata_cache(metadata_path: Path, metadata_cache: dict[str, Any]) -> None:
    """Persist metadata cache to disk atomically."""
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = metadata_path.with_suffix(".tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(metadata_cache, f)
    os.replace(tmp_path, metadata_path)


def _file_signature(path: str) -> tuple[int, int]:
    """Return size and mtime_ns for cache invalidation."""
    stat = os.stat(path)
    return stat.st_size, stat.st_mtime_ns


def _cache_key(path: str, size: int, mtime_ns: int) -> str:
    """Create a deterministic cache key for one eval file revision."""
    payload = f"{Path(path).resolve()}|{size}|{mtime_ns}|{_CACHE_VERSION}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _get_log_metadata_cached(
    log_path: str, metadata_cache: dict[str, Any]
) -> tuple[dict, int, bool]:
    """Return manifest + sample count for a log, using persistent cache."""
    abs_path = str(Path(log_path).resolve())
    size, mtime_ns = _file_signature(log_path)

    cached = metadata_cache.get(abs_path, {})
    if (
        cached.get("cache_version") == _CACHE_VERSION
        and cached.get("size") == size
        and cached.get("mtime_ns") == mtime_ns
        and "manifest" in cached
        and "sample_count" in cached
    ):
        return cached["manifest"], int(cached["sample_count"]), True

    manifest, sample_count = _read_log_metadata(log_path)
    metadata_cache[abs_path] = {
        "cache_version": _CACHE_VERSION,
        "size": size,
        "mtime_ns": mtime_ns,
        "manifest": manifest,
        "sample_count": sample_count,
    }
    return manifest, sample_count, False


def _sample_cache_file(log_path: str, samples_dir: Path) -> Path:
    """Return the cache file path for parsed samples of one log revision."""
    size, mtime_ns = _file_signature(log_path)
    return samples_dir / f"{_cache_key(log_path, size, mtime_ns)}.pkl.gz"


def _load_or_parse_samples(
    log_path: str, samples_dir: Path
) -> tuple[list[dict[str, Any]], bool]:
    """Load parsed raw samples from cache, or parse and cache them."""
    cache_file = _sample_cache_file(log_path, samples_dir)
    if cache_file.exists():
        try:
            with gzip.open(cache_file, "rb") as f:
                cached_samples = pickle.load(f)
            if isinstance(cached_samples, list):
                return cached_samples, True
        except (OSError, EOFError, pickle.UnpicklingError):
            pass

    parsed_samples = list(_iter_raw_samples(log_path))
    try:
        samples_dir.mkdir(parents=True, exist_ok=True)
        tmp_file = cache_file.with_suffix(".tmp")
        with gzip.open(tmp_file, "wb") as f:
            pickle.dump(parsed_samples, f, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp_file, cache_file)
    except OSError:
        pass

    return parsed_samples, False


def _select_best_logs(
    log_files: list[str], metadata_cache: dict[str, Any]
) -> tuple[list[tuple[str, dict]], int]:
    """Deduplicate log files: keep the one with the most samples per config.

    Returns a list of ``(log_path, manifest)`` tuples and metadata cache hits.
    """
    candidates: dict[str, list[tuple[int, str, dict]]] = defaultdict(list)
    cache_hits = 0

    for log_path in log_files:
        try:
            manifest, n_samples, was_cached = _get_log_metadata_cached(
                log_path, metadata_cache
            )
            if was_cached:
                cache_hits += 1
        except (OSError, zipfile.BadZipFile, KeyError, json.JSONDecodeError):
            continue

        eval_info = manifest.get("eval", {})
        model = eval_info.get("model", "unknown")
        reasoning = eval_info.get("model_generate_config", {}).get("reasoning_effort")
        config_key = f"{model}|{reasoning}"

        candidates[config_key].append((n_samples, log_path, manifest))

    selected = []
    for _config_key, entries in candidates.items():
        best = max(entries, key=lambda t: t[0])
        selected.append((best[1], best[2]))
        print(
            f"  Selected {os.path.basename(best[1])} "
            f"({best[0]} samples) for {_config_key}"
        )

    return selected, cache_hits


def _extract_condition_from_manifest(manifest: dict) -> dict[str, Any]:
    """Extract condition parameters from a raw manifest dict.

    Includes ``reasoning_effort`` so that models differing only by reasoning
    setting get distinct condition keys.
    """
    eval_info = manifest.get("eval", {})
    task_args = eval_info.get("task_args", {})
    config = eval_info.get("config", {})
    gen_config = eval_info.get("model_generate_config", {}) or {}

    # Look for grader model in model_roles
    grader_model = task_args.get("grader")
    if not grader_model:
        model_roles = eval_info.get("model_roles", {})
        grader_info = model_roles.get("grader")
        if isinstance(grader_info, dict):
            grader_model = grader_info.get("model")

    reasoning = (
        gen_config.get("reasoning_effort") if isinstance(gen_config, dict) else None
    )

    return {
        "max_context_messages": task_args.get("max_context_messages", 0),
        "max_windows": task_args.get("max_windows", 0),
        "limit": config.get("limit"),
        "grader_model": grader_model,
        "reasoning_effort": reasoning,
    }


def _dict_to_param_string(d: dict[str, Any]) -> str:
    """Return a compact key=value string representation of a dict for filenames."""
    sorted_items = sorted(d.items(), key=lambda item: str(item[0]))
    kv_pairs = []
    for k, v in sorted_items:
        if k == "limit":
            continue
        v_str = "None" if v is None else str(v)
        # Replace slashes to make filesystem safe without confusing HTTP servers
        v_safe = v_str.replace("/", "_").replace("\\", "_").replace(" ", "_")
        kv_pairs.append(f"{k}={v_safe}")
    return "&".join(kv_pairs)


def _extract_condition(log: EvalLog) -> Dict[str, Any]:
    """Extract condition parameters from log configuration."""
    task_args = log.eval.task_args or {}
    config = log.eval.config or {}

    grader_model = task_args.get("grader")
    if not grader_model:
        grader_config = log.eval.model_roles.get("grader")
        grader_model = grader_config.model if grader_config else None

    condition_dict = {
        "max_context_messages": task_args.get("max_context_messages", 0),
        "max_windows": task_args.get("max_windows", 0),
        "limit": getattr(config, "limit", None),
        "grader_model": grader_model,
    }

    if isinstance(condition_dict["limit"], list):
        condition_dict["limit"] = str(condition_dict["limit"])

    return condition_dict


def _get_excerpt(sample) -> str:
    """Extract text excerpt from a sample."""
    if hasattr(sample, "output") and sample.output:
        return str(sample.output.completion)
    if hasattr(sample, "messages") and sample.messages:
        last_msg = sample.messages[-1]
        if hasattr(last_msg, "content"):
            return str(last_msg.content)
    return ""


def _get_history(sample) -> list:
    """Extract conversation history from a sample."""
    history = []
    messages = getattr(sample, "input", [])
    if not messages and hasattr(sample, "messages"):
        messages = sample.messages[:-1]

    for msg in messages:
        if hasattr(msg, "role") and hasattr(msg, "content"):
            history.append({"role": str(msg.role), "content": str(msg.content)})
    return history


def _get_sample_code(sample, score_obj) -> str:
    """Determine the annotation code for a given sample."""
    code = "unknown"
    if sample.metadata and "harmful_annotations" in sample.metadata:
        annotations = sample.metadata["harmful_annotations"]
        if annotations:
            code = annotations[0]
    elif score_obj.explanation:
        try:
            explanation_data = json.loads(score_obj.explanation)
            code = explanation_data.get("annotation_id", "unknown")
        except (json.JSONDecodeError, TypeError, KeyError):
            pass

    return normalize_id(code)


def _update_aggregate(
    agg: dict, category: str, code: str, score_value: float | None
) -> None:
    """Update the score sums and counts for a single evaluation sample."""
    is_harm = category == "concerns harm"

    if score_value is not None:
        agg["codes"][code]["score_sum"] += score_value
        if is_harm:
            agg["harm_codes"][code]["score_sum"] += score_value
        else:
            agg["categories"][category]["score_sum"] += score_value
        agg["overall"]["score_sum"] += score_value

    agg["codes"][code]["count"] += 1
    if is_harm:
        agg["harm_codes"][code]["count"] += 1
    else:
        agg["categories"][category]["count"] += 1
    agg["overall"]["count"] += 1


def _process_sample(sample, metadata_map, agg, annotation_id_filter=None) -> None:
    """Process a single sample and aggregate scores."""
    if not sample.scores:
        return

    excerpt = _get_excerpt(sample)
    history = _get_history(sample)

    for score_obj in sample.scores.values():
        if not score_obj or score_obj.value is None:
            continue

        try:
            score_value = float(score_obj.value)
        except (ValueError, TypeError):
            continue

        if math.isnan(score_value):
            continue

        code = _get_sample_code(sample, score_obj)
        if annotation_id_filter and code not in annotation_id_filter:
            continue

        category = metadata_map[code].category if code in metadata_map else "unknown"
        _update_aggregate(agg, category, code, score_value)

        grader_answer = getattr(score_obj, "answer", "")
        grader_explanation = getattr(score_obj, "explanation", "")

        agg["samples"][code].append(
            {
                "sample_id": str(sample.id),
                "window_id": (
                    sample.metadata.get("window_id", "unknown")
                    if sample.metadata
                    else "unknown"
                ),
                "code": code,
                "category": category,
                "score": score_value,
                "history": history,
                "excerpt": excerpt,
                "grader_answer": str(grader_answer),
                "grader_explanation": str(grader_explanation),
            }
        )


def _process_raw_sample(
    sample: dict,
    metadata_map: dict,
    agg: dict,
    annotation_id_filter=None,
    row_context: dict[str, Any] | None = None,
) -> None:
    """Process a raw sample dict (from zip JSON) and aggregate scores."""
    scores = sample.get("scores", {})
    if not scores:
        return

    meta = sample.get("metadata", {}) or {}
    sample_id = sample.get("id", "")

    # Extract history and excerpt from the raw dict
    input_msgs = sample.get("input", [])
    history = []
    for msg in input_msgs:
        if isinstance(msg, dict) and "role" in msg and "content" in msg:
            history.append({"role": msg["role"], "content": str(msg["content"])})

    output = sample.get("output", {}) or {}
    excerpt = ""
    if isinstance(output, dict):
        excerpt = output.get("completion", "")
    if not excerpt:
        messages = sample.get("messages", [])
        if messages:
            last = messages[-1]
            if isinstance(last, dict):
                excerpt = str(last.get("content", ""))

    for _score_key, score_obj in scores.items():
        if not score_obj or not isinstance(score_obj, dict):
            continue
        raw_value = score_obj.get("value")
        if raw_value is None:
            continue
        try:
            score_value = float(raw_value)
        except (ValueError, TypeError):
            continue
        if math.isnan(score_value):
            continue

        # Determine code
        code = "unknown"
        if meta.get("harmful_annotations"):
            code = meta["harmful_annotations"][0]
        else:
            explanation = score_obj.get("explanation", "")
            if explanation:
                try:
                    expl_data = json.loads(explanation)
                    code = expl_data.get("annotation_id", "unknown")
                except (json.JSONDecodeError, TypeError):
                    pass
        code = normalize_id(code)

        if annotation_id_filter and code not in annotation_id_filter:
            continue

        category = metadata_map[code].category if code in metadata_map else "unknown"
        _update_aggregate(agg, category, code, score_value)

        agg["samples"][code].append(
            {
                "sample_id": str(sample_id),
                "window_id": meta.get("window_id", "unknown"),
                "code": code,
                "category": category,
                "score": score_value,
                "history": history,
                "excerpt": excerpt,
                "grader_answer": str(score_obj.get("answer", "")),
                "grader_explanation": str(score_obj.get("explanation", "")),
            }
        )

        if row_context is not None:
            raw_score = None
            explanation = score_obj.get("explanation")
            if explanation:
                try:
                    expl_data = json.loads(explanation)
                    raw_score = expl_data.get("raw_score")
                except (json.JSONDecodeError, TypeError, KeyError):
                    raw_score = None
            if raw_score is not None:
                try:
                    raw_score = float(raw_score)
                    if math.isnan(raw_score):
                        raw_score = None
                except (TypeError, ValueError):
                    raw_score = None

            row_context["rows"].append(
                {
                    "model": row_context["model"],
                    "reasoning_effort": row_context["reasoning_effort"],
                    "model_label": row_context["model_label"],
                    "annotation_id": code,
                    "code_short": code.removeprefix("bot-"),
                    "category": category,
                    "window_id": meta.get("window_id", "unknown"),
                    "turn_index": meta.get("turn_index"),
                    "score": score_value,
                    "raw_score": raw_score,
                    "sample_id": str(sample_id),
                }
            )


def _generate_html_viewer(output_dir: str) -> None:
    """Copies index.html, styles.css, and viewer.js to the output directory."""
    assets_dir = Path(__file__).parent / "report_assets"

    for filename in ["index.html", "styles.css", "viewer.js"]:
        src = assets_dir / filename
        dst = Path(output_dir) / filename
        if src.exists():
            shutil.copy(src, dst)
        else:
            print(f"WARNING: Could not find asset {src} to copy to report.")


def _plot_condition(cond_id, models_data, figures_dir):
    """Generate and save plots for a single condition."""
    models_data.sort(key=lambda x: x["model"], reverse=True)
    model_names = [m["model"] for m in models_data]

    category_keys = set()
    harm_keys = set()
    for m in models_data:
        category_keys.update(m.get("category_scores", {}).keys())
        harm_keys.update(m.get("harm_code_scores", {}).keys())

    metrics = sorted(list(category_keys)) + sorted(list(harm_keys))
    if not metrics:
        return

    formatted_titles = []
    for m in metrics:
        # Determine the sample count (n) across all models for this metric
        counts = []
        for model in models_data:
            if m in category_keys:
                n = model.get("category_scores", {}).get(m, {}).get("samples", 0)
            else:
                n = model.get("harm_code_scores", {}).get(m, {}).get("samples", 0)
            counts.append(n)

        max_n = max(counts) if counts else 0

        # Warn if there's a mismatch
        if len(set(c for c in counts if c > 0)) > 1:
            print(f"WARNING: Sample count mismatch for '{m}': counts={counts}")

        if m in category_keys:
            formatted_titles.append(f"<b>{m.title()} (n={max_n})</b>")
        else:
            wrapped = "<br>".join(textwrap.wrap(f"{m} (n={max_n})", width=20))
            formatted_titles.append(f"<span style='color:#e45756'>{wrapped}</span>")

    fig = make_subplots(
        rows=1,
        cols=len(metrics),
        shared_yaxes=True,
        horizontal_spacing=0.02,
        subplot_titles=formatted_titles,
    )

    for idx, metric in enumerate(metrics, start=1):
        is_cat = metric in category_keys
        color = "#4c78a8" if is_cat else "#e45756"
        values = []
        for m in models_data:
            if is_cat:
                val = m.get("category_scores", {}).get(metric, {}).get("mean", 0.0)
            else:
                val = m.get("harm_code_scores", {}).get(metric, {}).get("mean", 0.0)
            values.append(val)

        fig.add_trace(
            go.Bar(
                x=values,
                y=model_names,
                orientation="h",
                marker_color=color,
                showlegend=False,
                text=[f"{v * 100:.0f}%" if v > 0 else "" for v in values],
                textposition="auto",
                textfont={"size": 10},
            ),
            row=1,
            col=idx,
        )
        fig.update_xaxes(
            tickformat=".0%",
            range=[0, 1.1],
            dtick=0.5,
            row=1,
            col=idx,
            showgrid=True,
            gridwidth=1,
            gridcolor="LightGray",
            tickangle=0,
            title_font={"size": 10},
            tickfont={"size": 10},
        )

    width = max(800, len(metrics) * 220 + 200)
    height = max(300, len(model_names) * 60 + 150)
    fig.update_layout(
        width=width,
        height=height,
        margin={"l": 200, "r": 40, "t": 120, "b": 40},
        plot_bgcolor="white",
        paper_bgcolor="white",
        title={
            "text": f"Condition: {cond_id}",
            "y": 0.98,
            "x": 0.5,
            "xanchor": "center",
            "yanchor": "top",
            "font": {"size": 14, "color": "gray"},
        },
    )
    for ann in fig["layout"]["annotations"]:
        ann.update(
            {"yanchor": "bottom", "y": 1.02, "yref": "paper", "font": {"size": 11}}
        )
    fig.update_yaxes(showgrid=False)

    safe_cond_id = cond_id[:150]
    base_path = os.path.join(figures_dir, safe_cond_id)
    fig.write_image(f"{base_path}.png", scale=2)
    fig.write_image(f"{base_path}.pdf")
    fig.write_image(f"{base_path}.svg")


def _export_static_figures(report_data: list, output_dir: str) -> None:
    """Generates static PNG/PDF/SVG plots natively using Plotly and Kaleido."""
    figures_dir = os.path.join(output_dir, "figures")
    os.makedirs(figures_dir, exist_ok=True)

    conditions = defaultdict(list)
    for item in report_data:
        conditions[item["condition_id"]].append(item)

    for cond_id, models_data in conditions.items():
        _plot_condition(cond_id, models_data, figures_dir)


def _init_report_item(agg, condition_id):
    """Initialize a single report item dictionary."""
    model = agg["model"]
    reasoning = agg.get("reasoning_effort")
    return {
        "model": model,
        "model_label": _model_label(model, reasoning),
        "reasoning_effort": reasoning,
        "task": agg["task"],
        "condition": agg["condition"],
        "condition_id": condition_id,
        "code_scores": {},
        "category_scores": {},
        "harm_code_scores": {},
        "overall_score": 0.0,
        "total_samples": agg["overall"]["count"],
        "sample_paths": {},
        "category_to_codes": defaultdict(list),
    }


def _write_samples(output_dir, safe_model, condition_id, code, samples_list):
    """Write sample list to a JSON file."""
    code_dir = os.path.join(output_dir, "samples", safe_model, condition_id)
    os.makedirs(code_dir, exist_ok=True)
    file_path = os.path.join(code_dir, f"{code}.json")
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(samples_list, f, indent=2)
    return f"samples/{safe_model}/{condition_id}/{code}.json"


def _collect_scores_for_group(samples_dict: dict, keys: list[str]) -> list[float]:
    """Collect all binary scores from samples matching the given code keys."""
    scores = []
    for key in keys:
        for s in samples_dict.get(key, []):
            if s["score"] is not None:
                scores.append(s["score"])
    return scores


def _summarize_group_scores(
    samples_dict: dict[str, list[dict[str, Any]]],
    code_keys: list[str],
    *,
    compute_ci: bool,
) -> Optional[dict[str, float]]:
    """Return grouped prevalence stats for a list of code keys."""
    scores = _collect_scores_for_group(samples_dict, code_keys)
    if not scores:
        return None
    entry: dict[str, float] = {
        "mean": float(np.mean(scores)),
        "samples": float(len(scores)),
    }
    if compute_ci:
        ci_lo, ci_hi = _bootstrap_binary_ci(scores)
        entry["ci_lower"] = ci_lo
        entry["ci_upper"] = ci_hi
    return entry


def _format_report(
    evaluations: Dict[str, Any], output_dir: str, *, compute_ci: bool = True
) -> list:
    """Format aggregated statistics and write sample files."""
    report = []
    for model, conditions in evaluations.items():
        for _, agg in conditions.items():
            condition_id = _dict_to_param_string(agg["condition"])
            safe_model = model.replace("/", "_").replace("\\", "_")
            item = _init_report_item(agg, condition_id)

            for code, samples_list in agg["samples"].items():
                item["sample_paths"][code] = _write_samples(
                    output_dir, safe_model, condition_id, code, samples_list
                )

            for code, stats in agg["codes"].items():
                if stats["count"] > 0:
                    mean_val = stats["score_sum"] / stats["count"]
                    entry: Dict[str, Any] = {
                        "mean": mean_val,
                        "samples": stats["count"],
                    }
                    if compute_ci:
                        code_scores = [
                            s["score"]
                            for s in agg["samples"].get(code, [])
                            if s["score"] is not None
                        ]
                        ci_lo, ci_hi = _bootstrap_binary_ci(code_scores)
                        entry["ci_lower"] = ci_lo
                        entry["ci_upper"] = ci_hi
                    item["code_scores"][code] = entry
                    cat = (
                        agg["samples"][code][0]["category"]
                        if agg["samples"][code]
                        else "unknown"
                    )
                    if (
                        cat != "concerns harm"
                        and code not in item["category_to_codes"][cat]
                    ):
                        item["category_to_codes"][cat].append(code)

            for cat, stats in agg["categories"].items():
                if stats["count"] > 0 and cat != "concerns harm":
                    mean_val = stats["score_sum"] / stats["count"]
                    entry = {"mean": mean_val, "samples": stats["count"]}
                    if compute_ci:
                        cat_codes = [
                            c
                            for c, slist in agg["samples"].items()
                            if slist and slist[0]["category"] == cat
                        ]
                        cat_scores = _collect_scores_for_group(
                            agg["samples"], cat_codes
                        )
                        ci_lo, ci_hi = _bootstrap_binary_ci(cat_scores)
                        entry["ci_lower"] = ci_lo
                        entry["ci_upper"] = ci_hi
                    item["category_scores"][cat] = entry

            # Split old "concerns harm" into two aggregate categories, excluding
            # validates-* codes from both.
            for category_key, code_keys in _HARM_CATEGORY_CODE_GROUPS.items():
                entry = _summarize_group_scores(
                    agg["samples"],
                    code_keys,
                    compute_ci=compute_ci,
                )
                if not entry:
                    continue
                item["category_scores"][category_key] = entry
                for code in code_keys:
                    if (
                        agg["samples"].get(code)
                        and code not in item["category_to_codes"][category_key]
                    ):
                        item["category_to_codes"][category_key].append(code)

            for code, stats in agg["harm_codes"].items():
                if stats["count"] > 0:
                    mean_val = stats["score_sum"] / stats["count"]
                    entry = {"mean": mean_val, "samples": stats["count"]}
                    if compute_ci:
                        code_scores = [
                            s["score"]
                            for s in agg["samples"].get(code, [])
                            if s["score"] is not None
                        ]
                        ci_lo, ci_hi = _bootstrap_binary_ci(code_scores)
                        entry["ci_lower"] = ci_lo
                        entry["ci_upper"] = ci_hi
                    item["harm_code_scores"][code] = entry

            if agg["overall"]["count"] > 0:
                item["overall_score"] = (
                    agg["overall"]["score_sum"] / agg["overall"]["count"]
                )

            report.append(item)

    # Compute delta_from_original for each non-original model.
    # Original transcript condition_ids lack reasoning_effort, so we match
    # on the base condition (stripping reasoning_effort from model conditions).
    def _base_condition_id(condition: dict) -> str:
        base = {k: v for k, v in condition.items() if k != "reasoning_effort"}
        return _dict_to_param_string(base)

    orig_by_base: Dict[str, dict] = {}
    orig_items_by_base: Dict[str, dict] = {}
    for item in report:
        if item["model"] == "original_transcript":
            base_id = _base_condition_id(
                item.get("condition", item.get("condition", {}))
            )
            orig_items_by_base[base_id] = item
    for _, cond_agg in evaluations.get("original_transcript", {}).items():
        base_id = _dict_to_param_string(cond_agg["condition"])
        orig_by_base[base_id] = cond_agg

    for item in report:
        if item["model"] == "original_transcript":
            continue
        # Match to original via base condition (without reasoning_effort)
        cond = item.get("condition")
        if not cond:
            # Fallback: reconstruct from evaluations
            for _, cond_agg in evaluations.get(item["model"], {}).items():
                if _dict_to_param_string(cond_agg["condition"]) == item["condition_id"]:
                    cond = cond_agg["condition"]
                    break
        if not cond:
            continue
        base_id = _base_condition_id(cond)
        orig = orig_items_by_base.get(base_id)
        orig_agg = orig_by_base.get(base_id)
        if not orig or not orig_agg:
            continue

        # Build original sample scores keyed by group (categories + harm codes)
        orig_samples_by_group: Dict[str, list[float]] = {}
        for cat in orig.get("category_scores", {}):
            cat_codes = list((orig.get("category_to_codes", {}) or {}).get(cat, []))
            orig_samples_by_group[cat] = _collect_scores_for_group(
                orig_agg["samples"], cat_codes
            )
        for code in orig.get("harm_code_scores", {}):
            orig_samples_by_group[code] = [
                s["score"]
                for s in orig_agg["samples"].get(code, [])
                if s["score"] is not None
            ]

        # Find this model's raw agg
        model_agg = None
        for _, cond_agg in evaluations.get(item["model"], {}).items():
            if _dict_to_param_string(cond_agg["condition"]) == item["condition_id"]:
                model_agg = cond_agg
                break
        if not model_agg:
            continue

        delta_dict: Dict[str, Any] = {}
        for cat in item.get("category_scores", {}):
            cat_codes = list((item.get("category_to_codes", {}) or {}).get(cat, []))
            model_scores = _collect_scores_for_group(model_agg["samples"], cat_codes)
            orig_scores = orig_samples_by_group.get(cat, [])
            if model_scores and orig_scores:
                if compute_ci:
                    d, lo, hi = _bootstrap_delta_ci(model_scores, orig_scores)
                    delta_dict[cat] = {"delta": d, "ci_lower": lo, "ci_upper": hi}
                else:
                    d = float(np.mean(model_scores) - np.mean(orig_scores))
                    delta_dict[cat] = {"delta": d}

        for code in item.get("harm_code_scores", {}):
            model_scores = [
                s["score"]
                for s in model_agg["samples"].get(code, [])
                if s["score"] is not None
            ]
            orig_scores = orig_samples_by_group.get(code, [])
            if model_scores and orig_scores:
                if compute_ci:
                    d, lo, hi = _bootstrap_delta_ci(model_scores, orig_scores)
                    delta_dict[code] = {"delta": d, "ci_lower": lo, "ci_upper": hi}
                else:
                    d = float(np.mean(model_scores) - np.mean(orig_scores))
                    delta_dict[code] = {"delta": d}

        item["delta_from_original"] = delta_dict

    return report


def _load_rich_metadata() -> Dict[str, Any]:
    """Load rich metadata from annotations.csv."""

    metadata = {}
    with open(ANNOTATIONS_FILE, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            source_id = row["id"]
            code = normalize_id(source_id)
            metadata[code] = {
                "name": row.get("name", ""),
                "description": row.get("description", ""),
                "positive_examples": row.get("positive-examples", ""),
                "negative_examples": row.get("negative-examples", ""),
                "cutoff": _CUTOFFS.get(source_id, 7),
            }
    return metadata


def generate_report(
    logs_dir: str,
    output_dir: str,
    filter_criteria: ReportFilter,
    *,
    compute_ci: bool = True,
    log_files: list[str] | None = None,
) -> None:
    """Reads inspect_ai logs and generates an aggregated JSON report.

    Parameters
    ----------
    logs_dir:
        Directory to glob for ``*.eval`` files (ignored when *log_files* given).
    output_dir:
        Directory for the generated report artefacts.
    filter_criteria:
        Optional model/code/grader filters.
    compute_ci:
        Whether to run bootstrap CIs (slow).
    log_files:
        Explicit list of eval log paths. When provided, *logs_dir* glob and
        dedup are skipped entirely.
    """
    cache_dir = ".cache/generate_report"
    parse_workers: int | None = None

    timings: dict[str, float] = {}
    total_start = time.perf_counter()

    metadata_cache_path, samples_cache_dir = _cache_paths(cache_dir)
    metadata_cache = _load_metadata_cache(metadata_cache_path)
    metadata_cache_dirty = False
    metadata_cache_hits = 0

    stage_start = time.perf_counter()
    explicit_files = log_files is not None
    if not explicit_files:
        log_files = glob.glob(os.path.join(logs_dir, "*.eval"))
    log_files = log_files or []
    timings["discover_logs"] = time.perf_counter() - stage_start

    stage_start = time.perf_counter()
    metadata_map_raw = load_annotation_metadata_with_role_splits(ANNOTATIONS_FILE)
    metadata_map = {normalize_id(k): v for k, v in metadata_map_raw.items()}
    timings["load_annotation_metadata"] = time.perf_counter() - stage_start

    evaluations = defaultdict(lambda: defaultdict(dict))

    # Pre-load window data for original_transcript processing
    stage_start = time.perf_counter()
    windows_path = resolve_path(
        "LLM_DELUSIONS_WINDOWS_PATH",
        DEFAULT_WINDOWS_PATH,
        require_local=True,
    )
    window_map = {}
    excluded_window_ids: set[str] = set()
    excluded_participants: set[str] = set()
    try:
        excluded_participants, excluded_window_ids = resolve_excluded_window_ids(
            windows_path
        )
    except (OSError, ValueError, KeyError) as exc:
        print(
            "WARNING: could not resolve excluded participants from windows "
            f"data ({exc}); proceeding without exclusions."
        )
    if excluded_participants:
        print(
            "Report exclusion enabled for participants "
            f"{sorted(excluded_participants)} "
            f"({len(excluded_window_ids)} window IDs)"
        )

    try:
        df = pd.read_parquet(windows_path)
        for row in df.itertuples():
            window_ids = {
                str(build_window_id(row)).strip(),
                str(build_eval_subset_id_from_row(row)).strip(),
            }
            if excluded_window_ids and any(
                window_id in excluded_window_ids
                for window_id in window_ids
                if window_id
            ):
                continue
            messages = [
                (
                    dict(m)
                    if hasattr(m, "keys")
                    else {k: getattr(m, k) for k in m._fields}
                )
                for m in row.messages
            ]
            clean_msgs = [
                m for m in messages if m.get("role") in ("user", "assistant", "tool")
            ]
            for window_id in window_ids:
                if window_id:
                    window_map[window_id] = clean_msgs
    except (OSError, ValueError) as e:
        print("Warning: Could not load windows parquet: " + str(e))
    timings["load_windows"] = time.perf_counter() - stage_start

    # Resolve log files: explicit list or glob + dedup
    stage_start = time.perf_counter()
    if explicit_files:
        print(f"Using {len(log_files)} explicitly specified eval files")
        selected_logs = []
        for path in log_files:
            try:
                manifest, _sample_count, was_cached = _get_log_metadata_cached(
                    path, metadata_cache
                )
                if was_cached:
                    metadata_cache_hits += 1
                else:
                    metadata_cache_dirty = True
                selected_logs.append((path, manifest))
                print(f"  {os.path.basename(path)}")
            except (
                OSError,
                zipfile.BadZipFile,
                KeyError,
                json.JSONDecodeError,
            ) as exc:
                print(f"  WARNING: skipping {path}: {exc}")
    else:
        print(f"Found {len(log_files)} eval files, deduplicating...")
        selected_logs, metadata_cache_hits = _select_best_logs(
            log_files, metadata_cache
        )
        metadata_cache_dirty = metadata_cache_hits < len(log_files)
        print(f"Selected {len(selected_logs)} unique model configs")
    timings["select_logs"] = time.perf_counter() - stage_start

    if metadata_cache_dirty:
        try:
            _save_metadata_cache(metadata_cache_path, metadata_cache)
        except OSError as exc:
            print(f"WARNING: could not save metadata cache: {exc}")

    print(f"Metadata cache hits: {metadata_cache_hits}/{len(log_files)}")

    stage_start = time.perf_counter()
    parse_jobs: list[dict[str, Any]] = []
    eval_rows: list[dict[str, Any]] = []
    seen_original_samples = set()
    for log_path, manifest in selected_logs:
        eval_info = manifest.get("eval", {})
        model = eval_info.get("model", "unknown")
        task_name = eval_info.get("task", "unknown")
        gen_config = eval_info.get("model_generate_config", {}) or {}
        reasoning_effort = (
            gen_config.get("reasoning_effort") if isinstance(gen_config, dict) else None
        )
        model_label = _model_label(model, reasoning_effort)

        condition_dict = _extract_condition_from_manifest(manifest)

        if (
            filter_criteria.max_context is not None
            and condition_dict.get("max_context_messages")
            != filter_criteria.max_context
        ):
            continue
        if filter_criteria.models and model not in filter_criteria.models:
            continue
        grader_model = condition_dict.get("grader_model")
        if (
            filter_criteria.grader_models
            and grader_model not in filter_criteria.grader_models
        ):
            continue

        condition_key = json.dumps(condition_dict, sort_keys=True)
        if condition_key not in evaluations[model]:
            evaluations[model][condition_key] = {
                "model": model,
                "task": task_name,
                "condition": condition_dict,
                "reasoning_effort": reasoning_effort,
                "codes": defaultdict(lambda: {"score_sum": 0.0, "count": 0}),
                "harm_codes": defaultdict(lambda: {"score_sum": 0.0, "count": 0}),
                "categories": defaultdict(lambda: {"score_sum": 0.0, "count": 0}),
                "overall": {"score_sum": 0.0, "count": 0},
                "samples": defaultdict(list),
            }

        # Original transcript is reasoning-agnostic: key without reasoning_effort
        # so it is shared across all models with different reasoning settings.
        base_condition = {
            k: v for k, v in condition_dict.items() if k != "reasoning_effort"
        }
        base_key = json.dumps(base_condition, sort_keys=True)
        if window_map and base_key not in evaluations["original_transcript"]:
            evaluations["original_transcript"][base_key] = {
                "model": "original_transcript",
                "task": task_name,
                "condition": base_condition,
                "reasoning_effort": None,
                "codes": defaultdict(lambda: {"score_sum": 0.0, "count": 0}),
                "harm_codes": defaultdict(lambda: {"score_sum": 0.0, "count": 0}),
                "categories": defaultdict(lambda: {"score_sum": 0.0, "count": 0}),
                "overall": {"score_sum": 0.0, "count": 0},
                "samples": defaultdict(list),
            }

        parse_jobs.append(
            {
                "log_path": log_path,
                "model": model,
                "model_label": model_label,
                "reasoning_effort": reasoning_effort,
                "condition_key": condition_key,
                "base_key": base_key,
            }
        )
    timings["prepare_jobs"] = time.perf_counter() - stage_start

    stage_start = time.perf_counter()
    requested_workers = parse_workers
    if requested_workers is None or requested_workers <= 0:
        requested_workers = min(8, os.cpu_count() or 1)
    worker_count = max(1, min(requested_workers, len(parse_jobs))) if parse_jobs else 1

    sample_load_seconds = 0.0
    sample_aggregate_seconds = 0.0
    sample_cache_hits = 0
    failed_paths: set[str] = set()
    excluded_samples = 0

    if parse_jobs:
        print(
            f"Parsing {len(parse_jobs)} selected eval files with "
            f"{worker_count} worker(s)..."
        )
        remaining_by_path = Counter(job["log_path"] for job in parse_jobs)
        total_unique_paths = len(remaining_by_path)
        loaded_samples: dict[str, list[dict[str, Any]]] = {}

        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            futures_by_path = {
                log_path: executor.submit(
                    _load_or_parse_samples, log_path, samples_cache_dir
                )
                for log_path in remaining_by_path
            }

            for job in parse_jobs:
                log_path = job["log_path"]
                if log_path in failed_paths:
                    continue

                if log_path not in loaded_samples:
                    print(f"  Parsing {os.path.basename(log_path)} ...")
                    load_start = time.perf_counter()
                    try:
                        parsed_samples, was_cached = futures_by_path[log_path].result()
                    except (OSError, zipfile.BadZipFile) as exc:
                        print(f"  WARNING: skipping {log_path}: {exc}")
                        failed_paths.add(log_path)
                        continue
                    sample_load_seconds += time.perf_counter() - load_start
                    sample_cache_hits += int(was_cached)
                    loaded_samples[log_path] = parsed_samples

                agg = evaluations[job["model"]][job["condition_key"]]
                orig_agg = evaluations.get("original_transcript", {}).get(
                    job["base_key"]
                )

                aggregate_start = time.perf_counter()
                for sample in loaded_samples[log_path]:
                    meta = sample.get("metadata", {}) or {}
                    window_id = str(meta.get("window_id", "")).strip()
                    if excluded_window_ids and window_id in excluded_window_ids:
                        excluded_samples += 1
                        continue

                    _process_raw_sample(
                        sample,
                        metadata_map,
                        agg,
                        filter_criteria.annotation_ids,
                        row_context={
                            "model": job["model"],
                            "reasoning_effort": job["reasoning_effort"],
                            "model_label": job["model_label"],
                            "rows": eval_rows,
                        },
                    )

                    # Also process original transcript for the exact same sample.
                    if orig_agg is None or not meta:
                        continue

                    turn_index = meta.get("turn_index")
                    if not (
                        window_id and turn_index is not None and window_id in window_map
                    ):
                        continue

                    clean_msgs = window_map[window_id]
                    msg_idx = turn_index + 1
                    is_valid_assistant = (
                        msg_idx < len(clean_msgs)
                        and clean_msgs[msg_idx].get("role") == "assistant"
                    )

                    target_codes = filter_criteria.annotation_ids
                    if not target_codes:
                        target_codes = [
                            normalize_id(ann)
                            for ann in meta.get("harmful_annotations", ["unknown"])
                        ]

                    # Build history once from raw sample input.
                    raw_history = []
                    for msg in sample.get("input", []):
                        if isinstance(msg, dict) and "role" in msg and "content" in msg:
                            raw_history.append(
                                {"role": msg["role"], "content": str(msg["content"])}
                            )

                    for code in target_codes:
                        if code == "unknown":
                            continue

                        source_code = get_source_id(code)
                        sample_key = (
                            job["condition_key"],
                            window_id,
                            turn_index,
                            code,
                        )
                        if sample_key in seen_original_samples:
                            continue
                        seen_original_samples.add(sample_key)

                        raw_score = (
                            clean_msgs[msg_idx].get(source_code)
                            if is_valid_assistant
                            else None
                        )
                        category = (
                            metadata_map[code].category
                            if code in metadata_map
                            else "unknown"
                        )
                        cutoff = _CUTOFFS.get(source_code, 7)

                        if raw_score is not None and not pd.isna(raw_score):
                            binary_score = 1.0 if float(raw_score) >= cutoff else 0.0
                            excerpt = clean_msgs[msg_idx].get("content", "")
                            grader_ans = (
                                clean_msgs[msg_idx].get(f"rationale__{source_code}")
                                or "Extracted from dataset"
                            )

                            matches_val = clean_msgs[msg_idx].get(
                                f"matches__{source_code}", []
                            )
                            if hasattr(matches_val, "tolist"):
                                matches_val = matches_val.tolist()
                            elif isinstance(matches_val, (list, tuple)):
                                matches_val = list(matches_val)
                            else:
                                matches_val = []

                            expl = {
                                "raw_score": float(raw_score),
                                "cutoff": cutoff,
                                "matches": matches_val,
                                "rationale": clean_msgs[msg_idx].get(
                                    f"rationale__{source_code}", ""
                                ),
                            }
                        else:
                            binary_score = None
                            excerpt = ""
                            grader_ans = "No subsequent assistant message."
                            expl = {"error": "Missing original score"}

                        _update_aggregate(orig_agg, category, code, binary_score)

                        orig_agg["samples"][code].append(
                            {
                                "sample_id": str(sample.get("id", "")),
                                "window_id": window_id,
                                "code": code,
                                "category": category,
                                "score": binary_score,
                                "history": raw_history,
                                "excerpt": excerpt,
                                "grader_answer": grader_ans,
                                "grader_explanation": json.dumps(
                                    expl,
                                    default=lambda o: (
                                        o.tolist() if hasattr(o, "tolist") else str(o)
                                    ),
                                ),
                            }
                        )
                        eval_rows.append(
                            {
                                "model": "original_transcript",
                                "reasoning_effort": None,
                                "model_label": "Original transcript",
                                "annotation_id": code,
                                "code_short": code.removeprefix("bot-"),
                                "category": category,
                                "window_id": window_id,
                                "turn_index": turn_index,
                                "score": binary_score,
                                "raw_score": (
                                    float(raw_score)
                                    if raw_score is not None and not pd.isna(raw_score)
                                    else None
                                ),
                                "sample_id": str(sample.get("id", "")),
                            }
                        )
                sample_aggregate_seconds += time.perf_counter() - aggregate_start

                remaining_by_path[log_path] -= 1
                if remaining_by_path[log_path] <= 0:
                    del loaded_samples[log_path]

        print(f"Sample cache hits: {sample_cache_hits}/{total_unique_paths}")
        if excluded_window_ids:
            print(f"Excluded {excluded_samples} samples by participant filter.")

    timings["load_samples"] = sample_load_seconds
    timings["aggregate_samples"] = sample_aggregate_seconds
    timings["parse_and_aggregate"] = time.perf_counter() - stage_start

    stage_start = time.perf_counter()
    os.makedirs(output_dir, exist_ok=True)
    report = _format_report(evaluations, output_dir, compute_ci=compute_ci)
    summary_path = os.path.join(output_dir, "summary.json")
    eval_rows_path = os.path.join(output_dir, "eval_rows.parquet")
    rich_metadata = _load_rich_metadata()
    final_output = {"evaluations": report, "metadata": rich_metadata}
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(
            _replace_non_finite_floats(final_output),
            f,
            indent=2,
            allow_nan=False,
        )
    rows_columns = [
        "model",
        "reasoning_effort",
        "model_label",
        "annotation_id",
        "code_short",
        "category",
        "window_id",
        "turn_index",
        "score",
        "raw_score",
        "sample_id",
    ]
    rows_df = pd.DataFrame(eval_rows, columns=rows_columns)
    rows_df.to_parquet(eval_rows_path, index=False)
    print(f"Raw eval rows written to {eval_rows_path} ({len(rows_df)} rows).")

    print(f"Report summary generated at {summary_path} with {len(report)} items.")
    _generate_html_viewer(output_dir)
    print(f"Interactive HTML dashboard created at {output_dir}/index.html")
    _export_static_figures(report, output_dir)
    print(f"Static figures exported to {os.path.join(output_dir, 'figures/')}")
    timings["write_outputs"] = time.perf_counter() - stage_start

    timings["total"] = time.perf_counter() - total_start
    print("Timing summary:")
    for key in (
        "discover_logs",
        "load_annotation_metadata",
        "load_windows",
        "select_logs",
        "prepare_jobs",
        "load_samples",
        "aggregate_samples",
        "parse_and_aggregate",
        "write_outputs",
        "total",
    ):
        print(f"  {key}: {timings.get(key, 0.0):.2f}s")


def main() -> None:
    """Main entry point for generating evaluation reports."""
    parser = argparse.ArgumentParser(
        description="Generate evaluation report from Inspect AI logs."
    )
    parser.add_argument(
        "--logs-dir", type=str, default="logs", help="Directory containing logs"
    )
    parser.add_argument(
        "--output-dir", type=str, default="report", help="Path to output directory"
    )
    parser.add_argument(
        "--max-context-messages", type=int, default=None, help="Filter by context"
    )
    parser.add_argument("--models", action="append", help="Models to include")
    parser.add_argument(
        "--annotation-id", action="append", help="Annotation IDs to include"
    )
    parser.add_argument(
        "--grader-models", action="append", help="Grader models to include"
    )
    parser.add_argument(
        "--no-bootstrap",
        action="store_true",
        help="Skip bootstrap CI computation for faster runs.",
    )
    parser.add_argument(
        "--log-files",
        nargs="+",
        default=None,
        help="Explicit eval log files to use (skips glob and dedup).",
    )

    args = parser.parse_args()

    filter_criteria = ReportFilter(
        max_context=args.max_context_messages,
        models=args.models,
        annotation_ids=args.annotation_id,
        grader_models=args.grader_models,
    )

    generate_report(
        args.logs_dir,
        args.output_dir,
        filter_criteria,
        compute_ci=not args.no_bootstrap,
        log_files=args.log_files,
    )


if __name__ == "__main__":
    main()
