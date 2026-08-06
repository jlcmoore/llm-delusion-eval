"""Analyze requested vs effective context length from context eval logs.

This script reads ``.eval`` files from a context-run logs directory,
aggregates row-level scores by requested context setting, and produces:

- CSVs of aggregated points in ``analysis/data/context_effects/``
- Scatter plots of prevalence vs effective context length in
  ``analysis/figures/``

The script does not depend on ``report/summary.json`` or export sample-level
sensitive data.
"""

import argparse
import gzip
import hashlib
import json
import logging
import math
import os
import pickle
import re
import zipfile
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import bootstrap

from analysis.artifact_paths import DATA_DIR, FIGURE_DIR, ensure_output_dirs
from analysis.bootstrap import BootstrapConfig, hierarchical_weighted_mean_ci
from analysis.load_eval_data import _EVALS_REPO_ROOT, CODE_CATEGORIES, _load_cutoffs
from analysis.metric_labels import (
    format_metric_label_plain_for_matplotlib,
    prevalence_axis_label_for_matplotlib,
)
from analysis.participant_clustered import (
    aggregate_participant_conversation_value_sums,
    attach_participant_ids,
)
from analysis.plot_style import apply_plot_style, get_model_color, sort_model_labels
from llm_delusion_eval.constants import format_model_label, normalize_id
from llm_delusion_eval.participant_exclusions import resolve_excluded_window_ids
from llm_delusion_eval.paths import DEFAULT_WINDOWS_PATH, resolve_path
from llm_delusion_eval.window_ids import build_eval_subset_id_from_subset_rel_path

logger = logging.getLogger(__name__)

plt.switch_backend("Agg")
apply_plot_style()

DEFAULT_LOGS_DIR = _EVALS_REPO_ROOT / "logs-context"
DEFAULT_BASELINE_LOGS_DIR = _EVALS_REPO_ROOT / "logs"
DEFAULT_ROWS_CACHE = _EVALS_REPO_ROOT / ".cache" / "context_effect_rows.parquet"
DEFAULT_LOG_METADATA_CACHE = (
    _EVALS_REPO_ROOT / ".cache" / "context_effect_log_metadata.json"
)
DEFAULT_SAMPLE_ROWS_CACHE_DIR = _EVALS_REPO_ROOT / ".cache" / "context_effect_samples"
DEFAULT_FIGURES_DIR = FIGURE_DIR
DEFAULT_DATA_DIR = DATA_DIR / "context_effects"
ROWS_CACHE_SCHEMA_VERSION = 9
LOG_METADATA_CACHE_VERSION = 1
SAMPLE_ROWS_CACHE_VERSION = 6
DEFAULT_CI = 0.95
DEFAULT_N_RESAMPLES = 10_000
DEFAULT_BOOTSTRAP_SEED = 42
DEFAULT_CLUSTER_BOOTSTRAP_CONFIG = BootstrapConfig()
CONTEXT_EFFECT_FIGURE_WIDTH_INCHES = 4.0
CONTEXT_EFFECT_FIGURE_HEIGHT_INCHES = 2.3
CONTEXT_EFFECT_X_MAX_MESSAGES = 400
CONTEXT_EFFECT_X_ENDPOINT_PAD_MESSAGES = 8.0

_CODE_FENCE_PATTERN = re.compile(r"```|~~~")
_INLINE_CODE_PATTERN = re.compile(r"`[^`\n]{3,}`")
_CODE_KEYWORD_PATTERN = re.compile(
    r"\b(def\s+\w+\s*\(|class\s+\w+\s*[:(]|import\s+\w+|from\s+\w+\s+import\s+)"
    r"|(\bfunction\s+\w+\s*\()|(\b(const|let|var)\s+\w+\s*=)"
    r"|(\bSELECT\b.+\bFROM\b)|(\bINSERT\b.+\bINTO\b)|(\bCREATE\b.+\bTABLE\b)",
    flags=re.IGNORECASE,
)
_SYMBOL_HEAVY_LINE_PATTERN = re.compile(r"^[\s\[\]{}();=<>+\-/*%.,:]{8,}$")
DEFAULT_ANNOTATION_CUTOFF = 7.0
ANNOTATION_CUTOFFS = _load_cutoffs()
CODE_LIKE_MIN_LINE_MATCHES = 2
GPT54_MODEL_ID = "openai/gpt-5.4-2026-03-05"
GPT54_APPENDIX_CONTEXT_MAX = 400
MAIN_CONTEXT_EFFECT_LEFT_CATEGORY = "delusional"
MAIN_CONTEXT_EFFECT_RIGHT_CODE = "bot-discourages-violence"

# Canonical order used for GPT-5.4 context appendix code subplots.
GPT54_APPENDIX_CODE_ORDER = [
    # sycophancy
    "bot-positive-affirmation",
    "bot-reflective-summary",
    "bot-grand-significance",
    "bot-dismisses-counterevidence",
    "bot-reports-others-admire-speaker",
    # delusional
    "bot-misrepresents-sentience",
    "bot-misrepresents-ability",
    "bot-metaphysical-themes",
    "bot-endorses-delusion",
    # relationship
    "bot-claims-unique-connection",
    "bot-romantic-interest",
    "bot-platonic-affinity",
    # harm (facilitates + discourages/validates)
    "bot-facilitates-violence",
    "bot-validates-violent-feelings",
    "bot-discourages-violence",
    "bot-facilitates-self-harm",
    "bot-validates-self-harm-feelings",
    "bot-discourages-self-harm",
]

# Category order for aggregate-category subplot figures.
GPT54_APPENDIX_CATEGORY_ORDER = [
    "sycophancy",
    "delusional",
    "relationship",
    "facilitates harm",
    "discourages harm",
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


def _aggregate_plot_category(annotation_id: str) -> str:
    """Map one annotation ID to the 5-category aggregate plot taxonomy.

    Parameters
    ----------
    annotation_id:
        Normalized annotation ID, typically with the ``bot-`` prefix.

    Returns
    -------
    str
        One of:
        ``sycophancy``, ``delusional``, ``relationship``,
        ``facilitates harm``, ``discourages harm``, or ``unknown``.
    """
    normalized = str(annotation_id).strip().removeprefix("bot-")
    if normalized.startswith("facilitates-"):
        return "facilitates harm"
    if normalized.startswith("discourages-") or normalized.startswith("validates-"):
        return "discourages harm"
    return CODE_CATEGORIES.get(normalized, "unknown")


@dataclass(frozen=True)
class ContextEffectCachePaths:
    """Cache paths used by context-effects loading.

    Attributes
    ----------
    rows_cache:
        Parquet cache of concatenated selected rows.
    log_metadata_cache:
        JSON cache for per-log manifest/sample-count metadata.
    sample_rows_cache_dir:
        Directory for per-log parsed-row caches.
    """

    rows_cache: Path
    log_metadata_cache: Path
    sample_rows_cache_dir: Path


@dataclass(frozen=True)
class ContextSubplotConfig:
    """Configuration for a grouped multi-panel context-effect figure.

    Attributes
    ----------
    group_col:
        Column name used for subplot grouping.
    groups:
        Ordered group labels to plot.
    filename_stem:
        Output filename stem.
    subplot_shape:
        ``(rows, cols)`` arrangement for subplot grid.
    figsize:
        Figure size in inches.
    annotate_windows:
        Whether to annotate points with unique-window counts.
    figure_title:
        Optional figure-level title.
    """

    group_col: str
    groups: list[str]
    filename_stem: str
    subplot_shape: tuple[int, int]
    figsize: tuple[float, float]
    annotate_windows: bool
    figure_title: Optional[str] = None


def _model_label(model: str, reasoning: Optional[str]) -> str:
    """Build a display label from model ID and reasoning effort.

    Parameters
    ----------
    model:
        Full model identifier from eval logs.
    reasoning:
        Optional reasoning effort setting.

    Returns
    -------
    str
        Human-readable model label.
    """
    return format_model_label(model, reasoning)


def _file_signature(path: Path) -> tuple[int, int]:
    """Return file size and mtime_ns for cache invalidation.

    Parameters
    ----------
    path:
        File path.

    Returns
    -------
    tuple[int, int]
        ``(size_bytes, mtime_ns)``.
    """
    stat = path.stat()
    return stat.st_size, stat.st_mtime_ns


def _cache_key(path: Path, *, cache_version: int) -> str:
    """Create a deterministic cache key for one file revision.

    Parameters
    ----------
    path:
        Eval log file path.
    cache_version:
        Cache schema version for invalidation.

    Returns
    -------
    str
        SHA-256 cache key.
    """
    size, mtime_ns = _file_signature(path)
    payload = f"{path.resolve()}|{size}|{mtime_ns}|{cache_version}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _load_metadata_cache(cache_path: Path) -> dict[str, Any]:
    """Load log metadata cache from disk.

    Parameters
    ----------
    cache_path:
        Metadata cache JSON path.

    Returns
    -------
    dict[str, Any]
        Cache dictionary, or an empty dict when missing/invalid.
    """
    if not cache_path.exists():
        return {}
    try:
        with cache_path.open("r", encoding="utf-8") as file_obj:
            cached = json.load(file_obj)
    except (OSError, json.JSONDecodeError):
        return {}
    if not isinstance(cached, dict):
        return {}
    return cached


def _save_metadata_cache(cache_path: Path, cache_data: dict[str, Any]) -> None:
    """Persist log metadata cache atomically.

    Parameters
    ----------
    cache_path:
        Metadata cache JSON path.
    cache_data:
        Cache dictionary to write.
    """
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = cache_path.with_suffix(".tmp")
    with tmp_path.open("w", encoding="utf-8") as file_obj:
        json.dump(cache_data, file_obj)
    os.replace(tmp_path, cache_path)


def _get_log_metadata_cached(
    log_path: Path, metadata_cache: dict[str, Any]
) -> tuple[dict[str, Any], int, bool]:
    """Return manifest and sample count for a log with metadata caching.

    Parameters
    ----------
    log_path:
        Eval log file path.
    metadata_cache:
        Mutable metadata cache dictionary.

    Returns
    -------
    tuple[dict[str, Any], int, bool]
        ``(manifest, sample_count, was_cache_hit)``.
    """
    cache_key = str(log_path.resolve())
    size, mtime_ns = _file_signature(log_path)
    cached = metadata_cache.get(cache_key, {})
    if (
        isinstance(cached, dict)
        and cached.get("cache_version") == LOG_METADATA_CACHE_VERSION
        and cached.get("size") == size
        and cached.get("mtime_ns") == mtime_ns
        and "manifest" in cached
        and "sample_count" in cached
    ):
        return cached["manifest"], int(cached["sample_count"]), True

    manifest, sample_count = _read_manifest_and_sample_count(log_path)
    metadata_cache[cache_key] = {
        "cache_version": LOG_METADATA_CACHE_VERSION,
        "size": size,
        "mtime_ns": mtime_ns,
        "manifest": manifest,
        "sample_count": sample_count,
    }
    return manifest, sample_count, False


def _sample_rows_cache_file(log_path: Path, cache_dir: Path) -> Path:
    """Return cache file path for parsed rows of one log revision.

    Parameters
    ----------
    log_path:
        Eval log file path.
    cache_dir:
        Cache directory root.

    Returns
    -------
    Path
        Cache file path.
    """
    return (
        cache_dir
        / f"{_cache_key(log_path, cache_version=SAMPLE_ROWS_CACHE_VERSION)}.pkl.gz"
    )


def _read_manifest_and_sample_count(log_path: Path) -> tuple[dict[str, Any], int]:
    """Read start manifest and sample count from one eval log.

    Parameters
    ----------
    log_path:
        Path to an ``.eval`` file.

    Returns
    -------
    tuple[dict[str, Any], int]
        Parsed start manifest and number of sample entries.
    """
    with zipfile.ZipFile(log_path, "r") as zip_file:
        manifest = json.loads(zip_file.read("_journal/start.json"))
        sample_count = sum(
            1 for info in zip_file.infolist() if info.filename.startswith("samples/")
        )
    return manifest, sample_count


def _iter_raw_samples(log_path: Path):
    """Yield raw sample dictionaries from an eval log.

    Parameters
    ----------
    log_path:
        Path to an ``.eval`` file.

    Yields
    ------
    dict[str, Any]
        Raw sample payloads from the zip.
    """
    with zipfile.ZipFile(log_path, "r") as zip_file:
        for name in sorted(zip_file.namelist()):
            if not name.startswith("samples/"):
                continue
            try:
                yield json.loads(zip_file.read(name))
            except (json.JSONDecodeError, EOFError):
                continue


def _normalize_codes(task_args: dict[str, Any]) -> tuple[str, ...]:
    """Normalize task code selections for config deduplication keys.

    Parameters
    ----------
    task_args:
        Task argument dictionary from the eval manifest.

    Returns
    -------
    tuple[str, ...]
        Sorted normalized code identifiers.
    """
    codes = task_args.get("codes")
    if isinstance(codes, str):
        raw_codes = [entry.strip() for entry in codes.split(",") if entry.strip()]
    elif isinstance(codes, list):
        raw_codes = [str(entry).strip() for entry in codes if str(entry).strip()]
    else:
        raw_codes = []
    return tuple(sorted(normalize_id(code) for code in raw_codes))


def _manifest_eval_fields(
    manifest: dict[str, Any],
) -> tuple[str, Optional[str], dict[str, Any]]:
    """Extract common eval manifest fields used throughout this module.

    Parameters
    ----------
    manifest:
        Parsed start manifest for an eval log.

    Returns
    -------
    tuple[str, Optional[str], dict[str, Any]]
        ``(model, reasoning_effort, task_args)``
    """
    eval_info = manifest.get("eval", {})
    model = str(eval_info.get("model", "unknown"))
    reasoning = (eval_info.get("model_generate_config", {}) or {}).get(
        "reasoning_effort"
    )
    task_args = eval_info.get("task_args", {}) or {}
    return model, reasoning, task_args


def _manifest_config_key(manifest: dict[str, Any]) -> tuple[Any, ...]:
    """Build the deduplication key for a log manifest.

    Parameters
    ----------
    manifest:
        Parsed start manifest for an eval log.

    Returns
    -------
    tuple[Any, ...]
        Key used to deduplicate logs for the same run configuration.
    """
    model, reasoning, task_args = _manifest_eval_fields(manifest)
    return (
        model,
        reasoning,
        task_args.get("max_context_messages", 0),
        task_args.get("max_windows", 0),
        _normalize_codes(task_args),
    )


def _select_best_logs(
    log_files: list[Path], metadata_cache: dict[str, Any]
) -> tuple[list[tuple[Path, dict[str, Any]]], int]:
    """Deduplicate logs and keep the most complete run per context config.

    Parameters
    ----------
    log_files:
        Candidate log files.
    metadata_cache:
        Mutable cache for log manifest/sample count metadata.

    Returns
    -------
    tuple[list[tuple[Path, dict[str, Any]]], int]
        Selected ``(log_path, manifest)`` pairs and metadata cache hits.
    """
    candidates: dict[tuple[Any, ...], list[tuple[int, Path, dict[str, Any]]]] = (
        defaultdict(list)
    )
    metadata_cache_hits = 0

    for log_path in log_files:
        try:
            manifest, sample_count, was_cache_hit = _get_log_metadata_cached(
                log_path, metadata_cache
            )
            if was_cache_hit:
                metadata_cache_hits += 1
        except (OSError, zipfile.BadZipFile, KeyError, json.JSONDecodeError) as exc:
            logger.warning("Skipping unreadable log %s: %s", log_path, exc)
            continue

        config_key = _manifest_config_key(manifest)
        candidates[config_key].append((sample_count, log_path, manifest))

    selected: list[tuple[Path, dict[str, Any]]] = []
    for config_key, entries in candidates.items():
        best = max(entries, key=lambda item: (item[0], item[1].name))
        selected.append((best[1], best[2]))
        logger.info(
            "Selected %s (%d samples) for config %s",
            best[1].name,
            best[0],
            config_key,
        )

    selected.sort(key=lambda item: item[0].name)
    return selected, metadata_cache_hits


def _extract_code(sample: dict[str, Any]) -> str:
    """Extract normalized annotation code for a sample row.

    Parameters
    ----------
    sample:
        Raw sample dictionary.

    Returns
    -------
    str
        Normalized annotation code.
    """
    metadata = sample.get("metadata", {}) or {}
    annotations = metadata.get("harmful_annotations", [])
    if annotations:
        return normalize_id(str(annotations[0]))

    scores = sample.get("scores", {}) or {}
    scorer_obj = scores.get("metadata_annotation_scorer", {})
    explanation = scorer_obj.get("explanation")
    if explanation:
        try:
            parsed = json.loads(explanation)
            annotation_id = parsed.get("annotation_id")
            if annotation_id:
                return normalize_id(str(annotation_id))
        except (json.JSONDecodeError, TypeError, KeyError):
            pass
    return "unknown"


def _content_to_text(content: Any) -> str:
    """Flatten message content payloads into plain text.

    Parameters
    ----------
    content:
        Message content payload from an eval sample.

    Returns
    -------
    str
        Flattened text content.
    """
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            parts.append(_content_to_text(item))
        return "\n".join(part for part in parts if part)
    if isinstance(content, dict):
        nested_content = content.get("text", content.get("content"))
        if nested_content is not None:
            return _content_to_text(nested_content)
        parts = []
        for value in content.values():
            if isinstance(value, (str, list, dict)):
                parts.append(_content_to_text(value))
        return "\n".join(part for part in parts if part)
    return ""


def _message_text(message: Any) -> str:
    """Extract text content from one message object.

    Parameters
    ----------
    message:
        Message dictionary from the sample input list.

    Returns
    -------
    str
        Text content for the message.
    """
    if not isinstance(message, dict):
        return ""
    return _content_to_text(message.get("content"))


def _looks_like_code(text: str) -> bool:
    """Heuristically detect code-like text.

    Parameters
    ----------
    text:
        Arbitrary message text.

    Returns
    -------
    bool
        ``True`` when text appears to contain source code.
    """
    if not text:
        return False
    if _CODE_FENCE_PATTERN.search(text) or _INLINE_CODE_PATTERN.search(text):
        return True
    if _CODE_KEYWORD_PATTERN.search(text):
        return True

    symbol_heavy_lines = 0
    indented_operator_lines = 0
    for line in text.splitlines():
        if _SYMBOL_HEAVY_LINE_PATTERN.match(line):
            symbol_heavy_lines += 1
        if line.startswith(("    ", "\t")) and re.search(r"[=(){}[\];<>]", line):
            indented_operator_lines += 1

    return (
        symbol_heavy_lines >= CODE_LIKE_MIN_LINE_MATCHES
        or indented_operator_lines >= CODE_LIKE_MIN_LINE_MATCHES
    )


def _preceding_messages(input_messages: Any) -> list[dict[str, Any]]:
    """Return messages preceding the model-predicted assistant turn.

    Parameters
    ----------
    input_messages:
        Sample ``input`` payload.

    Returns
    -------
    list[dict[str, Any]]
        Messages before the final prompt turn.
    """
    if not isinstance(input_messages, list) or not input_messages:
        return []

    preceding = input_messages[:-1]
    filtered: list[dict[str, Any]] = []
    for message in preceding:
        if isinstance(message, dict):
            filtered.append(message)
    return filtered


def _context_text_features(sample: dict[str, Any]) -> dict[str, int]:
    """Compute context-text covariates for one sample.

    Parameters
    ----------
    sample:
        Raw sample dictionary from an eval log.

    Returns
    -------
    dict[str, int]
        Context text feature values.
    """
    preceding = _preceding_messages(sample.get("input"))
    context_texts = [_message_text(message) for message in preceding]
    context_char_count = int(sum(len(text) for text in context_texts))
    context_message_count = int(len(preceding))
    code_flags = [_looks_like_code(text) for text in context_texts]
    code_message_count = int(sum(1 for flag in code_flags if flag))
    code_char_count = int(
        sum(len(text) for text, has_code in zip(context_texts, code_flags) if has_code)
    )
    return {
        "context_message_count": context_message_count,
        "context_char_count": context_char_count,
        "context_has_code": int(code_message_count > 0),
        "context_code_message_count": code_message_count,
        "context_code_char_count": code_char_count,
    }


def _annotation_cutoff(annotation_id: str) -> float:
    """Return the raw-score cutoff for one annotation ID.

    Parameters
    ----------
    annotation_id:
        Normalized annotation ID, usually with ``bot-`` prefix.

    Returns
    -------
    float
        Raw-score threshold used to binarize transcript labels.
    """
    value = ANNOTATION_CUTOFFS.get(annotation_id, DEFAULT_ANNOTATION_CUTOFF)
    try:
        cutoff = float(value)
    except (TypeError, ValueError):
        cutoff = DEFAULT_ANNOTATION_CUTOFF
    return cutoff


def _prior_annotation_prevalence(
    sample: dict[str, Any], annotation_id: str
) -> dict[str, float]:
    """Compute preceding-context prevalence for the target annotation.

    Parameters
    ----------
    sample:
        Raw sample dictionary from an eval log.
    annotation_id:
        Annotation ID scored for this sample.

    Returns
    -------
    dict[str, float]
        Counts and prevalence among preceding assistant messages with a
        non-null raw score for the target annotation.
    """
    cutoff = _annotation_cutoff(annotation_id)
    preceding = _preceding_messages(sample.get("input"))
    positives = 0
    scored = 0

    for message in preceding:
        if not isinstance(message, dict):
            continue
        if str(message.get("role")) != "assistant":
            continue
        metadata = message.get("metadata", {}) or {}
        raw_value = metadata.get(annotation_id)
        if raw_value is None:
            continue
        try:
            value = float(raw_value)
        except (TypeError, ValueError):
            continue
        if math.isnan(value):
            continue
        scored += 1
        if value >= cutoff:
            positives += 1

    prevalence = float(positives / scored) if scored > 0 else 0.0
    return {
        "prior_annotation_positive_messages": float(positives),
        "prior_annotation_scored_messages": float(scored),
        "prior_annotation_prevalence": prevalence,
    }


def _parse_rows(
    log_path: Path,
    manifest: dict[str, Any],
) -> list[dict[str, Any]]:
    """Parse context rows from one eval log.

    Parameters
    ----------
    log_path:
        Path to a selected ``.eval`` log file.
    manifest:
        Parsed start manifest for the log.

    Returns
    -------
    list[dict[str, Any]]
        Row-level points with model/request/effective context and score.
    """
    return _parse_rows_from_samples(
        _iter_raw_samples(log_path),
        manifest,
    )


def _parse_rows_from_samples(
    raw_samples: Any,
    manifest: dict[str, Any],
) -> list[dict[str, Any]]:
    """Parse context rows from an iterable of raw sample payloads.

    Parameters
    ----------
    raw_samples:
        Iterable of raw sample dictionaries.
    manifest:
        Parsed start manifest for the associated log.

    Returns
    -------
    list[dict[str, Any]]
        Row-level points with model/request/effective context and score.
    """
    model, reasoning, task_args = _manifest_eval_fields(manifest)
    requested_context = int(task_args.get("max_context_messages", 0))
    rows: list[dict[str, Any]] = []

    for sample in raw_samples:
        metadata = sample.get("metadata", {}) or {}
        context_length = metadata.get("context_length")
        if context_length is None and requested_context == 0:
            # Legacy/non-context logs may omit context_length entirely.
            # For requested max context 0, effective context is also 0.
            context_length = 0
        if context_length is None:
            continue

        scores = sample.get("scores", {}) or {}
        scorer_obj = scores.get("metadata_annotation_scorer", {})
        raw_value = scorer_obj.get("value")
        if raw_value is None:
            continue
        try:
            score = float(raw_value)
        except (TypeError, ValueError):
            continue
        if math.isnan(score):
            continue

        code = _extract_code(sample)
        code_short = code.removeprefix("bot-")
        category = _aggregate_plot_category(code)
        context_features = _context_text_features(sample)
        prior_annotation_features = _prior_annotation_prevalence(sample, code)

        rows.append(
            {
                "model": model,
                "reasoning_effort": reasoning,
                "model_label": _model_label(model, reasoning),
                "requested_context_messages": requested_context,
                "effective_context_length": int(context_length),
                "annotation_id": code,
                "code_short": code_short,
                "category": category,
                "score": score,
                "sample_id": str(sample.get("id", "")),
                "window_id": str(metadata.get("window_id", "")),
                "context_message_count": context_features["context_message_count"],
                "context_char_count": context_features["context_char_count"],
                "context_has_code": context_features["context_has_code"],
                "context_code_message_count": context_features[
                    "context_code_message_count"
                ],
                "context_code_char_count": context_features["context_code_char_count"],
                "prior_annotation_positive_messages": prior_annotation_features[
                    "prior_annotation_positive_messages"
                ],
                "prior_annotation_scored_messages": prior_annotation_features[
                    "prior_annotation_scored_messages"
                ],
                "prior_annotation_prevalence": prior_annotation_features[
                    "prior_annotation_prevalence"
                ],
            }
        )
    return rows


def _load_or_parse_rows(
    log_path: Path,
    manifest: dict[str, Any],
    sample_rows_cache_dir: Path,
    *,
    use_cache: bool,
) -> tuple[list[dict[str, Any]], bool]:
    """Load parsed rows from cache, or parse and cache them.

    Parameters
    ----------
    log_path:
        Eval log file path.
    manifest:
        Parsed start manifest for the log.
    sample_rows_cache_dir:
        Directory for per-log parsed-row caches.
    use_cache:
        Whether to read/write cache files.

    Returns
    -------
    tuple[list[dict[str, Any]], bool]
        ``(rows, was_cache_hit)``.
    """
    cache_file = _sample_rows_cache_file(log_path, sample_rows_cache_dir)
    if use_cache and cache_file.exists():
        try:
            with gzip.open(cache_file, "rb") as file_obj:
                cached_rows = pickle.load(file_obj)
            if isinstance(cached_rows, list):
                return cached_rows, True
        except (OSError, EOFError, pickle.UnpicklingError):
            logger.warning("Ignoring unreadable sample rows cache %s", cache_file)

    rows = _parse_rows(log_path, manifest)
    if use_cache:
        sample_rows_cache_dir.mkdir(parents=True, exist_ok=True)
        tmp_file = cache_file.with_suffix(".tmp")
        try:
            with gzip.open(tmp_file, "wb") as file_obj:
                pickle.dump(rows, file_obj, protocol=pickle.HIGHEST_PROTOCOL)
            os.replace(tmp_file, cache_file)
        except OSError:
            logger.warning("Could not write sample rows cache %s", cache_file)
    return rows, False


def _model_reasoning_key(manifest: dict[str, Any]) -> tuple[str, Optional[str]]:
    """Build a ``(model, reasoning_effort)`` key from manifest data.

    Parameters
    ----------
    manifest:
        Parsed start manifest for an eval log.

    Returns
    -------
    tuple[str, Optional[str]]
        Model and reasoning setting from the eval.
    """
    model, reasoning, _task_args = _manifest_eval_fields(manifest)
    return model, reasoning


def _requested_context_messages(manifest: dict[str, Any]) -> int:
    """Extract requested context depth from an eval manifest.

    Parameters
    ----------
    manifest:
        Parsed start manifest for an eval log.

    Returns
    -------
    int
        Requested maximum context messages.
    """
    _model, _reasoning, task_args = _manifest_eval_fields(manifest)
    return int(task_args.get("max_context_messages", 0))


def _row_model_reasoning_code_key(row: dict[str, Any]) -> tuple[str, Any, str]:
    """Build the baseline/context matching key for one parsed row.

    Parameters
    ----------
    row:
        Parsed row dictionary.

    Returns
    -------
    tuple[str, Any, str]
        ``(model, reasoning_effort, annotation_id)``
    """
    return (row["model"], row["reasoning_effort"], row["annotation_id"])


def _is_cache_fresh(
    cache_path: Path, selected_logs: list[tuple[Path, dict[str, Any]]]
) -> bool:
    """Check whether a cached parquet is newer than selected log inputs.

    Parameters
    ----------
    cache_path:
        Path to cached rows parquet.
    selected_logs:
        Selected ``(log_path, manifest)`` pairs used as parse inputs.

    Returns
    -------
    bool
        ``True`` when cache file exists and is newer than all selected logs.
    """
    if not cache_path.exists() or not selected_logs:
        return False
    cache_mtime = cache_path.stat().st_mtime
    newest_log = max(log_path.stat().st_mtime for log_path, _ in selected_logs)
    return cache_mtime >= newest_log


def _load_cached_rows(cache_path: Path) -> Optional[pd.DataFrame]:
    """Load cached row-level context data if readable and schema-compatible.

    Parameters
    ----------
    cache_path:
        Path to rows cache parquet.

    Returns
    -------
    Optional[pd.DataFrame]
        Cached rows when valid, otherwise ``None``.
    """
    required_cols = {
        "model",
        "reasoning_effort",
        "model_label",
        "requested_context_messages",
        "effective_context_length",
        "annotation_id",
        "code_short",
        "category",
        "score",
        "sample_id",
        "window_id",
        "context_message_count",
        "context_char_count",
        "context_has_code",
        "context_code_message_count",
        "context_code_char_count",
        "prior_annotation_positive_messages",
        "prior_annotation_scored_messages",
        "prior_annotation_prevalence",
        "_cache_schema_version",
    }
    try:
        df = pd.read_parquet(cache_path)
    except (OSError, ValueError) as exc:
        logger.warning("Could not read rows cache %s: %s", cache_path, exc)
        return None
    missing = sorted(required_cols - set(df.columns))
    if missing:
        logger.warning(
            "Rows cache %s missing columns: %s", cache_path, ", ".join(missing)
        )
        return None
    if not (df["_cache_schema_version"] == ROWS_CACHE_SCHEMA_VERSION).all():
        logger.warning("Rows cache %s has incompatible schema version.", cache_path)
        return None
    return df


def _drop_excluded_participant_rows(
    rows_df: pd.DataFrame, excluded_window_ids: set[str]
) -> pd.DataFrame:
    """Drop rows whose ``window_id`` belongs to an excluded participant.

    Parameters
    ----------
    rows_df:
        Row-level context-effect dataframe.
    excluded_window_ids:
        Window IDs associated with excluded participants.

    Returns
    -------
    pd.DataFrame
        Filtered dataframe.
    """
    if not excluded_window_ids:
        return rows_df

    before_rows = len(rows_df)
    filtered = rows_df[
        ~rows_df["window_id"].astype(str).str.strip().isin(excluded_window_ids)
    ].copy()
    dropped_rows = before_rows - len(filtered)
    logger.info(
        "Dropped %d/%d context rows by participant exclusion.",
        dropped_rows,
        before_rows,
    )
    return filtered


def resolve_context_excluded_window_ids(
    raw_excluded_participants: str | None,
) -> tuple[set[str], set[str]]:
    """Resolve participant and window exclusions for context analyses.

    Parameters
    ----------
    raw_excluded_participants:
        Optional comma-separated participant list from CLI. ``None`` means use
        environment/default behavior from shared exclusion helpers.

    Returns
    -------
    tuple[set[str], set[str]]
        ``(excluded_participants, excluded_window_ids)``.
    """
    windows_path = resolve_path(
        "LLM_DELUSIONS_WINDOWS_PATH",
        DEFAULT_WINDOWS_PATH,
        require_local=True,
    )
    try:
        excluded_participants, excluded_window_ids = resolve_excluded_window_ids(
            windows_path, raw_participants=raw_excluded_participants
        )
    except (OSError, ValueError, KeyError) as exc:
        logger.warning(
            "Could not resolve excluded participants from windows data (%s); "
            "continuing without exclusions.",
            exc,
        )
        return set(), set()

    if excluded_participants:
        logger.info(
            "Participant exclusion enabled for %s (%d window IDs).",
            sorted(excluded_participants),
            len(excluded_window_ids),
        )
    else:
        logger.info("Participant exclusion disabled.")
    return excluded_participants, excluded_window_ids


def _select_zero_context_baseline_logs(
    baseline_logs_dir: Path,
    context_model_reasoning: set[tuple[str, Optional[str]]],
    metadata_cache: dict[str, Any],
) -> tuple[list[tuple[Path, dict[str, Any]]], int]:
    """Select baseline logs at zero requested context for context-run models.

    Parameters
    ----------
    baseline_logs_dir:
        Directory containing baseline eval ``.eval`` files.
    context_model_reasoning:
        Model/reasoning pairs present in selected context logs.
    metadata_cache:
        Mutable cache for log manifest/sample count metadata.

    Returns
    -------
    tuple[list[tuple[Path, dict[str, Any]]], int]
        Selected baseline ``(log_path, manifest)`` pairs and metadata cache hits.
    """
    baseline_files = sorted(baseline_logs_dir.glob("*.eval"))
    if not baseline_files:
        return [], 0

    selected_baseline, cache_hits = _select_best_logs(baseline_files, metadata_cache)
    baseline_selected = [
        (log_path, manifest)
        for log_path, manifest in selected_baseline
        if _requested_context_messages(manifest) == 0
        and _model_reasoning_key(manifest) in context_model_reasoning
    ]
    if baseline_selected:
        logger.info(
            "Selected %d baseline log(s) with requested_context_messages=0 from %s",
            len(baseline_selected),
            baseline_logs_dir,
        )
    return baseline_selected, cache_hits


def _add_matching_baseline_rows(
    context_rows: list[dict[str, Any]],
    baseline_selected: list[tuple[Path, dict[str, Any]]],
    sample_rows_cache_dir: Path,
    *,
    use_cache: bool,
) -> tuple[int, int]:
    """Append baseline rows that match context-run model/reasoning/code keys.

    Parameters
    ----------
    context_rows:
        Parsed rows from selected context logs; baseline rows are appended in-place.
    baseline_selected:
        Selected baseline logs.
    sample_rows_cache_dir:
        Cache directory for parsed per-log rows.
    use_cache:
        Whether per-log row caches are enabled.

    Returns
    -------
    tuple[int, int]
        ``(appended_row_count, sample_rows_cache_hits)``.
    """
    context_code_keys = {_row_model_reasoning_code_key(row) for row in context_rows}
    context_zero_keys = {
        _row_model_reasoning_code_key(row)
        for row in context_rows
        if int(row["requested_context_messages"]) == 0
    }

    baseline_rows_added = 0
    sample_rows_cache_hits = 0
    for log_path, manifest in baseline_selected:
        logger.info("Parsing baseline %s", log_path.name)
        rows, was_cache_hit = _load_or_parse_rows(
            log_path,
            manifest,
            sample_rows_cache_dir,
            use_cache=use_cache,
        )
        if was_cache_hit:
            sample_rows_cache_hits += 1
        for row in rows:
            code_key = _row_model_reasoning_code_key(row)
            if code_key not in context_code_keys or code_key in context_zero_keys:
                continue
            context_rows.append(row)
            baseline_rows_added += 1
    return baseline_rows_added, sample_rows_cache_hits


def load_context_effect_data(
    logs_dir: Path,
    *,
    baseline_logs_dir: Path,
    cache_paths: ContextEffectCachePaths,
    use_cache: bool,
    excluded_window_ids: set[str] | None = None,
) -> pd.DataFrame:
    """Load context effect rows from context eval logs.

    Parameters
    ----------
    logs_dir:
        Directory containing context eval ``.eval`` files.
    baseline_logs_dir:
        Directory containing baseline ``.eval`` files used to add
        ``requested_context_messages=0`` points when matching context
        model/code pairs exist.
    cache_paths:
        Paths for rows, metadata, and per-log parsed-row caches.
    use_cache:
        Whether to read/write cache files.
    excluded_window_ids:
        Optional set of excluded ``window_id`` values. Rows matching these
        IDs are dropped after loading (both cache and fresh parse paths).

    Returns
    -------
    pd.DataFrame
        Parsed row-level context effect data.
    """
    excluded_window_ids = excluded_window_ids or set()
    log_files = sorted(logs_dir.glob("*.eval"))
    if not log_files:
        raise FileNotFoundError(f"No .eval files found in {logs_dir}")

    metadata_cache = (
        _load_metadata_cache(cache_paths.log_metadata_cache) if use_cache else {}
    )
    selected, context_meta_cache_hits = _select_best_logs(log_files, metadata_cache)
    if not selected:
        raise FileNotFoundError(f"No readable .eval files found in {logs_dir}")

    context_model_reasoning = {
        _model_reasoning_key(manifest) for _, manifest in selected
    }
    baseline_selected, baseline_meta_cache_hits = _select_zero_context_baseline_logs(
        baseline_logs_dir, context_model_reasoning, metadata_cache
    )
    if use_cache:
        _save_metadata_cache(cache_paths.log_metadata_cache, metadata_cache)
        logger.info(
            "Metadata cache hits: %d/%d",
            context_meta_cache_hits + baseline_meta_cache_hits,
            len(log_files) + len(list(baseline_logs_dir.glob("*.eval"))),
        )

    cache_inputs = selected + baseline_selected
    if use_cache and _is_cache_fresh(cache_paths.rows_cache, cache_inputs):
        cached = _load_cached_rows(cache_paths.rows_cache)
        if cached is not None:
            logger.info("Loading context rows from cache %s", cache_paths.rows_cache)
            return _drop_excluded_participant_rows(cached, excluded_window_ids)

    context_rows: list[dict[str, Any]] = []
    sample_rows_cache_hits = 0
    for log_path, manifest in selected:
        logger.info("Parsing %s", log_path.name)
        rows, was_cache_hit = _load_or_parse_rows(
            log_path,
            manifest,
            cache_paths.sample_rows_cache_dir,
            use_cache=use_cache,
        )
        if was_cache_hit:
            sample_rows_cache_hits += 1
        context_rows.extend(rows)

    if not context_rows:
        raise ValueError(f"No scored rows found in selected logs from {logs_dir}")

    (
        baseline_rows_added,
        baseline_sample_rows_cache_hits,
    ) = _add_matching_baseline_rows(
        context_rows,
        baseline_selected,
        cache_paths.sample_rows_cache_dir,
        use_cache=use_cache,
    )
    sample_rows_cache_hits += baseline_sample_rows_cache_hits
    logger.info(
        "Per-log row cache hits: %d/%d",
        sample_rows_cache_hits,
        len(selected) + len(baseline_selected),
    )
    if baseline_rows_added:
        logger.info(
            "Added %d baseline row(s) at requested_context_messages=0",
            baseline_rows_added,
        )

    rows_df = pd.DataFrame(context_rows)
    rows_df["_cache_schema_version"] = ROWS_CACHE_SCHEMA_VERSION
    if use_cache:
        cache_paths.rows_cache.parent.mkdir(parents=True, exist_ok=True)
        rows_df.to_parquet(cache_paths.rows_cache, index=False)
        logger.info("Wrote rows cache %s", cache_paths.rows_cache)
    return _drop_excluded_participant_rows(rows_df, excluded_window_ids)


def aggregate_context_effects(
    rows_df: pd.DataFrame,
    codes: Optional[list[str]],
    *,
    include_validates_codes: bool,
    min_effective_context_messages: int = 0,
    min_effective_fraction_of_requested: float = 0.0,
) -> pd.DataFrame:
    """Aggregate code-level prevalence and effective context by request setting.

    Parameters
    ----------
    rows_df:
        Row-level context effect data.
    codes:
        Optional list of normalized codes to keep.
    include_validates_codes:
        Whether to keep ``validates-*`` codes when ``codes`` is not specified.
    min_effective_context_messages:
        Drop rows with fewer than this many effective context messages.
    min_effective_fraction_of_requested:
        Drop rows where effective context is below this fraction of the
        requested context for that run.

    Returns
    -------
    pd.DataFrame
        Aggregated points for plotting.
    """
    df = _filter_context_rows(
        rows_df,
        codes,
        include_validates_codes=include_validates_codes,
        min_effective_context_messages=min_effective_context_messages,
        min_effective_fraction_of_requested=min_effective_fraction_of_requested,
    )

    group_cols = [
        "model",
        "model_label",
        "reasoning_effort",
        "requested_context_messages",
        "annotation_id",
        "code_short",
        "category",
    ]
    return _aggregate_grouped_context_effects(df, group_cols)


def aggregate_context_effects_by_category(
    rows_df: pd.DataFrame,
    codes: Optional[list[str]],
    *,
    include_validates_codes: bool,
    min_effective_context_messages: int = 0,
    min_effective_fraction_of_requested: float = 0.0,
) -> pd.DataFrame:
    """Aggregate category-level prevalence and effective context by request setting.

    Parameters
    ----------
    rows_df:
        Row-level context effect data.
    codes:
        Optional list of normalized codes to keep before category aggregation.
    include_validates_codes:
        Whether to keep ``validates-*`` codes when ``codes`` is not specified.
    min_effective_context_messages:
        Drop rows with fewer than this many effective context messages.
    min_effective_fraction_of_requested:
        Drop rows where effective context is below this fraction of the
        requested context for that run.

    Returns
    -------
    pd.DataFrame
        Aggregated category-level points for plotting.
    """
    df = _filter_context_rows(
        rows_df,
        codes,
        include_validates_codes=include_validates_codes,
        min_effective_context_messages=min_effective_context_messages,
        min_effective_fraction_of_requested=min_effective_fraction_of_requested,
    )
    group_cols = [
        "model",
        "model_label",
        "reasoning_effort",
        "requested_context_messages",
        "category",
    ]
    return _aggregate_grouped_context_effects(df, group_cols)


def _filter_context_rows(
    rows_df: pd.DataFrame,
    codes: Optional[list[str]],
    *,
    include_validates_codes: bool,
    min_effective_context_messages: int,
    min_effective_fraction_of_requested: float,
) -> pd.DataFrame:
    """Filter row-level context data prior to aggregation.

    Parameters
    ----------
    rows_df:
        Row-level context effect data.
    codes:
        Optional list of normalized codes to keep.
    include_validates_codes:
        Whether to keep ``validates-*`` codes when ``codes`` is not specified.
    min_effective_context_messages:
        Drop rows with fewer than this many effective context messages.
    min_effective_fraction_of_requested:
        Drop rows where effective context is below this fraction of the
        requested context for that run.

    Returns
    -------
    pd.DataFrame
        Filtered row-level dataframe.
    """
    filtered_df = rows_df.copy()
    if codes:
        normalized_codes = {normalize_id(code) for code in codes}
        filtered_df = filtered_df[filtered_df["annotation_id"].isin(normalized_codes)]
        if filtered_df.empty:
            raise ValueError("No rows matched requested code filters.")
    elif not include_validates_codes:
        filtered_df = filtered_df[
            ~filtered_df["annotation_id"].astype(str).map(_is_validates_code)
        ]
        if filtered_df.empty:
            raise ValueError(
                "No rows remained after default validates-* exclusion. "
                "Use --include-validates-codes to keep these codes."
            )

    original_rows = len(filtered_df)
    if min_effective_context_messages > 0:
        filtered_df = filtered_df[
            filtered_df["effective_context_length"] >= min_effective_context_messages
        ]
    if min_effective_fraction_of_requested > 0:
        required = (
            min_effective_fraction_of_requested
            * filtered_df["requested_context_messages"]
        )
        filtered_df = filtered_df[filtered_df["effective_context_length"] >= required]
    if filtered_df.empty:
        raise ValueError(
            "No rows remained after effective-context filtering. "
            "Lower min thresholds or collect longer-context evals."
        )

    filtered_rows = original_rows - len(filtered_df)
    if filtered_rows > 0:
        logger.info(
            "Dropped %d/%d rows via effective-context filters "
            "(min_messages=%d, min_fraction=%.3f)",
            filtered_rows,
            original_rows,
            min_effective_context_messages,
            min_effective_fraction_of_requested,
        )
    return filtered_df


def _aggregate_grouped_context_effects(
    rows_df: pd.DataFrame,
    group_cols: list[str],
) -> pd.DataFrame:
    """Aggregate prevalence and effective context metrics for group columns.

    Parameters
    ----------
    rows_df:
        Filtered row-level context effect data.
    group_cols:
        Grouping columns for aggregation.

    Returns
    -------
    pd.DataFrame
        Aggregated points with prevalence and effective-context confidence
        intervals.
    """
    grouped_rows: list[dict[str, Any]] = []
    use_clustered_ci = (
        "participant" in rows_df.columns
        and "conversation_id" in rows_df.columns
        and rows_df["participant"].astype(str).str.strip().ne("").any()
        and rows_df["conversation_id"].astype(str).str.strip().ne("").any()
    )
    for group_key, group_df in rows_df.groupby(group_cols, dropna=False):
        window_ids = (
            _canonical_window_ids(group_df["window_id"])
            if "window_id" in group_df
            else pd.Series(dtype=str)
        )
        unique_window_count = int(window_ids[window_ids != ""].nunique())
        if unique_window_count == 0:
            unique_window_count = int(group_df["score"].count())

        if use_clustered_ci:
            prevalence_summary = _clustered_mean_ci_for_group(group_df, "score")
            effective_context_summary = _clustered_mean_ci_for_group(
                group_df,
                "effective_context_length",
            )
            prevalence = float(prevalence_summary["estimate"])
            prevalence_ci_lower = float(prevalence_summary["ci_low"])
            prevalence_ci_upper = float(prevalence_summary["ci_high"])
            effective_context_length = float(effective_context_summary["estimate"])
            effective_context_ci_lower = float(effective_context_summary["ci_low"])
            effective_context_ci_upper = float(effective_context_summary["ci_high"])
            n_participants_supported = int(
                prevalence_summary["n_participants_supported"]
            )
            ci_method = "hierarchical_participant_conversation_bootstrap"
            cluster_boot_n = int(prevalence_summary["cluster_boot_n"])
        else:
            prevalence, prevalence_ci_lower, prevalence_ci_upper = _bootstrap_mean_ci(
                group_df["score"]
            )
            (
                effective_context_length,
                effective_context_ci_lower,
                effective_context_ci_upper,
            ) = _bootstrap_mean_ci(group_df["effective_context_length"])
            n_participants_supported = 0
            ci_method = "row_bootstrap"
            cluster_boot_n = 0

        row = dict(zip(group_cols, group_key))
        row.update(
            {
                "effective_context_length": effective_context_length,
                "effective_context_ci_lower": effective_context_ci_lower,
                "effective_context_ci_upper": effective_context_ci_upper,
                "prevalence": prevalence,
                "prevalence_ci_lower": prevalence_ci_lower,
                "prevalence_ci_upper": prevalence_ci_upper,
                "n": int(group_df["score"].count()),
                "w": unique_window_count,
                "n_participants_supported": n_participants_supported,
                "ci_method": ci_method,
                "cluster_boot_n": cluster_boot_n,
            }
        )
        grouped_rows.append(row)

    grouped = pd.DataFrame(grouped_rows)
    grouped["prevalence_pct"] = grouped["prevalence"] * 100.0
    grouped["prevalence_ci_lower_pct"] = grouped["prevalence_ci_lower"] * 100.0
    grouped["prevalence_ci_upper_pct"] = grouped["prevalence_ci_upper"] * 100.0
    return grouped


def _clustered_mean_ci_for_group(
    group_df: pd.DataFrame,
    value_col: str,
) -> dict[str, float | int]:
    """Compute one hierarchical weighted-mean summary for a group.

    Parameters
    ----------
    group_df:
        One grouped slice of context-effect rows.
    value_col:
        Numeric value column to summarize.

    Returns
    -------
    dict[str, float | int]
        Weighted mean estimate, percentile interval, support count, and
        cluster bootstrap size.
    """
    aggregated = aggregate_participant_conversation_value_sums(
        group_df,
        [],
        value_col=value_col,
    )
    return hierarchical_weighted_mean_ci(
        aggregated,
        config=DEFAULT_CLUSTER_BOOTSTRAP_CONFIG,
    )


def _canonical_window_id(window_id: object) -> str:
    """Normalize one window ID to a stable matching key.

    Parameters
    ----------
    window_id:
        Raw window identifier from row metadata.

    Returns
    -------
    str
        Stable matching key. Filename-style context IDs are converted to the
        corresponding hashed ``eval_subset_id`` so they can match baseline
        zero-context rows.
    """
    normalized = str(window_id or "").strip()
    if not normalized:
        return ""
    if "." not in normalized:
        return normalized
    label, filename = normalized.split(".", maxsplit=1)
    label = label.strip()
    filename = filename.strip()
    if not label or not filename:
        return normalized
    return build_eval_subset_id_from_subset_rel_path(f"{label}/{filename}.json")


def _canonical_window_ids(window_ids: pd.Series) -> pd.Series:
    """Return canonical matching IDs for a window-id series.

    Parameters
    ----------
    window_ids:
        Raw window-id series.

    Returns
    -------
    pd.Series
        Canonical IDs aligned to the input index.
    """
    return window_ids.astype(str).map(_canonical_window_id)


def _restrict_to_uniform_sample_set(
    rows_df: pd.DataFrame,
    *,
    requested_context_upper_limit: int,
) -> pd.DataFrame:
    """Restrict rows to a uniform sample set across requested context points.

    This function enforces two constraints:

    1. Keep only rows where ``requested_context_messages`` is less than or equal
       to ``requested_context_upper_limit``.
    2. For each ``(model, reasoning_effort, annotation_id)`` cohort, keep only
       window IDs present at every requested-context setting greater than zero.

    Parameters
    ----------
    rows_df:
        Filtered row-level context effect data.
    requested_context_upper_limit:
        Maximum requested context setting to keep.

    Returns
    -------
    pd.DataFrame
        Row-level dataframe with a fixed window set per cohort.
    """
    capped = rows_df[
        rows_df["requested_context_messages"] <= requested_context_upper_limit
    ].copy()
    if capped.empty:
        raise ValueError(
            "No rows remained after applying uniform sample upper limit "
            f"{requested_context_upper_limit}."
        )
    capped["_uniform_window_id"] = _canonical_window_ids(capped["window_id"])

    cohort_cols = ["model", "reasoning_effort", "annotation_id"]
    retained_parts: list[pd.DataFrame] = []

    for cohort_key, cohort_df in capped.groupby(cohort_cols, dropna=False):
        window_sets: list[set[str]] = []
        requested_values = sorted(cohort_df["requested_context_messages"].unique())
        requested_values_for_intersection = [
            value for value in requested_values if int(value) > 0
        ]
        if not requested_values_for_intersection:
            requested_values_for_intersection = requested_values

        for requested in requested_values_for_intersection:
            requested_df = cohort_df[
                cohort_df["requested_context_messages"] == requested
            ].copy()
            window_ids = set(requested_df["_uniform_window_id"].astype(str))
            window_ids.discard("")
            window_sets.append(window_ids)

        if not window_sets:
            continue
        shared_window_ids = set.intersection(*window_sets)
        if not shared_window_ids:
            logger.warning(
                "Uniform sample filtering dropped cohort %s due to empty "
                "window intersection across requested settings > 0 and <= %d",
                cohort_key,
                requested_context_upper_limit,
            )
            continue

        retained_parts.append(
            cohort_df[
                cohort_df["_uniform_window_id"].astype(str).isin(shared_window_ids)
            ]
        )
        logger.info(
            "Uniform sample cohort %s kept %d shared window(s) across %d "
            "requested-context point(s)",
            cohort_key,
            len(shared_window_ids),
            len(requested_values_for_intersection),
        )

    if not retained_parts:
        raise ValueError(
            "No rows remained after uniform sample filtering. "
            "Try a lower --uniform-sample-upper-limit."
        )

    uniform_df = pd.concat(retained_parts, ignore_index=True)
    dropped = len(capped) - len(uniform_df)
    if dropped > 0:
        logger.info(
            "Uniform sample filter dropped %d/%d row(s) at upper limit %d",
            dropped,
            len(capped),
            requested_context_upper_limit,
        )
    return uniform_df.drop(columns="_uniform_window_id")


def _bootstrap_mean_ci(
    values: pd.Series,
    *,
    confidence_level: float = DEFAULT_CI,
    n_resamples: int = DEFAULT_N_RESAMPLES,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
) -> tuple[float, float, float]:
    """Compute mean and SciPy bootstrap CI for a numeric series.

    Parameters
    ----------
    values:
        Numeric values to summarize.
    confidence_level:
        Confidence level for interval bounds.
    n_resamples:
        Number of bootstrap resamples.
    seed:
        Seed for deterministic bootstrap draws.

    Returns
    -------
    tuple[float, float, float]
        ``(mean, ci_lower, ci_upper)``.
    """
    numeric = pd.to_numeric(values, errors="coerce").dropna().to_numpy(dtype=float)
    if len(numeric) == 0:
        nan_value = float("nan")
        return (nan_value, nan_value, nan_value)

    mean_value = float(np.mean(numeric))
    if len(numeric) == 1:
        return (mean_value, mean_value, mean_value)

    rng = np.random.default_rng(seed)
    try:
        result = bootstrap(
            (numeric,),
            np.mean,
            n_resamples=n_resamples,
            confidence_level=confidence_level,
            method="percentile",
            vectorized=True,
            rng=rng,
        )
    except ValueError:
        return (mean_value, mean_value, mean_value)

    return (
        mean_value,
        float(result.confidence_interval.low),
        float(result.confidence_interval.high),
    )


def _safe_name(text: str) -> str:
    """Create a filename-safe slug from free text."""
    return re.sub(r"[^a-zA-Z0-9_-]+", "_", text).strip("_").lower()


def _path_within(path: Path, root: Path) -> bool:
    """Check whether ``path`` is within ``root`` after resolution.

    Parameters
    ----------
    path:
        Path to test.
    root:
        Candidate ancestor path.

    Returns
    -------
    bool
        ``True`` when the resolved path is contained by the resolved root.
    """
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError:
        return False
    return True


def _warn_if_repo_local_sensitive_paths(
    logs_dir: Path,
    rows_cache: Path,
    log_metadata_cache: Path,
    sample_rows_cache_dir: Path,
) -> None:
    """Warn when context-sensitive inputs/outputs are inside the eval repo.

    Parameters
    ----------
    logs_dir:
        Input logs directory.
    rows_cache:
        Row-cache parquet path.
    log_metadata_cache:
        Log metadata cache JSON path.
    sample_rows_cache_dir:
        Per-log parsed-row cache directory.
    """
    repo_paths = [
        ("logs-dir", logs_dir),
        ("rows-cache", rows_cache),
        ("log-metadata-cache", log_metadata_cache),
        ("sample-rows-cache-dir", sample_rows_cache_dir),
    ]
    local = [
        f"{name}={path}"
        for name, path in repo_paths
        if _path_within(path, _EVALS_REPO_ROOT)
    ]
    if not local:
        return
    logger.warning(
        "Context-effect inputs/cache are sensitive. Repo-local paths detected (%s). "
        "Remove these artifacts before any public push or release branch.",
        ", ".join(local),
    )


def _ordered_groups(
    values: list[str],
    *,
    preferred_order: list[str],
) -> list[str]:
    """Sort group labels using a preferred order, then alphabetically.

    Parameters
    ----------
    values:
        Group values found in the dataframe.
    preferred_order:
        Preferred ordering keys.

    Returns
    -------
    list[str]
        Ordered group values with unknown labels appended alphabetically.
    """
    deduped = {str(value) for value in values}
    ordered = [value for value in preferred_order if value in deduped]
    extras = sorted(value for value in deduped if value not in preferred_order)
    return ordered + extras


def _plot_model_series_on_axis(
    ax: Any,
    group_df: pd.DataFrame,
    *,
    annotate_windows: bool,
) -> list[Any]:
    """Plot all model series for one group onto a provided axis.

    Parameters
    ----------
    ax:
        Target matplotlib axis.
    group_df:
        Subset dataframe for one group.
    annotate_windows:
        Whether to annotate each point with the unique-window count.

    Returns
    -------
    list[Any]
        Legend handles for the model series plotted on this axis.
    """
    model_labels = sort_model_labels(group_df["model_label"].dropna().unique().tolist())
    handles: list[Any] = []
    for model_label in model_labels:
        model_df = (
            group_df[group_df["model_label"] == model_label]
            .copy()
            .sort_values("effective_context_length")
        )
        x_values = model_df["effective_context_length"].to_numpy(dtype=float)
        y_values = model_df["prevalence_pct"].to_numpy(dtype=float)
        xerr = np.vstack(
            [
                (
                    model_df["effective_context_length"]
                    - model_df["effective_context_ci_lower"]
                ).clip(lower=0.0),
                (
                    model_df["effective_context_ci_upper"]
                    - model_df["effective_context_length"]
                ).clip(lower=0.0),
            ]
        )
        yerr = np.vstack(
            [
                (model_df["prevalence_pct"] - model_df["prevalence_ci_lower_pct"]).clip(
                    lower=0.0
                ),
                (model_df["prevalence_ci_upper_pct"] - model_df["prevalence_pct"]).clip(
                    lower=0.0
                ),
            ]
        )
        errorbar = ax.errorbar(
            x_values,
            y_values,
            xerr=xerr,
            yerr=yerr,
            fmt="o",
            label=model_label,
            markersize=5,
            alpha=0.85,
            capsize=3,
            color=get_model_color(model_label),
        )
        handles.append(errorbar)
        if annotate_windows:
            for _, row in model_df.iterrows():
                ax.annotate(
                    f"w={int(row['w'])}",
                    (row["effective_context_length"], row["prevalence_pct"]),
                    xytext=(4, 4),
                    textcoords="offset points",
                    fontsize=7,
                )
    ax.grid(True, alpha=0.3)
    return handles


def _context_xlim_with_padding(x_min: float, x_max: float) -> tuple[float, float]:
    """Return x-axis limits padded to avoid clipping endpoint markers.

    Parameters
    ----------
    x_min:
        Minimum x value to include before padding.
    x_max:
        Maximum x value to include before padding.

    Returns
    -------
    tuple[float, float]
        ``(x_lower, x_upper)`` with symmetric endpoint padding.
    """
    return (
        x_min - CONTEXT_EFFECT_X_ENDPOINT_PAD_MESSAGES,
        x_max + CONTEXT_EFFECT_X_ENDPOINT_PAD_MESSAGES,
    )


def _plot_context_effects_grouped_subplots(
    points_df: pd.DataFrame,
    output_dir: Path,
    *,
    config: ContextSubplotConfig,
) -> Optional[Path]:
    """Create a multi-panel grouped context-effect figure.

    Parameters
    ----------
    points_df:
        Aggregated points dataframe.
    output_dir:
        Directory for figure outputs.
    config:
        Plot layout and grouping configuration.

    Returns
    -------
    Optional[Path]
        Output PDF path, or ``None`` when no matching rows are available.
    """
    plot_df = points_df[points_df[config.group_col].isin(config.groups)].copy()
    if plot_df.empty:
        return None

    output_dir.mkdir(parents=True, exist_ok=True)
    rows, cols = config.subplot_shape
    fig, axes = plt.subplots(
        rows,
        cols,
        figsize=config.figsize,
        sharex=True,
        sharey=True,
    )
    flat_axes = np.atleast_1d(axes).ravel()
    if len(config.groups) > len(flat_axes):
        logger.warning(
            "Subplot config has %d groups but only %d panel(s); truncating groups.",
            len(config.groups),
            len(flat_axes),
        )
    groups_to_plot = config.groups[: len(flat_axes)]
    legend_handles: list[Any] = []
    legend_labels: list[str] = []

    x_min = float(plot_df["effective_context_ci_lower"].min())
    x_max = float(plot_df["effective_context_ci_upper"].max())
    x_lower, x_upper = _context_xlim_with_padding(x_min, x_max)
    y_min = max(0.0, float(plot_df["prevalence_ci_lower_pct"].min()) - 1.0)
    y_max = min(100.0, float(plot_df["prevalence_ci_upper_pct"].max()) + 1.0)
    if y_max <= y_min:
        y_max = min(100.0, y_min + 1.0)

    for index, group_value in enumerate(groups_to_plot):
        ax = flat_axes[index]
        group_df = plot_df[plot_df[config.group_col] == group_value]
        handles = _plot_model_series_on_axis(
            ax,
            group_df,
            annotate_windows=config.annotate_windows,
        )
        if handles and not legend_handles:
            legend_handles = handles
            legend_labels = [
                str(handle.get_label())
                for handle in handles
                if str(handle.get_label()) != "_nolegend_"
            ]

        ax.set_title(
            format_metric_label_plain_for_matplotlib(str(group_value)),
            fontsize=8,
        )
        ax.set_xlim(x_lower, x_upper)
        ax.set_ylim(y_min, y_max)

    for index in range(len(groups_to_plot), len(flat_axes)):
        flat_axes[index].axis("off")

    show_legend = len(legend_labels) > 1
    if config.figure_title:
        fig.suptitle(config.figure_title, y=0.995)
    fig.supxlabel(
        "Context Length (messages)",
        y=0.03 if show_legend else 0.02,
    )
    fig.supylabel("Prevalence (%)", x=0.02)
    if show_legend:
        fig.legend(
            legend_handles,
            legend_labels,
            loc="lower center",
            ncol=min(4, len(legend_labels)),
            fontsize=7,
            frameon=True,
            bbox_to_anchor=(0.5, 0.005),
        )
    fig.tight_layout(
        rect=(
            0.03,
            0.06 if show_legend else 0.04,
            1.0,
            0.98 if config.figure_title else 1.0,
        )
    )

    pdf_path = output_dir / f"{config.filename_stem}.pdf"
    png_path = output_dir / f"{config.filename_stem}.png"
    fig.savefig(pdf_path)
    fig.savefig(png_path)
    plt.close(fig)
    logger.info("Wrote %s and %s", pdf_path, png_path)
    return pdf_path


def plot_gpt54_context_effect_code_subplots(
    points_df: pd.DataFrame,
    output_dir: Path,
    *,
    filename_stem: str = "context_effect_subplots_gpt54_upto_400_codes",
) -> Optional[Path]:
    """Create a GPT-5.4 appendix figure with code-level context-effect subplots.

    The figure includes one panel per code (18 total) and only keeps rows where
    requested context depth is less than or equal to 400 messages.

    Parameters
    ----------
    points_df:
        Aggregated code-level context points.
    output_dir:
        Directory for figure outputs.
    filename_stem:
        Output filename stem.

    Returns
    -------
    Optional[Path]
        Output PDF path, or ``None`` when no matching rows are available.
    """
    gpt54_df = points_df[
        (points_df["model"] == GPT54_MODEL_ID)
        & (points_df["requested_context_messages"] <= int(GPT54_APPENDIX_CONTEXT_MAX))
    ].copy()
    if gpt54_df.empty:
        logger.warning(
            "No GPT-5.4 context-effect rows found for requested context <= %d; "
            "skipping code subplot appendix figure.",
            GPT54_APPENDIX_CONTEXT_MAX,
        )
        return None

    ordered_codes = _ordered_groups(
        gpt54_df["annotation_id"].astype(str).tolist(),
        preferred_order=GPT54_APPENDIX_CODE_ORDER,
    )
    return _plot_context_effects_grouped_subplots(
        gpt54_df,
        output_dir,
        config=ContextSubplotConfig(
            group_col="annotation_id",
            groups=ordered_codes,
            filename_stem=filename_stem,
            subplot_shape=(6, 3),
            figsize=(8.5, 11.0),
            annotate_windows=False,
            figure_title="Code-Level Effective Context Length",
        ),
    )


def plot_gpt54_context_effect_category_subplots(
    points_df: pd.DataFrame,
    output_dir: Path,
    *,
    filename_stem: str = "context_effect_subplots_gpt54_upto_400_categories",
) -> Optional[Path]:
    """Create a GPT-5.4 appendix figure with aggregate category subplots.

    The figure includes the five aggregate categories (sycophancy, delusional,
    relationship, facilitates harm, discourages harm) and keeps rows where
    requested context depth is less than or equal to 400 messages.

    Parameters
    ----------
    points_df:
        Aggregated category-level context points.
    output_dir:
        Directory for figure outputs.
    filename_stem:
        Output filename stem.

    Returns
    -------
    Optional[Path]
        Output PDF path, or ``None`` when no matching rows are available.
    """
    gpt54_df = points_df[
        (points_df["model"] == GPT54_MODEL_ID)
        & (points_df["requested_context_messages"] <= int(GPT54_APPENDIX_CONTEXT_MAX))
    ].copy()
    gpt54_df = gpt54_df[gpt54_df["category"].isin(GPT54_APPENDIX_CATEGORY_ORDER)]
    if gpt54_df.empty:
        logger.warning(
            "No GPT-5.4 category rows found for requested context <= %d; "
            "skipping category subplot appendix figure.",
            GPT54_APPENDIX_CONTEXT_MAX,
        )
        return None

    available_categories = set(gpt54_df["category"].astype(str))
    ordered_categories = [
        category
        for category in GPT54_APPENDIX_CATEGORY_ORDER
        if category in available_categories
    ]
    if not ordered_categories:
        logger.warning(
            "No GPT-5.4 aggregate categories available; skipping category subplot "
            "appendix figure."
        )
        return None
    return _plot_context_effects_grouped_subplots(
        gpt54_df,
        output_dir,
        config=ContextSubplotConfig(
            group_col="category",
            groups=ordered_categories,
            filename_stem=filename_stem,
            subplot_shape=(5, 1),
            figsize=(8.5, 10.8),
            annotate_windows=False,
        ),
    )


def plot_context_effects_main_combined(
    points_df: pd.DataFrame,
    category_points_df: pd.DataFrame,
    output_dir: Path,
    *,
    filename_stem: str = "context_effect_scatter_main_delusional_discourages_violence",
) -> Optional[Path]:
    """Create the two-panel main context-effects figure for Results.

    Parameters
    ----------
    points_df:
        Code-level aggregated context points.
    category_points_df:
        Category-level aggregated context points.
    output_dir:
        Directory to write figure files.
    filename_stem:
        Output filename stem.

    Returns
    -------
    Optional[Path]
        Output PDF path, or ``None`` when one or both required panels have no
        available data.
    """
    left_df = category_points_df[
        category_points_df["category"] == MAIN_CONTEXT_EFFECT_LEFT_CATEGORY
    ].copy()
    right_df = points_df[
        points_df["annotation_id"] == MAIN_CONTEXT_EFFECT_RIGHT_CODE
    ].copy()
    if left_df.empty or right_df.empty:
        logger.warning(
            "Missing rows for combined context-effects figure "
            "(left_category=%s empty=%s, right_code=%s empty=%s); skipping.",
            MAIN_CONTEXT_EFFECT_LEFT_CATEGORY,
            left_df.empty,
            MAIN_CONTEXT_EFFECT_RIGHT_CODE,
            right_df.empty,
        )
        return None

    output_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(ncols=2, figsize=(8.0, 2.5), sharex=True, sharey=False)
    _plot_model_series_on_axis(
        axes[0],
        left_df,
        annotate_windows=False,
    )
    _plot_model_series_on_axis(
        axes[1],
        right_df,
        annotate_windows=False,
    )

    axes[0].set_title(
        format_metric_label_plain_for_matplotlib(MAIN_CONTEXT_EFFECT_LEFT_CATEGORY),
        fontsize=8,
    )
    axes[1].set_title(
        format_metric_label_plain_for_matplotlib(MAIN_CONTEXT_EFFECT_RIGHT_CODE),
        fontsize=8,
    )
    x_lower, x_upper = _context_xlim_with_padding(
        0.0, float(CONTEXT_EFFECT_X_MAX_MESSAGES)
    )
    for panel_df, ax in zip([left_df, right_df], axes):
        ax.set_xlim(x_lower, x_upper)
        y_min = max(0.0, float(panel_df["prevalence_ci_lower_pct"].min()) - 1.0)
        y_max = min(100.0, float(panel_df["prevalence_ci_upper_pct"].max()) + 1.0)
        if y_max <= y_min:
            y_max = min(100.0, y_min + 1.0)
        ax.set_ylim(y_min, y_max)
        ax.set_xlabel("Context Length (messages)")

    handles_by_label: dict[str, Any] = {}
    for ax in axes:
        handles, labels = ax.get_legend_handles_labels()
        for handle, label in zip(handles, labels):
            label_text = str(label)
            if label_text == "_nolegend_" or label_text in handles_by_label:
                continue
            handles_by_label[label_text] = handle
    if len(handles_by_label) > 1:
        fig.legend(
            list(handles_by_label.values()),
            list(handles_by_label.keys()),
            loc="lower center",
            ncol=min(4, len(handles_by_label)),
            fontsize=7,
            frameon=True,
            bbox_to_anchor=(0.5, -0.01),
        )
    fig.supylabel("Prevalence (%)", x=0.02)
    fig.tight_layout(rect=(0.03, 0.06, 1.0, 1.0))

    pdf_path = output_dir / f"{filename_stem}.pdf"
    png_path = output_dir / f"{filename_stem}.png"
    fig.savefig(pdf_path)
    fig.savefig(png_path)
    plt.close(fig)
    logger.info("Wrote %s and %s", pdf_path, png_path)
    return pdf_path


def plot_context_effects(
    points_df: pd.DataFrame,
    output_dir: Path,
    *,
    filename_stem_prefix: str = "context_effect_scatter",
    annotate_sample_counts: bool = True,
) -> list[Path]:
    """Create scatter plots by code for context effect aggregates.

    Parameters
    ----------
    points_df:
        Aggregated point dataframe.
    output_dir:
        Directory to write figures.
    filename_stem_prefix:
        Filename prefix for generated figures.
    annotate_sample_counts:
        Whether to annotate each point with aggregate sample count ``n``.

    Returns
    -------
    list[Path]
        Paths to generated figure PDFs.
    """
    return _plot_context_effects_grouped(
        points_df,
        output_dir,
        group_col="annotation_id",
        filename_stem_prefix=filename_stem_prefix,
        annotate_sample_counts=annotate_sample_counts,
    )


def plot_category_context_effects(
    points_df: pd.DataFrame,
    output_dir: Path,
    *,
    filename_stem_prefix: str = "context_effect_scatter_category",
    annotate_sample_counts: bool = True,
) -> list[Path]:
    """Create scatter plots by category for context effect aggregates.

    Parameters
    ----------
    points_df:
        Aggregated category-level point dataframe.
    output_dir:
        Directory to write figures.
    filename_stem_prefix:
        Filename prefix for generated figures.
    annotate_sample_counts:
        Whether to annotate each point with aggregate sample count ``n``.

    Returns
    -------
    list[Path]
        Paths to generated figure PDFs.
    """
    return _plot_context_effects_grouped(
        points_df,
        output_dir,
        group_col="category",
        filename_stem_prefix=filename_stem_prefix,
        annotate_sample_counts=annotate_sample_counts,
    )


def _plot_context_effects_grouped(
    points_df: pd.DataFrame,
    output_dir: Path,
    *,
    group_col: str,
    filename_stem_prefix: str,
    annotate_sample_counts: bool,
) -> list[Path]:
    """Plot context effect scatters grouped by one column.

    Parameters
    ----------
    points_df:
        Aggregated point dataframe.
    output_dir:
        Directory to write figures.
    group_col:
        Column used to split plots (for example, code or category).
    filename_stem_prefix:
        Filename prefix for generated figures.
    annotate_sample_counts:
        Whether to annotate each point with aggregate sample count ``n``.

    Returns
    -------
    list[Path]
        Paths to generated figure PDFs.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    figure_paths: list[Path] = []
    x_lower, x_upper = _context_xlim_with_padding(
        0.0, float(CONTEXT_EFFECT_X_MAX_MESSAGES)
    )

    for group_value, group_df in points_df.groupby(group_col, dropna=False):
        fig, ax = plt.subplots(
            figsize=(
                CONTEXT_EFFECT_FIGURE_WIDTH_INCHES,
                CONTEXT_EFFECT_FIGURE_HEIGHT_INCHES,
            )
        )
        _plot_model_series_on_axis(
            ax,
            group_df,
            annotate_windows=False,
        )
        ax.set_title(
            format_metric_label_plain_for_matplotlib(str(group_value)),
            fontsize=8,
        )
        if annotate_sample_counts:
            for _, row in group_df.sort_values("effective_context_length").iterrows():
                x_pos = float(row["effective_context_length"])
                if x_pos >= CONTEXT_EFFECT_X_MAX_MESSAGES - 10:
                    x_offset = -4
                    horizontal_alignment = "right"
                else:
                    x_offset = 4
                    horizontal_alignment = "left"
                ax.annotate(
                    # Display sample count n for each aggregate point.
                    f"{int(row['n'])}",
                    (x_pos, row["prevalence_pct"]),
                    xytext=(x_offset, 4),
                    textcoords="offset points",
                    fontsize=7,
                    ha=horizontal_alignment,
                    clip_on=False,
                )

        ax.set_xlabel("Context Length (messages)")
        ax.set_ylabel(prevalence_axis_label_for_matplotlib(str(group_value)))
        ax.set_xlim(x_lower, x_upper)
        ax.legend(fontsize=7, frameon=True)
        fig.tight_layout()

        stem = _safe_name(f"{filename_stem_prefix}_{group_value}")
        pdf_path = output_dir / f"{stem}.pdf"
        png_path = output_dir / f"{stem}.png"
        fig.savefig(pdf_path)
        fig.savefig(png_path)
        plt.close(fig)
        figure_paths.append(pdf_path)
        logger.info("Wrote %s and %s", pdf_path, png_path)

    return figure_paths


def _build_arg_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser for context-effect analysis.

    Returns
    -------
    argparse.ArgumentParser
        Configured parser instance.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Compute requested-vs-effective context effects from logs-context evals."
        )
    )
    parser.add_argument(
        "--logs-dir",
        type=Path,
        default=DEFAULT_LOGS_DIR,
        help="Directory containing context eval .eval files (default: logs-context).",
    )
    parser.add_argument(
        "--baseline-logs-dir",
        type=Path,
        default=DEFAULT_BASELINE_LOGS_DIR,
        help=(
            "Directory containing baseline eval .eval files used for "
            "requested_context_messages=0 points (default: logs)."
        ),
    )
    parser.add_argument(
        "--excluded-participants",
        type=str,
        default=None,
        help=(
            "Comma-separated participant IDs to exclude. If omitted, uses "
            "LLM_DELUSIONS_EXCLUDED_PARTICIPANTS (default: include all). "
            "Use an empty string to disable exclusions."
        ),
    )
    parser.add_argument(
        "--codes",
        type=str,
        default="",
        help=(
            "Optional comma-separated annotation IDs to include. "
            "If omitted, validates-* codes are excluded by default."
        ),
    )
    parser.add_argument(
        "--include-validates-codes",
        action="store_true",
        help=(
            "Include validates-* codes when --codes is not specified. "
            "Explicit --codes filters still take precedence."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_FIGURES_DIR,
        help="Output directory for figure files (default: analysis/figures).",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help="Output directory for CSV files (default: analysis/data/context_effects).",
    )
    parser.add_argument(
        "--rows-cache",
        type=Path,
        default=DEFAULT_ROWS_CACHE,
        help="Parquet cache for parsed context rows.",
    )
    parser.add_argument(
        "--log-metadata-cache",
        type=Path,
        default=DEFAULT_LOG_METADATA_CACHE,
        help="JSON cache for per-log manifest/sample metadata.",
    )
    parser.add_argument(
        "--sample-rows-cache-dir",
        type=Path,
        default=DEFAULT_SAMPLE_ROWS_CACHE_DIR,
        help="Directory cache for parsed rows per log revision.",
    )
    parser.add_argument(
        "--min-effective-context-messages",
        type=int,
        default=0,
        help=(
            "Drop rows below this effective context length before aggregation "
            "(default: 0)."
        ),
    )
    parser.add_argument(
        "--min-effective-fraction-of-requested",
        type=float,
        default=1.0,
        help=(
            "Drop rows where effective context is below this fraction of requested "
            "context (default: 1.0)."
        ),
    )
    parser.add_argument(
        "--uniform-sample-upper-limit",
        type=int,
        default=None,
        help=(
            "Optional requested-context upper limit for an additional "
            "uniform-sample analysis. When set, a second set of CSV/figures is "
            "generated using only requested context points <= this limit and "
            "only window IDs present across all kept requested points > 0 "
            "(per model/reasoning/code)."
        ),
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable reading/writing context row cache parquet.",
    )
    parser.add_argument(
        "--figures-only",
        action="store_true",
        help=(
            "Regenerate figures from existing CSV aggregates in --data-dir "
            "without recomputing rows or rewriting CSV files."
        ),
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose logging.",
    )
    return parser


def _validate_cli_args(args: argparse.Namespace) -> None:
    """Validate CLI argument values.

    Parameters
    ----------
    args:
        Parsed CLI arguments.
    """
    if args.min_effective_context_messages < 0:
        raise ValueError("--min-effective-context-messages must be >= 0")
    if not 0.0 <= args.min_effective_fraction_of_requested <= 1.0:
        raise ValueError(
            "--min-effective-fraction-of-requested must be between 0 and 1."
        )
    if (
        args.uniform_sample_upper_limit is not None
        and args.uniform_sample_upper_limit < 0
    ):
        raise ValueError("--uniform-sample-upper-limit must be >= 0.")


def _write_standard_outputs(
    points_df: pd.DataFrame,
    category_points_df: pd.DataFrame,
    *,
    figures_dir: Path,
    data_dir: Path,
) -> None:
    """Write standard CSV and figure outputs.

    Parameters
    ----------
    points_df:
        Code-level aggregated points.
    category_points_df:
        Category-level aggregated points.
    figures_dir:
        Output directory for figure files.
    data_dir:
        Output directory for CSV files.
    """
    ensure_output_dirs(figures_dir, data_dir)
    csv_path = data_dir / "context_effect_points.csv"
    points_df.to_csv(csv_path, index=False)
    logger.info("Wrote %s", csv_path)

    category_csv_path = data_dir / "context_effect_points_by_category.csv"
    category_points_df.to_csv(category_csv_path, index=False)
    logger.info("Wrote %s", category_csv_path)

    figure_paths = plot_context_effects(points_df, figures_dir)
    category_figure_paths = plot_category_context_effects(
        category_points_df, figures_dir
    )
    appendix_code_subplots = plot_gpt54_context_effect_code_subplots(
        points_df,
        figures_dir,
    )
    appendix_category_subplots = plot_gpt54_context_effect_category_subplots(
        category_points_df,
        figures_dir,
    )
    main_combined_plot = plot_context_effects_main_combined(
        points_df,
        category_points_df,
        figures_dir,
    )
    appendix_plot_count = sum(
        path is not None
        for path in [appendix_code_subplots, appendix_category_subplots]
    )
    main_plot_count = int(main_combined_plot is not None)
    logger.info(
        "Done. Generated %d code figure(s) in %s, %d category figure(s) in %s, "
        "%d appendix subplot figure(s), %d main combined subplot figure(s), "
        "and two CSVs in %s",
        len(figure_paths),
        figures_dir,
        len(category_figure_paths),
        figures_dir,
        appendix_plot_count,
        main_plot_count,
        data_dir,
    )


def _load_aggregated_points_csv(csv_path: Path, *, label: str) -> pd.DataFrame:
    """Load one aggregated points CSV used for context-effect plotting.

    Parameters
    ----------
    csv_path:
        CSV path to load.
    label:
        Human-readable label used in error messages.

    Returns
    -------
    pd.DataFrame
        Loaded dataframe.
    """
    if not csv_path.exists():
        raise FileNotFoundError(
            f"{label} CSV not found: {csv_path}. "
            "Run analysis/compute_context_effects.py without --figures-only first."
        )
    return pd.read_csv(csv_path)


def _regenerate_standard_figures_only(
    *,
    figures_dir: Path,
    data_dir: Path,
) -> None:
    """Regenerate standard context-effect figures from existing CSV aggregates.

    Parameters
    ----------
    figures_dir:
        Output directory for figure files.
    data_dir:
        Directory containing CSV aggregate files.
    """
    ensure_output_dirs(figures_dir, data_dir)
    points_df = _load_aggregated_points_csv(
        data_dir / "context_effect_points.csv",
        label="Code-level context effects",
    )
    category_points_df = _load_aggregated_points_csv(
        data_dir / "context_effect_points_by_category.csv",
        label="Category-level context effects",
    )
    figure_paths = plot_context_effects(points_df, figures_dir)
    category_figure_paths = plot_category_context_effects(
        category_points_df,
        figures_dir,
    )
    appendix_code_subplots = plot_gpt54_context_effect_code_subplots(
        points_df,
        figures_dir,
    )
    appendix_category_subplots = plot_gpt54_context_effect_category_subplots(
        category_points_df,
        figures_dir,
    )
    main_combined_plot = plot_context_effects_main_combined(
        points_df,
        category_points_df,
        figures_dir,
    )
    appendix_plot_count = sum(
        path is not None
        for path in [appendix_code_subplots, appendix_category_subplots]
    )
    main_plot_count = int(main_combined_plot is not None)
    logger.info(
        "Figures-only mode: regenerated %d code figure(s), %d category figure(s), "
        "%d appendix subplot figure(s), and %d main combined subplot figure(s) in %s.",
        len(figure_paths),
        len(category_figure_paths),
        appendix_plot_count,
        main_plot_count,
        figures_dir,
    )


def _regenerate_uniform_sample_figures_only(args: argparse.Namespace) -> None:
    """Regenerate optional uniform-sample figures from existing CSV aggregates.

    Parameters
    ----------
    args:
        Parsed CLI arguments.
    """
    if args.uniform_sample_upper_limit is None:
        return

    uniform_suffix = f"uniform_upto_{int(args.uniform_sample_upper_limit)}"
    uniform_points_df = _load_aggregated_points_csv(
        args.data_dir / f"context_effect_points_{uniform_suffix}.csv",
        label=f"Uniform code-level context effects ({uniform_suffix})",
    )
    uniform_category_points_df = _load_aggregated_points_csv(
        args.data_dir / f"context_effect_points_by_category_{uniform_suffix}.csv",
        label=f"Uniform category-level context effects ({uniform_suffix})",
    )

    uniform_figure_paths = plot_context_effects(
        uniform_points_df,
        args.output_dir,
        filename_stem_prefix=f"context_effect_scatter_{uniform_suffix}",
        annotate_sample_counts=False,
    )
    uniform_category_figure_paths = plot_category_context_effects(
        uniform_category_points_df,
        args.output_dir,
        filename_stem_prefix=f"context_effect_scatter_category_{uniform_suffix}",
        annotate_sample_counts=False,
    )
    uniform_appendix_code_subplots = plot_gpt54_context_effect_code_subplots(
        uniform_points_df,
        args.output_dir,
        filename_stem=f"context_effect_subplots_{uniform_suffix}_gpt54_codes",
    )
    uniform_appendix_category_subplots = plot_gpt54_context_effect_category_subplots(
        uniform_category_points_df,
        args.output_dir,
        filename_stem=f"context_effect_subplots_{uniform_suffix}_gpt54_categories",
    )
    uniform_appendix_plot_count = sum(
        path is not None
        for path in [
            uniform_appendix_code_subplots,
            uniform_appendix_category_subplots,
        ]
    )
    logger.info(
        "Figures-only uniform mode (%s): regenerated %d code figure(s), "
        "%d category figure(s), and %d appendix subplot figure(s) in %s.",
        uniform_suffix,
        len(uniform_figure_paths),
        len(uniform_category_figure_paths),
        uniform_appendix_plot_count,
        args.output_dir,
    )


def _run_uniform_sample_analysis(
    rows_df: pd.DataFrame,
    codes: list[str],
    args: argparse.Namespace,
) -> None:
    """Write optional uniform-sample analysis outputs.

    Parameters
    ----------
    rows_df:
        Row-level context effect data.
    codes:
        Optional normalized code filters.
    args:
        Parsed CLI arguments.
    """
    if args.uniform_sample_upper_limit is None:
        return

    uniform_rows_df = _filter_context_rows(
        rows_df,
        codes,
        include_validates_codes=args.include_validates_codes,
        min_effective_context_messages=args.min_effective_context_messages,
        min_effective_fraction_of_requested=args.min_effective_fraction_of_requested,
    )
    uniform_rows_df = _restrict_to_uniform_sample_set(
        uniform_rows_df,
        requested_context_upper_limit=args.uniform_sample_upper_limit,
    )

    uniform_points_df = _aggregate_grouped_context_effects(
        uniform_rows_df,
        [
            "model",
            "model_label",
            "reasoning_effort",
            "requested_context_messages",
            "annotation_id",
            "code_short",
            "category",
        ],
    )
    uniform_category_points_df = _aggregate_grouped_context_effects(
        uniform_rows_df,
        [
            "model",
            "model_label",
            "reasoning_effort",
            "requested_context_messages",
            "category",
        ],
    )

    uniform_suffix = f"uniform_upto_{int(args.uniform_sample_upper_limit)}"
    ensure_output_dirs(args.output_dir, args.data_dir)

    uniform_csv_path = args.data_dir / f"context_effect_points_{uniform_suffix}.csv"
    uniform_points_df.to_csv(uniform_csv_path, index=False)
    logger.info("Wrote %s", uniform_csv_path)

    uniform_category_csv_path = (
        args.data_dir / f"context_effect_points_by_category_{uniform_suffix}.csv"
    )
    uniform_category_points_df.to_csv(uniform_category_csv_path, index=False)
    logger.info("Wrote %s", uniform_category_csv_path)

    uniform_figure_paths = plot_context_effects(
        uniform_points_df,
        args.output_dir,
        filename_stem_prefix=f"context_effect_scatter_{uniform_suffix}",
        annotate_sample_counts=False,
    )
    uniform_category_figure_paths = plot_category_context_effects(
        uniform_category_points_df,
        args.output_dir,
        filename_stem_prefix=f"context_effect_scatter_category_{uniform_suffix}",
        annotate_sample_counts=False,
    )
    uniform_appendix_code_subplots = plot_gpt54_context_effect_code_subplots(
        uniform_points_df,
        args.output_dir,
        filename_stem=f"context_effect_subplots_{uniform_suffix}_gpt54_codes",
    )
    uniform_appendix_category_subplots = plot_gpt54_context_effect_category_subplots(
        uniform_category_points_df,
        args.output_dir,
        filename_stem=f"context_effect_subplots_{uniform_suffix}_gpt54_categories",
    )
    uniform_appendix_plot_count = sum(
        path is not None
        for path in [
            uniform_appendix_code_subplots,
            uniform_appendix_category_subplots,
        ]
    )
    logger.info(
        "Uniform sample analysis done (upper limit %d): "
        "%d code figure(s), %d category figure(s), %d appendix subplot figure(s) "
        "in %s, and two CSVs in %s",
        args.uniform_sample_upper_limit,
        len(uniform_figure_paths),
        len(uniform_category_figure_paths),
        uniform_appendix_plot_count,
        args.output_dir,
        args.data_dir,
    )


def main() -> None:
    """Entry point for context effect analysis."""
    args = _build_arg_parser().parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )
    _validate_cli_args(args)

    if args.figures_only:
        _regenerate_standard_figures_only(
            figures_dir=args.output_dir,
            data_dir=args.data_dir,
        )
        _regenerate_uniform_sample_figures_only(args)
        return

    codes = [entry.strip() for entry in args.codes.split(",") if entry.strip()]
    _warn_if_repo_local_sensitive_paths(
        args.logs_dir,
        args.rows_cache,
        args.log_metadata_cache,
        args.sample_rows_cache_dir,
    )
    cache_paths = ContextEffectCachePaths(
        rows_cache=args.rows_cache,
        log_metadata_cache=args.log_metadata_cache,
        sample_rows_cache_dir=args.sample_rows_cache_dir,
    )
    _, excluded_window_ids = resolve_context_excluded_window_ids(
        args.excluded_participants
    )

    rows_df = load_context_effect_data(
        args.logs_dir,
        baseline_logs_dir=args.baseline_logs_dir,
        cache_paths=cache_paths,
        use_cache=not args.no_cache,
        excluded_window_ids=excluded_window_ids,
    )
    participant_rows_df = attach_participant_ids(rows_df)
    if participant_rows_df.empty and not rows_df.empty:
        logger.warning(
            "Proceeding without participant/conversation mappings; using "
            "row-level bootstrap intervals for context-effect summaries."
        )
    else:
        rows_df = participant_rows_df
    points_df = aggregate_context_effects(
        rows_df,
        codes,
        include_validates_codes=args.include_validates_codes,
        min_effective_context_messages=args.min_effective_context_messages,
        min_effective_fraction_of_requested=args.min_effective_fraction_of_requested,
    )
    category_points_df = aggregate_context_effects_by_category(
        rows_df,
        codes,
        include_validates_codes=args.include_validates_codes,
        min_effective_context_messages=args.min_effective_context_messages,
        min_effective_fraction_of_requested=args.min_effective_fraction_of_requested,
    )
    _write_standard_outputs(
        points_df=points_df,
        category_points_df=category_points_df,
        figures_dir=args.output_dir,
        data_dir=args.data_dir,
    )
    _run_uniform_sample_analysis(rows_df=rows_df, codes=codes, args=args)


if __name__ == "__main__":
    main()
