"""Participant metadata and aggregation helpers for analysis tables.

This module centralizes:

1. Resolving a local windows parquet suitable for participant mapping.
2. Attaching participant and conversation identifiers to row-level data.
3. Aggregating row-level values to participant- and conversation-level
   summaries.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from llm_delusion_eval.participant_exclusions import build_window_id_to_metadata_map
from llm_delusion_eval.paths import _EVALS_REPO_ROOT, DEFAULT_WINDOWS_PATH, resolve_path

logger = logging.getLogger(__name__)

_LOCAL_ITEMS_PATH = (
    _EVALS_REPO_ROOT.parent / "llm-delusions" / "subsets" / "items.parquet"
)


def resolve_participant_windows_path(explicit: str = "") -> Path:
    """Resolve a local windows parquet path for participant mapping.

    Parameters
    ----------
    explicit:
        Optional explicit path override.

    Returns
    -------
    Path
        Local parquet path used to derive ``window_id`` metadata.
    """
    if explicit:
        return Path(
            resolve_path(
                "LLM_DELUSIONS_WINDOWS_PATH",
                DEFAULT_WINDOWS_PATH,
                explicit=explicit,
                require_local=True,
            )
        )
    if _LOCAL_ITEMS_PATH.exists():
        return _LOCAL_ITEMS_PATH
    resolved = Path(
        resolve_path(
            "LLM_DELUSIONS_WINDOWS_PATH",
            DEFAULT_WINDOWS_PATH,
            explicit=explicit,
            require_local=True,
        )
    )
    return resolved


def attach_participant_ids(
    df: pd.DataFrame,
    *,
    windows_path: str | Path | None = None,
    excluded_participants: set[str] | None = None,
) -> pd.DataFrame:
    """Attach participant and conversation IDs using ``window_id`` metadata.

    Parameters
    ----------
    df:
        Row-level dataframe containing a ``window_id`` column.
    windows_path:
        Optional local windows parquet override used to build the mapping.
    excluded_participants:
        Optional participant IDs to drop after mapping.

    Returns
    -------
    pd.DataFrame
        Input rows with normalized ``participant`` and ``conversation_id``
        columns.
    """
    resolved_windows_path = resolve_participant_windows_path(str(windows_path or ""))
    metadata_map = build_window_id_to_metadata_map(resolved_windows_path)
    mapping = {
        window_id: metadata.get("participant", "")
        for window_id, metadata in metadata_map.items()
    }
    if not mapping:
        logger.warning(
            "Could not build any window-to-metadata mappings from %s.",
            resolved_windows_path,
        )
        mapped = df.iloc[0:0].copy()
        mapped["participant"] = pd.Series(dtype="object")
        mapped["conversation_id"] = pd.Series(dtype="object")
        return mapped

    mapped = df.copy()
    mapped["window_id"] = mapped["window_id"].astype(str).str.strip()
    mapped["participant"] = mapped["window_id"].map(mapping)
    mapped["participant"] = mapped["participant"].fillna("").astype(str).str.strip()
    mapped_conversation = mapped["window_id"].map(
        {
            window_id: metadata.get("conversation_id", "")
            for window_id, metadata in metadata_map.items()
        }
    )
    if "conversation_id" in mapped.columns:
        existing = mapped["conversation_id"].fillna("").astype(str).str.strip()
        mapped["conversation_id"] = existing.mask(existing.eq(""), mapped_conversation)
    else:
        mapped["conversation_id"] = mapped_conversation
    mapped["conversation_id"] = (
        mapped["conversation_id"].fillna("").astype(str).str.strip()
    )

    missing_participant = mapped["participant"].eq("")
    if missing_participant.any():
        logger.warning(
            "Dropping %d row(s) without participant mappings.",
            int(missing_participant.sum()),
        )
        mapped = mapped[~missing_participant].copy()

    if excluded_participants:
        mapped = mapped[~mapped["participant"].isin(excluded_participants)].copy()
    return mapped


def aggregate_participant_conversation_value_sums(
    df: pd.DataFrame,
    group_cols: list[str],
    *,
    value_col: str,
    participant_col: str = "participant",
    conversation_col: str = "conversation_id",
) -> pd.DataFrame:
    """Aggregate row-level values to participant-conversation summaries.

    Parameters
    ----------
    df:
        Row-level dataframe with participant and conversation IDs.
    group_cols:
        Columns defining the analysis cell.
    value_col:
        Numeric value column to summarize.
    participant_col:
        Column containing participant identifiers.
    conversation_col:
        Column containing conversation identifiers.

    Returns
    -------
    pd.DataFrame
        One row per ``participant``, ``conversation_id``, and analysis cell
        with ``value_sum`` and ``value_count`` columns.
    """
    required_cols = [participant_col, conversation_col, *group_cols, value_col]
    valid = df[required_cols].copy()
    valid[participant_col] = valid[participant_col].astype(str).str.strip()
    valid[conversation_col] = valid[conversation_col].astype(str).str.strip()
    valid = valid[valid[participant_col] != ""].copy()
    valid = valid[valid[conversation_col] != ""].copy()
    valid[value_col] = pd.to_numeric(valid[value_col], errors="coerce")
    valid = valid[valid[value_col].notna()].copy()
    if valid.empty:
        return pd.DataFrame(
            columns=[
                participant_col,
                conversation_col,
                *group_cols,
                "value_sum",
                "value_count",
            ]
        )

    grouped = (
        valid.groupby([participant_col, conversation_col, *group_cols], sort=False)[
            value_col
        ]
        .agg(["sum", "count"])
        .reset_index()
        .rename(columns={"sum": "value_sum", "count": "value_count"})
    )
    grouped["value_sum"] = grouped["value_sum"].astype(float)
    grouped["value_count"] = grouped["value_count"].astype(float)
    return grouped


def aggregate_participant_value_sums(
    df: pd.DataFrame,
    group_cols: list[str],
    *,
    value_col: str,
    participant_col: str = "participant",
) -> pd.DataFrame:
    """Aggregate row-level values to participant-level sums and counts.

    Parameters
    ----------
    df:
        Row-level dataframe with participant IDs.
    group_cols:
        Columns defining the analysis cell.
    value_col:
        Numeric value column to summarize.
    participant_col:
        Column containing participant identifiers.

    Returns
    -------
    pd.DataFrame
        One row per ``participant`` and analysis cell with ``value_sum`` and
        ``value_count`` columns.
    """
    required_cols = [participant_col, *group_cols, value_col]
    valid = df[required_cols].copy()
    valid[participant_col] = valid[participant_col].astype(str).str.strip()
    valid = valid[valid[participant_col] != ""].copy()
    valid[value_col] = pd.to_numeric(valid[value_col], errors="coerce")
    valid = valid[valid[value_col].notna()].copy()
    if valid.empty:
        return pd.DataFrame(
            columns=[participant_col, *group_cols, "value_sum", "value_count"]
        )

    grouped = (
        valid.groupby([participant_col, *group_cols], sort=False)[value_col]
        .agg(["sum", "count"])
        .reset_index()
        .rename(columns={"sum": "value_sum", "count": "value_count"})
    )
    grouped["value_sum"] = grouped["value_sum"].astype(float)
    grouped["value_count"] = grouped["value_count"].astype(float)
    return grouped


def build_conversation_value_lookup(
    aggregated: pd.DataFrame,
    group_cols: list[str],
    *,
    sum_col: str = "value_sum",
    count_col: str = "value_count",
) -> dict[tuple[Any, ...], pd.DataFrame]:
    """Build a grouped lookup of participant-conversation summaries.

    Parameters
    ----------
    aggregated:
        Output from :func:`aggregate_participant_conversation_value_sums`.
    group_cols:
        Grouping columns used in the aggregation.
    sum_col:
        Column containing conversation-level value sums.
    count_col:
        Column containing conversation-level row counts.

    Returns
    -------
    dict[tuple[Any, ...], pd.DataFrame]
        Mapping from grouped-cell key to the conversation-level summary rows
        for that cell.
    """
    if aggregated.empty:
        return {}

    base_columns = ["participant", "conversation_id", sum_col, count_col]
    if not group_cols:
        return {(): aggregated[base_columns].copy().reset_index(drop=True)}

    lookup: dict[tuple[Any, ...], pd.DataFrame] = {}
    for raw_key, group in aggregated.groupby(group_cols, sort=False):
        key = raw_key if isinstance(raw_key, tuple) else (raw_key,)
        lookup[key] = group[base_columns].copy().reset_index(drop=True)
    return lookup


def build_aligned_value_lookup(
    aggregated: pd.DataFrame,
    group_cols: list[str],
    *,
    participant_col: str = "participant",
    sum_col: str = "value_sum",
    count_col: str = "value_count",
) -> tuple[np.ndarray, dict[tuple[Any, ...], tuple[np.ndarray, np.ndarray]]]:
    """Align participant-level sums and counts for each grouped cell.

    Parameters
    ----------
    aggregated:
        Output from :func:`aggregate_participant_value_sums`.
    group_cols:
        Grouping columns used in the aggregation.
    participant_col:
        Column containing participant identifiers.
    sum_col:
        Column containing participant-level value sums.
    count_col:
        Column containing participant-level row counts.

    Returns
    -------
    tuple[np.ndarray, dict[tuple[Any, ...], tuple[np.ndarray, np.ndarray]]]
        Sorted participant IDs and a lookup from grouped cell key to aligned
        ``(value_sum_array, value_count_array)``.
    """
    if aggregated.empty:
        return np.array([], dtype=object), {}

    participants = np.array(
        sorted(aggregated[participant_col].astype(str).unique()),
        dtype=object,
    )
    participant_index = pd.Index(participants, name=participant_col)
    lookup: dict[tuple[Any, ...], tuple[np.ndarray, np.ndarray]] = {}

    for raw_key, group in aggregated.groupby(group_cols, sort=False):
        key = raw_key if isinstance(raw_key, tuple) else (raw_key,)
        indexed = (
            group.set_index(participant_col)[[sum_col, count_col]]
            .reindex(participant_index, fill_value=0.0)
            .sort_index()
        )
        lookup[key] = (
            indexed[sum_col].to_numpy(dtype=float),
            indexed[count_col].to_numpy(dtype=float),
        )
    return participants, lookup
