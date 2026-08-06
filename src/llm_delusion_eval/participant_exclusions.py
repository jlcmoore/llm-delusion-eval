"""Shared participant-exclusion helpers for report and analysis scripts.

This module centralizes:
1. Which participant IDs are excluded by default.
2. How exclusion settings are read from environment variables.
3. How eval ``window_id`` values map back to participant/conversation IDs.
4. Optional overrides for the sanitized items parquet path.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Any

import pandas as pd

from llm_delusion_eval.paths import DEFAULT_WINDOWS_PATH, resolve_path
from llm_delusion_eval.window_ids import build_eval_subset_id_from_row, build_window_id

EXCLUDED_PARTICIPANTS_ENV = "LLM_DELUSIONS_EXCLUDED_PARTICIPANTS"
DEFAULT_EXCLUDED_PARTICIPANTS: tuple[str, ...] = ()

ITEMS_SANITIZED_PATH_ENV = "LLM_DELUSIONS_ITEMS_SANITIZED_PATH"
DEFAULT_ITEMS_SANITIZED_PATH = DEFAULT_WINDOWS_PATH


def normalize_participant(value: Any) -> str:
    """Return a normalized participant identifier, or an empty string.

    Parameters
    ----------
    value:
        Raw participant value from parquet data.

    Returns
    -------
    str
        Normalized participant identifier, or ``""`` if missing.
    """
    if value is None:
        return ""
    text = str(value).strip()
    if not text or text in {"<NA>", "nan", "NaN", "None"}:
        return ""
    return text


def resolve_excluded_participants(raw_value: str | None = None) -> set[str]:
    """Resolve participant IDs excluded from analysis/report aggregation.

    Parameters
    ----------
    raw_value:
        Optional comma-separated participant list. When ``None``, this function
        reads ``LLM_DELUSIONS_EXCLUDED_PARTICIPANTS`` from the environment.
        If both are unset, default exclusions are returned (none by default).

    Returns
    -------
    set[str]
        Set of participant IDs to exclude.
    """
    if raw_value is None:
        raw_value = os.getenv(EXCLUDED_PARTICIPANTS_ENV)
    if raw_value is None:
        return set(DEFAULT_EXCLUDED_PARTICIPANTS)
    return {token.strip() for token in raw_value.split(",") if token.strip()}


def build_row_hash_eval_subset_id(row_index: int) -> str:
    """Return a deterministic row-hash eval subset ID.

    Parameters
    ----------
    row_index:
        Positional index in selected items rows.

    Returns
    -------
    str
        Short deterministic row-hash key.
    """
    return hashlib.sha256(f"row:{row_index}".encode("utf-8")).hexdigest()[:16]


def _normalize_metadata_text(value: Any) -> str:
    """Return a normalized non-empty identifier string, or ``""``.

    Parameters
    ----------
    value:
        Raw identifier value from parquet data.

    Returns
    -------
    str
        Normalized identifier string, or ``""`` if missing.
    """
    if value is None:
        return ""
    text = str(value).strip()
    if not text or text in {"<NA>", "nan", "NaN", "None"}:
        return ""
    return text


def _add_window_metadata(
    mapping: dict[str, dict[str, str]],
    rows: pd.DataFrame,
) -> None:
    """Populate ``mapping`` with window metadata from parquet rows.

    Parameters
    ----------
    mapping:
        Mutable mapping from window ID to metadata.
    rows:
        Dataframe whose rows may include participant, conversation, and window
        ID fields.
    """
    for row in rows.itertuples():
        participant = normalize_participant(getattr(row, "participant", ""))
        if not participant:
            continue
        conversation_id = _normalize_metadata_text(getattr(row, "conversation_id", ""))
        window_ids = {
            str(build_window_id(row)).strip(),
            str(build_eval_subset_id_from_row(row)).strip(),
        }
        for window_id in window_ids:
            if window_id:
                mapping[window_id] = {
                    "participant": participant,
                    "conversation_id": conversation_id,
                }


def _load_sidecar_items_dataframe(windows_path_obj: Path) -> pd.DataFrame | None:
    """Load sidecar ``items.parquet`` for local windows paths.

    Parameters
    ----------
    windows_path_obj:
        Local windows parquet path.

    Returns
    -------
    pd.DataFrame | None
        Sidecar items dataframe when available, otherwise ``None``.
    """
    if not windows_path_obj.exists():
        return None
    items_path = windows_path_obj.with_name("items.parquet")
    if not items_path.exists():
        return None
    return pd.read_parquet(items_path)


def build_window_id_to_metadata_map(
    windows_path: str | Path,
) -> dict[str, dict[str, str]]:
    """Build a deterministic mapping from ``window_id`` to window metadata.

    Parameters
    ----------
    windows_path:
        Path to windows parquet (typically items_sanitized or items parquet).

    Returns
    -------
    dict[str, dict[str, str]]
        Mapping from eval window IDs to metadata with ``participant`` and
        ``conversation_id`` keys.
    """
    mapping: dict[str, dict[str, str]] = {}
    windows_path_text = resolve_path(
        "LLM_DELUSIONS_WINDOWS_PATH",
        DEFAULT_WINDOWS_PATH,
        explicit=str(windows_path),
        require_local=True,
    )
    windows_path_obj = Path(windows_path_text)

    if not windows_path_obj.exists():
        return mapping

    windows_df = pd.read_parquet(windows_path_text)
    _add_window_metadata(mapping, windows_df)

    # Sidecar items.parquet joins only work for local filesystem paths.
    items_df = _load_sidecar_items_dataframe(windows_path_obj)
    if items_df is None:
        return mapping

    _add_window_metadata(mapping, items_df)

    required_columns = {"meets_code", "selected_for_eval", "participant"}
    if not required_columns.issubset(items_df.columns):
        return mapping

    selected_items = items_df[
        items_df["meets_code"].eq(True) & items_df["selected_for_eval"].eq(True)
    ]
    for row_index, row in enumerate(selected_items.itertuples()):
        participant = normalize_participant(getattr(row, "participant", ""))
        if not participant:
            continue
        mapping[build_row_hash_eval_subset_id(row_index)] = {
            "participant": participant,
            "conversation_id": _normalize_metadata_text(
                getattr(row, "conversation_id", "")
            ),
        }

    return mapping


def build_window_id_to_participant_map(windows_path: str | Path) -> dict[str, str]:
    """Build a deterministic mapping from ``window_id`` to participant.

    Parameters
    ----------
    windows_path:
        Path to windows parquet (typically items_sanitized or items parquet).

    Returns
    -------
    dict[str, str]
        Mapping from eval window IDs to participant IDs.
    """
    metadata_map = build_window_id_to_metadata_map(windows_path)
    mapping: dict[str, str] = {}
    for window_id, metadata in metadata_map.items():
        participant = normalize_participant(metadata.get("participant", ""))
        if participant:
            mapping[window_id] = participant

    return mapping


def resolve_excluded_window_ids(
    windows_path: str | Path,
    *,
    raw_participants: str | None = None,
) -> tuple[set[str], set[str]]:
    """Resolve excluded participants and matching window IDs.

    Parameters
    ----------
    windows_path:
        Path used to build the ``window_id -> participant`` map.
    raw_participants:
        Optional comma-separated participant list. ``None`` means use
        ``LLM_DELUSIONS_EXCLUDED_PARTICIPANTS`` with default behavior.

    Returns
    -------
    tuple[set[str], set[str]]
        ``(excluded_participants, excluded_window_ids)``.
    """
    excluded_participants = resolve_excluded_participants(raw_participants)
    if not excluded_participants:
        return set(), set()

    window_to_participant = build_window_id_to_participant_map(windows_path)
    excluded_window_ids = {
        window_id
        for window_id, participant in window_to_participant.items()
        if participant in excluded_participants
    }
    return excluded_participants, excluded_window_ids


def resolve_items_sanitized_path(explicit: str = "") -> Path:
    """Resolve items_sanitized parquet path from env override or default.

    Parameters
    ----------
    explicit:
        Optional explicit path argument that overrides environment/default.

    Returns
    -------
    Path
        Absolute path to items_sanitized parquet.
    """
    return Path(
        resolve_path(
            ITEMS_SANITIZED_PATH_ENV,
            DEFAULT_ITEMS_SANITIZED_PATH,
            explicit=explicit,
            require_local=True,
        )
    )
