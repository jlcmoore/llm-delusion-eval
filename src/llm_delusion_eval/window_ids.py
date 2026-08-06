"""Deterministic window identifier helpers used by eval and analysis code."""

from __future__ import annotations

import hashlib
from typing import Any


def build_synthetic_rel_path(row: Any) -> str:
    """Build a JSON-style relative path from a row-like object.

    Parameters
    ----------
    row:
        Row-like object with ``label``, ``participant``, ``filename_hash``,
        ``conversation_id``, and ``start_message_index`` fields.

    Returns
    -------
    str
        Relative path string used in subset exports.
    """
    chat_idx_int = int(str(row.conversation_id).replace("chat_", ""))
    filename = (
        f"{row.participant}_{row.filename_hash}_chat{chat_idx_int}"
        f"_win{row.start_message_index}.json"
    )
    return f"{row.label}/{filename}"


def build_eval_subset_id_from_subset_rel_path(subset_rel_path: str) -> str:
    """Return a stable anonymized eval ID from a subset relative path.

    Parameters
    ----------
    subset_rel_path:
        Subset path string.

    Returns
    -------
    str
        Deterministic 16-character hex identifier.
    """
    normalized = str(subset_rel_path or "").strip()
    if not normalized:
        return ""
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:16]


def build_eval_subset_id_from_row(row: Any) -> str:
    """Return a stable anonymized eval ID for a row-like object.

    Parameters
    ----------
    row:
        Row-like object from items-parquet-style data.

    Returns
    -------
    str
        Existing ``eval_subset_id`` when present, otherwise a deterministic ID.
    """
    existing = str(getattr(row, "eval_subset_id", "") or "").strip()
    if existing:
        return existing
    subset_rel_path = build_synthetic_rel_path(row)
    return build_eval_subset_id_from_subset_rel_path(subset_rel_path)


def build_window_id(row: Any) -> str:
    """Return the eval window ID used in logs and row-level outputs.

    Parameters
    ----------
    row:
        Row-like object from windows/items parquet data.

    Returns
    -------
    str
        ``eval_subset_id`` when available, else filename-derived ID.
    """
    eval_subset_id = str(getattr(row, "eval_subset_id", "") or "").strip()
    if eval_subset_id:
        return eval_subset_id
    rel_path = build_synthetic_rel_path(row)
    filename = rel_path.rsplit("/", maxsplit=1)[-1].removesuffix(".json")
    return f"{getattr(row, 'label')}.{filename}"
