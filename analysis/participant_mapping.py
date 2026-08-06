"""Build a mapping from eval_subset_id to participant.

The sanitized parquet uses hashed window IDs, while the original items
parquet contains participant identifiers.  Both datasets have 725 rows
in the same positional order (verified by matching labels), so we can
join them positionally.

Usage::

    from analysis.participant_mapping import get_participant_mapping
    mapping = get_participant_mapping()  # {eval_subset_id: participant_id}
"""

import logging

import pandas as pd

from llm_delusion_eval.paths import _EVALS_REPO_ROOT

logger = logging.getLogger(__name__)

_ITEMS_PATH = _EVALS_REPO_ROOT.parent / "llm-delusions" / "subsets" / "items.parquet"

_SANITIZED_PATH = (
    _EVALS_REPO_ROOT.parent / "llm-delusions" / "subsets" / "items_sanitized.parquet"
)


def get_participant_mapping() -> dict[str, str]:
    """Return a mapping from eval_subset_id to participant identifier.

    The mapping is derived by positionally joining the sanitized parquet
    (which uses hashed IDs) with the original items parquet (which
    contains participant IDs).

    Returns
    -------
    dict[str, str]
        Mapping from ``eval_subset_id`` to participant string.
    """
    if not _ITEMS_PATH.exists():
        logger.warning("items.parquet not found at %s", _ITEMS_PATH)
        return {}

    if not _SANITIZED_PATH.exists():
        logger.warning("items_sanitized.parquet not found at %s", _SANITIZED_PATH)
        return {}

    items = pd.read_parquet(
        _ITEMS_PATH,
        columns=["subset_id", "participant", "label", "selected_for_eval"],
    )
    sanitized = pd.read_parquet(
        _SANITIZED_PATH,
        columns=["eval_subset_id", "label"],
    )

    items_sel = items[items["selected_for_eval"]].reset_index(drop=True)

    if len(items_sel) != len(sanitized):
        logger.warning(
            "Row count mismatch: items_selected=%d, sanitized=%d",
            len(items_sel),
            len(sanitized),
        )
        return {}

    if not (items_sel["label"].values == sanitized["label"].values).all():
        logger.warning("Label mismatch between items and sanitized parquets")
        return {}

    mapping = dict(
        zip(
            sanitized["eval_subset_id"].values,
            items_sel["participant"].astype(str).values,
        )
    )
    logger.info("Built participant mapping for %d windows", len(mapping))
    return mapping


def add_participant_column(df: pd.DataFrame) -> pd.DataFrame:
    """Add a ``participant`` column to a DataFrame with ``window_id``.

    Parameters
    ----------
    df:
        DataFrame containing a ``window_id`` column.

    Returns
    -------
    pd.DataFrame
        Input DataFrame with an added ``participant`` column.
    """
    mapping = get_participant_mapping()
    df = df.copy()
    df["participant"] = df["window_id"].map(mapping)
    n_mapped = df["participant"].notna().sum()
    logger.info("Mapped %d/%d rows to participants", n_mapped, len(df))
    return df
