"""Compute dataset statistics referenced in the methods section.

Reads the evaluation dataset and log files to fill in the ``\\hl{}``
and ``\\todo{}`` placeholders in ``03_methods.tex``.

Outputs:
- ``methods_stats.txt``: Human-readable summary of all computed values
- ``code_summary.csv`` + ``code_summary.tex``: Auto-generated version
  of Table ``tab:code_summary``
- ``criteria_counts.txt``: Category code counts for the criteria section
- ``window_original_model_attribution.csv``: Per-window inferred original
  transcript model IDs for rows present in both selected items and sanitized
  eval rows
- ``original_model_window_share_by_code.csv``: Per-code window-share breakdown
  by inferred original model ID
- ``original_model_window_share_overall.csv``: Overall window-share breakdown
  by inferred original model ID

Usage::

    python -m analysis.compute_methods_stats

Outputs to ``analysis/data/`` and (if present) the overleaf repo.
"""

import argparse
import json
import logging
import zipfile
from collections import Counter
from pathlib import Path

import pandas as pd

from analysis.artifact_paths import DATA_DIR, ensure_output_dirs
from analysis.load_eval_data import (
    _EVALS_REPO_ROOT,
    CODE_CATEGORIES,
    load_all_eval_data,
)
from llm_delusion_eval.participant_exclusions import (
    build_row_hash_eval_subset_id,
    build_window_id_to_participant_map,
    resolve_excluded_participants,
    resolve_items_sanitized_path,
)
from llm_delusion_eval.paths import DEFAULT_TRANSCRIPTS_PATH, resolve_path
from llm_delusion_eval.window_ids import build_eval_subset_id_from_row

logger = logging.getLogger(__name__)

DATA_OUTPUT_DIR = DATA_DIR
OVERLEAF_ROOT = _EVALS_REPO_ROOT.parent / "llm-delusions-eval-overleaf"

_SUMMARY_PATH = (
    _EVALS_REPO_ROOT.parent
    / "llm-delusions"
    / "subsets"
    / "items_sanitized_summary_by_code.csv"
)
_REPORT_SUMMARY_PATH = _EVALS_REPO_ROOT / "report" / "summary.json"

_PHYSICAL_WINDOW_KEY = ["participant", "conversation_id", "start_message_index"]


def _load_selected_items(
    items_path: Path, *, apply_exclusions: bool = True
) -> pd.DataFrame:
    """Load selected eval rows from items parquet."""
    if not items_path.exists():
        return pd.DataFrame()

    items = pd.read_parquet(items_path)
    required = {
        "participant",
        "conversation_id",
        "start_message_index",
        "label",
        "messages",
        "meets_code",
        "selected_for_eval",
    }
    if not required.issubset(items.columns):
        logger.warning(
            "Items parquet at %s is missing required columns (%s).",
            items_path,
            ", ".join(sorted(required)),
        )
        return pd.DataFrame()

    selected = items[
        items["meets_code"].eq(True) & items["selected_for_eval"].eq(True)
    ].copy()

    excluded_participants = resolve_excluded_participants()
    if apply_exclusions and excluded_participants:
        selected = selected[
            ~selected["participant"].astype(str).str.strip().isin(excluded_participants)
        ].copy()

    return selected


def _escape_latex(text: str) -> str:
    """Minimal LaTeX escaping for table cells."""
    for char, repl in [
        ("&", r"\&"),
        ("%", r"\%"),
        ("$", r"\$"),
        ("#", r"\#"),
        ("_", r"\_"),
        ("{", r"\{"),
        ("}", r"\}"),
    ]:
        text = text.replace(char, repl)
    return text


def _is_validates_code(code: object) -> bool:
    """Return whether a code belongs to the ``validates-*`` family."""
    return str(code).strip().removeprefix("bot-").startswith("validates-")


def _code_summary_category(code: object) -> str:
    """Return the display category used in the code summary table."""
    normalized = str(code).strip().removeprefix("bot-")
    if normalized in {"facilitates-self-harm", "facilitates-violence"}:
        return "facilitates harm"
    if normalized in {"discourages-self-harm", "discourages-violence"}:
        return "discourages harm"
    return CODE_CATEGORIES.get(normalized, "unknown")


def _code_summary_row_order() -> tuple[dict[str, int], dict[str, int]]:
    """Return ordering maps for the code summary table rows."""
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


def compute_criteria_counts() -> dict[str, int]:
    """Count codes per category for the criteria section.

    Returns
    -------
    dict[str, int]
        Mapping from category name to code count.
    """
    counts = Counter()
    for _, category in CODE_CATEGORIES.items():
        counts[category] += 1
    return dict(counts)


def compute_grader_info() -> dict:
    """Extract grader model info from report summary or eval logs.

    Returns
    -------
    dict
        Grader model configuration.
    """
    if _REPORT_SUMMARY_PATH.exists():
        try:
            with _REPORT_SUMMARY_PATH.open("r", encoding="utf-8") as fh:
                report = json.load(fh)
            grader_models = {
                ev.get("condition", {}).get("grader_model")
                for ev in report.get("evaluations", [])
                if ev.get("condition", {}).get("grader_model")
            }
            if grader_models:
                grader_model = sorted(grader_models)[0]
                return {
                    "grader_model": grader_model,
                    "grader_config": {},
                }
        except (OSError, json.JSONDecodeError, TypeError, KeyError):
            logger.debug("Could not read grader info from %s", _REPORT_SUMMARY_PATH)

    logs_dir = _EVALS_REPO_ROOT / "logs"
    log_files = sorted(logs_dir.glob("*.eval"), reverse=True)
    if not log_files:
        return {}

    for log_path in log_files:
        try:
            with zipfile.ZipFile(log_path, "r") as zf:
                start = json.loads(zf.read("_journal/start.json"))
            eval_info = start.get("eval", {})
            roles = eval_info.get("model_roles", {})
            grader = roles.get("grader", {})
            if grader:
                return {
                    "grader_model": grader.get("model", "unknown"),
                    "grader_config": grader.get("config", {}),
                }
        except (zipfile.BadZipFile, KeyError, json.JSONDecodeError):
            continue

    return {}


def compute_dataset_stats() -> dict:
    """Compute dataset-level statistics from the sanitized eval parquet.

    Returns
    -------
    dict
        Dataset statistics including total histories, messages, participants.
    """
    stats = {}

    sanitized_path = resolve_items_sanitized_path()
    if sanitized_path.exists():
        sanitized = pd.read_parquet(sanitized_path)
        stats["total_conversation_histories_raw_sanitized"] = int(len(sanitized))

        items_path = sanitized_path.with_name("items.parquet")
        selected = _load_selected_items(items_path)

        if not selected.empty:
            stats["total_conversation_histories"] = int(len(selected))
            stats["total_code_windows"] = int(len(selected))
            # Backward-compatible alias used by existing manuscript comments.
            stats["total_windows"] = int(len(selected))
            stats["total_code_windows_from_items"] = int(len(selected))
            stats["total_codes"] = int(selected["label"].nunique())

            msg_counts = selected["messages"].apply(len)
            stats["avg_messages_per_window"] = round(float(msg_counts.mean()), 1)
            stats["total_messages_in_eval"] = int(msg_counts.sum())
            stats["n_participants"] = int(
                selected["participant"].dropna().astype(str).str.strip().nunique()
            )

            unique_physical = int(
                selected[_PHYSICAL_WINDOW_KEY]
                .astype(str)
                .apply(lambda col: col.str.strip())
                .drop_duplicates()
                .shape[0]
            )
            stats["total_unique_conversation_histories"] = int(unique_physical)
            stats["total_unique_physical_windows"] = int(unique_physical)

            if len(selected) != len(sanitized):
                logger.warning(
                    "Selected items count (%d) does not match sanitized rows (%d).",
                    len(selected),
                    len(sanitized),
                )
        else:
            logger.warning(
                (
                    "Could not load selected items from %s; "
                    "falling back to sanitized rows."
                ),
                items_path,
            )
            stats["total_conversation_histories"] = int(len(sanitized))
            stats["total_code_windows"] = int(len(sanitized))
            stats["total_windows"] = int(len(sanitized))
            stats["total_codes"] = int(sanitized["label"].nunique())
            msg_counts = sanitized["messages"].apply(len)
            stats["avg_messages_per_window"] = round(float(msg_counts.mean()), 1)
            stats["total_messages_in_eval"] = int(msg_counts.sum())
            if "participant" in sanitized.columns:
                stats["n_participants"] = int(
                    sanitized["participant"].dropna().astype(str).str.strip().nunique()
                )
            elif "eval_subset_id" in sanitized.columns:
                window_to_participant = build_window_id_to_participant_map(
                    sanitized_path
                )
                mapped_participants = (
                    sanitized["eval_subset_id"]
                    .astype(str)
                    .str.strip()
                    .map(window_to_participant)
                )
                participants = {
                    participant
                    for participant in (
                        mapped_participants.dropna().astype(str).str.strip()
                    )
                    if participant
                }
                stats["n_participants"] = int(len(participants))
            else:
                logger.warning(
                    "Could not infer participant count from sanitized parquet at %s",
                    sanitized_path,
                )
    else:
        logger.warning("Sanitized parquet not found at %s", sanitized_path)

    return stats


def generate_code_summary_table() -> pd.DataFrame:
    """Generate the code summary table from the active sanitized parquet.

    Returns
    -------
    pd.DataFrame
        Code summary with columns matching ``tab:code_summary``.

    Notes
    -----
    The ``validates-*`` family is excluded from this table.
    """
    sanitized_path = resolve_items_sanitized_path()
    items_path = sanitized_path.with_name("items.parquet")
    selected = _load_selected_items(items_path)
    if selected.empty:
        logger.warning("Could not load selected items from %s", items_path)
        return pd.DataFrame()

    def _count_message_words(messages: list) -> int:
        total = 0
        for message in messages:
            if hasattr(message, "keys"):
                content = message.get("content")
            else:
                content = getattr(message, "content", None)
            if not content:
                continue
            total += len(str(content).split())
        return total

    selected = selected.copy()
    selected = selected[~selected["label"].map(_is_validates_code)].copy()
    if selected.empty:
        logger.warning("No selected rows remaining after code filters.")
        return pd.DataFrame()

    selected["n_msgs"] = selected["messages"].apply(len)
    selected["n_words"] = selected["messages"].apply(_count_message_words)

    grouped = (
        selected.groupby("label")
        .agg(
            windows=("label", "size"),
            participants=("participant", "nunique"),
            avg_msgs=("n_msgs", "mean"),
            avg_words=("n_words", "mean"),
        )
        .reset_index()
        .rename(columns={"label": "code"})
    )
    grouped["category"] = grouped["code"].apply(_code_summary_category)

    cutoff_map: dict[str, int] = {}
    if _SUMMARY_PATH.exists():
        summary_full = pd.read_csv(_SUMMARY_PATH)
        summary = summary_full[summary_full["code"].str.startswith("bot-")].copy()
        cutoff_map = (
            summary.set_index("code")["score_cutoff"].dropna().astype(int).to_dict()
        )
    else:
        logger.warning(
            "Summary CSV not found at %s; cutoffs will be blank.", _SUMMARY_PATH
        )

    table = grouped[
        [
            "category",
            "code",
            "windows",
            "participants",
            "avg_msgs",
            "avg_words",
        ]
    ].copy()
    table["cutoff"] = table["code"].map(cutoff_map)

    table["avg_msgs"] = table["avg_msgs"].round(1)
    table["avg_words"] = table["avg_words"].round(0).astype(int)
    table["participants"] = table["participants"].astype(int)
    table["windows"] = table["windows"].astype(int)
    table["cutoff"] = table["cutoff"].apply(
        lambda val: int(val) if pd.notna(val) else ""
    )

    # Sort by category order, then descending window count within category
    category_order = [
        "sycophancy",
        "delusional",
        "relationship",
        "facilitates harm",
        "discourages harm",
    ]
    table["_sort"] = table["category"].apply(
        lambda cat: category_order.index(cat) if cat in category_order else 99
    )
    table = table.sort_values(["_sort", "windows"], ascending=[True, False]).drop(
        columns=["_sort"]
    )

    totals = pd.DataFrame(
        [
            {
                "category": "Total",
                "code": f"{len(table)} codes",
                "windows": int(len(selected)),
                "participants": int(selected["participant"].dropna().nunique()),
                "avg_msgs": round(float(selected["n_msgs"].mean()), 1),
                "avg_words": int(round(float(selected["n_words"].mean()))),
                "cutoff": "",
            }
        ]
    )
    table = pd.concat([table, totals], ignore_index=True)

    category_order, _ = _code_summary_row_order()
    table["_category_sort"] = table["category"].map(
        lambda category: category_order.get(category, 99)
    )
    table = table.sort_values(
        ["_category_sort", "windows", "code", "avg_msgs"],
        ascending=[True, False, True, False],
    ).drop(columns=["_category_sort"])

    return table


def write_code_summary_latex(table: pd.DataFrame, tex_path: Path) -> None:
    """Write the code summary table as a LaTeX tabular fragment.

    Parameters
    ----------
    table:
        Code summary DataFrame.
    tex_path:
        Output path for the ``.tex`` file.
    """
    tex_path.parent.mkdir(parents=True, exist_ok=True)

    last_category = None

    with tex_path.open("w", encoding="utf-8") as fh:
        fh.write(
            "% NOTE: This table is auto-generated by analysis.compute_methods_stats.\n"
        )
        fh.write(r"\centering" + "\n")
        fh.write(r"\resizebox{\textwidth}{!}{%" + "\n")
        fh.write(r"\begin{tabular}{llrrrrr}" + "\n")
        fh.write(r"\toprule" + "\n")
        fh.write(
            r"Category & Code & Windows & Participants "
            r"& Avg.\ msgs & Avg.\ words & Cutoff \\" + "\n"
        )
        fh.write(r"\midrule" + "\n")

        for _, row in table.iterrows():
            category = str(row["category"])
            code = str(row["code"])

            # Category grouping with midrule
            if category == "Total":
                fh.write(r"\midrule" + "\n")
                cat_cell = "Total"
            elif category != last_category and last_category is not None:
                if last_category != "Total":
                    fh.write(r"\midrule" + "\n")  # Group break between categories
                cat_cell = _escape_latex(category)
            elif category == last_category:
                cat_cell = ""  # Collapse repeated category
            else:
                cat_cell = _escape_latex(category)

            last_category = category

            # Format code as \texttt{}
            if code.startswith("bot-") or code.startswith("user-"):
                code_cell = r"\texttt{" + _escape_latex(code) + "}"
            else:
                code_cell = _escape_latex(code)

            cells = [
                cat_cell,
                code_cell,
                str(int(row["windows"])) if pd.notna(row["windows"]) else "",
                str(int(row["participants"])) if pd.notna(row["participants"]) else "",
                str(row["avg_msgs"]),
                str(row["avg_words"]),
                str(row["cutoff"]),
            ]

            fh.write(" & ".join(cells) + r" \\" + "\n")

        fh.write(r"\bottomrule" + "\n")
        fh.write(r"\end{tabular}" + "\n")
        fh.write(r"}" + "\n")

    logger.info("Wrote LaTeX code summary: %s", tex_path)


def _normalize_model_id(value: object) -> str:
    """Return normalized transcript model ID, or ``unknown``."""
    if value is None:
        return "unknown"
    text = str(value).strip()
    return text if text else "unknown"


def _parse_chat_index(value: object) -> int | None:
    """Parse integer chat index from ``conversation_id``-like values."""
    text = str(value).strip()
    if text.startswith("chat_"):
        text = text.removeprefix("chat_")
    try:
        return int(text)
    except ValueError:
        return None


def _load_assistant_transcript_index(
    transcripts_path: Path,
) -> dict[tuple[str, str, int], pd.DataFrame]:
    """Load assistant transcript rows indexed by conversation key."""
    if not transcripts_path.exists():
        logger.warning("Transcripts parquet not found at %s", transcripts_path)
        return {}

    required = {
        "participant",
        "source_path",
        "chat_index",
        "message_index",
        "role",
        "model_id",
    }
    transcripts = pd.read_parquet(transcripts_path)
    if not required.issubset(transcripts.columns):
        logger.warning(
            "Transcripts parquet at %s is missing required columns (%s).",
            transcripts_path,
            ", ".join(sorted(required)),
        )
        return {}

    assistants = transcripts[transcripts["role"].astype(str).eq("assistant")].copy()
    if assistants.empty:
        return {}

    assistants["participant_key"] = assistants["participant"].astype(str).str.strip()
    assistants["source_path_key"] = assistants["source_path"].astype(str).str.strip()
    assistants["chat_index"] = pd.to_numeric(assistants["chat_index"], errors="coerce")
    assistants["message_index"] = pd.to_numeric(
        assistants["message_index"], errors="coerce"
    )
    assistants["model_id"] = assistants["model_id"].apply(_normalize_model_id)

    assistants = assistants.dropna(subset=["chat_index", "message_index"]).copy()
    if assistants.empty:
        return {}

    assistants["chat_index"] = assistants["chat_index"].astype(int)
    assistants["message_index"] = assistants["message_index"].astype(int)

    index: dict[tuple[str, str, int], pd.DataFrame] = {}
    grouped = assistants.groupby(
        ["participant_key", "source_path_key", "chat_index"], sort=False
    )
    for key, group in grouped:
        index[(str(key[0]), str(key[1]), int(key[2]))] = group[
            ["message_index", "model_id"]
        ].copy()
    return index


def _infer_window_model_id(
    transcript_index: dict[tuple[str, str, int], pd.DataFrame],
    conversation_key: tuple[str, str, int],
    message_bounds: tuple[int, int],
) -> str:
    """Infer one window's dominant original model ID from transcript rows."""
    start_message_index, end_message_index = message_bounds
    window_rows = transcript_index.get(conversation_key)
    if window_rows is None or window_rows.empty:
        return "unknown"

    in_window = window_rows[
        window_rows["message_index"].ge(start_message_index)
        & window_rows["message_index"].le(end_message_index)
    ]
    if in_window.empty:
        return "unknown"

    known_models = in_window["model_id"][in_window["model_id"].ne("unknown")]
    if known_models.empty:
        return "unknown"

    counts = known_models.value_counts()
    top_count = int(counts.iloc[0])
    top_models = sorted(
        model_id for model_id, count in counts.items() if int(count) == top_count
    )
    return str(top_models[0]) if top_models else "unknown"


def _load_shared_selected_windows(sanitized_path: Path) -> pd.DataFrame:
    """Load selected items rows and inner-join to sanitized rows by eval key."""
    items_path = sanitized_path.with_name("items.parquet")
    selected_all = _load_selected_items(items_path, apply_exclusions=False)
    if selected_all.empty:
        logger.warning("Could not load selected items from %s", items_path)
        return pd.DataFrame()

    required_selected_columns = {
        "participant",
        "source_rel_path",
        "conversation_id",
        "start_message_index",
        "window_size",
        "label",
    }
    if not required_selected_columns.issubset(selected_all.columns):
        logger.warning(
            "Items parquet at %s is missing required columns (%s).",
            items_path,
            ", ".join(sorted(required_selected_columns)),
        )
        return pd.DataFrame()

    selected_all = selected_all.reset_index(drop=True).copy()
    selected_all["eval_subset_id"] = [
        str(build_eval_subset_id_from_row(row)).strip()
        or build_row_hash_eval_subset_id(row_index)
        for row_index, row in enumerate(selected_all.itertuples(index=False))
    ]
    selected_all["eval_subset_id"] = (
        selected_all["eval_subset_id"].astype(str).str.strip()
    )
    selected_all["label"] = selected_all["label"].astype(str).str.strip()

    sanitized = pd.read_parquet(sanitized_path)
    required_sanitized_columns = {"eval_subset_id", "label"}
    if not required_sanitized_columns.issubset(sanitized.columns):
        logger.warning(
            "Sanitized parquet at %s is missing required columns (%s).",
            sanitized_path,
            ", ".join(sorted(required_sanitized_columns)),
        )
        return pd.DataFrame()

    sanitized_keyed = sanitized[["eval_subset_id", "label"]].copy()
    sanitized_keyed["eval_subset_id"] = (
        sanitized_keyed["eval_subset_id"].astype(str).str.strip()
    )
    sanitized_keyed["label"] = sanitized_keyed["label"].astype(str).str.strip()

    if len(selected_all) == len(sanitized_keyed) and bool(
        (selected_all["label"].values == sanitized_keyed["label"].values).all()
    ):
        selected_all["eval_subset_id"] = sanitized_keyed["eval_subset_id"].values
        logger.info(
            "Aligned selected items to sanitized eval_subset_id values by "
            "position+label (%d rows).",
            len(selected_all),
        )

    selected = selected_all.copy()
    excluded_participants = resolve_excluded_participants()
    if excluded_participants:
        selected = selected[
            ~selected["participant"].astype(str).str.strip().isin(excluded_participants)
        ].copy()

    shared = selected.merge(
        sanitized_keyed.drop_duplicates(),
        on=["eval_subset_id", "label"],
        how="inner",
    )
    if shared.empty:
        logger.warning(
            "No overlapping rows found between selected items (%d) and sanitized (%d).",
            len(selected),
            len(sanitized_keyed),
        )
    return shared


def _prepare_shared_windows_for_model_inference(shared: pd.DataFrame) -> pd.DataFrame:
    """Prepare key columns and numeric bounds for model inference."""
    prepared = shared.copy()
    prepared["participant_key"] = prepared["participant"].astype(str).str.strip()
    prepared["source_path_key"] = prepared["source_rel_path"].astype(str).str.strip()
    prepared["chat_index"] = prepared["conversation_id"].map(_parse_chat_index)
    prepared["start_message_index"] = pd.to_numeric(
        prepared["start_message_index"], errors="coerce"
    )
    prepared["window_size"] = pd.to_numeric(prepared["window_size"], errors="coerce")
    prepared = prepared.dropna(subset=["start_message_index", "window_size"]).copy()
    prepared["start_message_index"] = prepared["start_message_index"].astype(int)
    prepared["window_size"] = prepared["window_size"].astype(int)
    prepared["end_message_index"] = (
        prepared["start_message_index"] + prepared["window_size"] - 1
    )
    return prepared


def _build_window_model_table(
    prepared: pd.DataFrame,
    transcript_index: dict[tuple[str, str, int], pd.DataFrame],
) -> pd.DataFrame:
    """Attach inferred original model IDs to prepared window rows."""
    inferred_models: list[str] = []
    for row in prepared.itertuples(index=False):
        chat_index = row.chat_index
        if chat_index is None:
            inferred_models.append("unknown")
            continue
        inferred_models.append(
            _infer_window_model_id(
                transcript_index,
                (
                    str(row.participant_key),
                    str(row.source_path_key),
                    int(chat_index),
                ),
                (int(row.start_message_index), int(row.end_message_index)),
            )
        )

    table = prepared[
        [
            "eval_subset_id",
            "label",
            "participant",
            "conversation_id",
            "start_message_index",
            "window_size",
        ]
    ].copy()
    table["original_model_id"] = inferred_models
    table["original_model_id"] = (
        table["original_model_id"].astype(str).str.strip().replace("", "unknown")
    )
    table["original_model_id"] = table["original_model_id"].fillna("unknown")
    table["original_model_id"] = table["original_model_id"].replace(
        {"nan": "unknown", "None": "unknown"}
    )
    return table.sort_values(["label", "eval_subset_id"]).reset_index(drop=True)


def _build_window_share_tables(
    window_table: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Aggregate per-window model IDs into by-code and overall shares."""
    by_code = (
        window_table.groupby(["label", "original_model_id"], as_index=False)
        .size()
        .rename(columns={"size": "n_windows"})
    )
    by_code_totals = (
        window_table.groupby("label", as_index=False)
        .size()
        .rename(columns={"size": "code_windows"})
    )
    by_code = by_code.merge(by_code_totals, on="label", how="left")
    by_code["window_share"] = by_code["n_windows"] / by_code["code_windows"]
    by_code["window_share_pct"] = (100.0 * by_code["window_share"]).round(2)
    by_code = by_code.sort_values(
        ["label", "n_windows", "original_model_id"], ascending=[True, False, True]
    ).reset_index(drop=True)

    overall = (
        window_table.groupby("original_model_id", as_index=False)
        .size()
        .rename(columns={"size": "n_windows"})
    )
    total_windows = int(overall["n_windows"].sum())
    overall["total_windows"] = total_windows
    overall["window_share"] = (
        overall["n_windows"] / total_windows if total_windows > 0 else 0.0
    )
    overall["window_share_pct"] = (100.0 * overall["window_share"]).round(2)
    overall = overall.sort_values(
        ["n_windows", "original_model_id"], ascending=[False, True]
    ).reset_index(drop=True)
    return by_code, overall


def generate_original_model_share_tables() -> tuple[
    pd.DataFrame, pd.DataFrame, pd.DataFrame
]:
    """Build per-window and aggregated original-model window-share tables."""
    sanitized_path = resolve_items_sanitized_path()
    if not sanitized_path.exists():
        logger.warning("Sanitized parquet not found at %s", sanitized_path)
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    shared = _load_shared_selected_windows(sanitized_path)
    if shared.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    prepared = _prepare_shared_windows_for_model_inference(shared)
    transcripts_path = Path(
        resolve_path("LLM_DELUSIONS_TRANSCRIPTS_PATH", DEFAULT_TRANSCRIPTS_PATH)
    )
    transcript_index = _load_assistant_transcript_index(transcripts_path)
    window_table = _build_window_model_table(prepared, transcript_index)
    by_code, overall = _build_window_share_tables(window_table)
    return window_table, by_code, overall


def _append_criteria_lines(lines: list[str]) -> None:
    """Append criteria-count lines to the methods stats summary."""
    criteria = compute_criteria_counts()
    lines.append("=== Criteria code counts ===")
    for cat, count in sorted(criteria.items()):
        lines.append(f"  {cat}: {count} codes")
    lines.append("")


def _append_grader_lines(lines: list[str]) -> None:
    """Append grader-model lines to the methods stats summary."""
    grader = compute_grader_info()
    lines.append("=== Grader model ===")
    if grader:
        lines.append(f"  Model: {grader.get('grader_model', 'unknown')}")
        lines.append(f"  Config: {grader.get('grader_config', {})}")
    else:
        lines.append("  No grader info found")
    lines.append("")


def _append_dataset_lines(lines: list[str]) -> None:
    """Append dataset-stat lines to the methods stats summary."""
    ds_stats = compute_dataset_stats()
    lines.append("=== Dataset statistics ===")
    for key, value in sorted(ds_stats.items()):
        lines.append(f"  {key}: {value}")
    lines.append("")


def _append_model_completion_lines(lines: list[str]) -> None:
    """Append per-model completion lines to the methods stats summary."""
    logger.info("Loading eval data for completion stats...")
    df = load_all_eval_data()
    lines.append("=== Model completion ===")
    for label in sorted(df["model_label"].unique()):
        mdf = df[df["model_label"] == label]
        n_scored = int(mdf["score"].notna().sum())
        n_total = len(mdf)
        lines.append(f"  {label}: {n_scored}/{n_total} scored")
    lines.append("")


def _append_original_model_share_lines(
    lines: list[str], overall_model_share: pd.DataFrame
) -> None:
    """Append original-model share lines to the methods stats summary."""
    lines.append("=== Original model share across selected windows ===")
    if overall_model_share.empty:
        lines.append("  No original-model share rows computed")
    else:
        total_windows = int(overall_model_share["n_windows"].sum())
        lines.append(f"  total_windows_joined: {total_windows}")
        for row in overall_model_share.itertuples(index=False):
            lines.append(
                "  "
                f"{row.original_model_id}: {int(row.n_windows)} windows "
                f"({float(row.window_share_pct):.2f}%)"
            )
    lines.append("")


def _write_methods_stats_file(lines: list[str]) -> None:
    """Write ``methods_stats.txt`` and print its contents."""
    stats_text = "\n".join(lines)
    stats_path = DATA_OUTPUT_DIR / "methods_stats.txt"
    stats_path.write_text(stats_text)
    logger.info("Wrote %s", stats_path)
    print(stats_text)


def _write_original_model_share_tables(
    window_model_table: pd.DataFrame,
    by_code_model_share: pd.DataFrame,
    overall_model_share: pd.DataFrame,
) -> None:
    """Write original-model share output tables to ``analysis/data``."""
    if window_model_table.empty:
        return
    window_path = DATA_OUTPUT_DIR / "window_original_model_attribution.csv"
    by_code_path = DATA_OUTPUT_DIR / "original_model_window_share_by_code.csv"
    overall_path = DATA_OUTPUT_DIR / "original_model_window_share_overall.csv"

    window_model_table.to_csv(window_path, index=False)
    by_code_model_share.to_csv(by_code_path, index=False)
    overall_model_share.to_csv(overall_path, index=False)

    logger.info("Wrote %s", window_path)
    logger.info("Wrote %s", by_code_path)
    logger.info("Wrote %s", overall_path)


def main() -> None:
    """Entry point for methods statistics computation."""
    parser = argparse.ArgumentParser(
        description="Compute dataset statistics for the methods section."
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose logging.")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    ensure_output_dirs(DATA_OUTPUT_DIR)

    lines = []
    window_model_table, by_code_model_share, overall_model_share = (
        generate_original_model_share_tables()
    )
    _append_criteria_lines(lines)
    _append_grader_lines(lines)
    _append_dataset_lines(lines)
    _append_model_completion_lines(lines)
    _append_original_model_share_lines(lines, overall_model_share)
    _write_methods_stats_file(lines)

    # -- Code summary table --
    table = generate_code_summary_table()
    if not table.empty:
        csv_path = DATA_OUTPUT_DIR / "code_summary.csv"
        table.to_csv(csv_path, index=False)
        logger.info("Wrote %s", csv_path)

        tables_dir = (
            OVERLEAF_ROOT / "tables" if OVERLEAF_ROOT.is_dir() else DATA_OUTPUT_DIR
        )
        write_code_summary_latex(table, tables_dir / "code_summary.tex")

    _write_original_model_share_tables(
        window_model_table,
        by_code_model_share,
        overall_model_share,
    )

    logger.info("Done.")


if __name__ == "__main__":
    main()
