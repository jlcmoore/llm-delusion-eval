"""Generate a methods figure from real eval data as a vector PDF.

This generator keeps the original TikZ-inspired structure: symbolic replay rows
plus one concrete real-data exchange. It writes two artifacts:
- a standalone TikZ source file
- a compiled PDF figure
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import tempfile
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

SCRIPT_PATH = Path(__file__).resolve()
ANALYSIS_DIR = SCRIPT_PATH.parent
EVALS_ROOT = ANALYSIS_DIR.parent
WORKSPACE_ROOT = EVALS_ROOT.parent
DELUSIONS_ROOT = WORKSPACE_ROOT / "llm-delusions"
WINDOW_CUTUP_TEMPLATE_PATH = (
    ANALYSIS_DIR / "templates" / "window_cutup_real_data_figure_template.tex"
)
DEFAULT_ANNOTATIONS_MATCHES_PATH = (
    DELUSIONS_ROOT / "annotations/all_annotations__matches.parquet"
)
TEMPLATE_SAMPLE_COUNT = 2
POSITIVE_BEHAVIOR_SCORE = 2
MIN_BEHAVIOR_PRESENT_OUTCOMES = 3
PREFERRED_TEMPLATE_SAMPLE_IDS = {
    ("bot-endorses-delusion", "openai/gpt-5.4-2026-03-05", "none"): (
        "71832541a71bddfd.turn6",
        "71832541a71bddfd.turn14",
    ),
}


@dataclass
class SampleTurn:
    """Container for one selected sample turn.

    Parameters
    ----------
    sample_id:
        Sample identifier in ``window_id.turnN`` format.
    turn_index:
        Numeric turn index parsed from ``sample_id``.
    history:
        Prefix history passed to the evaluated model.
    user_text:
        Current user message text.
    original_excerpt:
        Original transcript assistant output excerpt.
    new_excerpt:
        Evaluated model assistant output excerpt.
    original_score:
        Binarized score for original transcript output.
    new_score:
        Binarized score for evaluated model output.
    original_raw_score:
        Raw original transcript score in [0, 10], if available.
    new_raw_score:
        Raw evaluated model raw score in [0, 10], if available.
    original_cutoff:
        Cutoff used for original transcript binarization, if available.
    new_cutoff:
        Cutoff used for evaluated model binarization, if available.
    original_matches:
        Matched evidence snippets from the original judge explanation.
    new_matches:
        Matched evidence snippets from the evaluated-model judge explanation.
    """

    sample_id: str
    turn_index: int
    history: list[dict[str, Any]]
    user_text: str
    original_excerpt: str
    new_excerpt: str
    original_score: float
    new_score: float
    original_raw_score: float | None
    new_raw_score: float | None
    original_cutoff: float | None
    new_cutoff: float | None
    original_matches: list[str]
    new_matches: list[str]


@dataclass
class FigureTemplatePayload:
    """Container for template content substitutions.

    Parameters
    ----------
    original_excerpt_pairs:
        Two original-output excerpt strings for ``A1`` and ``A2``.
    evaluated_excerpt_pairs:
        Two evaluated-output excerpt strings for ``A'_1`` and ``A'_2``.
    behavior_labels:
        Four behavior labels for ``A1``, ``A2``, ``A'_1``, and ``A'_2``.
    user_match_pairs:
        Two user-side excerpt strings for ``U_1`` and ``U_2``.
    """

    original_excerpt_pairs: tuple[str, str]
    evaluated_excerpt_pairs: tuple[str, str]
    behavior_labels: tuple[str, str, str, str]
    user_match_pairs: tuple[str, str]


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns
    -------
    argparse.Namespace
        Parsed command-line arguments.
    """

    parser = argparse.ArgumentParser(
        description="Generate real-data replay figure as a PDF asset."
    )
    parser.add_argument("--code", default="bot-endorses-delusion")
    parser.add_argument("--model-id", default="openai/gpt-5.4-2026-03-05")
    parser.add_argument("--model-display", default="GPT-5.4")
    parser.add_argument("--reasoning-effort", default="none")
    parser.add_argument(
        "--items-parquet",
        default=str(DELUSIONS_ROOT / "subsets/items.parquet"),
    )
    parser.add_argument(
        "--transcripts-parquet",
        default=str(DELUSIONS_ROOT / "transcripts_data/transcripts.parquet"),
    )
    parser.add_argument(
        "--samples-root",
        default=str(EVALS_ROOT / "report/samples"),
    )
    parser.add_argument(
        "--output-figure-tex",
        default=str(ANALYSIS_DIR / "figures/window_cutup_real_data_figure.tex"),
    )
    parser.add_argument(
        "--output-pdf",
        default=str(ANALYSIS_DIR / "figures/window_cutup_real_data.pdf"),
    )
    parser.add_argument("--user-code", default="user-endorses-delusion")
    parser.add_argument(
        "--annotations-matches-parquet",
        default=str(DEFAULT_ANNOTATIONS_MATCHES_PATH),
    )
    return parser.parse_args()


def build_eval_subset_id_from_row(row: pd.Series) -> str:
    """Compute deterministic eval subset id from an items.parquet row.

    Parameters
    ----------
    row:
        One row from ``items.parquet``.

    Returns
    -------
    str
        Deterministic 16-character eval subset id.
    """

    chat_index = int(str(row["conversation_id"]).replace("chat_", ""))
    filename = (
        f"{row['participant']}_{row['filename_hash']}_chat{chat_index}"
        f"_win{int(row['start_message_index'])}.json"
    )
    subset_rel_path = f"{row['label']}/{filename}"
    return hashlib.sha256(subset_rel_path.encode("utf-8")).hexdigest()[:16]


def sample_json_path(
    samples_root: Path,
    model_id: str,
    reasoning_effort: str | None,
    code: str,
) -> Path:
    """Build the report sample JSON path for one model/config.

    Parameters
    ----------
    samples_root:
        Root report samples directory.
    model_id:
        Model identifier.
    reasoning_effort:
        Reasoning effort value, or ``None`` for original transcript.
    code:
        Annotation code.

    Returns
    -------
    Path
        Path to the code-specific sample JSON.
    """

    model_dir = model_id.replace("/", "_")
    base_run = (
        "grader_model=openai_gpt-5.1-2025-11-13&max_context_messages=0&max_windows=0"
    )
    if model_id == "original_transcript":
        run_dir = base_run
    else:
        run_dir = f"{base_run}&reasoning_effort={reasoning_effort}"
    return samples_root / model_dir / run_dir / f"{code}.json"


def load_sample_map(path: Path) -> dict[str, dict[str, Any]]:
    """Load sample JSON keyed by ``sample_id``.

    Parameters
    ----------
    path:
        Path to report sample JSON.

    Returns
    -------
    dict[str, dict[str, Any]]
        Mapping from sample id to row payload.
    """

    with path.open("r", encoding="utf-8") as infile:
        payload = json.load(infile)
    return {str(row["sample_id"]): row for row in payload}


def parse_turn_index(sample_id: str) -> int:
    """Extract the numeric turn index from ``sample_id``.

    Parameters
    ----------
    sample_id:
        Sample identifier like ``window.turn6``.

    Returns
    -------
    int
        Parsed turn index.
    """

    return int(sample_id.rsplit(".turn", maxsplit=1)[-1])


def parse_explanation_fields(text: str) -> tuple[float | None, float | None, list[str]]:
    """Parse score fields and evidence matches from grader explanation JSON.

    Parameters
    ----------
    text:
        Serialized explanation JSON string.

    Returns
    -------
    tuple[float | None, float | None, list[str]]
        ``(raw_score, cutoff, matches)`` triple.
    """

    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return None, None, []
    raw_value = parsed.get("raw_score")
    cutoff_value = parsed.get("cutoff")
    matches_value = parsed.get("matches")
    raw_score = float(raw_value) if raw_value is not None else None
    cutoff = float(cutoff_value) if cutoff_value is not None else None
    matches: list[str] = []
    if isinstance(matches_value, list):
        matches = [str(item) for item in matches_value if str(item).strip()]
    return raw_score, cutoff, matches


def choose_window_and_shared_samples(
    original_samples: dict[str, dict[str, Any]],
    new_samples: dict[str, dict[str, Any]],
) -> tuple[str, list[str]]:
    """Select one shared window with enough positive behavior for the template.

    Parameters
    ----------
    original_samples:
        Original transcript sample map.
    new_samples:
        Evaluated model sample map.

    Returns
    -------
    tuple[str, list[str]]
        Selected ``(window_id, sample_ids)``.
    """
    shared_ids = sorted(set(original_samples) & set(new_samples))
    if not shared_ids:
        raise ValueError("No shared samples found for requested inputs.")

    grouped: dict[str, list[str]] = {}
    for sample_id in shared_ids:
        window_id = sample_id.split(".turn", maxsplit=1)[0]
        grouped.setdefault(window_id, []).append(sample_id)

    window_candidates: list[tuple[str, list[str], int, int]] = []
    for window_id, window_sample_ids in grouped.items():
        ordered_ids = sorted(window_sample_ids, key=parse_turn_index)
        if len(ordered_ids) < TEMPLATE_SAMPLE_COUNT:
            continue
        scored_ids = sorted(
            [
                (
                    int(float(original_samples[sample_id].get("score", 0) or 0) > 0)
                    + int(float(new_samples[sample_id].get("score", 0) or 0) > 0),
                    sample_id,
                )
                for sample_id in ordered_ids
            ],
            key=lambda item: (-item[0], parse_turn_index(item[1]), item[1]),
        )
        pair_total = scored_ids[0][0] + scored_ids[1][0]
        shared_positive_count = sum(
            1 for score, _ in scored_ids if score == POSITIVE_BEHAVIOR_SCORE
        )
        window_candidates.append(
            (window_id, ordered_ids, pair_total, shared_positive_count)
        )

    if not window_candidates:
        raise ValueError("Template figure requires at least two shared turns.")

    chosen_window, chosen_ids, _, _ = sorted(
        window_candidates,
        key=lambda item: (
            -int(item[2] >= MIN_BEHAVIOR_PRESENT_OUTCOMES),
            -item[2],
            -item[3],
            -len(item[1]),
            min(parse_turn_index(sample_id) for sample_id in item[1]),
            item[0],
        ),
    )[0]
    return chosen_window, chosen_ids


def collapse_whitespace(text: str) -> str:
    """Collapse whitespace in figure text while preserving punctuation.

    Parameters
    ----------
    text:
        Input text.

    Returns
    -------
    str
        Single-line text with repeated whitespace collapsed.
    """

    return re.sub(r"\s+", " ", str(text)).strip()


def normalize_text(text: str) -> str:
    """Normalize user/model text to compact ASCII.

    Parameters
    ----------
    text:
        Input text.

    Returns
    -------
    str
        Collapsed ASCII text.
    """

    collapsed = collapse_whitespace(text)
    ascii_text = (
        unicodedata.normalize("NFKD", collapsed).encode("ascii", "ignore").decode()
    )
    return collapse_whitespace(ascii_text)


def truncate_text(text: str, max_chars: int) -> str:
    """Truncate text with ellipsis.

    Parameters
    ----------
    text:
        Input text.
    max_chars:
        Character budget.

    Returns
    -------
    str
        Possibly truncated text.
    """

    if len(text) <= max_chars:
        return text
    cutoff = max_chars - 3
    truncated = text[:cutoff].rstrip()
    boundary = max(
        truncated.rfind(" "),
        truncated.rfind("."),
        truncated.rfind(","),
        truncated.rfind(";"),
        truncated.rfind(":"),
    )
    if boundary >= cutoff // 2:
        truncated = truncated[:boundary].rstrip(" ,;:")
    return truncated + "..."


def escape_latex(text: str) -> str:
    """Escape LaTeX special characters in plain text.

    Parameters
    ----------
    text:
        Raw string.

    Returns
    -------
    str
        Escaped string safe for LaTeX text context.
    """

    escaped = text
    replacements = {
        "\\": r"\textbackslash{}",
        "{": r"\{",
        "}": r"\}",
        "%": r"\%",
        "$": r"\$",
        "&": r"\&",
        "#": r"\#",
        "_": r"\_",
        "^": r"\^{}",
        "~": r"\~{}",
    }
    for source, target in replacements.items():
        escaped = escaped.replace(source, target)
    return escaped


def find_items_row(items_path: Path, code: str, window_id: str) -> pd.Series:
    """Find the selected row in non-sanitized items parquet.

    Parameters
    ----------
    items_path:
        Path to ``items.parquet``.
    code:
        Annotation code.
    window_id:
        Selected eval window id.

    Returns
    -------
    pd.Series
        Matching row.
    """

    items = pd.read_parquet(items_path)
    code_rows = items[items["label"].eq(code)].copy()
    code_rows["eval_subset_id"] = code_rows.apply(build_eval_subset_id_from_row, axis=1)
    matches = code_rows[code_rows["eval_subset_id"].eq(window_id)]
    if matches.empty:
        raise ValueError(f"Window {window_id} not found in items parquet for {code}.")
    return matches.iloc[0]


def _coerce_match_sequence(value: Any) -> list[str]:
    """Convert a raw match field to a normalized string list.

    Parameters
    ----------
    value:
        Raw value from ``matches__<code>`` field.

    Returns
    -------
    list[str]
        Non-empty normalized match strings.
    """

    if value is None:
        return []
    if isinstance(value, str):
        return [value] if value.strip() else []
    if isinstance(value, (list, tuple)):
        return [str(item) for item in value if str(item).strip()]
    if hasattr(value, "tolist"):
        converted = value.tolist()
        if isinstance(converted, list):
            return [str(item) for item in converted if str(item).strip()]
    return [str(value)] if str(value).strip() else []


def extract_original_item_matches(
    item_row: pd.Series,
    code: str,
    turn_index: int,
) -> list[str]:
    """Extract original assistant matches from ``items.parquet`` window row.

    Parameters
    ----------
    item_row:
        Selected items row for the chosen window.
    code:
        Annotation code key (e.g., ``bot-endorses-delusion``).
    turn_index:
        Sample turn index (``window.turnN`` -> ``N``).

    Returns
    -------
    list[str]
        Match snippets from the assistant message tied to this turn.
    """

    messages = list(item_row.get("messages", []))
    if not messages:
        return []
    matches_key = f"matches__{code}"
    candidate_indexes = [turn_index + 1] + list(range(turn_index + 2, len(messages)))
    for index in candidate_indexes:
        if index >= len(messages):
            continue
        message = messages[index]
        if str(message.get("role", "")) != "assistant":
            continue
        matches = _coerce_match_sequence(message.get(matches_key))
        if matches:
            return matches
    return []


def _coerce_annotation_match_sequence(value: Any) -> list[str]:
    """Convert one annotation parquet ``matches`` cell to a string list.

    Parameters
    ----------
    value:
        Raw matches value from ``all_annotations__matches.parquet``.

    Returns
    -------
    list[str]
        Non-empty normalized match strings.
    """

    if value is None:
        return []
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return []
        try:
            parsed = json.loads(stripped)
        except json.JSONDecodeError:
            return [stripped]
        return _coerce_match_sequence(parsed)
    return _coerce_match_sequence(value)


def load_user_annotation_matches_by_turn(
    annotations_matches_path: Path,
    item_row: pd.Series,
    user_code: str,
    turn_indexes: list[int],
) -> dict[int, list[str]]:
    """Load user-match excerpts from annotations parquet for selected turns.

    Parameters
    ----------
    annotations_matches_path:
        Path to ``all_annotations__matches.parquet``.
    item_row:
        Selected items row for the chosen window.
    user_code:
        User-side annotation code key.
    turn_indexes:
        Relative window turn indexes to resolve.

    Returns
    -------
    dict[int, list[str]]
        Mapping from relative turn index to annotation match snippets.
    """

    if not turn_indexes:
        return {}
    if not annotations_matches_path.is_file():
        raise FileNotFoundError(
            f"Missing annotations parquet: {annotations_matches_path}"
        )

    participant = str(item_row["participant"])
    source_path = str(item_row["source_rel_path"])
    chat_index = int(str(item_row["conversation_id"]).replace("chat_", ""))
    start_index = int(item_row["start_message_index"])
    absolute_indexes = sorted({start_index + int(turn) for turn in turn_indexes})

    columns = ["message_index", "score_cutoff", "matches"]
    filters: list[tuple[str, str, Any]] = [
        ("annotation_id", "==", user_code),
        ("participant", "==", participant),
        ("source_path", "==", source_path),
        ("chat_index", "==", chat_index),
        ("role", "==", "user"),
        ("message_index", "in", absolute_indexes),
    ]
    matches_df = pd.read_parquet(
        annotations_matches_path,
        columns=columns,
        filters=filters,
    )
    if matches_df.empty:
        return {}

    by_turn: dict[int, list[str]] = {}
    sorted_df = matches_df.sort_values(
        by=["message_index", "score_cutoff"], ascending=[True, False]
    )
    for row in sorted_df.itertuples(index=False):
        turn_index = int(row.message_index) - start_index
        if turn_index in by_turn:
            continue
        matches = _coerce_annotation_match_sequence(row.matches)
        if matches:
            by_turn[turn_index] = matches
    return by_turn


def infer_original_model(transcripts_path: Path, item_row: pd.Series) -> str:
    """Infer original assistant model id for the selected window.

    Parameters
    ----------
    transcripts_path:
        Path to transcripts parquet.
    item_row:
        Selected items row.

    Returns
    -------
    str
        Most frequent assistant model id in the window.
    """

    transcripts = pd.read_parquet(transcripts_path)
    participant = str(item_row["participant"])
    source_path = str(item_row["source_rel_path"])
    chat_index = int(str(item_row["conversation_id"]).replace("chat_", ""))
    start_index = int(item_row["start_message_index"])
    window_size = int(item_row["window_size"])
    end_index = start_index + window_size - 1

    assistant_rows = transcripts[
        transcripts["participant"].astype(str).eq(participant)
        & transcripts["source_path"].astype(str).eq(source_path)
        & transcripts["chat_index"].eq(chat_index)
        & transcripts["message_index"].ge(start_index)
        & transcripts["message_index"].le(end_index)
        & transcripts["role"].eq("assistant")
    ]
    if assistant_rows.empty:
        return "unknown"

    counts = assistant_rows["model_id"].dropna().astype(str).value_counts()
    if counts.empty:
        return "unknown"
    return str(counts.index[0])


def collect_sample_turn(
    sample_id: str,
    original_samples: dict[str, dict[str, Any]],
    new_samples: dict[str, dict[str, Any]],
) -> SampleTurn:
    """Collect one selected sample turn and parsed score metadata.

    Parameters
    ----------
    sample_id:
        Selected sample identifier.
    original_samples:
        Original transcript sample map.
    new_samples:
        Evaluated model sample map.

    Returns
    -------
    SampleTurn
        Structured turn payload.
    """

    original = original_samples[sample_id]
    new = new_samples[sample_id]

    history = list(new.get("history", []))
    if not history:
        raise ValueError(f"Missing history for sample {sample_id}.")
    user_text = str(history[-1].get("content", ""))

    original_raw, original_cutoff, original_matches = parse_explanation_fields(
        str(original.get("grader_explanation", ""))
    )
    new_raw, new_cutoff, new_matches = parse_explanation_fields(
        str(new.get("grader_explanation", ""))
    )

    return SampleTurn(
        sample_id=sample_id,
        turn_index=parse_turn_index(sample_id),
        history=history,
        user_text=user_text,
        original_excerpt=str(original.get("excerpt", "")),
        new_excerpt=str(new.get("excerpt", "")),
        original_score=float(original.get("score", 0) or 0),
        new_score=float(new.get("score", 0) or 0),
        original_raw_score=original_raw,
        new_raw_score=new_raw,
        original_cutoff=original_cutoff,
        new_cutoff=new_cutoff,
        original_matches=original_matches,
        new_matches=new_matches,
    )


def load_window_cutup_template() -> str:
    """Load the fixed TikZ template used for window cutup figure export.

    Returns
    -------
    str
        Raw template body text with replacement markers.
    """

    if not WINDOW_CUTUP_TEMPLATE_PATH.is_file():
        raise FileNotFoundError(
            f"Missing figure template: {WINDOW_CUTUP_TEMPLATE_PATH}"
        )
    return WINDOW_CUTUP_TEMPLATE_PATH.read_text(encoding="utf-8")


def select_template_sample_ids(
    sample_ids: list[str],
    original_samples: dict[str, dict[str, Any]],
    new_samples: dict[str, dict[str, Any]],
) -> tuple[str, str]:
    """Select two turn ids to populate the fixed A1/A2 template.

    Parameters
    ----------
    sample_ids:
        Shared sample ids for the chosen window.
    original_samples:
        Original transcript sample map.
    new_samples:
        Evaluated model sample map.

    Returns
    -------
    tuple[str, str]
        Two selected ids in ascending turn order.
    """

    if len(sample_ids) < TEMPLATE_SAMPLE_COUNT:
        raise ValueError("Template figure requires at least two shared turns.")
    ordered_ids = sorted(sample_ids, key=parse_turn_index)
    positive_score = {
        sample_id: (
            int(float(original_samples[sample_id].get("score", 0) or 0) > 0)
            + int(float(new_samples[sample_id].get("score", 0) or 0) > 0)
        )
        for sample_id in ordered_ids
    }
    new_positive = {
        sample_id: int(float(new_samples[sample_id].get("score", 0) or 0) > 0)
        for sample_id in ordered_ids
    }
    orig_positive = {
        sample_id: int(float(original_samples[sample_id].get("score", 0) or 0) > 0)
        for sample_id in ordered_ids
    }

    def _counterevidence_score(sample_id: str) -> int:
        """Score how clearly a negative sample states non-endorsement."""

        _, _, matches = parse_explanation_fields(
            str(new_samples[sample_id].get("grader_explanation", ""))
        )
        if not matches:
            return 0
        text = normalize_text(" ".join(matches)).lower()
        keywords = (
            "cant",
            "cannot",
            "dont",
            "do not",
            "not",
            "tool",
            "validate",
            "confirm",
            "shouldnt",
            "should not",
            "grounded",
            "evidence",
        )
        score = 1
        score += sum(1 for keyword in keywords if keyword in text)
        score += min(len(text) // 120, 2)
        return score

    pair_candidates: list[tuple[int, int, int, int, int, str, str]] = []
    for first_index in range(len(ordered_ids)):
        for second_index in range(first_index + 1, len(ordered_ids)):
            first_id = ordered_ids[first_index]
            second_id = ordered_ids[second_index]
            pair_total = positive_score[first_id] + positive_score[second_id]
            absent_counterevidence_score = 0
            if new_positive[first_id] == 0:
                absent_counterevidence_score += _counterevidence_score(first_id)
            if new_positive[second_id] == 0:
                absent_counterevidence_score += _counterevidence_score(second_id)
            pair_candidates.append(
                (
                    pair_total,
                    new_positive[first_id],
                    new_positive[second_id],
                    orig_positive[first_id] + orig_positive[second_id],
                    absent_counterevidence_score,
                    first_id,
                    second_id,
                )
            )

    best_pair = sorted(
        pair_candidates,
        key=lambda item: (
            -item[0],
            -item[1],
            -item[2],
            -item[3],
            -item[4],
            parse_turn_index(item[5]),
            parse_turn_index(item[6]),
            item[5],
            item[6],
        ),
    )[0]
    behavior_total = best_pair[0]
    if behavior_total < MIN_BEHAVIOR_PRESENT_OUTCOMES:
        raise ValueError(
            "Template figure requires at least three behavior-present outcomes "
            "across A1, A2, A'_1, and A'_2."
        )
    return best_pair[5], best_pair[6]


def select_preferred_template_sample_ids(
    *,
    code: str,
    model_id: str,
    reasoning_effort: str | None,
    original_samples: dict[str, dict[str, Any]],
    new_samples: dict[str, dict[str, Any]],
) -> tuple[str, str] | None:
    """Return a preferred sample pair when one is explicitly configured.

    Parameters
    ----------
    code:
        Annotation code.
    model_id:
        Evaluated model identifier.
    reasoning_effort:
        Reasoning effort setting.
    original_samples:
        Original transcript sample map.
    new_samples:
        Evaluated model sample map.

    Returns
    -------
    tuple[str, str] | None
        Preferred sample ids when available, otherwise ``None``.
    """

    preferred_ids = PREFERRED_TEMPLATE_SAMPLE_IDS.get(
        (code, model_id, reasoning_effort)
    )
    if preferred_ids is None:
        return None
    if not all(
        sample_id in original_samples and sample_id in new_samples
        for sample_id in preferred_ids
    ):
        return None
    return preferred_ids


def behavior_label(score: float) -> str:
    """Render behavior label text from one binarized score.

    Parameters
    ----------
    score:
        Binarized behavior score.

    Returns
    -------
    str
        Label for present or absent behavior.
    """

    if score > 0:
        return r"Behavior\\Present\\[-0.2ex]$\checkmark$"
    return r"Behavior\\Absent\\[-0.2ex]$\times$"


def extract_match_text(
    sample_turn: SampleTurn,
    *,
    use_original: bool,
    max_chars: int,
) -> str:
    """Build one rendered message string from grader evidence matches.

    Parameters
    ----------
    sample_turn:
        Selected sample turn payload.
    use_original:
        Whether to draw from original-model fields instead of evaluated fields.
    max_chars:
        Max output length in characters.

    Returns
    -------
    str
        Normalized and truncated match text, or ``...`` when unavailable.
    """

    if use_original:
        matches = sample_turn.original_matches
    else:
        matches = sample_turn.new_matches

    normalized_matches = [
        normalize_text(item) for item in matches if normalize_text(item)
    ]
    if not normalized_matches:
        return "..."
    source_text = " ... ".join(normalized_matches)
    return truncate_text(normalize_text(source_text), max_chars)


def extract_assistant_excerpt(
    sample_turn: SampleTurn,
    *,
    use_original: bool,
    max_chars: int,
) -> str:
    """Build one rendered assistant excerpt from the raw response text.

    The figure should show actual response text rather than stitched match
    snippets. When grader matches are available, this anchors the excerpt on the
    first matched span so the quote stays relevant while preserving punctuation
    and quote marks from the original response.

    Parameters
    ----------
    sample_turn:
        Selected sample turn payload.
    use_original:
        Whether to draw from original-model fields instead of evaluated fields.
    max_chars:
        Max output length in characters.

    Returns
    -------
    str
        Normalized and truncated assistant excerpt.
    """

    if use_original:
        source_text = sample_turn.original_excerpt
        matches = sample_turn.original_matches
    else:
        source_text = sample_turn.new_excerpt
        matches = sample_turn.new_matches

    collapsed_source = collapse_whitespace(source_text)
    if not collapsed_source:
        return extract_match_text(
            sample_turn,
            use_original=use_original,
            max_chars=max_chars,
        )

    for match in matches:
        collapsed_match = collapse_whitespace(match)
        if not collapsed_match:
            continue
        start_index = collapsed_source.lower().find(collapsed_match.lower())
        if start_index == -1:
            continue
        snippet = collapsed_source[start_index:]
        if start_index > 0:
            snippet = "... " + snippet
        return truncate_text(snippet, max_chars)

    return truncate_text(collapsed_source, max_chars)


def extract_user_match_text(
    sample_turn: SampleTurn,
    item_row: pd.Series,
    user_matches_by_turn: dict[int, list[str]],
    max_chars: int,
) -> str:
    """Build one rendered user-message string from user code matches.

    Parameters
    ----------
    sample_turn:
        Selected sample turn payload.
    item_row:
        Selected items row for the chosen window.
    user_matches_by_turn:
        User annotation matches keyed by relative turn index.
    max_chars:
        Max output length in characters.

    Returns
    -------
    str
        Normalized and truncated user text.
    """

    matches = user_matches_by_turn.get(sample_turn.turn_index, [])
    normalized_matches = [
        normalize_text(item) for item in matches if normalize_text(item)
    ]
    if normalized_matches:
        return truncate_text(" ... ".join(normalized_matches), max_chars)

    messages = list(item_row.get("messages", []))
    if 0 <= sample_turn.turn_index < len(messages):
        message = messages[sample_turn.turn_index]
        if str(message.get("role", "")) == "user":
            content = normalize_text(str(message.get("content", "")))
            if content:
                return truncate_text(content, max_chars)

    fallback_text = normalize_text(sample_turn.user_text)
    if fallback_text:
        return truncate_text(fallback_text, max_chars)
    return "..."


def render_tikz_figure_body(
    code: str,
    model_display: str,
    original_model_id: str,
    payload: FigureTemplatePayload,
) -> str:
    """Render figure body from fixed LaTeX template and substitutions.

    Parameters
    ----------
    code:
        Annotation code label.
    model_display:
        Evaluated model display label.
    original_model_id:
        Original transcript model id or label.
    payload:
        Grouped template content values.

    Returns
    -------
    str
        TikZ body with placeholders replaced.
    """
    original_a1_excerpt, original_a2_excerpt = payload.original_excerpt_pairs
    evaluated_a1_excerpt, evaluated_a2_excerpt = payload.evaluated_excerpt_pairs
    (
        original_a1_behavior,
        original_a2_behavior,
        evaluated_a1_behavior,
        evaluated_a2_behavior,
    ) = payload.behavior_labels
    user_u1_match, user_u2_match = payload.user_match_pairs

    replacements = {
        "__ANNOTATION_CODE__": escape_latex(code),
        "__ORIGINAL_MODEL_VALUE__": escape_latex(original_model_id),
        "__EVALUATED_MODEL_VALUE__": escape_latex(model_display),
        "__USER_U1_MATCH__": escape_latex(user_u1_match),
        "__USER_U2_MATCH__": escape_latex(user_u2_match),
        "__ORIGINAL_A1_MATCH__": escape_latex(original_a1_excerpt),
        "__ORIGINAL_A2_MATCH__": escape_latex(original_a2_excerpt),
        "__EVALUATED_A1_MATCH__": escape_latex(evaluated_a1_excerpt),
        "__EVALUATED_A2_MATCH__": escape_latex(evaluated_a2_excerpt),
        "__ORIGINAL_A1_BEHAVIOR__": original_a1_behavior,
        "__ORIGINAL_A2_BEHAVIOR__": original_a2_behavior,
        "__EVALUATED_A1_BEHAVIOR__": evaluated_a1_behavior,
        "__EVALUATED_A2_BEHAVIOR__": evaluated_a2_behavior,
    }

    rendered = load_window_cutup_template()
    for marker, value in replacements.items():
        rendered = rendered.replace(marker, value)
    return rendered


def write_standalone_tikz_pdf(
    tikz_body: str,
    output_figure_tex: Path,
    output_pdf: Path,
) -> None:
    """Write standalone TikZ source and compile it into a PDF asset.

    Parameters
    ----------
    tikz_body:
        TikZ figure body.
    output_figure_tex:
        Path for saved standalone source.
    output_pdf:
        Target compiled PDF path.

    Returns
    -------
    None
        Writes source and compiled PDF.
    """

    document = "\n".join(
        [
            "\\documentclass[tikz,border=0pt]{standalone}",
            "\\usepackage[utf8]{inputenc}",
            "\\usepackage[dvipsnames]{xcolor}",
            "\\usepackage{amssymb}",
            "\\usepackage{tikz}",
            "\\usetikzlibrary{positioning,arrows.meta}",
            "\\begin{document}",
            tikz_body.rstrip(),
            "\\end{document}",
            "",
        ]
    )

    output_figure_tex.parent.mkdir(parents=True, exist_ok=True)
    output_figure_tex.write_text(document, encoding="utf-8")

    with tempfile.TemporaryDirectory(prefix="window_cutup_fig_") as temp_dir_str:
        temp_dir = Path(temp_dir_str)
        temp_tex = temp_dir / "figure.tex"
        temp_tex.write_text(document, encoding="utf-8")

        command = [
            "pdflatex",
            "-interaction=nonstopmode",
            "-halt-on-error",
            "figure.tex",
        ]
        result = subprocess.run(
            command,
            cwd=temp_dir,
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            raise RuntimeError(
                "Failed to compile standalone figure PDF.\n"
                f"stdout:\n{result.stdout}\n\n"
                f"stderr:\n{result.stderr}"
            )

        compiled_pdf = temp_dir / "figure.pdf"
        if not compiled_pdf.exists():
            raise RuntimeError("Standalone compile succeeded but figure.pdf missing.")

        output_pdf.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(compiled_pdf, output_pdf)


def main() -> None:
    """Run end-to-end generation for the replay figure.

    Returns
    -------
    None
        Writes standalone source and compiled PDF.
    """

    args = parse_args()

    samples_root = Path(args.samples_root)
    new_path = sample_json_path(
        samples_root=samples_root,
        model_id=args.model_id,
        reasoning_effort=args.reasoning_effort,
        code=args.code,
    )
    original_path = sample_json_path(
        samples_root=samples_root,
        model_id="original_transcript",
        reasoning_effort=None,
        code=args.code,
    )

    new_samples = load_sample_map(new_path)
    original_samples = load_sample_map(original_path)

    preferred_sample_ids = select_preferred_template_sample_ids(
        code=args.code,
        model_id=args.model_id,
        reasoning_effort=args.reasoning_effort,
        original_samples=original_samples,
        new_samples=new_samples,
    )
    if preferred_sample_ids is not None:
        first_sample_id, second_sample_id = preferred_sample_ids
        window_id = first_sample_id.split(".turn", maxsplit=1)[0]
    else:
        window_id, sample_ids = choose_window_and_shared_samples(
            original_samples=original_samples,
            new_samples=new_samples,
        )
        first_sample_id, second_sample_id = select_template_sample_ids(
            sample_ids,
            original_samples=original_samples,
            new_samples=new_samples,
        )
    first_turn = collect_sample_turn(
        sample_id=first_sample_id,
        original_samples=original_samples,
        new_samples=new_samples,
    )
    second_turn = collect_sample_turn(
        sample_id=second_sample_id,
        original_samples=original_samples,
        new_samples=new_samples,
    )

    item_row = find_items_row(Path(args.items_parquet), args.code, window_id)
    original_model_id = infer_original_model(Path(args.transcripts_parquet), item_row)
    user_matches_by_turn = load_user_annotation_matches_by_turn(
        annotations_matches_path=Path(args.annotations_matches_parquet),
        item_row=item_row,
        user_code=args.user_code,
        turn_indexes=[first_turn.turn_index, second_turn.turn_index],
    )
    original_first_excerpt = extract_assistant_excerpt(
        first_turn,
        use_original=True,
        max_chars=230,
    )
    original_second_excerpt = extract_assistant_excerpt(
        second_turn,
        use_original=True,
        max_chars=220,
    )
    evaluated_first_excerpt = extract_assistant_excerpt(
        first_turn,
        use_original=False,
        max_chars=260,
    )
    evaluated_second_excerpt = extract_assistant_excerpt(
        second_turn,
        use_original=False,
        max_chars=240,
    )
    user_first_match = extract_user_match_text(
        sample_turn=first_turn,
        item_row=item_row,
        user_matches_by_turn=user_matches_by_turn,
        max_chars=180,
    )
    user_second_match = extract_user_match_text(
        sample_turn=second_turn,
        item_row=item_row,
        user_matches_by_turn=user_matches_by_turn,
        max_chars=180,
    )
    behavior_labels = (
        behavior_label(first_turn.original_score),
        behavior_label(second_turn.original_score),
        behavior_label(first_turn.new_score),
        behavior_label(second_turn.new_score),
    )

    tikz_body = render_tikz_figure_body(
        code=args.code,
        model_display=args.model_display,
        original_model_id=original_model_id,
        payload=FigureTemplatePayload(
            original_excerpt_pairs=(original_first_excerpt, original_second_excerpt),
            evaluated_excerpt_pairs=(
                evaluated_first_excerpt,
                evaluated_second_excerpt,
            ),
            behavior_labels=behavior_labels,
            user_match_pairs=(user_first_match, user_second_match),
        ),
    )

    output_figure_tex = Path(args.output_figure_tex)
    output_pdf = Path(args.output_pdf)

    write_standalone_tikz_pdf(
        tikz_body=tikz_body,
        output_figure_tex=output_figure_tex,
        output_pdf=output_pdf,
    )

    print(f"Wrote standalone source: {output_figure_tex}")
    print(f"Wrote PDF asset: {output_pdf}")
    print(f"Selected window: {window_id}")
    print(f"Template samples: {first_sample_id}, {second_sample_id}")
    print(f"Original model inferred from transcripts: {original_model_id}")


if __name__ == "__main__":
    main()
