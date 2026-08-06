"""Export top-ranked eval traces to markdown files.

This script reads row-level eval data from ``report/eval_rows.parquet``,
selects top-ranked rows for one or more annotation codes, and enriches each
row with the original conversation, model reasoning trace, model output text,
and grader evidence pulled from a ``.eval`` zip log.
"""

from __future__ import annotations

import argparse
import json
import zipfile
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

DEFAULT_ROWS_PATH = Path("report/eval_rows.parquet")
DEFAULT_OUTPUT_DIR = Path("analysis/data")
DEFAULT_MODEL = "together/Qwen/Qwen3.5-397B-A17B"
DEFAULT_REASONING = "high"


@dataclass(frozen=True)
class ExportSettings:
    """Settings used to export markdown trace packs.

    Parameters
    ----------
    n:
        Number of examples to export per code.
    rank_by:
        Ranking strategy.
    reference_score:
        Reference score for distance-based ranking.
    model:
        Model id used for row filtering and metadata.
    reasoning_effort:
        Reasoning effort value used for filtering and metadata.
    log_path:
        Path to the source ``.eval`` zip log.
    output_dir:
        Directory where markdown outputs are written.
    """

    n: int
    rank_by: str
    reference_score: float
    model: str
    reasoning_effort: str
    log_path: Path
    output_dir: Path


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Parameters
    ----------
    None

    Returns
    -------
    argparse.Namespace
        Parsed command-line options.
    """
    parser = argparse.ArgumentParser(
        description="Export top-ranked samples with reasoning traces."
    )
    parser.add_argument(
        "--code",
        action="append",
        required=True,
        help=(
            "Annotation code to export. Repeat for multiple codes. "
            "Accepts values with or without the 'bot-' prefix."
        ),
    )
    parser.add_argument(
        "--n",
        type=int,
        default=10,
        help="Number of ranked samples to export per code.",
    )
    parser.add_argument(
        "--rank-by",
        choices=("raw_score_desc", "abs_distance_from_reference"),
        default="raw_score_desc",
        help=(
            "Ranking mode: raw score descending, or absolute distance from a "
            "reference score."
        ),
    )
    parser.add_argument(
        "--reference-score",
        type=float,
        default=5.0,
        help="Reference score for abs_distance_from_reference ranking.",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help="Model id filter from eval rows.",
    )
    parser.add_argument(
        "--reasoning-effort",
        default=DEFAULT_REASONING,
        help="Reasoning effort filter from eval rows.",
    )
    parser.add_argument(
        "--rows-path",
        type=Path,
        default=DEFAULT_ROWS_PATH,
        help="Path to eval rows parquet.",
    )
    parser.add_argument(
        "--log-path",
        type=Path,
        required=True,
        help="Path to .eval log zip used to fetch sample payloads.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for markdown outputs.",
    )
    return parser.parse_args()


def _normalize_codes(raw_codes: list[str]) -> list[str]:
    """Normalize code arguments to canonical ``bot-*`` values.

    Parameters
    ----------
    raw_codes:
        Values passed by the user, possibly comma-separated.

    Returns
    -------
    list[str]
        Ordered deduplicated codes with a ``bot-`` prefix.
    """
    normalized: list[str] = []
    seen: set[str] = set()
    for raw_code in raw_codes:
        for part in raw_code.split(","):
            code = part.strip()
            if not code:
                continue
            if not code.startswith("bot-"):
                code = f"bot-{code}"
            if code in seen:
                continue
            seen.add(code)
            normalized.append(code)
    return normalized


def _extract_text(content: object) -> str:
    """Extract plain text from Inspect content payload.

    Parameters
    ----------
    content:
        Message content, either a string or a structured list.

    Returns
    -------
    str
        Extracted text.
    """
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""

    parts: list[str] = []
    for item in content:
        if not isinstance(item, dict):
            continue
        item_type = item.get("type")
        if item_type == "text":
            text = item.get("text")
            if isinstance(text, str):
                parts.append(text)
        elif item_type == "reasoning":
            reasoning = item.get("reasoning")
            if isinstance(reasoning, str):
                parts.append(reasoning)
    return "\n\n".join(parts)


def _extract_reasoning_and_text(content: object) -> tuple[str, str]:
    """Split model output into reasoning trace and final text.

    Parameters
    ----------
    content:
        Generated assistant content payload.

    Returns
    -------
    tuple[str, str]
        ``(reasoning_trace, final_response_text)``.
    """
    if isinstance(content, str):
        return "", content
    if not isinstance(content, list):
        return "", ""

    reasoning_parts: list[str] = []
    text_parts: list[str] = []
    for item in content:
        if not isinstance(item, dict):
            continue
        item_type = item.get("type")
        if item_type == "reasoning":
            reasoning = item.get("reasoning")
            if isinstance(reasoning, str):
                reasoning_parts.append(reasoning)
        elif item_type == "text":
            text = item.get("text")
            if isinstance(text, str):
                text_parts.append(text)
    return "\n\n".join(reasoning_parts), "\n\n".join(text_parts)


def _render_input_messages(messages: object) -> str:
    """Render conversation input messages.

    Parameters
    ----------
    messages:
        Message list from the sample payload.

    Returns
    -------
    str
        Role-labeled markdown text.
    """
    if not isinstance(messages, list):
        return "(No input messages found)"

    lines: list[str] = []
    for idx, msg in enumerate(messages, start=1):
        if not isinstance(msg, dict):
            continue
        role = str(msg.get("role", "unknown")).upper()
        text = _extract_text(msg.get("content"))
        lines.append(f"{idx}. [{role}]\n{text}")
    return "\n\n".join(lines) if lines else "(No input messages found)"


def _render_quotes(score_answer_raw: object) -> str:
    """Render grader quote bullets from score payload.

    Parameters
    ----------
    score_answer_raw:
        Raw ``answer`` field from the scorer payload.

    Returns
    -------
    str
        Markdown bullets or fallback text.
    """
    if not isinstance(score_answer_raw, str):
        return "(No scorer quotes available)"

    try:
        parsed = json.loads(score_answer_raw)
    except json.JSONDecodeError:
        return "(Could not parse scorer answer JSON)"

    quotes = parsed.get("quotes")
    if not isinstance(quotes, list):
        return "(No scorer quotes available)"

    out_lines = [f"- {quote}" for quote in quotes if isinstance(quote, str)]
    return "\n".join(out_lines) if out_lines else "(No scorer quotes available)"


def _rank_rows(
    rows: pd.DataFrame, rank_by: str, reference_score: float
) -> pd.DataFrame:
    """Return rows with rank helper columns and ordering.

    Parameters
    ----------
    rows:
        Candidate rows for a single code.
    rank_by:
        Ranking mode.
    reference_score:
        Reference score for absolute-distance ranking.

    Returns
    -------
    pd.DataFrame
        Sorted rows with added helper columns.
    """
    ranked = rows.copy()
    ranked["raw_score_num"] = pd.to_numeric(ranked["raw_score"], errors="coerce")

    if rank_by == "abs_distance_from_reference":
        ranked["rank_metric"] = (ranked["raw_score_num"] - reference_score).abs()
        ranked = ranked.sort_values(
            ["rank_metric", "raw_score_num", "sample_id"],
            ascending=[False, False, True],
        )
        return ranked

    ranked["rank_metric"] = ranked["raw_score_num"]
    return ranked.sort_values(
        ["rank_metric", "sample_id"],
        ascending=[False, True],
    )


def _build_output_path(
    output_dir: Path, code: str, n: int, rank_by: str, reasoning_effort: str
) -> Path:
    """Build markdown output path for one code.

    Parameters
    ----------
    output_dir:
        Output directory.
    code:
        Annotation code id.
    n:
        Number of examples requested.
    rank_by:
        Ranking mode.
    reasoning_effort:
        Reasoning effort value used in filtering.

    Returns
    -------
    Path
        Path where markdown will be written.
    """
    short_code = code.removeprefix("bot-")
    rank_short = "raw" if rank_by == "raw_score_desc" else "absref"
    file_name = f"trace_pack_{short_code}_top{n}_{rank_short}_{reasoning_effort}.md"
    return output_dir / file_name


def _to_int_if_numeric(value: object) -> object:
    """Convert a numeric-like value to int when possible.

    Parameters
    ----------
    value:
        Any value.

    Returns
    -------
    object
        Integer for numeric inputs, unchanged otherwise.
    """
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return value


def export_code_markdown(
    rows: pd.DataFrame,
    code: str,
    settings: ExportSettings,
) -> Path:
    """Export one code-specific markdown trace pack.

    Parameters
    ----------
    rows:
        Full filtered rows for the chosen model/reasoning.
    code:
        Annotation code id.
    settings:
        Export configuration values.

    Returns
    -------
    Path
        Written markdown path.
    """
    code_rows = rows[rows["annotation_id"] == code].copy()
    ranked = _rank_rows(code_rows, settings.rank_by, settings.reference_score)
    top_rows = ranked.head(settings.n).copy()
    output_path = _build_output_path(
        settings.output_dir,
        code,
        settings.n,
        settings.rank_by,
        settings.reasoning_effort,
    )

    lines: list[str] = [
        f"# Trace pack: {code}",
        "",
        f"- Model: `{settings.model}`",
        f"- Reasoning effort: `{settings.reasoning_effort}`",
        f"- Code: `{code}`",
        f"- Ranking mode: `{settings.rank_by}`",
        f"- Reference score: `{settings.reference_score}`",
        f"- Source log: `{settings.log_path}`",
        f"- Total matching rows: `{len(code_rows)}`",
        f"- Exported rows: `{len(top_rows)}`",
        "",
    ]

    with zipfile.ZipFile(settings.log_path, "r") as archive:
        archive_names = set(archive.namelist())
        for rank, row in enumerate(top_rows.itertuples(index=False), start=1):
            sample_id = str(row.sample_id)
            sample_path = f"samples/{sample_id}_epoch_1.json"

            if sample_path not in archive_names:
                lines.extend(
                    [
                        f"## {rank}. `{sample_id}`",
                        "",
                        f"Missing sample payload: `{sample_path}`",
                        "",
                    ]
                )
                continue

            sample_json = json.loads(archive.read(sample_path))
            input_messages = sample_json.get("input", [])

            output_block = sample_json.get("output", {})
            choices = output_block.get("choices", [])
            first_choice = choices[0] if isinstance(choices, list) and choices else {}
            message = first_choice.get("message", {})
            reasoning_trace, final_response = _extract_reasoning_and_text(
                message.get("content")
            )

            score_payload = sample_json.get("scores", {}).get(
                "metadata_annotation_scorer", {}
            )
            score_answer = (
                score_payload.get("answer", "")
                if isinstance(score_payload, dict)
                else ""
            )
            score_explanation = (
                score_payload.get("explanation", "")
                if isinstance(score_payload, dict)
                else ""
            )

            lines.extend(
                [
                    f"## {rank}. `{sample_id}`",
                    "",
                    f"- window_id: `{row.window_id}`",
                    f"- turn_index: `{_to_int_if_numeric(row.turn_index)}`",
                    f"- raw_score: `{row.raw_score}`",
                    f"- binarized score: `{_to_int_if_numeric(row.score)}`",
                    f"- rank_metric: `{row.rank_metric}`",
                    "",
                    "### Original Conversation (Input to model)",
                    "",
                    _render_input_messages(input_messages),
                    "",
                    "### Model Reasoning Trace",
                    "",
                    "```text",
                    (
                        reasoning_trace
                        if reasoning_trace
                        else "(No reasoning trace found)"
                    ),
                    "```",
                    "",
                    "### Model Final Response",
                    "",
                    "```text",
                    final_response if final_response else "(No final text found)",
                    "```",
                    "",
                    "### Grader Evidence (score answer quotes)",
                    "",
                    _render_quotes(score_answer),
                    "",
                    "### Grader Explanation Payload",
                    "",
                    "```json",
                    score_explanation if isinstance(score_explanation, str) else "",
                    "```",
                    "",
                ]
            )

    settings.output_dir.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")
    return output_path


def main() -> None:
    """Run trace-pack export from command line.

    Parameters
    ----------
    None

    Returns
    -------
    None
    """
    args = parse_args()
    codes = _normalize_codes(args.code)
    if not codes:
        raise ValueError("No valid codes were provided.")

    if args.n < 1:
        raise ValueError("--n must be >= 1.")

    rows = pd.read_parquet(args.rows_path)
    filtered_rows = rows[
        (rows["model"] == args.model)
        & (rows["reasoning_effort"] == args.reasoning_effort)
    ].copy()
    settings = ExportSettings(
        n=args.n,
        rank_by=args.rank_by,
        reference_score=args.reference_score,
        model=args.model,
        reasoning_effort=args.reasoning_effort,
        log_path=args.log_path,
        output_dir=args.output_dir,
    )

    for code in codes:
        output_path = export_code_markdown(
            rows=filtered_rows,
            code=code,
            settings=settings,
        )
        print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
