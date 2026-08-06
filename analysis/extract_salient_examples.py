"""Extract salient eval examples for paper writing and LaTeX paste-in.

This script finds the strongest "back and forth" windows in
``report/eval_rows.parquet`` by category, then pulls the full
conversation window from ``subsets/items_sanitized.parquet`` so the
output includes the complete user/assistant exchange.

Outputs:
- a CSV with one row per selected window
- a LaTeX fragment with the same windows in a paste-ready format

The default selection is one top window per category.  For each selected
window, the export includes the full conversation plus a compact summary
of the high-scoring eval rows in that window.
"""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import json
import textwrap
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import pandas as pd
from pylatexenc.latexencode import unicode_to_latex

SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parents[1]

from analysis.artifact_paths import DATA_DIR, TABLE_DIR, ensure_output_dirs
from analysis.load_eval_data import _EVALS_REPO_ROOT, CODE_CATEGORIES
from llm_delusion_eval.participant_exclusions import resolve_items_sanitized_path

DEFAULT_ROWS_PATH = _EVALS_REPO_ROOT / "report" / "eval_rows.parquet"
DEFAULT_SUMMARY_PATH = _EVALS_REPO_ROOT / "report" / "summary.json"
DEFAULT_CSV_PATH = DATA_DIR / "salient_windows.csv"
DEFAULT_TEX_PATH = TABLE_DIR / "salient_windows.tex"

CATEGORY_ORDER = ["sycophancy", "relationship", "delusional", "concerns harm"]
CODE_EXAMPLES_SECTION_TITLE = "Programmatically Selected Qualitative Examples by Code"
CODE_EXAMPLES_SECTION_LABEL = "app:salient_code_examples"
CODE_EXAMPLE_TEXT_BLACKLISTS: dict[str, tuple[str, ...]] = {
    "discourages-violence": ("spank", "spanking", "son", "child", "kid"),
}


def _is_validates_code(code: str) -> bool:
    """Return whether a code belongs to the ``validates-*`` family."""
    return str(code).strip().removeprefix("bot-").startswith("validates-")


CODE_ORDER = [
    code
    for category in CATEGORY_ORDER
    for code, code_category in CODE_CATEGORIES.items()
    if code_category == category and not _is_validates_code(code)
]


def _escape_latex(text: Any) -> str:
    """Escape text for a LaTeX fragment.

    This escape route converts Unicode punctuation to LaTeX-safe sequences and
    replaces unmapped Unicode characters with ``?`` so exported fragments avoid
    raw Unicode.
    """
    if text is None:
        return ""
    return unicode_to_latex(
        str(text),
        unknown_char_policy=lambda _char: "?",
        unknown_char_warning=False,
    )


def _escape_latex_monospace(text: Any) -> str:
    """Escape text for monospace LaTeX output without smart quotes."""
    if text is None:
        return ""

    value = str(text)
    value = value.replace("“", '"').replace("”", '"')
    value = value.replace("‘", "'").replace("’", "'")
    value = value.replace("—", "--").replace("–", "-")
    value = value.replace("…", "...")
    for char, repl in [
        ("\\", r"\textbackslash{}"),
        ("&", r"\&"),
        ("%", r"\%"),
        ("$", r"\$"),
        ("#", r"\#"),
        ("_", r"\_"),
        ("{", r"\{"),
        ("}", r"\}"),
    ]:
        value = value.replace(char, repl)
    return value


def _truncate(text: Any, limit: int) -> str:
    """Truncate text to a fixed character budget with ellipsis."""
    if text is None:
        return ""
    value = str(text).strip()
    if len(value) <= limit:
        return value
    return textwrap.shorten(value, width=limit, placeholder=" ...")


@dataclass(frozen=True)
class SelectedWindow:
    category: str
    window_id: str
    n_pos: int
    n_models: int
    n_codes: int
    turn_min: int
    turn_max: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract salient back-and-forth eval examples."
    )
    parser.add_argument(
        "--rows-path",
        type=Path,
        default=DEFAULT_ROWS_PATH,
        help=f"Path to eval_rows.parquet (default: {DEFAULT_ROWS_PATH})",
    )
    parser.add_argument(
        "--summary-path",
        type=Path,
        default=DEFAULT_SUMMARY_PATH,
        help=f"Path to report/summary.json (default: {DEFAULT_SUMMARY_PATH})",
    )
    parser.add_argument(
        "--csv-out",
        type=Path,
        default=DEFAULT_CSV_PATH,
        help=f"Output CSV path (default: {DEFAULT_CSV_PATH})",
    )
    parser.add_argument(
        "--tex-out",
        type=Path,
        default=DEFAULT_TEX_PATH,
        help=f"Output LaTeX fragment path (default: {DEFAULT_TEX_PATH})",
    )
    parser.add_argument(
        "--examples-per-category",
        type=int,
        default=1,
        help="Number of windows to keep per category.",
    )
    parser.add_argument(
        "--group-by",
        choices=("category", "code"),
        default="category",
        help="Choose whether to extract one example per category window or per code.",
    )
    parser.add_argument(
        "--rows-per-window",
        type=int,
        default=6,
        help=(
            "How many positive eval rows to surface per selected window in the summary."
        ),
    )
    parser.add_argument(
        "--excerpt-chars",
        type=int,
        default=600,
        help="Character budget for summary snippets in the CSV/LaTeX output.",
    )
    parser.add_argument(
        "--items-path",
        type=str,
        default="",
        help=(
            "Path or HF source for items_sanitized.parquet. "
            "Defaults to LLM_DELUSIONS_ITEMS_SANITIZED_PATH or shared defaults."
        ),
    )
    return parser.parse_args()


def load_summary(summary_path: Path) -> dict[str, Any]:
    with summary_path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def build_sample_index(summary: dict[str, Any]) -> dict[tuple[str, str], Path]:
    """Map (model_label, code_short) -> sample JSON path."""
    index: dict[tuple[str, str], Path] = {}
    for ev in summary.get("evaluations", []):
        model_label = ev.get("model_label")
        sample_paths = ev.get("sample_paths", {})
        if not model_label or not isinstance(sample_paths, dict):
            continue
        for annotation_id, rel_path in sample_paths.items():
            if not isinstance(annotation_id, str) or not annotation_id.startswith(
                "bot-"
            ):
                continue
            code_short = annotation_id.removeprefix("bot-")
            index[(model_label, code_short)] = _EVALS_REPO_ROOT / "report" / rel_path
    return index


@lru_cache(maxsize=None)
def load_sample_entries(sample_path: Path) -> list[dict[str, Any]]:
    """Load a per-model/per-code sample JSON file."""
    with sample_path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, list):
        raise ValueError(f"Sample file must contain a list: '{sample_path}'")
    return [dict(item) for item in data if isinstance(item, dict)]


def find_sample_entry(sample_path: Path, sample_id: str) -> dict[str, Any] | None:
    """Return one exact sample entry from a sample JSON file."""
    for entry in load_sample_entries(sample_path):
        if str(entry.get("sample_id")) == sample_id:
            return entry
    return None


def load_window_messages(items_path: Path) -> dict[str, dict[str, Any]]:
    """Load the sanitized full-window transcripts keyed by window id."""
    frame = pd.read_parquet(items_path, columns=["eval_subset_id", "label", "messages"])
    frame = frame.drop_duplicates(subset=["eval_subset_id"]).copy()
    return {
        str(row.eval_subset_id): {
            "label": str(row.label),
            "messages": row.messages,
        }
        for row in frame.itertuples(index=False)
    }


def select_windows(
    df: pd.DataFrame, examples_per_category: int
) -> list[SelectedWindow]:
    """Select the top windows by category."""
    positive = df[df["score"] > 0].copy()
    summary = (
        positive.groupby(["category", "window_id"])
        .agg(
            n_pos=("score", "size"),
            n_models=("model_label", "nunique"),
            n_codes=("code_short", "nunique"),
            turn_min=("turn_index", "min"),
            turn_max=("turn_index", "max"),
        )
        .reset_index()
    )
    summary = summary[summary["category"].isin(CATEGORY_ORDER)].copy()
    summary["_cat_rank"] = summary["category"].map(
        {cat: idx for idx, cat in enumerate(CATEGORY_ORDER)}
    )
    summary = summary.sort_values(
        ["_cat_rank", "n_pos", "n_models", "n_codes", "turn_max"],
        ascending=[True, False, False, False, False],
    )

    selected: list[SelectedWindow] = []
    for category in CATEGORY_ORDER:
        cat_rows = summary[summary["category"] == category].head(examples_per_category)
        for row in cat_rows.itertuples(index=False):
            selected.append(
                SelectedWindow(
                    category=row.category,
                    window_id=row.window_id,
                    n_pos=int(row.n_pos),
                    n_models=int(row.n_models),
                    n_codes=int(row.n_codes),
                    turn_min=int(row.turn_min),
                    turn_max=int(row.turn_max),
                )
            )
    return selected


def select_code_examples(df: pd.DataFrame) -> pd.DataFrame:
    """Select the strongest positive row for each code across all models."""
    positive = df[df["score"] > 0].copy()
    output_rows: list[dict[str, Any]] = []

    for code_short in CODE_ORDER:
        if _is_validates_code(code_short):
            continue
        code_df = positive[positive["code_short"] == code_short].copy()
        if code_df.empty:
            continue
        code_df = code_df.sort_values(
            ["raw_score", "score", "turn_index", "model_label", "window_id"],
            ascending=[False, False, True, True, True],
            na_position="last",
        )
        row = code_df.iloc[0]
        output_rows.append(
            {
                "category": str(row["category"]),
                "code_short": str(row["code_short"]),
                "model_label": str(row["model_label"]),
                "window_id": str(row["window_id"]),
                "sample_id": str(row["sample_id"]),
                "turn_index": int(row["turn_index"]),
                "score": float(row["score"]),
                "raw_score": (
                    float(row["raw_score"]) if pd.notna(row["raw_score"]) else None
                ),
            }
        )

    return pd.DataFrame(output_rows)


def _example_contains_blacklisted_text(
    sample_entry: dict[str, Any], *, code_short: str
) -> bool:
    """Return whether a sample entry contains text we want to avoid."""
    text_parts: list[str] = []

    for field_name in ("excerpt", "grader_answer", "grader_explanation"):
        value = sample_entry.get(field_name, "")
        if value:
            text_parts.append(str(value))

    history = sample_entry.get("history", [])
    if isinstance(history, list):
        for message in history:
            if isinstance(message, dict):
                content = message.get("content", "")
            else:
                content = str(message)
            if content:
                text_parts.append(str(content))

    haystack = "\n".join(text_parts).lower()
    blacklist = CODE_EXAMPLE_TEXT_BLACKLISTS.get(code_short, ())
    return any(term in haystack for term in blacklist)


def _format_message_lines(messages: list[dict[str, Any]], limit: int) -> list[str]:
    """Format a full window as LaTeX-ready lines."""
    lines: list[str] = []
    for idx, msg in enumerate(messages):
        role = str(msg.get("role", "message")).title()
        content = _truncate(msg.get("content", ""), limit)
        lines.append(f"{role} {idx}: {content}")
    return lines


def _format_last_prompt(history: list[dict[str, Any]], limit: int) -> str:
    """Return the final prompt in a sample history, if present."""
    if not history:
        return ""
    last_msg = history[-1]
    role = str(last_msg.get("role", "message")).title()
    content = _truncate(last_msg.get("content", ""), limit)
    return f"{role}: {content}"


def build_selected_examples(
    df: pd.DataFrame,
    items_map: dict[str, dict[str, Any]],
    selected_windows: list[SelectedWindow],
    rows_per_window: int,
    excerpt_chars: int,
) -> pd.DataFrame:
    """Build a tidy table of representative example windows."""
    positive = df[df["score"] > 0].copy()
    output_rows: list[dict[str, Any]] = []

    for window in selected_windows:
        window_df = positive[positive["window_id"] == window.window_id].copy()
        full_window = items_map.get(window.window_id, {})
        messages = full_window.get("messages", [])
        messages_list = [
            (
                dict(msg)
                if isinstance(msg, dict)
                else {"role": "message", "content": str(msg)}
            )
            for msg in list(messages)
        ]
        positive_rows = (
            window_df.sort_values(
                ["turn_index", "raw_score", "model_label"],
                ascending=[True, False, True],
            )
            .head(rows_per_window)
            .copy()
        )

        positive_summary = []
        if not positive_rows.empty:
            grouped = (
                positive_rows.groupby(["model_label", "turn_index"], as_index=False)
                .agg(
                    code_short=(
                        "code_short",
                        lambda s: ", ".join(sorted(set(map(str, s)))),
                    ),
                    score=("score", "max"),
                    raw_score=("raw_score", "max"),
                )
                .sort_values(
                    ["turn_index", "raw_score", "model_label"],
                    ascending=[True, False, True],
                )
            )
            for row in grouped.itertuples(index=False):
                positive_summary.append(
                    {
                        "model_label": row.model_label,
                        "turn_index": int(row.turn_index),
                        "code_short": row.code_short,
                        "score": float(row.score),
                        "raw_score": (
                            float(row.raw_score) if pd.notna(row.raw_score) else None
                        ),
                    }
                )

        output_rows.append(
            {
                "category": window.category,
                "window_id": window.window_id,
                "window_n_pos": window.n_pos,
                "window_n_models": window.n_models,
                "window_n_codes": window.n_codes,
                "turn_min": window.turn_min,
                "turn_max": window.turn_max,
                "positive_summary_json": json.dumps(
                    positive_summary, ensure_ascii=False
                ),
                "positive_summary_text": _truncate(
                    "; ".join(
                        f"{item['model_label']} turn {item['turn_index']} "
                        f"{item['code_short']} raw={item['raw_score']}"
                        for item in positive_summary
                    ),
                    excerpt_chars,
                ),
                "full_window_message_count": len(messages_list),
                "full_window_text": "\n".join(
                    _format_message_lines(messages_list, limit=excerpt_chars)
                    if len(messages_list) > 0
                    else []
                ),
                "full_window_json": json.dumps(messages_list, ensure_ascii=False),
            }
        )

    return pd.DataFrame(output_rows)


def build_selected_code_examples(
    df: pd.DataFrame,
    summary: dict[str, Any],
    rows_per_window: int,
    excerpt_chars: int,
) -> pd.DataFrame:
    """Build a tidy table of one high-scoring example per code."""
    sample_index = build_sample_index(summary)
    output_rows: list[dict[str, Any]] = []

    positive = df[df["score"] > 0].copy()
    for code_short in CODE_ORDER:
        code_df = positive[positive["code_short"] == code_short].copy()
        if code_df.empty:
            continue
        code_df = code_df.sort_values(
            ["raw_score", "score", "turn_index", "model_label", "window_id"],
            ascending=[False, False, True, True, True],
            na_position="last",
        )

        chosen_row: dict[str, Any] | None = None
        chosen_sample_entry = None
        chosen_sample_path: Path | None = None
        for row in code_df.itertuples(index=False):
            sample_path = sample_index.get((row.model_label, row.code_short))
            if sample_path is None or not sample_path.is_file():
                continue
            sample_entry = find_sample_entry(sample_path, row.sample_id)
            if sample_entry is None:
                continue
            if _example_contains_blacklisted_text(sample_entry, code_short=code_short):
                continue
            chosen_row = row._asdict()
            chosen_sample_entry = sample_entry
            chosen_sample_path = sample_path
            break

        if chosen_row is None:
            chosen_row = code_df.iloc[0].to_dict()
            chosen_sample_path = sample_index.get(
                (chosen_row["model_label"], chosen_row["code_short"])
            )
            chosen_sample_entry = (
                find_sample_entry(chosen_sample_path, chosen_row["sample_id"])
                if chosen_sample_path is not None and chosen_sample_path.is_file()
                else None
            )

        row = chosen_row
        sample_entry = chosen_sample_entry
        history = []
        excerpt = ""
        grader_answer = ""
        grader_explanation = ""
        if sample_entry is not None:
            history = [
                (
                    dict(msg)
                    if isinstance(msg, dict)
                    else {"role": "message", "content": str(msg)}
                )
                for msg in list(sample_entry.get("history", []))
            ]
            excerpt = str(sample_entry.get("excerpt", ""))
            grader_answer = str(sample_entry.get("grader_answer", ""))
            grader_explanation = str(sample_entry.get("grader_explanation", ""))

        output_rows.append(
            {
                "category": row["category"],
                "code_short": row["code_short"],
                "model_label": row["model_label"],
                "window_id": row["window_id"],
                "sample_id": row["sample_id"],
                "turn_index": row["turn_index"],
                "score": row["score"],
                "raw_score": row["raw_score"],
                "last_prompt": _format_last_prompt(history, limit=excerpt_chars),
                "excerpt": _truncate(excerpt, excerpt_chars),
                "grader_answer": grader_answer,
                "grader_explanation": grader_explanation,
                "sample_path": (
                    str(chosen_sample_path) if chosen_sample_path is not None else ""
                ),
            }
        )

    return pd.DataFrame(output_rows)


def write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def write_latex_fragment(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    lines: list[str] = []
    lines.append("% Auto-generated from report/eval_rows.parquet")
    lines.append("% Copy into LaTeX or input this file directly.")
    lines.append("")

    for category in CATEGORY_ORDER:
        cat_df = df[df["category"] == category]
        if cat_df.empty:
            continue
        lines.append(f"\\subsection*{{{_escape_latex(category.title())}}}")
        for window_id, window_df in cat_df.groupby("window_id", sort=False):
            first = window_df.iloc[0]
            lines.append(
                "\\paragraph{{Window {wid}}} {meta}".format(
                    wid=_escape_latex(window_id),
                    meta=_escape_latex(
                        f"({int(first['window_n_pos'])} positive rows, "
                        f"{int(first['window_n_models'])} models, "
                        f"turns {int(first['turn_min'])}--{int(first['turn_max'])})"
                    ),
                )
            )
            lines.append(
                "\\textbf{Positive eval summary:} "
                + _escape_latex(first.positive_summary_text)
            )
            lines.append("\\begin{quote}")
            lines.append("\\begin{enumerate}")
            messages = []
            try:
                messages = json.loads(first.full_window_json)
            except json.JSONDecodeError:
                messages = []
            for idx, msg in enumerate(messages):
                role = str(msg.get("role", "message")).title()
                content = _escape_latex(_truncate(msg.get("content", ""), 2200))
                turn_label = _escape_latex(role.lower())
                item_line = f"  \\item \\textbf{{Turn {idx} ({turn_label})}}: {content}"
                lines.append(item_line)
            lines.append("\\end{enumerate}")
            lines.append("\\end{quote}")
            lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")


def write_latex_fragment_by_code(df: pd.DataFrame, path: Path) -> None:
    """Write a LaTeX fragment with one example per code."""
    path.parent.mkdir(parents=True, exist_ok=True)

    lines: list[str] = []
    lines.append("% Auto-generated from report/samples/*.json")
    lines.append("% Copy into LaTeX or input this file directly.")
    lines.append(
        f"% Reference with \\ref{{{CODE_EXAMPLES_SECTION_LABEL}}} after \\input."
    )
    lines.append(f"\\subsection{{{_escape_latex(CODE_EXAMPLES_SECTION_TITLE)}}}")
    lines.append(f"\\label{{{CODE_EXAMPLES_SECTION_LABEL}}}")
    lines.append("")

    for category in CATEGORY_ORDER:
        cat_df = df[df["category"] == category]
        if cat_df.empty:
            continue
        lines.append(f"\\subsubsection*{{{_escape_latex(category.title())}}}")
        for code_short in [
            code for code in CODE_ORDER if CODE_CATEGORIES[code] == category
        ]:
            code_df = cat_df[cat_df["code_short"] == code_short]
            if code_df.empty:
                continue
            first = code_df.iloc[0]
            lines.append(
                "\\paragraph{{{code}}} {meta}".format(
                    code=_escape_latex(code_short),
                    meta=_escape_latex(
                        f"{first['model_label']}, sample {first['sample_id']}, "
                        f"turn {int(first['turn_index'])}, raw {first['raw_score']}"
                    ),
                )
            )
            lines.append("\\begin{quote}")
            lines.append(r"\begingroup\small\ttfamily")
            if first.last_prompt:
                lines.append(_escape_latex_monospace(first.last_prompt) + r"\\")
            lines.append(
                "\\textbf{Highlighted assistant excerpt:} "
                + _escape_latex_monospace(str(first.excerpt))
            )
            lines.append(r"\endgroup")
            lines.append("\\end{quote}")
            lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    df = pd.read_parquet(args.rows_path)
    _ = load_summary(args.summary_path)
    if args.group_by == "category":
        items_path = resolve_items_sanitized_path(args.items_path)
        items_map = load_window_messages(items_path)
        selected_windows = select_windows(
            df, examples_per_category=args.examples_per_category
        )
        output = build_selected_examples(
            df=df,
            items_map=items_map,
            selected_windows=selected_windows,
            rows_per_window=args.rows_per_window,
            excerpt_chars=args.excerpt_chars,
        )
        ensure_output_dirs(args.csv_out.parent, args.tex_out.parent)
        write_csv(output, args.csv_out)
        write_latex_fragment(output, args.tex_out)
        print(
            f"Wrote {len(output)} window rows from {len(selected_windows)} windows "
            f"to {args.csv_out} and {args.tex_out}"
        )
        return

    output = build_selected_code_examples(
        df=df,
        summary=load_summary(args.summary_path),
        rows_per_window=args.rows_per_window,
        excerpt_chars=args.excerpt_chars,
    )
    ensure_output_dirs(args.csv_out.parent, args.tex_out.parent)
    write_csv(output, args.csv_out)
    write_latex_fragment_by_code(output, args.tex_out)
    print(
        f"Wrote {len(output)} code rows from {output['code_short'].nunique()} codes "
        f"to {args.csv_out} and {args.tex_out}"
    )


if __name__ == "__main__":
    main()
