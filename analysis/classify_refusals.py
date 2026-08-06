"""Classify eval messages with a Hugging Face refusal classifier.

This script uses the materialized report snapshot under ``report/samples``
rather than the raw ``.eval`` logs. That avoids duplicate log handling and
keeps the input aligned with the report data used elsewhere in analysis.

Workflow:
1. Load row-level scores from ``report/eval_rows.parquet``.
2. Load message text from ``report/samples`` and deduplicate by ``sample_id``.
3. Optionally subsample a small unique-message set for a quick smoke test.
4. Run ``Human-CentricAI/LLM-Refusal-Classifier`` on the assistant excerpt.
5. Merge predictions back onto the row-level eval table and write outputs.

Use ``--max-unique-samples`` to start small, then rerun with ``0`` or
``--all-unique-samples`` for the full report.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parents[1]
ANALYSIS_DATA_DIR = REPO_ROOT / "analysis" / "data"

logger = logging.getLogger(__name__)

DEFAULT_MODEL_ID = "Human-CentricAI/LLM-Refusal-Classifier"
DEFAULT_ROWS_PATH = REPO_ROOT / "report" / "eval_rows.parquet"
DEFAULT_SAMPLES_ROOT = REPO_ROOT / "report" / "samples"
DEFAULT_OUTPUT_PATH = ANALYSIS_DATA_DIR / "refusal_predictions.parquet"
DEFAULT_PREVIEW_PATH = ANALYSIS_DATA_DIR / "refusal_predictions_preview.csv"
DEFAULT_CLASS_SUMMARY_PATH = ANALYSIS_DATA_DIR / "refusal_class_summary.csv"
DEFAULT_DETAILED_SUMMARY_PATH = ANALYSIS_DATA_DIR / "refusal_detailed_summary.csv"
DEFAULT_BUCKET_SUMMARY_PATH = ANALYSIS_DATA_DIR / "refusal_bucket_summary.csv"

MODEL_LABELS = {
    0: "normal",
    1: "refusal_unethical",
    2: "disclaimer_unethical",
    3: "refusal_capability",
    4: "disclaimer_capability",
}
REFUSAL_LABEL_IDS = {1, 3}
DISCLAIMER_LABEL_IDS = {2, 4}


def _bucket_label(pred_id: Any) -> str:
    """Map the classifier label id to a coarse interpretation bucket."""
    try:
        value = int(pred_id)
    except (TypeError, ValueError):
        return "unknown"
    if value == 0:
        return "normal"
    if value in REFUSAL_LABEL_IDS:
        return "refusal"
    if value in DISCLAIMER_LABEL_IDS:
        return "disclaimer"
    return "unknown"


def ensure_output_dirs(*directories: Path) -> None:
    """Create output directories if needed."""
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Classify eval messages with a refusal classifier."
    )
    parser.add_argument(
        "--rows-path",
        type=Path,
        default=DEFAULT_ROWS_PATH,
        help=f"Path to report/eval_rows.parquet (default: {DEFAULT_ROWS_PATH})",
    )
    parser.add_argument(
        "--samples-root",
        type=Path,
        default=DEFAULT_SAMPLES_ROOT,
        help=f"Path to report/samples (default: {DEFAULT_SAMPLES_ROOT})",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help=f"Output parquet path (default: {DEFAULT_OUTPUT_PATH})",
    )
    parser.add_argument(
        "--preview-path",
        type=Path,
        default=DEFAULT_PREVIEW_PATH,
        help=f"Output CSV preview path (default: {DEFAULT_PREVIEW_PATH})",
    )
    parser.add_argument(
        "--class-summary-path",
        type=Path,
        default=DEFAULT_CLASS_SUMMARY_PATH,
        help=(
            "Output CSV path for the 5-way class summary "
            f"(default: {DEFAULT_CLASS_SUMMARY_PATH})"
        ),
    )
    parser.add_argument(
        "--detailed-summary-path",
        type=Path,
        default=DEFAULT_DETAILED_SUMMARY_PATH,
        help=(
            "Output CSV path for the detailed refusal/disclaimer summary "
            f"(default: {DEFAULT_DETAILED_SUMMARY_PATH})"
        ),
    )
    parser.add_argument(
        "--bucket-summary-path",
        type=Path,
        default=DEFAULT_BUCKET_SUMMARY_PATH,
        help=(
            "Output CSV path for the coarse normal/refusal/disclaimer summary "
            f"(default: {DEFAULT_BUCKET_SUMMARY_PATH})"
        ),
    )
    parser.add_argument(
        "--model-id",
        type=str,
        default=DEFAULT_MODEL_ID,
        help=f"Hugging Face model id (default: {DEFAULT_MODEL_ID})",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Inference batch size.",
    )
    parser.add_argument(
        "--max-unique-samples",
        type=int,
        default=200,
        help=(
            "Maximum number of unique sample_ids to classify. "
            "Use 0 or --all-unique-samples to process the full report."
        ),
    )
    parser.add_argument(
        "--all-unique-samples",
        action="store_true",
        help="Process all unique sample_ids instead of a small subset.",
    )
    parser.add_argument(
        "--sample-order",
        choices=["head", "random"],
        default="random",
        help="How to choose the small subset when max-unique-samples > 0.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used when sampling a subset.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Torch device override, e.g. cpu or cuda. Defaults to auto-detect.",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        help="Logging level (default: INFO).",
    )
    return parser.parse_args()


def _flatten_content(content: Any) -> str:
    """Convert report content blobs into plain text."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict):
                parts.append(str(item.get("text", item.get("content", ""))))
            else:
                parts.append(str(item))
        return "".join(parts)
    if content is None:
        return ""
    return str(content)


def _extract_excerpt(sample: dict[str, Any]) -> str:
    """Extract the assistant excerpt from one report sample row."""
    excerpt = str(sample.get("excerpt", "") or "").strip()
    if excerpt:
        return excerpt

    history = sample.get("history", [])
    if isinstance(history, list):
        for msg in reversed(history):
            if isinstance(msg, dict) and msg.get("role") == "assistant":
                return _flatten_content(msg.get("content", "")).strip()
    return ""


def load_report_samples(samples_root: Path) -> pd.DataFrame:
    """Load sample JSON exports from ``report/samples`` and dedupe by sample_id."""
    rows: list[dict[str, Any]] = []
    sample_files = sorted(samples_root.rglob("*.json"))
    if not sample_files:
        raise FileNotFoundError(f"No sample JSON files found under {samples_root}")

    for sample_path in sample_files:
        try:
            payload = json.loads(sample_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("Skipping %s: %s", sample_path, exc)
            continue

        if not isinstance(payload, list):
            logger.warning("Skipping %s: expected a JSON list", sample_path)
            continue

        rel_path = sample_path.relative_to(samples_root)
        for item in payload:
            if not isinstance(item, dict):
                continue
            sample_id = str(item.get("sample_id", "")).strip()
            if not sample_id:
                continue
            rows.append(
                {
                    "sample_id": sample_id,
                    "excerpt": _extract_excerpt(item),
                    "source_path": str(rel_path),
                    "code": str(item.get("code", "")),
                    "category": str(item.get("category", "")),
                }
            )

    samples_df = pd.DataFrame(rows)
    if samples_df.empty:
        raise ValueError(f"No sample rows found under {samples_root}")

    samples_df = samples_df.sort_values(["sample_id", "source_path", "code"]).copy()
    duplicates = samples_df.duplicated("sample_id", keep="first")
    if duplicates.any():
        logger.info(
            "Deduplicating %d repeated sample_id rows from report/samples",
            int(duplicates.sum()),
        )
    samples_df = samples_df.drop_duplicates("sample_id", keep="first")
    return samples_df[
        ["sample_id", "excerpt", "source_path", "code", "category"]
    ].copy()


def load_existing_predictions(output_path: Path) -> pd.DataFrame:
    """Load previously classified rows if the output file already exists."""
    if not output_path.exists():
        return pd.DataFrame()

    try:
        df = pd.read_parquet(output_path)
    except (OSError, ValueError) as exc:
        logger.warning(
            "Could not read existing predictions at %s: %s", output_path, exc
        )
        return pd.DataFrame()

    if "sample_id" not in df.columns:
        logger.warning(
            "Existing predictions at %s do not include sample_id.", output_path
        )
        return pd.DataFrame()

    df = df.drop_duplicates("sample_id", keep="first").copy()
    return df


def pick_unique_subset(
    unique_df: pd.DataFrame,
    max_unique_samples: int,
    *,
    sample_order: str,
    seed: int,
) -> pd.DataFrame:
    """Choose a small unique-sample subset for smoke testing."""
    if max_unique_samples <= 0 or max_unique_samples >= len(unique_df):
        return unique_df.copy()

    if sample_order == "head":
        return unique_df.head(max_unique_samples).copy()

    return unique_df.sample(n=max_unique_samples, random_state=seed).copy()


def _resolve_device(device: str | None) -> str:
    if device:
        return device
    return "cuda" if torch.cuda.is_available() else "cpu"


def classify_unique_messages(
    texts: list[str],
    model_id: str,
    *,
    batch_size: int,
    device: str,
) -> pd.DataFrame:
    """Run the HF refusal classifier on a list of message excerpts."""
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSequenceClassification.from_pretrained(model_id)
    model.to(device)
    model.eval()

    id2label = {int(k): str(v) for k, v in model.config.id2label.items()}
    refusal_ids = set(REFUSAL_LABEL_IDS)
    disclaimer_ids = set(DISCLAIMER_LABEL_IDS)

    rows: list[dict[str, Any]] = []
    for start in range(0, len(texts), batch_size):
        batch = [
            text if isinstance(text, str) else ""
            for text in texts[start : start + batch_size]
        ]
        encoded = tokenizer(
            batch,
            truncation=True,
            padding=True,
            return_tensors="pt",
        ).to(device)

        with torch.no_grad():
            logits = model(**encoded).logits
            probs = torch.softmax(logits, dim=-1).cpu()

        for row_prob_tensor in probs:
            row_probs = row_prob_tensor.tolist()
            pred_id = int(max(range(len(row_probs)), key=row_probs.__getitem__))
            pred_label = MODEL_LABELS.get(pred_id, id2label.get(pred_id, str(pred_id)))
            refusal_score = float(
                sum(row_probs[idx] for idx in refusal_ids if idx < len(row_probs))
            )
            disclaimer_score = float(
                sum(row_probs[idx] for idx in disclaimer_ids if idx < len(row_probs))
            )
            rows.append(
                {
                    "pred_id": pred_id,
                    "pred_label": pred_label,
                    "pred_score": float(row_probs[pred_id]),
                    "refusal_score": refusal_score,
                    "disclaimer_score": disclaimer_score,
                    "is_refusal": bool(pred_id in refusal_ids),
                    "is_disclaimer": bool(pred_id in disclaimer_ids),
                }
            )

    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO))

    rows_df = pd.read_parquet(args.rows_path)
    samples_df = load_report_samples(args.samples_root)
    existing_predictions = load_existing_predictions(args.output_path)

    merged = rows_df.merge(samples_df, on="sample_id", how="left", validate="m:1")
    unique_messages = merged.drop_duplicates("sample_id").copy()
    unique_messages["excerpt"] = unique_messages["excerpt"].fillna("")

    if not existing_predictions.empty:
        already_done = set(existing_predictions["sample_id"].astype(str))
        unique_messages = unique_messages[
            ~unique_messages["sample_id"].astype(str).isin(already_done)
        ].copy()
        logger.info(
            "Skipping %d already-classified sample_ids from existing %s",
            len(already_done),
            args.output_path,
        )

    subset = pick_unique_subset(
        unique_messages,
        0 if args.all_unique_samples else args.max_unique_samples,
        sample_order=args.sample_order,
        seed=args.seed,
    )

    if subset.empty:
        raise ValueError("No samples selected for classification.")

    device = _resolve_device(args.device)
    logger.info(
        "Classifying %d unique messages on %s using %s",
        len(subset),
        device,
        args.model_id,
    )

    preds = classify_unique_messages(
        subset["excerpt"].tolist(),
        args.model_id,
        batch_size=args.batch_size,
        device=device,
    )

    classified_unique = pd.concat(
        [subset.reset_index(drop=True), preds.reset_index(drop=True)],
        axis=1,
    )
    classified_rows = merged.merge(
        classified_unique[
            [
                "sample_id",
                "pred_id",
                "pred_label",
                "pred_score",
                "refusal_score",
                "disclaimer_score",
                "is_refusal",
                "is_disclaimer",
            ]
        ],
        on="sample_id",
        how="inner",
        validate="m:1",
    )

    if not existing_predictions.empty:
        combined = pd.concat([existing_predictions, classified_rows], ignore_index=True)
    else:
        combined = classified_rows

    ensure_output_dirs(args.output_path.parent, args.preview_path.parent)
    combined.to_parquet(args.output_path, index=False)
    preview_df = combined.drop_duplicates("sample_id", keep="first").copy()
    preview_df.sort_values(
        ["refusal_score", "pred_score"], ascending=[False, False]
    ).head(50).to_csv(args.preview_path, index=False)

    summary = (
        preview_df.groupby(["pred_label", "is_refusal", "is_disclaimer"], dropna=False)
        .size()
        .reset_index(name="n")
        .sort_values(["is_refusal", "n"], ascending=[False, False])
    )

    preview_df["bucket"] = preview_df["pred_id"].map(_bucket_label)
    class_summary = (
        preview_df.groupby(
            ["pred_id", "pred_label", "is_refusal", "is_disclaimer"], dropna=False
        )
        .size()
        .reset_index(name="n")
        .assign(pct=lambda frame: (frame["n"] / frame["n"].sum() * 100).round(1))
        .sort_values(["pred_id"], ascending=[True])
    )
    bucket_summary = (
        preview_df.groupby("bucket", dropna=False)
        .size()
        .reset_index(name="n")
        .assign(pct=lambda frame: (frame["n"] / frame["n"].sum() * 100).round(1))
        .sort_values(["bucket"], ascending=[True])
    )

    ensure_output_dirs(
        args.output_path.parent,
        args.preview_path.parent,
        args.class_summary_path.parent,
        args.detailed_summary_path.parent,
        args.bucket_summary_path.parent,
    )
    class_summary.to_csv(args.class_summary_path, index=False)
    summary.to_csv(args.detailed_summary_path, index=False)
    bucket_summary.to_csv(args.bucket_summary_path, index=False)

    print(f"Wrote row-level predictions to {args.output_path}")
    print(f"Wrote preview rows to {args.preview_path}")
    print(f"Wrote 5-way class summary to {args.class_summary_path}")
    print(f"Wrote detailed summary to {args.detailed_summary_path}")
    print(f"Wrote bucket summary to {args.bucket_summary_path}")
    print("Per-class summary:")
    print(class_summary.to_string(index=False))
    print("Detailed unique-message summary:")
    print(summary.to_string(index=False))
    print("Coarse unique-message summary:")
    print(bucket_summary.to_string(index=False))


if __name__ == "__main__":
    main()
