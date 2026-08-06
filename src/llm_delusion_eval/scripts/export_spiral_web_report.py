"""Export a generated report bundle into the spiral-web site repository.

This script copies the static artifacts produced by ``generate_report.py`` into a
spiral-web asset directory so the report viewer can be embedded from
``_research/evaluation.md`` without additional build steps.
"""

import argparse
import json
import shutil
from dataclasses import dataclass
from pathlib import Path

DEFAULT_TARGET_SUBDIR = "assets/evaluation/report/latest"
REQUIRED_REPORT_FILES = ("summary.json",)
UI_ASSET_FILES = ("viewer.js", "styles.css", "index.html")


@dataclass(frozen=True)
class ExportSettings:
    """Runtime options for exporting report assets.

    Parameters
    ----------
    clean_target
        Whether to delete the destination directory before copying.
    include_figures
        Whether to copy the optional ``figures/`` directory.
    aggregate_only
        Whether to export aggregate-only results with no sample/example data.
    """

    clean_target: bool
    include_figures: bool
    aggregate_only: bool


def _ensure_report_assets(report_dir: Path) -> None:
    """Validate the generated report directory contains required assets.

    Parameters
    ----------
    report_dir
        Directory produced by ``generate_report.py``.

    Returns
    -------
    None
    """
    missing = [
        filename
        for filename in REQUIRED_REPORT_FILES
        if not (report_dir / filename).is_file()
    ]
    if missing:
        missing_list = ", ".join(missing)
        raise FileNotFoundError(
            f"Missing required report files in '{report_dir}': {missing_list}"
        )


def _copy_file(src: Path, dst: Path) -> None:
    """Copy one file and ensure its parent directory exists.

    Parameters
    ----------
    src
        Source file path.
    dst
        Destination file path.

    Returns
    -------
    None
    """
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def _sanitize_summary_for_aggregate_only(summary_path: Path) -> None:
    """Remove per-sample data and example snippets from ``summary.json``.

    Parameters
    ----------
    summary_path
        Path to the copied ``summary.json`` file.

    Returns
    -------
    None
    """
    with open(summary_path, "r", encoding="utf-8") as summary_file:
        summary_data = json.load(summary_file)

    evaluations = summary_data.get("evaluations")
    if not isinstance(evaluations, list) and isinstance(summary_data, list):
        evaluations = summary_data

    if isinstance(evaluations, list):
        for item in evaluations:
            if isinstance(item, dict):
                item.pop("sample_paths", None)

    metadata = summary_data.get("metadata")
    if isinstance(metadata, dict):
        for code_meta in metadata.values():
            if isinstance(code_meta, dict):
                code_meta.pop("positive_examples", None)
                code_meta.pop("negative_examples", None)

    with open(summary_path, "w", encoding="utf-8") as summary_file:
        json.dump(summary_data, summary_file, indent=2)
        summary_file.write("\n")


def _copy_required_report_files(report_dir: Path, target_dir: Path) -> None:
    """Copy required report files from source to destination.

    Parameters
    ----------
    report_dir
        Source generated report directory.
    target_dir
        Destination directory in spiral-web.

    Returns
    -------
    None
    """
    for filename in REQUIRED_REPORT_FILES:
        src_file = report_dir / filename
        if src_file.is_file():
            _copy_file(src_file, target_dir / filename)


def _copy_ui_assets(report_dir: Path, ui_assets_dir: Path, target_dir: Path) -> None:
    """Copy viewer UI assets, preferring bundled script assets.

    Parameters
    ----------
    report_dir
        Source generated report directory.
    ui_assets_dir
        Path to static UI assets bundled with this script.
    target_dir
        Destination directory in spiral-web.

    Returns
    -------
    None
    """
    for filename in UI_ASSET_FILES:
        preferred_src = ui_assets_dir / filename
        fallback_src = report_dir / filename
        selected_src = preferred_src if preferred_src.is_file() else fallback_src
        if selected_src.is_file():
            _copy_file(selected_src, target_dir / filename)


def _copy_optional_tree(src_dir: Path, dst_dir: Path) -> None:
    """Copy a directory if it exists.

    Parameters
    ----------
    src_dir
        Source directory path.
    dst_dir
        Destination directory path.

    Returns
    -------
    None
    """
    if src_dir.is_dir():
        shutil.copytree(src_dir, dst_dir, dirs_exist_ok=True)


def _copy_optional_directories(
    report_dir: Path, target_dir: Path, settings: ExportSettings
) -> None:
    """Copy optional report directories based on export settings.

    Parameters
    ----------
    report_dir
        Source generated report directory.
    target_dir
        Destination directory in spiral-web.
    settings
        Export behavior flags.

    Returns
    -------
    None
    """
    if not settings.aggregate_only:
        _copy_optional_tree(report_dir / "samples", target_dir / "samples")
    if settings.include_figures:
        _copy_optional_tree(report_dir / "figures", target_dir / "figures")


def export_report_bundle(
    report_dir: Path,
    spiral_web_dir: Path,
    target_subdir: str,
    settings: ExportSettings,
) -> Path:
    """Copy report assets from ``llm-delusion-eval`` into ``spiral-web``.

    Parameters
    ----------
    report_dir
        Source generated report directory.
    spiral_web_dir
        Root directory of the spiral-web repository.
    target_subdir
        Destination path relative to ``spiral_web_dir``.
    settings
        Export behavior flags.

    Returns
    -------
    Path
        Absolute path to the destination directory.
    """
    report_dir = report_dir.resolve()
    spiral_web_dir = spiral_web_dir.resolve()
    target_dir = (spiral_web_dir / target_subdir).resolve()
    ui_assets_dir = Path(__file__).resolve().parent / "report_assets"

    if not report_dir.is_dir():
        raise NotADirectoryError(f"Report directory not found: '{report_dir}'")

    if not spiral_web_dir.is_dir():
        raise NotADirectoryError(f"Spiral-web directory not found: '{spiral_web_dir}'")

    _ensure_report_assets(report_dir)

    if settings.clean_target and target_dir.exists():
        shutil.rmtree(target_dir)

    target_dir.mkdir(parents=True, exist_ok=True)

    _copy_required_report_files(report_dir, target_dir)

    if settings.aggregate_only:
        _sanitize_summary_for_aggregate_only(target_dir / "summary.json")

    _copy_ui_assets(report_dir, ui_assets_dir, target_dir)
    _copy_optional_directories(report_dir, target_dir, settings)

    return target_dir


def main() -> None:
    """Parse CLI arguments and export the report bundle.

    Parameters
    ----------
    None

    Returns
    -------
    None
    """
    parser = argparse.ArgumentParser(
        description=(
            "Copy generated report artifacts into spiral-web for static embedding."
        )
    )
    parser.add_argument(
        "--report-dir",
        type=Path,
        default=Path("report"),
        help="Generated report directory (default: report)",
    )
    parser.add_argument(
        "--spiral-web-dir",
        type=Path,
        default=Path("../spiral-web"),
        help="Path to spiral-web repository root (default: ../spiral-web)",
    )
    parser.add_argument(
        "--target-subdir",
        type=str,
        default=DEFAULT_TARGET_SUBDIR,
        help=(f"Destination under spiral-web root (default: {DEFAULT_TARGET_SUBDIR})"),
    )
    parser.add_argument(
        "--keep-existing",
        action="store_true",
        help="Do not remove destination directory before copying.",
    )
    parser.add_argument(
        "--include-figures",
        action="store_true",
        help="Also copy report/figures if present.",
    )
    parser.add_argument(
        "--aggregate-only",
        action="store_true",
        help=(
            "Export aggregate stats only: do not copy samples and strip sample/example "
            "fields from summary.json."
        ),
    )
    args = parser.parse_args()

    exported_to = export_report_bundle(
        report_dir=args.report_dir,
        spiral_web_dir=args.spiral_web_dir,
        target_subdir=args.target_subdir,
        settings=ExportSettings(
            clean_target=not args.keep_existing,
            include_figures=args.include_figures,
            aggregate_only=args.aggregate_only,
        ),
    )

    print(f"Export complete: {exported_to}")


if __name__ == "__main__":
    main()
