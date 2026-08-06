"""Export selected analysis artifacts to the sibling Overleaf repository.

This module reads ``analysis/paper_export_manifest.json`` and copies only the
explicitly listed assets into ``../llm-delusions-eval-overleaf/``.

The manifest is organized by producer so export selection is decoupled from
artifact generation.
"""

import argparse
import json
import shutil
from pathlib import Path

from analysis.load_eval_data import _EVALS_REPO_ROOT

DEFAULT_MANIFEST_PATH = _EVALS_REPO_ROOT / "analysis" / "paper_export_manifest.json"
DEFAULT_OVERLEAF_ROOT = _EVALS_REPO_ROOT.parent / "llm-delusions-eval-overleaf"

PRODUCER_OUTPUT_DIRS = {
    "generate_figures": {
        "figures": _EVALS_REPO_ROOT / "analysis" / "figures",
        "tables": _EVALS_REPO_ROOT / "analysis" / "tables",
    },
    "classify_refusals": {
        "tables": _EVALS_REPO_ROOT / "analysis" / "data",
    },
    "generate_window_cutup_real_data": {
        "figures": _EVALS_REPO_ROOT / "analysis" / "figures",
    },
    "compute_context_effects": {
        "figures": _EVALS_REPO_ROOT / "analysis" / "figures",
    },
    "compute_context_code_controls": {
        "figures": _EVALS_REPO_ROOT / "analysis" / "figures",
    },
    "compute_participant_robustness": {
        "tables": _EVALS_REPO_ROOT / "analysis" / "tables",
    },
}

OVERLEAF_TARGET_DIRS = {
    "figures": "figures",
    "tables": "tables",
}


def _copy_file(src: Path, dst: Path) -> None:
    """Copy one file, creating destination directories as needed.

    Parameters
    ----------
    src:
        Source file path.
    dst:
        Destination file path.
    """
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def _load_export_manifest(manifest_path: Path) -> dict[str, dict[str, list[str]]]:
    """Load and validate the export manifest.

    Parameters
    ----------
    manifest_path:
        Path to the manifest JSON file.

    Returns
    -------
    dict[str, dict[str, list[str]]]
        Nested mapping keyed as ``producer -> artifact_kind -> filenames``.
    """
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Manifest file not found: '{manifest_path}'")

    try:
        with manifest_path.open("r", encoding="utf-8") as file_obj:
            manifest_data = json.load(file_obj)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in manifest: '{manifest_path}'") from exc

    producers = manifest_data.get("producers")
    if not isinstance(producers, dict):
        raise ValueError("Manifest must contain object key 'producers'.")

    normalized: dict[str, dict[str, list[str]]] = {}
    for producer_name, artifact_map in producers.items():
        normalized[producer_name] = _normalize_producer_entry(
            producer_name=producer_name,
            artifact_map=artifact_map,
        )

    return normalized


def _normalize_producer_entry(
    producer_name: str,
    artifact_map: object,
) -> dict[str, list[str]]:
    """Validate and normalize one producer block from the manifest.

    Parameters
    ----------
    producer_name:
        Producer key from the manifest.
    artifact_map:
        Artifact mapping object for the producer.

    Returns
    -------
    dict[str, list[str]]
        Normalized artifact lists keyed by artifact kind.
    """
    if producer_name not in PRODUCER_OUTPUT_DIRS:
        raise ValueError(f"Unsupported producer in manifest: '{producer_name}'")
    if not isinstance(artifact_map, dict):
        raise ValueError(
            f"Manifest producer entry must be an object: '{producer_name}'"
        )

    normalized_artifacts: dict[str, list[str]] = {}
    for artifact_kind, filenames in artifact_map.items():
        _validate_artifact_kind(
            producer_name=producer_name,
            artifact_kind=artifact_kind,
        )
        normalized_artifacts[artifact_kind] = _normalize_filename_list(
            filenames=filenames,
            producer_name=producer_name,
            artifact_kind=artifact_kind,
        )
    return normalized_artifacts


def _validate_artifact_kind(producer_name: str, artifact_kind: str) -> None:
    """Validate one artifact kind for a producer.

    Parameters
    ----------
    producer_name:
        Producer key from the manifest.
    artifact_kind:
        Artifact kind key from the manifest.
    """
    if artifact_kind not in PRODUCER_OUTPUT_DIRS[producer_name]:
        raise ValueError(
            "Unsupported artifact kind "
            f"'{artifact_kind}' for producer '{producer_name}'"
        )
    if artifact_kind not in OVERLEAF_TARGET_DIRS:
        raise ValueError(f"Unsupported overleaf target kind: '{artifact_kind}'")


def _normalize_filename_list(
    filenames: object,
    *,
    producer_name: str,
    artifact_kind: str,
) -> list[str]:
    """Normalize one artifact filename list from the manifest.

    Parameters
    ----------
    filenames:
        Manifest value expected to be a list of strings.
    producer_name:
        Producer key from the manifest.
    artifact_kind:
        Artifact kind key from the manifest.

    Returns
    -------
    list[str]
        Cleaned filenames with empty strings removed.
    """
    if not isinstance(filenames, list):
        raise ValueError(
            "Manifest artifact list must be an array: "
            f"'{producer_name}.{artifact_kind}'"
        )

    clean_filenames: list[str] = []
    for filename in filenames:
        if not isinstance(filename, str):
            raise ValueError(
                "Manifest filename entries must be strings in "
                f"'{producer_name}.{artifact_kind}'"
            )
        normalized_name = filename.strip()
        if normalized_name:
            clean_filenames.append(normalized_name)

    return clean_filenames


def export_assets_to_overleaf(manifest_path: Path, overleaf_root: Path) -> int:
    """Copy manifest-selected artifacts to the overleaf repository.

    Parameters
    ----------
    manifest_path:
        Path to manifest JSON.
    overleaf_root:
        Overleaf repository root path.

    Returns
    -------
    int
        Number of files copied.
    """
    overleaf_root = overleaf_root.resolve()
    if not overleaf_root.is_dir():
        raise NotADirectoryError(
            f"Overleaf repo directory not found: '{overleaf_root}'"
        )

    manifest = _load_export_manifest(manifest_path.resolve())

    copied_count = 0
    for producer_name, artifact_map in manifest.items():
        for artifact_kind, filenames in artifact_map.items():
            src_dir = PRODUCER_OUTPUT_DIRS[producer_name][artifact_kind]
            dst_dir = overleaf_root / OVERLEAF_TARGET_DIRS[artifact_kind]
            for filename in filenames:
                src_path = src_dir / filename
                if not src_path.is_file():
                    raise FileNotFoundError(
                        "Manifest-selected asset does not exist: "
                        f"'{src_path}' (producer={producer_name})"
                    )
                dst_path = dst_dir / filename
                _copy_file(src_path, dst_path)
                copied_count += 1

    return copied_count


def main() -> None:
    """Parse CLI args and export selected assets to Overleaf."""
    parser = argparse.ArgumentParser(
        description=(
            "Copy manifest-selected analysis artifacts into "
            "the sibling llm-delusions-eval-overleaf repository."
        )
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_MANIFEST_PATH,
        help=f"Path to export manifest JSON (default: {DEFAULT_MANIFEST_PATH})",
    )
    parser.add_argument(
        "--overleaf-root",
        type=Path,
        default=DEFAULT_OVERLEAF_ROOT,
        help=f"Path to overleaf repo root (default: {DEFAULT_OVERLEAF_ROOT})",
    )
    args = parser.parse_args()

    copied = export_assets_to_overleaf(
        manifest_path=args.manifest,
        overleaf_root=args.overleaf_root,
    )
    print(f"Export complete: copied {copied} file(s)")


if __name__ == "__main__":
    main()
