"""Shared path defaults and resolution for eval data directories."""

import os
from pathlib import Path

try:
    from huggingface_hub import hf_hub_download
except ImportError:
    hf_hub_download = None

# Resolve the evals repository root
# __file__ is src/llm_delusion_eval/paths.py
_EVALS_REPO_ROOT = Path(__file__).resolve().parent.parent.parent

# Default public sanitized data source (Hugging Face dataset file URI).
DEFAULT_WINDOWS_PATH = "hf://datasets/jlcmoore/delusioneval/items_sanitized.parquet"
_MIN_HF_DATASET_PATH_PARTS = 3

# Local-only context sources (kept for context-length experiments).
_LOCAL_DATA_REPO_ROOT = _EVALS_REPO_ROOT.parent / "llm-delusions"
DEFAULT_CONTEXT_WINDOWS_PATH = str(_LOCAL_DATA_REPO_ROOT / "subsets" / "items.parquet")
DEFAULT_TRANSCRIPTS_PATH = str(
    _LOCAL_DATA_REPO_ROOT / "transcripts_data" / "transcripts.parquet"
)


def _normalize_path_source(path_text: str) -> str:
    """Normalize supported path/URI strings without changing semantics."""
    normalized = str(path_text).strip()
    if normalized.startswith("hf:/") and not normalized.startswith("hf://"):
        return normalized.replace("hf:/", "hf://", 1)
    return normalized


def _abspath_from_repo_root(path_text: str) -> str:
    """Resolve relative paths from the eval repo root.

    Parameters
    ----------
    path_text:
        Candidate filesystem path from task args or environment.

    Returns
    -------
    str
        Absolute filesystem path.
    """
    candidate = Path(path_text).expanduser()
    if candidate.is_absolute():
        return str(candidate)
    return str((_EVALS_REPO_ROOT / candidate).resolve())


def _is_remote_path(path_text: str) -> bool:
    """Return whether a path-like value is a remote URI."""
    return _normalize_path_source(path_text).startswith("hf://")


def _parse_hf_dataset_source(source: str) -> tuple[str, str, str] | None:
    """Parse strict Hugging Face dataset sources into download components.

    Parameters
    ----------
    source:
        Input source string in ``hf://datasets/<owner>/<dataset>/<file>`` format.

    Returns
    -------
    tuple[str, str, str] | None
        ``(repo_id, filename, revision)`` when source matches the required HF
        dataset URI format, otherwise ``None``.
    """
    normalized = _normalize_path_source(source)
    if not normalized:
        return None

    hf_prefix = "hf://datasets/"
    if not normalized.startswith(hf_prefix):
        return None

    suffix = normalized.removeprefix(hf_prefix).strip("/")
    path_parts = suffix.split("/")
    if len(path_parts) < _MIN_HF_DATASET_PATH_PARTS:
        return None

    repo_id = f"{path_parts[0]}/{path_parts[1]}"
    filename = "/".join(path_parts[2:]).strip("/")
    if not filename:
        return None
    return repo_id, filename, "main"


def _materialize_hf_dataset_file(source: str) -> str:
    """Download an HF dataset file to local cache and return its local path.

    Parameters
    ----------
    source:
        HF dataset source URI.

    Returns
    -------
    str
        Absolute local path to the cached dataset file.
    """
    parsed = _parse_hf_dataset_source(source)
    if parsed is None:
        raise ValueError(
            "Unsupported remote source for local materialization: "
            f"{source}. Use hf://datasets/<owner>/<dataset>/<file>."
        )
    repo_id, filename, revision = parsed

    if hf_hub_download is None:
        raise ImportError(
            "huggingface_hub is required to materialize HF dataset files locally."
        )

    return str(
        Path(
            hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                revision=revision,
                repo_type="dataset",
            )
        ).resolve()
    )


def resolve_path(
    env_var: str,
    default_path: str,
    explicit: str = "",
    *,
    require_local: bool = False,
) -> str:
    """Resolve a path/URI from explicit value, env var, or default.

    Parameters
    ----------
    env_var:
        Environment variable name to check when ``explicit`` is empty.
    default_path:
        Default value to fall back on.
    explicit:
        If non-empty, used first.
    require_local:
        When ``True``, ensure the returned value is a local filesystem path.
        HF dataset URLs/URIs are materialized into local cache files.
    Returns
    -------
    str
        Resolved local path or remote URI.
    """
    if explicit:
        candidate = explicit
    else:
        from_env = os.environ.get(env_var, "")
        candidate = from_env if from_env else default_path

    candidate = _normalize_path_source(candidate)
    if not candidate:
        raise ValueError(f"Resolved empty value for {env_var}.")

    if candidate.startswith(("http://", "https://")):
        raise ValueError(
            "HTTP(S) dataset sources are not supported. "
            "Use hf://datasets/<owner>/<dataset>/<file>."
        )

    if _is_remote_path(candidate):
        if not require_local:
            return candidate
        return _materialize_hf_dataset_file(candidate)

    return _abspath_from_repo_root(candidate)
