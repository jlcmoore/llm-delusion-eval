"""Shared filesystem paths for analysis artifacts.

This module centralizes where analysis scripts write figures, tables, and
machine-readable data artifacts.
"""

from pathlib import Path

from analysis.load_eval_data import _EVALS_REPO_ROOT

ANALYSIS_DIR = _EVALS_REPO_ROOT / "analysis"
FIGURE_DIR = ANALYSIS_DIR / "figures"
DATA_DIR = ANALYSIS_DIR / "data"
TABLE_DIR = ANALYSIS_DIR / "tables"


def ensure_output_dirs(*directories: Path) -> None:
    """Create output directories if they do not already exist.

    Parameters
    ----------
    directories:
        One or more directory paths to create.

    Returns
    -------
    None
        This function mutates the filesystem by creating directories in place.
    """
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
