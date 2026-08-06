"""Display helpers for metric labels and optimization direction.

These helpers standardize how category/code labels are shown in figures and
encode whether each metric should be maximized or minimized.
"""

from typing import Final

_CATEGORY_DISPLAY_NAMES: Final[dict[str, str]] = {
    "sycophancy": "Sycophancy",
    "delusional": "Delusional",
    "relationship": "Relationship",
    "facilitates harm": "Facilitates harm",
    "discourages harm": "Discourages harm",
}

_MAXIMIZE_PREFIXES: Final[tuple[str, str]] = ("discourages-", "validates-")
_MAXIMIZE_EXACT_KEYS: Final[frozenset[str]] = frozenset({"discourages harm"})


def normalize_metric_key(metric_key: str) -> str:
    """Normalize one metric key for display/rule checks.

    Parameters
    ----------
    metric_key:
        Raw metric key (for example ``"bot-facilitates-violence"``).

    Returns
    -------
    str
        Normalized key with ``bot-`` removed.
    """
    return str(metric_key).strip().removeprefix("bot-")


def is_maximize_metric(metric_key: str) -> bool:
    """Return whether a metric should be interpreted as maximize.

    Parameters
    ----------
    metric_key:
        Raw or normalized metric key.

    Returns
    -------
    bool
        ``True`` when the metric should be maximized; otherwise minimized.
    """
    normalized = normalize_metric_key(metric_key)
    return normalized in _MAXIMIZE_EXACT_KEYS or normalized.startswith(
        _MAXIMIZE_PREFIXES
    )


def metric_direction_arrow_for_matplotlib(metric_key: str, *, axis: str = "x") -> str:
    """Return a display arrow for metric optimization direction.

    Parameters
    ----------
    metric_key:
        Raw or normalized metric key.
    axis:
        Axis orientation context. Use ``"y"`` for y-axis labels to render
        horizontal arrows.

    Returns
    -------
    str
        For non-y axes: ``"↑"`` maximize, ``"↓"`` minimize.
        For y axes: ``"→"`` maximize, ``"←"`` minimize.
    """
    maximize = is_maximize_metric(metric_key)
    if str(axis).lower() == "y":
        return "→" if maximize else "←"
    return "↑" if maximize else "↓"


def prevalence_axis_label_for_matplotlib(metric_key: str) -> str:
    """Build a prevalence axis label with optimization direction.

    Parameters
    ----------
    metric_key:
        Raw or normalized metric key.

    Returns
    -------
    str
        Axis label such as ``"Prevalence (%) (←)"``.
    """
    return (
        "Prevalence (%) "
        f"({metric_direction_arrow_for_matplotlib(metric_key, axis='y')})"
    )


def format_metric_label_for_matplotlib(metric_key: str) -> str:
    """Build a Matplotlib-ready display label with an objective arrow.

    Parameters
    ----------
    metric_key:
        Raw or normalized metric key.

    Returns
    -------
    str
        Display label such as ``"Sycophancy (↓)"``.
    """
    normalized = normalize_metric_key(metric_key)
    base = _CATEGORY_DISPLAY_NAMES.get(normalized, normalized.replace("-", " "))
    arrow = metric_direction_arrow_for_matplotlib(normalized)
    return f"{base} ({arrow})"


def format_metric_label_plain_for_matplotlib(metric_key: str) -> str:
    """Build a Matplotlib-ready display label without an objective arrow.

    Parameters
    ----------
    metric_key:
        Raw or normalized metric key.

    Returns
    -------
    str
        Display label such as ``"Sycophancy"``.
    """
    normalized = normalize_metric_key(metric_key)
    return _CATEGORY_DISPLAY_NAMES.get(normalized, normalized.replace("-", " "))
