"""Shared plotting style and model color mappings for analysis figures."""

from __future__ import annotations

import matplotlib.pyplot as plt
import seaborn as sns

MODEL_ORDER_PREFERRED = [
    "GPT-5.4 (high)",
    "GPT-5.4",
    "GPT-5.4 Mini",
    "GPT-5.4 Nano",
    "GPT-4.1",
    "GPT-4o",
    "GPT-4 Turbo",
    "Claude Opus 4.7",
    "Claude Sonnet 4.6",
    "Claude Haiku 4.5",
    "Gemini 3.1 Pro",
    "Gemini 3.1 Flash-Lite",
    "Gemini 2.5 Pro",
    "Gemini 2.5 Flash",
    "Gemini 2.5 Flash-Lite",
    "Qwen3.5-397B (high)",
    "Qwen3.5-397B (low)",
    "Qwen3.5-397B",
    "Qwen3.5-9B",
    "Grok 4.20",
    "Original transcript",
]

MODEL_COLORS = {
    "GPT-5.4": "#1f77b4",
    "GPT-5.4 (high)": "#4c78a8",
    "GPT-5.4 Mini": "#17becf",
    "GPT-5.4 Nano": "#9edae5",
    "GPT-4.1": "#ff7f0e",
    "GPT-4o": "#f2a65a",
    "GPT-4 Turbo": "#fdd0a2",
    "Qwen3.5-397B": "#2ca02c",
    "Qwen3.5-397B (low)": "#74c476",
    "Qwen3.5-397B (high)": "#006d2c",
    "Qwen3.5-9B": "#98df8a",
    "Qwen3.5-9B (low)": "#98df8a",
    "Gemini 3.1 Pro": "#9467bd",
    "Gemini 3.1 Pro (minimal)": "#9467bd",
    "Gemini 3.1 Flash-Lite": "#c5b0d5",
    "Gemini 3.1 Flash-Lite (minimal)": "#c5b0d5",
    "Gemini 2.5 Pro": "#6f4a8e",
    "Gemini 2.5 Pro (minimal)": "#6f4a8e",
    "Gemini 2.5 Flash": "#a283c5",
    "Gemini 2.5 Flash (minimal)": "#a283c5",
    "Gemini 2.5 Flash-Lite": "#cfbce5",
    "Gemini 2.5 Flash-Lite (minimal)": "#cfbce5",
    "Claude Opus 4.7": "#d62728",
    "Claude Sonnet 4.6": "#ff9896",
    "Claude Haiku 4.5": "#f7b6d2",
    "Grok 4.20": "#8c564b",
    "Original transcript": "#7f7f7f",
}

REASONING_COMPARISON_COLORS = {
    "GPT-5.4": "#4c78a8",
    "GPT-5.4 (high)": "#9ecae9",
    "Qwen3.5-397B": "#2ca02c",
    "Qwen3.5-397B (low)": "#74c476",
    "Qwen3.5-397B (high)": "#006d2c",
}

_FALLBACK_COLORS = [
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
    "#bcbd22",
    "#17becf",
]


def apply_plot_style() -> None:
    """Apply a shared plotting theme across analysis figure scripts.

    Returns
    -------
    None
        This function updates global seaborn/matplotlib style settings.
    """
    sns.set_theme(style="whitegrid", font_scale=1.0)
    plt.rcParams.update(
        {
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "font.family": "serif",
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
        }
    )


def sort_model_labels(labels: list[str]) -> list[str]:
    """Sort model labels by a preferred display order.

    Parameters
    ----------
    labels:
        Model labels to sort.

    Returns
    -------
    list[str]
        Labels ordered by ``MODEL_ORDER_PREFERRED`` then alphabetically.
    """
    order_map = {name: index for index, name in enumerate(MODEL_ORDER_PREFERRED)}

    def _model_sort_key(name: str) -> tuple[int, int, str]:
        if name == "Original transcript":
            return (10_000, 0, name)

        if name in order_map:
            return (order_map[name], 0, name)

        if name.endswith(")") and " (" in name:
            base_name, variant = name.rsplit(" (", maxsplit=1)
            variant = variant.removesuffix(")")
            if base_name in order_map:
                variant_rank = {
                    "high": 0,
                    "low": 1,
                    "minimal": 2,
                    "none": 3,
                    "n/a": 4,
                }.get(variant.lower(), 5)
                return (order_map[base_name], variant_rank, name)

        return (9_000, 9, name)

    return sorted(labels, key=_model_sort_key)


def get_model_color(model_label: str) -> str:
    """Return the canonical display color for a model label.

    Parameters
    ----------
    model_label:
        Human-readable model label.

    Returns
    -------
    str
        Hex color string.
    """
    if model_label in MODEL_COLORS:
        return MODEL_COLORS[model_label]
    fallback_index = sum(ord(char) for char in model_label) % len(_FALLBACK_COLORS)
    return _FALLBACK_COLORS[fallback_index]


def get_reasoning_model_color(model_label: str) -> str:
    """Return reasoning-comparison color for a model label.

    Parameters
    ----------
    model_label:
        Human-readable model label.

    Returns
    -------
    str
        Hex color string from the reasoning palette when available.
    """
    if model_label in REASONING_COMPARISON_COLORS:
        return REASONING_COMPARISON_COLORS[model_label]
    return get_model_color(model_label)
