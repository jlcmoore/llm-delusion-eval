"""Shared constants and mappings for the eval repo."""

# Mapping from source annotation IDs to preferred eval repo IDs.
#
# Some IDs in the llm-delusions source repo are being renamed in this
# repository to better reflect the underlying intent or for consistency.
ID_RENAME_MAPPING = {
    "user-suicidal-intent": "user-suicidal-thoughts",
    "user-violent-intent": "user-violent-thoughts",
}

# Inverse mapping for lookup when calling into packages that still use
# the original IDs (e.g., llm-delusions-annotations).
ID_RENAME_MAPPING_INV = {v: k for k, v in ID_RENAME_MAPPING.items()}

# Display names for models. Keys are full model IDs as logged in eval outputs.
MODEL_DISPLAY_NAMES = {
    "openai/gpt-5.4-2026-03-05": "GPT-5.4",
    "openai/gpt-5.4-mini-2026-03-17": "GPT-5.4 Mini",
    "openai/gpt-5.4-nano-2026-03-17": "GPT-5.4 Nano",
    "openai/gpt-4.1-2025-04-14": "GPT-4.1",
    "openai/gpt-4o-2024-11-20": "GPT-4o",
    "openai/gpt-4-turbo-2024-04-09": "GPT-4 Turbo",
    "together/Qwen/Qwen3.5-397B-A17B": "Qwen3.5-397B",
    "together/Qwen/Qwen3.5-9B": "Qwen3.5-9B",
    "google/vertex/gemini-3.1-flash-lite-preview": "Gemini 3.1 Flash-Lite",
    "google/vertex/gemini-3-flash-lite-preview": "Gemini 3.1 Flash-Lite",
    "google/vertex/gemini-3.1-pro-preview": "Gemini 3.1 Pro",
    "google/vertex/gemini-2.5-flash": "Gemini 2.5 Flash",
    "google/vertex/gemini-2.5-flash-lite": "Gemini 2.5 Flash-Lite",
    "google/vertex/gemini-2.5-pro": "Gemini 2.5 Pro",
    "anthropic/claude-haiku-4-5": "Claude Haiku 4.5",
    "anthropic/claude-opus-4-7": "Claude Opus 4.7",
    "anthropic/claude-sonnet-4-6": "Claude Sonnet 4.6",
    "grok/grok-4.20-0309-non-reasoning": "Grok 4.20",
}

_MODEL_DISPLAY_LABEL_ALIASES = {
    "claude opus 4": "Claude Opus 4.7",
    "claude opus 4.7": "Claude Opus 4.7",
    "claude sonnet 4.6": "Claude Sonnet 4.6",
    "claude haiku 4.5": "Claude Haiku 4.5",
    "grok 4": "Grok 4.20",
    "grok 4.20": "Grok 4.20",
    "gemini 2.5 pro": "Gemini 2.5 Pro",
    "gemini 2.5 flash": "Gemini 2.5 Flash",
    "gemini 2.5 flash-lite": "Gemini 2.5 Flash-Lite",
    "gemini-2.5-pro": "Gemini 2.5 Pro",
    "gemini-2.5-flash": "Gemini 2.5 Flash",
    "gemini-2.5-flash-lite": "Gemini 2.5 Flash-Lite",
    "claude-opus-4-7": "Claude Opus 4.7",
    "claude-sonnet-4-6": "Claude Sonnet 4.6",
    "claude-haiku-4-5": "Claude Haiku 4.5",
}


def normalize_id(code: str) -> str:
    """Normalize a source annotation ID to the preferred eval repo ID."""
    return ID_RENAME_MAPPING.get(code, code)


def get_source_id(code: str) -> str:
    """Get the original source annotation ID from an eval repo ID."""
    return ID_RENAME_MAPPING_INV.get(code, code)


def model_display_name(model: str) -> str:
    """Return the human-readable display name for a model ID.

    Parameters
    ----------
    model:
        Full model identifier from eval logs.

    Returns
    -------
    str
        Display name, or the trailing model path segment if unmapped.
    """
    if model == "original_transcript":
        return "Original transcript"
    return MODEL_DISPLAY_NAMES.get(model, model.split("/")[-1])


def format_model_label(model: str, reasoning: str | None) -> str:
    """Build a display label from model ID and reasoning effort.

    Parameters
    ----------
    model:
        Full model identifier from eval logs.
    reasoning:
        Optional reasoning effort value.

    Returns
    -------
    str
        Display label such as ``GPT-5.4 (high)``.
    """
    base = model_display_name(model)
    if reasoning and reasoning not in ("none", "N/A"):
        return f"{base} ({reasoning})"
    return base


def normalize_model_label(model_label: str) -> str:
    """Normalize a display label to the canonical model-label style.

    Parameters
    ----------
    model_label:
        Raw display label or model identifier from historical artifacts.

    Returns
    -------
    str
        Canonical model label with normalized capitalization/version suffixes.
    """
    normalized = str(model_label).strip()
    if not normalized:
        return normalized

    reasoning_suffix = ""
    if " (" in normalized and normalized.endswith(")"):
        normalized, suffix = normalized.rsplit(" (", maxsplit=1)
        suffix = suffix.removesuffix(")")
        if suffix.lower() not in ("none", "n/a"):
            reasoning_suffix = f" ({suffix})"

    if "/" in normalized:
        base = model_display_name(normalized)
    else:
        base = _MODEL_DISPLAY_LABEL_ALIASES.get(normalized.lower(), normalized)
        if "/" in base:
            base = model_display_name(base)

    return f"{base}{reasoning_suffix}"
