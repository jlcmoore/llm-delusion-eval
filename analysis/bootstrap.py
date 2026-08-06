"""Bootstrap confidence interval utilities for analysis summaries.

Provides flat bootstrap helpers for IID summaries and hierarchical bootstrap
helpers for participant-then-conversation resampling.
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
from scipy.stats import bootstrap

_PARTICIPANT_COL = "participant"
_CONVERSATION_COL = "conversation_id"


@dataclass
class BootstrapConfig:
    """Configuration for bootstrap confidence interval calculations.

    Attributes
    ----------
    n_boot:
        Number of bootstrap resamples.
    ci:
        Confidence level (default 0.95 for 95% CI).
    seed:
        Random seed for reproducibility.
    """

    n_boot: int = 10_000
    ci: float = 0.95
    seed: Optional[int] = 42


def bootstrap_binary_ci(
    values: np.ndarray,
    *,
    config: Optional[BootstrapConfig] = None,
) -> tuple[float, float, float]:
    """Compute a bootstrapped confidence interval for a binary proportion.

    Parameters
    ----------
    values:
        1-D array of binary (0/1) values.
    config:
        Bootstrap configuration. If None, uses default settings.

    Returns
    -------
    tuple[float, float, float]
        ``(mean, ci_lower, ci_upper)`` as proportions in [0, 1].
    """
    if config is None:
        config = BootstrapConfig()

    values = np.asarray(values, dtype=float)
    values = values[~np.isnan(values)]
    if len(values) == 0:
        return (np.nan, np.nan, np.nan)

    observed_mean = float(np.mean(values))
    if len(values) == 1:
        return (observed_mean, observed_mean, observed_mean)

    rng = np.random.default_rng(config.seed)
    result = bootstrap(
        (values,),
        np.mean,
        n_resamples=config.n_boot,
        confidence_level=config.ci,
        method="percentile",
        vectorized=True,
        rng=rng,
    )

    return (
        observed_mean,
        float(result.confidence_interval.low),
        float(result.confidence_interval.high),
    )


def bootstrap_grouped_ci(
    df: pd.DataFrame,
    *,
    group_col: str,
    score_col: str = "score",
    config: Optional[BootstrapConfig] = None,
) -> pd.DataFrame:
    """Compute bootstrapped CIs grouped by a column.

    Parameters
    ----------
    df:
        Input DataFrame containing at least ``group_col`` and ``score_col``.
    group_col:
        Column to group by (e.g. ``"code_short"`` or ``"category"``).
    score_col:
        Column containing binary 0/1 scores.
    config:
        Bootstrap configuration. If None, uses default settings.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: ``group_col``, ``mean``, ``ci_lower``,
        ``ci_upper``, ``n``.
    """
    if config is None:
        config = BootstrapConfig()

    results = []
    for group_value, group_df in df.groupby(group_col):
        values = group_df[score_col].dropna().values
        mean_val, lower, upper = bootstrap_binary_ci(values, config=config)
        results.append(
            {
                group_col: group_value,
                "mean": mean_val,
                "ci_lower": lower,
                "ci_upper": upper,
                "n": len(values),
            }
        )
    return pd.DataFrame(results)


def bootstrap_delta_ci(
    values_a: np.ndarray,
    values_b: np.ndarray,
    *,
    config: Optional[BootstrapConfig] = None,
) -> tuple[float, float, float]:
    """Compute a bootstrapped CI for the difference in means (a - b).

    Resamples the two groups independently on each iteration, then
    computes the difference of their means.

    Parameters
    ----------
    values_a:
        1-D array of binary (0/1) values for group A.
    values_b:
        1-D array of binary (0/1) values for group B.
    config:
        Bootstrap configuration. If None, uses default settings.

    Returns
    -------
    tuple[float, float, float]
        ``(observed_delta, ci_lower, ci_upper)`` as proportions.
    """
    if config is None:
        config = BootstrapConfig()

    values_a = np.asarray(values_a, dtype=float)
    values_b = np.asarray(values_b, dtype=float)
    values_a = values_a[~np.isnan(values_a)]
    values_b = values_b[~np.isnan(values_b)]

    if len(values_a) == 0 or len(values_b) == 0:
        return (np.nan, np.nan, np.nan)

    observed_delta = float(np.mean(values_a) - np.mean(values_b))
    if len(values_a) == 1 and len(values_b) == 1:
        return (observed_delta, observed_delta, observed_delta)

    rng = np.random.default_rng(config.seed)
    result = bootstrap(
        (values_a, values_b),
        _mean_delta_statistic,
        n_resamples=config.n_boot,
        confidence_level=config.ci,
        method="percentile",
        vectorized=True,
        paired=False,
        rng=rng,
    )

    return (
        observed_delta,
        float(result.confidence_interval.low),
        float(result.confidence_interval.high),
    )


def _mean_delta_statistic(
    values_a: np.ndarray, values_b: np.ndarray, axis: int
) -> np.ndarray:
    """Return mean(values_a) - mean(values_b) along ``axis``."""
    return np.mean(values_a, axis=axis) - np.mean(values_b, axis=axis)


def bootstrap_model_code_ci(
    df: pd.DataFrame,
    *,
    model_col: str = "model_label",
    code_col: str = "code_short",
    score_col: str = "score",
    config: Optional[BootstrapConfig] = None,
) -> pd.DataFrame:
    """Compute bootstrapped CIs for each (model, code) combination.

    Parameters
    ----------
    df:
        Input DataFrame.
    model_col:
        Column identifying the model.
    code_col:
        Column identifying the annotation code.
    score_col:
        Column containing binary 0/1 scores.
    config:
        Bootstrap configuration. If None, uses default settings.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: ``model_col``, ``code_col``, ``mean``,
        ``ci_lower``, ``ci_upper``, ``n``.
    """
    if config is None:
        config = BootstrapConfig()

    results = []
    for (model, code), group_df in df.groupby([model_col, code_col]):
        values = group_df[score_col].dropna().values
        mean_val, lower, upper = bootstrap_binary_ci(values, config=config)
        results.append(
            {
                model_col: model,
                code_col: code,
                "mean": mean_val,
                "ci_lower": lower,
                "ci_upper": upper,
                "n": len(values),
            }
        )
    return pd.DataFrame(results)


def hierarchical_weighted_mean_ci(
    aggregated: pd.DataFrame,
    *,
    config: Optional[BootstrapConfig] = None,
    sum_col: str = "value_sum",
    count_col: str = "value_count",
) -> dict[str, float | int]:
    """Compute a hierarchical CI for a weighted mean.

    Parameters
    ----------
    aggregated:
        Participant-conversation summary rows for one analysis cell.
    config:
        Optional bootstrap configuration.
    sum_col:
        Column containing conversation-level value sums.
    count_col:
        Column containing conversation-level row counts.

    Returns
    -------
    dict[str, float | int]
        Weighted mean estimate, percentile interval, supported participant
        count, and bootstrap draw count.
    """
    if config is None:
        config = BootstrapConfig()

    prepared = _prepare_hierarchical_mean_inputs(
        aggregated,
        sum_col=sum_col,
        count_col=count_col,
    )
    if not prepared:
        return _empty_ci_result(config)

    observed = _hierarchical_observed_mean(prepared)
    n_supported = len(prepared)
    if n_supported == 1:
        return _singleton_ci_result(observed, config)

    boot_stats = _bootstrap_hierarchical_mean(prepared, config)
    return _ci_result(observed, boot_stats, n_supported, config)


def hierarchical_weighted_delta_ci(
    aggregated_a: pd.DataFrame,
    aggregated_b: pd.DataFrame,
    *,
    config: Optional[BootstrapConfig] = None,
    sum_col: str = "value_sum",
    count_col: str = "value_count",
) -> dict[str, float | int]:
    """Compute a hierarchical CI for a weighted mean delta.

    Support is defined pairwise: only participants with non-zero counts in
    both analysis cells are retained.

    Parameters
    ----------
    aggregated_a:
        Participant-conversation summary rows for arm A.
    aggregated_b:
        Participant-conversation summary rows for arm B.
    config:
        Optional bootstrap configuration.
    sum_col:
        Column containing conversation-level value sums.
    count_col:
        Column containing conversation-level row counts.

    Returns
    -------
    dict[str, float | int]
        Delta estimate, percentile confidence interval, support count, and
        bootstrap draw count.
    """
    if config is None:
        config = BootstrapConfig()

    prepared = _prepare_hierarchical_delta_inputs(
        aggregated_a,
        aggregated_b,
        sum_col=sum_col,
        count_col=count_col,
    )
    if not prepared:
        return _empty_ci_result(config)

    observed = _hierarchical_observed_delta(prepared)
    n_supported = len(prepared)
    if n_supported == 1:
        return _singleton_ci_result(observed, config)

    boot_stats = _bootstrap_hierarchical_delta(prepared, config)
    return _ci_result(observed, boot_stats, n_supported, config)


def _prepare_hierarchical_mean_inputs(
    aggregated: pd.DataFrame,
    *,
    sum_col: str,
    count_col: str,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Prepare conversation-level arrays for hierarchical mean resampling."""
    valid = _normalize_hierarchical_frame(
        aggregated, sum_col=sum_col, count_col=count_col
    )
    if valid.empty:
        return {}

    prepared: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for participant_id, group in valid.groupby(_PARTICIPANT_COL, sort=False):
        prepared[participant_id] = (
            group[sum_col].to_numpy(dtype=float),
            group[count_col].to_numpy(dtype=float),
        )
    return prepared


def _prepare_hierarchical_delta_inputs(
    aggregated_a: pd.DataFrame,
    aggregated_b: pd.DataFrame,
    *,
    sum_col: str,
    count_col: str,
) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """Prepare aligned conversation arrays for hierarchical delta resampling."""
    prepared_a = _normalize_delta_input_frame(
        aggregated_a,
        sum_col=sum_col,
        count_col=count_col,
        suffix="a",
    )
    prepared_b = _normalize_delta_input_frame(
        aggregated_b,
        sum_col=sum_col,
        count_col=count_col,
        suffix="b",
    )
    if prepared_a.empty or prepared_b.empty:
        return {}

    totals_a = (
        prepared_a.groupby(_PARTICIPANT_COL, sort=False)[f"{count_col}_a"]
        .sum()
        .astype(float)
    )
    totals_b = (
        prepared_b.groupby(_PARTICIPANT_COL, sort=False)[f"{count_col}_b"]
        .sum()
        .astype(float)
    )
    supported_participants = [
        participant_id
        for participant_id in totals_a.index.intersection(totals_b.index)
        if totals_a[participant_id] > 0.0 and totals_b[participant_id] > 0.0
    ]
    if not supported_participants:
        return {}

    prepared: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}
    for participant_id in supported_participants:
        participant_a = prepared_a[prepared_a[_PARTICIPANT_COL] == participant_id]
        participant_b = prepared_b[prepared_b[_PARTICIPANT_COL] == participant_id]
        merged = participant_a.merge(
            participant_b,
            on=[_PARTICIPANT_COL, _CONVERSATION_COL],
            how="outer",
        ).fillna(0.0)
        prepared[participant_id] = (
            merged[f"{sum_col}_a"].to_numpy(dtype=float),
            merged[f"{count_col}_a"].to_numpy(dtype=float),
            merged[f"{sum_col}_b"].to_numpy(dtype=float),
            merged[f"{count_col}_b"].to_numpy(dtype=float),
        )
    return prepared


def _normalize_delta_input_frame(
    aggregated: pd.DataFrame,
    *,
    sum_col: str,
    count_col: str,
    suffix: str,
) -> pd.DataFrame:
    """Normalize one arm of delta inputs for hierarchical resampling."""
    renamed = _normalize_hierarchical_frame(
        aggregated, sum_col=sum_col, count_col=count_col
    )
    return renamed.rename(
        columns={
            sum_col: f"{sum_col}_{suffix}",
            count_col: f"{count_col}_{suffix}",
        }
    )


def _normalize_hierarchical_frame(
    aggregated: pd.DataFrame,
    *,
    sum_col: str,
    count_col: str,
) -> pd.DataFrame:
    """Normalize one hierarchical summary frame."""
    if aggregated.empty:
        return pd.DataFrame()

    normalized = aggregated[
        [_PARTICIPANT_COL, _CONVERSATION_COL, sum_col, count_col]
    ].copy()
    normalized[_PARTICIPANT_COL] = normalized[_PARTICIPANT_COL].astype(str).str.strip()
    normalized[_CONVERSATION_COL] = (
        normalized[_CONVERSATION_COL].astype(str).str.strip()
    )
    normalized[sum_col] = pd.to_numeric(normalized[sum_col], errors="coerce")
    normalized[count_col] = pd.to_numeric(normalized[count_col], errors="coerce")
    normalized = normalized[normalized[_PARTICIPANT_COL] != ""].copy()
    normalized = normalized[normalized[_CONVERSATION_COL] != ""].copy()
    normalized = normalized[normalized[count_col] > 0.0].copy()
    return normalized


def _hierarchical_observed_mean(
    prepared: dict[str, tuple[np.ndarray, np.ndarray]],
) -> float:
    """Return the observed weighted mean from prepared hierarchical inputs."""
    observed_sum = sum(float(sums.sum()) for sums, _ in prepared.values())
    observed_count = sum(float(counts.sum()) for _, counts in prepared.values())
    return float(observed_sum / observed_count)


def _hierarchical_observed_delta(
    prepared: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
) -> float:
    """Return the observed weighted delta from prepared hierarchical inputs."""
    observed_sum_a = sum(float(sums_a.sum()) for sums_a, _, _, _ in prepared.values())
    observed_count_a = sum(
        float(counts_a.sum()) for _, counts_a, _, _ in prepared.values()
    )
    observed_sum_b = sum(float(sums_b.sum()) for _, _, sums_b, _ in prepared.values())
    observed_count_b = sum(
        float(counts_b.sum()) for _, _, _, counts_b in prepared.values()
    )
    return float(
        (observed_sum_a / observed_count_a) - (observed_sum_b / observed_count_b)
    )


def _bootstrap_hierarchical_mean(
    prepared: dict[str, tuple[np.ndarray, np.ndarray]],
    config: BootstrapConfig,
) -> np.ndarray:
    """Return bootstrap draws for a hierarchical weighted mean."""
    rng = np.random.default_rng(config.seed)
    participant_ids = list(prepared)
    boot_stats = np.empty(config.n_boot, dtype=float)
    for draw_index in range(config.n_boot):
        total_sum = 0.0
        total_count = 0.0
        sampled_participants = rng.choice(
            participant_ids,
            size=len(participant_ids),
            replace=True,
        )
        for participant_id in sampled_participants:
            sums, counts = prepared[participant_id]
            sampled_sum, sampled_count = _resample_conversation_totals(
                sums,
                counts,
                rng=rng,
            )
            total_sum += sampled_sum
            total_count += sampled_count
        boot_stats[draw_index] = total_sum / total_count
    return boot_stats


def _bootstrap_hierarchical_delta(
    prepared: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
    config: BootstrapConfig,
) -> np.ndarray:
    """Return bootstrap draws for a hierarchical weighted delta."""
    rng = np.random.default_rng(config.seed)
    participant_ids = list(prepared)
    boot_stats = np.empty(config.n_boot, dtype=float)
    for draw_index in range(config.n_boot):
        sampled_participants = rng.choice(
            participant_ids,
            size=len(participant_ids),
            replace=True,
        )
        boot_stats[draw_index] = _resampled_delta_statistic(
            prepared,
            sampled_participants,
            rng,
        )
    return boot_stats


def _resampled_delta_statistic(
    prepared: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
    sampled_participants: np.ndarray,
    rng: np.random.Generator,
) -> float:
    """Return one hierarchical bootstrap delta draw."""
    total_sum_a = 0.0
    total_count_a = 0.0
    total_sum_b = 0.0
    total_count_b = 0.0
    for participant_id in sampled_participants:
        sums_a, counts_a, sums_b, counts_b = prepared[participant_id]
        conversation_draw_index = rng.integers(0, len(sums_a), size=len(sums_a))
        total_sum_a += float(sums_a[conversation_draw_index].sum())
        total_count_a += float(counts_a[conversation_draw_index].sum())
        total_sum_b += float(sums_b[conversation_draw_index].sum())
        total_count_b += float(counts_b[conversation_draw_index].sum())
    return float((total_sum_a / total_count_a) - (total_sum_b / total_count_b))


def _ci_result(
    observed: float,
    boot_stats: np.ndarray,
    n_supported: int,
    config: BootstrapConfig,
) -> dict[str, float | int]:
    """Return the standard CI payload for non-empty hierarchical summaries."""
    ci_low, ci_high = _percentile_interval(boot_stats, config.ci)
    return {
        "estimate": observed,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "n_participants_supported": n_supported,
        "cluster_boot_n": config.n_boot,
    }


def _resample_conversation_totals(
    sums: np.ndarray,
    counts: np.ndarray,
    *,
    rng: np.random.Generator,
) -> tuple[float, float]:
    """Resample conversations within one participant and return totals."""
    if len(sums) == 1:
        return float(sums[0]), float(counts[0])
    draw_index = rng.integers(0, len(sums), size=len(sums))
    return float(sums[draw_index].sum()), float(counts[draw_index].sum())


def _empty_ci_result(config: BootstrapConfig) -> dict[str, float | int]:
    """Return a standard empty-result payload for CI helpers."""
    return {
        "estimate": np.nan,
        "ci_low": np.nan,
        "ci_high": np.nan,
        "n_participants_supported": 0,
        "cluster_boot_n": config.n_boot,
    }


def _singleton_ci_result(
    observed: float,
    config: BootstrapConfig,
) -> dict[str, float | int]:
    """Return a standard singleton-support payload for CI helpers."""
    return {
        "estimate": observed,
        "ci_low": observed,
        "ci_high": observed,
        "n_participants_supported": 1,
        "cluster_boot_n": config.n_boot,
    }


def _percentile_interval(
    values: np.ndarray, confidence_level: float
) -> tuple[float, float]:
    """Return the percentile interval for a bootstrap sample."""
    alpha = (1.0 - confidence_level) / 2.0
    return (
        float(np.quantile(values, alpha)),
        float(np.quantile(values, 1.0 - alpha)),
    )
