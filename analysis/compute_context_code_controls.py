"""Estimate requested-depth and prior-annotation prevalence effects.

This analysis reuses row-level context data from ``compute_context_effects``
and fits linear probability models (LPMs):

``binary_score ~ requested_depth + code_prevalence``

where ``code_prevalence`` is the prevalence of the sample's target annotation
in preceding assistant messages (using annotation-specific cutoffs).

Each model is fit within a cohort.
Outputs are CSV summaries in ``analysis/data/context_effects/``.
"""

import argparse
import logging
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analysis.artifact_paths import DATA_DIR, FIGURE_DIR, ensure_output_dirs
from analysis.bootstrap import BootstrapConfig
from analysis.compute_context_effects import (
    DEFAULT_BASELINE_LOGS_DIR,
    DEFAULT_LOG_METADATA_CACHE,
    DEFAULT_LOGS_DIR,
    DEFAULT_ROWS_CACHE,
    DEFAULT_SAMPLE_ROWS_CACHE_DIR,
    ContextEffectCachePaths,
    _is_validates_code,
    load_context_effect_data,
    resolve_context_excluded_window_ids,
)
from analysis.participant_clustered import attach_participant_ids
from analysis.plot_style import apply_plot_style
from llm_delusion_eval.constants import normalize_id

logger = logging.getLogger(__name__)

DEFAULT_DATA_DIR = DATA_DIR / "context_effects"
DEFAULT_FIGURES_DIR = FIGURE_DIR
DEFAULT_MIN_EFFECTIVE_FRACTION = 1.0
MIN_UNIQUE_PREVALENCE_VALUES = 2
# The LPM path refits the model on every hierarchical bootstrap draw, so use a
# smaller default than the vectorized prevalence summaries.
DEFAULT_CLUSTER_BOOTSTRAP_CONFIG = BootstrapConfig(n_boot=2_000)

apply_plot_style()
plt.switch_backend("Agg")


def _filter_rows(
    rows_df: pd.DataFrame,
    codes: list[str],
    *,
    include_validates_codes: bool,
    min_effective_context_messages: int,
    min_effective_fraction_of_requested: float,
) -> pd.DataFrame:
    """Filter rows for code subset and effective-context requirements.

    Parameters
    ----------
    rows_df:
        Row-level context dataframe.
    codes:
        Optional normalized annotation IDs to keep.
    include_validates_codes:
        Whether to include ``validates-*`` codes when ``codes`` is not given.
    min_effective_context_messages:
        Minimum effective context length in messages.
    min_effective_fraction_of_requested:
        Minimum realized/requested context ratio.

    Returns
    -------
    pd.DataFrame
        Filtered rows.
    """
    filtered = rows_df.copy()
    if codes:
        normalized_codes = {normalize_id(code) for code in codes}
        filtered = filtered[filtered["annotation_id"].isin(normalized_codes)]
        if filtered.empty:
            raise ValueError("No rows matched requested code filters.")
    elif not include_validates_codes:
        filtered = filtered[
            ~filtered["annotation_id"].astype(str).map(_is_validates_code)
        ]
        if filtered.empty:
            raise ValueError(
                "No rows remained after default validates-* exclusion. "
                "Use --include-validates-codes to keep these codes."
            )

    if min_effective_context_messages > 0:
        filtered = filtered[
            filtered["effective_context_length"] >= min_effective_context_messages
        ]

    if min_effective_fraction_of_requested > 0:
        required = (
            min_effective_fraction_of_requested * filtered["requested_context_messages"]
        )
        filtered = filtered[filtered["effective_context_length"] >= required]

    if filtered.empty:
        raise ValueError("No rows remained after applying context filters.")
    return filtered


def _fit_linear_probability_model(group_df: pd.DataFrame) -> Optional[dict[str, float]]:
    """Fit an OLS linear probability model for one group.

    Parameters
    ----------
    group_df:
        Dataframe for one cohort.

    Returns
    -------
    Optional[dict[str, float]]
        Coefficients, standard errors, and fit diagnostics, or ``None`` when
        the model cannot be estimated for the cohort.
    """
    design = pd.DataFrame(index=group_df.index)
    design["intercept"] = 1.0
    design["requested_depth_100"] = (
        pd.to_numeric(group_df["requested_context_messages"], errors="coerce").fillna(
            0.0
        )
        / 100.0
    )
    design["prior_annotation_prevalence"] = pd.to_numeric(
        group_df["prior_annotation_prevalence"], errors="coerce"
    ).fillna(0.0)
    if group_df["score"].isna().all() or (
        design["prior_annotation_prevalence"].nunique(dropna=True)
        < MIN_UNIQUE_PREVALENCE_VALUES
    ):
        return None

    # Explicit binary target: score > 0 is treated as a positive label.
    numeric_score = pd.to_numeric(group_df["score"], errors="coerce")
    y = pd.Series(
        np.where(numeric_score.isna(), np.nan, (numeric_score > 0).astype(float)),
        index=group_df.index,
        dtype=float,
    )
    model_df = pd.concat([design, y.rename("score")], axis=1).dropna()
    if model_df.empty:
        return None

    x = model_df.drop(columns=["score"]).to_numpy(dtype=float)
    y_values = model_df["score"].to_numpy(dtype=float)
    n_rows, n_params = x.shape
    if n_rows <= n_params:
        return None

    try:
        beta, _, _, _ = np.linalg.lstsq(x, y_values, rcond=None)
    except np.linalg.LinAlgError:
        return None

    y_hat = x @ beta
    residual = y_values - y_hat
    rank = int(np.linalg.matrix_rank(x))
    dof = n_rows - rank
    if dof <= 0:
        return None

    rss = float(np.dot(residual, residual))
    centered = y_values - np.mean(y_values)
    tss = float(np.dot(centered, centered))
    r_squared = 1.0 - (rss / tss) if tss > 0 else 0.0

    sigma2 = rss / dof
    xtx_inv = np.linalg.pinv(x.T @ x)
    variances = sigma2 * np.diag(xtx_inv)
    se = np.sqrt(np.clip(variances, a_min=0.0, a_max=None))

    columns = model_df.drop(columns=["score"]).columns.to_list()
    coefficients = dict(zip(columns, beta))
    standard_errors = dict(zip(columns, se))

    return {
        "n_rows": float(n_rows),
        "r_squared": r_squared,
        "coef_requested_depth_100": float(coefficients["requested_depth_100"]),
        "se_requested_depth_100": float(standard_errors["requested_depth_100"]),
        "coef_prior_annotation_prevalence": float(
            coefficients["prior_annotation_prevalence"]
        ),
        "se_prior_annotation_prevalence": float(
            standard_errors["prior_annotation_prevalence"]
        ),
    }


def _cluster_bootstrap_lpm_ci(
    group_df: pd.DataFrame,
    *,
    config: BootstrapConfig,
) -> Optional[dict[str, float | int | str]]:
    """Bootstrap hierarchical confidence intervals for one LPM fit.

    Parameters
    ----------
    group_df:
        Dataframe for one fitted cohort.
    config:
        Cluster bootstrap configuration.

    Returns
    -------
    Optional[dict[str, float | int | str]]
        Hierarchical coefficient intervals and support metadata, or ``None`` when
        participant mappings are unavailable.
    """
    if (
        "participant" not in group_df.columns
        or "conversation_id" not in group_df.columns
    ):
        return None

    participant_df = group_df.copy()
    participant_df["participant"] = (
        participant_df["participant"].fillna("").astype(str).str.strip()
    )
    participant_df["conversation_id"] = (
        participant_df["conversation_id"].fillna("").astype(str).str.strip()
    )
    participant_df = participant_df[participant_df["participant"] != ""].copy()
    participant_df = participant_df[participant_df["conversation_id"] != ""].copy()
    participants = participant_df["participant"].unique().tolist()
    n_supported = len(participants)
    if n_supported == 0:
        return None

    fit = _fit_linear_probability_model(participant_df)
    if fit is None:
        return None

    if n_supported == 1:
        return {
            "coef_requested_depth_ci95_low_per_100_messages_pp": float(
                fit["coef_requested_depth_100"] * 100.0
            ),
            "coef_requested_depth_ci95_high_per_100_messages_pp": float(
                fit["coef_requested_depth_100"] * 100.0
            ),
            "coef_prior_annotation_prevalence_ci95_low_per_10pp": float(
                fit["coef_prior_annotation_prevalence"] * 10.0
            ),
            "coef_prior_annotation_prevalence_ci95_high_per_10pp": float(
                fit["coef_prior_annotation_prevalence"] * 10.0
            ),
            "n_participants_supported": n_supported,
            "cluster_boot_n": config.n_boot,
            "ci_method": "hierarchical_participant_conversation_bootstrap",
        }

    participant_groups = {
        participant: {
            conversation_id: conversation_group.copy()
            for conversation_id, conversation_group in participant_group.groupby(
                "conversation_id",
                sort=False,
            )
        }
        for participant, participant_group in participant_df.groupby(
            "participant",
            sort=False,
        )
    }
    rng = np.random.default_rng(config.seed)
    depth_boot: list[float] = []
    prevalence_boot: list[float] = []

    for _ in range(config.n_boot):
        sampled_participants = rng.choice(
            participants,
            size=n_supported,
            replace=True,
        )
        sampled_frames: list[pd.DataFrame] = []
        for participant in sampled_participants:
            conversation_groups = participant_groups[participant]
            conversation_ids = list(conversation_groups)
            sampled_conversations = rng.choice(
                conversation_ids,
                size=len(conversation_ids),
                replace=True,
            )
            sampled_frames.extend(
                conversation_groups[conversation_id]
                for conversation_id in sampled_conversations
            )
        sampled_df = pd.concat(sampled_frames, ignore_index=True)
        boot_fit = _fit_linear_probability_model(sampled_df)
        if boot_fit is None:
            continue
        depth_boot.append(float(boot_fit["coef_requested_depth_100"] * 100.0))
        prevalence_boot.append(
            float(boot_fit["coef_prior_annotation_prevalence"] * 10.0)
        )

    if not depth_boot or not prevalence_boot:
        return {
            "coef_requested_depth_ci95_low_per_100_messages_pp": float(
                fit["coef_requested_depth_100"] * 100.0
            ),
            "coef_requested_depth_ci95_high_per_100_messages_pp": float(
                fit["coef_requested_depth_100"] * 100.0
            ),
            "coef_prior_annotation_prevalence_ci95_low_per_10pp": float(
                fit["coef_prior_annotation_prevalence"] * 10.0
            ),
            "coef_prior_annotation_prevalence_ci95_high_per_10pp": float(
                fit["coef_prior_annotation_prevalence"] * 10.0
            ),
            "n_participants_supported": n_supported,
            "cluster_boot_n": config.n_boot,
            "ci_method": "hierarchical_participant_conversation_fallback_point",
        }

    return {
        "coef_requested_depth_ci95_low_per_100_messages_pp": float(
            np.quantile(depth_boot, 0.025)
        ),
        "coef_requested_depth_ci95_high_per_100_messages_pp": float(
            np.quantile(depth_boot, 0.975)
        ),
        "coef_prior_annotation_prevalence_ci95_low_per_10pp": float(
            np.quantile(prevalence_boot, 0.025)
        ),
        "coef_prior_annotation_prevalence_ci95_high_per_10pp": float(
            np.quantile(prevalence_boot, 0.975)
        ),
        "n_participants_supported": n_supported,
        "cluster_boot_n": config.n_boot,
        "ci_method": "hierarchical_participant_conversation_bootstrap",
    }


def _run_grouped_models(
    rows_df: pd.DataFrame,
    group_cols: list[str],
    *,
    min_rows: int,
) -> pd.DataFrame:
    """Fit per-group LPM controls and return a summary table.

    Parameters
    ----------
    rows_df:
        Filtered row-level context dataframe.
    group_cols:
        Grouping columns for model fitting.
    min_rows:
        Minimum rows per group to attempt estimation.

    Returns
    -------
    pd.DataFrame
        Group-level model summaries.
    """
    output_rows: list[dict[str, object]] = []
    for key, group in rows_df.groupby(group_cols, dropna=False):
        if len(group) < min_rows:
            continue

        fit = _fit_linear_probability_model(group)
        if fit is None:
            continue
        clustered_ci = _cluster_bootstrap_lpm_ci(
            group,
            config=DEFAULT_CLUSTER_BOOTSTRAP_CONFIG,
        )

        key_dict = dict(zip(group_cols, key))
        numeric_score = pd.to_numeric(group["score"], errors="coerce")
        binary_score = pd.Series(
            np.where(numeric_score.isna(), np.nan, (numeric_score > 0).astype(float)),
            index=group.index,
            dtype=float,
        )
        with_code = binary_score[
            pd.to_numeric(group["context_has_code"], errors="coerce") > 0
        ]
        without_code = binary_score[
            pd.to_numeric(group["context_has_code"], errors="coerce") <= 0
        ]
        prior_prevalence = pd.to_numeric(
            group["prior_annotation_prevalence"], errors="coerce"
        )
        requested_levels = sorted(
            pd.to_numeric(group["requested_context_messages"], errors="coerce")
            .dropna()
            .astype(int)
            .unique()
            .tolist()
        )

        if clustered_ci is None:
            z_value = 1.96
            coef_requested_depth_ci95_low_per_100_messages_pp = float(
                (
                    fit["coef_requested_depth_100"]
                    - z_value * fit["se_requested_depth_100"]
                )
                * 100.0
            )
            coef_requested_depth_ci95_high_per_100_messages_pp = float(
                (
                    fit["coef_requested_depth_100"]
                    + z_value * fit["se_requested_depth_100"]
                )
                * 100.0
            )
            coef_prior_annotation_prevalence_ci95_low_per_10pp = float(
                (
                    fit["coef_prior_annotation_prevalence"]
                    - z_value * fit["se_prior_annotation_prevalence"]
                )
                * 10.0
            )
            coef_prior_annotation_prevalence_ci95_high_per_10pp = float(
                (
                    fit["coef_prior_annotation_prevalence"]
                    + z_value * fit["se_prior_annotation_prevalence"]
                )
                * 10.0
            )
            n_participants_supported = 0
            cluster_boot_n = 0
            ci_method = "analytic_ols"
        else:
            coef_requested_depth_ci95_low_per_100_messages_pp = float(
                clustered_ci["coef_requested_depth_ci95_low_per_100_messages_pp"]
            )
            coef_requested_depth_ci95_high_per_100_messages_pp = float(
                clustered_ci["coef_requested_depth_ci95_high_per_100_messages_pp"]
            )
            coef_prior_annotation_prevalence_ci95_low_per_10pp = float(
                clustered_ci["coef_prior_annotation_prevalence_ci95_low_per_10pp"]
            )
            coef_prior_annotation_prevalence_ci95_high_per_10pp = float(
                clustered_ci["coef_prior_annotation_prevalence_ci95_high_per_10pp"]
            )
            n_participants_supported = int(clustered_ci["n_participants_supported"])
            cluster_boot_n = int(clustered_ci["cluster_boot_n"])
            ci_method = str(clustered_ci["ci_method"])

        row = {
            **key_dict,
            "n_rows": int(fit["n_rows"]),
            "requested_context_levels": ",".join(
                str(level) for level in requested_levels
            ),
            "score_mean": float(binary_score.mean()),
            "score_mean_with_code": (
                float(with_code.mean()) if not with_code.empty else np.nan
            ),
            "score_mean_without_code": (
                float(without_code.mean()) if not without_code.empty else np.nan
            ),
            "share_rows_with_code": float(
                pd.to_numeric(group["context_has_code"], errors="coerce").mean()
            ),
            "mean_prior_annotation_prevalence": float(prior_prevalence.mean()),
            "mean_prior_annotation_scored_messages": float(
                pd.to_numeric(
                    group["prior_annotation_scored_messages"], errors="coerce"
                ).mean()
            ),
            "coef_requested_depth_per_100_messages_pp": float(
                fit["coef_requested_depth_100"] * 100.0
            ),
            "coef_requested_depth_ci95_low_per_100_messages_pp": (
                coef_requested_depth_ci95_low_per_100_messages_pp
            ),
            "coef_requested_depth_ci95_high_per_100_messages_pp": (
                coef_requested_depth_ci95_high_per_100_messages_pp
            ),
            "coef_prior_annotation_prevalence_per_10pp": float(
                fit["coef_prior_annotation_prevalence"] * 10.0
            ),
            "coef_prior_annotation_prevalence_ci95_low_per_10pp": (
                coef_prior_annotation_prevalence_ci95_low_per_10pp
            ),
            "coef_prior_annotation_prevalence_ci95_high_per_10pp": (
                coef_prior_annotation_prevalence_ci95_high_per_10pp
            ),
            "n_participants_supported": n_participants_supported,
            "cluster_boot_n": cluster_boot_n,
            "ci_method": ci_method,
            "r_squared": float(fit["r_squared"]),
        }
        output_rows.append(row)

    return pd.DataFrame(output_rows)


def _save_category_forest_plot(
    category_results: pd.DataFrame,
    output_dir: Path,
) -> Path:
    """Write a forest plot of category-level model coefficients.

    Parameters
    ----------
    category_results:
        Category-level model summary table.
    output_dir:
        Directory where figure files are written.

    Returns
    -------
    Path
        Output PDF path.
    """
    if category_results.empty:
        raise ValueError("No category model rows available for forest plot.")

    df = category_results.copy()
    df["category_label"] = df["category"].astype(str)
    if "model_label" in df.columns and df["model_label"].nunique() > 1:
        df["category_label"] = (
            df["model_label"].astype(str) + " | " + df["category"].astype(str)
        )

    y_pos = np.arange(len(df))
    fig, axes = plt.subplots(ncols=2, figsize=(8.0, 3.8), sharey=True)

    depth = df["coef_requested_depth_per_100_messages_pp"].to_numpy(dtype=float)
    depth_low = df["coef_requested_depth_ci95_low_per_100_messages_pp"].to_numpy(
        dtype=float
    )
    depth_high = df["coef_requested_depth_ci95_high_per_100_messages_pp"].to_numpy(
        dtype=float
    )
    axes[0].errorbar(
        depth,
        y_pos,
        xerr=[depth - depth_low, depth_high - depth],
        fmt="o",
        color="#1f77b4",
        capsize=3,
    )
    axes[0].axvline(0.0, color="black", linewidth=1, alpha=0.7)
    axes[0].set_xlabel("Depth Effect (pp per +100 messages)")
    axes[0].set_yticks(y_pos)
    axes[0].set_yticklabels(df["category_label"].tolist())
    axes[0].grid(True, axis="x", alpha=0.3)

    prevalence = df["coef_prior_annotation_prevalence_per_10pp"].to_numpy(dtype=float)
    prevalence_low = df["coef_prior_annotation_prevalence_ci95_low_per_10pp"].to_numpy(
        dtype=float
    )
    prevalence_high = df[
        "coef_prior_annotation_prevalence_ci95_high_per_10pp"
    ].to_numpy(dtype=float)
    axes[1].errorbar(
        prevalence,
        y_pos,
        xerr=[prevalence - prevalence_low, prevalence_high - prevalence],
        fmt="o",
        color="#d62728",
        capsize=3,
    )
    axes[1].axvline(0.0, color="black", linewidth=1, alpha=0.7)
    axes[1].set_xlabel("Code Prevalence Effect (pp per +10pp)")
    axes[1].grid(True, axis="x", alpha=0.3)

    fig.tight_layout()
    output_pdf = output_dir / "context_code_control_forest_by_category.pdf"
    output_png = output_dir / "context_code_control_forest_by_category.png"
    fig.savefig(output_pdf)
    fig.savefig(output_png)
    plt.close(fig)
    logger.info("Wrote %s and %s", output_pdf, output_png)
    return output_pdf


def _write_forest_plot(category_results: pd.DataFrame, output_dir: Path) -> None:
    """Generate the forest-plot output for context code-control analysis.

    Parameters
    ----------
    category_results:
        Category-level model summary table.
    output_dir:
        Figure output directory.
    """
    ensure_output_dirs(output_dir)
    _save_category_forest_plot(category_results, output_dir)


def _build_arg_parser() -> argparse.ArgumentParser:
    """Create the CLI parser for context code-control analysis.

    Returns
    -------
    argparse.ArgumentParser
        Configured parser.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Estimate requested-depth and prior-annotation prevalence effects "
            "from context eval logs using per-group linear probability models."
        )
    )
    parser.add_argument(
        "--logs-dir",
        type=Path,
        default=DEFAULT_LOGS_DIR,
        help="Directory containing context eval .eval files (default: logs-context).",
    )
    parser.add_argument(
        "--baseline-logs-dir",
        type=Path,
        default=DEFAULT_BASELINE_LOGS_DIR,
        help=(
            "Directory containing baseline eval .eval files for requested-context "
            "zero points (default: logs)."
        ),
    )
    parser.add_argument(
        "--excluded-participants",
        type=str,
        default=None,
        help=(
            "Comma-separated participant IDs to exclude. If omitted, uses "
            "LLM_DELUSIONS_EXCLUDED_PARTICIPANTS (default: include all). "
            "Use an empty string to disable exclusions."
        ),
    )
    parser.add_argument(
        "--codes",
        type=str,
        default="",
        help=(
            "Optional comma-separated annotation IDs to include. "
            "If omitted, validates-* codes are excluded by default."
        ),
    )
    parser.add_argument(
        "--include-validates-codes",
        action="store_true",
        help=(
            "Include validates-* codes when --codes is not specified. "
            "Explicit --codes filters still take precedence."
        ),
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help="Output directory for CSV files (default: analysis/data/context_effects).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_FIGURES_DIR,
        help="Output directory for figure files (default: analysis/figures).",
    )
    parser.add_argument(
        "--rows-cache",
        type=Path,
        default=DEFAULT_ROWS_CACHE,
        help="Parquet cache for parsed context rows.",
    )
    parser.add_argument(
        "--log-metadata-cache",
        type=Path,
        default=DEFAULT_LOG_METADATA_CACHE,
        help="JSON cache for per-log manifest/sample metadata.",
    )
    parser.add_argument(
        "--sample-rows-cache-dir",
        type=Path,
        default=DEFAULT_SAMPLE_ROWS_CACHE_DIR,
        help="Directory cache for parsed rows per log revision.",
    )
    parser.add_argument(
        "--min-effective-context-messages",
        type=int,
        default=0,
        help="Drop rows below this effective context length before model fitting.",
    )
    parser.add_argument(
        "--min-effective-fraction-of-requested",
        type=float,
        default=DEFAULT_MIN_EFFECTIVE_FRACTION,
        help=(
            "Drop rows where effective context is below this fraction of requested "
            "context (default: 1.0)."
        ),
    )
    parser.add_argument(
        "--min-rows",
        type=int,
        default=80,
        help="Minimum rows per cohort required to fit a model (default: 80).",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable reading/writing context row cache files.",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose logging.",
    )
    return parser


def _validate_cli_args(args: argparse.Namespace) -> None:
    """Validate CLI values.

    Parameters
    ----------
    args:
        Parsed CLI arguments.
    """
    if args.min_effective_context_messages < 0:
        raise ValueError("--min-effective-context-messages must be >= 0.")
    if not 0.0 <= args.min_effective_fraction_of_requested <= 1.0:
        raise ValueError(
            "--min-effective-fraction-of-requested must be between 0 and 1."
        )
    if args.min_rows <= 0:
        raise ValueError("--min-rows must be > 0.")


def main() -> None:
    """Entry point for context code-control analysis."""
    args = _build_arg_parser().parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )
    _validate_cli_args(args)

    codes = [entry.strip() for entry in args.codes.split(",") if entry.strip()]
    cache_paths = ContextEffectCachePaths(
        rows_cache=args.rows_cache,
        log_metadata_cache=args.log_metadata_cache,
        sample_rows_cache_dir=args.sample_rows_cache_dir,
    )
    _, excluded_window_ids = resolve_context_excluded_window_ids(
        args.excluded_participants
    )

    rows_df = load_context_effect_data(
        args.logs_dir,
        baseline_logs_dir=args.baseline_logs_dir,
        cache_paths=cache_paths,
        use_cache=not args.no_cache,
        excluded_window_ids=excluded_window_ids,
    )
    participant_rows_df = attach_participant_ids(rows_df)
    if participant_rows_df.empty and not rows_df.empty:
        logger.warning(
            "Proceeding without participant/conversation mappings; using "
            "analytic OLS intervals for context code-control summaries."
        )
    else:
        rows_df = participant_rows_df
    rows_df = _filter_rows(
        rows_df,
        codes,
        include_validates_codes=args.include_validates_codes,
        min_effective_context_messages=args.min_effective_context_messages,
        min_effective_fraction_of_requested=args.min_effective_fraction_of_requested,
    )

    code_results = _run_grouped_models(
        rows_df,
        [
            "model",
            "model_label",
            "reasoning_effort",
            "annotation_id",
            "code_short",
            "category",
        ],
        min_rows=args.min_rows,
    )
    category_results = _run_grouped_models(
        rows_df,
        ["model", "model_label", "reasoning_effort", "category"],
        min_rows=args.min_rows,
    )

    ensure_output_dirs(args.data_dir, args.output_dir)
    code_output = args.data_dir / "context_code_control_lpm_by_code.csv"
    category_output = args.data_dir / "context_code_control_lpm_by_category.csv"
    code_results.to_csv(code_output, index=False)
    category_results.to_csv(category_output, index=False)
    _write_forest_plot(category_results=category_results, output_dir=args.output_dir)

    logger.info("Wrote %s", code_output)
    logger.info("Wrote %s", category_output)
    logger.info(
        "Done. Fitted %d code-level model(s) and %d category-level model(s).",
        len(code_results),
        len(category_results),
    )


if __name__ == "__main__":
    main()
