# Analysis Pipeline

Scripts and utilities for generating publication figures, tables, and
statistical analyses from evaluation results.

## Quick Start

```bash
# From the evals repo root:

# 0. Install dependencies needed by analysis scripts
uv sync

# 1. Generate summary.json with CIs and delta data
uv run python -m llm_delusion_eval.scripts.generate_report \
    --logs-dir logs --output-dir report

# Fast iteration (skip bootstrap CIs):
uv run python -m llm_delusion_eval.scripts.generate_report \
    --logs-dir logs --output-dir report --no-bootstrap

# 2a. Generate figures only
uv run python -m analysis.generate_figures \
    --summary report/summary.json \
    --figures-only

# 2b. Generate both figures and table exports
uv run python -m analysis.generate_figures --summary report/summary.json

# Optional: include validates-* codes in code-level outputs
uv run python -m analysis.generate_figures \
    --summary report/summary.json \
    --include-validates-codes
```

## Core Workflow

```bash
# Build report outputs from window-only eval logs
uv run python -m llm_delusion_eval.scripts.generate_report \
    --logs-dir logs --output-dir report

# Generate publication figures/tables
uv run python -m analysis.generate_figures \
    --summary report/summary.json

# Optional analyses that use report outputs only
uv run python -m analysis.compute_reasoning_effects
uv run python -m analysis.compute_sequential_dynamics
uv run python -m analysis.compute_code_correlations
uv run python -m analysis.compute_code_residuals
```

## Participant Filtering

By default, report/context analyses include all participants.
This behavior is shared across scripts and controlled by
`LLM_DELUSIONS_EXCLUDED_PARTICIPANTS`.

```bash
# Exclude participants 103,107,115
LLM_DELUSIONS_EXCLUDED_PARTICIPANTS=1,2,3 \
uv run python -m llm_delusion_eval.scripts.generate_report \
    --logs-dir logs --output-dir report

# Include all participants (default behavior)
LLM_DELUSIONS_EXCLUDED_PARTICIPANTS="" \
uv run python -m llm_delusion_eval.scripts.generate_report \
    --logs-dir logs --output-dir report
```

You can also pass `--excluded-participants` directly to
`compute_context_effects` and `compute_context_code_controls`.
Use `--excluded-participants ""` to disable exclusions explicitly.

For dataset-level methods stats, you can override the sanitized parquet path:

```bash
LLM_DELUSIONS_ITEMS_SANITIZED_PATH=hf://datasets/jlcmoore/delusioneval/items_sanitized.parquet \
uv run python -m analysis.compute_methods_stats
```

## Participant Robustness

The appendix-facing participant robustness tables are generated with:

```bash
uv run python -m analysis.compute_participant_robustness
```

## Specialized Analyses

These are independent of the core pipeline; each can be run on its own once
`report/eval_rows.parquet` (or `summary.json`, where applicable) is in place.

```bash
# Reasoning-effect deltas with bootstrap CIs and turn-bin split
uv run python -m analysis.compute_reasoning_effects

# Per-model code residuals against each model's category mean
uv run python -m analysis.compute_code_residuals

# Sample counts and descriptive stats per model
uv run python -m analysis.compute_methods_stats

# Inter-code correlations (phi, co-occurrence)
uv run python -m analysis.compute_code_correlations

# Within-window sequential effects (onset, persistence)
uv run python -m analysis.compute_sequential_dynamics

# Per-participant code prevalence and within-model variance
uv run python -m analysis.compute_participant_profiles

# Per-participant summary table plus clustered-bootstrap / leave-one-out
# robustness for the main category-level model comparisons
uv run python -m analysis.compute_participant_robustness

# Context controls: score>0 ~ requested_depth + code_prevalence
uv run python -m analysis.compute_context_code_controls \
    --logs-dir logs-context
```

Scripts below remain internal workflows because they require local source
data from outside this repository:

- `analysis.compute_methods_stats` (uses selected `items.parquet` and transcript paths for full outputs)
- `analysis.compute_participant_profiles` (requires participant mapping from local source parquets)
- `analysis.generate_window_cutup_real_data` (uses local source items/transcripts and annotations matches)
- `analysis.export_overleaf_assets` (requires sibling overleaf repository)

## Context Effects (No Report Export)

For context-depth experiments, analyze logs directly without generating
`report/summary.json`:

```bash
# Requested vs effective context length vs prevalence
uv run python -m analysis.compute_context_effects \
    --logs-dir logs-context

# Optional: include validates-* codes in default outputs
uv run python -m analysis.compute_context_effects \
    --logs-dir logs-context \
    --include-validates-codes

# Optional: filter out short-context rows before aggregation
uv run python -m analysis.compute_context_effects \
    --logs-dir logs-context \
    --min-effective-context-messages 300

# Optional: require effective context to reach a fraction of requested context
uv run python -m analysis.compute_context_effects \
    --logs-dir logs-context \
    --min-effective-fraction-of-requested 1.0

# Optional: restrict to one or more codes
uv run python -m analysis.compute_context_effects \
    --logs-dir logs-context \
    --codes bot-discourages-self-harm

# Optional: generate an additional "uniform sample set" analysis up to
# a requested-context upper limit. This keeps only shared window IDs
# across requested-context points > 0 and <= 350 (per model/reasoning/code).
uv run python -m analysis.compute_context_effects \
    --logs-dir logs-context \
    --uniform-sample-upper-limit 350
```

The script also pulls a `requested_context_messages=0` baseline from `logs/`
for matching `(model, reasoning, code)` points when available. Override with:

```bash
uv run python -m analysis.compute_context_effects \
    --logs-dir logs-context \
    --baseline-logs-dir logs
```

By default, the analysis keeps only rows with full realized context
(`effective_context_length >= requested_context_messages`).
When `--codes` is omitted, `validates-*` codes are excluded by default.
Use `--include-validates-codes` to keep them.

Outputs:
- `analysis/data/context_effects/context_effect_points.csv`
- `analysis/data/context_effects/context_effect_points_by_category.csv`
- `analysis/figures/context_effect_scatter_<code>.pdf/.png`
- `analysis/figures/context_effect_scatter_category_<category>.pdf/.png`
- `analysis/figures/context_effect_subplots_gpt54_upto_400_codes.pdf/.png`
- `analysis/figures/context_effect_subplots_gpt54_upto_400_categories.pdf/.png`

Additional control analysis outputs:
- `analysis/data/context_effects/context_code_control_lpm_by_code.csv`
- `analysis/data/context_effects/context_code_control_lpm_by_category.csv`
- `analysis/figures/context_code_control_forest_by_category.pdf/.png`

Control-model details:
- Outcome per sample is `1[score > 0]`.
- Regressor 1 (`requested_depth`) is the requested prior-context message count.
- Regressor 2 (`code_prevalence`) is the share of preceding assistant turns in
  that sample's context that pass the same code's annotation cutoff.
- Models are fit within cohorts (`model`, `reasoning_effort`, and category/code)
  using a linear probability model:
  `binary_score ~ requested_depth + code_prevalence`.
- Forest-plot scaling: left panel is percentage points per +100 requested
  messages; right panel is percentage points per +10pp in prior-code prevalence.

Each plotted point is labeled as `w=<count>`, where `w` is the number of
unique windows included in that aggregate point. The CSVs also include:
- `n` = number of samples
- `w` = number of unique windows

When `--uniform-sample-upper-limit N` is set, an additional set of outputs is
generated (while keeping the original outputs above):
- `analysis/data/context_effects/context_effect_points_uniform_upto_<N>.csv`
- `analysis/data/context_effects/context_effect_points_by_category_uniform_upto_<N>.csv`
- `analysis/figures/context_effect_scatter_uniform_upto_<N>_<code>.pdf/.png`
- `analysis/figures/context_effect_scatter_category_uniform_upto_<N>_<category>.pdf/.png`
- `analysis/figures/context_effect_subplots_uniform_upto_<N>_gpt54_codes.pdf/.png`
- `analysis/figures/context_effect_subplots_uniform_upto_<N>_gpt54_categories.pdf/.png`

You can override output paths with:
- `--output-dir` for figures
- `--data-dir` for CSVs

The script also caches parsed rows to:
- `.cache/context_effect_rows.parquet`

For repeated runs, it also caches per-log metadata and parsed rows so unchanged
`.eval` files are not re-parsed:
- `.cache/context_effect_log_metadata.json`
- `.cache/context_effect_samples/`

## Trace Packs and Reasoning-Trace Features

Use trace-pack export when you want ranked, human-readable examples that include
the original conversation, model reasoning trace, model response, and grader
evidence in one file per code.

```bash
# Top-N by raw score (descending)
uv run python -m analysis.export_top_trace_examples \
    --code bot-facilitates-self-harm \
    --code bot-grand-significance \
    --code bot-endorses-delusion \
    --n 10 \
    --rank-by raw_score_desc \
    --log-path logs/2026-04-26T16-03-11+00-00_delusions-eval_kdPDHMWaW9wJDGXBcAsQKa.eval
```

Outputs:
- `analysis/data/trace_pack_<code>_top<N>_raw_<reasoning>.md`
- `analysis/data/trace_pack_<code>_top<N>_absref_<reasoning>.md`

CLI notes:
- `--code` can be repeated and accepts values with or without the `bot-` prefix.
- `--model`, `--reasoning-effort`, `--rows-path`, and `--output-dir` are
  optional filters/overrides.
- Ranking ties are broken by `sample_id` ascending.

## Data Flow

```
.eval log files (logs/)
        |
        v
generate_report.py  -->  report/summary.json
                    +  report/eval_rows.parquet
        |                    (model_label, reasoning_effort,
        |                     per-code/category/harm CIs,
        |                     delta_from_original with CIs)
        v
generate_figures.py            --> analysis/figures/ + analysis/data/ + analysis/tables/
                                   (fig1-fig7 as PDF, CSVs, LaTeX tables)
compute_context_effects.py     --> analysis/figures/ + analysis/data/context_effects/
                                   (context-effect PDFs/PNGs + CSVs)
compute_context_code_controls.py --> analysis/figures/ + analysis/data/context_effects/
                                     (LPM coefficient CSVs + forest plot)
compute_reasoning_effects.py   --> analysis/data/reasoning_effects/
compute_code_residuals.py      --> analysis/data/code_residuals_*.csv
compute_methods_stats.py       --> analysis/data/methods_stats.txt
compute_code_correlations.py   --> analysis/data/code_phi_correlation_*.csv,
                                   code_cooccurrence_*.csv,
                                   model_prevalence_correlation.csv
compute_sequential_dynamics.py --> analysis/data/onset_statistics.csv,
                                   persistence_rates.csv
compute_participant_profiles.py --> analysis/data/participant_*.csv
compute_participant_robustness.py --> analysis/data/participant_summary.csv,
                                      participant_clustered_*.csv,
                                      participant_leave_one_out_*.csv
export_top_trace_examples.py   --> analysis/data/trace_pack_*.md
export_overleaf_assets.py      --> ../llm-delusions-eval-overleaf/
                                   (manifest-selected figures/tables)
```

`generate_figures.py` reads **`summary.json`** for all figures except
`fig3_turn_position`, which requires per-turn row-level data and uses
`load_all_eval_data()` (which requires `report/eval_rows.parquet`).

Alongside `prevalence_by_model_category.csv`, `generate_figures.py` also
writes `prevalence_deviation_from_original.csv`, the same table with
extra `<category>_dev` columns giving signed deviation in percentage points
from the `Original transcript` row.

## Figures

| Figure | Description | Data source |
|--------|-------------|-------------|
| fig1 | Category prevalence heatmap with separate `discourages harm` panel (inverse color direction for that panel) | summary.json |
| fig2 | Per-code prevalence heatmap | summary.json |
| fig3 | Prevalence by turn position | report/eval_rows.parquet |
| fig4 | Scaling effects (GPT-5.4 family) | summary.json |
| fig5 | Temporal effects + GPT-5.4 reasoning variants (combined) | summary.json |
| fig6 | Reasoning effects (GPT-5.4 variants) | summary.json |
| fig7 | Delta from original (forest plot) | summary.json |
| figA_scaling_model_sizes | Model-size appendix scaling comparison | summary.json |
| figA_reasoning_gpt54 | Reasoning effects (GPT-5.4 variants), appendix companion | summary.json |
| figA_reasoning_qwen397b | Reasoning effects (Qwen3.5-397B variants), appendix | summary.json |
| figA_main_heatmap_harm_codes | Compatibility alias of `fig2_code_heatmap` (same content, different filename) | summary.json |

Additional appendix figures produced by `generate_figures.py`:
- `figA_scaling_model_sizes.pdf` (Qwen3.5-9B vs Qwen3.5-397B, Gemini 3.1 Flash-Lite vs Gemini 3.1 Pro, GPT-5.4 Nano/Mini/5.4)
- `figA_reasoning_gpt54.pdf` (GPT-5.4 reasoning variants)
- `figA_reasoning_qwen397b.pdf` (Qwen3.5-397B reasoning variants)

Bar charts (fig4, fig5, fig6) show five aggregate categories on the
x-axis with 95% CIs: sycophancy, delusional, relationship,
facilitates harm, and discourages harm. Harm is split so facilitates and
discourages patterns do not collapse into one category.

## Key Modules

### Utilities (imported by other modules, not run directly)

- **`bootstrap.py`** -- Bootstrap CI utilities: `bootstrap_binary_ci`,
  `bootstrap_delta_ci`, `bootstrap_grouped_ci`, `bootstrap_model_code_ci`.
- **`load_eval_data.py`** -- Loads row-level eval data from
  `report/eval_rows.parquet`.
  Used directly for fig3 and other row-level analysis scripts.
- **`artifact_paths.py`** -- Canonical paths for figures, data, and tables.
- **`participant_mapping.py`** -- Window-to-participant mapping helpers
  shared by participant-profile analyses.
- **`plot_style.py`** -- Shared matplotlib styling, color palettes, and
  per-model color helpers.

### Producer scripts
- **`generate_figures.py`** -- Main entry point for figure generation.
  CLI: `--summary`, `--figures-only`, `--include-validates-codes`, `--verbose`.
- **`compute_methods_stats.py`** -- Sample counts and descriptive stats.
- **`compute_code_correlations.py`** -- Inter-code correlation analysis.
- **`compute_sequential_dynamics.py`** -- Within-window sequential effects.
- **`compute_context_effects.py`** -- Requested vs effective context analysis
  from `logs-context/*.eval` without report generation. Excludes
  `validates-*` codes by default unless `--include-validates-codes` or
  explicit `--codes ...` filters are provided. Category-level outputs use
  the five aggregate categories (including facilitates harm and discourages harm).
- **`compute_context_code_controls.py`** -- Fits linear probability controls
  (`score > 0 ~ requested_depth + code_prevalence`) within code/category
  cohorts and exports coefficient tables plus
  `context_code_control_forest_by_category.pdf/.png`. Excludes
  `validates-*` codes by default unless explicitly requested.
- **`compute_participant_profiles.py`** -- Per-participant code prevalence,
  window counts, heatmap data, and within-model variance. Outputs to
  `analysis/data/participant_*.csv`.
- **`compute_participant_robustness.py`** -- Per-participant contribution
  summary from selected windows, plus participant-clustered bootstrap and
  leave-one-participant-out reruns for the main category-level model
  comparisons.
- **`compute_reasoning_effects.py`** -- Per-code and per-category prevalence
  deltas (high-reasoning minus baseline) with 95% bootstrap CIs and a
  `significant_95ci` flag, plus a turn-bin (early/mid/late thirds within each
  window) split. Configured for GPT-5.4 (default vs. high) and Qwen3.5-397B
  (low vs. high). Outputs to `analysis/data/reasoning_effects/`.
- **`export_top_trace_examples.py`** -- Exports top-ranked sample trace packs
  per code with full conversation input, reasoning trace, model output, and
  grader evidence. Supports ranking by `raw_score` or absolute distance from a
  reference score. Outputs to `analysis/data/trace_pack_*.md`.
- **`compute_code_residuals.py`** -- Per-model code-level residuals against
  each model's category mean. Reads
  `analysis/data/prevalence_by_model_code.csv` and writes:
  - `analysis/data/code_residuals_by_model.csv` (long-form: code, model,
    prevalence, category_mean, residual)
  - `analysis/data/code_residuals_top_by_model.csv` (top-3 positive and
    top-3 negative residuals per model)

  Supports the item-level analyses in the paper's Results section by
  isolating code-level heterogeneity from overall prevalence differences.
