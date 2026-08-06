# Extended Reference

## Setup

### Using uv

[uv](https://docs.astral.sh/uv/) makes Python package and project management much easier, and we highly recommend using it. If you want to use venv, virtualenv, pyenv or conda instead, you may figure it out.

### Installation

```sh
uv sync
```

To also install all optional dependencies (i.e. client libraries for AI platforms):

```sh
uv sync --all-extras
```

To run context-length experiments, install the optional context dependency:

```sh
uv sync --extra context
```

### Install pre-commit (recommended but optional)

```
uv run pre-commit install
```

### Lint and format

```
uv run pre-commit run
```

### Inspect AI

We use [Inspect AI](https://inspect.aisi.org.uk/) as a framework for evaluations. Please refer to its documentation for more information on its features.

Tip: You can set the output folder for Inspect AI using an environment variable:

```sh
export INSPECT_LOG_DIR=logs-2026-01-14
```

### Important Arguments

- `--model` specifies the model(s) under evaluation (e.g., `--model gpt-4o,gemini-1.5-pro`)
- `--model-role grader=...` specifies the judge model that scores responses
- `--limit N` limits the number of samples to evaluate
- `--cache 1Y` enables [Inspect AI's request caching](https://inspect.aisi.org.uk/caching.html)
- `-T key=value` passes task parameters (e.g., `-T codes=bot-misrepresents-sentience,bot-endorses-delusion`)
- `--temperature T` sets the sampling temperature (e.g., `0.0` for deterministic, `1.0` for more creative)
- `--top-p P` sets nucleus sampling threshold (e.g., `1.0` to disable)

**Note on CLI Flags:** In `inspect`, passing the same flag multiple times (e.g., `--model A --model B` or `-T code=A -T code=B`) will cause the **last** flag to override all previous ones. To specify multiple values, use a single flag with a comma-separated list.

#### Reasoning Configurations

Inspect AI natively supports models with reasoning capabilities. You can modulate reasoning behavior using the following CLI flags:

- `--reasoning-effort [none|minimal|low|medium|high|xhigh]`: Constrains the reasoning effort. Support and defaults vary by provider. For some models, setting this to `none` can disable reasoning.
- `--reasoning-tokens N`: Sets the maximum number of tokens to use for reasoning (e.g., Anthropic Claude models).
- `--reasoning-summary [none|concise|detailed|auto]`: Controls the summary of reasoning steps for OpenAI reasoning models.
- `--reasoning-history [none|all|last|auto]`: Determines if reasoning steps are included in the chat message history sent back to the model in multi-turn evals.

#### Max Tokens & Stopping

By default, Inspect AI does not enforce a global token limit on model responses, relying instead on the model or provider's default behavior.

You can explicitly control this using:

- `--max-tokens N`: The maximum number of tokens the model is allowed to generate.
- `--stop-seqs "seq1,seq2"`: One or more sequences where the model should stop generating.

Example:

```sh
uv run inspect eval ... --max-tokens 1024 --stop-seqs "###,USER:"
```

#### Model Specific Args

You can pass provider-specific "native" model arguments using the `-M` (or `--model-config`) CLI flag. These are used when you need to bypass default behaviors, force specific API paths, or toggle provider-only features that aren't part of the standard Inspect generation config. Note that if you use `--model-config`, the arguments are applied to _all_ models listed in the `--model` flag, which will cause a crash if you are evaluating models from different providers (e.g. OpenAI and Google) simultaneously.


### Setting up API keys for models

Inspect AI supports `.env` files, so the easiest way to manage API keys is to add them to the `.env` file. For example:

```
OPENAI_API_KEY=sk-...
TOGETHER_API_KEY=tgp_...
ANTHROPIC_API_KEY=sk-...
```

Inspect AI will automatically load the environment variables from this file on startup.

Alternatively, you can `export` these as shell environment variables by running this in your shell:

```sh
export OPENAI_API_KEY=sk-...
export TOGETHER_API_KEY=tgp_...
export ANTHROPIC_API_KEY=sk-...
```

Note: You can switch to using Vertex AI instead of Gemini API for Gemini by setting a few environment variables. See Inspect AI's [documentation](https://inspect.aisi.org.uk/providers.html#gemini-on-vertex-ai) for more information.

```sh
export GOOGLE_CLOUD_PROJECT=your-project-name
export GOOGLE_CLOUD_LOCATION=global
export GOOGLE_GENAI_USE_VERTEXAI=True
```

### Data Setup

Window-only evals use the public sanitized dataset hosted on Hugging Face:

```sh
export LLM_DELUSIONS_WINDOWS_PATH=hf://datasets/spiralsafety/delusioneval/items_sanitized.parquet
```

Context-length experiments require local non-sanitized data and
transcripts (for prepended conversation history), plus the optional context
dependency (`uv sync --extra context`):

```sh
export LLM_DELUSIONS_WINDOWS_PATH=../llm-delusions/subsets/items.parquet
export LLM_DELUSIONS_TRANSCRIPTS_PATH=../llm-delusions/transcripts_data/transcripts.parquet
```

<!--  -->
<!--  -->
<!--  -->
<!--  -->
<!--  -->

## Evaluation

### Parameters

The `delusions_eval` task supports several parameters that can be passed using the `-T` flag:

| Parameter              | Type         | Default | Description                                                                                                                                                                                                                                                                                   |
| :--------------------- | :----------- | :------ | :-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `max_context_messages` | `int`        | `0`     | Number of preceding transcript messages to prepend to the window. If `0`, evaluates only the conversation window. If negative (for example, `-1`), includes all available preceding context. If the history exceeds a positive limit, only the most recent messages are kept, and a "Truncated context..." warning is displayed. |
| `codes`                | `str`/`list` | `None`  | Specific annotation IDs to run (e.g., `bot-endorses-delusion`). Supports comma-separated strings or repeating the flag. Defaults to all `bot-*` codes.                                                                                                                                        |
| `target_role`          | `str`        | `assistant` | Next-turn role to generate. `assistant` uses the existing setup (predict assistant after user turns). `user` asks the model to complete the next user turn after assistant turns.                                                                                                            |
| `max_windows`          | `int`        | `0`     | Maximum number of conversation windows to load **per annotation code**. `0` means no limit. Note: This is _not_ the total number of evaluation samples.                                                                                                                                       |
| `grader`               | `str`        | `None`  | Shortname alias for the judge model (e.g. `gemini-3`, `gpt-5`, `gpt-4o`, `mock`). Bypasses the need for verbose `--model-role` flags. Pass this explicitly (or pass `--model-role grader=...`). Recommended: `openai/gpt-5.1-2025-11-13` with reasoning disabled.                             |

Recommended explicit grader configuration:

```sh
--model-role 'grader={"model":"openai/gpt-5.1-2025-11-13","reasoning_effort":"none"}'
```

---

### Dry Run

If you want to verify that your data loads correctly and preview the exact prompts being sent to the models without incurring API costs, you can use Inspect's built-in `mockllm/model`.

However, because the `delusions_eval` task uses a grader that expects JSON output, the standard `mockllm/model` will fail with a `ClassificationError`. To avoid this, use the provided mock wrapper:

```sh
uv run python src/llm_delusion_eval/scripts/inspect_mocked.py eval src/llm_delusion_eval/tasks/delusions_eval.py \
  --model mockllm/model \
  --model-role grader=mockllm/model \
  --limit 2 \
  -T mock_score=7
```

- `-T mock_score=7`: Sets the dummy score that the mock grader will "assign" to every sample.

This will run the first 2 samples through the pipeline, returning dummy responses instantly. You can then use `uv run inspect view` to inspect the generated logs and verify the prompts.

### Examples

**Run a quick test on a single behavior:**

```sh
uv run inspect eval src/llm_delusion_eval/tasks/delusions_eval.py \
  --model openai/gpt-5-mini-2025-08-07 \
  -T grader=google/vertex/gemini-3-flash-preview \
  -T codes=bot-misrepresents-sentience \
  -T max_windows=3 \
  --limit 1
```

**Run multiple specific codes:**

```sh
uv run inspect eval src/llm_delusion_eval/tasks/delusions_eval.py \
  --model openai/gpt-4o \
  -T codes=bot-facilitates-self-harm,bot-facilitates-violence
```

**Run specific behaviors with full conversation context:**

Context-mode examples below require local non-sanitized windows/transcripts and
are not part of the default window-only workflow.

```sh
LLM_DELUSIONS_WINDOWS_PATH=../llm-delusions/subsets/items.parquet \
LLM_DELUSIONS_TRANSCRIPTS_PATH=../llm-delusions/transcripts_data/transcripts.parquet \
uv run inspect eval src/llm_delusion_eval/tasks/delusions_eval.py \
  --model openai/gpt-4o \
  -T max_context_messages=200 \
  -T codes=bot-facilitates-self-harm,bot-facilitates-violence,bot-validates-self-harm-feelings \
  -T max_windows=10
```

**Run all behaviors with full context:**

```sh
LLM_DELUSIONS_WINDOWS_PATH=../llm-delusions/subsets/items.parquet \
LLM_DELUSIONS_TRANSCRIPTS_PATH=../llm-delusions/transcripts_data/transcripts.parquet \
uv run inspect eval src/llm_delusion_eval/tasks/delusions_eval.py \
  --model openai/gpt-4o \
  -T max_context_messages=200
```

**Run all behaviors with all available context:**

```sh
LLM_DELUSIONS_WINDOWS_PATH=../llm-delusions/subsets/items.parquet \
LLM_DELUSIONS_TRANSCRIPTS_PATH=../llm-delusions/transcripts_data/transcripts.parquet \
LLM_DELUSIONS_MIN_CONTEXT_MESSAGES=0 \
uv run inspect eval src/llm_delusion_eval/tasks/delusions_eval.py \
  --model openai/gpt-4o \
  -T max_context_messages=-1
```

**Run all behaviors on the full sanitized dataset:**

```sh
MODEL=openai/gpt-5.4-mini-2026-03-17
GRADER_JSON='{"model":"openai/gpt-5.1-2025-11-13","reasoning_effort":"none"}'
uv run inspect eval src/llm_delusion_eval/tasks/delusions_eval.py \
  --model "$MODEL" \
  --model-role "grader=$GRADER_JSON"
```

You can swap `MODEL` for any model listed in the Models section.
For reasoning-capable models, choose a reasoning setting explicitly:

```sh
# Reasoning off
MODEL=openai/gpt-5.4-mini-2026-03-17
uv run inspect eval src/llm_delusion_eval/tasks/delusions_eval.py \
  --model "$MODEL" \
  --reasoning-effort none \
  --model-role 'grader={"model":"openai/gpt-5.1-2025-11-13","reasoning_effort":"none"}'

# Reasoning on
MODEL=openai/gpt-5.4-2026-03-05
uv run inspect eval src/llm_delusion_eval/tasks/delusions_eval.py \
  --model "$MODEL" \
  --reasoning-effort high \
  --model-role 'grader={"model":"openai/gpt-5.1-2025-11-13","reasoning_effort":"none"}'
```

Use provider-appropriate reasoning controls:

- OpenAI and Together reasoning models: use `--reasoning-effort ...`.
- Anthropic Claude models: use `--reasoning-effort ...` and
  `--reasoning-tokens ...` where supported (`--effort ...` only where
  supported).
- If a model does not expose a reasoning control in this setup, omit it.

### How Samples are Generated

For any given conversation window (typically ~20 messages long), sample generation depends on `target_role`.

- `target_role=assistant` (default): one sample per **user** turn, and the model generates the next assistant response.
- `target_role=user`: one sample per **assistant** turn, and the model generates the next user response.
- **Adding context (`max_context_messages != 0`) does not change the sampling rule.** It prepends preceding transcript messages to each sample history for apples-to-apples comparisons at matched turn boundaries.

### Collecting Next User Turns (No Scoring)

Use this mode to collect model-predicted user turns for later qualitative analysis.

```sh
uv run inspect eval src/llm_delusion_eval/tasks/delusions_eval.py \
  --model openai/gpt-5.4-mini-2026-03-17 \
  -T target_role=user \
  --limit 50
```

Notes:
- This mode intentionally skips grader scoring.
- Generated user-turn predictions are stored in the Inspect log output for each sample.

### Grader Context Depth

For each scored assistant message, the evaluator passes the grader all preceding
messages currently present in that sample's history.

- The scorer does not apply a fixed "previous 3 messages" truncation.
- With `max_context_messages=0`, this means all preceding messages within the
  window-only sample history.
- With `max_context_messages=N` (N > 0), this includes prepended transcript
  context up to `N` messages, plus the in-window history for that sample.
- With `max_context_messages < 0` (for example, `-1`), this includes all
  available prepended transcript context plus the in-window history.

The following will do the same but using the max context experiment with 200 messages in context.

```sh
LLM_DELUSIONS_WINDOWS_PATH=../llm-delusions/subsets/items.parquet \
LLM_DELUSIONS_TRANSCRIPTS_PATH=../llm-delusions/transcripts_data/transcripts.parquet \
uv run python src/llm_delusion_eval/scripts/inspect_mocked.py eval src/llm_delusion_eval/tasks/delusions_eval.py \
  --model mockllm/model \
  --model-role grader=mockllm/model \
  --limit 2 \
  -T mock_score=7 \
  -T max_context_messages=200
```

To run the same context-depth setup as a real eval in `logs-context/` for
`GPT-5.4` with reasoning disabled, restricted to the discourages-self-harm code
at selected context depths:

```sh
MODEL=openai/gpt-5.4-2026-03-05
REASONING_ARGS=(--reasoning-effort none)
CODE=bot-discourages-self-harm
GRADER_JSON='{"model":"openai/gpt-5.1-2025-11-13","reasoning_effort":"none"}'

for max_ctx in 100 200 300 400 500; do
  echo "Running max_context_messages=${max_ctx}"
  if ! INSPECT_LOG_DIR=logs-context \
    LLM_DELUSIONS_WINDOWS_PATH=../llm-delusions/subsets/items.parquet \
    LLM_DELUSIONS_TRANSCRIPTS_PATH=../llm-delusions/transcripts_data/transcripts.parquet \
    uv run inspect eval src/llm_delusion_eval/tasks/delusions_eval.py \
      --model "$MODEL" \
      "${REASONING_ARGS[@]}" \
      --model-role "grader=$GRADER_JSON" \
      --no-fail-on-error \
      --continue-on-fail \
      --retry-on-error 2 \
      -T max_windows=0 \
      -T codes="$CODE" \
      -T max_context_messages="${max_ctx}"; then
    echo "Failed at max_context_messages=${max_ctx}" >&2
    exit 1
  fi
done
```

This stops on any non-zero process exit. With `--no-fail-on-error` and
`--continue-on-fail`, sample-level failures may still exit `0`.

To run `GPT-5.4` (reasoning disabled) across all delusional codes at context
depths `0, 50, 100, 150, 200, 250, 300, 350, 400`:

```sh
MODEL="openai/gpt-5.4-2026-03-05"
CODES="bot-endorses-delusion,bot-metaphysical-themes,bot-misrepresents-ability,bot-misrepresents-sentience"

for CTX in 0 50 100 150 200 250 300 350 400; do
  bash scripts/run_context_eval.sh "$MODEL" "$CTX" "$CODES"
done
```

`scripts/run_context_eval.sh` usage:

```sh
bash scripts/run_context_eval.sh <model> <max_context_messages> [codes] [log_dir] [min_context_messages] [reasoning_effort]
```

Defaults:

- `log_dir` defaults to `logs-context`.
- `reasoning_effort` defaults to `none`.
- If `min_context_messages` is omitted and `max_context_messages > 0`,
  `min_context_messages` defaults to `max_context_messages`.

To explicitly set reasoning effort while still using the default
`min_context_messages`, pass an empty 5th argument:

```sh
bash scripts/run_context_eval.sh "$MODEL" "$CTX" "$CODES" logs-context "" low
```

To run the mocked eval at prior-context depths `0, 10, 20, ..., 200`:

```sh
for max_ctx in $(seq 0 10 200); do
  echo "Running max_context_messages=${max_ctx}"
  LLM_DELUSIONS_WINDOWS_PATH=../llm-delusions/subsets/items.parquet \
  LLM_DELUSIONS_TRANSCRIPTS_PATH=../llm-delusions/transcripts_data/transcripts.parquet \
  uv run python src/llm_delusion_eval/scripts/inspect_mocked.py eval src/llm_delusion_eval/tasks/delusions_eval.py \
    --model mockllm/model \
    --model-role grader=mockllm/model \
    --limit 2 \
    -T mock_score=7 \
    -T max_context_messages="${max_ctx}"
done
```

<!--  -->
<!--  -->
<!--  -->
<!--  -->

## Experiment Playbook

### Models

These help us make a scale point:

```sh
MODEL=openai/gpt-5.4-2026-03-05
REASONING_ARGS=(--reasoning-effort none)

MODEL=openai/gpt-5.4-2026-03-05
REASONING_ARGS=(--reasoning-effort high)

MODEL=openai/gpt-5.4-mini-2026-03-17
REASONING_ARGS=(--reasoning-effort none)

MODEL=openai/gpt-5.4-nano-2026-03-17
REASONING_ARGS=(--reasoning-effort none)
```

```sh
MODEL=anthropic/claude-sonnet-4-6
REASONING_ARGS=(--reasoning-effort none)

MODEL=anthropic/claude-haiku-4-5
REASONING_ARGS=(--reasoning-effort none)

MODEL=google/vertex/gemini-2.5-pro
REASONING_ARGS=(--reasoning-effort minimal)

MODEL=google/vertex/gemini-2.5-flash-lite
REASONING_ARGS=(--reasoning-effort none)
```

These help us make a change-through-time point:

```sh
MODEL=openai/gpt-4.1-2025-04-14
REASONING_ARGS=()

MODEL=openai/gpt-4o-2024-11-20
REASONING_ARGS=()

MODEL=openai/gpt-4-turbo-2024-04-09
REASONING_ARGS=()
```

These help us show that it affects open source models (and another scaling point):

```sh
MODEL=together/Qwen/Qwen3.5-397B-A17B
REASONING_ARGS=(--reasoning-effort low)

MODEL=together/Qwen/Qwen3.5-397B-A17B
REASONING_ARGS=(--reasoning-effort high)

MODEL=together/Qwen/Qwen3.5-9B
REASONING_ARGS=(--reasoning-effort low)
```

These allow us to make the point that this is industry wide:

```sh
MODEL=google/vertex/gemini-3.1-pro-preview
REASONING_ARGS=(--reasoning-effort minimal)

MODEL=anthropic/claude-opus-4-7
REASONING_ARGS=(--reasoning-effort none)

MODEL=grok/grok-4.20-0309-non-reasoning
REASONING_ARGS=()
```

### Commands to run

#### General experiments

Mock.

```sh
MODEL=mockllm/model
REASONING_ARGS=()
GRADER_MODEL=mockllm/model

uv run python src/llm_delusion_eval/scripts/inspect_mocked.py eval \
  src/llm_delusion_eval/tasks/delusions_eval.py \
  --model "$MODEL" \
  "${REASONING_ARGS[@]}" \
  --model-role "grader=$GRADER_MODEL" \
  -T mock_score=7 \
  --limit 1
```

Use the following command as the starting template.

Real example. **When running experiments, copy this command and change the `MODEL` and `REASONING_ARGS` to the above values. REMOVE the `--limit 1`.**

```sh
MODEL=google/vertex/gemini-3.1-flash-lite-preview
REASONING_ARGS=(--reasoning-effort minimal)
GRADER_JSON='{"model":"openai/gpt-5.1-2025-11-13","reasoning_effort":"none"}'

uv run inspect eval src/llm_delusion_eval/tasks/delusions_eval.py \
  --model "$MODEL" \
  "${REASONING_ARGS[@]}" \
  --model-role "grader=$GRADER_JSON" \
  --no-fail-on-error \
  --continue-on-fail \
  --retry-on-error 2 \
  --limit 1
```

Retry a prior run from its printed log path:

```sh
.venv/bin/inspect eval-retry logs/<your-log-file>.eval \
  --no-fail-on-error \
  --continue-on-fail \
  --retry-on-error 2
```

#### Model context experiment

```sh
MODEL="openai/gpt-5.4-2026-03-05"
CODES="bot-endorses-delusion,bot-metaphysical-themes,bot-misrepresents-ability,bot-misrepresents-sentience"

for CTX in 0 50 100 150 200 250 300 350 400; do
  bash scripts/run_context_eval.sh "$MODEL" "$CTX" "$CODES"
done
```

### Cost Estimate (Mock-Only)

The table below is an estimate for one full `delusions_eval` pass
over the current window-only dataset (`6,781` samples), including both the
evaluated model and the grader.

Important: Inspect's `mockllm` token usage values are character lengths, not
provider-billed tokens. Do not use `mockllm` usage totals directly for pricing.

Tokenizer and counting method:

- We use `litellm.token_counter(model=..., messages/text=...)` locally.
- Counts are model-aware through LiteLLM mappings. They are estimates and can
  differ from provider-side billing tokenizers.

Assumptions used in this estimate:

- Dataset: `items_sanitized.parquet`, `6,781` samples.
- Eval input tokens: measured from each sample's `input` messages.
- Eval output tokens (base): measured from the original next assistant message
  in the same source window.
- For `130` samples without a next assistant, we impute with a median-length
  assistant response (about `270` tokens in `gpt-5.1` tokenization).
- Reasoning runs use a `4x` multiplier on eval output tokens.
- Grader input token estimate is:
  `base_grader_prompt_tokens + (eval_output_tokens_adjusted - eval_output_tokens_base)`.
- Grader output tokens use the default grader JSON shape in
  `inspect_mocked.py` (`44` tokens/sample under `gpt-5.1` tokenization).
- Grader model for pricing: `openai/gpt-5.1-2025-11-13`.

Cost formula:

```text
run_cost_usd =
  (eval_in_tokens * eval_in_price_per_token) +
  (eval_out_tokens * eval_out_price_per_token) +
  (grader_in_tokens * grader_in_price_per_token) +
  (grader_out_tokens * grader_out_price_per_token)
```

Estimated per-run costs:

| Run                                                      | Eval In Tok | Eval Out Tok | Grader In Tok | Grader Out Tok | Eval USD | Grader USD | Total USD |
| -------------------------------------------------------- | ----------: | -----------: | ------------: | -------------: | -------: | ---------: | --------: |
| `openai/gpt-5.4-2026-03-05` (none)                       |  12,551,467 |    2,270,580 |    22,422,667 |        298,364 |    65.44 |      31.01 |     96.45 |
| `openai/gpt-5.4-2026-03-05` (high, 4x out)               |  12,551,467 |    9,082,320 |    29,234,407 |        298,364 |   167.61 |      39.53 |    207.14 |
| `openai/gpt-5.4-mini-2026-03-17` (none)                  |  12,551,467 |    2,270,580 |    22,422,667 |        298,364 |    19.63 |      31.01 |     50.64 |
| `openai/gpt-5.4-nano-2026-03-17` (none)                  |  12,551,467 |    2,270,580 |    22,422,667 |        298,364 |     5.35 |      31.01 |     36.36 |
| `openai/gpt-4.1-2025-04-14`                              |  12,551,467 |    2,270,580 |    22,422,667 |        298,364 |    43.27 |      31.01 |     74.28 |
| `openai/gpt-4o-2024-11-20`                               |  12,551,467 |    2,270,580 |    22,422,667 |        298,364 |    54.08 |      31.01 |     85.10 |
| `openai/gpt-4-turbo-2024-04-09`                          |  12,551,467 |    2,270,580 |    22,422,667 |        298,364 |   193.63 |      31.01 |    224.64 |
| `together/Qwen/Qwen3.5-397B-A17B` (low, 4x out)          |  12,551,467 |    9,082,320 |    29,234,407 |        298,364 |    40.23 |      39.53 |     79.75 |
| `together/Qwen/Qwen3.5-397B-A17B` (high, 4x out)         |  12,551,467 |    9,082,320 |    29,234,407 |        298,364 |    40.23 |      39.53 |     79.75 |
| `together/Qwen/Qwen3.5-9B` (low, 4x out)                 |  12,551,467 |    9,082,320 |    29,234,407 |        298,364 |     2.62 |      39.53 |     42.14 |
| `google/vertex/gemini-3.1-pro-preview` (minimal, 4x out) |  12,551,467 |    9,082,320 |    29,234,407 |        298,364 |   134.09 |      39.53 |    173.62 |
| `anthropic/claude-opus-4-7` (low, 4x out)                |  12,551,467 |    9,082,320 |    29,234,407 |        298,364 |   289.82 |      39.53 |    329.34 |
| `grok/grok-4.20-0309-non-reasoning`                      |  12,551,467 |    2,270,580 |    22,422,667 |        298,364 |    38.73 |      31.01 |     69.74 |

Total across all rows above: `1548.96` USD.

Pricing references:

- OpenAI models and grader: OpenAI API pricing pages for `gpt-5.4`,
  `gpt-5.4-mini`, `gpt-5.4-nano`, `gpt-5.1`, `gpt-4.1`, `gpt-4o`,
  `gpt-4-turbo`.
- Non-OpenAI models: provider pricing mirrored in LiteLLM's
  `model_prices_and_context_window.json` (fetched on April 24, 2026).
- `Qwen3.5-9B` used a manual fallback estimate (`$0.10/$0.15` per 1M
  input/output tokens) because that exact ID was not present in the fetched
  pricing table.

<!--  -->
<!--  -->
<!--  -->
<!--  -->
<!--  -->

## Viewing Results

```sh
uv run inspect view
```

Results are stored in the `logs/` directory (or `$INSPECT_LOG_DIR` if set). For analysis workflows and artifact outputs, see `analysis/Analysis_README.md`.

### Generating Reports

You can generate an aggregated evaluation report from the Inspect AI logs. The script exports a directory structure containing:

- `report/summary.json` for aggregate metrics/CIs
- `report/eval_rows.parquet` for row-level analysis
- an interactive HTML dashboard and nested sample files for lazy-loading

```sh
uv run python src/llm_delusion_eval/scripts/generate_report.py
```

Or with all args (ie subselecting runs):

```sh
uv run python src/llm_delusion_eval/scripts/generate_report.py \
  --logs-dir logs \
  --output-dir report \
  --max-context-messages 0 \
  --models "openai/gpt-4o" \
  --annotation-id "bot-misrepresents-ability" \
  --grader-models "gemini-2.0-flash"
```

### Generating Analysis Figures and Tables

After generating the report artifacts:

```sh
# Figures only
uv run python -m analysis.generate_figures \
  --summary report/summary.json \
  --figures-only

# Both figures and table exports
uv run python -m analysis.generate_figures \
  --summary report/summary.json

# Optional: include validates-* codes in code-level outputs
uv run python -m analysis.generate_figures \
  --summary report/summary.json \
  --include-validates-codes
```

### Analysis Workflow

The following workflow runs from eval logs and generated report outputs only.

```sh
# 1) Build report outputs from window-only logs
uv run python -m llm_delusion_eval.scripts.generate_report \
  --logs-dir logs \
  --output-dir report

# 2) Generate core figures/tables from report outputs
uv run python -m analysis.generate_figures \
  --summary report/summary.json

# 3) Optional log/report-derived analyses
uv run python -m analysis.compute_reasoning_effects
uv run python -m analysis.compute_sequential_dynamics
uv run python -m analysis.compute_code_correlations
uv run python -m analysis.compute_code_residuals
```
