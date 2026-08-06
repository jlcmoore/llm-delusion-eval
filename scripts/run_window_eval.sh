#!/usr/bin/env bash
set -euo pipefail

# Window-only eval with configurable model and reasoning effort.
#
# Usage:
#   bash scripts/run_window_eval.sh <model> [reasoning_effort]
#
# Examples:
#   bash scripts/run_window_eval.sh openai/gpt-5.4-2026-03-05 none
#   bash scripts/run_window_eval.sh openai/gpt-5.4-2026-03-05 high
#   bash scripts/run_window_eval.sh grok/grok-4.20-0309-non-reasoning

MODEL="${1:?Usage: bash scripts/run_window_eval.sh <model> [reasoning_effort]}"
REASONING_EFFORT="${2:-}"
GRADER_JSON='{"model":"openai/gpt-5.1-2025-11-13","reasoning_effort":"none"}'

REASONING_ARGS=()
if [ -n "$REASONING_EFFORT" ]; then
  REASONING_ARGS=(--reasoning-effort "$REASONING_EFFORT")
fi

echo "Model: $MODEL"
echo "Reasoning: ${REASONING_EFFORT:-<default>}"

uv run inspect eval src/llm_delusion_eval/tasks/delusions_eval.py \
  --model "$MODEL" \
  ${REASONING_ARGS[@]+"${REASONING_ARGS[@]}"} \
  --model-role "grader=$GRADER_JSON"
