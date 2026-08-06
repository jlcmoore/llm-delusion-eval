#!/usr/bin/env bash
set -euo pipefail

# Context + window eval with sanitized windows and configurable context depth.
#
# Usage:
#   bash scripts/run_context_eval.sh <model> <max_context_messages> [codes] [log_dir] [min_context_messages] [reasoning_effort]
#
# Examples:
#   bash scripts/run_context_eval.sh openai/gpt-5.4-mini-2026-03-17 200
#   bash scripts/run_context_eval.sh openai/gpt-5.4-mini-2026-03-17 0 bot-misrepresents-sentience
#   bash scripts/run_context_eval.sh openai/gpt-5.4-mini-2026-03-17 200 "" logs-context-200
#   bash scripts/run_context_eval.sh openai/gpt-5.4-2026-03-05 500 bot-discourages-self-harm logs-context 300
#   bash scripts/run_context_eval.sh anthropic/claude-opus-4-7 200 "" logs-context 200 none
#
# If min_context_messages is omitted, context-mode runs default to requiring
# full context: min_context_messages = max_context_messages.

MODEL="${1:?Usage: bash scripts/run_context_eval.sh <model> <max_context_messages> [codes] [log_dir] [min_context_messages] [reasoning_effort]}"
MAX_CONTEXT_MESSAGES="${2:?Usage: bash scripts/run_context_eval.sh <model> <max_context_messages> [codes] [log_dir] [min_context_messages] [reasoning_effort]}"
CODES="${3:-}"
LOG_DIR="${4:-logs-context}"
MIN_CONTEXT_MESSAGES="${5:-${MIN_CONTEXT_MESSAGES:-}}"
REASONING_EFFORT="${6:-${REASONING_EFFORT:-none}}"

if [ -z "$MIN_CONTEXT_MESSAGES" ]; then
  if [ "$MAX_CONTEXT_MESSAGES" -gt 0 ]; then
    MIN_CONTEXT_MESSAGES="$MAX_CONTEXT_MESSAGES"
  else
    MIN_CONTEXT_MESSAGES="0"
  fi
fi

if [ -n "${LLM_DELUSIONS_WINDOWS_PATH:-}" ]; then
  export LLM_DELUSIONS_WINDOWS_PATH
elif [ "$MAX_CONTEXT_MESSAGES" -ne 0 ]; then
  # Context mode needs the non-sanitized parquet because it filters on
  # `selected_for_eval` before loading the context windows.
  # max_context_messages < 0 means "all available context".
  export LLM_DELUSIONS_WINDOWS_PATH="../llm-delusions/subsets/items.parquet"
else
  export LLM_DELUSIONS_WINDOWS_PATH="hf://datasets/jlcmoore/delusioneval/items_sanitized.parquet"
fi
export LLM_DELUSIONS_TRANSCRIPTS_PATH="${LLM_DELUSIONS_TRANSCRIPTS_PATH:-../llm-delusions/transcripts_data/transcripts.parquet}"
if [ -n "$LOG_DIR" ]; then
  export INSPECT_LOG_DIR="$LOG_DIR"
fi
GRADER_JSON='{"model":"openai/gpt-5.1-2025-11-13","reasoning_effort":"none"}'
REASONING_ARGS=(--reasoning-effort "$REASONING_EFFORT")
TASK_ARGS=(-T max_context_messages="$MAX_CONTEXT_MESSAGES")
export LLM_DELUSIONS_MIN_CONTEXT_MESSAGES="$MIN_CONTEXT_MESSAGES"
if [ -n "$CODES" ]; then
  TASK_ARGS+=(-T "codes=$CODES")
fi

echo "Model: $MODEL"
echo "Max context messages: $MAX_CONTEXT_MESSAGES"
echo "Min context messages: $MIN_CONTEXT_MESSAGES"
echo "Windows path: $LLM_DELUSIONS_WINDOWS_PATH"
echo "Transcripts path: $LLM_DELUSIONS_TRANSCRIPTS_PATH"
echo "Log dir: ${INSPECT_LOG_DIR:-<default>}"
echo "Reasoning effort: $REASONING_EFFORT"
echo "Codes: ${CODES:-<all>}"

uv run inspect eval src/llm_delusion_eval/tasks/delusions_eval.py \
  --model "$MODEL" \
  "${REASONING_ARGS[@]}" \
  --model-role "grader=$GRADER_JSON" \
  "${TASK_ARGS[@]}"
