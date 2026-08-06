#!/usr/bin/env bash
set -euo pipefail

# Sync large eval artifacts between local disk and a GCS bucket.
#
# Usage:
#   bash scripts/sync_artifacts_gcs.sh <upload|download> <bucket> [prefix]
#
# Examples:
#   bash scripts/sync_artifacts_gcs.sh upload llm-delusion-eval-artifacts
#   bash scripts/sync_artifacts_gcs.sh download llm-delusion-eval-artifacts
#   bash scripts/sync_artifacts_gcs.sh upload llm-delusion-eval-artifacts release-2026-05-04
#
# Synced directories:
#   logs/
#   logs-context/
#   report/samples/

MODE="${1:-}"
BUCKET="${2:-${LLM_DELUSIONS_EVALS_ARTIFACT_BUCKET:-}}"
PREFIX="${3:-${LLM_DELUSIONS_EVALS_ARTIFACT_PREFIX:-}}"

if [[ "$MODE" != "upload" && "$MODE" != "download" ]]; then
  echo "Usage: bash scripts/sync_artifacts_gcs.sh <upload|download> <bucket> [prefix]" >&2
  exit 1
fi

if [[ -z "$BUCKET" ]]; then
  echo "Missing bucket. Pass it as arg 2 or set LLM_DELUSIONS_EVALS_ARTIFACT_BUCKET." >&2
  exit 1
fi

if ! command -v gcloud >/dev/null 2>&1; then
  echo "gcloud is required but not found in PATH." >&2
  exit 1
fi

ARTIFACT_DIRS=(
  "logs"
  "logs-context"
  "report/samples"
)

for dir in "${ARTIFACT_DIRS[@]}"; do
  local_path="${dir}/"
  if [[ -n "$PREFIX" ]]; then
    remote_path="gs://${BUCKET}/${PREFIX}/${dir}/"
  else
    remote_path="gs://${BUCKET}/${dir}/"
  fi

  if [[ "$MODE" == "upload" ]]; then
    if [[ ! -d "$dir" ]]; then
      echo "Skipping missing local directory: $dir"
      continue
    fi
    echo "Uploading $local_path -> $remote_path"
    gcloud storage rsync --recursive "$local_path" "$remote_path"
  else
    if ! gcloud storage ls "$remote_path" >/dev/null 2>&1; then
      echo "Skipping missing remote directory: $remote_path"
      continue
    fi
    mkdir -p "$dir"
    echo "Downloading $remote_path -> $local_path"
    gcloud storage rsync --recursive "$remote_path" "$local_path"
  fi
done
