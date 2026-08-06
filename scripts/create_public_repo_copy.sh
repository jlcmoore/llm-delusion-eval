#!/usr/bin/env bash
set -euo pipefail

# Create a public-ready copy of the repository from tracked files only.
#
# Usage:
#   bash scripts/create_public_repo_copy.sh [dest] [excludes_file]
#
# Examples:
#   bash scripts/create_public_repo_copy.sh
#   bash scripts/create_public_repo_copy.sh /private/tmp/llm-delusion-eval-public
#   bash scripts/create_public_repo_copy.sh /private/tmp/llm-delusion-eval-public scripts/public_export_excludes.txt

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
DEFAULT_DEST="/private/tmp/llm-delusion-eval-public"
DEFAULT_EXCLUDES_FILE="${SCRIPT_DIR}/public_export_excludes.txt"

DEST="${1:-${DEFAULT_DEST}}"
EXCLUDES_FILE="${2:-${DEFAULT_EXCLUDES_FILE}}"

if [[ ! -f "${EXCLUDES_FILE}" ]]; then
  echo "Exclusion file not found: ${EXCLUDES_FILE}" >&2
  exit 1
fi

if [[ -e "${DEST}" ]]; then
  echo "Destination already exists: ${DEST}" >&2
  echo "Remove it or pass a different destination path." >&2
  exit 1
fi

mkdir -p "${DEST}"

TRACKED_FILE_LIST="$(mktemp)"
cleanup() {
  rm -f "${TRACKED_FILE_LIST}"
}
trap cleanup EXIT

# Copy exactly tracked files from the current working tree.
git -C "${REPO_ROOT}" ls-files -z > "${TRACKED_FILE_LIST}"
rsync -a --from0 --files-from="${TRACKED_FILE_LIST}" "${REPO_ROOT}/" "${DEST}/"

# Apply deterministic excludes (one glob/path per line, comments supported).
while IFS= read -r raw_line || [[ -n "${raw_line}" ]]; do
  line_without_comment="${raw_line%%#*}"
  exclude_pattern="$(printf '%s' "${line_without_comment}" | sed -e 's/^[[:space:]]*//' -e 's/[[:space:]]*$//')"

  if [[ -z "${exclude_pattern}" ]]; then
    continue
  fi

  while IFS= read -r match_path; do
    if [[ -z "${match_path}" ]]; then
      continue
    fi
    rm -rf "${DEST}/${match_path}"
  done < <(
    (
      cd "${DEST}"
      compgen -G "${exclude_pattern}" || true
    )
  )
done < "${EXCLUDES_FILE}"

echo "Created public copy at ${DEST}"
echo "Excluded paths from ${EXCLUDES_FILE}"
