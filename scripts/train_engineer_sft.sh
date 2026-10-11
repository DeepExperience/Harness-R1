#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
CONFIG="${1:-${HARNESS_R1_SFT_CONFIG:-${ROOT}/configs/sft/qwen35_9b_engineer_coldstart.yaml}}"
LLAMAFACTORY_CLI="${LLAMAFACTORY_CLI:-llamafactory-cli}"
source "$SCRIPT_DIR/path_helpers.sh"
CONFIG="$(harness_abspath "$CONFIG")"
harness_freeze_pythonpath

if ! command -v "${LLAMAFACTORY_CLI}" >/dev/null 2>&1; then
  echo "LlamaFactory executable not found: ${LLAMAFACTORY_CLI}" >&2
  exit 2
fi
LLAMAFACTORY_CLI="$(harness_command_path "$LLAMAFACTORY_CLI")"

cd "${ROOT}"
exec "${LLAMAFACTORY_CLI}" train "${CONFIG}"
