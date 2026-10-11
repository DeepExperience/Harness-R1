#!/usr/bin/env bash
set -euo pipefail

MODE="${1:?Usage: bash reproduce_alfworld_main_table.sh replay|generate [additional evaluator arguments]}"
shift
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
REFERENCE="$ROOT/reproduction/alfworld_e0"
AGENTBENCH_DIR="${AGENTBENCH_DIR:-$ROOT/code/life-harness/AgentBench}"
EVAL_PYTHON="${EVAL_PYTHON:?Set EVAL_PYTHON to the evaluator Python}"
AGENTBENCH_PYTHON="${AGENTBENCH_PYTHON:?Set AGENTBENCH_PYTHON to the controller Python}"
ALFWORLD_WORKER_PYTHON="${ALFWORLD_WORKER_PYTHON:?Set ALFWORLD_WORKER_PYTHON}"
TARGET_BASE_URL="${TARGET_BASE_URL:?Set TARGET_BASE_URL}"
OUTPUT_ROOT="${OUTPUT_ROOT:?Set OUTPUT_ROOT to a NEW experiment directory}"
CONTROLLER_PORT_BASE="${CONTROLLER_PORT_BASE:?Set an unused controller port range}"
WORKER_PORT_BASE="${WORKER_PORT_BASE:?Set an unused worker port range}"

if [[ -e "$OUTPUT_ROOT" ]]; then
  printf 'Refusing to overwrite existing output: %s\n' "$OUTPUT_ROOT" >&2
  exit 2
fi

"$EVAL_PYTHON" - "$ROOT/data/alfworld/test_set.json" "$AGENTBENCH_DIR/data/alfworld/test_set.json" <<'PY'
import json
import sys
from pathlib import Path

reference, installed = map(Path, sys.argv[1:])
if not installed.is_file():
    raise SystemExit("Install data/alfworld/test_set.json in the AgentBench data directory first.")
if json.loads(reference.read_text()) != json.loads(installed.read_text()):
    raise SystemExit("Installed ALFWorld test_set differs from the published main-table split/order.")
PY

mode_args=()
case "$MODE" in
  replay)
    mode_args+=(--patch-source-root "$REFERENCE/outputs")
    ;;
  generate)
    mode_args+=(
      --engineer-base-url "${ENGINEER_BASE_URL:?Set ENGINEER_BASE_URL}"
      --engineer-model "${ENGINEER_MODEL:?Set ENGINEER_MODEL}"
      --engineer-api-key-env "${ENGINEER_API_KEY_ENV:-}"
    )
    case "${ENGINEER_BACKEND:-sglang}" in
      sglang) ;;
      vllm) mode_args+=(--engineer-extra-body '{"add_generation_prompt":false}') ;;
      *) printf 'ENGINEER_BACKEND must be sglang or vllm\n' >&2; exit 2 ;;
    esac
    ;;
  *) printf 'Mode must be replay or generate\n' >&2; exit 2 ;;
esac

exec "$EVAL_PYTHON" "$ROOT/code/Relax/examples/harness_r1/eval_alfworld_patches.py" \
  --input "$REFERENCE/data/engineer_inputs.jsonl" \
  --output-root "$OUTPUT_ROOT" --run-id "alfworld_e0_repro_$(date -u +%Y%m%d_%H%M%S)" \
  --engineer-timeout 1200 --engineer-max-tokens 12288 \
  --engineer-concurrency "${ENGINEER_CONCURRENCY:-8}" \
  --engineer-temperature 0 --engineer-top-p 1 \
  --engineer-chat-template-kwargs '{"enable_thinking":true}' \
  --engineer-assistant-prefill $'<think>\n' \
  --engineer-retries 3 --engineer-retry-sleep 10 \
  --response-protocol prefill_think_patch \
  --target-base-url "$TARGET_BASE_URL" --target-model "${TARGET_MODEL:-Qwen3.5-9B}" \
  --target-agent-name qwen35-9b-alfworld-main-table-reproduction \
  --alfworld-split test_set --alfworld-max-step 50 \
  --alfworld-worker-python "$ALFWORLD_WORKER_PYTHON" \
  --rollout-concurrency "${ROLLOUT_CONCURRENCY:-10}" --rollout-auto-retry \
  --rollout-max-tokens 4096 --rollout-tool-choice auto \
  --rollout-chat-template-kwargs '{"enable_thinking":false}' \
  --startup-timeout 900 --rollout-timeout 3600 --eval-timeout 7200 \
  --controller-port-base "$CONTROLLER_PORT_BASE" --worker-port-base "$WORKER_PORT_BASE" \
  --agentbench-dir "$AGENTBENCH_DIR" --agentbench-python "$AGENTBENCH_PYTHON" \
  "${mode_args[@]}" "$@"
