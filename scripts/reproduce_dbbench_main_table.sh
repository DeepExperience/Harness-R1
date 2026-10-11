#!/usr/bin/env bash
set -euo pipefail

MODE="${1:?Usage: bash reproduce_dbbench_main_table.sh replay|generate [additional evaluator arguments]}"
shift
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
REFERENCE="$ROOT/reproduction/dbbench_e0"
EVAL_PYTHON="${EVAL_PYTHON:?Set EVAL_PYTHON to the evaluator Python}"
AGENTBENCH_PYTHON="${AGENTBENCH_PYTHON:?Set AGENTBENCH_PYTHON to the controller Python}"
DBBENCH_WORKER_PYTHON="${DBBENCH_WORKER_PYTHON:?Set DBBENCH_WORKER_PYTHON}"
TARGET_BASE_URL="${TARGET_BASE_URL:?Set TARGET_BASE_URL}"
DBBENCH_MYSQL_HOST="${DBBENCH_MYSQL_HOST:?Set host[:port] of the prepared MySQL 8 instance}"
OUTPUT_ROOT="${OUTPUT_ROOT:?Set OUTPUT_ROOT to a NEW experiment directory}"
CONTROLLER_PORT_BASE="${CONTROLLER_PORT_BASE:?Set an unused controller port range}"
WORKER_PORT_BASE="${WORKER_PORT_BASE:?Set an unused worker port range}"

if [[ -e "$OUTPUT_ROOT" ]]; then
  printf 'Refusing to overwrite existing output: %s\n' "$OUTPUT_ROOT" >&2
  exit 2
fi

"$EVAL_PYTHON" - "$ROOT/data/dbbench/standard.jsonl" "$REFERENCE/data/provenance.json" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

data, provenance = map(Path, sys.argv[1:])
expected = json.loads(provenance.read_text())["test_data_sha256"]
if hashlib.sha256(data.read_bytes()).hexdigest() != expected:
    raise SystemExit("DBBench standard.jsonl differs from the archived main-table test file.")
PY

mode_args=()
case "$MODE" in
  replay)
    mode_args+=(--patch-source-root "$REFERENCE/outputs" --engineer-model harness-r1)
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

exec "$EVAL_PYTHON" "$ROOT/code/Relax/examples/harness_r1/eval_dbbench_gpt_patches.py" \
  --prepared-input "$REFERENCE/data/engineer_inputs.jsonl" \
  --output-root "$OUTPUT_ROOT" --run-id "dbbench_e0_repro_$(date -u +%Y%m%d_%H%M%S)" \
  --engineer-api-key EMPTY --engineer-api-key-env '' \
  --engineer-timeout 1200 --engineer-max-tokens 12288 \
  --engineer-concurrency "${ENGINEER_CONCURRENCY:-8}" \
  --engineer-temperature 0 --engineer-top-p 1 \
  --engineer-chat-template-kwargs '{"enable_thinking":true}' \
  --engineer-assistant-prefill $'<think>\n' --engineer-reasoning-effort '' \
  --engineer-retries 3 --engineer-retry-sleep 10 --empty-response-retries 1 \
  --response-protocol prefill_think_patch \
  --target-base-url "$TARGET_BASE_URL" --target-model "${TARGET_MODEL:-Qwen3.5-9B}" \
  --target-agent-name qwen35-9b-dbbench-main-table-reproduction \
  --rollout-concurrency "${ROLLOUT_CONCURRENCY:-10}" \
  --rollout-max-tokens 4096 --rollout-http-timeout 900 --rollout-tool-choice auto \
  --rollout-chat-template-kwargs '{"enable_thinking":false}' \
  --dbbench-env-driver manual --dbbench-manual-mysql-host "$DBBENCH_MYSQL_HOST" \
  --dbbench-data-file "$ROOT/data/dbbench/standard.jsonl" --dbbench-max-round 20 \
  --startup-timeout 300 --rollout-timeout 3600 --eval-timeout 5400 \
  --controller-port-base "$CONTROLLER_PORT_BASE" --worker-port-base "$WORKER_PORT_BASE" \
  --agentbench-dir "${AGENTBENCH_DIR:-$ROOT/code/life-harness/AgentBench}" \
  --agentbench-python "$AGENTBENCH_PYTHON" --dbbench-worker-python "$DBBENCH_WORKER_PYTHON" \
  "${mode_args[@]}" "$@"
