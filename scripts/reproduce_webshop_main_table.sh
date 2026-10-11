#!/usr/bin/env bash
set -euo pipefail

MODE="${1:?Usage: bash reproduce_webshop_main_table.sh replay|generate [additional evaluator arguments]}"
shift
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "${SCRIPT_DIR}/.." && pwd)"
BUNDLE="$REPO/reproduction/webshop_e0"
EVAL_PYTHON="${EVAL_PYTHON:?Set EVAL_PYTHON to the evaluator Python}"
AGENTBENCH_PYTHON="${AGENTBENCH_PYTHON:?Set AGENTBENCH_PYTHON to the controller Python}"
WEBSHOP_WORKER_PYTHON="${WEBSHOP_WORKER_PYTHON:?Set WEBSHOP_WORKER_PYTHON}"
TARGET_BASE_URL="${TARGET_BASE_URL:?Set TARGET_BASE_URL}"
OUTPUT_ROOT="${OUTPUT_ROOT:?Set OUTPUT_ROOT to a NEW experiment directory}"
CONTROLLER_PORT_BASE="${CONTROLLER_PORT_BASE:?Set an unused controller port range (50 ports)}"
WORKER_PORT_BASE="${WORKER_PORT_BASE:?Set an unused worker port range (50 ports)}"

if [[ -e "$OUTPUT_ROOT" ]]; then
  printf 'Refusing to overwrite existing output: %s\n' "$OUTPUT_ROOT" >&2
  exit 2
fi

mode_args=()
case "$MODE" in
  replay)
    mode_args+=(--patch-source-root "$BUNDLE/patches")
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

exec "$EVAL_PYTHON" "$REPO/code/Relax/examples/harness_r1/eval_webshop_checkpoint_patches.py" \
  --input "$BUNDLE/data/engineer_inputs.jsonl" \
  --output-root "$OUTPUT_ROOT" \
  --run-id "webshop_e0_repro_$(date -u +%Y%m%d_%H%M%S)" \
  --engineer-timeout 1200 --engineer-max-tokens 12288 \
  --engineer-concurrency "${ENGINEER_CONCURRENCY:-8}" \
  --engineer-temperature 0 --engineer-top-p 1 \
  --engineer-chat-template-kwargs '{"enable_thinking":true}' \
  --engineer-assistant-prefill $'<think>\n' \
  --engineer-retries 3 --engineer-retry-sleep 10 \
  --target-base-url "$TARGET_BASE_URL" --target-model "${TARGET_MODEL:-Qwen3.5-9B}" \
  --target-agent-name qwen35-9b-webshop-main-table-reproduction \
  --reward-metric delta_pass_rate --rollout-concurrency "${ROLLOUT_CONCURRENCY:-10}" \
  --rollout-auto-retry --rollout-max-tokens 4096 --rollout-tool-choice auto \
  --rollout-chat-template-kwargs '{"enable_thinking":false}' \
  --webshop-rounds 20 --webshop-goal-seed 233 \
  --require-webshop-goal-seed --require-webshop-identity \
  --startup-timeout 900 --rollout-timeout 3600 --eval-timeout 7200 \
  --controller-port-base "$CONTROLLER_PORT_BASE" --worker-port-base "$WORKER_PORT_BASE" \
  --require-code-hook-only --expect-benchmark webshop \
  --expect-schema-style webshop_code_only \
  --expect-response-protocol prefill_think_patch --response-protocol prefill_think_patch \
  --agentbench-dir "$REPO/code/life-harness/AgentBench" \
  --agentbench-python "$AGENTBENCH_PYTHON" --webshop-worker-python "$WEBSHOP_WORKER_PYTHON" \
  "${mode_args[@]}" "$@"
