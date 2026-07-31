#!/usr/bin/env bash
set -euo pipefail

INPUT="${1:?usage: eval_webshop.sh INPUT_JSONL OUTPUT_DIR}"
OUTPUT="${2:?usage: eval_webshop.sh INPUT_JSONL OUTPUT_DIR}"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="$(command -v "${PYTHON_BIN:-python}")"
AGENTBENCH_PYTHON="${AGENTBENCH_PYTHON:-${PYTHON_BIN}}"
WEBSHOP_WORKER_PYTHON="${WEBSHOP_WORKER_PYTHON:-${PYTHON_BIN}}"
RUN_ID="${RUN_ID:-webshop_$(date +%Y%m%d_%H%M%S)}"

ENGINEER_ASSISTANT_PREFILL="${ENGINEER_ASSISTANT_PREFILL:-$'<think>\n'}"
OPTIONAL=()
[[ -n "${PATCH_SOURCE_ROOT:-}" ]] && OPTIONAL+=(--patch-source-root "${PATCH_SOURCE_ROOT}")
[[ "${GENERATE_ONLY:-0}" == "1" ]] && OPTIONAL+=(--generate-only)
[[ "${RESUME:-0}" == "1" ]] && OPTIONAL+=(--resume)
[[ "${RESUME_RERUN_EVAL_FAILED:-0}" == "1" ]] && OPTIONAL+=(--resume-rerun-eval-failed)
[[ -n "${LIMIT:-}" ]] && OPTIONAL+=(--limit "${LIMIT}")
OPTIONAL+=(--webshop-worker-python "${WEBSHOP_WORKER_PYTHON}")

cd "${ROOT}"
exec "${PYTHON_BIN}" code/Relax/examples/harness_r1/eval_webshop_checkpoint_patches.py \
  --input "${INPUT}" \
  --output-root "${OUTPUT}" \
  --run-id "${RUN_ID}" \
  --engineer-base-url "${ENGINEER_BASE_URL:-http://127.0.0.1:8142/v1}" \
  --engineer-model "${ENGINEER_MODEL:-harness-engineer}" \
  --engineer-api-key-env "${ENGINEER_API_KEY_ENV:-HARNESS_R1_ENGINEER_API_KEY}" \
  --engineer-max-tokens "${ENGINEER_MAX_TOKENS:-12288}" \
  --engineer-concurrency "${ENGINEER_CONCURRENCY:-8}" \
  --engineer-temperature "${ENGINEER_TEMPERATURE:-0.0}" \
  --engineer-top-p "${ENGINEER_TOP_P:-1.0}" \
  --engineer-chat-template-kwargs "${ENGINEER_CHAT_TEMPLATE_KWARGS:-{\"enable_thinking\":true}}" \
  --engineer-assistant-prefill "${ENGINEER_ASSISTANT_PREFILL}" \
  --engineer-retries "${ENGINEER_RETRIES:-3}" \
  --response-protocol prefill_think_patch \
  --require-code-hook-only \
  --expect-benchmark webshop \
  --expect-schema-style "${WEBSHOP_SCHEMA_STYLE:-webshop_life_multihook_v1}" \
  --expect-response-protocol prefill_think_patch \
  --target-base-url "${TARGET_BASE_URL:-http://127.0.0.1:8110/v1}" \
  --target-model "${TARGET_MODEL:-Qwen3.5-9B}" \
  --target-agent-name "${TARGET_AGENT_NAME:-qwen35-9b-nothink}" \
  --reward-metric "${REWARD_METRIC:-delta_average_reward}" \
  --rollout-concurrency "${ROLLOUT_CONCURRENCY:-16}" \
  --rollout-max-tokens "${ROLLOUT_MAX_TOKENS:-4096}" \
  --rollout-tool-choice "${ROLLOUT_TOOL_CHOICE:-auto}" \
  --rollout-chat-template-kwargs "${ROLLOUT_CHAT_TEMPLATE_KWARGS:-{\"enable_thinking\":false}}" \
  --rollout-auto-retry \
  --webshop-rounds "${WEBSHOP_ROUNDS:-20}" \
  --webshop-goal-seed "${WEBSHOP_GOAL_SEED:-233}" \
  --require-webshop-identity \
  --require-webshop-goal-seed \
  --agentbench-dir "${AGENTBENCH_DIR:-${ROOT}/code/life-harness/AgentBench}" \
  --agentbench-python "${AGENTBENCH_PYTHON}" \
  "${OPTIONAL[@]}"
