#!/bin/bash

# Copyright (c) 2026 Relax Authors. All Rights Reserved.
#
# Qwen3.5-9B Harness-R1 mixed-benchmark single-step GRPO.
#
# Required environment:
#   QWEN35_HF       Hugging Face checkpoint used for actor/reference init
#   HARNESS_R1_DATA grouped RL JSONL
#   NUM_ROLLOUT     number of rollout batches to consume
#
# Optional:
#   HARNESS_R1_CONFIG, SAVE_PATH, LOAD_PATH, WANDB_API_KEY, RESOURCE_JSON

set -euo pipefail

now=$(date "+%Y%m%d-%H%M%S")

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
HARNESS_ROOT="$(cd -- "${SCRIPT_DIR}/.." &>/dev/null && pwd)"
RELAX_ROOT="${HARNESS_ROOT}/code/Relax"

export RELAX="${RELAX:-${RELAX_ROOT}}"
export MEGATRON="${MEGATRON:-${RELAX_ROOT}/deps/Megatron-LM}"
export SGLANG="${SGLANG:-${RELAX_ROOT}/deps/sglang/python}"
export PYTHONPATH="${RELAX}:${MEGATRON}:${SGLANG}:${RELAX}:${PYTHONPATH:-}"
export MODEL_CONFIG_DIR="${MODEL_CONFIG_DIR:-${RELAX_ROOT}/scripts/models}"
export PYTHONUNBUFFERED=1
export CUDA_DEVICE_MAX_CONNECTIONS=1
export RELAX_MASTER_PORT_BASE_ACTOR="${RELAX_MASTER_PORT_BASE_ACTOR:-29000}"
export RELAX_MASTER_PORT_BASE_ACTOR_FWD="${RELAX_MASTER_PORT_BASE_ACTOR_FWD:-30000}"
export RELAX_MASTER_PORT_BASE_REFERENCE="${RELAX_MASTER_PORT_BASE_REFERENCE:-31000}"
export RELAX_MASTER_PORT_BASE_CRITIC="${RELAX_MASTER_PORT_BASE_CRITIC:-32000}"
export RELAX_SERVE_PORT="${RELAX_SERVE_PORT:-18080}"
export NCCL_CUMEM_ENABLE="${NCCL_CUMEM_ENABLE:-0}"
export NCCL_NVLS_ENABLE="${NCCL_NVLS_ENABLE:-0}"
export NCCL_ALGO="${NCCL_ALGO:-^NVLS}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
export NCCL_RAS_ENABLE="${NCCL_RAS_ENABLE:-0}"
export NCCL_COLLNET_ENABLE="${NCCL_COLLNET_ENABLE:-0}"
export NCCL_SOCKET_RETRY_CNT="${NCCL_SOCKET_RETRY_CNT:-50}"
export NCCL_SOCKET_RETRY_SLEEP_MSEC="${NCCL_SOCKET_RETRY_SLEEP_MSEC:-100}"
PYTHON_BIN="${PYTHON_BIN:-$(command -v python3)}"
RAY_BIN="${RAY_BIN:-$(command -v ray)}"
export RUN_DIRECT="${RUN_DIRECT:-1}"
HOST_IP="$(hostname -I 2>/dev/null | awk '{print $1}')"
NO_PROXY_VALUE="${NO_PROXY_VALUE:-127.0.0.1,localhost,0.0.0.0,${MASTER_ADDR:-127.0.0.1},${HOST_IP}}"
export NO_PROXY="${NO_PROXY:-${NO_PROXY_VALUE}}"
export no_proxy="${no_proxy:-${NO_PROXY_VALUE}}"

source "${MODEL_CONFIG_DIR}/qwen35-9B.sh"
cd "${RELAX_ROOT}"

if [ "${RUN_DIRECT:-0}" != "1" ] && ! timeout "${RAY_STATUS_TIMEOUT:-10}" "${RAY_BIN}" status >/dev/null 2>&1; then
   if [ "${START_LOCAL_RAY:-0}" = "1" ]; then
      "${RAY_BIN}" start --head \
         --node-ip-address "${MASTER_ADDR:-127.0.0.1}" \
         --num-gpus "${NUM_GPUS:-4}" \
         --disable-usage-stats \
         --dashboard-host=0.0.0.0 \
         --dashboard-port="${RAY_DASHBOARD_PORT:-8265}"
   else
      echo "Ray is not running. Start Ray first, or rerun with START_LOCAL_RAY=1." >&2
      exit 1
   fi
fi

export HAS_NVLINK="${HAS_NVLINK:-0}"
export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"

if [ -z "${WANDB_API_KEY:-}" ] && [ -f "${HOME}/.netrc" ]; then
   WANDB_API_KEY="$("${PYTHON_BIN}" - <<'PY'
import netrc
import os

try:
    auth = netrc.netrc(os.path.expanduser("~/.netrc")).authenticators("api.wandb.ai")
except Exception:
    auth = None
if auth and auth[2]:
    print(auth[2], end="")
PY
)"
   if [ -n "${WANDB_API_KEY}" ]; then
      export WANDB_API_KEY
   fi
fi
export RUNTIME_ENV_JSON="${RUNTIME_ENV_JSON:-{
\"worker_process_setup_hook\": \"relax.utils.logging_utils.install_asyncio_noise_filter\",
\"env_vars\": {
   \"PYTHONUNBUFFERED\": \"1\",
   \"PATH\": \"${PATH}\",
   \"PYTHONPATH\": \"${PYTHONPATH}\",
   \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
   \"RAY_OVERRIDE_JOB_RUNTIME_ENV\": \"1\",
   \"NCCL_NVLS_ENABLE\": \"${NCCL_NVLS_ENABLE}\",
   \"NCCL_CUMEM_ENABLE\": \"${NCCL_CUMEM_ENABLE}\",
   \"NCCL_ALGO\": \"${NCCL_ALGO}\",
   \"NCCL_IB_DISABLE\": \"${NCCL_IB_DISABLE}\",
   \"NCCL_RAS_ENABLE\": \"${NCCL_RAS_ENABLE}\",
   \"NCCL_COLLNET_ENABLE\": \"${NCCL_COLLNET_ENABLE}\",
   \"NCCL_SOCKET_RETRY_CNT\": \"${NCCL_SOCKET_RETRY_CNT}\",
   \"NCCL_SOCKET_RETRY_SLEEP_MSEC\": \"${NCCL_SOCKET_RETRY_SLEEP_MSEC}\",
   \"RELAX_MASTER_PORT_BASE_ACTOR\": \"${RELAX_MASTER_PORT_BASE_ACTOR}\",
	   \"RELAX_MASTER_PORT_BASE_ACTOR_FWD\": \"${RELAX_MASTER_PORT_BASE_ACTOR_FWD}\",
	   \"RELAX_MASTER_PORT_BASE_REFERENCE\": \"${RELAX_MASTER_PORT_BASE_REFERENCE}\",
	   \"RELAX_MASTER_PORT_BASE_CRITIC\": \"${RELAX_MASTER_PORT_BASE_CRITIC}\",
	   \"RELAX_SERVE_PORT\": \"${RELAX_SERVE_PORT}\",
	   \"MASTER_ADDR\": \"${MASTER_ADDR}\",
   \"WANDB_BASE_URL\": \"${WANDB_BASE_URL:-https://api.wandb.ai}\",
   \"NO_PROXY\": \"${NO_PROXY}\",
   \"no_proxy\": \"${no_proxy}\",
   \"SGLANG_HEALTH_CHECK_TIMEOUT\": \"${SGLANG_HEALTH_CHECK_TIMEOUT:-180}\"
}
}}"

PROJECT_NAME="${PROJECT_NAME:=Harness-R1/mixed-codepatch}"
EXP_DIR="${EXP_DIR:-${HARNESS_ROOT}}"
SAVE_DIR="${SAVE_DIR:-${EXP_DIR}/outputs/checkpoints}"
: "${QWEN35_HF:?Set QWEN35_HF to the actor/reference Hugging Face checkpoint}"
: "${HARNESS_R1_DATA:?Set HARNESS_R1_DATA to a grouped Harness-R1 JSONL}"
: "${NUM_ROLLOUT:?Set NUM_ROLLOUT to the number of rollout batches}"
HARNESS_R1_CONFIG="${HARNESS_R1_CONFIG:-${EXP_DIR}/configs/rl/mixed_codepatch.yaml}"
CUSTOM_RM_PATH="${CUSTOM_RM_PATH:-examples.harness_r1.reward_mixed_codepatch.reward_func}"
SAVE_PATH="${SAVE_PATH:-${SAVE_DIR}/qwen35-9b-harness-r1-${now}}"
if [ -z "${APPLY_CHAT_TEMPLATE_KWARGS:-}" ]; then
   APPLY_CHAT_TEMPLATE_KWARGS='{"enable_thinking":true}'
fi
N_SAMPLES_PER_PROMPT="${N_SAMPLES_PER_PROMPT:=8}"
ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:=4}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:=32}"
NUM_ITERS_PER_TRAIN_UPDATE="${NUM_ITERS_PER_TRAIN_UPDATE:-4}"
REWARD_MAX_CONCURRENCY="${REWARD_MAX_CONCURRENCY:=10}"
REWARD_NUM_WORKERS="${REWARD_NUM_WORKERS:=10}"
if [ -z "${RESOURCE_JSON:-}" ]; then
   RESOURCE_JSON='{"actor":[1,4],"rollout":[1,2],"reference":[1,1],"actor_fwd":[1,1],"advantages":[1,0]}'
fi
if [ -z "${REF_ACTOR_CONFIG_JSON:-}" ]; then
   REF_ACTOR_CONFIG_JSON='{"tensor_model_parallel_size":1,"max_tokens_per_gpu":40960,"sequence_parallel":false,"only_load_weight":true}'
fi
TENSOR_MODEL_PARALLEL_SIZE="${TENSOR_MODEL_PARALLEL_SIZE:-4}"
SEQUENCE_PARALLEL="${SEQUENCE_PARALLEL:-1}"
PIPELINE_MODEL_PARALLEL_SIZE="${PIPELINE_MODEL_PARALLEL_SIZE:-1}"
CONTEXT_PARALLEL_SIZE="${CONTEXT_PARALLEL_SIZE:-1}"
MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-1}"
MAX_TOKENS_PER_GPU="${MAX_TOKENS_PER_GPU:-40960}"
ROLLOUT_MAX_CONTEXT_LEN="${ROLLOUT_MAX_CONTEXT_LEN:-40960}"
ROLLOUT_MAX_PROMPT_LEN="${ROLLOUT_MAX_PROMPT_LEN:-28672}"
ROLLOUT_MAX_RESPONSE_LEN="${ROLLOUT_MAX_RESPONSE_LEN:-12288}"
ROLLOUT_NUM_GPUS_PER_ENGINE="${ROLLOUT_NUM_GPUS_PER_ENGINE:-2}"
SGLANG_MEM_FRACTION_STATIC="${SGLANG_MEM_FRACTION_STATIC:-0.60}"
if (( ROLLOUT_BATCH_SIZE <= 0 || N_SAMPLES_PER_PROMPT <= 0 || GLOBAL_BATCH_SIZE <= 0 )); then
   echo "ROLLOUT_BATCH_SIZE, N_SAMPLES_PER_PROMPT, and GLOBAL_BATCH_SIZE must all be positive." >&2
   exit 2
fi
if (( NUM_ITERS_PER_TRAIN_UPDATE <= 0 )); then
   echo "NUM_ITERS_PER_TRAIN_UPDATE must be positive." >&2
   exit 2
fi
if (( GLOBAL_BATCH_SIZE % (NUM_ITERS_PER_TRAIN_UPDATE * N_SAMPLES_PER_PROMPT) != 0 )); then
   echo "GLOBAL_BATCH_SIZE must be divisible by NUM_ITERS_PER_TRAIN_UPDATE * N_SAMPLES_PER_PROMPT." >&2
   echo "Got GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE}, NUM_ITERS_PER_TRAIN_UPDATE=${NUM_ITERS_PER_TRAIN_UPDATE}, N_SAMPLES_PER_PROMPT=${N_SAMPLES_PER_PROMPT}." >&2
   exit 2
fi

CKPT_ARGS=(
   --hf-checkpoint "${QWEN35_HF}"
   --ref-load "${QWEN35_HF}"
   --megatron-to-hf-mode bridge
   --save "${SAVE_PATH}"
   --save-interval "${SAVE_INTERVAL:-16}"
   --no-save-optim
   --no-load-optim
)
if [ -n "${LOAD_PATH:-}" ]; then
   CKPT_ARGS+=(--load "${LOAD_PATH}")
fi

ROLLOUT_ARGS=(
   --prompt-data "${HARNESS_R1_DATA}"
   --input-key prompt
   --label-key label
   --metadata-key metadata
   --apply-chat-template
   --apply-chat-template-kwargs "${APPLY_CHAT_TEMPLATE_KWARGS}"

   --custom-rm-path "${CUSTOM_RM_PATH}"
   --custom-config-path "${HARNESS_R1_CONFIG}"
   --reward-key score
   --reward-max-concurrency "${REWARD_MAX_CONCURRENCY}"
   --reward-num-workers "${REWARD_NUM_WORKERS}"

   --num-rollout "${NUM_ROLLOUT}"
   --rollout-batch-size "${ROLLOUT_BATCH_SIZE}"
   --n-samples-per-prompt "${N_SAMPLES_PER_PROMPT}"
   --rollout-max-context-len "${ROLLOUT_MAX_CONTEXT_LEN}"
   --rollout-max-prompt-len "${ROLLOUT_MAX_PROMPT_LEN}"
   --rollout-max-response-len "${ROLLOUT_MAX_RESPONSE_LEN}"
   --rollout-temperature "${ROLLOUT_TEMPERATURE:-0.7}"
   --rollout-top-p "${ROLLOUT_TOP_P:-0.95}"

   --global-batch-size "${GLOBAL_BATCH_SIZE}"
   --use-fault-tolerance
)
if [ "${ROLLOUT_SHUFFLE:-1}" = "1" ]; then
   ROLLOUT_ARGS+=(--rollout-shuffle)
fi
if [ -n "${START_ROLLOUT_ID:-}" ]; then
   ROLLOUT_ARGS+=(--start-rollout-id "${START_ROLLOUT_ID}")
fi
if [ "${USE_DYNAMIC_SAMPLING:-0}" = "1" ] && [ -z "${DYNAMIC_SAMPLING_FILTER_PATH:-}" ]; then
   DYNAMIC_SAMPLING_FILTER_PATH="relax.engine.filters.dynamic_sampling_filters.check_reward_nonzero_std"
fi
if [ -n "${DYNAMIC_SAMPLING_FILTER_PATH:-}" ]; then
   ROLLOUT_ARGS+=(
      --dynamic-sampling-filter-path "${DYNAMIC_SAMPLING_FILTER_PATH}"
   )
fi
if [ -n "${OVER_SAMPLING_BATCH_SIZE:-}" ]; then
   ROLLOUT_ARGS+=(
      --over-sampling-batch-size "${OVER_SAMPLING_BATCH_SIZE}"
   )
fi

PERF_ARGS=(
   --tensor-model-parallel-size "${TENSOR_MODEL_PARALLEL_SIZE}"
   --pipeline-model-parallel-size "${PIPELINE_MODEL_PARALLEL_SIZE}"
   --context-parallel-size "${CONTEXT_PARALLEL_SIZE}"
   --expert-model-parallel-size 1
   --expert-tensor-parallel-size 1

   --recompute-granularity full
   --recompute-method uniform
   --recompute-num-layers 1

   --micro-batch-size "${MICRO_BATCH_SIZE}"
   --max-tokens-per-gpu "${MAX_TOKENS_PER_GPU}"
)
if [ "${SEQUENCE_PARALLEL}" = "1" ]; then
   PERF_ARGS+=(--sequence-parallel)
fi

GRPO_ARGS=(
   --advantage-estimator grpo
   --entropy-coef "${ENTROPY_COEF:-0.00}"
   --eps-clip 0.2
   --eps-clip-high 0.28
   --use-tis
)
if [ "${USE_KL_LOSS:-0}" = "1" ]; then
   GRPO_ARGS+=(
      --use-kl-loss
      --kl-loss-coef "${KL_LOSS_COEF:-0.00}"
      --kl-loss-type low_var_kl
   )
fi

OPTIMIZER_ARGS=(
   --optimizer adam
   --lr "${LR:-1e-6}"
   --lr-decay-style constant
   --weight-decay 0.1
   --adam-beta1 0.9
   --adam-beta2 0.98
   --optimizer-cpu-offload
   --overlap-cpu-optimizer-d2h-h2d
   --use-precision-aware-optimizer
)

SGLANG_ARGS=(
   --rollout-num-gpus-per-engine "${ROLLOUT_NUM_GPUS_PER_ENGINE}"
   --sglang-mem-fraction-static "${SGLANG_MEM_FRACTION_STATIC}"
)

TRACKING_ARGS=(
   --use-metrics-service
   --tb-project-name "${PROJECT_NAME}"
   --tb-experiment-name "qwen35-9b-harness-r1-${now}"
)
if [ "${USE_CLEARML:-0}" = "1" ]; then
   TRACKING_ARGS+=(--use-clearml)
fi
if [ "${USE_WANDB:-1}" = "1" ]; then
   TRACKING_ARGS+=(
      --use-wandb
      --wandb-mode "${WANDB_MODE:-online}"
      --wandb-project "${WANDB_PROJECT:-Harness-R1}"
      --wandb-group "${WANDB_GROUP:-mixed-codepatch-qwen35-9b-${now}}"
      --wandb-dir "${WANDB_DIR:-${EXP_DIR}/outputs/wandb}"
      --disable-wandb-random-suffix
   )
fi

MISC_ARGS=(
   --attention-dropout 0.0
   --hidden-dropout 0.0
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   --attention-backend flash
   --no-rope-fusion
)

mkdir -p "${RELAX_ROOT}/log" "${SAVE_DIR}" "${EXP_DIR}/outputs/relax_harness_r1"
if [ "${RUN_DIRECT:-0}" = "1" ]; then
   "${PYTHON_BIN}" -m relax.entrypoints.train \
      --resource "${RESOURCE_JSON}" \
      --max-staleness "${MAX_STALENESS:-4}" \
      --num-data-storage-units 1 \
      --num-iters-per-train-update "${NUM_ITERS_PER_TRAIN_UPDATE}" \
      --ref-actor-config "${REF_ACTOR_CONFIG_JSON}" \
      --fully-async \
      --use-health-check \
      "${MODEL_ARGS[@]}" \
      "${CKPT_ARGS[@]}" \
      "${ROLLOUT_ARGS[@]}" \
      "${OPTIMIZER_ARGS[@]}" \
      "${GRPO_ARGS[@]}" \
      "${TRACKING_ARGS[@]}" \
      "${PERF_ARGS[@]}" \
      "${SGLANG_ARGS[@]}" \
      "${MISC_ARGS[@]}" 2>&1 | tee "log/qwen35-9b-harness-r1-direct-${now}.log"
   exit ${PIPESTATUS[0]}
fi
"${RAY_BIN}" job submit ${RAY_NO_WAIT:+--no-wait} --address="http://127.0.0.1:${RAY_DASHBOARD_PORT:-8265}" \
   ${WORKING_DIR:+--working-dir "${WORKING_DIR}"} \
   --runtime-env-json="${RUNTIME_ENV_JSON}" \
   -- "${PYTHON_BIN}" -m relax.entrypoints.train \
   --resource "${RESOURCE_JSON}" \
   --max-staleness "${MAX_STALENESS:-4}" \
   --num-data-storage-units 1 \
   --num-iters-per-train-update "${NUM_ITERS_PER_TRAIN_UPDATE}" \
   --ref-actor-config "${REF_ACTOR_CONFIG_JSON}" \
   --fully-async \
   --use-health-check \
   "${MODEL_ARGS[@]}" \
   "${CKPT_ARGS[@]}" \
   "${ROLLOUT_ARGS[@]}" \
   "${OPTIMIZER_ARGS[@]}" \
   "${GRPO_ARGS[@]}" \
   "${TRACKING_ARGS[@]}" \
   "${PERF_ARGS[@]}" \
   "${SGLANG_ARGS[@]}" \
   "${MISC_ARGS[@]}" 2>&1 | tee "log/qwen35-9b-harness-r1-async-${now}.log"
