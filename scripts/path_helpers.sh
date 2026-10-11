#!/usr/bin/env bash

# Freeze caller-relative paths before a wrapper changes its working directory.
harness_abspath() {
  case "$1" in
    /*) printf '%s\n' "$1" ;;
    '~') printf '%s\n' "$HOME" ;;
    '~/'*) printf '%s/%s\n' "$HOME" "${1:2}" ;;
    *) printf '%s/%s\n' "$PWD" "$1" ;;
  esac
}

harness_command_path() {
  local executable
  case "$1" in
    */*|'~') set -- "$(harness_abspath "$1")" ;;
  esac
  executable="$(command -v -- "$1")" || {
    printf 'Executable not found: %s\n' "$1" >&2
    return 2
  }
  harness_abspath "$executable"
}

harness_freeze_pythonpath() {
  [[ "${PYTHONPATH+x}" == x ]] || return 0
  local remaining="$PYTHONPATH" component normalized=""
  while :; do
    component="${remaining%%:*}"
    normalized+="${normalized:+:}$(harness_abspath "${component:-.}")"
    [[ "$remaining" == *:* ]] || break
    remaining="${remaining#*:}"
  done
  export PYTHONPATH="$normalized"
}

harness_freeze_eval_paths() {
  local name executable
  INPUT="$(harness_abspath "$INPUT")"
  OUTPUT="$(harness_abspath "$OUTPUT")"
  for name in PATCH_SOURCE_ROOT PROMPT_PREFIX_REFERENCE AGENTBENCH_DIR DBBENCH_DATA_FILE; do
    if [[ -n "${!name:-}" ]]; then
      printf -v "$name" '%s' "$(harness_abspath "${!name}")"
    fi
  done
  for name in PYTHON_BIN AGENTBENCH_PYTHON "$@"; do
    if [[ -n "${!name:-}" ]]; then
      executable="$(harness_command_path "${!name}")" || return
      printf -v "$name" '%s' "$executable"
    fi
  done
  harness_freeze_pythonpath
}
