#!/usr/bin/env bash
# Shared preempt-resilient loop. Sourced, not executed.
# Re-runs $RUN_CMD until it exits with code 0 (training complete) or 130 (user Ctrl-C).
# Any other exit code is treated as a preempt/OOM and triggers a restart after
# $RESTART_DELAY seconds.
#
# Exports on success: none. Caller is expected to inspect the output dir.

set -o pipefail

: "${RESTART_DELAY:=15}"
: "${MAX_RESTARTS:=100}"

restart_loop() {
  local attempt=0
  while (( attempt < MAX_RESTARTS )); do
    echo "[resume-loop] attempt=$((attempt+1)) cmd: $*"
    "$@"
    local rc=$?
    if (( rc == 0 )); then
      echo "[resume-loop] completed cleanly (rc=0)"
      return 0
    elif (( rc == 130 )); then
      echo "[resume-loop] interrupted by user (rc=130); stopping"
      return 130
    else
      attempt=$((attempt+1))
      echo "[resume-loop] exited rc=$rc; sleeping ${RESTART_DELAY}s then resuming (attempt $attempt/$MAX_RESTARTS)"
      sleep "${RESTART_DELAY}"
    fi
  done
  echo "[resume-loop] hit MAX_RESTARTS=$MAX_RESTARTS; giving up"
  return 1
}
