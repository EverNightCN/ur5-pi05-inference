#!/usr/bin/env bash
# On-demand user service. Installation and start are separate operations.
set -euo pipefail
DEPLOY_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SERVICE=ur5-pi05-inference.service
CHECKPOINT="$DEPLOY_ROOT/.runtime/checkpoint-current"
ACTION="${1:-status}"
if (( $# )); then shift; fi
TIMEOUT=600
while (( $# )); do
  case "$1" in
    --timeout) TIMEOUT="${2:?}"; shift 2 ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done
if [[ ! "$TIMEOUT" =~ ^[1-9][0-9]*$ ]]; then
  echo '--timeout must be a positive integer' >&2
  exit 2
fi
health() { curl --noproxy '*' -fsS --max-time 2 http://127.0.0.1:8000/healthz >/dev/null 2>&1; }
wait_health() {
  local deadline=$((SECONDS + TIMEOUT))
  local next_report=$SECONDS
  while (( SECONDS < deadline )); do
    if health; then return 0; fi
    if ! systemctl --user is-active --quiet "$SERVICE"; then
      echo "Service exited. Read $DEPLOY_ROOT/.runtime/logs/service.log" >&2
      return 1
    fi
    if (( SECONDS >= next_report )); then
      echo "Waiting for model load, compilation and warmup ($((SECONDS - deadline + TIMEOUT))s)"
      next_report=$((SECONDS + 10))
    fi
    sleep 2
  done
  echo "Service health check exceeded ${TIMEOUT}s. Read $DEPLOY_ROOT/.runtime/logs/service.log" >&2
  return 1
}
case "$ACTION" in
  start)
    for name in model.safetensors config.json norm_stats.json paligemma_tokenizer.model source_model_spec.json conversion_audit.json; do
      test -f "$CHECKPOINT/$name" || { echo "Missing checkpoint artifact: $CHECKPOINT/$name" >&2; exit 3; }
    done
    test -f "$DEPLOY_ROOT/.runtime/warmup.npz"
    systemctl --user start "$SERVICE"
    wait_health
    echo 'Ready at ws://127.0.0.1:8000'
    ;;
  stop) systemctl --user stop "$SERVICE" ;;
  restart)
    systemctl --user stop "$SERVICE"
    exec bash "$0" start --timeout "$TIMEOUT"
    ;;
  status)
    systemctl --user show "$SERVICE" --property=ActiveState,SubState,UnitFileState
    echo "Checkpoint: $CHECKPOINT"
    if health; then echo 'Health: ready'; else echo 'Health: down'; fi
    ;;
  *) echo 'Usage: bash start_inference.sh {start|stop|restart|status} [--timeout 600]' >&2; exit 2 ;;
esac
