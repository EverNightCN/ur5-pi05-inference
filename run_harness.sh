#!/usr/bin/env bash
set -euo pipefail
DEPLOY_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HARNESS_PYTHON="${HARNESSVLA_PYTHON:-$DEPLOY_ROOT/.venv-harness/bin/python}"
export UR5_REPO="${UR5_REPO:-$DEPLOY_ROOT}"
export PYTHONPATH="$DEPLOY_ROOT:$UR5_REPO:$DEPLOY_ROOT/.runtime/RPent:$DEPLOY_ROOT/vendor/openpi/packages/openpi-client/src"
ACTION="${1:-help}"
if (( $# )); then shift; fi
case "$ACTION" in
  help|-h|--help) exec "$HARNESS_PYTHON" "$DEPLOY_ROOT/harness_vla.py" --help ;;
  check) exec "$HARNESS_PYTHON" "$DEPLOY_ROOT/harness_vla.py" --check "$@" ;;
  run) exec "$HARNESS_PYTHON" "$DEPLOY_ROOT/harness_vla.py" --mode record --home-at-start --execute --i-understand "$@" ;;
  dry-run) exec "$HARNESS_PYTHON" "$DEPLOY_ROOT/harness_vla.py" --mode dry-run "$@" ;;
  *) echo 'Usage: bash run_harness.sh {check|dry-run|run} [options]' >&2; exit 2 ;;
esac
