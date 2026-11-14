#!/usr/bin/env bash
set -euo pipefail
DEPLOY_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Site-specific paths only; API credentials remain in the private Python dotenv profile.
if [[ -f "$DEPLOY_ROOT/harness.local.env" ]]; then
  source "$DEPLOY_ROOT/harness.local.env"
fi
HARNESS_PYTHON="${HARNESSVLA_PYTHON:-$DEPLOY_ROOT/.venv-harness/bin/python}"
export UR5_REPO="${UR5_REPO:-$DEPLOY_ROOT}"
export PYTHONPATH="$DEPLOY_ROOT:$UR5_REPO:${RPENT_ROOT:-$DEPLOY_ROOT/.runtime/RPent}:${OPENPI_CLIENT_ROOT:-$DEPLOY_ROOT/vendor/openpi/packages/openpi-client/src}"
ACTION="${1:-help}"
if (( $# )); then shift; fi
case "$ACTION" in
  help|-h|--help) exec "$HARNESS_PYTHON" "$DEPLOY_ROOT/harness_vla.py" --help ;;
  explore|evaluate|review|distill|publish|freeze|summarize)
    exec "$HARNESS_PYTHON" "$DEPLOY_ROOT/harness_workflow.py" "$ACTION" "$@" ;;
  check) exec "$HARNESS_PYTHON" "$DEPLOY_ROOT/harness_vla.py" --check "$@" ;;
  run) exec "$HARNESS_PYTHON" "$DEPLOY_ROOT/harness_vla.py" --mode record --home-at-start --execute --i-understand "$@" ;;
  dry-run) exec "$HARNESS_PYTHON" "$DEPLOY_ROOT/harness_vla.py" --mode dry-run "$@" ;;
  *) echo 'Usage: bash run_harness.sh {check|dry-run|run|explore|review|distill|publish|freeze|evaluate|summarize} [options]' >&2; exit 2 ;;
esac
