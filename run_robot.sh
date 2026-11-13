#!/usr/bin/env bash
# On-demand real-robot entry. `run` and `home` issue physical motion commands.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEPLOY_ROOT="${PI05_DEPLOY_ROOT:-$SCRIPT_DIR}"
ACTION="${1:-status}"
if (( $# )); then shift; fi
usage() {
  cat <<'HELP'
Usage: bash run_robot.sh {status|observe|dry-run|home|run} [rollout options]

  status    Show model service readiness.
  observe   Read robot, gripper and cameras.
  dry-run   Start inference and log policy outputs using read-only hardware connections.
  home      Move the arm to the configured HOME joint position.
  run       Wait for inference, move HOME, run the policy, then hold and close hardware.

Task settings come from the UR5 TOML [inference] section. CLI options override them.
Examples: bash run_robot.sh run --duration 60
          bash run_robot.sh run --help
Use Ctrl-C to stop the rollout and hold the current position.
HELP
}
if [[ "$ACTION" == help || "$ACTION" == -h || "$ACTION" == --help ]]; then
  usage
  exit 0
fi
if [[ "${1:-}" == -h || "${1:-}" == --help ]]; then
  exec bash "$DEPLOY_ROOT/run.sh" "$DEPLOY_ROOT/policy_rollout.py" --help
fi
case "$ACTION" in
  status)
    exec bash "$DEPLOY_ROOT/start_inference.sh" status "$@"
    ;;
  observe)
    exec bash "$DEPLOY_ROOT/run.sh" "$DEPLOY_ROOT/policy_rollout.py" --mode observe "$@"
    ;;
  dry-run)
    bash "$DEPLOY_ROOT/start_inference.sh" start
    exec bash "$DEPLOY_ROOT/run.sh" "$DEPLOY_ROOT/policy_rollout.py" --mode dry-run "$@"
    ;;
  home)
    exec bash "$DEPLOY_ROOT/run.sh" "$DEPLOY_ROOT/policy_rollout.py" --mode home --execute --i-understand "$@"
    ;;
  run)
    printf '%s\n' '[1/2] Waiting for the complete pi0.5 inference service.' >&2
    bash "$DEPLOY_ROOT/start_inference.sh" start
    printf '%s\n' '[2/2] Starting HOME and real-robot policy rollout. Ctrl-C requests hold and shutdown.' >&2
    exec bash "$DEPLOY_ROOT/run.sh" "$DEPLOY_ROOT/policy_rollout.py" --mode record \
      --home-at-start --execute --i-understand "$@"
    ;;
  *)
    usage >&2
    exit 2
    ;;
esac
