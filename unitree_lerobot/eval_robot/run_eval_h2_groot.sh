#!/usr/bin/env bash
# Run H2 + Sharpa GR00T policy inference.
#
# Usage:
#   bash run_eval_h2_groot.sh "<task description>" <total_steps> [extra args...]
#
# Examples:
#   bash run_eval_h2_groot.sh "pick up the apple" 60
#   bash run_eval_h2_groot.sh "pick up the apple" 60 --head-eye right
#   bash run_eval_h2_groot.sh "pick up the apple" 60 --policy-port 5556
#   # Skip the init-posture step entirely (start each episode from arm-home zeros):
#   bash run_eval_h2_groot.sh "pick up the apple" 60 --init-state-yaml ""
#
# Prereqs:
#   - sharpa_dds_bridge --side both running on Thor
#   - GR00T policy server reachable at --policy-host:--policy-port (default localhost:5555)
#   - tv conda env active, native DDS camera publisher running on Thor domain 10
#   - $INIT_STATE_YAML exists (defaults to ~/init_state.yaml). Generate with:
#       bash ~/binliu/xr_teleoperate/teleop/utils/run_compute_average_initial_state.sh \
#            <dataset_dir> -- --output ~/init_state.yaml
#
# Override any default by passing the matching flag as an extra argument:
#   --policy-host, --policy-port, --head-eye, --hold-s, --action-horizon,
#   --init-state-yaml, --init-state-hold-s, --frequency, --dds-domain,
#   --camera-dds-domain, --camera-network-interface, ...
set -euo pipefail

if [[ $# -lt 2 ]]; then
    echo "Usage: $0 \"<task description>\" <total_steps> [extra args...]" >&2
    exit 1
fi

TASK="$1"
STEPS="$2"
shift 2

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Init posture: averaged from recorded training data. Override with the
# INIT_STATE_YAML env var, or by passing --init-state-yaml <path> as an extra arg.
INIT_STATE_YAML="${INIT_STATE_YAML:-$HOME/init_post_0512.yaml}"
if [[ ! -f "$INIT_STATE_YAML" ]]; then
    echo "WARN: init-state YAML not found at $INIT_STATE_YAML — robot will start from arm zero pose." >&2
    echo "      Generate one with run_compute_average_initial_state.sh, or override via" >&2
    echo "      INIT_STATE_YAML=/path/to/file.yaml bash $0 ..." >&2
    INIT_STATE_ARGS=()
else
	INIT_STATE_ARGS=(--init-state-yaml "$INIT_STATE_YAML" --init-state-hold-s 10) 
fi

python "${SCRIPT_DIR}/eval_h2_groot.py" \
    --policy-host localhost \
    --policy-port 5555 \
    --task "${TASK}" \
    --steps "${STEPS}" \
    --motion \
    --head-eye left \
    --hands-source dds \
    --image-source dds \
    --camera-dds-domain 10 \
    --hold-s 20 \
    --action-horizon 15 \
    "${INIT_STATE_ARGS[@]}" \
    "$@"
