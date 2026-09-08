#!/usr/bin/env bash
# Run H2 + Sharpa against a GR00T N2 policy server.
#
# Usage:
#   bash run_eval_h2_groot_n2.sh "<task description>" <total_steps> [extra args...]
#
# N2 has no RTC.  This launcher uses synchronous chunk inference and the Python
# client maintains the checkpoint-requested video/state history at 30 Hz.
set -euo pipefail

if [[ $# -lt 2 ]]; then
    echo "Usage: $0 \"<task description>\" <total_steps> [extra args...]" >&2
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

exec bash "${SCRIPT_DIR}/run_eval_h2_groot.sh" "$1" "$2" \
    --policy-format n2 \
    --head-eye right \
    --swap-wrist-cams \
    --timeout-ms 60000 \
    "${@:3}"
