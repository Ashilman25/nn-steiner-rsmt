#!/usr/bin/env bash
# Score one trained model on the 100 uniform test problems, at a threshold picked fairly.

set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

if [ $# -lt 2 ]; then
    sed -n '2,12p' "$0" | sed 's/^# \{0,1\}//'
    exit 1
fi

MODEL="${1#work/}"; NAME="$2"; shift 2
THRESHOLDS=("$@")
[ ${#THRESHOLDS[@]} -gt 0 ] || THRESHOLDS=(0.6 0.7 0.8 0.9 0.95)
[ -f "$WORK/$MODEL" ] || [ -f "$MODEL" ] || { echo "!! No model at work/$MODEL"; exit 1; }

EXP="$UPSTREAM/exp_out"
SET200="upstream/points/point200_10000x10000-uniform-100-pt/"
SET100="upstream/points/point100_10000x10000-uniform-100-pt/"

# Solves one set at one threshold (reused if done); prints % over optimal
score() {
    local set="$1" golden="$2" out="$3" threshold="$4"
    local solved="$WORK/exp_out/$out-solved.txt"
    if [ ! -f "$solved" ] || [ "$(wc -l < "$solved")" -ne 100 ]; then
        if ! "$REPO/scripts/arora.sh" scale 'flow=[nn_exp]' nn_exp.model="$MODEL" \
                nn_exp.threshold="$threshold" nn_exp.test_set="$set" \
                nn_exp.output="exp_out/$out" > "$WORK/exp_out/$out.log" 2>&1; then
            echo "!! Solving failed; last lines of work/exp_out/$out.log:" >&2
            tail -n 5 "$WORK/exp_out/$out.log" >&2
            exit 1
        fi
    fi
    "$PYTHON" "$UPSTREAM/evaluator/evaluateRatio.py" "$solved" "$EXP/$golden" | awk '{print $NF}'
}

echo "== $NAME ($MODEL)"
echo "200-point set (picks the threshold):"
BEST_T=""; BEST=""
for t in "${THRESHOLDS[@]}"; do
    pct="$(score "$SET200" 200_10000x10000-uniform-golden.txt "thr200-$NAME-t$t" "$t")"
    echo "  threshold $t: $pct%"
    if [ -z "$BEST" ] || awk -v a="$pct" -v b="$BEST" 'BEGIN { exit !(a < b) }'; then
        BEST_T="$t"; BEST="$pct"
    fi
done

echo "picked threshold $BEST_T ($BEST% on the 200-point set)"
if [ ${#THRESHOLDS[@]} -gt 1 ] && { [ "$BEST_T" = "${THRESHOLDS[0]}" ] || [ "$BEST_T" = "${THRESHOLDS[${#THRESHOLDS[@]}-1]}" ]; }; then
    echo "note: that's the edge of the list; the best may lie beyond it, so try more thresholds"
fi
pct="$(score "$SET100" 100_10000x10000-uniform-golden.txt "thr-$NAME-t$BEST_T" "$BEST_T")"
echo "100-point set at $BEST_T: $pct%   <- the score to report"
echo "trees: work/exp_out/thr-$NAME-t$BEST_T-solved.txt"
