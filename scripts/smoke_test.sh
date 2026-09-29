#!/usr/bin/env bash
# End-to-end check at tiny scale: generate temporary train/val/test data
# (skipped if it already exists), then train for a few epochs with configs/smoke.yaml.
# Extra arguments are passed to the training run, e.g. train.epochs=10.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

ARORA="$REPO/scripts/arora.sh"

# name  seed  num_trees  batch   (must match the paths in configs/smoke.yaml)
make_set() {
    local dir="$WORK/data/batch$3_100x100-uniform-$4-tensor"
    if [ -d "$dir" ] && [ -n "$(ls -A "$dir")" ]; then
        echo "==> $1 set exists: $(basename "$dir")"
    else
        echo "==> Generating $1 set: $(basename "$dir")"
        "$ARORA" smoke 'flow=[data_gen]' data_gen.seed="$2" data_gen.num_trees="$3" data_gen.batch="$4"
    fi
}

make_set train 1000 4 80
make_set val   2000 2 20
make_set test  3000 1 10

echo "==> Training"
"$ARORA" smoke 'flow=[train]' "$@"
