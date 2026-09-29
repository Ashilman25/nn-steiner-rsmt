#!/usr/bin/env bash
# Run the authors' NN-Steiner code with one of our configs.
#
#   scripts/arora.sh <config> [hydra overrides...]
#   scripts/arora.sh smoke 'flow=[train]' train.epochs=10
#
# <config> is a file in configs/ without the .yaml. Quote overrides that contain
# [ ] so the shell doesn't treat them as file patterns. Everything the run
# creates lands in work/ (Hydra logs and checkpoints in work/outputs/<date>/<time>/).
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

if [ $# -lt 1 ]; then
    sed -n '2,9p' "$0" | sed 's/^# \{0,1\}//'
    exit 1
fi
[ -d "$UPSTREAM/.git" ] || { echo "!! Run scripts/setup.sh first"; exit 1; }

CONFIG="$1"; shift
make_work_dirs

# Pick the device for inference unless one was given: CUDA > Apple MPS > CPU.
# (Training picks its device the same way inside our patched trainer.)
EXTRA=()
if ! printf '%s\n' "$@" | grep -q '^model\.device='; then
    DEVICE="$("$PYTHON" -c 'import torch; print("cuda:0" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")')"
    EXTRA+=("model.device=$DEVICE")
fi

# Let PyTorch fall back to CPU for any op Apple's GPU doesn't support.
export PYTORCH_ENABLE_MPS_FALLBACK=1

cd "$WORK"
PYTHONPATH="$UPSTREAM${PYTHONPATH:+:$PYTHONPATH}" exec "$PYTHON" -m arora \
    --config-dir "$REPO/configs" --config-name "$CONFIG" \
    ${EXTRA[@]+"${EXTRA[@]}"} "$@"
