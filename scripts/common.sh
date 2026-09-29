# Shared settings for the scripts in this folder. Sourced, not run directly.

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
UPSTREAM="$REPO/third_party/NN-Steiner"
WORK="$REPO/work"

# The authors' code, pinned so everyone runs exactly the same version.
UPSTREAM_URL="https://github.com/ABKGroup/NN-Steiner.git"
UPSTREAM_SHA="5fd75c5e437bd7651aee2740dadec1cdc6636895"

# Override with e.g. PYTHON=.venv/bin/python if python3 isn't the one you want.
PYTHON="${PYTHON:-python3}"

# Folders the authors' code writes into, relative to where it is launched (work/).
make_work_dirs() {
    mkdir -p "$WORK/data" "$WORK/points" "$WORK/models" "$WORK/outputs" "$WORK/exp_out"
    ln -sfn "$UPSTREAM" "$WORK/upstream"
}
