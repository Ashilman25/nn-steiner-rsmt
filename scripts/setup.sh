#!/usr/bin/env bash
# Get a machine (Colab or macOS) ready to run NN-Steiner:
#   1. download the authors' code at a pinned commit into third_party/NN-Steiner
#   2. apply our fixes from third_party/patches/
#   3. install system and Python dependencies
#   4. build the C++ parts (GeoSteiner and RMST)
#   5. create the work/ folders
# Safe to re-run: steps that are already done are skipped.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

LOG_DIR="$WORK/setup_logs"
mkdir -p "$LOG_DIR"

# Run a command quietly; on failure show the end of its log.
run_logged() {
    local name="$1"; shift
    if ! "$@" >"$LOG_DIR/$name.log" 2>&1; then
        echo "!! $name failed. Last lines of $LOG_DIR/$name.log:"
        tail -n 40 "$LOG_DIR/$name.log"
        exit 1
    fi
}

OS="$(uname)"
if [ "$OS" = Darwin ]; then NPROC="$(sysctl -n hw.ncpu)"; else NPROC="$(nproc)"; fi

# --- Python version check ----------------------------------------------------
PY_VERSION="$("$PYTHON" -c 'import sys; print("%d.%d" % sys.version_info[:2])')"
case "$PY_VERSION" in
    3.10|3.11|3.12|3.13) ;;
    *)
        echo "!! $PYTHON is Python $PY_VERSION. Use 3.10-3.13 (Hydra breaks on 3.14)."
        echo "   On macOS: python3.10 -m venv .venv && source .venv/bin/activate"
        exit 1
        ;;
esac
PY_EXE="$("$PYTHON" -c 'import sys; print(sys.executable)')"
echo "==> Using $PY_EXE (Python $PY_VERSION)"

# --- 1. Download the authors' code -------------------------------------------
if [ ! -d "$UPSTREAM/.git" ]; then
    echo "==> Downloading NN-Steiner @ ${UPSTREAM_SHA:0:7}"
    # Skip the 770 MB pretrained model here; scripts/get_pretrained.sh fetches it on demand.
    export GIT_LFS_SKIP_SMUDGE=1
    run_logged clone git clone --quiet "$UPSTREAM_URL" "$UPSTREAM"
    run_logged checkout git -C "$UPSTREAM" -c advice.detachedHead=false checkout "$UPSTREAM_SHA"
    run_logged submodules git -C "$UPSTREAM" submodule update --init --recursive
else
    echo "==> NN-Steiner already downloaded"
fi

# --- 2. Apply our patches ----------------------------------------------------
echo "==> Applying patches"
for patch in "$REPO"/third_party/patches/*.patch; do
    if git -C "$UPSTREAM" apply --reverse --check "$patch" 2>/dev/null; then
        echo "    already applied: $(basename "$patch")"
    else
        git -C "$UPSTREAM" apply "$patch"
        echo "    applied: $(basename "$patch")"
    fi
done

# --- 3. Dependencies ---------------------------------------------------------
echo "==> Installing system packages"
if [ "$OS" = Linux ]; then
    SUDO=""; [ "$(id -u)" -ne 0 ] && SUDO="sudo"
    run_logged apt-update $SUDO apt-get update -qq
    run_logged apt-install $SUDO apt-get install -y -qq \
        build-essential cmake libgmp-dev libtool libtool-bin autoconf automake flex bison
    BUILD_CFLAGS="-fPIC"
    BUILD_LDFLAGS=""
elif [ "$OS" = Darwin ]; then
    command -v brew >/dev/null || { echo "!! Install Homebrew first: https://brew.sh"; exit 1; }
    # libtool provides GNU "glibtool"; Apple's own libtool can't build GeoSteiner.
    for formula in gmp cmake libtool; do
        brew list --versions "$formula" >/dev/null || run_logged "brew-$formula" brew install "$formula"
    done
    BREW="$(brew --prefix)"
    BUILD_CFLAGS="-fPIC -I$BREW/include"
    BUILD_LDFLAGS="-L$BREW/lib"
    # CMake links plain "-lgmp"; tell the linker where Homebrew keeps it.
    export LIBRARY_PATH="$BREW/lib${LIBRARY_PATH:+:$LIBRARY_PATH}"
    export CPATH="$BREW/include${CPATH:+:$CPATH}"
else
    echo "!! Unsupported OS: $OS"; exit 1
fi

echo "==> Installing Python packages"
run_logged pip "$PYTHON" -m pip install -r "$UPSTREAM/requirements.txt"

# --- 4. Build GeoSteiner and RMST --------------------------------------------
module_works() {
    (cd "$UPSTREAM" && "$PYTHON" -c "$1") >/dev/null 2>&1
}

# Build a pybind11 module with CMake. CMAKE_POLICY_VERSION_MINIMUM lets CMake 4.x
# accept the authors' old cmake_minimum_required(VERSION 3.0.0).
build_module() {
    local dir="$1" name="$2"
    rm -rf "$dir/build"
    mkdir -p "$dir/build"
    run_logged "$name-cmake" cmake -S "$dir" -B "$dir/build" \
        -DCMAKE_POLICY_VERSION_MINIMUM=3.5 \
        -DPYTHON_EXECUTABLE="$PY_EXE" -DPython_EXECUTABLE="$PY_EXE"
    run_logged "$name-make" make -C "$dir/build" -j"$NPROC"
}

GEO_IMPORT="from arora.quadtree.lib import find_RSMT_GEO"
if module_works "$GEO_IMPORT"; then
    echo "==> GeoSteiner already built"
else
    GEO_SRC="$UPSTREAM/arora/quadtree/lib/geosteiner-5.3"
    if [ ! -f "$GEO_SRC/libgeosteiner.a" ] || [ ! -f "$GEO_SRC/lp_solve_2.3/libLPS.a" ]; then
        echo "==> Compiling GeoSteiner 5.3 (without CPLEX; takes a few minutes)"
        (
            cd "$GEO_SRC"
            run_logged geosteiner-configure env CFLAGS="$BUILD_CFLAGS" LDFLAGS="$BUILD_LDFLAGS" \
                ./configure --without-cplex
            # Serial on purpose: the Makefile is missing a dependency on
            # lp_solve_2.3/libLPS.a, so a parallel build (-j) fails at random.
            run_logged geosteiner-make make
        )
    fi
    echo "==> Building GeoSteiner Python module"
    build_module "$UPSTREAM/arora/quadtree/lib" geosteiner-module
fi

RMST_IMPORT="from arora.quadtree.rmst.build.rmst import find_RMST"
if module_works "$RMST_IMPORT"; then
    echo "==> RMST already built"
else
    echo "==> Building RMST Python module"
    build_module "$UPSTREAM/arora/quadtree/rmst" rmst-module
fi

# --- 5. Work folders + final check ------------------------------------------
make_work_dirs

module_works "$GEO_IMPORT" || { echo "!! GeoSteiner module does not import"; exit 1; }
module_works "$RMST_IMPORT" || { echo "!! RMST module does not import"; exit 1; }

DEVICE="$("$PYTHON" -c 'import torch; print("CUDA GPU" if torch.cuda.is_available() else "Apple GPU (MPS)" if torch.backends.mps.is_available() else "CPU only")')"
echo
echo "Setup complete. Training/inference device: $DEVICE"
echo "Next: scripts/get_pretrained.sh (optional, 770 MB) or scripts/smoke_test.sh"
