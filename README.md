# nn-steiner-rsmt

CMPE 257 course project reproducing and extending **NN-Steiner** (Kahng et al., AAAI 2024), a mixed neural-algorithmic approach to the Rectilinear Steiner Minimum Tree problem. We build on the authors' code, [ABKGroup/NN-Steiner](https://github.com/ABKGroup/NN-Steiner), which our setup script downloads at a pinned commit.

- Project plan and roles: [docs/Team_Responsibilities.md](docs/Team_Responsibilities.md)
- Proposal: [docs/Project_Proposal.md](docs/Project_Proposal.md)

## Layout

```
configs/        our run settings; each inherits the authors' conf/config.yaml and lists only what changes
scripts/        setup.sh, arora.sh (runs the authors' code with a config), and helpers
notebooks/      colab.ipynb: the Colab entry point
third_party/
  patches/      our fixes to the authors' code, applied by setup.sh
  NN-Steiner/   the authors' code (downloaded, not committed)
results/        small final tables and figures (committed)
work/           everything runs generate: data/, points/, models/, outputs/, exp_out/ (not committed)
```

## Quick start

**Colab:** open [notebooks/colab.ipynb](https://colab.research.google.com/github/Ashilman25/nn-steiner-rsmt/blob/main/notebooks/colab.ipynb), choose a GPU runtime, and run the cells in order. `work/` is linked to the shared Drive folder.

**macOS (Apple Silicon):** training and inference run on the Mac's GPU (MPS). Requires [Homebrew](https://brew.sh) and Python 3.10–3.13 (Hydra breaks on 3.14).

```bash
python3.10 -m venv .venv && source .venv/bin/activate
scripts/setup.sh              # build everything (a few minutes the first time)
scripts/smoke_test.sh         # tiny data + a few training epochs, about 1 minute
scripts/get_pretrained.sh     # optional: the authors' model, ~770 MB
scripts/arora.sh pretrained_solve
```

## Running experiments

```bash
scripts/arora.sh <config> [overrides...]
scripts/arora.sh smoke 'flow=[train]' train.epochs=10 train.lr=3e-4
```

- `<config>` is a file in `configs/` (without `.yaml`). Start a new experiment by copying `smoke.yaml`.
- Relative paths in configs resolve from `work/`. The authors' bundled test point sets are under `upstream/points/...`.
- Each run writes its logs, config, TensorBoard events, and checkpoints to `work/outputs/<date>/<time>/`. Trained models are in `train/model/`.
- Quote overrides that contain `[ ]`. Hydra can't parse `=` inside a command-line value, so don't use `=` in file names (e.g. `m15_kb4.pt`, not `m=15_kb=4.pt`).

## Our patches to the authors' code

| Patch | Why |
|---|---|
| `0001-rmst-include-cstddef` | RMST doesn't compile on newer GCC (Colab) without `<cstddef>` |
| `0002-single-device-training` | Upstream training requires 2+ NVIDIA GPUs. With the patch it also runs on 1 GPU (Colab), Apple MPS, or CPU; checkpoints load without CUDA |
| `0003-rmst-64bit-and-clang-fixes` | RMST stored pointers in 32-bit `int`s (unsafe on 64-bit machines), and a missing `<stdlib.h>` broke the clang build |
| `0004-restore-leaf-refinement` | The authors' final commit left a cleanup step (`refine_leaves`) switched off, which is one of their ablations. Turning it back on reproduces their reported results |

GeoSteiner is built without CPLEX (the authors used CPLEX). It still solves exactly, but solve times will differ from the paper.
