# Findings

A running log of the team's important findings: results, surprises, and problems that affect how we run experiments or what we write in the report. Add new findings to the right section, and move items out of "Open questions" once they're answered.

**How to add a finding:** date, who found it, what we found, the evidence (numbers, run folder, command), and why it matters.

---

## Training results

### Medium training run: the model learns (2026-09-28, Andrew)

Trained on the authors' default data sizes (1,200 train / 96 val / 32 test samples, 100x100 grid, m=15, kb=4) for 10 epochs on a Mac (Apple M5 Max GPU). Config: `configs/medium.yaml`. Run folder: `work/outputs/2026-09-28/18-40-42/`.

| Epoch | 0 (before training) | 2 | 4 | 6 | 8 | 10 |
|---|---|---|---|---|---|---|
| Validation F1 | 0.058 | 0.149 | 0.164 | 0.171 | 0.196 | 0.202 |

- **Test F1: 0.196.** It's close to validation (0.202), so no sign of overfitting.
- **Speed:** about 49 seconds per epoch, 8.3 minutes total.
- **F1 improved at every check and was still rising at epoch 10,** so the model hadn't finished learning.

**Why it matters:** this confirms the training pipeline works and the model learns. The smoke test only reached F1 of about 0.12.

### 30-epoch run: still improving at epoch 30 (2026-09-28, Andrew)

Same data and settings as the medium run, with 30 epochs (`scripts/arora.sh medium train.epochs=30`). Run folder: `work/outputs/2026-09-28/19-11-43/`. About 49 seconds per epoch, 24.5 minutes total.

| Epoch | 0 | 5 | 10 | 15 | 20 | 25 | 29/30 |
|---|---|---|---|---|---|---|---|
| Train loss (lower is better) | 0.809 | 0.597 | 0.471 | 0.381 | 0.297 | 0.254 | 0.235 |
| Train F1 | 0.150 | 0.189 | 0.273 | 0.336 | 0.409 | 0.454 | 0.479 |
| Validation F1 (checked every 2 epochs) | 0.058 | 0.171 (ep 6) | 0.202 | 0.236 (ep 16) | 0.255 | 0.279 (ep 26) | **0.304** |

- **Validation F1 hadn't leveled off.** It rose at almost every check, from 0.058 to 0.304; the only pause was epochs 12-14. It's still climbing about 0.01 per check, so more epochs would help.
- **The model is starting to memorize.** The gap between train and validation F1 grew from 0.07 at epoch 10 to about 0.18 at epoch 30. That isn't a problem yet, because validation F1 is still rising. It's the expected effect of a small dataset (1,200 samples, 1% of the paper's), and more data should help more than more epochs.
- **Test F1 was only 0.219,** well below validation F1 (0.304). The test set is too narrow to trust (see "The built-in test step is a narrow check" below), so we shouldn't over-read this gap.

### Training runs are exactly reproducible (2026-09-28, Andrew)

The first 10 epochs of the 30-epoch run matched the 10-epoch run to every digit (e.g. validation F1 0.14915637... and 0.2022702...), because the seed is fixed (`train.seed=42`).

**Why it matters:** on the same machine, differences between two experiments come from the settings we changed, not from randomness. Results on a Mac and a Colab GPU still differ slightly.

### Smoke tests are pipeline checks only (2026-09-28, Andrew)

The smoke test (80 training samples, 4 epochs) gives test F1 **0.119 on the Mac** and **0.122 on a Colab T4**. The difference comes from small numerical differences between GPUs. It only checks that everything runs, so it says nothing about model quality.

---

## Reproducing the authors' results

### What the eval costs mean (2026-09-28, Andrew)

`flow=eval` prints five costs for one test case. Two of them don't depend on the model at all:

| Cost | What it is | Depends on the model? |
|---|---|---|
| `golden_cost` | The optimal tree, from GeoSteiner | No |
| `adapted_cost` | Tree built from the *correct* portals, before cleanup | No (assumes a perfect network) |
| `processed_cost` | Same, after cleanup: the best NN-Steiner can possibly do on this case | No (assumes a perfect network) |
| `predict_cost` | Tree built from the model's predicted portals, before cleanup | Yes |
| `final_cost` | The model's actual result, after cleanup | **Yes. This is the number that matters** |

The authors' results files (`exp_out/*-solved.txt`, written by `flow=nn_exp`) record the **final** cost.

### Wirelength on one test case: our model vs the authors' (2026-09-28, Andrew)

Test case `point100_10000x10000-uniform-1.txt` (100 points), threshold 0.95. Lower is better.

| Result | Wirelength | vs optimal |
|---|---|---|
| Optimal tree (GeoSteiner) | 73,838 | - |
| Best possible with a perfect network (`processed_cost`) | 75,621 | +2.4% |
| Authors' recorded result for their model (`exp_out/...m=15_kb=4-threshold=0.95-solved.txt`) | 75,641 | +2.4% |
| Our run of the authors' pretrained model | 78,541 | +6.4% |
| **Our 30-epoch medium model** (`outputs/2026-09-28/19-11-43`) | **83,077** | **+12.5%** |

- **Our medium model works, but it's clearly weaker than the pretrained one** (+12.5% vs +6.4% over optimal). That's expected: it has 1% of the paper's data and a validation F1 of 0.30.
- **We have not matched the authors' recorded number yet.** Our run of their own model gives 78,541 on this case, not 75,641. The same 78,541 came out on the Mac and on Colab, so it isn't a hardware difference. One test case is too few to judge, though; see "Open questions".

### F1 is a training signal, not the final quality measure (2026-09-28, Andrew)

After the network predicts portals, NN-Steiner hands them to an exact solver (GeoSteiner) that repairs many mistakes. On the test case above:
- The pretrained model has F1 **0.327**, and its final tree is +6.4% over optimal.
- Our medium model has F1 **0.121**, and its final tree is +12.5% over optimal (+15.6% before cleanup).

A higher F1 does mean a better tree, but the relationship isn't simple.

**Why it matters:** use F1 to track training progress, but judge models by final wirelength (Shraddha's evaluation). Also, where the logs print `acc`, the value is actually F1.

---

## Compute and hardware

### The Mac is faster than a Colab T4 for our workloads (2026-09-28, Andrew)

Same smoke test on both machines:

| Step | Mac (M5 Max) | Colab T4 |
|---|---|---|
| Training, 4 epochs | 37 s | 59 s |
| Data generation, 110 samples | about 13 s | about 100 s |

Data generation is CPU-heavy (GeoSteiner labels every sample), and the Colab VM has only 2 CPU cores.

### Full-scale training may not fit in Colab's memory (2026-09-28, Andrew)

The authors' trainer loads the **entire training set into memory** before training. The 1,200-sample training set is 194 MB, so the paper's 120,000 samples would be about **19 GB**. That fits on a 64 GB Mac but likely not on free Colab (about 12-13 GB of RAM).

Scaling the medium run's speed up to that size, one epoch would take roughly **80 minutes** on the Mac.

**Why it matters:** the full dataset size decides where training can happen. It's a team decision before Jaspreet generates the full dataset. The options are:
- train on the Mac,
- use a smaller dataset,
- pay for a high-RAM Colab machine,
- or change the data loader.

---

## Problems in the authors' code (and our fixes)

Our fixes are patch files in `third_party/patches/`, applied automatically by `scripts/setup.sh`.

- **Training required 2 or more NVIDIA GPUs** (a hard check in the trainer), so it couldn't run on Colab's single GPU or a Mac. Fixed so it runs on 1 GPU, an Apple GPU, or CPU (patch 0002).
- **A memory bug in RMST** (the C++ routine that connects the final tree) stored memory addresses in 32-bit numbers, which is unsafe on 64-bit machines. RMST is part of the main pipeline, not just a baseline. Fixed in patch 0003.
- **GeoSteiner is built without CPLEX**, the commercial solver the authors used. It still finds exact optimal trees but runs slower. The report should mention this.
- **Build problems on Colab and Mac** (a missing C++ include, Apple's incompatible `libtool`, a race in GeoSteiner's parallel build) are all handled by the setup script.

---

## Things that can silently go wrong

- **Checkpoint names are misleading.** In each run's `train/model/`, `nnArora.pt` is the **best** model by validation F1, and `nnArora_best.pt` is just the **latest** one. The authors' end-of-training test loads `nnArora_best.pt`.
- **The built-in test step is a narrow check.** The authors' end-of-training test only scores the **first batch** of the test set. With our test sets that's 32 samples of a **single quadtree shape**, compared with 96 samples across 3 shapes for validation, so `test acc` is noisy. Judge models by validation F1 and, above all, by final wirelength.
- **Dataset settings must line up with batch sizes (important for data generation).**
  - The dataset folder name doesn't include the seed, so train, val and test sets need different sizes or they overwrite each other.
  - The training batch size must equal `batch / num_trees` (samples per quadtree shape), or batches mix different tree shapes. For example, 1200 / 24 = 50.
  - Details are in the "Upstream behaviors that bite" section of `CLAUDE.md`.

---

## Open questions

- **Do we reproduce the authors' pretrained results across all 100 test cases?** On one case we got 78,541 vs their recorded 75,641. Run `flow=nn_exp` with the pretrained model on `upstream/points/point100_10000x10000-uniform-100-pt/` and compare the output against `exp_out/100_10000x10000-uniform-m=15_kb=4-threshold=0.95-solved.txt` using `evaluator/evaluateRatio.py`. (Shraddha / Andrew)
- **How big should the full training set be,** given the memory limit above? (Team, with Jaspreet)
- **Where does validation F1 level off?** It was still rising at 30 epochs (0.304). Next: a longer run, or more training data. (Andrew)
- **Does loading the latest checkpoint instead of the best one** change the authors' reported test numbers? (Andrew)
