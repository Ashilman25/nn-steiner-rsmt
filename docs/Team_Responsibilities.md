# Team Responsibilities & Work Plan

**Project:** NN-Steiner: A Mixed Neural-Algorithmic Approach for the Rectilinear Steiner Minimum Tree Problem
**Team:** Jaspreet Aujla (Data), Andrew Shilman (Modeling), Shraddha Debata (Evaluation)
**Related:** [Project_Proposal.md](Project_Proposal.md)

This document turns the roles in the proposal into concrete steps. It is meant to be general enough to adapt as we learn more, but specific enough that each person knows what to do next and what they owe the others.

---

## 1. How the Pieces Fit Together

We are reproducing (and then experimenting with) the authors' code at [ABKGroup/NN-Steiner](https://github.com/ABKGroup/NN-Steiner). Almost everything runs through one entry point, `python -m arora`, with a `flow=` setting that picks the stage. Each stage corresponds roughly to one person:

```
 JASPREET (Data)                 ANDREW (Modeling)               SHRADDHA (Evaluation)
 ───────────────                 ─────────────────               ─────────────────────
 flow=data_gen  (tensor) ──────▶ flow=train ───── model .pt ───▶ flow=nn_exp   (NN-Steiner)
                                 flow=solve (single case)        flow=geo_exp  (GeoSteiner, optimal)
 flow=data_gen  (pt) / points/ ─────── test point sets ────────▶ flow=mst_exp  (RMST baseline)
                                                                 flute_out/    (FLUTE baseline)
                                                                 evaluator/evaluateRatio.py
                                                                 evaluator/evaluateSTT.py
```

- **Training data** = tensors built from synthetic point sets on a 100×100 grid, labeled using GeoSteiner's exact solutions (which portals are used).
- **The model** = small MLPs (bottom-up and top-down passes over the quadtree) that predict which portals the Steiner tree should pass through.
- **Evaluation** = run NN-Steiner on test point sets (50–5000 points, several distributions), compare total wirelength against GeoSteiner (optimal), FLUTE, and RMST, and measure runtime.

### Key settings everyone should know

| Setting | Meaning | Paper / repo default |
|---|---|---|
| `quadtree.m` | Portals per cell side (excluding corners); must be 2ⁿ − 1 | 15 |
| `quadtree.kb` | Max terminals (points) allowed in one leaf cell | 4 |
| `train.portal_weight` | Loss weight for positive portals is `(portal_weight + 1) : 1` | 15 |
| `data_gen.dist` | Point distribution: `uniform`, `normal`, `mix-normal`, `non-isotropic` | `uniform` |
| `nn_exp.threshold` | Confidence threshold for selecting a portal at inference | 0.95 |

---

## 2. Shared Setup (Everyone)

Everyone should be able to run the code themselves, not just the person who built it.

1. **Get access** to the shared Google Drive folder `FA26CMPE257Sec02MachineLearningProject/ProjectWorkUpdates` and to this GitHub repo.
2. **Set up with one command.** On Colab, open `notebooks/colab.ipynb` and run the cells in order. On a Mac, see the README (`scripts/setup.sh`). Setup downloads the authors' code, applies our fixes from `third_party/patches/`, and builds GeoSteiner and RMST. Colab wipes `/content` when a session ends, so re-run it each session. If you hit a new build problem, fix it in the script, not just in your own notebook.
3. **Run everything through `scripts/arora.sh <config>`**, with configs in `configs/`. Everything a run generates lands in `work/`. See the README.
4. **Pick hardware.** The authors' training code required 2+ NVIDIA GPUs. Our patch lets it run on one Colab GPU or on an Apple Silicon Mac's GPU. CPU works only for tiny tests. Data generation and GeoSteiner baselines run on CPU.
5. **Get the pretrained model** with `scripts/get_pretrained.sh` (770 MB; on Colab it's saved to Drive, so it downloads once). It's saved as `m15_kb4.pt`. Avoid `=` in file names, because Hydra can't parse `=` inside command-line values.
6. **Know the known deviation.** The paper's numbers use GeoSteiner compiled with CPLEX; ours is built `--without-cplex`. It still finds exact optimal trees but runs slower. Note this in the report.
7. **Keep outputs on Drive.** On Colab the notebook links `work/` to the shared Drive folder automatically. Commit code, configs, and small result tables to GitHub, but not datasets or model files.
8. **Record every run**: config overrides used, seed, runtime type (CPU/GPU), date. A shared run log (a sheet, or `docs/run_log.md`) is enough.

---

## 3. Jaspreet Aujla — Data Collection & Preprocessing

**Goal:** Produce reproducible training, validation, and test data in the format the model and evaluation code expect, and describe it well enough for the report.

### Steps

1. **Get set up and learn the data-generation flow**
   - Run the shared setup script and generate a tiny dataset with `flow=data_gen` and `data_gen.output=tensor` to confirm everything works.
   - Andrew is using his own small temporary data for early training tests, so this isn't blocking anyone. The goal is to get to the full dataset.

2. **Understand the fixed-shape batching options before scaling up**
   - The proposal notes that varying quadtree shapes break GPU batching. The repo handles this with `data_gen.level`, `data_gen.constraint`, and `data_gen.num_trees`. Read `arora/flow/dataGen.py` and `arora/points/` to confirm what these do, and write a short explanation for the team.

3. **Generate the full training data**
   - Aim to match the paper's setup: roughly 120,000 synthetic point sets on a 100×100 grid, `m=15`, `kb=4`, `fst=1` (optimal GeoSteiner labels). Start from the values in the repo's `conf/config.yaml`.
   - Produce separate train, val, and test sets with **different seeds**.
   - Budget time for this: GeoSteiner labels every sample, so generation can be slow. Consider running it in chunks and saving each chunk to Drive.
   - Tell Andrew where the sets are stored. Switching training over to them should only require changing the dataset paths in the training config.

4. **Prepare evaluation point sets (with Shraddha)**
   - The repo already includes test point sets in `points/` (100 instances each; 50–5000 points on a 10000×10000 canvas; uniform, normal, mix-normal, and non-isotropic). Confirm these load and agree with Shraddha on which ones to use.
   - Generate extra sets with `data_gen.output=pt` only if we need sizes or distributions that aren't already included.

5. **Explore and describe the data**
   - Stats: number of samples, points per sample (mean/std), quadtree depth, and especially the **fraction of portals labeled positive**. This measures the class imbalance, and Andrew needs it to tune `portal_weight`.
   - Visuals: a few example point sets with their quadtree and optimal tree (`flow=plot`).

6. **Decide on the public ISPD datasets (2018/2019)**
   - Check whether the ISPD contest nets can be turned into point files (one `x y` pair per line, integer coordinates) and scaled to our canvas.
   - Recommend to the team whether to include them as an extra real-world test set. If yes, write the conversion script. If no, write one paragraph explaining why for the report.

7. **Write the dataset card** (for the report and the repo): generation commands and parameters, seeds, sizes, storage locations, preprocessing (normalization, quadtree partitioning, padding), and known limitations.

### Deliverables
- [ ] Full train/val/test tensor datasets on Drive
- [ ] Agreed-upon evaluation point sets (with Shraddha)
- [ ] Data stats, including portal label imbalance, plus example plots
- [ ] ISPD recommendation (and converter, if used)
- [ ] Dataset section draft for the report

---

## 4. Andrew Shilman — Modeling & Training

**Goal:** Understand the NN-Steiner architecture, get a trained model we can trust, and run a small, well-documented set of experiments to improve or explain its performance.

### Steps

1. **Study the architecture**
   - Read the paper's method section alongside `arora/models/` (for example `base.py`, `dp.py`, `top.py`, `retrieve.py`, `nnArora.py`) and `arora/training/`, all under `third_party/NN-Steiner/`.
   - Write a one-page summary (with a diagram, if possible) of the bottom-up and top-down passes, what each MLP takes in and outputs, how portals are chosen at inference (threshold + GeoSteiner on the top-`k` candidates), and how the weighted BCE loss handles class imbalance. This becomes the model section of the report.

2. **Check inference with the pretrained model**
   - Run `scripts/get_pretrained.sh`, then `scripts/arora.sh pretrained_solve` (one 100-point case). Add `solve.plot=true` to get pictures.
   - Find out which reported cost corresponds to the paper's numbers. On test case #1, `flow=eval` reports a *processed* cost of 75,621, which matches the authors' recorded result (75,641; optimum 73,838), and a *final* cost of 78,541.
   - Tell Shraddha once this works, so the evaluation pipeline can be built on the pretrained model while training is still in progress.

3. **Smoke-test training**
   - Don't wait for the full dataset. `scripts/smoke_test.sh` generates a small **temporary** train/val/test set (110 samples) with the same `flow=data_gen` command Jaspreet will use at scale, then trains for 4 epochs. Switching to the real data later only means changing the dataset paths in a config.
   - Confirm that loss decreases, checkpoints save, and TensorBoard logs appear (under the run's `train/` folder). This already works on a Mac; confirm it on Colab's GPU too.
   - Note an upstream quirk: `train/model/nnArora.pt` is the **best** model by validation F1, while `nnArora_best.pt` is just the latest one, saved at the start of each validation. The authors' post-training test loads `nnArora_best.pt`.

4. **Full training run**
   - Train on the full dataset using the paper defaults (`m=15`, `kb=4`, `lr=1e-4`, `portal_weight=15`, `emb_size=16`, `hidden_size=4096`, `dropout=0.1`) with early stopping (`max_no_update`).
   - **Save checkpoints to Drive.** Colab sessions time out, so use `train.checkpoint` to resume.
   - Compare our trained model against the pretrained one on the same test set (with Shraddha). If they're close, the reproduction worked.

5. **Targeted experiments (keep the scope small)**
   Choose 2–3 questions to answer, run each as a controlled comparison, and log the results. Good options, several of which the authors also tested:
   - Max points per leaf cell: `kb` ∈ {1, 4, 7}
   - Portal density: `m` ∈ {3, 7, 15}
   - Class-imbalance weighting: `portal_weight` (including 0 = unweighted, to show why weighting matters)
   - Inference threshold: `threshold` ∈ {0.1 … 0.975}
   - Generalization: train on uniform, test on the other distributions

6. **Hand off models cleanly**
   - For each model, save the `.pt` file, the exact config and overrides, the training curves, and a one-line description, all named consistently (for example `m15_kb4_pw15_seed42.pt`, with no `=`).

7. **Write** the methodology/model section and the training details (hardware, training time, hyperparameters, early-stopping behavior).

### Deliverables
- [ ] Architecture summary and diagram
- [x] Pretrained model running with `scripts/arora.sh pretrained_solve` (unblocks Shraddha) (needs push + tell Shraddha)
- [x] Training smoke test passing on temporary data (Mac ✅, Colab pending)
- [ ] Our trained `m=15, kb=4` model and its training curves
- [ ] 2–3 experiment comparisons, with models and configs
- [ ] Model and training section draft for the report

---

## 5. Shraddha Debata — Evaluation & Analysis

**Goal:** Build a repeatable evaluation pipeline, compare NN-Steiner to the baselines on wirelength, validity, runtime, and scalability, and produce the figures and tables for the report.

### Steps

1. **Understand the metrics and tools**
   - `evaluator/evaluateRatio.py <comp.txt> <base.txt>` prints the **average % wirelength error** of one method relative to another. Each file has one total length per line, one line per test instance.
   - `evaluator/evaluateSTT.py <points.txt> <result.txt>` checks that a produced tree is a **valid** Steiner tree (connected and covering all terminals).
   - Read `exp/nn.sh`, `exp/geo.sh`, `exp/mst.sh`, `exp/snapped.sh`, `exp/ratio_flute.sh`, and `exp/ratio_rest.sh` (under `third_party/NN-Steiner/`). These are the authors' own evaluation loops and are good templates. Our versions should call `scripts/arora.sh` instead of `python -m arora`.

2. **Set up the baselines**
   - **GeoSteiner (optimal):** `flow=geo_exp` on the test point sets. It gets slow for large inputs (the authors' `exp/geo.sh` stops at 1000 points), so decide on a size cutoff and note it.
   - **RMST:** `flow=mst_exp`, a simple lower-quality baseline.
   - **FLUTE and REST:** precomputed results are included in `flute_out/` and `REST_out/`, so they can be used as-is (and we can say so in the report).

3. **Build the pipeline using the pretrained model**
   - Run `flow=nn_exp` with the pretrained `m=15_kb=4` model across the test sizes (50 → 5000) and distributions. Script this like `exp/nn.sh` so it can be re-run with any model.
   - Compare the results against the paper's reported numbers and the included `exp_out/` files. This checks that *our evaluation setup* is correct before we evaluate *our model*.

4. **Evaluate the team's trained models** as soon as Andrew hands them off, using the same scripts and test sets.

5. **Check validity** by running `evaluateSTT.py` on a sample of outputs for each size and distribution, and report the percentage that are valid.

6. **Measure runtime**
   - Record wall-clock time per instance for NN-Steiner, GeoSteiner, and RMST on the same hardware, and note CPU vs. GPU.
   - Show how runtime grows with the number of points (scalability).

7. **Build the results tables and figures**
   - Table: % wirelength error vs. GeoSteiner and vs. FLUTE, by number of points and by distribution.
   - Plot: error vs. number of points, one line per method.
   - Plot: runtime vs. number of points (probably a log scale).
   - Plots or tables for Andrew's experiments (for example error vs. `kb`, `m`, threshold).
   - Two or three visual examples of NN-Steiner trees next to the optimal ones (`flow=solve plot=true` or `flow=plot`).

8. **Analyze, not just report.** Where does NN-Steiner do well or poorly? How does it behave as the number of points grows, and on distributions it wasn't trained on? How does our reproduction compare with the paper, and why might it differ (for example, GeoSteiner built without CPLEX)?

### Deliverables
- [ ] Re-runnable evaluation scripts (one command per experiment)
- [ ] Baseline results: GeoSteiner, RMST, FLUTE/REST
- [ ] Pretrained-model results compared with the paper's numbers
- [ ] Results for our trained model(s)
- [ ] Validity and runtime results
- [ ] Final figures and tables
- [ ] Results and analysis section draft for the report

---

## 6. Everyone — Report & Presentation

| Report section | Lead | Contributors |
|---|---|---|
| Problem statement & motivation | All (adapted from proposal) | — |
| Dataset & preprocessing | Jaspreet | Andrew |
| Methodology & model | Andrew | Jaspreet |
| Experimental setup | Shraddha | Andrew |
| Results & analysis | Shraddha | All |
| Strengths, limitations, future work | All | — |
| References & AI tool usage | All | — |

Steps:
1. Each lead drafts their section as soon as their deliverables are done, without waiting until the end.
2. One person merges the drafts into a single document and makes the terminology consistent (portal, leaf cell, `m`, `kb`, and so on).
3. Everyone reviews the full report for clarity, correct numbers, and citations.
4. Update the **AI Tool Usage** section to list every AI tool used during the project, not just for the proposal.
5. Slides: each person presents their own section, plus a shared intro and conclusion. Do at least one timed practice run together.

---

## 7. Handoffs & Dependencies

| From → To | What | Why it matters |
|---|---|---|
| Andrew → Shraddha | Confirmation that the pretrained model runs | Lets evaluation be built in parallel with training |
| Jaspreet → Shraddha | Agreed test point sets | Every method must be evaluated on the same inputs |
| Jaspreet → Andrew | Full datasets + portal imbalance stats | Needed for the real training run and for tuning `portal_weight` |
| Andrew → Shraddha | Trained models + configs | The main results |
| Shraddha → Andrew | Early error results | Tells us which experiments are worth running |
| All → Report | Section drafts | Final deliverable |

---

## 8. Suggested Phases

Fill in target dates to match the course deadlines.

| Phase | Focus | Exit criteria | Target date |
|---|---|---|---|
| 1. Setup & smoke test | Everyone can run the code; temporary small dataset; pretrained model runs; training runs a few epochs | Each person has run their own stage end-to-end at small scale | |
| 2. Full data & baselines | Full datasets; GeoSteiner/RMST/FLUTE baselines; pretrained-model evaluation | Evaluation numbers for the pretrained model roughly match the paper | |
| 3. Training & experiments | Our trained model; 2–3 targeted experiments | Results tables filled in for our models | |
| 4. Report & presentation | Section drafts, figures, review, slides | Report submitted; practice run done | |

---

## 9. Open Decisions (to settle as a team)

- **ISPD datasets:** include them as a real-world test set, or stay synthetic-only? (Jaspreet recommends.)
- **Compute:** is free Colab enough for full training and data generation, or do we need Colab Pro or another GPU source?
- **Training strategy:** train from scratch to reproduce the paper, fine-tune from the pretrained model, or both?
- **Experiment scope:** which 2–3 experiments do we commit to? (Andrew proposes; the team agrees.)
- **GeoSteiner size cutoff:** the largest input size at which we compute exact optimal baselines. (Shraddha proposes.)
