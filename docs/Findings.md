# Findings

The team's running record of the project: where things stand, what each of us needs to know, every important finding (results, surprises, and problems that affect how we run experiments or what we write in the report), and how to reproduce it all. Most entries so far are mine (Andrew, Modeling).

**New here?** Read [Where things stand](#where-things-stand) and [What you need to know or do](#what-you-need-to-know-or-do) first.

**How to add a finding:** give the date, who found it, what we found, the evidence (numbers, run folder, command), and why it matters. Put it in the right section. Move items out of "Open questions" once they're answered, and update "Where things stand" if the big picture changes.

- Background on the paper: [Project_Proposal.md](Project_Proposal.md)
- Who owns what: [Team_Responsibilities.md](Team_Responsibilities.md)

*Last updated: 2026-10-09 (Andrew)*

---

## Contents

- [Where things stand](#where-things-stand)
- [What you need to know or do](#what-you-need-to-know-or-do)
- [Timeline](#timeline)
- [All results in one place](#all-results-in-one-place)
- [What the results mean](#what-the-results-mean)
- [Reproducing the authors' results](#reproducing-the-authors-results)
- [Training results](#training-results)
- [Data](#data)
- [How we measure models](#how-we-measure-models)
- [Compute and hardware](#compute-and-hardware)
- [Problems in the authors' code (our patches)](#problems-in-the-authors-code-our-patches)
- [Things that can silently go wrong](#things-that-can-silently-go-wrong)
- [The data-scaling study (done)](#the-data-scaling-study-done)
- [What comes next](#what-comes-next)
- [Our scripts, and what each one does](#our-scripts-and-what-each-one-does)
- [How to reproduce our numbers](#how-to-reproduce-our-numbers)
- [Where everything is](#where-everything-is)
- [Open questions](#open-questions)
- [Words we use](#words-we-use)

---

## Where things stand

| Question | Answer |
|---|---|
| Can everyone run the authors' code? | **Yes.** It takes one setup command on a Mac or on Colab. Their training needed 2+ NVIDIA GPUs; it now runs on 1 GPU, an Apple GPU, or CPU |
| Is the authors' trained model as good as the paper says? | **Yes.** On the 100 uniform 100-point test problems it scores **2.648%** longer than optimal, vs their published **2.646%**. Getting there needed a fix (patch 0004) |
| Can we train a model that good ourselves? | **Not yet.** Trained on 1,200 examples: **3.77%**. On 12,000: **3.38%**. On 36,000: **3.38%** again. The authors' model (120,000 examples): **2.65%** |
| Is more training data the fix? | **No, not anymore.** 1,200 → 12,000 helped; 12,000 → 36,000 changed nothing (a tie, each model wins 50 of 100 problems). So we're skipping the full 120,000 |
| What's running right now? | Nothing. The data-scaling study is done |
| What's next? | Find what else differs from the authors' training. Hand our best model to Shraddha. Pick 1–2 more experiments, trained on 12,000 examples (same score as 36,000, a third of the time) |

"% longer than optimal" is the real score: how much more wire our tree uses than the perfect tree from GeoSteiner. Lower is better. 3.38% means: where the perfect tree needs 100 m of wire, ours uses 103.38 m.

---

## What you need to know or do

### Everyone

- **Pull `main` and re-run `scripts/setup.sh`.** Two patches landed on Sep 29 (0004 and 0005). The script is safe to re-run: it only applies patches that aren't applied yet. On Colab the notebook clones fresh every session, so it picks them up on its own.
- **Judge models by "% longer than optimal", not by F1.** F1 tracks training; it isn't the result. See [What the results mean](#what-the-results-mean).

### Shraddha

- **Any NN-Steiner numbers made before Sep 29 measured the wrong setup.** The authors' final code has a cleanup step (`refine_leaves`) switched off. With it off, their model scores 6.87%, not 2.65%. Patch 0004 switches it back on. Please re-run anything from before then.
- **Known-good checks for your pipeline:**
  - `scripts/arora.sh pretrained_solve` should print `final_cost: 75641.0`.
  - The authors' model on `upstream/points/point100_10000x10000-uniform-100-pt/` should score **2.648%** against the authors' answer file, `third_party/NN-Steiner/exp_out/100_10000x10000-uniform-golden.txt`. 96 of the 100 lengths match the authors' own result file exactly.
- **Our own trained model is ready** (36,000 examples, 3.38%). I'll share the `.pt` file (about 830 MB), its exact settings and its training curves. It's still 0.74 points behind the authors' model on 100-point problems, so for "does NN-Steiner work" questions, test the authors' model too.
- **So far I've only tested 100-point uniform problems.** The paper's main claim is that NN-Steiner beats FLUTE at 500+ points. We haven't tested that yet; it's part of your evaluation.

---

## Timeline

**Sep 21 (team):** submitted the [proposal](Project_Proposal.md).

**Sep 28 (Andrew): got the authors' code running.**
- Built our repo as a wrapper around their code. `scripts/setup.sh` downloads it at a fixed commit, applies our patches and builds everything. `scripts/arora.sh` runs it with our configs. `notebooks/colab.ipynb` does the same on Colab. See [Our scripts](#our-scripts-and-what-each-one-does).
- Fixed what didn't work (patches 0001–0003): their training needed 2+ NVIDIA GPUs, and RMST didn't build and had a 64-bit memory bug.
- Smoke test passed on the Mac and on a Colab T4. The Mac turned out faster, so I train locally.
- First real training runs (1,200 examples, 10 and 30 epochs): the model learns.
- One test problem with the authors' model didn't match their recorded result (78,541 vs 75,641).
- Wrote this file and updated `Team_Responsibilities.md`.

**Sep 29 (Andrew): matched the paper, and set up the scaling study.**
- Found why we didn't match: the authors left the cleanup step switched off. Patch 0004 fixes it, and their model now scores 2.648% vs their 2.646%.
- Found that the cleanup step does a lot of the work, and scored my 30-epoch model properly (3.74%).
- Found a validation leak in the default seeds, and made clean validation and test sets.
- Measured how precise a score is (±0.1 points).
- Designed the data-scaling study (`configs/scale.yaml`) and generated all its data. Data generation froze on a rare example; patch 0005 fixes that.
- Trained and scored the 1,200 model (3.77%), then started the 12,000 run.

**Sep 30 (Andrew):** the 12,000 model finished and scored 3.38%. It needed many more learning steps to peak, so I raised the 36,000 run's epoch cap from 30 to 60.

**Oct 8 (Andrew):** started the 36,000 run.

**Oct 9 (Andrew):** the 36,000 run used all 60 epochs (22 h 23 min) and scored 3.38%, a tie with 12,000. More data stopped helping, so the scaling study is done and we're not making 120,000. Then I scored each run's last checkpoint too: it's 0.14–0.31 points worse than the best one every time, so the checkpoint you score matters.

---

## All results in one place

All scores are on the same exam: the authors' 100 test problems (100 uniformly random points each, on a 10,000 × 10,000 canvas), with threshold 0.95, compared with the perfect GeoSteiner trees. Lower is better.

| Method | Training examples | Cleanup off | **Cleanup on** (the paper's method) |
|---|---|---|---|
| FLUTE (fast standard method, authors' result file) | – | – | 1.25% |
| Authors' model, their published result file | 120,000 | 6.87% | 2.65% (2.646) |
| Authors' model, run by me | 120,000 | 6.87% | **2.65%** (2.648) |
| My 30-epoch model (old leaky validation) | 1,200 | 10.63% | 3.74% |
| **Scaling study** | 1,200 | – | **3.77%** |
| **Scaling study** | 12,000 | – | **3.38%** |
| **Scaling study** | 36,000 | – | **3.38%** |

FLUTE beating NN-Steiner here is expected. The paper's claim is that NN-Steiner wins on **bigger** problems (500+ points), which we haven't tested yet.

**The scaling study in detail:**

| Training examples | Run folder | Best val F1 | Epoch of best | Stopped at | Train time | % longer than optimal | vs the authors' model |
|---|---|---|---|---|---|---|---|
| 1,200 | `2026-09-29/16-01-59` | 0.226 | 32 | 42 (early stop) | 33.6 min | **3.77** | 1.12 ± 0.11 points behind; they win 88 of 100 |
| 12,000 | `2026-09-29/17-07-18` | 0.427 | 44 | 54 (early stop) | 7 h 8 min | **3.38** | 0.74 ± 0.10 points behind; they win 76 of 100 |
| 36,000 | `2026-10-08/11-43-16` | 0.459 | 56 | 60 (hit the cap) | 22 h 23 min | **3.38** | 0.74 ± 0.12 points behind; they win 74 of 100 |
| Authors' model | – | – | – | – | – | **2.65** | – |

**The average hides a range.** Per problem, the 1,200 model is 1.80% to 6.20% longer than optimal, the 12,000 model 1.25% to 5.77%, the 36,000 model 1.33% to 7.20%, and the authors' model 1.03% to 5.22%. None hit the optimum exactly.

---

## What the results mean

1. **The paper's results are real, and our setup measures them correctly.** The authors' model matches their published number (2.648% vs 2.646%), with 96 of 100 problems giving identical lengths. So our build, our patches and our scoring all behave like theirs.

2. **More training data helped once, then stopped helping.** Going from 1,200 to 12,000 examples (10×) improved the score by 0.38 points and closed about a third of the gap to the authors. Going from 12,000 to 36,000 (3×) changed nothing: −0.00 ± 0.11 points, and each model wins exactly 50 of the 100 problems. So in our setup the score levels off at about 3.38%, still 0.74 points behind the authors. The remaining gap comes from something other than the amount of data, and generating 120,000 examples isn't worth days of compute.

3. **Bigger datasets need longer training.** The number of learning steps needed to peak went from 768 (1,200 examples) to 10,560 (12,000) to at least 40,320 (36,000). If the epoch cap doesn't grow with the data, runs get cut off early and the bigger dataset looks worse than it is. The 36,000 run did hit its 60-epoch cap. A longer run might have found a checkpoint a few tenths better (point 6), but not enough to close the 0.74-point gap.

4. **The cleanup step does a lot of the work.** Without it, the authors' model goes from 2.65% to 6.87%. NN-Steiner is a team effort: the network does the big-picture routing, and the exact solver fixes small local pieces.

5. **F1 is a training signal, not the result.** Validation F1 nearly doubled from 1,200 to 12,000 examples (0.226 → 0.427), but the score improved much less (3.77% → 3.38%). Better F1 does mean shorter trees, but not in proportion, because the cleanup repairs many wrong guesses. From 12,000 to 36,000 examples, validation F1 rose again (best 0.427 → 0.459; averaged over each run's last 6 checks, 0.403 → 0.435), and the score didn't move at all. Within a run, though, the checkpoint with the best validation F1 did beat the last checkpoint every time. So F1 is a rough guide: good for picking a checkpoint, not for predicting the score. Only "% longer than optimal" tells us how good a model is.

6. **How sure we can be.** Each score is accurate to about ±0.1 points, but that only covers the luck of the 100 test problems. **Which checkpoint you score adds more:** each run's best and last checkpoints differ by 0.14–0.31 points ([details](#which-checkpoint-you-score-moves-the-score-by-up-to-03-points-2026-10-09-andrew)). So a difference under about 0.3 points between two training runs isn't trustworthy from one checkpoint each. All three gaps to the authors (1.12, 0.74 and 0.74) are well above that, and they win most problems, so the gaps are real. Two comparisons are ties: 1,200 vs the old 30-epoch model (+0.03), and 36,000 vs 12,000 (−0.00, 50 wins each).

7. **What these results don't cover yet:**
   - Bigger problems (500–5,000 points), where the paper's main claim lives.
   - Other point patterns (normal, mix-normal, non-isotropic).
   - Runtime. Ours isn't comparable to the paper's anyway: different hardware, and GeoSteiner without CPLEX.

---

## Reproducing the authors' results

### The authors' model matches the paper once a switched-off cleanup step is restored (2026-09-29, Andrew)

- I ran the authors' pretrained model on all 100 uniform 100-point test problems (`upstream/points/point100_10000x10000-uniform-100-pt/`). It scored **6.87%**, far from the paper's 2.65%.
- **The cause:** the authors' last commit left one line commented out, the call to `refine_leaves` in `SteinerTree.post_process`. That's their "tree" ablation from the paper, not their actual method. Their own ablation result file (`exp_out/...threshold=0.95-tree-solved.txt`) also scores exactly 6.87%.
- **Patch 0004 puts the line back.** Their model then scores **2.648% vs their published 2.646%**. 96 of 100 problems give identical lengths; 2 are slightly shorter and 2 slightly longer.
- Evidence: `work/exp_out/check-refine-on-100-uniform-solved.txt` and `check-refine-off-...`, compared with `third_party/NN-Steiner/exp_out/100_10000x10000-uniform-m=15_kb=4-threshold=0.95-solved.txt` using `evaluator/evaluateRatio.py`.

**Why it matters:** the first half of the reproduction is confirmed: the authors' model is as good as the paper says, and our build and scoring give the same trees as theirs. Anything scored before the patch measured the wrong setup.

### The cleanup step does a lot of the work (2026-09-29, Andrew)

| Model | Cleanup off | Cleanup on |
|---|---|---|
| Authors' model | 6.87% | 2.65% |
| My 30-epoch model (1,200 examples) | 10.63% | 3.74% |

**Why it matters:** a big share of NN-Steiner's quality comes from the exact-solver cleanup, not only from the network. That's good report material.

### One test case, before and after the fix (2026-09-28, updated 2026-09-29, Andrew)

Test case `point100_10000x10000-uniform-1.txt` (100 points), threshold 0.95. Lower is better.

| Result | Wirelength | vs optimal |
|---|---|---|
| Optimal tree (GeoSteiner) | 73,838 | – |
| Best possible with a perfect network (`processed_cost`) | 75,621 | +2.4% |
| Authors' recorded result for their model | 75,641 | +2.4% |
| Authors' model, run by me, cleanup off (before patch 0004) | 78,541 | +6.4% |
| **Authors' model, run by me, cleanup on** | **75,641** | **+2.4%** (matches their record) |
| My 30-epoch model, cleanup off | 83,077 | +12.5% |
| **My 30-epoch model, cleanup on** | **77,662** | **+5.2%** |

On Sep 28 our run of their model gave 78,541 on both the Mac and Colab, so it wasn't a hardware difference. The cause turned out to be the switched-off cleanup step above.

---

## Training results

### Data-scaling study, 1,200 examples: 3.77% (2026-09-29, Andrew)

Run folder `work/outputs/2026-09-29/16-01-59`, config `scale`, `train.epochs=200` (a cap; early stopping decides).

- Validation F1 reached about 0.22 by epoch 16–20 and stayed flat. The best was **0.226 at epoch 32**.
- Early stopping ended the run at epoch 42 (33.6 min, about 49 s per epoch).
- Training F1 kept climbing (0.41 → 0.53 → 0.62), so the extra epochs were pure memorizing.
- **Score: 3.767%.** That's the same as my old 30-epoch model (3.739%): the difference is +0.03 ± 0.09 points, a tie (the new one wins 56 problems, loses 43).
- vs the authors' model: 1.12 ± 0.11 points behind; they win 88 of 100.

**Why it matters:** the sanity check passes. The new setup (clean seeds, new validation set, early stopping) reproduces the old result. The leak made the old validation F1 look better, but it didn't change the final quality. At 1,200 examples the ceiling is about 3.75%.

### Data-scaling study, 12,000 examples: 3.38% (2026-09-30, Andrew)

Run folder `work/outputs/2026-09-29/17-07-18`, config `scale`, `train.epochs=60`.

- Early stopping ended it at epoch 54 after 7 h 8 min (7.9 min per epoch). The best validation F1 was **0.427 at epoch 44**.
- **Score: 3.385%,** a real improvement:
  - vs the 1,200 model: 0.38 ± 0.10 points better, with shorter trees on 66 of 100 problems.
  - vs the authors' model: still 0.74 ± 0.10 points behind (they win 76 of 100).
  - The gap to the authors went from 1.12 to 0.74 points, so about a third of it closed.

**Why it matters:** more data helps, and it isn't done helping.

### Bigger datasets need more training steps to peak (2026-09-30, Andrew)

- One epoch is 24 learning steps on 1,200 examples, 240 on 12,000 and 720 on 36,000 (batches of 50).
- The 1,200 model peaked after **768 steps** (epoch 32). The 12,000 model needed **10,560 steps** (epoch 44).
- The 36,000 run was planned with a 30-epoch cap (21,600 steps). If it needs a few times more steps again, that cap could cut it off while it's still improving. **So I raised the cap to 60 epochs.**
- Update (Oct 9): the 36,000 run's best came at **40,320 steps** (epoch 56), and it reached the 60-epoch cap before early stopping could end it. So even 60 was a little short. A longer run might find a checkpoint a few tenths better, but not enough to close the gap to the authors.

**Why it matters:** if the cap doesn't grow with the data, a bigger dataset can look worse than it is. If a run hits its cap while validation F1 is still rising, mark it "hit the cap", not finished.

### Data-scaling study, 36,000 examples: 3.38%, no better than 12,000 (2026-10-09, Andrew)

Run folder `work/outputs/2026-10-08/11-43-16`, config `scale`, `train.epochs=60`.

- It ran all 60 epochs (22 h 23 min, 22.2 min per epoch). **It hit the cap:** the best validation F1 was **0.459 at epoch 56**, and early stopping needs 10 epochs with no new best.
- Validation F1 was ahead of the 12,000 run at almost every check, and its best is higher (0.459 vs 0.427). That isn't one lucky spike: averaged over each run's last 6 checks it's 0.435 vs 0.403.
- **Score: 3.384%,** the same as the 12,000 model (3.385%):
  - vs the 12,000 model: −0.00 ± 0.11 points; each wins exactly 50 of 100 problems. The trees aren't identical (they differ by 0.85 points per problem on average), but the wins and losses cancel out.
  - vs the 1,200 model: 0.38 ± 0.11 points better, shorter on 62 of 100.
  - vs the authors' model: 0.74 ± 0.12 points behind (they win 74 of 100), the same gap as at 12,000.
- The gap between training F1 and validation F1 didn't shrink with more data: 0.31, 0.34 and 0.37 at each run's best epoch (training F1 reached 0.83). The network has 207 million weights, enough to memorize even 36,000 examples.

**Why it matters:** in our setup, more data stops helping by 12,000 examples. The remaining 0.74-point gap to the authors comes from something else, so we're not generating 120,000. A practical bonus: later experiments can train on 12,000 examples (about 7 h) instead of 36,000 (about 22 h) with no loss in score.

**Caveat:** because the run hit its cap, it may not have been fully trained. Its last checkpoint (epoch 60) scores 3.70%, 0.31 points worse than the best one, so the score moves a lot between checkpoints ([details](#which-checkpoint-you-score-moves-the-score-by-up-to-03-points-2026-10-09-andrew)). A longer run might find a checkpoint a few tenths better. That still wouldn't close the 0.74-point gap.

### First real training runs: the model learns (2026-09-28, Andrew)

Trained on the authors' default data sizes (1,200 train / 96 val / 32 test examples, 100×100 grid, `m=15`, `kb=4`) on my Mac's GPU. Config: `configs/medium.yaml`.

**10 epochs** (`work/outputs/2026-09-28/18-40-42/`, about 49 s per epoch, 8.3 min total):

| Epoch | 0 (before training) | 2 | 4 | 6 | 8 | 10 |
|---|---|---|---|---|---|---|
| Validation F1 | 0.058 | 0.149 | 0.164 | 0.171 | 0.196 | 0.202 |

**30 epochs** (`scripts/arora.sh medium train.epochs=30`, `work/outputs/2026-09-28/19-11-43/`, 24.5 min total):

| Epoch | 0 | 5 | 10 | 15 | 20 | 25 | 29/30 |
|---|---|---|---|---|---|---|---|
| Train loss (lower is better) | 0.809 | 0.597 | 0.471 | 0.381 | 0.297 | 0.254 | 0.235 |
| Train F1 | 0.150 | 0.189 | 0.273 | 0.336 | 0.409 | 0.454 | 0.479 |
| Validation F1 (checked every 2 epochs) | 0.058 | 0.171 (ep 6) | 0.202 | 0.236 (ep 16) | 0.255 | 0.279 (ep 26) | **0.304** |

- Validation F1 rose at almost every check and was still climbing at epoch 30.
- The model was starting to memorize: the gap between train and validation F1 grew from 0.07 at epoch 10 to about 0.18 at epoch 30. That's the expected effect of a small dataset (1% of the paper's).
- **Later corrections:** this validation set was leaky, so its F1 was inflated (see [the leak finding](#the-default-seeds-leaked-validation-data-into-training-2026-09-29-andrew)). Scored properly with cleanup on, the 30-epoch model is at **3.74%**.

**Why it matters:** the training pipeline works and the model learns. The smoke test only reached F1 of about 0.12.

### Training runs are exactly reproducible (2026-09-28, Andrew)

The first 10 epochs of the 30-epoch run matched the 10-epoch run to every digit (e.g. validation F1 0.14915637... and 0.2022702...), because the seed is fixed (`train.seed=42`).

**Why it matters:** on the same machine, differences between two experiments come from the settings we changed, not from randomness. Results on a Mac and a Colab GPU still differ slightly.

### Smoke tests are pipeline checks only (2026-09-28, Andrew)

The smoke test (80 training examples, 4 epochs) gives test F1 **0.119 on my Mac** and **0.122 on a Colab T4**. The difference comes from GPUs rounding slightly differently. It only checks that everything runs, so it says nothing about model quality.

---

## Data

### The default seeds leaked validation data into training (2026-09-29, Andrew)

- Each example's random points come from `seed + example number`.
- The authors' default training seed is 4200, so 1,200 training examples use seeds 4200–5399. That range covers all 96 validation seeds (5200–5295).
- So 96 training examples were near-copies of the validation examples (94.7% the same points).
- **The fix:** training seeds now start at 100000, far from validation (5200) and test (6200). I also made new, bigger validation (128 examples, 4 tree shapes) and test (64, 2 shapes) sets. All of this is in `configs/scale.yaml`.

**Why it matters:** validation F1 numbers from before Sep 29 look better than they should. The final scores weren't affected (see the 1,200 entry above).

### GeoSteiner can get stuck forever on rare examples (2026-09-29, Andrew)

- Making the 12,000 set, data generation froze at 11,908 of 12,000. Example #6588 never finished: GeoSteiner's LP solver hit a "divide by zero" warning, and even on its own it ran for 25+ minutes. (We build GeoSteiner without CPLEX, the commercial solver the authors used.)
- **Patch 0005 fixes it.** If nothing finishes for 2 minutes, the stuck examples are stopped and redone with new random points of the same tree shape (seed + attempt × 10,000,000), and a `GeoSteiner stalled on samples [...]` warning is logged.
- With the patch, the 12,000 set finished in 7.5 minutes. Only #6588 was replaced; the other 11,908 files are byte-for-byte identical to before.
- The 36,000 set took 18.5 minutes, and 3 examples were redone automatically (#6588, #33426, #35055).

**Why it matters:** about 1 in 12,000 examples does this. It's now handled automatically, so the warning is normal.

### The scaling-study training sets are nested (2026-09-29, Andrew)

The first 12,000 examples of the 36,000 set are byte-for-byte the 12,000 set (including the replaced #6588), and the first 1,200 are the 1,200 set.

**Why it matters:** each bigger set is "the same data plus more", so the amount of data is the only difference between the scaling runs.

### Dataset statistics (Andrew)

From the 1,200 training set, for the dataset card:
- 180 points are scattered per example, and some are dropped to fit the quadtree shape, leaving **about 165**.
- **157–185 quadtree cells** and **2,652–3,128 portals** per example.
- **Only 3.1% of portals are labeled "used".** That's the class imbalance, and it's why the loss counts each "used" portal 16 times (`portal_weight=15`, so 15 + 1).
- Disk size: the 1,200 set is 189 MB, 12,000 is 1.9 GB, 36,000 is 5.6 GB.

Generation times on my Mac (18 CPU cores):

| Set | Time |
|---|---|
| 1,200 | about 30 s |
| 12,000 | about 7.5 min |
| 36,000 | about 18.5 min |

---

## How we measure models

### A score is accurate to about ±0.1 points (2026-09-29, Andrew)

With 100 test problems, "% longer than optimal" is accurate to about **±0.1 percentage points** (the standard error of the per-problem differences). Differences under about 0.3 points are too close to call. Because every model takes the same exam, I also compare problem by problem and count how many problems each model wins. The command is in [How to reproduce our numbers](#how-to-reproduce-our-numbers).

### Which checkpoint you score moves the score by up to 0.3 points (2026-10-09, Andrew)

Training keeps two checkpoints: `nnArora.pt` (best validation F1, the one we always score) and `nnArora_best.pt` (the **last** one, despite its name). I scored the last one for all three scaling runs:

| Training examples | Best checkpoint (val F1 → score) | Last checkpoint (val F1 → score) | Last minus best |
|---|---|---|---|
| 1,200 | epoch 32: 0.226 → **3.77%** | epoch 42: 0.220 → 4.00% | +0.24 ± 0.08 points |
| 12,000 | epoch 44: 0.427 → **3.38%** | epoch 54: 0.380 → 3.53% | +0.14 ± 0.09 points |
| 36,000 | epoch 56: 0.459 → **3.38%** | epoch 60: 0.425 → 3.70% | +0.31 ± 0.12 points |

- The best-F1 checkpoint wins all 3 times, so choosing by validation F1 works.
- But F1 is a coarse guide. At 1,200 a tiny F1 drop (0.226 → 0.220) cost 0.24 points. The 12,000 best (F1 0.427) and the 36,000 last (F1 0.425) differ by 0.31 points.
- The scaling result still holds either way. 12,000 beats 1,200 with both checkpoints (−0.38 best, −0.48 last). 36,000 never beats 12,000 (−0.00 best, +0.17 ± 0.13 last).

**Why it matters:** the ±0.1 above only covers the luck of the 100 test problems. Comparing two **training runs** has a second source of luck: which checkpoint happened to be saved. So a difference under about 0.3 points between two training setups isn't trustworthy from one checkpoint each. For the next experiments, I'll score both checkpoints of every run (2 min each). A sturdier option is to keep the top 3 checkpoints by validation F1 and average their scores; that needs a small trainer patch and about 830 MB per checkpoint.

### What the eval costs mean (2026-09-28, Andrew)

`flow=eval` prints five costs for one test case. Two of them don't depend on the model at all:

| Cost | What it is | Depends on the model? |
|---|---|---|
| `golden_cost` | The optimal tree, from GeoSteiner | No |
| `adapted_cost` | Tree built from the *correct* portals, before cleanup | No (assumes a perfect network) |
| `processed_cost` | Same, after cleanup: the best NN-Steiner can possibly do on this case | No (assumes a perfect network) |
| `predict_cost` | Tree built from the model's predicted portals, before cleanup | Yes |
| `final_cost` | The model's actual result, after cleanup | **Yes. This is the number that matters** |

The authors' result files (`exp_out/*-solved.txt`, written by `flow=nn_exp`) record the **final** cost.

### F1 is a training signal, not the final quality measure (2026-09-28, updated 2026-10-08, Andrew)

After the network predicts portals, NN-Steiner hands them to an exact solver (GeoSteiner) that repairs many mistakes.
- On test case 1: the authors' model has F1 **0.327** and ends +2.4% over optimal. My 30-epoch model has F1 **0.121** and ends +5.2% (+15.6% before cleanup).
- In the scaling study, validation F1 nearly doubled from 1,200 to 12,000 examples (0.226 → 0.427), but the score only went from 3.77% to 3.38%.

Higher F1 does mean a better tree, but the relationship isn't simple. Use F1 to track training, and judge models by final wirelength. Also, where the logs print `acc`, the value is actually F1.

---

## Compute and hardware

### My Mac is faster than a Colab T4 for our workloads (2026-09-28, Andrew)

Same smoke test on both machines:

| Step | Mac (M5 Max) | Colab T4 |
|---|---|---|
| Training, 4 epochs | 37 s | 59 s |
| Data generation, 110 examples | about 13 s | about 100 s |

Data generation is CPU-heavy (GeoSteiner labels every example), and the Colab VM has only 2 CPU cores. So I train and generate data on my Mac.

Measured training speed on the Mac: about 49 s per epoch on 1,200 examples, 7.9 min on 12,000, and about 22.5 min on 36,000.

### Full-scale training may not fit in Colab's memory (2026-09-28, Andrew)

The authors' trainer loads the **entire training set into memory** before training. The 1,200 set is 189 MB, so the paper's 120,000 examples would be about **19 GB**. That fits on a 64 GB Mac but likely not on free Colab (about 12–13 GB of RAM). At the measured speeds, one epoch on 120,000 would take roughly **80 minutes** on my Mac.

**Why it matters:** the full dataset size decides where training can happen. Update (Oct 9): the data-scaling study says 120,000 isn't worth it (36,000 scored no better than 12,000), so this limit no longer blocks us. A 12,000 set is 1.9 GB on disk.

---

## Problems in the authors' code (our patches)

The authors' code is downloaded fresh by `setup.sh`, so we never edit it by hand. Every change is a patch file in `third_party/patches/`, applied automatically in order.

| Patch | Problem | Fix |
|---|---|---|
| `0001-rmst-include-cstddef` | RMST didn't compile on newer GCC (Colab) | Adds a missing C++ include |
| `0002-single-device-training` | Training required 2+ NVIDIA GPUs (a hard check in the trainer) | Runs on 1 GPU, an Apple GPU, or CPU. Checkpoints load without CUDA |
| `0003-rmst-64bit-and-clang-fixes` | RMST stored memory addresses in 32-bit ints, which is unsafe on 64-bit machines. A missing include also broke the Mac build | Uses proper pointer-sized ints and adds the include. RMST is on the main NN-Steiner path, not only the baseline |
| `0004-restore-leaf-refinement` | The authors' final commit left the cleanup step `refine_leaves` switched off (their "tree" ablation) | Switches it back on. **This is why we now match the paper.** It only affects solving, not data generation or training |
| `0005-data-gen-stall-retry` | GeoSteiner (without CPLEX) can run forever on a rare point set, and data generation then froze | After 2 minutes with no progress, stuck examples are redone with new points, with a logged warning. Every other example is unchanged |

`setup.sh` also handles build problems that aren't patches: on the Mac it uses Homebrew's `glibtool` (Apple's `libtool` fails), tells CMake 4 to accept the authors' old CMake files, and builds GeoSteiner with plain `make`, because a parallel build races.

**A known difference from the paper:** we build GeoSteiner without CPLEX, the commercial solver the authors used. It still finds the exact optimal trees, just more slowly. The report should mention this.

---

## Things that can silently go wrong

| Trap | What to do |
|---|---|
| **Checkpoint names are backwards.** In `train/model/`, `nnArora.pt` is the **best** model by validation F1, and `nnArora_best.pt` is just the **latest** | Always use `nnArora.pt`. The authors' own end-of-training test loads the latest one |
| **The default seeds leak.** Examples use `seed + example number`, and the default training seeds (4200 onward) overlap the validation seeds (5200 onward) | Keep training seeds far from validation and test (100000 vs 5200 / 6200) |
| **Dataset folder names don't include the seed** (`batch<num_trees>_100x100-uniform-<batch>-tensor`) | Train, validation and test sets must differ in size, or they overwrite each other |
| **`data_gen.batch` is the number of examples, not a batch size** | `data_gen.batch=36000` means "make 36,000 examples" |
| **Batch size must equal examples per tree shape.** Examples sharing a quadtree shape are stored next to each other | Set `train.batch_size` to `data_gen.batch / data_gen.num_trees` (50 for the scaling sets), or batches mix shapes |
| **The `test acc` printed at the end of training is a narrow check.** It only scores the first test batch (a single tree shape), and "acc" is really F1 | Judge models by validation F1 during training, and by % longer than optimal in the end |
| **In `flow=eval`, only `predict_cost` and `final_cost` depend on the model.** `adapted_cost` and `processed_cost` are built from the correct portals | `final_cost` is the model's real result |
| **Hydra can't read `=` inside a command-line value** | No `=` in file names. That's why the authors' `m=15_kb=4.pt` is saved as `m15_kb4.pt`. Also put quotes around overrides with brackets: `'flow=[nn_exp]'` |
| **Python 3.14 breaks Hydra** | Use Python 3.10–3.13 |
| **Colab clones the repo from GitHub** | It only sees pushed changes |
| **Mac and Colab differ in the last digits** | Normal: GPUs round differently. Compare models on the same machine |

---

## The data-scaling study (done)

Finished Oct 9. **Result: the score leveled off.** 1,200 → 3.77%, 12,000 → 3.38%, 36,000 → 3.38%. That's the third row of the outcome table below.

**The question:** our model is behind the authors'. Is that because it saw 100× less data?

**The plan:** train the same model with the same settings on 1,200, 12,000 and 36,000 examples, and score each on the same 100 test problems. Only the amount of data changes, so any change in the score comes from the data. Each run is one point on a "data vs score" graph.

**This isn't hyperparameter tuning.** Every model and training setting stays fixed. The epoch cap is only a safety limit; early stopping decides when each run really ends.

**Design choices, and why:**

| Choice | Why |
|---|---|
| Sizes go ×10, then ×3 | Learning improves in multiples, not in fixed steps |
| Each bigger set contains the smaller ones | Checked byte-for-byte, so the only difference is "more data" |
| Early stopping: stop after 10 epochs with no new best validation F1 | Every size gets to reach its own best |
| Epoch caps of 200 / 60 / 60 | Bigger sets need fewer passes but more total steps. The 36,000 cap went from 30 to 60 after the 12,000 run |
| Clean validation and test sets (seeds 5200 and 6200, training from 100000) | No leak |
| Same 100 test problems, same threshold (0.95), same answer file every time | Makes the scores comparable |
| Why not 120,000 right away? | About 19 GB of memory and roughly 80 min per epoch, so days of training. The trend tells us first whether it's worth it |

**What the outcome would mean:**

| If the score… | It means | Next |
|---|---|---|
| keeps dropping, and 36,000 is still above 2.65% | Data is the bottleneck | Generate 120,000 (fits on my Mac) |
| reaches about 2.65% | Full reproduction, with less data than the paper | Use this model for everything else |
| levels off above 2.65% | Something other than data is holding it back | Look at training length and learning rate |

**Status:** all three are scored, and the score leveled off above 2.65%. Details are in [Training results](#training-results).

---

## What comes next

**Now that the scaling study is done** (the score leveled off, so no 120,000):
1. ~~Quick check: score the 36,000 run's last checkpoint.~~ Done: the last checkpoint of every run is 0.14–0.31 points worse than its best ([details](#which-checkpoint-you-score-moves-the-score-by-up-to-03-points-2026-10-09-andrew)). From now on I score both checkpoints of every run.
2. **Graph it:** training examples (log scale) vs % longer than optimal, with a dashed line at the authors' 2.65%. Then write a report section. This is one of my 2–3 experiments: in our setup, NN-Steiner stops improving by 12,000 examples.
3. **Find what else differs from the authors' training.** Training length and learning rate are the obvious candidates. A longer run might find a checkpoint a few tenths better, but that alone wouldn't close 0.74 points. Also check what their training examples looked like: ours use upstream's default `data_gen` settings (`num_points: 180`, a 100 × 100 canvas).
4. **Hand the best model to Shraddha** with a clear name (`m15_kb4_n36000.pt`; the 12,000 and 36,000 models tie, and this one saw the most data), its settings and its curves, so she can test it on bigger problems and other point patterns.

**After that:**
- **1–2 more experiments,** each compared against the best scaling-study model and trained on 12,000 examples (same score as 36,000, a third of the time). The candidates (the authors tested several of these, so we can check against their result files):
  - `kb` (max points per leaf cell): 1, 4 or 7
  - `m` (portals per cell side): 3, 7 or 15
  - `portal_weight`, including 0, to show why the weighting matters
  - `threshold` (how sure the network must be to use a portal)
  - Train on uniform points, test on other point patterns
- **Report:** I lead Methodology & model, and help with Dataset and Experimental setup. I still owe a one-page architecture summary with a diagram.

---

## Our scripts, and what each one does

**Which one do I run?**

| I want to… | Run |
|---|---|
| Set up a new machine, or update after pulling new patches | `scripts/setup.sh` |
| Check that nothing is broken | `scripts/smoke_test.sh`, then `scripts/arora.sh pretrained_solve` |
| Get the authors' trained model | `scripts/get_pretrained.sh` |
| Make data, train, solve or score anything | `scripts/arora.sh <config> [settings...]` |
| Do all of this on Colab | `notebooks/colab.ipynb` |

All of them run from the repo root, with the virtual environment active on a Mac (`source .venv/bin/activate`).

### `scripts/setup.sh`: get a machine ready

Run it once per machine (on Colab, once per session, since Colab wipes everything). The first run takes a few minutes. Re-running is safe: it skips anything already done. Steps:

1. **Checks the Python version.** It must be 3.10–3.13, because Hydra breaks on 3.14. Otherwise it stops and tells you how to make a venv.
2. **Downloads the authors' code** into `third_party/NN-Steiner/`, at a fixed commit so we all run the same version. It skips the 770 MB model (`get_pretrained.sh` fetches that). Skipped if the code is already there.
3. **Applies our patches** from `third_party/patches/` in number order, printing `applied` or `already applied` for each. This is why re-running it after a pull picks up new patches.
4. **Installs dependencies:**
   - Colab (Linux): build tools, GMP and friends via `apt`.
   - Mac: `gmp`, `cmake` and `libtool` via Homebrew. Homebrew's `libtool` provides `glibtool`, which GeoSteiner needs; Apple's own `libtool` can't build it.
   - Then the authors' Python packages (`requirements.txt`).
5. **Builds the C/C++ parts:** GeoSteiner 5.3 (without CPLEX), plus the Python connectors for GeoSteiner and RMST. Each one is skipped if it already works.
6. **Creates the `work/` folders,** checks that both connectors load, and prints the device it found, e.g. `Setup complete. Training/inference device: Apple GPU (MPS)`.

It prints one line per step. The full output of each step goes to `work/setup_logs/<step>.log`. If a step fails, it prints the last 40 lines of that log and stops.

### `scripts/arora.sh`: run anything

The one launcher for everything: making data, training, solving and scoring.

```bash
scripts/arora.sh <config> [settings...]
scripts/arora.sh scale 'flow=[nn_exp]' nn_exp.model=outputs/2026-09-29/17-07-18/train/model/nnArora.pt nn_exp.output=exp_out/scale-12000
```

What it does:
1. Checks that `setup.sh` has been run.
2. Makes sure the `work/` folders exist. The authors' data generator crashes if they don't.
3. **Picks the device for solving:** NVIDIA GPU (`cuda:0`), then Apple GPU (`mps`), then CPU, unless you pass `model.device=...` yourself. Training picks its own device the same way (patch 0002).
4. Lets PyTorch fall back to the CPU for any operation Apple's GPU doesn't support, instead of crashing.
5. Moves into `work/` and runs the authors' program (`python -m arora`). The final settings are stacked like this (later wins):

```
authors' conf/config.yaml  +  configs/<config>.yaml  +  your command-line settings
```

- **`flow=` decides what runs:** `data_gen` (make data), `train`, `solve` / `eval` (one test problem), `nn_exp` (solve a whole test folder), `geo_exp` (GeoSteiner baseline), `mst_exp` (RMST baseline), and a few more.
- **Every run gets its own folder,** `work/outputs/<date>/<start time>/`. It holds the log, the exact settings used (`.hydra/`), and for training the curves (`train/`) and checkpoints (`train/model/`).
- **Paths in configs are relative to `work/`:** `data/...`, `outputs/...`, `models/...`. The authors' test problems are under `upstream/points/...` (`work/upstream` is a shortcut to their code).
- To see the final settings without running anything: `scripts/arora.sh <config> --cfg job`.
- Running it with no arguments prints how to use it.

**The configs it can use** (`configs/`; each lists only what differs from the authors' defaults):

| Config | Use it for |
|---|---|
| `smoke` | The tiny pipeline check (used by `smoke_test.sh`) |
| `medium` | My first training runs on the authors' default sizes (1,200 / 96 / 32). Superseded by `scale` |
| `pretrained_solve` | The authors' model on one 100-point test problem. Should print `final_cost: 75641.0` |
| `scale` | The data-scaling study: clean validation and test sets, batch size 50, early stopping, and the 100 test problems. **Use this one for new training.** The training set and epoch cap go on the command line |

### `scripts/smoke_test.sh`: quick health check (about a minute)

1. **Makes three tiny datasets** if they don't exist yet: training (80 examples, seed 1000), validation (20, seed 2000) and test (10, seed 3000).
2. **Trains for 4 epochs** with `configs/smoke.yaml`.

It should end with `test acc: 0.11926605504587157` on my Mac, or `0.12177729018102029` on a Colab T4. A different number on the same machine means something changed. Extra arguments go to the training run, e.g. `scripts/smoke_test.sh train.epochs=10`. It only checks that everything runs, never model quality.

### `scripts/get_pretrained.sh`: the authors' trained model

1. Downloads the authors' model (a 770 MB archive) straight from GitHub, for the same pinned commit. No `git-lfs` needed.
2. Checks the file's SHA-256 checksum, so a broken download is caught.
3. Unpacks it and renames `m=15_kb=4.pt` to **`work/models/pretrained/m15_kb4.pt`**, because Hydra can't handle `=` in file names.

It does nothing if the model is already there. On Colab, `work/models` lives on Drive, so it only downloads once for the whole team.

### `scripts/common.sh`: shared settings (never run directly)

The other scripts load it. It defines:
- Where the repo, the authors' code and `work/` are.
- **The pinned commit of the authors' code** (`5fd75c5`). Changing it here changes it for everyone.
- Which Python to use (`python3` by default; override with e.g. `PYTHON=.venv/bin/python`).
- `make_work_dirs`, which creates `work/data`, `points`, `models`, `outputs` and `exp_out`, plus the `work/upstream` shortcut.

### `notebooks/colab.ipynb`: the same, on Colab

Run the cells top to bottom at the start of every session, on a GPU runtime (T4):
1. **Mount Drive.**
2. **Get the code:** clones our repo from GitHub, or pulls if it's already there. It only sees **pushed** changes.
3. **Keep outputs on Drive:** links `work/data`, `points`, `models`, `outputs` and `exp_out` to the shared folder `FA26CMPE257Sec02MachineLearningProject/nn-steiner-work`, so datasets, models and results survive when Colab resets. It stops with an error if one of those is a real folder instead of a link.
4. **Set up:** runs `setup.sh`.
5. **Pretrained model:** runs `get_pretrained.sh`, then `pretrained_solve`.
6. **Smoke test:** runs `smoke_test.sh`.
7. **Training curves:** opens TensorBoard inside the notebook.

The last cell explains how to run your own config: add it to `configs/`, push, pull in step 2, then `!bash scripts/arora.sh <config> [settings]`.

### The authors' tools we use

| Tool (in `third_party/NN-Steiner/`) | What it does |
|---|---|
| `evaluator/evaluateRatio.py <ours> <perfect>` | **The scoring script.** Both files have one total wire length per test problem, in the same order. It divides ours by perfect for each line, averages, subtracts 1 and multiplies by 100. It prints `average error percentage of 100 cases: ...`, which is "% longer than optimal" |
| `evaluator/evaluateSTT.py` | Checks that a tree is valid: connected, and covers every point. I haven't needed it yet; it's part of the evaluation work |
| `exp/*.sh` (`nn.sh`, `geo.sh`, `mst.sh`, …) | The authors' own experiment loops. We don't run them directly, because they call `python -m arora` instead of our launcher, but they're good templates for evaluation scripts |

---

## How to reproduce our numbers

Run from the repo root, with the virtual environment active (`source .venv/bin/activate`).

**Health checks** (run these after changing anything):
```bash
scripts/smoke_test.sh              # ends with: test acc: 0.11926605504587157 (Mac)
scripts/arora.sh pretrained_solve  # prints:    final_cost: 75641.0
```

**Make the scaling-study data:**
```bash
scripts/arora.sh scale 'flow=[data_gen]' data_gen.seed=5200   data_gen.num_trees=4   data_gen.batch=128    # validation
scripts/arora.sh scale 'flow=[data_gen]' data_gen.seed=6200   data_gen.num_trees=2   data_gen.batch=64     # test
scripts/arora.sh scale 'flow=[data_gen]' data_gen.seed=100000 data_gen.num_trees=24  data_gen.batch=1200
scripts/arora.sh scale 'flow=[data_gen]' data_gen.seed=100000 data_gen.num_trees=240 data_gen.batch=12000
scripts/arora.sh scale 'flow=[data_gen]' data_gen.seed=100000 data_gen.num_trees=720 data_gen.batch=36000
```

**Train** (`caffeinate` keeps a Mac awake; drop it elsewhere):
```bash
caffeinate -i scripts/arora.sh scale train.train_set=data/batch24_100x100-uniform-1200-tensor   train.epochs=200
caffeinate -i scripts/arora.sh scale train.train_set=data/batch240_100x100-uniform-12000-tensor train.epochs=60
caffeinate -i scripts/arora.sh scale train.train_set=data/batch720_100x100-uniform-36000-tensor train.epochs=60
```
Watch training live with `tensorboard --logdir work/outputs`, then open http://localhost:6006.

**Score a model** (example: the 12,000 model):
```bash
# 1. Solve the 100 test problems with the trained model (about 2 min)
scripts/arora.sh scale 'flow=[nn_exp]' nn_exp.model=outputs/2026-09-29/17-07-18/train/model/nnArora.pt nn_exp.output=exp_out/scale-12000

# 2. Compare with the perfect trees. Prints "average error percentage" = % longer than optimal
python third_party/NN-Steiner/evaluator/evaluateRatio.py \
    work/exp_out/scale-12000-solved.txt third_party/NN-Steiner/exp_out/100_10000x10000-uniform-golden.txt
```

**Compare two models problem by problem** (prints both scores, the difference with its ±, and who wins how many problems):
```bash
python - <<'PY'
import numpy as np
g = np.loadtxt("third_party/NN-Steiner/exp_out/100_10000x10000-uniform-golden.txt")
a = np.loadtxt("work/exp_out/scale-12000-solved.txt") / g - 1         # model A
b = np.loadtxt("work/exp_out/scale-pretrained-solved.txt") / g - 1    # model B
d = (a - b) * 100
print(f"A: {a.mean()*100:.2f}%  B: {b.mean()*100:.2f}%  A-B: {d.mean():+.2f} ± {d.std(ddof=1)/10:.2f} points  "
      f"A shorter on {(d<0).sum()}, B shorter on {(d>0).sum()} of 100")
PY
```
For the 12,000 model vs the authors' model it prints `A: 3.38%  B: 2.65%  A-B: +0.74 ± 0.10 points  A shorter on 24, B shorter on 76 of 100`.

---

## Where everything is

Everything under `work/` is on my Mac only (it isn't in git or on Drive). Tell me if you need something.

**Runs** (each folder is named by its start date and time; to see what a run was, use `cat work/outputs/<date>/<time>/.hydra/overrides.yaml`):

| Run folder | What it was |
|---|---|
| `2026-09-28/16-50-57`, `16-54-21`, `17-02-31` | Smoke tests (80 examples, 4 epochs) |
| `2026-09-28/18-40-42` | Medium run: 1,200 examples, 10 epochs (old leaky validation) |
| `2026-09-28/19-11-43` | Medium run: 1,200 examples, 30 epochs, the 3.74% model (old leaky validation) |
| `2026-09-29/16-01-59` | **Scaling study: 1,200 examples** (3.77%) |
| `2026-09-29/17-07-18` | **Scaling study: 12,000 examples** (3.38%) |
| `2026-10-08/11-43-16` | **Scaling study: 36,000 examples** (3.38%) |

Inside a training run: `train/` holds the TensorBoard curves, and `train/model/nnArora.pt` is the best model.

**Score files** (`work/exp_out/`, one total wire length per test problem):

| File | Model |
|---|---|
| `scale-pretrained-solved.txt` | Authors' model (2.65%) |
| `scale-1200-solved.txt` | 1,200 examples (3.77%) |
| `scale-12000-solved.txt` | 12,000 examples (3.38%) |
| `scale-36000-solved.txt` | 36,000 examples (3.38%) |
| `scale-1200-last-`, `scale-12000-last-`, `scale-36000-ep60-solved.txt` | Each run's **last** checkpoint (4.00%, 3.53%, 3.70%) |
| `check-refine-on/off-100-uniform-solved.txt` | Authors' model with cleanup on / off |
| `check-medium30-refine-on/off-100-uniform-solved.txt` | My 30-epoch model with cleanup on / off |

**Datasets** (`work/data/`):

| Folder | What | Size on disk |
|---|---|---|
| `batch4_100x100-uniform-128-tensor` | Validation, seed 5200 | 21 MB |
| `batch2_100x100-uniform-64-tensor` | Test, seed 6200 | 11 MB |
| `batch24_100x100-uniform-1200-tensor` | Training, 1,200 | 189 MB |
| `batch240_100x100-uniform-12000-tensor` | Training, 12,000 | 1.9 GB |
| `batch720_100x100-uniform-36000-tensor` | Training, 36,000 | 5.6 GB |

**Files in the repo:**

| File | What it is |
|---|---|
| `scripts/setup.sh`, `common.sh`, `arora.sh`, `smoke_test.sh`, `get_pretrained.sh` | Setup, the run launcher, and helpers. Details in [Our scripts](#our-scripts-and-what-each-one-does) |
| `configs/smoke.yaml`, `medium.yaml`, `pretrained_solve.yaml`, `scale.yaml` | Settings for each kind of run (table in [Our scripts](#our-scripts-and-what-each-one-does)) |
| `notebooks/colab.ipynb` | The Colab entry point |
| `third_party/patches/0001–0005` | Our fixes to the authors' code |
| `docs/Project_Proposal.md`, `Team_Responsibilities.md`, this file | The proposal, the plan, and this record |

---

## Open questions

- **What causes the remaining 0.74-point gap to the authors,** if not the amount of data? Candidates: training length, learning rate, and what their training examples looked like. (Andrew)
- **Does NN-Steiner beat FLUTE on bigger problems (500+ points) in our setup,** as the paper claims? (Shraddha)
- **Which 1–2 more experiments do we commit to?** Candidates are in [What comes next](#what-comes-next). (Andrew proposes, the team agrees)
- **Does loading the latest checkpoint instead of the best one** change the authors' reported test numbers? (Andrew)

**Answered:**
- *Do we reproduce the authors' pretrained results across all 100 test cases?* Yes, once the cleanup step is restored: 2.648% vs 2.646% ([details](#the-authors-model-matches-the-paper-once-a-switched-off-cleanup-step-is-restored-2026-09-29-andrew)).
- *Where does validation F1 level off?* On 1,200 examples, around epoch 16–20 (about 0.22 on the clean validation set). The old run's "still rising" 0.304 was on the leaky set. Bigger datasets level off later and higher.
- *Does the data trend continue at 36,000, or level off?* It levels off: 36,000 scores 3.38%, the same as 12,000 ([details](#data-scaling-study-36000-examples-338-no-better-than-12000-2026-10-09-andrew)).
- *How big should the full training set be?* 12,000 examples is enough in our setup; 36,000 didn't help, so we're not making 120,000. That also avoids the memory limit in [Compute and hardware](#compute-and-hardware).

---

## Words we use

| Word | Meaning |
|---|---|
| Portal | A fixed crossing point on a quadtree cell's edge (15 per side). The network predicts which portals the optimal tree passes through |
| Leaf cell / `kb` | A smallest quadtree cell / the most points allowed in one (4) |
| Cleanup (`refine_leaves`) | After the network's guess, re-solving small groups of points exactly with GeoSteiner. Patch 0004 switched it back on |
| % longer than optimal | How much more wire our tree uses than GeoSteiner's perfect tree. **The real score** (the authors call it "error") |
| Golden file | The file of perfect GeoSteiner lengths for the test problems: the answer key for scoring |
| F1 | How well the network guesses portals. A training signal, not the final score. Logs print it as "acc" |
| Epoch / step | One pass through the training data / one weight update from one batch of 50 examples |
| Early stopping | Ending training once validation F1 hasn't improved for 10 epochs |
| Ablation | Switching off one part of a method to see how much it matters |
| Data leak | Validation or test data sneaking into training, which makes scores look too good |
| Seed | The number that fixes "random" choices, so a run can be repeated exactly |
