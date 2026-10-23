# Findings

The team's running record of the project: where things stand, what each of us needs to know, every important finding (results, surprises, and problems that affect how we run experiments or what we write in the report), and how to reproduce it all. Most entries so far are mine (Andrew, Modeling).

**New here?** Read [Where things stand](#where-things-stand) and [What you need to know or do](#what-you-need-to-know-or-do) first.

**How to add a finding:** give the date, who found it, what we found, the evidence (numbers, run folder, command), and why it matters. Put it in the right section. Move items out of "Open questions" once they're answered, and update "Where things stand" if the big picture changes.

- Background on the paper: [Project_Proposal.md](Project_Proposal.md)
- Who owns what: [Team_Responsibilities.md](Team_Responsibilities.md)

*Last updated: 2026-10-20 (Andrew)*

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
- [The threshold experiment (done)](#the-threshold-experiment-done)
- [Comparing the two networks directly (done)](#comparing-the-two-networks-directly-done)
- [Training on test-shaped examples (done)](#training-on-test-shaped-examples-done)
- [How the paper trained its model](#how-the-paper-trained-its-model)
- [The paper's full 120,000 examples (done)](#the-papers-full-120000-examples-done)
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
| Can we train a model that good ourselves? | **Yes.** Trained on the paper's full 120,000 paper-style examples (one fixed tree shape, batch 500, 140 epochs), our model scores **2.73%** vs the authors' **2.65%**: +0.08 ± 0.09 points, and ours is shorter on 52 of the 100 problems. That's a tie. Earlier runs: 1,200 examples **3.77%**, 12,000 **3.38%**, 36,000 **3.38%**, 12,000 test-shaped **3.28%**, 12,000 paper-style trained much longer **3.45%** ([details](#the-papers-full-120000-examples-done)) |
| Is more training data the fix? | **For paper-style data, yes.** 120,000 examples gave 2.73%: 0.55 ± 0.11 points better than our previous best (3.28%), and 0.72 ± 0.09 better than the same setup on 12,000 (3.45%). For corner-shaped data it wasn't: 12,000 → 36,000 changed nothing (each model wins 50 of 100 problems) |
| Is the portal threshold the problem? | **No, but each model needs its own.** For our corner-shaped model the default 0.95 was already best, on both the 100- and 200-point sets. Models trained differently are sure of themselves in different ways: the test-shaped model's best is 0.8, the 120,000 model's 0.9, both picked on the 200-point set. The sweep also showed **where** the gap was: over a "no network" baseline (threshold 1.0, 3.77%), the corner-shaped network improved the score by only 0.39 points, the authors' by 1.07. The 120,000 model's network adds 0.99 ([details](#the-threshold-experiment-done)) |
| Does the authors' network guess portals better than ours? | **Depends on the data's shape.** On our corner-shaped validation data, ours wins (F1 0.46 vs 0.29). On validation data shaped like the test problems, theirs wins clearly (0.37 vs 0.25). Our corner-shaped training examples put all points in one corner (quadtrees 9–11 levels deep); the test problems don't (2–4) ([details](#comparing-the-two-networks-directly-done)). Our 120,000 model now beats theirs on test-shaped data (0.42 vs 0.37), though the trees only tie ([details](#the-papers-full-120000-examples-done)) |
| Is the shape of our training data the fix? | **It helps a little; it isn't the big fix.** Training on 12,000 examples shaped like the test problems gave 3.62% at the default threshold 0.95, because that network is rarely 95% sure. Its own threshold sweep picked 0.8: **3.28%**, a tie with the corner-shaped 3.38% on the 100 test problems (−0.10 ± 0.12), and a clearer gain on the 200-point set (−0.25 ± 0.09). That still left it 0.63 points behind the authors; the rest of the gap was the number of examples ([details](#training-on-test-shaped-examples-done)) |
| How did the authors train their model? | **About 4,000× longer than our test-shaped model's best checkpoint** (and 36× longer than our 120,000 run): 120,000 examples, 5,000 epochs, batches of 5,000, one fixed tree shape, no early stopping mentioned. Everything else matches our setup, and their data was test-shaped too ([details](#how-the-paper-trained-its-model)) |
| Does training much longer help? | **Not on 12,000 examples.** A paper-style run (one tree shape, batch 500, no early stopping) memorized them: by epoch 600 training F1 was 0.98 and validation F1 had fallen from 0.24 to 0.18. Its trees got worse too: **3.45%** at epoch 50 (its best), 3.68% at epoch 600 (+0.22 ± 0.11). Even the best is behind the test-shaped 3.28% ([details](#scoring-the-long-run-longer-training-on-12000-examples-doesnt-help-2026-10-19-andrew)). **On 120,000 examples it does:** the network no longer memorizes (training F1 only 0.76 after 140 epochs), and the trees improve from 3.62% at epoch 5 to 2.74% at epoch 40, then hold steady ([details](#the-papers-full-120000-examples-done)) |
| What's running right now? | Nothing. The 120,000 run is finished and scored |
| What's next? | Hand the 120,000 model to Shraddha (at threshold 0.9) for bigger problems and other point patterns. Then the report, and `portal_weight` if there's time. Details in [What comes next](#what-comes-next) |

"% longer than optimal" is the real score: how much more wire our tree uses than the perfect tree from GeoSteiner. Lower is better. 3.38% means: where the perfect tree needs 100 m of wire, ours uses 103.38 m.

---

## What you need to know or do

### Everyone

- **Pull `main` and re-run `scripts/setup.sh`.** Two patches landed on Sep 29 (0004 and 0005), and two more on Oct 18 (0006: numbered checkpoints; 0007: fixes the paper's `level` data mode). The script is safe to re-run: it only applies patches that aren't applied yet. On Colab the notebook clones fresh every session, so it picks them up on its own.
- **Judge models by "% longer than optimal", not by F1.** F1 tracks training; it isn't the result. See [What the results mean](#what-the-results-mean).
- **Give every model its own threshold.** Pick it on the 200-point set and report the 100-point score, which is what `scripts/score_checkpoint.sh` does. The default 0.95 isn't every model's best.

### Shraddha

- **Any NN-Steiner numbers made before Sep 29 measured the wrong setup.** The authors' final code has a cleanup step (`refine_leaves`) switched off. With it off, their model scores 6.87%, not 2.65%. Patch 0004 switches it back on. Please re-run anything from before then.
- **Known-good checks for your pipeline:**
  - `scripts/arora.sh pretrained_solve` should print `final_cost: 75641.0`.
  - The authors' model on `upstream/points/point100_10000x10000-uniform-100-pt/` should score **2.648%** against the authors' answer file, `third_party/NN-Steiner/exp_out/100_10000x10000-uniform-golden.txt`. 96 of the 100 lengths match the authors' own result file exactly.
- **Our own trained model is ready, and it ties the authors'** (120,000 paper-style examples, 2.73% vs their 2.65%). I'll share the `.pt` file (about 830 MB), its exact settings and its training curves. **Run it with `nn_exp.threshold=0.9`:** that's the threshold picked for it on the 200-point set (0.95 was a little worse there). Test the authors' model alongside it.
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

**Oct 9 (Andrew):** the 36,000 run used all 60 epochs (22 h 23 min) and scored 3.38%, a tie with 12,000. More data stopped helping, so the scaling study is done and we're not making 120,000 (of corner-shaped data; 120,000 paper-style examples later closed the gap, Oct 19–20). Then I scored each run's last checkpoint too: it's 0.14–0.31 points worse than the best one every time, so the checkpoint you score matters.

**Oct 9–10 (Andrew): the threshold experiment.** Solved the 100- and 200-point test sets with our 36,000 model at thresholds from 0.1 to 1.0. 0.95 (the default) is our best, so our score stays 3.38%. Over the "no network" baseline, our network adds only about a third of what the authors' does. Spot-checked the authors' model at two thresholds: it matches their result files, so their threshold curves are comparable with ours.

**Oct 10 (Andrew): graded the two networks' portal guesses directly.** Wrote `scripts/val_f1.py`. On our validation set the authors' network scores a lower F1 than ours (0.29 vs 0.46), yet builds better trees. Then found that our training examples are shaped very differently from the test problems (quadtrees 9–11 levels deep vs 2–4). On a validation set shaped like the test problems the ranking flips: the authors' network 0.37, ours 0.25. Set up `configs/shape.yaml` for a test-shaped training run.

**Oct 13–18 (Andrew): the test-shaped run.** Made the test-shaped test set (seed 8200) and 12,000 test-shaped training examples, trained (early-stopped at epoch 22, 1 h 55 min) and scored it: **3.62%**, no better than the corner-shaped 3.38%. Then graded its portal guesses on the test-shaped test set: they're about as often right as the authors', but rarely 95% sure, so at the default threshold the solver barely uses them. Its own threshold sweep picked 0.8 on the 200-point set: **3.28%** on the 100 test problems, a tie with 3.38% there and a clearer gain on the 200-point set. The shape helps a little; it isn't the big fix. Then read the paper's training recipe: everything matches ours except they trained about 4,000× longer, on one fixed tree shape, with no early stopping mentioned. Set up a paper-style run: patch 0006 (numbered checkpoints), patch 0007 (the paper's data mode crashed in the released code) and `configs/paper.yaml`, all tested. Made its 12,000 examples and ran a speed test: batch 500 for 1,600 epochs (about 21.5 h) is the plan.

**Oct 18–19 (Andrew): the long paper-style run.** Trained on the 12,000 paper-style examples at batch 500 with no early stopping. By epoch 609 it was memorizing (training F1 0.98, validation F1 down from 0.24 at epoch 120 to 0.18), so I stopped it at epoch 611. Wrote `scripts/score_checkpoint.sh` and scored 5 of its checkpoints: the trees got a little worse as it memorized (3.45% at epoch 50, 3.68% at epoch 600), and all are behind the test-shaped 3.28%. Training longer on 12,000 examples doesn't help; next is the paper's full 120,000.

**Oct 19–20 (Andrew): the paper's full 120,000 examples.** Made 120,000 paper-style examples (9.6 GB) and trained them for 140 epochs at batch 500. It ran in three parts: I paused it twice (at epochs 42 and 123) and restarted from the last numbered checkpoint with `train.checkpoint`. About 22.5 h in all. Validation F1 rose to 0.43 with no memorizing. Scored 6 checkpoints with `scripts/score_checkpoint.sh`. The one picked on the 200-point set (epoch 140, threshold 0.9) scores **2.73%**, a tie with the authors' 2.65% (+0.08 ± 0.09). The number of examples was the gap.

---

## All results in one place

All scores are on the same exam: the authors' 100 test problems (100 uniformly random points each, on a 10,000 × 10,000 canvas), with threshold 0.95 unless a row says otherwise, compared with the perfect GeoSteiner trees. Lower is better.

| Method | Training examples | Cleanup off | **Cleanup on** (the paper's method) |
|---|---|---|---|
| FLUTE (fast standard method, authors' result file) | – | – | 1.25% |
| Authors' model, their published result file | 120,000 | 6.87% | 2.65% (2.646) |
| Authors' model, run by me | 120,000 | 6.87% | **2.65%** (2.648) |
| My 30-epoch model (old leaky validation) | 1,200 | 10.63% | 3.74% |
| **Scaling study** | 1,200 | – | **3.77%** |
| **Scaling study** | 12,000 | – | **3.38%** |
| **Scaling study** | 36,000 | – | **3.38%** |
| **Test-shaped training** (threshold 0.8, picked on the 200-point set) | 12,000 | – | **3.28%** (3.62% at 0.95) |
| **Paper-style training, much longer** (one tree shape, batch 500; best checkpoint, epoch 50, threshold 0.8) | 12,000 | – | **3.45%** (3.68% after 600 epochs) |
| **Paper-style training, the paper's full 120,000 examples** (140 epochs, batch 500; epoch 140 and threshold 0.9, both picked on the 200-point set) | 120,000 | – | **2.73%** (2.74% already at epoch 40) |
| No network (threshold 1.0), authors' model | – | – | 3.72% |
| No network (threshold 1.0), our 36,000 model | – | – | 3.77% |
| No network (threshold 1.0), our test-shaped model | – | – | 3.78% |

The two "no network" rows use threshold 1.0, so almost no portal passes and the tree comes mostly from the cleanup step. They show how much each network adds ([details](#the-threshold-experiment-done)).

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

2. **More training data helped once, then stopped helping.** Going from 1,200 to 12,000 examples (10×) improved the score by 0.38 points and closed about a third of the gap to the authors. Going from 12,000 to 36,000 (3×) changed nothing: −0.00 ± 0.11 points, and each model wins exactly 50 of the 100 problems. So with corner-shaped data the score levels off at about 3.38%, still 0.74 points behind the authors, and generating 120,000 of those examples isn't worth days of compute. (Points 9–10 reopen this for paper-style data, which is much cheaper: with 120,000 of those, the gap closes.)

3. **Bigger datasets need longer training.** The number of learning steps needed to peak went from 768 (1,200 examples) to 10,560 (12,000) to at least 40,320 (36,000). If the epoch cap doesn't grow with the data, runs get cut off early and the bigger dataset looks worse than it is. The 36,000 run did hit its 60-epoch cap. A longer run might have found a checkpoint a few tenths better (point 6), but not enough to close the 0.74-point gap.

4. **The cleanup step does a lot of the work.** Without it, the authors' model goes from 2.65% to 6.87%. NN-Steiner is a team effort: the network does the big-picture routing, and the exact solver fixes small local pieces.

5. **F1 is a training signal, not the result.** Validation F1 nearly doubled from 1,200 to 12,000 examples (0.226 → 0.427), but the score improved much less (3.77% → 3.38%). Better F1 does mean shorter trees, but not in proportion, because the cleanup repairs many wrong guesses. From 12,000 to 36,000 examples, validation F1 rose again (best 0.427 → 0.459; averaged over each run's last 6 checks, 0.403 → 0.435), and the score didn't move at all. Within a run, though, the checkpoint with the best validation F1 did beat the last checkpoint every time. So F1 is a rough guide: good for picking a checkpoint, not for predicting the score. Between models trained on different data it can point the wrong way entirely: on our validation data the authors' model scores 0.29 vs our 0.46, yet builds much better trees ([details](#comparing-the-two-networks-directly-done)). Our 120,000 model is the reverse case: a higher F1 than the authors' on test-shaped data (0.41 vs 0.35), but trees that only tie. Only "% longer than optimal" tells us how good a model is.

6. **How sure we can be.** Each score is accurate to about ±0.1 points, but that only covers the luck of the 100 test problems. **Which checkpoint you score adds more:** each run's best and last checkpoints differ by 0.14–0.31 points ([details](#which-checkpoint-you-score-moves-the-score-by-up-to-03-points-2026-10-09-andrew)). So a difference under about 0.3 points between two training runs isn't trustworthy from one checkpoint each. All three scaling-study gaps to the authors (1.12, 0.74 and 0.74) are well above that, and they win most problems, so those gaps are real. The 120,000 model's gap (+0.08) is well below it: a tie. Two comparisons are ties: 1,200 vs the old 30-epoch model (+0.03), and 36,000 vs 12,000 (−0.00, 50 wins each).

7. **The gap is in the network itself, not the threshold.** At threshold 1.0 almost no portal passes, so that's a "no network" baseline: 3.77% for our model, 3.72% for the authors'. Our network improves on it by 0.39 points; theirs by 1.07. So ours delivers about a third of their network's benefit, and the 200-point set gives the same ratio (0.24 vs 0.63). The default threshold (0.95) is already our best, so changing it can't close the gap. Also, using the portals our network is 60–90% sure about makes trees no better, or even worse, than using no network. Using the authors' portals in that range helps ([details](#the-threshold-experiment-done)). That holds for our corner-shaped model. For the test-shaped model the threshold did matter (point 8), and the 120,000 model's best is 0.9.

8. **Our training data didn't match the test problems, but fixing that didn't close the gap.** Our training examples put all points in one corner of the quadtree's box, giving trees 9–11 levels deep; the test problems' trees are 2–4 levels deep. On validation data shaped like ours, our network guesses portals better than the authors' (F1 0.46 vs 0.29). On validation data shaped like the test problems, theirs is clearly better (0.37 vs 0.25) ([details](#comparing-the-two-networks-directly-done)). Training on test-shaped examples scored 3.62% at threshold 0.95, because the new network is rarely 95% sure (though its portals are about as often right as the authors'). At the threshold that suits it (0.8, picked on the 200-point set) it scores **3.28%**: a tie with 3.38% on the 100 test problems, a clearer gain on the 200-point set, and its network now adds 0.50 points over no network instead of 0.39 ([details](#training-on-test-shaped-examples-done)). So the shape helps a little, but the gap to the authors was still 0.63 points (point 10 closes it). Also: each model needs its own threshold, picked on the 200-point set.

9. **Training much longer on 12,000 examples doesn't help; the network memorizes them.** The paper trained about 4,000× longer than our best checkpoint ([details](#how-the-paper-trained-its-model)), so I ran a paper-style copy (one tree shape, batch 500, no early stopping) for 611 epochs. It learned its 12,000 examples by heart (training F1 0.98) while validation F1 fell, and the trees got a little worse with it: 3.45% at epoch 50, 3.68% at epoch 600 (+0.22 ± 0.11). Even its best checkpoint is behind the test-shaped model (+0.17 ± 0.11 on the 100 test problems, +0.19 ± 0.07 on the 200-point set) ([details](#scoring-the-long-run-longer-training-on-12000-examples-doesnt-help-2026-10-19-andrew)). Both of our whole-box runs memorized 12,000 examples early, which pointed at the paper's 10× more examples. Point 10 confirms it.

10. **With the paper's full 120,000 examples, our own model matches the authors'.** Same paper-style setup as point 9, with 10× the examples, for 140 epochs: 33,600 learning steps of 500, about 16.8 million examples processed, 36× less than the paper. It scores **2.73%** vs the authors' 2.65%: +0.08 ± 0.09 points, and each model wins about half the problems (52 vs 48). That's 0.55 ± 0.11 better than our previous best (3.28%), well past the 0.3 points set in advance, so **the number of examples was the gap**. The extra examples don't make early learning faster: at the same 1,200 steps, 120,000 examples score no better than 12,000 (3.62% vs 3.45%, +0.17 ± 0.10). What they do is stop the memorizing, so the network can keep learning. Training F1 only reached 0.76 (vs 0.98 on 12,000), and the trees went from 3.62% at epoch 5 to 2.74% at epoch 40, then held steady to epoch 140. Our network now adds about 93% of what the authors' adds over no network (0.99 vs 1.07 points) ([details](#the-papers-full-120000-examples-done)).

11. **What these results don't cover yet:**
   - Bigger problems (500–5,000 points), where the paper's main claim lives.
   - Other point patterns (normal, mix-normal, non-isotropic).
   - Runtime. Ours isn't comparable to the paper's anyway: different hardware, and GeoSteiner without CPLEX.
   - Run-to-run luck. Every result is one training run with one seed (42).

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

**Why it matters:** in our setup, more data stops helping by 12,000 examples. The remaining 0.74-point gap to the authors comes from something else, so we're not generating 120,000. (That holds for corner-shaped data. 120,000 paper-style examples later closed the gap: [details](#the-papers-full-120000-examples-done).) A practical bonus: later experiments can train on 12,000 examples (about 7 h) instead of 36,000 (about 22 h) with no loss in score.

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
- **157–185 quadtree cells** and **2,652–3,128 portals** per example. The quadtrees are **9–11 levels deep**, because the points cover only 0–99 inside the 10,000 × 10,000 quadtree box ([details](#our-training-examples-are-shaped-very-differently-from-the-test-problems-2026-10-10-andrew)).
- **Only 3.1% of portals are labeled "used".** That's the class imbalance, and it's why the loss counts each "used" portal 16 times (`portal_weight=15`, so 15 + 1).
- Disk size: the 1,200 set is 189 MB, 12,000 is 1.9 GB, 36,000 is 5.6 GB. Paper-style examples are about half the size: 12,000 is 984 MB, 120,000 is 9.6 GB.

Generation times on my Mac (18 CPU cores):

| Set | Time |
|---|---|
| 1,200 | about 30 s |
| 12,000 | about 7.5 min |
| 36,000 | about 18.5 min |
| Paper-style 12,000 | 2 min 46 s |
| Paper-style 120,000 | 30 min 42 s (mean 163.8 points per example, like the 12,000 set) |

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

**Why it matters:** the ±0.1 above only covers the luck of the 100 test problems. Comparing two **training runs** has a second source of luck: which checkpoint happened to be saved. So a difference under about 0.3 points between two training setups isn't trustworthy from one checkpoint each. **Plan before any more training:** a small trainer patch (0006) that saves a checkpoint at every validation, so I can score the top 3 by validation F1 and average them. That's about 830 MB per checkpoint, about 25 GB for a 12,000 run, deleted after scoring.

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

**F1 and solving use different cutoffs.** Training's F1 counts a portal as "used" when the network is at least **50%** sure, but solving only uses portals it's at least **95%** sure about (`nn_exp.threshold: 0.95`). For our model, portals in the 50–90% range make trees worse ([details](#the-threshold-experiment-done)). So F1 partly grades guesses that never help the final tree. That may be part of why F1 tracks the score so loosely.

---

## Compute and hardware

### My Mac is faster than a Colab T4 for our workloads (2026-09-28, Andrew)

Same smoke test on both machines:

| Step | Mac (M5 Max) | Colab T4 |
|---|---|---|
| Training, 4 epochs | 37 s | 59 s |
| Data generation, 110 examples | about 13 s | about 100 s |

Data generation is CPU-heavy (GeoSteiner labels every example), and the Colab VM has only 2 CPU cores. So I train and generate data on my Mac.

Measured training speed on the Mac: about 49 s per epoch on 1,200 examples, 7.9 min on 12,000, and about 22.5 min on 36,000 (all at batch 50). Paper-style at batch 500: about 57 s per epoch on 12,000, and 8–10 min on 120,000.

### Full-scale training may not fit in Colab's memory (2026-09-28, Andrew)

The authors' trainer loads the **entire training set into memory** before training. The 1,200 set is 189 MB, so the paper's 120,000 examples would be about **19 GB**. That fits on a 64 GB Mac but likely not on free Colab (about 12–13 GB of RAM). At the measured speeds, one epoch on 120,000 would take roughly **80 minutes** on my Mac.

**Why it matters:** the full dataset size decides where training can happen. Update (Oct 9): the data-scaling study says 120,000 isn't worth it (36,000 scored no better than 12,000), so this limit no longer blocks us. A 12,000 set is 1.9 GB on disk. Update (Oct 19): paper-style examples are about half that size (12,000 = 984 MB), so 120,000 of them is about 10 GB, and at batch 500 an epoch of 120,000 takes about 8–9 min. That's Experiment 5. Update (Oct 20): it ran on my Mac with no memory problems: 9.6 GB on disk, about 22.5 h for 140 epochs ([details](#the-papers-full-120000-examples-done)).

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
| `0006-keep-numbered-checkpoints` | Training only keeps the best-by-F1 and the latest checkpoint, so a long run's history is lost | New setting `train.keep_every` (0 = off, the default): every N epochs, at validation time, also save `train/model/nnArora_ep<epoch>.pt`. N must be a multiple of `train.eval` |
| `0007-data-gen-level-kb` | `data_gen.constraint: level` (one shared tree shape, the paper's way of making training data) crashed with `Key 'kb' is not in struct`: it read `kb` from the `data_gen` settings, where it doesn't exist | Reads it from the `quadtree` settings. Without this, the paper's training data can't be generated from the released code |

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
| **Batch size must equal examples per tree shape.** Examples sharing a quadtree shape are stored next to each other | Set `train.batch_size` to `data_gen.batch / data_gen.num_trees` (50 for the scaling sets), or batches mix shapes. With `constraint: level` (paper-style) every example shares one shape, so any batch size works |
| **The `test acc` printed at the end of training is a narrow check.** It only scores the first test batch (a single tree shape), it uses the **last** checkpoint (`nnArora_best.pt`), not the best, and "acc" is really F1 | Judge models by validation F1 during training, and by % longer than optimal in the end |
| **A stopped run can't be resumed exactly.** `train.checkpoint=<file>` starts a **new** run from those weights: epoch numbers, numbered checkpoints, TensorBoard, Adam's momentum and the best-F1 tracking all start over, so its first `model saved` line always prints | Stop right after a numbered checkpoint is written. Add the starting epoch to every number in the new run, and compare its best with the old run's yourself |
| **`tee` overwrites its log file** | Give each part of a restarted run a new log name, or use `tee -a` |
| **The default threshold (0.95) isn't every model's best** | Sweep it per model, picked on the 200-point set: `scripts/score_checkpoint.sh` |
| **In `flow=eval`, only `predict_cost` and `final_cost` depend on the model.** `adapted_cost` and `processed_cost` are built from the correct portals | `final_cost` is the model's real result |
| **Hydra can't read `=` inside a command-line value** | No `=` in file names. That's why the authors' `m=15_kb=4.pt` is saved as `m15_kb4.pt`. Also put quotes around overrides with brackets: `'flow=[nn_exp]'` |
| **Python 3.14 breaks Hydra** | Use Python 3.10–3.13 |
| **Colab clones the repo from GitHub** | It only sees pushed changes |
| **Mac and Colab differ in the last digits** | Normal: GPUs round differently. Compare models on the same machine |
| **Upstream's `data_gen.constraint: level` mode crashes** (`Key 'kb' is not in struct`) | Fixed by patch 0007. Re-run `scripts/setup.sh` after pulling |
| **The quadtree box is always 10,000 × 10,000** (`quadtree.bbox`), even when `data_gen` makes points on a 100 × 100 canvas | Default training data puts all points in one corner, so its quadtrees (9–11 levels) look nothing like the test problems' (2–4 levels). Set `data_gen.x_range` / `y_range` to 10000 for test-shaped data |

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

## The threshold experiment (done)

Finished Oct 10. **Result: the default threshold (0.95) is already our best, so our score stays 3.38%.** But the experiment showed where the gap to the authors is: in what our network learned.

**The question:** the threshold is how sure the network must be before a portal is used (default 0.95). It was chosen for the authors' model. Ours may be less sure of itself, so a different threshold might suit it better and close part of the gap. It costs no training: about 2 minutes per threshold on the 100-point set, 3.5 on the 200-point set.

**Fairness rule:** picking the best threshold on the 100 test problems would be tuning on the exam. So I picked it on a second set, the authors' 100 problems with **200** points each (`upstream/points/point200_10000x10000-uniform-100-pt/`, answer file `200_10000x10000-uniform-golden.txt`), and report the 100-point score at that threshold. The rule, fixed in advance: lowest average on the 200-point set wins.

**Results** (our 36,000 model; the authors' columns come from their own result files):

| Threshold | 0.1 | 0.3 | 0.5 | 0.6 | 0.7 | 0.8 | 0.9 | **0.95** | 0.975 | 1.0 |
|---|---|---|---|---|---|---|---|---|---|---|
| Ours, 100-point set | 4.69% | 4.28% | 3.92% | 3.81% | 3.71% | 3.62% | 3.46% | **3.38%** | 3.42% | 3.77% |
| Authors', 100-point set | 3.74% | – | – | 2.60% | 2.62% | 2.66% | 2.69% | **2.65%** | 2.76% | 3.72% |
| Ours, 200-point set | 5.34% | 4.61% | 4.20% | 4.08% | 4.01% | 3.88% | 3.66% | **3.53%** | 3.56% | 3.76% |
| Authors', 200-point set | 4.13% | 3.50% | – | 3.28% | 3.24% | 3.16% | 3.14% | **3.06%** | 3.11% | 3.70% |

(The authors' files also have 0.2 and 0.4: 100-point 0.4 → 2.70%; 200-point 0.2 → 3.79%, 0.4 → 3.49%.)

**What I found:**

1. **0.95 wins on the 200-point set, for both models.** For ours, 0.975 ties with it (+0.03 ± 0.05 points) and 0.9 is clearly worse (+0.13 ± 0.05). So the reported score stays **3.38%**. On the 100-point set the authors' best looks like 0.6, but on 200 points it's 0.95: the best threshold on any one set of 100 problems is partly luck.
2. **A "no network" baseline.** At threshold 1.0 almost no portal passes, so the tree comes mostly from the cleanup step. Our model and theirs score almost the same there (100-point set: 3.77% vs 3.72%, +0.05 ± 0.03 points, half the trees identical). It isn't exactly "no network": the solver keeps portals with confidence ≥ the threshold, and a few round to exactly 100%.
3. **Our network adds about a third of what theirs does:**

   | Set | Our network's gain over 1.0 | Authors' network's gain over 1.0 |
   |---|---|---|
   | 100-point | **0.39 ± 0.11 points** (helps on 64 of 100) | **1.07 ± 0.10 points** (86 of 100) |
   | 200-point | **0.24 ± 0.08 points** (62 of 100) | **0.63 ± 0.07 points** (84 of 100) |

   The gap at 0.95 is smaller on 200 points (0.47 ± 0.09 vs 0.74 ± 0.12), but so is their network's gain, so ours isn't catching up.
4. **Our curve isn't flat like theirs.** Their model stays at 2.60–2.76% from 0.4 to 0.975 on the 100-point set. Ours gets steadily worse below 0.9. At 0.6–0.7 it's no better than no network at all on 100 points (+0.04 / −0.06 ± 0.12). On 200 points it's clearly worse than no network (0.7: +0.25 ± 0.10; 0.6: +0.31 ± 0.10), while theirs beats no network down to 0.3.
5. **Repeatable, and comparable with the authors' files.** Our 0.95 run gave exactly 3.3836 again. Their model, run by me, matches their files: 3.7193% at 1.0 (all 100 trees identical) and 2.6103% vs their 2.6029% at 0.6 (95 of 100 identical, +0.01 ± 0.01 points). That's the same tiny difference as our original reproduction, probably a few portals right at the cutoff rounding differently on my Mac than on their GPU.

**Why it matters:** the threshold isn't why we're behind; the network is. Our network's confident guesses (95%+) help the trees, but adding its 60–90% guesses doesn't, while adding the authors' does. That points at **how our network was trained**. (It's not simply that our mid-confidence portals are wrong more often; see [Comparing the two networks directly](#comparing-the-two-networks-directly-done).) It also gives the report a clean way to state the gap: their network improves on "no network" by 1.07 points, ours by 0.39.

---

## Comparing the two networks directly (done)

### On our validation data, the authors' network guesses portals worse than ours (2026-10-10, Andrew)

I graded the portal guesses of the authors' model and our three scaling models on our validation set with `scripts/val_f1.py` ([how it works](#scriptsval_f1py-grade-portal-guesses-directly-about-7-s-per-model)). For our models it reproduces training's logged validation F1 exactly, so it grades the same way.

| Model | Val F1 (0.5 cutoff) | Precision / recall at 0.95 | Share really used, band 0.6–0.7 / 0.95–1.0 | Trees on the 100 test problems |
|---|---|---|---|---|
| Authors' | 0.291 | 0.35 / 0.23 | 12.8% / 34.7% | **2.65%** |
| Ours, 36,000 | **0.459** | **0.52 / 0.37** | 19.5% / 52.2% | 3.38% |
| Ours, 12,000 | 0.427 | 0.49 / 0.31 | 20.1% / 49.2% | 3.38% |
| Ours, 1,200 | 0.226 | 0.25 / 0.07 | 13.5% / 24.9% | 3.77% |

- The authors' model scores a **lower** F1 than our 12,000 and 36,000 models, and its portals are less often right in **every** confidence band. Yet its trees are much shorter.
- F1 ranks our own models sensibly (they were all trained on the same kind of data), but not theirs.

**Why it matters:** F1 on our validation data can't explain the gap to the authors; it even points the wrong way. Two possible reasons:
1. **Our validation set is home ground for our network and unfamiliar ground for theirs.** It's made the same way as our training data, which is shaped very differently from the test problems ([next finding](#our-training-examples-are-shaped-very-differently-from-the-test-problems-2026-10-10-andrew)). If the authors trained on test-shaped examples, their network would look worse here and better on the test.
2. **F1 is too strict.** The answer key marks only the portal nearest to where one perfect tree crosses a cell edge. A neighbouring portal, or a different but equally short tree, counts as fully wrong, though the cleanup step can use it just as well.

**Update (same day):** reason 1 checks out. On test-shaped validation data the authors' network wins clearly ([below](#on-test-shaped-data-the-ranking-flips-the-authors-network-wins-2026-10-10-andrew)).

It also corrects an easy misreading of the threshold experiment: our 60–90% portals don't make trees worse because they're "wrong more often than the authors'". On this data, the authors' are wrong even more often.

### Our training examples are shaped very differently from the test problems (2026-10-10, Andrew)

- Our examples scatter 180 points over coordinates 0–99, but the quadtree's box is always 10,000 × 10,000 (`quadtree.bbox` in the authors' config). So every quadtree first zooms about 6 levels into the bottom-left corner, adding 3 empty cells per level, before reaching the points.
- Result: our training and validation quadtrees are **9–11 levels deep** (157–185 cells; 50 examples checked). The 100-point test problems spread over the whole box: **2–4 levels deep** (61–89 cells; all 100 checked). The 200-point set: 3–4 levels.
- The network's structure follows the quadtree, and its `top` part only handles the whole area. In our training data, the lines splitting the whole area are never crossed (all points sit in one corner), so `top` only ever learned "not used" for those portals. The answer keys confirm it: 0 used portals on those lines in all 128 corner-shaped validation examples, vs 3–12 (average 7.3) in all 128 test-shaped ones. In the test problems, those lines are always crossed.
- The authors' README example uses the same 100 × 100 canvas and 10,000 × 10,000 box, so I can't tell yet whether they trained this way too.

**Why it matters:** this is the strongest lead so far for the 0.74-point gap. It's a classic distribution shift: our network learned from examples that don't look like the problems it's tested on. The next finding tests it.

### On test-shaped data, the ranking flips: the authors' network wins (2026-10-10, Andrew)

I made a small validation set shaped like the test problems: 128 examples, 100 points spread over the whole 10,000 × 10,000 box (about 88 left after fitting each example to its tree shape), quadtrees 2–3 levels deep, seed 7200. Then I graded both networks on it with `scripts/val_f1.py`.

| Model | Val F1, our corner-shaped set | Val F1, **test-shaped set** | Precision / recall at 0.95, test-shaped | Share really used, bands 0.5–0.95 / 0.95–1.0, test-shaped |
|---|---|---|---|---|
| Authors' | 0.291 | **0.373** | **0.43 / 0.29** | **23–27% / 42.8%** |
| Ours, 36,000 | **0.459** | 0.252 | 0.31 / 0.16 | 14–22% / 31.1% |

- On data shaped like ours, our network wins. On data shaped like the test problems, the authors' wins clearly, in every confidence band.
- It also makes the threshold experiment make sense: on test-shaped data, the authors' 0.5–0.95 portals are right about 1 time in 4, ours about 1 in 6. That fits their mid-confidence portals helping the trees and ours not.

**Why it matters:** strong evidence that the shape of our training data is a big part of the 0.74-point gap. Our network learned corner-shaped problems; the test problems aren't corner-shaped. **Next:** train on 12,000 test-shaped examples (`configs/shape.yaml`) and score it on the 100 test problems.

**Update (Oct 18):** that run didn't close the gap ([next section](#training-on-test-shaped-examples-done)). **Update (Oct 20):** 120,000 paper-style examples did ([details](#the-papers-full-120000-examples-done)).

---

## Training on test-shaped examples (done)

### 12,000 test-shaped examples: 3.62%, no better than corner-shaped (2026-10-18, Andrew)

Run folder `work/outputs/2026-10-18/01-58-52`, config `shape`, `train.epochs=60`. It's the 12,000 corner-shaped run with one change: the points spread over the whole 10,000 × 10,000 box (`data_gen.x_range` and `y_range` = 10000). Same 180 points per example (about 165 after fitting each example to its tree shape), same seeds (100000+), 240 tree shapes of 50 examples each. Validation (seed 7200) and the end-of-training test set (seed 8200) are test-shaped too, 100 points each.

- **The data now has the right shape.** The training quadtrees are 3–5 levels deep (185 of 240 shapes have 3 levels, 54 have 4, 1 has 5), vs 9–11 for the corner-shaped set. The 100-point test problems are 2–4.
- **Training was short.** Early stopping ended it at epoch 22 after 1 h 55 min (about 5.3 min per epoch vs 7.9 for corner-shaped, since shallower trees have fewer cells). The best validation F1 was **0.280 at epoch 12** (2,880 learning steps, vs 10,560 for the corner-shaped run). After that it stayed at 0.26–0.28 while training F1 kept climbing (0.38 → 0.51), so the rest was memorizing.
- On the same test-shaped validation set, 0.280 sits between our 36,000 model (0.252) and the authors' (0.373). It was picked as the best of 12 checks on that set, so it's slightly flattering.
- **Score: 3.621%.**
  - vs corner-shaped 12,000 (3.38%): **+0.24 ± 0.10 points**, worse on 60 of 100 problems.
  - vs corner-shaped 36,000 (3.38%): +0.24 ± 0.10, worse on 57 of 100.
  - vs the authors' model (2.65%): **+0.97 ± 0.10**, they win 80 of 100.
  - vs the no-network runs (threshold 1.0): 0.15 ± 0.05 better than our 36,000 model's (3.77%), 0.10 ± 0.06 better than the authors' (3.72%). So this network adds very little.

**Why it matters:** at the default threshold, fixing the shape didn't help. The fix did what it should (the trees look like the test problems now), but the score didn't improve. It may be slightly worse, but 0.24 is within the 0.3 points a single checkpoint can move ([details](#which-checkpoint-you-score-moves-the-score-by-up-to-03-points-2026-10-09-andrew)), so the safe reading is "no better". **Update (same day):** the next finding shows the solver barely uses this network at 0.95, so a threshold sweep has to come first. Other leads:
1. **The authors trained differently in some other way** (points per example, number of examples, epochs, learning rate, data settings). The paper should say.
2. **12,000 test-shaped examples may be too few.** The run peaked after only 12 epochs and then just memorized, the classic sign of too little data. Corner-shaped examples are all much alike, so 12,000 was enough for them; test-shaped ones vary far more. The authors' model saw 120,000.

I skipped scoring its last checkpoint: the gap to the authors is 0.97, far above checkpoint luck, so it can't change the conclusion.

### On the test-shaped test set, the new network is good but under-confident (2026-10-18, Andrew)

I graded portal guesses with `scripts/val_f1.py` on the test-shaped test set (seed 8200: 64 examples, 82,688 portals, 3.8% really used). Unlike the validation set, it never picked a checkpoint, so it's the fairest F1 comparison.

| Model | F1 (0.5 cutoff) | Portals kept at 0.95 | Precision / recall at 0.95 | Share really used, bands 0.8–0.9 / 0.9–0.95 / 0.95–1.0 |
|---|---|---|---|---|
| Authors' | **0.347** | 2,060 | 0.42 / **0.28** | 23.6% / 26.3% / 42.1% |
| **Test-shaped 12,000** | 0.277 | **274** | 0.43 / 0.04 | 25.8% / 30.0% / 42.7% |
| Corner-shaped 12,000 | 0.236 | 1,363 | 0.29 / 0.13 | 16.6% / 20.7% / 29.2% |
| Corner-shaped 36,000 | 0.233 | 1,660 | 0.29 / 0.15 | 15.5% / 20.9% / 29.2% |

- **Where the new network is confident, it's as often right as the authors'**, in every band from 0.8 up, and clearly better than our corner-shaped models.
- **But it's rarely 95% sure.** At the solving threshold it keeps only 274 portals and finds 3.7% of the real ones, vs 28% for the authors'. So the solver mostly works without it, which is why its trees score close to the no-network baseline.
- Its best F1 cutoff is 0.7 (0.299). At 0.8 it keeps 3,101 portals and finds 28% of the real ones (the authors' recall at 0.95), though fewer are right (28% vs 42%).
- The two corner-shaped models score the same F1 here (0.236 vs 0.233), which matches their tied trees (3.38% each).

**Why it matters:** the 3.62% result says more about the threshold than about the shape. 0.95 was the right cutoff for our corner-shaped model ([details](#the-threshold-experiment-done)), but this network is less sure of itself, maybe because it stopped after only 2,880 learning steps. Its 0.8–0.95 portals are about as often right as the authors', and the authors' portals in that range help the trees. So I swept its threshold (next finding). The rule, set in advance: if its best threshold beats 3.38% by a real margin (0.3+ points), the shape helps; if not, it doesn't help much.

### Threshold sweep for the test-shaped model: 0.8 wins, 3.28% (2026-10-18, Andrew)

Same rule as the first threshold experiment: the lowest average on the 200-point set picks the threshold, then I report the 100-point score at that threshold.

| Threshold | 0.5 | 0.6 | 0.7 | 0.75 | **0.8** | 0.85 | 0.9 | 0.95 | 1.0 |
|---|---|---|---|---|---|---|---|---|---|
| 200-point set (picks) | 4.51% | 3.88% | 3.49% | 3.36% | **3.28%** | 3.40% | 3.49% | 3.70% | 3.78% |
| 100-point set (reports) | 4.20% | 3.77% | 3.40% | 3.34% | **3.28%** | 3.34% | 3.61% | 3.62% | 3.78% |

- **0.8 is lowest on both sets.** On the 100-point set its neighbours tie with it (0.75 and 0.85: +0.06 ± 0.08 each). The 0.95 run repeated 3.621 exactly.
- **Reported score: 3.282%.**
  - vs corner-shaped 12,000 (3.38%): **−0.10 ± 0.12 points**, shorter on 52 of 100. A tie.
  - vs corner-shaped 36,000 (3.38%): −0.10 ± 0.12, shorter on 60 of 100. Also a tie.
  - vs the authors' model (2.65%): **+0.63 ± 0.11**, they win 72 of 100 (was 0.74).
- **On the 200-point set the gain is clearer:** 3.28% vs our 36,000 model's 3.53% (−0.25 ± 0.09, shorter on 60), and the gap to the authors' 3.06% shrinks from 0.47 to 0.22 ± 0.08. That set also picked both models' thresholds, which flatters both a little.
- **The network now adds more.** Over its own no-network score (threshold 1.0): 0.50 ± 0.11 points on the 100-point set and 0.49 ± 0.06 on the 200-point set. Our corner-shaped 36,000 model: 0.39 and 0.24. The authors': 1.07 and 0.63. So on the 200-point set our network now gives about 80% of the authors' network's benefit, up from about 40%.

**Why it matters:** once the threshold suits the model, the shape helps a little, but it isn't the big fix. By the rule set in advance it's a tie on the 100 test problems, and the gap to the authors there is still 0.63 points. Test-shaped data stays our default (it's the setup that makes sense, and it's no worse; update Oct 20: paper-style data with 120,000 examples is now our best setup, [details](#the-papers-full-120000-examples-done)), and **every new model gets its own threshold sweep**, since models trained differently are sure of themselves in different ways. Next: read the paper's training details ([next section](#how-the-paper-trained-its-model)).

---

## How the paper trained its model

### The paper trained about 4,000× longer than our best checkpoint (2026-10-18, Andrew)

From the paper ([arXiv 2312.10589](https://arxiv.org/abs/2312.10589), Section 3.2 "Training" and Section 4 "Implementation"), next to our test-shaped run:

| Setting | The paper | Our test-shaped run | Same? |
|---|---|---|---|
| Training examples | **120,000** | 12,000 | 10× fewer |
| Points per example | 180 uniform, trimmed to ≤4 per leaf cell (about 160 left) | Same (about 165 left) | Yes |
| Canvas and quadtree box | 100 × 100 grid; the box must equal the canvas (checked in the code, below) | 10,000 × 10,000, box = canvas | Same idea: points fill the whole box |
| Tree shapes | **One:** a full depth-3 quadtree for every example (64 leaf cells; `constraint: level`, `level: 3`) | 240 shapes (`constraint: batch`) | Different |
| Batch size | **5,000** | 50 | 100× smaller |
| Epochs | **5,000** | Best at 12, stopped at 22 | Different |
| Learning steps | **120,000** | 2,880 to the best checkpoint | 40× fewer |
| Examples processed | **600 million** | about 144,000 to the best checkpoint | **about 4,000× fewer** |
| Validation / early stopping | Not mentioned | Stop after 10 epochs with no new best validation F1 | Different |
| `m`, `kb`, optimizer, learning rate, dropout, network size, loss weight, labels, threshold | 15, 4, Adam, 1e-4, 0.1, 4-layer MLPs with 4,096 hidden, m+1 = 16 for used portals, nearest portal to each crossing of GeoSteiner's tree, 0.95 | Same | Yes |
| Hardware | 3 × A100 80 GB | M5 Max Mac | – |

- **Their training data was test-shaped too.** I built paper-style examples with the authors' code (180 points on a 100 × 100 canvas, a full depth-3 structure, trimmed to ≤4 points per leaf). With a 100 × 100 quadtree box they give exactly the full depth-3 tree (85 cells). With the default 10,000 × 10,000 box, the same points zoom 9–10 levels deep. So the paper's "depth 3" means the box was the canvas. (The code logs that tree as "level 2". It's about the size of the test problems' trees, 61–89 cells.)
- **The README's toy example (`level3_100x100`) is a small copy of the real recipe.**
- **Even our 36,000 run** (40,320 steps of 50, about 2 million examples processed) trained 300× less than the paper.
- **Not stated:** validation set, early stopping, training time.

**Why the recipe matters:** the settings that shape the model all match. What differs is mostly **how much training** it gets: each of their examples was seen about 5,000 times, ours about 12 times. Our test-shaped network being rarely 95% sure ([details](#on-the-test-shaped-test-set-the-new-network-is-good-but-under-confident-2026-10-18-andrew)) fits being under-trained. And our early stopping ends runs when validation F1 flattens, but F1 is only a loose guide to tree quality, so it may stop runs long before the trees stop improving. Copying the recipe exactly isn't possible on my Mac: at our measured speed (about 38 examples a second at batch 50), 600 million examples would take about half a year. I tested that with a scaled-down copy: on 12,000 examples, much longer training doesn't help ([below](#scoring-the-long-run-longer-training-on-12000-examples-doesnt-help-2026-10-19-andrew)). On 120,000 it reproduces their result with 36× less training ([details](#the-papers-full-120000-examples-done)).

### A paper-style run is set up and tested (2026-10-18, Andrew)

- **Patch 0007:** upstream's `level` data mode, the paper's way of making training data, crashed in the released code (`Key 'kb' is not in struct`). A one-line fix.
- **Patch 0006:** `train.keep_every=N` keeps `nnArora_ep<epoch>.pt` every N epochs, so a long run's history survives.
- **`configs/paper.yaml`:** one full depth-3 tree for every example (`data_gen.constraint: level`, `level: 3`), points over the whole 10,000 × 10,000 box, batch size 500, no early stopping (`max_no_update: 100000`), validation every 10 epochs, a numbered checkpoint every 50. Same validation and test sets as `shape`.
- **Checks:** the smoke test still ends with `test acc: 0.11926605504587157`, with `keep_every` off and with `keep_every=2` (which kept `nnArora_ep2.pt` and `nnArora_ep4.pt`). All 7 patches pass `setup.sh`'s already-applied check, and 0006 / 0007 apply cleanly on top of 0002 / 0005 from scratch. A tiny paper-style set (20 examples, about 164 points each) gave the same 85-cell tree for all 20, and 2 epochs of training on it at batch 10 worked and kept numbered checkpoints. At the 10,000 scale the full depth-3 tree comes out without any extra setting.

**Why it matters:** we can now copy the paper's recipe, scaled down. It's also worth a sentence in the report: the released code couldn't generate the paper's training data as published.

### Paper-style data and speed test: batch 500 for 1,600 epochs (2026-10-18, Andrew)

**Data:** `work/data/level3_10000x10000-uniform-12000-tensor`, 12,000 examples, seed 100000 (the same seeds as our other 12,000 sets), made in 2 min 46 s. About 164 points per example (mean 163.8, stdev 3.8), 984 MB. GeoSteiner printed two "divide by zero … trying to recover" warnings and recovered; nothing stalled.

**Speed test:** 1 epoch at each batch size (12,000 examples, plus one validation). No memory errors.

| Batch size | Seconds per epoch | Examples a second | Steps per epoch | In about 22 h |
|---|---|---|---|---|
| 50 | 220 | 55 | 240 | 350 epochs = 84,000 steps, 4.2 million examples |
| 250 | 66 | 181 | 48 | 1,150 epochs = 55,200 steps, 13.8 million |
| **500** | **49** | **247** | **24** | **1,600 epochs = 38,400 steps, 19.2 million** |
| 1,000 | 44 | 273 | 12 | 1,800 epochs = 21,600 steps, 21.6 million |

- **Batch 500, not 1,000:** 1,000 is only 10% faster per example but takes half as many learning steps. With a fixed learning rate each step moves the weights by about the same amount, so 500 gets about 1.8× more learning in the same time. Below 500, too much speed is lost. (24 steps per epoch also matches the paper: 120,000 ÷ 5,000.)
- At batch 50 these one-shape examples train 1.4× faster than our test-shaped ones (55 vs 38 a second), since their trees are smaller.
- **The planned run** (1,600 epochs at batch 500) processes about 130× the examples and 13× the learning steps of our test-shaped model's best checkpoint. That's still only about 3% of the paper's examples and a third of its steps.

### Interim: the long run is memorizing (2026-10-19, Andrew)

Run folder `work/outputs/2026-10-18/23-16-42`, checked at epoch 609 of 1,600 (about 57 s per epoch).

| Epoch | 50 | 110–120 | 200 | 400 | 600 |
|---|---|---|---|---|---|
| Training F1 | 0.40 | about 0.6 | 0.86 | 0.96 | **0.98** |
| Validation F1 | 0.22 | **0.2415 (peak)** | 0.20 | 0.19 | 0.18 |

**Why it matters:** classic overfitting. The network's 207 million weights can memorize 12,000 examples, and validation F1 falls as it does. F1 isn't tree length, so this doesn't settle the question, but the numbered checkpoints already on disk (epochs 50–600, plus `nnArora.pt` from epoch 120) can. So I stopped it at epoch 611 (the last numbered checkpoint is epoch 600) and will score epochs 50, 120, 200, 400 and 600 with `scripts/score_checkpoint.sh`. A stopped run can't be resumed exactly: `train.checkpoint` starts a new run from the weights, with the epoch count, Adam's momentum and the best-F1 tracking reset.

### Scoring the long run: longer training on 12,000 examples doesn't help (2026-10-19, Andrew)

Five checkpoints of the long run, each scored with `scripts/score_checkpoint.sh`: its own threshold, picked on the 200-point set, then the score to report on the 100 test problems.

| Checkpoint | Training F1 | Validation F1 | Picked threshold | 200-point set | **100-point set** |
|---|---|---|---|---|---|
| Epoch 50 | 0.40 | 0.22 | 0.8 | **3.47%** | **3.45%** |
| Epoch 120 (`nnArora.pt`, best validation F1) | about 0.6 | **0.2415** | 0.9 | 3.51% | 3.54% |
| Epoch 200 | 0.86 | 0.20 | 0.9 | 3.62% | 3.63% |
| Epoch 400 | 0.96 | 0.19 | 0.6 (lowest tried) | 3.60% | 3.51% |
| Epoch 600 | 0.98 | 0.18 | 0.9 | 3.65% | 3.68% |
| *Test-shaped model, for comparison* | | *0.280* | *0.8* | *3.28%* | *3.28%* |

- **The trees get worse as it memorizes.** Epoch 600 vs epoch 50: +0.22 ± 0.11 points on the 100 test problems (epoch 50 shorter on 56 of 100), +0.18 ± 0.08 on the 200-point set. On the 200-point set the climb is almost steady: 3.47 → 3.51 → 3.62 → 3.60 → 3.65.
- **The best-validation-F1 checkpoint (epoch 120) is no better than epoch 50:** +0.09 ± 0.09 and +0.04 ± 0.07, a tie.
- **Every checkpoint is behind the test-shaped model.** The best (epoch 50) by +0.17 ± 0.11 on 100 and +0.19 ± 0.07 on 200; the test-shaped one is shorter on 61 of 100 on both. Epoch 120 had exactly as many learning steps (2,880) as the test-shaped model's best checkpoint, with 10× bigger batches, and is still +0.26 ± 0.12 / +0.23 ± 0.08 behind. So at 12,000 examples, one tree shape with batch 500 does worse than 240 shapes with batch 50. This run can't say which of the two is to blame.
- **vs the authors':** +0.81 ± 0.09 at best (they win 84 of 100).
- Epoch 400's best threshold was the lowest one tried (0.6). I didn't extend it: it was 0.13 behind epoch 50 there, its scores moved less than 0.1 per step near 0.6, and epochs 200 and 600 show the decline anyway.

**Why it matters:** "train much longer" was the leading suspect, and on 12,000 examples it's ruled out. The network learns them by heart and the trees get worse, as validation F1 said they would. Both of our whole-box runs (test-shaped and paper-style) memorized their 12,000 examples early, which points at the number of examples: the paper had 120,000. Paper-style data is cheap to make (12,000 in under 3 minutes), so the next test is the paper's full 120,000 ([result](#the-papers-full-120000-examples-done)). Also for the report: validation F1 found the right region here (its peak and the best trees both came early), though not the single best checkpoint.

---

## The paper's full 120,000 examples (done)

### 120,000 paper-style examples: 2.73%, a tie with the authors' model (2026-10-20, Andrew)

**What:** the long paper-style run (`configs/paper.yaml`) again, with the paper's own number of examples: 120,000 instead of 12,000. Everything else is the same: one full depth-3 tree for every example, batch 500 (240 learning steps per epoch), no early stopping, and validation plus a numbered checkpoint every 5 epochs, for 140 epochs. The data (seed 100000, the same seed as the 12,000 set) is 9.6 GB and fits in memory.

**Training:** about 22.5 h on my Mac, 8–10 min per epoch. It ran in three parts because I paused it twice. Each part restarted from the last numbered checkpoint with `train.checkpoint` (see the note at the end).

| Part | Run folder | Overall epochs | Started from |
|---|---|---|---|
| 1 | `2026-10-19/15-15-43` | 0–40 (stopped at 42) | random weights |
| 2 | `2026-10-20/00-11-40` | 40–120 (stopped at 123) | part 1's `nnArora_ep40.pt` |
| 3 | `2026-10-20/15-31-53` | 120–140 | part 2's `nnArora_ep80.pt` |

**Scores:** six checkpoints, each through `scripts/score_checkpoint.sh`: its own threshold picked on the 200-point set, then scored once on the 100 test problems. The checkpoint to report is also picked on the 200-point set, where epoch 140 is the lowest.

| Overall epoch | File | Training F1 | Validation F1 | Picked threshold | 200-point set | **100-point set** |
|---|---|---|---|---|---|---|
| 5 | part 1 `nnArora_ep5.pt` | 0.32 | 0.267 | 0.8 | 3.59% | 3.62% |
| 40 | part 1 `nnArora_ep40.pt` | 0.54 | 0.387 | 0.9 | 3.05% | 2.74% |
| 80 | part 2 `nnArora_ep40.pt` | 0.66 | 0.416 | 0.9 | 3.09% | 2.82% |
| 110 | part 2 `nnArora_ep70.pt` | 0.71 | 0.4265 | 0.9 | 3.04% | 2.76% |
| 135 | part 3 `nnArora_ep15.pt` (best validation F1) | 0.75 | **0.430** | 0.9 | 2.97% | 2.75% |
| **140** | part 3 `nnArora_ep20.pt` (**picked**) | 0.76 | 0.417 | 0.9 | **2.95%** | **2.73%** |
| *Authors' model* | | | *0.373* | *0.95* | *3.06%* | *2.65%* |
| *Our previous best (test-shaped, 12,000)* | | | *0.280* | *0.8* | *3.28%* | *3.28%* |

All validation F1s are on the same test-shaped validation set (seed 7200).

- **vs the authors' model: a tie.** +0.08 ± 0.09 points on the 100 test problems, with ours shorter on 52 and theirs on 48. Per problem, ours ranges from 0.62% to 5.26% longer than optimal (theirs 1.03% to 5.22%). On the 200-point set ours is a little shorter (−0.12 ± 0.08, 59 of 100), but that set picked our checkpoint and threshold, which flatters us there. Their 0.95 is also what the 200-point set picks for their model ([details](#the-threshold-experiment-done)), so both thresholds were chosen the same way.
- **vs our previous best (test-shaped 12,000, 3.28%): −0.55 ± 0.11** (ours shorter on 69 of 100), and −0.33 ± 0.08 on the 200-point set. **vs the 12,000 paper-style run (3.45%): −0.72 ± 0.09** (84 of 100).
- **What the network adds:** over the authors' no-network run (threshold 1.0, 3.72%), ours is −0.99 ± 0.10 (shorter on 85 of 100). Theirs adds 1.07. So ours now gives about 93% of their network's benefit, up from about a third (corner-shaped) and a half (test-shaped).
- **More examples don't speed up early learning.** Epoch 5 has 1,200 learning steps, the same as the 12,000 run's best checkpoint (epoch 50). It scores 3.62% vs 3.45% (+0.17 ± 0.10), no better. The gain comes with more training: epoch 40 is 0.89 ± 0.11 better than epoch 5.
- **It levels off after about epoch 40.** On the 100 test problems, epochs 40–140 all tie (2.73–2.82%; epoch 140 vs 40: −0.01 ± 0.10). The 200-point set shows a slow further gain (3.05% → 2.95%; epoch 140 vs 40: −0.10 ± 0.07). So most of the benefit came in the first 6 h of training.
- **No memorizing.** Training F1 rose slowly to 0.76, and validation F1 to 0.43, staying near its peak to the end. With 12,000 examples, training F1 hit 0.98 and validation F1 fell.
- **Checkpoint luck is small here.** The last five checkpoints are within 0.09 points of each other on the 100 test problems, so picking a different one wouldn't change the conclusion.
- **Its door guesses are now better than the authors'**, yet the trees only tie. On the test-shaped test set (seed 8200), F1 is 0.413 vs 0.347. At each model's own threshold it keeps about as many portals as theirs (2,337 at 0.9 vs 2,060 at 0.95), and more of them are right (precision 0.448 vs 0.421). That fits point 5 of [What the results mean](#what-the-results-mean): F1 only roughly predicts tree length.

**Why it matters:** this answers the project's main open question. The 0.63-point gap between our best model and the authors' was the number of training examples. With the paper's 120,000, our own training reproduces the authors' result (2.73% vs 2.65%, a tie), with about 36× less training than the paper (16.8 million examples processed vs 600 million). The extra examples didn't make the network learn faster. They let it train much longer without memorizing. By the rule set in advance (0.3+ better than 3.28%), this is the "the number of examples was the gap" outcome. Caveats: it's one training run with one seed, and so far only 100-point uniform problems. Bigger problems and other point patterns are next (Shraddha).

**Restarting a stopped run:** a run can't be resumed exactly. `train.checkpoint=<file>` starts a new run from those weights, so the epoch count, numbered checkpoints, TensorBoard, Adam's momentum and the best-F1 tracking all start over. Part 2's `nnArora_ep<N>.pt` is overall epoch N + 40, and part 3's is N + 120. Each restart's first validation matched the checkpoint it came from (0.3868 and 0.4217), so the weights loaded correctly. Adam's reset cost one slightly worse epoch (training loss 0.278 → 0.297, then down again). Pausing loses the epochs after the last numbered checkpoint (2 and 3 epochs here).

---

## What comes next

**Now that Experiments 1–5 are done** (data scaling, threshold, data shape, long training, and the paper's full 120,000 examples), in this order:
1. **Hand the 120,000 model to Shraddha** as `m15_kb4_n120000.pt` (part 3's `nnArora_ep20.pt`, epoch 140), with its settings, its training curves (three run folders; I'll stitch them into one) and **threshold 0.9**. It ties the authors' model on 100-point problems, so it's the one to test on bigger problems (500–5,000 points) and other point patterns.
2. **Writing:** the scaling plot (training examples on a log scale vs % longer than optimal, with dashed lines at the authors' 2.65% and the "no network" 3.77%), now with the 120,000 point, and its report paragraph. Plus my one-page architecture summary with a diagram.
3. **Experiment 6, if there's time: `portal_weight`.** One run of the 120,000 recipe with `train.portal_weight=0` (no extra weight for "used" portals), about 20 h, scored the same way. Only the loss changes. It shows whether the paper's weighting for the class imbalance (3.1% of portals used) matters.

**Later, if there's time** (the authors' result files cover the first three, so we can check against theirs):
- `kb` (max points per leaf cell): 1, 4 or 7. Needs new data.
- `m` (portals per cell side): 3, 7 or 15. Needs new data.
- Train on uniform points, test on other point patterns. Overlaps with Shraddha's baselines.
- 120,000 test-shaped examples: does the tree shape still matter once there's enough data?

**Report:** I lead Methodology & model, and help with Dataset and Experimental setup.

---

## Our scripts, and what each one does

**Which one do I run?**

| I want to… | Run |
|---|---|
| Set up a new machine, or update after pulling new patches | `scripts/setup.sh` |
| Check that nothing is broken | `scripts/smoke_test.sh`, then `scripts/arora.sh pretrained_solve` |
| Get the authors' trained model | `scripts/get_pretrained.sh` |
| Make data, train, solve or score anything | `scripts/arora.sh <config> [settings...]` |
| Grade models' portal guesses (F1, precision, recall) without solving | `python scripts/val_f1.py <model.pt> [more models...]` |
| Score a trained model on the 100 test problems, threshold picked on the 200-point set | `scripts/score_checkpoint.sh <model.pt> <name> [thresholds...]` |
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
| `scale` | The data-scaling study: clean validation and test sets, batch size 50, early stopping, and the 100 test problems. The training set and epoch cap go on the command line |
| `shape` | Same as `scale`, but points spread over the whole 10,000 × 10,000 box, with test-shaped validation and test sets (Experiment 3) |
| `paper` | The paper's recipe, scaled down: one full depth-3 tree for every example, points over the whole box, batch size 500, no early stopping, a numbered checkpoint every 50 epochs. Same validation and test sets as `shape`. **Use this one for new training** (Experiments 4 and 5). The epoch count goes on the command line; for 120,000 examples I also set `train.eval=5 train.keep_every=5` |

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

### `scripts/val_f1.py`: grade portal guesses directly (about 7 s per model)

`nn_exp` grades a model by the **trees** it builds. This script grades the **portal guesses themselves**, before any tree is built: the same grading training does every 2 epochs, plus more detail.

```bash
python scripts/val_f1.py work/models/pretrained/m15_kb4.pt work/outputs/2026-10-08/11-43-16/train/model/nnArora.pt
python scripts/val_f1.py --data work/data/<folder> --batch-size <examples per tree shape> <model.pt> ...   # another dataset
```

**What it does:**
1. Loads the dataset (default: our validation set, `batch4_100x100-uniform-128-tensor`, 128 examples) in batches of 32, one tree shape per batch, exactly like training.
2. For each model: builds the network from the authors' `conf/config.yaml`, loads the checkpoint (DDP or not), and switches to eval mode (dropout off, no learning).
3. Runs every batch through the network, giving a probability for each of the 376,448 portals (3.1% of them used in the answer key).
4. At each cutoff (0.5, 0.6, …, 0.95), a portal counts as "picked" if its probability is at least the cutoff. It counts TP (picked and used), FP (picked, not used) and FN (used, not picked), then computes:
   - **precision** = TP / (TP + FP): of the portals picked, the share that are right
   - **recall** = TP / (TP + FN): of the right portals, the share that were picked
   - **F1** = 2·TP / (2·TP + FP + FN)
5. Sorts portals into confidence bands (0.5–0.6, …, 0.95–1.0) and reports the share in each band that's really used.

**What it prints:**

| Line | Meaning |
|---|---|
| `portals: 376,448, truly used: 3.1%` | Portals graded, and the share used in the answer key (same for every model) |
| `val F1 as training logs it` | Mean of the 4 per-batch F1s at 0.5. Exactly training's `val/f1`: our 36,000 / 12,000 / 1,200 models give 0.4591 / 0.4268 / 0.2264, matching training's logs |
| Cutoff table | F1, precision, recall with all batches pooled (so 0.5 can differ in the 4th digit), and how many portals pass |
| Band table | How many portals fall in each band, and the share of those really used (precision within the band) |

**The confidence numbers aren't real probabilities.** Training weights "used" portals 16× (`portal_weight=15`), which pushes every probability up. With that weighting, a perfectly trained network would say 0.95 for portals that are used about 54% of the time (0.9 → 36%, 0.8 → 20%, 0.5 → 6%). Compare models band by band, not against the band's label.

**Limits:** it grades against **one** perfect tree's portals, on the dataset you give it. It doesn't build trees, so it doesn't replace `nn_exp` scoring.

### `scripts/score_checkpoint.sh`: score a model the fair way (about 20 min per model)

```bash
scripts/score_checkpoint.sh <model.pt> <name> [thresholds...]
scripts/score_checkpoint.sh outputs/2026-10-18/23-16-42/train/model/nnArora_ep200.pt paper-ep200
```

1. **Solves the 200-point set** at each threshold (default 0.6, 0.7, 0.8, 0.9, 0.95) and grades each against the perfect lengths.
2. **Picks the threshold with the lowest average there.** Picking on the 200-point set keeps the 100 test problems out of the choice, the same rule as the threshold experiments.
3. **Solves the 100 test problems at that threshold only** and prints the score to report.

- The model path is relative to `work/` (a leading `work/` is fine).
- Results go to `work/exp_out/thr200-<name>-t<t>-solved.txt` and `thr-<name>-t<t>-solved.txt`, with each solve's log next to them (`.log`). Finished files are reused, so an interrupted run can just be started again.
- If the pick is the lowest or highest threshold tried, it prints a note: the best may lie beyond the list, so try more (pass them as extra arguments).
- Check: `scripts/score_checkpoint.sh outputs/2026-10-18/01-58-52/train/model/nnArora.pt shape12000 0.7 0.75 0.8 0.85` reuses the test-shaped model's sweep and prints `picked threshold 0.8` and `3.282%`.

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

**Grade portal guesses directly** (no solving, about 7 s per model):
```bash
python scripts/val_f1.py work/models/pretrained/m15_kb4.pt work/outputs/2026-10-08/11-43-16/train/model/nnArora.pt \
  work/outputs/2026-09-29/17-07-18/train/model/nnArora.pt work/outputs/2026-09-29/16-01-59/train/model/nnArora.pt
```
For our three models it should print `val F1 as training logs it` of 0.4591, 0.4268 and 0.2264; for the authors' model, 0.2906.

**Test-shaped validation data** (100 points over the whole 10,000 × 10,000 box, like the test problems). For the authors' model it prints 0.3730, for our 36,000 model 0.2519:
```bash
scripts/arora.sh scale 'flow=[data_gen]' data_gen.seed=7200 data_gen.num_points=100 data_gen.x_range=10000 data_gen.y_range=10000 data_gen.num_trees=4 data_gen.batch=128
python scripts/val_f1.py --data work/data/batch4_10000x10000-uniform-128-tensor work/models/pretrained/m15_kb4.pt work/outputs/2026-10-08/11-43-16/train/model/nnArora.pt
```

**Experiment 3: the test-shaped run** (`configs/shape.yaml`; the validation set above already exists). The last line prints `average error percentage of 100 cases: 3.621`:
```bash
scripts/arora.sh shape 'flow=[data_gen]' data_gen.seed=8200 data_gen.num_points=100 data_gen.num_trees=2 data_gen.batch=64     # test set
scripts/arora.sh shape 'flow=[data_gen]' data_gen.seed=100000 data_gen.num_trees=240 data_gen.batch=12000                       # training set
caffeinate -i scripts/arora.sh shape train.train_set=data/batch240_10000x10000-uniform-12000-tensor train.epochs=60
scripts/arora.sh shape 'flow=[nn_exp]' nn_exp.model=outputs/2026-10-18/01-58-52/train/model/nnArora.pt nn_exp.output=exp_out/shape-12000
python third_party/NN-Steiner/evaluator/evaluateRatio.py work/exp_out/shape-12000-solved.txt third_party/NN-Steiner/exp_out/100_10000x10000-uniform-golden.txt
```
Training's run folder name is its start time, so yours will differ from `2026-10-18/01-58-52`.

**Experiment 4: the paper-style run** (`configs/paper.yaml`). Make the data, run the speed test (batch 50 / 250 / 500 / 1,000 should take about 220 / 66 / 49 / 44 s per epoch on an M5 Max), then the long run (batch 500 is the config's default):
```bash
scripts/arora.sh paper 'flow=[data_gen]' data_gen.seed=100000 data_gen.batch=12000
for b in 50 250 500 1000; do
  echo "== batch $b"
  scripts/arora.sh paper train.batch_size=$b train.epochs=1 train.eval=1 train.keep_every=0 2>&1 \
    | tr '\r' '\n' | grep -E "1/1 \[|Error|out of memory"
done
caffeinate -i scripts/arora.sh paper train.epochs=1600
```
Numbered checkpoints land in `<run folder>/train/model/nnArora_ep<epoch>.pt`. I stopped mine at epoch 611 (run folder `2026-10-18/23-16-42`) and score its checkpoints like this:
```bash
R=outputs/2026-10-18/23-16-42/train/model
scripts/score_checkpoint.sh $R/nnArora_ep50.pt  paper-ep50
scripts/score_checkpoint.sh $R/nnArora.pt       paper-ep120    # best validation F1
scripts/score_checkpoint.sh $R/nnArora_ep200.pt paper-ep200
scripts/score_checkpoint.sh $R/nnArora_ep400.pt paper-ep400
scripts/score_checkpoint.sh $R/nnArora_ep600.pt paper-ep600
```
The scores to report (100-point set) are 3.4543% (epoch 50, threshold 0.8), 3.5449% (120, 0.9), 3.6314% (200, 0.9), 3.5149% (400, 0.6) and 3.6762% (600, 0.9).

**Experiment 5: the paper-style run with 120,000 examples** (about 30 min for the data, about 22.5 h to train). Data generation ends with `points summary: mean: 163.84...`:
```bash
scripts/arora.sh paper 'flow=[data_gen]' data_gen.seed=100000 data_gen.batch=120000
caffeinate -i scripts/arora.sh paper train.train_set=data/level3_10000x10000-uniform-120000-tensor \
  train.epochs=140 train.eval=5 train.keep_every=5
```
I paused mine twice and continued each time from the last numbered checkpoint, with only the epochs left (each is a new run folder, and its epochs restart at 0):
```bash
caffeinate -i scripts/arora.sh paper train.train_set=data/level3_10000x10000-uniform-120000-tensor \
  train.checkpoint=outputs/2026-10-19/15-15-43/train/model/nnArora_ep40.pt train.epochs=100 train.eval=5 train.keep_every=5
caffeinate -i scripts/arora.sh paper train.train_set=data/level3_10000x10000-uniform-120000-tensor \
  train.checkpoint=outputs/2026-10-20/00-11-40/train/model/nnArora_ep80.pt train.epochs=20 train.eval=5 train.keep_every=5
```
Scoring, named by overall epoch:
```bash
P1=outputs/2026-10-19/15-15-43/train/model    # part 1: epoch N = N
P2=outputs/2026-10-20/00-11-40/train/model    # part 2: epoch N = N + 40
P3=outputs/2026-10-20/15-31-53/train/model    # part 3: epoch N = N + 120
scripts/score_checkpoint.sh $P1/nnArora_ep5.pt  p120k-ep5
scripts/score_checkpoint.sh $P1/nnArora_ep40.pt p120k-ep40
scripts/score_checkpoint.sh $P2/nnArora_ep40.pt p120k-ep80
scripts/score_checkpoint.sh $P2/nnArora_ep70.pt p120k-ep110
scripts/score_checkpoint.sh $P3/nnArora_ep15.pt p120k-ep135
scripts/score_checkpoint.sh $P3/nnArora_ep20.pt p120k-ep140
```
The 200-point scores at the picked thresholds are 3.5937% (epoch 5, 0.8), 3.0507% (40, 0.9), 3.0862% (80, 0.9), 3.0446% (110, 0.9), 2.965% (135, 0.9) and **2.9474% (140, 0.9, the lowest)**. The scores to report (100-point set) are 3.6219%, 2.7365%, 2.8155%, 2.7614%, 2.7463% and **2.7304%**.

**Threshold sweep for the test-shaped model** (pick on the 200-point set, report on the 100-point set). It should print 3.2819 at 0.8 on the 200-point set (the lowest), and 3.282 at 0.8 and 3.621 at 0.95 on the 100-point set:
```bash
M=outputs/2026-10-18/01-58-52/train/model/nnArora.pt
E=third_party/NN-Steiner/exp_out
for t in 0.5 0.6 0.7 0.75 0.8 0.85 0.9 0.95 1.0; do
  scripts/arora.sh shape 'flow=[nn_exp]' nn_exp.model=$M nn_exp.threshold=$t \
    nn_exp.test_set=upstream/points/point200_10000x10000-uniform-100-pt/ nn_exp.output=exp_out/thr200-shape12000-t$t
  python $E/../evaluator/evaluateRatio.py work/exp_out/thr200-shape12000-t$t-solved.txt $E/200_10000x10000-uniform-golden.txt
done
for t in 0.5 0.6 0.7 0.75 0.8 0.85 0.9 0.95 1.0; do
  scripts/arora.sh shape 'flow=[nn_exp]' nn_exp.model=$M nn_exp.threshold=$t nn_exp.output=exp_out/thr-shape12000-t$t
  python $E/../evaluator/evaluateRatio.py work/exp_out/thr-shape12000-t$t-solved.txt $E/100_10000x10000-uniform-golden.txt
done
```

**Grade portal guesses on the test-shaped test set** (seed 8200, 32 examples per tree shape; it never picked a checkpoint, so it's the fairest F1 comparison):
```bash
python scripts/val_f1.py --data work/data/batch2_10000x10000-uniform-64-tensor --batch-size 32 \
  work/models/pretrained/m15_kb4.pt work/outputs/2026-09-29/17-07-18/train/model/nnArora.pt \
  work/outputs/2026-10-08/11-43-16/train/model/nnArora.pt work/outputs/2026-10-18/01-58-52/train/model/nnArora.pt
```
It prints `val F1 as training logs it` of 0.3470 (authors'), 0.2360 (12,000), 0.2329 (36,000) and 0.2768 (test-shaped 12,000). Adding the 120,000 model (`work/outputs/2026-10-20/15-31-53/train/model/nnArora_ep20.pt`) gives 0.4126.

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
For the 12,000 model vs the authors' model it prints `A: 3.38%  B: 2.65%  A-B: +0.74 ± 0.10 points  A shorter on 24, B shorter on 76 of 100`. With model A = `thr-p120k-ep140-t0.9-solved.txt` (the 120,000 model) it prints `A: 2.73%  B: 2.65%  A-B: +0.08 ± 0.09 points  A shorter on 52, B shorter on 48 of 100`.

**The threshold experiment** (about 2 min per threshold on the 100-point set, 3.5 on the 200-point set):
```bash
M=outputs/2026-10-08/11-43-16/train/model/nnArora.pt    # our 36,000 model
E=third_party/NN-Steiner/exp_out
# 100-point set (the usual test problems)
for t in 0.1 0.3 0.5 0.6 0.7 0.8 0.9 0.95 0.975 1.0; do
  scripts/arora.sh scale 'flow=[nn_exp]' nn_exp.model=$M nn_exp.threshold=$t nn_exp.output=exp_out/thr-36000-t$t
  python $E/../evaluator/evaluateRatio.py work/exp_out/thr-36000-t$t-solved.txt $E/100_10000x10000-uniform-golden.txt
done
# 200-point set (used to pick the threshold)
for t in 0.1 0.3 0.5 0.6 0.7 0.8 0.9 0.95 0.975 1.0; do
  scripts/arora.sh scale 'flow=[nn_exp]' nn_exp.model=$M nn_exp.threshold=$t \
    nn_exp.test_set=upstream/points/point200_10000x10000-uniform-100-pt/ nn_exp.output=exp_out/thr200-36000-t$t
  python $E/../evaluator/evaluateRatio.py work/exp_out/thr200-36000-t$t-solved.txt $E/200_10000x10000-uniform-golden.txt
done
```
The authors' spot-check is the 100-point loop with `M=models/pretrained/m15_kb4.pt`, thresholds `0.6 1.0`, and output `exp_out/thr-pretrained-t$t`. Their own threshold files are `$E/100_10000x10000-uniform-m=15_kb=4-threshold=<t>-solved.txt` (and `200_...` for the 200-point set).

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
| `2026-10-18/01-58-52` | **Test-shaped training: 12,000 examples** (3.28% at threshold 0.8) |
| `2026-10-18/23-16-42` | **Paper-style long run: 12,000 examples, batch 500**, stopped at epoch 611 (memorizing); numbered checkpoints every 50 epochs. Best trees at epoch 50 (3.45%) |
| `2026-10-19/15-15-43` | **120,000 paper-style examples, part 1:** epochs 0–40 (paused at 42) |
| `2026-10-20/00-11-40` | **Part 2:** from part 1's `nnArora_ep40.pt`; overall epochs 40–120 (its `ep<N>` is epoch N + 40; paused at 123) |
| `2026-10-20/15-31-53` | **Part 3:** from part 2's `nnArora_ep80.pt`; overall epochs 120–140 (its `ep<N>` is epoch N + 120). Its `nnArora_ep20.pt` (epoch 140) is our best model: **2.73%** at threshold 0.9 |

Inside a training run: `train/` holds the TensorBoard curves, and `train/model/nnArora.pt` is the best model by validation F1. For runs with numbered checkpoints (`nnArora_ep<N>.pt`) I pick by scoring instead (`score_checkpoint.sh`).

**Score files** (`work/exp_out/`, one total wire length per test problem):

| File | Model |
|---|---|
| `scale-pretrained-solved.txt` | Authors' model (2.65%) |
| `scale-1200-solved.txt` | 1,200 examples (3.77%) |
| `scale-12000-solved.txt` | 12,000 examples (3.38%) |
| `scale-36000-solved.txt` | 36,000 examples (3.38%) |
| `shape-12000-solved.txt` | 12,000 test-shaped examples at threshold 0.95 (3.62%) |
| `thr-shape12000-t<threshold>-`, `thr200-shape12000-t<threshold>-solved.txt` | Threshold sweep of the test-shaped model on the 100- and 200-point sets (best: 0.8, 3.28% on both) |
| `thr-paper-ep<N>-t<threshold>-`, `thr200-paper-ep<N>-t<threshold>-solved.txt` | The long paper-style run's checkpoints (epochs 50, 120, 200, 400, 600), from `score_checkpoint.sh` (best: epoch 50 at 0.8, 3.45%) |
| `thr-p120k-ep<N>-t<threshold>-`, `thr200-p120k-ep<N>-t<threshold>-solved.txt` | The 120,000 run's checkpoints (overall epochs 5, 40, 80, 110, 135, 140), from `score_checkpoint.sh` (picked: epoch 140 at 0.9, 2.73%) |
| `scale-1200-last-`, `scale-12000-last-`, `scale-36000-ep60-solved.txt` | Each run's **last** checkpoint (4.00%, 3.53%, 3.70%) |
| `check-refine-on/off-100-uniform-solved.txt` | Authors' model with cleanup on / off |
| `check-medium30-refine-on/off-100-uniform-solved.txt` | My 30-epoch model with cleanup on / off |
| `thr-36000-t<threshold>-solved.txt` | Threshold experiment: our 36,000 model on the 100-point set, thresholds 0.1–1.0 |
| `thr200-36000-t<threshold>-solved.txt` | Same, on the 200-point set |
| `thr-pretrained-t0.6-`, `thr-pretrained-t1.0-solved.txt` | Authors' model at thresholds 0.6 and 1.0 (2.61%, 3.72%) |

**Datasets** (`work/data/`):

| Folder | What | Size on disk |
|---|---|---|
| `batch4_100x100-uniform-128-tensor` | Validation, seed 5200 | 21 MB |
| `batch2_100x100-uniform-64-tensor` | Test, seed 6200 | 11 MB |
| `batch24_100x100-uniform-1200-tensor` | Training, 1,200 | 189 MB |
| `batch240_100x100-uniform-12000-tensor` | Training, 12,000 | 1.9 GB |
| `batch720_100x100-uniform-36000-tensor` | Training, 36,000 | 5.6 GB |
| `batch4_10000x10000-uniform-128-tensor` | Test-shaped validation, seed 7200, 100 points over the whole 10,000 × 10,000 box | – |
| `batch2_10000x10000-uniform-64-tensor` | Test-shaped test, seed 8200, 100 points | – |
| `batch240_10000x10000-uniform-12000-tensor` | Test-shaped training, 12,000, seed 100000, 180 points | – |
| `level3_10000x10000-uniform-12000-tensor` | Paper-style training, 12,000, seed 100000: one full depth-3 tree for every example, about 164 points | 984 MB |
| `level3_10000x10000-uniform-120000-tensor` | Paper-style training, 120,000, same settings and seed | 9.6 GB |

**Files in the repo:**

| File | What it is |
|---|---|
| `scripts/setup.sh`, `common.sh`, `arora.sh`, `smoke_test.sh`, `get_pretrained.sh`, `val_f1.py`, `score_checkpoint.sh` | Setup, the run launcher, and helpers. Details in [Our scripts](#our-scripts-and-what-each-one-does) |
| `configs/smoke.yaml`, `medium.yaml`, `pretrained_solve.yaml`, `scale.yaml`, `shape.yaml`, `paper.yaml` | Settings for each kind of run (table in [Our scripts](#our-scripts-and-what-each-one-does)) |
| `notebooks/colab.ipynb` | The Colab entry point |
| `third_party/patches/0001–0007` | Our fixes to the authors' code |
| `docs/Project_Proposal.md`, `Team_Responsibilities.md`, this file | The proposal, the plan, and this record |

---

## Open questions

- **Does NN-Steiner beat FLUTE on bigger problems (500+ points) in our setup,** as the paper claims? And does our 120,000 model hold up there and on other point patterns, like the authors' does? (Shraddha, with the model I hand over)
- **Is our 2.73% repeatable?** It's one training run with one seed. A second seed would show how much run-to-run luck there is (about 20 h). (Andrew, if there's time)
- **Which more experiment do we commit to?** Five are done (data scaling, threshold, data shape, long training, the paper's full 120,000 examples). I propose `portal_weight` if there's time; see [What comes next](#what-comes-next). (Andrew proposes, the team agrees)
- **Does loading the latest checkpoint instead of the best one** change the authors' reported test numbers? (Andrew)

**Answered:**
- *Do 10× more examples close the gap?* Yes, for paper-style data: 120,000 examples score 2.73%, a tie with the authors' 2.65% (+0.08 ± 0.09) ([details](#the-papers-full-120000-examples-done)).
- *Do we reproduce the authors' pretrained results across all 100 test cases?* Yes, once the cleanup step is restored: 2.648% vs 2.646% ([details](#the-authors-model-matches-the-paper-once-a-switched-off-cleanup-step-is-restored-2026-09-29-andrew)).
- *Where does validation F1 level off?* On 1,200 examples, around epoch 16–20 (about 0.22 on the clean validation set). The old run's "still rising" 0.304 was on the leaky set. Bigger datasets level off later and higher.
- *Does the data trend continue at 36,000, or level off?* It levels off: 36,000 scores 3.38%, the same as 12,000 ([details](#data-scaling-study-36000-examples-338-no-better-than-12000-2026-10-09-andrew)).
- *How big should the full training set be?* For corner-shaped data, 12,000 is enough: 36,000 didn't help. For paper-style data, the paper's 120,000 closes the gap to the authors ([details](#the-papers-full-120000-examples-done)).
- *Does much longer training close the gap?* Not on 12,000 examples: the network memorizes them and the trees get a little worse (3.45% at epoch 50, 3.68% at epoch 600) ([details](#scoring-the-long-run-longer-training-on-12000-examples-doesnt-help-2026-10-19-andrew)). On 120,000 examples, yes: 3.62% at epoch 5, 2.74% by epoch 40 ([details](#the-papers-full-120000-examples-done)).
- *Does early stopping on validation F1 stop our runs too soon?* Not in the long run: the best trees came by epoch 50–120, around validation F1's peak, and got worse after.
- *Does a different threshold suit our model better?* For the corner-shaped model, no: 0.95 is its best on both the 100- and 200-point sets ([details](#the-threshold-experiment-done)). Later models, yes: 0.8 for the test-shaped model, 0.9 for the 120,000 model. So every model now gets its own sweep.
- *Do the authors' threshold result files match our setup?* Yes. Their model, run by me, matches at 0.6 and 1.0 ([details](#the-threshold-experiment-done)).
- *How did the authors train their model?* 120,000 examples, one full depth-3 tree shape on a 100 × 100 canvas, batches of 5,000, 5,000 epochs; everything else as ours ([details](#how-the-paper-trained-its-model)).
- *Does test-shaped training data close the gap?* A little. At its own best threshold (0.8) it scores 3.28%, a tie with 3.38% on the 100 test problems and a clearer gain on the 200-point set; the gap to the authors was still 0.63 ([details](#training-on-test-shaped-examples-done)).
- *Does the authors' network score a higher F1 than ours?* On our corner-shaped validation set, no (0.29 vs 0.46). On test-shaped validation data, yes (0.37 vs 0.25) ([details](#comparing-the-two-networks-directly-done)). Our 120,000 model beats it there (0.42 vs 0.37), yet the trees only tie ([details](#the-papers-full-120000-examples-done)).

---

## Words we use

| Word | Meaning |
|---|---|
| Portal | A fixed crossing point on a quadtree cell's edge (15 per side). The network predicts which portals the optimal tree passes through |
| Leaf cell / `kb` | A smallest quadtree cell / the most points allowed in one (4) |
| Cleanup (`refine_leaves`) | After the network's guess, re-solving small groups of points exactly with GeoSteiner. Patch 0004 switched it back on |
| % longer than optimal | How much more wire our tree uses than GeoSteiner's perfect tree. **The real score** (the authors call it "error") |
| Golden file | The file of perfect GeoSteiner lengths for the test problems: the answer key for scoring |
| Threshold | How sure the network must be (0.95 = 95%) before a portal is used when solving. Set with `nn_exp.threshold`. Only affects solving, so trying values needs no training |
| RMST | Rectilinear Minimum Spanning Tree: the shortest tree that only connects the points to each other directly, with horizontal/vertical wires and no extra junctions. Fast, typically around 10% longer than optimal. NN-Steiner builds one through the points plus the network's chosen portals, then cleans it up with GeoSteiner |
| No-network baseline | Solving with threshold 1.0: almost no portal passes, so the tree comes from RMST plus GeoSteiner cleanup alone (the same as the authors' commented-out `feasible_portals = []` ablation, except a few portals rounding to exactly 1.0). 3.72% for the authors' model, 3.77% for ours. Score at 0.95 minus this = what the network adds |
| F1 | How well the network guesses portals. A training signal, not the final score. Logs print it as "acc" |
| Epoch / learning step | One pass through the training data / one weight update from one batch (50 examples in most runs, 500 in paper-style runs) |
| Batch size | How many examples the network looks at before one weight update. Bigger batches run faster per example but give fewer steps per epoch |
| Early stopping | Ending training once validation F1 hasn't improved for 10 epochs |
| Ablation | Switching off one part of a method to see how much it matters |
| Data leak | Validation or test data sneaking into training, which makes scores look too good |
| Seed | The number that fixes "random" choices, so a run can be repeated exactly |
| `constraint: batch` / `level` | How `data_gen` picks tree shapes. `batch`: `num_trees` different shapes from random points (the scaling and test-shaped runs). `level`: one full quadtree `level` deep for every example (the paper's way, `level: 3`; our paper-style runs) |
| Precision / recall | Of the portals a model picks, the share that are right / of the right portals, the share it picks. F1 combines the two |
| Confidence band | A range of the network's output, like 0.6–0.7. The share of portals in a band that are really used shows what that confidence means in practice |
| Quadtree depth | How many times the area is split into 4 before every cell has at most `kb` points. Our corner-shaped training examples: 9–11. Test-shaped ones (`configs/shape.yaml`): 3–5. Paper-style ones: always 3 (85 cells). The 100-point test problems: 2–4 |
| Corner-shaped / test-shaped / paper-style data | Our three kinds of training data. **Corner-shaped:** the code's defaults, all points in one 100 × 100 corner of the 10,000 × 10,000 box (`configs/scale.yaml`). **Test-shaped:** points over the whole box, 240 tree shapes built from the points, batch 50 (`configs/shape.yaml`). **Paper-style:** points over the whole box, one fixed full depth-3 tree for every example, batch 500, no early stopping (`configs/paper.yaml`), the paper's recipe |
| Numbered checkpoint | `train/model/nnArora_ep<N>.pt`, saved every `train.keep_every` epochs (patch 0006), so a run's history survives |
| Restarting from a checkpoint | `train.checkpoint=<file>` starts a new run from saved weights. It isn't an exact resume: epochs, best-F1 tracking and Adam's momentum start over |
| 200-point set / 100-point set | The authors' two uniform test sets, 100 problems each. The 200-point set picks each model's threshold and checkpoint; the 100-point set is the exam we report, touched once |
| Distribution shift | When training examples look different from the problems a model is tested on. Models usually do worse on the unfamiliar kind |
