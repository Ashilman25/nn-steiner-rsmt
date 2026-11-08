#!/usr/bin/env python3
#Checks a training set made with data_gen.turns

from __future__ import annotations

import argparse
import filecmp
import os
import re
import sys
from typing import Dict, List, Tuple

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
UPSTREAM = os.path.join(REPO, "third_party", "NN-Steiner")
WORK = os.path.join(REPO, "work")

sys.path.insert(0, UPSTREAM)
sys.path.insert(0, os.path.join(REPO, "lib"))

import torch  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402

from arora.data import get_transform  # noqa: E402
from arora.flow.dataGen import make_name, simplify_quadtree  # noqa: E402
from arora.flow.utils import get_golden_stt, get_quadtree, get_quadtreedata_like  # noqa: E402
from arora.points import SimpleQuadTree, get_pointGen, get_struct_quadtree  # noqa: E402
from arora.quadtree import QuadTree  # noqa: E402
from arora.utils import get_dtype  # noqa: E402
from turns import TURN_NAMES, sample_idx, turn_points  # noqa: E402

Point = Tuple[int, int]


# Upstream's config with ours on top, as Hydra would compose it
def load_config(name: str) -> Dict:
    base = OmegaConf.load(os.path.join(UPSTREAM, "conf", "config.yaml"))
    ours = OmegaConf.load(os.path.join(REPO, "configs", f"{name}.yaml"))
    ours.pop("defaults", None)
    return OmegaConf.to_container(OmegaConf.merge(base, ours))


# The one tree every example shares, built the way data_gen builds it
def shared_tree(data_args: Dict, quadtree_args: Dict) -> QuadTree:
    simple = SimpleQuadTree.get_level_quadtree(data_args["x_range"], data_args["y_range"], data_args["level"])
    point_gen = get_pointGen(data_args["dist"], data_args, data_args["seed"])
    points = get_struct_quadtree(point_gen.get_points(data_args["num_points"]), simple, quadtree_args["kb"])
    return get_quadtree(points, quadtree_args)


# A base problem's points before turning, as data_gen's worker makes them
def base_points(base: int, data_args: Dict, quadtree_args: Dict, tree: QuadTree) -> List[Point]:
    point_gen = get_pointGen(data_args["dist"], data_args, data_args["seed"] + base)
    points = point_gen.get_points(data_args["num_points"])
    return get_struct_quadtree(points, simplify_quadtree(tree), quadtree_args["kb"])


# Reads the points back out of a file's leaf cells (stored relative to each cell)
def points_in_file(tens: Dict[str, torch.Tensor], tree: QuadTree) -> set[Point]:
    points = set()
    for cell, row in zip(tree.cells, tens["terminal_tens"].tolist()):
        x0, y0, width = cell.bbox.x_min.to_float(), cell.bbox.y_min.to_float(), cell.bbox.width.to_float()
        for i in range(0, len(row), 2):
            if row[i] != -1:
                points.add((round(x0 + row[i] * width), round(y0 + row[i + 1] * width)))
                
    return points


def main() -> None:
    parser = argparse.ArgumentParser(description="Check a training set of turned copies.")
    parser.add_argument("--set", default="data/level3_10000x10000-uniform-96000-turns8-tensor", help="turned set, relative to work/")
    parser.add_argument("--plain", default="data/level3_10000x10000-uniform-12000-tensor", help="plain set with the same seed")
    parser.add_argument("--config", default="symmetry", help="config the turned set was made with")
    parser.add_argument("--seed", type=int, default=100000, help="data_gen.seed the set was made with")
    parser.add_argument("--bases", type=int, default=20, help="base problems to check labels on")
    args = parser.parse_args()

    match = re.search(r"-(\d+)-turns(\d+)-tensor/?$", args.set)
    if match is None:
        sys.exit(f"!! {args.set} isn't a turned set (expected ...-<batch>-turns<N>-tensor)")
    batch, turns = int(match.group(1)), int(match.group(2))
    num_base = batch // turns

    conf = load_config(args.config)
    data_args, quadtree_args = conf["data_gen"], conf["quadtree"]
    data_args["seed"] = args.seed
    name = make_name(data_args)
    turned_dir, plain_dir = os.path.join(WORK, args.set), os.path.join(WORK, args.plain)
    turned_file = lambda idx: os.path.join(turned_dir, f"{name}-{idx}.pt")
    plain_file = lambda idx: os.path.join(plain_dir, f"{name}-{idx}.pt")

    count = len(os.listdir(turned_dir))
    print(f"{args.set}: {count} files, {num_base} base problems x {turns} turns")
    if count != batch:
        sys.exit(f"!! expected {batch} files")

    # 1. turn 0 of base b vs file b of the plain set
    differ = [b for b in range(num_base) if not filecmp.cmp(turned_file(sample_idx(b, 0, num_base, turns)), plain_file(b), shallow=False)]
    print(f"turn-0 files identical to {args.plain}: {num_base - len(differ)} of {num_base}")
    if differ:
        print(f"differ: base problems {differ[:20]}{' ...' if len(differ) > 20 else ''}")
        print("(only expected for base problems the data_gen log says it redid after a stall)")

    # 2. labels on a few base problems spread over the set (skipping any redone ones)
    tree = shared_tree(data_args, quadtree_args)
    transform = get_transform(data_args["type"], get_dtype(data_args["precision"]))
    tree_struct = torch.load(plain_file(0))["tree_struct"]
    picks = [b for b in range(0, num_base, max(1, num_base // args.bases)) if b not in differ][: args.bases]
    n = data_args["x_range"] - 1
    failures = 0

    print(f"\nlabels on {len(picks)} base problems (portals used by GeoSteiner's tree, per turn):")
    print(f"{'base':>6} {'shortest':>9}  " + " ".join(f"t{k:<3}" for k in range(turns)) + "  check")
    for b in picks:
        points = base_points(b, data_args, quadtree_args, tree)
        lengths, used, problems = [], [], []
        for k in range(turns):
            turned = turn_points(k, points, n)
            saved = torch.load(turned_file(sample_idx(b, k, num_base, turns)))
            rebuilt = transform.transform(get_quadtreedata_like(turned, tree, data_args["fst"]))

            lengths.append(get_golden_stt(turned, data_args["fst"]).length())
            used.append(int(saved["golden"].sum()))
            if points_in_file(saved, tree) != set(turned):
                problems.append(f"t{k} points")
                
            if not torch.equal(saved["tree_struct"], tree_struct):
                problems.append(f"t{k} tree")
                
            if any(not torch.equal(saved[key], rebuilt[key]) for key in saved):
                problems.append(f"t{k} labels")
                
        if len(set(lengths)) != 1:
            problems.append(f"lengths {sorted(set(lengths))}")

        failures += bool(problems)
        print(f"{b:>6} {lengths[0]:>9.0f}  " + " ".join(f"{u:<4}" for u in used) + "  " + (", ".join(problems) or "ok"))

    print(f"\nturns: {', '.join(f't{k} = {TURN_NAMES[k]}' for k in range(turns))}")
    ok = not differ and failures == 0
    print("ALL CHECKS PASSED" if ok else f"!! {len(differ)} turn-0 mismatches, {failures} base problems with label problems")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
