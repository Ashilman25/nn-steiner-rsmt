#!/usr/bin/env python3
from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import sys
from typing import Dict, List, Tuple

from omegaconf import OmegaConf

import best_of_8 
from best_of_8 import TURN_NAMES, WORK, golden_file, prepare, read_lengths

from arora.flow.utils import get_quadtree, get_quadtreedata
from arora.points import read_points
from arora.solver import NNSteiner

CONF = OmegaConf.load(os.path.join(best_of_8.UPSTREAM, "conf", "config.yaml"))
QUADTREE, K, FST = CONF["quadtree"], CONF["nn_exp"]["k"], CONF["nn_exp"]["fst"]
MODELS = [("ours", "p120k-thr0.9"), ("authors'", "pretrained-thr0.95")]


# One problem's tree lengths without a network: correct doors, then no doors at all
def solve_without_network(path: str) -> Tuple[float, float]:
    points = read_points(path)
    data = get_quadtreedata(points, QUADTREE, FST)
    
    _, perfect = NNSteiner.build_stt(data.feasible_portals, data, K, FST)
    _, cleanup_only = NNSteiner.build_stt([], get_quadtree(points, QUADTREE), K, FST)
    
    perfect.check()
    cleanup_only.check()
    return perfect.length(), cleanup_only.length()


# Average % over optimal, average spread across turns, and problems where all turns tie
def summarize(lengths: List[List[float]], golden: List[float]) -> Dict[str, float]:
    pct = [[(x / g - 1) * 100 for x in row] for row, g in zip(lengths, golden)]
    return {
        "average": sum(map(sum, pct)) / sum(map(len, pct)),
        "spread": sum(max(r) - min(r) for r in pct) / len(pct),
        "all_same": sum(len(set(row)) == 1 for row in lengths),
    }


# Network-free lengths for every problem and turn, [problem][turn], saved to a CSV and reused
def no_network_lengths(size: int, root: str, problems: int, jobs: int) -> Tuple[List[List[float]], List[List[float]]]:
    table = os.path.join(WORK, "exp_out", f"orient-{size}pt.csv")
    if not os.path.exists(table) or len(open(table).readlines()) != problems * 8 + 1:
        names = sorted(os.listdir(os.path.join(root, "turn0")))
        paths = [os.path.join(root, f"turn{k}", name) for name in names for k in range(8)]
        print(f"solving {len(paths)} turned copies without a network, {jobs} at a time", flush=True)
        
        with mp.Pool(jobs) as pool:
            results = pool.map(solve_without_network, paths, chunksize=4)
            
        with open(table, "w") as f:
            f.write("problem,turn,perfect_doors,no_doors\n")
            for i, (perfect, cleanup_only) in enumerate(results):
                f.write(f"{i // 8},{i % 8},{perfect},{cleanup_only}\n")

    rows = [line.strip().split(",") for line in open(table).readlines()[1:]]
    perfect = [[float(r[2]) for r in rows[p * 8:p * 8 + 8]] for p in range(problems)]
    cleanup_only = [[float(r[3]) for r in rows[p * 8:p * 8 + 8]] for p in range(problems)]
    return perfect, cleanup_only


def main() -> None:
    parser = argparse.ArgumentParser(description="Does NN-Steiner's tree change with orientation even without the network? Solves every turned copy with the correct doors and with no doors.")
    parser.add_argument("--sizes", type=int, nargs="+", default=[100], help="uniform test sets (default 100)")
    parser.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) - 2), help="problems solved at once")
    args = parser.parse_args()

    for size in args.sizes:
        root, problems = prepare(size, 8)
        golden = read_lengths(golden_file(size))
        perfect, cleanup_only = no_network_lengths(size, root, problems, args.jobs)

        rows = [("correct doors (perfect network)", perfect), ("no doors (cleanup steps only)", cleanup_only)]
        for name, stem in MODELS:
            files = [os.path.join(WORK, "exp_out", f"best8-{stem}-{size}pt-turn{k}-solved.txt") for k in range(8)]
            if all(os.path.exists(f) for f in files):
                rows.append((f"{name} network", [list(r) for r in zip(*[read_lengths(f) for f in files])]))

        print(f"== {size}-point set, {problems} problems x {len(TURN_NAMES)} turns")
        for name, lengths in rows:
            r = summarize(lengths, golden)
            print(f"{name}: average {r['average']:.2f}%, spread {r['spread']:.2f} (longest minus shortest turn, % of optimal), all 8 the same on {r['all_same']} of {problems}")
        print(f"table: work/exp_out/orient-{size}pt.csv")


if __name__ == "__main__":
    main()
