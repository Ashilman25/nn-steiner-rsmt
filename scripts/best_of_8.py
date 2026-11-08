#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import os
import re
import subprocess
import sys
import tempfile
import time
from typing import Dict, List, Optional

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
UPSTREAM = os.path.join(REPO, "third_party", "NN-Steiner")
WORK = os.path.join(REPO, "work")
ARORA = os.path.join(REPO, "scripts", "arora.sh")

# The authors' code and our lib/ have to be findable before importing from them
sys.path.insert(0, UPSTREAM)
sys.path.insert(0, os.path.join(REPO, "lib"))

from omegaconf import OmegaConf
from arora.fix import fix
from arora.quadtree import QuadTree, Rect, Terminal
from turns import TURN_NAMES, turn_point, turn_points


# Reads a point file like upstream: "x y" per line, stopping at a blank line
def read_point_file(path: str) -> List[tuple[int, int]]:
    points = []
    with open(path) as f:
        for line in f:
            parts = line.split()
            if not parts:
                break
            points.append((int(parts[0]), int(parts[1])))
    return points


# Writes turned copies of every point file into dst/turn0 .. turn<turns-1> - same file names
def turn_folder(src: str, dst: str, turns: int) -> None:
    names = sorted(os.listdir(src))
    for k in range(turns):
        folder = os.path.join(dst, f"turn{k}")
        os.makedirs(folder, exist_ok=True)
        stray = sorted(set(os.listdir(folder)) - set(names))

        if stray:
            raise ValueError(f"{folder} has files that aren't in {src}: {stray}. Delete them and rerun")

        for name in names:
            if k == 0:
                with open(os.path.join(src, name), "rb") as f:
                    data = f.read()
            else:
                points = turn_points(k, read_point_file(os.path.join(src, name)))
                data = "".join(f"{x} {y}\n" for x, y in points).encode()
            write_if_changed(os.path.join(folder, name), data, dst)


def write_if_changed(path: str, data: bytes, tmp_dir: str) -> None:
    if os.path.exists(path):
        with open(path, "rb") as f:
            if f.read() == data:
                return

    fd, tmp = tempfile.mkstemp(dir=tmp_dir, suffix=".tmp")
    with os.fdopen(fd, "wb") as f:
        f.write(data)
    os.replace(tmp, path)


#check if each turn's quadtree is the exact mirror image of the original's
def exact_turns(points: List[tuple[int, int]], turns: int = 8) -> List[bool]:
    args = OmegaConf.load(os.path.join(UPSTREAM, "conf", "config.yaml"))["quadtree"]
    width, height = args["bbox"]["width"], args["bbox"]["height"]
    points = list(dict.fromkeys(points))  # upstream drops repeated points

    def leaf_boxes(pts: List[tuple[int, int]]) -> List[tuple[float, ...]]:
        qt = QuadTree(Terminal.terminals_from_point(pts), Rect(fix(0), fix(0), fix(width), fix(height)), args)
        boxes = [qt.cells[c].bbox for c in qt.terminal2cell]
        return [(b.x_min.to_float(), b.y_min.to_float(), b.x_max.to_float(), b.y_max.to_float()) for b in boxes]

    def turn_box(k: int, box: tuple[float, ...]) -> tuple[float, ...]:
        (ax, ay), (bx, by) = turn_point(k, box[0], box[1], width), turn_point(k, box[2], box[3], width)
        return (min(ax, bx), min(ay, by), max(ax, bx), max(ay, by))

    base = leaf_boxes(points)
    return [leaf_boxes(turn_points(k, points)) == [turn_box(k, b) for b in base] for k in range(turns)]


# Each problem's shortest length among its first k turns
def best_of(lengths: List[List[float]], k: int) -> List[float]:
    return [min(row[:k]) for row in lengths]


# For each problem, every turn that ties for the shortest tree
def winners(lengths: List[List[float]]) -> List[List[int]]:
    return [[i for i, v in enumerate(row) if v == min(row)] for row in lengths]


# Compares two sets of trees problem by problem, in % over optimal
def compare(a: List[float], b: List[float], golden: List[float]) -> Dict[str, float]:
    a_pct = [(x / g - 1) * 100 for x, g in zip(a, golden)]
    b_pct = [(x / g - 1) * 100 for x, g in zip(b, golden)]

    d = [x - y for x, y in zip(a_pct, b_pct)]
    n = len(d)

    mean = sum(d) / n
    se = math.sqrt(sum((x - mean) ** 2 for x in d) / (n - 1) / n) if n > 1 else 0.0

    return {
        "a_pct": sum(a_pct) / n,
        "b_pct": sum(b_pct) / n,
        "diff": mean,
        "se": se,
        "shorter": sum(x < y for x, y in zip(a, b)),
        "same": sum(x == y for x, y in zip(a, b)),
        "longer": sum(x > y for x, y in zip(a, b)),
    }


# The solve time upstream logs ("Runtime: 109.627 secs"), or None if there isn't one
def read_runtime(log: str) -> Optional[float]:
    if not os.path.exists(log):
        return None

    with open(log, errors="replace") as f:
        found = re.findall(r"Runtime: ([\d.]+) secs", f.read())
    return float(found[-1]) if found else None


def read_lengths(path: str) -> List[float]:
    with open(path) as f:
        return [float(line) for line in f if line.strip()]


def line_count(path: str) -> int:
    if not os.path.exists(path):
        return -1
    return len(read_lengths(path))


def point_set(size: int) -> str:
    return f"point{size}_10000x10000-uniform-100-pt"


def golden_file(size: int) -> str:
    return os.path.join(UPSTREAM, "exp_out", f"{size}_10000x10000-uniform-golden.txt")


# Makes the turned copies of one test set; returns (turned root, number of problems)
def prepare(size: int, turns: int) -> tuple[str, int]:
    src = os.path.join(UPSTREAM, "points", point_set(size))
    if not os.path.isdir(src) or not os.path.exists(golden_file(size)):
        sys.exit(f"!! No uniform {size}-point test set with an optimal file in {UPSTREAM}")

    root = os.path.join(WORK, "points", "turned", point_set(size))
    turn_folder(src, root, turns)

    problems = len(os.listdir(src))
    if problems != line_count(golden_file(size)):
        sys.exit(f"!! {src} has {problems} problems but its golden file has {line_count(golden_file(size))}")
    return root, problems


# Which turned copies are exact mirror images, [problem][turn]; worked out once per set
def exact_table(size: int, root: str) -> List[List[bool]]:
    src = os.path.join(UPSTREAM, "points", point_set(size))
    names = sorted(os.listdir(src))
    path = os.path.join(root, "exact.csv")

    if os.path.exists(path):
        with open(path) as f:
            rows = [line.strip().split(",") for line in f][1:]
        if [r[0] for r in rows] == names:
            return [[x == "1" for x in r[1:]] for r in rows]

    print("checking which turned copies are exact mirror images (once per set)", flush=True)
    table = [exact_turns(read_point_file(os.path.join(src, name))) for name in names]
    with open(path, "w") as f:
        f.write("file," + ",".join(f"turn{k}" for k in range(8)) + "\n")
        f.writelines(name + "," + ",".join(str(int(x)) for x in row) + "\n" for name, row in zip(names, table))
    return table


# Starts scripts/arora.sh in the background, its output going to a log
def start_arora(overrides: List[str], log: str) -> subprocess.Popen:
    env = dict(os.environ)
    env.setdefault("PYTHON", sys.executable)
    with open(log, "w") as f:
        return subprocess.Popen([ARORA, "scale", *overrides], stdout=f, stderr=subprocess.STDOUT, env=env)


# Runs jobs (label, overrides, log) a few at a time. Ctrl-C stops the running ones too
def run_all(jobs: List[tuple[str, List[str], str]], workers: int, start=start_arora) -> None:
    begin = time.time()
    waiting, running, failed = list(jobs), [], []

    try:
        while waiting or running:
            while waiting and len(running) < workers:
                label, overrides, log = waiting.pop(0)
                running.append((label, log, start(overrides, log)))

            for job in list(running):
                label, log, proc = job
                if proc.poll() is None:
                    continue

                running.remove(job)
                if proc.returncode == 0:
                    print(f"{label} done ({time.time() - begin:.0f} s so far)", flush=True)
                    continue

                failed.append(label)
                with open(log, errors="replace") as f:
                    tail = "\n".join(f.read().splitlines()[-5:])

                print(f"!! {label} failed; last lines of {os.path.relpath(log, REPO)}:\n{tail}", file=sys.stderr, flush=True)
            time.sleep(0.2 if running else 0)
    except KeyboardInterrupt:
        for _, _, proc in running:
            proc.terminate()
        for _, _, proc in running:
            proc.wait()

        sys.exit("!! Stopped. Finished turns are kept, so the same command carries on from here")
    if failed:
        sys.exit(f"!! {len(failed)} solve(s) failed: {', '.join(failed)}")


def tries_to_report(turns: int) -> List[int]:
    return sorted({k for k in (1, 2, 4, 8) if k <= turns} | {turns})


def cmd_check(args: argparse.Namespace) -> None:
    os.makedirs(os.path.join(WORK, "exp_out"), exist_ok=True)
    all_ok = True

    for size in args.sizes:
        root, problems = prepare(size, args.turns)
        golden = read_lengths(golden_file(size))
        jobs = []
        outs = []

        for k in range(args.turns):
            out = f"exp_out/best8-check-{size}pt-turn{k}"
            outs.append(os.path.join(WORK, f"{out}-golden.txt"))
            if os.path.exists(outs[-1]):
                os.remove(outs[-1])

            test_set = os.path.relpath(os.path.join(root, f"turn{k}"), WORK) + "/"
            jobs.append((
                f"{size}-point turn {k} (GeoSteiner)",
                ["flow=[geo_exp]", f"geo_exp.test_set={test_set}", f"geo_exp.output={out}"],
                os.path.join(WORK, f"{out}.log")
            ))

        print(f"== check: {size}-point set, {args.turns} turns, GeoSteiner vs the optimal file", flush=True)
        run_all(jobs, args.jobs)
        for k, path in enumerate(outs):
            got = read_lengths(path)
            same = sum(x == g for x, g in zip(got, golden))
            ok = same == problems == len(got)
            all_ok &= ok
            print(f"turn {k} ({TURN_NAMES[k]}): {same}/{problems} lengths match{'' if ok else ' <== MISMATCH'}")
    print("check passed: every turned copy has exactly the optimal length" if all_ok else "!! check FAILED")
    sys.exit(0 if all_ok else 1)


def cmd_solve(args: argparse.Namespace) -> None:
    model = args.model[len("work/"):] if args.model.startswith("work/") else args.model
    if not os.path.exists(os.path.join(WORK, model)):
        sys.exit(f"!! No model at work/{model}")

    float(args.threshold)  # fail early on a typo
    os.makedirs(os.path.join(WORK, "exp_out"), exist_ok=True)

    for size in args.sizes:
        root, problems = prepare(size, args.turns)
        stem = f"exp_out/best8-{args.name}-thr{args.threshold}-{size}pt"
        solved = [os.path.join(WORK, f"{stem}-turn{k}-solved.txt") for k in range(args.turns)]
        logs = [os.path.join(WORK, f"{stem}-turn{k}.log") for k in range(args.turns)]
        jobs = []

        for k in range(args.turns):
            if line_count(solved[k]) != problems:
                test_set = os.path.relpath(os.path.join(root, f"turn{k}"), WORK) + "/"
                jobs.append((
                    f"{size}-point turn {k}",
                    ["flow=[nn_exp]", f"nn_exp.model={model}", f"nn_exp.threshold={args.threshold}",
                     f"nn_exp.test_set={test_set}", f"nn_exp.output={stem}-turn{k}"],
                    logs[k]
                ))

        print(f"== {args.name} ({model}), threshold {args.threshold}, {size}-point set ({problems} problems)",
              flush=True)
        if jobs:
            print(f"solving {len(jobs)} turned cop{'y' if len(jobs) == 1 else 'ies'}, {args.jobs} at a time", flush=True)
        run_all(jobs, args.jobs)
        report(size, problems, stem, solved, logs, args.turns, exact_table(size, root))


# Prints the scores, writes the combined trees and a per-problem table
def report(size: int, problems: int, stem: str, solved: List[str], logs: List[str], turns: int, exact: List[List[bool]]) -> None:
    golden = read_lengths(golden_file(size))
    per_turn = [read_lengths(p) for p in solved]
    for path, got in zip(solved, per_turn):
        if len(got) != problems:
            sys.exit(f"!! {path} has {len(got)} lines, not {problems}. Delete it and rerun")

    exact = [row[:turns] for row in exact]
    lengths = [list(row) for row in zip(*per_turn)]  # [problem][turn]
    single = best_of(lengths, 1)

    for k in range(turns):
        pct = compare(per_turn[k], per_turn[k], golden)["a_pct"]
        runtime = read_runtime(logs[k])
        print(f"turn {k} ({TURN_NAMES[k]}): {pct:.2f}%" + (f", {runtime:.0f} s" if runtime else ""))

    avg = sum(compare(t, t, golden)["a_pct"] for t in per_turn) / turns
    print(f"average single try: {avg:.2f}%")

    for k in tries_to_report(turns):
        best = best_of(lengths, k)
        r = compare(best, single, golden)
        line = f"best of {k}: {r['a_pct']:.2f}%"
        if k > 1:
            line += (f" (vs turn 0: {r['diff']:+.2f} +- {r['se']:.2f} points; shorter on {r['shorter']}, same on {r['same']} of {problems})")
            with open(os.path.join(WORK, f"{stem}-best{k}-solved.txt"), "w") as f:
                f.writelines(f"{x}\n" for x in best)
        print(line)

    wins = winners(lengths)
    spread = [(max(row) - min(row)) / g * 100 for row, g in zip(lengths, golden)]
    if turns > 1:
        all_exact = [i for i, row in enumerate(exact) if all(row)]
        print(f"exact mirror images: {sum(map(sum, exact))} of {turns * problems} turned copies; all {turns} exact on {len(all_exact)} of {problems} problems")
        invariance(lengths, spread, turns, list(range(problems)), f"all {problems} problems")
        if 0 < len(all_exact) < problems:
            invariance(lengths, spread, turns, all_exact, f"only the {len(all_exact)} with all {turns} exact")

    times = [read_runtime(log) for log in logs]
    if turns > 1 and all(times):
        print(f"solve time: {sum(times):.0f} s for {turns} tries ({sum(times) / turns:.0f} s per try)")

    names = sorted(os.listdir(os.path.join(UPSTREAM, "points", point_set(size))))
    table = os.path.join(WORK, f"{stem}.csv")
    with open(table, "w") as f:
        f.write("problem,file,optimal," + ",".join(f"turn{k}" for k in range(turns)) + ",best,winners,spread_pct,exact_turns\n")
        for i, (name, row, g, w, s, e) in enumerate(zip(names, lengths, golden, wins, spread, exact)):
            f.write(f"{i},{name},{g}," + ",".join(str(x) for x in row) + f",{min(row)},{' '.join(map(str, w))},{s:.4f},{' '.join(str(k) for k in range(turns) if e[k])}\n")
    print(f"trees: work/{stem}-best{turns}-solved.txt, table: work/{stem}.csv" if turns > 1 else f"table: work/{stem}.csv")


# Prints how much turning changed the trees, over some of the problems
def invariance(lengths: List[List[float]], spread: List[float], turns: int, rows: List[int], label: str) -> None:
    wins = winners([lengths[i] for i in rows])
    alone = [w[0] for w in wins if len(w) == 1]
    counts = ", ".join(f"turn {k}: {alone.count(k)}" for k in range(turns))
    part = [spread[i] for i in rows]
    print(f"{label}: all {turns} the same length on {sum(len(w) == turns for w in wins)}, one turn shortest on its own on {len(alone)}")
    print(f"{label}, wins by turn: {counts}")
    print(f"{label}, spread (longest minus shortest, % of optimal): average {sum(part) / len(part):.2f}, largest {max(part):.2f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--sizes", type=int, nargs="+", default=[100], help="uniform test sets by points per problem: 50 100 200 500 800 1000 (default 100)")
    common.add_argument("--turns", type=int, default=8, choices=range(1, 9), metavar="K", help="use only the first K turns (default 8)")
    common.add_argument("--jobs", type=int, default=1, help="turned copies to solve at once (default 1; more is faster but skews solve times)")

    check = sub.add_parser("check", parents=[common], help="GeoSteiner on the turned copies = optimal file")
    check.set_defaults(func=cmd_check)

    solve = sub.add_parser("solve", parents=[common], help="solve the turned copies with a model")
    solve.add_argument("model", help="model file, relative to work/ (e.g. models/m15_kb4_n120000.pt)")
    solve.add_argument("name", help="label for the output files, one per model (e.g. p120k)")
    solve.add_argument("threshold", help="the model's own threshold (ours 0.9, the authors' 0.95)")
    solve.set_defaults(func=cmd_solve)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
