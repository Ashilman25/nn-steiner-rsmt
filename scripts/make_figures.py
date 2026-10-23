from __future__ import annotations

import argparse
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
UPSTREAM = os.path.join(REPO, "third_party", "NN-Steiner")
WORK = os.path.join(REPO, "work")
RESULTS = os.path.join(REPO, "results")

# % longer than optimal on the authors' 100 uniform 100-point test problems 
DEFAULT_DATA = [(1_200, 3.77), (12_000, 3.38), (36_000, 3.38)]
TEST_SHAPED = [(12_000, 3.28)]
PAPER_SETUP = [(12_000, 3.45), (120_000, 2.73)]
AUTHORS = 2.65
NO_NETWORK = 3.72

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 10,
    "text.color": INK,
    "axes.labelcolor": INK_2,
    "axes.edgecolor": AXIS,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
})


# Score vs number of training examples, one line per kind of training data
def scaling_plot(path: str) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.2))

    for y, label, dy, va in [(AUTHORS, f"Authors' model, {AUTHORS}%", -0.03, "top"), (NO_NETWORK, f"No network (cleanup step only), {NO_NETWORK}%", 0.03, "bottom")]:
        ax.axhline(y, color=MUTED, lw=1.2, ls=(0, (4, 3)), zorder=1)
        ax.text(220_000, y + dy, label, color=INK_2, ha="right", va=va, fontsize=9)

    series = [
        (DEFAULT_DATA, ORANGE, "s", "Upstream default data"),
        (TEST_SHAPED, AQUA, "D", "Test-shaped data"),
        (PAPER_SETUP, BLUE, "o", "Paper's setup"),
    ]
    for points, color, marker, label in series:
        xs, ys = zip(*points)
        ax.plot(xs, ys, color=color, lw=2, marker=marker, ms=8, mec=SURFACE, mew=1.5, label=label, zorder=3)

    ax.annotate("2.73%", PAPER_SETUP[-1], xytext=(9, 0), textcoords="offset points", ha="left", va="center", color=INK, fontsize=10)
    ax.set_xscale("log")
    ax.set_xticks([1_200, 12_000, 36_000, 120_000], ["1,200", "12,000", "36,000", "120,000"])
    ax.minorticks_off()
    ax.set_xlim(900, 230_000)
    ax.set_ylim(2.3, 4.0)
    ax.set_xlabel("Training examples (log scale)")
    ax.set_ylabel("% longer than the optimal tree (lower is better)")
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(length=0)

    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=3, frameon=False, fontsize=9, labelcolor=INK_2, handlelength=2.2, columnspacing=1.6, borderaxespad=0.2)
    fig.suptitle("With the paper's 120,000 examples, our model ties the authors'", x=0.02, ha="left", fontsize=12.5)

    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


# Draws one tree's edges and terminals; L-shaped edges bend at (src.x, tgt.y) like upstream
def draw_tree(ax, stt, title: str) -> None:
    for edge in stt.edges:
        src, tgt = stt.get_point(edge.src), stt.get_point(edge.tgt)
        xs = [src.x.to_float(), src.x.to_float(), tgt.x.to_float()]
        ys = [src.y.to_float(), tgt.y.to_float(), tgt.y.to_float()]
        ax.plot(xs, ys, color=INK_2, lw=1.2, solid_capstyle="round", zorder=1)
        
    xs = [t.x.to_float() for t in stt.terminals]
    ys = [t.y.to_float() for t in stt.terminals]
    ax.scatter(xs, ys, s=22, color=BLUE, edgecolors=SURFACE, linewidths=0.8, zorder=2)

    ax.set_xlim(-200, 10_200)
    ax.set_ylim(-200, 10_200)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_color(GRID)
    ax.set_title(title, loc="left", fontsize=10.5, pad=8)


# Our tree from a solve run next to GeoSteiner's optimal tree for the same points
def tree_plot(solve_dir: str, path: str) -> None:
    sys.path.insert(0, UPSTREAM)
    from arora.quadtree import SteinerTree, Terminal  # noqa: E402
    from arora.points import read_points  # noqa: E402

    conf = OmegaConf.load(os.path.join(solve_dir, ".hydra", "config.yaml"))
    point_file = os.path.join(WORK, conf["solve"]["testcase"])
    ours = SteinerTree.read(point_file, os.path.join(solve_dir, "solve", "final.txt"))
    terminals = Terminal.terminals_from_point(read_points(point_file))
    best = SteinerTree.get_RSMT_GEO(terminals, conf["solve"]["fst"])

    ours_len, best_len = ours.length(), best.length()
    pct = (ours_len / best_len - 1) * 100
    case = os.path.splitext(os.path.basename(point_file))[0].rsplit("-", 1)[-1]

    fig, (left, right) = plt.subplots(1, 2, figsize=(9, 4.9))
    draw_tree(left, ours, f"NN-Steiner, our model\nlength {ours_len:,.0f} ({pct:.1f}% longer)")
    draw_tree(right, best, f"Optimal tree (GeoSteiner)\nlength {best_len:,.0f}")
    fig.suptitle(f"Test problem {case}: {len(terminals)} random points", x=0.02, ha="left", fontsize=12.5)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print(f"case {case}: ours {ours_len:.0f}, optimal {best_len:.0f}, {pct:.2f}% longer")


def main() -> None:
    parser = argparse.ArgumentParser(description="Draw the README figures into results/.")
    parser.add_argument("--solve", help="a solve run folder, e.g. work/outputs/<date>/<time>")
    args = parser.parse_args()

    os.makedirs(RESULTS, exist_ok=True)
    scaling_plot(os.path.join(RESULTS, "scaling.png"))
    print("wrote results/scaling.png")
    if args.solve:
        tree_plot(args.solve, os.path.join(RESULTS, "tree.png"))
        print("wrote results/tree.png")


if __name__ == "__main__":
    main()
