from __future__ import annotations

from typing import Iterable, List, Tuple

N_MAX = 9999  # largest coordinate in the 10000 x 10000 test sets

TURN_NAMES = [
    "none",
    "90",
    "180",
    "270",
    "mirror left-right",
    "mirror top-bottom",
    "flip on the diagonal",
    "flip on the other diagonal",
]


# Turns one point with turn k (0-7); see TURN_NAMES
def turn_point(k: int, x: int, y: int, n: int = N_MAX) -> Tuple[int, int]:
    if not (0 <= x <= n and 0 <= y <= n):
        raise ValueError(f"point ({x}, {y}) is outside 0..{n}")

    return [
        (x, y),
        (n - y, x),
        (n - x, n - y),
        (y, n - x),
        (n - x, y),
        (x, n - y),
        (y, x),
        (n - y, n - x),
    ][k]


# Undoes turn k: rotations 90 and 270 undo each other, the rest undo themselves
def unturn_point(k: int, x: int, y: int, n: int = N_MAX) -> Tuple[int, int]:
    return turn_point([0, 3, 2, 1, 4, 5, 6, 7][k], x, y, n)


# Turns a whole point set, keeping the order of the points
def turn_points(k: int, points: Iterable[Tuple[int, int]], n: int = N_MAX) -> List[Tuple[int, int]]:
    return [turn_point(k, x, y, n) for x, y in points]
