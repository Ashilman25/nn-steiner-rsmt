"""Tests for the 8 turns of the square (lib/turns.py).

Run all tests: python -m unittest discover -s tests
claude assisted
"""
from __future__ import annotations

import os
import random
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "lib"))

from turns import N_MAX, TURN_NAMES, turn_point, turn_points, unturn_point  # noqa: E402

RNG = random.Random(257)
POINTS = [(RNG.randint(0, N_MAX), RNG.randint(0, N_MAX)) for _ in range(200)]


def l1(a: tuple[int, int], b: tuple[int, int]) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


class TurnTest(unittest.TestCase):
    def test_table_values(self):
        # (x, y) = (1, 2) under each turn, in TURN_NAMES order
        n = N_MAX
        expected = [(1, 2), (n - 2, 1), (n - 1, n - 2), (2, n - 1),
                    (n - 1, 2), (1, n - 2), (2, 1), (n - 2, n - 1)]
        self.assertEqual([turn_point(k, 1, 2) for k in range(8)], expected)
        self.assertEqual(len(TURN_NAMES), 8)

    def test_corners_stay_inside_the_box(self):
        corners = {(0, 0), (0, N_MAX), (N_MAX, 0), (N_MAX, N_MAX)}
        for k in range(8):
            self.assertEqual({turn_point(k, x, y) for x, y in corners}, corners)

    def test_distances_unchanged(self):
        for k in range(8):
            turned = turn_points(k, POINTS)
            for i in range(0, len(POINTS) - 1):
                self.assertEqual(l1(turned[i], turned[i + 1]), l1(POINTS[i], POINTS[i + 1]))

    def test_unturn_undoes_turn(self):
        for k in range(8):
            for x, y in POINTS:
                self.assertEqual(unturn_point(k, *turn_point(k, x, y)), (x, y))

    def test_all_eight_differ(self):
        self.assertEqual(len({turn_point(k, 1, 2) for k in range(8)}), 8)

    def test_rejects_points_outside_the_box(self):
        for bad in [(-1, 5), (5, N_MAX + 1), (10000, 0)]:
            with self.assertRaises(ValueError):
                turn_point(1, *bad)


if __name__ == "__main__":
    unittest.main()
