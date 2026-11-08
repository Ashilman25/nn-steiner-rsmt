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

from turns import N_MAX, TURN_NAMES, sample_idx, sample_turn, turn_point, turn_points, unturn_point  # noqa: E402

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


class SampleTurnTest(unittest.TestCase):
    # small stand-in for 12,000 bases x 8 turns; batch 5 divides the 40-sample blocks
    NUM_BASE, TURNS, BATCH = 40, 8, 5

    def all_pairs(self, num_base: int, turns: int) -> list[tuple[int, int]]:
        return [sample_turn(i, num_base, turns) for i in range(num_base * turns)]

    def test_every_base_and_turn_exactly_once(self):
        pairs = self.all_pairs(self.NUM_BASE, self.TURNS)
        self.assertEqual(sorted(pairs), [(b, t) for b in range(self.NUM_BASE) for t in range(self.TURNS)])

    def test_sample_idx_undoes_sample_turn(self):
        for i in range(self.NUM_BASE * self.TURNS):
            self.assertEqual(sample_idx(*sample_turn(i, self.NUM_BASE, self.TURNS), self.NUM_BASE, self.TURNS), i)

    def test_each_block_has_every_base_once(self):
        pairs = self.all_pairs(self.NUM_BASE, self.TURNS)
        for block in range(self.TURNS):
            bases = [b for b, _ in pairs[block * self.NUM_BASE:(block + 1) * self.NUM_BASE]]
            self.assertEqual(bases, list(range(self.NUM_BASE)))

    def test_batches_have_distinct_bases_and_mixed_turns(self):
        pairs = self.all_pairs(self.NUM_BASE, self.TURNS)
        for start in range(0, len(pairs), self.BATCH):
            batch = pairs[start:start + self.BATCH]
            self.assertEqual(len({b for b, _ in batch}), self.BATCH)
            self.assertEqual(len({t for _, t in batch}), self.BATCH)

    def test_paper_size_turn_zero_positions(self):
        # 12,000 bases x 8: base 0 turn 0 is sample 0, base 1 turn 0 sits in the last block
        self.assertEqual(sample_turn(0, 12000, 8), (0, 0))
        self.assertEqual(sample_idx(1, 0, 12000, 8), 7 * 12000 + 1)
        self.assertEqual(sample_idx(8, 0, 12000, 8), 8)

    def test_one_turn_is_the_identity(self):
        self.assertEqual(self.all_pairs(30, 1), [(i, 0) for i in range(30)])

    def test_rejects_out_of_range(self):
        with self.assertRaises(ValueError):
            sample_turn(self.NUM_BASE * self.TURNS, self.NUM_BASE, self.TURNS)
        with self.assertRaises(ValueError):
            sample_idx(0, self.TURNS, self.NUM_BASE, self.TURNS)


if __name__ == "__main__":
    unittest.main()
