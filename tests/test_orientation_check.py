"""Tests for scripts/orientation_check.py.

Run all tests: python -m unittest discover -s tests
"""
from __future__ import annotations

import os
import sys
import unittest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "scripts"))

import best_of_8  # noqa: E402
from orientation_check import solve_without_network, summarize  # noqa: E402


class SummarizeTest(unittest.TestCase):
    def test_average_spread_and_ties(self):
        golden = [100.0, 200.0]
        lengths = [[101.0, 103.0], [204.0, 204.0]]   # [problem][turn]
        r = summarize(lengths, golden)
        self.assertAlmostEqual(r["average"], (1 + 3 + 2 + 2) / 4)
        self.assertAlmostEqual(r["spread"], (2 + 0) / 2)
        self.assertEqual(r["all_same"], 1)


class SolveTest(unittest.TestCase):
    def test_both_trees_are_valid_and_never_beat_optimal(self):
        folder = os.path.join(best_of_8.UPSTREAM, "points", best_of_8.point_set(100))
        name = sorted(os.listdir(folder))[0]
        optimal = best_of_8.read_lengths(best_of_8.golden_file(100))[0]
        perfect, cleanup_only = solve_without_network(os.path.join(folder, name))
        self.assertGreaterEqual(perfect, optimal)
        self.assertGreaterEqual(cleanup_only, optimal)


if __name__ == "__main__":
    unittest.main()
