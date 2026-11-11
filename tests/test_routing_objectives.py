"""Tests for the congestion/delay scoring (lib/routing_objectives.py).

Run all tests: python -m unittest discover -s tests
Claude assisted
"""
from __future__ import annotations

import os
import sys
import unittest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "lib"))

import routing_objectives as ro  # noqa: E402


# Small grid with flat capacity and background, for hand-checkable congestion
def flat_scenario(background=0.0, bins=2, extent=10.0):
    grid = lambda v: [[[v] * bins for _ in range(bins)] for _ in range(2)]
    return {"bins": bins, "extent": extent, "capacity": grid(1.0), "background": grid(background),
            "r_per_unit": 0.01, "c_per_unit": 0.02, "driver_r": 1.0, "sink_c": 1.0, "driver_terminal": 0}


class ValidateTreeTest(unittest.TestCase):
    def test_accepts_tree(self):
        tree = {"points": [(0, 0), (5, 0), (5, 5)], "edges": [(0, 1), (1, 2)], "num_terminals": 3}
        pts, edges, adj = ro.validate_tree(tree)
        self.assertEqual(adj[1], [0, 2])

    def test_rejects_disconnected(self):
        tree = {"points": [(0, 0), (1, 0), (0, 1), (9, 9)], "edges": [(0, 1), (1, 2), (2, 0)], "num_terminals": 4}
        with self.assertRaisesRegex(ValueError, "Disconnected"):
            ro.validate_tree(tree)

    def test_rejects_zero_length_edge(self):
        tree = {"points": [(0, 0), (0, 0)], "edges": [(0, 1)], "num_terminals": 2}
        with self.assertRaisesRegex(ValueError, "Zero-length"):
            ro.validate_tree(tree)


class SegmentsTest(unittest.TestCase):
    tree = {"points": [(1, 1), (9, 7)], "edges": [(0, 1)], "num_terminals": 2}

    def test_hv_goes_horizontal_first(self):
        self.assertEqual(ro.segments(self.tree, "HV"), [((1, 1), (9, 1)), ((9, 1), (9, 7))])

    def test_vh_goes_vertical_first(self):
        self.assertEqual(ro.segments(self.tree, "VH"), [((1, 1), (1, 7)), ((1, 7), (9, 7))])

    def test_straight_edge_has_one_segment(self):
        tree = {"points": [(1, 1), (9, 1)], "edges": [(0, 1)], "num_terminals": 2}
        self.assertEqual(ro.segments(tree, "VH"), [((1, 1), (9, 1))])


class CongestionTest(unittest.TestCase):
    # One horizontal wire from x=1 to 9 in bins 5 wide: 0.8 of a track in each of two bins
    tree = {"points": [(1, 1), (9, 1)], "edges": [(0, 1)], "num_terminals": 2}

    def test_density(self):
        demand = ro.wire_density(self.tree, flat_scenario())
        self.assertAlmostEqual(demand[0][0][0], 0.8)
        self.assertAlmostEqual(demand[0][0][1], 0.8)
        self.assertEqual(demand[1], [[0.0, 0.0], [0.0, 0.0]])

    def test_no_overflow_under_capacity(self):
        got = ro.congestion(self.tree, flat_scenario(background=0.0))
        self.assertEqual(got["congestion_proxy"], 0.0)
        self.assertAlmostEqual(got["density_sum"], 1.6)

    def test_counts_only_extra_squared_overflow(self):
        # Background 0.5 + wire 0.8 = 1.3 against capacity 1 in two bins: 2 * 0.3^2
        got = ro.congestion(self.tree, flat_scenario(background=0.5))
        self.assertAlmostEqual(got["congestion_proxy"], 0.18)
        self.assertAlmostEqual(got["peak_utilization_proxy"], 1.3)


class DelayTest(unittest.TestCase):
    def test_star_by_hand(self):
        # Root C = 2 sinks + wire 0.02*(100+50) = 5; sink 1: 5 + 0.01*100*(1 + 0.5*0.02*100) = 7
        tree = {"points": [(0, 0), (100, 0), (0, 50)], "edges": [(0, 1), (0, 2)], "num_terminals": 3}
        got = ro.delay(tree, flat_scenario(extent=1000.0))
        self.assertAlmostEqual(got["delay_proxy"], 7.0)
        self.assertEqual(got["worst_driver_sink_path"], 100)

    def test_chain_sees_downstream_load(self):
        # No wire C or driver R: first edge drives both sinks (10*2), second one (10*1)
        tree = {"points": [(0, 0), (10, 0), (20, 0)], "edges": [(0, 1), (1, 2)], "num_terminals": 3}
        sc = dict(flat_scenario(extent=100.0), r_per_unit=1.0, c_per_unit=0.0, driver_r=0.0)
        self.assertAlmostEqual(ro.delay(tree, sc)["delay_proxy"], 30.0)


class ChooseTest(unittest.TestCase):
    @staticmethod
    def candidate(cid, wl, cong, dly):
        return {"id": cid, "metrics": {"wirelength": wl, "congestion_proxy": cong,
                                       "density_sum": 10.0, "delay_proxy": dly}}

    def setUp(self):
        self.candidates = [
            self.candidate("turn0-t0.9-HV", 100.0, 5.0, 10.0),  # baseline
            self.candidate("a", 105.0, 0.0, 5.0),               # within the 10% budget
            self.candidate("b", 120.0, 0.0, 0.0),               # best score but too long
        ]

    def test_respects_wirelength_budget(self):
        best, cost, _ = ro.choose(self.candidates, None, (1.0, 1.0, 1.0))
        self.assertEqual(best["id"], "a")
        self.assertAlmostEqual(cost, 1.05 + 0.0 + 0.5)

    def test_wire_only_keeps_shortest(self):
        best, _, _ = ro.choose(self.candidates, None, (1.0, 0.0, 0.0))
        self.assertEqual(best["id"], "turn0-t0.9-HV")

    def test_rejects_bad_weights(self):
        with self.assertRaises(ValueError):
            ro.choose(self.candidates, None, (0.0, 0.0, 0.0))
        with self.assertRaises(ValueError):
            ro.choose(self.candidates, None, (1.0, -1.0, 0.0))


class TurnTest(unittest.TestCase):
    def test_unturn_undoes_turn(self):
        for k in range(8):
            self.assertEqual(ro.unturn(k, *ro.turn(k, 12, 345)), (12, 345))


if __name__ == "__main__":
    unittest.main()
