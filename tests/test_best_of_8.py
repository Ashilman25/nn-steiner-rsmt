"""Tests for the best-of-8 bookkeeping (scripts/best_of_8.py).

Run all tests: python -m unittest discover -s tests
Claude assisted
"""
from __future__ import annotations

import contextlib
import io
import os
import sys
import tempfile
import unittest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "scripts"))
sys.path.insert(0, os.path.join(REPO, "lib"))

import best_of_8  # noqa: E402
from best_of_8 import best_of, compare, exact_turns, read_runtime, run_all, turn_folder, winners  # noqa: E402
from turns import turn_points  # noqa: E402


class TurnFolderTest(unittest.TestCase):
    def test_writes_same_names_with_turned_points(self):
        with tempfile.TemporaryDirectory() as tmp:
            src = os.path.join(tmp, "src")
            os.mkdir(src)
            with open(os.path.join(src, "a-0.txt"), "w") as f:
                f.write("1 2\n30 40\n")
            with open(os.path.join(src, "a-1.txt"), "w") as f:
                f.write("0 9999\n")
            dst = os.path.join(tmp, "turned")
            turn_folder(src, dst, 8)
            for k in range(8):
                folder = os.path.join(dst, f"turn{k}")
                self.assertEqual(sorted(os.listdir(folder)), ["a-0.txt", "a-1.txt"])
                with open(os.path.join(folder, "a-0.txt")) as f:
                    got = [tuple(map(int, line.split())) for line in f]
                self.assertEqual(got, turn_points(k, [(1, 2), (30, 40)]))

    def test_turn0_is_a_byte_identical_copy(self):
        with tempfile.TemporaryDirectory() as tmp:
            src = os.path.join(tmp, "src")
            os.mkdir(src)
            with open(os.path.join(src, "p.txt"), "w") as f:
                f.write("6019 1757\n2790 7408\n")
            turn_folder(src, os.path.join(tmp, "turned"), 1)
            with open(os.path.join(tmp, "turned", "turn0", "p.txt")) as f:
                self.assertEqual(f.read(), "6019 1757\n2790 7408\n")


    def test_stray_file_in_a_turned_folder_is_an_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            src = os.path.join(tmp, "src")
            os.mkdir(src)
            with open(os.path.join(src, "p.txt"), "w") as f:
                f.write("1 2\n")
            dst = os.path.join(tmp, "turned")
            turn_folder(src, dst, 2)
            with open(os.path.join(dst, "turn1", "p copy.txt"), "w") as f:
                f.write("1 2\n")
            with self.assertRaises(ValueError):
                turn_folder(src, dst, 2)

    def test_unchanged_files_are_not_rewritten(self):
        # A second run must not rewrite a file another run may be reading
        with tempfile.TemporaryDirectory() as tmp:
            src = os.path.join(tmp, "src")
            os.mkdir(src)
            with open(os.path.join(src, "p.txt"), "w") as f:
                f.write("1 2\n30 40\n")
            dst = os.path.join(tmp, "turned")
            turn_folder(src, dst, 2)
            files = [os.path.join(dst, f"turn{k}", "p.txt") for k in range(2)]
            for path in files:
                os.utime(path, ns=(10**18, 10**18))
            turn_folder(src, dst, 2)
            self.assertEqual([os.stat(path).st_mtime_ns for path in files], [10**18, 10**18])


class ExactTest(unittest.TestCase):
    def test_point_next_to_a_half_split_line_breaks_the_mirror(self):
        # 5 points in the 625-wide corner square split it at 312.5; x = 312 sits just left of that line
        points = [(100, 100), (312, 100), (500, 100), (100, 500), (500, 500)]
        exact = exact_turns(points)
        self.assertTrue(exact[0])
        self.assertFalse(exact[4])   # 9999 - 312 = 9687 lands left of the mirrored line at 9687.5
        self.assertTrue(exact[6])    # swapping x and y needs no mirroring

    def test_a_100_point_test_problem_is_exact_in_all_turns(self):
        folder = os.path.join(best_of_8.UPSTREAM, "points", best_of_8.point_set(100))
        points = best_of_8.read_point_file(os.path.join(folder, sorted(os.listdir(folder))[0]))
        self.assertEqual(exact_turns(points), [True] * 8)


class FakeRun:
    # Stands in for a running solve: poll() gives the exit code, or None while running
    def __init__(self, codes):
        self.codes = list(codes)
        self.terminated = False

    def poll(self):
        code = self.codes.pop(0) if len(self.codes) > 1 else self.codes[0]
        if code == "ctrl-c":
            raise KeyboardInterrupt
        self.returncode = code
        return code

    def terminate(self):
        self.terminated = True

    def wait(self):
        return self.returncode if hasattr(self, "returncode") else -15


class RunAllTest(unittest.TestCase):
    def jobs(self, tmp, n):
        return [(f"turn {k}", [f"x={k}"], os.path.join(tmp, f"{k}.log")) for k in range(n)]

    def test_ctrl_c_stops_the_running_solve_and_starts_no_more(self):
        started = []
        first = FakeRun([None, "ctrl-c"])

        def start(overrides, log):
            started.append(overrides)
            return first

        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()), \
                contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                run_all(self.jobs(tmp, 4), 1, start=start)
        self.assertEqual(started, [["x=0"]])
        self.assertTrue(first.terminated)

    def test_a_failed_solve_stops_with_its_name(self):
        def start(overrides, log):
            with open(log, "w") as f:
                f.write("Traceback: boom\n")
            return FakeRun([1 if overrides == ["x=1"] else 0])

        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()), \
                contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as stop:
                run_all(self.jobs(tmp, 3), 2, start=start)
        self.assertIn("turn 1", str(stop.exception))


class BestOfTest(unittest.TestCase):
    # 3 problems x 4 turns of tree lengths
    LENGTHS = [[10.0, 12.0, 9.0, 11.0],
               [20.0, 20.0, 20.0, 20.0],
               [30.0, 29.0, 29.0, 31.0]]

    def test_best_of_uses_the_first_k_turns(self):
        self.assertEqual(best_of(self.LENGTHS, 1), [10.0, 20.0, 30.0])
        self.assertEqual(best_of(self.LENGTHS, 2), [10.0, 20.0, 29.0])
        self.assertEqual(best_of(self.LENGTHS, 4), [9.0, 20.0, 29.0])

    def test_winners_lists_every_turn_that_ties_for_shortest(self):
        self.assertEqual(winners(self.LENGTHS), [[2], [0, 1, 2, 3], [1, 2]])

    def test_compare_gives_mean_difference_and_counts(self):
        golden = [10.0, 10.0, 10.0, 10.0]
        a = [10.0, 10.2, 10.1, 10.5]          # 0, 2, 1, 5 % over
        b = [10.1, 10.2, 10.3, 10.5]          # 1, 2, 3, 5 % over
        r = compare(a, b, golden)
        self.assertAlmostEqual(r["a_pct"], 2.0)
        self.assertAlmostEqual(r["b_pct"], 2.75)
        self.assertAlmostEqual(r["diff"], -0.75)
        # differences -1, 0, -2, 0: sample std 0.957..., / sqrt(4)
        self.assertAlmostEqual(r["se"], 0.4787135538781691)
        self.assertEqual((r["shorter"], r["same"], r["longer"]), (2, 2, 0))


class ReportTest(unittest.TestCase):
    def test_writes_each_problems_shortest_tree_and_a_table(self):
        golden = best_of_8.read_lengths(best_of_8.golden_file(100))
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            stem = os.path.join(tmp, "fake")
            solved, logs = [], []
            for k in range(8):
                solved.append(f"{stem}-turn{k}-solved.txt")
                logs.append(f"{stem}-turn{k}.log")
                with open(solved[-1], "w") as f:
                    f.writelines(f"{g + (7 * i + 3 * k) % 11}\n" for i, g in enumerate(golden))
            exact = [[True] * 8 for _ in golden]
            best_of_8.report(100, 100, stem, solved, logs, 8, exact)
            got = best_of_8.read_lengths(f"{stem}-best8-solved.txt")
            want = [g + min((7 * i + 3 * k) % 11 for k in range(8)) for i, g in enumerate(golden)]
            self.assertEqual(got, want)
            with open(f"{stem}.csv") as f:
                self.assertEqual(len(f.readlines()), 101)

    def test_a_solved_file_with_the_wrong_length_is_an_error(self):
        golden = best_of_8.read_lengths(best_of_8.golden_file(100))
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            stem = os.path.join(tmp, "fake")
            solved = [f"{stem}-turn{k}-solved.txt" for k in range(2)]
            for k, path in enumerate(solved):
                with open(path, "w") as f:
                    f.writelines(f"{g}\n" for g in golden + ([1.0] if k == 1 else []))
            with self.assertRaises(SystemExit):
                best_of_8.report(100, 100, stem, solved, [f"{stem}.log"] * 2, 2, [[True] * 2 for _ in golden])


class RuntimeTest(unittest.TestCase):
    def test_reads_runtime_from_a_solve_log(self):
        with tempfile.TemporaryDirectory() as tmp:
            log = os.path.join(tmp, "x.log")
            with open(log, "w") as f:
                f.write("100%|...| 100/100\n[2026-10-20 20:34:03,506][nn_exp][INFO] - Runtime: 109.627 secs\n")
            self.assertAlmostEqual(read_runtime(log), 109.627)
            self.assertIsNone(read_runtime(os.path.join(tmp, "missing.log")))


if __name__ == "__main__":
    unittest.main()
