"""Unit tests for the GUI's progress bookkeeping (analog_designer/gui/
progress_tracker.py -- pure, no Tk) and the planning side that feeds it
(run_sim.plan_progress()): weighted fractions, the three segment kinds,
reaching 100% on a clean finish, and the ETA's countdown/calibration."""
import unittest
from pathlib import Path

from analog_designer.core import workspace
from analog_designer.gui.progress_tracker import ProgressTracker
from analog_designer.sim import run_sim

# Next to a standalone core checkout, or next to the PRO repo that nests it
# as a submodule.
REFERENCE_ROOT = next(
    (p / "ihp_mh_ip__cmos_vref" for p in Path(__file__).resolve().parents[2:4]
     if (p / "ihp_mh_ip__cmos_vref").exists()),
    Path(__file__).resolve().parents[2] / "ihp_mh_ip__cmos_vref",
)


class FakeClock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now


class ProgressTrackerTests(unittest.TestCase):
    def setUp(self):
        self.clock = FakeClock()
        self.tracker = ProgressTracker(cpu_budget=1, clock=self.clock)
        # "slow" is 3x the time of "fast" despite having fewer conditions.
        for line in ("@PROGRESS PLAN v1 fast 2 10.0", "@PROGRESS PLAN v1 slow 1 30.0"):
            self.assertTrue(self.tracker.feed(line))

    def test_fractions_are_weighted_by_estimated_time(self):
        self.tracker.feed("@PROGRESS STEP v1 slow ok")
        ok, fail, skip = self.tracker.fractions()
        self.assertAlmostEqual(ok, 0.75)
        self.assertEqual((fail, skip), (0.0, 0.0))
        self.assertTrue(self.tracker.label().startswith("1/3 (75%)"))

    def test_failed_step_is_red(self):
        self.tracker.feed("@PROGRESS STEP v1 fast fail")
        self.assertAlmostEqual(self.tracker.fractions()[1], 0.125)
        self.assertIn("1 failed", self.tracker.label())

    def test_testfail_recolors_ok_steps(self):
        self.tracker.feed("@PROGRESS STEP v1 fast ok")
        self.tracker.feed("@PROGRESS STEP v1 fast ok")
        self.tracker.feed("@PROGRESS TESTFAIL v1 fast")
        ok, fail, _ = self.tracker.fractions()
        self.assertAlmostEqual(ok, 0.0)
        self.assertAlmostEqual(fail, 0.25)

    def test_skipped_tests_are_gray(self):
        self.tracker.feed("@PROGRESS STEP v1 fast ok")
        self.tracker.feed("@PROGRESS STEP v1 fast ok")
        self.tracker.feed("@PROGRESS SKIPPED v1 slow")
        self.assertAlmostEqual(sum(self.tracker.fractions()), 1.0)
        self.assertAlmostEqual(self.tracker.fractions()[2], 0.75)
        self.assertIn("1 skipped", self.tracker.label())

    def test_leftover_at_done_is_red(self):
        self.tracker.feed("@PROGRESS STEP v1 fast ok")
        self.tracker.variation_done("v1")
        ok, fail, skip = self.tracker.fractions()
        self.assertAlmostEqual(ok, 0.125)
        self.assertAlmostEqual(fail, 0.875)
        self.assertEqual(skip, 0.0)

    def test_clean_finish_fills_gray_to_100(self):
        self.tracker.feed("@PROGRESS STEP v1 fast ok")
        self.tracker.finish(True)
        self.assertAlmostEqual(sum(self.tracker.fractions()), 1.0)
        self.assertIsNone(self.tracker.eta())

    def test_cancel_leaves_bar_where_it_stopped(self):
        self.tracker.feed("@PROGRESS STEP v1 fast ok")
        self.tracker.finish(False)
        self.assertAlmostEqual(sum(self.tracker.fractions()), 0.125)
        self.assertIsNone(self.tracker.eta())

    def test_eta_counts_down_and_calibrates(self):
        seconds, guess = self.tracker.eta()
        self.assertTrue(guess)
        self.assertAlmostEqual(seconds, 40.0)
        self.clock.now = 5.0
        self.assertAlmostEqual(self.tracker.eta()[0], 35.0)
        # This run is twice as slow as history: 10 estimated seconds took 20.
        self.clock.now = 20.0
        self.tracker.feed("@PROGRESS STEP v1 fast ok")
        self.tracker.feed("@PROGRESS STEP v1 fast ok")
        seconds, guess = self.tracker.eta()
        self.assertFalse(guess)
        # k = (20 + 1*w0) / (10 + w0), w0 = 40/3 -> ~1.43, x 30 left.
        w0 = 40 / 3
        self.assertAlmostEqual(seconds, (20 + w0) / (10 + w0) * 30)
        self.clock.now = 25.0
        self.assertAlmostEqual(self.tracker.eta()[0], (20 + w0) / (10 + w0) * 30 - 5)

    def test_no_history_means_step_count_and_no_upfront_eta(self):
        tracker = ProgressTracker(clock=self.clock)
        tracker.feed("@PROGRESS PLAN v2 a 3 -")
        tracker.feed("@PROGRESS PLAN v2 b 1 -")
        self.assertIsNone(tracker.eta())
        self.clock.now = 4.0
        tracker.feed("@PROGRESS STEP v2 a ok")
        self.assertAlmostEqual(tracker.fractions()[0], 0.25)
        self.assertIsNotNone(tracker.eta())

    def test_duplicate_plan_is_ignored_and_extra_steps_grow_the_plan(self):
        self.tracker.feed("@PROGRESS PLAN v1 slow 1 30.0")
        self.assertEqual(self.tracker.n_total, 3)
        self.tracker.feed("@PROGRESS STEP v1 slow ok")
        self.tracker.feed("@PROGRESS STEP v1 slow ok")
        self.assertEqual(self.tracker.n_total, 4)
        self.assertLessEqual(sum(self.tracker.fractions()), 1.0)

    def test_other_lines_are_not_consumed(self):
        for line in ("@PROGRESS DONE v1 ok 3.0", "@PROGRESS RUNNING v1 fast x", "running fast ...",
                     "@PROGRESS STEP malformed"):
            self.assertFalse(self.tracker.feed(line), line)


class PlanProgressTests(unittest.TestCase):
    def setUp(self):
        if not REFERENCE_ROOT.exists():
            raise unittest.SkipTest(f"reference project not checked out at {REFERENCE_ROOT}")
        workspace.open_folder(str(REFERENCE_ROOT), block="cmos_vref", topology="default")
        self.block_cfg = workspace.CONFIG["blocks"]["cmos_vref"]["topologies"]["default"]
        self.tests = workspace.CONFIG["tests"]["cmos_vref"]
        self.defaults = workspace.CONFIG["defaults"]

    def _plan(self, history):
        return run_sim.plan_progress(self.block_cfg, self.tests, self.defaults, [], "vX", False, history)

    def test_no_history_gives_no_estimate(self):
        plan = self._plan({})
        self.assertEqual([t for t, _, _ in plan], list(self.tests))
        self.assertTrue(all(est is None and n > 0 for _, n, est in plan))

    def test_missing_test_falls_back_to_per_condition_mean(self):
        first = next(iter(self.tests))
        plan = {t: (n, est) for t, n, est in self._plan({first: 12.0})}
        n_first = plan[first][0]
        self.assertEqual(plan[first][1], 12.0)
        for test, (n, est) in plan.items():
            if test != first:
                self.assertAlmostEqual(est, 12.0 / n_first * n)


if __name__ == "__main__":
    unittest.main()
