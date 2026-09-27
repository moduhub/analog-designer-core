"""Tk-free state behind the main window's progress bar -- fed the
"@PROGRESS PLAN/STEP/TESTFAIL/SKIPPED" lines a simulation job prints (see
analog_designer/sim/run_sim.py's emit_progress_plan/emit_progress_step/
emit_progress_testfail/emit_progress_skipped) plus the per-variation DONE
signal, and turned into three weighted fractions (ok/fail/skip) and an ETA.

Every planned (variation, test) gets a slice of the bar proportional to its
own estimated seconds (plan_progress()), split evenly across its conditions,
so a slow transient test fills more of the bar than a quick op point -- the
bar reads as "share of the expected time", not "share of the step count".
Each slice ends up green (simulated ok), red (failed) or gray (planned but
never run), so a finished job always reaches 100% even when --skip-on-fail
cut most of it short.

Kept separate from app.py so the bookkeeping is unit-testable without a
display (tests/test_progress_tracker.py)."""

import time

# Seconds-per-estimated-second weight given to the upfront guess (see
# ProgressTracker.eta()) -- measured in "average steps' worth" of real
# progress, so it stops mattering after a handful of steps.
_PRIOR_STEPS = 1.0


class ProgressTracker:
    def __init__(self, cpu_budget=1, clock=time.monotonic):
        self._cpu_budget = max(1, cpu_budget)
        self._clock = clock
        self.reset()

    def reset(self):
        # (variation, test) -> {"n", "w" (weight per condition), "ok", "fail", "skip"}
        self._plan = {}
        self._by_variation = {}
        self._has_history = False
        self._finished = False
        self._start_ts = self._clock()
        self._last_step_ts = self._start_ts
        self.w_total = self.w_ok = self.w_fail = self.w_skip = 0.0
        self.n_total = self.n_ok = self.n_fail = self.n_skip = 0

    @property
    def active(self):
        return self.n_total > 0

    @property
    def w_left(self):
        return max(0.0, self.w_total - self.w_ok - self.w_fail - self.w_skip)

    def feed(self, line):
        """True if `line` was one of the lines this tracker owns (caller
        doesn't log it), False otherwise -- DONE/RUNNING/CONTAINER stay the
        caller's business (see variation_done() for DONE's share)."""
        head, _, rest = line.partition(" ")
        if head != "@PROGRESS":
            return False
        kind, _, rest = rest.partition(" ")
        fields = rest.split()
        try:
            if kind == "PLAN":
                variation, test, n, est = fields
                self._add_plan(variation, test, int(n), None if est == "-" else float(est))
            elif kind == "STEP":
                variation, test, status = fields
                self._step(variation, test, status == "ok")
            elif kind == "TESTFAIL":
                variation, test = fields
                self._testfail(variation, test)
            elif kind == "SKIPPED":
                variation, *tests = fields
                for test in tests:
                    self._close(self._plan.get((variation, test)), "skip")
            else:
                return False
        except ValueError:
            return False
        return True

    def variation_done(self, variation):
        """Anything `variation` still hadn't stepped/skipped by the time its
        DONE line arrives never ran because something went wrong (an
        unexpected exception mid-variation -- SKIPPED already covered the
        deliberate cases), so it's filled red."""
        for key in self._by_variation.get(variation, ()):
            self._close(self._plan[key], "fail")

    def finish(self, completed):
        """Job ended. A clean exit fills whatever's left gray (planned, never
        run) so the bar reaches 100%; a cancel/crash leaves it where it
        stopped, which is the honest picture of how far it got. Either
        way there's nothing left to count down."""
        self._finished = True
        if completed:
            for entry in self._plan.values():
                self._close(entry, "skip")

    def fractions(self):
        """(ok, fail, skip) as fractions of the whole bar."""
        if self.w_total <= 0:
            return 0.0, 0.0, 0.0
        return self.w_ok / self.w_total, self.w_fail / self.w_total, self.w_skip / self.w_total

    def eta(self):
        """(seconds_left, is_upfront_guess) or None.

        k (wall seconds per estimated second) starts at the upfront guess
        1/min(cpu_budget, variations planned) -- historical durations were
        each measured as one test's own wall time, and up to cpu_budget
        variations run side by side -- and converges to this run's measured
        pace (elapsed / estimated-seconds actually simulated) as real steps
        arrive, which absorbs machine load and the real overlap. The
        countdown runs off the last step's timestamp, so it keeps ticking
        down between steps instead of freezing."""
        w_left = self.w_left
        if w_left <= 0 or not self.active or self._finished:
            return None
        w_run = self.w_ok + self.w_fail
        if w_run <= 0 and not self._has_history:
            return None  # weights are plain step counts, not seconds -- nothing to show yet
        k0 = 1.0 / min(self._cpu_budget, max(1, len(self._by_variation)))
        w0 = _PRIOR_STEPS * self.w_total / self.n_total
        elapsed_ref = self._last_step_ts - self._start_ts
        k = (elapsed_ref + k0 * w0) / (w_run + w0)
        left = k * w_left - (self._clock() - self._last_step_ts)
        return max(0.0, left), w_run <= 0

    def label(self):
        if not self.active:
            return ""
        done = self.n_ok + self.n_fail + self.n_skip
        pct = 100 * (self.w_total - self.w_left) / self.w_total
        text = f"{done}/{self.n_total} ({pct:.0f}%)"
        if self.n_fail:
            text += f" · {self.n_fail} failed"
        if self.n_skip:
            text += f" · {self.n_skip} skipped"
        eta = self.eta()
        if eta is not None:
            seconds, guess = eta
            text += f" · ETA {int(seconds // 60):02d}:{int(seconds % 60):02d}"
            if guess:
                text += " (est.)"
        return text

    def _add_plan(self, variation, test, n, est):
        key = (variation, test)
        if key in self._plan or n <= 0:
            # The same variation planned twice (two hierarchical jobs sharing
            # a sub-block) runs only once -- the second finds it fresh --
            # so the duplicate must not reserve bar space of its own.
            return
        if est is not None:
            self._has_history = True
        w = (est if est is not None and est > 0 else float(n)) / n
        self._plan[key] = {"n": n, "w": w, "ok": 0, "fail": 0, "skip": 0}
        self._by_variation.setdefault(variation, []).append(key)
        self.w_total += w * n
        self.n_total += n

    def _step(self, variation, test, ok):
        entry = self._plan.get((variation, test))
        if entry is None or entry["ok"] + entry["fail"] + entry["skip"] >= entry["n"]:
            # A step the plan didn't count (e.g. --force re-running a
            # duplicate) -- grow the plan rather than overflowing past 100%.
            if entry is None:
                w = self.w_total / self.n_total if self.n_total else 1.0
                entry = self._plan[(variation, test)] = {"n": 0, "w": w, "ok": 0, "fail": 0, "skip": 0}
                self._by_variation.setdefault(variation, []).append((variation, test))
            entry["n"] += 1
            self.w_total += entry["w"]
            self.n_total += 1
        kind = "ok" if ok else "fail"
        entry[kind] += 1
        self._add(kind, 1, entry["w"])
        self._last_step_ts = self._clock()

    def _testfail(self, variation, test):
        entry = self._plan.get((variation, test))
        if entry is None or not entry["ok"]:
            return
        n = entry["ok"]
        entry["ok"], entry["fail"] = 0, entry["fail"] + n
        self._add("ok", -n, entry["w"])
        self._add("fail", n, entry["w"])

    def _close(self, entry, kind):
        if entry is None:
            return
        left = entry["n"] - entry["ok"] - entry["fail"] - entry["skip"]
        if left > 0:
            entry[kind] += left
            self._add(kind, left, entry["w"])

    def _add(self, kind, n, w):
        setattr(self, f"n_{kind}", getattr(self, f"n_{kind}") + n)
        setattr(self, f"w_{kind}", getattr(self, f"w_{kind}") + n * w)
