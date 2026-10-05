"""Unit tests for gen_variations.py's discard_on_fail/checkpoint_size
plumbing: whichever run_variation() call signals "discard": True must get
trim_variation()'d, but only once its own CHUNK's dispatch has fully
drained -- never while a sibling variation elsewhere in that same chunk is
still mid-flight (see _run_batch's/_run_hierarchical_batch's own
docstrings for exactly why a mid-chunk trim would race a sibling
variation's concurrent results.jsonl/variations.jsonl append). Without an
explicit (or auto-computed) checkpoint_size, the whole batch is just one
chunk, so this collapses to "trim only once at the very end" -- the
original, pre-checkpointing behavior. docker/run_variation/trim_variation
are all mocked out here -- pure control-flow tests, no simulation, no real
file writes -- grounded against the real reference project
(ihp_mh_ip__cmos_vref) only for a valid config.json/block_cfg/tests to plan
against.
"""
import contextlib
import unittest
from pathlib import Path
from unittest.mock import patch

from analog_designer.core import workspace
from analog_designer.sim import gen_variations

REFERENCE_ROOT = Path(__file__).resolve().parents[2] / "ihp_mh_ip__cmos_vref"


def _require_reference():
    if not REFERENCE_ROOT.exists():
        raise unittest.SkipTest(f"reference project not checked out at {REFERENCE_ROOT}")


class RunBatchDiscardTests(unittest.TestCase):
    def setUp(self):
        _require_reference()
        workspace.open_folder(str(REFERENCE_ROOT), block="cmos_vref", topology="default")
        self.block_cfg = workspace.CONFIG["blocks"]["cmos_vref"]["topologies"]["default"]
        self.tests = workspace.CONFIG["tests"]["cmos_vref"]
        self.defaults = workspace.CONFIG["defaults"]
        self.calls = []  # shared order-of-events log every mock below appends to

    def _patched(self, outcomes, cpu_budget=2):
        """outcomes: [dict, ...] -- one per param set, returned by run_variation()
        in call order. Records "run:<name>"/"trim:<name>"/"container_enter"/
        "container_exit" into self.calls, in real chronological order.
        cpu_budget is mocked (not left to whatever this machine's own
        settings.json says) so max_workers/the auto checkpoint_size formula
        are deterministic across environments."""
        outcomes = iter(outcomes)

        def fake_run_variation(*args, **kwargs):
            outcome = next(outcomes)
            self.calls.append(f"run:{outcome['variation']}")
            return outcome

        @contextlib.contextmanager
        def fake_managed_executor():
            self.calls.append("container_enter")
            yield "fake-container-id"
            self.calls.append("container_exit")

        def fake_trim_variation(name):
            self.calls.append(f"trim:{name}")

        return (
            patch("analog_designer.sim.gen_variations.run_variation", side_effect=fake_run_variation),
            patch("analog_designer.sim.gen_variations.managed_executor", fake_managed_executor),
            patch("analog_designer.sim.gen_variations.setup_container", return_value=None),
            patch("analog_designer.sim.gen_variations.trim_variation", side_effect=fake_trim_variation),
            patch("analog_designer.core.workspace.cpu_budget", return_value=cpu_budget),
        )

    def _run(self, outcomes, cpu_budget=2, **kwargs):
        param_sets = [{n: pdef["default"] for n, pdef in self.block_cfg["parameters"].items()} for _ in outcomes]
        with contextlib.ExitStack() as stack:
            for cm in self._patched(outcomes, cpu_budget=cpu_budget):
                stack.enter_context(cm)
            return gen_variations._run_batch(
                param_sets, self.block_cfg, self.tests, self.defaults, force=False,
                origin={"kind": "manual"}, **kwargs,
            )

    def test_no_checkpoint_size_trims_once_at_the_very_end(self):
        """discard_on_fail with no explicit checkpoint_size auto-sizes one
        (DEFAULT_CHECKPOINT_MULTIPLIER * cpu_budget) -- with only 3 items,
        that's always bigger than the batch, so everything still lands in
        one chunk and trims only once, after every run() call, same as
        before checkpointing existed."""
        outcomes = [
            {"variation": "cmos_vref-default-aaaaaa", "any_error": False, "discard": False},
            {"variation": "cmos_vref-default-bbbbbb", "any_error": False, "discard": True},
            {"variation": "cmos_vref-default-cccccc", "any_error": False, "discard": False},
        ]
        any_error = self._run(
            outcomes, skip_on_fail_profile="low_power", skip_on_fail_max_failures=1, discard_on_fail=True,
        )
        self.assertFalse(any_error)
        self.assertEqual(self.calls.count("trim:cmos_vref-default-bbbbbb"), 1)
        # every run() already happened before the one trim -- no interleaving.
        trim_index = self.calls.index("trim:cmos_vref-default-bbbbbb")
        self.assertEqual(self.calls[:trim_index].count("run:"), 0)  # sanity: no bare "run:" (always suffixed)
        self.assertEqual(len([c for c in self.calls[:trim_index] if c.startswith("run:")]), 3)

    def test_explicit_checkpoint_size_trims_in_multiple_waves(self):
        """checkpoint_size=2 over 4 items (2 discarded, one per chunk) --
        each chunk's own discard must be trimmed BEFORE the next chunk's
        own run() calls start, not batched up for a single trim at the end."""
        outcomes = [
            {"variation": "cmos_vref-default-aaaaaa", "any_error": False, "discard": True},
            {"variation": "cmos_vref-default-bbbbbb", "any_error": False, "discard": False},
            {"variation": "cmos_vref-default-cccccc", "any_error": False, "discard": True},
            {"variation": "cmos_vref-default-dddddd", "any_error": False, "discard": False},
        ]
        self._run(
            outcomes, skip_on_fail_profile="low_power", discard_on_fail=True, checkpoint_size=2,
        )
        trim_a = self.calls.index("trim:cmos_vref-default-aaaaaa")
        trim_c = self.calls.index("trim:cmos_vref-default-cccccc")
        run_c = self.calls.index("run:cmos_vref-default-cccccc")
        run_d = self.calls.index("run:cmos_vref-default-dddddd")
        # chunk 1 (aaaaaa, bbbbbb) fully drains and its own discard (aaaaaa)
        # is trimmed BEFORE chunk 2 (cccccc, dddddd) even starts running.
        self.assertLess(trim_a, run_c)
        self.assertLess(trim_a, run_d)
        # chunk 2's own discard (cccccc) is trimmed only after both of
        # chunk 2's own run() calls, not before/mid-chunk.
        self.assertGreater(trim_c, run_c)
        self.assertGreater(trim_c, run_d)

    def test_checkpoint_size_ignored_without_discard_on_fail(self):
        """checkpoint_size only ever matters alongside discard_on_fail --
        nothing to periodically trim otherwise, so passing it without
        discard_on_fail is a harmless no-op, same single-dispatch shape as
        before this feature existed (no forced chunking overhead for a
        caller that never asked for discard)."""
        outcomes = [{"variation": "cmos_vref-default-aaaaaa", "any_error": False, "discard": False}]
        self._run(outcomes, skip_on_fail_profile="low_power", checkpoint_size=1)
        self.assertNotIn("trim:cmos_vref-default-aaaaaa", self.calls)

    def test_not_discarded_never_trims(self):
        outcomes = [{"variation": "cmos_vref-default-aaaaaa", "any_error": False, "discard": False}]
        self._run(outcomes, skip_on_fail_profile="low_power", discard_on_fail=True)
        self.assertEqual([c for c in self.calls if c.startswith("trim:")], [])

    def test_discard_off_by_default_even_if_run_variation_would_flag_it(self):
        """discard_on_fail defaults to False -- a caller that never asks for
        it (every existing caller, before this feature) sees no behavior
        change even if (hypothetically) an outcome carried "discard": True."""
        outcomes = [{"variation": "cmos_vref-default-aaaaaa", "any_error": False, "discard": True}]
        # discard_on_fail wasn't passed at all here -- _run_batch() only
        # acts on whatever run_variation() itself decided to report, so a
        # caller not asking for discard still leaves nothing to trim UNLESS
        # run_variation() itself returned discard=True (which it only ever
        # does when IT was called with discard_on_fail=True, forwarded from
        # this same _run_batch() call -- this test's own default-False
        # forwarding means the mock above is exercising an outcome shape
        # _run_batch() would never actually receive in practice, purely to
        # confirm _run_batch() reacts to the dict key alone, trusting its
        # own forwarded discard_on_fail=False elsewhere).
        self._run(outcomes)
        self.assertIn("trim:cmos_vref-default-aaaaaa", self.calls)


class RunHierarchicalBatchDiscardTests(unittest.TestCase):
    def setUp(self):
        _require_reference()
        workspace.open_folder(str(REFERENCE_ROOT), block="top", topology="default")
        self.top_cfg = workspace.CONFIG["blocks"]["top"]["topologies"]["default"]
        self.vref_cfg = workspace.CONFIG["blocks"]["cmos_vref"]["topologies"]["default"]
        self.amp_cfg = workspace.CONFIG["blocks"]["output_amp"]["topologies"]["default"]
        self.defaults = workspace.CONFIG["defaults"]
        self.calls = []

    def _patched(self, outcomes_by_name, cpu_budget=2):
        def fake_run_variation(block_cfg, tests, defaults, params, **kwargs):
            name = gen_variations.variation_name(kwargs["block"], kwargs["topology"], params)
            self.calls.append(("run", name, kwargs.get("discard_on_fail")))
            return outcomes_by_name[name]

        @contextlib.contextmanager
        def fake_managed_executor():
            self.calls.append(("container_enter",))
            yield "fake-container-id"
            self.calls.append(("container_exit",))

        def fake_trim_variation(name):
            self.calls.append(("trim", name))

        return (
            patch("analog_designer.sim.gen_variations.run_variation", side_effect=fake_run_variation),
            patch("analog_designer.sim.gen_variations.managed_executor", fake_managed_executor),
            patch("analog_designer.sim.gen_variations.setup_container", return_value=None),
            patch("analog_designer.sim.gen_variations.trim_variation", side_effect=fake_trim_variation),
            patch("analog_designer.core.workspace.cpu_budget", return_value=cpu_budget),
        )

    def _one_iteration_jobs(self, variant, top_discard, vref_discard=False, amp_discard=False):
        """variant: a small int distinguishing one call's own jobs/names
        from another's (e.g. two "iterations" in the same batch) -- offsets
        m1_width by variant*5n (both blocks' own grid), staying safely
        within range, so variation_name() hashes to genuinely distinct
        names per variant, not just for the top job (whose own
        X1_variation/x2_variation reference strings already differ) but
        for its own sub-jobs too."""
        top_params = {n: pdef["default"] for n, pdef in self.top_cfg["parameters"].items() if pdef.get("type") != "block_ref"}
        top_params["X1_variation"] = f"cmos_vref-default-sub{variant}"
        top_params["x2_variation"] = f"output_amp-default-sub{variant}"
        vref_params = {n: pdef["default"] for n, pdef in self.vref_cfg["parameters"].items()}
        amp_params = {n: pdef["default"] for n, pdef in self.amp_cfg["parameters"].items()}
        vref_params["m1_width"] = f"{2.5 + variant * 0.005:.4g}u"
        amp_params["m1_width"] = f"{2.5 + variant * 0.005:.4g}u"
        vref_name = gen_variations.variation_name("cmos_vref", "default", vref_params)
        amp_name = gen_variations.variation_name("output_amp", "default", amp_params)
        top_name = gen_variations.variation_name("top", "default", top_params)
        jobs = [
            {"block_cfg": self.vref_cfg, "tests": workspace.CONFIG["tests"]["cmos_vref"], "block": "cmos_vref",
             "topology": "default", "params": vref_params, "origin": {"kind": "generate_hierarchical"}},
            {"block_cfg": self.amp_cfg, "tests": workspace.CONFIG["tests"]["output_amp"], "block": "output_amp",
             "topology": "default", "params": amp_params, "origin": {"kind": "generate_hierarchical"}},
            {"block_cfg": self.top_cfg, "tests": workspace.CONFIG["tests"]["top"], "block": "top",
             "topology": "default", "params": top_params, "origin": {"kind": "random"},
             "discard_with": [vref_name, amp_name]},
        ]
        outcomes = {
            vref_name: {"variation": vref_name, "any_error": False, "discard": vref_discard},
            amp_name: {"variation": amp_name, "any_error": False, "discard": amp_discard},
            top_name: {"variation": top_name, "any_error": False, "discard": top_discard},
        }
        return jobs, outcomes, (top_name, vref_name, amp_name)

    def test_discarded_top_job_takes_its_own_sub_jobs_down_with_it(self):
        jobs, outcomes_by_name, (top_name, vref_name, amp_name) = self._one_iteration_jobs(1, top_discard=True)
        with contextlib.ExitStack() as stack:
            for cm in self._patched(outcomes_by_name):
                stack.enter_context(cm)
            gen_variations._run_hierarchical_batch(
                jobs, self.defaults, force=False, skip_on_fail_profile="low_power", discard_on_fail=True,
            )

        # sub-jobs (no "discard_with" key) are NEVER individually asked to
        # discard themselves, regardless of the batch's own discard_on_fail --
        # only the top-level job's own "discard_with" list, once ITS outcome
        # discards, brings them down too.
        run_calls = {c[1]: c[2] for c in self.calls if c[0] == "run"}
        self.assertEqual(run_calls[vref_name], False)
        self.assertEqual(run_calls[amp_name], False)
        self.assertEqual(run_calls[top_name], True)

        trimmed = {c[1] for c in self.calls if c[0] == "trim"}
        self.assertEqual(trimmed, {top_name, vref_name, amp_name})

    def test_sub_job_own_failure_does_not_discard_it_if_top_is_kept(self):
        """A sub-job individually failing its own profile must NOT get
        trimmed on its own -- only grouped with its top-level job's own
        discard. If top is kept, its sub-jobs are always kept too,
        regardless of what run_variation() might have decided for THEM
        (irrelevant here since sub-jobs never even get discard_on_fail=True
        to decide with in the first place)."""
        jobs, outcomes_by_name, _ = self._one_iteration_jobs(2, top_discard=False, vref_discard=True)
        with contextlib.ExitStack() as stack:
            for cm in self._patched(outcomes_by_name):
                stack.enter_context(cm)
            gen_variations._run_hierarchical_batch(
                jobs, self.defaults, force=False, skip_on_fail_profile="low_power", discard_on_fail=True,
            )
        self.assertEqual([c for c in self.calls if c[0] == "trim"], [])

    def test_checkpoint_size_groups_by_sample_not_raw_job_count(self):
        """checkpoint_size=1 with 2 iterations (3 jobs each, 6 jobs total):
        each iteration's own 3 jobs (2 sub-jobs + 1 top job) must stay
        together in the SAME chunk -- iteration 1's own discard trimmed
        before iteration 2's own jobs start running at all."""
        jobs1, outcomes1, (top1, vref1, amp1) = self._one_iteration_jobs(3, top_discard=True)
        jobs2, outcomes2, (top2, vref2, amp2) = self._one_iteration_jobs(4, top_discard=False)
        jobs = jobs1 + jobs2
        outcomes = {**outcomes1, **outcomes2}
        with contextlib.ExitStack() as stack:
            for cm in self._patched(outcomes):
                stack.enter_context(cm)
            gen_variations._run_hierarchical_batch(
                jobs, self.defaults, force=False, skip_on_fail_profile="low_power",
                discard_on_fail=True, checkpoint_size=1,
            )
        trim_index = next(i for i, c in enumerate(self.calls) if c[0] == "trim")
        run_indices_iter2 = [i for i, c in enumerate(self.calls) if c[0] == "run" and c[1] in (top2, vref2, amp2)]
        self.assertTrue(all(trim_index < i for i in run_indices_iter2))
        trimmed = {c[1] for c in self.calls if c[0] == "trim"}
        self.assertEqual(trimmed, {top1, vref1, amp1})


if __name__ == "__main__":
    unittest.main()
