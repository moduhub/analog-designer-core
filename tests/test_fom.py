"""Unit tests for analog_designer/results/fom.py's {typical, min, max}
metric handling -- pure Python, synthetic metric dicts, no docker/results.jsonl
involved. Covers metrics_to_variables()'s <slug>/<slug>_observed_min/
<slug>_observed_max variable set and constraint_satisfied()'s both-direction
(minimum checked against observed min, maximum checked against observed
max) bound checking -- the fix for a metric only being able to satisfy one
direction when it reported a single float."""
import unittest

from analog_designer.results.fom import classify, constraint_satisfied, constraints_violated, metrics_to_variables, slugify


def _metric(name, typical, lo, hi):
    return {"metric": name, "typical": typical, "min": lo, "max": hi}


def _mc_metric(name, mean, std, lo, hi):
    """A Monte Carlo metric (see tb/_shared/parser_common.mc_stats()) --
    typical is None, no single run is representative."""
    return {"metric": name, "typical": None, "mean": mean, "std": std, "min": lo, "max": hi}


class MetricsToVariablesTests(unittest.TestCase):
    def test_slug_is_the_typical_value(self):
        variables = metrics_to_variables([_metric("PSRR @ 1kHz", 65.0, 60.0, 70.0)])
        self.assertEqual(variables[slugify("PSRR @ 1kHz")], 65.0)

    def test_observed_min_max_are_exposed_under_non_colliding_names(self):
        variables = metrics_to_variables([_metric("PSRR @ 1kHz", 65.0, 60.0, 70.0)])
        slug = slugify("PSRR @ 1kHz")
        self.assertEqual(variables[f"{slug}_observed_min"], 60.0)
        self.assertEqual(variables[f"{slug}_observed_max"], 70.0)

    def test_mc_metric_omits_plain_slug_but_keeps_observed_range(self):
        """A Monte Carlo metric's typical is None -- metrics_to_variables()
        must not fake a <slug> value (e.g. by silently using the mean), but
        <slug>_observed_min/_observed_max (needed for constraint checking)
        stay present regardless."""
        variables = metrics_to_variables([_mc_metric("Vref output voltage (mismatch)", 1.2, 0.01, 1.17, 1.23)])
        slug = slugify("Vref output voltage (mismatch)")
        self.assertNotIn(slug, variables)
        self.assertEqual(variables[f"{slug}_observed_min"], 1.17)
        self.assertEqual(variables[f"{slug}_observed_max"], 1.23)

    def test_mc_metric_exposes_mean_and_std(self):
        variables = metrics_to_variables([_mc_metric("Vref output voltage (mismatch)", 1.2, 0.01, 1.17, 1.23)])
        slug = slugify("Vref output voltage (mismatch)")
        self.assertEqual(variables[f"{slug}_mean"], 1.2)
        self.assertEqual(variables[f"{slug}_std"], 0.01)

    def test_point_metric_has_no_mean_or_std(self):
        variables = metrics_to_variables([_metric("PSRR @ 1kHz", 65.0, 60.0, 70.0)])
        slug = slugify("PSRR @ 1kHz")
        self.assertNotIn(f"{slug}_mean", variables)
        self.assertNotIn(f"{slug}_std", variables)


class ConstraintSatisfiedTests(unittest.TestCase):
    def test_maximum_bound_checked_against_observed_max_not_typical(self):
        """A metric whose typical reading is well within spec but whose
        worst-case (max) observed value breaches a "maximum" bound must
        fail -- a single float couldn't represent this before."""
        variables = metrics_to_variables([_metric("current", 0.5, 0.4, 1.5)])
        slug = slugify("current")
        self.assertFalse(constraint_satisfied(slug, {"maximum": 1.0}, variables))

    def test_minimum_bound_checked_against_observed_min_not_typical(self):
        variables = metrics_to_variables([_metric("psrr", 65.0, 40.0, 70.0)])
        slug = slugify("psrr")
        self.assertFalse(constraint_satisfied(slug, {"minimum": 50.0}, variables))

    def test_both_directions_satisfied_when_whole_range_is_in_bounds(self):
        variables = metrics_to_variables([_metric("bias_current", 100.0, 85.0, 115.0)])
        slug = slugify("bias_current")
        self.assertTrue(constraint_satisfied(slug, {"minimum": 80.0, "maximum": 120.0}, variables))

    def test_unknown_slug_counts_against_it(self):
        self.assertFalse(constraint_satisfied("nonexistent", {"maximum": 1.0}, {}))

    def test_mc_metric_still_checkable_despite_no_plain_slug(self):
        """constraint_satisfied() must key off the metric's presence (via
        _observed_min/_observed_max), not the plain <slug> -- which a Monte
        Carlo metric never has (regression: it used to check `slug in
        variables`, which would wrongly reject every MC metric)."""
        variables = metrics_to_variables([_mc_metric("vbg (mismatch)", 1.2, 0.01, 1.17, 1.23)])
        slug = slugify("vbg (mismatch)")
        self.assertTrue(constraint_satisfied(slug, {"minimum": 1.1, "maximum": 1.3}, variables))
        self.assertFalse(constraint_satisfied(slug, {"minimum": 1.2}, variables))


class ClassifyTests(unittest.TestCase):
    def test_figure_of_merit_formula_uses_typical_value(self):
        block_cfg = {
            "profiles": {
                "demo": {
                    "constraints": {"current": {"maximum": 1.0}},
                    "figure_of_merit": "current_max / current",
                }
            }
        }
        metrics = [_metric("current", 0.5, 0.4, 0.6)]
        results = classify(block_cfg, metrics)
        self.assertEqual(len(results), 1)
        self.assertAlmostEqual(results[0]["fom"], 1.0 / 0.5)
        self.assertTrue(results[0]["matched"])

    def test_profile_fails_when_worst_case_breaches_bound_even_if_typical_passes(self):
        block_cfg = {"profiles": {"demo": {"constraints": {"current": {"maximum": 1.0}}}}}
        metrics = [_metric("current", 0.5, 0.4, 1.5)]
        results = classify(block_cfg, metrics)
        self.assertFalse(results[0]["matched"])
        self.assertEqual(results[0]["n_satisfied"], 0)

    def test_missing_metric_is_not_satisfied_but_not_failed_either(self):
        """A test not run yet (or skipped by --skip-on-fail) leaves its
        metric absent: it can't count as passed, but it isn't a measured
        failure -- n_failed counts only out-of-bounds measurements."""
        block_cfg = {"profiles": {"demo": {"constraints": {
            "current": {"maximum": 1.0}, "psrr": {"minimum": 50.0}, "noise": {"maximum": 1.0},
        }}}}
        metrics = [_metric("current", 0.5, 0.4, 0.6), _metric("psrr", 45.0, 40.0, 48.0)]
        result = classify(block_cfg, metrics)[0]
        self.assertEqual((result["n_satisfied"], result["n_failed"], result["n_missing"]), (1, 1, 1))
        self.assertEqual(result["n_constraints"], 3)


class ConstraintsViolatedTests(unittest.TestCase):
    """max_failures's own tolerance (default 0 == the original "any single
    violation ends it" behavior) -- see run_variation()'s own skip_on_fail
    early exit, the sole caller of this function."""

    def _variables(self, **maxima):
        """One violated/satisfied constraint per kwarg: current=1.5 with a
        maximum=1.0 bound is violated (observed_max breaches it), current=0.5
        is satisfied."""
        metrics = [_metric(name, value, value, value) for name, value in maxima.items()]
        return metrics_to_variables(metrics)

    def _constraints(self, *names):
        return {name: {"maximum": 1.0} for name in names}

    def test_default_max_failures_stops_at_the_first_violation(self):
        constraints = self._constraints("a", "b")
        variables = self._variables(a=1.5, b=0.5)  # a violated, b satisfied
        self.assertTrue(constraints_violated(constraints, variables))

    def test_no_violations_is_never_flagged_regardless_of_max_failures(self):
        constraints = self._constraints("a", "b")
        variables = self._variables(a=0.5, b=0.5)
        self.assertFalse(constraints_violated(constraints, variables, max_failures=5))

    def test_violations_within_max_failures_are_tolerated(self):
        constraints = self._constraints("a", "b", "c")
        variables = self._variables(a=1.5, b=1.5, c=0.5)  # 2 violated, 1 satisfied
        self.assertFalse(constraints_violated(constraints, variables, max_failures=2))

    def test_violations_beyond_max_failures_still_trip_it(self):
        constraints = self._constraints("a", "b", "c")
        variables = self._variables(a=1.5, b=1.5, c=1.5)  # 3 violated
        self.assertTrue(constraints_violated(constraints, variables, max_failures=2))

    def test_a_metric_not_simulated_yet_never_counts_as_a_violation(self):
        constraints = {"a": {"maximum": 1.0}, "not_run_yet": {"maximum": 1.0}}
        variables = self._variables(a=0.5)
        self.assertFalse(constraints_violated(constraints, variables, max_failures=0))


if __name__ == "__main__":
    unittest.main()
