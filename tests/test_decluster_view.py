"""Unit tests for the pure (non-Tk) helpers in analog_designer/gui/
decluster_view.py -- _negate() and _all_criteria(). The Toplevel window
itself is Tk-widget wiring, verified manually against a real project
(same convention as variations_table.py/variation_detail.py, which have
no dedicated test files of their own either)."""
import unittest

from analog_designer.gui.decluster_view import _all_criteria, _negate


class NegateTests(unittest.TestCase):
    def test_negates_a_plain_float(self):
        self.assertEqual(_negate(3.5), -3.5)

    def test_negates_a_tuple_elementwise(self):
        self.assertEqual(_negate((2, -1.0, 5)), (-2, 1.0, -5))

    def test_double_negation_is_identity(self):
        self.assertEqual(_negate(_negate((1, 2.5))), (1, 2.5))


class AllCriteriaTests(unittest.TestCase):
    def _summary(self, name, stats, profiles=()):
        return {"variation": name, "metric_stats": stats, "profiles": list(profiles)}

    def test_includes_every_measured_metric_field(self):
        summaries = [self._summary("v1", {("dc", "current"): {"typical": 1.0, "min": 0.9, "max": 1.1}})]
        entries = _all_criteria(summaries, {"profiles": {}})
        labels = [label for label, _, _ in entries]
        self.assertIn("current (dc): typical, lowest first", labels)
        self.assertIn("current (dc): typical, highest first", labels)
        self.assertIn("current (dc): min, lowest first", labels)
        self.assertNotIn("current (dc): mean, lowest first", labels)

    def test_mean_std_appear_only_when_actually_reported(self):
        summaries = [
            self._summary("v1", {("mc", "vref"): {"typical": None, "mean": 1.2, "std": 0.01}}),
        ]
        entries = _all_criteria(summaries, {"profiles": {}})
        labels = [label for label, _, _ in entries]
        self.assertIn("vref (mc): mean, lowest first", labels)
        self.assertIn("vref (mc): std, highest first", labels)

    def test_unions_fields_across_summaries(self):
        """One variation only has typical/min/max so far, another already
        has mean/std -- the combined list must offer mean/std too, since
        SOME variation can be ranked by it."""
        summaries = [
            self._summary("v1", {("mc", "vref"): {"typical": None, "min": 1.1, "max": 1.3}}),
            self._summary("v2", {("mc", "vref"): {"typical": None, "mean": 1.2, "std": 0.02, "min": 1.0, "max": 1.4}}),
        ]
        entries = _all_criteria(summaries, {"profiles": {}})
        labels = {label for label, _, _ in entries}
        self.assertIn("vref (mc): mean, lowest first", labels)

    def test_includes_every_declared_profile(self):
        entries = _all_criteria([], {"profiles": {"low_power": {}, "high_perf": {}}})
        subjects = {c["profile"] for _, c, _ in entries}
        self.assertEqual(subjects, {"low_power", "high_perf"})

    def test_profile_entries_cover_fom_and_constraints(self):
        entries = _all_criteria([], {"profiles": {"low_power": {}}})
        fields = {c["field"] for _, c, _ in entries}
        self.assertEqual(fields, {"fom", "passed", "failed"})

    def test_metrics_sorted_for_a_stable_combobox_order(self):
        summaries = [self._summary("v1", {
            ("z_test", "m"): {"typical": 1.0}, ("a_test", "m"): {"typical": 1.0},
        })]
        entries = _all_criteria(summaries, {"profiles": {}})
        subjects_in_order = [(c["test"]) for _, c, _ in entries]
        self.assertEqual(subjects_in_order, sorted(subjects_in_order))


if __name__ == "__main__":
    unittest.main()
