"""Unit tests for ihp_mh_ip__cmos_vref/tb/_shared/parser_common.py's
{typical, min, max} reshaping helpers (typical_min_max(), range_pass(),
value_at()) -- the shared logic every tb/<block>/tb_*.py parser's
evaluate() now uses to reduce a flat `runs` list into one uniform metric
instead of each parser hand-rolling its own worst-case/nominal-only/
suffixed-name convention. parser_common.py lives in the sibling IP-project
repo and is loaded dynamically at runtime (see run_sim.load_parser()), not
importable as a normal package -- added to sys.path here the same way
test_derived_params.py reaches into that same sibling repo for its own
reference fixtures."""
import sys
import unittest
from pathlib import Path

_SHARED_DIR = Path(__file__).resolve().parents[2] / "ihp_mh_ip__cmos_vref" / "tb" / "_shared"


def _require_shared():
    if not (_SHARED_DIR / "parser_common.py").exists():
        raise unittest.SkipTest(f"sibling IP project not checked out at {_SHARED_DIR}")
    if str(_SHARED_DIR) not in sys.path:
        sys.path.insert(0, str(_SHARED_DIR))


class TypicalMinMaxTests(unittest.TestCase):
    def setUp(self):
        _require_shared()
        import parser_common
        self.parser_common = parser_common

    def _runs(self):
        return [
            {"conditions": {"corner": "tt", "temperature": "25"}, "value": 100.0},
            {"conditions": {"corner": "ss", "temperature": "25"}, "value": 80.0},
            {"conditions": {"corner": "ff", "temperature": "25"}, "value": 130.0},
        ]

    def test_typical_is_the_run_matching_typical_conditions(self):
        typical = {"corner": "tt", "temperature": "25"}
        result = self.parser_common.typical_min_max(self._runs(), typical, lambda r: r["value"])
        self.assertEqual(result["typical"], 100.0)

    def test_min_max_span_every_run_regardless_of_typical(self):
        typical = {"corner": "tt", "temperature": "25"}
        result = self.parser_common.typical_min_max(self._runs(), typical, lambda r: r["value"])
        self.assertEqual(result["min"], 80.0)
        self.assertEqual(result["max"], 130.0)

    def test_match_keys_absent_from_a_run_are_skipped(self):
        """A run whose test excludes an axis from its outer grid entirely
        (e.g. temperature swept internally, not across runs) shouldn't
        fail to match on a key it doesn't carry."""
        runs = [{"conditions": {"corner": "tt"}, "value": 42.0}]
        typical = {"corner": "tt", "temperature": "25"}
        result = self.parser_common.typical_min_max(runs, typical, lambda r: r["value"])
        self.assertEqual(result["typical"], 42.0)

    def test_no_matching_run_raises(self):
        typical = {"corner": "tt", "temperature": "25"}
        runs = [{"conditions": {"corner": "ss", "temperature": "25"}, "value": 80.0}]
        with self.assertRaises(ValueError):
            self.parser_common.typical_min_max(runs, typical, lambda r: r["value"])

    def test_more_than_one_matching_run_raises(self):
        """A wrong pick here could silently corrupt an import_metrics sizing
        reference -- must raise rather than pick either run."""
        typical = {"corner": "tt", "temperature": "25"}
        runs = [
            {"conditions": {"corner": "tt", "temperature": "25"}, "value": 100.0},
            {"conditions": {"corner": "tt", "temperature": "25"}, "value": 101.0},
        ]
        with self.assertRaises(ValueError):
            self.parser_common.typical_min_max(runs, typical, lambda r: r["value"])


class RangePassTests(unittest.TestCase):
    def setUp(self):
        _require_shared()
        import parser_common
        self.parser_common = parser_common

    def test_passes_when_both_extremes_are_in_spec(self):
        result = {"typical": 100.0, "min": 90.0, "max": 110.0}
        self.assertTrue(self.parser_common.range_pass(result, {"minimum": 80.0, "maximum": 120.0}))

    def test_fails_when_max_breaches_maximum_even_if_typical_is_fine(self):
        result = {"typical": 100.0, "min": 90.0, "max": 130.0}
        self.assertFalse(self.parser_common.range_pass(result, {"maximum": 120.0}))

    def test_fails_when_min_breaches_minimum(self):
        result = {"typical": 100.0, "min": 70.0, "max": 110.0}
        self.assertFalse(self.parser_common.range_pass(result, {"minimum": 80.0}))


class ValueAtTests(unittest.TestCase):
    def setUp(self):
        _require_shared()
        import parser_common
        self.parser_common = parser_common

    def test_exact_match(self):
        xs, ys = [-40, 0, 25, 100], [1.0, 1.1, 1.2, 1.3]
        self.assertEqual(self.parser_common.value_at(xs, ys, 25), 1.2)

    def test_closest_sample_when_no_exact_match(self):
        xs, ys = [-40, 0, 50, 100], [1.0, 1.1, 1.25, 1.3]
        self.assertEqual(self.parser_common.value_at(xs, ys, 20), 1.1)  # 0 is closer to 20 than 50 is


class McStatsTests(unittest.TestCase):
    def setUp(self):
        _require_shared()
        import parser_common
        self.parser_common = parser_common

    def _runs(self, values):
        # conditions are irrelevant to mc_stats() (unlike typical_min_max()) --
        # every run is the same fixed corner/temperature, only mc_seed differs.
        return [{"conditions": {"corner": "tt_mismatch", "temperature": "25", "mc_seed": str(i)}, "value": v}
                for i, v in enumerate(values)]

    def test_typical_is_none(self):
        result = self.parser_common.mc_stats(self._runs([1.0, 2.0, 3.0]), lambda r: r["value"])
        self.assertIsNone(result["typical"])

    def test_mean_and_min_max(self):
        result = self.parser_common.mc_stats(self._runs([1.0, 2.0, 3.0]), lambda r: r["value"])
        self.assertAlmostEqual(result["mean"], 2.0)
        self.assertEqual(result["min"], 1.0)
        self.assertEqual(result["max"], 3.0)

    def test_sample_std_uses_n_minus_1_denominator(self):
        # values [1,2,3]: mean=2, sum((v-mean)**2)=2, /(3-1)=1.0, sqrt=1.0
        result = self.parser_common.mc_stats(self._runs([1.0, 2.0, 3.0]), lambda r: r["value"])
        self.assertAlmostEqual(result["std"], 1.0)

    def test_single_run_has_zero_std_not_a_crash(self):
        result = self.parser_common.mc_stats(self._runs([5.0]), lambda r: r["value"])
        self.assertEqual(result["std"], 0.0)
        self.assertEqual(result["mean"], 5.0)


if __name__ == "__main__":
    unittest.main()
