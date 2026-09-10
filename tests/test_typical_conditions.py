"""Unit tests for run_sim.typical_conditions() -- the {axis: value}
resolution a parser's evaluate() uses (via tb/_shared/parser_common's
typical_min_max()) to pick which simulated run counts as "typical",
instead of a hardcoded convention (corner=="tt", temperature=="25")
duplicated ad hoc across parser files. No docker involved."""
import unittest

from analog_designer.sim.run_sim import typical_conditions

_DEFAULTS = {"corner": "tt", "temperature": "25", "vdd": "1.8"}


class TypicalConditionsTests(unittest.TestCase):
    def test_falls_back_to_defaults_when_test_declares_no_override(self):
        result = typical_conditions({"conditions": {}}, _DEFAULTS)
        self.assertEqual(result, _DEFAULTS)

    def test_test_specific_override_wins_over_defaults(self):
        test_cfg = {"conditions": {"typical": {"corner": "ss"}}}
        result = typical_conditions(test_cfg, _DEFAULTS)
        self.assertEqual(result["corner"], "ss")
        self.assertEqual(result["temperature"], "25")  # untouched default

    def test_missing_conditions_key_entirely_still_falls_back(self):
        result = typical_conditions({}, _DEFAULTS)
        self.assertEqual(result, _DEFAULTS)

    def test_result_is_a_superset_carrying_every_default_axis(self):
        """Deliberately carries axes (e.g. vdd) this particular test's own
        condition_matrix() never varies -- callers (typical_min_max())
        only match on the keys present in a given run's own conditions
        dict, so the extra keys are harmless."""
        result = typical_conditions({"conditions": {}}, _DEFAULTS)
        self.assertIn("vdd", result)


if __name__ == "__main__":
    unittest.main()
