"""Explicit finger count via derived_parameters.constants + a `ceil(...)`
formula (ng = ceil(w / w_finger_max)) -- pure Python, no docker."""
import unittest

from analog_designer.sim import check_params
from analog_designer.sim.run_sim import formula_names, resolve_formulas

_CFG = {"derived_parameters": {
    "constants": {"w_finger_max": {"value": "10u"}},
    "formulas": {"m1_ng": {"expr": "ceil(w / w_finger_max)", "unit_suffix": ""}},
}}


def _ng(width):
    return resolve_formulas(_CFG, {"w": width})["m1_ng"]


class CeilFormulaTest(unittest.TestCase):
    def test_narrow_device_is_one_finger(self):
        self.assertEqual(_ng("2.5u"), "1")

    def test_wide_device_is_split_into_fingers(self):
        self.assertEqual(_ng("52.33u"), "6")

    def test_exact_multiples_do_not_round_up(self):
        self.assertEqual(_ng("10u"), "1")
        self.assertEqual(_ng("20u"), "2")
        self.assertEqual(_ng("30u"), "3")

    def test_just_over_the_limit_needs_another_finger(self):
        self.assertEqual(_ng("10.5u"), "2")

    def test_constant_is_exposed_and_free_params_pass_through(self):
        out = resolve_formulas(_CFG, {"w": "5u"})
        self.assertEqual((out["w"], out["w_finger_max"]), ("5u", "10u"))

    def test_called_function_name_is_not_a_referenced_name(self):
        self.assertEqual(formula_names("ceil(w / w_finger_max)"), {"w", "w_finger_max"})

    def test_unknown_function_is_rejected(self):
        cfg = {"derived_parameters": {"formulas": {"x": {"expr": "sqrt(w)"}}}}
        with self.assertRaises(ValueError):
            resolve_formulas(cfg, {"w": "4u"})


class CheckCeilFormulaTest(unittest.TestCase):
    _PARAMS = {"w": {"default": "5u", "min": "1u", "max": "40u"}}
    _SCH = "w='w'\nng='m1_ng'\nm=1\n"

    def test_valid_setup_reports_no_problem(self):
        orphans, missing, errors = check_params.check(self._PARAMS, self._SCH, _CFG["derived_parameters"])
        self.assertEqual((orphans, missing, errors), ([], [], []))

    def test_unreferenced_constant_is_flagged(self):
        cfg = {"constants": {"w_finger_max": {"value": "10u"}}, "formulas": {}}
        _, _, errors = check_params.check(self._PARAMS, "w='w'\n", cfg)
        self.assertTrue(any("w_finger_max" in msg for _, msg in errors))

    def test_formula_referencing_an_undeclared_name_is_flagged(self):
        cfg = {"constants": {}, "formulas": {"m1_ng": {"expr": "ceil(nope / 10)"}}}
        _, _, errors = check_params.check(self._PARAMS, self._SCH, cfg)
        self.assertTrue(any("nope" in msg for _, msg in errors))


if __name__ == "__main__":
    unittest.main()
