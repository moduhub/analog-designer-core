"""Unit tests for the derived_parameters (width_base + integer factor) scheme
-- pure Python, no docker needed. Covers analog_designer/sim/run_sim.py's
resolve_derived_params()/StaleParameterSchema and
analog_designer/sim/check_params.py's derived-parameters-aware check(),
grounded against the same real reference project as tests/test_layout_devices.py
where useful (confirms the migration is a byte-identical no-op for the
default variation's M6/M7/M8/M9 widths), plus small synthetic fixtures for
the parts that don't need a real project checked out.
"""
import json
import unittest
from pathlib import Path

from analog_designer.sim import check_params
from analog_designer.sim.run_sim import StaleParameterSchema, check_unresolved, resolve_derived_params, substitute_params

REFERENCE_PARAMS = (
    Path(__file__).resolve().parents[2] / "ihp_mh_ip__cmos_vref" / "params" / "cmos_vref" / "default.json"
)
REFERENCE_SCH = Path(__file__).resolve().parents[2] / "ihp_mh_ip__cmos_vref" / "sch" / "cmos_vref" / "cmos_vref_default.sch"

# A minimal two-group block_cfg, independent of any real project -- mirrors
# the shape params/cmos_vref/default.json's derived_parameters/parameters
# sections splice into block_cfg (see workspace.resolve_parameters_files()).
_FAKE_BLOCK_CFG = {
    "parameters": {
        "unit_width": {"default": "5u", "min": "0.3u", "max": "10u", "integer": False},
        "a_factor": {"default": "20", "min": "1", "max": "20", "integer": True},
        "b_factor": {"default": "1", "min": "1", "max": "20", "integer": True},
    },
    "derived_parameters": {
        "width_groups": [
            {
                "id": "fake_group",
                "base": "unit_width",
                "members": {
                    "a_width": {"factor": "a_factor"},
                    "b_width": {"factor": "b_factor"},
                },
            }
        ]
    },
}


def _require_reference():
    if not REFERENCE_PARAMS.exists() or not REFERENCE_SCH.exists():
        raise unittest.SkipTest(f"reference project not checked out at {REFERENCE_PARAMS.parent}")


class ResolveDerivedParamsTests(unittest.TestCase):
    def test_multiplies_base_by_factor(self):
        params = {"unit_width": "5u", "a_factor": "20", "b_factor": "1"}
        derived = resolve_derived_params(_FAKE_BLOCK_CFG, params)
        self.assertEqual(derived["a_width"], "100u")
        self.assertEqual(derived["b_width"], "5u")

    def test_free_params_pass_through_unchanged(self):
        params = {"unit_width": "5u", "a_factor": "20", "b_factor": "1"}
        derived = resolve_derived_params(_FAKE_BLOCK_CFG, params)
        self.assertEqual(derived["unit_width"], "5u")
        self.assertEqual(derived["a_factor"], "20")

    def test_no_width_groups_is_a_no_op(self):
        params = {"m1_width": "2u"}
        derived = resolve_derived_params({"parameters": {}}, params)
        self.assertEqual(derived, params)

    def test_missing_base_raises_stale_parameter_schema(self):
        old_style_params = {"a_width": "100u", "b_width": "5u"}
        with self.assertRaises(StaleParameterSchema):
            resolve_derived_params(_FAKE_BLOCK_CFG, old_style_params)

    def test_missing_factor_raises_stale_parameter_schema(self):
        params = {"unit_width": "5u", "a_factor": "20"}  # b_factor missing
        with self.assertRaises(StaleParameterSchema):
            resolve_derived_params(_FAKE_BLOCK_CFG, params)

    def test_reference_project_default_reproduces_todays_exact_widths(self):
        """The whole point of this migration's chosen default/min/max values
        (see the params/cmos_vref/default.json plan) is that it's a no-op
        for the default variation -- m6_width/m7_width/m8_width/m9_width
        must come out byte-identical to what they were before the
        migration (100u/5u/10u/10u)."""
        _require_reference()
        doc = json.loads(REFERENCE_PARAMS.read_text(encoding="utf-8"))
        block_cfg = {"parameters": doc["parameters"], "derived_parameters": doc["derived_parameters"]}
        params = {name: pdef["default"] for name, pdef in doc["parameters"].items()}
        derived = resolve_derived_params(block_cfg, params)
        self.assertEqual(derived["m6_width"], "100u")
        self.assertEqual(derived["m7_width"], "5u")
        self.assertEqual(derived["m8_width"], "10u")
        self.assertEqual(derived["m9_width"], "10u")

    def test_reference_project_materializes_with_no_unresolved_tokens(self):
        _require_reference()
        doc = json.loads(REFERENCE_PARAMS.read_text(encoding="utf-8"))
        block_cfg = {"parameters": doc["parameters"], "derived_parameters": doc["derived_parameters"]}
        params = {name: pdef["default"] for name, pdef in doc["parameters"].items()}
        derived = resolve_derived_params(block_cfg, params)
        materialized = substitute_params(REFERENCE_SCH.read_text(encoding="utf-8"), derived)
        check_unresolved(materialized, "cmos_vref_default.sch")  # raises/exits on leftover placeholders


class CheckParamsDerivedTests(unittest.TestCase):
    """check_params.check() against a schematic snippet using both groups'
    derived names as literal '<name>' tokens, mirroring what
    cmos_vref_default.sch actually contains for M6/M7."""

    _SCH_TEXT = "w='a_width' w='b_width'"

    def test_base_and_factor_are_not_flagged_as_orphans(self):
        orphans, missing, errors = check_params.check(
            _FAKE_BLOCK_CFG["parameters"], self._SCH_TEXT, _FAKE_BLOCK_CFG["derived_parameters"],
        )
        self.assertEqual(orphans, [])
        self.assertEqual(errors, [])

    def test_derived_names_are_not_flagged_as_missing(self):
        orphans, missing, errors = check_params.check(
            _FAKE_BLOCK_CFG["parameters"], self._SCH_TEXT, _FAKE_BLOCK_CFG["derived_parameters"],
        )
        self.assertEqual(missing, [])

    def test_typo_in_base_is_caught(self):
        bad_cfg = {"width_groups": [{
            "id": "fake_group", "base": "no_such_param",
            "members": {"a_width": {"factor": "a_factor"}, "b_width": {"factor": "b_factor"}},
        }]}
        _, _, errors = check_params.check(_FAKE_BLOCK_CFG["parameters"], self._SCH_TEXT, bad_cfg)
        self.assertTrue(any("no_such_param" in msg for _, msg in errors))

    def test_typo_in_factor_is_caught(self):
        bad_cfg = {"width_groups": [{
            "id": "fake_group", "base": "unit_width",
            "members": {"a_width": {"factor": "no_such_factor"}, "b_width": {"factor": "b_factor"}},
        }]}
        _, _, errors = check_params.check(_FAKE_BLOCK_CFG["parameters"], self._SCH_TEXT, bad_cfg)
        self.assertTrue(any("no_such_factor" in msg for _, msg in errors))

    def test_duplicate_derived_name_across_groups_is_caught(self):
        dup_cfg = {"width_groups": [
            {"id": "group1", "base": "unit_width", "members": {"a_width": {"factor": "a_factor"}}},
            {"id": "group2", "base": "unit_width", "members": {"a_width": {"factor": "b_factor"}}},
        ]}
        _, _, errors = check_params.check(_FAKE_BLOCK_CFG["parameters"], self._SCH_TEXT, dup_cfg)
        self.assertTrue(any("produced by both" in msg for _, msg in errors))

    def test_non_integer_factor_is_caught(self):
        params_with_float_factor = dict(_FAKE_BLOCK_CFG["parameters"])
        params_with_float_factor["a_factor"] = {"default": "20", "min": "1", "max": "20"}  # no "integer": True
        _, _, errors = check_params.check(
            params_with_float_factor, self._SCH_TEXT, _FAKE_BLOCK_CFG["derived_parameters"],
        )
        self.assertTrue(any("should declare" in msg for _, msg in errors))

    def test_reference_project_has_zero_derived_parameter_problems(self):
        _require_reference()
        doc = json.loads(REFERENCE_PARAMS.read_text(encoding="utf-8"))
        sch_text = REFERENCE_SCH.read_text(encoding="utf-8")
        orphans, missing, errors = check_params.check(doc["parameters"], sch_text, doc["derived_parameters"])
        self.assertEqual(orphans, [])
        self.assertEqual(missing, [])
        self.assertEqual(errors, [])


if __name__ == "__main__":
    unittest.main()
