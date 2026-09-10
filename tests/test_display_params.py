"""Unit tests for run_sim.py's read-only parameter-preview helpers
(resolve_display_params(), calculated_param_descriptions(),
resolve_sub_block_params()) -- the GUI's Parameters panel uses these to
show every calculated (derived_parameters) value alongside the free ones,
without ever writing sch/<block>.sch the way the real materialization path
(resolve_materialization_params()/materialize_sub_blocks()) does. Pure
Python plus some synthetic fixtures, grounded against the real reference
project (ihp_mh_ip__cmos_vref) where useful -- no docker needed."""
import json
import unittest
from pathlib import Path

from analog_designer.core import workspace
from analog_designer.sim.run_sim import (
    StaleParameterSchema, calculated_param_descriptions, resolve_display_params,
)

REFERENCE_ROOT = Path(__file__).resolve().parents[2] / "ihp_mh_ip__cmos_vref"

_FAKE_TOPOLOGY_CFG = {
    "parameters": {
        "unit_width": {"default": "5u", "min": "0.3u", "max": "10u", "integer": False},
        "a_factor": {"default": "20", "min": "1", "max": "20", "integer": True},
        "b_factor": {"default": "1", "min": "1", "max": "20", "integer": True},
    },
    "derived_parameters": {
        "width_groups": [
            {
                "id": "fake_group",
                "description": "A and B share a unit width, scaled by their own integer factor.",
                "base": "unit_width",
                "members": {
                    "a_width": {"factor": "a_factor"},
                    "b_width": {"factor": "b_factor"},
                },
            }
        ]
    },
}

_FAKE_TOPOLOGY_CFG_NO_GROUP_DESCRIPTION = {
    "parameters": _FAKE_TOPOLOGY_CFG["parameters"],
    "derived_parameters": {
        "width_groups": [{
            "id": "fake_group", "base": "unit_width",
            "members": {"a_width": {"factor": "a_factor"}},
        }]
    },
}


def _require_reference():
    if not REFERENCE_ROOT.exists():
        raise unittest.SkipTest(f"reference project not checked out at {REFERENCE_ROOT}")


class CalculatedParamDescriptionsTests(unittest.TestCase):
    def test_width_group_member_inherits_group_description(self):
        descriptions = calculated_param_descriptions(_FAKE_TOPOLOGY_CFG)
        self.assertEqual(descriptions["a_width"], "A and B share a unit width, scaled by their own integer factor.")
        self.assertEqual(descriptions["b_width"], descriptions["a_width"])

    def test_width_group_member_falls_back_to_base_times_factor_when_group_has_no_description(self):
        descriptions = calculated_param_descriptions(_FAKE_TOPOLOGY_CFG_NO_GROUP_DESCRIPTION)
        self.assertEqual(descriptions["a_width"], "a_width = unit_width * a_factor")

    def test_free_parameters_are_not_included(self):
        descriptions = calculated_param_descriptions(_FAKE_TOPOLOGY_CFG)
        self.assertNotIn("unit_width", descriptions)
        self.assertNotIn("a_factor", descriptions)

    def test_no_derived_parameters_is_empty(self):
        self.assertEqual(calculated_param_descriptions({"parameters": {}}), {})

    def test_import_params_entry_uses_its_own_description(self):
        topology_cfg = {"derived_parameters": {"import_params": {
            "x1_m3_width": {"from": "X1", "source": "m3_width", "description": "cmos_vref's own M3 width"},
        }}}
        descriptions = calculated_param_descriptions(topology_cfg)
        self.assertEqual(descriptions["x1_m3_width"], "cmos_vref's own M3 width")

    def test_import_params_entry_falls_back_to_source_reference(self):
        topology_cfg = {"derived_parameters": {"import_params": {
            "x1_pbias_length": {"from": "X1", "source": "pbias_length"},
        }}}
        descriptions = calculated_param_descriptions(topology_cfg)
        self.assertEqual(descriptions["x1_pbias_length"], "X1.pbias_length")

    def test_import_metrics_entry_uses_its_own_description(self):
        topology_cfg = {"derived_parameters": {"import_metrics": {
            "x1_ref_current_typical": {
                "from": "X1", "test": "reference_current", "metric": "Core reference current (M3 branch)",
                "description": "X1's own measured reference current",
            },
        }}}
        descriptions = calculated_param_descriptions(topology_cfg)
        self.assertEqual(descriptions["x1_ref_current_typical"], "X1's own measured reference current")

    def test_import_metrics_entry_falls_back_to_a_readable_summary(self):
        topology_cfg = {"derived_parameters": {"import_metrics": {
            "x1_ref_current_typical": {
                "from": "X1", "test": "reference_current", "metric": "Core reference current (M3 branch)",
            },
        }}}
        descriptions = calculated_param_descriptions(topology_cfg)
        self.assertIn("Core reference current (M3 branch)", descriptions["x1_ref_current_typical"])
        self.assertIn("reference_current", descriptions["x1_ref_current_typical"])

    def test_formula_entry_falls_back_to_its_own_expr(self):
        topology_cfg = {"derived_parameters": {"formulas": {
            "r1_length": {"expr": "rfeedback_total * 0.1"},
        }}}
        descriptions = calculated_param_descriptions(topology_cfg)
        self.assertEqual(descriptions["r1_length"], "rfeedback_total * 0.1")


class ResolveDisplayParamsTests(unittest.TestCase):
    def test_free_and_calculated_values_both_present(self):
        params = {"unit_width": "5u", "a_factor": "20", "b_factor": "1"}
        resolved = resolve_display_params(_FAKE_TOPOLOGY_CFG, params)
        self.assertEqual(resolved["unit_width"], "5u")
        self.assertEqual(resolved["a_width"], "100u")
        self.assertEqual(resolved["b_width"], "5u")

    def test_stale_schema_degrades_to_free_params_only_instead_of_raising(self):
        """A variation whose stored params predate this topology's own
        width_groups migration (missing a_factor/b_factor) must still show
        SOMETHING in the GUI, not blow up the whole panel."""
        old_style_params = {"unit_width": "5u"}
        resolved = resolve_display_params(_FAKE_TOPOLOGY_CFG, old_style_params)
        self.assertEqual(resolved, old_style_params)
        self.assertNotIn("a_width", resolved)

    def test_no_derived_parameters_at_all_is_a_plain_passthrough(self):
        params = {"m1_width": "2u"}
        resolved = resolve_display_params({"parameters": {}}, params)
        self.assertEqual(resolved, params)


class ResolveDisplayParamsHierarchicalTests(unittest.TestCase):
    """resolve_display_params() against a synthetic hierarchical (sub_blocks)
    topology, to cover the "imported metric not available yet" graceful
    degradation without needing real sim/results.jsonl data or docker."""

    _CONFIG = {
        "blocks": {
            "leaf": {"topologies": {"default": {
                "parameters": {"m3_width": {"default": "50u"}},
            }}},
            "top": {"topologies": {"default": {
                "sub_blocks": {"X1": {"block": "leaf", "topology": "default"}},
                "parameters": {"amp_bias_factor": {"default": "3"}},
                "derived_parameters": {
                    "import_params": {
                        # a pure passthrough imports DIRECTLY under the name the
                        # schematic needs -- no formula wrapper required, and
                        # crucially resolves in the import_params step, which
                        # runs (and can succeed) independently of import_metrics,
                        # so it survives even when a DIFFERENT derived value's
                        # own import_metrics entry has no sim data yet (see
                        # test_unresolvable_import_metric_is_omitted_not_raised
                        # below).
                        "pbias_length_copy": {"from": "X1", "source": "m3_width"},
                        "x1_m3_width": {"from": "X1", "source": "m3_width"},
                    },
                    "import_metrics": {
                        "x1_core_current": {"from": "X1", "test": "reference_current", "metric": "Core current"},
                    },
                    "formulas": {
                        "amp_bias_width": {"expr": "x1_m3_width * 100 / x1_core_current", "unit_suffix": "u"},
                    },
                },
            }}},
        },
    }

    def setUp(self):
        self._saved_config = workspace.CONFIG
        self.addCleanup(lambda: setattr(workspace, "CONFIG", self._saved_config))
        workspace.CONFIG = self._CONFIG

    def test_metric_less_import_resolves_without_any_sim_data(self):
        topology_cfg = self._CONFIG["blocks"]["top"]["topologies"]["default"]
        params = {"amp_bias_factor": "3"}
        resolved = resolve_display_params(topology_cfg, params)
        self.assertEqual(resolved["pbias_length_copy"], "50u")

    def test_unresolvable_import_metric_is_omitted_not_raised(self):
        """No sim/results.jsonl entry exists for "reference_current" here --
        resolve_import_metrics() would raise MissingCrossBlockMetric;
        resolve_display_params() must swallow that and still return the
        OTHER (import_params-resolved) value it already resolved."""
        topology_cfg = self._CONFIG["blocks"]["top"]["topologies"]["default"]
        params = {"amp_bias_factor": "3"}
        resolved = resolve_display_params(topology_cfg, params)
        self.assertNotIn("amp_bias_width", resolved)
        self.assertIn("pbias_length_copy", resolved)

    def test_calculated_names_include_both_resolved_and_unresolved(self):
        topology_cfg = self._CONFIG["blocks"]["top"]["topologies"]["default"]
        descriptions = calculated_param_descriptions(topology_cfg)
        self.assertIn("pbias_length_copy", descriptions)
        self.assertIn("amp_bias_width", descriptions)


class ReferenceProjectSmokeTests(unittest.TestCase):
    def test_cmos_vref_default_calculated_widths_match_known_values(self):
        _require_reference()
        doc = json.loads((REFERENCE_ROOT / "params" / "cmos_vref" / "default.json").read_text(encoding="utf-8"))
        topology_cfg = {"parameters": doc["parameters"], "derived_parameters": doc["derived_parameters"]}
        params = {name: pdef["default"] for name, pdef in doc["parameters"].items()}
        resolved = resolve_display_params(topology_cfg, params)
        self.assertEqual(resolved["m6_width"], "100u")
        self.assertEqual(resolved["m7_width"], "5u")
        self.assertEqual(resolved["m8_width"], "10u")
        self.assertEqual(resolved["m9_width"], "10u")

    def test_top_default_never_raises_regardless_of_sim_data_presence(self):
        """Whether or not sim/results.jsonl has cmos_vref's reference_current/
        temp_sweep results for X1's own chosen variation, resolve_display_params()
        must return cleanly -- amp_bias_width/rbot_nominal/r1..r8_length just
        won't be in the result if that data isn't there yet."""
        _require_reference()
        workspace.open_folder(str(REFERENCE_ROOT), block="top", topology="default")
        topology_cfg = workspace.CONFIG["blocks"]["top"]["topologies"]["default"]
        params = {n: pdef["default"] for n, pdef in topology_cfg["parameters"].items()}
        resolved = resolve_display_params(topology_cfg, params)  # must not raise
        self.assertEqual(resolved["pbias_length"], "10u")  # a plain passthrough, always resolvable


if __name__ == "__main__":
    unittest.main()
