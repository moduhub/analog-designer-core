"""Unit tests for the "generator"-backed topology mechanism -- pure Python,
no CSXCAD/openEMS/Docker needed. Covers resolve_generator_params() (the
fast, no-FDTD path used at materialization time and for GUI display) against
a small stub generator module, independent of any real project's actual
GF180MCU spiral generator.
"""
import tempfile
import unittest
from pathlib import Path

from analog_designer.core import workspace
from analog_designer.sim import check_params
from analog_designer.sim.run_sim import resolve_generator_params

_STUB_GENERATOR_SRC = '''
def geometry_from_params(params):
    return {"width_um": float(params["width_um"])}


def load_stack(corner="tt"):
    return {"corner": corner}


def fit_electrical_params(geometry, stack, em_result=None):
    if em_result is None:
        return {"r": 1.0, "c": 2e-12}
    return {"r": em_result["r"], "c": em_result["c"]}
'''


class ResolveGeneratorParamsTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self._saved_root = workspace.PROJECT_ROOT
        self.addCleanup(lambda: setattr(workspace, "PROJECT_ROOT", self._saved_root))
        workspace.PROJECT_ROOT = Path(self._tmp.name)
        (workspace.PROJECT_ROOT / "fake_generator.py").write_text(_STUB_GENERATOR_SRC, encoding="utf-8")

    def test_no_generator_key_is_a_no_op(self):
        block_cfg = {}
        params = {"width_um": "5"}
        resolved = resolve_generator_params(block_cfg, params)
        self.assertEqual(resolved, params)
        self.assertIsNot(resolved, params)  # returns a copy, not the same dict

    def test_merges_placeholder_fit_when_generator_key_present(self):
        block_cfg = {"generator": "fake_generator.py"}
        params = {"width_um": "5"}
        resolved = resolve_generator_params(block_cfg, params)
        self.assertEqual(resolved["width_um"], "5")
        self.assertIn("r", resolved)
        self.assertIn("c", resolved)
        # Values come back as plain SPICE-parseable strings (via
        # format_spice_value), not raw floats -- same convention every
        # other derived_parameters kind (formulas/width_groups) already
        # uses, so substitute_params()'s plain text.replace() works.
        self.assertIsInstance(resolved["r"], str)
        self.assertEqual(float(resolved["r"]), 1.0)
        self.assertAlmostEqual(float(resolved["c"]), 2e-12)

    def test_never_touches_a_real_fdtd_path(self):
        # fit_electrical_params() above only returns em_result-shaped
        # values when em_result is explicitly non-None -- resolve_
        # generator_params() must always call it with em_result=None, i.e.
        # never claim to have run a real EM characterization.
        block_cfg = {"generator": "fake_generator.py"}
        resolved = resolve_generator_params(block_cfg, {"width_um": "5"})
        self.assertEqual(float(resolved["r"]), 1.0)
        self.assertNotEqual(float(resolved["r"]), 999.0)  # sanity: not some em_result-only sentinel


class CheckParamsGeneratorBackedTests(unittest.TestCase):
    """A generator-backed topology's own `parameters` (free geometric
    inputs, e.g. inner_radius_um) are consumed by the generator module's
    own Python code, never as literal '<name>' schematic tokens -- so
    check() must not flag them as orphans when generator_backed=True, the
    same way it already exempts "type": "block_ref" parameters."""

    _PARAM_DEFS = {
        "width_um": {"default": "5", "min": "1", "max": "10", "grid": "0.5"},
        "turns": {"default": "3", "min": "1", "max": "6", "grid": "1", "integer": True},
    }
    _SCH_TEXT = "C {res.sym} 0 0 0 0 {name=R1 value='rs'}\n"
    _DERIVED_CFG = {"generator": {"rs": {"unit": "ohm", "description": "fitted"}}}

    def test_free_params_not_flagged_when_generator_backed(self):
        orphans, missing, errors = check_params.check(
            self._PARAM_DEFS, self._SCH_TEXT, self._DERIVED_CFG, generator_backed=True,
        )
        self.assertEqual(orphans, [])
        self.assertEqual(missing, [])
        self.assertEqual(errors, [])

    def test_same_inputs_flagged_as_orphans_when_not_generator_backed(self):
        # Sanity check: the exemption is opt-in, not accidentally global --
        # without generator_backed=True the same free params are (correctly,
        # for every OTHER topology kind) still real orphans.
        orphans, _, _ = check_params.check(
            self._PARAM_DEFS, self._SCH_TEXT, self._DERIVED_CFG, generator_backed=False,
        )
        self.assertEqual(orphans, ["turns", "width_um"])


if __name__ == "__main__":
    unittest.main()
