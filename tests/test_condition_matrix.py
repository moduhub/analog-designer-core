"""Unit tests for run_sim.py's outer per-run condition grid:
condition_matrix() (which (test, condition) combinations get a separate
simulation) and fixed_tb_params() (which conditions{} keys are instead a
single substituted value, never varied). Covers the generalization that
lets an arbitrary conditions{} key with more than one value (e.g. an
'ibias' current-source token) become an outer sweep axis exactly like
corner/temperature already were, instead of erroring -- no docker, no
schematic materialization involved."""
import unittest

from analog_designer.sim.run_sim import MOS_CORNER_SECTION, RES_CORNER_SECTION, condition_matrix, fixed_tb_params

_DEFAULTS = {"corner": "tt", "temperature": "25"}


class ConditionMatrixTests(unittest.TestCase):
    def test_corner_and_temperature_only_is_unchanged(self):
        test_cfg = {"conditions": {"corner": ["tt", "ss"], "temperature": ["-40", "25"]}}
        combos = list(condition_matrix(test_cfg, _DEFAULTS, sweep_axis=None))
        self.assertEqual(len(combos), 4)
        self.assertIn({"corner": "tt", "temperature": "-40"}, combos)
        self.assertIn({"corner": "ss", "temperature": "25"}, combos)

    def test_missing_corner_and_temperature_fall_back_to_defaults(self):
        combos = list(condition_matrix({"conditions": {}}, _DEFAULTS, sweep_axis=None))
        self.assertEqual(combos, [{"corner": "tt", "temperature": "25"}])

    def test_single_valued_extra_key_does_not_join_the_grid(self):
        """A conditions{} key with exactly one value (e.g. a PSRR
        testbench's single-point 'frequency_start') stays a plain fixed
        value -- adding it to the grid would rename every run's label/
        directory for no reason."""
        test_cfg = {"conditions": {"corner": ["tt"], "temperature": ["25"], "frequency_start": ["10"]}}
        combos = list(condition_matrix(test_cfg, _DEFAULTS, sweep_axis=None))
        self.assertEqual(combos, [{"corner": "tt", "temperature": "25"}])

    def test_multi_valued_extra_key_becomes_an_outer_axis(self):
        """The generalization this test file exists for: 'ibias' with 3
        values multiplies the grid by 3, same mechanism as corner/temperature,
        with no schematic change needed."""
        test_cfg = {"conditions": {
            "corner": ["tt", "ss"], "temperature": ["25"], "ibias": ["80n", "100n", "120n"],
        }}
        combos = list(condition_matrix(test_cfg, _DEFAULTS, sweep_axis=None))
        self.assertEqual(len(combos), 6)
        self.assertIn({"corner": "tt", "temperature": "25", "ibias": "80n"}, combos)
        self.assertIn({"corner": "ss", "temperature": "25", "ibias": "120n"}, combos)

    def test_internal_sweep_axis_is_excluded_entirely(self):
        test_cfg = {"conditions": {"corner": ["tt", "ss"], "temperature": ["25"], "iload": ["0", "2e-8"]}}
        combos = list(condition_matrix(test_cfg, _DEFAULTS, sweep_axis=("iload", "iload")))
        self.assertEqual(len(combos), 2)
        self.assertTrue(all("iload" not in c for c in combos))

    def test_temperature_as_internal_sweep_axis_collapses_to_corner_only(self):
        test_cfg = {"conditions": {"corner": ["tt", "ss", "ff"], "temperature": ["-40", "25", "125"]}}
        combos = list(condition_matrix(test_cfg, _DEFAULTS, sweep_axis=("temperature", "temperature")))
        self.assertEqual(combos, [{"corner": "tt"}, {"corner": "ss"}, {"corner": "ff"}])

    def test_typical_key_is_reserved_metadata_not_a_sweep_axis(self):
        """conditions.typical is a dict ({"corner": "tt", "temperature":
        "25"}), not a list of values to sweep -- condition_matrix() must not
        mistake its 2 dict keys for 2 axis values (regression: it used to
        pass the whole conditions{} dict, "typical" included, through the
        "any remaining multi-value key becomes an outer axis" generalization,
        producing bogus extra runs like {"typical": "corner"})."""
        test_cfg = {"conditions": {
            "corner": ["tt", "ss"], "temperature": ["-40", "25"],
            "typical": {"corner": "tt", "temperature": "25"},
        }}
        combos = list(condition_matrix(test_cfg, _DEFAULTS, sweep_axis=None))
        self.assertEqual(len(combos), 4)
        self.assertTrue(all("typical" not in c for c in combos))

    def test_mc_seed_becomes_an_outer_axis_like_any_other_multi_valued_key(self):
        """A Monte Carlo (mismatch/stat) test's mc_seed list doesn't
        correspond to any real testbench token -- its only job is making
        this generalization spawn N separate ngspice processes, each of
        which draws its own independent random sample (confirmed
        empirically: ngspice reseeds its RNG per process). No dedicated
        code needed -- it's the same multi-valued-key generalization
        'ibias' already exercises."""
        test_cfg = {"conditions": {"corner": ["tt_mismatch"], "mc_seed": ["1", "2", "3"]}}
        combos = list(condition_matrix(test_cfg, _DEFAULTS, sweep_axis=None))
        self.assertEqual(len(combos), 3)
        self.assertEqual({c["mc_seed"] for c in combos}, {"1", "2", "3"})


class CornerSectionTests(unittest.TestCase):
    """MOS_CORNER_SECTION/RES_CORNER_SECTION are plain lookup tables from a
    friendly conditions.corner/res_corner value to the PDK's own .LIB
    section name (cornerMOShv.lib/cornerMOSlv.lib/cornerRES.lib) -- a typo
    here silently breaks netlisting (check_unresolved()/xschem would catch
    a missing section at simulate time, but not a WRONG section name that
    still resolves), so pin the exact strings confirmed against the real
    PDK files."""

    def test_mos_mismatch_and_stat_sections(self):
        self.assertEqual(MOS_CORNER_SECTION["tt_mismatch"], "mos_tt_mismatch")
        self.assertEqual(MOS_CORNER_SECTION["tt_stat"], "mos_tt_stat")

    def test_mos_deterministic_corners_unchanged(self):
        self.assertEqual(MOS_CORNER_SECTION["tt"], "mos_tt")
        self.assertEqual(MOS_CORNER_SECTION["ss"], "mos_ss")
        self.assertEqual(MOS_CORNER_SECTION["ff"], "mos_ff")

    def test_res_corner_sections(self):
        self.assertEqual(RES_CORNER_SECTION["typ"], "res_typ")
        self.assertEqual(RES_CORNER_SECTION["typ_mismatch"], "res_typ_mismatch")
        self.assertEqual(RES_CORNER_SECTION["typ_stat"], "res_typ_stat")
        self.assertEqual(RES_CORNER_SECTION["bcs"], "res_bcs")
        self.assertEqual(RES_CORNER_SECTION["wcs"], "res_wcs")


class FixedTbParamsTests(unittest.TestCase):
    def test_single_valued_key_referenced_in_schematic_is_pulled(self):
        test_cfg = {"conditions": {"frequency_start": ["10"]}}
        params = fixed_tb_params(test_cfg, "value='frequency_start'", sweep_axis=None)
        self.assertEqual(params, {"frequency_start": "10"})

    def test_multi_valued_key_is_skipped_not_an_error(self):
        """Used to sys.exit() here -- now condition_matrix() handles it as
        an outer axis instead, so fixed_tb_params() just leaves it out."""
        test_cfg = {"conditions": {"ibias": ["80n", "100n", "120n"]}}
        params = fixed_tb_params(test_cfg, "value='ibias'", sweep_axis=None)
        self.assertEqual(params, {})

    def test_key_not_referenced_in_schematic_is_skipped(self):
        test_cfg = {"conditions": {"unused_key": ["1"]}}
        params = fixed_tb_params(test_cfg, "value='something_else'", sweep_axis=None)
        self.assertEqual(params, {})

    def test_non_fixed_condition_keys_are_always_skipped(self):
        test_cfg = {"conditions": {"vdd": ["1.8"]}}
        params = fixed_tb_params(test_cfg, "value='vdd'", sweep_axis=None)
        self.assertEqual(params, {})


if __name__ == "__main__":
    unittest.main()
