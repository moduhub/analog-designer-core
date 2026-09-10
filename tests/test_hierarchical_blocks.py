"""Unit tests for hierarchical block composition ("top" instantiating
cmos_vref + output_amp as sub_blocks) -- pure Python, no docker needed.
Covers analog_designer/sim/run_sim.py's resolve_import_params()/
resolve_import_metrics()/materialize_sub_blocks()/
resolve_materialization_params() and analog_designer/sim/check_params.py's
sub_blocks-aware import_params/import_metrics validation, grounded against
the real reference project (ihp_mh_ip__cmos_vref) where useful -- confirms
"top"'s own amp_bias_width/pbias_length come out correctly derived from
cmos_vref's real m3_width/pbias_length defaults.
"""
import json
import random
import tempfile
import unittest
from pathlib import Path

from analog_designer.core import workspace
from analog_designer.results import data
from analog_designer.sim import check_params, gen_variations
from analog_designer.sim.run_sim import (
    MissingCrossBlockMetric, _read_jsonl, lookup_cross_block_metric, materialize_sub_blocks,
    resolve_formulas, resolve_import_metrics, resolve_import_params, resolve_materialization_params,
    variation_name,
)
from analog_designer.sim.spice_value import parse_spice_value

REFERENCE_ROOT = Path(__file__).resolve().parents[2] / "ihp_mh_ip__cmos_vref"


def _iter_jsonl(path):
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            yield json.loads(line)

_FAKE_TOP_BLOCK_CFG = {
    "parameters": {
        "amp_bias_factor": {"default": "3", "min": "1", "max": "20", "integer": True},
    },
    "derived_parameters": {
        "import_params": {
            "x1_m3_width": {"from": "X1", "source": "m3_width"},
            "x1_pbias_length": {"from": "X1", "source": "pbias_length"},
        },
        "formulas": {
            "amp_bias_width": {"expr": "x1_m3_width * amp_bias_factor", "unit_suffix": "u"},
            "pbias_length": {"expr": "x1_pbias_length", "unit_suffix": "u"},
        },
    },
}


def _require_reference():
    if not REFERENCE_ROOT.exists():
        raise unittest.SkipTest(f"reference project not checked out at {REFERENCE_ROOT}")


class ResolveImportParamsTests(unittest.TestCase):
    """resolve_import_params() is a PURE fetch (no math at all -- any
    scaling, like the old "factor" multiply, is an ordinary resolve_formulas()
    expr now, see test_combined_with_formula_reproduces_the_old_factor_math
    below)."""

    def test_fetches_source_value_unchanged(self):
        params = {"amp_bias_factor": "3"}
        sub_block_resolved = {"X1": {"m3_width": "50u", "pbias_length": "10u"}}
        resolved = resolve_import_params(_FAKE_TOP_BLOCK_CFG, params, sub_block_resolved)
        self.assertEqual(resolved["x1_m3_width"], "50u")
        self.assertEqual(resolved["x1_pbias_length"], "10u")

    def test_own_free_params_pass_through_unchanged(self):
        params = {"amp_bias_factor": "3"}
        sub_block_resolved = {"X1": {"m3_width": "50u", "pbias_length": "10u"}}
        resolved = resolve_import_params(_FAKE_TOP_BLOCK_CFG, params, sub_block_resolved)
        self.assertEqual(resolved["amp_bias_factor"], "3")

    def test_no_import_params_entries_is_a_no_op(self):
        resolved = resolve_import_params({"parameters": {}}, {"a": "1u"}, {})
        self.assertEqual(resolved, {"a": "1u"})

    def test_combined_with_formula_reproduces_the_old_factor_math(self):
        params = {"amp_bias_factor": "3"}
        sub_block_resolved = {"X1": {"m3_width": "50u", "pbias_length": "10u"}}
        resolved = resolve_import_params(_FAKE_TOP_BLOCK_CFG, params, sub_block_resolved)
        resolved = resolve_formulas(_FAKE_TOP_BLOCK_CFG, resolved)
        self.assertEqual(resolved["amp_bias_width"], "150u")
        self.assertEqual(resolved["pbias_length"], "10u")


class ResolveImportMetricsTests(unittest.TestCase):
    """resolve_import_metrics()/lookup_cross_block_metric() -- like
    resolve_import_params(), a PURE fetch (no target/invert ratio math --
    that's an ordinary resolve_formulas() expr now, see
    test_combined_with_formula_reproduces_the_old_scale_to_target_math
    below), against a synthetic project (no docker, no real project's sim/
    state) so it doesn't depend on any test having actually been run
    against the reference project."""

    _BLOCK_CFG = {
        "derived_parameters": {
            "import_params": {
                "x1_m3_width": {"from": "X1", "source": "m3_width"},
            },
            "import_metrics": {
                "x1_ref_current_typical": {
                    "from": "X1", "test": "reference_current", "metric": "Core reference current (M3 branch)",
                },
            },
            "formulas": {
                "amp_bias_width": {"expr": "x1_m3_width * 100 / x1_ref_current_typical", "unit_suffix": "u"},
            },
        }
    }

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self._saved_root = workspace.PROJECT_ROOT
        self.addCleanup(lambda: setattr(workspace, "PROJECT_ROOT", self._saved_root))
        workspace.PROJECT_ROOT = Path(self._tmp.name)
        (workspace.PROJECT_ROOT / "sim").mkdir()

    def _write_results(self, rows):
        path = workspace.PROJECT_ROOT / "sim" / "results.jsonl"
        with path.open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row) + "\n")

    def test_fetches_the_measured_value_unchanged(self):
        self._write_results([
            {"variation": "cmos_vref-x", "test": "reference_current", "metric": "Core reference current (M3 branch)",
             "typical": 20, "min": 20, "max": 20},
        ])
        resolved = resolve_import_metrics(self._BLOCK_CFG, {}, {"X1": "cmos_vref-x"})
        self.assertEqual(resolved["x1_ref_current_typical"], "20")

    def test_combined_with_formula_reproduces_the_old_scale_to_target_math(self):
        self._write_results([
            {"variation": "cmos_vref-x", "test": "reference_current", "metric": "Core reference current (M3 branch)",
             "typical": 20, "min": 20, "max": 20},
        ])
        sub_block_resolved = {"X1": {"m3_width": "50u"}}
        resolved = resolve_import_params(self._BLOCK_CFG, {}, sub_block_resolved)
        resolved = resolve_import_metrics(self._BLOCK_CFG, resolved, {"X1": "cmos_vref-x"})
        resolved = resolve_formulas(self._BLOCK_CFG, resolved)
        # target=100, measured=20 -> 5x scale-up: 50u * (100/20) = 250u
        self.assertAlmostEqual(parse_spice_value(resolved["amp_bias_width"]), parse_spice_value("250u"))

    def test_can_scale_down_unlike_the_old_integer_factor(self):
        self._write_results([
            {"variation": "cmos_vref-x", "test": "reference_current", "metric": "Core reference current (M3 branch)",
             "typical": 400, "min": 400, "max": 400},
        ])
        sub_block_resolved = {"X1": {"m3_width": "50u"}}
        resolved = resolve_import_params(self._BLOCK_CFG, {}, sub_block_resolved)
        resolved = resolve_import_metrics(self._BLOCK_CFG, resolved, {"X1": "cmos_vref-x"})
        resolved = resolve_formulas(self._BLOCK_CFG, resolved)
        # target=100, measured=400 -> 0.25x: 50u * (100/400) = 12.5u, impossible
        # with the old integer-factor (min 1) mechanism this replaced.
        self.assertAlmostEqual(parse_spice_value(resolved["amp_bias_width"]), parse_spice_value("12.5u"))

    def test_passes_through_params_already_resolved_upstream(self):
        self._write_results([
            {"variation": "cmos_vref-x", "test": "reference_current", "metric": "Core reference current (M3 branch)",
             "typical": 20, "min": 20, "max": 20},
        ])
        resolved = resolve_import_metrics(
            self._BLOCK_CFG, {"pbias_length": "10u"}, {"X1": "cmos_vref-x"},
        )
        self.assertEqual(resolved["pbias_length"], "10u")

    def test_missing_result_raises_missing_cross_block_metric(self):
        self._write_results([])
        with self.assertRaises(MissingCrossBlockMetric):
            resolve_import_metrics(self._BLOCK_CFG, {}, {"X1": "cmos_vref-x"})

    def test_defaults_sentinel_raises_missing_cross_block_metric(self):
        with self.assertRaises(MissingCrossBlockMetric):
            resolve_import_metrics(self._BLOCK_CFG, {}, {"X1": "defaults"})

    def test_non_positive_measured_value_raises(self):
        self._write_results([
            {"variation": "cmos_vref-x", "test": "reference_current", "metric": "Core reference current (M3 branch)",
             "typical": 0, "min": 0, "max": 0},
        ])
        with self.assertRaises(MissingCrossBlockMetric):
            resolve_import_metrics(self._BLOCK_CFG, {}, {"X1": "cmos_vref-x"})

    def test_lookup_returns_the_latest_matching_row(self):
        self._write_results([
            {"variation": "cmos_vref-x", "test": "reference_current", "metric": "Core reference current (M3 branch)",
             "typical": 20, "min": 20, "max": 20},
            {"variation": "cmos_vref-x", "test": "reference_current", "metric": "Core reference current (M3 branch)",
             "typical": 30, "min": 30, "max": 30},
        ])
        self.assertEqual(
            lookup_cross_block_metric("cmos_vref-x", "reference_current", "Core reference current (M3 branch)"),
            30,
        )

    def test_lookup_can_select_min_or_max_stat_instead_of_typical(self):
        self._write_results([
            {"variation": "cmos_vref-x", "test": "reference_current", "metric": "Core reference current (M3 branch)",
             "typical": 20, "min": 15, "max": 25},
        ])
        self.assertEqual(
            lookup_cross_block_metric("cmos_vref-x", "reference_current", "Core reference current (M3 branch)", stat="min"),
            15,
        )
        self.assertEqual(
            lookup_cross_block_metric("cmos_vref-x", "reference_current", "Core reference current (M3 branch)", stat="max"),
            25,
        )


class MaterializeSubBlocksTests(unittest.TestCase):
    def test_reference_project_resolves_and_materializes_both_sub_blocks(self):
        """Reproduces exactly what "top"'s own sub_blocks declares in the
        real config.json: X1 -> cmos_vref/default (defaults), x2 ->
        output_amp/default (defaults)."""
        _require_reference()
        workspace.open_folder(str(REFERENCE_ROOT), block="top", topology="default")
        config = workspace.CONFIG
        block_cfg = config["blocks"]["top"]["topologies"]["default"]
        sub_blocks = block_cfg["sub_blocks"]
        params = {n: pdef["default"] for n, pdef in block_cfg["parameters"].items()}

        resolved_by_instance = materialize_sub_blocks(config, sub_blocks, params)

        self.assertIn("X1", resolved_by_instance)
        self.assertIn("x2", resolved_by_instance)
        # cmos_vref's own real defaults (m3_width=50u, pbias_length=10u,
        # m6_width derived to 100u -- see tests/test_derived_params.py)
        self.assertEqual(resolved_by_instance["X1"]["m3_width"], "50u")
        self.assertEqual(resolved_by_instance["X1"]["pbias_length"], "10u")
        self.assertEqual(resolved_by_instance["X1"]["m6_width"], "100u")

        # materialize_sub_blocks() actually wrote sch/cmos_vref.sch/sch/output_amp.sch
        # with no unresolved 'name' placeholders left.
        import re
        for block_name in ("cmos_vref", "output_amp"):
            text = (workspace.PROJECT_ROOT / "sch" / f"{block_name}.sch").read_text(encoding="utf-8")
            self.assertEqual(re.findall(r"'([A-Za-z_][A-Za-z0-9_]*)'", text), [])


class ResolveMaterializationParamsTopTests(unittest.TestCase):
    def test_top_amp_bias_width_scales_to_hit_output_amps_target_current(self):
        """End-to-end against the real project: amp_bias_width is no longer
        a free-factor multiple of cmos_vref's m3_width (that could only
        scale UP, never down, relative to whatever current cmos_vref's own
        core happened to produce) -- it's calculated via an import_metrics
        entry (x1_ref_current_typical) plus a formula so M1's mirrored
        current lands on output_amp's 100nA design point, using whichever
        registered cmos_vref/default variation X1_variation selects' own
        MEASURED reference_current result (not just its m3_width parameter
        value). pbias_length stays a plain passthrough (import_params +
        a pure-passthrough formula), reused verbatim, unaffected by this
        change."""
        _require_reference()
        workspace.open_folder(str(REFERENCE_ROOT), block="top", topology="default")
        config = workspace.CONFIG
        variations = [
            r for r in _iter_jsonl(REFERENCE_ROOT / "sim" / "variations.jsonl")
            if r["block"] == "cmos_vref" and r["topology"] == "default"
        ]
        results = list(_iter_jsonl(REFERENCE_ROOT / "sim" / "results.jsonl"))
        chosen = next(
            (r for r in variations if any(
                row["variation"] == r["name"] and row["test"] == "reference_current"
                for row in results
            )),
            None,
        )
        if chosen is None:
            raise unittest.SkipTest(
                "no registered cmos_vref/default variation with a reference_current "
                "result -- run that test for one before this test can exercise "
                "import_metrics + formula end-to-end"
            )
        # last match, exactly like lookup_cross_block_metric() itself --
        # a variation can have been re-simulated more than once, appending
        # a newer row rather than replacing the old one.
        measured_na = [
            row["typical"] for row in results
            if row["variation"] == chosen["name"] and row["test"] == "reference_current"
        ][-1]

        block_cfg = config["blocks"]["top"]["topologies"]["default"]
        params = {name: pdef["default"] for name, pdef in block_cfg["parameters"].items()}
        params["X1_variation"] = chosen["name"]

        resolved = resolve_materialization_params(block_cfg, params)

        expected_width = parse_spice_value(chosen["parameters"]["m3_width"]) * (100 / measured_na)
        actual_width = parse_spice_value(resolved["amp_bias_width"])
        self.assertAlmostEqual(actual_width, expected_width, delta=expected_width * 1e-3)
        self.assertEqual(resolved["pbias_length"], chosen["parameters"]["pbias_length"])


class CheckParamsCrossBlockTests(unittest.TestCase):
    _SCH_TEXT = "w='amp_bias_width' l='pbias_length'"
    _SUB_BLOCKS = {"X1": {"block": "cmos_vref", "topology": "default", "variation": None}}

    def test_import_and_formula_derived_names_are_not_flagged_as_missing(self):
        orphans, missing, errors = check_params.check(
            _FAKE_TOP_BLOCK_CFG["parameters"], self._SCH_TEXT,
            _FAKE_TOP_BLOCK_CFG["derived_parameters"], self._SUB_BLOCKS,
        )
        self.assertEqual(missing, [])
        self.assertEqual(errors, [])

    def test_imported_names_are_not_flagged_as_orphans(self):
        orphans, _, _ = check_params.check(
            _FAKE_TOP_BLOCK_CFG["parameters"], self._SCH_TEXT,
            _FAKE_TOP_BLOCK_CFG["derived_parameters"], self._SUB_BLOCKS,
        )
        self.assertEqual(orphans, [])

    def test_typo_in_import_params_from_is_caught(self):
        bad_cfg = {"import_params": {
            "x1_m3_width": {"from": "no_such_instance", "source": "m3_width"},
        }}
        _, _, errors = check_params.check(
            _FAKE_TOP_BLOCK_CFG["parameters"], self._SCH_TEXT, bad_cfg, self._SUB_BLOCKS,
        )
        self.assertTrue(any("no_such_instance" in msg for _, msg in errors))

    def test_missing_source_is_caught(self):
        bad_cfg = {"import_params": {"x1_m3_width": {"from": "X1"}}}
        _, _, errors = check_params.check(
            _FAKE_TOP_BLOCK_CFG["parameters"], self._SCH_TEXT, bad_cfg, self._SUB_BLOCKS,
        )
        self.assertTrue(any("source" in msg for _, msg in errors))

    def test_import_metrics_is_accepted_with_required_keys(self):
        cfg = {
            "import_metrics": {
                "x1_ref_current_typical": {
                    "from": "X1", "test": "reference_current", "metric": "Core reference current (M3 branch)",
                },
            },
            "formulas": {
                "amp_bias_width": {"expr": "x1_ref_current_typical", "unit_suffix": "u"},
            },
        }
        _, _, errors = check_params.check({}, self._SCH_TEXT, cfg, self._SUB_BLOCKS)
        self.assertEqual(errors, [])

    def test_typo_in_import_metrics_from_is_caught(self):
        cfg = {"import_metrics": {
            "x1_ref_current_typical": {
                "from": "no_such_instance", "test": "reference_current", "metric": "m",
            },
        }}
        _, _, errors = check_params.check({}, self._SCH_TEXT, cfg, self._SUB_BLOCKS)
        self.assertTrue(any("no_such_instance" in msg for _, msg in errors))

    def test_import_metrics_missing_keys_is_caught(self):
        cfg = {"import_metrics": {
            "x1_ref_current_typical": {"from": "X1", "test": "reference_current"},
        }}
        _, _, errors = check_params.check({}, self._SCH_TEXT, cfg, self._SUB_BLOCKS)
        self.assertTrue(any("metric" in msg for _, msg in errors))

    def test_reference_project_top_has_zero_problems(self):
        _require_reference()
        doc = json.loads((REFERENCE_ROOT / "config.json").read_text(encoding="utf-8"))
        block_cfg = doc["blocks"]["top"]["topologies"]["default"]
        params_doc = json.loads((REFERENCE_ROOT / block_cfg["parameters_file"]).read_text(encoding="utf-8"))
        sch_text = (REFERENCE_ROOT / "sch" / block_cfg["schematic"]).read_text(encoding="utf-8")
        orphans, missing, errors = check_params.check(
            params_doc["parameters"], sch_text, params_doc["derived_parameters"], block_cfg["sub_blocks"], doc,
        )
        self.assertEqual(orphans, [])
        self.assertEqual(missing, [])
        self.assertEqual(errors, [])


class CheckParamsStructuralSubBlockTests(unittest.TestCase):
    """structural_sub_block_errors()/check()'s "sub_block" tag support --
    NOT a sub_blocks/block_ref instance (no separate block/topology, no
    {instance}_variation, no registered variations of its own): a
    "sub_block"-tagged free parameter is still an ordinary literal '<name>'
    token in the parent's own schematic (on a structural_sub_blocks
    instance's own call-site attribute line, e.g. half_qvco_cell's X1/X2 in
    sch/vco/vco_quadrature_lc.sch), so -- unlike block_ref_names() -- it
    stays subject to the normal used/orphan check, only the tag ITSELF gets
    cross-checked against config.json's own structural_sub_blocks."""

    _STRUCTURAL_SUB_BLOCKS = {
        "X1": {"schematic": "vco/half_qvco_cell.sch"},
        "X2": {"schematic": "vco/half_qvco_cell.sch"},
    }
    _PARAM_DEFS = {
        "m1m2_width": {
            "default": "10u", "min": "0.22u", "max": "100u",
            "sub_block": ["X1", "X2"],
        },
    }
    _SCH_TEXT = "m1m2_width='m1m2_width'"

    def test_valid_tag_produces_no_errors(self):
        _, _, errors = check_params.check(
            self._PARAM_DEFS, self._SCH_TEXT, structural_sub_blocks=self._STRUCTURAL_SUB_BLOCKS,
        )
        self.assertEqual(errors, [])

    def test_tagged_param_used_as_a_token_is_not_flagged_as_orphan(self):
        orphans, missing, _ = check_params.check(
            self._PARAM_DEFS, self._SCH_TEXT, structural_sub_blocks=self._STRUCTURAL_SUB_BLOCKS,
        )
        self.assertEqual(orphans, [])
        self.assertEqual(missing, [])

    def test_unknown_instance_in_tag_is_caught(self):
        bad_defs = {
            "m1m2_width": {
                "default": "10u", "min": "0.22u", "max": "100u",
                "sub_block": ["X9"],
            },
        }
        _, _, errors = check_params.check(
            bad_defs, "m1m2_width='m1m2_width'", structural_sub_blocks=self._STRUCTURAL_SUB_BLOCKS,
        )
        self.assertTrue(any("X9" in msg for _, msg in errors))

    def test_tag_with_no_structural_sub_blocks_declared_is_caught(self):
        _, _, errors = check_params.check(self._PARAM_DEFS, self._SCH_TEXT, structural_sub_blocks=None)
        self.assertTrue(any("structural_sub_blocks" in msg for _, msg in errors))

    def test_reference_project_vco_has_zero_problems(self):
        project_root = Path(__file__).resolve().parents[2] / "gf180mcu_mh_ip__nfrac_pll"
        if not project_root.exists():
            raise unittest.SkipTest(f"gf180mcu_mh_ip__nfrac_pll not checked out at {project_root}")
        doc = json.loads((project_root / "config.json").read_text(encoding="utf-8"))
        block_cfg = doc["blocks"]["vco"]["topologies"]["quadrature_lc"]
        params_doc = json.loads((project_root / block_cfg["parameters_file"]).read_text(encoding="utf-8"))
        sch_text = (project_root / "sch" / block_cfg["schematic"]).read_text(encoding="utf-8")
        orphans, missing, errors = check_params.check(
            params_doc["parameters"], sch_text, params_doc.get("derived_parameters"),
            block_cfg.get("sub_blocks"), doc, block_cfg.get("structural_sub_blocks"),
        )
        self.assertEqual(orphans, [])
        self.assertEqual(missing, [])
        self.assertEqual(errors, [])


class DynamicSubBlockChoiceTests(unittest.TestCase):
    """materialize_sub_blocks() reading the sub-block CHOICE out of the
    parent's own params[f"{instance}_variation"] (a "block_ref" parameter)
    instead of a static config.json field."""

    def test_missing_or_defaults_sentinel_uses_block_defaults(self):
        _require_reference()
        workspace.open_folder(str(REFERENCE_ROOT), block="top", topology="default")
        config = workspace.CONFIG
        sub_blocks = config["blocks"]["top"]["topologies"]["default"]["sub_blocks"]

        for params in ({}, {"X1_variation": "defaults", "x2_variation": "defaults"}):
            resolved = materialize_sub_blocks(config, sub_blocks, params)
            self.assertEqual(resolved["X1"]["m3_width"], "50u")

    def test_real_variation_name_is_used_over_defaults(self):
        """output_amp variations aren't affected by cmos_vref's own
        width_base+factor migration, so a real registered one can be
        re-materialized here without hitting StaleParameterSchema -- picks
        whichever real output_amp/default variation is registered in the
        reference project and confirms materialize_sub_blocks() actually
        used ITS OWN m1_width, not output_amp/default's config.json
        default, proving the params-driven choice is really being read."""
        _require_reference()
        workspace.open_folder(str(REFERENCE_ROOT), block="top", topology="default")
        config = workspace.CONFIG
        sub_blocks = config["blocks"]["top"]["topologies"]["default"]["sub_blocks"]
        variations = [
            r for r in _iter_jsonl(REFERENCE_ROOT / "sim" / "variations.jsonl")
            if r["block"] == "output_amp" and r["topology"] == "default"
        ]
        if not variations:
            raise unittest.SkipTest("no registered output_amp/default variations in the reference project")
        chosen = variations[0]

        resolved = materialize_sub_blocks(config, sub_blocks, {"x2_variation": chosen["name"]})
        self.assertEqual(resolved["x2"]["m1_width"], chosen["parameters"]["m1_width"])


class MatchingVariationsTests(unittest.TestCase):
    def test_unfiltered_returns_every_registered_variation(self):
        _require_reference()
        workspace.open_folder(str(REFERENCE_ROOT), block="top", topology="default")
        names = data.matching_variations("cmos_vref", "default")
        all_names = {
            r["name"] for r in _iter_jsonl(REFERENCE_ROOT / "sim" / "variations.jsonl")
            if r["block"] == "cmos_vref" and r["topology"] == "default"
        }
        self.assertEqual(set(names), all_names)

    def test_no_profiles_declared_degrades_to_unfiltered(self):
        """output_amp has no "profiles" in config.json at all -- a spec
        filter has nothing to match against, so this must NOT return an
        empty list."""
        _require_reference()
        workspace.open_folder(str(REFERENCE_ROOT), block="top", topology="default")
        unfiltered = data.matching_variations("output_amp", "default")
        spec_filtered = data.matching_variations("output_amp", "default", "low_power")
        self.assertEqual(set(spec_filtered), set(unfiltered))
        self.assertTrue(spec_filtered)

    def test_matching_a_real_profile_is_a_real_subset(self):
        _require_reference()
        workspace.open_folder(str(REFERENCE_ROOT), block="top", topology="default")
        if not any(_iter_jsonl(REFERENCE_ROOT / "sim" / "results.jsonl")):
            raise unittest.SkipTest(
                "no simulated results in the reference project -- run cmos_vref's "
                "tests for at least one registered variation before this test can "
                "exercise real profile classification"
            )
        unfiltered = data.matching_variations("cmos_vref", "default")
        low_power = data.matching_variations("cmos_vref", "default", "low_power")
        self.assertTrue(set(low_power) <= set(unfiltered))
        self.assertLess(len(low_power), len(unfiltered))


class GenerateHierarchicalParamSetTests(unittest.TestCase):
    """generate_hierarchical_param_set() -- pure sampling + JSONL
    registration, no docker, no .sch files touched at all. Runs against a
    throwaway synthetic project in a temp dir (NOT the real reference
    project's own sim/variations.jsonl -- this function's whole job is to
    WRITE new variation registrations, and the real project's history isn't
    something a test run should be mutating)."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        root = Path(self._tmp.name)
        (root / "sim").mkdir()
        (root / "config.json").write_text("{}", encoding="utf-8")

        self._saved = (workspace.PROJECT_ROOT, workspace.CONFIG, workspace.BLOCK, workspace.TOPOLOGY)
        self.addCleanup(self._restore_workspace)
        workspace.PROJECT_ROOT = root
        workspace.BLOCK, workspace.TOPOLOGY = "top", "default"

        self.config = {
            "blocks": {"sub": {"topologies": {"d": {"parameters": {
                "w": {"default": "1u", "min": "0.5u", "max": "2u"},
            }}}}},
            "tests": {"sub": {"area": {}}},
        }
        self.block_cfg = {"parameters": {
            "A_variation": {"type": "block_ref", "block": "sub", "topology": "d", "default": "defaults"},
        }}

    def _restore_workspace(self):
        workspace.PROJECT_ROOT, workspace.CONFIG, workspace.BLOCK, workspace.TOPOLOGY = self._saved

    def test_registers_one_fresh_sub_block_variation_per_block_ref(self):
        rng = random.Random(1)
        top_params, sub_jobs = gen_variations.generate_hierarchical_param_set(
            self.config, self.block_cfg, "top", "default", rng, None, "random",
        )
        self.assertEqual(len(sub_jobs), 1)
        job = sub_jobs[0]
        self.assertEqual((job["block"], job["topology"]), ("sub", "d"))
        expected_name = variation_name("sub", "d", job["params"])
        self.assertEqual(top_params["A_variation"], expected_name)

        registered = list(_read_jsonl(workspace.PROJECT_ROOT / "sim" / "variations.jsonl"))
        self.assertEqual(len(registered), 1)
        self.assertEqual(registered[0]["name"], expected_name)
        self.assertEqual(registered[0]["block"], "sub")
        self.assertEqual(registered[0]["origin"]["kind"], "generate_hierarchical")
        self.assertEqual(registered[0]["origin"]["parent_kind"], "random")

    def test_two_calls_register_two_independent_sub_block_variations(self):
        """"os sub blocos poderiam ser gerados na hora" -- no pooling/reuse
        across iterations, each call gets its own fresh pick."""
        rng = random.Random(1)
        top_params_a, _ = gen_variations.generate_hierarchical_param_set(
            self.config, self.block_cfg, "top", "default", rng, None, "random",
        )
        top_params_b, _ = gen_variations.generate_hierarchical_param_set(
            self.config, self.block_cfg, "top", "default", rng, None, "random",
        )
        registered = list(_read_jsonl(workspace.PROJECT_ROOT / "sim" / "variations.jsonl"))
        self.assertEqual(len(registered), 2)
        self.assertEqual(len({r["name"] for r in registered}), 2)


if __name__ == "__main__":
    unittest.main()
