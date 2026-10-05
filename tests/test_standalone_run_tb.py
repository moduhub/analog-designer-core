"""Tests for analog_designer/standalone/run_tb.py -- the stdlib-only,
docker-free testbench runner that gets vendored into project repos -- and
for its export command.

Two kinds of checks, both on a small self-contained fixture project:
  * parity: the standalone copy of each pure pipeline function (parameter
    resolution, condition grid, testbench placeholders, definition hash,
    PDK corner tables, Xyce netlist fixups) gives exactly what run_sim.py
    gives for the same input, so a vendored runner can't silently drift
    from the core;
  * end to end: a full run against fake xschem/ngspice executables (tiny
    Python scripts), covering --dry leaving the project untouched and the
    default mode writing sim/ results the core itself sees as fresh.
"""
import contextlib
import io
import json
import os
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path
from unittest import mock

from analog_designer.core import workspace
from analog_designer.sim import run_sim
from analog_designer.standalone import export
from analog_designer.standalone import run_tb

_CONFIG = {
    "container": {"image": "eda-env-designer:ihp-fake"},
    "defaults": {"vdd": "3.3", "ibias": "100n", "corner": "tt", "temperature": "25"},
    "blocks": {
        "core": {
            "topologies": {"a": {"schematic": "core/core_a.sch", "parameters_file": "sch/core/core_a.params.json"}},
        },
        "top": {
            "topologies": {
                "default": {
                    "schematic": "top/top_default.sch",
                    "parameters_file": "sch/top/top_default.params.json",
                    "sub_blocks": {"X1": {"block": "core", "topology": "a"}},
                },
            },
        },
    },
    "tests": {
        "core": {
            "area": {"simulator": "netlist", "testbench": "tb/core/tb_core.sch", "parser": "tb/core/tb_area.py",
                     "outputs": [{"name": "Area", "unit": "um2"}]},
            "level": {
                "simulator": "ngspice", "testbench": "tb/core/tb_core.sch", "parser": "tb/core/tb_level.py",
                "conditions": {"corner": ["tt", "ss"], "temperature": ["-40", "25"], "ibias": ["80n", "120n"],
                               "vdd": ["3.3"], "typical": {"corner": "tt", "temperature": "25", "ibias": "80n"}},
                "outputs": [{"name": "Level", "unit": "V"}],
            },
            "tsweep": {
                "simulator": "ngspice", "testbench": "tb/core/tb_tsweep.sch", "parser": "tb/core/tb_level.py",
                "conditions": {"corner": ["tt", "ff"], "temperature": ["-40", "0", "125"], "vdd": ["3.3", "1.8"]},
                "outputs": [{"name": "Level", "unit": "V"}],
            },
        },
        "top": {
            "level": {"simulator": "ngspice", "testbench": "tb/top/tb_top.sch", "parser": "tb/core/tb_level.py",
                      "conditions": {"corner": ["tt"]}, "outputs": [{"name": "Level", "unit": "V"}]},
        },
    },
}

_CORE_PARAMS = {
    "parameters": {
        "w_base": {"default": "1u"},
        "w_factor": {"default": "3"},
        "l_total": {"default": "20u"},
    },
    "derived_parameters": {
        "width_groups": [{"id": "g", "base": "w_base", "members": {"w_m1": {"factor": "w_factor"}}}],
        "constants": {"l_max": {"value": "10u"}},
        "formulas": {
            "l_ng": {"expr": "ceil(l_total / l_max)"},
            "l_seg": {"expr": "l_total / l_ng", "unit_suffix": "u"},
        },
    },
}

_TOP_PARAMS = {
    "parameters": {
        "X1_variation": {"type": "block_ref", "default": "defaults"},
        "bias_factor": {"default": "2"},
    },
    "derived_parameters": {
        "import_params": {"x1_w": {"from": "X1", "source": "w_m1"}},
        "import_metrics": {"x1_level": {"from": "X1", "test": "level", "metric": "Level"}},
        "formulas": {
            "bias_w": {"expr": "x1_w * bias_factor", "unit_suffix": "u"},
            "r_top": {"expr": "x1_level * 1e6"},
        },
    },
}

_TB_CORE = textwrap.dedent("""\
    v {xschem version=3.4.4 file_version=1.2}
    C {sch/core.sym} 0 0 0 0 {name=X1}
    .lib 'models_dir'/cornerMOShv.lib 'mos_corner'
    .temp 'temperature'
    vdd vdd 0 'Vavdd'
    ib ib 0 'ibias'
    .control
    wrdata 'simpath'/'filename'_'N'.data v(out)
    write 'simpath'/'filename'_'N'.raw
    .endc
""")

_TB_TSWEEP = _TB_CORE.replace(".temp 'temperature'", ".dc temp 'temp_min' 'temp_max' 5")

_PARSER_LEVEL = textwrap.dedent("""\
    from parser_common import read_value


    def extract(data_path):
        return {"value": read_value(data_path)}


    def evaluate(runs, outputs, typical, plot_base=None):
        values = [r["value"] for r in runs]
        typ = next((r["value"] for r in runs
                    if all(r["conditions"].get(k) == v for k, v in typical.items() if k in r["conditions"])),
                   values[0])
        return [{"name": outputs[0]["name"], "unit": outputs[0]["unit"],
                 "typical": typ, "min": min(values), "max": max(values)}]
""")

_PARSER_AREA = textwrap.dedent("""\
    def extract(data_path):
        return {"n": open(data_path).read().count("MOSFET")}


    def evaluate(runs, outputs, typical, plot_base=None):
        n = runs[0]["n"]
        return [{"name": "Area", "unit": "um2", "typical": n, "min": n, "max": n}]
""")

# Fake xschem: the "netlist" is the testbench text plus every materialized
# block schematic found under ./sch (cwd = the rcfile's directory), so the
# parsers and assertions can see what was substituted where.
_FAKE_XSCHEM = textwrap.dedent("""\
    #!{python}
    import sys
    from pathlib import Path
    args = sys.argv[1:]
    out = Path(args[args.index("-o") + 1])
    sch = Path(args[-1])
    text = sch.read_text()
    for block in sorted(Path("sch").glob("*.sch")):
        text += "\\n.subckt " + block.stem + "\\n" + block.read_text() + "\\n.ends\\n"
    (out / (sch.stem + ".spice")).write_text(text)
""")

# Fake ngspice: writes every wrdata/write target; the .data value encodes
# the conditions (temperature + ibias + corner) so evaluate() has something
# to reduce.
_FAKE_NGSPICE = textwrap.dedent("""\
    #!{python}
    import re, sys
    from pathlib import Path
    text = Path(sys.argv[-1]).read_text()
    temp = float((re.search(r"^\\.temp (\\S+)", text, re.M) or [0, "0"])[1])
    ib = re.search(r"^ib ib 0 (\\S+)", text, re.M)[1]
    corner = re.search(r"cornerMOShv.lib (\\S+)", text)[1]
    value = 1.0 + temp / 1000 + (0.1 if ib == "120n" else 0) + (0.01 if corner == "mos_ss" else 0)
    for cmd, path in re.findall(r"^(wrdata|write) (\\S+)", text, re.M):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(f"0 {{value}}\\n" if cmd == "wrdata" else "raw")
    print("fake ngspice done")
""")


def _write(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _make_project(root):
    _write(root / "config.json", json.dumps(_CONFIG))
    _write(root / "sch" / "core.sym", "symbol core")
    _write(root / "sch" / "top.sym", "symbol top")
    _write(root / "sch" / "core" / "core_a.sch", "MOSFET w='w_m1' l='l_seg' ng='l_ng' base='w_base'\n")
    _write(root / "sch" / "core" / "core_a.params.json", json.dumps(_CORE_PARAMS))
    _write(root / "sch" / "top" / "top_default.sch", "MOSFET w='bias_w'\nMOSFET r='r_top'\nC {sch/core.sym}\n")
    _write(root / "sch" / "top" / "top_default.params.json", json.dumps(_TOP_PARAMS))
    _write(root / "tb" / "core" / "tb_core.sch", _TB_CORE)
    _write(root / "tb" / "core" / "tb_tsweep.sch", _TB_TSWEEP)
    _write(root / "tb" / "top" / "tb_top.sch", _TB_CORE.replace("sch/core.sym", "sch/top.sym"))
    _write(root / "tb" / "core" / "tb_level.py", _PARSER_LEVEL)
    _write(root / "tb" / "core" / "tb_area.py", _PARSER_AREA)
    _write(root / "tb" / "_shared" / "parser_common.py",
           "def read_value(path):\n    return float(open(path).read().split()[1])\n")


def _make_tools(base):
    pdk = base / "pdk" / "ihp-fake" / "libs.tech"
    _write(pdk / "xschem" / "xschemrc", "# fake xschemrc\n")
    (pdk / "ngspice" / "models").mkdir(parents=True)
    for name in ("psp103", "psp103_nqs", "r3_cmc", "mosvar"):
        _write(pdk / "ngspice" / "osdi" / f"{name}.osdi", "")
    tools = {}
    for name, src in (("xschem", _FAKE_XSCHEM), ("ngspice", _FAKE_NGSPICE)):
        path = base / "bin" / name
        _write(path, src.format(python=sys.executable))
        path.chmod(0o755)
        tools[name] = str(path)
    return ["--pdk-root", str(base / "pdk"), "--pdk", "ihp-fake",
            "--xschem", tools["xschem"], "--ngspice", tools["ngspice"], "--jobs", "4"]


def _snapshot(root):
    return {str(p.relative_to(root)): p.stat().st_mtime_ns for p in root.rglob("*")}


class _FixtureCase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        base = Path(self._tmp.name)
        self.root = base / "proj"
        _make_project(self.root)
        self.tool_args = _make_tools(base)
        self.base = base
        patcher = mock.patch.object(workspace, "_LAST_FOLDER_FILE", base / "last_folder.txt")
        patcher.start()
        self.addCleanup(patcher.stop)
        self.addCleanup(self._tmp.cleanup)
        saved_path = list(sys.path)
        self.addCleanup(lambda: sys.path.__setitem__(slice(None), saved_path))

    def run_tb(self, *args):
        out = io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(out):
            rc = run_tb.main(["--project-root", str(self.root), *self.tool_args, *args])
        return rc, out.getvalue()

    def open_core(self, block, topology):
        workspace.open_folder(str(self.root), block=block, topology=topology)
        return workspace.CONFIG["blocks"][block]["topologies"][topology]


class ParityTests(_FixtureCase):
    def test_condition_grid_and_testbench_params_match_run_sim(self):
        defaults = _CONFIG["defaults"]
        cases = [
            ({"conditions": {}}, ""),
            (_CONFIG["tests"]["core"]["level"], _TB_CORE),
            (_CONFIG["tests"]["core"]["tsweep"], _TB_TSWEEP),
            ({"conditions": {"vdd": ["2.97", "3.63"], "corner": ["tt", "ss", "ff"]}}, "'vdd_min' 'vdd_max'"),
            ({"conditions": {"corner": ["tt_mismatch"], "mc_seed": ["1", "2", "3"], "res_corner": ["typ_mismatch"]}},
             "'mc_seed' 'res_corner'"),
            ({"conditions": {"frequency_start": ["10"], "typical": {"corner": "tt"}}}, "'frequency_start'"),
        ]
        for test_cfg, tb_text in cases:
            with self.subTest(test_cfg=test_cfg):
                axis = run_sim.internal_sweep_axis(test_cfg, tb_text)
                self.assertEqual(run_tb.internal_sweep_axis(test_cfg, tb_text), axis)
                self.assertEqual(run_tb.fixed_tb_params(test_cfg, tb_text, axis),
                                 run_sim.fixed_tb_params(test_cfg, tb_text, axis))
                core_grid = list(run_sim.condition_matrix(test_cfg, defaults, axis))
                self.assertEqual(list(run_tb.condition_matrix(test_cfg, defaults, axis)), core_grid)
                self.assertEqual([run_tb.condition_label(c) for c in core_grid],
                                 [run_sim.condition_label(c) for c in core_grid])
                self.assertEqual(run_tb.typical_conditions(test_cfg, defaults),
                                 run_sim.typical_conditions(test_cfg, defaults))

    def test_corner_tables_match_run_sim(self):
        self.assertEqual(run_tb.MOS_CORNER_SECTION, run_sim.MOS_CORNER_SECTION)
        self.assertEqual(run_tb.MOS_CORNER_SECTION_GF180MCU, run_sim.MOS_CORNER_SECTION_GF180MCU)
        self.assertEqual(run_tb.RES_CORNER_SECTION, run_sim.RES_CORNER_SECTION)
        self.assertEqual(run_tb._RES_SECTION_CMOS5L, run_sim._RES_SECTION_CMOS5L)
        self.assertEqual(run_tb.INTERNAL_SWEEP_AXES, run_sim.INTERNAL_SWEEP_AXES)
        self.assertEqual(run_tb._NON_FIXED_CONDITION_KEYS, run_sim._NON_FIXED_CONDITION_KEYS)

    def test_materialization_and_definition_hash_match_run_sim(self):
        block_cfg = self.open_core("core", "a")
        project = run_tb.Project(self.root, self.root / "sim", persist=True)
        sa_cfg = project.topology_cfg("core", "a")
        params = run_tb.default_params(sa_cfg)
        self.assertEqual(params, {n: p["default"] for n, p in block_cfg["parameters"].items()})
        self.assertEqual(run_tb.variation_name("core", "a", params), run_sim.variation_name("core", "a", params))
        core_resolved = run_sim.resolve_materialization_params(block_cfg, params, sch_dir=self.base / "core_sch")
        sa_resolved = run_tb.resolve_materialization_params(project, sa_cfg, params, self.base / "sa_sch")
        self.assertEqual(sa_resolved, core_resolved)
        self.assertEqual(sa_resolved["w_m1"], "3u")
        self.assertEqual(sa_resolved["l_seg"], "10u")
        for test_name, test_cfg in _CONFIG["tests"]["core"].items():
            with self.subTest(test=test_name):
                self.assertEqual(run_tb.compute_definition_hash(project, sa_cfg, test_cfg),
                                 run_sim.compute_definition_hash(block_cfg, test_cfg))

    def test_hierarchical_materialization_matches_run_sim(self):
        core_sub = run_sim.variation_name("core", "a", {"w_base": "2u", "w_factor": "2", "l_total": "5u"})
        sim = self.root / "sim"
        _write(sim / "variations.jsonl", json.dumps({
            "name": core_sub, "block": "core", "topology": "a",
            "parameters": {"w_base": "2u", "w_factor": "2", "l_total": "5u"}}) + "\n")
        _write(sim / "results.jsonl", json.dumps({
            "variation": core_sub, "test": "level", "metric": "Level",
            "typical": 1.2, "min": 1.1, "max": 1.3}) + "\n")
        block_cfg = self.open_core("top", "default")
        project = run_tb.Project(self.root, sim, persist=True)
        params = {"X1_variation": core_sub, "bias_factor": "2"}
        core_dir, sa_dir = self.base / "core_sch", self.base / "sa_sch"
        core_dir.mkdir()
        sa_dir.mkdir()
        core_resolved = run_sim.resolve_materialization_params(block_cfg, params, sch_dir=core_dir)
        sa_resolved = run_tb.resolve_materialization_params(
            project, project.topology_cfg("top", "default"), params, sa_dir)
        self.assertEqual(sa_resolved, core_resolved)
        self.assertEqual(sa_resolved["bias_w"], "8u")
        self.assertEqual((sa_dir / "core.sch").read_text(), (core_dir / "core.sch").read_text())

    def test_xyce_netlist_fixups_match_run_sim(self):
        netlist = textwrap.dedent("""\
            .lib /m/cornerCAP.lib cap_typ
            .save i(vmeas)
            XC1 a b cap_cmomf w=10u l=5u mmin=1 mmax=3 m=2
            XC2 c d cap_cmomf w=1u l=1u
            .end
        """)
        core_path, sa_path = self.base / "core.spice", self.base / "sa.spice"
        core_path.write_text(netlist)
        sa_path.write_text(netlist)
        run_sim._strip_ngspice_save_lines(core_path)
        run_sim._lower_cmomf_for_xyce(core_path)
        run_tb._prepare_xyce_netlist(sa_path)
        self.assertEqual(sa_path.read_text(), core_path.read_text())

    def test_ngspice_error_detection_matches_log_diagnostics(self):
        from analog_designer.sim import log_diagnostics
        log = "Error: no such vector v(x)\n  tran simulation(s) aborted\ndoAnalyses: TRAN:  Timestep too small\nWarning: meh\n"
        core = sorted(d["message"] for d in log_diagnostics.parse(log, "ngspice") if d["severity"] == "error")
        self.assertEqual(sorted(run_tb._ngspice_errors(log)), core)


class EndToEndTests(_FixtureCase):
    def test_dry_run_leaves_the_project_untouched(self):
        before = _snapshot(self.root)
        summary_path = self.base / "summary.json"
        rc, out = self.run_tb("--block", "core", "--dry", "--json", str(summary_path))
        self.assertEqual(rc, 0, out)
        self.assertEqual(_snapshot(self.root), before)
        summary = json.loads(summary_path.read_text())
        level = summary["tests"]["level"]["metrics"][0]
        self.assertEqual(level["typical"], 1.025)  # tt / 25C / ibias=80n
        self.assertAlmostEqual(level["max"], 1.0 + 25 / 1000 + 0.1 + 0.01)
        self.assertEqual(summary["tests"]["area"]["metrics"][0]["typical"], 1)
        self.assertEqual(summary["tests"]["tsweep"]["status"], "success")

    def test_default_mode_writes_results_the_core_sees_as_fresh(self):
        rc, out = self.run_tb("--block", "core")
        self.assertEqual(rc, 0, out)
        block_cfg = self.open_core("core", "a")
        params = {n: p["default"] for n, p in block_cfg["parameters"].items()}
        name = run_sim.variation_name("core", "a", params)
        fresh, to_run, _ = run_sim._test_freshness(
            workspace.CONFIG["tests"]["core"], block_cfg, run_sim.load_results(), name, False)
        self.assertEqual(sorted(fresh), ["area", "level", "tsweep"])
        self.assertEqual(to_run, {})
        self.assertTrue(any(r["name"] == name for r in run_sim._read_jsonl(self.root / "sim" / "variations.jsonl")))
        runs = run_sim._read_jsonl(self.root / "sim" / name / "runs.jsonl")
        self.assertEqual(len(runs), 1 + 8 + 4)
        # Materialized only into the per-variation shadow tree, never sch/.
        self.assertFalse((self.root / "sch" / "core.sch").exists())
        self.assertIn("w=3u", (self.root / "sim" / name / "_src" / "sch" / "core.sch").read_text())
        # The internal temperature sweep bounds reached the testbench.
        tsweep = next((self.root / "sim" / name / "tsweep").glob("*/tb_tsweep.spice")).read_text()
        self.assertIn(".dc temp -40.0 125.0 5", tsweep)
        # Aux outputs (the .raw) are dropped after a successful run.
        self.assertEqual(list((self.root / "sim" / name).rglob("*.raw")), [])

        rc, out = self.run_tb("--block", "core")
        self.assertEqual(rc, 0, out)
        self.assertIn("skipping level: fresh result", out)

    def test_where_filter_runs_a_subset_and_records_nothing(self):
        rc, out = self.run_tb("--block", "core", "--test", "level", "--where", "corner=tt", "--where", "ibias=80n")
        self.assertEqual(rc, 0, out)
        self.assertIn("running 2 simulation(s)", out)
        self.assertFalse((self.root / "sim" / "results.jsonl").exists())

    def test_hierarchical_block_needs_and_uses_sub_block_results(self):
        rc, out = self.run_tb("--block", "top", "--dry")
        self.assertEqual(rc, 2)
        self.assertIn("X1_variation", out)
        rc, out = self.run_tb("--block", "top", "--dry", "--import-metric", "x1_level=1.5")
        self.assertEqual(rc, 0, out)

        rc, out = self.run_tb("--block", "core", "--test", "level")
        self.assertEqual(rc, 0, out)
        core_name = run_tb.variation_name("core", "a", {"w_base": "1u", "w_factor": "3", "l_total": "20u"})
        summary_path = self.base / "top.json"
        rc, out = self.run_tb("--block", "top", "--param", f"X1_variation={core_name}", "--dry",
                              "--keep", "--json", str(summary_path))
        self.assertEqual(rc, 0, out)
        work = Path(out.split("kept work directory: ")[1].split()[0])
        self.addCleanup(lambda: __import__("shutil").rmtree(work, ignore_errors=True))
        top_sch = next(work.glob("top-default-*/_src/sch/top.sch")).read_text()
        self.assertIn("w=6u", top_sch)          # x1_w (3u) * bias_factor (2)
        self.assertIn("r=1.025e+06", top_sch)   # stored X1 Level typical * 1e6

    def test_missing_tools_are_reported_before_running(self):
        rc, out = self.run_tb("--block", "core", "--dry", "--ngspice", str(self.base / "nope"), "--pdk", "absent")
        self.assertEqual(rc, 2)
        self.assertIn("PDK directory not found", out)

    def test_list(self):
        rc, out = self.run_tb("--list")
        self.assertEqual(rc, 0)
        self.assertIn("tsweep", out)
        self.assertIn("4 condition(s)", out)  # temperature swept internally: corner x vdd only


class ExportTests(_FixtureCase):
    def test_export_writes_a_stamped_copy_that_check_accepts(self):
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(export.main([str(self.root), "--check"]), 1)
            self.assertEqual(export.main([str(self.root)]), 0)
            self.assertEqual(export.main([str(self.root), "--check"]), 0)
        dest = self.root / "tools" / "run_tb.py"
        lines = dest.read_text().splitlines()
        self.assertTrue(lines[0].startswith("#!"))
        self.assertTrue(lines[1].startswith(export.STAMP_PREFIX))
        self.assertTrue(os.access(dest, os.X_OK))
        dest.write_text(dest.read_text() + "# local edit\n")
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(export.main([str(self.root), "--check"]), 1)


if __name__ == "__main__":
    unittest.main()
