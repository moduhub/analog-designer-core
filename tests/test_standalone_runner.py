"""Tests for host/docker execution (analog_designer/core/executor.py and
workspace's execution settings) and for the standalone runner: the CLI
(analog_designer/standalone/cli.py) and the tools/run_tb.py bundle export.py
generates from it.

The end-to-end checks run the real run_sim.py pipeline in host mode on a
small fixture project, against fake xschem/ngspice executables (tiny Python
scripts), so they need no EDA tools: --dry leaving the project untouched,
the default mode writing sim/ results the core sees as fresh, --where,
hierarchical import_metrics, and the generated bundle running in a fresh
interpreter that can't import this checkout.
"""
import contextlib
import io
import json
import os
import shutil
import subprocess
import sys
import tempfile
import textwrap
import time
import unittest
from pathlib import Path
from unittest import mock

from analog_designer.core import executor, settings, workspace
from analog_designer.sim import run_sim
from analog_designer.standalone import cli, export

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
    import os
    args = sys.argv[1:]
    # Real xschemrc files resolve paths against $env(PWD), not the cwd.
    assert os.environ.get("PWD") == os.getcwd(), (os.environ.get("PWD"), os.getcwd())
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
        base = Path(self._tmp.name).resolve()
        self.root = base / "proj"
        _make_project(self.root)
        self.tool_args = _make_tools(base)
        self.base = base
        for target, attr, value in (
            (workspace, "_LAST_FOLDER_FILE", base / "last_folder.txt"),
            (settings, "_SETTINGS_FILE", base / "settings.json"),
        ):
            patcher = mock.patch.object(target, attr, value)
            patcher.start()
            self.addCleanup(patcher.stop)
        env = mock.patch.dict(os.environ, {"PDK_ROOT": "", "PDK": "", "ANALOG_DESIGNER_EXECUTION": ""})
        env.start()
        self.addCleanup(env.stop)
        self.addCleanup(self._tmp.cleanup)
        self.addCleanup(workspace.reset_overrides)
        saved_path = list(sys.path)
        self.addCleanup(lambda: sys.path.__setitem__(slice(None), saved_path))

    def run_cli(self, *args):
        out = io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(out):
            try:
                rc = cli.main(["--project-root", str(self.root), *self.tool_args, *args])
            except SystemExit as exc:
                rc = exc.code
        return rc, out.getvalue()

    def open_core(self, block, topology):
        workspace.open_folder(str(self.root), block=block, topology=topology, remember=False)
        return workspace.CONFIG["blocks"][block]["topologies"][topology]


class ExecutionSettingsTests(_FixtureCase):
    def test_docker_is_the_default_and_settings_choose_host(self):
        self.open_core("core", "a")
        self.assertEqual(workspace.execution_mode(), "docker")
        self.assertEqual(workspace.container_project_root(), "/home/moduhub/work/proj")
        stored = settings.load()
        stored["execution"]["mode"] = "host"
        settings.save(stored)
        self.assertEqual(workspace.execution_mode(), "host")
        self.assertEqual(workspace.container_project_root(), self.root.as_posix())

    def test_override_beats_environment_beats_settings(self):
        self.open_core("core", "a")
        with mock.patch.dict(os.environ, {"ANALOG_DESIGNER_EXECUTION": "host"}):
            self.assertEqual(workspace.execution_mode(), "host")
            workspace.set_overrides(mode="docker")
            self.assertEqual(workspace.execution_mode(), "docker")

    def test_exec_path_maps_into_the_container_and_refuses_outside_paths(self):
        self.open_core("core", "a")
        self.assertEqual(workspace.exec_path(self.root / "sim" / "v" / "t"), "/home/moduhub/work/proj/sim/v/t")
        with self.assertRaises(ValueError):
            workspace.exec_path(self.base / "elsewhere")
        workspace.set_overrides(mode="host")
        self.assertEqual(workspace.exec_path(self.base / "elsewhere"), (self.base / "elsewhere").as_posix())

    def test_host_pdk_falls_back_to_the_image_tag(self):
        self.open_core("core", "a")
        self.assertEqual(workspace.host_pdk(), ("", "ihp-fake"))
        with mock.patch.dict(os.environ, {"PDK_ROOT": "/pdks", "PDK": "ihp-other"}):
            self.assertEqual(workspace.host_pdk(), ("/pdks", "ihp-other"))
            workspace.set_overrides(pdk="gf180mcuD")
            self.assertEqual(workspace.host_pdk(), ("/pdks", "gf180mcuD"))

    def test_settings_written_before_execution_modes_still_load(self):
        settings._SETTINGS_FILE.write_text(json.dumps({"container": {"image": "x:y", "max_parallel": 3}}))
        loaded = settings.load()
        self.assertEqual(loaded["execution"], {"mode": "docker"})
        self.assertEqual(loaded["container"]["cpu_budget"], 3)


#: Host mode and the generated runner need a POSIX bash; on Windows
#: Popen(["bash"]) finds the WSL launcher in System32 first.
needs_posix_bash = unittest.skipIf(os.name == "nt", "needs a POSIX bash")


@needs_posix_bash
class HostExecutorTests(unittest.TestCase):
    def test_runs_bash_in_its_own_directory(self):
        result = executor.HostExecutor().run('cd /tmp && echo "$PWD" && exit 3')
        self.assertEqual(result.returncode, 3)
        self.assertEqual(result.stdout.strip(), "/tmp")

    def test_timeout_kills_the_whole_process_group(self):
        marker = Path(tempfile.mkdtemp()) / "survivor"
        start = time.monotonic()
        result = executor.HostExecutor().run(f'(sleep 2; touch "{marker}") & sleep 5', timeout=1)
        self.assertEqual(result.returncode, 124)
        self.assertLess(time.monotonic() - start, 4)
        time.sleep(1.5)
        self.assertFalse(marker.exists(), "a child of the timed-out script kept running")

    def test_sigterm_takes_running_host_scripts_down_too(self):
        """What the GUI's Cancel relies on in host mode."""
        marker = Path(tempfile.mkdtemp()) / "survivor"
        job = subprocess.Popen([sys.executable, "-c", textwrap.dedent(f"""
            from analog_designer.core import executor
            executor.kill_host_jobs_on_terminate()
            print("started", flush=True)
            executor.HostExecutor().run('sleep 3; touch "{marker}"', timeout=30)
        """)], stdout=subprocess.PIPE, text=True)
        self.assertEqual(job.stdout.readline().strip(), "started")
        time.sleep(0.5)
        job.terminate()
        job.wait(timeout=10)
        time.sleep(3.5)
        self.assertFalse(marker.exists(), "the simulator kept running after its job was terminated")

    def test_docker_exec_dispatches_to_an_executor(self):
        self.assertEqual(run_sim.docker_exec(executor.HostExecutor(), "echo hi").stdout, "hi\n")


@needs_posix_bash
class StandaloneCliTests(_FixtureCase):
    def test_dry_run_leaves_the_project_untouched(self):
        before = _snapshot(self.root)
        summary_path = self.base / "summary.json"
        rc, out = self.run_cli("--block", "core", "--dry", "--json", str(summary_path))
        self.assertEqual(rc, 0, out)
        self.assertEqual(_snapshot(self.root), before)
        summary = json.loads(summary_path.read_text())
        level = summary["tests"]["level"]["metrics"][0]
        self.assertEqual(level["typical"], 1.025)  # tt / 25C / ibias=80n
        self.assertAlmostEqual(level["max"], 1.0 + 25 / 1000 + 0.1 + 0.01)
        self.assertEqual(summary["tests"]["area"]["metrics"][0]["typical"], 1)
        self.assertEqual(summary["tests"]["tsweep"]["status"], "success")

    def test_default_mode_writes_results_the_core_sees_as_fresh(self):
        rc, out = self.run_cli("--block", "core")
        self.assertEqual(rc, 0, out)
        block_cfg = self.open_core("core", "a")
        params = {n: p["default"] for n, p in block_cfg["parameters"].items()}
        name = run_sim.variation_name("core", "a", params)
        fresh, to_run, _ = run_sim._test_freshness(
            workspace.CONFIG["tests"]["core"], block_cfg, run_sim.load_results(), name, False)
        self.assertEqual(sorted(fresh), ["area", "level", "tsweep"])
        self.assertEqual(to_run, {})
        runs = run_sim._read_jsonl(self.root / "sim" / name / "runs.jsonl")
        self.assertEqual(len(runs), 1 + 8 + 4)
        # Shadow materialization only: the tracked sch/ tree is never written.
        self.assertFalse((self.root / "sch" / "core.sch").exists())
        self.assertIn("w=3u", (self.root / "sim" / name / "_src" / "sch" / "core.sch").read_text())
        tsweep = next((self.root / "sim" / name / "tsweep").glob("*/tb_tsweep.spice")).read_text()
        self.assertIn(".dc temp -40.0 125.0 5", tsweep)
        self.assertEqual(list((self.root / "sim" / name).rglob("*.raw")), [])  # aux outputs dropped

        rc, out = self.run_cli("--block", "core")
        self.assertEqual(rc, 0, out)
        self.assertIn("SKIPPED, fresh result", out)

    def test_where_runs_a_subset_and_records_nothing(self):
        rc, out = self.run_cli("--block", "core", "--test", "level", "--where", "corner=tt", "--where", "ibias=80n")
        self.assertEqual(rc, 0, out)
        self.assertIn("not recorded", out)
        self.assertEqual(len(run_sim._read_jsonl(next((self.root / "sim").glob("core-a-*/runs.jsonl")))), 2)
        self.assertFalse((self.root / "sim" / "results.jsonl").exists())

    def test_hierarchical_block_uses_stored_or_given_sub_block_metrics(self):
        rc, out = self.run_cli("--block", "top", "--dry")
        self.assertNotEqual(rc, 0)
        self.assertIn("X1_variation", out)
        rc, out = self.run_cli("--block", "top", "--dry", "--import-metric", "x1_level=1.5")
        self.assertEqual(rc, 0, out)

        rc, out = self.run_cli("--block", "core", "--test", "level")
        self.assertEqual(rc, 0, out)
        core_name = run_sim.variation_name("core", "a", {"w_base": "1u", "w_factor": "3", "l_total": "20u"})
        rc, out = self.run_cli("--block", "top", "--param", f"X1_variation={core_name}", "--dry", "--keep")
        self.assertEqual(rc, 0, out)
        work = Path(out.split("kept work directory: ")[1].split()[0])
        self.addCleanup(shutil.rmtree, work, True)
        top_sch = next(work.glob("sim/top-default-*/_src/sch/top.sch")).read_text()
        self.assertIn("w=6u", top_sch)          # x1_w (3u) * bias_factor (2)
        self.assertIn("r=1.025e+06", top_sch)   # stored X1 Level typical * 1e6

    def test_missing_tools_are_reported_before_running(self):
        rc, out = self.run_cli("--block", "core", "--dry", "--ngspice", str(self.base / "nope"), "--pdk", "absent")
        self.assertEqual(rc, 2)
        self.assertIn("PDK directory", out)
        self.assertIn("ngspice not found", out)

    def test_list(self):
        rc, out = self.run_cli("--list")
        self.assertEqual(rc, 0, out)
        self.assertIn("4 condition(s)", out)  # tsweep: temperature swept inside the testbench


class ProjectSwitchTests(_FixtureCase):
    def test_parsers_use_the_open_projects_shared_helpers(self):
        """The GUI's Open Folder keeps one process across projects: a parser
        of the second project must not import the first one's
        tb/_shared/parser_common.py (sys.path + sys.modules)."""
        other = self.base / "other"
        _make_project(other)
        _write(other / "tb" / "_shared" / "parser_common.py", "def read_value(path):\n    return -1.0\n")
        data = self.base / "x.data"
        data.write_text("0 2.5\n")
        workspace.open_folder(str(other), remember=False)
        self.assertEqual(run_sim.load_parser("tb/core/tb_level.py").extract(data), {"value": -1.0})
        workspace.open_folder(str(self.root), remember=False)
        self.assertEqual(run_sim.load_parser("tb/core/tb_level.py").extract(data), {"value": 2.5})


class BundleTests(_FixtureCase):
    def test_bundle_covers_the_pipeline_and_nothing_gui(self):
        files = export.bundled_files()
        for needed in ("analog_designer/sim/run_sim.py", "analog_designer/core/executor.py",
                       "analog_designer/standalone/cli.py", "analog_designer/sim/openems_generator_runner.py"):
            self.assertIn(needed, files)
        self.assertFalse([f for f in files if "/gui/" in f])

    def test_bundle_hash_ignores_line_endings(self):
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "a.py").write_bytes(b"x = 1\ny = 2\n")
            with mock.patch.object(export, "SOURCE_ROOT", Path(tmp)):
                lf = export.bundle_hash(["a.py"])
                (Path(tmp) / "a.py").write_bytes(b"x = 1\r\ny = 2\r\n")
                self.assertEqual(export.bundle_hash(["a.py"]), lf)

    @needs_posix_bash
    def test_generated_runner_works_without_this_checkout(self):
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(export.main([str(self.root)]), 0)
        runner = self.root / "tools" / "run_tb.py"
        before = _snapshot(self.root)
        env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
        result = subprocess.run(
            [sys.executable, "-I", str(runner), "--block", "core", "--test", "level", "--dry", *self.tool_args],
            cwd=self.base, env=env, capture_output=True, text=True, timeout=120,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("Level", result.stdout)
        self.assertEqual(_snapshot(self.root), before)
        self.assertFalse(list(Path(tempfile.gettempdir()).glob("run_tb_src_*")), "bundle sources left behind")

    def test_status_follows_the_tool_version(self):
        dest = self.root / "tools" / "run_tb.py"
        self.assertEqual(export.runner_status(self.root), "missing")
        export.export_runner(self.root)
        self.assertEqual(export.runner_status(self.root), "current")
        self.assertTrue(os.access(dest, os.X_OK))
        dest.write_text(dest.read_text().replace("#|", "#|# changed\n#|", 1))
        self.assertEqual(export.runner_status(self.root), "stale")
        self.assertEqual(export.sync_runner(self.root), dest)
        self.assertEqual(export.runner_status(self.root), "current")
        self.assertIsNone(export.sync_runner(self.root))

    def test_hand_written_files_are_never_overwritten(self):
        dest = self.root / "tools" / "run_tb.py"
        _write(dest, "print('mine')\n")
        self.assertEqual(export.runner_status(self.root), "foreign")
        self.assertIsNone(export.sync_runner(self.root))
        with self.assertRaises(SystemExit), contextlib.redirect_stdout(io.StringIO()):
            export.main([str(self.root)])
        self.assertEqual(dest.read_text(), "print('mine')\n")
        _write(dest, "#!/usr/bin/env python3\n# vendored from analog-designer-core abc123\n")
        self.assertEqual(export.runner_status(self.root), "stale")  # the first, hand-synced runner


if __name__ == "__main__":
    unittest.main()
