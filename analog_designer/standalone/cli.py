"""Command line of the standalone testbench runner -- the entry point of the
tools/run_tb.py that `python -m analog_designer.standalone.export` writes
into a project repo (a bundle of this module plus the exact run_sim.py
pipeline it calls, see export.py). Also runnable in place:
`python -m analog_designer.standalone.cli --project-root PROJECT ...`.

Everything simulation-related is run_sim.run_variation() itself; this
module only maps flags onto it:
  * where simulators run: this machine (default -- e.g. a shell inside the
    EDA image) or, with --execution docker, a container of the project's
    image like the GUI does;
  * where output goes: the project's sim/ (default, same files the GUI
    reads, fresh tests skipped) or, with --dry, a temporary directory --
    the project tree is then only read (materialization always goes to the
    per-variation shadow sim/<variation>/_src/, never to sch/);
  * what it reports: each test's metrics as it finishes, a summary of all of
    them at the end, and the figures (the GUI's sim/<variation>/<test>/*.png,
    listed on the console; --no-plots skips them). Interactive viewing is the
    GUI's job;
  * what runs: --test, --where (a subset of conditions, never recorded),
    --variation/--param, --import-metric.
"""
import argparse
import json
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path

# Loading the project's tb/*.py parsers must not leave __pycache__ in the
# project (--dry promises no trace), nor try to open a window.
sys.dont_write_bytecode = True
os.environ.setdefault("MPLBACKEND", "Agg")

from analog_designer.core import workspace  # noqa: E402
from analog_designer.sim import run_sim  # noqa: E402

_IHP_OSDI = ("psp103", "psp103_nqs", "r3_cmc", "mosvar")


class UsageError(Exception):
    """Printed without a traceback, exit status 2."""


def _default_project_root():
    here = Path(sys.argv[0]).resolve().parent
    for candidate in (Path.cwd(), here, here.parent):
        if (candidate / "config.json").is_file():
            return candidate
    return Path.cwd()


def _pairs(values, flag):
    out = {}
    for item in values:
        key, sep, value = item.partition("=")
        if not sep or not key:
            raise UsageError(f"{flag} expects NAME=VALUE, got {item!r}")
        out[key] = value
    return out


def list_project():
    config = workspace.CONFIG
    defaults = config.get("defaults", {})
    for block, block_entry in config.get("blocks", {}).items():
        print(block)
        for topology in block_entry.get("topologies") or {}:
            print(f"  topology {topology}")
        for test_name, test_cfg in config.get("tests", {}).get(block, {}).items():
            tb_path = workspace.PROJECT_ROOT / test_cfg["testbench"]
            tb_text = tb_path.read_text(encoding="utf-8") if tb_path.exists() else ""
            n = len(list(run_sim.condition_matrix(test_cfg, defaults, run_sim.internal_sweep_axis(test_cfg, tb_text))))
            simulator = test_cfg.get("simulator", "ngspice")
            print(f"    test {test_name:<24} {simulator:<8} {n:>4} condition(s)  {test_cfg['testbench']}")


def doctor(executor, simulators):
    """What the executor sees: PDK, tools, OSDI models. Returns a list of
    problems (empty: ready)."""
    tools = run_sim._resolve_tools(executor)
    probe = executor.run(
        'echo "$PDK_ROOT"; echo "$PDK"; '
        f'for t in "{tools["xschem"]}" "{tools["ngspice"]}" "{tools["xyce"]}"; do command -v "$t" >/dev/null && echo yes || echo no; done; '
        'test -d "$PDK_ROOT/$PDK" && echo yes || echo no; '
        'test -f "$PDK_ROOT/$PDK/libs.tech/xschem/xschemrc" && echo yes || echo no; '
        + "".join(f'test -f "$PDK_ROOT/$PDK/libs.tech/ngspice/osdi/{n}.osdi" && echo yes || echo no; ' for n in _IHP_OSDI)
    )
    lines = probe.stdout.split("\n")
    pdk_root, pdk = lines[0], lines[1]
    has_xschem, has_ngspice, has_xyce, has_pdk, has_rc = (l == "yes" for l in lines[2:7])
    osdi = dict(zip(_IHP_OSDI, (l == "yes" for l in lines[7:7 + len(_IHP_OSDI)])))

    def mark(ok):
        return "" if ok else "  MISSING"
    print(f"  execution         {executor.kind}" + (f" ({executor})" if executor.kind == "docker" else ""))
    print(f"  PDK               {pdk or '?'} @ {pdk_root or '?'}{mark(has_pdk)}")
    print(f"  xschem            {tools['xschem']}{mark(has_xschem)}")
    print(f"  ngspice           {tools['ngspice']}{mark(has_ngspice)}")
    print(f"  Xyce              {tools['xyce']}{mark(has_xyce)}")
    print(f"  X display         {tools['display'] or 'none (xschem -x needs none)'}")
    problems = []
    if not pdk:
        problems.append("PDK name unknown: set $PDK or pass --pdk")
    elif not has_pdk:
        problems.append(f"PDK directory {pdk_root}/{pdk} not found: set $PDK_ROOT/$PDK or pass --pdk-root/--pdk")
    elif not has_rc and not (workspace.PROJECT_ROOT / "xschemrc").exists():
        problems.append("no xschemrc in the project nor in the PDK's libs.tech/xschem")
    if not has_xschem:
        problems.append("xschem not found (PATH, $XSCHEM or --xschem)")
    if not has_ngspice and "ngspice" in simulators:
        problems.append("ngspice not found (PATH, $NGSPICE or --ngspice)")
    if not has_xyce and "xyce" in simulators:
        problems.append("Xyce not found (PATH, $XYCE or --xyce), but a selected test uses it")
    if has_pdk and not pdk.startswith("gf180mcu"):
        print("  osdi              " + ", ".join(f"{n}{'' if ok else ' (MISSING)'}" for n, ok in osdi.items()))
        missing = [n for n, ok in osdi.items() if not ok]
        if missing and "ngspice" in simulators:
            problems.append(f"OSDI models missing from $PDK_ROOT/$PDK/libs.tech/ngspice/osdi: {', '.join(missing)} "
                            "(compile them with openvaf; needs ngspice >= 44 for OSDI 0.4)")
    return problems


def run(args):
    config = workspace.CONFIG
    block, topology = workspace.BLOCK, workspace.TOPOLOGY
    block_cfg = config["blocks"][block]["topologies"][topology]
    all_tests = config.get("tests", {}).get(block, {})
    unknown = [t for t in args.test if t not in all_tests]
    if unknown:
        raise UsageError(f"unknown test(s) for {block}: {', '.join(unknown)} (have: {', '.join(all_tests)})")
    tests = {t: all_tests[t] for t in (args.test or all_tests)}

    if args.variation:
        params = dict(run_sim._variation_params(args.variation))
    else:
        params = {n: pdef["default"] for n, pdef in block_cfg["parameters"].items()}
    for key, value in _pairs(args.param, "--param").items():
        if key not in block_cfg["parameters"]:
            raise UsageError(f"--param {key}: not a parameter of {block}/{topology}")
        params[key] = value

    with run_sim.managed_executor() as executor:
        problems = doctor(executor, {cfg.get("simulator", "ngspice") for cfg in tests.values()})
        if problems:
            raise UsageError("environment not ready:\n  - " + "\n  - ".join(problems))
        ctx = run_sim.setup_container(executor)
        try:
            return run_sim.run_variation(
                block_cfg, tests, config["defaults"], params, force=args.force or args.dry, container_ctx=ctx,
                origin={"kind": "manual", "tool": "run_tb"}, shadow=True, where=_pairs(args.where, "--where") or None,
            )
        except run_sim.MissingCrossBlockMetric as exc:
            raise UsageError(
                f"{exc}\n  pick the sub-block variation with --param <instance>_variation=<name> (run that "
                f"sub-block's test first, without --dry), or give the value with --import-metric NAME=VALUE"
            ) from None


def _num(value):
    return "-" if value is None else f"{value:.6g}"


def format_summary(outcome):
    """The end-of-run recap: one block per test with each metric's typical
    value and range (mean +- std for a Monte Carlo metric), or the first line
    of its error. The same numbers the run printed as each test finished,
    gathered after the simulator chatter."""
    lines = [f"summary: {outcome['variation']}"]
    for test, result in outcome["tests"].items():
        status = result["status"]
        if status == "error":
            reason = next((l.strip() for l in str(result.get("error", "")).splitlines() if l.strip()), "")
            lines.append(f"  {test}: ERROR {reason}")
            continue
        note = {"fresh": " (stored result, not rerun)", "partial": " (some conditions failed)"}.get(status, "")
        lines.append(f"  {test}{note}")
        for m in result["metrics"]:
            central = f"{_num(m.get('mean'))} +- {_num(m.get('std'))}" if m["typical"] is None else _num(m["typical"])
            lines.append(f"    {m['name']}: {central} {m.get('unit', '')} [{_num(m['min'])} .. {_num(m['max'])}]")
    return "\n".join(lines)


def make_plots(args, outcome):
    """Writes each test's figures next to its results, sim/<variation>/<test>/
    (the files the GUI shows), and lists them. The run never fails because of
    a figure: a parser's plotting error or a missing matplotlib is reported
    and skipped. Nothing is drawn for --dry (it writes nothing) or --where
    (a subset of conditions) -- the console says so."""
    if args.no_plots:
        return
    if args.dry or args.where:
        print(f"figures: not generated with {'--dry' if args.dry else '--where'} "
              f"({'nothing is written' if args.dry else 'partial run'}); run without it to get them in sim/")
        return
    config, block = workspace.CONFIG, workspace.BLOCK
    variation = outcome["variation"]
    written = []
    for test, result in outcome["tests"].items():
        if result["status"] == "error":
            continue
        try:
            made = run_sim.generate_plot(variation, test, config["tests"][block][test], config["defaults"])
        except ImportError as exc:
            print(f"figures: skipped, {exc}")
            return
        except Exception as exc:
            print(f"figures: {test} failed ({type(exc).__name__}: {exc})")
            continue
        if made:
            written += sorted((workspace.sim_root() / variation / test).glob(f"{test}*.png"))
    if written:
        print("figures:")
        for path in written:
            try:
                path = path.relative_to(workspace.PROJECT_ROOT)
            except ValueError:
                pass
            print(f"  {path.as_posix()}")


def build_parser():
    ap = argparse.ArgumentParser(
        prog="run_tb.py", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--project-root", help="project folder (default: auto-detected from the cwd / script location)")
    ap.add_argument("--block", help="block (default: the first one in config.json)")
    ap.add_argument("--topology", help="topology (default: the block's first one)")
    ap.add_argument("--test", action="append", default=[], help="test to run (repeatable; default: all of the block's)")
    ap.add_argument("--variation", help="a registered variation (sim/variations.jsonl) instead of the defaults")
    ap.add_argument("--param", action="append", default=[], metavar="NAME=VALUE", help="override one free parameter")
    ap.add_argument("--import-metric", action="append", default=[], metavar="NAME=VALUE",
                    help="value for a hierarchical block's derived_parameters.import_metrics entry")
    ap.add_argument("--where", action="append", default=[], metavar="AXIS=VALUE",
                    help="only conditions with this value (e.g. corner=tt); such partial runs are never recorded")
    ap.add_argument("--dry", action="store_true",
                    help="output to a temporary directory, deleted at the end; nothing is written to the project")
    ap.add_argument("--keep", action="store_true", help="with --dry: keep the temporary directory and print it")
    ap.add_argument("--force", action="store_true", help="rerun tests whose stored result is still fresh")
    ap.add_argument("--json", help="write a JSON summary of the results here")
    ap.add_argument("--no-plots", action="store_true",
                    help="skip the figures (default: written to sim/<variation>/<test>/ and listed at the end)")
    ap.add_argument("--jobs", type=int, default=None, help="CPU cores to use (default: all)")
    ap.add_argument("--keep-aux", action="store_true", help="keep auxiliary simulator outputs (.raw etc.)")
    ap.add_argument("--execution", choices=("host", "docker"), default="host",
                    help="run the tools on this machine (default) or in a container of config.json's image")
    ap.add_argument("--pdk-root", help="host mode: PDK root (default: $PDK_ROOT)")
    ap.add_argument("--pdk", help="host mode: PDK name (default: $PDK, else the tag of config.json's container.image)")
    ap.add_argument("--xschem", help="xschem binary (default: $XSCHEM, else PATH)")
    ap.add_argument("--ngspice", help="ngspice binary (default: $NGSPICE, else PATH)")
    ap.add_argument("--xyce", help="Xyce binary (default: $XYCE, else PATH)")
    ap.add_argument("--list", action="store_true", help="list blocks, topologies and tests, then exit")
    ap.add_argument("--doctor", action="store_true", help="report what was found (PDK, tools, models), then exit")
    return ap


def main(argv=None):
    args = build_parser().parse_args(argv)
    root = Path(args.project_root or _default_project_root()).resolve()
    tmp = None
    saved_env = dict(os.environ)
    try:
        if not (root / "config.json").is_file():
            raise UsageError(f"no config.json in {root} -- not a project folder (use --project-root)")
        for flag, var in ((args.xschem, "XSCHEM"), (args.ngspice, "NGSPICE"), (args.xyce, "XYCE")):
            if flag:
                os.environ[var] = flag
        if args.keep_aux:
            os.environ[run_sim.KEEP_AUX_ENV] = "1"
        workspace.set_overrides(mode=args.execution, pdk_root=args.pdk_root, pdk=args.pdk, cpu_budget=args.jobs)
        if args.dry or args.list or args.doctor:
            if args.execution == "docker" and args.dry:
                raise UsageError("--dry needs --execution host (a container only sees the project folder)")
            tmp = Path(tempfile.mkdtemp(prefix="run_tb_"))
            sim_root = tmp / "sim"
            sim_root.mkdir()
            # Read-only copies, so a dry run can still resolve a registered
            # --variation or a sub-block's stored import_metrics result.
            for name in ("variations.jsonl", "results.jsonl"):
                if (root / "sim" / name).is_file():
                    shutil.copyfile(root / "sim" / name, sim_root / name)
            workspace.set_overrides(sim_root=sim_root, read_only=True)
        try:
            workspace.open_folder(root, block=args.block, topology=args.topology, remember=False)
        except (KeyError, ValueError) as exc:
            raise UsageError(f"cannot open {root}: {exc}") from None
        if not workspace.CONFIG["blocks"][workspace.BLOCK].get("topologies"):
            raise UsageError(f"block {workspace.BLOCK!r} has no topologies declared yet")
        run_sim.IMPORT_METRIC_OVERRIDES.update(_pairs(args.import_metric, "--import-metric"))

        if args.list:
            list_project()
            return 0
        if args.doctor:
            with run_sim.managed_executor() as executor:
                problems = doctor(executor, {"ngspice", "xyce"})
            for problem in problems:
                print(f"  ! {problem}")
            return 1 if problems else 0

        start = time.monotonic()
        outcome = run(args)
        print(f"\n{format_summary(outcome)}")
        make_plots(args, outcome)
        print(f"\ndone in {time.monotonic() - start:.1f}s" + ("  (with errors)" if outcome["any_error"] else ""))
        if args.json:
            Path(args.json).write_text(json.dumps(outcome, indent=2, default=str) + "\n", encoding="utf-8")
        return 1 if outcome["any_error"] else 0
    except UsageError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    finally:
        os.environ.clear()
        os.environ.update(saved_env)
        workspace.reset_overrides()
        run_sim.IMPORT_METRIC_OVERRIDES.clear()
        if tmp:
            if args.keep and args.dry:
                print(f"kept work directory: {tmp}")
            else:
                shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
