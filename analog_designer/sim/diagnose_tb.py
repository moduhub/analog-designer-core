#!/usr/bin/env python3
"""Time every (test, condition) simulation for one variation in isolation,
to find which testbench/condition is slow to converge or hangs until the
ngspice timeout -- the root cause of a GUI "Update Range" job that looks
frozen (see analog_designer/gui/run_trigger.py): the batch itself runs in a
background thread/subprocess, so the GUI never actually locks up, but a
single non-convergent (test, condition) pair can sit for up to ~290s with
zero visible feedback, which reads exactly like a freeze.

Unlike run_sim.py/update_variations.py, this never writes into
sim/<variation>/ (runs.jsonl) or sim/results.jsonl -- it materializes into
its own sim/_diagnose-<run_id>/ scratch directory (deleted afterward unless
--keep is given) so a diagnostic scan can never be mistaken for -- or
interfere with -- real run history. Kept a flat, single-level sim/<name>
directory (not sim/_diagnose/<run_id>) to match
run_sim.materialize_variation_shadow()'s own assumption that the
container-side sim dir is always exactly sim/<sim_dir.name> -- an extra
path segment there would silently point its returned --rcfile at a path
that doesn't exist inside the container.

Usage:
    python -m analog_designer.sim.diagnose_tb [variation] [--test NAME ...]
        [--limit N] [--ngspice-timeout SECONDS] [--keep]

Examples:
    # Full scan of every test/condition for the config.json defaults.
    python -m analog_designer.sim.diagnose_tb

    # Just the suspect mismatch test, first 5 mc_seed values, fast-fail
    # after 30s instead of waiting out the full 290s per seed.
    python -m analog_designer.sim.diagnose_tb --test vref_mismatch --limit 5 --ngspice-timeout 30
"""
import argparse
import shutil
import sys
import time

from analog_designer.core import workspace
from analog_designer.sim import run_sim

#: A run that took at least this fraction of --ngspice-timeout is flagged
#: as "approaching timeout" even if it happened to finish successfully --
#: worth investigating before it tips over into an outright non-convergent
#: hang under slightly different conditions (a different corner, a
#: different mc_seed draw, ...).
_SLOW_FRACTION = 0.8


def main():
    arg_parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    arg_parser.add_argument(
        "variation", nargs="?", default=None,
        help="existing variation name from sim/variations.jsonl to diagnose; omit for the config.json default parameters",
    )
    arg_parser.add_argument("--project-root", default=None, help="project folder to operate on; defaults to the last-opened folder, else CWD")
    arg_parser.add_argument("--block", default=None, help="block to operate on; defaults to the first declared in config.json")
    arg_parser.add_argument("--topology", default=None, help="topology to operate on; defaults to the first declared for --block")
    arg_parser.add_argument(
        "--test", action="append", dest="tests", default=None,
        help="restrict the scan to this test name (repeatable); default: every test declared for the block",
    )
    arg_parser.add_argument(
        "--limit", type=int, default=None,
        help="only run the first N conditions per test -- e.g. to scan a 50-value mc_seed sweep without waiting for all of it",
    )
    arg_parser.add_argument(
        "--ngspice-timeout", type=int, default=290,
        help="per-(test,condition) ngspice timeout in seconds (default: 290, same ceiling a real run uses); "
             "lower this for a fast scan that just wants to know WHICH conditions are slow, not their exact real-run duration",
    )
    arg_parser.add_argument(
        "--keep", action="store_true",
        help="keep sim/_diagnose-<run_id>/ afterward (netlists + ngspice.log per condition) instead of deleting it",
    )
    args = arg_parser.parse_args()

    workspace.open_folder(args.project_root, block=args.block, topology=args.topology)
    config = workspace.CONFIG
    defaults = config["defaults"]
    block_cfg = config["blocks"][workspace.BLOCK]["topologies"][workspace.TOPOLOGY]
    tests = config["tests"][workspace.BLOCK]
    if args.tests:
        unknown = sorted(set(args.tests) - set(tests))
        if unknown:
            sys.exit(f"unknown test(s) {unknown}; declared tests for {workspace.BLOCK}: {sorted(tests)}")
        tests = {t: tests[t] for t in args.tests}

    params = (
        run_sim._variation_params(args.variation) if args.variation
        else {n: pdef["default"] for n, pdef in block_cfg["parameters"].items()}
    )

    sim_dir = workspace.sim_root() / f"_diagnose-{int(time.time())}"
    container_sim_dir = workspace.exec_path(sim_dir)

    print(
        f"diagnosing {len(tests)} test(s) for {workspace.BLOCK}/{workspace.TOPOLOGY} "
        f"({'variation ' + args.variation if args.variation else 'config.json defaults'}), "
        f"ngspice timeout {args.ngspice_timeout}s"
    )

    timings = []  # (test_name, label, elapsed_seconds, status, ngspice_exit_code)
    try:
        container_rcfile = run_sim.materialize_variation_shadow(sim_dir, block_cfg, params, block=workspace.BLOCK)
        with run_sim.managed_executor() as container:
            ctx = run_sim.setup_container(container)
            for test_name, test_cfg in tests.items():
                simulator = test_cfg.get("simulator", "ngspice")
                runner = run_sim.SIMULATOR_RUNNERS.get(simulator)
                if runner is None:
                    print(f"  {test_name}: skipped (simulator {simulator!r} not implemented)")
                    continue

                tb_source = workspace.PROJECT_ROOT / test_cfg["testbench"]
                tb_text = tb_source.read_text(encoding="utf-8")
                sweep_axis = run_sim.internal_sweep_axis(test_cfg, tb_text)

                tb_params_base = dict(defaults)
                tb_params_base["Vavdd"] = defaults["vdd"]
                tb_params_base["filename"] = test_name
                tb_params_base["N"] = "0"
                tb_params_base["models_dir"] = ctx.xyce_models_dir if simulator == "xyce" else ctx.models_dir
                tb_params_base["stdcell_dir"] = ctx.stdcell_dir
                if sweep_axis:
                    key, prefix = sweep_axis
                    values = [float(v) for v in test_cfg.get("conditions", {}).get(key, [])]
                    tb_params_base[f"{prefix}_min"] = min(values)
                    tb_params_base[f"{prefix}_max"] = max(values)
                tb_params_base.update(run_sim.fixed_tb_params(test_cfg, tb_text, sweep_axis))

                test_dir = sim_dir / test_name
                conditions_list = list(run_sim.condition_matrix(test_cfg, defaults, sweep_axis))
                if args.limit:
                    conditions_list = conditions_list[:args.limit]
                print(f"  {test_name}: {len(conditions_list)} condition(s)")

                for conditions in conditions_list:
                    label = run_sim.condition_label(conditions)
                    run_dir = test_dir / label
                    start = time.perf_counter()
                    outcome = runner(
                        container, test_name, tb_source, conditions, tb_params_base,
                        run_dir, f"{container_sim_dir}/{test_name}/{label}", container_rcfile, ctx,
                        sim_timeout=args.ngspice_timeout,
                    )
                    elapsed = time.perf_counter() - start
                    exit_code = outcome.get("ngspice_exit_code") or outcome.get("xyce_exit_code")
                    timings.append((test_name, label, elapsed, outcome["status"], exit_code))
                    flag = "" if outcome["status"] == "success" else f"  <-- {outcome['status'].upper()}"
                    print(f"    {label}: {elapsed:6.1f}s{flag}")
    finally:
        if not args.keep:
            shutil.rmtree(sim_dir, ignore_errors=True)
        elif sim_dir.exists():
            print(f"\nkept netlists/logs at {sim_dir}")

    if not timings:
        print("nothing ran")
        return

    print("\n--- slowest first ---")
    for test_name, label, elapsed, status, exit_code in sorted(timings, key=lambda t: -t[2]):
        risk = ""
        if exit_code == 124:
            risk = "  ** TIMED OUT (non-convergent / hung) **"
        elif elapsed >= _SLOW_FRACTION * args.ngspice_timeout:
            risk = "  ** approaching timeout **"
        print(f"{test_name:20s} {label:30s} {elapsed:7.1f}s  {status:8s}{risk}")

    failed = [t for t in timings if t[3] != "success"]
    if failed:
        print(f"\n{len(failed)}/{len(timings)} condition(s) failed or timed out -- see above")
        sys.exit(1)


if __name__ == "__main__":
    main()
