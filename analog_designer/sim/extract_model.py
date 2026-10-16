#!/usr/bin/env python3
"""Build a circuit model for one block/topology out of SEVERAL generator runs, instead of one.

Some devices cannot be modelled from a single simulation. A differential inductor is the worked
example: driving one arm, then the other, then the structure with one arm shorted, gives three
sweeps whose combination separates the two self-inductances from their mutual -- no single run
can, because a single run only ever sees their sum. The same shape appears well outside EM: a
varactor C-V model from a handful of SPICE bias points, a resistor model from a geometric sweep,
a capacitance model from a quasi-static solve per corner.

So this script knows nothing about inductors, meshes, FDTD or openEMS. It knows four optional
functions a generator module may expose, all phrased in the generator's own parameters:

    extraction_plan(params)   -> [{'name', 'params', 'required', 'note'}, ...]
        The setups a complete model needs. Each entry's 'params' is handed straight back to
        geometry_from_params(), so only the generator ever decides what a setup means.
    combine_extraction(params, runs) -> {'params': {...}, 'provenance': {...}}
        runs maps a setup name to the path of that setup's cached result. Raises, with what is
        missing, when it cannot build a model yet.
    installed_model(params) -> dict or None
    install_model(params, model, provenance) -> path

A generator without them is simply one whose model comes from a single run; nothing changes for
it. NOTHING here needs a config.json entry: the project already names the generator module there
(blocks.<block>.topologies.<topology>.generator), and everything device-specific lives in it.

This is deliberately NOT one of config.json's "tests". A test measures a model that already
exists; this produces the model. Keeping the two apart is what stops a project's config from
accumulating simulator-specific campaign bookkeeping.

Running it is opt-in and explicit, because a plan can be very expensive -- the reference six-setup
inductor campaign was 60+ hours of FDTD. With no flags it only REPORTS the plan and what is
already done. --run starts the missing setups, one at a time.

Usage:
  python -m analog_designer.sim.extract_model <variation> [--project-root PATH]
  python -m analog_designer.sim.extract_model <variation> --run [--only arm_a]
  python -m analog_designer.sim.extract_model <variation> --combine [--install]
"""
import argparse
import json
import os
import sys

from analog_designer.core import workspace
from analog_designer.sim.run_sim import (
    _load_generator_module, generator_cache_dir, generator_cache_key, run_one_openems,
)

#: Retired spelling of the per-setup key, still found in results cached before 2026-10-06 -- see
#: _migrate_legacy_cache_keys().
_LEGACY_SETUP_KEY = "variant"


def _load_variation(name):
    path = workspace.sim_root() / "variations.jsonl"
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row["name"] == name:
            return row
    sys.exit(f"no variation named {name!r} in {path}")


def _migrate_legacy_cache_keys(generator, cache_dir):
    """Re-keys cached results whose parameters still use a retired key spelling.

    A generator is free to rename its own parameters, but the cache is keyed by a hash OF those
    parameters, so a rename silently orphans every result already on disk -- and for a campaign
    measured in tens of hours, "silently orphans" means "re-runs". Rather than freezing generator
    vocabulary forever, this normalises old entries by asking the generator itself to re-derive
    the parameters (geometry_from_params accepts both spellings during a transition) and moving
    the file to the key that re-derivation produces. Idempotent, and a no-op for a cache that has
    never seen a rename."""
    if not cache_dir.is_dir():
        return 0
    moved = 0
    for path in sorted(cache_dir.glob("*.json")):
        try:
            doc = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        params = doc.get("geometry")
        if not isinstance(params, dict) or _LEGACY_SETUP_KEY not in params:
            continue
        try:
            renamed = generator.geometry_from_params(params)
        except Exception as exc:  # noqa: BLE001 -- a cache entry we cannot re-derive is left alone
            print(f"  (left {path.name} alone: {type(exc).__name__}: {exc})")
            continue
        new_key = generator_cache_key(renamed)
        if path.stem == new_key:
            continue
        doc["geometry"] = renamed
        (cache_dir / f"{new_key}.json").write_text(json.dumps(doc), encoding="utf-8")
        for sibling in cache_dir.glob(f"{path.stem}__*"):
            sibling.rename(cache_dir / sibling.name.replace(path.stem, new_key, 1))
        path.unlink()
        moved += 1
        print(f"  re-keyed cached result {path.stem} -> {new_key} "
              f"(parameters renamed since it was run)")
    return moved


def _plan_status(generator, params, cache_dir):
    """[(entry, cache_path or None), ...] for every setup in the plan."""
    out = []
    for entry in generator.extraction_plan(params):
        resolved = generator.geometry_from_params(entry["params"])
        cache_file = cache_dir / f"{generator_cache_key(resolved)}.json"
        out.append((entry, resolved, cache_file if cache_file.exists() else None))
    return out


def _print_status(status, params, generator):
    installed = None
    if hasattr(generator, "installed_model"):
        try:
            installed = generator.installed_model(params)
        except Exception as exc:  # noqa: BLE001
            print(f"warning: installed_model() raised {type(exc).__name__}: {exc}")
    done = sum(1 for _, _, cache in status if cache)
    required_missing = [e["name"] for e, _, cache in status if cache is None and e["required"]]

    print(f"\nextraction plan: {done}/{len(status)} setups already run")
    print(f"{'setup':<15} {'required':<9} {'status':<9} cached result")
    for entry, _resolved, cache in status:
        print(f"{entry['name']:<15} {'yes' if entry['required'] else 'no':<9} "
              f"{'done' if cache else 'MISSING':<9} {cache.name if cache else '--'}")
        if entry.get("note"):
            print(f"{'':<15} note: {entry['note']}")
    print()
    if installed is not None:
        print("a model is already installed for this device "
              "(the materialize flow is using it, not placeholders)")
    if required_missing:
        print(f"cannot build a model yet -- required setups still missing: {required_missing}")
    else:
        print("every required setup is present: --combine can build a model")
    return required_missing


def _run_missing(status, variation, block, topology, block_cfg, only, n_threads):
    run_root = workspace.sim_root() / variation / "_extraction"
    for entry, _resolved, cache in status:
        name = entry["name"]
        if cache is not None or (only and name != only):
            continue
        run_dir = run_root / name
        run_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n=== running setup {name!r} ({'required' if entry['required'] else 'optional'}) ===")
        if entry.get("note"):
            print(f"    note from the generator: {entry['note']}")
        result = run_one_openems(
            container=None, test_name=f"extract_{name}", tb_source=None, conditions={},
            tb_params_base=entry["params"], run_dir=run_dir,
            container_run_dir=workspace.exec_path(run_dir), container_rcfile=None, ctx=None,
            block_cfg=block_cfg, block=block, topology=topology, n_threads=n_threads,
        )
        if result["status"] != "success":
            print(f"setup {name!r} did not finish: {result.get('error', '')[-800:]}")
            return False
        print(f"setup {name!r} finished")
    return True


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("variation", help="a registered variation (see sim/variations.jsonl)")
    parser.add_argument("--project-root", default=None)
    parser.add_argument("--block", default=None)
    parser.add_argument("--topology", default=None)
    parser.add_argument("--run", action="store_true",
                        help="actually run the missing setups, one at a time (default: only report)")
    parser.add_argument("--only", default=None, metavar="SETUP",
                        help="with --run, run just this one setup")
    parser.add_argument("--combine", action="store_true",
                        help="combine the setups already run into model parameters and print them")
    parser.add_argument("--install", action="store_true",
                        help="with --combine, write the model where the materialize flow picks it "
                             "up. Opt-in: it changes what every later simulation of this device sees")
    args = parser.parse_args()

    workspace.open_folder(args.project_root, block=args.block, topology=args.topology)
    row = _load_variation(args.variation)
    block, topology = row["block"], row["topology"]
    block_cfg = workspace.CONFIG["blocks"][block]["topologies"][topology]
    generator = _load_generator_module(block_cfg)
    if generator is None or not hasattr(generator, "extraction_plan"):
        sys.exit(f"{block}/{topology} has no extraction_plan() -- its model comes from a single "
                 f"run, so there is no campaign to orchestrate. Use run_sim.py.")

    params = row["parameters"]
    cache_dir = generator_cache_dir(block, topology)
    _migrate_legacy_cache_keys(generator, cache_dir)
    status = _plan_status(generator, params, cache_dir)
    required_missing = _print_status(status, params, generator)

    if args.run:
        if workspace.execution_mode() == "host":
            sys.exit("--run needs docker execution mode: a campaign relies on the detached "
                     "container that only that mode provides.")
        # FDTD is memory-bandwidth bound: on the reference host more than 8 threads gave no speedup
        # and caused stalls, so the default is 8 (capped at cpu_count - 2), not cpu_count - 2.
        # Override with ANALOG_DESIGNER_OPENEMS_THREADS.
        n_threads = int(os.environ.get("ANALOG_DESIGNER_OPENEMS_THREADS")
                        or min(8, max(1, (os.cpu_count() or 2) - 2)))
        if not _run_missing(status, args.variation, block, topology, block_cfg,
                            args.only, n_threads):
            sys.exit(1)
        status = _plan_status(generator, params, cache_dir)
        required_missing = _print_status(status, params, generator)

    if args.combine:
        if required_missing:
            sys.exit(f"--combine needs the required setups first: {required_missing}")
        runs = {entry["name"]: str(cache) for entry, _r, cache in status if cache}
        built = generator.combine_extraction(params, runs)
        print("\nmodel parameters:")
        print(json.dumps(built["params"], indent=2))
        print("\nprovenance:")
        print(json.dumps(built["provenance"], indent=2, default=str))
        if args.install:
            dest = generator.install_model(params, built["params"], built["provenance"])
            print(f"\ninstalled -> {dest}")
            print("every later materialize of this device now uses it instead of placeholders.")
        else:
            print("\n(not installed -- pass --install to make the materialize flow use it)")


if __name__ == "__main__":
    main()
