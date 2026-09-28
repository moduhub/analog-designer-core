#!/usr/bin/env python3
"""Re-check every already-registered variation of the open project's
block/topology against its tests' current definitions and re-run whichever
are stale -- the batch counterpart to analog_designer/sim/run_sim.py's
single-variation Update. Reuses run_variation()'s own per-variation
freshness check unchanged (see its `fresh`/`to_run` split): a variation
that's already fully fresh costs nothing here, run_variation() returns
before ever touching the container. No pre-filtering step is needed to
find "the stale ones" -- running the whole range through unconditionally
already only-does-work-where-needed.

Sequential by default, same as gen_variations.py -- raise the global
container.cpu_budget setting (analog_designer/core/settings.py, see the GUI's Docker
Settings dialog) above 1 to update several variations at once against one
shared container instead; see gen_variations._run_batch/run_variation's
shadow= param for how concurrent variations avoid racing on the
materialized schematic.

--from/--to narrow the batch to a contiguous slice of sim/variations.jsonl
(its append -- i.e. creation -- order), by variation name; omit either
(or both) to leave that end of the range unbounded. --name (repeatable) is
the alternative form: exactly the named variations, in whatever combination
-- what the GUI's Variations table drives via a Shift/Ctrl-click selection
(analog_designer/gui/app.py's _update_range), since a table sorted by some
other column has no relationship to creation order any more. Give one form
or the other, not both.

Usage: python -m analog_designer.sim.update_variations [--from NAME] [--to NAME] [--force] [--project-root PATH]
       python -m analog_designer.sim.update_variations --name NAME [--name NAME ...] [--force] [--project-root PATH]
"""
import argparse
import sys

from analog_designer.core import workspace
from analog_designer.sim.gen_variations import _run_batch, add_skip_on_fail_tolerance_args, skip_on_fail_batch_kwargs
from analog_designer.sim.run_sim import _read_jsonl, validate_skip_on_fail_profile, validate_skip_on_fail_tolerance


def select_variations(name_from, name_to):
    """Rows from sim/variations.jsonl for the current block/topology, in
    creation order, sliced to [name_from, name_to] inclusive (either bound
    omitted leaves that end of the range open). sys.exit()s on a bound
    that doesn't name an existing variation of this block/topology, or a
    --from that comes after --to -- both almost certainly a mistake, not
    something to silently work around."""
    rows = [
        r for r in _read_jsonl(workspace.PROJECT_ROOT / "sim" / "variations.jsonl")
        if r["block"] == workspace.BLOCK and r["topology"] == workspace.TOPOLOGY
    ]
    names = [r["name"] for r in rows]
    start, end = 0, len(rows)
    if name_from is not None:
        if name_from not in names:
            sys.exit(f"--from {name_from!r}: no such variation for {workspace.BLOCK}/{workspace.TOPOLOGY}")
        start = names.index(name_from)
    if name_to is not None:
        if name_to not in names:
            sys.exit(f"--to {name_to!r}: no such variation for {workspace.BLOCK}/{workspace.TOPOLOGY}")
        end = names.index(name_to) + 1
    if start >= end:
        sys.exit(f"--from {name_from!r} does not come before --to {name_to!r} in creation order")
    return rows[start:end]


def select_variations_by_name(names):
    """Rows from sim/variations.jsonl for the current block/topology, exactly
    the ones named in `names` -- the --name sibling of select_variations()'s
    --from/--to creation-order slice, for a caller that already knows exactly
    which variations it wants (e.g. the GUI's Variations table, driven by a
    Shift/Ctrl-click selection that may be in ANY order, including one a
    column sort produced with no relationship to creation order). Returned in
    variations.jsonl's own creation order regardless of `names`' own order,
    for a stable, deterministic batch/log sequence. sys.exit()s on any name
    that doesn't exist for this block/topology, same as select_variations()'s
    own --from/--to validation."""
    rows = [
        r for r in _read_jsonl(workspace.PROJECT_ROOT / "sim" / "variations.jsonl")
        if r["block"] == workspace.BLOCK and r["topology"] == workspace.TOPOLOGY
    ]
    by_name = {r["name"]: r for r in rows}
    missing = [n for n in names if n not in by_name]
    if missing:
        sys.exit(f"--name {missing[0]!r}: no such variation for {workspace.BLOCK}/{workspace.TOPOLOGY}")
    wanted = set(names)
    return [r for r in rows if r["name"] in wanted]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--from", dest="name_from", default=None,
        help="first variation (creation order) to include; omit to start from the very first",
    )
    parser.add_argument(
        "--to", dest="name_to", default=None,
        help="last variation (creation order, inclusive) to include; omit to run through the very last",
    )
    parser.add_argument(
        "--name", action="append", default=None,
        help="include exactly this variation (repeatable) -- alternative to --from/--to, any combination/order; "
             "give one form or the other, not both",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="re-run every test in range even if results.jsonl already has a fresh (matching definition_hash) result",
    )
    parser.add_argument(
        "--skip-on-fail", default=None, metavar="PROFILE",
        help="stop simulating a variation's remaining tests the moment they'd already disqualify PROFILE "
             "(a config.json blocks.<block>.profiles name) -- opt-in, off by default",
    )
    add_skip_on_fail_tolerance_args(parser, allow_discard=False)
    parser.add_argument("--project-root", default=None, help="project folder to operate on; defaults to the last-opened folder, else CWD")
    parser.add_argument("--block", default=None, help="block to operate on; defaults to the first declared in config.json")
    parser.add_argument("--topology", default=None, help="topology to operate on; defaults to the first declared for --block")
    args = parser.parse_args()

    workspace.open_folder(args.project_root, block=args.block, topology=args.topology)
    validate_skip_on_fail_profile(args.skip_on_fail)
    validate_skip_on_fail_tolerance(
        args.skip_on_fail, args.skip_on_fail_max_failures, args.discard_on_fail, args.checkpoint_size,
    )
    config = workspace.CONFIG
    defaults = config["defaults"]
    block_cfg = config["blocks"][workspace.BLOCK]["topologies"][workspace.TOPOLOGY]
    tests = config["tests"][workspace.BLOCK]

    if args.name and (args.name_from or args.name_to):
        sys.exit("--name cannot be combined with --from/--to -- use one form or the other")

    if args.name:
        selected = select_variations_by_name(args.name)
    else:
        selected = select_variations(args.name_from, args.name_to)
    if not selected:
        sys.exit(f"no variations registered yet for {workspace.BLOCK}/{workspace.TOPOLOGY} -- nothing to update")

    if args.name:
        print(f"updating {len(selected)} variation(s): {', '.join(r['name'] for r in selected)}")
    else:
        print(f"updating {len(selected)} variation(s): {selected[0]['name']} .. {selected[-1]['name']}")
    param_sets = [row["parameters"] for row in selected]
    any_error = _run_batch(param_sets, block_cfg, tests, defaults, args.force, origin=None, **skip_on_fail_batch_kwargs(args))

    sys.exit(1 if any_error else 0)


if __name__ == "__main__":
    main()
