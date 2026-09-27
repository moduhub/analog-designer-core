#!/usr/bin/env python3
"""Register/simulate one manually-specified parameter set for the open
project's block/topology: any parameter not given via --param falls back to
its config.json default. Not randomized for ordinary numeric parameters --
no --seed/--pct/count, exactly one variation, simulated through
analog_designer/sim/run_sim.py's existing per-variation pipeline
(materialize -> netlist -> simulate -> parse -> store).

A "block_ref" parameter (e.g. "top"'s own "X1_variation" -- see
analog_designer.sim.run_sim.materialize_sub_blocks) selects WHICH already-
registered variation of a sub-block composes this one, not a number:
--param NAME=VALUE still works for an exact variation name (or the reserved
value "defaults"), and --param-spec NAME=PROFILE additionally draws ONE
random variation of that sub-block whose primary_profile matches PROFILE
(PROFILE="any" for an unfiltered draw among every registered variation of
that sub-block) -- resolved once, right here, into a concrete name before
the variation is ever registered/materialized, so the CLI never depends on
a GUI having "rolled" a value beforehand.

--base NAME is purely a provenance label -- it does NOT seed the parameter
set (every parameter not given via --param still falls back to its
config.json default, not to NAME's own values); the caller is responsible
for passing NAME's own current parameters as --param overrides if that's
the intent (see analog_designer/gui/create_variation_dialog.py's "From
Parent" tab and App._vary_param's "exact" mode). Recorded as
origin={"kind": "manual", "base": NAME} instead of the bare
{"kind": "manual"} used when omitted, so pro's own design-space viewer's
parent-line visualization and the origin field in general can show where a
hand-edited variation actually came from.

Usage: python -m analog_designer.sim.manual_variation [--param NAME=VALUE ...] [--param-spec NAME=PROFILE ...] [--base NAME] [--force] [--seed SEED] [--project-root PATH]
"""
import argparse
import random
import sys

from analog_designer.core import workspace
from analog_designer.results import data
from analog_designer.sim.gen_variations import _clip, _render, _run_batch, parse_spice_value
from analog_designer.sim.run_sim import BLOCK_REF_DEFAULT, validate_skip_on_fail_profile, validate_skip_on_fail_tolerance


def cmd_basic(args, config):
    block_cfg = config["blocks"][workspace.BLOCK]["topologies"][workspace.TOPOLOGY]
    defaults = config["defaults"]
    tests = config["tests"][workspace.BLOCK]
    param_defs = block_cfg["parameters"]

    overrides = {}
    for item in args.param or []:
        if "=" not in item:
            sys.exit(f"--param must be NAME=VALUE, got: {item!r}")
        name, value = item.split("=", 1)
        if name not in param_defs:
            sys.exit(f"unknown parameter: {name!r}")
        overrides[name] = value

    rng = random.Random(args.seed)
    for item in args.param_spec or []:
        if "=" not in item:
            sys.exit(f"--param-spec must be NAME=PROFILE, got: {item!r}")
        name, profile = item.split("=", 1)
        pdef = param_defs.get(name)
        if pdef is None or pdef.get("type") != "block_ref":
            sys.exit(f"--param-spec {name!r}: not a declared block_ref parameter")
        if name in overrides:
            sys.exit(f"{name!r} given via both --param and --param-spec -- pick one")
        profile_name = None if profile == "any" else profile
        candidates = data.matching_variations(pdef["block"], pdef["topology"], profile_name)
        if not candidates:
            sys.exit(
                f"--param-spec {name!r}={profile!r}: no registered variations of "
                f"{pdef['block']}/{pdef['topology']} exist yet"
            )
        overrides[name] = rng.choice(candidates)

    params = {}
    for name, pdef in param_defs.items():
        if pdef.get("type") == "block_ref":
            # a variation name (or "defaults"), never a numeric quantity --
            # skip parse_spice_value/_clip/_render entirely, those assume a
            # SPICE-value string.
            params[name] = overrides.get(name, pdef.get("default", BLOCK_REF_DEFAULT))
            continue
        params[name] = _render(_clip(parse_spice_value(overrides.get(name, pdef["default"])), pdef), pdef)
    origin = {"kind": "manual", "base": args.base} if args.base else {"kind": "manual"}
    any_error = _run_batch(
        [params], block_cfg, tests, defaults, args.force, origin=origin, skip_on_fail_profile=args.skip_on_fail,
        skip_on_fail_max_failures=args.skip_on_fail_max_failures, discard_on_fail=args.discard_on_fail,
    )
    sys.exit(1 if any_error else 0)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--param", action="append", metavar="NAME=VALUE",
        help="override one parameter (repeatable); parameters not given fall back to their config.json default",
    )
    parser.add_argument(
        "--param-spec", action="append", metavar="NAME=PROFILE",
        help='for a "block_ref" parameter only (repeatable): draw one random already-registered '
             'variation of that sub-block matching PROFILE (or "any" for unfiltered) instead of '
             'giving an exact name via --param',
    )
    parser.add_argument(
        "--base", default=None, metavar="NAME",
        help="provenance only -- records origin.base=NAME, does not seed any parameter value "
             "(pass the base's own values via --param yourself if that's the intent)",
    )
    parser.add_argument("--force", action="store_true", help="re-run tests even if a fresh result already exists")
    parser.add_argument(
        "--skip-on-fail", default=None, metavar="PROFILE",
        help="stop simulating a variation's remaining tests the moment they'd already disqualify PROFILE "
             "(a config.json blocks.<block>.profiles name) -- opt-in, off by default",
    )
    parser.add_argument(
        "--skip-on-fail-max-failures", type=int, default=0, metavar="N",
        help="tolerate up to N already-violated constraints of --skip-on-fail's own PROFILE before actually "
             "stopping (default 0: any single violation stops it) -- requires --skip-on-fail",
    )
    parser.add_argument(
        "--discard-on-fail", action="store_true",
        help="when --skip-on-fail (beyond --skip-on-fail-max-failures) actually stops this variation, trim it "
             "entirely instead of leaving it registered with partial results -- requires --skip-on-fail",
    )
    parser.add_argument("--seed", type=int, default=None, help="random seed for --param-spec draws, for reproducibility")
    parser.add_argument("--project-root", default=None, help="project folder to operate on; defaults to the last-opened folder, else CWD")
    parser.add_argument("--block", default=None, help="block to operate on; defaults to the first declared in config.json")
    parser.add_argument("--topology", default=None, help="topology to operate on; defaults to the first declared for --block")
    args = parser.parse_args()

    workspace.open_folder(args.project_root, block=args.block, topology=args.topology)
    validate_skip_on_fail_profile(args.skip_on_fail)
    validate_skip_on_fail_tolerance(args.skip_on_fail, args.skip_on_fail_max_failures, args.discard_on_fail)
    cmd_basic(args, workspace.CONFIG)


if __name__ == "__main__":
    main()
