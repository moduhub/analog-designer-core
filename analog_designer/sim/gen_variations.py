#!/usr/bin/env python3
"""Generate N random variations of the open project's block/topology --
each parameter independently sampled uniformly between its config.json
min/max, or within +/-PCT%% of its default if --spread is given -- and
simulate each one through analog_designer/sim/run_sim.py's existing per-variation
pipeline (materialize -> netlist -> simulate -> parse -> store).

Sequential by default (one variation fully simulated before the next
starts, same single-writer assumption analog_designer/sim/run_sim.py's own CLI makes) --
raise the global container.cpu_budget setting (analog_designer/core/settings.py, see
the GUI's Docker Settings dialog) above 1 to simulate several variations at
once against one shared container instead; see _run_batch/run_variation's
shadow= param for how concurrent variations avoid racing on the
materialized schematic.

Usage: python -m analog_designer.sim.gen_variations N [--spread PCT] [--force] [--seed SEED] [--project-root PATH]
"""
import argparse
import concurrent.futures
import random
import sys
import time
import traceback

from analog_designer.core import workspace
from analog_designer.sim.run_sim import (
    BLOCK_REF_DEFAULT, _read_jsonl, emit_progress_total, emit_progress_total_seconds,
    emit_progress_variation_done, ensure_variation_registered, historical_test_durations,
    load_results, managed_container, plan_seconds, plan_steps, run_variation,
    setup_container, validate_skip_on_fail_profile, variation_name,
)
from analog_designer.sim.spice_value import _match, format_spice_value, parse_spice_value


def _grid_value(pdef):
    """Quantization grid for a parameter, in base (unsuffixed) units -- None
    if it's unconstrained/continuous. "integer": true always snaps to whole
    numbers (grid=1, and _render() below drops the SI suffix entirely for
    it -- a device multiplier/finger count isn't a physical quantity with
    units); a continuous parameter may instead declare an explicit "grid"
    (e.g. a PDK's minimum W/L layout increment) to snap to. A parameter
    with neither is sampled/perturbed at full float precision, same as
    before this existed."""
    if pdef.get("integer"):
        return 1.0
    if "grid" in pdef:
        return parse_spice_value(pdef["grid"])
    return None


def _clip(value, pdef):
    """value clipped to a parameter's configured [min, max] and snapped to
    its grid (see _grid_value), if any -- shared by every variation-
    generating command (basic, Monte Carlo, and pro's own perturb/combine/
    directed-step commands) since they all need to keep sampled/perturbed/combined
    values both within config.json's declared range AND at whatever
    precision that parameter is physically meaningful at (a whole number of
    fingers, a manufacturable W/L increment, ...). Re-clips after rounding
    to grid since a boundary value can round just outside [min, max]."""
    lo = parse_spice_value(pdef["min"])
    hi = parse_spice_value(pdef["max"])
    value = min(max(value, lo), hi)
    grid = _grid_value(pdef)
    if grid is not None:
        value = min(max(round(value / grid) * grid, lo), hi)
    return value


def _render(value, pdef):
    """value rendered back as the string config.json/the schematic expect:
    a bare integer (no SI suffix) for "integer": true parameters, else the
    same unit suffix as the parameter's own default, for readability."""
    if pdef.get("integer"):
        return str(int(round(value)))
    _, default_suffix = _match(pdef["default"])
    return format_spice_value(value, default_suffix)


def random_params(param_defs, rng, spread_pct=None):
    """One parameter set: each parameter independently sampled uniform(min,
    max), then clipped/grid-snapped and rendered back via _clip()/_render()
    (same unit suffix as its config.json default -- min/max can use a
    different suffix, e.g. min='500n' next to default='1u', parse_spice_value
    handles that).

    spread_pct narrows the per-parameter sampling range to
    default +/- spread_pct% (intersected with [min, max]) instead of the
    full configured range -- still independent uniform sampling per
    parameter, just centered and narrower, for a "defined spread" Monte
    Carlo batch instead of "whole range"."""
    params = {}
    for name, pdef in param_defs.items():
        if pdef.get("type") == "block_ref":
            # Not a numeric quantity -- a block_ref parameter selects WHICH
            # already-registered variation of a sub_blocks instance composes
            # this one (see analog_designer.sim.run_sim.materialize_sub_blocks);
            # random_params() has no opinion on that (see
            # gen_variations.py's own hierarchical batch path, which handles
            # block_ref sampling on its own terms -- generating and
            # registering a fresh sub-block variation, not picking a number).
            # Passed through unchanged so this function stays safely callable
            # even for a hierarchical block's OWN parameters dict.
            params[name] = pdef.get("default", BLOCK_REF_DEFAULT)
            continue
        lo = parse_spice_value(pdef["min"])
        hi = parse_spice_value(pdef["max"])
        if spread_pct is not None:
            center = parse_spice_value(pdef["default"])
            lo = max(lo, center * (1 - spread_pct / 100))
            hi = min(hi, center * (1 + spread_pct / 100))
        params[name] = _render(_clip(rng.uniform(lo, hi), pdef), pdef)
    return params


def _run_batch(param_sets, block_cfg, tests, defaults, force, origin, skip_on_fail_profile=None):
    """Simulate each of param_sets through run_variation, reusing one
    on-demand container across the whole batch. Sequential when
    workspace.cpu_budget() is 1 (the only case for a single-item batch like
    manual_variation.py's) -- identical behavior to before this was
    parallelized, writing the shared, visible sch/<block>.sch. Above 1,
    dispatches up to that many variations at once against the one
    container, each materializing its own per-variation shadow schematic
    instead (see run_variation's shadow= param) so concurrent netlisting
    can't race on that shared file -- the actual per-job thread count each
    one spends once running is governed separately, by
    workspace.core_pool() (see run_sim.py's THREAD_POLICY), not by this
    max_workers. Shared by every variation-generating command that
    produces more than one param set at a time. skip_on_fail_profile is
    forwarded to run_variation() unchanged -- a profile name or None/off,
    see its own docstring.

    origin is normally one dict, broadcast to every item in the batch (every
    existing caller: manual_variation.py, update_variations.py, this
    module's own Monte Carlo, and pro's own variation-generating commands
    -- the whole batch genuinely shares one origin). Pass a list the same
    length as param_sets instead when each item's provenance differs -- e.g.
    pro's own crossover command, where every resulting variation has
    its own (parent_a, parent_b) pair."""
    max_workers = workspace.cpu_budget()
    shadow = max_workers > 1
    existing_names = {r["name"] for r in _read_jsonl(workspace.PROJECT_ROOT / "sim" / "variations.jsonl")}
    any_error = False

    # Progress-bar total for the WHOLE batch, computed up front (pure file
    # reads, no docker) -- one plan_steps() call per param set, reusing the
    # exact freshness/condition-count logic run_variation() itself uses, so
    # this can never drift from the real work each variation ends up doing.
    existing_results = load_results()
    emit_progress_total(sum(
        plan_steps(
            block_cfg, tests, defaults, existing_results,
            variation_name(workspace.BLOCK, workspace.TOPOLOGY, params), force,
        )
        for params in param_sets
    ))
    historical_durations = historical_test_durations(existing_results, workspace.BLOCK)
    emit_progress_total_seconds(sum(
        plan_seconds(
            block_cfg, tests, defaults, existing_results,
            variation_name(workspace.BLOCK, workspace.TOPOLOGY, params), force, historical_durations,
        )
        for params in param_sets
    ))

    def _run_one(i, params):
        name = variation_name(workspace.BLOCK, workspace.TOPOLOGY, params)
        item_origin = origin[i] if isinstance(origin, list) else origin
        # only tagged in parallel mode (shadow==True) -- sequential mode's
        # console output is otherwise untouched, and there's nothing to
        # attribute lines to when only one variation runs at a time.
        log_prefix = f"[{name}] " if shadow else ""
        if name in existing_names:
            print(f"\n{log_prefix}=== variation {i + 1}/{len(param_sets)}: {name} already exists, re-checking freshness ===")
        else:
            print(f"\n{log_prefix}=== variation {i + 1}/{len(param_sets)} ===")
        start_ts = time.monotonic()
        try:
            return run_variation(
                block_cfg, tests, defaults, params, force=force,
                container_ctx=container_ctx, origin=item_origin, shadow=shadow, log_prefix=log_prefix,
                skip_on_fail_profile=skip_on_fail_profile,
            )
        except Exception as exc:
            # One variation's unexpected failure (not a StaleParameterSchema
            # skip or an ordinary simulation error -- run_variation() already
            # handles both of those itself and never raises for them; this is
            # anything else, e.g. MissingCrossBlockMetric from a hierarchical
            # variation still pointed at the "defaults" sentinel for a
            # sub_blocks instance instead of a registered variation) must
            # NOT be allowed to propagate up through future.result() --
            # ThreadPoolExecutor(max_workers>1)'s own `with` block would tear
            # down every OTHER still-running/pending variation's future the
            # moment this one raises, killing the entire batch (and making
            # the GUI's progress bar/ETA look frozen, since no more @PROGRESS
            # lines would ever arrive) over a single bad variation. Reported
            # the same way an ordinary ERROR result already is (log line +
            # any_error=True), so the batch keeps going. The full traceback
            # goes to the log too (not just str(exc)) -- a bare exception
            # message alone (e.g. a bare "Invalid control character at: line
            # 1 column 3") gives no clue which call raised it or why, and
            # this path is exactly the "something truly unexpected happened"
            # case where that matters most.
            print(f"{log_prefix}{name}: ERROR (unexpected exception: {exc})")
            print(f"{log_prefix}{traceback.format_exc()}")
            emit_progress_variation_done(name, time.monotonic() - start_ts, True)
            return {"variation": name, "any_error": True}

    with managed_container() as container:
        container_ctx = setup_container(container)
        if max_workers <= 1:
            for i, params in enumerate(param_sets):
                outcome = _run_one(i, params)
                existing_names.add(outcome["variation"])
                any_error = any_error or outcome["any_error"]
        else:
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as pool:
                futures = [pool.submit(_run_one, i, params) for i, params in enumerate(param_sets)]
                for future in concurrent.futures.as_completed(futures):
                    any_error = any_error or future.result()["any_error"]

    return any_error


def generate_hierarchical_param_set(config, block_cfg, block, topology, rng, spread_pct, origin_kind):
    """One free-parameter set for a hierarchical block (e.g. "top"), PLUS
    -- for every "block_ref" parameter it declares -- a freshly sampled
    sub-block parameter set of its own, registered right here as its own
    independent variation (pure JSONL append via ensure_variation_registered,
    no docker touched yet) before this function returns, so the returned
    top-level params dict already references a REAL, stable variation name
    (never a placeholder that could still change) by the time its own
    variation_name() gets computed.

    Returns (top_params, sub_jobs) -- top_params is this block's own params
    dict (with every block_ref entry replaced by the sub-block variation
    name just registered for it); sub_jobs is [{"block_cfg", "tests",
    "block", "topology", "params", "origin"}, ...], one per freshly
    generated sub-block, for the caller to ALSO simulate standalone (see
    _run_hierarchical_batch()) -- "os sub blocos ... passam a constar como
    blocos isolados também", not just embedded inside the parent's own
    materialization."""
    top_params = random_params(block_cfg["parameters"], rng, spread_pct=spread_pct)
    sub_jobs = []
    for name, pdef in block_cfg["parameters"].items():
        if pdef.get("type") != "block_ref":
            continue
        sub_block, sub_topology = pdef["block"], pdef["topology"]
        sub_block_cfg = config["blocks"][sub_block]["topologies"][sub_topology]
        sub_params = random_params(sub_block_cfg["parameters"], rng)
        sub_name = variation_name(sub_block, sub_topology, sub_params)
        sub_origin = {"kind": "generate_hierarchical", "parent_block": block, "parent_topology": topology, "parent_kind": origin_kind}
        ensure_variation_registered(sub_name, sub_block, sub_topology, sub_params, origin=sub_origin)
        top_params[name] = sub_name
        sub_jobs.append({
            "block_cfg": sub_block_cfg, "tests": config["tests"][sub_block], "block": sub_block,
            "topology": sub_topology, "params": sub_params, "origin": sub_origin,
        })
    return top_params, sub_jobs


def _run_hierarchical_batch(jobs, defaults, force, skip_on_fail_profile=None):
    """Like _run_batch(), but for a heterogeneous list of jobs spanning
    MORE than one (block, topology) at once -- needed for a hierarchical
    block's own Monte Carlo generation (see generate_hierarchical_param_set()),
    where each top-level iteration also needs its own freshly generated
    cmos_vref/output_amp sub-block simulated standalone, alongside "top"
    itself, all sharing the one container this function opens. Each job
    carries its OWN {"block_cfg", "tests", "params", "block", "topology",
    "origin"} explicitly -- run_variation()'s own block=/topology=
    parameters (not workspace.BLOCK/TOPOLOGY) are what make this safe even
    under a parallel (cpu_budget>1) batch: two jobs for different blocks
    running on different threads never share/race on mutable global state,
    since neither one ever touches workspace.BLOCK/TOPOLOGY at all.

    Deliberately NOT a generalization of _run_batch() itself -- populations
    homogeneous in (block, topology) are the overwhelmingly common case,
    and _run_batch() is used everywhere already; a separate function keeps
    that one simple and unchanged instead of threading per-item
    block/topology through code that has never needed it before.

    skip_on_fail_profile is forwarded unchanged to every job's own
    run_variation() call -- each job's own block=job["block"] is what makes
    this safe even though jobs span different blocks (cmos_vref/output_amp
    under a "top" run, each with their own declared profiles): a profile
    name not declared for a given job's own block just resolves to an empty
    constraints dict there (see run_variation()'s own profile lookup),
    never disqualifying that job -- a deliberate, permissive default for a
    name that simply doesn't apply to that sub-block."""
    max_workers = workspace.cpu_budget()
    shadow = max_workers > 1
    existing_names = {r["name"] for r in _read_jsonl(workspace.PROJECT_ROOT / "sim" / "variations.jsonl")}
    any_error = False

    existing_results = load_results()
    emit_progress_total(sum(
        plan_steps(
            job["block_cfg"], job["tests"], defaults, existing_results,
            variation_name(job["block"], job["topology"], job["params"]), force,
        )
        for job in jobs
    ))
    # Recomputed per job (not cached per distinct block) -- a hierarchical
    # batch's job list is small (a handful of entries per `count` iteration)
    # and both functions are pure/cheap, not worth the extra bookkeeping of
    # a per-block cache for an input this size.
    emit_progress_total_seconds(sum(
        plan_seconds(
            job["block_cfg"], job["tests"], defaults, existing_results,
            variation_name(job["block"], job["topology"], job["params"]), force,
            historical_test_durations(existing_results, job["block"]),
        )
        for job in jobs
    ))

    def _run_one(i, job):
        name = variation_name(job["block"], job["topology"], job["params"])
        log_prefix = f"[{job['block']}:{name}] " if shadow else f"[{job['block']}] "
        if name in existing_names:
            print(f"\n{log_prefix}=== job {i + 1}/{len(jobs)}: {name} already exists, re-checking freshness ===")
        else:
            print(f"\n{log_prefix}=== job {i + 1}/{len(jobs)} ===")
        start_ts = time.monotonic()
        try:
            return run_variation(
                job["block_cfg"], job["tests"], defaults, job["params"], force=force,
                container_ctx=container_ctx, origin=job["origin"], shadow=shadow, log_prefix=log_prefix,
                block=job["block"], topology=job["topology"], skip_on_fail_profile=skip_on_fail_profile,
            )
        except Exception as exc:
            # See _run_batch's own identical try/except for why this can't
            # be allowed to propagate -- one job's unexpected failure must
            # not tear down every other concurrently-running/pending job's
            # future in the same batch. Full traceback logged too, same
            # reasoning as _run_batch's own copy of this comment.
            print(f"{log_prefix}{name}: ERROR (unexpected exception: {exc})")
            print(f"{log_prefix}{traceback.format_exc()}")
            emit_progress_variation_done(name, time.monotonic() - start_ts, True)
            return {"variation": name, "any_error": True}

    with managed_container() as container:
        container_ctx = setup_container(container)
        if max_workers <= 1:
            for i, job in enumerate(jobs):
                outcome = _run_one(i, job)
                existing_names.add(outcome["variation"])
                any_error = any_error or outcome["any_error"]
        else:
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as pool:
                futures = [pool.submit(_run_one, i, job) for i, job in enumerate(jobs)]
                for future in concurrent.futures.as_completed(futures):
                    any_error = any_error or future.result()["any_error"]

    return any_error


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("count", type=int, help="number of random variations to generate and simulate")
    parser.add_argument(
        "--spread", type=float, default=None,
        help="sample each parameter within +/-PCT%% of its config.json default instead of its full [min, max] range",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="re-run a generated variation's tests even if (by hash coincidence) a fresh result already exists",
    )
    parser.add_argument(
        "--skip-on-fail", default=None, metavar="PROFILE",
        help="stop simulating a variation's remaining tests the moment they'd already disqualify PROFILE "
             "(a config.json blocks.<block>.profiles name) -- opt-in, off by default",
    )
    parser.add_argument("--seed", type=int, default=None, help="random seed, for reproducible batches")
    parser.add_argument("--project-root", default=None, help="project folder to operate on; defaults to the last-opened folder, else CWD")
    parser.add_argument("--block", default=None, help="block to operate on; defaults to the first declared in config.json")
    parser.add_argument("--topology", default=None, help="topology to operate on; defaults to the first declared for --block")
    args = parser.parse_args()

    workspace.open_folder(args.project_root, block=args.block, topology=args.topology)
    validate_skip_on_fail_profile(args.skip_on_fail)
    config = workspace.CONFIG
    defaults = config["defaults"]
    block_cfg = config["blocks"][workspace.BLOCK]["topologies"][workspace.TOPOLOGY]
    tests = config["tests"][workspace.BLOCK]
    rng = random.Random(args.seed)

    if block_cfg.get("sub_blocks"):
        # A hierarchical block (e.g. "top"): each of the `count` iterations
        # gets its OWN freshly generated cmos_vref/output_amp pair, not a
        # reused/pooled one -- "os sub blocos poderiam ser gerados na hora"
        # -- registered as independent variations and simulated standalone
        # right alongside "top" itself (same shared container), not just
        # used internally. Sampled/registered sequentially up front (same
        # "rng isn't thread-safe, --seed must stay reproducible regardless
        # of cpu_budget()" reasoning _run_batch's own caller below already
        # follows) before _run_hierarchical_batch may then simulate the
        # whole flattened job list concurrently.
        jobs = []
        for _ in range(args.count):
            top_params, sub_jobs = generate_hierarchical_param_set(
                config, block_cfg, workspace.BLOCK, workspace.TOPOLOGY, rng, args.spread, "random",
            )
            jobs.extend(sub_jobs)
            jobs.append({
                "block_cfg": block_cfg, "tests": tests, "block": workspace.BLOCK,
                "topology": workspace.TOPOLOGY, "params": top_params, "origin": {"kind": "random"},
            })
        any_error = _run_hierarchical_batch(jobs, defaults, args.force, skip_on_fail_profile=args.skip_on_fail)
    else:
        # sampled sequentially (rng.uniform() isn't thread-safe) before handing
        # off to _run_batch, which may then simulate them concurrently --
        # keeps --seed reproducible regardless of workspace.cpu_budget().
        param_sets = [random_params(block_cfg["parameters"], rng, spread_pct=args.spread) for _ in range(args.count)]
        any_error = _run_batch(param_sets, block_cfg, tests, defaults, args.force, origin={"kind": "random"}, skip_on_fail_profile=args.skip_on_fail)

    sys.exit(1 if any_error else 0)


if __name__ == "__main__":
    main()
