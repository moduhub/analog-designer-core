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
    BLOCK_REF_DEFAULT, _read_jsonl, emit_progress_plan, emit_progress_trimmed,
    emit_progress_variation_done, ensure_variation_registered, historical_test_durations,
    load_results, managed_container, plan_progress, run_variation,
    setup_container, trim_variation, validate_skip_on_fail_profile, validate_skip_on_fail_tolerance,
    variation_name,
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


#: checkpoint_size's own "auto" default (see _run_batch/_run_hierarchical_batch)
#: is this multiplied by workspace.cpu_budget() -- e.g. 3 * 10 cores = checkpoint
#: every 30 items. Chosen (not just 1x) so a checkpoint's own drain-the-chunk
#: barrier doesn't starve faster cores waiting on one chunk's slowest straggler
#: too often -- purely a starting point, override with --checkpoint-size.
DEFAULT_CHECKPOINT_MULTIPLIER = 3


def _trim_all(names):
    """trim_variation() each of `names` -> (trimmed, failed). A trim that
    still fails after run_sim._replace_with_retry()'s own retries (a file
    held open by another process) is reported and handed back for the next
    checkpoint to retry, instead of raising out of the batch and killing
    every variation still queued behind it. trim_variation() is
    idempotent, so a half-done trim (variations.jsonl rewritten,
    results.jsonl not) simply finishes on the retry."""
    trimmed, failed = [], []
    for name in names:
        try:
            trim_variation(name)
            trimmed.append(name)
        except OSError as exc:
            print(f"checkpoint: could not discard {name} yet ({exc}) -- will retry at the next checkpoint")
            failed.append(name)
    return trimmed, failed


def _run_batch(param_sets, block_cfg, tests, defaults, force, origin, skip_on_fail_profile=None,
                skip_on_fail_max_failures=0, discard_on_fail=False, checkpoint_size=None):
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
    produces more than one param set at a time. skip_on_fail_profile/
    skip_on_fail_max_failures are forwarded to run_variation() unchanged --
    see its own docstring.

    discard_on_fail is ALSO forwarded to every run_variation() call, but
    the actual trim_variation() a discarded variation needs happens HERE,
    never inside run_variation() itself (see its own discard_on_fail
    docstring for exactly why: trim_variation()'s whole-file rewrite would
    race a SIBLING variation's concurrent append on the very same two
    files, in whichever of them wins that race losing whatever the other
    one just wrote) -- the real invariant trim_variation() needs is just
    "no other thread is CURRENTLY appending", which holds at every chunk
    boundary below, not only once at the very end of the whole batch.

    checkpoint_size splits param_sets into chunks of that many items,
    each dispatched to its own short-lived ThreadPoolExecutor (parallel
    mode) or, in sequential mode, just a checkpoint boundary every that
    many items of the one long-running loop -- either way, this batch's
    OWN discovered-since-the-last-checkpoint discards are trim_variation()'d
    right after each chunk's own executor has fully drained (or, in
    sequential mode, immediately -- there's only ever the one thread
    there), instead of letting a whole afternoon-long low-hit-rate Monte
    Carlo search's worth of garbage variations sit on disk until the
    entire batch finishes. This costs some parallelism -- a chunk's
    fastest worker still has to wait for that SAME chunk's slowest
    straggler before the next chunk's work can even start dispatching --
    a real, accepted tradeoff (disk usage over raw core utilization) for a
    search whose useful-candidate hit rate is low enough that most
    generated variations are getting discarded anyway.

    None (the default) means "no periodic checkpointing at all" UNLESS
    discard_on_fail is also True, in which case it defaults to
    DEFAULT_CHECKPOINT_MULTIPLIER * workspace.cpu_budget() -- discard_on_fail
    is the whole reason this exists, so turning it on already protects disk
    usage with no extra flag required; pass an explicit checkpoint_size to
    override that auto value (still only takes effect alongside
    discard_on_fail -- nothing to periodically trim otherwise, so this is
    silently a no-op, exactly like a plain skip_on_fail_profile with no
    discard was already before this feature existed). A single time-based
    ("every N minutes") alternative was considered instead but dropped: it
    needs its own timer thread coordinating a mid-chunk drain, whereas a
    sample count needs nothing extra -- ThreadPoolExecutor's own `with`
    block already blocks until every submitted item in a chunk is done, a
    checkpoint barrier "for free". The tradeoff: a batch whose remaining
    items in the CURRENT chunk are individually very slow (few of them,
    each expensive) won't checkpoint any sooner than that chunk's own
    completion, regardless of how much wall-clock time has passed --
    a non-issue for the many-cheap-samples Monte Carlo search this was
    built for, but worth knowing if a future caller's own workload looks
    different (few, expensive items).

    origin is normally one dict, broadcast to every item in the batch (every
    existing caller: manual_variation.py, update_variations.py, this
    module's own Monte Carlo, and pro's own variation-generating commands
    -- the whole batch genuinely shares one origin). Pass a list the same
    length as param_sets instead when each item's provenance differs -- e.g.
    pro's own crossover command, where every resulting variation has
    its own (parent_a, parent_b) pair."""
    max_workers = workspace.cpu_budget()
    shadow = max_workers > 1
    if discard_on_fail and checkpoint_size is None:
        checkpoint_size = max(1, DEFAULT_CHECKPOINT_MULTIPLIER * max_workers)
    elif not discard_on_fail:
        checkpoint_size = None
    existing_names = {r["name"] for r in _read_jsonl(workspace.PROJECT_ROOT / "sim" / "variations.jsonl")}
    any_error = False

    # Progress plan for the WHOLE batch, computed up front (pure file
    # reads, no docker) -- one plan_progress() call per param set, reusing
    # the exact freshness/condition-count logic run_variation() itself uses,
    # so this can never drift from the real work each variation ends up doing.
    existing_results = load_results()
    historical_durations = historical_test_durations(existing_results, workspace.BLOCK)
    for params in param_sets:
        name = variation_name(workspace.BLOCK, workspace.TOPOLOGY, params)
        emit_progress_plan(name, plan_progress(
            block_cfg, tests, defaults, existing_results, name, force, historical_durations,
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
                skip_on_fail_max_failures=skip_on_fail_max_failures, discard_on_fail=discard_on_fail,
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
            return {"variation": name, "any_error": True, "discard": False}

    to_discard = []
    discarded_total = 0

    def _checkpoint():
        nonlocal to_discard, discarded_total
        if not to_discard:
            return
        to_discard, failed = _trim_all(to_discard)
        if not to_discard:
            to_discard = failed
            return
        discarded_total += len(to_discard)
        print(f"checkpoint: discarded {len(to_discard)} variation(s) that disqualified "
              f"{skip_on_fail_profile!r} beyond {skip_on_fail_max_failures} allowed failure(s) "
              f"({discarded_total} total so far): {', '.join(to_discard)}")
        emit_progress_trimmed(to_discard)
        to_discard = failed

    with managed_container() as container:
        container_ctx = setup_container(container)
        if max_workers <= 1:
            for i, params in enumerate(param_sets):
                outcome = _run_one(i, params)
                existing_names.add(outcome["variation"])
                any_error = any_error or outcome["any_error"]
                if outcome.get("discard"):
                    to_discard.append(outcome["variation"])
                if checkpoint_size and (i + 1) % checkpoint_size == 0:
                    _checkpoint()
        else:
            # No checkpointing at all (checkpoint_size is None): one single
            # dispatch across the whole batch, unchanged from before this
            # feature existed. Checkpointing: one short-lived
            # ThreadPoolExecutor per chunk instead -- its own `with` block
            # already blocks until every item IN THAT CHUNK is done, which
            # is exactly the drain barrier a checkpoint needs, for free.
            chunks = (
                [param_sets[start:start + checkpoint_size] for start in range(0, len(param_sets), checkpoint_size)]
                if checkpoint_size else [param_sets]
            )
            offset = 0
            for chunk in chunks:
                with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as pool:
                    futures = [pool.submit(_run_one, offset + j, params) for j, params in enumerate(chunk)]
                    for future in concurrent.futures.as_completed(futures):
                        outcome = future.result()
                        any_error = any_error or outcome["any_error"]
                        if outcome.get("discard"):
                            to_discard.append(outcome["variation"])
                offset += len(chunk)
                # Safe here: this chunk's own executor has fully drained
                # (its `with` block just exited), so no thread from it could
                # still be appending to variations.jsonl/results.jsonl --
                # the next chunk hasn't started dispatching yet either.
                _checkpoint()

    # Final flush for whatever's left since the last checkpoint boundary (or
    # everything, if checkpoint_size was never set) -- same safety as every
    # earlier _checkpoint() call, now that the whole batch/container is done.
    _checkpoint()

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


def _iterations(jobs):
    """Group a flat hierarchical `jobs` list (see main()'s own hierarchical
    branch) back into per-SAMPLE groups -- each group is one top-level
    composed attempt's own freshly-generated sub-block jobs, followed by
    its own top-level job (the exact shape main() builds: jobs.extend(sub_jobs)
    then jobs.append(top_job)). A top-level job is any job carrying
    "discard_with" (see generate_hierarchical_param_set()); every job
    since the previous one (or the start) belongs to that same sample.
    Needed so _run_hierarchical_batch()'s own checkpoint_size groups by
    SAMPLE (one whole composed attempt -- what "a cada N amostras" means
    to whoever launched the search), not by raw job count, which would
    otherwise risk splitting one sample's own sub-jobs from its own
    top-level job across two different checkpoint chunks for no reason."""
    group = []
    for job in jobs:
        group.append(job)
        if "discard_with" in job:
            yield group
            group = []
    if group:
        # No top-level job ever closed this final group (shouldn't happen
        # given how main() builds `jobs` today, but don't silently drop
        # jobs if some future caller's own shape ever differs) -- surface
        # whatever's left as its own last, incomplete group rather than
        # dropping it.
        yield group


def _run_hierarchical_batch(jobs, defaults, force, skip_on_fail_profile=None,
                             skip_on_fail_max_failures=0, discard_on_fail=False, checkpoint_size=None):
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

    skip_on_fail_profile/skip_on_fail_max_failures are forwarded unchanged
    to every job's own run_variation() call -- each job's own
    block=job["block"] is what makes this safe even though jobs span
    different blocks (cmos_vref/output_amp under a "top" run, each with
    their own declared profiles): a profile name not declared for a given
    job's own block just resolves to an empty constraints dict there (see
    run_variation()'s own profile lookup), never disqualifying that job --
    a deliberate, permissive default for a name that simply doesn't apply
    to that sub-block.

    discard_on_fail is NEVER forwarded to a sub-block job's own
    run_variation() call -- only a job carrying its own "discard_with" key
    (a top-level "top" job, added by main() right where it already knows
    that iteration's freshly-generated sub-job names, see
    generate_hierarchical_param_set()) can trigger a discard, and it does
    so for BOTH itself and every name in its own "discard_with" list --
    those sub-block variations were sampled JUST for this one composed
    attempt (never reused across iterations), so a discarded "top" always
    takes its own ingredient sub-blocks down with it, whatever their own
    individual pass/fail happened to be. A plain sub-job (no "discard_with"
    key) is therefore only ever discarded as part of its parent's own
    outcome, never on its own -- same deferred trim_variation() timing as
    _run_batch(), for the identical concurrency reason (see that
    function's own docstring: safe the moment a chunk's own worker(s) have
    all finished, not only once at the very end of the whole batch).

    checkpoint_size -- see _run_batch()'s own docstring for the general
    idea (periodic trim_variation() of this batch's own discards instead
    of leaving them all until the very end, auto-enabled at
    DEFAULT_CHECKPOINT_MULTIPLIER * workspace.cpu_budget() the moment
    discard_on_fail is True, unless overridden). The one difference here:
    it counts in SAMPLES (one whole composed "top" attempt, sub-jobs
    included), not raw jobs -- see _iterations()'s own docstring for why a
    plain job-count chunk boundary would risk splitting one sample's own
    sub-jobs from its own top-level job across two different checkpoints
    for no reason."""
    max_workers = workspace.cpu_budget()
    shadow = max_workers > 1
    if discard_on_fail and checkpoint_size is None:
        checkpoint_size = max(1, DEFAULT_CHECKPOINT_MULTIPLIER * max_workers)
    elif not discard_on_fail:
        checkpoint_size = None
    existing_names = {r["name"] for r in _read_jsonl(workspace.PROJECT_ROOT / "sim" / "variations.jsonl")}
    any_error = False

    existing_results = load_results()
    # Recomputed per job (not cached per distinct block) -- a hierarchical
    # batch's job list is small (a handful of entries per `count` iteration)
    # and both functions are pure/cheap, not worth the extra bookkeeping of
    # a per-block cache for an input this size.
    for job in jobs:
        name = variation_name(job["block"], job["topology"], job["params"])
        emit_progress_plan(name, plan_progress(
            job["block_cfg"], job["tests"], defaults, existing_results, name, force,
            historical_test_durations(existing_results, job["block"]),
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
                skip_on_fail_max_failures=skip_on_fail_max_failures,
                discard_on_fail=discard_on_fail and "discard_with" in job,
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
            return {"variation": name, "any_error": True, "discard": False}

    def _collect_discard(job, outcome):
        # Gated on "discard_with" in job (a top-level job only), NOT just
        # outcome.get("discard") alone -- belt-and-suspenders alongside
        # _run_one()'s own discard_on_fail=discard_on_fail and "discard_with"
        # in job: a sub-job's own run_variation() call is always forced
        # discard_on_fail=False, so its own outcome should never actually
        # carry "discard": True in practice, but this function must not
        # ALSO independently trim a sub-job on its own if it somehow did --
        # only a top-level job's own discard takes its "discard_with" list
        # down with it (see this function's own caller's docstring).
        if "discard_with" not in job or not outcome.get("discard"):
            return []
        return [outcome["variation"], *job["discard_with"]]

    to_discard = []
    discarded_total = 0

    def _checkpoint():
        nonlocal to_discard, discarded_total
        if not to_discard:
            return
        to_discard, failed = _trim_all(to_discard)
        if not to_discard:
            to_discard = failed
            return
        discarded_total += len(to_discard)
        print(f"checkpoint: discarded {len(to_discard)} variation(s) (composed job(s) that disqualified "
              f"{skip_on_fail_profile!r} plus their own freshly-generated sub-blocks, {discarded_total} "
              f"total so far): {', '.join(to_discard)}")
        emit_progress_trimmed(to_discard)
        to_discard = failed

    # Both branches below chunk the SAME way: groups of `checkpoint_size`
    # consecutive samples (_iterations(jobs) -- each group is one composed
    # attempt's own sub-jobs + its own top-level job), flattened back into
    # a plain job list per chunk. checkpoint_size=None (discard_on_fail
    # off) collapses to exactly one chunk holding everything -- unchanged
    # from before this feature existed.
    groups = list(_iterations(jobs)) if checkpoint_size else [jobs]
    chunks = (
        [groups[start:start + checkpoint_size] for start in range(0, len(groups), checkpoint_size)]
        if checkpoint_size else [groups]
    )

    with managed_container() as container:
        container_ctx = setup_container(container)
        offset = 0
        for chunk_groups in chunks:
            chunk = [job for group in chunk_groups for job in group]
            if max_workers <= 1:
                for j, job in enumerate(chunk):
                    outcome = _run_one(offset + j, job)
                    existing_names.add(outcome["variation"])
                    any_error = any_error or outcome["any_error"]
                    to_discard.extend(_collect_discard(job, outcome))
            else:
                with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as pool:
                    futures = {pool.submit(_run_one, offset + j, job): job for j, job in enumerate(chunk)}
                    for future in concurrent.futures.as_completed(futures):
                        outcome = future.result()
                        any_error = any_error or outcome["any_error"]
                        to_discard.extend(_collect_discard(futures[future], outcome))
            offset += len(chunk)
            # Safe here: every job in this chunk has finished (sequential
            # mode never had concurrency to begin with; parallel mode's
            # executor `with` block just exited, fully drained) -- see
            # _run_batch()'s own identical checkpoint comment. Every chunk
            # (there's always at least one, even with checkpoint_size=None)
            # ends with this same call, so nothing is ever left over for a
            # separate final flush after the loop, unlike _run_batch()'s own
            # sequential branch (which only checkpoints every Nth item).
            _checkpoint()

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
    parser.add_argument(
        "--skip-on-fail-max-failures", type=int, default=0, metavar="N",
        help="tolerate up to N already-violated constraints of --skip-on-fail's own PROFILE before actually "
             "stopping (default 0: any single violation stops it) -- requires --skip-on-fail",
    )
    parser.add_argument(
        "--discard-on-fail", action="store_true",
        help="when --skip-on-fail (beyond --skip-on-fail-max-failures) actually stops a variation, trim it "
             "entirely (and, for a hierarchical block, its own freshly-generated sub-block variations too) "
             "instead of leaving it registered with partial results -- requires --skip-on-fail",
    )
    parser.add_argument(
        "--checkpoint-size", type=int, default=None, metavar="N",
        help="with --discard-on-fail, trim discarded variations every N samples instead of waiting for the "
             f"whole batch to finish -- keeps disk usage bounded on a long, low-hit-rate search. Auto-sizes to "
             f"{DEFAULT_CHECKPOINT_MULTIPLIER} * the container.cpu_budget setting when --discard-on-fail is on "
             "and this is left unset; pass an explicit N to override. Requires --discard-on-fail",
    )
    parser.add_argument("--seed", type=int, default=None, help="random seed, for reproducible batches")
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
                # This top-level job's own freshly-generated sub-block
                # variations (just registered above, single-use for this
                # one composed attempt) -- present ONLY on a top-level job,
                # never a sub-job itself, so _run_hierarchical_batch() can
                # tell the two apart (see its own discard_on_fail docstring).
                "discard_with": [
                    variation_name(job["block"], job["topology"], job["params"]) for job in sub_jobs
                ],
            })
        any_error = _run_hierarchical_batch(
            jobs, defaults, args.force, skip_on_fail_profile=args.skip_on_fail,
            skip_on_fail_max_failures=args.skip_on_fail_max_failures, discard_on_fail=args.discard_on_fail,
            checkpoint_size=args.checkpoint_size,
        )
    else:
        # sampled sequentially (rng.uniform() isn't thread-safe) before handing
        # off to _run_batch, which may then simulate them concurrently --
        # keeps --seed reproducible regardless of workspace.cpu_budget().
        param_sets = [random_params(block_cfg["parameters"], rng, spread_pct=args.spread) for _ in range(args.count)]
        any_error = _run_batch(
            param_sets, block_cfg, tests, defaults, args.force, origin={"kind": "random"},
            skip_on_fail_profile=args.skip_on_fail,
            skip_on_fail_max_failures=args.skip_on_fail_max_failures, discard_on_fail=args.discard_on_fail,
            checkpoint_size=args.checkpoint_size,
        )

    sys.exit(1 if any_error else 0)


if __name__ == "__main__":
    main()
