#!/usr/bin/env python3
"""Materialize + (re-)simulate one block/topology variation through xschem +
ngspice inside the EDA docker container: the variation named on the command
line if given -- re-checking its existing tests against their current
definitions and re-running whichever are stale, same as any other run -- or
the config.json default-parameter variation if omitted (this script's only
mode before named-variation support was added). Every test declared under
config.json["tests"][<block>] runs the first time a variation is
registered. Operates on whichever project folder workspace.open_folder()
resolved (see analog_designer/core/workspace.py) -- defaults to the first block/topology
declared in that folder's config.json unless overridden.

Three separate logs, different granularity, don't confuse them:
  sim/variations.jsonl        - identity registry. One line per unique
      (block, topology, parameters) tuple, written once, never duplicated.
  sim/<variation>/runs.jsonl  - raw execution history for that variation.
      One line per (test, condition) simulation attempt. Expected to grow
      every time you run something, including reruns. Kept per-variation
      (not global) so parallel execution across variations -- the plan for
      large parameter sweeps -- never contends writing to the same file.
  sim/results.jsonl           - final answer table. One line per
      (variation, test, metric), tagged with a definition_hash so a stale
      result (test/schematic/parser changed since it was computed) is
      detectable without needing git.

One block/topology per session (whichever workspace.open_folder() resolved).
No sweeps, no CLI-selectable topology yet (that's the "genparams" phase) --
but, within that one topology, any already-registered parameter set can be
targeted by name, not just the config.json defaults.
"""
import argparse
import ast
import collections
import concurrent.futures
import contextlib
import datetime
import hashlib
import importlib.util
import itertools
import json
import math
import operator
import os
import re
import shlex
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

from analog_designer.core import executor as executors
from analog_designer.core import console
from analog_designer.core import console
from analog_designer.core import workspace
from analog_designer.results import fom
from analog_designer.sim import log_diagnostics
from analog_designer.sim import raw_peaks
from analog_designer.sim import soa_check
from analog_designer.sim import spice_devices
from analog_designer.sim.spice_value import _match, format_spice_value, parse_spice_value

print = console.atomic_print  # worker threads share stdout, see core/console.py

print = console.atomic_print  # worker threads share stdout, see core/console.py

PARAM_TOKEN_RE = re.compile(r"'([A-Za-z_][A-Za-z0-9_]*)'")

# matplotlib's mathtext/pyparsing internals aren't thread-safe -- when
# workspace.cpu_budget() > 1, gen_variations.py (and pro's own batch
# variation-generating scripts) run several variations at once via
# ThreadPoolExecutor, each landing in
# run_test() below and calling its own parser's evaluate() (every parser
# in a project may plot). Concurrent calls corrupt pyparsing's shared
# parser state and raise a spurious ParseException from deep inside
# matplotlib (e.g. "Expected end of text, found '$'" while laying out a
# log-axis tick label) that has nothing to do with the plot's actual data.
# Serializing every evaluate() call behind this lock costs nothing in the
# common cpu_budget==1 case and fixes it in the parallel one.
_PLOT_LOCK = threading.Lock()

# HISTORY: materialize_sub_blocks() used to always write a hierarchical
# block's sub-block schematics (e.g. "top"'s X1/X2 -> sch/cmos_vref.sch,
# sch/output_amp.sch) to that SAME shared, global location, regardless of
# whether the CALLING variation materialized itself into an isolated
# shadow/ location -- shadow=True only isolated the block's OWN top-level
# schematic (see materialize_variation_shadow), never its sub_blocks. Under
# workspace.cpu_budget() > 1 (gen_variations.py's ThreadPoolExecutor), two
# different "top" variations running concurrently -- each with its own
# X1_variation/X2_variation choice -- would race to overwrite those same
# shared files while the OTHER one was still netlisting/simulating against
# them, corrupting whichever variation lost the race (seen in practice as a
# non-deterministic "unknown subckt: cmos_vref" ngspice error that cleared
# up on a later, non-racing retry).
#
# FIX: materialize_variation_shadow() now forwards its own per-variation
# shadow sch/ dir through resolve_materialization_params() into
# materialize_sub_blocks() too (see both docstrings), so a shadow=True
# hierarchical variation's sub_blocks land in that SAME isolated location
# as its own top-level .sch -- no shared mutable file left to race on, for
# that path. This lock (below) is kept ONLY as a safety net for
# shadow=False (a lone, non-batch run) -- see run_variation()'s own
# materialize_lock construction for exactly when it's still taken. Costs
# nothing for the common case (a block with no sub_blocks never touches
# this lock at all).
_HIERARCHICAL_MATERIALIZE_LOCK = threading.Lock()

# _HIERARCHICAL_MATERIALIZE_LOCK only protects against another THREAD of
# THIS SAME Python process -- it does nothing against a SEPARATE
# run_sim.py/manual_variation.py process started concurrently (e.g. two
# manual CLI invocations, or a CLI run overlapping a GUI-spawned one),
# since each process gets its own, independent Lock object. Confirmed in
# practice (back when this guarded EVERY hierarchical variation, shadow or
# not -- see the HISTORY note above): a second, unrelated process's
# cmos_vref sub-block variation got written to the shared sch/cmos_vref.sch
# mid-simulation of a "top" variation's own 144-condition trim_range test,
# silently baking that OTHER variation's startup_cap_mult into ONE
# condition's netlist -- not caught by check_unresolved() or any
# definition_hash staleness check, since the netlist was syntactically
# valid, just materialized against the wrong sub-block content for that one
# condition (every other test of the same variation, netlisted outside the
# race window, showed the correct value). Now that shadow=True variations
# never touch the shared path at all, this specific failure mode can only
# still happen between two shadow=False (lone) runs -- rarer, but the same
# plain lockfile closes that gap too, without adding a new dependency or an
# fcntl/msvcrt platform split (atomic exclusive-create is enough, see
# _hierarchical_materialize_lock() below).
_HIERARCHICAL_MATERIALIZE_LOCKFILE_NAME = ".hierarchical_materialize.lock"
# Generous relative to any hierarchical variation's own longest observed
# materialize-through-simulate window (a 144-condition test has taken up to
# ~20 minutes in practice) -- long enough that no legitimate holder is ever
# mistaken for abandoned, short enough that a holder killed without
# cleanup (e.g. a task-manager kill mid-run) doesn't deadlock every future
# hierarchical run indefinitely.
_HIERARCHICAL_MATERIALIZE_STALE_SECONDS = 3600


@contextlib.contextmanager
def _hierarchical_materialize_lock():
    """Cross-PROCESS mutual exclusion for materialize_sub_blocks()'s shared
    sch/<sub-block>.sch scratch files -- see _HIERARCHICAL_MATERIALIZE_LOCK's
    own comment above for why this window must be exclusive, and why that
    in-process threading.Lock alone isn't enough. Layered UNDER the
    in-process lock (cheap, and still correct/necessary for same-process
    threads) so two threads of this same process never even race to
    create/poll the lockfile."""
    with _HIERARCHICAL_MATERIALIZE_LOCK:
        lock_path = workspace.PROJECT_ROOT / _HIERARCHICAL_MATERIALIZE_LOCKFILE_NAME
        _acquire_file_lock(lock_path)
        try:
            yield
        finally:
            lock_path.unlink(missing_ok=True)


def _acquire_file_lock(lock_path, poll_seconds=0.5):
    """Blocks until `lock_path` can be created exclusively (O_EXCL is
    atomic even across processes/platforms, unlike a check-then-write) --
    an abandoned lockfile (holder died without reaching the `finally`
    above) older than _HIERARCHICAL_MATERIALIZE_STALE_SECONDS is cleared
    rather than left to deadlock every future hierarchical run forever."""
    while True:
        try:
            fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(fd, str(os.getpid()).encode())
            os.close(fd)
            return
        except FileExistsError:
            try:
                age = time.time() - lock_path.stat().st_mtime
            except FileNotFoundError:
                continue  # released between our failed create and this stat -- retry immediately
            if age > _HIERARCHICAL_MATERIALIZE_STALE_SECONDS:
                lock_path.unlink(missing_ok=True)
                continue
            time.sleep(poll_seconds)

# Maps a config.json conditions[] key to the tb parameter name prefix a
# testbench uses for sweeping it internally in one ngspice run (e.g.
# `dc TEMP 'temp_min' 'temp_max' 5`) instead of it being a separate
# per-run condition. 'temperature' predates this table and kept its
# historical 'temp' prefix; any other key defaults to itself as the
# prefix (see internal_sweep_axis()).
INTERNAL_SWEEP_AXES = {"temperature": "temp"}


def internal_sweep_axis(test_cfg, tb_text):
    """(conditions_key, tb_param_prefix) for the axis this testbench
    sweeps internally via its own ngspice .dc directive -- one run then
    covers the whole conditions[] list for that key, so it's excluded
    from the outer per-run condition grid. Detected by an actual
    '<prefix>_min'/'<prefix>_max' parameter pair in the schematic's
    stimuli code, not just by the key's presence in conditions{}, so a
    conditions entry that ISN'T swept internally (e.g. a fixed vdd value)
    still gets treated as a plain fixed default. None if this testbench
    doesn't sweep anything internally."""
    for key in test_cfg.get("conditions", {}):
        prefix = INTERNAL_SWEEP_AXES.get(key, key)
        if f"'{prefix}_min'" in tb_text and f"'{prefix}_max'" in tb_text:
            return key, prefix
    return None


# conditions{} keys already covered elsewhere (global defaults, the
# internal-sweep-axis machinery, or -- "typical" -- reserved metadata naming
# the nominal value of each axis, not itself a testbench parameter), so
# fixed_tb_params() never re-derives them.
_NON_FIXED_CONDITION_KEYS = {"corner", "temperature", "vdd", "Cload", "Rload", "typical"}


def fixed_tb_params(test_cfg, tb_text, sweep_axis):
    """Fixed (non-swept) testbench parameters pulled straight from this
    test's own conditions{} -- e.g. a PSRR testbench's single-point
    'frequency'. Only keys whose exact '<key>' token appears in the
    testbench text are pulled, and only when conditions[key] holds exactly
    one value -- more than one means it's meant to vary, which is now
    condition_matrix()'s own outer-grid job (see there) for any key other
    than the internal-sweep-axis one, so it's skipped here rather than
    treated as an error."""
    skip = set(_NON_FIXED_CONDITION_KEYS)
    if sweep_axis:
        skip.add(sweep_axis[0])
    params = {}
    for key, values in test_cfg.get("conditions", {}).items():
        if key in skip or f"'{key}'" not in tb_text or len(values) > 1:
            continue
        params[key] = values[0]
    return params


_DESKTOP_PROCESS_PATTERN = (
    "xfce4|dbus-daemon|at-spi|gvfs|Thunar|tumblerd|ssh-agent|gpg-agent|pulseaudio|wrapper-2\\.0"
)


def _kill_unused_desktop_session(container_id):
    """The image's own ENTRYPOINT (/home/moduhub/start.sh) ignores the
    "sleep infinity" this module passes it entirely -- confirmed by reading
    the script: it has no `"$@"`/`exec "$@"` anywhere, it just always runs
    `vncserver $DISPLAY ...` and keeps itself alive with its own `tail -f
    /dev/null`. vncserver's own convention then runs ~/.vnc/xstartup, which
    launches a FULL xfce4 desktop session (xfce4-session, dbus, gvfs,
    Thunar, the panel, screensaver, pulseaudio, tumbler -- 20+ background
    processes) on top of the bare Xvnc server. None of that is reachable or
    useful for an unattended batch job -- confirmed via `docker exec ... ps
    aux` mid-job that nothing in it is ever touched -- so it's pure
    startup-time and steady-state CPU/memory overhead, paid by every single
    ephemeral container this module creates.

    Xvnc itself CANNOT be killed here -- run_test()'s own netlist_cmd
    hardcodes `export DISPLAY=:1` before every xschem invocation (xschem is
    Tk-based and needs a real, even if headless/off-screen, X connection
    even in its `-x -q` batch-netlist mode), so the container needs Xvnc
    alive for the whole job. Only the desktop SHELL layered on top of it
    (_DESKTOP_PROCESS_PATTERN, built from the exact process names observed
    in a live container) gets killed, via a plain pattern-matched `pkill`
    -9 -- Xvnc/vncserver/start.sh/tail are deliberately not matched by that
    pattern. Best-effort: a `docker exec` into a container that's still
    finishing its own startup can occasionally fail/no-op if xfce4-session
    hasn't forked yet, which is fine -- there's nothing to kill yet in that
    case, and the marginal CPU cost of a few stray processes for the first
    fraction of a second is negligible next to running the whole desktop
    for the job's entire duration."""
    subprocess.run(
        ["docker", "exec", container_id, "bash", "-lc", f"pkill -9 -f '{_DESKTOP_PROCESS_PATTERN}'"],
        capture_output=True, text=True,
    )


_NGSPICE_SPINIT_PATH = "/usr/local/share/ngspice/share/ngspice/scripts/spinit"


def _cap_ngspice_threads(container_id):
    """OMP_NUM_THREADS=1 (see managed_container()'s own docstring) turned
    out NOT to be enough -- confirmed via `docker exec ... ps -eLf` mid-run
    that ngspice was still spawning 8 OS threads, each independently
    burning up to ~100% CPU on their own core, EVEN with that env var set.
    Root cause: the image's own ngspice ships a global init script
    (_NGSPICE_SPINIT_PATH, loaded unconditionally on every single
    invocation, confirmed by reading it) containing a bare `set
    num_threads=8` -- ngspice's own "set" command calls the OpenMP runtime
    API (omp_set_num_threads()) directly, which OVERRIDES whatever
    OMP_NUM_THREADS initialized the process with, by OpenMP's own design
    (an explicit API call always wins over the env var that only sets the
    initial default). A personal ~/.spiceinit with `set num_threads=1` was
    tried first and did NOT override it (confirmed via the same live
    thread-count check) -- so the fix has to edit the global script
    in-place instead, which DOES verifiably cap it to 1 thread. Requires
    `docker exec -u root` (the container's own moduhub user has no write
    access to this path) -- only touches this ONE ephemeral container's
    own filesystem layer, never the underlying image, so nothing else
    running from that image (a manually-started VNC session, say) is
    affected.

    This is now a SAFETY FLOOR, not the real thread control: run_one_ngspice()
    writes its own `set num_threads=<N>` into each run's own per-run-dir
    .spiceinit (read AFTER this global spinit script, so it overrides the
    "1" set here with whatever workspace.core_pool() actually granted that
    run) -- see THREAD_POLICY. This patch stays anyway, capping the
    fallback for any ngspice invocation that -- for whatever reason -- ever
    runs without going through that per-run mechanism, to 1 rather than an
    uncontrolled 8."""
    subprocess.run(
        [
            "docker", "exec", "-u", "root", container_id, "sed", "-i",
            "s/set num_threads=8/set num_threads=1/", _NGSPICE_SPINIT_PATH,
        ],
        capture_output=True, text=True,
    )


@contextlib.contextmanager
def managed_executor():
    """Where this job's simulators run, for the length of one `with` block
    (one script invocation): a HostExecutor in host mode (see
    workspace.execution_mode() and analog_designer/core/executor.py) -- the
    tools on this machine, with the PDK from workspace.host_pdk() -- or, in
    docker mode, a fresh container, see managed_container()."""
    if workspace.execution_mode() == "host":
        pdk_root, pdk = workspace.host_pdk()
        env = {"OMP_NUM_THREADS": "1"}
        if pdk_root:
            env["PDK_ROOT"] = pdk_root
        if pdk:
            env["PDK"] = pdk
        executors.kill_host_jobs_on_terminate()
        yield executors.HostExecutor(env)
        return
    with managed_container() as container:
        yield container


@contextlib.contextmanager
def managed_container():
    """Container lifetime = one `with` block, matching one script
    invocation (a CLI run or one GUI-triggered job) -- never one simulation,
    and never left running forever. Always starts a fresh container of its
    own (-d --rm, so it's cleaned up the moment it's stopped) rather than
    reusing whatever else might already be running from the same image --
    a container the user started by hand for their own manual work (e.g.
    poking at xschem directly) is left completely alone, since sharing it
    would risk this script's writes (materialized schematic, xschemrc,
    sim/*.jsonl) racing against whatever unrelated thing is happening in
    it. Mounts this project's folder exactly where
    workspace.container_project_root() expects it, and stops it again when
    this block exits -- success or not.

    OMP_NUM_THREADS=1: the container's ngspice is linked against libgomp
    (confirmed via ldd) and defaults to spawning one OpenMP thread per core
    it sees (nproc inside the container, unrestricted -- no --cpus given
    here) when this isn't set. That's fine for a single simulation, but
    every SIMULATOR_RUNNERS entry already gets its OWN parallelism from
    workspace.cpu_budget() (gen_variations._run_batch's own
    ThreadPoolExecutor, N docker_exec calls into this ONE shared
    container at once, now further fanned out per-condition too -- see
    run_test()) -- stacking OpenMP's per-process threading on top of that
    oversubscribes badly (concurrent jobs x nproc threads each competing
    for nproc cores), which is what actually saturated every core on a
    28-core machine, not just a single slow simulation. This container-wide
    env var is only this mechanism's SAFETY FLOOR now, not the real control
    -- every actual docker_exec that runs a simulator prefixes its own
    command with `export OMP_NUM_THREADS=<N>` for whatever N
    workspace.core_pool().reserve() granted THAT run (see THREAD_POLICY),
    so the per-run value is what really governs threading; this flat "1"
    only still protects any exec that -- for whatever reason -- bypasses
    that per-run override.

    Also kills the unattended, unreachable xfce4 desktop session the
    image's own entrypoint always launches on top of Xvnc -- see
    _kill_unused_desktop_session()'s own docstring for why that's real,
    separate overhead from the OpenMP issue above, and why Xvnc itself has
    to stay. And caps ngspice's OWN internal thread count -- see
    _cap_ngspice_threads()'s own docstring for why OMP_NUM_THREADS=1 above
    turned out not to be sufficient on its own."""
    image = workspace.container_image()
    docker_args = [
        "docker", "run", "-d", "--rm",
        "-e", "OMP_NUM_THREADS=1",
        "-v", f"{workspace.PROJECT_ROOT}:{workspace.container_project_root()}",
    ]
    docker_args += ["-w", workspace.container_project_root(), image, "sleep", "infinity"]
    result = subprocess.run(docker_args, capture_output=True, text=True)
    if result.returncode != 0:
        sys.exit(f"could not start a container from image {image!r}:\n{result.stderr}")
    container_id = result.stdout.strip()
    _kill_unused_desktop_session(container_id)
    _cap_ngspice_threads(container_id)
    emit_progress_container(container_id)
    try:
        yield executors.DockerExecutor(container_id)
    finally:
        subprocess.run(["docker", "stop", container_id], capture_output=True, text=True)


def docker_exec(container, script, timeout=120):
    """Runs one bash script wherever this job's simulators run -- `container`
    is the executor managed_executor() yielded (a bare container id string,
    from older callers, still means that docker container). Name kept for
    compatibility: it is the one entry point for both execution modes."""
    if not isinstance(container, str):
        return container.run(script, timeout=timeout)
    try:
        return subprocess.run(
            ["docker", "exec", container, "bash", "-lc", script],
            capture_output=True, text=True, timeout=timeout,
        )
    except subprocess.TimeoutExpired as exc:
        # A stuck command inside the container (e.g. a non-converging
        # ngspice run) must not raise past here -- every caller already
        # treats a non-zero returncode as an ordinary {"status": "error"}
        # result for just that one (test, condition); letting TimeoutExpired
        # propagate instead would blow through run_variation()'s
        # ThreadPoolExecutor (gen_variations._run_batch) and tear down the
        # ONE shared container out from under every other variation
        # currently running in that same parallel batch. 124 is the
        # conventional shell "command timed out" exit code -- synthesizing
        # a normal CompletedProcess keeps every caller's contract unchanged.
        return subprocess.CompletedProcess(
            args=exc.cmd, returncode=124,
            stdout=exc.stdout or "",
            stderr=(exc.stderr or "") + f"\n[docker exec timed out after {timeout}s]",
        )


def get_pdk_dir(container, subpath):
    result = docker_exec(container, f'echo "$PDK_ROOT/$PDK/{subpath}"')
    path = result.stdout.strip()
    if result.returncode != 0 or not path:
        sys.exit(f"Could not resolve PDK path {subpath}:\n{result.stderr}")
    return path


def get_pdk_name(container):
    """The container's own $PDK (e.g. "ihp-sg13g2", "ihp-sg13cmos5l") --
    baked into the image at build time (see eda-env's Dockerfile), so this
    is the single source of truth for which PDK a project's container.image
    (config.json) selected. Used to derive PDK-specific naming (Xyce plugin
    filename, stdcell dir) instead of hardcoding one PDK's names."""
    result = docker_exec(container, 'echo "$PDK"')
    pdk_name = result.stdout.strip()
    if result.returncode != 0 or not pdk_name:
        sys.exit(f"Could not resolve $PDK from container:\n{result.stderr}")
    return pdk_name


def project_xschemrc():
    """The xschemrc every netlist run uses: the project's own (gitignored,
    copied from the PDK on first use), or -- when the project tree must not
    be written (workspace.read_only_project()) and has none yet -- a copy
    under workspace.sim_root()."""
    rc_path = workspace.PROJECT_ROOT / "xschemrc"
    if rc_path.exists() or not workspace.read_only_project():
        return rc_path
    return workspace.sim_root() / "xschemrc"


def ensure_xschemrc(container):
    rc_path = project_xschemrc()
    if rc_path.exists():
        return
    result = docker_exec(container, 'cat "$PDK_ROOT/$PDK/libs.tech/xschem/xschemrc"')
    if result.returncode != 0 or not result.stdout.strip():
        sys.exit(f"Could not fetch default xschemrc from the PDK:\n{result.stderr}")
    rc_path.parent.mkdir(parents=True, exist_ok=True)
    rc_path.write_text(result.stdout)
    print(f"created {rc_path}")


def substitute_params(text, params):
    for name, value in params.items():
        text = text.replace(f"'{name}'", str(value))
    return text


class StaleParameterSchema(Exception):
    """A variation's stored `parameters` dict predates a derived_parameters
    width group declared in the current params/<block>/<topology>.json (it's
    missing that group's `base` free parameter) -- raised by
    resolve_derived_params() instead of letting a raw KeyError propagate.
    Caught in run_variation() so a batch caller (analog_designer.sim.update_variations,
    the GUI's "Update Range") can skip just this one stale-schema variation
    and keep going, instead of the whole batch dying on the first
    pre-migration name it happens to hit."""


def resolve_derived_params(block_cfg, params):
    """params (free parameters only -- exactly what every sampler/perturber/
    stepper produces, and what variation_name()/ensure_variation_registered()
    hash and store) expanded with every derived_parameters.width_groups
    entry's computed width (e.g. m6_width = m6m7_width_base * m6_factor),
    evaluated in Python BEFORE substitute_params() ever sees the dict --
    ngspice/the materialized schematic must only ever see a plain numeric
    literal, never a formula, exactly like every other parameter. Derived
    values are a pure function of already-hashed free parameters, so they're
    deliberately NOT added to `params` before variation_name()/
    ensure_variation_registered() run -- only at the two materialization call
    sites (run_variation's shadow=False branch, materialize_variation_shadow)
    -- otherwise variation identity would depend on redundant, recomputable
    data instead of just the real degrees of freedom."""
    derived = dict(params)
    for group in block_cfg.get("derived_parameters", {}).get("width_groups", []):
        base_name = group["base"]
        if base_name not in params:
            raise StaleParameterSchema(
                f"{group['id']}: missing {base_name!r} -- this variation predates "
                f"the width_base+factor migration for this group and can no longer "
                f"be re-materialized (its stored widths don't fit an integer ratio)"
            )
        base_value = parse_spice_value(params[base_name])
        _, unit_suffix = _match(params[base_name])
        for derived_name, member in group["members"].items():
            factor_name = member["factor"]
            if factor_name not in params:
                raise StaleParameterSchema(
                    f"{group['id']}: missing {factor_name!r} -- this variation predates "
                    f"the width_base+factor migration for this group and can no longer "
                    f"be re-materialized (its stored widths don't fit an integer ratio)"
                )
            factor = int(round(parse_spice_value(params[factor_name])))
            derived[derived_name] = format_spice_value(base_value * factor, unit_suffix)
    return derived


class MissingCrossBlockMetric(Exception):
    """An import_metrics entry (see resolve_import_metrics()) needs a
    sub-block metric that isn't in sim/results.jsonl yet -- either that
    sub-block variation's own test has never been run, or the referencing
    parameter's sub_blocks instance is still pointed at the BLOCK_REF_DEFAULT
    ("defaults") sentinel instead of a registered variation, which has no
    sim/<name>/ of its own to have a result in."""


def lookup_cross_block_metric(from_variation, test_name, metric, stat="typical"):
    """The already-persisted (sim/results.jsonl) {typical,min,max} value for
    one (variation, test, metric) row -- what resolve_import_metrics() reads
    as another block's already-simulated ground truth, instead of just
    another one of its parameters (see resolve_import_params() for that
    simpler, parameter-only sibling). `stat` picks which of that row's
    typical/min/max columns to read (an import_metrics entry's own "stat"
    field, defaulting to "typical" -- a single design-point reading, not a
    range across the whole sweep). Returns the row's raw numeric value, in
    whatever unit that test's own outputs[].unit is -- any unit handling is
    the importing formula's own concern (see resolve_formulas()), not this
    function's."""
    matches = [
        row for row in load_results()
        if row["variation"] == from_variation and row["test"] == test_name and row["metric"] == metric
    ]
    if not matches:
        raise MissingCrossBlockMetric(
            f"no result for variation {from_variation!r}, test {test_name!r}, metric {metric!r} "
            f"-- run that test for that variation before materializing a variation that depends on it"
        )
    return matches[-1][stat]


#: {import_metrics name: value} given by hand (the standalone runner's
#: --import-metric), used instead of looking the metric up in
#: sim/results.jsonl -- the only way to materialize a hierarchical block in a
#: fresh checkout that has no stored sub-block results yet.
IMPORT_METRIC_OVERRIDES = {}


def resolve_import_metrics(block_cfg, params, sub_block_variation_names):
    """derived_parameters.import_metrics: pure fetch of an already-registered
    sub_blocks instance variation's own stored test result (sim/results.jsonl)
    into a local name -- no math, see resolve_formulas() for that (e.g.
    "top"'s own rbot_nominal formula multiplies/divides an imported metric
    by other already-resolved names directly, instead of this function doing
    any ratio/scaling itself). `entry["from"]` names a sub_blocks instance
    (params/top/default.json), `entry["test"]`/`entry["metric"]` name which
    stored result row to read (see lookup_cross_block_metric()), `stat`
    defaults to "typical". Requires that sub-block variation to already have
    a result for the named test -- this function never triggers a nested
    simulation, it only reads what's already in sim/results.jsonl.

    Replaces the old "scale_to_target" mechanism (and its "self" sentinel,
    needed only because that mechanism couldn't otherwise reach this block's
    own free parameters -- a plain formulas expr already can, no import
    required) -- a formula referencing both an imported metric and this
    block's own free parameters/other derived names does directly what
    "invert": true/false used to do implicitly, just as ordinary,
    unambiguous arithmetic in the formula's own `expr` string."""
    resolved = dict(params)
    for name, entry in block_cfg.get("derived_parameters", {}).get("import_metrics", {}).items():
        if name in IMPORT_METRIC_OVERRIDES:
            resolved[name] = IMPORT_METRIC_OVERRIDES[name]
            continue
        from_instance = entry["from"]
        from_variation = sub_block_variation_names[from_instance]
        if from_variation == BLOCK_REF_DEFAULT:
            raise MissingCrossBlockMetric(
                f"{name}: import_metrics needs a registered {from_instance}_variation, not "
                f"the {BLOCK_REF_DEFAULT!r} sentinel -- register that sub-block's parameters "
                f"as a real variation first (e.g. via manual_variation.py)"
            )
        measured = lookup_cross_block_metric(
            from_variation, entry["test"], entry["metric"], stat=entry.get("stat", "typical"),
        )
        if measured <= 0:
            raise MissingCrossBlockMetric(
                f"{name}: measured value of {entry['metric']!r} for variation "
                f"{from_variation!r} is not positive ({measured!r}) -- check probe polarity"
            )
        resolved[name] = str(measured)
    return resolved


def resolve_import_params(block_cfg, params, sub_block_resolved):
    """derived_parameters.import_params: pure fetch of an already-resolved
    sub_blocks instance's own parameter value into a local name -- no math,
    see resolve_formulas() for that (e.g. "top"'s own pbias_length formula
    is just the imported name passed through unchanged; anything needing
    scaling multiplies/divides the imported name by another already-resolved
    name directly in its own formula's `expr`). `entry["from"]` names a
    sub_blocks instance (params/top/default.json), `entry["source"]` names
    that sub-block's own already-resolved parameter (see
    materialize_sub_blocks())."""
    resolved = dict(params)
    for name, entry in block_cfg.get("derived_parameters", {}).get("import_params", {}).items():
        resolved[name] = sub_block_resolved[entry["from"]][entry["source"]]
    return resolved


#: Sentinel value for a "block_ref" parameter (see materialize_sub_blocks())
#: meaning "that sub-block's own config.json defaults, no registered
#: variation required" -- the same thing an absent/None choice used to mean
#: back when the sub-block choice was a static config.json field instead of
#: a per-variation parameter.
BLOCK_REF_DEFAULT = "defaults"


def materialize_sub_blocks(config, sub_blocks, params, sch_dir=None):
    """For every {instance: {block, topology}} in a parent block's own
    `sub_blocks` (e.g. "top"'s X1 -> cmos_vref, x2 -> output_amp): resolve
    that sub-block's own free parameters -- its block/topology's config.json
    defaults if the PARENT's own params[f"{instance}_variation"] is missing
    or BLOCK_REF_DEFAULT, or a specific already-registered variation's
    stored params otherwise (see _variation_params()) -- expand them with
    resolve_derived_params() (that sub-block's own within-block
    derived_parameters, exactly like a standalone run of it would), and
    materialize <sch_dir>/<block>.sch in place.

    sch_dir defaults to the shared project-root sch/ -- the SAME scratch
    location a standalone run of that block already writes to, and the SAME
    location xschem's own hierarchy resolution reads from when netlisting
    the parent schematic's sub-circuit instance in the non-shadow case. Pass
    the parent's own per-variation shadow sch/ dir instead (see
    materialize_variation_shadow) when the PARENT was itself materialized
    into isolation -- see run_variation()'s own comment on
    _HIERARCHICAL_MATERIALIZE_LOCK for why writing here unconditionally to
    the shared location, regardless of the caller's own isolation, used to
    be exactly the race this function created.

    `params` is the PARENT's (e.g. "top"'s) own free parameters -- the
    sub-block CHOICE lives there now (a "block_ref"-typed parameter, e.g.
    "X1_variation"), not in config.json's `sub_blocks` itself anymore
    (that's purely structural: which schematic instance maps to which
    block, fixed by the .sch hierarchy -- never varies per variation).

    Returns {instance: resolved_params}, so the parent block's own
    resolve_import_params() can pull a named source value (e.g.
    cmos_vref's own "m3_width") out of the sub-block it actually belongs
    to."""
    default_sch_dir = workspace.PROJECT_ROOT / "sch"
    sch_dir = sch_dir or default_sch_dir
    resolved_by_instance = resolve_sub_block_params(config, sub_blocks, params)
    for instance, ref in sub_blocks.items():
        resolved = resolved_by_instance[instance]
        sub_block_cfg = config["blocks"][ref["block"]]["topologies"][ref["topology"]]
        materialized_name = f"{ref['block']}.sch"
        topology_sch = workspace.PROJECT_ROOT / "sch" / sub_block_cfg["schematic"]
        materialized = substitute_params(topology_sch.read_text(encoding="utf-8"), resolved)
        check_unresolved(materialized, materialized_name)
        sch_dir.mkdir(parents=True, exist_ok=True)
        (sch_dir / materialized_name).write_text(materialized, encoding="utf-8")
        if sch_dir != default_sch_dir:
            # A shadow sch_dir starts empty -- unlike the shared project sch/,
            # it doesn't already have this sub-block's own checked-in .sym
            # sitting next to where we just wrote its materialized .sch, and
            # xschem pairs them by matching basename in the same directory
            # (same convention materialize_variation_shadow() already relies
            # on for the PARENT block's own .sym -- see its docstring).
            sym_name = f"{ref['block']}.sym"
            shutil.copyfile(default_sch_dir / sym_name, sch_dir / sym_name)
    return resolved_by_instance


def resolve_sub_block_params(config, sub_blocks, params):
    """Pure (no file I/O) half of materialize_sub_blocks(): resolves each
    sub_blocks instance's own free+within-block-derived parameters, without
    writing sch/<block>.sch -- shared by materialize_sub_blocks() (which
    does the writing afterward, for real simulation) and
    resolve_display_params() (the GUI's read-only preview, which must never
    have that side effect since it can run on every variation selection,
    including while a real simulation is materializing a DIFFERENT
    variation concurrently)."""
    resolved_by_instance = {}
    for instance, ref in sub_blocks.items():
        sub_block_cfg = config["blocks"][ref["block"]]["topologies"][ref["topology"]]
        chosen = params.get(f"{instance}_variation")
        if chosen and chosen != BLOCK_REF_DEFAULT:
            sub_params = _variation_params(chosen)
        else:
            sub_params = {n: pdef["default"] for n, pdef in sub_block_cfg["parameters"].items()}
        resolved_by_instance[instance] = resolve_formulas(
            sub_block_cfg,
            resolve_generator_params(sub_block_cfg, resolve_derived_params(sub_block_cfg, sub_params)),
        )
    return resolved_by_instance


#: ast node types resolve_formulas()'s evaluator allows -- deliberately just
#: enough arithmetic (+, -, *, /, **, unary +/-, parentheses, numeric
#: literals, bare names) to combine a handful of already-resolved parameters
#: into one derived number. Not a general eval(): derived_parameters.formulas
#: entries are developer-authored JSON, but there's no reason to allow
#: arbitrary Python (attribute access, calls, comprehensions, ...) just to
#: compute a resistor length from a few other parameters.
_FORMULA_BINOPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
}
_FORMULA_UNARYOPS = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}
#: One-argument functions a formula may call. ceil/floor tolerate 1e-9 of
#: float noise so a ratio that is mathematically an integer never rounds
#: the wrong way (e.g. 20u/10u evaluating to 2.0000000000000004 -> ceil 3).
_FORMULA_FUNCS = {
    "ceil": lambda x: math.ceil(x - 1e-9),
    "floor": lambda x: math.floor(x + 1e-9),
}


def formula_names(expr):
    """Every bare name a derived_parameters.formulas `expr` string
    references (e.g. "rfeedback_total", "rbot_nominal") -- used both by
    resolve_formulas() (to know which of `params` to look up) and by
    check_params.py's own derived_errors() (to check each one resolves to a
    real declared/derived name, and to count it as real usage of a free
    parameter that's otherwise only ever consumed here, never as a literal
    schematic token)."""
    tree = ast.parse(expr, mode="eval")
    called = {node.func.id for node in ast.walk(tree) if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)}
    return {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)} - called


def _eval_formula(expr, values):
    """Evaluate one derived_parameters.formulas `expr` against `values`
    ({name: float}, already parsed out of their SPICE-value strings) -- see
    _FORMULA_BINOPS/_FORMULA_UNARYOPS/_FORMULA_FUNCS above for exactly what's allowed."""
    def _visit(node):
        if isinstance(node, ast.Expression):
            return _visit(node.body)
        if isinstance(node, ast.BinOp) and type(node.op) in _FORMULA_BINOPS:
            return _FORMULA_BINOPS[type(node.op)](_visit(node.left), _visit(node.right))
        if isinstance(node, ast.UnaryOp) and type(node.op) in _FORMULA_UNARYOPS:
            return _FORMULA_UNARYOPS[type(node.op)](_visit(node.operand))
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in _FORMULA_FUNCS                 and len(node.args) == 1 and not node.keywords:
            return _FORMULA_FUNCS[node.func.id](_visit(node.args[0]))
        if isinstance(node, ast.Name):
            return values[node.id]
        raise ValueError(f"unsupported expression node in formula {expr!r}: {ast.dump(node)}")
    return _visit(ast.parse(expr, mode="eval"))


def resolve_formulas(block_cfg, params):
    """params (this block's own free params, already expanded by
    resolve_derived_params() and -- for a hierarchical block -- by
    resolve_import_params()/resolve_import_metrics()) expanded with every
    derived_parameters.formulas entry -- pure arithmetic (see
    _eval_formula()) combining SEVERAL of this SAME block's own
    already-resolved values (free parameters, imports, and other formulas'
    own outputs alike), unlike width_groups (one base*factor) or
    import_params/import_metrics (one pure fetch, no math at all -- that's
    what formulas is for). Runs LAST in resolve_materialization_params(),
    after every other derived kind, since a formula's own expr can
    reference an import_params/import_metrics result (e.g. "top"'s own
    rbot_nominal, which multiplies/divides an imported x1_vref_typical
    metric by its own free parameters directly) that doesn't exist until
    those have already run -- and, since `resolved` is threaded through
    entry by entry (not computed from a frozen snapshot of `params`), a
    formula can also reference an earlier formula's own derived name, as
    long as it's declared first in derived_parameters.formulas.

    Exists so a schematic's own w=/l=/etc. attributes never have to carry a
    raw ngspice {...} expression mixing multiple named parameters -- xschem's
    own parser/GUI turned out not to render those reliably (component
    instances silently vanishing on reload, sch/top/top_default.sch's own
    R1-R8) -- every substituted schematic token stays a single plain numeric
    literal after resolve_formulas() runs, exactly like every other derived
    kind (width_groups, import_params, import_metrics) already guarantees."""
    resolved = dict(params)
    # derived_parameters.constants: named literals (e.g. w_finger_max) a
    # formula's expr can reference -- declared in the topology's own JSON so
    # nothing a formula depends on is hidden in Python or another file.
    for name, entry in block_cfg.get("derived_parameters", {}).get("constants", {}).items():
        resolved[name] = entry["value"]
    for name, entry in block_cfg.get("derived_parameters", {}).get("formulas", {}).items():
        values = {}
        for n, v in resolved.items():
            try:
                values[n] = parse_spice_value(v)
            except (ValueError, TypeError):
                continue  # not a plain SPICE-value string (e.g. a block_ref's variation name) -- not formula-usable anyway
        resolved[name] = format_spice_value(_eval_formula(entry["expr"], values), entry.get("unit_suffix", ""))
    return resolved


def resolve_materialization_params(block_cfg, params, sch_dir=None):
    """Single choke point turning a block's own free `params` into
    everything substitute_params() needs to fully materialize its
    schematic: this block's own within-block derived_parameters
    (resolve_derived_params()), plus -- for a hierarchical block like "top"
    that declares its own `sub_blocks` -- every referenced sub-block
    materialized first (materialize_sub_blocks(), writing their own
    <sch_dir>/<block>.sch in place, BEFORE this block's own schematic is
    materialized, since xschem's hierarchy resolution reads whatever's
    currently on disk there), then this block's own import_params/
    import_metrics entries resolved against them -- both PURE fetches (a
    sub-block's own resolved parameter, or its own stored test result) into
    local names, no math at all; resolve_formulas() (below, always last)
    does every actual computation, uniformly, on free parameters + these
    imports + earlier formulas' own outputs. A non-hierarchical block (no
    `sub_blocks` declared) is unaffected by any of this -- reduces to
    exactly resolve_derived_params() + resolve_formulas(), same as before
    hierarchical composition existed. import_metrics additionally needs to
    know which REGISTERED variation (or the BLOCK_REF_DEFAULT sentinel) was
    chosen for each sub_blocks instance, since that decides which
    sim/results.jsonl row it reads -- read directly out of this block's own
    `params` here (the same params[f"{instance}_variation"]
    materialize_sub_blocks() already consulted), not out of
    materialize_sub_blocks()'s own return value, to avoid changing that
    function's (separately tested) return shape.

    sch_dir is forwarded to materialize_sub_blocks() unchanged (None ->
    the shared project sch/) -- pass the caller's own shadow sch/ dir (see
    materialize_variation_shadow) so a sub-block materialized on THIS
    block's behalf lands in the same isolated location as this block's own
    schematic, instead of racing every other concurrently-running
    hierarchical variation on the shared path."""
    resolved = resolve_derived_params(block_cfg, params)
    resolved = resolve_generator_params(block_cfg, resolved)
    sub_blocks = block_cfg.get("sub_blocks")
    if sub_blocks:
        sub_block_resolved = materialize_sub_blocks(workspace.CONFIG, sub_blocks, params, sch_dir=sch_dir)
        resolved = resolve_import_params(block_cfg, resolved, sub_block_resolved)
        sub_block_variation_names = {
            instance: params.get(f"{instance}_variation") or BLOCK_REF_DEFAULT
            for instance in sub_blocks
        }
        resolved = resolve_import_metrics(block_cfg, resolved, sub_block_variation_names)
    resolved = resolve_formulas(block_cfg, resolved)
    return resolved


def calculated_param_descriptions(topology_cfg):
    """{name: description} for every derived_parameters name a topology
    declares (width_groups' members, import_params, import_metrics,
    formulas, generator) -- independent of whether resolve_display_params() can
    actually compute a value for each one right now. Used by the GUI to
    know which parameter rows to mark as "calculated" (equation icon) and
    what to show for them even when it couldn't resolve a value this time
    (e.g. an import_metrics entry whose sub-block hasn't been simulated
    yet) -- the row (name + description) still shows, just with no value
    yet, rather than not showing at all.

    A width_groups MEMBER (e.g. "m6_width") has no description of its
    own in params/<block>/default.json -- only the whole GROUP does, since
    every member exists purely to be that group's own base*factor -- so
    each member inherits its group's description, falling back to a plain
    "name = base * factor" if the group itself has none. import_params,
    import_metrics, and formulas entries always carry their own description
    directly."""
    derived_cfg = topology_cfg.get("derived_parameters", {})
    descriptions = {}
    for group in derived_cfg.get("width_groups", []):
        group_description = group.get("description", "")
        for member_name, member in group.get("members", {}).items():
            descriptions[member_name] = group_description or f"{member_name} = {group['base']} * {member['factor']}"
    for name, entry in derived_cfg.get("import_params", {}).items():
        descriptions[name] = entry.get("description") or f"{entry['from']}.{entry['source']}"
    for name, entry in derived_cfg.get("import_metrics", {}).items():
        descriptions[name] = entry.get("description") or f"{entry['from']}.{entry['test']}.{entry['metric']}"
    for name, entry in derived_cfg.get("formulas", {}).items():
        descriptions[name] = entry.get("description") or entry.get("expr", "")
    for name, entry in derived_cfg.get("generator", {}).items():
        descriptions[name] = entry.get("description", "")
    return descriptions


def resolve_display_params(topology_cfg, params):
    """Best-effort {name: value} preview of a variation's own free params
    PLUS every calculated (derived_parameters) value this topology
    declares, for read-only display (the GUI's Parameters panel) -- unlike
    resolve_materialization_params(), NEVER writes sch/<block>.sch (via
    materialize_sub_blocks()), so it's always safe to call on every
    variation selection, including while a real simulation is materializing
    a DIFFERENT variation concurrently.

    A calculated value that can't be resolved right now -- a stale
    parameter schema, a sub-block reference that no longer resolves, or an
    unresolved import_metrics entry (its sub-block variation hasn't been
    simulated yet, see MissingCrossBlockMetric) -- is simply left out of the
    returned dict rather than raised, so the panel still renders every OTHER
    value instead of failing to show anything at all. Compare against
    calculated_param_descriptions(topology_cfg) to tell "genuinely a free
    parameter" apart from "a calculated one this call just couldn't resolve
    yet"."""
    resolved = dict(params)
    try:
        resolved = resolve_derived_params(topology_cfg, params)
        resolved = resolve_generator_params(topology_cfg, resolved)
    except (StaleParameterSchema, KeyError, ValueError, TypeError):
        return resolved

    sub_blocks = topology_cfg.get("sub_blocks")
    if not sub_blocks:
        try:
            return resolve_formulas(topology_cfg, resolved)
        except (KeyError, ValueError, TypeError):
            return resolved

    try:
        sub_block_resolved = resolve_sub_block_params(workspace.CONFIG, sub_blocks, params)
        resolved = resolve_import_params(topology_cfg, resolved, sub_block_resolved)
        sub_block_variation_names = {
            instance: params.get(f"{instance}_variation") or BLOCK_REF_DEFAULT
            for instance in sub_blocks
        }
        resolved = resolve_import_metrics(topology_cfg, resolved, sub_block_variation_names)
        resolved = resolve_formulas(topology_cfg, resolved)
    except (SystemExit, MissingCrossBlockMetric, KeyError, ValueError, TypeError):
        pass
    return resolved


def check_unresolved(text, label):
    leftover = sorted(set(PARAM_TOKEN_RE.findall(text)))
    if leftover:
        sys.exit(
            f"{label}: unresolved parameter placeholder(s) left after substitution: "
            f"{', '.join(leftover)} -- add them to config.json or check the schematic."
        )


def variation_name(block, topology, params):
    digest = hashlib.sha1(json.dumps(params, sort_keys=True).encode()).hexdigest()[:6]
    return f"{block}-{topology}-{digest}"


def _read_jsonl(path):
    """One dict per non-blank line -- skips (with a stderr warning) any
    line that fails to parse instead of raising, same reasoning as
    analog_designer/results/data.py's own identical copy of this function
    (duplicated rather than imported to avoid a data.py<->run_sim.py import
    cycle, see this module's own _read_jsonl-adjacent comments): a torn
    write from two threads racing on this file (see _append_jsonl's own
    docstring -- now fixed with a lock, but old damage can still be on
    disk) shouldn't make every reader of the whole file fail."""
    if not path.exists():
        return []
    rows = []
    with path.open(encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                print(f"warning: skipping unparseable line {lineno} in {path}: {exc}", file=sys.stderr)
    return rows


_JSONL_APPEND_LOCK = threading.Lock()  # see _append_jsonl's own docstring


def _append_jsonl(path, row):
    """Append one JSON line to `path`, serialized against every OTHER
    _append_jsonl() call in this process via _JSONL_APPEND_LOCK. Needed
    because ensure_variation_registered()/append_results() write to the
    SAME shared sim/variations.jsonl / sim/results.jsonl from every worker
    thread of gen_variations._run_batch's ThreadPoolExecutor at once under
    workspace.cpu_budget() > 1 -- Python's own file.write() is not
    guaranteed atomic against a concurrent writer the way POSIX O_APPEND
    is (and isn't even reliably that on Windows), so two threads' writes
    landing at the same moment can genuinely interleave and leave a torn,
    unparseable line (observed in practice: a `cross`-generated variation's
    trailing `"}` ending up alone on its own line, splitting what should
    have been one JSON object into two -- see analog_designer/results/data.py's
    _read_jsonl, which has no recovery for a single bad line: it raises for
    the WHOLE file, not just that one row). append_run()'s own per-variation
    runs.jsonl doesn't strictly need this (concurrent variations write to
    DIFFERENT files there), but sharing one lock for every jsonl append is
    simpler than reasoning about which call sites need it and which don't,
    and the cost (briefly serializing a handful of tiny appends) is
    negligible next to the docker_exec each of these calls sits next to."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with _JSONL_APPEND_LOCK:
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")


def _rewrite_jsonl(path, keep):
    """Overwrite path with only the rows keep(row) accepts. Unlike
    _append_jsonl (the only other writer touching these files), this
    rewrites the whole file -- written to a sibling temp file first and
    swapped in with os.replace (atomic on POSIX and Windows) so a crash
    mid-write can't leave a truncated/corrupt log."""
    if not path.exists():
        return
    rows = [row for row in _read_jsonl(path) if keep(row)]
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    _replace_with_retry(tmp_path, path)


def _replace_with_retry(src, dst, timeout_s=10.0):
    """os.replace(), retried on PermissionError for up to timeout_s. On
    Windows the swap fails ("Access is denied") whenever ANY other process
    has `dst` open at that instant -- the GUI's own 1s new-row poll and
    per-variation DONE refresh read results.jsonl/variations.jsonl
    throughout a batch, and editors/indexers/antivirus take brief handles
    too. Those reads are short, so a few retries get through; a
    discard-on-fail checkpoint used to crash the whole batch on the first
    collision instead."""
    deadline = time.monotonic() + timeout_s
    delay = 0.05
    while True:
        try:
            os.replace(src, dst)
            return
        except PermissionError:
            if time.monotonic() >= deadline:
                raise
            time.sleep(delay)
            delay = min(delay * 2, 0.5)


def ensure_variation_registered(name, block, topology, params, origin=None):
    """sim/variations.jsonl is an identity registry, not a run log: write
    once per unique (block, topology, parameters), never duplicate.
    origin records how this variation's parameters were produced --
    {"kind": "manual"|"random"|"generate"|"combine", ...} (plus a couple of
    additional kinds pro's own variation generators tag), kind-
    specific extra keys (e.g. "base" for generate, "parent_a"/"parent_b" for
    combine). Defaults to {"kind": "manual"} for direct run_sim.py
    invocations, which don't go through a generator script. Existing rows
    predate this field entirely -- readers must use row.get("origin"), not
    row["origin"]."""
    path = workspace.sim_root() / "variations.jsonl"
    if any(r["name"] == name for r in _read_jsonl(path)):
        return
    _append_jsonl(path, {
        "name": name,
        "block": block,
        "topology": topology,
        "parameters": params,
        "origin": origin or {"kind": "manual"},
        "created": datetime.datetime.now().isoformat(timespec="seconds"),
    })


def append_run(variation, test_name, label, conditions, outcome):
    """sim/<variation>/runs.jsonl: raw execution history, one line per
    simulation attempt. Per-variation (not a single global file) so
    parallel workers running different variations never write to the same
    file.

    "exit_code" replaces the old hardcoded "ngspice_exit_code" key -- that
    name silently recorded None for every Xyce failure, since
    run_one_xyce() returns "xyce_exit_code" instead. Nothing reads this
    file's historical rows (confirmed: load_runs() had zero callers before
    this), so there's no old-key compatibility to preserve."""
    exit_code = outcome.get("ngspice_exit_code")
    if exit_code is None:
        exit_code = outcome.get("xyce_exit_code")
    _append_jsonl(workspace.sim_root() / variation / "runs.jsonl", {
        "variation": variation,
        "test": test_name,
        "condition": label,
        "conditions": conditions,
        "status": outcome["status"],
        "exit_code": exit_code,
        "error": outcome.get("error"),
        "diagnostics": outcome.get("diagnostics", []),
        "created": datetime.datetime.now().isoformat(timespec="seconds"),
    })


def compute_definition_hash(block_cfg, test_cfg):
    """Hash of everything that defines HOW this test is run and measured --
    independent of which parameter VALUES were simulated (that's the
    variation hash). Changes when: the test's own config.json subtree
    changes, the testbench .sch changes, the parser .py (or its own
    directory's _common.py, or tb/_shared/'s parser_common.py) changes, or
    the topology .sch template changes (catches structural schematic edits,
    like a body-tie fix, that don't touch any parameter value but do change
    what gets simulated)."""
    parts = [json.dumps(test_cfg, sort_keys=True)]
    topology_sch = workspace.PROJECT_ROOT / "sch" / block_cfg["schematic"]
    parts.append(topology_sch.read_text(encoding="utf-8"))
    if "generator" in block_cfg:
        # A "generator"-backed topology's own electrical values come from
        # resolve_generator_params()/openems_generator_runner.py, not from
        # the schematic template's own literal text (which just has
        # 'l'/'rs'/... placeholder tokens) -- so a real code change to the
        # generator SHOULD invalidate cached results the same way a
        # topology .sch edit already does for every other topology, hashed
        # here IN ADDITION TO the schematic template above (both matter
        # now that "schematic" is a real static template again, not a
        # per-variation generated output). "generator" is a project-repo-
        # relative file path (same convention as "testbench"/"parser"),
        # not an importable dotted module -- it's project-specific code
        # that lives in the open project's own repo, not in this shared
        # tool.
        try:
            gen_path = workspace.PROJECT_ROOT / block_cfg["generator"]
            parts.append(gen_path.read_text(encoding="utf-8"))
        except Exception:
            # best-effort: don't let a resolution hiccup here crash the
            # whole freshness check for every test in the block.
            parts.append(block_cfg["generator"])
    if "testbench" in test_cfg:
        parts.append((workspace.PROJECT_ROOT / test_cfg["testbench"]).read_text(encoding="utf-8"))
    parser_path = workspace.PROJECT_ROOT / test_cfg["parser"]
    parts.append(parser_path.read_text(encoding="utf-8"))
    common_path = parser_path.parent / "_common.py"
    if common_path.exists():
        parts.append(common_path.read_text(encoding="utf-8"))
    # tb/_shared/parser_common.py: the cross-block-shared parser helpers
    # (read_data/in_spec/add_spec_bounds/... -- originally near-identical
    # per-block _common.py copies, factored out once cmos_vref/output_amp/top
    # all needed the same PSRR/current-consumption/etc. logic). Hashed
    # unconditionally alongside the per-directory _common.py above (not
    # instead of it) so an edit there invalidates every test that actually
    # imports from it, the same way a per-directory _common.py edit already
    # does for tests that use that instead.
    shared_common_path = workspace.PROJECT_ROOT / "tb" / "_shared" / "parser_common.py"
    if shared_common_path.exists():
        parts.append(shared_common_path.read_text(encoding="utf-8"))
    return hashlib.sha256("\n".join(parts).encode("utf-8")).hexdigest()[:12]


def git_info():
    """Informational only -- NOT the staleness mechanism (that's
    definition_hash, which works with or without git). Returns
    (commit_hash_or_None, dirty_bool_or_None)."""
    commit = subprocess.run(
        ["git", "-C", str(workspace.PROJECT_ROOT), "rev-parse", "HEAD"],
        capture_output=True, text=True,
    )
    if commit.returncode != 0:
        return None, None
    status = subprocess.run(
        ["git", "-C", str(workspace.PROJECT_ROOT), "status", "--porcelain"],
        capture_output=True, text=True,
    )
    return commit.stdout.strip(), bool(status.stdout.strip())


def load_results():
    return _read_jsonl(workspace.sim_root() / "results.jsonl")


def is_test_fresh(results, name, test_name, definition_hash):
    return any(
        r["variation"] == name and r["test"] == test_name and r["definition_hash"] == definition_hash
        for r in results
    )


def _test_freshness(tests, block_cfg, existing_results, name, force):
    """(fresh_names, to_run_dict, definition_hashes) for one variation --
    single source of truth shared by run_variation() (which actually
    simulates `to_run`) and plan_progress() (which only plans them, for a
    progress bar), so the two can never disagree about what's stale."""
    definition_hashes = {t: compute_definition_hash(block_cfg, cfg) for t, cfg in tests.items()}
    fresh = {
        t for t in tests
        if not force and is_test_fresh(existing_results, name, t, definition_hashes[t])
    }
    to_run = {t: cfg for t, cfg in tests.items() if t not in fresh}
    return fresh, to_run, definition_hashes


def _n_conditions(test_cfg, defaults):
    tb_text = (workspace.PROJECT_ROOT / test_cfg["testbench"]).read_text(encoding="utf-8")
    return len(list(condition_matrix(test_cfg, defaults, internal_sweep_axis(test_cfg, tb_text))))


def historical_test_durations(results, block):
    """{test_name: mean(duration_seconds)} mined from `results` rows for
    `block` -- the data plan_progress() below turns into progress weights.
    Pooled across every topology of `block` rather than kept separate per
    topology: config["tests"][block] is declared once per block, shared by
    every topology (same testbench/parser, just a different materialized
    schematic) -- pooling maximizes the (often small) sample size for a
    more stable average, at the acceptable cost of not distinguishing "this
    topology's own `power` test happens to run slower than that one's".

    Dedupes to exactly one sample per (variation, test) pair before
    averaging -- append_results() writes the SAME duration_seconds value
    once per metric row of a test, so summing/averaging raw rows here would
    wildly overcount (see append_results()'s own docstring). Rows with a
    missing/None duration_seconds (older data written before this field
    existed, or a test that errored before append_results() could even run)
    are EXCLUDED, not treated as 0 -- treating them as 0 would silently drag
    every average toward zero the moment any history predates this field.
    A test skipped by --skip-on-fail is never append_results()'d at all
    (see run_variation()'s own skip_on_fail_profile branch), so it contributes zero
    samples here too, never a wrong/short one -- a skipped test shows up
    on the GUI side as its own gray segment instead (see
    emit_progress_skipped())."""
    samples = {}
    for row in results:
        if row.get("block") != block or row.get("duration_seconds") is None:
            continue
        key = (row["variation"], row["test"])
        samples.setdefault(row["test"], {})[key] = row["duration_seconds"]
    return {
        test_name: sum(by_key.values()) / len(by_key)
        for test_name, by_key in samples.items()
    }


def plan_progress(block_cfg, tests, defaults, existing_results, name, force, historical_durations):
    """[(test_name, n_conditions, est_seconds), ...] for every test
    run_variation() would actually simulate for this variation right now,
    without running anything -- reuses _test_freshness()/_n_conditions() so
    the GUI's progress plan can never drift from the real work done. Pure
    (file reads only, no docker) -- cheap to call for every variation in a
    batch before any simulation starts. Feeds emit_progress_plan().

    est_seconds comes from historical_durations (see
    historical_test_durations() above) -- NOT scaled by the condition count:
    a test's own historical average already reflects however many
    conditions it runs, since run_variation() times the whole run_test()
    call once per test. A test with no sample of its own yet falls back to
    the per-condition mean of the tests that DO have one (x its own
    condition count). With no history at all for this block, est_seconds
    is None -- the GUI then weighs every condition equally (plain step
    counting, the old behavior) and shows no ETA until real steps arrive.
    Never 0: a zero-weight test would be invisible on the bar and in the ETA."""
    _, to_run, _ = _test_freshness(tests, block_cfg, existing_results, name, force)
    entries = [(t, _n_conditions(cfg, defaults)) for t, cfg in to_run.items()]
    known = [
        (historical_durations[t], _n_conditions(tests[t], defaults))
        for t in historical_durations if t in tests
    ]
    known_conditions = sum(n for _, n in known)
    if not known_conditions:
        return [(t, n, None) for t, n in entries]
    per_condition = sum(s for s, _ in known) / known_conditions
    return [
        (t, n, historical_durations[t] if t in historical_durations else per_condition * n)
        for t, n in entries
    ]


def _progress_enabled():
    return os.environ.get("ANALOG_DESIGNER_PROGRESS") == "1"


def emit_progress_plan(variation, entries):
    """Printed up front, once per (variation, test) a job will simulate, by
    whichever entry point (main() for a single Update,
    gen_variations._run_batch()/_run_hierarchical_batch() for every batch
    path) knows the full job before simulating anything -- see
    plan_progress() for `entries`. The GUI sizes each test's slice of the
    progress bar by est_seconds, split evenly across its n_conditions
    STEP lines. Gated behind ANALOG_DESIGNER_PROGRESS so a plain CLI
    invocation's output is unchanged; analog_designer/gui/run_trigger.py
    sets it for GUI-launched jobs."""
    if _progress_enabled():
        for test_name, n_conditions, est_seconds in entries:
            est = "-" if est_seconds is None else f"{est_seconds:.3f}"
            print(f"@PROGRESS PLAN {variation} {test_name} {n_conditions} {est}")


def emit_progress_step(variation, test_name, ok):
    """Printed once per completed (test, condition) simulation attempt --
    see run_test()'s condition loop. Counts every attempt, success or not,
    matching exactly what plan_progress() counted; `ok` picks the green vs
    red segment of the GUI's bar."""
    if _progress_enabled():
        print(f"@PROGRESS STEP {variation} {test_name} {'ok' if ok else 'fail'}")


def emit_progress_testfail(variation, test_name):
    """Printed when a whole test ends up "error" AFTER its conditions were
    stepped (every condition failed, or extract()/evaluate() raised) -- the
    GUI recolors that test's already-green steps red, since a condition
    that simulated fine but whose test produced no result is still a
    failure from the reader's point of view."""
    if _progress_enabled():
        print(f"@PROGRESS TESTFAIL {variation} {test_name}")


def emit_progress_container(container_id):
    """Printed once, right after managed_container() starts a fresh
    container, so analog_designer/gui/run_trigger.py can learn its id without any IPC
    beyond the stdout pipe it already reads every line from. Lets
    RunTrigger.cancel() `docker stop` that specific container instead of
    only killing the local subprocess -- terminate() alone (SIGKILL-strength
    on Windows, via TerminateProcess) never gives managed_container()'s own
    `finally: docker stop` a chance to run, so a stuck ngspice run (e.g. a
    non-convergent testbench) would otherwise keep running in an orphaned
    container indefinitely after a cancel. Gated behind ANALOG_DESIGNER_PROGRESS
    like every other @PROGRESS line, so a plain CLI invocation's output is
    unchanged."""
    if _progress_enabled():
        print(f"@PROGRESS CONTAINER {container_id}")


def emit_progress_running(variation, test_name, label):
    """Printed once per (test, condition) attempt, right before the
    potentially-slow netlist+simulate call (see run_test()) -- lets
    analog_designer/gui/app.py highlight `variation`'s own row in the
    Variations table (and, if that variation's detail panel happens to be
    open, `test_name`'s own row there too) as currently in flight, alongside
    a per-row live elapsed-time readout (see emit_progress_variation_done's
    own docstring for why that's tracked per row, not as one global
    indicator). `label` is a condition_label() value, which never contains a
    space, so a plain 3-way str.split(" ", 2) on the GUI side is enough to
    recover all three fields. Gated behind ANALOG_DESIGNER_PROGRESS like
    every other @PROGRESS line."""
    if _progress_enabled():
        print(f"@PROGRESS RUNNING {variation} {test_name} {label}")


def emit_progress_skipped(variation, test_names):
    """Printed when planned tests of `variation` will never run -- a
    skip_on_fail_profile break (the tests after the disqualifying one, see
    run_variation()'s own per-test loop) or a StaleParameterSchema skip
    (all of them). emit_progress_plan() is an UPPER BOUND (every planned
    test running to completion), so this is how the GUI fills those tests'
    slices gray instead of leaving the bar short of 100% once the job
    finishes. Gated behind ANALOG_DESIGNER_PROGRESS like every other
    @PROGRESS line."""
    if _progress_enabled() and test_names:
        print(f"@PROGRESS SKIPPED {variation} {' '.join(test_names)}")


def emit_progress_trimmed(names):
    """Printed right after a discard-on-fail checkpoint trim_variation()'d
    `names` (see gen_variations._run_batch/_run_hierarchical_batch's own
    _checkpoint()) -- the GUI only ever adds/updates Variations table rows
    mid-job, so without this the discarded rows would linger on screen until
    the whole batch ends. Gated behind ANALOG_DESIGNER_PROGRESS like every
    other @PROGRESS line."""
    if _progress_enabled() and names:
        print(f"@PROGRESS TRIMMED {' '.join(names)}")


def emit_progress_variation_done(variation, elapsed_seconds, any_error):
    """Printed once per variation, right as run_variation() returns (every
    exit path -- see its own call sites) -- the definitive "this one's
    finished" signal emit_progress_running()'s own per-condition lines can't
    provide on their own (there's no reliable way to tell "no more RUNNING
    lines for this variation" apart from "still between conditions" purely
    from their absence, especially with several variations interleaving
    under workspace.cpu_budget() > 1). Lets the GUI freeze that variation's
    row at its own final elapsed time with a pass/fail marker instead of
    leaving it stuck on a live-ticking "still running" readout, which is
    what actually answers "is this done, and how long did it take" per
    sample -- a SINGLE global "currently running" indicator can't represent
    more than one in-flight variation at once, and used to visibly jump
    between whichever of up to cpu_budget() rows last printed something,
    which read as random flicker rather than useful progress."""
    if _progress_enabled():
        status = "error" if any_error else "ok"
        print(f"@PROGRESS DONE {variation} {status} {elapsed_seconds:.1f}")


def append_results(name, block, topology, test_name, definition_hash, metrics, git_commit, git_dirty, duration_seconds=None):
    """One row per metric in `metrics` -- so duration_seconds (how long the
    WHOLE test, every one of its conditions, took to simulate -- see
    run_variation()'s own timing around its run_test() call) ends up
    repeated identically across every metric row of this one (variation,
    test) run. A reader mining results.jsonl for per-test timing (see
    historical_test_durations()) MUST dedupe by (variation, test) before
    averaging, never sum/average raw duration_seconds values across metric
    rows, or they'll wildly overcount. Optional/None (not a required
    positional) even though every call site today provides it -- matches
    this file's existing convention for per-metric fields that not every
    caller can supply (mean/std/minimum/maximum are all .get()-based
    already)."""
    path = workspace.sim_root() / "results.jsonl"
    created = datetime.datetime.now().isoformat(timespec="seconds")
    for metric in metrics:
        _append_jsonl(path, {
            "variation": name,
            "block": block,
            "topology": topology,
            "test": test_name,
            "metric": metric["name"],
            "typical": metric["typical"],
            "min": metric["min"],
            "max": metric["max"],
            "mean": metric.get("mean"),
            "std": metric.get("std"),
            "unit": metric.get("unit"),
            "minimum": metric.get("minimum"),
            "maximum": metric.get("maximum"),
            "definition_hash": definition_hash,
            "git_commit": git_commit,
            "git_dirty": git_dirty,
            "created": created,
            "duration_seconds": duration_seconds,
        })


def trim_variation(name):
    """Delete a variation entirely: its identity row from variations.jsonl,
    every result row for it in results.jsonl, and its sim/<name>/ directory
    (runs.jsonl, per-test run dirs, plots). Synchronous local file I/O only
    -- no docker -- but must not run concurrently with a docker-backed
    writer (run_sim.py/gen_variations.py/manual_variation.py, or any of
    pro's own variation generators), since those
    append to the same two JSONL files this rewrites; callers are
    responsible for that mutual exclusion (see analog_designer/gui/app.py)."""
    _rewrite_jsonl(workspace.sim_root() / "variations.jsonl", lambda r: r["name"] != name)
    _rewrite_jsonl(workspace.sim_root() / "results.jsonl", lambda r: r["variation"] != name)
    sim_dir = workspace.sim_root() / name
    if sim_dir.exists():
        shutil.rmtree(sim_dir)


def purge_stale_results(names):
    """Permanently remove every results.jsonl row for a (variation, test) key
    whose freshest stored result no longer matches what that test currently
    hashes to -- the same staleness analog_designer.results.data.latest_results()
    flags at read time (its own freshest_hash/current_hashes logic,
    duplicated here rather than imported, since data.py imports this module
    and importing data.py back would be a cycle -- same reason this module
    keeps its own _read_jsonl instead of data.py's). Scoped to `names` (an
    iterable of variation names) -- other variations' stale results are
    untouched. Also deletes that (variation, test)'s now-orphaned
    sim/<variation>/<test>/ dir (per-condition outputs and plot PNGs) --
    nothing reads it once its results rows are gone, and a rerun rewrites it
    from scratch anyway. Unlike trim_variation, leaves the variation and any
    still-fresh (variation, test) results alone. Returns how many
    (variation, test) keys were purged."""
    names = set(names)
    results_path = workspace.sim_root() / "results.jsonl"
    rows = [r for r in _read_jsonl(results_path) if r["variation"] in names]
    if not rows:
        return 0

    config = json.loads((workspace.PROJECT_ROOT / "config.json").read_text(encoding="utf-8"))
    workspace.resolve_parameters_files(workspace.PROJECT_ROOT, config)

    freshest = {}  # (variation, test) -> {"created", "hash", "block", "topology"}
    for row in rows:
        key = (row["variation"], row["test"])
        if key not in freshest or row["created"] > freshest[key]["created"]:
            freshest[key] = {
                "created": row["created"], "hash": row["definition_hash"],
                "block": row["block"], "topology": row["topology"],
            }

    current_hashes = {}
    stale_keys = set()
    for key, info in freshest.items():
        hash_key = (info["block"], info["topology"], key[1])
        if hash_key not in current_hashes:
            try:
                block_cfg = config["blocks"][info["block"]]["topologies"][info["topology"]]
                test_cfg = config["tests"][info["block"]][key[1]]
                current_hashes[hash_key] = compute_definition_hash(block_cfg, test_cfg)
            except (KeyError, FileNotFoundError):
                current_hashes[hash_key] = None
        if current_hashes[hash_key] != info["hash"]:
            stale_keys.add(key)

    if not stale_keys:
        return 0

    _rewrite_jsonl(results_path, lambda r: (r["variation"], r["test"]) not in stale_keys)
    for variation, test in stale_keys:
        test_dir = workspace.sim_root() / variation / test
        if test_dir.is_dir():
            shutil.rmtree(test_dir)
    return len(stale_keys)


def purge_aux_outputs(names):
    """Delete auxiliary simulator outputs left in sim/<variation>/<test>/
    <condition>/ run dirs for each name in `names`: every .raw, and every
    '<test>_0_*.data' side dump (e.g. startup's '_diag.data') -- never the
    canonical '<test>_0.data' the parser and lazy plots read. New runs
    don't leave these behind at all (see _AUX_SCRATCH_ROOT); this clears
    ones written before that, or kept via KEEP_AUX_ENV. Returns
    (files removed, bytes freed)."""
    removed = freed = 0
    for name in names:
        sim_dir = workspace.sim_root() / name
        if not sim_dir.is_dir():
            continue
        for test_dir in (p for p in sim_dir.iterdir() if p.is_dir()):
            for path in test_dir.glob("*/*"):
                if path.suffix == ".raw" or (path.suffix == ".data" and path.name.startswith(f"{test_dir.name}_0_")):
                    freed += path.stat().st_size
                    path.unlink()
                    removed += 1
    return removed, freed


def purge_plots(names):
    """Delete every cached plot PNG under sim/<variation>/*/ for each name in
    `names` -- forces the next view to regenerate it (see generate_plot()
    below). Useful when only a parser's own plotting code changed, which
    isn't tracked by definition_hash and so wouldn't be caught by
    purge_stale_results(). Returns how many PNG files were removed."""
    removed = 0
    for name in names:
        sim_dir = workspace.sim_root() / name
        if not sim_dir.exists():
            continue
        for png in sim_dir.glob("*/*.png"):
            png.unlink()
            removed += 1
    return removed


def _forget_other_projects_shared_helpers(shared_dir):
    """Parsers import tb/_shared/ helpers by bare module name
    (`from parser_common import ...`), resolved through sys.path and cached
    in sys.modules -- so in a process that has opened another project
    before (the GUI's Open Folder, a test run), that project's
    tb/_shared/ entry and its already-imported parser_common would silently
    win over this one's. Drop both before loading this project's parser."""
    def _is_other(directory):
        return Path(directory).parts[-2:] == ("tb", "_shared") and str(directory) != shared_dir
    sys.path[:] = [p for p in sys.path if not _is_other(p)]
    for name, module in list(sys.modules.items()):
        module_file = getattr(module, "__file__", None)
        if module_file and _is_other(Path(module_file).parent):
            del sys.modules[name]


def load_parser(relpath):
    module_path = workspace.PROJECT_ROOT / relpath
    module_dir = str(module_path.parent)
    if module_dir not in sys.path:
        sys.path.insert(0, module_dir)
    # tb/_shared/: cross-block parser helpers (parser_common.py) -- see
    # compute_definition_hash()'s own comment. Inserted AFTER the parser's
    # own directory (not before) so a same-named module sitting next to the
    # parser itself would still win over the shared one -- not expected to
    # ever happen in practice (the shared module is deliberately named
    # parser_common, not _common, precisely so the two can never collide on
    # sys.path at the same time), but this keeps the more-specific location
    # taking precedence if it ever did.
    shared_dir = str(workspace.PROJECT_ROOT / "tb" / "_shared")
    _forget_other_projects_shared_helpers(shared_dir)
    if shared_dir not in sys.path:
        sys.path.append(shared_dir)
    spec = importlib.util.spec_from_file_location(module_path.stem, module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_generator_module(block_cfg):
    """Dynamically loads a "generator"-backed topology's own project-repo-
    relative Python module (block_cfg["generator"]) by file path -- same
    pattern as load_parser() above. This is project-specific code living
    in the open project's own repo, not an importable package of this
    tool."""
    gen_path = workspace.PROJECT_ROOT / block_cfg["generator"]
    spec = importlib.util.spec_from_file_location(gen_path.stem, gen_path)
    generator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(generator)
    return generator


def resolve_generator_params(block_cfg, params):
    """params (already expanded by resolve_derived_params()) expanded with
    a "generator"-backed topology's own derived ELECTRICAL parameters --
    dynamically loads block_cfg["generator"] and calls its own
    geometry_from_params() + load_stack() + fit_electrical_params(...,
    em_result=None) -- the fast, no-FDTD, placeholder-quality path (real
    Cox from geometry+stack, generic placeholder for the rest -- see that
    module's own fit_electrical_params() docstring). A no-op for every
    topology without a "generator" key, exactly like resolve_formulas()/
    resolve_derived_params() are no-ops when a topology declares no
    width_groups/formulas.

    This is what lets a "generator"-backed topology's own schematic go
    back to being a normal STATIC template (e.g. inductor_spiral.sch, with
    'l'/'rs'/'cox'/... substitute_params() tokens) like every other
    topology -- no more bespoke per-variation .sch-writing function; the
    existing substitute_params()/check_unresolved() pipeline does the rest
    uniformly.

    NEVER runs the real (hours-long) FDTD characterization -- that only
    happens inside the container via openems_generator_runner.py, invoked
    from run_one_openems(). Called from resolve_materialization_params()/
    resolve_sub_block_params()/resolve_display_params(), all of which must
    stay fast and synchronous.

    Fixed at corner "tt": materialization/display shows ONE set of derived
    values shared by every corner/temperature condition a later test might
    sweep (same as every other topology's static .sch), so there's no
    single "right" per-condition corner to pick here -- "tt" matches this
    topology's own single-condition test default in config.json."""
    if "generator" not in block_cfg:
        return dict(params)
    generator = _load_generator_module(block_cfg)
    geometry = generator.geometry_from_params(params)
    stack = generator.load_stack("tt")
    fitted = generator.fit_electrical_params(geometry, stack, em_result=None)
    resolved = dict(params)
    resolved.update({name: format_spice_value(value, "") for name, value in fitted.items()})
    return resolved


MOS_CORNER_SECTION = {
    "tt": "mos_tt", "ss": "mos_ss", "ff": "mos_ff",
    # Local (intra-die) device mismatch and global (inter-die) process
    # variation layered on top of the "tt" process point -- see
    # tb/_shared/parser_common.mc_stats() and cornerMOShv.lib/cornerMOSlv.lib's
    # own "mos_tt_mismatch"/"mos_tt_stat" .LIB sections (both PDK corner
    # files define these, since "top"'s own testbenches .lib-include both
    # HV and LV libraries against the same 'mos_corner' token).
    "tt_mismatch": "mos_tt_mismatch", "tt_stat": "mos_tt_stat",
}

# GF180MCU's own corner-section names, from its single libs.tech/{ngspice,
# xyce}/sm141064.spice deck (".lib typical"/".lib ff"/".lib ss"/".lib fs"/
# ".lib sf", confirmed by grepping that file) -- unlike IHP's split
# cornerMOShv.lib/cornerMOSlv.lib, there's no "mos_" prefix and no unified
# mismatch/stat section per corner (GF180MCU's statistical .LIB blocks are
# split per device family instead, e.g. "res_stat"/"bjt_stat"/"nfet_03v3_stat"
# -- add here if/when a gf180mcu testbench needs Monte Carlo). Selected by
# setup_container() based on the container's own $PDK (get_pdk_name()).
MOS_CORNER_SECTION_GF180MCU = {
    "tt": "typical", "ss": "ss", "ff": "ff",
}

#: Resistor-model corner (e.g. rhigh's sheet resistance), decoupled from
#: MOS_CORNER_SECTION on purpose -- every EXISTING testbench hardcodes a
#: literal "res_typ" in its own stimuli text (never a substituted
#: 'res_corner' token), so this table and the substitution below only ever
#: apply to a testbench that opts in by referencing 'res_corner' itself
#: (see tb/cmos_vref/tb_vref_mismatch.sch, tb/top/tb_top_mismatch.sch) --
#: zero effect on any of the other, already-validated testbenches. Kept as
#: an axis independent from "corner" (not folded into MOS_CORNER_SECTION's
#: own values) so a future study can mix them freely -- e.g. a "worst
#: resistor at typical MOS" or "fast MOS with slow resistor" point -- via
#: plain cross-product conditions{} lists (corner: [...] x res_corner: [...]),
#: the same generalization condition_matrix() already gives any other
#: multi-valued conditions{} key.
RES_CORNER_SECTION = {
    "typ": "res_typ", "bcs": "res_bcs", "wcs": "res_wcs",
    "typ_mismatch": "res_typ_mismatch", "bcs_mismatch": "res_bcs_mismatch", "wcs_mismatch": "res_wcs_mismatch",
    "typ_stat": "res_typ_stat",
}
_RES_SECTION_CMOS5L = {"res_typ_stat": "res_stat"}


def condition_matrix(test_cfg, defaults, sweep_axis):
    """Cartesian product of every conditions{} key that could vary from run
    to run -- corner/temperature always participate (falling back to the
    project's own default when the test doesn't declare them), and any
    OTHER conditions{} key with more than one value becomes an outer axis
    too, using the exact same "run the whole testbench again" mechanism:
    _netlist() copies every non-corner/temperature key straight into the
    substituted tb_params, so a plain fixed testbench token (e.g. an
    'ibias' current-source value) can vary run-to-run with no schematic
    change, same as corner/temperature already do for any test that
    doesn't sweep them internally. A conditions{} key with exactly one
    value stays a plain fixed_tb_params() substitution, not an axis here --
    this keeps every existing test's run/label layout unchanged (adding a
    stray single-valued key to the product would rename its run
    directories for no reason).

    sweep_axis, if given as (conditions_key, tb_param_prefix) from
    internal_sweep_axis(), names the axis this testbench already sweeps
    INTERNALLY in one ngspice run (e.g. load_reg's own 'iload_min'/
    'iload_max' .dc sweep) -- excluded entirely from this outer grid, never
    a fixed value either. When it's specifically 'temperature', the outer
    grid additionally collapses to corner-only (temp_sweep's own special
    case, unchanged from before this was generalized).

    conditions{}'s own "typical" key (see typical_conditions()) is reserved
    metadata -- a dict naming the nominal value of each axis, not itself an
    axis to sweep -- so it's popped off before the "any remaining
    multi-value key becomes an outer axis" generalization below, which
    would otherwise misread its 2 dict keys ("corner", "temperature") as 2
    sweep VALUES for a stray "typical" axis."""
    conditions = dict(test_cfg.get("conditions", {}))
    conditions.pop("typical", None)
    if sweep_axis:
        conditions.pop(sweep_axis[0], None)
    corners = conditions.pop("corner", [defaults["corner"]])
    if sweep_axis and sweep_axis[0] == "temperature":
        # Still generalizes to any OTHER multi-valued conditions{} key left
        # after popping corner/typical/the swept axis itself (e.g. a vdd
        # list carrying both the nominal 3v3 supply and an informational
        # corner-case supply, folded into temp_sweep's own conditions
        # instead of a separately-duplicated test) -- same mechanism as the
        # non-collapsed path below, just crossed with corner only, never
        # with temperature (that's still swept internally, one .dc per run).
        axes = {"corner": corners}
        axes.update({key: values for key, values in conditions.items() if len(values) > 1})
        keys = list(axes)
        for combo in itertools.product(*(axes[key] for key in keys)):
            yield dict(zip(keys, combo))
        return
    temperatures = conditions.pop("temperature", [defaults["temperature"]])
    axes = {"corner": corners, "temperature": temperatures}
    axes.update({key: values for key, values in conditions.items() if len(values) > 1})
    keys = list(axes)
    for combo in itertools.product(*(axes[key] for key in keys)):
        yield dict(zip(keys, combo))


def condition_label(conditions):
    return "_".join(f"{k}-{v}" for k, v in conditions.items())


def typical_conditions(test_cfg, defaults):
    """Merged {axis: value} naming which single value of every axis is this
    test's nominal/typical point -- what a parser's evaluate() matches
    against each run's own "conditions" dict (see
    tb/_shared/parser_common.typical_min_max()) to pick the one run that
    counts as "typical" instead of a hardcoded convention (corner=="tt",
    temperature=="25") duplicated ad hoc across parser files. Starts from
    the project's own defaults{} (already one value per axis) and layers
    this test's own conditions.typical{} override on top, for any axis
    whose nominal point differs from the project default -- mirrors
    condition_matrix()'s own defaults-fallback for corner/temperature.
    Deliberately a superset of whatever axes this test's own
    condition_matrix() actually varies (also carries vdd/Cload/ibias/...
    even when those aren't outer axes here) -- callers match only the keys
    present in one run's own conditions dict, so the extra keys are
    harmless."""
    typical = dict(defaults)
    typical.update(test_cfg.get("conditions", {}).get("typical", {}))
    return typical


# Confirmed live (2026-09-08): concurrent `xschem -x -q` batch-netlist
# invocations against the SAME container (see run_test()'s per-condition
# ThreadPoolExecutor) occasionally produce no .spice at all, even though
# the identical command against the same schematic, run serially (one
# docker_exec at a time -- either workspace.cpu_budget()==1, or standalone
# outside any pool), reproduced 0 failures across 40+ attempts. Whatever
# xschem/Xvnc-internal resource this trips on when several batch instances
# race (a scratch/lock file under its own $HOME, an X11-connection hiccup
# on the shared DISPLAY=:1 -- not chased down further, since it never
# reproduces serially), it's clearly transient, not a real data/content
# problem (unlike the "IS MISSING" branch below, which is deterministic
# and NOT worth retrying) -- so a few quick retries absorb it cheaply
# without giving up the concurrency this was worth adding for.
_NETLIST_MAX_ATTEMPTS = 3
_NETLIST_RETRY_DELAY_SECONDS = 0.5


def _missing_project_subckts(netlist_text):
    """Names of this project's own blocks (config.json "blocks") that the
    netlist instantiates (an X line whose model is that block) but never
    defines with a .subckt -- xschem intermittently emits a netlist with a
    hierarchical block's expansion silently absent (see run_one_ngspice()'s
    own docstring). ngspice at least fails loudly on that ("unknown
    subckt"), but a static "netlist" test (area) would read ZERO devices
    and record 0.0 as a valid result -- confirmed live, 14 of 236
    ihp_mh_ip__cmos_vref area results (2026-09-27). PDK devices
    (sg13_hv_nmos, ...) are subckts too, but come from the .lib files, not
    the netlist, so only project block names are checked."""
    blocks = {name.lower() for name in workspace.CONFIG.get("blocks", {})}
    lines = []
    for raw in netlist_text.splitlines():
        if raw.startswith("+") and lines:
            lines[-1] += " " + raw[1:]
        else:
            lines.append(raw)
    defined, used = set(), set()
    for line in lines:
        tokens = line.split()
        if not tokens:
            continue
        head = tokens[0].lower()
        if head == ".subckt" and len(tokens) > 1:
            defined.add(tokens[1].lower())
        elif head.startswith("x"):
            model = next((tok for tok in reversed(tokens[1:]) if "=" not in tok), None)
            if model is not None and model.lower() in blocks:
                used.add(model.lower())
    return used - defined


def _netlist(container, test_name, tb_source, conditions, tb_params_base,
             run_dir, container_run_dir, container_rcfile, ctx):
    """Materialize + netlist one testbench through xschem -- the half of
    run_one_ngspice that's common to any SIMULATOR_RUNNERS entry, since
    netlisting is needed whether or not ngspice ever runs afterward (see
    run_one_netlist, which stops here and hands the .spice straight to a
    parser for static analysis instead of simulating it). Returns the
    netlist Path on success, or an {"status": "error", ...} dict."""
    run_dir.mkdir(parents=True, exist_ok=True)

    tb_params = dict(tb_params_base)
    tb_params["mos_corner"] = ctx.mos_corner_section[conditions["corner"]]
    for key, value in conditions.items():
        if key != "corner":
            tb_params[key] = value
    if "vdd" in conditions:
        tb_params["Vavdd"] = conditions["vdd"]  # outer vdd axis: keep the cmos_vref alias in step
    # Only a testbench that opts in by referencing 'res_corner' itself ever
    # has this key at all (from fixed_tb_params() for a single-valued
    # conditions.res_corner, or from the outer-axis copy above for a
    # multi-valued one) -- translate whichever friendly name (RES_CORNER_SECTION's
    # own keys, e.g. "typ_mismatch") got in there to the real PDK .LIB
    # section name, same as 'mos_corner' above. A no-op for every other
    # (existing) testbench, which never sets this key.
    if "res_corner" in tb_params:
        tb_params["res_corner"] = RES_CORNER_SECTION[tb_params["res_corner"]]
        # CMOS5L's cornerRES.lib names its "typical + statistical" section
        # res_stat (sg13g2: res_typ_stat) -- otherwise ngspice aborts with
        # "section definition res_typ_stat not found".
        if ctx.pdk_name == "ihp-sg13cmos5l":
            tb_params["res_corner"] = _RES_SECTION_CMOS5L.get(tb_params["res_corner"], tb_params["res_corner"])
    tb_params["simpath"] = container_run_dir

    tb_text = substitute_params(tb_source.read_text(encoding="utf-8"), tb_params)
    check_unresolved(tb_text, f"{test_name} {condition_label(conditions)} ({tb_source.name})")
    (run_dir / tb_source.name).write_text(tb_text, encoding="utf-8")

    # cd to the RCFILE's own directory before invoking xschem: the
    # project's xschemrc builds XSCHEM_LIBRARY_PATH with `append
    # XSCHEM_LIBRARY_PATH :$env(PWD)` -- NOT "directory the rcfile lives
    # in" (that was this comment's own prior, WRONG assumption, and
    # materialize_variation_shadow()'s old docstring's too -- see
    # BUG_shadow_materialization_ignored.md). Every docker_exec into this
    # container otherwise inherits the SAME cwd (managed_container()'s own
    # `docker run -w <project_root>`), so $env(PWD) -- and therefore every
    # bare "sch/<name>" symbol/schematic reference -- ALWAYS resolved
    # against the shared project root before this fix, no matter which
    # --rcfile or shadow sch/ copy existed on disk: confirmed live, a
    # shadow-materialized override was silently netlisted against the
    # stale SHARED sch/<block>.sch instead. Explicitly cd-ing here makes
    # $env(PWD) match wherever container_rcfile actually lives (the shadow
    # sim/<variation>/_src/ for a shadow=True run, unchanged project root
    # otherwise), so XSCHEM_LIBRARY_PATH's :$env(PWD) entry -- and hence
    # every "sch/<name>" reference -- resolves from the RIGHT tree.
    rcfile_dir = container_rcfile.rsplit("/", 1)[0]
    # DISPLAY: the docker image's Xvnc (:1); in host mode whatever display
    # this machine has, or none -- xschem -x only netlists (see
    # executor.HostExecutor).
    display = f"export DISPLAY={ctx.display}; " if ctx.display else ""
    netlist_cmd = (
        f'cd "{rcfile_dir}" && {display}'
        f'"{ctx.xschem}" --rcfile "{container_rcfile}" '
        f'-n -x -q -o "{container_run_dir}" "{container_run_dir}/{tb_source.name}"'
    )
    netlist_path = run_dir / f"{tb_source.stem}.spice"
    # A leftover netlist (earlier run, or run_one_ngspice()'s own retry)
    # would satisfy the exists() check below even if xschem wrote nothing.
    netlist_path.unlink(missing_ok=True)
    for attempt in range(1, _NETLIST_MAX_ATTEMPTS + 1):
        result = docker_exec(container, netlist_cmd)
        if netlist_path.exists():
            missing = _missing_project_subckts(netlist_path.read_text(encoding="utf-8"))
            if not missing:
                break
            failure = (
                f"netlist is missing the .subckt expansion of {', '.join(sorted(missing))} "
                f"after {attempt} attempt(s) (see _missing_project_subckts())."
            )
            netlist_path.unlink()
        else:
            failure = (
                f"netlist failed, no .spice produced after {attempt} attempt(s) "
                f"(see _netlist()'s own comment on this being a known transient "
                f"xschem/concurrency flake, not a content problem)."
            )
        if attempt == _NETLIST_MAX_ATTEMPTS:
            return {"status": "error", "error": f"{failure}\n{result.stdout}\n{result.stderr}"}
        time.sleep(_NETLIST_RETRY_DELAY_SECONDS)
    netlist_text = netlist_path.read_text(encoding="utf-8")
    if "IS MISSING" in netlist_text:
        missing = [l for l in netlist_text.splitlines() if "IS MISSING" in l]
        return {"status": "error", "error": "netlist has unresolved symbols:\n" + "\n".join(missing)}
    return netlist_path


# Auxiliary simulator outputs -- every `write`/`wrdata` target in a
# testbench's .control block OTHER than the canonical '<test>_0.data' its
# parser reads (the `save all` .raw feeding the SOA check, debug dumps like
# tb_vref_startup's '_diag.data') -- are redirected by
# _redirect_aux_outputs() to container-local scratch under this root instead
# of the project bind mount. Nothing reads them after the run itself: the
# .raw is reduced to SOA peaks inside the container (raw_peaks.py) and the
# lazy plots (generate_plot()) replay only the .data. Measured on
# ihp_mh_ip__cmos_vref's startup test (2026-09-26): one condition's ASCII
# .raw is ~40MB, and writing it through Docker Desktop's bind mount took
# that run from 2.0s to 7.9s -- ~9GB across 174 variations, and the
# dominant I/O contention in a parallel batch. The container is --rm (see
# managed_container()), so anything a crashed run leaves here goes with it.
_AUX_SCRATCH_ROOT = "/tmp/analog_designer_aux"

# Set to a non-empty, non-"0" value to copy a run's auxiliary outputs back
# into its run dir anyway (debugging a testbench). They're always kept for a
# run whose ngspice exited non-zero.
KEEP_AUX_ENV = "ANALOG_DESIGNER_KEEP_SIM_AUX"

_AUX_OUTPUT_RE = re.compile(r"^(?P<cmd>write|wrdata)\s+(?P<path>\S+)(?P<rest>.*)$", re.IGNORECASE)
_SOA_PEAKS_MARKER = "@@analog_designer:soa_peaks@@"


def _keep_aux_outputs():
    return os.environ.get(KEEP_AUX_ENV, "") not in ("", "0")


def _redirect_aux_outputs(netlist_path, container_run_dir, scratch_dir, primary_name):
    """Rewrites every `write`/`wrdata` line targeting a file directly in
    container_run_dir, other than primary_name, to target scratch_dir
    instead (see _AUX_SCRATCH_ROOT). If any `write` (a .raw) moved, also
    flips `set filetype=ascii` to binary -- ~3x smaller and far cheaper to
    both write and parse, and raw_peaks.read_raw() reads either; the ASCII
    setting only existed for the old host-side reader. Returns
    {"write": [names], "wrdata": [names]} of what moved."""
    lines = netlist_path.read_text(encoding="utf-8").splitlines()
    prefix = container_run_dir.rstrip("/") + "/"
    moved = {"write": [], "wrdata": []}
    for i, line in enumerate(lines):
        match = _AUX_OUTPUT_RE.match(line.strip())
        if not match or not match.group("path").startswith(prefix):
            continue
        name = match.group("path")[len(prefix):]
        if name == primary_name or "/" in name:
            continue
        lines[i] = f"{match.group('cmd')} {scratch_dir}/{name}{match.group('rest')}"
        moved[match.group("cmd").lower()].append(name)
    if not (moved["write"] or moved["wrdata"]):
        return moved
    if moved["write"]:
        lines = ["set filetype=binary" if l.strip().lower() == "set filetype=ascii" else l for l in lines]
    netlist_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return moved


def _soa_setup(netlist_path):
    """(devices, limits, pairs) for the SOA (Safe Operating Area) check of
    one run -- see soa_check.py's own docstring for why this reads real
    node-voltage waveforms instead of the earlier PSP103-model-warning-based
    approach. None if config.json has no technology.mosfet_limits to compare
    against."""
    limits = (workspace.CONFIG or {}).get("technology", {}).get("mosfet_limits")
    if not limits:
        return None
    netlist_text = netlist_path.read_text(encoding="utf-8")
    devices = spice_devices.extract_mosfets(netlist_text)
    subckt_name = spice_devices.find_dut_subckt(netlist_text)
    if subckt_name:
        devices = spice_devices.resolve_hierarchical_nets(netlist_text, devices, subckt_name)
    return devices, limits, soa_check.soa_pairs(devices, limits)


def _soa_peaks_command(raw_path, pairs):
    """Shell snippet (appended to the ngspice exec) that runs raw_peaks.py
    against raw_path inside the container and prints its JSON after
    _SOA_PEAKS_MARKER -- the script's own source travels in a heredoc, so
    nothing needs installing in the image beyond python3."""
    source = Path(raw_peaks.__file__).read_text(encoding="utf-8")
    return (
        f'if [ -f "{raw_path}" ]; then echo "{_SOA_PEAKS_MARKER}"; '
        f'python3 - "{raw_path}" {shlex.quote(json.dumps(pairs))} 2>&1 <<\'AD_RAW_PEAKS_PY\'\n'
        f"{source}\nAD_RAW_PEAKS_PY\nfi"
    )


def _soa_diagnostics(soa, peaks_output):
    """Diagnostics from raw_peaks.py's in-container output. A reducer
    failure (no python3 in some other image, a malformed .raw) becomes one
    visible warning rather than silently skipping the check."""
    devices, limits, _ = soa
    try:
        peaks = {(a, b): peak for a, b, peak in json.loads(peaks_output)}
    except (ValueError, TypeError):
        first_line = (peaks_output.strip().splitlines() or ["no output"])[-1]
        return [{
            "severity": "warning", "category": "soa_check_failed", "key": "soa_check_failed",
            "message": f"SOA check could not read the .raw: {first_line}", "count": 1,
        }]
    return soa_check.check_soa(devices, peaks, limits)


def run_one_ngspice(container, test_name, tb_source, conditions, tb_params_base,
                     run_dir, container_run_dir, container_rcfile, ctx,
                     block_cfg=None, block=None, topology=None, sim_timeout=290, n_threads=1):
    """One ngspice condition: netlist + simulate, retried once when ngspice
    reports an "unknown subckt" -- xschem intermittently emits a netlist
    with a hierarchical block's .subckt expansion silently absent (~0.7% of
    all runs across ihp_mh_ip__cmos_vref's history, spread over every test,
    never reproducible on its own -- 84/84 clean under 28-way concurrent
    netlisting, 2026-09-26), which _netlist()'s "IS MISSING" check can't
    see. A genuinely missing subckt just fails the same way twice."""
    args = (container, test_name, tb_source, conditions, tb_params_base,
            run_dir, container_run_dir, container_rcfile, ctx, sim_timeout, n_threads)
    outcome = _run_one_ngspice_attempt(*args)
    if outcome["status"] == "error" and "unknown subckt" in (outcome.get("error") or ""):
        outcome = _run_one_ngspice_attempt(*args)
    return outcome


def _run_one_ngspice_attempt(container, test_name, tb_source, conditions, tb_params_base,
                             run_dir, container_run_dir, container_rcfile, ctx, sim_timeout, n_threads):
    run_dir.mkdir(parents=True, exist_ok=True)
    # `set num_threads=<n_threads>` here is what actually governs this run's
    # thread count (see THREAD_POLICY/run_test()): read AFTER the
    # container's own global spinit script (which _cap_ngspice_threads()
    # already pinned to num_threads=1 as a safety floor), so this line's
    # value wins for ngspice's own OpenMP call, per-run, exactly matching
    # whatever workspace.core_pool().reserve() granted this particular job.
    spiceinit_text = ctx.spiceinit_text + f"\nset num_threads={n_threads}\n"
    (run_dir / ".spiceinit").write_text(spiceinit_text, encoding="utf-8")

    netlist_result = _netlist(container, test_name, tb_source, conditions, tb_params_base,
                               run_dir, container_run_dir, container_rcfile, ctx)
    if isinstance(netlist_result, dict):
        return netlist_result
    netlist_path = netlist_result

    # Delete any stale <test>_0.data left over from a PREVIOUS run of this
    # exact (variation, test, condition) before simulating -- the success
    # check below only verifies the file EXISTS afterward, which a broken
    # testbench (e.g. a .control block's wrdata referencing a vector that
    # no longer resolves, confirmed live: 5 top-level testbenches silently
    # kept "succeeding" this way for days after a wire lost its explicit
    # net label, since ngspice itself still exits 0 after printing "Error:
    # no such vector ..." to its own log) can satisfy by simply leaving an
    # OLD, unrelated-to-this-run file untouched. Without this, a genuinely
    # broken re-simulation of an EXISTING variation is indistinguishable
    # from a real success; only a variation simulated for the first time
    # (no stale file to fall back on) was ever failing loudly.
    data_file = run_dir / f"{test_name}_0.data"
    data_file.unlink(missing_ok=True)

    # Unique per (variation, test, condition) since it mirrors the run
    # dir's own container path -- concurrent conditions never share one.
    scratch_dir = f"{_AUX_SCRATCH_ROOT}/{container_run_dir.lstrip('/')}"
    aux = _redirect_aux_outputs(netlist_path, container_run_dir, scratch_dir, data_file.name)
    soa_raw = f"{test_name}_0.raw"
    soa = _soa_setup(netlist_path) if soa_raw in aux["write"] else None

    # `timeout` runs INSIDE the container so a non-converging ngspice run
    # actually gets killed (SIGTERM, then SIGKILL if it ignores that) --
    # docker_exec()'s own timeout= only kills the local `docker exec`
    # client on this side, never the process it started inside the
    # container, which would otherwise keep consuming CPU/RAM for the rest
    # of a long batch (especially bad with a shared container across
    # several concurrent variations, see gen_variations._run_batch). Set a
    # few seconds under docker_exec()'s own timeout so this fires first in
    # the normal case; that outer timeout is just a fallback for docker
    # exec itself hanging (e.g. a stalled docker daemon), not the usual
    # path. sim_timeout defaults to the real-run value (290s) for every
    # existing caller (run_test()); analog_designer/sim/diagnose_tb.py is
    # the only caller that ever shortens it, to scan for slow/non-converging
    # (test, condition) combinations faster than waiting out a full timeout
    # on each one.
    # export OMP_NUM_THREADS=<n_threads>: belt-and-suspenders alongside the
    # .spiceinit override above -- redundant for ngspice itself (whose OWN
    # `set num_threads` call already wins over the env var, see
    # _cap_ngspice_threads()'s docstring), but keeps this exec's declared
    # thread budget visible/correct for anything else in the process tree
    # that DOES read OMP_NUM_THREADS normally (e.g. a BLAS library).
    # Auxiliary outputs (see _AUX_SCRATCH_ROOT) are reduced and cleaned up
    # in this SAME exec, right after ngspice -- one docker exec per
    # condition, same as before they existed; `exit $rc` keeps ngspice's
    # own exit code as the exec's.
    sim_cmd = (
        f'cd "{container_run_dir}" && export OMP_NUM_THREADS={n_threads}; '
        f'timeout {sim_timeout} "{ctx.ngspice}" -b {tb_source.stem}.spice'
    )
    if aux["write"] or aux["wrdata"]:
        keep_back = "true" if _keep_aux_outputs() else '[ "$rc" -ne 0 ]'
        sim_cmd = "\n".join(filter(None, [
            f'mkdir -p "{scratch_dir}"',
            sim_cmd,
            "rc=$?",
            _soa_peaks_command(f"{scratch_dir}/{soa_raw}", soa[2]) if soa else None,
            f'if {keep_back}; then cp -r "{scratch_dir}/." "{container_run_dir}/"; fi',
            f'rm -rf "{scratch_dir}"',
            "exit $rc",
        ]))
    sim_result = docker_exec(container, sim_cmd, timeout=sim_timeout + 30)
    stdout, _, soa_output = sim_result.stdout.partition(_SOA_PEAKS_MARKER + "\n")
    log_text = stdout + "\n" + sim_result.stderr
    (run_dir / "ngspice.log").write_text(log_text, encoding="utf-8")
    diagnostics = log_diagnostics.parse(log_text, "ngspice")
    if soa and soa_output:
        diagnostics += _soa_diagnostics(soa, soa_output)
    # returncode/file-existence alone isn't enough: ngspice can exit 0 and
    # still leave behind a data file after an analysis-level failure --
    # confirmed live for two distinct real cases, both already caught by
    # log_diagnostics.parse() as severity="error" but previously never
    # consulted here: (1) "Error: no such vector ..." from a testbench
    # whose wrdata references a net that no longer resolves (see this
    # function's own comment above, by the data_file.unlink() call, on why
    # that used to be invisible on a variation's first-ever run), and (2)
    # a .tran whose adaptive timestep collapses on a misbehaving nonlinear
    # element, printing "tran simulation(s) aborted" after writing only a
    # handful of rows instead of the full sweep (gf180mcu_mh_ip__nfrac_pll's
    # tb_startup.sch, 2026-09-09 -- see log_diagnostics.py's own
    # _NGSPICE_ABORTED_RE/_NGSPICE_TIMESTEP_RE comments). Either way the
    # data file is present but meaningless, so any error-severity
    # diagnostic now fails the run outright instead of silently becoming a
    # near-empty plot downstream with nothing flagging why.
    has_error_diagnostic = any(d["severity"] == "error" for d in diagnostics)
    if sim_result.returncode != 0 or not data_file.exists() or has_error_diagnostic:
        return {
            "status": "error",
            "ngspice_exit_code": sim_result.returncode,
            "error": log_text[-2000:],
            "diagnostics": diagnostics,
        }
    return {
        "status": "success", "ngspice_exit_code": sim_result.returncode, "data_file": data_file,
        "diagnostics": diagnostics,
    }


def run_one_netlist(container, test_name, tb_source, conditions, tb_params_base,
                     run_dir, container_run_dir, container_rcfile, ctx,
                     block_cfg=None, block=None, topology=None, sim_timeout=290, n_threads=1):
    """Static counterpart to run_one_ngspice: netlists the testbench through
    xschem exactly the same way, but never invokes ngspice -- for tests
    whose parser reads geometry/structure straight out of the expanded
    .spice text (area estimation, e.g.) instead of simulation output.
    sim_timeout/n_threads/block_cfg/block/topology are accepted only to
    keep SIMULATOR_RUNNERS' entries call-compatible; unused here since
    nothing simulates (see THREAD_POLICY's own "netlist" entry,
    minimum=preferred=0). ctx IS used, for _netlist()'s own
    mos_corner_section lookup."""
    netlist_result = _netlist(container, test_name, tb_source, conditions, tb_params_base,
                               run_dir, container_run_dir, container_rcfile, ctx)
    if isinstance(netlist_result, dict):
        return netlist_result
    return {"status": "success", "data_file": netlist_result}


# ONE combined plugin (PSP103 + r3_cmc + mosvar built together), not one
# .so per model -- confirmed live (2026-09-05) that loading more than one
# independently-built ADMS plugin into the same Xyce process (-plugin a.so
# -plugin b.so) corrupts PSP103's device parameter/terminal interface (the
# same "Unrecognized parameter D for device YPSP103_VA!M1" symptom the
# buildxyceplugin DESTDIR fix, see eda-env/docker/ihp-open-pdk/install.sh,
# was chasing -- but a DIFFERENT bug: reproduces even with fresh,
# correctly-isolated single-model plugins). eda-env's own install.sh builds
# each PDK's Verilog-A models into exactly this one combined plugin instead
# of three separate ones -- see that script's own comment for how this was
# diagnosed and confirmed fixed (a PSP103 MOSFET + an r3_cmc-based rhigh
# resistor in one netlist, e.g. ihp_mh_ip__cmos_vref's "top" block,
# converges cleanly through it). Named per-PDK ($PDK with "-" -> "_", same
# transform eda-env's install.sh applies) since a project's container.image
# (config.json) selects which PDK's plugin actually exists in the image.
def xyce_plugin_so(pdk_name):
    return f"libXyce_Plugin_{pdk_name.replace('-', '_')}.so"


_CMOMF_LINE_RE = re.compile(r"^(?P<name>[Xx]\S+)\s+(?P<plus>\S+)\s+(?P<minus>\S+)\s+cap_cmomf\s+(?P<params>.*)$")


_CMOM_INSTANCE_RE = re.compile(r"^[Xx]\S+\s+.*\bcap_cmom[fi]\b")
_CORNERCAP_LIB_RE = re.compile(r"^\.lib\s+\S*cornerCAP\.lib\b", re.IGNORECASE)


def _lower_cmomf_for_xyce(netlist_path):
    """Rewrite every `cap_cmomf` instance in an Xyce netlist as an ideal
    capacitor of the same value. The IHP CMOS5L cap_cmomf is a Verilog-A/OSDI
    model that only ngspice loads (Xyce's combined plugin is PSP103 + r3_cmc
    + mosvar), but the device is a pure low-frequency capacitance --
    C = m * areacap(mmin, mmax) * w * l, no series R/L, no substrate node --
    so an ideal C is an exact equivalent (formula from the PDK's own
    cap_cmomf.lib header, checked against ngspice's model: M1-M4 =
    1.287 fF/um^2, M2-M3 = 0.61 fF/um^2):
        areacap = base + (mmax - mmin) * 0.305  [fF/um^2],
        base = 0.372 if mmin == 1 else 0.305."""
    lines = netlist_path.read_text(encoding="utf-8").splitlines()
    changed = False
    for i, line in enumerate(lines):
        match = _CMOMF_LINE_RE.match(line)
        if not match:
            continue
        kv = dict(item.split("=", 1) for item in match.group("params").split() if "=" in item)
        w_um = parse_spice_value(kv["w"]) * 1e6
        l_um = parse_spice_value(kv["l"]) * 1e6
        mmin, mmax = int(float(kv.get("mmin", 1))), int(float(kv.get("mmax", 4)))
        m = float(kv.get("m", 1))
        areacap = (0.372 if mmin == 1 else 0.305) + (mmax - mmin) * 0.305
        farads = m * areacap * 1e-15 * w_um * l_um
        lines[i] = f"C_{match.group('name')} {match.group('plus')} {match.group('minus')} {farads:.6e}"
        changed = True
    if changed:
        # Xyce's PDK mirror ships no cornerCAP.lib (it only holds the OSDI
        # cap models' cards), and nothing left in the netlist needs it.
        if not any(_CMOM_INSTANCE_RE.match(line) for line in lines):
            lines = [line for line in lines if not _CORNERCAP_LIB_RE.match(line)]
        netlist_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_one_xyce(container, test_name, tb_source, conditions, tb_params_base,
                  run_dir, container_run_dir, container_rcfile, ctx,
                  block_cfg=None, block=None, topology=None, sim_timeout=290, n_threads=1):
    run_dir.mkdir(parents=True, exist_ok=True)
    # No .spiceinit equivalent for Xyce -- plugins load via -plugin on the
    # command line below, not a container-cwd init file ngspice relies on.

    netlist_result = _netlist(container, test_name, tb_source, conditions, tb_params_base,
                               run_dir, container_run_dir, container_rcfile, ctx)
    if isinstance(netlist_result, dict):
        return netlist_result
    netlist_path = netlist_result
    _strip_ngspice_save_lines(netlist_path)
    _lower_cmomf_for_xyce(netlist_path)

    # See run_one_ngspice's own identical line for why this must happen
    # before simulating, not just be checked for existence afterward.
    data_file = run_dir / f"{test_name}_0.data"
    data_file.unlink(missing_ok=True)

    # ctx.xyce_plugins_dir is None for GF180MCU: its BSIM4 devices are
    # native to Xyce, not a compiled Verilog-A plugin like IHP's PSP103 --
    # xyce_plugin_so(ctx.pdk_name) would just name a .so that was never
    # built for this PDK family.
    plugin_flag = f'-plugin "{ctx.xyce_plugins_dir}/{xyce_plugin_so(ctx.pdk_name)}" ' if ctx.xyce_plugins_dir else ""
    # Same server-side-timeout pattern as run_one_ngspice (see its comment
    # above): `timeout` runs INSIDE the container so a non-converging/hung
    # Xyce run is actually killed, not left running in a shared batch
    # container.
    # export OMP_NUM_THREADS=<n_threads>: today's eda-env Xyce build is
    # serial (no MPI, no explicit OpenMP -- see its own install script), so
    # this has no observable effect yet, but costs nothing and keeps Xyce
    # runs governed by the same per-run thread budget as ngspice/openEMS if
    # a future build ever links a threaded BLAS underneath it.
    sim_cmd = (
        f'cd "{container_run_dir}" && export OMP_NUM_THREADS={n_threads}; '
        f'timeout {sim_timeout} "{ctx.xyce}" {plugin_flag}{tb_source.stem}.spice'
    )
    sim_result = docker_exec(container, sim_cmd, timeout=sim_timeout + 10)
    log_text = sim_result.stdout + "\n" + sim_result.stderr
    (run_dir / "xyce.log").write_text(log_text, encoding="utf-8")
    diagnostics = log_diagnostics.parse(log_text, "xyce")

    if sim_result.returncode != 0 or not data_file.exists():
        return {
            "status": "error",
            "xyce_exit_code": sim_result.returncode,
            "error": log_text[-2000:],
            "diagnostics": diagnostics,
        }
    _strip_xyce_print_header(data_file)
    return {
        "status": "success", "xyce_exit_code": sim_result.returncode, "data_file": data_file,
        "diagnostics": diagnostics,
    }


def _strip_ngspice_save_lines(netlist_path):
    """xschem's devices/ammeter.sym emits a bare `.save i(<name>)` line for
    every ammeter in a schematic (cmos_vref.sch's Vmeas_ana/Vm_b1/etc.,
    needed by OTHER, ngspice-only tests like current_consumption) --
    ngspice-specific syntax to mark a source current for availability.
    Xyce doesn't understand this form ("Unrecognized field in .SAVE line",
    confirmed live) and doesn't need it: Xyce can print I(<source>) in a
    .PRINT/.NOISE line with no .save declaration at all. Since _netlist()
    is shared, simulator-agnostic infrastructure (the .sch files themselves
    have no simulator awareness), neutralizing this ngspice-only line is
    run_one_xyce()'s job, not _netlist()'s -- same category of fixup as
    _strip_xyce_print_header, just on the way in instead of the way out."""
    text = netlist_path.read_text(encoding="utf-8")
    lines = [
        line for line in text.splitlines()
        if line.strip().lower().split() and line.strip().lower().split()[0] != ".save"
    ]
    netlist_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _strip_xyce_print_header(data_file):
    """Xyce's `.print ... file=...` always writes one header line (signal
    names) and always prepends an Index column before whatever signals were
    actually requested -- confirmed general to Xyce's .print output, not
    noise-specific, so this normalization applies unconditionally to any
    future Xyce-backed test's output, regardless of which/how-many signals
    its own .print line requests. parser_common.read_data() is a naive
    whitespace-float-table reader with zero header tolerance, so this must
    run before data_file is handed back as a success result. Drops any
    trailing line that doesn't parse as an all-float row (defensive against
    a stray blank/summary line) rather than assuming an exact row count."""
    lines = data_file.read_text(encoding="utf-8").splitlines()
    out_lines = []
    for line in lines[1:]:  # drop the header line unconditionally
        tokens = line.split()[1:]  # drop the leading Index column
        if not tokens:
            continue
        try:
            [float(t) for t in tokens]
        except ValueError:
            continue
        out_lines.append(" ".join(tokens))
    data_file.write_text("\n".join(out_lines) + "\n", encoding="utf-8")


_OPENEMS_F_MAX_HZ = 20e9
_OPENEMS_N_FREQ = 201


def _openems_placeholder_cache_doc(generator, geometry, corner="tt", n_freq=_OPENEMS_N_FREQ, f_max_hz=_OPENEMS_F_MAX_HZ):
    """Fabricates a cache_doc in the exact shape run_one_openems() writes
    for a real run (freqs_hz/re/im/q/srf_ghz/peak_q/peak_q_freq_ghz) --
    used only under ANALOG_DESIGNER_OPENEMS_PLACEHOLDER, see
    run_one_openems()'s own docstring. Derives Y11(f) from the SAME simple
    series-RL model fit_electrical_params() itself assumes at low frequency
    (Y11 = 1/(Rs + jwL)), using its own placeholder-path l/rs -- so this
    curve is smoothly inductive (monotonically rising Q, no SRF within the
    swept range, same as a real un-fitted low-loss inductor's own low-
    frequency behavior would look before self-resonance), plausible enough
    to exercise plots/parsers without ever being mistaken for FDTD-accurate
    data (peak_q's own value alone -- a bare series-RL model's Q keeps
    rising forever, never actually peaking -- is a tell an FDTD run's own
    result never has).

    `generator` is the already-loaded module (see run_one_openems -- the
    caller loads it once and passes it in here, rather than this helper
    re-resolving block_cfg["generator"] itself)."""
    import math

    stack = generator.load_stack(corner)
    fitted = generator.fit_electrical_params(geometry, stack, em_result=None)
    rs_total = 2 * fitted["rs"]
    l_total = 2 * fitted["l"]

    freqs = [1e6 + i * (f_max_hz - 1e6) / (n_freq - 1) for i in range(n_freq)]
    re, im, q = [], [], []
    for f in freqs:
        w = 2 * math.pi * f
        denom = complex(rs_total, w * l_total)
        y11 = 1.0 / denom
        re.append(y11.real)
        im.append(y11.imag)
        q.append(-y11.imag / y11.real)

    return {
        "geometry": geometry,
        "freqs_hz": freqs, "re": re, "im": im, "q": q,
        "srf_ghz": None, "peak_q": q[-1], "peak_q_freq_ghz": freqs[-1] / 1e9,
    }


def run_one_openems(container, test_name, tb_source, conditions, tb_params_base,
                     run_dir, container_run_dir, container_rcfile, ctx,
                     block_cfg=None, block=None, topology=None, sim_timeout=172800, n_threads=1):
    """Runner for "generator"-backed topologies (e.g. inductor.spiral) --
    there is no xschem/.sch testbench at all (`tb_source` is unused; this
    test's config.json entry has no "testbench" key), the openEMS FDTD run
    itself IS the measurement. Self-contained caching keyed by a hash of the
    resolved geometry params (`tb_params_base`, the same resolved-parameter
    dict every other runner already receives): a FDTD run costs hours, so a
    second variation with identical geometry must not re-run it.

    sim_timeout default is 172800s (48h), not the 290s every other runner
    here uses -- bumped from an earlier 36000s (10h) default 2026-09-16
    after it killed a real, cleanly-converging 40x10um/1um loop run mid-
    flight (reached -20.47dB of the required -40dB EndCriteria, energy
    still trending down, no instability) purely because it needed more
    than 10h -- ~10h of compute lost for nothing (the raw port_it_*/
    port_ut_* time-domain files survived and were separately recovered via
    a direct CalcPort() call, skipping FDTD.Run() entirely, but that's a
    manual recovery path, not something this runner does on its own).
    Never passed explicitly by any call site (run_test()'s own `runner(...)`
    call has no sim_timeout kwarg), so this default IS the effective value
    for every real run -- keep that in mind if tuning it again.

    The generator script itself is project-specific GF180MCU code, so it
    lives in the OPEN PROJECT's own repo (block_cfg["generator"], a
    project-repo-relative file path -- same convention as "parser"/
    "testbench") rather than in this shared tool. `block_cfg` is threaded
    in from run_variation() via run_test() specifically so this generic
    path resolution works for ANY generator-backed block/topology, not
    just this one -- `block`/`topology` are threaded the same way, purely
    to keep the on-disk cache directory generic too
    (sim/_generator_cache/<block>/<topology>/) instead of a hardcoded
    "inductor"/"spiral_openems" pair.

    The actual FDTD run happens inside the container via a GENERIC script
    this tool owns, openems_generator_runner.py -- copied into run_dir
    (already mounted in-container at container_run_dir, so no extra bind-
    mount/PYTHONPATH is needed, unlike an earlier version that hardcoded a
    dotted `analog_designer_pro.modeling.*` module path -- see
    managed_container() git history if that's ever relevant again) and
    invoked there by container-relative file path. That script dynamically
    loads the SAME project generator module and does the actual
    `FDTD.Run()`/`CalcPort()`/Y11-Q-SRF extraction (openEMS-specific, not
    PDK-specific, so it doesn't belong in the project's own generator.py
    at all) before calling the generator's own fit_electrical_params()
    with the real result.

    ANALOG_DESIGNER_OPENEMS_PLACEHOLDER=1 (env var, checked below): skips
    the real (hours-long) FDTD run entirely (no docker_exec at all) and
    writes a placeholder Y11(f) sweep instead (see
    _openems_placeholder_cache_doc()) -- for exercising this runner/the
    parser/the GUI's plots end to end without paying for openEMS.
    Deliberately an env var, not a config.json or GUI toggle: opt-in per
    shell session, impossible to leave silently enabled inside a project's
    own checked-in config where it could taint a real characterization run
    without anyone noticing."""
    run_dir.mkdir(parents=True, exist_ok=True)
    generator = _load_generator_module(block_cfg)
    geometry = generator.geometry_from_params(tb_params_base)

    cache_key = hashlib.sha1(json.dumps(geometry, sort_keys=True).encode()).hexdigest()[:16]
    cache_dir = workspace.sim_root() / "_generator_cache" / block / topology
    cache_file = cache_dir / f"{cache_key}.json"
    cache_field_png = cache_dir / f"{cache_key}__field.png"
    data_file = run_dir / f"{test_name}_0.json"
    # tb_yparam_spiral.py's extract() derives this same "__field.png"
    # sibling name from data_path itself -- keep the two conventions in
    # sync if either ever changes.
    data_field_png = run_dir / f"{test_name}_0__field.png"
    # Per-layer dumps (metal4/via4/substrate/...), 2026-09-15: same
    # cache_dir/run_dir pairing as the single field.png above, but keyed by
    # whatever labels the generator's own FIELD_DUMP_NAMES declares (see
    # inductor_spiral_generator.py) -- "__field_<label>.png" siblings,
    # matched by tb_yparam_spiral.py's extract() globbing for that pattern.
    cache_field_png_dir = cache_dir / f"{cache_key}__field_dumps"

    if cache_file.exists():
        data_file.write_text(cache_file.read_text(encoding="utf-8"), encoding="utf-8")
        if cache_field_png.exists():
            data_field_png.write_bytes(cache_field_png.read_bytes())
        if cache_field_png_dir.is_dir():
            for cached_png in cache_field_png_dir.glob("*.png"):
                (run_dir / f"{test_name}_0__field_{cached_png.stem}.png").write_bytes(cached_png.read_bytes())
        return {"status": "success", "data_file": data_file, "diagnostics": []}

    cache_dir.mkdir(parents=True, exist_ok=True)

    if os.environ.get("ANALOG_DESIGNER_OPENEMS_PLACEHOLDER"):
        # No real FDTD run happens in placeholder mode, so there's no
        # field dump to render either -- tb_yparam_spiral.py's
        # _save_field_plot() falls back to its own "not available"
        # placeholder text when data_field_png doesn't exist, same as it
        # already does before any run has ever happened at all.
        cache_doc = _openems_placeholder_cache_doc(generator, geometry)
        cache_text = json.dumps(cache_doc)
        cache_file.write_text(cache_text, encoding="utf-8")
        data_file.write_text(cache_text, encoding="utf-8")
        return {"status": "success", "data_file": data_file, "diagnostics": []}

    runner_src = Path(__file__).with_name("openems_generator_runner.py")
    (run_dir / "openems_generator_runner.py").write_bytes(runner_src.read_bytes())
    container_runner_py = f"{container_run_dir}/openems_generator_runner.py"
    container_generator_py = workspace.exec_path(workspace.PROJECT_ROOT / block_cfg["generator"])
    container_geometry_json = f"{container_run_dir}/geometry.json"
    container_params_json = f"{container_run_dir}/fitted.params.json"
    container_cache_json = f"{container_run_dir}/result.json"
    container_field_png = f"{container_run_dir}/field.png"
    container_field_png_dir = f"{container_run_dir}/field_dumps"
    (run_dir / "geometry.json").write_text(json.dumps(geometry), encoding="utf-8")

    # Redirected straight to a file INSIDE container_run_dir (already
    # bind-mounted at run_dir on the host) instead of letting docker_exec()
    # capture it -- docker_exec()'s own subprocess.run(capture_output=True)
    # only hands stdout/stderr back once the ENTIRE command exits, so for
    # every OTHER simulator (seconds-scale runs) that's invisible, but for
    # a multi-hour FDTD run it meant openems_run.log plain didn't exist on
    # disk at all until the run was already over -- exactly what someone
    # tailing it mid-run (the same "periodic energy checks" practice this
    # project's own standalone diagnostic scripts already established) was
    # missing. `stdbuf -oL -eL` forces line-buffered output even though
    # stdout/stderr are no longer a TTY once redirected to a file, so each
    # openEMS "[@ ...] Timestep: ... Energy: ...dB" line lands on disk as
    # it's printed, not batched behind libc's own full-buffering default
    # for non-TTY output.
    container_log = f"{container_run_dir}/openems_run.log"
    sim_cmd = (
        f'mkdir -p "{container_run_dir}" && '
        f'export OMP_NUM_THREADS={n_threads} && '
        f'stdbuf -oL -eL timeout {sim_timeout} python3 "{container_runner_py}" '
        f'--generator "{container_generator_py}" --geometry-json "{container_geometry_json}" '
        f'--corner tt --f-max {_OPENEMS_F_MAX_HZ} '
        f'--out "{container_params_json}" --cache-out "{container_cache_json}" '
        f'--field-png "{container_field_png}" --field-png-dir "{container_field_png_dir}" '
        f'> "{container_log}" 2>&1'
    )
    sim_result = docker_exec(container, sim_cmd, timeout=sim_timeout + 30)
    log_path = run_dir / "openems_run.log"
    if log_path.exists():
        log_text = log_path.read_text(encoding="utf-8", errors="replace")
    else:
        # Only happens if the command never reached the redirected part at
        # all (e.g. `mkdir -p` itself failed) -- fall back to whatever
        # bash's own stdout/stderr captured, and make sure a log file
        # exists either way so callers never have to special-case this.
        log_text = sim_result.stdout + "\n" + sim_result.stderr
        log_path.write_text(log_text, encoding="utf-8")

    out_json = run_dir / "result.json"
    if sim_result.returncode != 0 or not out_json.exists():
        return {
            "status": "error", "openems_exit_code": sim_result.returncode,
            "error": log_text[-2000:], "diagnostics": [],
        }
    cache_file.write_text(out_json.read_text(encoding="utf-8"), encoding="utf-8")
    data_file.write_text(out_json.read_text(encoding="utf-8"), encoding="utf-8")
    # The field PNG is genuinely optional (render_field_dump() is
    # best-effort inside openems_generator_runner.py -- see its own
    # comment on why a dump/render failure must never take the real Y11
    # result down with it), so its absence here is expected, not an error.
    out_field_png = run_dir / "field.png"
    if out_field_png.exists():
        cache_field_png.write_bytes(out_field_png.read_bytes())
        data_field_png.write_bytes(out_field_png.read_bytes())
    # Per-layer dumps (see container_field_png_dir above) -- same
    # optional/best-effort posture as the single field.png: openems_generator_runner.py
    # already skipped/logged anything that failed to render, so whatever
    # shows up here (0 or more PNGs) is exactly what's available.
    out_field_png_dir = run_dir / "field_dumps"
    if out_field_png_dir.is_dir():
        cache_field_png_dir.mkdir(parents=True, exist_ok=True)
        for rendered_png in out_field_png_dir.glob("*.png"):
            cache_field_png_dir.joinpath(rendered_png.name).write_bytes(rendered_png.read_bytes())
            (run_dir / f"{test_name}_0__field_{rendered_png.stem}.png").write_bytes(rendered_png.read_bytes())
    return {"status": "success", "data_file": data_file, "diagnostics": []}


SIMULATOR_RUNNERS = {
    "ngspice": run_one_ngspice, "netlist": run_one_netlist, "xyce": run_one_xyce,
    "openems": run_one_openems,
}

# How many CPU cores each simulator's OWN invocation should ask
# workspace.core_pool() for (see CpuBudget.reserve()): "minimum" is the
# least this job can usefully run with (reserve() blocks only while fewer
# than this are free), "preferred" is the most it can usefully use (it
# gets this many only if that many happen to be free right now -- see
# reserve()'s own adaptive-grant docstring). ngspice/Xyce condition runs
# are cheap and want to fan out as widely as possible, so both are 1;
# "netlist" never simulates at all (see run_one_netlist), so it doesn't
# need to wait on the budget for anything. "openems" wants the opposite
# shape: a small "minimum" (it still has to run even when busy) but a
# large "preferred", `max(1, workspace.core_pool().total() - 2)`, since a
# single FDTD run benefits from nearly the whole machine when nothing else
# is competing for it.
THREAD_POLICY = {
    "ngspice": {"minimum": 1, "preferred": 1},
    "xyce": {"minimum": 1, "preferred": 1},
    "netlist": {"minimum": 0, "preferred": 0},
    "openems": {"minimum": 1, "preferred": max(1, workspace.core_pool().total() - 2)},
}


def run_test(container, variation, test_name, test_cfg, defaults, ctx,
             sim_dir, container_sim_dir, container_rcfile, block_params=None,
             block_cfg=None, block=None, topology=None, where=None):
    simulator = test_cfg.get("simulator", "ngspice")
    runner = SIMULATOR_RUNNERS.get(simulator)
    if runner is None:
        sys.exit(
            f"{test_name}: simulator {simulator!r} is not implemented "
            f"(available: {sorted(SIMULATOR_RUNNERS)})"
        )
    thread_policy = THREAD_POLICY.get(simulator, {"minimum": 1, "preferred": 1})

    tb_source = workspace.PROJECT_ROOT / test_cfg["testbench"]
    tb_text = tb_source.read_text(encoding="utf-8")
    sweep_axis = internal_sweep_axis(test_cfg, tb_text)

    # Every declared default is a valid testbench placeholder under its own
    # name (e.g. output_amp's testbenches reference 'vdd', 'amp_vcm', 'ibias'
    # directly) -- "Vavdd" stays a separate alias for defaults["vdd"] since
    # cmos_vref's existing testbenches were authored against that name
    # instead ('avdd' is reserved for the top-level supply split, 'vdd' is
    # the block-scope convention). A test's own conditions{} can still
    # override any of these below via fixed_tb_params().
    tb_params_base = dict(defaults)
    tb_params_base["Vavdd"] = defaults["vdd"]
    tb_params_base["filename"] = test_name
    tb_params_base["N"] = "0"
    # Xyce's PDK model-lib mirror is structurally different from ngspice's
    # even though filenames match (see ContainerCtx/setup_container()), so
    # this must track the test's own declared simulator, not always ngspice's.
    tb_params_base["models_dir"] = ctx.xyce_models_dir if simulator == "xyce" else ctx.models_dir
    tb_params_base["stdcell_dir"] = ctx.stdcell_dir
    if simulator == "openems":
        # run_one_openems() needs the BLOCK's own free parameters (e.g.
        # inner_radius_um/n_turns/track_width_um/spacing_um), not testbench
        # placeholders sourced from config.json's global `defaults` -- every
        # other simulator gets its block-specific values already baked into
        # the materialized sch/<block>.sch (substitute_params(), see
        # run_variation()) instead, so this merge is openems-only, not a
        # general tb_params_base <- block_params channel.
        tb_params_base.update(block_params or {})
    if sweep_axis:
        key, prefix = sweep_axis
        values = [float(v) for v in test_cfg.get("conditions", {}).get(key, [])]
        if not values:
            sys.exit(f"{test_name}: testbench sweeps '{prefix}' internally but config.json has no conditions.{key} list")
        tb_params_base[f"{prefix}_min"] = min(values)
        tb_params_base[f"{prefix}_max"] = max(values)
    tb_params_base.update(fixed_tb_params(test_cfg, tb_text, sweep_axis))
    # A single-valued conditions.vdd (e.g. the informational *_1v8 twins of a
    # 3v3 project) must actually reach the testbench: fixed_tb_params() skips
    # "vdd" (see _NON_FIXED_CONDITION_KEYS), and cmos_vref's testbenches read
    # the 'Vavdd' alias, which is otherwise frozen to defaults["vdd"]. A
    # multi-valued list is either the internal sweep axis (line_reg's
    # vdd_min/vdd_max, handled above) or an outer axis (see _netlist()).
    vdd_values = test_cfg.get("conditions", {}).get("vdd")
    if vdd_values and len(vdd_values) == 1 and not (sweep_axis and sweep_axis[0] == "vdd"):
        tb_params_base["vdd"] = tb_params_base["Vavdd"] = vdd_values[0]

    test_dir = sim_dir / test_name

    def _run_one_condition(conditions):
        # Reserved for the ENTIRE docker_exec (netlist + simulate), same
        # granularity run_variation()'s own hierarchical materialize lock
        # used to span -- releases automatically (see CpuBudget.reserve())
        # whether this condition succeeds, errors, or raises. THREAD_POLICY
        # (module-level) never changes per-call, so this is safe to read
        # from multiple worker threads at once.
        label = condition_label(conditions)
        run_dir = test_dir / label
        emit_progress_running(variation, test_name, label)
        with workspace.core_pool().reserve(**thread_policy) as n_threads:
            outcome = runner(
                container, test_name, tb_source, conditions, tb_params_base,
                run_dir, f"{container_sim_dir}/{test_name}/{label}", container_rcfile, ctx,
                block_cfg=block_cfg, block=block, topology=topology, n_threads=n_threads,
            )
        append_run(variation, test_name, label, conditions, outcome)
        emit_progress_step(variation, test_name, outcome["status"] == "success")
        return conditions, outcome

    # One ThreadPoolExecutor per test, fanning out every condition of THIS
    # test at once -- separate from (and nested under, when this test's own
    # run_variation() call is itself one of several concurrent variations)
    # gen_variations._run_batch's own variation-level pool. Both levels
    # submit their real work through the SAME workspace.core_pool() above,
    # so however many variations/conditions end up overlapping, the total
    # thread count in flight across all of them never exceeds cpu_budget().
    # max_workers is generous (not itself the throttle -- core_pool() is):
    # worker threads mostly block inside docker_exec or waiting on
    # reserve(), so having more of them than cores just means more waiting
    # threads, not more actual CPU spent. Sized to the condition count so a
    # small test never pays ThreadPoolExecutor overhead for workers it will
    # never use.
    all_conditions = list(condition_matrix(test_cfg, defaults, sweep_axis))
    if where:
        # Only the conditions matching every {axis: value} in `where` (an
        # axis a condition doesn't have is no constraint) -- see
        # run_variation()'s where.
        all_conditions = [
            c for c in all_conditions
            if all(str(c[k]) == str(v) for k, v in where.items() if k in c)
        ]
        if not all_conditions:
            return {"status": "error", "error": f"no condition matches {where}"}
    outcomes = []
    if len(all_conditions) <= 1:
        outcomes = [_run_one_condition(c) for c in all_conditions]
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(all_conditions)) as pool:
            outcomes = [f.result() for f in [pool.submit(_run_one_condition, c) for c in all_conditions]]

    n_conditions = len(outcomes)
    runs = []
    n_ok = 0
    parser_module = load_parser(test_cfg["parser"])
    for conditions, outcome in outcomes:
        if outcome["status"] != "success":
            continue
        # A condition ngspice itself reported "success" for (zero exit code,
        # data file written) can still hold non-finite (nan/inf) samples --
        # a diverged/spuriously-converged solve that never raised its own
        # exit code, so outcome["status"] alone can't catch it (see
        # tb/_shared/parser_common.py's read_data(), which now raises
        # ValueError the moment it finds one, instead of silently passing
        # nan/inf through to a blank downstream plot). Treated exactly like
        # a simulate-time failure: this ONE condition is dropped from `runs`
        # (dragging the test's own status down to "partial" via n_ok below,
        # same as a condition that failed to simulate at all) instead of
        # letting the exception propagate up through run_variation() and
        # abort every OTHER condition's already-good data along with it.
        try:
            raw = parser_module.extract(outcome["data_file"])
        except Exception as exc:
            print(f"{test_name} {condition_label(conditions)}: ERROR extracting data ({exc})")
            continue
        n_ok += 1
        runs.append({"conditions": conditions, **raw})

    if not runs:
        return {"status": "error", "error": f"every condition failed to simulate, see sim/{variation}/runs.jsonl"}

    # A stale plot (or set of plots, under a since-changed naming convention)
    # from a previous run -- or from a lazy generate_plot() view of the
    # PREVIOUS result -- shouldn't look current, so this cleanup stays even
    # though plotting itself is no longer eager below. No PNG is generated
    # here at all anymore: generate_plot() materializes one lazily, the
    # first time a view actually needs it (analog_designer/gui/variation_detail.py),
    # from the exact same raw .data this call just wrote -- so a variation
    # that's never opened in the GUI (the vast majority of a large batch)
    # never pays for a PNG that's never looked
    # at. See generate_plot()'s own docstring for the replay it does.
    for stale in test_dir.glob(f"{test_name}*.png"):
        stale.unlink()
    typical = typical_conditions(test_cfg, defaults)
    # Same reasoning as the extract() try/except above, one level up: a
    # parser's own evaluate() -- e.g. parser_common.typical_min_max(),
    # which deliberately RAISES rather than silently substituting a
    # different run when the nominal/"typical" condition itself isn't
    # among the survivors (see its own docstring: a wrong silent pick here
    # could corrupt a cross_block sizing reference) -- can still fail even
    # after every individual condition's own extract() succeeded, e.g.
    # when extract() correctly dropped the FEW conditions that had a real
    # error-severity diagnostic (see run_one_ngspice()'s own comment) and
    # the typical one happened to be among them. Confirmed live
    # (gf180mcu_mh_ip__nfrac_pll's tb_startup, 2026-09-09): 2 of 3 vtune
    # conditions correctly failed on a real tran-abort, one of them being
    # the typical vtune=2.5 point, and this call crashed run_variation()
    # (and, for a bare run_sim.py CLI invocation with no batch-level
    # try/except around it, the whole process) instead of just failing
    # THIS one test.
    try:
        outcome = parser_module.evaluate(runs, test_cfg["outputs"], typical, plot_base=None)
    except Exception as exc:
        return {"status": "error", "error": f"{test_name}: evaluate() failed: {exc}"}
    return {
        "status": "success" if n_ok == n_conditions else "partial",
        "result": outcome,
    }


def generate_plot(variation, test_name, test_cfg, defaults):
    """Lazily materializes the plot PNG(s) for one already-simulated
    (variation, test) straight from its existing raw output on disk --
    never touches docker, never re-simulates anything. This is the on-demand
    counterpart to run_test()'s own plot_base=None above: replays exactly
    the "read what a condition already wrote" half of that function's loop
    (same condition_matrix()/internal_sweep_axis() to find each condition's
    run_dir, same load_parser()/extract() to rebuild the same `runs` list
    run_test() built at simulation time) and then the same
    evaluate(..., plot_base=...) call under the same _PLOT_LOCK. A condition
    whose output file is missing (trimmed, or genuinely never simulated) is
    simply skipped, same as run_test()'s own success/error split.

    Called from analog_designer/gui/variation_detail.py the first time a
    view needs a plot that isn't on disk yet; the result lands at the exact
    same canonical path run_test() used to write eagerly
    (sim/<variation>/<test>/<test>*.png), so every later view is an
    ordinary cache hit -- data.plot_paths_for() doesn't need to know this
    function exists. Returns True if a plot was written, False if no
    condition has output on disk yet (nothing to plot from)."""
    tb_source = workspace.PROJECT_ROOT / test_cfg["testbench"]
    tb_text = tb_source.read_text(encoding="utf-8")
    sweep_axis = internal_sweep_axis(test_cfg, tb_text)
    simulator = test_cfg.get("simulator", "ngspice")

    test_dir = workspace.sim_root() / variation / test_name
    parser_module = load_parser(test_cfg["parser"])
    runs = []
    for conditions in condition_matrix(test_cfg, defaults, sweep_axis):
        run_dir = test_dir / condition_label(conditions)
        # Matches run_one_ngspice()/run_one_xyce()/run_one_netlist()/
        # run_one_openems()'s own data_file naming exactly -- ngspice's
        # `wrdata` and every Xyce testbench's `.print ... file=` both
        # target the identical 'filename'_'N'.data convention (N is always
        # "0", see run_test()'s tb_params_base; run_one_xyce() normalizes
        # Xyce's own output to that same path/shape), so both share the
        # ".data" branch. A "netlist" simulator test (e.g. area estimation)
        # differs, reading the expanded .spice netlist text directly
        # instead of a .data file. "openems" differs too -- run_one_openems()
        # writes its cached Y11-sweep result as '<test_name>_0.json' (there
        # is no ngspice/Xyce-style .data output at all for this simulator,
        # the FDTD run itself IS the measurement) -- if a future Xyce
        # test's own .print line ever targets a different filename, or a
        # future simulator needs yet another convention, this branch needs
        # revisiting again.
        if simulator == "netlist":
            data_file = run_dir / f"{tb_source.stem}.spice"
        elif simulator == "openems":
            data_file = run_dir / f"{test_name}_0.json"
        else:
            data_file = run_dir / f"{test_name}_0.data"
        if not data_file.exists():
            continue
        raw = parser_module.extract(data_file)
        runs.append({"conditions": conditions, **raw})

    if not runs:
        return False

    for stale in test_dir.glob(f"{test_name}*.png"):
        stale.unlink()
    plot_base = test_dir / test_name
    typical = typical_conditions(test_cfg, defaults)
    with _PLOT_LOCK:
        parser_module.evaluate(runs, test_cfg["outputs"], typical, plot_base=plot_base)
    return True


def print_metrics(test_name, metrics, note="", log_prefix=""):
    """log_prefix (e.g. "[<variation>] ") tags every line with which
    variation it belongs to -- a no-op ("") for the default single-variation
    case, only set by callers that may have several variations printing
    concurrently (see run_variation's own log_prefix), whose output would
    otherwise interleave with no way to tell which line came from which
    one. No PASS/FAIL verdict printed -- there is no absolute per-test spec
    to judge against anymore (same "purely informative" convention
    fom.py/variation_detail.py's own docstrings already describe); a
    metric's pass/fail is only meaningful relative to a chosen design
    PROFILE's own constraints, which this console reporter has no context
    for even if it wanted one (see analog_designer/gui/variation_detail.py's
    metrics table, which computes that live from whichever profile is
    currently selected)."""
    print(f"{log_prefix}  {test_name}{note}")
    for m in metrics:
        # A Monte Carlo metric (see tb/_shared/parser_common.mc_stats())
        # has no single "typical" run -- report mean+-std instead of a
        # central value in that case.
        central = f"mean {m['mean']} +- {m['std']}" if m["typical"] is None else str(m["typical"])
        print(f"{log_prefix}    {m['name']}: {central} {m.get('unit', '')} (range {m['min']}..{m['max']})")


ContainerCtx = collections.namedtuple(
    "ContainerCtx",
    "container pdk_name models_dir stdcell_dir spiceinit_text xyce_models_dir xyce_plugins_dir mos_corner_section "
    "xschem ngspice xyce display",
    # Tool paths as resolved by _resolve_tools(); defaults are what the
    # docker image has on its login-shell PATH (and its Xvnc display).
    defaults=("/usr/local/share/xschem/bin/xschem", "ngspice", "Xyce", ":1"),
)

# $XSCHEM/$NGSPICE/$XYCE override the binaries in either execution mode
# (the standalone runner's --xschem/--ngspice/--xyce set them); otherwise
# whatever the executor's PATH has, falling back to the docker image's
# xschem install dir, which isn't on a non-login PATH.
_TOOLS_SCRIPT = (
    'echo "${XSCHEM:-$(command -v xschem || echo /usr/local/share/xschem/bin/xschem)}"; '
    'echo "${NGSPICE:-$(command -v ngspice || echo ngspice)}"; '
    'echo "${XYCE:-$(command -v Xyce || echo Xyce)}"'
)


def _resolve_tools(container):
    lines = docker_exec(container, _TOOLS_SCRIPT).stdout.split("\n")
    xschem, ngspice, xyce = (lines + ["", "", ""])[:3]
    return {
        "xschem": xschem or "xschem", "ngspice": ngspice or "ngspice", "xyce": xyce or "Xyce",
        "display": getattr(container, "display", ":1"),
    }


def setup_container(container):
    """PDK path resolution + spiceinit text for an already-running
    container (see managed_container()). Reusable across however many
    variations get simulated against that one container (see
    analog_designer/sim/gen_variations.py, which resolves this once for a whole batch
    instead of once per variation). Returns a ContainerCtx -- every
    SIMULATOR_RUNNERS entry gets the whole thing rather than one field at a
    time, so adding a 4th simulator later doesn't mean touching every call
    site's positional signature again (it did, twice, for xyce)."""
    return _setup_container(container)._replace(**_resolve_tools(container))


def _setup_container(container):
    ensure_xschemrc(container)
    pdk_name = get_pdk_name(container)

    # GF180MCU ships a single flat model file per simulator directly under
    # libs.tech/ngspice and libs.tech/xyce (no /models subdirectory), pure
    # BSIM4 SPICE (no Verilog-A/OSDI, no compiled Xyce plugin) -- structurally
    # different enough from IHP's layout that it needs its own branch here
    # rather than another get_pdk_dir() subpath tweak.
    if pdk_name.startswith("gf180mcu"):
        models_dir = get_pdk_dir(container, "libs.tech/ngspice")
        xyce_models_dir = get_pdk_dir(container, "libs.tech/xyce")
        return ContainerCtx(
            container, pdk_name, models_dir,
            stdcell_dir="",  # no gf180mcu testbench references 'stdcell_dir' yet
            spiceinit_text="",  # no OSDI models to load
            xyce_models_dir=xyce_models_dir,
            xyce_plugins_dir=None,  # no Xyce plugin needed -- see run_one_xyce
            mos_corner_section=MOS_CORNER_SECTION_GF180MCU,
        )

    # PDK's own libs.ref/<short>_stdcell dir is named by a short form of
    # $PDK with the "ihp-" family prefix stripped (confirmed against both
    # ihp-sg13g2 -> sg13g2_stdcell and ihp-sg13cmos5l -> sg13cmos5l_stdcell
    # in IHP-Open-PDK) -- NOT the same transform as xyce_plugin_so()'s,
    # which keeps the full "ihp_" prefix. Falls back to pdk_name unchanged
    # for any future PDK family without an "ihp-" prefix (e.g. sky130).
    pdk_short = pdk_name[4:] if pdk_name.startswith("ihp-") else pdk_name
    models_dir = get_pdk_dir(container, "libs.tech/ngspice/models")
    osdi_dir = get_pdk_dir(container, "libs.tech/ngspice/osdi")
    # <short>_stdcell's own SPICE subcircuit library -- only "top" (its trim/
    # enable buffers, <short>_buf_1) needs this so far; every purely-analog
    # testbench simply never references the 'stdcell_dir' placeholder this
    # feeds into run_test()'s own tb_params_base, so resolving it
    # unconditionally here (once per container, like models_dir/osdi_dir
    # already are) costs nothing extra for those.
    stdcell_dir = get_pdk_dir(container, f"libs.ref/{pdk_short}_stdcell/spice")
    # Every OSDI (compiled Verilog-A) compact model the PDK's own reference
    # libs.tech/ngspice/.spiceinit loads, not just the two MOSFET ones
    # (psp103/psp103_nqs) -- r3_cmc backs the rhigh/rsil/rppd resistor
    # models (confirmed live: without it, ngspice fails with "Unable to
    # find definition of model ...:rmod_rhigh" for any schematic using an
    # rhigh resistor, e.g. top.sch's own R1-R8 -- the model card IS present
    # via cornerRES.lib, but ngspice can't actually resolve the r3_cmc
    # MODEL TYPE it references without this being loaded first, same as
    # psp103 for MOSFETs). mosvar has no known consumer in this project
    # yet, but costs nothing to load unconditionally and keeps this list a
    # straight mirror of the PDK's own reference file instead of a
    # per-device allowlist someone has to remember to extend again.
    osdi_names = ["psp103", "psp103_nqs", "r3_cmc", "mosvar"]
    # cap_cmomi/cap_cmomf (MOM capacitors) only exist in the CMOS5L PDK (no
    # MIM there), and their model cards live in cornerCAP.lib -- same "model
    # type unresolvable without its OSDI loaded" failure as r3_cmc above
    # ("Unable to find definition of model ...:cap_cmomf_mod"). Checked for
    # existence instead of loaded unconditionally so sg13g2 (which ships no
    # such objects) keeps its exact previous .spiceinit.
    for optional in ("cap_cmomi", "cap_cmomf"):
        if docker_exec(container, f'test -f "{osdi_dir}/{optional}.osdi"').returncode == 0:
            osdi_names.append(optional)
    spiceinit_text = "\n".join([f"osdi {osdi_dir}/{name}.osdi" for name in osdi_names] + [""])
    # Xyce's own PDK model-lib mirror -- differently-structured .lib files
    # than ngspice's despite matching filenames (Xyce's instantiate the
    # YPSP103_VA plugin device directly instead of loading OSDI), so a Xyce
    # test must never fall back to the ngspice models_dir above -- and its
    # plugin (.so) directory, parallel to ngspice's own osdi_dir.
    xyce_models_dir = get_pdk_dir(container, "libs.tech/xyce/models")
    xyce_plugins_dir = get_pdk_dir(container, "libs.tech/xyce/plugins")
    return ContainerCtx(container, pdk_name, models_dir, stdcell_dir, spiceinit_text,
                         xyce_models_dir, xyce_plugins_dir, MOS_CORNER_SECTION)


@contextlib.contextmanager
def _resolve_container_ctx(container_ctx):
    """ContainerCtx for run_variation()'s simulate step -- yields
    container_ctx as-is if the caller already resolved one (batch callers
    share a single managed_container() across many variations), otherwise
    manages an on-demand container of its own for just this one variation's
    simulate step."""
    if container_ctx is not None:
        yield container_ctx
        return
    with managed_executor() as container:
        yield setup_container(container)


def materialize_variation_shadow(sim_dir, block_cfg, params, block=None):
    """Per-variation alternative to writing directly into the shared,
    visible sch/<block>.sch: materializes into sim/<variation>/_src/sch/
    instead, alongside a copy of the project's xschemrc.

    _src/sch/ starts as a full RECURSIVE COPY of the project's entire
    sch/ tree, not just this one block's own .sch/.sym -- see
    BUG_shadow_materialization_ignored.md (confirmed live, both on this
    project's own vco/quadrature_lc and cross-checked against
    ihp_mh_ip__cmos_vref's hierarchical top/default) for the full
    investigation. ROOT CAUSE (traced precisely, unlike that bug report's
    own "not yet done" section): the project's own xschemrc builds
    XSCHEM_LIBRARY_PATH with `append XSCHEM_LIBRARY_PATH :$env(PWD)` --
    NOT "the directory the rcfile lives in" (an earlier, wrong assumption
    both here and in that bug report). Every docker_exec into
    managed_container()'s container shares the SAME cwd (`docker run -w
    <project_root>`) unless a command explicitly cd's first -- so
    $env(PWD), and therefore every bare "sch/<name>" reference, ALWAYS
    resolved against the shared project root, no matter which --rcfile was
    passed or what a shadow sch/ copy contained. _netlist() now `cd`s to
    container_rcfile's own directory before invoking xschem, which is what
    actually fixes the resolution (confirmed live: an override that used
    to netlist as the stale shared value now netlists correctly) -- but
    that only works if _src/sch/ itself has everything a "sch/<name>"
    reference might need once it becomes the effective $PWD, hence the
    full tree copy here: this block's own .sch/.sym, every OTHER block's
    .sch/.sym, and every structural_sub_blocks schematic a topology's own
    .sch might reference (e.g. sch/vco/half_qvco_cell.sch). Overwriting
    this block's own entry with the real substitution below (and letting
    resolve_materialization_params() below overwrite any sub_blocks the
    same way) is what actually makes the copy correct, not just present.

    block defaults to workspace.BLOCK (every existing caller's behavior,
    unchanged) -- pass it explicitly when materializing a DIFFERENT block
    than whatever workspace.BLOCK currently holds (e.g. run_variation()
    called for a hierarchical block's own sub_blocks entry, from inside
    materialize_sub_blocks() -- see that function's own comment for why
    reading the global here instead would silently materialize under the
    wrong block's own .sch/.sym filenames).

    Exists only so multiple variations can netlist concurrently (see
    run_variation's shadow= param): each gets its own materialized
    schematic instead of racing to overwrite the one shared file. Passes
    this SAME sch_dir through to resolve_materialization_params() (which
    forwards it to materialize_sub_blocks() for a hierarchical block), so a
    "top"-like block's own sub_blocks land in this shadow sch/ too instead
    of the shared, global one -- otherwise two concurrent hierarchical
    variations, each with its OWN sub-block choice, would still race on
    sch/<sub-block>.sch even though their own top-level .sch was already
    isolated (see run_variation()'s own comment on
    _HIERARCHICAL_MATERIALIZE_LOCK for the corruption this used to cause).
    Returns the container-side path to this shadow xschemrc, to pass as
    --rcfile instead of the shared project-root one. Raises
    StaleParameterSchema (see resolve_derived_params) if `params` predates a
    derived_parameters width group the current config.json declares --
    caller's responsibility to catch, same as run_variation()'s own
    non-shadow branch does."""
    block = block or workspace.BLOCK
    src_dir = sim_dir / "_src"
    sch_dir = src_dir / "sch"
    if sch_dir.exists():
        shutil.rmtree(sch_dir)
    shutil.copytree(workspace.PROJECT_ROOT / "sch", sch_dir)

    materialized_name = f"{block}.sch"
    topology_sch = workspace.PROJECT_ROOT / "sch" / block_cfg["schematic"]
    materialized = substitute_params(
        topology_sch.read_text(encoding="utf-8"),
        resolve_materialization_params(block_cfg, params, sch_dir=sch_dir),
    )
    check_unresolved(materialized, materialized_name)
    (sch_dir / materialized_name).write_text(materialized, encoding="utf-8")

    shutil.copyfile(project_xschemrc(), src_dir / "xschemrc")
    return workspace.exec_path(src_dir / "xschemrc")


def run_variation(block_cfg, tests, defaults, params, force=False, container_ctx=None, origin=None, shadow=False, log_prefix="", block=None, topology=None, skip_on_fail_profile=None, skip_on_fail_max_failures=0, discard_on_fail=False, where=None):
    """Materialize + simulate one (block, topology, params) variation --
    block/topology default to whichever workspace.open_folder() resolved
    (workspace.BLOCK/workspace.TOPOLOGY), unchanged for every existing
    caller. Pass them explicitly to register/materialize/simulate a
    DIFFERENT block than the currently-open one -- e.g. a hierarchical
    block's own sub_blocks entry (cmos_vref, output_amp), run as its own
    independent variation from inside gen_variations.py's hierarchical
    batch path, while workspace.BLOCK is still "top". Every internal use of
    "which block/topology is this" (variation identity, registration,
    result logging, the materialized .sch/.sym filenames) uses this
    parameter, not the global, so two concurrent run_variation() calls for
    different blocks -- even under gen_variations.py's parallel
    cpu_budget>1 batch path -- can never race on workspace.BLOCK/TOPOLOGY
    mutable global state, because neither of them ever touches it:
    registers in variations.jsonl, skips whichever tests already have a
    fresh result in results.jsonl (unless force), runs the rest and appends
    their metrics. container_ctx is an optional pre-resolved
    ContainerCtx from setup_container() --
    if omitted, an on-demand container (see managed_container()) is
    started/stopped around just this one variation's simulate step, and
    NOT touched at all when every test is already fresh (so a `run_sim.py`
    invocation with nothing new to simulate never has to touch docker).
    shadow=True materializes into a per-variation shadow location instead
    of the shared sch/<block>.sch (see materialize_variation_shadow) -- set
    by callers that may be running more than one variation's simulate step
    at the same time (see analog_designer/sim/gen_variations.py's parallel batch path);
    the default (shadow=False) keeps writing the shared, visible file so a
    single-variation run still leaves it open-able in xschem afterward.
    origin is forwarded to ensure_variation_registered() -- ignored if this
    variation is already registered (identity is write-once). log_prefix
    (e.g. "[<variation>] "), default "", tags every print from this call so
    concurrent variations' interleaved console/log output stays
    attributable to whichever one printed each line. skip_on_fail_profile
    (a profile name from config.json's blocks.<block>.profiles, or
    None/off, opt-in, default off) stops iterating `to_run` the moment the
    metrics gathered SO FAR would already disqualify that named profile
    (fom.constraints_violated(), against that profile's own real
    "constraints" dict) -- deliberately NOT the old per-test outputs[i]
    minimum/maximum bound, which this codebase no longer treats as a real
    spec (see fom.py's own module docstring: "there is no absolute
    per-test pass/fail spec ... it duplicated and sometimes conflicted
    with profile-level judgment"). Ties the early-exit decision to
    whichever profile the caller is actually targeting, instead of an
    arbitrary bound that may have nothing to do with that goal.
    config.json declares each block's expensive Monte Carlo sweeps last,
    so this naturally skips exactly those once a cheap earlier test has
    already shown the design can't qualify, without needing to know which
    tests are "expensive" explicitly.

    skip_on_fail_max_failures (default 0, i.e. the original "any single
    violation ends it" behavior) tolerates that many already-violated
    constraints of skip_on_fail_profile before the early exit actually
    triggers -- see fom.constraints_violated()'s own max_failures. Ignored
    when skip_on_fail_profile is None.

    discard_on_fail (default False, requires skip_on_fail_profile) changes
    what happens AT that same early-exit point: instead of merely stopping
    (leaving this variation registered in variations.jsonl with whatever
    partial results it already collected, same as skip-on-fail alone
    always has), the returned dict carries "discard": True and this
    variation's name -- a signal for the CALLER to trim_variation() it,
    deliberately NOT done here. trim_variation() rewrites the whole of
    variations.jsonl/results.jsonl, which is documented as unsafe to run
    while another variation elsewhere in the same parallel batch
    (gen_variations.py's cpu_budget>1 path) is still mid-flight appending
    to those same two files -- only a caller that knows the whole batch's
    ThreadPoolExecutor has already finished (ceased ALL concurrent
    writers) can trim safely. This function itself has no such knowledge
    (it's one worker among possibly many), so it only ever signals the
    intent and leaves the actual deletion to gen_variations.py's
    post-batch cleanup pass (or, for a lone CLI/manual_variation.py call
    with no batch/executor around it at all, to that script's own main(),
    which trims immediately since nothing else could be writing
    concurrently there).

    where ({condition axis: value}, default None) simulates only the matching
    conditions of each test (see run_test()) -- a quick partial check. Such a
    run covers only part of a test's condition grid, so its metrics are
    printed and returned but never appended to results.jsonl.

    Returns {"variation": name, "any_error": bool, "discard": bool,
    "tests": {test: {"status": "fresh"|"success"|"partial"|"error",
    "metrics": [...]} or {"status": "error", "error": str}}}
    ("discard" is always present, False unless this call's own early exit
    just happened with discard_on_fail=True)."""
    block = block or workspace.BLOCK
    topology = topology or workspace.TOPOLOGY
    name = variation_name(block, topology, params)
    ensure_variation_registered(name, block, topology, params, origin=origin)

    existing_results = load_results()
    fresh, to_run, definition_hashes = _test_freshness(tests, block_cfg, existing_results, name, force)

    print(f"variation: {name}")
    any_error = False
    discard = False
    tests_out = {}

    # Seeded here (rather than starting empty) so a skip_on_fail_profile
    # constraint referencing an already-fresh/cached test's metric is
    # usable immediately, not just metrics from tests THIS call actually
    # simulates -- same rows the loop just below already re-derives per
    # test, just gathered once up front instead.
    accumulated_metrics = [
        r for r in existing_results
        if r["variation"] == name and r["test"] in definition_hashes
        and r["definition_hash"] == definition_hashes[r["test"]]
    ]

    for test_name in sorted(fresh):
        rows = [
            r for r in existing_results
            if r["variation"] == name and r["test"] == test_name
            and r["definition_hash"] == definition_hashes[test_name]
        ]
        print_metrics(test_name, [
            {
                "name": r["metric"], "typical": r["typical"], "min": r["min"], "max": r["max"],
                "mean": r.get("mean"), "std": r.get("std"),
                "unit": r["unit"],
            }
            for r in rows
        ], note=" (SKIPPED, fresh result already in sim/results.jsonl)", log_prefix=log_prefix)
        tests_out[test_name] = {"status": "fresh", "metrics": [
            {"name": r["metric"], **{k: r.get(k) for k in ("typical", "min", "max", "mean", "std", "unit")}}
            for r in rows
        ]}

    if not to_run:
        return {"variation": name, "any_error": any_error, "discard": discard, "tests": tests_out}

    # Real work starts here (materialize + simulate) -- see
    # emit_progress_variation_done's own docstring for why elapsed time is
    # tracked from here, per variation, rather than as one global indicator.
    start_ts = time.monotonic()

    sim_dir = workspace.sim_root() / name
    sim_dir.mkdir(parents=True, exist_ok=True)
    container_sim_dir = workspace.exec_path(sim_dir)

    # See _HIERARCHICAL_MATERIALIZE_LOCK's own comment for the history here:
    # materialize_sub_blocks() USED TO always materialize into shared,
    # global sch/<sub-block>.sch files no matter how the calling variation
    # itself was materialized, so this lock had to serialize EVERY
    # hierarchical variation against every other one, shadow or not.
    # materialize_variation_shadow() now forwards its own per-variation
    # shadow sch/ dir all the way through to materialize_sub_blocks() (see
    # resolve_materialization_params()), so a shadow=True hierarchical
    # variation no longer touches the shared path at all -- nothing left to
    # race against another shadow=True variation on. The lock is only still
    # needed for shadow=False: a block with no sub_blocks never enters this
    # branch at all, and a shadow=True one skips it too, so this now only
    # ever guards the rarer case of two SEPARATE non-shadow invocations
    # (e.g. two manual CLI runs, or a CLI run overlapping a GUI one) racing
    # on the shared sch/<sub-block>.sch -- _hierarchical_materialize_lock()
    # (not the bare in-process Lock) is what actually makes that exclusive
    # across separate processes too, see its own docstring.
    materialize_lock = (
        _hierarchical_materialize_lock()
        if (block_cfg.get("sub_blocks") and not shadow)
        else contextlib.nullcontext()
    )
    with materialize_lock:
        try:
            if shadow:
                container_rcfile = materialize_variation_shadow(sim_dir, block_cfg, params, block=block)
            else:
                # materialize the chosen topology into sch/<block>.sch (co-located with <block>.sym)
                materialized_name = f"{block}.sch"
                topology_sch = workspace.PROJECT_ROOT / "sch" / block_cfg["schematic"]
                materialized = substitute_params(topology_sch.read_text(encoding="utf-8"), resolve_materialization_params(block_cfg, params))
                check_unresolved(materialized, materialized_name)
                (workspace.PROJECT_ROOT / "sch" / materialized_name).write_text(materialized, encoding="utf-8")
                container_rcfile = workspace.exec_path(project_xschemrc())
        except StaleParameterSchema as exc:
            print(f"{log_prefix}{name}: SKIPPED (pre-migration parameter schema, see {exc})")
            emit_progress_skipped(name, list(to_run))
            emit_progress_variation_done(name, time.monotonic() - start_ts, any_error)
            return {"variation": name, "any_error": any_error, "discard": discard, "tests": tests_out}

        git_commit, git_dirty = git_info()
        # A list, not the plain dict-items() iteration this used to be, so a
        # skip_on_fail_profile break below can slice "everything not yet
        # reached" (to_run_items[i + 1:]) to tell the GUI exactly which
        # planned tests will now never run -- see emit_progress_skipped()'s
        # own docstring.
        to_run_items = list(to_run.items())
        # block_cfg (this function's own parameter) is TOPOLOGY-scoped
        # (config["blocks"][block]["topologies"][topology]) and has no
        # "profiles" key -- profiles live one level up, at
        # config["blocks"][block]["profiles"] (same place fom.classify()/
        # variation_detail.py/create_variation_dialog.py already read them
        # from) -- so this reaches workspace.CONFIG directly instead of
        # block_cfg, keyed by `block` (this call's own resolved target,
        # not necessarily workspace.BLOCK -- see this function's own
        # docstring on why block/topology are parameters, not globals).
        profiles = workspace.CONFIG["blocks"][block].get("profiles", {})
        with _resolve_container_ctx(container_ctx) as ctx:
            for i, (test_name, test_cfg) in enumerate(to_run_items):
                tb_text = (workspace.PROJECT_ROOT / test_cfg["testbench"]).read_text(encoding="utf-8")
                n_conditions = len(list(condition_matrix(
                    test_cfg, defaults, internal_sweep_axis(test_cfg, tb_text),
                )))
                print(f"{log_prefix}running {test_name} ({test_cfg['testbench']}, {n_conditions} condition(s)) ...")
                test_start_ts = time.monotonic()
                result = run_test(
                    ctx.container, name, test_name, test_cfg, defaults, ctx,
                    sim_dir, container_sim_dir, container_rcfile,
                    block_params=params, block_cfg=block_cfg, block=block, topology=topology, where=where,
                )
                test_elapsed = time.monotonic() - test_start_ts
                if result["status"] == "error":
                    print(f"{log_prefix}  {test_name}: ERROR ({result['error']})")
                    emit_progress_testfail(name, test_name)
                    tests_out[test_name] = {"status": "error", "error": result["error"]}
                    any_error = True
                    continue
                tests_out[test_name] = {"status": result["status"], "metrics": result["result"]}
                if not where:
                    append_results(
                        name, block, topology, test_name, definition_hashes[test_name], result["result"],
                        git_commit, git_dirty, duration_seconds=test_elapsed,
                    )
                note = f" (some conditions failed to simulate, see sim/{name}/runs.jsonl)" if result["status"] == "partial" else ""
                if where:
                    note += f" (only conditions matching {where}; not recorded)"
                print_metrics(test_name, result["result"], note, log_prefix=log_prefix)
                accumulated_metrics.extend(
                    {
                        "metric": m["name"], "typical": m["typical"], "min": m["min"], "max": m["max"],
                        "mean": m.get("mean"), "std": m.get("std"),
                    }
                    for m in result["result"]
                )
                # Keyed off skip_on_fail_profile's OWN constraints (a real
                # design spec) -- NOT the "error" status branch above, which
                # already continue()s to the next test on its own regardless
                # of this. A skipped test is simply never append_results()'d,
                # so it stays "stale" and will be picked up by a later run
                # (with or without --skip-on-fail), same as any other
                # never-yet-run test.
                if skip_on_fail_profile and fom.constraints_violated(
                    profiles.get(skip_on_fail_profile, {}).get("constraints", {}),
                    fom.metrics_to_variables(accumulated_metrics),
                    max_failures=skip_on_fail_max_failures,
                ):
                    # Every test from here on was planned into the upfront
                    # emit_progress_plan() but will now never run -- the GUI
                    # fills their slices gray.
                    emit_progress_skipped(name, [t for t, _ in to_run_items[i + 1:]])
                    if discard_on_fail:
                        discard = True
                        print(f"{log_prefix}  DISCARDING: {test_name} disqualifies profile {skip_on_fail_profile!r} "
                              f"beyond the {skip_on_fail_max_failures} allowed failure(s) (--discard-on-fail)")
                    else:
                        print(f"{log_prefix}  SKIPPING remaining test(s): {test_name} disqualifies profile {skip_on_fail_profile!r} (--skip-on-fail)")
                    break

    emit_progress_variation_done(name, time.monotonic() - start_ts, any_error)
    return {"variation": name, "any_error": any_error, "discard": discard, "tests": tests_out}


def _variation_params(name):
    for row in _read_jsonl(workspace.sim_root() / "variations.jsonl"):
        if row["name"] == name:
            return row["parameters"]
    sys.exit(f"no such variation in sim/variations.jsonl: {name!r}")


def validate_skip_on_fail_profile(profile_name):
    """sys.exit()s clearly if `profile_name` (a --skip-on-fail CLI value,
    or None/off) isn't one of the ACTIVE block's own declared profiles --
    called once by every skip-on-fail-accepting script's main(), right
    after workspace.open_folder() resolves workspace.BLOCK, instead of
    duplicating this check at each of their own argparse layers (same
    "validate once, up front" precedent as
    update_variations.select_variations_by_name's own sys.exit() on an
    unknown --name). Only validates the CURRENTLY ACTIVE block -- a
    hierarchical batch (gen_variations._run_hierarchical_batch) also
    simulates sub-block jobs (cmos_vref, output_amp under a "top" run)
    whose own declared profiles may differ; run_variation()'s own profile
    lookup for THOSE jobs just resolves to an empty constraints dict if
    `profile_name` isn't declared there (never disqualifies, since nothing
    references it) -- a deliberate, permissive default for a name that
    simply doesn't apply to that sub-block, not a gap this function needs
    to also catch."""
    if profile_name is None:
        return
    profiles = workspace.CONFIG["blocks"][workspace.BLOCK].get("profiles", {})
    if profile_name not in profiles:
        sys.exit(
            f"--skip-on-fail {profile_name!r}: not a declared profile for block "
            f"{workspace.BLOCK!r} (available: {sorted(profiles)})"
        )


def validate_skip_on_fail_tolerance(profile_name, max_failures, discard, checkpoint_size=None):
    """sys.exit()s clearly if --skip-on-fail-max-failures/--discard-on-fail
    are given without a --skip-on-fail profile to apply them to (both are
    meaningless on their own -- there is no "first fail" to tolerate or
    discard against without one), or if max_failures is negative. Called
    alongside validate_skip_on_fail_profile() by every accepting script's
    main(), same "validate once, up front" precedent.

    checkpoint_size (only gen_variations.py's own --checkpoint-size takes
    this at all -- run_sim.py/manual_variation.py each only ever run ONE
    variation, nothing to periodically checkpoint) must be a positive int
    if given, and requires --discard-on-fail too -- see
    gen_variations._run_batch()'s own docstring for why it's a silent no-op
    otherwise (nothing to periodically trim without discard_on_fail); this
    still refuses it loudly rather than silently ignoring it, same
    "meaningless combo, say so" precedent as the profile-less checks
    above."""
    if max_failures < 0:
        sys.exit(f"--skip-on-fail-max-failures must be >= 0, got {max_failures}")
    if profile_name is None and (max_failures > 0 or discard):
        sys.exit("--skip-on-fail-max-failures/--discard-on-fail require --skip-on-fail PROFILE to also be given")
    if checkpoint_size is not None:
        if checkpoint_size < 1:
            sys.exit(f"--checkpoint-size must be >= 1, got {checkpoint_size}")
        if not discard:
            sys.exit("--checkpoint-size requires --discard-on-fail (nothing to periodically trim otherwise)")


def main():
    arg_parser = argparse.ArgumentParser(description=__doc__)
    arg_parser.add_argument(
        "variation", nargs="?", default=None,
        help="existing variation name from sim/variations.jsonl to re-check/re-run; "
             "omit to target the config.json default-parameter variation",
    )
    arg_parser.add_argument(
        "--force", action="store_true",
        help="re-run every test even if results.jsonl already has a fresh (matching definition_hash) result",
    )
    arg_parser.add_argument(
        "--skip-on-fail", default=None, metavar="PROFILE",
        help="stop simulating a variation's remaining tests the moment they'd already disqualify PROFILE "
             "(a config.json blocks.<block>.profiles name) -- saves time on expensive Monte Carlo sweeps "
             "for a design that can't qualify anyway; opt-in, off by default. PROFILE must be declared "
             "for the active block",
    )
    arg_parser.add_argument(
        "--skip-on-fail-max-failures", type=int, default=0, metavar="N",
        help="tolerate up to N already-violated constraints of --skip-on-fail's own PROFILE before actually "
             "stopping (default 0: any single violation stops it, the original behavior) -- requires --skip-on-fail",
    )
    arg_parser.add_argument(
        "--discard-on-fail", action="store_true",
        help="when --skip-on-fail (beyond --skip-on-fail-max-failures) actually stops this variation, also "
             "trim_variation() it (delete its variations.jsonl/results.jsonl rows and sim/<name>/ dir) instead "
             "of just leaving it registered with whatever partial results it collected -- requires --skip-on-fail",
    )
    arg_parser.add_argument(
        "--project-root", default=None,
        help="project folder to operate on (needs a config.json); defaults to the last-opened folder, else CWD",
    )
    arg_parser.add_argument("--block", default=None, help="block to operate on; defaults to the first declared in config.json")
    arg_parser.add_argument("--topology", default=None, help="topology to operate on; defaults to the first declared for --block")
    arg_parser.add_argument(
        "--execution", choices=("docker", "host"), default=None,
        help="where simulators run: a docker container of the project's image, or this machine's own "
             "tools (default: settings.json execution.mode, see the GUI's Simulation Settings)",
    )
    args = arg_parser.parse_args()

    workspace.set_overrides(mode=args.execution)
    workspace.open_folder(args.project_root, block=args.block, topology=args.topology)
    validate_skip_on_fail_profile(args.skip_on_fail)
    validate_skip_on_fail_tolerance(args.skip_on_fail, args.skip_on_fail_max_failures, args.discard_on_fail)
    config = workspace.CONFIG
    defaults = config["defaults"]
    block_cfg = config["blocks"][workspace.BLOCK]["topologies"][workspace.TOPOLOGY]
    tests = config["tests"][workspace.BLOCK]
    if args.variation:
        params = _variation_params(args.variation)
    else:
        params = {n: pdef["default"] for n, pdef in block_cfg["parameters"].items()}

    name = variation_name(workspace.BLOCK, workspace.TOPOLOGY, params)
    existing_results = load_results()
    historical_durations = historical_test_durations(existing_results, workspace.BLOCK)
    emit_progress_plan(name, plan_progress(block_cfg, tests, defaults, existing_results, name, args.force, historical_durations))
    outcome = run_variation(
        block_cfg, tests, defaults, params, force=args.force, skip_on_fail_profile=args.skip_on_fail,
        skip_on_fail_max_failures=args.skip_on_fail_max_failures, discard_on_fail=args.discard_on_fail,
    )
    if outcome["discard"]:
        # Safe to trim immediately: this is a single, standalone
        # run_variation() call with no batch/executor around it, so no
        # OTHER concurrent writer could be appending to variations.jsonl/
        # results.jsonl right now (see run_variation()'s own discard_on_fail
        # docstring for why this can't be done unconditionally there).
        trim_variation(outcome["variation"])
        print(f"{outcome['variation']}: DISCARDED (--discard-on-fail)")
    if outcome["any_error"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
