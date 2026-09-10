"""Work-folder resolution: which project directory (config.json + sch/ tb/
sim/) this tool session operates on. Call open_folder() once at startup
(CLI main() or GUI launch) before anything else touches PROJECT_ROOT/BLOCK/
TOPOLOGY/CONFIG -- every other module reads these as workspace.<NAME>
attributes (not by import-time value) so a folder opened after import still
takes effect everywhere, and the same process can, in principle, re-open a
different folder later (e.g. a GUI "Open Folder" action).
"""
import contextlib
import json
import threading
from pathlib import Path

from analog_designer.core import settings

_LAST_FOLDER_FILE = Path.home() / ".mh-analog-designer" / "last_folder.txt"

PROJECT_ROOT = None
CONFIG = None
BLOCK = None
TOPOLOGY = None
_CORE_POOL = None


def _remembered_folder():
    if _LAST_FOLDER_FILE.exists():
        text = _LAST_FOLDER_FILE.read_text(encoding="utf-8").strip()
        if text and Path(text).is_dir():
            return Path(text)
    return None


def _remember_folder(path):
    _LAST_FOLDER_FILE.parent.mkdir(parents=True, exist_ok=True)
    _LAST_FOLDER_FILE.write_text(str(path), encoding="utf-8")


def open_folder(path=None, block=None, topology=None):
    """Resolve and set the work folder: explicit path arg -> remembered last
    folder -> CWD. Loads config.json from it and resolves BLOCK/TOPOLOGY:
    explicit args, else the first block/topology declared in config.json.
    Raises FileNotFoundError if the resolved folder has no config.json (not
    a valid project folder). Remembers the resolved folder for next time,
    but only once it's confirmed valid."""
    global PROJECT_ROOT, CONFIG, BLOCK, TOPOLOGY, _CORE_POOL
    root = Path(path).resolve() if path else (_remembered_folder() or Path.cwd())
    config_path = root / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"no config.json in {root} -- not a valid project folder")

    PROJECT_ROOT = root
    CONFIG = json.loads(config_path.read_text(encoding="utf-8"))
    resolve_parameters_files(root, CONFIG)
    _remember_folder(root)
    _CORE_POOL = None  # re-sized lazily from the CURRENT cpu_budget() on next core_pool() call

    blocks = CONFIG.get("blocks", {})
    if not blocks:
        raise ValueError(f"{config_path}: no blocks declared")
    BLOCK = block or next(iter(blocks))
    topologies = blocks[BLOCK].get("topologies", {})
    if not topologies:
        raise ValueError(f"{config_path}: block {BLOCK!r} has no topologies declared")
    TOPOLOGY = topology or next(iter(topologies))
    return PROJECT_ROOT


def resolve_parameters_files(root, config):
    """Splice each topology's external `parameters_file` (if declared) back
    into the in-memory config dict as if `parameters`/`symmetry`/
    `derived_parameters` had been inline all along -- every existing reader
    keeps working against config["blocks"][b]["topologies"][t]["parameters"]
    unchanged, unaware the file was split. A topology with no parameters_file
    keeps its inline `parameters` exactly as before (backward compatible, no
    forced migration). `symmetry` (islands/correlated_groups/critical_nets)
    and `derived_parameters` (width_groups -- see
    analog_designer.sim.run_sim.resolve_derived_params) only exist via
    parameters_file -- there's no inline equivalent to fall back to, so a
    topology without parameters_file simply has no symmetry/derived-parameter
    data (analog_designer.layout.placement treats every instance as a loose
    module in that case; run_variation() substitutes params unexpanded).
    Public (not just open_folder()'s own helper): analog_designer.results.data.load_config()
    does its own independent re-read of config.json (so the GUI picks up
    changes -- e.g. a newly registered topology -- without a restart) and
    must apply this same splice, or every reader downstream of it
    (the design-space scatter-plot viewer, app.py's dialogs) breaks on a topology using
    parameters_file, since they all still expect an inline "parameters" key."""
    for block in config.get("blocks", {}).values():
        for topology in block.get("topologies", {}).values():
            rel_path = topology.get("parameters_file")
            if not rel_path:
                continue
            params_path = root / rel_path
            params_doc = json.loads(params_path.read_text(encoding="utf-8"))
            topology["parameters"] = params_doc.get("parameters", {})
            topology["symmetry"] = params_doc.get("symmetry", {})
            topology["derived_parameters"] = params_doc.get("derived_parameters", {})


def container_image():
    """Global settings.py's container.image, overridable per-project via
    config.json's own container.image key."""
    global_default = settings.load()["container"]["image"]
    return CONFIG.get("container", {}).get("image", global_default)


def container_project_root():
    """The container mounts the host's project directories under a template
    path, keyed by the host folder's name -- e.g.
    '/home/moduhub/work/{name}'. Sourced from global settings.py's
    container.project_root_template, overridable per-project via
    config.json's own container.project_root_template if a project's mount
    layout differs."""
    global_default = settings.load()["container"]["project_root_template"]
    template = CONFIG.get("container", {}).get("project_root_template", global_default)
    return template.format(name=PROJECT_ROOT.name)


def cpu_budget():
    """How many CPU cores this host is willing to spend on simulation work
    at once -- purely a machine/hardware fact (this host's CPU/RAM
    budget), so it's global-only, not project-overridable. Defaults to
    this machine's own core count (see settings.DEFAULTS) rather than a
    conservative "1", since core_pool()'s CpuBudget adapts each job's own
    thread count to whatever's actually free, instead of needing a small
    job-COUNT ceiling to avoid oversubscription the way the older
    max_parallel (a variation count, not a core count) did."""
    return settings.load()["container"]["cpu_budget"]


class CpuBudget:
    """A pool of `total` interchangeable CPU cores, handed out in
    variable-sized chunks via reserve() instead of one-at-a-time (stdlib's
    Semaphore only ever acquires 1 unit per call, which can't express "give
    me up to N, but at least M"). Every simulator invocation that spends
    real CPU (see run_sim.py's THREAD_POLICY and its runner functions)
    reserves from the SAME shared instance (core_pool(), below) before
    running, whether it's one of many condition-level jobs inside one
    test, one of several variation-level jobs inside one batch, or (once
    wired up) a single expensive openEMS run -- so however these different
    calling layers overlap, the actual thread count in flight across ALL
    of them combined never exceeds `total`."""

    def __init__(self, total):
        self._total = max(1, total)
        self._available = self._total
        self._cond = threading.Condition()

    def total(self):
        return self._total

    @contextlib.contextmanager
    def reserve(self, preferred, minimum=1):
        """Blocks only while fewer than `minimum` cores are free, then
        grants as many as are free right now, up to `preferred` (never
        more than `total`, and never less than `minimum`) -- yields the
        granted count. This is what makes the pool adaptive: many
        concurrent minimum=1/preferred=1 jobs (e.g. ngspice condition runs)
        each get exactly 1 and all run at once up to `total` of them; one
        minimum=2/preferred=(total-2) job (e.g. a future openEMS run)
        grabs nearly everything when it's alone, but automatically shrinks
        toward `minimum` if other jobs are already holding cores, rather
        than blocking for its full preferred share."""
        minimum = max(0, min(minimum, self._total))
        with self._cond:
            while self._available < minimum:
                self._cond.wait()
            granted = max(minimum, min(preferred, self._available))
            self._available -= granted
        try:
            yield granted
        finally:
            with self._cond:
                self._available += granted
                self._cond.notify_all()


def core_pool():
    """The process-wide CpuBudget every simulator invocation shares (see
    CpuBudget's own docstring) -- sized from cpu_budget() the first time
    this is called after open_folder() (see open_folder()'s own
    _CORE_POOL reset), not re-sized live if the Docker Settings value
    changes mid-run, same "resolved once, for this run" contract
    gen_variations._run_batch's own max_workers already had for the older
    max_parallel()."""
    global _CORE_POOL
    if _CORE_POOL is None:
        _CORE_POOL = CpuBudget(cpu_budget())
    return _CORE_POOL
