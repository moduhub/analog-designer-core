"""Global, machine-level settings: which EDA docker image to use, where it
mounts a project directory, and how many CPU cores to budget for
simulation work. Distinct from a project's own config.json (design intent
-- blocks/topologies/tests) since these are facts about this
machine/environment, not about any one circuit design. Stored at
~/.mh-analog-designer/settings.json, sibling to workspace.py's own
last_folder.txt.

load() is called fresh on every workspace.container_image()/
container_project_root()/cpu_budget() call -- in particular, once per
variation inside run_sim.run_variation() (via container_project_root()),
not just once per batch -- so a long batch keeps seeing a Docker Settings
change the user makes mid-run without needing a restart. That only works
safely if save() below is atomic: a plain write_text() truncates the file
in place, and a load() landing mid-write (e.g. the user hits Save in the
GUI's Docker Settings dialog -- explicitly documented as safe to open while
a job is running -- while a batch's per-variation load() fires) would read
a torn, invalid-JSON file and blow up that one variation with something
like "json.decoder.JSONDecodeError: Invalid control character...". See
analog_designer/sim/run_sim.py's _rewrite_jsonl for the identical
temp-file-then-os.replace fix, same reasoning.

cpu_budget replaces the older max_parallel (a count of concurrent
variations) -- it's a count of CPU cores instead, spent by
workspace.core_pool() across BOTH variation-level and (now)
condition-level concurrency, and per-simulator-invocation thread counts
(see run_sim.py's THREAD_POLICY). Defaults to the machine's own core
count rather than a conservative "1", since the whole point of the new
CpuBudget mechanism is that it adapts each job's own thread count to
however many cores are actually free at that moment, instead of a fixed
per-job thread count needing a conservative ceiling on job COUNT to avoid
oversubscription."""
import json
import os
from pathlib import Path

_SETTINGS_FILE = Path.home() / ".mh-analog-designer" / "settings.json"

#: execution.mode: where simulators run -- "docker" (a fresh container of
#: container.image per job, the original and default behavior) or "host"
#: (xschem/ngspice/Xyce called directly on this machine, e.g. when this
#: tool itself runs inside the EDA image). host.pdk_root/host.pdk: the PDK
#: for host mode; empty means $PDK_ROOT/$PDK from the environment, and an
#: unset PDK falls back to the tag of the project's container.image. See
#: analog_designer/core/executor.py. cpu_budget stays under "container" for
#: compatibility with existing settings.json files, though it applies to
#: both modes.
DEFAULTS = {
    "container": {
        "image": "eda-env-designer:ihp-sg13g2",
        "project_root_template": "/home/moduhub/work/{name}",
        "cpu_budget": os.cpu_count() or 1,
    },
    "execution": {"mode": "docker"},
    "host": {"pdk_root": "", "pdk": ""},
}
EXECUTION_MODES = ("docker", "host")


def load():
    """Global settings merged over DEFAULTS -- a key missing from an
    existing settings.json (e.g. one written before a new setting was
    introduced) falls back to its default instead of KeyError.

    One-time migration: an existing settings.json predating cpu_budget
    only has the older "max_parallel" (a variation count, not a core
    count) -- carried over as cpu_budget's own initial value here (rather
    than silently reverting to the new os.cpu_count() default and losing
    whatever the user had deliberately set) so long as "cpu_budget" itself
    isn't ALSO already present (which would mean this migration already
    ran, or the user set it directly -- either way, "max_parallel" is a
    stale leftover key at that point, not a value to keep re-applying)."""
    settings = json.loads(json.dumps(DEFAULTS))
    if _SETTINGS_FILE.exists():
        stored = json.loads(_SETTINGS_FILE.read_text(encoding="utf-8"))
        stored_container = stored.get("container", {})
        if "max_parallel" in stored_container and "cpu_budget" not in stored_container:
            stored_container["cpu_budget"] = stored_container["max_parallel"]
        settings["container"].update(stored_container)
        for section in ("execution", "host"):
            settings[section].update(stored.get(section, {}))
    return settings


def save(settings):
    _SETTINGS_FILE.parent.mkdir(parents=True, exist_ok=True)
    # Written to a sibling temp file first and swapped in with os.replace
    # (atomic on POSIX and Windows) -- a concurrent load() (see module
    # docstring) then always sees either the complete old file or the
    # complete new one, never a truncated/torn one mid-write.
    tmp_path = _SETTINGS_FILE.with_suffix(_SETTINGS_FILE.suffix + ".tmp")
    tmp_path.write_text(json.dumps(settings, indent=2), encoding="utf-8")
    os.replace(tmp_path, _SETTINGS_FILE)
