"""Read-only access to config.json, sim/variations.jsonl, sim/results.jsonl
and sim/<variation>/runs.jsonl of the open project folder (analog_designer.core.workspace.
PROJECT_ROOT). No Tkinter import, no side effects -- safe to call from any
panel or from a background thread."""
import json
import sys

from analog_designer.results import fom as fom_module
from analog_designer.sim import run_sim as run_sim_module
from analog_designer.core import workspace


def _read_jsonl(path):
    """One dict per non-blank line. A line that fails to parse (e.g. a torn
    write from two threads racing on the same file under
    workspace.cpu_budget() > 1 -- see run_sim._append_jsonl's own
    docstring for the actual bug this guards against, now fixed with a
    lock, but old damage from before that fix -- or from any other
    corruption -- can still be sitting on disk) is skipped with a warning
    to stderr rather than raising: one bad row out of hundreds shouldn't
    make every single reader of this file (GUI table, pro's own design-space
    viewer, CLI batches...) refuse to load at all."""
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


def _filter_scope(rows, all_topologies):
    """Every row is always narrowed to workspace.BLOCK -- blocks never mix
    (different tests/metrics entirely). Topology narrows further to
    workspace.TOPOLOGY unless all_topologies=True (used only by views that
    are safe to show multiple topologies of the SAME block together, since
    those share tests/metrics -- see analog_designer/gui/app.py's docstring
    on _show_all_topologies)."""
    rows = [r for r in rows if r["block"] == workspace.BLOCK]
    if not all_topologies:
        rows = [r for r in rows if r["topology"] == workspace.TOPOLOGY]
    return rows


def load_variations(all_topologies=False):
    rows = _read_jsonl(workspace.PROJECT_ROOT / "sim" / "variations.jsonl")
    return _filter_scope(rows, all_topologies)


def load_results(all_topologies=False):
    rows = _read_jsonl(workspace.PROJECT_ROOT / "sim" / "results.jsonl")
    return _filter_scope(rows, all_topologies)


def load_runs(variation_name):
    return _read_jsonl(workspace.PROJECT_ROOT / "sim" / variation_name / "runs.jsonl")


def diagnostics_for_variation(variation_name, test_name=None):
    """Every unique diagnostic (severity/category/key from
    analog_designer.sim.log_diagnostics.parse/soa_check.check_soa,
    persisted per-run by run_sim.append_run) seen across this variation's
    runs.jsonl, merged across every matching run: the SAME (severity,
    category, key) can appear in more than one run (different
    conditions/corners of the same test) -- counts are summed and the
    worst "value" (when the rule tracks one, e.g. SOA) is kept, exactly the
    same merge policy log_diagnostics.parse itself already uses within one
    run. `test_name`, if given, restricts to that one test's own runs
    (e.g. for a per-test icon or a "this test only" log view); None
    aggregates the whole variation.

    Only the LATEST run per (test, condition) counts -- runs.jsonl is an
    append-only execution history (see append_run's own docstring), so a
    condition simulated more than once (a plain re-run, or an old row from
    BEFORE some testbench/parser/diagnostic-rule change) leaves every prior
    attempt sitting in the file forever. Confirmed live (2026-09-06): a
    variation re-simulated after log_diagnostics.py's OSDI-based SOA rule
    was replaced by soa_check.py's waveform-based one still had the OLD
    rule's stale entries (implausible values, a completely different "key"
    shape) mixed in with the new ones, because they were never superseded
    in this merge, only added alongside. Same "keep the freshest" principle
    latest_results() already applies to results.jsonl, just keyed by
    (test, condition) instead of (variation, test, metric)."""
    runs = load_runs(variation_name)
    if test_name is not None:
        runs = [r for r in runs if r["test"] == test_name]

    latest_by_condition = {}
    for run in runs:
        key = (run["test"], run["condition"])
        if key not in latest_by_condition or run["created"] > latest_by_condition[key]["created"]:
            latest_by_condition[key] = run

    merged = {}
    for run in latest_by_condition.values():
        for d in run.get("diagnostics") or []:
            key = (d["severity"], d["category"], d["key"])
            entry = merged.setdefault(key, {**d, "count": 0})
            entry["count"] += d["count"]
            if "value" in d and abs(d["value"]) > abs(entry.get("value") or 0):
                entry["value"] = d["value"]
                entry["message"] = d["message"]

    return sorted(merged.values(), key=lambda d: (d["severity"] != "error", -d["count"]))


def variation_problem_counts(variation_name):
    """{"errors": n, "warnings": n} -- distinct diagnostic count (not raw
    occurrence count) of each severity across the whole variation, for the
    GUI's per-variation icon (VariationsTable) and per-test icon
    (VariationDetail.metrics_tree, called once per distinct test with
    test_name set instead)."""
    diagnostics = diagnostics_for_variation(variation_name)
    return {
        "errors": sum(1 for d in diagnostics if d["severity"] == "error"),
        "warnings": sum(1 for d in diagnostics if d["severity"] == "warning"),
    }


def _current_definition_hashes(config, results):
    """{(block, topology, test): current_definition_hash} for every distinct
    (block, topology, test) combination appearing in results -- computed once
    per combination (not per row, since the hash only depends on the test's
    definition, not which variation ran it) by re-hashing the same schematic/
    testbench/parser/config.json subtree run_sim.py hashes when deciding
    whether a test needs to be re-run (analog_designer/sim/run_sim.py:compute_definition_hash).
    None for a combination whose block/topology/test/files no longer exist in
    the current config -- that can never match a stored hash, so it's always
    flagged stale rather than crashing latest_results()."""
    cache = {}
    for row in results:
        key = (row["block"], row["topology"], row["test"])
        if key in cache:
            continue
        try:
            block_cfg = config["blocks"][row["block"]]["topologies"][row["topology"]]
            test_cfg = config["tests"][row["block"]][row["test"]]
            cache[key] = run_sim_module.compute_definition_hash(block_cfg, test_cfg)
        except (KeyError, FileNotFoundError):
            cache[key] = None
    return cache


def latest_results(results):
    """Collapse results.jsonl to one row per (variation, test, metric): keep
    only rows whose definition_hash matches the freshest hash seen for that
    (variation, test) -- results.jsonl can hold superseded rows from a
    previous test/schematic/parser definition -- then keep the most recent
    row per (variation, test, metric), since a fresh definition can still be
    re-run more than once (e.g. --force).

    Each returned row also carries "stale": True when its definition_hash
    doesn't match what the test's definition hashes to *right now* -- e.g.
    the parser was edited but this variation hasn't been re-simulated since.
    The row is still returned with its old value (nothing is hidden), just
    flagged so the GUI can tell the user it needs a re-run instead of
    silently presenting possibly-outdated numbers as current."""
    freshest_hash = {}
    for row in results:
        key = (row["variation"], row["test"])
        if key not in freshest_hash or row["created"] > freshest_hash[key][0]:
            freshest_hash[key] = (row["created"], row["definition_hash"])

    current_hashes = _current_definition_hashes(load_config(), results)

    latest_per_metric = {}
    for row in results:
        key = (row["variation"], row["test"])
        if row["definition_hash"] != freshest_hash[key][1]:
            continue
        metric_key = (row["variation"], row["test"], row["metric"])
        if metric_key not in latest_per_metric or row["created"] > latest_per_metric[metric_key]["created"]:
            hash_key = (row["block"], row["topology"], row["test"])
            latest_per_metric[metric_key] = {
                **row,
                "stale": current_hashes.get(hash_key) != row["definition_hash"],
            }

    return list(latest_per_metric.values())


def plot_paths_for(variation, test):
    """[(label, Path), ...] for every plot PNG a parser generated for this
    (variation, test), sorted by filename. A file named exactly
    "{test}.png" gets the generic label "plot"; "{test}__<suffix>.png"
    gets <suffix> prettified (underscores -> spaces, title-cased) as its
    label -- lets a parser generate any number of named views (0, 1, or
    many) without the GUI needing to know in advance which ones exist."""
    test_dir = workspace.PROJECT_ROOT / "sim" / variation / test
    if not test_dir.exists():
        return []
    results = []
    for path in sorted(test_dir.glob(f"{test}*.png")):
        rest = path.stem[len(test):]
        label = rest[2:].replace("_", " ").title() if rest.startswith("__") else "plot"
        results.append((label, path))
    return results


def load_config():
    config = json.loads((workspace.PROJECT_ROOT / "config.json").read_text(encoding="utf-8"))
    workspace.resolve_parameters_files(workspace.PROJECT_ROOT, config)
    return config


def variation_summaries(all_topologies=False, names=None):
    """One row per variation, for the top-level table: identity fields from
    variations.jsonl, how many metrics have been measured, and its
    design-profile classification (per config.json blocks.<block>.profiles),
    computed on demand from its latest metrics -- nothing about it is
    stored on disk.

    `names`, if given, restricts the result to just those variation names --
    e.g. analog_designer/gui/app.py recomputing one variation's own fresh
    classification the moment IT finishes simulating, instead of every
    variation in the workspace. results.jsonl is still read in full either
    way (a flat log, not indexed by variation), but the per-variation
    fom.classify() loop below only runs for the requested subset."""
    variations = load_variations(all_topologies)
    if names is not None:
        names = set(names)
        variations = [v for v in variations if v["name"] in names]
    metrics_by_variation = {}
    for row in latest_results(load_results(all_topologies)):
        metrics_by_variation.setdefault(row["variation"], []).append(row)
    config = load_config()

    summaries = []
    for variation in variations:
        rows = metrics_by_variation.get(variation["name"], [])
        block_cfg = config.get("blocks", {}).get(variation["block"], {})
        profiles = fom_module.classify(block_cfg, rows)
        primary_profile = next((p for p in profiles if p["matched"]), None)
        summaries.append({
            "variation": variation["name"],
            "block": variation["block"],
            "topology": variation["topology"],
            "n_total": len(rows),
            "profiles": profiles,
            "primary_profile": primary_profile,
            "metrics_by_description": {r["metric"]: r["typical"] for r in rows},
            "has_stale": any(r.get("stale") for r in rows),
            "problems": variation_problem_counts(variation["name"]),
            "created": variation["created"],
        })
    return summaries


def matching_variations(block, topology, profile_name=None):
    """Every registered variation name of (block, topology) whose
    primary_profile matches profile_name -- the block/topology-EXPLICIT
    sibling of variation_summaries() (which is implicitly scoped to
    workspace.BLOCK/workspace.TOPOLOGY via _filter_scope()). Needed to
    answer "which existing cmos_vref/default variations satisfy the
    low_power profile" from OUTSIDE that block -- e.g. while
    workspace.BLOCK == "top", composing a hierarchical variation's own
    "block_ref" parameter (see analog_designer.gui.create_variation_dialog
    and analog_designer.sim.manual_variation's --param-spec), where
    workspace.BLOCK/TOPOLOGY point at "top", not at the sub-block whose
    variations are actually being drawn from.

    profile_name=None returns every registered variation's name regardless
    of profile (an unfiltered draw -- also manual_variation.py's own
    "--param-spec NAME=any" case). If the block declares no profiles at
    all, or none of its variations match the requested profile_name, this
    degrades to the same unfiltered set rather than returning an empty
    list -- "atender uma especificação" is a preference when one can be
    honored, not a hard requirement that can strand a block with no (or no
    matching) profiles."""
    variations = [
        r for r in _read_jsonl(workspace.PROJECT_ROOT / "sim" / "variations.jsonl")
        if r["block"] == block and r["topology"] == topology
    ]
    if profile_name is None:
        return [v["name"] for v in variations]

    results = [
        r for r in _read_jsonl(workspace.PROJECT_ROOT / "sim" / "results.jsonl")
        if r["block"] == block and r["topology"] == topology
    ]
    metrics_by_variation = {}
    for row in latest_results(results):
        metrics_by_variation.setdefault(row["variation"], []).append(row)

    block_cfg = load_config().get("blocks", {}).get(block, {})
    matched = []
    for v in variations:
        rows = metrics_by_variation.get(v["name"], [])
        profiles = fom_module.classify(block_cfg, rows)
        if any(p["matched"] and p["profile"] == profile_name for p in profiles):
            matched.append(v["name"])

    return matched if matched else [v["name"] for v in variations]


def _configured_scales(config):
    """{description: scale} for every test output across all blocks/tests that
    declares an explicit "scale" in config.json -- metrics without one are
    absent here; metric_options() defaults those to "linear" (today's implicit
    behavior, so no config change is needed for any other metric)."""
    scales = {}
    for block_tests in config.get("tests", {}).values():
        for test_cfg in block_tests.values():
            for output in test_cfg.get("outputs", []):
                if "scale" in output:
                    scales[output["description"]] = output["scale"]
    return scales


def metric_options():
    """Sorted {"description", "unit", "scale"} for every distinct metric
    currently in results.jsonl -- for populating pro's own design-space
    viewer's X/Y dropdowns with human-readable labels (slugs stay internal
    to analog_designer/results/fom.py) and their
    default axis scale (config.json-driven, "linear" unless overridden)."""
    seen = {}
    for row in latest_results(load_results()):
        seen.setdefault(row["metric"], row.get("unit", ""))
    scales = _configured_scales(load_config())
    return [
        {"description": d, "unit": u, "scale": scales.get(d, "linear")}
        for d, u in sorted(seen.items())
    ]
