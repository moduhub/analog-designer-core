#!/usr/bin/env python3
"""Presentable snapshot for an external evaluator: one chosen variation per
block, its materialized schematic under release/xschem/sch/, and a results
writeup under release/doc/<block>/README.md -- both committed to the design
repo (unlike sim/, which is entirely gitignored scratch).

export_release() renders the schematic fresh from config.json + the
variation's own stored parameters (analog_designer.sim.run_sim's own
non-shadow materialization path: resolve_materialization_params() +
substitute_params(), the same two calls run_variation() itself makes) rather
than trusting anything already on disk -- sim/<variation>/_src/ only exists
for a variation that happened to run through gen_variations.py's parallel
(shadow=True) batch path, and the shared sch/<block>.sch scratch file
reflects whichever variation was materialized there MOST RECENTLY, neither
of which is reliably "this variation" for an arbitrary release pick.
Materializing a hierarchical block (e.g. "top") has the same side effect an
ordinary Update run already has: its own sub_blocks get (re)written to that
shared sch/<sub block>.sch location too (run_sim.materialize_sub_blocks,
called from inside resolve_materialization_params), which this module then
copies from -- same disposable-scratch convention the rest of this tool
already relies on, not a new one invented here.

Called from analog_designer/gui/app.py's Release... button (single selected
variation, synchronous -- unlike App._trim_selected's own pure-local-I/O
"no docker" pattern, export_release() DOES open one docker container for its
whole call, purely to rasterize each exported block's schematic to a PNG via
a real xschem invocation; see export_release()'s own docstring) and from the
CLI below (python -m analog_designer.release.export <variation>
[--project-root PATH]).

Exporting one block only touches that block's own release files -- except a
hierarchical block's dependencies (e.g. "top" exporting its own X1/x2
sub-block choices), which get exported too, EVEN IF a different variation
was separately chosen as "the" release pick for that sub-block: there is
only one <block>.sch slot in the flat release/xschem/sch/ directory
(mirroring the live project's own sch/ convention), so whichever variation
materializes there last wins, and release/doc/<sub block>/README.md would
silently go stale/misleading relative to it otherwise -- exporting the
parent re-exports each real (non-"defaults") dependency too, keeping its
own doc page and schematic in sync with each other again. A "defaults"
dependency (config.json's own default parameters, no registered variation)
has no doc page of its own to refresh -- top's own "Sub-block dependencies"
line is the only record of that choice.
"""
import argparse
import datetime
import re
import shutil
import sys

from analog_designer.core import workspace
from analog_designer.results import data, fom
from analog_designer.sim import run_sim

RELEASE_XSCHEM_SCH = ("release", "xschem", "sch")
RELEASE_DOC = ("release", "doc")

_META_RE = re.compile(r"\*\*Variation ID:\*\* `([^`]+)`.*?\*\*Exported:\*\* (\S+)", re.S)


def _find_variation(variation_name):
    for row in run_sim._read_jsonl(workspace.PROJECT_ROOT / "sim" / "variations.jsonl"):
        if row["name"] == variation_name:
            return row
    sys.exit(f"no such variation in sim/variations.jsonl: {variation_name!r}")


def _materialize_schematic(block, topology, params, dest_sch_dir):
    """Render `block`'s own schematic from `params` and copy it + its
    symbol into dest_sch_dir; for a hierarchical block, also copy each
    sub_blocks entry's own materialized schematic, freshly (re)written to
    the shared sch/<sub block>.sch scratch location as a side effect of
    resolve_materialization_params() below -- see module docstring."""
    block_cfg = workspace.CONFIG["blocks"][block]["topologies"][topology]
    resolved = run_sim.resolve_materialization_params(block_cfg, params)
    topology_sch = workspace.PROJECT_ROOT / "sch" / block_cfg["schematic"]
    materialized = run_sim.substitute_params(topology_sch.read_text(encoding="utf-8"), resolved)
    run_sim.check_unresolved(materialized, f"{block}.sch")

    dest_sch_dir.mkdir(parents=True, exist_ok=True)
    (dest_sch_dir / f"{block}.sch").write_text(materialized, encoding="utf-8")
    shutil.copyfile(workspace.PROJECT_ROOT / "sch" / f"{block}.sym", dest_sch_dir / f"{block}.sym")

    for ref in block_cfg.get("sub_blocks", {}).values():
        sub_block = ref["block"]
        shutil.copyfile(workspace.PROJECT_ROOT / "sch" / f"{sub_block}.sch", dest_sch_dir / f"{sub_block}.sch")
        shutil.copyfile(workspace.PROJECT_ROOT / "sch" / f"{sub_block}.sym", dest_sch_dir / f"{sub_block}.sym")


def _export_xschem(block, topology, params):
    dest_sch_dir = workspace.PROJECT_ROOT.joinpath(*RELEASE_XSCHEM_SCH)
    _materialize_schematic(block, topology, params, dest_sch_dir)

    # xschemrc is machine/container-specific (fetched from inside the docker
    # image at real-simulation time, see run_sim.ensure_xschemrc) -- copied
    # best-effort for anyone with a matching PDK/docker setup, but the doc
    # page's own note below is the reliable statement of what's needed.
    root_xschemrc = workspace.PROJECT_ROOT / "xschemrc"
    if root_xschemrc.exists():
        shutil.copyfile(root_xschemrc, dest_sch_dir.parent / "xschemrc")
    sch_xschemrc = workspace.PROJECT_ROOT / "sch" / "xschemrc"
    if sch_xschemrc.exists():
        shutil.copyfile(sch_xschemrc, dest_sch_dir / "xschemrc")


def _render_schematic_png(container, rcfile, block, dest_doc_dir):
    """Render release/xschem/sch/<block>.sch (already materialized by
    _export_xschem) to a white-background PNG under dest_doc_dir -- the only
    step in this module that actually touches docker/xschem instead of pure
    Python file I/O, since there's no way to rasterize a schematic without
    the real renderer. `--preinit` runs before xschem.tcl's own top-level
    `set_ne dark_colorscheme 1` (set-if-not-exists), so forcing it to 0 here
    wins and gives a light/white background regardless of whatever theme a
    live xschem session has open; `-x`/`--no_x` is deliberately never passed
    since `--png` silently no-ops without a real X connection (has_x==0).
    Returns True on success; a render problem (missing PDK library paths,
    docker hiccup) is logged and returns False rather than blocking the rest
    of the release export, which is still useful without the picture."""
    container_sch = f"{workspace.container_project_root()}/release/xschem/sch/{block}.sch"
    container_png = f"{workspace.container_project_root()}/release/doc/{block}/schematic.png"
    dest_png = dest_doc_dir / "schematic.png"
    cmd = (
        f'export DISPLAY=:1; '
        f'/usr/local/share/xschem/bin/xschem --rcfile "{rcfile}" '
        f'--preinit "set dark_colorscheme 0" --plotfile "{container_png}" '
        f'--png -q "{container_sch}"'
    )
    result = run_sim.docker_exec(container, cmd)
    if not dest_png.exists():
        print(f"warning: schematic PNG render failed for {block}:\n{result.stdout}\n{result.stderr}")
        return False
    return True


def _format_sig(value, digits=4):
    if value is None:
        return ""
    return f"{value:.{digits}g}"


def _spec_text(test_cfg, metric_description):
    output = next((o for o in test_cfg.get("outputs", []) if o["description"] == metric_description), None)
    if output is None:
        return ""
    unit = output.get("unit", "")
    if "minimum" in output and "maximum" in output:
        return f"{output['minimum']}–{output['maximum']} {unit}".strip()
    if "minimum" in output:
        return f"≥ {output['minimum']} {unit}".strip()
    if "maximum" in output:
        return f"≤ {output['maximum']} {unit}".strip()
    return ""


def _export_plots(variation_name, block, metrics, dest_plots_dir):
    """{test: [(label, filename), ...]} for every test with at least one
    plot -- generating it on demand (run_sim.generate_plot, same on-demand
    path variation_detail.py's own panel uses) if this variation's test ran
    before a plot was ever viewed for it, then copying the PNG(s) into
    dest_plots_dir."""
    tests_cfg = workspace.CONFIG.get("tests", {}).get(block, {})
    defaults = workspace.CONFIG["defaults"]
    by_test = {}
    for test_name in sorted({m["test"] for m in metrics}):
        plots = data.plot_paths_for(variation_name, test_name)
        if not plots:
            test_cfg = tests_cfg.get(test_name)
            if test_cfg is not None:
                run_sim.generate_plot(variation_name, test_name, test_cfg, defaults)
            plots = data.plot_paths_for(variation_name, test_name)
        if not plots:
            continue
        dest_plots_dir.mkdir(parents=True, exist_ok=True)
        entries = []
        for label, path in plots:
            shutil.copyfile(path, dest_plots_dir / path.name)
            entries.append((label, path.name))
        by_test[test_name] = entries
    return by_test


def _render_readme(block, topology, variation_name, params, metrics, profiles, primary, plots_by_test, dependency_note, exported_at, schematic_rendered=False):
    block_cfg = workspace.CONFIG["blocks"][block]
    topology_cfg = block_cfg["topologies"][topology]
    tests_cfg = workspace.CONFIG.get("tests", {}).get(block, {})

    lines = [f"# {block} — {topology}", ""]
    lines.append(f"**Variation ID:** `{variation_name}`  ")
    lines.append(f"**Exported:** {exported_at}")
    lines.append("")
    description = topology_cfg.get("description", "")
    if description:
        lines += [description, ""]
    references = topology_cfg.get("references")
    if references:
        lines.append("**References:**")
        lines += [f"- {ref}" for ref in references]
        lines.append("")
    if dependency_note:
        lines += [dependency_note, ""]

    if schematic_rendered:
        lines += ["## Schematic", "", "![schematic](schematic.png)", ""]

    lines += ["## Parameters", "", "| Parameter | Value | Description |", "|---|---|---|"]
    free_descriptions = {n: p.get("description", "") for n, p in topology_cfg.get("parameters", {}).items()}
    for name, value in params.items():
        lines.append(f"| {name} | {value} | {free_descriptions.get(name, '')} |")
    calc_descriptions = run_sim.calculated_param_descriptions(topology_cfg)
    if calc_descriptions:
        resolved = run_sim.resolve_display_params(topology_cfg, params)
        for name, calc_description in calc_descriptions.items():
            value = resolved.get(name, "—")
            lines.append(f"| {name} (ƒx) | {value} | {calc_description} |")
    lines.append("")

    if block_cfg.get("profiles"):
        lines += ["## Design profile", ""]
        label = "Matched profile" if primary["matched"] else "Closest profile (not fully matched)"
        fom_text = primary["fom_error"] or ("" if primary["fom"] is None else fom.format_fom(primary["fom"]))
        lines.append(
            f"{label}: **{primary['profile']}** "
            f"({primary['n_satisfied']}/{primary['n_constraints']} constraints) — FOM: {fom_text}"
        )
        lines += ["", "| Profile | Score | FOM | Description |", "|---|---|---|---|"]
        for p in profiles:
            fom_text = p["fom_error"] or ("" if p["fom"] is None else fom.format_fom(p["fom"]))
            lines.append(f"| {p['profile']} | {p['n_satisfied']}/{p['n_constraints']} | {fom_text} | {p['description']} |")
        lines.append("")

    lines += [
        "## Test results", "",
        "| Test | Metric | Typical | Min | Max | Mean | Std | Unit | Spec | Pass |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    variables = fom.metrics_to_variables(metrics)
    constraints = primary["constraints"] if primary else {}
    for r in sorted(metrics, key=lambda r: (r["test"], r["metric"])):
        slug = fom.slugify(r["metric"])
        bounds = constraints.get(slug)
        pass_text = "" if bounds is None else ("PASS" if fom.constraint_satisfied(slug, bounds, variables) else "FAIL")
        spec = _spec_text(tests_cfg.get(r["test"], {}), r["metric"])
        lines.append(
            f"| {r['test']} | {r['metric']} | {_format_sig(r['typical'])} | {_format_sig(r['min'])} | "
            f"{_format_sig(r['max'])} | {_format_sig(r.get('mean'))} | {_format_sig(r.get('std'))} | "
            f"{r.get('unit', '')} | {spec} | {pass_text} |"
        )
    lines.append("")

    if plots_by_test:
        lines.append("## Plots")
        for test_name, entries in sorted(plots_by_test.items()):
            for label, filename in entries:
                lines += ["", f"### {test_name} — {label}", "", f"![{label}](plots/{filename})"]
        lines.append("")

    return "\n".join(lines)


def _rebuild_index():
    """release/README.md: a table of every block that's ever been exported,
    rediscovered by re-reading each release/doc/<block>/README.md's own
    "Variation ID"/"Exported" lines rather than a separate sidecar file --
    the doc page is already the single source of truth for that block's
    latest export, so there's nothing else to keep in sync."""
    doc_root = workspace.PROJECT_ROOT.joinpath(*RELEASE_DOC)
    # PDK name from the project's own container.image tag (config.json,
    # falls back to settings.py's global default) rather than a hardcoded
    # PDK -- this doc text is shared by every project regardless of which
    # PDK its own container.image selects.
    pdk_tag = workspace.container_image().rsplit(":", 1)[-1]
    lines = [
        "# Release", "",
        "One chosen variation per block, exported for review. Schematics: "
        f"[release/xschem/sch/](xschem/sch/) (requires the {pdk_tag} open-source "
        "PDK's xschem device library on XSCHEM_LIBRARY_PATH to render component symbols).",
        "", "| Block | Variation | Exported |", "|---|---|---|",
    ]
    if doc_root.exists():
        for block_dir in sorted(p for p in doc_root.iterdir() if p.is_dir()):
            readme = block_dir / "README.md"
            if not readme.exists():
                continue
            match = _META_RE.search(readme.read_text(encoding="utf-8"))
            if not match:
                continue
            variation_name, exported_at = match.groups()
            lines.append(f"| [{block_dir.name}](doc/{block_dir.name}/README.md) | `{variation_name}` | {exported_at} |")
    (workspace.PROJECT_ROOT / "release" / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _export_one(project_root, block, topology, variation_name, params, container, rcfile, dependency_note=""):
    """Materialize + write the doc page for exactly one (block, variation) --
    the reusable unit export_release() calls once for the block the user
    actually picked, and again for each of its own real (non-"defaults")
    sub-block dependencies, so every doc page under release/doc/ stays
    truthful about whatever schematic is currently sitting in the shared
    release/xschem/sch/ directory. Returns {"block", "topology", "variation"}.

    Re-opens workspace scoped to (block, topology) itself -- every
    analog_designer.results.data helper below (load_results, latest_results,
    load_config) filters to workspace.BLOCK/TOPOLOGY internally, and a
    caller iterating several (block, variation) pairs in one export_release()
    call can't rely on whichever one happened to be active last.

    container/rcfile: the one docker container + xschemrc export_release()
    opens for its whole call (see that function's own docstring) -- shared
    across every (block, variation) pair here so a hierarchical export
    doesn't pay container startup cost once per sub-block."""
    workspace.open_folder(project_root, block=block, topology=topology)
    _export_xschem(block, topology, params)

    doc_dir = workspace.PROJECT_ROOT.joinpath(*RELEASE_DOC, block)
    doc_dir.mkdir(parents=True, exist_ok=True)
    schematic_rendered = _render_schematic_png(container, rcfile, block, doc_dir)

    metrics = [r for r in data.latest_results(data.load_results(all_topologies=True)) if r["variation"] == variation_name]
    block_cfg = workspace.CONFIG["blocks"][block]
    profiles = fom.classify(block_cfg, metrics)
    # Best-scoring profile, matched or not -- same default variation_detail.py's
    # own Profiles panel selects (sorted by n_satisfied, not filtered to
    # matched=True), so the doc page's Pass column reflects the same profile
    # a GUI user would see by default, not an all-blank column whenever
    # nothing happens to fully match.
    primary = sorted(profiles, key=lambda p: p["n_satisfied"], reverse=True)[0] if profiles else None

    dest_plots_dir = workspace.PROJECT_ROOT.joinpath(*RELEASE_DOC, block, "plots")
    plots_by_test = _export_plots(variation_name, block, metrics, dest_plots_dir)

    exported_at = datetime.datetime.now().isoformat(timespec="seconds")
    readme_text = _render_readme(
        block, topology, variation_name, params, metrics, profiles, primary, plots_by_test, dependency_note, exported_at,
        schematic_rendered=schematic_rendered,
    )
    (doc_dir / "README.md").write_text(readme_text, encoding="utf-8")
    return {"block": block, "topology": topology, "variation": variation_name}


def export_release(project_root, variation_name):
    """Export `variation_name` as its block's release pick -- and, if it's a
    hierarchical block, re-export each of its own sub_blocks dependencies
    that names a real registered variation too (see module docstring for
    why). Returns the top-level {"block", "topology", "variation"}.
    project_root=None reopens the last-opened folder (or CWD), same default
    as every other CLI entry point in this codebase.

    Opens exactly one docker container for the whole call (see
    run_sim.managed_container) -- needed only to rasterize each exported
    block's schematic to a PNG via a real xschem invocation, everything else
    here is pure Python file I/O. Not the "no docker" synchronous export this
    module's own header docstring describes for the rest of its work; the
    GUI's Release button accepts the resulting few-seconds-per-export delay
    (see analog_designer/gui/app.py's own _export_release)."""
    workspace.open_folder(project_root)
    row = _find_variation(variation_name)
    block, topology, params = row["block"], row["topology"], row["parameters"]
    sub_blocks = workspace.CONFIG["blocks"][block]["topologies"][topology].get("sub_blocks")

    dependency_note = ""
    if sub_blocks:
        deps = ", ".join(
            f"{instance} ({ref['block']}): `{params.get(f'{instance}_variation') or run_sim.BLOCK_REF_DEFAULT}`"
            for instance, ref in sub_blocks.items()
        )
        dependency_note = (
            f"**Sub-block dependencies:** {deps} — their materialized schematics (and, for a "
            f"real registered variation, their own release/doc/ page) are kept in sync with "
            f"these exact choices every time this block is exported."
        )

    with run_sim.managed_container() as container:
        run_sim.ensure_xschemrc(container)
        rcfile = f"{workspace.container_project_root()}/xschemrc"

        result = _export_one(project_root, block, topology, variation_name, params, container, rcfile, dependency_note)

        if sub_blocks:
            for instance, ref in sub_blocks.items():
                chosen = params.get(f"{instance}_variation")
                if not chosen or chosen == run_sim.BLOCK_REF_DEFAULT:
                    continue
                sub_row = _find_variation(chosen)
                _export_one(project_root, sub_row["block"], sub_row["topology"], chosen, sub_row["parameters"], container, rcfile)

    _rebuild_index()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("variation", help="registered variation name (from sim/variations.jsonl) to export as its block's release pick")
    parser.add_argument("--project-root", default=None, help="project folder to operate on; defaults to the last-opened folder, else CWD")
    args = parser.parse_args()

    result = export_release(args.project_root, args.variation)
    print(f"exported {result['block']}/{result['topology']} variation {result['variation']!r} to release/xschem/sch/ and release/doc/{result['block']}/")


if __name__ == "__main__":
    main()
