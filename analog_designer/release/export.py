#!/usr/bin/env python3
"""Presentable snapshot for an external evaluator: one chosen variation per
block, its materialized schematic under release/xschem/sch/, its drawn
layout (GDS + a raster of it) under release/layout/<block>/, the evidence
that that layout passes LVS under release/lvs/<block>/, and a results
writeup under release/doc/<block>/README.md tying all four together -- all
committed to the design repo (unlike sim/, which is entirely gitignored
scratch).

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

The layout half is published differently from the schematic one, on
purpose: there is nothing to re-render from parameters, because Magic
itself owns the .mag cells (see analog_designer_pro.layout.magic_layout's
own docstring) and this module ships in the public core distribution,
which never imports pro. So layout/<block>/<variation-or-topology>/'s own
layout.gds + layout_snapshot.png and sim/<variation>/layout_verify/'s own
LVS outputs are COPIED exactly as the pro Layout tab last left them -- a
cheap read of a path convention, the same way
analog_designer.results.data's own has_layout glob is written. What that
makes stale is REPORTED as stale (each file's mtime against the newest
.mag in the folder) instead of being published as current, and a block
with no layout drawn yet simply gets no layout/LVS section at all.

Post-layout results (analog_designer.sim.run_extracted's own
extracted_result.json + curve PNGs, also in the gitignored sim/ tree) are
published the same way, and presented NEXT TO the schematic-level ones --
an extra column in the results table, a two-column plot row -- rather than
merged into them: a post-layout run covers the typical condition only, so
its numbers are not interchangeable with the swept ones feeding the
Min/Max/Spec/Pass columns. See _export_post_layout.

release/lvs/<block>/ is a re-runnable kit, not just a verdict: netgen's
own report, BOTH netlists it compared, the self-contained netgen setup
file, and a run_lvs.sh that re-runs that exact comparison. "LVS is clean"
in a writeup is unverifiable on its own, and re-deriving it would
otherwise mean installing this tool, docker and the PDK first -- with the
kit it takes netgen alone, against files that are in the release.

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
import json
import re
import shutil
import sys

from analog_designer.core import workspace
from analog_designer.results import data, fom
from analog_designer.sim import run_sim

RELEASE_XSCHEM_SCH = ("release", "xschem", "sch")
RELEASE_DOC = ("release", "doc")
RELEASE_LAYOUT = ("release", "layout")
RELEASE_LVS = ("release", "lvs")

#: How many mismatched nets/instances the doc page tabulates before
#: deferring to the published lvs_report.txt -- a real mismatch can run to
#: hundreds of groups (confirmed on this project's own `top`), and a page
#: that long stops being a readable summary of the problem.
_LVS_TABLE_LIMIT = 25

_META_RE = re.compile(r"\*\*Variation ID:\*\* `([^`]+)`.*?\*\*Exported:\*\* (\S+)", re.S)
# Same single-source-of-truth idea as _META_RE (see _rebuild_index): each
# block's own doc page is where its layout/LVS state is written down, so the
# index re-reads these two header lines back out of it rather than keeping a
# sidecar file of its own in sync.
_LAYOUT_LINE_RE = re.compile(r"^\*\*Layout:\*\* (.+?)\s*$", re.M)
_LVS_LINE_RE = re.compile(r"^\*\*LVS:\*\* (.+?)\s*$", re.M)


def _find_variation(variation_name):
    for row in run_sim._read_jsonl(workspace.sim_root() / "variations.jsonl"):
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
    container_sch = workspace.exec_path(workspace.PROJECT_ROOT / "release" / "xschem" / "sch" / f"{block}.sch")
    container_png = workspace.exec_path(workspace.PROJECT_ROOT / "release" / "doc" / block / "schematic.png")
    dest_png = dest_doc_dir / "schematic.png"
    tools = run_sim._resolve_tools(container)
    # --png needs a real X display: the docker image's Xvnc, or in host
    # mode whatever this machine has (none: the render fails, see above).
    display = f"export DISPLAY={tools['display']}; " if tools["display"] else ""
    cmd = (
        f'{display}'
        f'"{tools["xschem"]}" --rcfile "{rcfile}" '
        f'--preinit "set dark_colorscheme 0" --plotfile "{container_png}" '
        f'--png -q "{container_sch}"'
    )
    result = run_sim.docker_exec(container, cmd)
    if not dest_png.exists():
        print(f"warning: schematic PNG render failed for {block}:\n{result.stdout}\n{result.stderr}")
        return False
    return True


def _layout_source_dir(block, variation_name):
    """The layout folder this block's release pick publishes from:
    layout/<block>/<variation>/ when a .mag cell is actually drawn there,
    else layout/<block>/<topology>/ (the default-parameter folder pro's
    gen_spec.layout_target falls back to when no variation is named), else
    None -- a block whose layout nobody has started yet.

    A glob over a path convention, deliberately NOT an import of
    analog_designer_pro.layout.*: layout is a pro-only feature and this
    module ships in the public core distribution (the same reason
    analog_designer.results.data's own has_layout check is written this
    way). Keyed on a real *.mag rather than on layout_spec.json, which pro
    regenerates unconditionally as spec INPUT and which says nothing about
    whether Magic has ever drawn anything from it.

    A --layout-name scratch folder is never picked up here even when it is
    the only one with a .mag in it: that is a throwaway side copy of a
    layout (see gen_spec.set_layout_name), not the layout belonging to the
    variation whose results this page reports."""
    for target in (variation_name, workspace.TOPOLOGY):
        candidate = workspace.PROJECT_ROOT / "layout" / block / target
        if any(candidate.glob("*.mag")):
            return candidate
    return None


def _newest_mag_mtime(layout_dir):
    """mtime of the most recently written .mag cell in layout_dir, or None
    if there are none -- the reference point everything else published out
    of that folder (GDS, raster, LVS run) is called stale against."""
    mtimes = [path.stat().st_mtime for path in layout_dir.glob("*.mag")]
    return max(mtimes) if mtimes else None


def _prune_stale_publications(paths, empty_dirs=()):
    """Deletes each of `paths` that still exists, then each of `empty_dirs`
    that this left empty -- the leftovers of an EARLIER export of this same
    block that this one has nothing to replace them with (its layout was
    deleted or moved, its sim/ scratch purged, a GDS written but the raster
    not re-rendered, ...).

    Without this, release/lvs/<block>/ keeps a kit whose own doc page now
    says "LVS: not run" and release/layout/ keeps a GDS nothing links to:
    the exact "the release says one thing and the files say another" state
    the rest of this module goes out of its way to avoid (see the module
    docstring on re-exporting a hierarchical block's dependencies for the
    same reason). Only ever called with paths THIS module publishes, by
    their published names -- never a glob of whatever happens to be in the
    directory, so a file someone else put there (a reviewer's own
    lvs_report_rerun.txt, say) is left alone.

    `empty_dirs` is passed explicitly rather than taken as every pruned
    path's own parent: release/doc/<block>/ holds a pruned file (the
    layout raster) but is NOT this function's to remove -- its README.md
    is still being written at that point, and removing the directory from
    under the caller would turn a deleted raster into a failed export."""
    for path in paths:
        if path.exists():
            path.unlink()
    for directory in empty_dirs:
        if directory.exists() and not any(directory.iterdir()):
            directory.rmdir()


def _export_layout(block, variation_name, dest_doc_dir):
    """Publishes the drawn layout of `block`: layout.gds ->
    release/layout/<block>/<block>.gds, and layout_snapshot.png ->
    release/doc/<block>/layout.png, the raster the doc page embeds.

    The GDS is the portable half of the deliverable -- one self-contained
    file any layout viewer opens -- which is also why the .mag cell set is
    NOT copied next to it: those need Magic plus the PDK's own tech file to
    open at all, and they already sit in the project's own tracked layout/
    tree, so copying them into release/ would duplicate committed source
    rather than preserve something otherwise unreachable (the opposite of
    release/xschem/sch/, whose schematics are materialized from gitignored
    scratch and would be lost otherwise).

    Both files are whatever pro's Layout tab last wrote (Load, Place and
    Refresh Snapshot each rewrite both -- see
    analog_designer_pro.layout.magic_layout._run_stage); nothing here runs
    Magic or KLayout, so a .mag edited in Magic since then leaves a
    GDS/raster that PREDATES the layout it claims to show. That case is
    reported as stale in the returned status -- which the doc page's own
    "**Layout:**" line and the release index then carry verbatim -- rather
    than published as current. That mtime comparison is the whole reason
    this returns a status at all instead of just copying two files.

    Returns {"status", "note", "source", "gds", "raster"}, where "source"
    is the project-relative layout folder it published from."""
    layout_dest = workspace.PROJECT_ROOT.joinpath(*RELEASE_LAYOUT, block)
    published_gds = layout_dest / f"{block}.gds"
    published_raster = dest_doc_dir / "layout.png"

    source = _layout_source_dir(block, variation_name)
    if source is None:
        _prune_stale_publications([published_gds, published_raster], empty_dirs=(layout_dest,))
        return {"status": "not drawn yet", "note": "", "source": None, "gds": None, "raster": None}

    rel_source = source.relative_to(workspace.PROJECT_ROOT).as_posix()
    newest_mag = _newest_mag_mtime(source)
    gds_src = source / "layout.gds"
    raster_src = source / "layout_snapshot.png"
    info = {"source": rel_source, "gds": None, "raster": None}

    if gds_src.exists():
        layout_dest.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(gds_src, published_gds)
        info["gds"] = f"{block}.gds"
    if raster_src.exists():
        dest_doc_dir.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(raster_src, published_raster)
        info["raster"] = "layout.png"
    # Whichever of the two this export could not produce must not be left
    # behind from the last one -- it would show a different layout than
    # the one next to it.
    _prune_stale_publications(
        [path for path, published in ((published_gds, info["gds"]), (published_raster, info["raster"]))
         if published is None],
        empty_dirs=(layout_dest,),
    )

    if info["gds"] is None and info["raster"] is None:
        return {
            **info, "status": "no GDS/raster yet",
            "note": f"`{rel_source}/` holds the .mag cells but neither layout.gds nor "
                    "layout_snapshot.png -- run the Layout tab's Refresh Snapshot (or "
                    "`--stage snapshot`) and export again to publish the layout itself.",
        }

    candidates = (("GDS", gds_src), ("raster", raster_src))
    stale = [
        label for label, path in candidates
        if path.exists() and newest_mag is not None and path.stat().st_mtime < newest_mag
    ]
    missing = [label for label, path in candidates if not path.exists()]
    notes = []
    if stale:
        plural = len(stale) > 1
        notes.append(
            f"**Stale:** the published {' and '.join(stale)} {'predate' if plural else 'predates'} the "
            f"newest .mag cell in `{rel_source}/`, so {'they show' if plural else 'it shows'} an older "
            "state of this layout -- re-run the Layout tab's Refresh Snapshot and export again to "
            "publish the current one."
        )
    if missing:
        notes.append(f"Not published: {', '.join(missing)} (not in `{rel_source}/`).")
    return {**info, "status": "published (stale)" if stale else "published", "note": " ".join(notes)}


_RUN_LVS_SH = """#!/bin/sh
# Re-runs EXACTLY the netgen comparison this block's own release page
# reports under "## LVS", from the two netlists sitting next to this
# script -- so that claim can be checked independently, without the
# mh-analog-designer tool, without docker, and without the project's
# gitignored sim/ tree the original run happened in.
#
# Needs netgen on PATH and nothing else: netgen_setup.tcl here is
# self-contained (the PDK's own LVS setup file, Apache-2.0 header
# included, plus the tool's own device-class patch appended), so no
# $PDK_ROOT lookup happens at all.
#
#   {extracted}
#       ext2spice output of the layout cell -- connectivity only, no
#       parasitics (`ext2spice lvs`).
#   {schematic}
#       xschem netlist of the same block's materialized schematic.
#   lvs_report.txt / lvs_report.json
#       the report the release page quotes. This script writes
#       lvs_report_rerun.txt/.json instead, so nothing published is
#       overwritten and the two can be diffed.
set -e
cd "$(dirname "$0")"
netgen -batch lvs "{extracted} {cell}" "{schematic} {cell}" netgen_setup.tcl lvs_report_rerun.txt -json
echo "wrote lvs_report_rerun.txt -- diff it against lvs_report.txt"
"""


def _export_lvs(block, variation_name):
    """Publishes the LVS evidence for this block's release pick into
    release/lvs/<block>/: netgen's own report (text and its -json form),
    BOTH netlists it compared, the netgen setup file it used, and a
    run_lvs.sh that re-runs that exact comparison (see _RUN_LVS_SH).

    All of it is read out of sim/<variation>/layout_verify/, where pro's
    magic_layout.stage_lvs writes it (see that function's own docstring for
    why none of it lands in layout/) -- i.e. from inside the gitignored
    sim/ scratch tree, so unlike the layout GDS it has to be copied here or
    it does not survive a clone of the design repo at all.

    Publishing the whole kit rather than just the verdict is the point: a
    results writeup claiming "LVS is clean" is unverifiable on its own, and
    re-deriving it from a release would otherwise mean installing this
    tool, docker and the PDK first.

    Only the variation's own plain layout_verify/ is read, never a
    layout_verify__<name>/ that pro's --layout-name flag produces: that one
    verifies a throwaway side copy of the layout, not the release pick (the
    same reason _layout_source_dir ignores those folders).

    Returns {"status", "stale", "result", "files"}; status is "not run"
    when there is nothing on disk, in which case the doc page prints the
    exact command that produces it instead of quietly leaving the section
    out. `result` is stage_lvs's own lvs_result.json, as-is."""
    verify_dir = workspace.sim_root() / variation_name / "layout_verify"
    result_path = verify_dir / "lvs_result.json"
    dest_dir = workspace.PROJECT_ROOT.joinpath(*RELEASE_LVS, block)
    # Every name this function can publish -- the set _prune_stale_
    # publications is allowed to delete from a previous export, and
    # nothing else in that directory.
    publishable = [
        "lvs_report.txt", "lvs_report.json", f"{block}_extracted_lvs.spice",
        f"{block}_schematic.spice", "netgen_setup.tcl", "run_lvs.sh",
    ]

    info = {"status": "not run", "stale": False, "result": None, "files": []}
    if not result_path.exists():
        _prune_stale_publications([dest_dir / name for name in publishable], empty_dirs=(dest_dir,))
        return info
    try:
        result = json.loads(result_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        _prune_stale_publications([dest_dir / name for name in publishable], empty_dirs=(dest_dir,))
        return {**info, "status": "unreadable", "error": str(exc)}

    # stage_lvs's three shapes, kept distinct rather than flattened into a
    # bare pass/fail: "success" carries its own lvs_status (clean/mismatch),
    # while "skipped" (no netgen setup file found for this PDK) and "error"
    # each need something different found and fixed before there is any LVS
    # verdict to publish at all -- so the reader has to be told which it is.
    if result.get("status") == "success":
        info["status"] = result.get("lvs_status") or "unknown"
    else:
        info["status"] = result.get("status") or "unknown"
    info["result"] = result

    source = _layout_source_dir(block, variation_name)
    newest_mag = _newest_mag_mtime(source) if source is not None else None
    info["stale"] = newest_mag is not None and result_path.stat().st_mtime < newest_mag

    # _netgen_setup_local.tcl is published under a name without the leading
    # underscore that marked it as stage_lvs's own scratch: in the release
    # it is a first-class input, named directly by run_lvs.sh.
    for src_name, dest_name in (
        ("lvs_report.txt", "lvs_report.txt"),
        ("lvs_report.json", "lvs_report.json"),
        (f"{block}_extracted_lvs.spice", f"{block}_extracted_lvs.spice"),
        (f"{block}_schematic.spice", f"{block}_schematic.spice"),
        ("_netgen_setup_local.tcl", "netgen_setup.tcl"),
    ):
        src = verify_dir / src_name
        if not src.exists():
            continue
        dest_dir.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dest_dir / dest_name)
        info["files"].append(dest_name)

    # Only write the script once all three files it names are actually
    # published -- a run_lvs.sh that dies on a missing file would be worse
    # than none at all, since it is the one file here meant to be trusted
    # and run blind.
    needed = {f"{block}_extracted_lvs.spice", f"{block}_schematic.spice", "netgen_setup.tcl"}
    if needed <= set(info["files"]):
        script = dest_dir / "run_lvs.sh"
        # newline="\n" explicitly: Path.write_text()/open() translate "\n"
        # to the platform's own line ending, and a CRLF shell script fails
        # on the Linux box a reviewer is most likely to run it on, with
        # nothing but "bad interpreter: /bin/sh^M" to go on (confirmed:
        # this file came out CRLF when exported from Windows). Markdown
        # written elsewhere in this module doesn't care; a #! line does.
        with script.open("w", encoding="utf-8", newline="\n") as handle:
            handle.write(_RUN_LVS_SH.format(
                cell=block,
                extracted=f"{block}_extracted_lvs.spice",
                schematic=f"{block}_schematic.spice",
            ))
        # The one file in a release a reviewer is meant to EXECUTE; a
        # mode-644 script that only works via `sh run_lvs.sh` is exactly
        # the friction publishing the kit is supposed to remove. No-op on
        # Windows, where the execute bit is ignored either way.
        script.chmod(0o755)
        info["files"].append("run_lvs.sh")

    _prune_stale_publications(
        [dest_dir / name for name in publishable if name not in info["files"]], empty_dirs=(dest_dir,),
    )
    return info


def _layout_doc_lines(block, layout_info):
    """The doc page's own "## Layout" section: the raster, where the Magic
    cells and the GDS are, and whatever note _export_layout attached (a
    stale or half-published layout). Nothing at all for a block with no
    layout drawn yet -- the "**Layout:** not drawn yet" header line above
    already says that, and an empty section would only suggest something
    went missing."""
    if layout_info["status"] == "not drawn yet":
        return []
    lines = ["## Layout", ""]
    if layout_info["raster"]:
        lines += [f"![layout]({layout_info['raster']})", ""]
    source = layout_info["source"]
    # Relative from release/doc/<block>/: "../.." is release/, "../../.."
    # the project root -- so these links work in a GitHub/VS Code preview
    # of the committed release tree, not just on a local path.
    lines.append(f"- Magic cells (what a designer edits): [`{source}/`](../../../{source}/)")
    if layout_info["gds"]:
        gds = layout_info["gds"]
        lines.append(f"- GDS of that cell: [`release/layout/{block}/{gds}`](../../layout/{block}/{gds})")
    lines.append("")
    if layout_info["note"]:
        lines += [layout_info["note"], ""]
    return lines


def _lvs_fragment_cell(fragments):
    """One side of a netgen mismatch group as a single markdown table
    cell: "<net-or-instance> — <pin = fanout, ...>" per fragment, <br>
    separated (the only line break a markdown table cell takes). Keeps
    _parse_lvs_json's own "detail" string, which is what names the pins
    actually involved -- without it a row says only THAT something
    mismatched, which is the part the reader already knows."""
    if not fragments:
        return "—"
    return "<br>".join(
        " — ".join(part for part in (frag["name"], frag.get("detail")) if part).replace("|", "\\|")
        for frag in fragments
    )


def _lvs_doc_lines(block, variation_name, lvs_info):
    """The doc page's own "## LVS" section: the verdict, the device/net
    counts behind it, netgen's own mismatch groups as tables, and -- the
    reason the kit is published at all -- the one command that re-runs the
    comparison from the files right there in the release.

    Always emitted, including for a variation whose LVS was never run:
    "no LVS section" and "LVS not run" look identical to a reader
    otherwise, and the command that fixes it is worth printing where the
    question comes up."""
    status = lvs_info["status"]
    lines = ["## LVS", ""]
    if status == "not run":
        return lines + [
            "Not run for this variation, so this release makes no layout-vs-schematic claim. "
            "Run it (Layout tab -> Verification -> Run LVS, or `python -m "
            f"analog_designer_pro.layout.magic_layout {variation_name} --stage lvs --block {block}`) "
            "and export again: the report, both netlists and a re-runnable `run_lvs.sh` are "
            "published here when it has.",
            "",
        ]
    if status == "unreadable":
        return lines + [f"This variation's own lvs_result.json could not be read: {lvs_info.get('error')}", ""]

    result = lvs_info["result"] or {}
    headline = {
        "clean": f"netgen reports the layout and the schematic of `{block}` **match**.",
        "mismatch": f"netgen reports a **mismatch** between the layout and the schematic of `{block}`.",
        "skipped": f"LVS could not run: {result.get('reason', '(no reason recorded)')}",
        "error": f"LVS failed before producing a verdict: {result.get('error', '(no error recorded)')}",
    }.get(status, f"LVS status: {status}")
    lines += [headline, ""]
    if lvs_info["stale"]:
        lines += [
            "**Stale:** this LVS run predates the newest .mag cell in the layout folder above, so it "
            "was not run against the layout published here -- re-run LVS and export again.",
            "",
        ]

    if lvs_info["files"]:
        if "run_lvs.sh" in lvs_info["files"]:
            lines += [
                f"**Check it yourself:** `cd release/lvs/{block} && ./run_lvs.sh` -- that re-runs this exact "
                "comparison with netgen alone (no PDK install, no docker, no simulator), from the two "
                "netlists published next to the script, and writes lvs_report_rerun.txt to diff against "
                "the published lvs_report.txt.",
                "",
            ]
        lines.append("Published next to this page:")
        lines += [f"- [`release/lvs/{block}/{name}`](../../lvs/{block}/{name})" for name in lvs_info["files"]]
        lines.append("")

    net_counts = result.get("net_counts") or {}
    if net_counts.get("extracted") is not None or net_counts.get("schematic") is not None:
        lines += [
            f"Nets: {net_counts.get('extracted', '—')} in the layout, "
            f"{net_counts.get('schematic', '—')} in the schematic.",
            "",
        ]

    device_classes = result.get("device_classes") or {}
    counts = {}
    for side in ("extracted", "schematic"):
        for entry in device_classes.get(side) or []:
            counts.setdefault(entry["class"], {})[side] = entry["count"]
    if counts:
        lines += ["| Device class | Layout | Schematic |", "|---|---|---|"]
        for name in sorted(counts):
            side_counts = counts[name]
            lines.append(f"| {name} | {side_counts.get('extracted', '—')} | {side_counts.get('schematic', '—')} |")
        lines.append("")

    for title, key in (("Net mismatches", "bad_nets"), ("Instance mismatches", "bad_elements")):
        groups = result.get(key) or []
        if not groups:
            continue
        lines += [f"### {title} ({len(groups)})", "", "| Kind | Layout | Schematic |", "|---|---|---|"]
        for group in groups[:_LVS_TABLE_LIMIT]:
            lines.append(
                f"| {group['kind']} | {_lvs_fragment_cell(group.get('extracted'))} "
                f"| {_lvs_fragment_cell(group.get('schematic'))} |"
            )
        lines.append("")
        if len(groups) > _LVS_TABLE_LIMIT:
            lines += [
                f"{len(groups) - _LVS_TABLE_LIMIT} further group(s) not shown -- see "
                f"[lvs_report.txt](../../lvs/{block}/lvs_report.txt) for the complete list.",
                "",
            ]
    return lines


def _extracted_plot_paths(test_dir, test_name):
    """[(label, Path), ...] for the post-layout curve PNG(s) of one test --
    the same filename-to-label convention analog_designer.results.data's
    own plot_paths_for() applies to schematic-level plots, against
    run_extracted's own "<test>_extracted" plot_base instead ("plot" for
    the bare name, the "__<suffix>" part prettified otherwise), so the two
    sets of labels line up and can be paired on the page."""
    prefix = f"{test_name}_extracted"
    paths = []
    for path in sorted(test_dir.glob(f"{prefix}*.png")):
        rest = path.stem[len(prefix):]
        label = rest[2:].replace("_", " ").title() if rest.startswith("__") else "plot"
        paths.append((label, path))
    return paths


def _is_stale(result_path, extracted_netlist_created, newest_mag):
    """Whether one post-layout run says anything about the CURRENT layout:
    both the run and the netlist it ran against have to postdate the
    newest .mag cell. `extracted_netlist_created` is the ISO timestamp
    run_extracted records for the netlist it spliced (None for a result
    file written before that field existed -- then only the run's own
    mtime is checked, which is the old, weaker test rather than a guess)."""
    if result_path.stat().st_mtime < newest_mag:
        return True
    if not extracted_netlist_created:
        return False
    try:
        return datetime.datetime.fromisoformat(extracted_netlist_created).timestamp() < newest_mag
    except ValueError:
        return False


def _export_post_layout(block, variation_name, dest_doc_dir):
    """Publishes the post-layout (extracted-netlist) results of this
    block's release pick: the metrics of every test
    analog_designer.sim.run_extracted has run for it, and its curve PNG(s)
    copied into release/doc/<block>/plots_post_layout/.

    Read from sim/<variation>/layout_verify/<test>/extracted_result.json --
    run_extracted's own output, so nothing here simulates anything (same
    copy-what-is-on-disk contract as _export_layout/_export_lvs). Those
    files live in the gitignored sim/ tree, so they have to be copied to
    survive a clone.

    Deliberately NOT merged into the `metrics` the rest of the page is
    built from (data.latest_results, which feeds the spec/Pass columns and
    fom.classify): a post-layout run covers the typical condition only, no
    corner/sweep matrix, so its numbers are not interchangeable with a
    schematic-level row and must not silently replace one. They are
    reported next to it instead -- which is the whole point, since what a
    reviewer wants to know is how far the layout moved each number.

    Returns {"status", "tests", "metrics", "plots", "parasitics", "stale"}:
    `metrics` keyed by (test, metric description) for _render_readme's own
    side-by-side column, `plots` by test name."""
    verify_root = workspace.sim_root() / variation_name / "layout_verify"
    dest_plots_dir = dest_doc_dir / "plots_post_layout"
    info = {"status": "not run", "tests": {}, "metrics": {}, "plots": {}, "parasitics": None, "stale": False}
    if not verify_root.exists():
        _prune_published_tree(dest_plots_dir)
        return info

    source = _layout_source_dir(block, variation_name)
    newest_mag = _newest_mag_mtime(source) if source is not None else None

    published = {}
    for result_path in sorted(verify_root.glob("*/extracted_result.json")):
        try:
            result = json.loads(result_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        test_name = result.get("test") or result_path.parent.name
        info["tests"][test_name] = {
            "status": result.get("status"),
            "error": result.get("error"),
            "created": result.get("created"),
            "conditions": result.get("conditions") or {},
        }
        info["parasitics"] = info["parasitics"] or result.get("parasitics")
        # Stale if EITHER the run itself or the extracted netlist it
        # spliced predates the newest .mag: re-running the simulation
        # against a netlist extracted from an older layout produces a
        # perfectly fresh result file for a layout that no longer exists
        # (confirmed on this project's own cmos_vref -- netlist extracted
        # 10-01, layout last saved 10-04, post-layout run 10-22), so the
        # result file's own mtime alone would call that current.
        if newest_mag is not None and _is_stale(result_path, result.get("extracted_netlist_created"), newest_mag):
            info["stale"] = True
        for metric in result.get("metrics") or []:
            info["metrics"][(test_name, metric["name"])] = metric

        entries = []
        # Only a run that actually produced numbers gets its curves
        # published: a FAILED run leaves the PREVIOUS run's PNG sitting in
        # the same directory (run_extracted only clears those just before
        # evaluating, which a failure never reaches), and pairing that
        # curve next to the schematic one on a page that also says the
        # post-layout run failed is the page contradicting itself.
        # Confirmed on this project's own cmos_vref, whose extracted
        # netlist stopped splicing after a layout edit.
        plots = _extracted_plot_paths(result_path.parent, test_name) if result.get("status") == "success" else []
        for label, path in plots:
            dest_plots_dir.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, dest_plots_dir / path.name)
            published[path.name] = True
            entries.append((label, path.name))
        if entries:
            info["plots"][test_name] = entries

    if info["tests"]:
        info["status"] = "published" if any(
            t["status"] == "success" for t in info["tests"].values()) else "run failed"
    # Curves of a test that has since been removed from config.json, or
    # whose post-layout run was re-run under a different label, would
    # otherwise stay behind in the release pointing at nothing.
    _prune_published_tree(dest_plots_dir, keep=published)
    return info


def _prune_published_tree(directory, keep=()):
    """Deletes every file directly in `directory` that isn't named in
    `keep`, and the directory itself once empty -- the plots-directory
    counterpart of _prune_stale_publications (same reason: a release that
    keeps a curve its own page no longer references is a release that
    disagrees with itself). Only ever pointed at a directory this module
    owns outright (release/doc/<block>/plots_post_layout/), never at one
    holding anything it didn't write."""
    if not directory.exists():
        return
    for path in sorted(directory.iterdir()):
        if path.is_file() and path.name not in keep:
            path.unlink()
    if not any(directory.iterdir()):
        directory.rmdir()


def _delta_text(schematic_value, post_value):
    """"<post> (<+/-x.y%>)" for one metric's post-layout column, or just
    the value when a percentage would be meaningless or misleading: a
    missing side, a zero schematic-level reference to divide by, or a
    non-numeric (Monte Carlo "typical": None) reading. The percentage is
    what makes the column answer "did the layout move this?" at a glance
    instead of asking the reader to divide two numbers in their head."""
    if post_value is None:
        return ""
    post_text = _format_sig(post_value)
    if schematic_value is None or not isinstance(schematic_value, (int, float)) or schematic_value == 0:
        return post_text
    delta = (post_value - schematic_value) / abs(schematic_value) * 100.0
    return f"{post_text} ({delta:+.1f}%)"


def _post_layout_note(block, post_layout_info):
    """The paragraph above the results table explaining what the extra
    column is -- and, as importantly, what it is NOT: one condition, and
    whichever parasitics the extraction actually carried (today
    capacitance only; see magic_layout.EXTRACTED_PARASITICS). A reader who
    takes a C-only number for a full-RC one would draw conclusions the
    data does not support, so the page says it every time.

    Emitted whenever post-layout runs EXIST, even when none of them
    produced a usable number and there is therefore no column to explain
    (the first version only printed it alongside the column, which left a
    block whose every post-layout run failed looking exactly like one
    that was never simulated -- confirmed on this project's own cmos_vref,
    whose extracted netlist stopped splicing). Each failure is quoted,
    since "it failed" without the reason sends the reader back to a sim/
    tree that isn't in the release."""
    tests = ", ".join(f"`{name}`" for name in sorted(post_layout_info["tests"]))
    parasitics = post_layout_info["parasitics"] or "unrecorded"
    if post_layout_info["metrics"]:
        lead = (
            f"**Post-layout column:** from the extracted netlist of this block's layout, for the typical "
            f"condition only (no corner/sweep matrix -- so these are NOT comparable to the Min/Max "
            f"columns, which sweep every condition). Parasitics: {parasitics}. "
            f"Tests re-run post-layout: {tests}."
        )
    else:
        lead = (
            f"**Post-layout:** this block's layout was re-simulated from its extracted netlist "
            f"(parasitics: {parasitics}), but no test produced a usable number, so there is no "
            f"post-layout column below. Tests attempted: {tests}."
        )
    lines = [lead]
    failed = sorted(name for name, test in post_layout_info["tests"].items() if test["status"] != "success")
    if failed:
        lines.append(
            "Post-layout run FAILED for " + ", ".join(f"`{name}`" for name in failed)
            + " -- those rows have no post-layout value:"
        )
        # Blank lines around the bullets: a list glued to the paragraph
        # above it, or a paragraph glued below it, renders as a lazy
        # continuation of the wrong block in CommonMark.
        lines.append("")
        for name in failed:
            error = (post_layout_info["tests"][name].get("error") or "(no reason recorded)").strip()
            lines.append(f"- `{name}`: {error.splitlines()[0][:400]}")
        lines.append("")
    if post_layout_info["stale"]:
        lines.append(
            "**Stale:** at least one post-layout run -- or the extracted netlist it ran against -- "
            "predates the newest .mag cell in the layout folder above, so it does not describe the "
            "layout published here."
        )
    return lines


def _plots_doc_lines(block, plots_by_test, post_layout_info):
    """The "## Plots" section: schematic-level curve and post-layout curve
    of the same test SIDE BY SIDE in a two-column table whenever both
    exist, since the comparison is the point -- one above the other down a
    long page is a comparison nobody actually makes. A test with only one
    of the two keeps the plain single image it had before."""
    post_plots = post_layout_info["plots"]
    if not plots_by_test and not post_plots:
        return []
    lines = ["## Plots", ""]
    for test_name in sorted(set(plots_by_test) | set(post_plots)):
        schematic_entries = dict(plots_by_test.get(test_name) or [])
        post_entries = dict(post_plots.get(test_name) or [])
        for label in sorted(set(schematic_entries) | set(post_entries)):
            lines.append(f"### {test_name} — {label}")
            lines.append("")
            schematic_file = schematic_entries.get(label)
            post_file = post_entries.get(label)
            if schematic_file and post_file:
                lines += [
                    "| Schematic | Post-layout (extracted) |",
                    "|---|---|",
                    f"| ![{label}](plots/{schematic_file}) | ![{label} post-layout](plots_post_layout/{post_file}) |",
                    "",
                ]
            elif schematic_file:
                lines += [f"![{label}](plots/{schematic_file})", ""]
            else:
                lines += ["Post-layout only:", "", f"![{label} post-layout](plots_post_layout/{post_file})", ""]
    return lines


def _lvs_status_text(lvs_info):
    """The compact LVS token the doc page's "**LVS:**" header line and the
    release index both carry -- one string, written once, so the index can
    be rebuilt purely by re-reading the doc pages (see _rebuild_index)."""
    return f"{lvs_info['status']} (stale)" if lvs_info["stale"] else lvs_info["status"]


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


def _render_readme(block, topology, variation_name, params, metrics, profiles, primary, plots_by_test, dependency_note, exported_at, schematic_rendered=False, layout_info=None, lvs_info=None, post_layout_info=None):
    block_cfg = workspace.CONFIG["blocks"][block]
    topology_cfg = block_cfg["topologies"][topology]
    tests_cfg = workspace.CONFIG.get("tests", {}).get(block, {})

    lines = [f"# {block} — {topology}", ""]
    lines.append(f"**Variation ID:** `{variation_name}`  ")
    lines.append(f"**Exported:** {exported_at}  ")
    # Two header lines a reader (and _rebuild_index, re-reading them back
    # out of this page) gets without scrolling: whether this release
    # carries the layout at all, and whether that layout is known to match
    # the schematic. Both are the same compact tokens _export_layout/
    # _lvs_status_text return.
    lines.append(f"**Layout:** {layout_info['status']}  ")
    lines.append(f"**LVS:** {_lvs_status_text(lvs_info)}")
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

    # Right after the schematic: the two pictures of the same block belong
    # together. The section that says whether they AGREE -- "## LVS" -- goes
    # at the very end of the page instead: its verdict is already in the
    # header lines above, and its mismatch tables are long enough to push
    # the parameters/results a reader came for off the screen.
    lines += _layout_doc_lines(block, layout_info)

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

    lines += ["## Test results", ""]
    # The post-layout column only appears for a block that HAS post-layout
    # numbers -- an always-present empty column would just be noise on
    # every block whose layout hasn't been simulated (or drawn) yet.
    post_metrics = post_layout_info["metrics"]
    if post_layout_info["tests"]:
        lines += _post_layout_note(block, post_layout_info) + [""]
    if post_metrics:
        header = "| Test | Metric | Typical | Post-layout (Δ) | Min | Max | Mean | Std | Unit | Spec | Pass |"
        lines += [header, "|---|---|---|---|---|---|---|---|---|---|---|"]
    else:
        lines += [
            "| Test | Metric | Typical | Min | Max | Mean | Std | Unit | Spec | Pass |",
            "|---|---|---|---|---|---|---|---|---|---|",
        ]
    variables = fom.metrics_to_variables(metrics)
    constraints = primary["constraints"] if primary else {}

    def _post_cell(test_name, metric_description, schematic_typical):
        """The post-layout column's own cell, empty for a metric no
        post-layout run measured -- and ONLY written when the table has
        that column at all."""
        if not post_metrics:
            return ""
        post = post_metrics.get((test_name, metric_description))
        return f" {_delta_text(schematic_typical, post['typical'])} |" if post else "  |"

    for r in sorted(metrics, key=lambda r: (r["test"], r["metric"])):
        slug = fom.slugify(r["metric"])
        bounds = constraints.get(slug)
        pass_text = "" if bounds is None else ("PASS" if fom.constraint_satisfied(slug, bounds, variables) else "FAIL")
        spec = _spec_text(tests_cfg.get(r["test"], {}), r["metric"])
        lines.append(
            f"| {r['test']} | {r['metric']} | {_format_sig(r['typical'])} |"
            f"{_post_cell(r['test'], r['metric'], r['typical'])}"
            f" {_format_sig(r['min'])} | "
            f"{_format_sig(r['max'])} | {_format_sig(r.get('mean'))} | {_format_sig(r.get('std'))} | "
            f"{r.get('unit', '')} | {spec} | {pass_text} |"
        )
    # A metric measured post-layout but with no schematic-level row to sit
    # next to (its test was never run at schematic level, or its results
    # were purged) still gets a row of its own -- dropping it would hide a
    # measurement that was actually taken.
    measured = {(r["test"], r["metric"]) for r in metrics}
    for (test_name, metric_description), post in sorted(post_metrics.items()):
        if (test_name, metric_description) in measured:
            continue
        lines.append(
            f"| {test_name} | {metric_description} |  | {_delta_text(None, post['typical'])} |  |  |  |  "
            f"| {post.get('unit', '')} | {_spec_text(tests_cfg.get(test_name, {}), metric_description)} |  |"
        )
    lines.append("")

    lines += _plots_doc_lines(block, plots_by_test, post_layout_info)

    lines += _lvs_doc_lines(block, variation_name, lvs_info)

    return "\n".join(lines)


def _rebuild_index():
    """release/README.md: a table of every block that's ever been exported,
    rediscovered by re-reading each release/doc/<block>/README.md's own
    "Variation ID"/"Exported"/"Layout"/"LVS" header lines rather than a
    separate sidecar file -- the doc page is already the single source of
    truth for that block's latest export, so there's nothing else to keep
    in sync. The Layout/LVS columns are what make this table worth reading
    for a reviewer rather than just an inventory: "published (stale)" or
    "not run" on a block is visible before opening its page at all."""
    doc_root = workspace.PROJECT_ROOT.joinpath(*RELEASE_DOC)
    # PDK name from the project's own container.image tag (config.json,
    # falls back to settings.py's global default) rather than a hardcoded
    # PDK -- this doc text is shared by every project regardless of which
    # PDK its own container.image selects.
    pdk_tag = workspace.container_image().rsplit(":", 1)[-1]
    intro = [
        "One chosen variation per block, exported for review. Schematics: "
        f"[release/xschem/sch/](xschem/sch/) (requires the {pdk_tag} open-source "
        "PDK's xschem device library on XSCHEM_LIBRARY_PATH to render component symbols).",
    ]
    # Only advertised once something was actually published there -- a
    # project with no layout drawn yet would otherwise link two empty
    # directories that don't exist in the tree at all.
    if workspace.PROJECT_ROOT.joinpath(*RELEASE_LAYOUT).exists():
        intro.append("Layout GDS, one self-contained file per block: [release/layout/](layout/).")
    if workspace.PROJECT_ROOT.joinpath(*RELEASE_LVS).exists():
        intro.append(
            "LVS evidence -- netgen's own report, both netlists it compared, and a run_lvs.sh that "
            "re-runs the comparison with netgen alone: [release/lvs/](lvs/)."
        )
    lines = [
        "# Release", "",
        " ".join(intro),
        "", "| Block | Variation | Exported | Layout | LVS |", "|---|---|---|---|---|",
    ]
    if doc_root.exists():
        for block_dir in sorted(p for p in doc_root.iterdir() if p.is_dir()):
            readme = block_dir / "README.md"
            if not readme.exists():
                continue
            readme_text = readme.read_text(encoding="utf-8")
            match = _META_RE.search(readme_text)
            if not match:
                continue
            variation_name, exported_at = match.groups()
            layout_match = _LAYOUT_LINE_RE.search(readme_text)
            lvs_match = _LVS_LINE_RE.search(readme_text)
            lines.append(
                f"| [{block_dir.name}](doc/{block_dir.name}/README.md) | `{variation_name}` | {exported_at} "
                # "--" for a page exported before these two lines existed:
                # that page genuinely says nothing either way, which is not
                # the same claim as "not drawn yet"/"not run".
                f"| {layout_match.group(1) if layout_match else '--'} "
                f"| {lvs_match.group(1) if lvs_match else '--'} |"
            )
    (workspace.PROJECT_ROOT / "release" / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _export_one(project_root, block, topology, variation_name, params, container, rcfile, dependency_note=""):
    """Materialize + write the doc page for exactly one (block, variation) --
    the reusable unit export_release() calls once for the block the user
    actually picked, and again for each of its own real (non-"defaults")
    sub-block dependencies, so every doc page under release/doc/ stays
    truthful about whatever schematic is currently sitting in the shared
    release/xschem/sch/ directory. Returns {"block", "topology",
    "variation", "layout", "lvs"}.

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
    # Pure local file I/O, unlike the schematic render above -- see
    # _export_layout/_export_lvs on why the layout side is copied as the
    # pro Layout tab left it rather than regenerated here.
    layout_info = _export_layout(block, variation_name, doc_dir)
    lvs_info = _export_lvs(block, variation_name)
    post_layout_info = _export_post_layout(block, variation_name, doc_dir)

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
        schematic_rendered=schematic_rendered, layout_info=layout_info, lvs_info=lvs_info,
        post_layout_info=post_layout_info,
    )
    (doc_dir / "README.md").write_text(readme_text, encoding="utf-8")
    return {
        "block": block, "topology": topology, "variation": variation_name,
        # Carried back out so the caller can SAY what it published without
        # re-reading the doc page -- the GUI's own Release dialog reports
        # these two verbatim, which is how a designer finds out the layout
        # snapshot was stale or LVS was never run for the pick they just
        # released (see analog_designer.gui.app.App._export_release).
        "layout": layout_info["status"], "lvs": _lvs_status_text(lvs_info),
        "post_layout": f"{post_layout_info['status']} (stale)" if post_layout_info["stale"]
                       else post_layout_info["status"],
    }


def export_release(project_root, variation_name):
    """Export `variation_name` as its block's release pick -- and, if it's a
    hierarchical block, re-export each of its own sub_blocks dependencies
    that names a real registered variation too (see module docstring for
    why). Returns the top-level {"block", "topology", "variation",
    "layout", "lvs", "post_layout"}, the last three being compact status tokens
    the doc page's own header lines carry (so a caller can report what was
    published, including "not drawn yet"/"not run"/"...(stale)", without
    re-reading the page -- see _export_one).
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

    with run_sim.managed_executor() as container:
        run_sim.ensure_xschemrc(container)
        rcfile = workspace.exec_path(run_sim.project_xschemrc())

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
    print(f"  layout: {result['layout']}    LVS: {result['lvs']}    post-layout: {result['post_layout']}")


if __name__ == "__main__":
    main()
