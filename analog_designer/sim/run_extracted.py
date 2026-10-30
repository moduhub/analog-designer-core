#!/usr/bin/env python3
"""Re-runs one of a block's existing testbenches with the block's own
extracted (post-layout, parasitic) netlist spliced in, for the "typical"
condition only, so the resulting curve can be compared against the
schematic-level one an ordinary run_sim.py test run already produced --
the GUI's Layout tab Verification sub-tab's "Run Tests (Extracted)" button
(see analog_designer_pro.layout.magic_layout.stage_extract_lvs, which is
what produces the extracted netlist this reads).

Deliberately narrower than run_sim.py's own run_test(): only the typical
condition (not the full corner/sweep condition_matrix -- that's real,
useful future work, not built here) and only simulator="ngspice"
testbenches (every other SIMULATOR_RUNNERS entry -- xyce/openems/netlist --
is refused with a clear "unsupported" status rather than silently doing the
wrong thing). Keeps the common case (one test, one look at whether the
layout's parasitics visibly change the curve) cheap; a full-matrix version
can reuse the same _splice_extracted_subckt() once someone needs it.

Writes its own metrics to sim/<variation>/layout_verify/<test>/
extracted_result.json (see _write_extracted_result) rather than to
sim/results.jsonl: these are NOT comparable to a schematic-level result
row -- typical condition only, no corner/sweep matrix -- and anything
reading results.jsonl (the Variations table, fom.classify, the release
page's own spec/Pass columns) would silently start mixing the two if they
landed in the same log. A separate file per (variation, test) keeps them
side by side instead, which is exactly how the release page presents them
(analog_designer.release.export._export_post_layout).

Usage: python -m analog_designer.sim.run_extracted TEST_NAME [VARIATION]
[--project-root PATH] [--block NAME] [--topology NAME]
"""
import argparse
import datetime
import json
import re
import sys

from analog_designer.core import workspace
from analog_designer.sim.run_sim import (
    _netlist,
    _resolve_container_ctx,
    _variation_params,
    docker_exec,
    fixed_tb_params,
    internal_sweep_axis,
    load_parser,
    materialize_variation_shadow,
    typical_conditions,
    variation_name,
)

_SUBCKT_RE_TEMPLATE = r'^\.subckt\s+{block}\b.*?^\.ends\b[^\n]*\n?'
_SUBCKT_HEADER_RE = re.compile(r'^\.subckt\s+(\S+)\s*([^\n]*)\n', re.IGNORECASE | re.MULTILINE)


def _splice_extracted_subckt(netlist_text, block, extracted_text):
    """Replaces the schematic-derived `.subckt <block> ... .ends` in
    netlist_text with the one from extracted_text (Magic's ext2spice output
    for the same block) -- the standard post-layout-simulation trick: keep
    the testbench's own sources/other blocks exactly as netlisted, swap in
    only the DUT's own subckt body. Assumes xschem's netlister never nests
    one .subckt's body inside another's (confirmed true for every .spice
    this project's own xschem invocations produce -- a flat sequence of
    .subckt/.ends pairs), so a non-greedy match up to the very next .ends
    is always this subckt's own closing line, not some other block's.

    Verifies pin agreement between the two subckts beyond both matching
    `block`'s own name: a clean Extract+LVS run confirms DEVICE
    connectivity matches, not that ext2spice emitted its .subckt header's
    port list in the same ORDER the schematic-derived one uses. ngspice
    matches subckt ports positionally, so a real order mismatch here would
    silently simulate a mis-wired circuit rather than erroring -- confirmed
    live 2026-10-01 against a real ihp-sg13cmos5l extraction for output_amp
    that this does happen (extracted ".subckt output_amp vo vdd vss ibias
    vp vn" vs. schematic ".subckt output_amp vdd vo vp vn ibias vss", same
    6 ports, genuinely different order -- ext2spice's own port ordering
    comes from Magic's internal port/label bookkeeping, not the schematic
    symbol's declared pin order). If the two port lists are the same SET
    of names but a different order, only the extracted header's own port
    list is rewritten to match the schematic's order before splicing --
    the body's internal connectivity references nodes by name, not
    position, so it's unaffected. A different SET of names (not just a
    reorder) is a real mismatch, not something to paper over, so that
    raises instead."""
    pattern = re.compile(_SUBCKT_RE_TEMPLATE.format(block=re.escape(block)), re.IGNORECASE | re.MULTILINE | re.DOTALL)
    dut_match = pattern.search(netlist_text)
    if dut_match is None:
        raise ValueError(f"testbench netlist has no .subckt {block} ... .ends")
    extracted_match = pattern.search(extracted_text)
    if extracted_match is None:
        raise ValueError(f"extracted netlist has no .subckt {block} ... .ends")

    dut_header = _SUBCKT_HEADER_RE.match(dut_match.group(0))
    extracted_header = _SUBCKT_HEADER_RE.match(extracted_match.group(0))
    dut_ports = dut_header.group(2).split()
    extracted_ports = extracted_header.group(2).split()
    if sorted(dut_ports) != sorted(extracted_ports):
        raise ValueError(
            f".subckt {block}'s port lists differ between the schematic-derived netlist {dut_ports} "
            f"and the extracted one {extracted_ports} -- not just a reorder, can't safely splice"
        )

    extracted_body = extracted_match.group(0)
    if extracted_ports != dut_ports:
        new_header_line = f".subckt {extracted_header.group(1)} {' '.join(dut_ports)}\n"
        extracted_body = new_header_line + extracted_body[extracted_header.end():]

    return netlist_text[:dut_match.start()] + extracted_body + netlist_text[dut_match.end():]


#: Which parasitics the extracted netlist actually carries, when
#: layout_verify/extract_result.json doesn't say (every file written before
#: magic_layout.stage_extract started recording it). Capacitance only:
#: `extract do resistance`/`ext2spice extresist on` have never been enabled
#: in that stage -- see its own comment on the Magic segfault that blocks
#: them -- so no interconnect R is in any extracted netlist to date.
PARASITICS_UNRECORDED = "capacitance only (no interconnect resistance)"


def _write_extracted_result(verify_dir, sim_dir, block, topology, name, test_name, extracted_path, conditions, status, metrics=None, error=None):
    """extracted_result.json next to this run's own curve PNG(s): what the
    post-layout run measured, in the same per-metric shape run_sim's own
    append_results() rows use (name/typical/min/max/unit/...), so a reader
    can line the two up metric by metric without reformatting either.

    Written for a failed run too (status="error"), on purpose: "the
    post-layout run of this test failed" is a result a release page or the
    GUI should be able to state, and the alternative -- no file -- is
    indistinguishable from "never run" (the same reason
    analog_designer.release.export's own LVS section prints "not run"
    rather than omitting itself).

    `parasitics` records WHICH parasitics the netlist this ran against
    carried, read from magic_layout.stage_extract's own extract_result.json
    -- a post-layout number means something different with and without
    interconnect R, and the release page says which one it is rather than
    letting a reader assume full RC."""
    extract_result_path = sim_dir / "layout_verify" / "extract_result.json"
    parasitics = PARASITICS_UNRECORDED
    if extract_result_path.exists():
        try:
            parasitics = json.loads(extract_result_path.read_text(encoding="utf-8")).get("parasitics") or parasitics
        except (OSError, ValueError):
            pass
    payload = {
        "status": status,
        "variation": name,
        "block": block,
        "topology": topology,
        "test": test_name,
        "conditions": conditions,
        "parasitics": parasitics,
        # Which netlist produced these numbers, and when it was extracted:
        # enough for a reader to tell a post-layout result that predates
        # the layout it is attributed to (the same staleness question the
        # release page asks of the GDS/raster and the LVS run).
        "extracted_netlist": extracted_path.name,
        "extracted_netlist_created": datetime.datetime.fromtimestamp(
            extracted_path.stat().st_mtime).isoformat(timespec="seconds") if extracted_path.exists() else None,
        "created": datetime.datetime.now().isoformat(timespec="seconds"),
        "metrics": metrics or [],
    }
    if error:
        payload["error"] = error
    (verify_dir / "extracted_result.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def run_extracted(block_cfg, params, test_name, variation=None):
    block, topology = workspace.BLOCK, workspace.TOPOLOGY
    test_cfg = workspace.CONFIG["tests"][block][test_name]
    simulator = test_cfg.get("simulator", "ngspice")
    if simulator != "ngspice":
        return {
            "status": "error",
            "error": f"run_extracted only supports ngspice testbenches so far (this test declares {simulator!r})",
        }

    defaults = workspace.CONFIG["defaults"]
    name = variation_name(block, topology, params)
    sim_dir = workspace.sim_root() / name
    # analog_designer_pro.layout.magic_layout.stage_extract's own output --
    # the FULL parasitic netlist, gated there on a clean LVS -- not
    # stage_lvs's own connectivity-only <block>_extracted_lvs.spice.
    extracted_path = sim_dir / "layout_verify" / f"{block}_extracted.spice"
    if not extracted_path.exists():
        return {"status": "error", "error": f"no {extracted_path.name} -- run LVS then Extract for this variation first"}
    extracted_text = extracted_path.read_text(encoding="utf-8")

    verify_dir = sim_dir / "layout_verify" / test_name
    verify_dir.mkdir(parents=True, exist_ok=True)

    def _record(status, metrics=None, error=None):
        """Overwrite this (variation, test)'s own extracted_result.json
        with what THIS run found -- including a failure. Defined this
        early on purpose: every failure below (a netlist that won't
        splice, a netlist/ngspice run that dies, a parser that raises)
        has to replace a previous run's result rather than leave it
        standing. Confirmed the hard way on this project's own cmos_vref:
        a re-extraction whose top-level port list had changed (`vss` ->
        `vss_uq3`, a real split net in the layout) failed to splice, and
        with only the parser-level paths recording, the release page went
        on showing the PREVIOUS run's successful number for a netlist
        that no longer existed."""
        _write_extracted_result(
            verify_dir, sim_dir, block, topology, name, test_name, extracted_path, typical,
            status, metrics=metrics, error=error,
        )

    tb_source = workspace.PROJECT_ROOT / test_cfg["testbench"]
    tb_text = tb_source.read_text(encoding="utf-8")
    sweep_axis = internal_sweep_axis(test_cfg, tb_text)
    typical = typical_conditions(test_cfg, defaults)

    with _resolve_container_ctx(None) as ctx:
        container_rcfile = materialize_variation_shadow(sim_dir, block_cfg, params)

        # Same tb_params_base shape as run_test() -- see that function's own
        # comments for why each of these is set the way it is.
        tb_params_base = dict(defaults)
        tb_params_base["Vavdd"] = defaults["vdd"]
        tb_params_base["filename"] = test_name
        tb_params_base["N"] = "0"
        tb_params_base["models_dir"] = ctx.models_dir
        tb_params_base["stdcell_dir"] = ctx.stdcell_dir
        if sweep_axis:
            key, prefix = sweep_axis
            values = [float(v) for v in test_cfg.get("conditions", {}).get(key, [])]
            if values:
                tb_params_base[f"{prefix}_min"] = min(values)
                tb_params_base[f"{prefix}_max"] = max(values)
        tb_params_base.update(fixed_tb_params(test_cfg, tb_text, sweep_axis))
        vdd_values = test_cfg.get("conditions", {}).get("vdd")
        if vdd_values and len(vdd_values) == 1 and not (sweep_axis and sweep_axis[0] == "vdd"):
            tb_params_base["vdd"] = tb_params_base["Vavdd"] = vdd_values[0]

        container_sim_dir = workspace.exec_path(sim_dir)
        container_verify_dir = f"{container_sim_dir}/layout_verify/{test_name}"
        netlist_result = _netlist(
            ctx.container, test_name, tb_source, typical, tb_params_base,
            verify_dir, container_verify_dir, container_rcfile, ctx,
        )
        if isinstance(netlist_result, dict):
            _record("error", error=netlist_result.get("error"))
            return netlist_result
        netlist_path = netlist_result

        try:
            spliced = _splice_extracted_subckt(netlist_path.read_text(encoding="utf-8"), block, extracted_text)
        except ValueError as exc:
            _record("error", error=str(exc))
            return {"status": "error", "error": str(exc)}
        netlist_path.write_text(spliced, encoding="utf-8")
        # ngspice reads .spiceinit from its own cwd at startup -- this is
        # where run_one_ngspice() (run_sim.py's own normal-flow runner)
        # loads the PDK's compact-model OSDI plugins (psp103, r3_cmc,
        # cap_cmomf/i when present) from, via ctx.spiceinit_text. Missing
        # here entirely until 2026-10-01: confirmed live every MOSFET
        # model (even on a schematic-identical splice) failed with
        # "Unable to find definition of model ...:sg13g2_hv_pmos_psp" --
        # not a netlist problem, ngspice simply never knew to load the
        # plugin that defines it.
        (verify_dir / ".spiceinit").write_text(ctx.spiceinit_text, encoding="utf-8")

        data_file = verify_dir / f"{test_name}_0.data"
        data_file.unlink(missing_ok=True)
        sim_result = docker_exec(
            ctx.container, f'cd "{container_verify_dir}" && timeout 290 "{ctx.ngspice}" -b {netlist_path.name}', timeout=300,
        )
        if not data_file.exists():
            error = (sim_result.stdout + sim_result.stderr)[-2000:]
            _record("error", error=error)
            return {"status": "error", "error": error}

    parser_module = load_parser(test_cfg["parser"])
    try:
        raw = parser_module.extract(data_file)
    except Exception as exc:  # noqa: BLE001 -- report, same as run_test()'s own extract() guard
        error = f"{test_name}: extract() failed on the extracted-netlist run: {exc}"
        _record("error", error=error)
        return {"status": "error", "error": error}

    runs = [{"conditions": typical, **raw}]
    plot_base = verify_dir / f"{test_name}_extracted"
    for stale in verify_dir.glob(f"{test_name}_extracted*.png"):
        stale.unlink()
    try:
        outcome = parser_module.evaluate(runs, test_cfg["outputs"], typical, plot_base=plot_base)
    except Exception as exc:  # noqa: BLE001 -- report, same as run_test()'s own evaluate() guard
        error = f"{test_name}: evaluate() failed on the extracted-netlist run: {exc}"
        _record("error", error=error)
        return {"status": "error", "error": error}
    _record("success", metrics=outcome)
    return {"status": "success", "variation": name, "test": test_name, "result": outcome, "plot": str(plot_base) + ".png"}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("test_name")
    parser.add_argument(
        "variation", nargs="?", default=None,
        help="existing variation name from sim/variations.jsonl; omit to target the config.json default-parameter variation",
    )
    parser.add_argument("--project-root", default=None, help="project folder to operate on; defaults to the last-opened folder, else CWD")
    parser.add_argument("--block", default=None, help="block to operate on; defaults to the first declared in config.json")
    parser.add_argument("--topology", default=None, help="topology to operate on; defaults to the first declared for --block")
    args = parser.parse_args()

    workspace.open_folder(args.project_root, block=args.block, topology=args.topology)
    config = workspace.CONFIG
    block_cfg = config["blocks"][workspace.BLOCK]["topologies"][workspace.TOPOLOGY]
    params = _variation_params(args.variation) if args.variation else {
        n: pdef["default"] for n, pdef in block_cfg["parameters"].items()
    }

    result = run_extracted(block_cfg, params, args.test_name, variation=args.variation)
    print(json.dumps(result, indent=2))
    if result["status"] != "success":
        sys.exit(1)


if __name__ == "__main__":
    main()
