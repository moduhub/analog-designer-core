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

Usage: python -m analog_designer.sim.run_extracted TEST_NAME [VARIATION]
[--project-root PATH] [--block NAME] [--topology NAME]
"""
import argparse
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

    Does NOT verify pin name/order agreement between the two subckts
    beyond both matching `block`'s own name -- a clean Extract+LVS run
    confirms DEVICE connectivity matches, not that ext2spice emitted its
    .subckt header's port list in the same order the schematic-derived one
    uses. ngspice matches subckt ports positionally, so a real order
    mismatch here would silently simulate a mis-wired circuit rather than
    erroring -- verify this against a real extracted netlist for this PDK
    before trusting results from this function, per this feature's own
    implementation plan."""
    pattern = re.compile(_SUBCKT_RE_TEMPLATE.format(block=re.escape(block)), re.IGNORECASE | re.MULTILINE | re.DOTALL)
    dut_match = pattern.search(netlist_text)
    if dut_match is None:
        raise ValueError(f"testbench netlist has no .subckt {block} ... .ends")
    extracted_match = pattern.search(extracted_text)
    if extracted_match is None:
        raise ValueError(f"extracted netlist has no .subckt {block} ... .ends")
    return netlist_text[:dut_match.start()] + extracted_match.group(0) + netlist_text[dut_match.end():]


def run_extracted(block_cfg, params, test_name, variation=None):
    block, topology = workspace.BLOCK, workspace.TOPOLOGY
    test_cfg = workspace.CONFIG["tests"][block][test_name]
    simulator = test_cfg.get("simulator", "ngspice")
    if simulator != "ngspice":
        return {
            "status": "error",
            "error": f"run_extracted only supports ngspice testbenches so far (this test declares {simulator!r})",
        }

    layout_target = variation or topology
    layout_dir = workspace.PROJECT_ROOT / "layout" / block / layout_target
    extracted_path = layout_dir / f"{block}_extracted.spice"
    if not extracted_path.exists():
        return {"status": "error", "error": f"no {extracted_path.name} -- run Extract + LVS for this variation first"}
    extracted_text = extracted_path.read_text(encoding="utf-8")

    defaults = workspace.CONFIG["defaults"]
    name = variation_name(block, topology, params)
    sim_dir = workspace.PROJECT_ROOT / "sim" / name
    verify_dir = sim_dir / "layout_verify" / test_name
    verify_dir.mkdir(parents=True, exist_ok=True)

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

        container_sim_dir = f"{workspace.container_project_root()}/sim/{name}"
        container_verify_dir = f"{container_sim_dir}/layout_verify/{test_name}"
        netlist_result = _netlist(
            ctx.container, test_name, tb_source, typical, tb_params_base,
            verify_dir, container_verify_dir, container_rcfile, ctx,
        )
        if isinstance(netlist_result, dict):
            return netlist_result
        netlist_path = netlist_result

        try:
            spliced = _splice_extracted_subckt(netlist_path.read_text(encoding="utf-8"), block, extracted_text)
        except ValueError as exc:
            return {"status": "error", "error": str(exc)}
        netlist_path.write_text(spliced, encoding="utf-8")

        data_file = verify_dir / f"{test_name}_0.data"
        data_file.unlink(missing_ok=True)
        sim_result = docker_exec(
            ctx.container, f'cd "{container_verify_dir}" && timeout 290 ngspice -b {netlist_path.name}', timeout=300,
        )
        if not data_file.exists():
            return {"status": "error", "error": (sim_result.stdout + sim_result.stderr)[-2000:]}

    parser_module = load_parser(test_cfg["parser"])
    try:
        raw = parser_module.extract(data_file)
    except Exception as exc:  # noqa: BLE001 -- report, same as run_test()'s own extract() guard
        return {"status": "error", "error": f"{test_name}: extract() failed on the extracted-netlist run: {exc}"}

    runs = [{"conditions": typical, **raw}]
    plot_base = verify_dir / f"{test_name}_extracted"
    for stale in verify_dir.glob(f"{test_name}_extracted*.png"):
        stale.unlink()
    try:
        outcome = parser_module.evaluate(runs, test_cfg["outputs"], typical, plot_base=plot_base)
    except Exception as exc:  # noqa: BLE001 -- report, same as run_test()'s own evaluate() guard
        return {"status": "error", "error": f"{test_name}: evaluate() failed on the extracted-netlist run: {exc}"}
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
