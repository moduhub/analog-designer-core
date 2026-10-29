#!/usr/bin/env python3
"""Sweep one MOSFET through ngspice and persist its gm/Id design table.

    python -m analog_designer.sim.characterize_device --device nfet_03v3

This produces a model rather than measuring one, so -- following
sim/extract_model.py's precedent -- it is deliberately NOT one of
config.json's `tests`: no block, no topology, no xschem schematic, no
parser module, no results.jsonl row. What it needs from the project is one
declarative block, technology.characterization, read by
analog_designer/devices/gmid.py.

The output is a cached JSON table under sim/_device_tables/, keyed by a
content hash of the sweep definition and the PDK's own model files. sim/ is
gitignored by every project here, which is the point: the table is derived
data that any clone can regenerate in seconds, not an artifact to version.

Runs wherever this project's simulations run -- managed_executor() gives a
throwaway container in docker mode and the host tools in host mode -- so no
docker-specific code appears below.
"""
import argparse
import sys
from pathlib import Path

from analog_designer.core import workspace
from analog_designer.devices import gmid
from analog_designer.sim import run_sim

#: One DC sweep of a single transistor is trivial work; the timeout only has
#: to be generous enough that a pathological bias point cannot hang a run.
SWEEP_TIMEOUT_S = 120


def models_signature(executor, models_dir, spec):
    """Size and mtime of every model file this sweep includes, as one string.

    Folded into the cache key so that editing the PDK's own .spice deck --
    or pointing at a different PDK release that happens to use the same file
    names -- invalidates the table instead of silently reusing it."""
    names = list(spec["models_include"]) + [pair[0] for pair in spec["models_lib"]]
    unique = sorted(set(names))
    paths = " ".join(f'"{models_dir}/{name}"' for name in unique)
    # -L: gf180mcu ships sm141064.spice as a symlink into another PDK
    # revision's directory, so stat'ing the link itself would report the
    # link's own mtime and miss an edit to the file it points at.
    result = run_sim.docker_exec(executor, f"stat -L -c '%n %s %Y' {paths}", timeout=30)
    if result.returncode != 0:
        sys.exit(f"could not stat the PDK model files under {models_dir}:\n{result.stderr}")
    return result.stdout.strip()


def run_one_length(executor, spec, l_value, models_dir, work_dir):
    """One ngspice DC sweep. Returns the parsed wrdata columns."""
    tag = f"{spec['device']}_{spec['corner']}_L{l_value * 1e9:.0f}n"
    deck_path = work_dir / f"{tag}.spice"
    data_path = work_dir / f"{tag}.data"
    log_path = work_dir / f"{tag}.log"
    data_path.unlink(missing_ok=True)

    deck = gmid.build_deck(spec, l_value, models_dir, workspace.exec_path(data_path))
    deck_path.write_text(deck, encoding="utf-8")

    container_dir = workspace.exec_path(work_dir)
    command = (
        f'cd "{container_dir}" && export OMP_NUM_THREADS=1; '
        f'ngspice -b "{deck_path.name}" > "{log_path.name}" 2>&1'
    )
    result = run_sim.docker_exec(executor, command, timeout=SWEEP_TIMEOUT_S)
    log_text = log_path.read_text(encoding="utf-8", errors="replace") if log_path.exists() else ""
    if result.returncode != 0 or not data_path.exists():
        sys.exit(
            f"ngspice failed for {spec['device']} at L={l_value * 1e6:g}um "
            f"(exit {result.returncode}).\n{log_text[-2000:]}"
        )
    # ngspice exits 0 on plenty of real failures, so the log is checked too.
    for line in log_text.splitlines():
        if line.strip().lower().startswith("error"):
            sys.exit(f"ngspice reported an error for L={l_value * 1e6:g}um:\n{log_text[-2000:]}")
    return gmid.parse_wrdata(data_path.read_text(encoding="utf-8"))


def characterize(device, corner="tt", temperature=25.0, force=False, quiet=False,
                 w_char=None):
    """Build (or reuse) the gm/Id table for one device. Returns (table, path).

    `w_char` overrides the characterization width from config.json. The whole
    methodology rests on gm/Id, ft and gm.ro being independent of W (double
    the width and Id, gm and Cgg all double, so every ratio holds), and that
    assumption is worth re-measuring rather than trusting: sweeping the same
    device at two widths and comparing is what this override is for. It feeds
    the cache key, so the extra tables sit beside the real one instead of
    overwriting it."""
    spec = gmid.spec_from_config(workspace.CONFIG, device, corner, temperature)
    if w_char is not None:
        spec["w_char"] = float(w_char)
    work_dir = workspace.sim_root() / "_device_tables" / "_work"
    work_dir.mkdir(parents=True, exist_ok=True)

    with run_sim.managed_executor() as executor:
        pdk_name = run_sim.get_pdk_name(executor)
        models_dir = run_sim.get_pdk_dir(executor, "libs.tech/ngspice")
        signature = models_signature(executor, models_dir, spec)
        path = gmid.cache_path(workspace.sim_root(), spec, pdk_name, signature)
        if path.exists() and not force:
            if not quiet:
                print(f"reusing cached table {path}")
            return gmid.load_table(path), path

        sweeps = []
        with workspace.core_pool().reserve(1):
            for l_value in spec["l_values"]:
                if not quiet:
                    print(f"  sweeping {device} L={l_value * 1e6:g}um ...", flush=True)
                sweeps.append((l_value, run_one_length(executor, spec, l_value, models_dir, work_dir)))

    table = gmid.table_from_sweeps(spec, sweeps, pdk_name, signature)
    gmid.save_table(table, path)
    if not quiet:
        print(f"wrote {path}")
    return table, path


def summarize(table):
    """The per-length numbers worth reading at a glance, as text rows."""
    spec = table["spec"]
    phit = gmid.thermal_voltage(spec["temperature"])
    nan = float("nan")
    lines = [
        f"{spec['device']}  corner={spec['corner']}  T={spec['temperature']:g}C  "
        f"VDS={spec['vds']:g}V  VSB={spec['vsb']:g}V  W={spec['w_char'] * 1e6:g}um  "
        f"nf={spec['nf']}  phi_t={phit * 1e3:.3f}mV",
        "",
        "design charts, read at gm/Id = 15 S/A:",
        f"{'L (um)':>8} {'ft (GHz)':>9} {'gm.ro':>8} {'Id/W (uA/um)':>13} {'VGS (V)':>8} {'Vov (mV)':>9}",
    ]
    for entry in table["lengths"]:
        try:
            row = (gmid.lookup(table, entry["l"], 15.0, "ft") / 1e9,
                   gmid.lookup(table, entry["l"], 15.0, "gmro"),
                   gmid.lookup(table, entry["l"], 15.0, "idw"),
                   gmid.lookup(table, entry["l"], 15.0, "vgs"),
                   gmid.lookup(table, entry["l"], 15.0, "vov") * 1e3)
        except ValueError:
            row = (nan,) * 5
        lines.append(f"{entry['l'] * 1e6:8.2f} {row[0]:9.2f} {row[1]:8.1f} "
                     f"{row[2]:13.4f} {row[3]:8.4f} {row[4]:9.1f}")

    lines += [
        "",
        "ACM parameters, two independent extractions:",
        f"{'L (um)':>8} {'gm/Id max':>10} {'S (mV/dec)':>11} {'n weak':>8} {'n fit':>8} "
        f"{'gap':>7} {'I_SQ weak':>10} {'I_SQ fit':>9} {'spread':>7} {'V_T (V)':>8}",
    ]
    for entry in table["lengths"]:
        acm = entry.get("acm") or {}
        lines.append(
            f"{entry['l'] * 1e6:8.2f} {entry['gmid_max']:10.2f} "
            f"{acm.get('swing_mv_dec', nan):11.2f} "
            f"{acm.get('n_weak', nan):8.4f} {acm.get('n_fit', nan):8.4f} "
            f"{acm.get('n_gap', nan) * 100:6.1f}% "
            f"{acm.get('i_sq_weak', nan) * 1e9:10.2f} "
            f"{acm.get('i_sq_fit', nan) * 1e9:9.2f} "
            f"{acm.get('spread', nan):7.4f} "
            f"{entry.get('vt_acm') or nan:8.4f}"
        )
    lines += [
        "",
        "I_SQ in nA. 'n weak' is 1/(phi_t * peak gm/Id); 'n fit' is the slope factor",
        "that flattens I_S = Id/i_f through moderate inversion. 'gap' and 'spread'",
        "together say how far a single-n ACM is from describing this device.",
    ]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--project-root")
    parser.add_argument("--block")
    parser.add_argument("--topology")
    parser.add_argument("--device", required=True,
                        help="a key of technology.characterization.devices in config.json")
    parser.add_argument("--corner", default="tt")
    parser.add_argument("--temperature", type=float, default=25.0)
    parser.add_argument("--w-char", type=float, metavar="METRES",
                        help="override the characterization width (for the "
                             "W-independence check; see characterize())")
    parser.add_argument("--force", action="store_true",
                        help="re-sweep even when a matching cached table exists")
    parser.add_argument("--plots", metavar="DIR", nargs="?", const="",
                        help="also write the design charts as PNG (default: "
                             "sim/_device_tables/plots/<device>-<corner>)")
    args = parser.parse_args()

    workspace.open_folder(args.project_root, block=args.block, topology=args.topology)
    table, path = characterize(args.device, args.corner, args.temperature, args.force,
                               w_char=args.w_char)
    print()
    print(summarize(table))

    if args.plots is not None:
        from analog_designer.devices import gmid_plots
        target = Path(args.plots) if args.plots else (
            workspace.sim_root() / "_device_tables" / "plots" / f"{args.device}-{args.corner}"
        )
        written = gmid_plots.write_charts(table, target)
        print()
        for item in written:
            print(f"wrote {item}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
