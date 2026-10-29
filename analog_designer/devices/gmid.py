#!/usr/bin/env python3
"""gm/Id device tables: emit the ngspice deck that sweeps one MOSFET, turn
the resulting wrdata columns into a table, and solve a sizing problem
against it.

This is the gm/Id-based design methodology reduced to three lookup curves
per channel length -- transit frequency, intrinsic gain and current density,
all against transconductance efficiency gm/Id -- plus the ACM (Advanced
Compact Model) parameters n, I_SQ and V_T, which fall out of the same sweep
for free because the gm/Id curve IS the ACM extraction curve.

Tk-free and side-effect-free, same contract as analog_designer/results/data.py:
the GUI calculator, the CLI and any batch caller share this module. Nothing
here executes anything -- build_deck() returns netlist text and
table_from_sweeps() consumes wrdata columns; running them is
analog_designer/sim/characterize_device.py's job.

Nothing here is PDK-specific either. The device names, the model files and
section names, and -- critically -- the name of the transistor *inside* the
PDK's subcircuit wrapper all come from the project's config.json under
technology.characterization, so a second PDK is a second config.json rather
than a second branch in this file.

Two ngspice facts govern the deck, both measured against gf180mcu's
nfet_03v3 on ngspice-46 (2026-10-28), and both silent when you get them
wrong:

  1. `save` MUST list every @m.<inst>[<param>] BEFORE the analysis runs.
     Without it ngspice still accepts `let gm = @m.xm1.m0[gm]` and still
     writes a full-length column -- but every row holds the SAME value, the
     one from the final sweep point. No error, no warning, and the table
     looks plausible until you notice gm/Id came out constant.

  2. `cgg` is the INTRINSIC gate capacitance only. The overlap terms are
     separate ask parameters and must be added by hand:
         Cgg_total = cgg + cgso + cgdo + cgbo
     Verified against an AC measurement of Im(I_g)/(2*pi*f) at VGS=1.2V,
     VDS=1.65V: AC gave 13.0206 fF, cgg alone 6.888 fF, the four-term sum
     13.0206 fF. Using cgg alone overstates ft by 1.89x at that bias.
     (The four-term sum is what Hspice reports as cggbo(), which is what the
     gm/Id literature means by Cgg.)
"""
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from analog_designer.sim.spice_value import parse_spice_value

BOLTZMANN = 1.380649e-23
ELECTRON_CHARGE = 1.602176634e-19

#: wrdata column order after the sweep variable (vgs). See build_deck().
SWEEP_COLUMNS = ("id", "gm", "gds", "cgg", "vth", "vdsat")

#: Instance parameters that must be `save`d for the sweep to be real rather
#: than frozen -- see fact 1 above. cgg's three overlap companions are here
#: because of fact 2.
_SAVED_PARAMS = ("id", "gm", "gds", "cgg", "cgso", "cgdo", "cgbo", "vth", "vdsat")


def thermal_voltage(temperature_c):
    """phi_t = kT/q, for a temperature given in Celsius."""
    return BOLTZMANN * (temperature_c + 273.15) / ELECTRON_CHARGE


# ----------------------------------------------------------------- spec ---

def spec_from_config(config, device, corner="tt", temperature=25.0):
    """The sweep definition for one device, read out of a project's
    config.json. Everything PDK-specific enters the engine here and nowhere
    else.

    Raises with a message naming what is missing, because a half-populated
    technology.characterization block is the likeliest first mistake."""
    char = config.get("technology", {}).get("characterization")
    if not char:
        raise KeyError(
            "config.json has no technology.characterization block -- the gm/Id "
            "engine needs it to know this PDK's model files and device names"
        )
    devices = char.get("devices", {})
    if device not in devices:
        known = ", ".join(sorted(devices)) or "(none declared)"
        raise KeyError(f"device {device!r} is not in technology.characterization.devices: {known}")
    dev = devices[device]

    corners = char.get("corner_sections", {})
    if corner not in corners:
        known = ", ".join(sorted(corners)) or "(none declared)"
        raise ValueError(f"corner {corner!r} has no section name in corner_sections: {known}")

    vdd = float(dev["vdd"])
    return {
        "device": device,
        "type": dev.get("type", "n"),
        "corner": corner,
        "corner_section": corners[corner],
        "temperature": float(temperature),
        "vdd": vdd,
        # The gm/Id charts are conventionally extracted at VDS = VDD/2: far
        # enough above vdsat that ro is not collapsing. They are only valid
        # near that VDS -- intrinsic gain can fall 4-5x close to vdsat -- so
        # the value travels with the table and gets shown to the user rather
        # than assumed.
        "vds": float(dev.get("vds", vdd / 2.0)),
        "vsb": float(dev.get("vsb", 0.0)),
        "w_char": parse_spice_value(dev["w_char"]),
        "nf": int(dev.get("nf", 1)),
        "w_min": parse_spice_value(dev["w_min"]),
        "l_min": parse_spice_value(dev["l_min"]),
        "l_values": [parse_spice_value(v) for v in dev["l_sweep"]],
        "vgs_start": float(dev.get("vgs_start", 0.0)),
        "vgs_stop": float(dev.get("vgs_stop", vdd)),
        "vgs_step": float(dev.get("vgs_step", 0.005)),
        "inner_device": char["inner_device"],
        "models_include": list(char.get("models_include", [])),
        "models_lib": [list(pair) for pair in char.get("models_lib", [])],
        "options": list(char.get("options", ["warn=1", "rshunt=1e12"])),
    }


def cache_key(spec, pdk_name, models_signature=""):
    """Content hash over everything that could change a number in the table.

    `models_signature` is whatever the caller can cheaply learn about the
    model files themselves (sizes and mtimes, or a digest). Without it,
    editing the PDK's own .spice deck would silently reuse a stale table."""
    payload = json.dumps(
        {"spec": spec, "pdk": pdk_name, "models": models_signature},
        sort_keys=True, separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]


def cache_path(sim_root, spec, pdk_name, models_signature=""):
    key = cache_key(spec, pdk_name, models_signature)
    return Path(sim_root) / "_device_tables" / f"{spec['device']}-{spec['corner']}-{key}.json"


# ----------------------------------------------------------------- deck ---

def build_deck(spec, l_value, models_dir, out_data_path):
    """The whole ngspice input for ONE channel length, as text.

    The DUT is instantiated directly -- no xschem, no symbol, no block or
    topology. That follows sim/extract_model.py's precedent: a test measures
    a model that already exists, this produces one, so it has no business in
    config.json's `tests`.

    Polarity: an n-type device sits with its source at ground and sweeps VGS
    up; a p-type one is mirrored into negative supplies so the same column
    order comes back. Signs are normalized on the Python side, in
    table_from_sweeps().

    Deliberately NO `.option gmin`. Raising gmin with a large m= injected
    ~389uA of phantom DC current in this house's VCO work and read a real
    conductance as exactly 0.0000 mS (tb/vco/tb_zout.sch). This DUT has m=1,
    but the habit is the rule."""
    sign = -1.0 if spec["type"] == "p" else 1.0
    inner = spec["inner_device"]
    vds = sign * abs(spec["vds"])
    vgs_start = sign * abs(spec["vgs_start"])
    vgs_stop = sign * abs(spec["vgs_stop"])
    step = sign * abs(spec["vgs_step"])
    vbulk = -sign * abs(spec["vsb"])

    lines = [
        "* gm/Id characterization: {dev} L={l:.6g} W={w:.6g} VDS={vds:.6g} "
        "VSB={vsb:.6g} corner={sec} T={t:.6g}C".format(
            dev=spec["device"], l=l_value, w=spec["w_char"], vds=vds,
            vsb=spec["vsb"], sec=spec["corner_section"], t=spec["temperature"],
        ),
    ]
    for name in spec["models_include"]:
        lines.append(f".include {models_dir}/{name}")
    for name, section in spec["models_lib"]:
        lines.append(f".lib {models_dir}/{name} {section.replace('<corner>', spec['corner_section'])}")
    lines.append(f".option TEMP={spec['temperature']:.6g}")
    lines.extend(f".option {opt}" for opt in spec["options"])
    lines += [
        f"xm1 d g s b {spec['device']} w={spec['w_char']:.6g} l={l_value:.6g} nf={spec['nf']}",
        f"vds d 0 {vds:.6g}",
        "vsrc s 0 0",
        f"vbulk b 0 {vbulk:.6g}",
        f"vgs g 0 {vgs_start:.6g}",
        ".control",
        "save " + " ".join(f"@m.xm1.{inner}[{p}]" for p in _SAVED_PARAMS),
        f"dc vgs {vgs_start:.6g} {vgs_stop:.6g} {step:.6g}",
        f"let o_id  = @m.xm1.{inner}[id]",
        f"let o_gm  = @m.xm1.{inner}[gm]",
        f"let o_gds = @m.xm1.{inner}[gds]",
        # Fact 2 of the module docstring: intrinsic cgg plus the three overlaps.
        f"let o_cgg = @m.xm1.{inner}[cgg] + @m.xm1.{inner}[cgso]"
        f" + @m.xm1.{inner}[cgdo] + @m.xm1.{inner}[cgbo]",
        f"let o_vth = @m.xm1.{inner}[vth]",
        f"let o_vds = @m.xm1.{inner}[vdsat]",
        "set wr_singlescale",
        f"wrdata {out_data_path} o_id o_gm o_gds o_cgg o_vth o_vds",
        "quit",
        ".endc",
        ".end",
        "",
    ]
    return "\n".join(lines)


def parse_wrdata(text):
    """ngspice `wrdata` output under `set wr_singlescale`: whitespace-separated
    ASCII, sweep variable first, then one column per written vector.

    A non-finite value is an error rather than a NaN quietly entering a
    design chart -- the same stance tb/_shared/parser_common.py's read_data
    takes on a diverged run."""
    rows = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            values = [float(token) for token in line.split()]
        except ValueError:
            continue
        if not all(math.isfinite(value) for value in values):
            raise ValueError(f"non-finite value in ngspice output: {line!r}")
        rows.append(values)
    if not rows:
        raise ValueError("ngspice produced no usable data rows")
    width = 1 + len(SWEEP_COLUMNS)
    if len(rows[0]) != width:
        raise ValueError(f"expected {width} columns from wrdata, got {len(rows[0])}")
    array = np.asarray(rows, dtype=float)
    columns = {"vgs": array[:, 0]}
    for index, name in enumerate(SWEEP_COLUMNS, start=1):
        columns[name] = array[:, index]
    return columns


# ------------------------------------------------------------------ ACM ---

def fit_acm(drain_current, gmid, width, length, phit,
            n_range=(1.0, 1.8), n_steps=801, band=(0.1, 100.0)):
    """Extract the ACM slope factor n and sheet specific current I_SQ from one
    gm/Id sweep, two independent ways, and report how far apart they land.

    In saturation the ACM gives a single closed form,

        gm/Id = 2 / ( n*phit*(1 + sqrt(1 + i_f)) ),   i_f = Id / (I_SQ * W/L)

    whose two anchors are what the classical extraction uses:

      * Deep in weak inversion i_f -> 0 and gm/Id -> 1/(n*phit), so the peak
        of the measured gm/Id curve gives n outright. (Equivalently, the
        subthreshold swing: S = ln(10)*n*phit.)
      * At i_f = 3, sqrt(1+i_f) = 2 and gm/Id is exactly 2/3 of its peak. So
        the bias where gm/Id has fallen to two thirds of the peak is, by the
        model's own algebra, where Id = 3*I_S -- which gives I_S with no
        fitting at all.

    That pair (the "weak" fields) is the primary result: it is the classical
    method, and it makes the peak of the measured curve and the reported n
    agree by construction rather than by luck.

    The second way (the "fit" fields) is a least-squares cross-check. The form
    inverts exactly, i_f = (2/(n*phit*(gm/Id)) - 1)**2 - 1, so I_S = Id/i_f
    MUST be constant along the sweep if one n describes the device; scanning n
    for the flattest I_S through moderate inversion measures how well it does.

    The two are not expected to match exactly -- n really does drift with gate
    bias, as the depletion capacitance falls against Cox -- and the gap is
    reported rather than averaged away, because its size is what decides
    whether a single-n ACM is usable for this process.

    Returns None when the sweep never reaches moderate inversion."""
    gmid = np.asarray(gmid, dtype=float)
    drain_current = np.asarray(drain_current, dtype=float)
    result = {}

    # --- the classical two-anchor extraction ---------------------------
    peak = int(np.argmax(gmid))
    gmid_peak = float(gmid[peak])
    n_weak = 1.0 / (gmid_peak * phit)
    result["n_weak"] = n_weak
    result["swing_mv_dec"] = float(math.log(10.0) * n_weak * phit * 1e3)

    # Forward from the peak, gm/Id falls monotonically, so the 2/3 crossing is
    # unique; Id there is 3*I_S.
    target = (2.0 / 3.0) * gmid_peak
    tail_g, tail_i = gmid[peak:], drain_current[peak:]
    below = np.nonzero(tail_g <= target)[0]
    if below.size and below[0] > 0:
        index = int(below[0])
        g0, g1 = tail_g[index - 1], tail_g[index]
        i0, i1 = tail_i[index - 1], tail_i[index]
        at_if3 = i0 if g0 == g1 else i0 + (target - g0) * (i1 - i0) / (g1 - g0)
        result["i_s_weak"] = float(at_if3) / 3.0
        result["i_sq_weak"] = result["i_s_weak"] * length / width

    # --- the least-squares collapse, as a cross-check ------------------
    best = None
    for n in np.linspace(n_range[0], n_range[1], n_steps):
        # gm/Id can never exceed 1/(n*phit), so x = 2/(n*phit*gm/Id) >= 2.
        # Samples violating that are out of the model's reach for this n.
        x = 2.0 / (n * phit * gmid)
        usable = x > 2.0
        if usable.sum() < 20:
            continue
        i_f = (x[usable] - 1.0) ** 2 - 1.0
        i_s = drain_current[usable] / i_f
        window = (i_f >= band[0]) & (i_f <= band[1]) & (i_s > 0)
        if window.sum() < 20:
            continue
        logs = np.log(i_s[window])
        spread = float(np.std(logs))
        if best is None or spread < best["spread"]:
            best = {
                "n_fit": float(n),
                "i_s_fit": float(np.exp(np.median(logs))),
                "spread": spread,
                "samples": int(window.sum()),
            }
    if best is None:
        return result or None
    best["i_sq_fit"] = best["i_s_fit"] * length / width
    result.update(best)
    result["n_gap"] = (result["n_fit"] - result["n_weak"]) / result["n_weak"]

    # What every other module should use: the classical pair, because it is
    # anchored to measured features of the curve rather than to whatever
    # window the fit was handed.
    result["n"] = result["n_weak"]
    result["i_sq"] = result.get("i_sq_weak", best["i_sq_fit"])
    return result


def acm_threshold(vgs, gmid):
    """V_T in the ACM sense, read off the gm/Id curve: the gate voltage where
    gm/Id has fallen to half its weak-inversion peak. (In the ACM, V_T is
    where drift and diffusion contribute equally; with n roughly constant
    that lands at the half-peak point of gm/Id.)

    Searched forward from the peak, where gm/Id is monotonically falling, so
    the crossing is unique."""
    vgs = np.asarray(vgs, dtype=float)
    gmid = np.asarray(gmid, dtype=float)
    peak = int(np.argmax(gmid))
    target = gmid[peak] / 2.0
    tail_v, tail_g = vgs[peak:], gmid[peak:]
    below = np.nonzero(tail_g <= target)[0]
    if below.size == 0:
        return None
    index = int(below[0])
    if index == 0:
        return float(tail_v[0])
    # Linear interpolation between the straddling samples.
    g0, g1 = tail_g[index - 1], tail_g[index]
    v0, v1 = tail_v[index - 1], tail_v[index]
    if g0 == g1:
        return float(v1)
    return float(v0 + (target - g0) * (v1 - v0) / (g1 - g0))


# ---------------------------------------------------------------- cache ---

def save_table(table, path):
    """Write a table to its cache file. The directory is created; the write
    is atomic so an interrupted run cannot leave a half-parsed table that
    the next one would happily load."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(table), encoding="utf-8")
    temporary.replace(path)
    return path


def load_table(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


# Table assembly, lookup and the sizing solver live in _gmid_table.py (file
# size only); they are part of this module's surface. Imported last so that
# everything _gmid_table reaches back for is already defined.
from analog_designer.devices._gmid_table import (  # noqa: E402
    STRONG_LIMIT,
    WEAK_LIMIT,
    derived,
    entry_for_length,
    gmid_for_ft,
    inversion_label,
    lookup,
    solve,
    table_from_sweeps,
)


def find_cached_tables(sim_root, device, corner=None):
    """Every cached table for a device, newest first.

    The cache key folds in the PDK name and the model files' own
    size/mtime, both of which cost a container round-trip to learn. A GUI
    opening a table has no business paying that, so it looks the cache up by
    name instead and reports what it found. Anything that must be certain the
    table matches the PDK on disk goes through characterize() and its real
    key."""
    directory = Path(sim_root) / "_device_tables"
    if not directory.is_dir():
        return []
    found = []
    for path in directory.glob(f"{device}-*.json"):
        try:
            table = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        spec = table.get("spec", {})
        if spec.get("device") != device:
            continue
        if corner and spec.get("corner") != corner:
            continue
        found.append((table.get("created", ""), path, table))
    found.sort(key=lambda row: row[0], reverse=True)
    return [(path, table) for _, path, table in found]


def pick_cached_table(sim_root, spec):
    """The cached table that best matches a spec built from config.json.

    Prefers one swept at the same finger width, drain bias and channel
    lengths -- a table left behind by a --w-char experiment is a real table,
    but it is not the one the project means by "the charts"."""
    candidates = find_cached_tables(sim_root, spec["device"], spec["corner"])
    if not candidates:
        return None, None
    keys = ("w_char", "vds", "vsb", "temperature", "l_values", "nf")
    for path, table in candidates:
        other = table.get("spec", {})
        if all(other.get(key) == spec.get(key) for key in keys):
            return path, table
    return candidates[0]
