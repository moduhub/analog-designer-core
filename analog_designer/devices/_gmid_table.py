#!/usr/bin/env python3
"""Table assembly, cache I/O and the sizing solver for gm/Id design charts.

Split from analog_designer/devices/gmid.py purely for file size: gmid.py owns
the ngspice side (what the deck must say, what the columns mean) and the ACM
extraction, this owns what a caller does with the result. Import both through
analog_designer.devices.gmid, which re-exports these names.
"""
import datetime

import numpy as np

from analog_designer.devices import gmid as _gmid

#: Boundaries of the moderate-inversion band, in ACM inversion coefficient.
#: Below weak, above strong -- the conventional decade-either-side of i_f = 1.
WEAK_LIMIT = 1.0
STRONG_LIMIT = 100.0


def inversion_label(i_f):
    if i_f is None:
        return ""
    if i_f < WEAK_LIMIT:
        return "weak"
    if i_f > STRONG_LIMIT:
        return "strong"
    return "moderate"


# ---------------------------------------------------------------- build ---

def table_from_sweeps(spec, sweeps, pdk_name="", models_signature=""):
    """Assemble the persisted table from one wrdata column-set per channel
    length.

    `sweeps` is [(l_value, columns)] as parse_wrdata() returns them. Signs are
    normalized here, once, so every consumer downstream sees positive currents
    and capacitances whether the device is n- or p-type.

    Only the raw sweep is stored. Everything a chart or the calculator needs
    is recomputed by derived() on load, so a change to those formulas can
    never be shadowed by a stale cached column."""
    phit = _gmid.thermal_voltage(spec["temperature"])
    entries = []
    for l_value, columns in sorted(sweeps, key=lambda pair: pair[0]):
        drain_current = np.abs(columns["id"])
        transconductance = np.abs(columns["gm"])
        with np.errstate(divide="ignore", invalid="ignore"):
            gmid = np.where(drain_current > 0, transconductance / drain_current, 0.0)
        acm = _gmid.fit_acm(drain_current, gmid, spec["w_char"], l_value, phit)
        entry = {
            "l": float(l_value),
            "vgs": [float(v) for v in np.abs(columns["vgs"])],
            "id": [float(v) for v in drain_current],
            "gm": [float(v) for v in transconductance],
            "gds": [float(v) for v in np.abs(columns["gds"])],
            "cgg": [float(v) for v in np.abs(columns["cgg"])],
            "vth": [float(v) for v in np.abs(columns["vth"])],
            "vdsat": [float(v) for v in np.abs(columns["vdsat"])],
            "acm": acm,
            "gmid_max": float(np.max(gmid)),
            "vt_acm": _gmid.acm_threshold(np.abs(columns["vgs"]), gmid),
        }
        entries.append(entry)
    return {
        "format": 1,
        "created": datetime.datetime.now().isoformat(timespec="seconds"),
        "pdk": pdk_name,
        "models_signature": models_signature,
        "spec": spec,
        "lengths": entries,
    }


def derived(table, entry):
    """The design-chart quantities, recomputed from the stored raw sweep.

    Returns arrays restricted to the usable, strictly monotonic part of the
    curve: from the gm/Id peak onward, where gm/Id falls as VGS rises. Below
    that peak the device is in deep subthreshold and gm/Id turns over, which
    would make the inverse lookup the calculator depends on ambiguous.

    `saturated` is a mask, not a filter: the caller decides whether to drop
    the points where the applied VDS has fallen below vdsat, because a chart
    may legitimately want to draw them greyed rather than not at all."""
    spec = table["spec"]
    vgs = np.asarray(entry["vgs"], dtype=float)
    drain_current = np.asarray(entry["id"], dtype=float)
    transconductance = np.asarray(entry["gm"], dtype=float)
    output_conductance = np.asarray(entry["gds"], dtype=float)
    gate_capacitance = np.asarray(entry["cgg"], dtype=float)
    vth = np.asarray(entry["vth"], dtype=float)
    vdsat = np.asarray(entry["vdsat"], dtype=float)

    with np.errstate(divide="ignore", invalid="ignore"):
        gmid = np.where(drain_current > 0, transconductance / drain_current, 0.0)
    start = int(np.argmax(gmid))
    sl = slice(start, None)

    gmid = gmid[sl]
    drain_current = drain_current[sl]
    transconductance = transconductance[sl]
    output_conductance = output_conductance[sl]
    gate_capacitance = gate_capacitance[sl]

    with np.errstate(divide="ignore", invalid="ignore"):
        ft = np.where(gate_capacitance > 0,
                      transconductance / (2.0 * np.pi * gate_capacitance), 0.0)
        gmro = np.where(output_conductance > 0,
                        transconductance / output_conductance, 0.0)

    out = {
        "vgs": vgs[sl],
        "id": drain_current,
        "gm": transconductance,
        "gds": output_conductance,
        "cgg": gate_capacitance,
        "vth": vth[sl],
        "vdsat": vdsat[sl],
        "gmid": gmid,
        "ft": ft,
        "gmro": gmro,
        "idw": drain_current / spec["w_char"],
        "vov": vgs[sl] - vth[sl],
        "saturated": abs(spec["vds"]) >= vdsat[sl],
    }
    acm = entry.get("acm")
    if acm:
        phit = _gmid.thermal_voltage(spec["temperature"])
        # Two inversion coefficients, and the difference between them is the
        # whole point rather than a detail.
        #
        # i_f is read back out of the gm/Id measurement by inverting the ACM's
        # own closed form. It is what labels a bias point weak/moderate/strong
        # for a designer, and it is what the solver reports -- but it is a
        # restatement of gm/Id, not an independent quantity. Plotting gm/Id
        # against it would reproduce the ACM curve by construction and test
        # nothing.
        #
        # i_f_current comes from the drain current instead: Id/I_S, with I_S
        # pinned by the i_f = 3 anchor. Plotting the measured gm/Id against
        # THAT is a real test, because the two axes then come from two
        # independent measurements.
        with np.errstate(divide="ignore", invalid="ignore"):
            x = 2.0 / (acm["n"] * phit * gmid)
        out["i_f"] = np.where(x > 2.0, (x - 1.0) ** 2 - 1.0, np.nan)
        if acm.get("i_s_weak"):
            out["i_f_current"] = drain_current / acm["i_s_weak"]
    return out


def entry_for_length(table, length, tolerance=1e-12):
    """The sweep for one channel length.

    No interpolation across L, deliberately. The L grid is the designer's own
    choice of buildable lengths; silently returning a blend of 0.28um and
    0.35um would hand back a device nobody asked for and nobody can draw."""
    for entry in table["lengths"]:
        if abs(entry["l"] - length) <= max(tolerance, 1e-9 * abs(length)):
            return entry
    available = ", ".join(f"{e['l'] * 1e6:g}u" for e in table["lengths"])
    raise ValueError(f"L={length * 1e6:g}u is not in this table; it has {available}")


# --------------------------------------------------------------- lookup ---

def _ascending(gmid, values):
    """gm/Id falls as VGS rises, so the usable slice is descending in gm/Id.
    np.interp needs an ascending abscissa -- reverse both."""
    return gmid[::-1], values[::-1]


def lookup(table, length, gmid, quantity):
    """One chart reading: the value of `quantity` at a given gm/Id and L.

    Refuses to extrapolate. Off the end of the measured curve there is no
    answer, and returning the endpoint would quietly hand back a number the
    simulation never produced."""
    entry = entry_for_length(table, length)
    curves = derived(table, entry)
    if quantity not in curves:
        raise KeyError(f"unknown quantity {quantity!r}")
    x, y = _ascending(curves["gmid"], np.asarray(curves[quantity], dtype=float))
    if gmid < x[0] or gmid > x[-1]:
        raise ValueError(
            f"gm/Id = {gmid:.4g} S/A is outside this device's measured range "
            f"({x[0]:.4g} to {x[-1]:.4g} S/A at L={length * 1e6:g}u)"
        )
    return float(np.interp(gmid, x, y))


def gmid_for_ft(table, length, ft_target):
    """Invert the ft chart: the transconductance efficiency that buys a given
    transit frequency at this L. ft falls monotonically as gm/Id rises, which
    is what makes the inverse single-valued."""
    entry = entry_for_length(table, length)
    curves = derived(table, entry)
    gmid = np.asarray(curves["gmid"], dtype=float)
    ft = np.asarray(curves["ft"], dtype=float)
    order = np.argsort(ft)
    ft_sorted, gmid_sorted = ft[order], gmid[order]
    if ft_target < ft_sorted[0] or ft_target > ft_sorted[-1]:
        raise ValueError(
            f"ft = {ft_target / 1e9:.4g} GHz is outside this device's reach at "
            f"L={length * 1e6:g}u ({ft_sorted[0] / 1e9:.4g} to {ft_sorted[-1] / 1e9:.4g} GHz)"
        )
    return float(np.interp(ft_target, ft_sorted, gmid_sorted))


# ---------------------------------------------------------------- solve ---

def solve(table, length, *, gm=None, drain_current=None, gmid=None, ft=None,
          finger_width=None):
    """Size a transistor from any two of {gm, Id, gm/Id, ft} plus L.

    This is the flow of the gm/Id literature's worked example -- gain fixes
    gm, load and bandwidth fix Cgg and so ft, the ft chart turns that into
    gm/Id, gm/Id and gm give Id, and the current-density chart turns Id into
    W -- made usable from either end, because in practice the known pair is
    as often (Id, L) as it is (gm, ft).

    Over-specification raises rather than resolving silently: three givens
    are usually a mistake, and the one that gets ignored would be invisible."""
    given = {
        "gm": gm, "id": drain_current, "gmid": gmid, "ft": ft,
    }
    named = [key for key, value in given.items() if value is not None]
    if len(named) != 2:
        raise ValueError(
            f"give exactly two of gm, id, gmid, ft (got {len(named)}: {', '.join(named) or 'none'})"
        )

    # Step 1: the bias point, as a gm/Id value.
    if gmid is not None:
        operating_gmid = float(gmid)
    elif ft is not None:
        operating_gmid = gmid_for_ft(table, length, float(ft))
    else:
        operating_gmid = float(gm) / float(drain_current)

    # Step 2: the current.
    if drain_current is not None:
        current = float(drain_current)
    elif gm is not None:
        current = float(gm) / operating_gmid
    else:
        raise ValueError(
            "gm/Id and ft alone fix the bias point but not the size -- add gm or Id"
        )

    # Step 3: the width, from current density.
    density = lookup(table, length, operating_gmid, "idw")
    width = current / density

    spec = table["spec"]
    entry = entry_for_length(table, length)
    result = {
        "device": spec["device"],
        "l": float(length),
        "w": width,
        "id": current,
        "gm": current * operating_gmid,
        "gmid": operating_gmid,
        "idw": density,
        "ft": lookup(table, length, operating_gmid, "ft"),
        "gmro": lookup(table, length, operating_gmid, "gmro"),
        "vgs": lookup(table, length, operating_gmid, "vgs"),
        "vov": lookup(table, length, operating_gmid, "vov"),
        "vth": lookup(table, length, operating_gmid, "vth"),
        "cgg": lookup(table, length, operating_gmid, "cgg") * width / spec["w_char"],
        "vds": spec["vds"],
        "vsb": spec["vsb"],
        "corner": spec["corner"],
        "temperature": spec["temperature"],
    }
    acm = entry.get("acm")
    if acm:
        phit = _gmid.thermal_voltage(spec["temperature"])
        x = 2.0 / (acm["n"] * phit * operating_gmid)
        inversion = (x - 1.0) ** 2 - 1.0 if x > 2.0 else None
        result["i_f"] = inversion
        result["inversion"] = inversion_label(inversion)
        result["n"] = acm["n"]
        result["i_sq"] = acm["i_sq"]

    # Step 4: fingering -- and this is where the table stops being an
    # approximation. Measured on gf180mcu's nfet_03v3 (2026-10-28):
    # gm(w=16u, nf=4) equals 4 * gm(w=4u, nf=1) to seven significant digits,
    # so putting fingers in parallel scales the device exactly, which is the
    # parallel-unit-device argument the methodology rests on. Changing the
    # FINGER width is the operation that is not free: 0.03% across 20u->30u
    # inside one model bin, but up to 11.8% across a bin edge (the width bins
    # are [0.22, 0.5], [0.5, 1.2], [1.2, 10], [10, 50] um, and the model
    # selects on width per finger, not total width).
    #
    # So the honest answer is a device built from an integer number of fingers
    # of exactly the width the table was taken at. W is quantized to that, and
    # the quantization is carried through to Id and gm rather than left as a
    # silent discrepancy between what was asked for and what was sized.
    per_finger = float(finger_width) if finger_width else spec["w_char"]
    fingers = max(1, int(round(width / per_finger)))
    actual_width = fingers * per_finger
    result["nf"] = fingers
    result["w_per_finger"] = per_finger
    result["w_ideal"] = width
    result["w"] = actual_width
    # Re-derive what the quantized device actually delivers. gm/Id is a bias
    # property and does not move; Id and gm scale with the realized width.
    result["id"] = density * actual_width
    result["gm"] = result["id"] * operating_gmid
    result["cgg"] = lookup(table, length, operating_gmid, "cgg") * actual_width / spec["w_char"]
    result["w_quantization_pct"] = (actual_width - width) / width * 100.0

    warnings = []
    if per_finger < spec["w_min"]:
        warnings.append(
            f"finger width {per_finger * 1e6:.4g}um is below this device's minimum "
            f"{spec['w_min'] * 1e6:.4g}um"
        )
    if finger_width and abs(per_finger - spec["w_char"]) > 1e-12:
        warnings.append(
            f"the charts were taken at a {spec['w_char'] * 1e6:g}um finger; at "
            f"{per_finger * 1e6:g}um they are an extrapolation, and across a model "
            f"width bin they can be ~10% out"
        )
    if warnings:
        result["warnings"] = warnings
    return result
