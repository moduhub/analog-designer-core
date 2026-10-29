#!/usr/bin/env python3
"""The gm/Id design charts, as PNG.

Four of them are the working set of the methodology -- the transit-frequency
and intrinsic-gain charts you read L and gm/Id off, the current-density chart
you read W off, and gm/Id against VGS to see where the bias actually sits.
The fifth is the ACM check: every channel length normalized onto one curve,
which either collapses (and then n and I_SQ are real design parameters) or
does not (which is itself the finding).

Separate from gmid.py so that module stays import-light -- a GUI that only
solves a sizing problem has no reason to pull matplotlib in.
"""
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from analog_designer.devices import gmid  # noqa: E402

_FIGSIZE = (7.2, 5.0)
_DPI = 150


def _label(entry):
    return f"L = {entry['l'] * 1e6:g} um"


def _conditions(table):
    spec = table["spec"]
    return (f"{spec['device']}  |  corner {spec['corner']}  |  {spec['temperature']:g} C  |  "
            f"VDS = {spec['vds']:g} V  |  VSB = {spec['vsb']:g} V  |  "
            f"W = {spec['w_char'] * 1e6:g} um, nf = {spec['nf']}")


def _finish(fig, ax, table, title, xlabel, ylabel, legend=True):
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, which="both", alpha=0.3)
    if legend:
        ax.legend(fontsize=8, ncol=2)
    # The extraction conditions ride on every chart. A gm/Id chart read at the
    # wrong VDS is wrong by up to 5x in gm.ro, and the only defence is that
    # the reader can always see which VDS it was taken at.
    fig.text(0.5, 0.005, _conditions(table), ha="center", fontsize=7, color="0.35")
    fig.tight_layout(rect=(0, 0.03, 1, 1))


def _saturated(curves):
    """Points where the applied VDS still exceeds vdsat. Outside that the
    device is in the linear region and the charts do not mean what they say."""
    return np.asarray(curves["saturated"], dtype=bool)


def _design_window(curves, floor=0.01):
    """Saturated points, trimmed to where the device is still a usable device.

    Past the gm/Id plateau the transistor is in deep subthreshold: gm/Id stops
    rising while ft collapses towards zero, which on a log axis is ten decades
    of nothing that squashes the two or three decades anyone designs in. The
    cut is ft below `floor` of that length's own peak -- a property of the
    curve rather than a hand-picked gm/Id, so it travels to other devices and
    other PDKs."""
    mask = _saturated(curves)
    ft = np.asarray(curves["ft"], dtype=float)
    peak = ft[mask].max() if mask.any() else 0.0
    return mask & (ft >= floor * peak)


def write_charts(table, out_dir):
    """Write the five charts into `out_dir`. Returns the paths written."""
    from pathlib import Path

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    entries = table["lengths"]
    written = []

    # --- gm/Id and Vov against VGS -------------------------------------
    fig, (top, bottom) = plt.subplots(2, 1, figsize=(7.2, 6.4), sharex=True)
    for entry in entries:
        curves = gmid.derived(table, entry)
        mask = _saturated(curves)
        top.plot(curves["vgs"][mask], curves["gmid"][mask], label=_label(entry))
        bottom.plot(curves["vgs"][mask], curves["vov"][mask], label=_label(entry))
    top.set_ylabel("gm/Id  [S/A]")
    top.grid(True, alpha=0.3)
    top.legend(fontsize=8, ncol=2)
    top.set_title("Transconductance efficiency and overdrive against gate bias")
    bottom.set_ylabel("Vov = VGS - Vth  [V]")
    bottom.set_xlabel("VGS  [V]")
    bottom.grid(True, alpha=0.3)
    bottom.axhline(0.0, color="0.6", lw=0.8)
    fig.text(0.5, 0.005, _conditions(table), ha="center", fontsize=7, color="0.35")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    path = out_dir / "gmid_vgs.png"
    fig.savefig(path, dpi=_DPI)
    plt.close(fig)
    written.append(path)

    # --- ft against gm/Id ----------------------------------------------
    fig, ax = plt.subplots(figsize=_FIGSIZE)
    for entry in entries:
        curves = gmid.derived(table, entry)
        mask = _design_window(curves)
        ax.plot(curves["gmid"][mask], curves["ft"][mask] / 1e9, label=_label(entry))
    ax.set_yscale("log")
    _finish(fig, ax, table, "Transit frequency against transconductance efficiency",
            "gm/Id  [S/A]", "ft  [GHz]")
    path = out_dir / "ft.png"
    fig.savefig(path, dpi=_DPI)
    plt.close(fig)
    written.append(path)

    # --- intrinsic gain against gm/Id ----------------------------------
    fig, ax = plt.subplots(figsize=_FIGSIZE)
    for entry in entries:
        curves = gmid.derived(table, entry)
        mask = _design_window(curves)
        ax.plot(curves["gmid"][mask], curves["gmro"][mask], label=_label(entry))
    _finish(fig, ax, table, "Intrinsic gain against transconductance efficiency",
            "gm/Id  [S/A]", "gm.ro  [V/V]")
    path = out_dir / "gain.png"
    fig.savefig(path, dpi=_DPI)
    plt.close(fig)
    written.append(path)

    # --- current density against gm/Id ---------------------------------
    fig, ax = plt.subplots(figsize=_FIGSIZE)
    for entry in entries:
        curves = gmid.derived(table, entry)
        mask = _design_window(curves)
        # A/m -> uA/um is a factor of 1 exactly (1 A/m = 1 uA/um); the
        # multiplication is written out so the unit on the axis is checkable.
        ax.plot(curves["gmid"][mask], curves["idw"][mask] * 1e6 / 1e6, label=_label(entry))
    ax.set_yscale("log")
    _finish(fig, ax, table, "Current density against transconductance efficiency",
            "gm/Id  [S/A]", "Id/W  [uA/um]")
    path = out_dir / "idw.png"
    fig.savefig(path, dpi=_DPI)
    plt.close(fig)
    written.append(path)

    # --- the ACM collapse ----------------------------------------------
    spec = table["spec"]
    phit = gmid.thermal_voltage(spec["temperature"])
    fig, ax = plt.subplots(figsize=_FIGSIZE)
    any_acm = False
    for entry in entries:
        acm = entry.get("acm")
        curves = gmid.derived(table, entry)
        # The abscissa MUST be the current-derived inversion coefficient. The
        # other one (curves["i_f"]) is the ACM form solved backwards from the
        # very gm/Id on the ordinate, so plotting against it would draw the
        # model's own curve whatever the device did. Id/I_S is an independent
        # measurement, so a collapse onto the dashed line is evidence.
        if not acm or "i_f_current" not in curves:
            continue
        any_acm = True
        inversion = np.asarray(curves["i_f_current"], dtype=float)
        mask = _design_window(curves) & np.isfinite(inversion) & (inversion > 0)
        ax.plot(inversion[mask], curves["gmid"][mask] * acm["n"] * phit,
                label=f"{_label(entry)}  (n = {acm['n']:.3f})", lw=1.0)
    if any_acm:
        # The ACM's own closed form. Everything should lie on it if one slope
        # factor and one specific current describe the device.
        reference = np.logspace(-2, 4, 400)
        ax.plot(reference, 2.0 / (1.0 + np.sqrt(1.0 + reference)),
                "k--", lw=1.8, label="ACM: 2/(1+sqrt(1+i_f))")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.axvline(gmid.WEAK_LIMIT, color="0.7", lw=0.8)
        ax.axvline(gmid.STRONG_LIMIT, color="0.7", lw=0.8)
        _finish(fig, ax, table,
                "ACM check: measured efficiency against current-derived inversion coefficient",
                "inversion coefficient i_f = Id / I_S", "(gm/Id) . n . phi_t")
        path = out_dir / "acm.png"
        fig.savefig(path, dpi=_DPI)
        written.append(path)
    plt.close(fig)

    return written
