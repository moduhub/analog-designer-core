#!/usr/bin/env python3
"""Generic, project-agnostic openEMS FDTD runner -- executes INSIDE the
eda-env-designer container (CSXCAD/openEMS Python bindings only exist
there), invoked by run_one_openems() via docker_exec() after being copied
into the per-run scratch directory (already bind-mounted at
container_run_dir -- no separate mount/PYTHONPATH needed for this file
itself; it dynamically loads the PROJECT's own generator module the same
way, by file path).

This script owns everything that's genuinely "run a structure through
openEMS and extract Y11(f)/Q(f)/SRF" -- FDTD.Run(), port.CalcPort(), the
conjugate sign-convention fix openEMS's own DFT needs (exp(+jwt) vs. this
codebase's exp(-jwt) circuit-theory convention), and Q/SRF-from-Y11 math --
NONE of which is PDK/geometry-specific, so it is NOT part of the project's
own generator module. That module only ever needs to supply two functions,
called below by a fixed contract:

  geometry_from_params(params) -> dict
      Type-casts a project's own free geometric parameters out of a raw
      {name: "value"} dict.

  build_openems_structure(geometry, stack, f_max_hz, dump_field=False) -> (FDTD, port)
      Builds the CSXCAD/openEMS structure for one geometry. Does NOT run
      the simulation -- that's this script's job, below. dump_field=True
      (always passed by this script) additionally adds a frequency-domain
      current-density dump property named FIELD_DUMP_NAME (a module
      constant the generator exports) -- see render_field_dump() below for
      how its HDF5 output gets turned into a PNG, generically.

  fit_electrical_params(geometry, stack, em_result) -> dict
      Derives the electrical parameters a project's own schematic template
      substitutes in, from the real FDTD result this script produces
      (`em_result`, the same {'freqs','y11','q','srf_ghz','peak_q',
      'peak_q_freq_ghz'} shape this script itself builds below).

Also needs `load_stack(corner)` from the same module (PDK stack data
lookup) -- also project-specific, also unrelated to openEMS mechanics.

Usage:
  python3 openems_generator_runner.py \\
      --generator /path/to/project/generator.py \\
      --geometry-json /path/to/geometry.json \\
      --corner tt --f-max 20e9 \\
      --out /path/to/fitted.json --cache-out /path/to/result.json
"""
import argparse
import importlib.util
import json
import os


def _load_generator(path):
    spec = importlib.util.spec_from_file_location(os.path.splitext(os.path.basename(path))[0], path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run_and_extract(generator, geometry, stack, sim_path, f_max_hz, n_freq=201):
    """Generic FDTD run + Y11/Q/SRF extraction against a structure the
    PROJECT's own build_openems_structure() built -- see module docstring
    for why this is here and not in the project's generator module.
    dump_field=True is always requested (see build_openems_structure()'s
    own contract note) -- the field dump adds negligible overhead relative
    to a real multi-hour FDTD run, so there's no reason to make it opt-in
    per call."""
    import numpy as np

    FDTD, port = generator.build_openems_structure(geometry, stack, f_max_hz, dump_field=True)
    # numThreads=0 ("max", the Run() default) was observed pinning a single
    # core during the actual timestepping engine despite the operator-setup
    # phase correctly using all of them -- auto-detection inside a cgroup
    # --cpus-limited container is unreliable (a known class of OpenMP/hwloc
    # issue), so force it explicitly to match the container's own CPU limit
    # instead of trusting "max" to mean anything useful here.
    FDTD.Run(sim_path, cleanup=True, numThreads=int(os.environ.get("OMP_NUM_THREADS", 8)))

    freqs = np.linspace(1e6, f_max_hz, n_freq)
    port.CalcPort(sim_path, freqs, ref_impedance=50)
    # 2026-09-16: REMOVED a np.conj() that used to sit here -- reverting a
    # fix from an earlier session (originally justified as correcting
    # openEMS's DFT time-convention, exp(+jwt) vs. this codebase's exp(-jwt)
    # circuit-theory assumption) that turns out to have been wrong for the
    # in-line planar LumpedPort topology every current generator (spiral,
    # loop) actually uses. That original validation was against a VERTICAL
    # port design and, per this project's own later investigation, was
    # likely only ever checked against Re(Y11)/DC resistance, never Im(Y11)'s
    # sign -- see openems_inductor_status.md's "2026-09-15" and "2026-09-16"
    # entries for the full trail. Decisive evidence this conjugate was
    # backwards, from TWO independent, real FDTD runs on the SAME in-line
    # planar port topology:
    #   - 100x50um/2um loop (fully converged): WITH the conjugate, Im(Z11)
    #     is negative and falling with no self-resonance across the whole
    #     1MHz-20GHz sweep -- a pure-capacitor signature, physically absurd
    #     for a modest-size Metal5 loop. WITHOUT it, Im(Z11) rises smoothly
    #     and monotonically (0->29.6 ohm), Q rises smoothly (0->2.4), no
    #     anomalies -- textbook lossy-inductor-below-SRF behavior. Also
    #     independently confirmed correct via a REAL ngspice run of the
    #     actual schematic (not just Python): y11_full_pi_model()'s closed
    #     form matched real SPICE to 4e-9 relative error using the
    #     UN-conjugated convention.
    #   - 40x10um/1um loop (the ORIGINALLY-VALIDATED-for-Re(Y11)-only
    #     default geometry, this time fully converged to -42.15dB): WITH
    #     the conjugate, the low-frequency Z11 slope gives a NEGATIVE fitted
    #     series inductance (-29.9pH) -- not just "off", sign-impossible for
    #     a physical inductor, and it crashed downstream fitting code
    #     (pi_model_fit.fit_eddy_branch()'s bounds assume l>0).
    # Confirms the issue is a generic sign bug in this extraction, not a
    # geometry-size-dependent physical effect (both a large and the
    # original small/validated geometry show the identical pattern).
    y11 = port.if_tot / port.uf_tot

    re = np.real(y11)
    im = np.imag(y11)
    q = -im / re

    srf_hz = None
    for i in range(1, len(freqs)):
        if im[i - 1] < 0 <= im[i]:
            frac = -im[i - 1] / (im[i] - im[i - 1])
            srf_hz = freqs[i - 1] + frac * (freqs[i] - freqs[i - 1])
            break

    peak_idx = int(np.argmax(q))
    return {
        "freqs": freqs, "y11": y11, "q": q,
        "srf_ghz": (srf_hz / 1e9) if srf_hz else None,
        "peak_q": float(q[peak_idx]), "peak_q_freq_ghz": float(freqs[peak_idx] / 1e9),
    }


def render_field_dump(h5_path, png_path):
    """Renders a frequency-domain TOTAL CURRENT DENSITY dump (see the
    project generator's own add_field_dump()) into a 2D magnitude PNG --
    generic: works for any generator's dump, as long as it's a
    frequency-domain box named consistently with the generator's own
    FIELD_DUMP_NAME. HDF5 layout confirmed empirically against openEMS's
    real output (a minimal standalone probe run, not a full spiral --
    see inductor_spiral_generator.py's add_field_dump() docstring for the
    exact structure and its own caveat about not yet being confirmed on a
    real spiral run): group FieldData/FD holds dataset pairs f{N}_real/
    f{N}_imag (one pair per dumped frequency, shape (3, nz, ny, nx) --
    3 field/current components x the dump box's own z/y/x mesh-line
    counts), and Mesh/x, Mesh/y give that grid's coordinates IN METERS
    (confirmed -- NOT the bare micron numbers the generator itself passes
    to AddBox/AddLine, openEMS reports the real SI-scaled values here; the
    x1e6 conversion below undoes that back to micron for axis labels
    matching every other plot in this project).

    Only the FIRST dumped frequency (f0) is rendered, even if a future
    generator dumps more than one -- reasonable since this generator only
    ever dumps its own single excitation center frequency today. Every
    z-slice the dump box spans (nz, usually 1-2 for a box matching one
    metal layer's own thickness) is averaged into one 2D image -- a
    thicker box would blur through-thickness variation, but for a single
    thin metal layer that's not expected to matter.

    2026-09-16: crops the outermost ~15% of the dump box's own xy extent
    (on each side) before plotting/color-scaling -- every current generator
    (see add_field_dump()'s own box argument) sizes its dump box to match
    the FULL simulation domain, whose own outer edge sits right at the
    PML absorbing boundary. Confirmed on a real converged 40x10um/1um loop
    substrate dump: a bright ring appeared right at the plotted edge
    (~92% out from center), NOT at the loop's own footprint (~77% out for
    that geometry) -- 60,000x weaker than the metal layer's own peak
    current at the SAME color scale, and physically the wrong location to
    be real loop-induced current. This is a PML-adjacent near-field/
    imperfect-absorption artifact, not signal. Cropped by FRACTION of the
    domain's own actual coordinate range (not a fixed index count or um
    value) so this works for any geometry/mesh without needing to know the
    structure's own physical size -- the mesh is non-uniformly graded
    (fine near the structure, coarse near the boundary), so index-based
    cropping would not correspond to a consistent physical margin the way
    coordinate-based cropping does."""
    import h5py
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    with h5py.File(h5_path, "r") as f:
        real = f["FieldData/FD/f0_real"][:]
        imag = f["FieldData/FD/f0_imag"][:]
        x_um = f["Mesh/x"][:] * 1e6
        y_um = f["Mesh/y"][:] * 1e6

    magnitude = np.sqrt(real ** 2 + imag ** 2)  # per-component |J|, shape (3, nz, ny, nx)
    total = np.sqrt(np.sum(magnitude ** 2, axis=0))  # combine x/y/z components -> (nz, ny, nx)
    total_2d = total.mean(axis=0)  # average over whatever z-slices the dump box spans -> (ny, nx)

    crop_frac = 0.85
    x_keep = np.abs(x_um) <= crop_frac * np.abs(x_um).max()
    y_keep = np.abs(y_um) <= crop_frac * np.abs(y_um).max()
    x_um, y_um = x_um[x_keep], y_um[y_keep]
    total_2d = total_2d[np.ix_(y_keep, x_keep)]

    fig, ax = plt.subplots(figsize=(5, 5))
    im = ax.imshow(
        total_2d, origin="lower", extent=[x_um.min(), x_um.max(), y_um.min(), y_um.max()],
        aspect="equal", cmap="inferno",
    )
    fig.colorbar(im, ax=ax, label="|J| (A/m^2, arbitrary excitation amplitude)")
    ax.set_xlabel("x (um)")
    ax.set_ylabel("y (um)")
    ax.set_title("Current density concentration")
    fig.tight_layout()
    fig.savefig(png_path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--generator", required=True, help="Path (inside the container) to the project's own generator module")
    parser.add_argument("--geometry-json", required=True, help="Path to a JSON file with this generator's own free geometric parameters")
    parser.add_argument("--corner", default="tt", choices=["tt", "ss", "ff"])
    parser.add_argument("--f-max", type=float, default=20e9)
    parser.add_argument("--out", required=True, help="Output path for the fitted electrical-parameters JSON")
    parser.add_argument("--cache-out", required=True,
                         help="Output path for the raw Y11(f) sweep JSON (freqs_hz/re/im/q/srf_ghz/peak_q/peak_q_freq_ghz)")
    parser.add_argument("--field-png", default=None,
                         help="Optional output path for a current-density-concentration PNG rendered from the "
                              "generator's own field dump (see render_field_dump()) -- best-effort: a render "
                              "failure is logged and skipped, never lets a real FDTD result go unsaved over it")
    parser.add_argument("--field-png-dir", default=None,
                         help="Optional output DIRECTORY for one current-density PNG per entry in the generator's "
                              "own FIELD_DUMP_NAMES dict (e.g. metal5/metal4/via4/substrate -- see "
                              "inductor_spiral_generator.py's own FIELD_DUMP_NAMES docstring for why a project may "
                              "dump more than one layer). Written as '<dir>/<label>.png'. Generators without a "
                              "FIELD_DUMP_NAMES attribute are skipped entirely (nothing to render); each entry is "
                              "independently best-effort, same reasoning as --field-png")
    args = parser.parse_args()

    generator = _load_generator(args.generator)
    with open(args.geometry_json) as f:
        geometry = json.load(f)
    stack = generator.load_stack(args.corner)

    sim_path = os.path.splitext(args.out)[0] + "_sim"
    result = run_and_extract(generator, geometry, stack, sim_path, args.f_max)

    srf_str = f"{result['srf_ghz']:.3f} GHz" if result["srf_ghz"] is not None else "not found in swept range"
    print(f"SRF = {srf_str}, peak Q = {result['peak_q']:.2f} @ {result['peak_q_freq_ghz']:.3f} GHz")

    if args.field_png:
        # Best-effort, deliberately isolated from everything above/below:
        # this dump/render pipeline was only ever validated on a minimal
        # standalone probe structure (see render_field_dump()'s own
        # docstring), not this generator's real spiral -- if the HDF5
        # layout, dump box placement, or a missing container library turns
        # out to be wrong on a real run, that must NEVER cost the
        # multi-hour FDTD result itself (already computed and about to be
        # written below) -- it only costs the one optional visualization.
        try:
            h5_path = os.path.join(sim_path, generator.FIELD_DUMP_NAME + ".h5")
            if os.path.exists(h5_path):
                render_field_dump(h5_path, args.field_png)
                print(f"Field/current-density PNG written to {args.field_png}")
            else:
                print(f"No field dump found at {h5_path} -- skipping field PNG")
        except Exception as exc:  # noqa: BLE001 -- see comment above
            print(f"Field dump render failed (non-fatal, Y11 result unaffected): {exc}")

    if args.field_png_dir:
        # Same best-effort isolation as the single --field-png block above,
        # per entry: one generator (e.g. inductor_spiral_generator.py) may
        # dump metal5/metal4/via4/substrate all in the same run -- render
        # whichever of those actually produced an .h5 (a generator whose
        # FIELD_DUMP_NAMES only has 'metal5', like inductor_loop_generator.py,
        # or an older generator with no FIELD_DUMP_NAMES at all, simply
        # renders fewer/none here -- never an error on its own). This is
        # ADDITIONAL to --field-png above, not a replacement -- that fixed
        # single-path output (always the 'metal5' dump, by convention) is
        # left completely alone so existing callers/cache-copy logic keep
        # working unchanged.
        dump_names = getattr(generator, "FIELD_DUMP_NAMES", {})
        os.makedirs(args.field_png_dir, exist_ok=True)
        for label, dump_name in dump_names.items():
            png_path = os.path.join(args.field_png_dir, f"{label}.png")
            try:
                h5_path = os.path.join(sim_path, dump_name + ".h5")
                if os.path.exists(h5_path):
                    render_field_dump(h5_path, png_path)
                    print(f"Field/current-density PNG ({label}) written to {png_path}")
                else:
                    print(f"No field dump found at {h5_path} -- skipping {label} field PNG")
            except Exception as exc:  # noqa: BLE001 -- see comment above
                print(f"Field dump render failed for {label} (non-fatal, Y11 result unaffected): {exc}")

    # numpy arrays/complex Y11 aren't JSON-serializable directly -- re/im as
    # plain float lists is what run_one_openems()'s own cache/data-file
    # consumers (e.g. tb_yparam_spiral.py's extract()) expect.
    cache_doc = {
        "geometry": geometry,
        "freqs_hz": [float(f) for f in result["freqs"]],
        "re": [float(v) for v in result["y11"].real],
        "im": [float(v) for v in result["y11"].imag],
        "q": [float(v) for v in result["q"]],
        "srf_ghz": result["srf_ghz"],
        "peak_q": result["peak_q"],
        "peak_q_freq_ghz": result["peak_q_freq_ghz"],
    }
    with open(args.cache_out, "w") as f:
        json.dump(cache_doc, f)
    print(f"Y11(f) cache JSON written to {args.cache_out}")

    fitted = generator.fit_electrical_params(geometry, stack, em_result=result)
    with open(args.out, "w") as f:
        json.dump(fitted, f, indent=4)
    print(f"Fitted electrical-parameters JSON written to {args.out}")


if __name__ == "__main__":
    main()
