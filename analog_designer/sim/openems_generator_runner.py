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
import math
import os
import sys
import threading
import time

# openEMS polls for a file of this name in its own working directory (== sim_path, which Run()
# chdir()s into) and stops the timestepping cleanly when it appears -- confirmed against the real
# binary, which then prints 'openEMS::CheckAbortCond(): Found file "ABORT", aborting simulation...'.
# This is what makes a cancel cheap: the port probes are already on disk (see CheckpointWriter), so
# CalcPort() still works afterwards and the partial sweep is kept instead of thrown away.
ABORT_FILENAME = "ABORT"

# How often the partial Y11 checkpoint below is refreshed. 10 min against a run measured in HOURS is
# negligible overhead (one CalcPort over a few-thousand-sample probe file) and bounds what a hard
# kill -- docker kill, an OOM, a host reboot, none of which give openEMS a chance to see ABORT --
# can cost: at most this much progress, never the whole run.
CHECKPOINT_INTERVAL_S = int(os.environ.get("ANALOG_DESIGNER_OPENEMS_CHECKPOINT_S", "600"))

# Shortest run, as a multiple of the excitation pulse's own length, whose port DFT is worth
# believing. openEMS integrates the port probes over the run's whole length, so a run that stops
# soon after the excitation has a badly truncated transform -- and it fails QUIETLY: the sweep comes
# out smooth and plausible, just wrong.
#
# Measured on this project's own runs, all at the same excitation length (2.865e-10s):
#   N=3 W=5um  d220, stopped at -60dB: 4.56 excitations -> Ldiff 7.80nH, SRF 8.39GHz   (sound)
#   N=4 W=5um  d250, stopped at -60dB: 6.87 excitations -> Ldiff 13.09nH, SRF 5.34GHz  (sound)
#   N=3 W=15um d350, stopped at -40dB: 1.44 excitations -> Ldiff 0.32nH, no SRF        (nonsense:
#       25x too little inductance, a fifth of the DC resistance the winding's own geometry forces,
#       and Im(Z) saturating instead of rising with frequency)
#
# The energy criterion alone does not bound this, which is the trap: it is a RELATIVE measure, so
# the same -40dB lands at 2.84 excitations on one coil and 1.44 on another. A low-resistance
# winding reaches any given energy sooner in units of its own ringing. 4.0 sits below both sound
# runs and well above the failure.
MIN_RUN_EXCITATIONS = float(os.environ.get("ANALOG_DESIGNER_OPENEMS_MIN_EXCITATIONS", "4.0"))


def excitation_length_s(f_max_hz):
    """Length of openEMS's Gaussian excitation, in seconds, for SetGaussExcite(f_max/2, f_max/2) --
    which is what every generator in this project uses. Verified against three real run logs, all of
    which print exactly this value for f_max=20GHz (2.86479e-10s)."""
    return 9.0 / (math.pi * (f_max_hz / 2.0))


def port_run_length_s(sim_path):
    """How much simulated time the port probes actually cover: the last timestamp they recorded.
    Read from the probe files rather than from the engine, so it reflects what the DFT really saw."""
    last = 0.0
    for name in os.listdir(sim_path):
        if not name.startswith("port_"):
            continue
        with open(os.path.join(sim_path, name), encoding="utf-8", errors="replace") as fh:
            for line in fh:
                if line.startswith("%") or not line.strip():
                    continue
                try:
                    last = max(last, float(line.split()[0]))
                except (ValueError, IndexError):
                    continue
    return last


def _load_generator(path):
    spec = importlib.util.spec_from_file_location(os.path.splitext(os.path.basename(path))[0], path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_csxcad_xml(FDTD, xml_path):
    """Dumps the structure the generator just built as CSXCAD XML, for
    inspection in AppCSXCAD (`AppCSXCAD <file>.xml`) -- geometry, materials
    and the mesh grid, no excitation/ports. Written BEFORE FDTD.Run() so it is
    on disk (and openable) while the run is still going, and next to
    result.json rather than inside sim_path. (NOT because Run() would delete
    it: cleanup=True only clears sim_path BEFORE the run starts -- et/ht/the
    field-dump .h5/port_* all survive in <out>_sim afterwards.)
    Best-effort, same posture as the field-dump PNGs: a failure here is
    logged and skipped, never allowed to cost the real FDTD run."""
    try:
        FDTD.GetCSX().Write2XML(xml_path)
        print(f"CSXCAD geometry XML written to {xml_path}")
    except Exception as exc:  # noqa: BLE001 -- see docstring
        print(f"CSXCAD XML export failed (non-fatal, FDTD run unaffected): {exc}")


def write_vtr(path, x_um, y_um, z_um, point_arrays):
    """Writes one VTK RectilinearGrid (.vtr) file ParaView opens directly -- plain ASCII XML, built
    by hand because the container has no vtk/pyevtk/meshio (checked) and this format is small enough
    not to justify adding one.

    `point_arrays` maps a name to an array shaped either (nz, ny, nx) (scalar) or (nz, ny, nx, 3)
    (vector). VTK orders points with x varying fastest, then y, then z, and a vector's own components
    fastest of all -- which is exactly C-order ravel() of those shapes, so nothing is transposed here
    beyond what the caller already did.

    Coordinates are written in MICRON, matching every other plot/preview in this codebase, not the
    metres openEMS's own HDF5 mesh datasets use."""
    import numpy as np

    nx, ny, nz = len(x_um), len(y_um), len(z_um)
    extent = "0 %d 0 %d 0 %d" % (nx - 1, ny - 1, nz - 1)

    def fmt(values):
        return " ".join("%.6g" % v for v in np.asarray(values).ravel())

    lines = [
        '<?xml version="1.0"?>',
        '<VTKFile type="RectilinearGrid" version="0.1" byte_order="LittleEndian">',
        '  <RectilinearGrid WholeExtent="%s">' % extent,
        '    <Piece Extent="%s">' % extent,
        '      <Coordinates>',
        '        <DataArray type="Float32" Name="x_um" format="ascii">%s</DataArray>' % fmt(x_um),
        '        <DataArray type="Float32" Name="y_um" format="ascii">%s</DataArray>' % fmt(y_um),
        '        <DataArray type="Float32" Name="z_um" format="ascii">%s</DataArray>' % fmt(z_um),
        '      </Coordinates>',
        '      <PointData>',
    ]
    for name, array in point_arrays.items():
        array = np.asarray(array)
        ncomp = 3 if array.ndim == 4 else 1
        lines.append(
            '        <DataArray type="Float32" Name="%s" NumberOfComponents="%d" format="ascii">'
            '%s</DataArray>' % (name, ncomp, fmt(array)))
    lines += ['      </PointData>', '    </Piece>', '  </RectilinearGrid>', '</VTKFile>', '']
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))


def convert_field_dump_to_vtr(h5_path, vtr_path):
    """Re-publishes a frequency-domain field dump openEMS ALREADY wrote (see render_field_dump() for
    the confirmed HDF5 layout) as a ParaView-readable .vtr. Pure post-processing of an existing file:
    it adds NOTHING to the FDTD run, unlike asking openEMS for a second, VTK-file-type dump of the
    same box -- that would make the engine accumulate a second DFT over every timestep. Measured
    alternative, for the record: a probe run with openEMS's own VTK dumps cost 4-15% more wall time
    (noisy at that size), versus 0% here.

    Writes the first dumped frequency only (same convention as render_field_dump()): the magnitude as
    a scalar, plus the real and imaginary vector fields, so ParaView can show |J| directly and still
    reconstruct phase from the two vector arrays."""
    import h5py
    import numpy as np

    with h5py.File(h5_path, "r") as f:
        real = f["FieldData/FD/f0_real"][:]           # (3, nz, ny, nx)
        imag = f["FieldData/FD/f0_imag"][:]
        x_um = f["Mesh/x"][:] * 1e6                   # openEMS reports metres here, not micron
        y_um = f["Mesh/y"][:] * 1e6
        z_um = f["Mesh/z"][:] * 1e6

    # (3, nz, ny, nx) -> (nz, ny, nx, 3): component fastest, then x, then y, then z == VTK's order.
    real_v = np.transpose(real, (1, 2, 3, 0))
    imag_v = np.transpose(imag, (1, 2, 3, 0))
    magnitude = np.sqrt(np.sum(real_v ** 2 + imag_v ** 2, axis=-1))

    write_vtr(vtr_path, x_um, y_um, z_um,
              {"J_mag": magnitude, "J_real": real_v, "J_imag": imag_v})


def y11_result(port, sim_path, freqs):
    """Y11(f)/Q/SRF from whatever port time-domain data is on disk RIGHT NOW -- factored out of
    run_and_extract() so the mid-run checkpoint (see CheckpointWriter) derives its partial sweep
    through exactly the same math as the final one, rather than a second, drifting copy of it.

    The Y11 sign convention (un-conjugated port.if_tot/port.uf_tot) is the one documented at length
    in run_and_extract() -- see the evidence trail there before changing it."""
    import numpy as np

    port.CalcPort(sim_path, freqs, ref_impedance=50)
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


def cache_doc_from_result(geometry, result, **extra):
    """The on-disk Y11 sweep document (numpy arrays/complex Y11 aren't JSON-serializable directly;
    re/im as plain float lists is what run_one_openems()'s own cache/data-file consumers -- e.g.
    tb_yparam_em.py's extract() -- expect). `extra` carries the checkpoint/abort markers."""
    doc = {
        "geometry": geometry,
        "freqs_hz": [float(f) for f in result["freqs"]],
        "re": [float(v) for v in result["y11"].real],
        "im": [float(v) for v in result["y11"].imag],
        "q": [float(v) for v in result["q"]],
        "srf_ghz": result["srf_ghz"],
        "peak_q": result["peak_q"],
        "peak_q_freq_ghz": result["peak_q_freq_ghz"],
    }
    doc.update(extra)
    return doc


def write_json_atomic(path, doc):
    """Write-then-rename, so a reader (or a kill) never sees a half-written file. Matters most for
    the checkpoint, which is rewritten repeatedly while the very thing it protects against -- an
    abrupt kill -- can land in the middle of any one of those writes."""
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(doc, fh)
    os.replace(tmp, path)


class CheckpointWriter(threading.Thread):
    """Periodically derives a PARTIAL Y11 sweep from the port probe files openEMS is still writing,
    and saves it next to the final result.

    Why this is possible at all: openEMS writes port_it_1/port_ut_1 CONTINUOUSLY during the run, not
    at the end -- confirmed by watching a probe run's sim_path while it was still timestepping. So a
    run that dies without ever reaching its EndCriteria (host reboot, OOM, docker kill, the 48h
    timeout) still leaves a usable, if under-converged, sweep behind. The raw probe files would
    survive that anyway, but turning them into Y11 afterwards is a manual recovery exercise -- this
    makes it automatic.

    A checkpoint is NEVER a result: the DFT behind it runs over a truncated time window, so an early
    one is numerically meaningless (measured on a probe run: peak Q jumping between 0 and 300 across
    consecutive checkpoints while the structure was still ringing). It is marked converged=False and
    carries its own `samples` count so how much of the run it represents is visible. Treat a late
    checkpoint as a salvage of an interrupted run, never as a substitute for one that finished.

    The main thread is blocked inside FDTD.Run() the whole time this runs, so there is no contention
    over the port object; stop() is called and joined before the FINAL CalcPort, so the two never
    overlap. The probe files themselves ARE being appended to concurrently, which is why each pass
    reads a private snapshot rather than the live files -- see _snapshot_port_files()."""

    # NOT named _stop: threading.Thread already has an internal _stop() that join() calls, and
    # shadowing it with an Event made every join() raise "'Event' object is not callable".
    def __init__(self, port, sim_path, freqs, geometry, out_path, interval_s=CHECKPOINT_INTERVAL_S):
        super().__init__(daemon=True)
        self._port = port
        self._sim_path = sim_path
        self._freqs = freqs
        self._geometry = geometry
        self._out_path = out_path
        self._interval_s = interval_s
        self._stop_event = threading.Event()
        self.count = 0

    def stop(self):
        self._stop_event.set()

    def _snapshot_port_files(self, snap_dir):
        """Copies the port probe files aside before reading them: openEMS appends to them while this
        runs, so CalcPort() straight off the live files can hit a half-written final line. Returns
        the number of samples in the snapshot, the honest measure of how complete it is."""
        os.makedirs(snap_dir, exist_ok=True)
        samples = 0
        for name in os.listdir(self._sim_path):
            if not name.startswith("port_"):
                continue
            src, dst = os.path.join(self._sim_path, name), os.path.join(snap_dir, name)
            with open(src, "rb") as fh:
                data = fh.read()
            # Drop a trailing partial line rather than letting the parser trip over it.
            cut = data.rfind(b"\n")
            data = data[:cut + 1] if cut >= 0 else b""
            with open(dst, "wb") as fh:
                fh.write(data)
            samples = max(samples, data.count(b"\n"))
        return samples

    def run(self):
        started = time.time()
        snap_dir = self._sim_path.rstrip("/\\") + "_checkpoint"
        while not self._stop_event.wait(self._interval_s):
            try:
                samples = self._snapshot_port_files(snap_dir)
                if samples < 2:
                    print("[checkpoint] only %d port samples on disk yet -- skipping" % samples,
                          flush=True)
                    continue
                result = y11_result(self._port, snap_dir, self._freqs)
            except Exception as exc:  # noqa: BLE001
                # Expected early on (no probe samples yet) and possible at any time. Never fatal:
                # this is a safety net, not the result.
                print("[checkpoint] not available yet (%s: %s)" % (type(exc).__name__, exc),
                      flush=True)
                continue
            self.count += 1
            write_json_atomic(self._out_path, cache_doc_from_result(
                self._geometry, result, checkpoint=True, converged=False, samples=samples,
                elapsed_s=round(time.time() - started, 1)))
            print("[checkpoint] partial Y11 written to %s (%d samples, peak Q so far %.2f -- "
                  "NOT converged)" % (self._out_path, samples, result["peak_q"]), flush=True)


def run_and_extract(generator, geometry, stack, sim_path, f_max_hz, n_freq=201, xml_path=None,
                    checkpoint_path=None):
    """Generic FDTD run + Y11/Q/SRF extraction against a structure the
    PROJECT's own build_openems_structure() built -- see module docstring
    for why this is here and not in the project's generator module.
    dump_field=True is always requested (see build_openems_structure()'s
    own contract note) -- the field dump adds negligible overhead relative
    to a real multi-hour FDTD run, so there's no reason to make it opt-in
    per call."""
    import numpy as np

    FDTD, port = generator.build_openems_structure(geometry, stack, f_max_hz, dump_field=True)
    if xml_path:
        write_csxcad_xml(FDTD, xml_path)
    # numThreads=0 ("max", the Run() default) was observed pinning a single
    # core during the actual timestepping engine despite the operator-setup
    # phase correctly using all of them -- auto-detection inside a cgroup
    # --cpus-limited container is unreliable (a known class of OpenMP/hwloc
    # issue), so force it explicitly to match the container's own CPU limit
    # instead of trusting "max" to mean anything useful here.
    freqs = np.linspace(1e6, f_max_hz, n_freq)
    # Started BEFORE Run(), because Run() blocks for the entire simulation -- see CheckpointWriter.
    # cleanup=True wipes sim_path at the START of the run, so a checkpoint must only look at it after
    # that has happened; the first one is a full interval in, which is well past it.
    checkpoint = None
    if checkpoint_path:
        checkpoint = CheckpointWriter(port, sim_path, freqs, geometry, checkpoint_path)
        checkpoint.start()
    try:
        FDTD.Run(sim_path, cleanup=True, numThreads=int(os.environ.get("OMP_NUM_THREADS", 8)))
    finally:
        if checkpoint is not None:
            checkpoint.stop()
            checkpoint.join(timeout=60)
            # The snapshot copies have served their purpose; the real probe files stay in sim_path.
            snap_dir = sim_path.rstrip("/\\") + "_checkpoint"
            if os.path.isdir(snap_dir):
                for name in os.listdir(snap_dir):
                    try:
                        os.remove(os.path.join(snap_dir, name))
                    except OSError:
                        pass
                try:
                    os.rmdir(snap_dir)
                except OSError:
                    pass

    # openEMS stops early and cleanly when it finds this file (see ABORT_FILENAME) -- a cancel, not a
    # crash. Everything below still runs: the sweep is real, just taken before the energy reached
    # EndCriteria, so it gets reported and marked rather than silently passed off as converged.
    aborted = os.path.exists(os.path.join(sim_path, ABORT_FILENAME))
    if aborted:
        print("openEMS was stopped early by an ABORT file -- extracting the partial sweep",
              flush=True)
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
    result = y11_result(port, sim_path, freqs)
    result["aborted"] = aborted

    # See MIN_RUN_EXCITATIONS: a run that stopped too soon after its own excitation produces a
    # smooth, plausible and wrong sweep, so measure what the probes actually cover and let main()
    # refuse it rather than caching it.
    excitation_s = excitation_length_s(f_max_hz)
    covered_s = port_run_length_s(sim_path)
    result["excitations"] = (covered_s / excitation_s) if excitation_s > 0 else None
    result["run_length_s"] = covered_s
    if result["excitations"] is not None:
        print(f"run covered {covered_s:.4g}s = {result['excitations']:.2f} x the excitation "
              f"({excitation_s:.4g}s); minimum for a trustworthy DFT is "
              f"{MIN_RUN_EXCITATIONS:.2f}", flush=True)
    return result


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
    parser.add_argument("--vtr-dir", default=None,
                         help="Optional output DIRECTORY for one ParaView-readable '<label>.vtr' per "
                              "FIELD_DUMP_NAMES entry, converted from the SAME HDF5 dumps the PNGs come "
                              "from (see convert_field_dump_to_vtr) -- costs no FDTD time, only disk")
    parser.add_argument("--checkpoint-out", default=None,
                         help="Optional output path for a PARTIAL Y11 sweep rewritten every "
                              "CHECKPOINT_INTERVAL_S while the run is still going (see CheckpointWriter) "
                              "-- what survives a kill that never lets openEMS see an ABORT file")
    args = parser.parse_args()

    generator = _load_generator(args.generator)
    with open(args.geometry_json) as f:
        geometry = json.load(f)
    stack = generator.load_stack(args.corner)

    sim_path = os.path.splitext(args.out)[0] + "_sim"
    # Next to --out (the per-variation run dir), so the geometry sits beside
    # geometry.json/result.json/openems_run.log instead of inside sim_path.
    xml_path = os.path.join(os.path.dirname(os.path.abspath(args.out)), "csxcad_geometry.xml")
    result = run_and_extract(generator, geometry, stack, sim_path, args.f_max, xml_path=xml_path,
                             checkpoint_path=args.checkpoint_out)

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

    if args.vtr_dir:
        # The SAME HDF5 dumps the PNGs above come from, re-published for ParaView -- see
        # convert_field_dump_to_vtr() for why this is a conversion rather than a second openEMS dump.
        # Best-effort per entry, same posture as every other visualization here.
        dump_names = getattr(generator, "FIELD_DUMP_NAMES", {})
        os.makedirs(args.vtr_dir, exist_ok=True)
        for label, dump_name in dump_names.items():
            vtr_path = os.path.join(args.vtr_dir, f"{label}.vtr")
            try:
                h5_path = os.path.join(sim_path, dump_name + ".h5")
                if os.path.exists(h5_path):
                    convert_field_dump_to_vtr(h5_path, vtr_path)
                    print(f"ParaView field dump ({label}) written to {vtr_path}")
                else:
                    print(f"No field dump found at {h5_path} -- skipping {label} .vtr")
            except Exception as exc:  # noqa: BLE001 -- see comment above
                print(f"ParaView export failed for {label} (non-fatal, Y11 result unaffected): {exc}")

    excitations = result.get("excitations")
    if excitations is not None and excitations < MIN_RUN_EXCITATIONS and not result.get("aborted"):
        # Converged on energy, but over too short a window for the port DFT to mean anything. NOT
        # cached, for the same reason an aborted run is not: run_one_openems copies --cache-out
        # straight into the generator cache, where this would be served forever as a measurement.
        short_path = os.path.splitext(args.cache_out)[0] + ".undertime.json"
        write_json_atomic(short_path, cache_doc_from_result(
            geometry, result, undertime=True, converged=False,
            excitations=excitations, end_criteria=getattr(generator, "END_CRITERIA", None)))
        print(f"REJECTED: the run covered only {excitations:.2f} excitation lengths, below the "
              f"{MIN_RUN_EXCITATIONS:.2f} needed for the port DFT to be trustworthy. The sweep is "
              f"saved at {short_path} for inspection and was NOT cached as a result.")
        print(f"The stop criterion is energy-relative, so it does not bound run LENGTH -- lower "
              f"END_CRITERIA (e.g. 1e-6) for this structure and run it again.")
        return 3

    if result.get("aborted"):
        # A cancelled run. The sweep is real but under-converged, so it must NOT land in --cache-out:
        # run_one_openems() copies that file straight into sim/_generator_cache/, where it would be
        # served forever as if it were a finished characterization. Saved beside it instead, under a
        # name nothing reads automatically, and reported as a failure so the test is not marked
        # green. Everything else the run produced (field dumps, .vtr, the raw et/ht/port_* in
        # <out>_sim) is already on disk and is deliberately left there.
        partial_path = os.path.splitext(args.cache_out)[0] + ".aborted.json"
        write_json_atomic(partial_path, cache_doc_from_result(
            geometry, result, aborted=True, converged=False,
            end_criteria=getattr(generator, "END_CRITERIA", None)))
        print(f"Run was ABORTED -- partial Y11(f) saved to {partial_path} (NOT cached as a result)")
        print(f"SRF/peak-Q above are from that partial sweep and are not converged values.")
        return 2

    # The stop criterion is NOT part of the geometry, so it does not reach the cache KEY -- two runs
    # of the same setup stopped at different energies would otherwise be indistinguishable on disk.
    # Recorded in the document instead, so anything combining several runs can see what it is mixing.
    write_json_atomic(args.cache_out, cache_doc_from_result(
        geometry, result, end_criteria=getattr(generator, "END_CRITERIA", None)))
    print(f"Y11(f) cache JSON written to {args.cache_out}")

    fitted = generator.fit_electrical_params(geometry, stack, em_result=result)
    with open(args.out, "w") as f:
        json.dump(fitted, f, indent=4)
    print(f"Fitted electrical-parameters JSON written to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
