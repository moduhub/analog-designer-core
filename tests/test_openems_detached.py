"""Unit tests for the detached-openEMS plumbing -- pure Python, no docker/CSXCAD/openEMS needed.

Covers the three pieces that are easy to get silently wrong and expensive to discover on a run
measured in hours: the energy-line regex (its format differs between small and large readings), the
ETA bound (which must stay a BOUND rather than a prediction), and the hand-written VTK writer's
point ordering.
"""
import math
import tempfile
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

from analog_designer.sim.run_sim import (
    _fmt_hms, _OPENEMS_ENERGY_RE, _openems_eta_seconds,
)
from analog_designer.sim.openems_generator_runner import write_vtr


def _parse_energy(line):
    match = _OPENEMS_ENERGY_RE.search(line)
    if not match:
        return None
    return int(match.group(1)), float(match.group(3)) * (-1 if match.group(2) else 1)


class EnergyLineTests(unittest.TestCase):
    """Lines copied verbatim from a real run's openems_run.log."""

    def test_small_reading_has_the_sign_split_off_the_number(self):
        # "(- 0.00dB)" -- openEMS right-aligns the number, so the minus ends up detached. A regex
        # written only against "(-34.59dB)" silently never matches these.
        line = ("[@     4m15s] Timestep:         7273 || Speed:  116.8 MC/s (3.509e-02 s/TS) "
                "|| Energy: ~1.49e-12 (- 0.00dB)")
        self.assertEqual(_parse_energy(line), (7273, -0.0))

    def test_ordinary_reading(self):
        line = ("[@  5h24m08s] Timestep:       574567 || Speed:  122.4 MC/s (3.350e-02 s/TS) "
                "|| Energy: ~5.18e-08 (-34.59dB)")
        self.assertEqual(_parse_energy(line), (574567, -34.59))

    def test_non_energy_lines_do_not_match(self):
        self.assertIsNone(_parse_energy("Time for 1520057 iterations with 4098860.00 cells : 65420.23 sec"))
        self.assertIsNone(_parse_energy("Create FDTD engine (compressed SSE + multi-threading)"))


class EtaBoundTests(unittest.TestCase):
    def test_too_little_data_gives_no_estimate(self):
        self.assertIsNone(_openems_eta_seconds([(0, -1.0), (10, -2.0)], -60.0))

    def test_flat_energy_gives_no_estimate(self):
        samples = [(t, -5.0) for t in range(0, 600, 60)]
        self.assertIsNone(_openems_eta_seconds(samples, -60.0))

    def test_rising_energy_gives_no_estimate(self):
        samples = [(t, -20.0 + t / 100.0) for t in range(0, 600, 60)]
        self.assertIsNone(_openems_eta_seconds(samples, -60.0))

    def test_clean_exponential_decay_is_recovered(self):
        # A perfectly steady 0.1 dB/s decay. The last sample sits at t=580s, -58.0dB, so the
        # remaining 2.0dB to the -60dB target must come out as 20s.
        samples = [(t, -t / 10.0) for t in range(200, 600, 20)]
        self.assertAlmostEqual(samples[-1][1], -58.0)
        eta = _openems_eta_seconds(samples, -60.0)
        self.assertIsNotNone(eta)
        self.assertAlmostEqual(eta, 20.0, delta=0.5)

    def test_already_past_the_target_is_zero(self):
        samples = [(t, -t / 10.0) for t in range(200, 700, 20)]
        self.assertEqual(_openems_eta_seconds(samples, -60.0), 0.0)

    def test_pre_decay_plateau_is_ignored(self):
        # openEMS reports "- 0.00dB" for a long while before the energy starts falling; including
        # those points would flatten the fitted slope and inflate the estimate.
        plateau = [(t, -0.0) for t in range(0, 300, 20)]
        decay = [(t, -(t - 300) / 10.0) for t in range(300, 700, 20)]
        self.assertAlmostEqual(_openems_eta_seconds(plateau + decay, -60.0),
                               _openems_eta_seconds(decay, -60.0), delta=1.0)

    def test_it_is_a_lower_bound_on_a_decelerating_decay(self):
        """The real failure mode: a decay that slows down. Measured on a real 18h run, the estimate
        always came out SHORT (-27% to -83%). Anything that overshoots here would be presented to a
        user as a deadline it then blows through."""
        # dB(t) = -60 * (1 - exp(-t/3000)) -- approaches the target asymptotically, never early.
        samples = [(t, -60.0 * (1 - math.exp(-t / 3000.0))) for t in range(0, 3000, 100)]
        eta = _openems_eta_seconds(samples, -59.0)
        true_remaining = -3000.0 * math.log(1 - 59.0 / 60.0) - 2900
        self.assertIsNotNone(eta)
        self.assertLess(eta, true_remaining, "estimate must stay on the optimistic side")


class FmtHmsTests(unittest.TestCase):
    def test_sub_hour_shows_seconds(self):
        self.assertEqual(_fmt_hms(75), "1m15s")

    def test_hours_drop_seconds(self):
        self.assertEqual(_fmt_hms(3 * 3600 + 5 * 60 + 9), "3h05m")

    def test_negative_is_clamped(self):
        self.assertEqual(_fmt_hms(-10), "0m00s")


class WriteVtrTests(unittest.TestCase):
    """The ordering contract: VTK varies x fastest, then y, then z, with a vector's components
    fastest of all. Getting this wrong produces a file that opens fine in ParaView and shows a
    scrambled field, which is exactly the kind of error nobody catches by looking."""

    def _write_and_parse(self, arrays, nx=3, ny=4, nz=2):
        x = np.arange(nx, dtype=float)
        y = np.arange(ny, dtype=float) * 10
        z = np.arange(nz, dtype=float) * 100
        with tempfile.TemporaryDirectory() as tmp:
            path = str(Path(tmp) / "f.vtr")
            write_vtr(path, x, y, z, arrays)
            root = ET.parse(path).getroot()
        piece = root.find(".//Piece")
        out = {d.get("Name"): (np.fromstring(d.text, sep=" "), int(d.get("NumberOfComponents")))
               for d in piece.find("PointData").findall("DataArray")}
        coords = {d.get("Name"): np.fromstring(d.text, sep=" ")
                  for d in piece.find("Coordinates").findall("DataArray")}
        return piece, coords, out

    def test_scalar_ordering_is_x_fastest(self):
        nx, ny, nz = 3, 4, 2
        scalar = np.arange(nz * ny * nx, dtype=float).reshape(nz, ny, nx)
        piece, coords, arrays = self._write_and_parse({"s": scalar}, nx, ny, nz)
        self.assertEqual(piece.get("Extent"), "0 2 0 3 0 1")
        values, ncomp = arrays["s"]
        self.assertEqual(ncomp, 1)
        self.assertEqual(values.size, nx * ny * nz)
        for iz in range(nz):
            for iy in range(ny):
                for ix in range(nx):
                    self.assertEqual(values[ix + nx * (iy + ny * iz)], scalar[iz, iy, ix])

    def test_vector_components_are_fastest(self):
        nx, ny, nz = 3, 4, 2
        vector = np.arange(nz * ny * nx * 3, dtype=float).reshape(nz, ny, nx, 3)
        _, _, arrays = self._write_and_parse({"v": vector}, nx, ny, nz)
        values, ncomp = arrays["v"]
        self.assertEqual(ncomp, 3)
        self.assertEqual(values.size, 3 * nx * ny * nz)
        for iz in range(nz):
            for iy in range(ny):
                for ix in range(nx):
                    base = 3 * (ix + nx * (iy + ny * iz))
                    for c in range(3):
                        self.assertEqual(values[base + c], vector[iz, iy, ix, c])

    def test_coordinates_round_trip(self):
        _, coords, _ = self._write_and_parse({"s": np.zeros((2, 4, 3))})
        np.testing.assert_allclose(coords["x_um"], [0, 1, 2])
        np.testing.assert_allclose(coords["y_um"], [0, 10, 20, 30])
        np.testing.assert_allclose(coords["z_um"], [0, 100])


if __name__ == "__main__":
    unittest.main()
