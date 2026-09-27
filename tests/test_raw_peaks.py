""".raw reduction for the SOA check (raw_peaks, soa_check) and the
auxiliary-output redirect/purge around it (run_sim)."""
import struct
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from analog_designer.sim import raw_peaks, run_sim, soa_check

VARIABLES = ["time", "v(d)", "v(g)"]
ROWS = [[0.0, 0.0, 0.0], [1e-9, 2.0, -1.0], [2e-9, 1.0, 3.5]]


def _header(kind):
    lines = ["Title: t", "Plotname: Transient Analysis", "Flags: real",
             f"No. Variables: {len(VARIABLES)}", f"No. Points: {len(ROWS)}", "Variables:"]
    lines += [f"\t{i}\t{name.upper() if i else name}\tvoltage" for i, name in enumerate(VARIABLES)]
    return "\n".join(lines + [f"{kind}:"]) + "\n"


def _write(tmp, name, data):
    path = Path(tmp) / name
    path.write_bytes(data)
    return path


class ReadRawTest(unittest.TestCase):
    def test_ascii_and_binary_agree(self):
        ascii_body = "".join(
            f" {i}\t{row[0]:e}\n" + "".join(f"\t{v:e}\n" for v in row[1:]) + "\n" for i, row in enumerate(ROWS)
        )
        binary_body = b"".join(struct.pack("<3d", *row) for row in ROWS)
        with tempfile.TemporaryDirectory() as tmp:
            ascii_raw = raw_peaks.read_raw(_write(tmp, "a.raw", (_header("Values") + ascii_body).encode()))
            binary_raw = raw_peaks.read_raw(_write(tmp, "b.raw", _header("Binary").encode() + binary_body))
        for raw in (ascii_raw, binary_raw):
            self.assertEqual(raw["variables"], VARIABLES)  # lowercased
            self.assertEqual([list(c) for c in raw["columns"]], [list(c) for c in zip(*ROWS)])

    def test_binary_partial_last_point_dropped(self):
        body = b"".join(struct.pack("<3d", *row) for row in ROWS) + struct.pack("<d", 9.0)
        with tempfile.TemporaryDirectory() as tmp:
            raw = raw_peaks.read_raw(_write(tmp, "b.raw", _header("Binary").encode() + body))
        self.assertEqual(len(raw["columns"][0]), len(ROWS))


class SignedPeaksTest(unittest.TestCase):
    RAW = {"variables": VARIABLES, "columns": [list(c) for c in zip(*ROWS)]}

    def test_signed_peak_ground_and_missing(self):
        peaks = raw_peaks.signed_peaks(self.RAW, [("v(g)", "v(d)"), ("v(d)", None), ("v(x)", None)])
        self.assertEqual(peaks[("v(g)", "v(d)")], -3.0)  # 2e-9: 3.5-1.0=2.5; 1e-9: -1-2=-3
        self.assertEqual(peaks[("v(d)", None)], 2.0)
        self.assertNotIn(("v(x)", None), peaks)


class CheckSoaTest(unittest.TestCase):
    DEVICE = {"instance": "XM1", "family": "hv", "d": "d", "g": "g", "s": "GND", "b": "0"}
    LIMITS = {"hv": {"vgs_max": 3.3, "vds_max": 1.5}}

    def test_pairs_cover_every_param_and_skip_unlimited_families(self):
        pairs = soa_check.soa_pairs([self.DEVICE, {**self.DEVICE, "family": "lv"}], self.LIMITS)
        self.assertEqual(set(pairs), {("v(g)", None), ("v(d)", None), (None, None)})

    def test_violation_from_peaks(self):
        peaks = raw_peaks.signed_peaks(SignedPeaksTest.RAW, soa_check.soa_pairs([self.DEVICE], self.LIMITS))
        diagnostics = soa_check.check_soa([self.DEVICE], peaks, self.LIMITS)
        self.assertEqual([d["key"] for d in diagnostics], ["XM1|Vgs", "XM1|Vds"])

    def test_device_with_unsaved_net_skipped(self):
        diagnostics = soa_check.check_soa([self.DEVICE], {("v(g)", None): 9.0}, self.LIMITS)
        self.assertEqual(diagnostics, [])


class RedirectAuxOutputsTest(unittest.TestCase):
    RUN = "/w/sim/v/startup/c"

    def _redirect(self, control):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "tb.spice"
            path.write_text(control, encoding="utf-8")
            moved = run_sim._redirect_aux_outputs(path, self.RUN, "/tmp/s", "startup_0.data")
            return moved, path.read_text(encoding="utf-8")

    def test_raw_and_side_dumps_move_primary_stays(self):
        moved, text = self._redirect(
            f"save all\nwrdata {self.RUN}/startup_0.data V(vref)\n"
            f"wrdata {self.RUN}/startup_0_diag.data v(a) v(b)\nset filetype=ascii\nwrite {self.RUN}/startup_0.raw\n"
        )
        self.assertEqual(moved, {"write": ["startup_0.raw"], "wrdata": ["startup_0_diag.data"]})
        self.assertIn(f"wrdata {self.RUN}/startup_0.data V(vref)", text)
        self.assertIn("wrdata /tmp/s/startup_0_diag.data v(a) v(b)", text)
        self.assertIn("write /tmp/s/startup_0.raw", text)
        self.assertIn("set filetype=binary", text)

    def test_nothing_to_move_leaves_file_alone(self):
        control = f"set filetype=ascii\nwrdata {self.RUN}/startup_0.data V(vref)\nwrite /elsewhere/x.raw\n"
        moved, text = self._redirect(control)
        self.assertEqual(moved, {"write": [], "wrdata": []})
        self.assertEqual(text, control)


class PurgeAuxOutputsTest(unittest.TestCase):
    def test_only_raw_and_side_dumps_removed(self):
        with tempfile.TemporaryDirectory() as tmp:
            cond = Path(tmp) / "sim" / "v1" / "startup" / "c1"
            cond.mkdir(parents=True)
            keep = ["startup_0.data", "tb.spice", "ngspice.log", "startup_0__field.png"]
            drop = ["startup_0.raw", "startup_0_diag.data"]
            for name in keep + drop:
                (cond / name).write_text("x")
            with mock.patch.object(run_sim.workspace, "PROJECT_ROOT", Path(tmp)):
                removed, freed = run_sim.purge_aux_outputs(["v1", "missing"])
            self.assertEqual((removed, freed), (2, 2))
            self.assertEqual(sorted(p.name for p in cond.iterdir()), sorted(keep))


if __name__ == "__main__":
    unittest.main()
