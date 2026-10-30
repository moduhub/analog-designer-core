"""Unit tests for run_extracted's own extracted_result.json contract --
the file the release page's post-layout column and the GUI read back (see
analog_designer.release.export._export_post_layout). Pure file I/O: the
simulation itself is not exercised here.

The two things that matter to a reader downstream: the parasitics flavor
is always stated (falling back to PARASITICS_UNRECORDED for a netlist
extracted before magic_layout started recording it, never left blank or
guessed as full RC), and a failed run REPLACES a previous run's numbers
instead of leaving them standing."""
import json
import tempfile
import unittest
from pathlib import Path

from analog_designer.sim import run_extracted

_METRICS = [{"name": "PSRR @ 1kHz", "typical": 45.4, "min": 45.4, "max": 45.4, "unit": "dB"}]


class ExtractedResultFileTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)
        self.sim_dir = self.root / "sim" / "amp-default-abc123"
        self.verify_dir = self.sim_dir / "layout_verify" / "psrr"
        self.verify_dir.mkdir(parents=True)
        self.extracted = self.sim_dir / "layout_verify" / "amp_extracted.spice"
        self.extracted.write_text(".subckt amp vdd vss\n.ends\n", encoding="utf-8")

    def _write(self, status, metrics=None, error=None):
        run_extracted._write_extracted_result(
            self.verify_dir, self.sim_dir, "amp", "default", "amp-default-abc123", "psrr",
            self.extracted, {"vdd": 1.8}, status, metrics=metrics, error=error,
        )
        return json.loads((self.verify_dir / "extracted_result.json").read_text(encoding="utf-8"))

    def test_success_payload_names_the_netlist_it_ran_against(self):
        payload = self._write("success", metrics=_METRICS)
        self.assertEqual(payload["status"], "success")
        self.assertEqual(payload["metrics"], _METRICS)
        self.assertEqual(payload["extracted_netlist"], "amp_extracted.spice")
        self.assertIsNotNone(payload["extracted_netlist_created"])
        self.assertEqual(payload["block"], "amp")
        self.assertEqual(payload["conditions"], {"vdd": 1.8})

    def test_parasitics_falls_back_when_extraction_never_recorded_it(self):
        payload = self._write("success", metrics=_METRICS)
        self.assertEqual(payload["parasitics"], run_extracted.PARASITICS_UNRECORDED)
        self.assertIn("capacitance only", payload["parasitics"])

    def test_parasitics_comes_from_the_extraction_when_it_did(self):
        (self.sim_dir / "layout_verify" / "extract_result.json").write_text(
            json.dumps({"status": "success", "parasitics": "full RC"}), encoding="utf-8")
        payload = self._write("success", metrics=_METRICS)
        self.assertEqual(payload["parasitics"], "full RC")

    def test_unreadable_extraction_result_does_not_break_the_run(self):
        (self.sim_dir / "layout_verify" / "extract_result.json").write_text("{ not json", encoding="utf-8")
        payload = self._write("success", metrics=_METRICS)
        self.assertEqual(payload["parasitics"], run_extracted.PARASITICS_UNRECORDED)

    def test_a_failure_replaces_an_earlier_success(self):
        self._write("success", metrics=_METRICS)
        payload = self._write("error", error="port lists differ: ['vdd', 'vss'] vs ['vdd', 'vss_uq3']")
        self.assertEqual(payload["status"], "error")
        self.assertEqual(payload["metrics"], [])
        self.assertIn("vss_uq3", payload["error"])


if __name__ == "__main__":
    unittest.main()
