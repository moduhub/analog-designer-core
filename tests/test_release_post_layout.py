"""Unit tests for the post-layout half of analog_designer/release/export.py
-- publishing analog_designer.sim.run_extracted's own extracted_result.json
+ curves, and presenting them NEXT TO the schematic-level numbers (an extra
results column with the delta, a two-column plot row) instead of merged
into them. Pure local file I/O with synthetic fixtures: no docker, no
ngspice, no Magic.

What these pin down is mostly about not misleading a reader: that a
post-layout number never lands in a column that sweeps conditions, that a
failed or stale post-layout run says so, and that the parasitics the
extraction actually carried (capacitance only today) are stated on the
page rather than assumed to be full RC."""
import json
import os
import tempfile
import unittest
from pathlib import Path

from analog_designer.core import workspace
from analog_designer.release import export

_CONFIG = {
    "defaults": {"vdd": 1.8},
    "blocks": {
        "amp": {
            "topologies": {
                "default": {
                    "schematic": "amp/amp_default.sch",
                    "parameters": {"w": {"default": "1u"}},
                }
            }
        }
    },
    "tests": {
        "amp": {
            "psrr": {"outputs": [{"description": "PSRR at 1kHz", "unit": "dB", "minimum": 40}]},
            "openloop_ac": {"outputs": [{"description": "DC gain", "unit": "dB"}]},
        }
    },
}

_SCHEMATIC_METRICS = [
    {"test": "psrr", "metric": "PSRR at 1kHz", "typical": 50.0, "min": 45.0, "max": 55.0, "unit": "dB"},
    {"test": "openloop_ac", "metric": "DC gain", "typical": 80.0, "min": 78.0, "max": 82.0, "unit": "dB"},
]


class ReleasePostLayoutTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)
        (self.root / "config.json").write_text(json.dumps(_CONFIG), encoding="utf-8")
        workspace.open_folder(str(self.root), block="amp", topology="default", remember=False)
        self.variation = "amp-default-abc123"
        self.doc_dir = self.root / "release" / "doc" / "amp"
        self.doc_dir.mkdir(parents=True)

    def _write(self, rel, text="x", mtime=None):
        path = self.root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        if mtime is not None:
            os.utime(path, (mtime, mtime))
        return path

    def _write_post_layout(self, test="psrr", typical=47.5, status="success", labels=("",),
                           mtime=None, parasitics="capacitance only (no interconnect resistance)",
                           metric="PSRR at 1kHz", unit="dB", error=None):
        base = f"sim/{self.variation}/layout_verify/{test}"
        payload = {
            "status": status, "variation": self.variation, "block": "amp", "topology": "default",
            "test": test, "conditions": {"vdd": 1.8, "temp": 27}, "parasitics": parasitics,
            "extracted_netlist": "amp_extracted.spice",
            "extracted_netlist_created": "2026-10-20T10:00:00",
            "created": "2026-10-21T10:00:00",
            "metrics": [] if status != "success" else [
                {"name": metric, "typical": typical, "min": typical, "max": typical, "unit": unit},
            ],
        }
        if error:
            payload["error"] = error
        self._write(f"{base}/extracted_result.json", json.dumps(payload), mtime=mtime)
        for label in labels:
            suffix = f"__{label}" if label else ""
            self._write(f"{base}/{test}_extracted{suffix}.png", "PNG")

    def _render(self, post_layout_info, metrics=None, plots_by_test=None):
        return export._render_readme(
            "amp", "default", self.variation, {"w": "1u"},
            _SCHEMATIC_METRICS if metrics is None else metrics,
            [], None, plots_by_test or {}, "", "2026-10-22T12:00:00",
            schematic_rendered=False,
            layout_info={"status": "not drawn yet", "note": "", "source": None, "gds": None, "raster": None},
            lvs_info={"status": "not run", "stale": False, "result": None, "files": []},
            post_layout_info=post_layout_info,
        )

    # ---------- reading what run_extracted left ----------

    def test_nothing_run_publishes_nothing(self):
        info = export._export_post_layout("amp", self.variation, self.doc_dir)
        self.assertEqual(info["status"], "not run")
        self.assertEqual(info["metrics"], {})
        self.assertFalse((self.doc_dir / "plots_post_layout").exists())

    def test_metrics_and_curves_are_published(self):
        self._write_post_layout(labels=("",))
        info = export._export_post_layout("amp", self.variation, self.doc_dir)

        self.assertEqual(info["status"], "published")
        self.assertEqual(info["metrics"][("psrr", "PSRR at 1kHz")]["typical"], 47.5)
        self.assertEqual(info["plots"]["psrr"], [("plot", "psrr_extracted.png")])
        self.assertTrue((self.doc_dir / "plots_post_layout" / "psrr_extracted.png").exists())
        self.assertEqual(info["parasitics"], "capacitance only (no interconnect resistance)")

    def test_multi_label_curves_keep_their_labels(self):
        self._write_post_layout(test="openloop_ac", metric="DC gain", typical=78.0,
                                labels=("typical", "worst_case"))
        info = export._export_post_layout("amp", self.variation, self.doc_dir)
        self.assertEqual(
            info["plots"]["openloop_ac"],
            [("Typical", "openloop_ac_extracted__typical.png"),
             ("Worst Case", "openloop_ac_extracted__worst_case.png")],
        )

    def test_failed_run_is_reported_not_hidden(self):
        self._write_post_layout(status="error", error="ngspice aborted")
        info = export._export_post_layout("amp", self.variation, self.doc_dir)
        self.assertEqual(info["status"], "run failed")
        self.assertEqual(info["metrics"], {})
        note = "\n".join(export._post_layout_note("amp", info))
        self.assertIn("Post-layout run FAILED", note)
        self.assertIn("`psrr`", note)

    def test_a_failed_run_publishes_no_curve(self):
        """A failure leaves the previous run's PNG in place; publishing it
        next to the schematic curve would contradict the page's own
        "post-layout run FAILED" line."""
        self._write_post_layout(labels=("",))
        export._export_post_layout("amp", self.variation, self.doc_dir)
        self.assertTrue((self.doc_dir / "plots_post_layout" / "psrr_extracted.png").exists())

        self._write_post_layout(status="error", error="port lists differ", labels=("",))
        info = export._export_post_layout("amp", self.variation, self.doc_dir)

        self.assertEqual(info["plots"], {})
        self.assertFalse((self.doc_dir / "plots_post_layout").exists())

    def test_run_older_than_the_mag_is_stale(self):
        self._write(f"layout/amp/{self.variation}/amp.mag", mtime=9000)
        self._write_post_layout(mtime=1000)
        info = export._export_post_layout("amp", self.variation, self.doc_dir)
        self.assertTrue(info["stale"])
        self.assertIn("**Stale:**", "\n".join(export._post_layout_note("amp", info)))

    def test_a_fresh_run_against_an_old_netlist_is_stale(self):
        """The run can postdate the layout while the netlist it spliced
        does not -- extract 10-01, layout saved 10-04, simulate 10-22."""
        mag = self._write(f"layout/amp/{self.variation}/amp.mag")
        import datetime
        netlist_time = datetime.datetime.fromtimestamp(mag.stat().st_mtime - 3600).isoformat(timespec="seconds")
        self._write_post_layout()
        path = self.root / f"sim/{self.variation}/layout_verify/psrr/extracted_result.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["extracted_netlist_created"] = netlist_time
        path.write_text(json.dumps(payload), encoding="utf-8")

        info = export._export_post_layout("amp", self.variation, self.doc_dir)

        self.assertTrue(info["stale"])
        self.assertIn("extracted netlist it ran against", "\n".join(export._post_layout_note("amp", info)))

    def test_a_run_and_netlist_newer_than_the_layout_are_current(self):
        self._write(f"layout/amp/{self.variation}/amp.mag", mtime=1000)
        self._write_post_layout()
        info = export._export_post_layout("amp", self.variation, self.doc_dir)
        self.assertFalse(info["stale"])

    def test_an_old_result_file_without_the_netlist_field_still_works(self):
        self._write(f"layout/amp/{self.variation}/amp.mag", mtime=1000)
        self._write_post_layout()
        path = self.root / f"sim/{self.variation}/layout_verify/psrr/extracted_result.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        del payload["extracted_netlist_created"]
        path.write_text(json.dumps(payload), encoding="utf-8")
        info = export._export_post_layout("amp", self.variation, self.doc_dir)
        self.assertFalse(info["stale"])

    def test_curves_of_a_removed_test_are_pruned(self):
        self._write_post_layout()
        export._export_post_layout("amp", self.variation, self.doc_dir)
        stray = self.doc_dir / "plots_post_layout" / "old_test_extracted.png"
        stray.write_text("PNG", encoding="utf-8")

        export._export_post_layout("amp", self.variation, self.doc_dir)

        self.assertFalse(stray.exists())
        self.assertTrue((self.doc_dir / "plots_post_layout" / "psrr_extracted.png").exists())

    # ---------- how the page presents them ----------

    def test_results_table_puts_post_layout_next_to_typical_with_a_delta(self):
        self._write_post_layout(typical=47.5)
        info = export._export_post_layout("amp", self.variation, self.doc_dir)
        readme = self._render(info)

        self.assertIn("| Test | Metric | Typical | Post-layout (Δ) | Min | Max |", readme)
        self.assertIn("| psrr | PSRR at 1kHz | 50 | 47.5 (-5.0%) | 45 |", readme)
        # A metric with no post-layout run of its own keeps an empty cell,
        # never the schematic value repeated.
        self.assertIn("| openloop_ac | DC gain | 80 |  | 78 |", readme)
        # The swept columns must stay where they were.
        self.assertIn("Parasitics: capacitance only (no interconnect resistance)", readme)
        self.assertIn("typical condition only", readme)

    def test_an_all_failed_block_still_says_so_on_the_page(self):
        """Not the same as never simulated: the page has to carry the
        failure (and its reason), even with no column to explain."""
        self._write_post_layout(status="error", error="port lists differ: vss vs vss_uq3")
        info = export._export_post_layout("amp", self.variation, self.doc_dir)
        readme = self._render(info)

        self.assertIn("no test produced a usable number", readme)
        self.assertIn("vss_uq3", readme)
        self.assertNotIn("Post-layout (", readme)

    def test_no_post_layout_leaves_the_table_exactly_as_before(self):
        info = export._export_post_layout("amp", self.variation, self.doc_dir)
        readme = self._render(info)
        self.assertIn("| Test | Metric | Typical | Min | Max | Mean | Std | Unit | Spec | Pass |", readme)
        self.assertNotIn("Post-layout", readme)

    def test_plots_are_paired_side_by_side(self):
        self._write_post_layout(labels=("",))
        info = export._export_post_layout("amp", self.variation, self.doc_dir)
        readme = self._render(info, plots_by_test={"psrr": [("plot", "psrr.png")]})

        self.assertIn("| Schematic | Post-layout (extracted) |", readme)
        self.assertIn("| ![plot](plots/psrr.png) | ![plot post-layout](plots_post_layout/psrr_extracted.png) |",
                      readme)

    def test_a_schematic_only_plot_stays_a_single_image(self):
        info = export._export_post_layout("amp", self.variation, self.doc_dir)
        readme = self._render(info, plots_by_test={"psrr": [("plot", "psrr.png")]})
        self.assertIn("![plot](plots/psrr.png)", readme)
        self.assertNotIn("| Schematic | Post-layout (extracted) |", readme)

    def test_post_layout_only_metric_still_gets_a_row(self):
        """A test never run at schematic level (or whose results were
        purged) must not drop the measurement that WAS taken."""
        self._write_post_layout(test="psrr", typical=47.5)
        info = export._export_post_layout("amp", self.variation, self.doc_dir)
        readme = self._render(info, metrics=[_SCHEMATIC_METRICS[1]])
        self.assertIn("| psrr | PSRR at 1kHz |  | 47.5 |", readme)

    # ---------- the delta itself ----------

    def test_delta_percentage_edge_cases(self):
        self.assertEqual(export._delta_text(50.0, 47.5), "47.5 (-5.0%)")
        self.assertEqual(export._delta_text(40.0, 44.0), "44 (+10.0%)")
        # No schematic reference, a zero reference, or a Monte Carlo
        # metric with no single typical: value only, never a bogus %.
        self.assertEqual(export._delta_text(None, 44.0), "44")
        self.assertEqual(export._delta_text(0.0, 44.0), "44")
        self.assertEqual(export._delta_text(50.0, None), "")
        # A negative reference compares by magnitude, so a number that got
        # more negative still reads as a rise, not a fall.
        self.assertEqual(export._delta_text(-10.0, -11.0), "-11 (-10.0%)")


if __name__ == "__main__":
    unittest.main()
