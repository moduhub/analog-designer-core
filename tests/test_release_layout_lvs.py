"""Unit tests for the layout/LVS half of analog_designer/release/export.py
-- the part that publishes a block's drawn layout (GDS + raster) and the
evidence that it passes LVS, by COPYING what pro's Layout tab left on disk
(see _export_layout/_export_lvs's own docstrings on why nothing is
re-rendered there). Pure local file I/O with synthetic fixtures: no docker,
no Magic, no netgen, and no pro import -- which is exactly the contract
under test, since this module ships in the public core distribution.

Covers the three things that can go quietly wrong: picking the wrong
layout folder, publishing a GDS/raster that predates the .mag it claims to
show, and a run_lvs.sh written next to netlists that were never copied."""
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
                    "description": "A synthetic block, just enough config for the doc page.",
                    "parameters": {"w": {"default": "1u"}},
                }
            }
        }
    },
}

_LVS_RESULT = {
    "stage": "lvs",
    "variation": "amp-default-abc123",
    "status": "success",
    "lvs_status": "mismatch",
    "net_counts": {"extracted": 34, "schematic": 34},
    "device_classes": {
        "extracted": [{"class": "nmos", "count": 4}],
        "schematic": [{"class": "nmos", "count": 3}],
    },
    "bad_nets": [{
        "kind": "short",
        "extracted": [{"name": "vout", "detail": "nmos/D|S = 2"}],
        "schematic": [{"name": "vout", "detail": "nmos/D = 1"}],
    }],
    "bad_elements": [{
        "kind": "extra device (layout only)",
        "extracted": [{"name": "amp/nmos:0", "detail": "D = 3"}],
        "schematic": [{"name": "(no matching instance)", "detail": ""}],
    }],
    "report": "netgen report text",
}


class ReleaseLayoutLvsTests(unittest.TestCase):
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

    def _layout_dir(self, target=None):
        return f"layout/amp/{target or self.variation}"

    # ---------- which folder gets published ----------

    def test_variation_folder_wins_over_topology_folder(self):
        self._write(f"{self._layout_dir()}/amp.mag")
        self._write(f"{self._layout_dir('default')}/amp.mag")
        self.assertEqual(
            export._layout_source_dir("amp", self.variation),
            self.root / "layout" / "amp" / self.variation,
        )

    def test_topology_folder_is_the_fallback(self):
        self._write(f"{self._layout_dir('default')}/amp.mag")
        self.assertEqual(
            export._layout_source_dir("amp", self.variation),
            self.root / "layout" / "amp" / "default",
        )

    def test_layout_spec_without_a_mag_is_not_a_layout(self):
        """gen_spec regenerates layout_spec.json unconditionally as spec
        INPUT -- it says nothing about whether Magic has drawn anything."""
        self._write(f"{self._layout_dir()}/layout_spec.json", "{}")
        self.assertIsNone(export._layout_source_dir("amp", self.variation))

    def test_nothing_drawn_publishes_no_layout_section(self):
        info = export._export_layout("amp", self.variation, self.doc_dir)
        self.assertEqual(info["status"], "not drawn yet")
        self.assertEqual(export._layout_doc_lines("amp", info), [])

    # ---------- publishing the layout ----------

    def test_gds_and_raster_are_published_and_linked(self):
        self._write(f"{self._layout_dir()}/amp.mag", mtime=1000)
        self._write(f"{self._layout_dir()}/layout.gds", "GDS", mtime=2000)
        self._write(f"{self._layout_dir()}/layout_snapshot.png", "PNG", mtime=2000)

        info = export._export_layout("amp", self.variation, self.doc_dir)

        self.assertEqual(info["status"], "published")
        self.assertEqual(info["note"], "")
        self.assertEqual((self.root / "release" / "layout" / "amp" / "amp.gds").read_text(), "GDS")
        self.assertEqual((self.doc_dir / "layout.png").read_text(), "PNG")
        section = "\n".join(export._layout_doc_lines("amp", info))
        self.assertIn("![layout](layout.png)", section)
        self.assertIn("(../../layout/amp/amp.gds)", section)
        self.assertIn(f"(../../../layout/amp/{self.variation}/)", section)

    def test_raster_older_than_the_mag_is_published_as_stale(self):
        self._write(f"{self._layout_dir()}/layout.gds", "GDS", mtime=1000)
        self._write(f"{self._layout_dir()}/layout_snapshot.png", "PNG", mtime=1000)
        self._write(f"{self._layout_dir()}/amp.mag", mtime=5000)

        info = export._export_layout("amp", self.variation, self.doc_dir)

        self.assertEqual(info["status"], "published (stale)")
        self.assertIn("GDS and raster", info["note"])
        self.assertIn("Refresh Snapshot", info["note"])
        # Still published -- an old picture of the layout beats none, as
        # long as the page says it is old.
        self.assertTrue((self.doc_dir / "layout.png").exists())

    def test_mag_without_a_render_says_what_to_run(self):
        self._write(f"{self._layout_dir()}/amp.mag")
        info = export._export_layout("amp", self.variation, self.doc_dir)
        self.assertEqual(info["status"], "no GDS/raster yet")
        self.assertIn("Refresh Snapshot", info["note"])
        self.assertFalse((self.root / "release" / "layout").exists())

    def test_half_a_render_reports_what_is_missing(self):
        self._write(f"{self._layout_dir()}/amp.mag", mtime=1000)
        self._write(f"{self._layout_dir()}/layout_snapshot.png", "PNG", mtime=2000)
        info = export._export_layout("amp", self.variation, self.doc_dir)
        self.assertEqual(info["status"], "published")
        self.assertIn("Not published: GDS", info["note"])

    # ---------- publishing the LVS kit ----------

    def _write_lvs_outputs(self, mtime=None, **overrides):
        verify = f"sim/{self.variation}/layout_verify"
        result = {**_LVS_RESULT, **overrides}
        self._write(f"{verify}/lvs_result.json", json.dumps(result), mtime=mtime)
        self._write(f"{verify}/lvs_report.txt", "report")
        self._write(f"{verify}/lvs_report.json", "[]")
        self._write(f"{verify}/amp_extracted_lvs.spice", ".subckt amp\n.ends\n")
        self._write(f"{verify}/amp_schematic.spice", ".subckt amp\n.ends\n")
        self._write(f"{verify}/_netgen_setup_local.tcl", "# setup")

    def test_lvs_kit_is_copied_with_a_runnable_script(self):
        self._write_lvs_outputs()
        info = export._export_lvs("amp", self.variation)

        self.assertEqual(info["status"], "mismatch")
        self.assertFalse(info["stale"])
        kit = self.root / "release" / "lvs" / "amp"
        self.assertEqual(
            sorted(p.name for p in kit.iterdir()),
            ["amp_extracted_lvs.spice", "amp_schematic.spice", "lvs_report.json",
             "lvs_report.txt", "netgen_setup.tcl", "run_lvs.sh"],
        )
        script = (kit / "run_lvs.sh").read_text(encoding="utf-8")
        # The script must name the files it actually sits next to, with the
        # cell name on both sides -- that argument shape is netgen's own.
        self.assertIn('netgen -batch lvs "amp_extracted_lvs.spice amp" "amp_schematic.spice amp" '
                      'netgen_setup.tcl lvs_report_rerun.txt', script)
        self.assertNotIn("lvs_report.txt lvs_report_rerun", script)

    def test_no_script_when_a_netlist_is_missing(self):
        """A run_lvs.sh that dies on a missing file is worse than none."""
        self._write_lvs_outputs()
        (self.root / f"sim/{self.variation}/layout_verify/amp_schematic.spice").unlink()
        info = export._export_lvs("amp", self.variation)
        self.assertNotIn("run_lvs.sh", info["files"])
        self.assertFalse((self.root / "release" / "lvs" / "amp" / "run_lvs.sh").exists())

    def test_lvs_older_than_the_mag_is_stale(self):
        self._write(f"{self._layout_dir()}/amp.mag", mtime=9000)
        self._write_lvs_outputs(mtime=1000)
        info = export._export_lvs("amp", self.variation)
        self.assertTrue(info["stale"])
        self.assertEqual(export._lvs_status_text(info), "mismatch (stale)")

    def test_never_run_lvs_prints_the_command_that_runs_it(self):
        info = export._export_lvs("amp", self.variation)
        self.assertEqual(info["status"], "not run")
        section = "\n".join(export._lvs_doc_lines("amp", self.variation, info))
        self.assertIn("--stage lvs", section)
        self.assertIn(self.variation, section)

    def test_skipped_and_error_keep_their_own_shape(self):
        for status, field, text in (("skipped", "reason", "no netgen setup .tcl"),
                                    ("error", "error", "netgen produced no report")):
            with self.subTest(status=status):
                self._write_lvs_outputs(status=status, lvs_status=None, **{field: text})
                info = export._export_lvs("amp", self.variation)
                self.assertEqual(info["status"], status)
                self.assertIn(text, "\n".join(export._lvs_doc_lines("amp", self.variation, info)))

    def test_mismatch_tables_name_the_nets_and_pins(self):
        self._write_lvs_outputs()
        info = export._export_lvs("amp", self.variation)
        section = "\n".join(export._lvs_doc_lines("amp", self.variation, info))
        self.assertIn("Net mismatches (1)", section)
        self.assertIn("Instance mismatches (1)", section)
        # The pin detail is the actionable half; "|" inside it has to be
        # escaped or it closes the markdown table cell early.
        self.assertIn(r"vout — nmos/D\|S = 2", section)
        self.assertIn("| nmos | 4 | 3 |", section)
        self.assertIn("./run_lvs.sh", section)

    # ---------- leftovers from an earlier export ----------

    def test_deleted_layout_takes_its_published_files_with_it(self):
        """A release that says "not drawn yet" must not still carry the
        GDS/raster of a layout that has since been deleted."""
        self._write(f"{self._layout_dir()}/amp.mag")
        self._write(f"{self._layout_dir()}/layout.gds", "GDS")
        self._write(f"{self._layout_dir()}/layout_snapshot.png", "PNG")
        export._export_layout("amp", self.variation, self.doc_dir)
        for path in (self.root / f"{self._layout_dir()}").iterdir():
            path.unlink()

        info = export._export_layout("amp", self.variation, self.doc_dir)

        self.assertEqual(info["status"], "not drawn yet")
        self.assertFalse((self.doc_dir / "layout.png").exists())
        self.assertFalse((self.root / "release" / "layout" / "amp").exists())

    def test_pruning_the_raster_never_removes_the_doc_directory(self):
        """release/doc/<block>/ holds the raster but also the README.md the
        caller is still about to write -- removing it as an "empty
        directory" would turn a deleted raster into a failed export."""
        export._export_layout("amp", self.variation, self.doc_dir)
        self.assertTrue(self.doc_dir.is_dir())

    def test_purged_lvs_scratch_takes_the_published_kit_with_it(self):
        self._write_lvs_outputs()
        export._export_lvs("amp", self.variation)
        for path in (self.root / f"sim/{self.variation}/layout_verify").iterdir():
            path.unlink()

        info = export._export_lvs("amp", self.variation)

        self.assertEqual(info["status"], "not run")
        self.assertFalse((self.root / "release" / "lvs" / "amp").exists())

    def test_a_kit_that_loses_a_netlist_loses_its_script_too(self):
        self._write_lvs_outputs()
        export._export_lvs("amp", self.variation)
        (self.root / f"sim/{self.variation}/layout_verify/amp_schematic.spice").unlink()

        export._export_lvs("amp", self.variation)

        kit = self.root / "release" / "lvs" / "amp"
        self.assertFalse((kit / "amp_schematic.spice").exists())
        self.assertFalse((kit / "run_lvs.sh").exists())
        self.assertTrue((kit / "lvs_report.txt").exists())

    def test_pruning_leaves_files_this_module_never_published(self):
        """A reviewer's own re-run output lives in that directory too."""
        self._write_lvs_outputs()
        export._export_lvs("amp", self.variation)
        rerun = self.root / "release" / "lvs" / "amp" / "lvs_report_rerun.txt"
        rerun.write_text("their own run", encoding="utf-8")
        for path in (self.root / f"sim/{self.variation}/layout_verify").iterdir():
            path.unlink()

        export._export_lvs("amp", self.variation)

        self.assertTrue(rerun.exists())

    # ---------- the doc page + the index ----------

    def test_doc_page_header_lines_feed_the_index(self):
        self._write(f"{self._layout_dir()}/amp.mag", mtime=1000)
        self._write(f"{self._layout_dir()}/layout.gds", "GDS", mtime=2000)
        self._write(f"{self._layout_dir()}/layout_snapshot.png", "PNG", mtime=2000)
        self._write_lvs_outputs()
        layout_info = export._export_layout("amp", self.variation, self.doc_dir)
        lvs_info = export._export_lvs("amp", self.variation)

        readme = export._render_readme(
            "amp", "default", self.variation, {"w": "1u"}, [], [], None, {}, "",
            "2026-10-22T12:00:00", schematic_rendered=False,
            layout_info=layout_info, lvs_info=lvs_info,
            post_layout_info=export._export_post_layout("amp", self.variation, self.doc_dir),
        )
        self.assertIn("**Layout:** published  ", readme)
        self.assertIn("**LVS:** mismatch", readme)
        self.assertIn("## Layout", readme)
        self.assertIn("## LVS", readme)
        (self.doc_dir / "README.md").write_text(readme, encoding="utf-8")

        export._rebuild_index()
        index = (self.root / "release" / "README.md").read_text(encoding="utf-8")
        self.assertIn("| Block | Variation | Exported | Layout | LVS |", index)
        self.assertIn(f"| [amp](doc/amp/README.md) | `{self.variation}` | 2026-10-22T12:00:00 "
                      "| published | mismatch |", index)
        self.assertIn("[release/layout/](layout/)", index)
        self.assertIn("[release/lvs/](lvs/)", index)

    def test_index_columns_are_dashes_for_a_page_from_before_this_feature(self):
        (self.doc_dir / "README.md").write_text(
            "# amp — default\n\n**Variation ID:** `amp-default-old`  \n"
            "**Exported:** 2026-09-02T15:24:42\n",
            encoding="utf-8",
        )
        export._rebuild_index()
        index = (self.root / "release" / "README.md").read_text(encoding="utf-8")
        self.assertIn("| `amp-default-old` | 2026-09-02T15:24:42 | -- | -- |", index)


if __name__ == "__main__":
    unittest.main()
