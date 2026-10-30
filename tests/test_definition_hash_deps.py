"""compute_definition_hash() must go stale when a schematic/symbol the
topology pulls in changes (wrapper subcircuits, block symbols), without the
variation identity (variation_name, parameters only) moving, and must NOT
depend on the per-run materialized sch/<block>.sch scratch copies."""
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from analog_designer.core import workspace
from analog_designer.sim import run_sim

_TOP = (
    "v {xschem version=3.4.7 file_version=1.2}\n"
    "C {sch/res_wrap/rw2.sym} 0 0 0 0 {name=XR1 w='w' l='l'}\n"
    "C {/home/moduhub/work/proj/sch/res_wrap/rw4.sym} 0 100 0 0 {name=XR2}\n"
    "C {sg13g2_pr/rhigh.sym} 0 200 0 0 {name=R9}\n"
    "C {sch/amp.sym} 0 300 0 0 {name=x2}\n"
)


class DefinitionHashDependencyTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)
        for rel, text in {
            "sch/top/top_default.sch": _TOP,
            "sch/res_wrap/rw2.sym": "sym rw2\n",
            "sch/res_wrap/rw2.sch": "C {sch/res_wrap/leaf.sym} 0 0 0 0 {name=L}\n",
            "sch/res_wrap/leaf.sym": "sym leaf\n",
            "sch/res_wrap/rw4.sym": "sym rw4\n",
            "sch/amp.sym": "sym amp\n",
            "sch/amp.sch": "materialized scratch A\n",
            "tb/tb.sch": "C {sch/top.sym} 0 0 0 0 {name=DUT}\n",
            "sch/top.sym": "sym top\n",
            "sch/top.sch": "materialized scratch top\n",
            "tb/parser.py": "x = 1\n",
        }.items():
            path = self.root / rel
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text, encoding="utf-8")
        patches = [
            mock.patch.object(workspace, "PROJECT_ROOT", self.root),
            mock.patch.object(workspace, "CONFIG", {"blocks": {"top": {}, "amp": {}}}),
        ]
        for p in patches:
            p.start()
            self.addCleanup(p.stop)
        self.block_cfg = {"schematic": "top/top_default.sch"}
        self.test_cfg = {"testbench": "tb/tb.sch", "parser": "tb/parser.py"}

    def _hash(self):
        return run_sim.compute_definition_hash(self.block_cfg, self.test_cfg)

    def _write(self, rel, text):
        (self.root / rel).write_text(text, encoding="utf-8")

    def test_dependencies_found_transitively_and_library_symbols_ignored(self):
        rels = [rel for rel, _ in run_sim.schematic_dependency_files(self.root / "sch/top/top_default.sch")]
        self.assertEqual(rels, [
            "sch/amp.sym",
            "sch/res_wrap/leaf.sym",
            "sch/res_wrap/rw2.sch",
            "sch/res_wrap/rw2.sym",
            "sch/res_wrap/rw4.sym",  # absolute container path resolved into the project
        ])

    def test_scratch_block_schematics_are_not_dependencies(self):
        rels = [rel for rel, _ in run_sim.schematic_dependency_files(
            self.root / "sch/top/top_default.sch", self.root / "tb/tb.sch")]
        self.assertNotIn("sch/amp.sch", rels)
        self.assertNotIn("sch/top.sch", rels)
        self.assertIn("sch/top.sym", rels)

    def test_hash_is_stable(self):
        self.assertEqual(self._hash(), self._hash())

    def test_wrapper_edit_goes_stale(self):
        before = self._hash()
        self._write("sch/res_wrap/rw2.sch", "C {sch/res_wrap/leaf.sym} 0 0 0 0 {name=L2}\n")
        self.assertNotEqual(before, self._hash())

    def test_transitive_symbol_edit_goes_stale(self):
        before = self._hash()
        self._write("sch/res_wrap/leaf.sym", "sym leaf v2\n")
        self.assertNotEqual(before, self._hash())

    def test_block_symbol_edit_goes_stale(self):
        before = self._hash()
        self._write("sch/top.sym", "sym top v2\n")
        self.assertNotEqual(before, self._hash())

    def test_scratch_rewrite_does_not_go_stale(self):
        before = self._hash()
        self._write("sch/amp.sch", "materialized scratch B (another variation)\n")
        self._write("sch/top.sch", "materialized scratch top, another run\n")
        self.assertEqual(before, self._hash())

    def test_variation_identity_ignores_schematics(self):
        params = {"w": "1u"}
        before = run_sim.variation_name("top", "default", params)
        self._write("sch/res_wrap/rw2.sym", "sym rw2 v2\n")
        self.assertEqual(before, run_sim.variation_name("top", "default", params))


if __name__ == "__main__":
    unittest.main()
