"""cap_cmomf -> ideal capacitor rewrite for Xyce (run_sim._lower_cmomf_for_xyce)."""
import tempfile
import unittest
from pathlib import Path

from analog_designer.sim.run_sim import _lower_cmomf_for_xyce


def _lower(text):
    path = Path(tempfile.mkdtemp()) / "n.spice"
    path.write_text(text, encoding="utf-8")
    _lower_cmomf_for_xyce(path)
    return path.read_text(encoding="utf-8")


class LowerCmomfTest(unittest.TestCase):
    def test_m1_to_m4_matches_ngspice_measured_density(self):
        # 10x10um, m=2 -> 200um^2 * 1.287 fF/um^2 (measured with ngspice's own model)
        out = _lower("XC1 a b cap_cmomf w=10u l=10u mmin=1 mmax=4 subblock=0 m=2 mm_ok=1\n")
        self.assertEqual(out.split()[:3], ["C_XC1", "a", "b"])
        self.assertAlmostEqual(float(out.split()[3]), 200 * 1.287e-15, delta=1e-17)

    def test_non_m1_bottom_plate_uses_lower_base(self):
        out = _lower("XC1 a b cap_cmomf w=10u l=10u mmin=2 mmax=3 subblock=0 m=1 mm_ok=1\n")
        self.assertAlmostEqual(float(out.split()[3]), 100 * 0.61e-15, delta=1e-17)

    def test_other_lines_untouched(self):
        line = "XM1 a b c d sg13_hv_nmos w=1u l=1u\n"
        self.assertEqual(_lower(line), line)


if __name__ == "__main__":
    unittest.main()


class StripCornerCapLibTest(unittest.TestCase):
    def test_cornercap_lib_dropped_once_no_cmom_instance_remains(self):
        out = _lower(
            ".lib /pdk/xyce/models/cornerCAP.lib cap_typ\n"
            ".lib /pdk/xyce/models/cornerMOShv.lib mos_tt\n"
            "XC1 a b cap_cmomf w=5u l=5u mmin=1 mmax=4 subblock=0 m=1 mm_ok=1\n"
        )
        self.assertNotIn("cornerCAP", out)
        self.assertIn("cornerMOShv", out)

    def test_netlist_without_cmomf_keeps_cornercap(self):
        text = ".lib /pdk/xyce/models/cornerCAP.lib cap_typ\nXM1 a b c d sg13_hv_nmos w=1u l=1u\n"
        self.assertEqual(_lower(text), text)
