"""run_sim._missing_project_subckts(): xschem intermittently emits a
testbench netlist whose DUT .subckt expansion is silently absent. A static
"netlist" test (area) would read zero devices and record 0.0 as a valid
result, so _netlist() re-netlists whenever a project block is instantiated
but never defined."""
import unittest
from unittest.mock import patch

from analog_designer.core import workspace
from analog_designer.sim import run_sim

_CONFIG = {"blocks": {"cmos_vref": {}, "output_amp": {}, "top": {}}}

_TB = """** sch_path: tb_vref_power.sch
Vavdd avdd GND dc 3.3
X1 vref net1 GND net2 GND cmos_vref
.lib cornerMOShv.lib mos_tt
"""

_DUT = """.subckt cmos_vref vref vdd vss pbias SUB
XM1 m1_mid vref vss SUB sg13_hv_nmos w=25.41u l=9.18u ng=3 m=1 mm_ok=1
XM2 vref vref m2_mid SUB sg13_hv_nmos w=7.08u
+ l=8.085u ng=1 m=1
.ends
"""


class MissingProjectSubcktTests(unittest.TestCase):
    def setUp(self):
        patcher = patch.object(workspace, "CONFIG", _CONFIG)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_dropped_expansion_is_detected(self):
        self.assertEqual(run_sim._missing_project_subckts(_TB), {"cmos_vref"})

    def test_complete_netlist_passes(self):
        self.assertEqual(run_sim._missing_project_subckts(_TB + _DUT), set())

    def test_pdk_devices_and_parameters_are_not_project_blocks(self):
        # sg13_hv_nmos comes from the PDK .lib; trailing name=value
        # parameters must not be mistaken for the model name.
        self.assertEqual(run_sim._missing_project_subckts(_DUT), set())

    def test_case_insensitive(self):
        self.assertEqual(run_sim._missing_project_subckts("x1 a b CMOS_VREF\n.SUBCKT cmos_vref a b\n"), set())


if __name__ == "__main__":
    unittest.main()
