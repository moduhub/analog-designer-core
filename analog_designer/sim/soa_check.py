"""Checks a transistor's Vgs/Vds/Vgb/Vdb/Vsb -- read directly from the
actual simulated node-voltage waveforms (analog_designer.sim.raw_reader),
using each device's real gate/drain/source/bulk net (analog_designer.sim.
spice_devices) -- against explicit, sourced technology limits, replacing
the earlier approach of trusting the PSP103 model's own internal SOA check
(SWSOA=1, OSDI(debug) ... has exceeded ... messages parsed by
log_diagnostics.py).

That internal check was abandoned for two reasons, both confirmed live
(2026-09-05/06):
1. It fires on every Newton-Raphson iteration, not just the accepted/
   converged one -- needed a different, non-obvious filtering strategy for
   .op (trust only the solve's final iterations) vs .tran (every accepted
   timestep is a real instant, so trust the peak instead), each with its
   own edge cases.
2. Its own VDB_MAX/VSB_MAX values are wrong for HV devices: identical to
   LV's (1.6V) in all 4 HV .model blocks in the shipped
   sg13g2_moshv_parm.lib, while IHP's own published process spec
   (ihp-open-pdk-docs, process_control_params) gives HV a 3.3V nominal
   supply rating and a 12V TARGET junction breakdown (BVNPWhv/BVPNWhv) --
   nowhere near 1.6V. A plain HV inverter at 3.3V would trip it on every
   gate even though that's normal, documented HV operation.

Reading real node voltages sidesteps #1 entirely (a .raw file only ever
contains values ngspice actually accepted, .op or .tran alike -- no search
noise to filter), and #2 by sourcing config.json's limits from IHP's own
docs instead of the model's stale defaults:

  vgs_max = vds_max = vgb_max: 1.65V (LV) / 3.3V (HV) -- IHP's own
  published nominal supply rating (LV/HV-NMOS/PMOS-Specs: "VGS <= 1.65V
  @125C" LV, "VGS <= 3.3V @27C" HV).

  vdb_max = vsb_max: an explicit ENGINEERING MARGIN (not an IHP number) --
  50% of the ~12V junction-breakdown TARGET IHP documents for both LV and
  HV alike (BVNPW/BVPNW/BVNPWhv/BVPNWhv). No worst-case/minimum junction
  breakdown is published (unlike Vgs or BVDSS, which do have a MIN), so
  there's nothing traceable to use directly -- this margin is a project
  decision, meant to be easy to find and revise here if it turns out too
  loose or too tight."""

# Node names that collapse to the simulator's global ground reference
# (never a distinct saved variable, even under "save all") -- confirmed
# live, 2026-09-06: a PSP103 device's bulk/source wired to a net literally
# named "GND" (this project's own convention for tying vss/substrate to
# ground) shows up nowhere in a real .raw file's variable list, unlike
# every other internal net (which gets an "x1."-style prefix instead).
_GROUND_NAMES = {"gnd", "0"}

_SOA_PARAMS = ("vgs", "vds", "vgb", "vdb", "vsb")


def _node_voltages(net, raw_data, var_index):
    if net.lower() in _GROUND_NAMES:
        return [0.0] * len(raw_data["rows"])
    idx = var_index.get(f"v({net.lower()})")
    if idx is None:
        return None
    return [row[idx] for row in raw_data["rows"]]


def check_soa(devices, raw_data, limits):
    """devices: resolved list from spice_devices (real net names, not local
    ones). raw_data: {"variables", "rows"} from raw_reader.read_ascii_raw().
    limits: {"lv": {"vgs_max":..., ...}, "hv": {...}} (config.json's own
    technology.mosfet_limits, passed in as-is -- this module has no
    fallback/default of its own, a missing family or param simply isn't
    checked). Returns diagnostics in the same {severity, category, message,
    count, key} shape analog_designer.sim.log_diagnostics.parse() already
    produces, so the existing runs.jsonl/GUI Problems-panel pipeline needs
    no changes to consume this."""
    var_index = {name.lower(): i for i, name in enumerate(raw_data["variables"])}
    diagnostics = []

    for dev in devices:
        family_limits = limits.get(dev["family"])
        if not family_limits:
            continue

        pin_voltages = {}
        skip = False
        for pin in ("d", "g", "s", "b"):
            voltages = _node_voltages(dev[pin], raw_data, var_index)
            if voltages is None:
                skip = True
                break
            pin_voltages[pin] = voltages
        if skip:
            continue

        peak = {p: 0.0 for p in _SOA_PARAMS}
        for i in range(len(raw_data["rows"])):
            vd, vg, vs, vb = (pin_voltages[p][i] for p in ("d", "g", "s", "b"))
            candidates = {
                "vgs": vg - vs, "vds": vd - vs,
                "vgb": vg - vb, "vdb": vd - vb, "vsb": vs - vb,
            }
            for param, value in candidates.items():
                if abs(value) > abs(peak[param]):
                    peak[param] = value

        for param, value in peak.items():
            limit = family_limits.get(f"{param}_max")
            if limit is None or abs(value) <= limit:
                continue
            pretty = param[0].upper() + param[1:]  # "vgs" -> "Vgs"
            diagnostics.append({
                "severity": "warning",
                "category": "soa_violation",
                "key": f"{dev['instance']}|{pretty}",
                "message": f"{dev['instance']}: {pretty}={value:g} exceeds {pretty}_max={limit:g}",
                "count": 1,
            })

    return diagnostics
