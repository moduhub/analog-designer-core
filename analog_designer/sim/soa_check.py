"""Checks a transistor's Vgs/Vds/Vgb/Vdb/Vsb -- read directly from the
actual simulated node-voltage waveforms (analog_designer.sim.raw_peaks),
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

# param -> (plus pin, minus pin)
_SOA_PARAMS = {"vgs": ("g", "s"), "vds": ("d", "s"), "vgb": ("g", "b"), "vdb": ("d", "b"), "vsb": ("s", "b")}


def _raw_name(net):
    """The .raw variable holding `net`'s voltage, or None for ground (see
    raw_peaks.signed_peaks())."""
    return None if net.lower() in _GROUND_NAMES else f"v({net.lower()})"


def _device_pairs(dev):
    return {param: (_raw_name(dev[plus]), _raw_name(dev[minus])) for param, (plus, minus) in _SOA_PARAMS.items()}


def soa_pairs(devices, limits):
    """Every (a, b) node-voltage pair check_soa() will need a peak for --
    what to ask raw_peaks.signed_peaks() for. Devices whose family has no
    limits are left out, same as check_soa() skips them."""
    pairs = set()
    for dev in devices:
        if limits.get(dev["family"]):
            pairs.update(_device_pairs(dev).values())
    return sorted(pairs, key=lambda pair: (pair[0] or "", pair[1] or ""))


def check_soa(devices, peaks, limits):
    """devices: resolved list from spice_devices (real net names, not local
    ones). peaks: {(a, b): signed peak of v(a) - v(b)} from
    raw_peaks.signed_peaks() over soa_pairs(devices, limits) -- a device with
    any pair missing (a net not saved in the .raw) is skipped entirely.
    limits: {"lv": {"vgs_max":..., ...}, "hv": {...}} (config.json's own
    technology.mosfet_limits, passed in as-is -- this module has no
    fallback/default of its own, a missing family or param simply isn't
    checked). Returns diagnostics in the same {severity, category, message,
    count, key} shape analog_designer.sim.log_diagnostics.parse() already
    produces, so the existing runs.jsonl/GUI Problems-panel pipeline needs
    no changes to consume this."""
    diagnostics = []

    for dev in devices:
        family_limits = limits.get(dev["family"])
        if not family_limits:
            continue
        pairs = _device_pairs(dev)
        if any(pair not in peaks for pair in pairs.values()):
            continue
        peak = {param: peaks[pair] for param, pair in pairs.items()}

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
