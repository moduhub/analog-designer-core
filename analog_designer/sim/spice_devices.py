"""Per-instance terminal/net extraction from an xschem-expanded netlist --
`analog_designer.sim.soa_check` needs to know which NET is attached to each
transistor's gate/drain/source/bulk to compute Vgs/Vds/Vgb/Vdb/Vsb straight
from saved node-voltage waveforms (see raw_reader.py), rather than trusting
the PSP103 model's own internal SOA check (confirmed live, 2026-09-06: that
check's VDB_MAX/VSB_MAX values are wrong for HV -- see soa_check.py's own
docstring).

Pin order is fixed at `d g s b` for every device family here, confirmed
against the PDK's own device wrapper (docker exec, sg13g2_mos{hv,lv}_mod.lib:
".subckt sg13_hv_nmos d g s b" / ".subckt sg13_lv_nmos d g s b", PMOS same
order) -- xschem's own expansion (confirmed live against a real generated
netlist, ihp_mh_ip__cmos_vref/sim/.../tb_vref_startup.spice) already emits
`XM1 vptat vref vss SUB sg13_hv_nmos w=... l=...` with exactly these 4 nets
positional right after the instance name, before the model name -- no
simulation-time-only expansion involved for this part.

Same `X<name> <pin> ... <model> <key>=<value> ...` line shape as
ihp_mh_ip__cmos_vref/tb/_shared/parser_common.py's own
parse_sized_devices() -- that one discards the pin/net tokens
(tokens[1:first_param-1]), keeping only model/W/L; this module keeps them.
Not reused directly since that function lives in the PROJECT's tb/_shared
(dynamically loaded alongside per-test parsers), while this runs from
run_sim.py itself -- tool-side code, same package as log_diagnostics.py,
project-agnostic beyond the PDK device-naming convention baked into
_MODEL_RE below."""
import re

# Registry of known PSP103 MOSFET models this project's PDK ships -- family
# ("lv"/"hv") and type ("nmos"/"pmos") are both derived from the name itself
# rather than hand-listing every (model, family, type) triple, so a new
# model name following the same "sg13_<family>_<type>" convention needs no
# code change here, only a broader/adjusted _MODEL_RE if the convention
# itself ever changes.
_MODEL_RE = re.compile(r"^sg13_(?P<family>lv|hv)_(?P<type>nmos|pmos)$", re.IGNORECASE)


def extract_mosfets(netlist_text):
    """Every X-instance line whose model matches _MODEL_RE, returned as
    [{"instance", "model", "family", "type", "d", "g", "s", "b"}, ...] --
    "d"/"g"/"s"/"b" are the LOCAL net names as written in this exact
    netlist text (still needing resolve_hierarchical_nets() below if this
    text is a subcircuit body, not the top-level netlist). A line with
    fewer than 4 pin tokens before its model name is skipped rather than
    raising -- same defensive posture as parse_sized_devices()'s own
    first_param<2 guard, for whatever non-MOSFET X-lines this netlist also
    contains (the DUT subcircuit call itself, ammeters, etc.)."""
    devices = []
    for line in netlist_text.splitlines():
        line = line.strip()
        if not line.startswith("X"):
            continue
        tokens = line.split()
        first_param = next((i for i, t in enumerate(tokens) if "=" in t), None)
        if first_param is None or first_param < 6:
            continue
        model = tokens[first_param - 1]
        m = _MODEL_RE.match(model)
        if not m:
            continue
        d, g, s, b = tokens[first_param - 5:first_param - 1]
        devices.append({
            "instance": tokens[0],
            "model": model,
            "family": m.group("family").lower(),
            "type": m.group("type").lower(),
            "d": d, "g": g, "s": s, "b": b,
        })
    return devices


def find_dut_subckt(netlist_text, instance_name="x1"):
    """The subcircuit name that the top-level `X<instance_name> ...` line
    instantiates -- e.g. `X1 vref net1 GND net2 GND cmos_vref` -> "cmos_vref"
    -- so callers (run_sim.py) don't need to hand-supply the block name per
    test/block; this project's testbenches always wire the block-under-test
    in as a single top-level `X1` instance. Returns None if no such line is
    found (e.g. a testbench with no DUT subcircuit at all)."""
    m = re.search(rf"^{re.escape(instance_name)}\s+.*\s+(\S+)\s*$", netlist_text, re.IGNORECASE | re.MULTILINE)
    return m.group(1) if m else None


def resolve_hierarchical_nets(netlist_text, devices, subckt_name, instance_name="x1"):
    """Rewrites each device's "d"/"g"/"s"/"b" from its LOCAL (subcircuit-
    internal) net name to the name ngspice will actually use once it
    flattens the circuit for simulation -- confirmed live (2026-09-05,
    container inspiring_ishizaka, a synthetic 1-level subcircuit): a net
    that IS one of the subcircuit's own ports keeps the CALLER's net name
    in a `save all` dump (a port named "na" wired to "na1" at the call site
    shows up as plain "v(na1)", never "v(x1.na)"), while a net that is NOT
    a port gets the calling instance's name prepended, lowercased, with a
    dot ("v(x1.internal_net)").

    `subckt_name`/`instance_name` identify which `.subckt <subckt_name>
    <port1> <port2> ...` block `devices` came from and which single
    `X<instance_name> <arg1> <arg2> ... <subckt_name>` line instantiates it
    -- this project's own cmos_vref testbenches always call it as `X1`
    (`instance_name` defaults accordingly). Only ONE level of hierarchy is
    resolved: a device inside a subcircuit that itself instantiates
    `subckt_name` (e.g. a future "top" test wiring cmos_vref in as its own
    sub-block) needs this called again, once per level, chaining the
    resulting prefix -- not needed for any test this project runs today, so
    not implemented until it is."""
    subckt_match = re.search(
        rf"^\.subckt\s+{re.escape(subckt_name)}\s+(.+)$", netlist_text, re.IGNORECASE | re.MULTILINE,
    )
    if subckt_match is None:
        return devices
    ports = [p.lower() for p in subckt_match.group(1).split()]

    call_match = re.search(
        rf"^{re.escape(instance_name)}\s+(.+?)\s+{re.escape(subckt_name)}(?:\s|$)",
        netlist_text, re.IGNORECASE | re.MULTILINE,
    )
    if call_match is None:
        return devices
    call_args = call_match.group(1).split()
    port_to_caller_net = dict(zip(ports, call_args))

    resolved = []
    for dev in devices:
        new_dev = dict(dev)
        for pin in ("d", "g", "s", "b"):
            local_net = dev[pin].lower()
            if local_net in port_to_caller_net:
                new_dev[pin] = port_to_caller_net[local_net]
            else:
                new_dev[pin] = f"{instance_name.lower()}.{local_net}"
        resolved.append(new_dev)
    return resolved
