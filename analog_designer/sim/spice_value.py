#!/usr/bin/env python3
"""Pure SPICE-value string <-> float helpers -- parsing/formatting the
"100u"-style literals config.json/params/*.json parameters use, with no
dependency on workspace/docker/anything else. Split out of
analog_designer/sim/gen_variations.py (which still re-exports these names for
every existing importer) so analog_designer/sim/run_sim.py can use them too:
gen_variations.py imports FROM run_sim.py at its own module top level, so
run_sim.py importing back from gen_variations.py would deadlock on partial
module initialization (a real circular import, not just an ordering nuisance)
-- this module has zero dependents of its own, so both sides can import it
freely.
"""
import re

_SUFFIX_MULT = {
    "f": 1e-15, "p": 1e-12, "n": 1e-9, "u": 1e-6, "m": 1e-3,
    "k": 1e3, "meg": 1e6, "g": 1e9, "t": 1e12,
}
_VALUE_RE = re.compile(r"^([+-]?\d*\.?\d+(?:[eE][+-]?\d+)?)\s*([a-zA-Z]*)$")


def _match(text):
    m = _VALUE_RE.match(text.strip())
    if not m:
        raise ValueError(f"cannot parse SPICE value: {text!r}")
    return m.groups()


def parse_spice_value(text):
    number, suffix = _match(text)
    suffix = suffix.lower()
    if suffix == "":
        return float(number)
    if suffix not in _SUFFIX_MULT:
        raise ValueError(f"unknown SPICE suffix in {text!r}: {suffix!r}")
    return float(number) * _SUFFIX_MULT[suffix]


def format_spice_value(base_value, unit_suffix):
    suffix = unit_suffix.lower()
    mult = 1.0 if suffix == "" else _SUFFIX_MULT[suffix]
    return f"{base_value / mult:.4g}{unit_suffix}"
