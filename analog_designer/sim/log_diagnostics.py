"""Regex-based extraction of structured diagnostics (errors/warnings) from
raw ngspice/Xyce simulation log text -- one shared rule registry so
run_sim.py's runners don't need per-testbench log-parsing code, and any new
test automatically gets diagnostics for free.

Grouped by a normalized signature, not returned once per raw line: a single
recurring problem (a convergence warning) typically prints once per
Newton iteration/timestep. Numbers embedded in an otherwise-identical
message (e.g. "at time 1.2e-6") are collapsed to a placeholder before
grouping, so those still count as ONE recurring issue, not one per
distinct timestamp.

Xyce rules are best-effort, seeded from one real xyce.log
(sim/cmos_vref-default-004296/noise/corner-tt_temperature-25/xyce.log,
2026-09-05) rather than documentation -- Xyce's own error-line convention
was not observed in that log (it had only warnings) and may need
adjustment once a real Xyce error is seen.

NOTE: this module used to also detect SOA (Safe Operating Area) violations
by parsing PSP103's own OSDI(debug) ... has exceeded ... messages
(SWSOA=1). That was abandoned (2026-09-06) for analog_designer.sim.
soa_check's waveform-based check instead -- see that module's own
docstring for why (the OSDI message fires on every Newton-Raphson
iteration, not just the accepted one, and its own VDB_MAX/VSB_MAX values
are wrong for HV devices, contradicted by IHP's published process spec)."""
import re

_NUMBER_RE = re.compile(r"[-+]?\d+\.?\d*(?:[eE][-+]?\d+)?")


def _normalize(msg):
    return _NUMBER_RE.sub("#", msg).strip()


_NGSPICE_ERROR_RE = re.compile(r"^Error:\s*(.+)$", re.MULTILINE)
_NGSPICE_WARNING_RE = re.compile(r"^Warning:\s*(.+)$", re.MULTILINE)

# ngspice's own hard-failure messages for an analysis that never actually
# completed -- confirmed live (gf180mcu_mh_ip__nfrac_pll's tb_startup.sch,
# 2026-09-09): a .tran whose adaptive timestep collapses on a misbehaving
# nonlinear element (there, a moscap's own internal E-source) prints
# "doAnalyses: TRAN: Timestep too small; time = ..., timestep = ...:
# trouble with node ..." followed by "tran simulation(s) aborted" --
# NEITHER line starts with "Error:"/"Warning:", so _NGSPICE_ERROR_RE/
# _NGSPICE_WARNING_RE above miss both entirely. ngspice's own process exit
# code was still 0 and it had already written a (nearly empty -- 1-2 rows
# instead of the thousands a full .tran would produce) .data file, so
# run_one_ngspice()'s own returncode/file-existence check couldn't catch
# this either -- these two patterns exist specifically so severity="error"
# here (see parse()'s ngspice branch) can force run_one_ngspice() to treat
# it as the real failure it is, instead of silently downstream becoming an
# empty-looking plot with no diagnostic anywhere. "<analysis> simulation(s)
# aborted" covers every analysis type (op/dc/ac/tran/...), not just tran.
_NGSPICE_ABORTED_RE = re.compile(r"^\s*(\S+ simulation\(s\) aborted)\s*$", re.MULTILINE)
_NGSPICE_TIMESTEP_RE = re.compile(r"^(doAnalyses:.*(?:[Tt]imestep too small|not converge).*)$", re.MULTILINE)

# "Netlist warning: Voltage Node (X1:NET4) does not have a DC path to
# ground" -- single-line form, seen live.
_XYCE_NETLIST_WARNING_RE = re.compile(r"^Netlist warning:\s*(.+)$", re.MULTILINE)
# "Netlist warning in file\n <path>\n at or near line N\n <message, itself
# sometimes WORD-WRAPPED across another line/N>" -- multi-line block form,
# also seen live (PDK model parameters PSP103 doesn't recognize). Every
# continuation line in this block starts with exactly one leading space, so
# _collapse_xyce_wraps() below joins them onto one line before this runs;
# file/line context is dropped for now (grouping by message text alone is
# enough to dedupe/count these -- refine if a use case needs the file:line
# back).
_XYCE_IGNORED_PARAM_RE = re.compile(r"No model parameter .+? parameter ignored\.")
# Not empirically observed (the one real xyce.log checked had no errors) --
# best-effort guess at Xyce's own convention, matching ngspice's. Revisit
# once a real Xyce error log is available.
_XYCE_ERROR_RE = re.compile(r"^Error:\s*(.+)$", re.MULTILINE)


def _collapse_xyce_wraps(log_text):
    """Xyce word-wraps a long warning message across multiple lines, each
    continuation indented by exactly one leading space (confirmed live,
    sim/cmos_vref-default-004296/noise/corner-tt_temperature-25/xyce.log) --
    joins those back into one line so _XYCE_IGNORED_PARAM_RE (and any
    future multi-line Xyce block) can match with a plain non-greedy regex
    instead of needing to know how many lines a given message wraps across.
    Used only for parsing here, never for the log_text written to disk."""
    return re.sub(r"\n ", " ", log_text)


def parse(log_text, simulator):
    """log_text: full stdout+stderr of one simulation run (the same text
    already written to ngspice.log/xyce.log). simulator: "ngspice" or
    "xyce". Returns a list of {severity, category, message, count, key},
    one per unique (severity, category, key) seen in THIS run -- "key" is a
    plain string (never a tuple) so the result round-trips through JSON
    (runs.jsonl) and can be merged across runs later (see
    analog_designer.results.data.diagnostics_for_variation)."""
    found = {}

    def _add(severity, category, key, message, value=None):
        entry = found.setdefault(
            (severity, category, key),
            {"severity": severity, "category": category, "key": key, "message": message, "count": 0},
        )
        entry["count"] += 1
        if value is not None:
            entry["value"] = value
            entry["message"] = message

    if simulator == "ngspice":
        for m in _NGSPICE_ERROR_RE.finditer(log_text):
            msg = m.group(1).strip()
            _add("error", "ngspice_error", _normalize(msg), msg)
        for m in _NGSPICE_ABORTED_RE.finditer(log_text):
            msg = m.group(1).strip()
            _add("error", "ngspice_error", _normalize(msg), msg)
        for m in _NGSPICE_TIMESTEP_RE.finditer(log_text):
            msg = m.group(1).strip()
            _add("error", "ngspice_error", _normalize(msg), msg)
        for m in _NGSPICE_WARNING_RE.finditer(log_text):
            msg = m.group(1).strip()
            _add("warning", "ngspice_warning", _normalize(msg), msg)
    elif simulator == "xyce":
        collapsed = _collapse_xyce_wraps(log_text)
        for m in _XYCE_NETLIST_WARNING_RE.finditer(collapsed):
            msg = m.group(1).strip()
            _add("warning", "xyce_warning", _normalize(msg), msg)
        for m in _XYCE_IGNORED_PARAM_RE.finditer(collapsed):
            msg = m.group(0).strip()
            _add("warning", "xyce_warning", _normalize(msg), msg)
        for m in _XYCE_ERROR_RE.finditer(collapsed):
            msg = m.group(1).strip()
            _add("error", "xyce_error", _normalize(msg), msg)

    return sorted(found.values(), key=lambda d: (d["severity"] != "error", -d["count"]))
