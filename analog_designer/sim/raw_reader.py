"""Reader for ngspice's ASCII `.raw` format (`set filetype=ascii` before
`write`) -- NOT a plain whitespace table, confirmed live (2026-09-05,
container inspiring_ishizaka, both a `.op` and a `.tran` run): the header
declares one `<index> <name> <type>` line per saved variable, then each
data point is its OWN multi-line block -- the point's index and its FIRST
variable's value share one line, every subsequent variable's value sits
alone on its own line, blank line separates points. Real example (`.op`,
3 variables):

    Variables:
    \t0\tv(d)\tvoltage
    \t1\tv(g)\tvoltage
    \t2\ti(vd)\tcurrent
    Values:
     0\t1.000000000000000e+00
    \t5.000000000000000e-01
    \t-5.000000000000000e-04

A naive per-line float reader (e.g. this project's own
parser_common.read_data(), built for `wrdata`'s plain-table output) would
see 3 separate 1-column "rows" here instead of 1 row of 3 columns --
that's why this needs its own reader rather than reusing that one."""


def read_ascii_raw(path):
    """Returns {"variables": [name, ...], "rows": [[float, ...], ...]} --
    "variables" in declared order (index 0 is "time" for a .tran, or
    whatever the first `save`d signal was for a .op -- no special-casing
    needed by callers, they look up by name by name via .index()).
    Variable names come out exactly as ngspice wrote them (lowercase,
    e.g. "v(x1.vptat)"), matching soa_check.py's own net-name lookups."""
    with open(path, encoding="utf-8") as f:
        lines = f.read().splitlines()

    n_vars = None
    for line in lines:
        if line.startswith("No. Variables:"):
            n_vars = int(line.split(":", 1)[1].strip())
            break
    if n_vars is None:
        raise ValueError(f"{path}: no 'No. Variables:' header line found")

    var_start = next(i for i, line in enumerate(lines) if line.strip() == "Variables:") + 1
    variables = [lines[var_start + i].split()[1] for i in range(n_vars)]

    values_start = next(i for i, line in enumerate(lines) if line.strip() == "Values:") + 1
    flat_values = [
        float(line.split()[-1])
        for line in lines[values_start:]
        if line.strip()
    ]
    rows = [flat_values[i:i + n_vars] for i in range(0, len(flat_values), n_vars)]
    return {"variables": variables, "rows": rows}
