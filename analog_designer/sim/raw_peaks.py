"""Reads an ngspice `.raw` file (ASCII or binary, real data) and reduces it
to the one thing soa_check.py needs from it: for each requested pair of
node voltages (a, b), the signed value of v(a) - v(b) at the timepoint where
its magnitude peaks.

Deliberately standalone -- stdlib only, no analog_designer imports -- because
run_sim.run_one_ngspice() ships this file's own source into the simulation
container and runs it THERE, right next to the .raw ngspice just wrote to
container-local scratch space. Only the few-hundred-byte JSON result crosses
the (slow, on Docker Desktop) project bind mount; the .raw itself (tens of
MB for a `save all` transient) never does. The host imports the same
functions for tests and for any .raw kept on disk for debugging.

`python3 - <raw_path> <pairs_json>` (the in-container form) prints
`[[a, b, peak], ...]` as JSON. A pair name is a raw variable name exactly as
ngspice writes it (lowercase, e.g. "v(x1.vptat)"), or null for ground --
ground is never a saved variable, even under `save all`. A pair whose
variable isn't in the file is simply left out of the result."""
import json
import sys
from array import array


def read_raw(path):
    """{"variables": [name, ...], "columns": [array('d'), ...]} -- one
    column per variable, in declared order, names lowercased. Handles both
    `set filetype=ascii` and ngspice's default binary format. Only the first
    plot in the file is read (a testbench's `write` only ever writes the
    current one), and only real data -- a complex (.ac) plot raises."""
    with open(path, "rb") as f:
        blob = f.read()

    variables = []
    n_vars = None
    pos = 0
    while True:
        end = blob.index(b"\n", pos)
        line = blob[pos:end].decode("utf-8", "replace").rstrip("\r")
        pos = end + 1
        if line.startswith("No. Variables:"):
            n_vars = int(line.split(":", 1)[1])
        elif line.startswith("Flags:") and "complex" in line:
            raise ValueError(f"{path}: complex data not supported")
        elif line.strip() == "Variables:":
            for _ in range(n_vars):
                end = blob.index(b"\n", pos)
                variables.append(blob[pos:end].decode("utf-8", "replace").split()[1].lower())
                pos = end + 1
        elif line.strip() in ("Values:", "Binary:"):
            binary = line.strip() == "Binary:"
            break

    if binary:
        flat = array("d")
        # Truncated to whole points: an aborted run can leave a partial
        # final point behind.
        n_bytes = len(blob) - pos
        flat.frombytes(blob[pos:pos + n_bytes - n_bytes % (8 * n_vars)])
        if sys.byteorder != "little":
            flat.byteswap()
    else:
        # Each point is its own block: "<index>\t<first value>" then one
        # value per line for every remaining variable, blank line between.
        flat = array("d", (float(line.split()[-1]) for line in blob[pos:].decode("utf-8").splitlines() if line.strip()))
        flat = flat[:len(flat) - len(flat) % n_vars]

    columns = [flat[i::n_vars] for i in range(n_vars)]
    return {"variables": variables, "columns": columns}


def signed_peaks(raw_data, pairs):
    """{(a, b): signed peak of v(a) - v(b)} for every pair whose variables
    are present (None = ground). Ties keep the earliest timepoint."""
    index = {name: i for i, name in enumerate(raw_data["variables"])}
    n_points = len(raw_data["columns"][0]) if raw_data["columns"] else 0
    zeros = array("d", bytes(8 * n_points))

    def column(name):
        if name is None:
            return zeros
        i = index.get(name)
        return None if i is None else raw_data["columns"][i]

    peaks = {}
    for a, b in pairs:
        col_a, col_b = column(a), column(b)
        if col_a is None or col_b is None:
            continue
        peak = 0.0
        for va, vb in zip(col_a, col_b):
            diff = va - vb
            if abs(diff) > abs(peak):
                peak = diff
        peaks[(a, b)] = peak
    return peaks


if __name__ == "__main__":
    pairs = [tuple(pair) for pair in json.loads(sys.argv[2])]
    peaks = signed_peaks(read_raw(sys.argv[1]), pairs)
    print(json.dumps([[a, b, peak] for (a, b), peak in peaks.items()]))
