#!/usr/bin/env python3
"""Vendor the standalone testbench runner (run_tb.py, next to this file)
into a project repo, so the project's testbenches can be run on any machine
that has xschem/ngspice/the PDK -- typically inside the EDA container --
without docker orchestration and without this tool installed:

    python -m analog_designer.standalone.export PROJECT_ROOT
    python -m analog_designer.standalone.export PROJECT_ROOT --check

The copy is byte-identical to run_tb.py except for one stamp line (which
core commit it came from), so --check can tell whether a vendored copy is
behind the core's current pipeline (exit status 1) -- e.g. from a project's
CI.
"""
import argparse
import subprocess
import sys
from pathlib import Path

SOURCE = Path(__file__).with_name("run_tb.py")
DEFAULT_DEST = "tools/run_tb.py"
STAMP_PREFIX = "# vendored from analog-designer-core"


def _core_commit():
    result = subprocess.run(
        ["git", "-C", str(SOURCE.parent), "rev-parse", "--short", "HEAD"], capture_output=True, text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def _strip_stamp(text):
    return "".join(line for line in text.splitlines(keepends=True) if not line.startswith(STAMP_PREFIX))


def render():
    """run_tb.py's source with the stamp line inserted after the shebang."""
    lines = SOURCE.read_text(encoding="utf-8").splitlines(keepends=True)
    stamp = f"{STAMP_PREFIX} {_core_commit()} -- regenerate with: python -m analog_designer.standalone.export <project>\n"
    return "".join(lines[:1] + [stamp] + lines[1:])


def is_current(dest):
    return dest.is_file() and _strip_stamp(dest.read_text(encoding="utf-8")) == SOURCE.read_text(encoding="utf-8")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("project_root")
    ap.add_argument("--dest", default=DEFAULT_DEST, help=f"path inside the project (default: {DEFAULT_DEST})")
    ap.add_argument("--check", action="store_true", help="only report whether the vendored copy is current")
    args = ap.parse_args(argv)

    root = Path(args.project_root).resolve()
    if not (root / "config.json").is_file():
        sys.exit(f"no config.json in {root} -- not a project folder")
    dest = root / args.dest
    if args.check:
        current = is_current(dest)
        print(f"{dest}: {'current' if current else 'missing or out of date'}")
        return 0 if current else 1
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(render(), encoding="utf-8")
    dest.chmod(0o755)
    print(f"wrote {dest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
