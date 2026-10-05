#!/usr/bin/env python3
"""Standalone testbench runner for an analog-designer project repository.

Runs a project's testbenches (config.json + sch/ + tb/) directly on the
machine it is executed on -- typically *inside* an EDA container that
already has xschem + ngspice (+ Xyce) + the PDK -- with no docker
orchestration and no dependency on the analog-designer tool itself. One
file, Python standard library only (the project's own tb/*.py parsers may
still import numpy/matplotlib -- whatever they import must be installed).

This file is GENERATED from analog-designer-core
(analog_designer/standalone/run_tb.py) and vendored into project repos by
`python -m analog_designer.standalone.export <project_root>`. Edit it there
and re-export rather than editing a vendored copy, so every project stays
in step with the core's own simulation pipeline (see run_sim.py, whose
behavior this mirrors: same parameter resolution, condition grid,
testbench placeholders, PDK corner names, parser contract, and -- in the
default "sim" mode -- the same sim/ folder layout and jsonl formats, so the
GUI picks standalone results up as fresh).

Two output modes:
  default   writes into <project>/sim exactly like the core does
            (sim/variations.jsonl, sim/results.jsonl, sim/<variation>/...),
            skipping tests whose result is already fresh (--force reruns).
            Materialized schematics go to sim/<variation>/_src/, never to
            the tracked sch/ tree.
  --dry     everything goes to a temporary directory that is deleted at
            the end (--keep to inspect it); the project tree is only read,
            nothing is written to it (not even __pycache__). Results are
            printed, and written to --json if given.

Examples:
  python3 tools/run_tb.py --doctor
  python3 tools/run_tb.py --list
  python3 tools/run_tb.py --block cmos_vref --test temp_sweep --dry
  python3 tools/run_tb.py --block cmos_vref --where corner=tt --where temperature=25
  python3 tools/run_tb.py --block top --param X1_variation=cmos_vref-default-1a2b3c

Tool/PDK discovery (flags override environment, environment overrides
defaults): --pdk-root/$PDK_ROOT, --pdk/$PDK (default: the tag of
config.json's container.image, e.g. "eda-env-designer:ihp-sg13cmos5l" ->
"ihp-sg13cmos5l"), $XSCHEM/$NGSPICE/$XYCE binaries (default: from PATH).
"""
import argparse
import ast
import concurrent.futures
import datetime
import hashlib
import importlib.util
import itertools
import json
import math
import operator
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

# Loading the project's tb/*.py parsers must not leave __pycache__ behind in
# the project tree (--dry promises no trace), and must never try to open a
# GUI window from inside a container.
sys.dont_write_bytecode = True
os.environ.setdefault("MPLBACKEND", "Agg")

STANDALONE_VERSION = 1

# ---------------------------------------------------------------------------
# SPICE values (mirror of analog_designer/sim/spice_value.py)
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Project / config
# ---------------------------------------------------------------------------

class RunError(Exception):
    """A user-facing configuration/setup problem -- printed without a
    traceback."""


class Project:
    """A project folder: config.json (with every topology's parameters_file
    spliced in, as workspace.resolve_parameters_files() does) + where
    results go. results_dir is <root>/sim in the default mode and a
    throwaway directory in --dry mode."""

    def __init__(self, root, results_dir, persist):
        self.root = Path(root).resolve()
        config_path = self.root / "config.json"
        if not config_path.exists():
            raise RunError(f"no config.json in {self.root} -- not a project folder (use --project-root)")
        self.config = json.loads(config_path.read_text(encoding="utf-8"))
        for block in self.config.get("blocks", {}).values():
            for topology in block.get("topologies", {}).values():
                rel_path = topology.get("parameters_file")
                if not rel_path:
                    continue
                doc = json.loads((self.root / rel_path).read_text(encoding="utf-8"))
                topology["parameters"] = doc.get("parameters", {})
                topology["symmetry"] = doc.get("symmetry", {})
                topology["derived_parameters"] = doc.get("derived_parameters", {})
                topology["layout"] = doc.get("layout", {})
        self.results_dir = Path(results_dir)
        self.persist = persist
        self.import_metric_overrides = {}

    def topology_cfg(self, block, topology):
        try:
            return self.config["blocks"][block]["topologies"][topology]
        except KeyError:
            raise RunError(f"unknown block/topology {block}/{topology}") from None

    def tests(self, block):
        return self.config.get("tests", {}).get(block, {})

    @property
    def defaults(self):
        return self.config.get("defaults", {})

    def read_jsonl(self, name):
        """Rows of <project>/sim/<name> -- read from the REAL project sim/
        even in --dry mode (read-only), so a dry run can still resolve a
        registered variation name or a sub-block's stored metric."""
        path = self.root / "sim" / name
        if self.persist:
            path = self.results_dir / name
        return _read_jsonl(path)


def _read_jsonl(path):
    if not path.exists():
        return []
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return rows


_JSONL_LOCK = threading.Lock()


def _append_jsonl(path, row):
    path.parent.mkdir(parents=True, exist_ok=True)
    with _JSONL_LOCK, path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")


# ---------------------------------------------------------------------------
# Parameter resolution (mirror of run_sim.py's resolve_* chain)
# ---------------------------------------------------------------------------

PARAM_TOKEN_RE = re.compile(r"'([A-Za-z_][A-Za-z0-9_]*)'")
BLOCK_REF_DEFAULT = "defaults"


def substitute_params(text, params):
    for name, value in params.items():
        text = text.replace(f"'{name}'", str(value))
    return text


def check_unresolved(text, label):
    leftover = sorted(set(PARAM_TOKEN_RE.findall(text)))
    if leftover:
        raise RunError(
            f"{label}: unresolved parameter placeholder(s) left after substitution: "
            f"{', '.join(leftover)} -- add them to config.json or check the schematic."
        )


def variation_name(block, topology, params):
    digest = hashlib.sha1(json.dumps(params, sort_keys=True).encode()).hexdigest()[:6]
    return f"{block}-{topology}-{digest}"


def default_params(topology_cfg):
    return {name: pdef["default"] for name, pdef in topology_cfg.get("parameters", {}).items()}


def resolve_derived_params(block_cfg, params):
    derived = dict(params)
    for group in block_cfg.get("derived_parameters", {}).get("width_groups", []):
        base_name = group["base"]
        if base_name not in params:
            raise RunError(f"{group['id']}: missing {base_name!r} (pre-migration parameter schema)")
        base_value = parse_spice_value(params[base_name])
        _, unit_suffix = _match(params[base_name])
        for derived_name, member in group["members"].items():
            factor_name = member["factor"]
            if factor_name not in params:
                raise RunError(f"{group['id']}: missing {factor_name!r} (pre-migration parameter schema)")
            factor = int(round(parse_spice_value(params[factor_name])))
            derived[derived_name] = format_spice_value(base_value * factor, unit_suffix)
    return derived


def resolve_generator_params(project, block_cfg, params):
    if "generator" not in block_cfg:
        return dict(params)
    generator = _load_module(project.root / block_cfg["generator"])
    geometry = generator.geometry_from_params(params)
    stack = generator.load_stack("tt")
    fitted = generator.fit_electrical_params(geometry, stack, em_result=None)
    resolved = dict(params)
    resolved.update({name: format_spice_value(value, "") for name, value in fitted.items()})
    return resolved


_FORMULA_BINOPS = {
    ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
    ast.Div: operator.truediv, ast.Pow: operator.pow,
}
_FORMULA_UNARYOPS = {ast.UAdd: operator.pos, ast.USub: operator.neg}
_FORMULA_FUNCS = {
    "ceil": lambda x: math.ceil(x - 1e-9),
    "floor": lambda x: math.floor(x + 1e-9),
}


def _eval_formula(expr, values):
    def _visit(node):
        if isinstance(node, ast.Expression):
            return _visit(node.body)
        if isinstance(node, ast.BinOp) and type(node.op) in _FORMULA_BINOPS:
            return _FORMULA_BINOPS[type(node.op)](_visit(node.left), _visit(node.right))
        if isinstance(node, ast.UnaryOp) and type(node.op) in _FORMULA_UNARYOPS:
            return _FORMULA_UNARYOPS[type(node.op)](_visit(node.operand))
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in _FORMULA_FUNCS
                and len(node.args) == 1 and not node.keywords):
            return _FORMULA_FUNCS[node.func.id](_visit(node.args[0]))
        if isinstance(node, ast.Name):
            return values[node.id]
        raise ValueError(f"unsupported expression node in formula {expr!r}: {ast.dump(node)}")
    return _visit(ast.parse(expr, mode="eval"))


def resolve_formulas(block_cfg, params):
    resolved = dict(params)
    for name, entry in block_cfg.get("derived_parameters", {}).get("constants", {}).items():
        resolved[name] = entry["value"]
    for name, entry in block_cfg.get("derived_parameters", {}).get("formulas", {}).items():
        values = {}
        for n, v in resolved.items():
            try:
                values[n] = parse_spice_value(v)
            except (ValueError, TypeError, AttributeError):
                continue
        resolved[name] = format_spice_value(_eval_formula(entry["expr"], values), entry.get("unit_suffix", ""))
    return resolved


def _variation_params(project, name):
    for row in project.read_jsonl("variations.jsonl"):
        if row["name"] == name:
            return row["parameters"]
    raise RunError(f"variation {name!r} is not registered in sim/variations.jsonl")


def resolve_sub_block_params(project, sub_blocks, params):
    resolved_by_instance = {}
    for instance, ref in sub_blocks.items():
        sub_block_cfg = project.topology_cfg(ref["block"], ref["topology"])
        chosen = params.get(f"{instance}_variation")
        if chosen and chosen != BLOCK_REF_DEFAULT:
            sub_params = _variation_params(project, chosen)
        else:
            sub_params = default_params(sub_block_cfg)
        resolved_by_instance[instance] = resolve_formulas(
            sub_block_cfg,
            resolve_generator_params(project, sub_block_cfg, resolve_derived_params(sub_block_cfg, sub_params)),
        )
    return resolved_by_instance


def materialize_sub_blocks(project, sub_blocks, params, sch_dir):
    resolved_by_instance = resolve_sub_block_params(project, sub_blocks, params)
    for instance, ref in sub_blocks.items():
        sub_block_cfg = project.topology_cfg(ref["block"], ref["topology"])
        materialized_name = f"{ref['block']}.sch"
        topology_sch = project.root / "sch" / sub_block_cfg["schematic"]
        materialized = substitute_params(topology_sch.read_text(encoding="utf-8"), resolved_by_instance[instance])
        check_unresolved(materialized, materialized_name)
        (sch_dir / materialized_name).write_text(materialized, encoding="utf-8")
    return resolved_by_instance


def resolve_import_params(block_cfg, params, sub_block_resolved):
    resolved = dict(params)
    for name, entry in block_cfg.get("derived_parameters", {}).get("import_params", {}).items():
        resolved[name] = sub_block_resolved[entry["from"]][entry["source"]]
    return resolved


def resolve_import_metrics(project, block_cfg, params, sub_block_variation_names):
    """Same lookup as run_sim.resolve_import_metrics() -- a sub-block
    variation's stored result in sim/results.jsonl -- plus
    --import-metric NAME=VALUE overrides, which is the only way to satisfy
    an import_metrics entry in a fresh checkout / --dry run with no stored
    results yet."""
    resolved = dict(params)
    for name, entry in block_cfg.get("derived_parameters", {}).get("import_metrics", {}).items():
        if name in project.import_metric_overrides:
            resolved[name] = project.import_metric_overrides[name]
            continue
        from_instance = entry["from"]
        from_variation = sub_block_variation_names[from_instance]
        hint = (
            f"-- run test {entry['test']!r} for that sub-block variation first (without --dry), "
            f"or pass --import-metric {name}=<value>"
        )
        if from_variation == BLOCK_REF_DEFAULT:
            raise RunError(
                f"{name}: import_metrics needs a registered {from_instance}_variation, not "
                f"{BLOCK_REF_DEFAULT!r} (use --param {from_instance}_variation=<name>) {hint}"
            )
        stat = entry.get("stat", "typical")
        matches = [
            row for row in project.read_jsonl("results.jsonl")
            if row["variation"] == from_variation and row["test"] == entry["test"] and row["metric"] == entry["metric"]
        ]
        if not matches:
            raise RunError(f"{name}: no stored result for {from_variation!r} {entry['test']!r}/{entry['metric']!r} {hint}")
        measured = matches[-1][stat]
        if measured <= 0:
            raise RunError(f"{name}: measured value {measured!r} of {entry['metric']!r} is not positive")
        resolved[name] = str(measured)
    return resolved


def resolve_materialization_params(project, block_cfg, params, sch_dir):
    resolved = resolve_derived_params(block_cfg, params)
    resolved = resolve_generator_params(project, block_cfg, resolved)
    sub_blocks = block_cfg.get("sub_blocks")
    if sub_blocks:
        sub_block_resolved = materialize_sub_blocks(project, sub_blocks, params, sch_dir)
        resolved = resolve_import_params(block_cfg, resolved, sub_block_resolved)
        names = {instance: params.get(f"{instance}_variation") or BLOCK_REF_DEFAULT for instance in sub_blocks}
        resolved = resolve_import_metrics(project, block_cfg, resolved, names)
    return resolve_formulas(block_cfg, resolved)


def materialize_variation(project, block, block_cfg, params, src_dir, xschemrc_text):
    """materialize_variation_shadow()'s equivalent, and the ONLY
    materialization mode here: a full copy of the project's sch/ tree into
    src_dir/sch with this block (and its sub_blocks) substituted, plus an
    xschemrc next to it. The project's own sch/ is never written."""
    sch_dir = src_dir / "sch"
    if sch_dir.exists():
        shutil.rmtree(sch_dir)
    shutil.copytree(project.root / "sch", sch_dir)
    materialized_name = f"{block}.sch"
    topology_sch = project.root / "sch" / block_cfg["schematic"]
    materialized = substitute_params(
        topology_sch.read_text(encoding="utf-8"),
        resolve_materialization_params(project, block_cfg, params, sch_dir),
    )
    check_unresolved(materialized, materialized_name)
    (sch_dir / materialized_name).write_text(materialized, encoding="utf-8")
    rcfile = src_dir / "xschemrc"
    rcfile.write_text(xschemrc_text, encoding="utf-8")
    return rcfile


# ---------------------------------------------------------------------------
# Conditions (mirror of run_sim.py)
# ---------------------------------------------------------------------------

INTERNAL_SWEEP_AXES = {"temperature": "temp"}
_NON_FIXED_CONDITION_KEYS = {"corner", "temperature", "vdd", "Cload", "Rload", "typical"}


def internal_sweep_axis(test_cfg, tb_text):
    for key in test_cfg.get("conditions", {}):
        prefix = INTERNAL_SWEEP_AXES.get(key, key)
        if f"'{prefix}_min'" in tb_text and f"'{prefix}_max'" in tb_text:
            return key, prefix
    return None


def fixed_tb_params(test_cfg, tb_text, sweep_axis):
    skip = set(_NON_FIXED_CONDITION_KEYS)
    if sweep_axis:
        skip.add(sweep_axis[0])
    params = {}
    for key, values in test_cfg.get("conditions", {}).items():
        if key in skip or f"'{key}'" not in tb_text or len(values) > 1:
            continue
        params[key] = values[0]
    return params


def condition_matrix(test_cfg, defaults, sweep_axis):
    conditions = dict(test_cfg.get("conditions", {}))
    conditions.pop("typical", None)
    if sweep_axis:
        conditions.pop(sweep_axis[0], None)
    corners = conditions.pop("corner", [defaults["corner"]])
    if sweep_axis and sweep_axis[0] == "temperature":
        axes = {"corner": corners}
        axes.update({key: values for key, values in conditions.items() if len(values) > 1})
    else:
        temperatures = conditions.pop("temperature", [defaults["temperature"]])
        axes = {"corner": corners, "temperature": temperatures}
        axes.update({key: values for key, values in conditions.items() if len(values) > 1})
    keys = list(axes)
    for combo in itertools.product(*(axes[key] for key in keys)):
        yield dict(zip(keys, combo))


def condition_label(conditions):
    return "_".join(f"{k}-{v}" for k, v in conditions.items())


def typical_conditions(test_cfg, defaults):
    typical = dict(defaults)
    typical.update(test_cfg.get("conditions", {}).get("typical", {}))
    return typical


def compute_definition_hash(project, block_cfg, test_cfg):
    """Byte-for-byte the same hash as run_sim.compute_definition_hash(), so
    results written in the default mode count as fresh for the GUI too."""
    root = project.root
    parts = [json.dumps(test_cfg, sort_keys=True)]
    parts.append((root / "sch" / block_cfg["schematic"]).read_text(encoding="utf-8"))
    if "generator" in block_cfg:
        try:
            parts.append((root / block_cfg["generator"]).read_text(encoding="utf-8"))
        except Exception:
            parts.append(block_cfg["generator"])
    if "testbench" in test_cfg:
        parts.append((root / test_cfg["testbench"]).read_text(encoding="utf-8"))
    parser_path = root / test_cfg["parser"]
    parts.append(parser_path.read_text(encoding="utf-8"))
    common_path = parser_path.parent / "_common.py"
    if common_path.exists():
        parts.append(common_path.read_text(encoding="utf-8"))
    shared_common_path = root / "tb" / "_shared" / "parser_common.py"
    if shared_common_path.exists():
        parts.append(shared_common_path.read_text(encoding="utf-8"))
    return hashlib.sha256("\n".join(parts).encode("utf-8")).hexdigest()[:12]


# ---------------------------------------------------------------------------
# PDK / tools
# ---------------------------------------------------------------------------

MOS_CORNER_SECTION = {
    "tt": "mos_tt", "ss": "mos_ss", "ff": "mos_ff",
    "tt_mismatch": "mos_tt_mismatch", "tt_stat": "mos_tt_stat",
}
MOS_CORNER_SECTION_GF180MCU = {"tt": "typical", "ss": "ss", "ff": "ff"}
RES_CORNER_SECTION = {
    "typ": "res_typ", "bcs": "res_bcs", "wcs": "res_wcs",
    "typ_mismatch": "res_typ_mismatch", "bcs_mismatch": "res_bcs_mismatch", "wcs_mismatch": "res_wcs_mismatch",
    "typ_stat": "res_typ_stat",
}
_RES_SECTION_CMOS5L = {"res_typ_stat": "res_stat"}
# Every OSDI model the IHP PDKs' own reference .spiceinit loads; cap_cmom*
# only exist in CMOS5L.
_IHP_OSDI = ["psp103", "psp103_nqs", "r3_cmc", "mosvar"]
_IHP_OSDI_OPTIONAL = ["cap_cmomi", "cap_cmomf"]


class Env:
    """Everything the container-side ContainerCtx resolves, resolved
    locally instead: PDK paths, OSDI loads, corner tables, binaries."""

    def __init__(self, project, args):
        self.problems = []
        image = project.config.get("container", {}).get("image", "")
        self.pdk_root = Path(args.pdk_root or os.environ.get("PDK_ROOT") or "/foss/pdks")
        self.pdk_name = args.pdk or os.environ.get("PDK") or (image.rsplit(":", 1)[1] if ":" in image else "")
        if not self.pdk_name:
            self.problems.append("PDK name unknown: set $PDK or pass --pdk")
        self.pdk_dir = self.pdk_root / self.pdk_name
        if not self.pdk_dir.is_dir():
            self.problems.append(f"PDK directory not found: {self.pdk_dir} (set $PDK_ROOT/$PDK or --pdk-root/--pdk)")

        self.xschem = args.xschem or os.environ.get("XSCHEM") or _which("xschem", "/usr/local/share/xschem/bin/xschem")
        self.ngspice = args.ngspice or os.environ.get("NGSPICE") or _which("ngspice")
        self.xyce = args.xyce or os.environ.get("XYCE") or _which("Xyce")
        self.xvfb_run = _which("xvfb-run") if args.xvfb else None
        if args.xvfb and not self.xvfb_run:
            self.problems.append("--xvfb given but xvfb-run not found")
        # xschem only netlists here (-x), which recent versions do without
        # any X server. Older ones want one, so reuse a running server if
        # the container has one (e.g. the EDA image's own Xvnc on :1, which
        # run_sim.py hardcodes as DISPLAY=:1) even when this shell lacks
        # $DISPLAY.
        self.display = os.environ.get("DISPLAY")
        if not self.display and not self.xvfb_run:
            sockets = sorted(Path("/tmp/.X11-unix").glob("X*")) if Path("/tmp/.X11-unix").is_dir() else []
            if sockets:
                self.display = ":" + sockets[0].name[1:]

        tech = self.pdk_dir / "libs.tech"
        self.gf180 = self.pdk_name.startswith("gf180mcu")
        if self.gf180:
            self.models_dir = tech / "ngspice"
            self.xyce_models_dir = tech / "xyce"
            self.stdcell_dir = ""
            self.xyce_plugin = None
            self.mos_corner_section = MOS_CORNER_SECTION_GF180MCU
            self.spiceinit_text = ""
        else:
            pdk_short = self.pdk_name[4:] if self.pdk_name.startswith("ihp-") else self.pdk_name
            self.models_dir = tech / "ngspice" / "models"
            self.xyce_models_dir = tech / "xyce" / "models"
            self.stdcell_dir = self.pdk_dir / "libs.ref" / f"{pdk_short}_stdcell" / "spice"
            plugin = tech / "xyce" / "plugins" / f"libXyce_Plugin_{self.pdk_name.replace('-', '_')}.so"
            self.xyce_plugin = Path(args.xyce_plugin) if args.xyce_plugin else plugin
            self.mos_corner_section = MOS_CORNER_SECTION
            osdi_dirs = [Path(d) for d in (args.osdi_dir or [])] + [tech / "ngspice" / "osdi"]
            loads = []
            for name in _IHP_OSDI + _IHP_OSDI_OPTIONAL:
                found = next((d / f"{name}.osdi" for d in osdi_dirs if (d / f"{name}.osdi").is_file()), None)
                if found:
                    loads.append(f"osdi {found}")
                elif name in _IHP_OSDI:
                    self.problems.append(f"OSDI model {name}.osdi not found in {', '.join(map(str, osdi_dirs))} "
                                         f"(compile it with openvaf, or pass --osdi-dir)")
            self.spiceinit_text = "\n".join(loads + [""])

        rc = project.root / "xschemrc"
        if not rc.is_file():
            rc = tech / "xschem" / "xschemrc"
        self.xschemrc = rc if rc.is_file() else None
        if self.xschemrc is None:
            self.problems.append(f"no xschemrc in the project root nor at {tech / 'xschem' / 'xschemrc'}")
        if not self.xschem:
            self.problems.append("xschem not found (PATH or $XSCHEM)")

    def xschemrc_text(self):
        """The rcfile copied next to the materialized sch/ tree. PDK_ROOT/PDK
        are exported to xschem's environment too, so an rcfile using
        $env(PDK_ROOT) works even if the caller didn't export them."""
        if self.xschemrc is None:
            raise RunError("no xschemrc available (see --doctor)")
        return self.xschemrc.read_text(encoding="utf-8")

    def tool_env(self, n_threads=1):
        env = dict(os.environ)
        env["PDK_ROOT"] = str(self.pdk_root)
        env["PDK"] = self.pdk_name
        env["OMP_NUM_THREADS"] = str(n_threads)
        if self.display:
            env["DISPLAY"] = self.display
        return env

    def report(self, simulators):
        rows = [
            ("project PDK", f"{self.pdk_name} @ {self.pdk_dir}"),
            ("xschem", self.xschem or "MISSING"),
            ("xschemrc", self.xschemrc or "MISSING"),
            ("X display", f"xvfb-run ({self.xvfb_run})" if self.xvfb_run else (self.display or "none (xschem -x)")),
            ("ngspice", self.ngspice or "MISSING"),
            ("Xyce", self.xyce or "MISSING"),
            ("models (ngspice)", self.models_dir),
        ]
        if self.spiceinit_text:
            rows.append(("osdi", ", ".join(Path(l.split()[1]).name for l in self.spiceinit_text.split("\n") if l)))
        for k, v in rows:
            print(f"  {k:<17} {v}")
        problems = list(self.problems)
        if "ngspice" in simulators and not self.ngspice:
            problems.append("ngspice not found (PATH or $NGSPICE) but the selected tests need it")
        if "xyce" in simulators and not self.xyce:
            problems.append("Xyce not found (PATH or $XYCE) but the selected tests need it")
        return problems


def _which(name, *fallbacks):
    found = shutil.which(name)
    if found:
        return found
    return next((f for f in fallbacks if Path(f).is_file()), None)


def _run(cmd, cwd, env, timeout):
    # subprocess's cwd= does not update $PWD, and xschemrc files read
    # $env(PWD) (the IHP one aborts without it, before adding the PDK's
    # symbol paths; a project one resolves sch/ against it) -- run_sim.py
    # gets the same effect from `cd <dir> && ...` in a shell.
    env = dict(env, PWD=str(cwd))
    try:
        return subprocess.run(cmd, cwd=cwd, env=env, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired as exc:
        out = exc.stdout.decode(errors="replace") if isinstance(exc.stdout, bytes) else (exc.stdout or "")
        err = exc.stderr.decode(errors="replace") if isinstance(exc.stderr, bytes) else (exc.stderr or "")
        return subprocess.CompletedProcess(cmd, 124, out, err + f"\n[timed out after {timeout}s]")


# ---------------------------------------------------------------------------
# Netlist + simulate (mirror of run_sim.py's _netlist / run_one_*)
# ---------------------------------------------------------------------------

_NETLIST_MAX_ATTEMPTS = 3
_NGSPICE_ERROR_RE = re.compile(r"^Error:\s*(.+)$", re.MULTILINE)
_NGSPICE_ABORTED_RE = re.compile(r"^\s*(\S+ simulation\(s\) aborted)\s*$", re.MULTILINE)
_NGSPICE_TIMESTEP_RE = re.compile(r"^(doAnalyses:.*(?:[Tt]imestep too small|not converge).*)$", re.MULTILINE)
_AUX_OUTPUT_RE = re.compile(r"^(?P<cmd>write|wrdata)\s+(?P<path>\S+)(?P<rest>.*)$", re.IGNORECASE)


def _ngspice_errors(log_text):
    """The error-severity subset of log_diagnostics.parse(..., "ngspice"):
    ngspice can exit 0 with a data file on disk after an analysis-level
    failure, which the core treats as a failed run."""
    found = []
    for regex in (_NGSPICE_ERROR_RE, _NGSPICE_ABORTED_RE, _NGSPICE_TIMESTEP_RE):
        found += [m.group(1).strip() for m in regex.finditer(log_text)]
    return found


def _missing_project_subckts(project, netlist_text):
    blocks = {name.lower() for name in project.config.get("blocks", {})}
    lines = []
    for raw in netlist_text.splitlines():
        if raw.startswith("+") and lines:
            lines[-1] += " " + raw[1:]
        else:
            lines.append(raw)
    defined, used = set(), set()
    for line in lines:
        tokens = line.split()
        if not tokens:
            continue
        head = tokens[0].lower()
        if head == ".subckt" and len(tokens) > 1:
            defined.add(tokens[1].lower())
        elif head.startswith("x"):
            model = next((tok for tok in reversed(tokens[1:]) if "=" not in tok), None)
            if model is not None and model.lower() in blocks:
                used.add(model.lower())
    return used - defined


class Job:
    """One (test, condition) simulation."""

    def __init__(self, test_name, test_cfg, simulator, tb_source, conditions, tb_params_base, run_dir):
        self.test_name = test_name
        self.test_cfg = test_cfg
        self.simulator = simulator
        self.tb_source = tb_source
        self.conditions = conditions
        self.label = condition_label(conditions)
        self.tb_params_base = tb_params_base
        self.run_dir = run_dir


def _netlist(project, env, job, rcfile):
    run_dir = job.run_dir
    run_dir.mkdir(parents=True, exist_ok=True)
    tb_params = dict(job.tb_params_base)
    tb_params["mos_corner"] = env.mos_corner_section[job.conditions["corner"]]
    for key, value in job.conditions.items():
        if key != "corner":
            tb_params[key] = value
    if "vdd" in job.conditions:
        tb_params["Vavdd"] = job.conditions["vdd"]
    if "res_corner" in tb_params:
        tb_params["res_corner"] = RES_CORNER_SECTION[tb_params["res_corner"]]
        if env.pdk_name == "ihp-sg13cmos5l":
            tb_params["res_corner"] = _RES_SECTION_CMOS5L.get(tb_params["res_corner"], tb_params["res_corner"])
    tb_params["simpath"] = str(run_dir)

    tb_text = substitute_params(job.tb_source.read_text(encoding="utf-8"), tb_params)
    check_unresolved(tb_text, f"{job.test_name} {job.label} ({job.tb_source.name})")
    (run_dir / job.tb_source.name).write_text(tb_text, encoding="utf-8")

    # cwd = the rcfile's own directory: project xschemrc files resolve bare
    # "sch/<name>" references against $env(PWD) (see run_sim._netlist()).
    cmd = [env.xschem, "--rcfile", str(rcfile), "-n", "-x", "-q", "-o", str(run_dir), str(run_dir / job.tb_source.name)]
    tenv = env.tool_env()
    if env.xvfb_run:
        cmd = [env.xvfb_run, "-a"] + cmd
    netlist_path = run_dir / f"{job.tb_source.stem}.spice"
    netlist_path.unlink(missing_ok=True)
    for attempt in range(1, _NETLIST_MAX_ATTEMPTS + 1):
        result = _run(cmd, cwd=rcfile.parent, env=tenv, timeout=120)
        if netlist_path.exists():
            missing = _missing_project_subckts(project, netlist_path.read_text(encoding="utf-8"))
            if not missing:
                break
            failure = f"netlist is missing the .subckt expansion of {', '.join(sorted(missing))}"
            netlist_path.unlink()
        else:
            failure = "netlist failed, no .spice produced"
        if attempt == _NETLIST_MAX_ATTEMPTS:
            return {"status": "error", "error": f"{failure} after {attempt} attempt(s)\n{result.stdout}\n{result.stderr}"}
        time.sleep(0.5)
    netlist_text = netlist_path.read_text(encoding="utf-8")
    if "IS MISSING" in netlist_text:
        missing = [l for l in netlist_text.splitlines() if "IS MISSING" in l]
        return {"status": "error", "error": "netlist has unresolved symbols:\n" + "\n".join(missing)}
    return netlist_path


def _redirect_aux_outputs(netlist_path, run_dir, scratch_dir, primary_name):
    """Every write/wrdata target in run_dir other than the parser's own
    <test>_0.data goes to scratch_dir (deleted after the run unless it
    failed or --keep-aux) -- the same policy as run_sim's
    _redirect_aux_outputs(), which exists because a full `save all` .raw
    can be tens of MB per condition."""
    lines = netlist_path.read_text(encoding="utf-8").splitlines()
    prefix = str(run_dir).rstrip("/") + "/"
    moved = False
    for i, line in enumerate(lines):
        match = _AUX_OUTPUT_RE.match(line.strip())
        if not match or not match.group("path").startswith(prefix):
            continue
        name = match.group("path")[len(prefix):]
        if name == primary_name or "/" in name:
            continue
        lines[i] = f"{match.group('cmd')} {scratch_dir}/{name}{match.group('rest')}"
        moved = True
    if moved:
        netlist_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return moved


def run_ngspice(project, env, job, rcfile, opts):
    if not env.ngspice:
        return {"status": "error", "error": "ngspice not found"}
    run_dir = job.run_dir
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / ".spiceinit").write_text(env.spiceinit_text + "\nset num_threads=1\n", encoding="utf-8")
    netlist = _netlist(project, env, job, rcfile)
    if isinstance(netlist, dict):
        return netlist
    data_file = run_dir / f"{job.test_name}_0.data"
    data_file.unlink(missing_ok=True)
    scratch = Path(tempfile.mkdtemp(prefix="run_tb_aux_"))
    try:
        _redirect_aux_outputs(netlist, run_dir, scratch, data_file.name)
        result = _run([env.ngspice, "-b", netlist.name], cwd=run_dir, env=env.tool_env(), timeout=opts.timeout)
        log_text = result.stdout + "\n" + result.stderr
        (run_dir / "ngspice.log").write_text(log_text, encoding="utf-8")
        errors = _ngspice_errors(log_text)
        failed = result.returncode != 0 or not data_file.exists() or errors
        if (failed or opts.keep_aux) and any(scratch.iterdir()):
            shutil.copytree(scratch, run_dir, dirs_exist_ok=True)
    finally:
        shutil.rmtree(scratch, ignore_errors=True)
    if failed:
        return {"status": "error", "exit_code": result.returncode, "error": log_text[-2000:],
                "diagnostics": [{"severity": "error", "message": e} for e in errors]}
    return {"status": "success", "exit_code": result.returncode, "data_file": data_file}


def run_netlist(project, env, job, rcfile, opts):
    netlist = _netlist(project, env, job, rcfile)
    if isinstance(netlist, dict):
        return netlist
    return {"status": "success", "data_file": netlist}


_CMOMF_LINE_RE = re.compile(r"^(?P<name>[Xx]\S+)\s+(?P<plus>\S+)\s+(?P<minus>\S+)\s+cap_cmomf\s+(?P<params>.*)$")
_CMOM_INSTANCE_RE = re.compile(r"^[Xx]\S+\s+.*\bcap_cmom[fi]\b")
_CORNERCAP_LIB_RE = re.compile(r"^\.lib\s+\S*cornerCAP\.lib\b", re.IGNORECASE)


def _prepare_xyce_netlist(netlist_path):
    """run_sim's _strip_ngspice_save_lines() + _lower_cmomf_for_xyce():
    Xyce rejects ngspice's bare `.save i(...)`, and has no cap_cmomf model
    (an ideal C of the same value is exact for it)."""
    lines = [
        line for line in netlist_path.read_text(encoding="utf-8").splitlines()
        if not (line.strip().lower().split() and line.strip().lower().split()[0] == ".save")
    ]
    changed = False
    for i, line in enumerate(lines):
        match = _CMOMF_LINE_RE.match(line)
        if not match:
            continue
        kv = dict(item.split("=", 1) for item in match.group("params").split() if "=" in item)
        w_um = parse_spice_value(kv["w"]) * 1e6
        l_um = parse_spice_value(kv["l"]) * 1e6
        mmin, mmax = int(float(kv.get("mmin", 1))), int(float(kv.get("mmax", 4)))
        m = float(kv.get("m", 1))
        areacap = (0.372 if mmin == 1 else 0.305) + (mmax - mmin) * 0.305
        lines[i] = f"C_{match.group('name')} {match.group('plus')} {match.group('minus')} {m * areacap * 1e-15 * w_um * l_um:.6e}"
        changed = True
    if changed and not any(_CMOM_INSTANCE_RE.match(line) for line in lines):
        lines = [line for line in lines if not _CORNERCAP_LIB_RE.match(line)]
    netlist_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _strip_xyce_print_header(data_file):
    out_lines = []
    for line in data_file.read_text(encoding="utf-8").splitlines()[1:]:
        tokens = line.split()[1:]
        if not tokens:
            continue
        try:
            [float(t) for t in tokens]
        except ValueError:
            continue
        out_lines.append(" ".join(tokens))
    data_file.write_text("\n".join(out_lines) + "\n", encoding="utf-8")


def run_xyce(project, env, job, rcfile, opts):
    if not env.xyce:
        return {"status": "error", "error": "Xyce not found"}
    netlist = _netlist(project, env, job, rcfile)
    if isinstance(netlist, dict):
        return netlist
    _prepare_xyce_netlist(netlist)
    data_file = job.run_dir / f"{job.test_name}_0.data"
    data_file.unlink(missing_ok=True)
    cmd = [env.xyce]
    if env.xyce_plugin and env.xyce_plugin.is_file():
        cmd += ["-plugin", str(env.xyce_plugin)]
    cmd.append(netlist.name)
    result = _run(cmd, cwd=job.run_dir, env=env.tool_env(), timeout=opts.timeout)
    log_text = result.stdout + "\n" + result.stderr
    (job.run_dir / "xyce.log").write_text(log_text, encoding="utf-8")
    if result.returncode != 0 or not data_file.exists():
        return {"status": "error", "exit_code": result.returncode, "error": log_text[-2000:]}
    _strip_xyce_print_header(data_file)
    return {"status": "success", "exit_code": result.returncode, "data_file": data_file}


RUNNERS = {"ngspice": run_ngspice, "netlist": run_netlist, "xyce": run_xyce}


# ---------------------------------------------------------------------------
# Parsers
# ---------------------------------------------------------------------------

_PARSER_LOCK = threading.Lock()  # matplotlib (used by many parsers) is not thread-safe


def _load_module(path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_parser(project, relpath):
    module_path = project.root / relpath
    if str(module_path.parent) not in sys.path:
        sys.path.insert(0, str(module_path.parent))
    shared_dir = str(project.root / "tb" / "_shared")
    if shared_dir not in sys.path:
        sys.path.append(shared_dir)
    return _load_module(module_path)


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def build_jobs(project, env, test_name, test_cfg, sim_dir, where):
    tb_source = project.root / test_cfg["testbench"]
    tb_text = tb_source.read_text(encoding="utf-8")
    sweep_axis = internal_sweep_axis(test_cfg, tb_text)
    defaults = project.defaults
    simulator = test_cfg.get("simulator", "ngspice")

    tb_params_base = dict(defaults)
    tb_params_base["Vavdd"] = defaults["vdd"]
    tb_params_base["filename"] = test_name
    tb_params_base["N"] = "0"
    tb_params_base["models_dir"] = str(env.xyce_models_dir if simulator == "xyce" else env.models_dir)
    tb_params_base["stdcell_dir"] = str(env.stdcell_dir)
    if sweep_axis:
        key, prefix = sweep_axis
        values = [float(v) for v in test_cfg.get("conditions", {}).get(key, [])]
        if not values:
            raise RunError(f"{test_name}: testbench sweeps '{prefix}' internally but config.json has no conditions.{key} list")
        tb_params_base[f"{prefix}_min"] = min(values)
        tb_params_base[f"{prefix}_max"] = max(values)
    tb_params_base.update(fixed_tb_params(test_cfg, tb_text, sweep_axis))
    vdd_values = test_cfg.get("conditions", {}).get("vdd")
    if vdd_values and len(vdd_values) == 1 and not (sweep_axis and sweep_axis[0] == "vdd"):
        tb_params_base["vdd"] = tb_params_base["Vavdd"] = vdd_values[0]

    jobs = []
    for conditions in condition_matrix(test_cfg, defaults, sweep_axis):
        if any(k in conditions and str(conditions[k]) != v for k, v in where.items()):
            continue
        jobs.append(Job(test_name, test_cfg, simulator, tb_source, conditions, tb_params_base,
                        sim_dir / test_name / condition_label(conditions)))
    return jobs


def evaluate_test(project, test_name, test_cfg, outcomes):
    """run_sim.run_test()'s post-simulation half: extract() every
    successful condition, then evaluate() them together."""
    parser = load_parser(project, test_cfg["parser"])
    runs, n_ok = [], 0
    for job, outcome in outcomes:
        if outcome["status"] != "success":
            continue
        try:
            raw = parser.extract(outcome["data_file"])
        except Exception as exc:
            print(f"  {test_name} {job.label}: ERROR extracting data ({exc})")
            continue
        n_ok += 1
        runs.append({"conditions": job.conditions, **raw})
    if not runs:
        return {"status": "error", "error": "every condition failed to simulate"}
    typical = typical_conditions(test_cfg, project.defaults)
    try:
        with _PARSER_LOCK:
            metrics = parser.evaluate(runs, test_cfg["outputs"], typical, plot_base=None)
    except Exception as exc:
        return {"status": "error", "error": f"evaluate() failed: {exc}"}
    return {"status": "success" if n_ok == len(outcomes) else "partial", "result": metrics}


def _fmt(value):
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.5g}"
    return str(value)


def print_metrics(test_name, status, metrics):
    print(f"\n{test_name}  [{status}]")
    for m in metrics:
        stats = f"typ {_fmt(m.get('typical'))}  min {_fmt(m.get('min'))}  max {_fmt(m.get('max'))}"
        if m.get("std") is not None:
            stats += f"  mean {_fmt(m.get('mean'))}  std {_fmt(m.get('std'))}"
        print(f"  {m['name']:<44} {stats}  {m.get('unit') or ''}")


def _git_info(root):
    commit = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True, text=True)
    if commit.returncode != 0:
        return None, None
    status = subprocess.run(["git", "-C", str(root), "status", "--porcelain"], capture_output=True, text=True)
    return commit.stdout.strip(), bool(status.stdout.strip())


def run(project, env, args):
    block = args.block or next(iter(project.config["blocks"]))
    block_entry = project.config["blocks"].get(block)
    if block_entry is None:
        raise RunError(f"unknown block {block!r} (have: {', '.join(project.config['blocks'])})")
    if not block_entry.get("topologies"):
        raise RunError(f"block {block!r} has no topologies declared in config.json yet")
    topology = args.topology or next(iter(block_entry["topologies"]))
    block_cfg = project.topology_cfg(block, topology)

    if args.variation:
        params = dict(_variation_params(project, args.variation))
    else:
        params = default_params(block_cfg)
    for override in args.param:
        key, _, value = override.partition("=")
        if key not in block_cfg.get("parameters", {}):
            raise RunError(f"--param {key}: not a parameter of {block}/{topology}")
        params[key] = value
    name = variation_name(block, topology, params)

    all_tests = project.tests(block)
    selected = args.test or list(all_tests)
    unknown = [t for t in selected if t not in all_tests]
    if unknown:
        raise RunError(f"unknown test(s) for block {block}: {', '.join(unknown)} (have: {', '.join(all_tests)})")
    tests = {t: all_tests[t] for t in selected}
    unsupported = [t for t, cfg in tests.items() if cfg.get("simulator", "ngspice") not in RUNNERS]
    for t in unsupported:
        print(f"skipping {t}: simulator {tests[t].get('simulator')!r} is not supported standalone")
        del tests[t]

    hashes = {t: compute_definition_hash(project, block_cfg, cfg) for t, cfg in tests.items()}
    if project.persist and not args.force and not args.where:
        fresh = {
            r["test"] for r in project.read_jsonl("results.jsonl")
            if r["variation"] == name and r["test"] in hashes and r["definition_hash"] == hashes[r["test"]]
        }
        for t in sorted(fresh):
            print(f"skipping {t}: fresh result already in sim/results.jsonl (--force to rerun)")
            del tests[t]

    print(f"variation {name}  ({block}/{topology})  ->  {project.results_dir / name}")
    if not tests:
        return {"variation": name, "tests": {}}

    problems = env.report({cfg.get("simulator", "ngspice") for cfg in tests.values()})
    if problems:
        raise RunError("environment not ready:\n  - " + "\n  - ".join(problems))

    sim_dir = project.results_dir / name
    sim_dir.mkdir(parents=True, exist_ok=True)
    if project.persist:
        if not any(r["name"] == name for r in project.read_jsonl("variations.jsonl")):
            _append_jsonl(project.results_dir / "variations.jsonl", {
                "name": name, "block": block, "topology": topology, "parameters": params,
                "origin": {"kind": "manual", "tool": "run_tb"},
                "created": datetime.datetime.now().isoformat(timespec="seconds"),
            })
    rcfile = materialize_variation(project, block, block_cfg, params, sim_dir / "_src", env.xschemrc_text())

    where = dict(w.partition("=")[::2] for w in args.where)
    jobs_by_test = {t: build_jobs(project, env, t, cfg, sim_dir, where) for t, cfg in tests.items()}
    jobs = [j for js in jobs_by_test.values() for j in js]
    print(f"\nrunning {len(jobs)} simulation(s) across {len(tests)} test(s) with {args.jobs} job(s)")

    started = {t: None for t in tests}
    finished = {t: None for t in tests}
    outcomes = {t: [] for t in tests}
    lock = threading.Lock()
    done = [0]

    def _one(job):
        with lock:
            started[job.test_name] = started[job.test_name] or time.monotonic()
        outcome = RUNNERS[job.simulator](project, env, job, rcfile, args)
        with lock:
            done[0] += 1
            finished[job.test_name] = time.monotonic()
            mark = "ok " if outcome["status"] == "success" else "ERR"
            print(f"  [{done[0]:>4}/{len(jobs)}] {mark} {job.test_name} {job.label}", flush=True)
            if outcome["status"] != "success" and args.verbose:
                print("        " + outcome.get("error", "").strip().replace("\n", "\n        "))
        if project.persist:
            _append_jsonl(sim_dir / "runs.jsonl", {
                "variation": name, "test": job.test_name, "condition": job.label,
                "conditions": job.conditions, "status": outcome["status"],
                "exit_code": outcome.get("exit_code"), "error": outcome.get("error"),
                "diagnostics": outcome.get("diagnostics", []),
                "created": datetime.datetime.now().isoformat(timespec="seconds"),
            })
        return job, outcome

    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, args.jobs)) as pool:
        for job, outcome in pool.map(_one, jobs):
            outcomes[job.test_name].append((job, outcome))

    git_commit, git_dirty = _git_info(project.root)
    summary = {"variation": name, "block": block, "topology": topology, "parameters": params, "tests": {}}
    for test_name, test_cfg in tests.items():
        if not outcomes[test_name]:
            print(f"\n{test_name}: no condition matches --where")
            continue
        result = evaluate_test(project, test_name, test_cfg, outcomes[test_name])
        failed = [{"condition": j.label, "error": o.get("error")} for j, o in outcomes[test_name] if o["status"] != "success"]
        entry = {"status": result["status"], "failed_conditions": failed}
        if result["status"] == "error":
            print(f"\n{test_name}  [ERROR] {result['error']}")
            for f in failed[:3]:
                print(f"  {f['condition']}: {(f['error'] or '').strip().splitlines()[-1:] or ''}")
            entry["error"] = result["error"]
        else:
            print_metrics(test_name, result["status"], result["result"])
            entry["metrics"] = result["result"]
            # A --where-filtered run covers only part of the condition grid:
            # never record it as the test's result.
            if project.persist and not args.where:
                created = datetime.datetime.now().isoformat(timespec="seconds")
                duration = (finished[test_name] or 0) - (started[test_name] or 0)
                for m in result["result"]:
                    _append_jsonl(project.results_dir / "results.jsonl", {
                        "variation": name, "block": block, "topology": topology, "test": test_name,
                        "metric": m["name"], "typical": m["typical"], "min": m["min"], "max": m["max"],
                        "mean": m.get("mean"), "std": m.get("std"), "unit": m.get("unit"),
                        "minimum": m.get("minimum"), "maximum": m.get("maximum"),
                        "definition_hash": hashes[test_name], "git_commit": git_commit, "git_dirty": git_dirty,
                        "created": created, "duration_seconds": duration,
                    })
        summary["tests"][test_name] = entry
    return summary


def list_project(project):
    defaults = project.defaults
    for block, block_entry in project.config.get("blocks", {}).items():
        print(f"{block}")
        for topology in block_entry.get("topologies") or {}:
            print(f"  topology {topology}")
        for test_name, test_cfg in project.tests(block).items():
            tb_path = project.root / test_cfg["testbench"]
            tb_text = tb_path.read_text(encoding="utf-8") if tb_path.exists() else ""
            n = len(list(condition_matrix(test_cfg, defaults, internal_sweep_axis(test_cfg, tb_text))))
            sim = test_cfg.get("simulator", "ngspice")
            note = "" if sim in RUNNERS else "  (not supported standalone)"
            print(f"    test {test_name:<24} {sim:<8} {n:>4} condition(s)  {test_cfg['testbench']}{note}")


def main(argv=None):
    here = Path(__file__).resolve().parent
    default_root = next((p for p in (Path.cwd(), here, here.parent) if (p / "config.json").is_file()), Path.cwd())
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--project-root", default=str(default_root), help="project folder (default: auto-detected)")
    ap.add_argument("--block", help="block (default: the first one in config.json)")
    ap.add_argument("--topology", help="topology (default: the block's first one)")
    ap.add_argument("--test", action="append", default=[], help="test to run (repeatable; default: all of the block's tests)")
    ap.add_argument("--variation", help="a registered variation name (sim/variations.jsonl) instead of the defaults")
    ap.add_argument("--param", action="append", default=[], metavar="NAME=VALUE", help="override one free parameter")
    ap.add_argument("--import-metric", action="append", default=[], metavar="NAME=VALUE",
                    help="value for a derived_parameters.import_metrics entry (hierarchical blocks)")
    ap.add_argument("--where", action="append", default=[], metavar="KEY=VALUE",
                    help="only run conditions with this value (e.g. corner=tt); such partial runs are never recorded")
    ap.add_argument("--dry", action="store_true", help="work in a temporary directory; leave no trace in the project")
    ap.add_argument("--keep", action="store_true", help="with --dry: keep (and print) the temporary directory")
    ap.add_argument("--workdir", help="with --dry: use this directory instead of a temporary one")
    ap.add_argument("--force", action="store_true", help="rerun tests whose stored result is still fresh")
    ap.add_argument("--json", help="write a JSON summary of the results here")
    ap.add_argument("--jobs", type=int, default=os.cpu_count() or 1, help="parallel simulations (default: CPU count)")
    ap.add_argument("--timeout", type=int, default=290, help="per-simulation timeout in seconds (default 290)")
    ap.add_argument("--keep-aux", action="store_true", help="keep auxiliary simulator outputs (.raw etc.)")
    ap.add_argument("--pdk-root"), ap.add_argument("--pdk")
    ap.add_argument("--osdi-dir", action="append", help="extra directory to look for *.osdi in (repeatable)")
    ap.add_argument("--xvfb", action="store_true", help="run xschem under xvfb-run (for an xschem that needs X)")
    ap.add_argument("--xschem"), ap.add_argument("--ngspice"), ap.add_argument("--xyce"), ap.add_argument("--xyce-plugin")
    ap.add_argument("--list", action="store_true", help="list blocks/topologies/tests and exit")
    ap.add_argument("--doctor", action="store_true", help="report tool/PDK discovery and exit")
    ap.add_argument("-v", "--verbose", action="store_true", help="print each failed condition's error")
    args = ap.parse_args(argv)

    tmp = None
    try:
        root = Path(args.project_root)
        if args.dry:
            if args.workdir:
                results_dir = Path(args.workdir).resolve()
            else:
                tmp = tempfile.mkdtemp(prefix="run_tb_")
                results_dir = Path(tmp)
            project = Project(root, results_dir, persist=False)
        else:
            project = Project(root, root.resolve() / "sim", persist=True)
        project.import_metric_overrides = dict(m.partition("=")[::2] for m in args.import_metric)

        if args.list:
            list_project(project)
            return 0
        env = Env(project, args)
        if args.doctor:
            problems = env.report({"ngspice", "xyce"})
            for p in problems:
                print(f"  ! {p}")
            return 1 if problems else 0

        t0 = time.monotonic()
        summary = run(project, env, args)
        any_error = any(t["status"] == "error" for t in summary["tests"].values())
        print(f"\ndone in {time.monotonic() - t0:.1f}s" + ("  (with errors)" if any_error else ""))
        if args.json:
            Path(args.json).write_text(json.dumps(summary, indent=2, default=str) + "\n", encoding="utf-8")
        return 1 if any_error else 0
    except RunError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    finally:
        if tmp:
            if args.keep:
                print(f"kept work directory: {tmp}")
            else:
                shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
