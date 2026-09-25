#!/usr/bin/env python3
"""Cross-checks one block/topology's config.json parameter declarations
against what its schematic actually uses -- catches the two ways they drift
apart without needing a docker/xschem round-trip: a declared parameter no
device in the schematic references any more (orphaned -- dead weight in
every sampler/model, safe to delete), or a live 'placeholder' token in the
schematic that config.json never declares (would otherwise only surface as
analog_designer/sim/run_sim.py's "unresolved parameter placeholder" error, at netlist
time, inside docker). Also does light schema sanity-checking on every
declared parameter: default within [min, max], and an "integer": true
default that's actually a whole number. A topology's `derived_parameters` --
`width_groups` (base*factor within one block, see
analog_designer.sim.run_sim.resolve_derived_params), `import_params`/
`import_metrics` (a PURE fetch -- no math -- of an already-resolved
sub_blocks instance's own parameter, or its own stored test result, into a
local name; e.g. "top"'s own x1_m3_width imports cmos_vref's m3_width, see
resolve_import_params/resolve_import_metrics), and `formulas` (pure
arithmetic combining SEVERAL of this same block's own already-resolved
values -- free parameters, imports, and other derived names alike -- see
resolve_formulas() -- e.g. "top"'s own amp_bias_width/rbot_nominal/
r1_length..r8_length, so a schematic's own w=/l=/etc. attributes never have
to carry a raw multi-parameter ngspice {...} expression themselves) -- all
count as real declarations/usage too: a group's/entry's base/factor/expr free
parameters are only ever referenced there, not as literal schematic tokens,
and their own derived names (e.g. "m6_width", "amp_bias_width",
"r1_length") are only ever schematic tokens, not `parameters` entries -- plus
typo/consistency checks specific to derived_parameters itself (see
derived_errors()).

Usage: python -m analog_designer.sim.check_params [--project-root PATH] [--block NAME] [--topology NAME]
Exit code 1 if anything is flagged, 0 if the schematic and config.json agree.
"""
import argparse
import ast
import re
import sys

from analog_designer.core import workspace
from analog_designer.sim.gen_variations import parse_spice_value

PARAM_TOKEN_RE = re.compile(r"'([A-Za-z_][A-Za-z0-9_]*)'")


def _formula_names(expr):
    """Every bare name a derived_parameters.formulas `expr` string
    references -- the same extraction analog_designer.sim.run_sim.
    formula_names() does for resolve_formulas() itself, reimplemented here
    (rather than imported) so this module's own "no docker/xschem" self-
    containment (see module docstring) doesn't grow a dependency on run_sim's
    much heavier materialization machinery just for this one AST walk."""
    tree = ast.parse(expr, mode="eval")
    called = {node.func.id for node in ast.walk(tree) if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)}
    return {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)} - called


def _safe_formula_names(expr):
    """_formula_names(), but tolerant of a missing/unparseable `expr` --
    returns an empty set instead of raising, for call sites (derived_names()'s
    own "used" expansion) that just want best-effort names and let
    derived_errors() be the one place that actually reports a bad expr."""
    if expr is None:
        return set()
    try:
        return _formula_names(expr)
    except SyntaxError:
        return set()


def used_params(sch_text):
    return set(PARAM_TOKEN_RE.findall(sch_text))


def schema_errors(param_defs):
    """[(name, message), ...] -- declared parameters whose own definition
    is internally inconsistent, independent of the schematic. "block_ref"
    parameters (see block_ref_errors()) are skipped here -- default/min/max
    only make sense for a numeric parameter."""
    errors = []
    for name, pdef in param_defs.items():
        if pdef.get("type") == "block_ref":
            continue
        missing_keys = [k for k in ("default", "min", "max") if k not in pdef]
        if missing_keys:
            errors.append((name, f"missing {', '.join(missing_keys)}"))
            continue
        try:
            default = parse_spice_value(pdef["default"])
            lo = parse_spice_value(pdef["min"])
            hi = parse_spice_value(pdef["max"])
        except ValueError as exc:
            errors.append((name, str(exc)))
            continue
        if not (lo <= default <= hi):
            errors.append((name, f"default {pdef['default']} outside [{pdef['min']}, {pdef['max']}]"))
        if pdef.get("integer") and not float(default).is_integer():
            errors.append((name, f"integer=true but default {pdef['default']} isn't a whole number"))
    return errors


def derived_names(derived_cfg):
    """Every name a derived_parameters.width_groups, import_params,
    import_metrics, formulas, OR generator entry produces (e.g. 'm6_width',
    "top"'s own 'x1_m3_width'/'amp_bias_width'/'rbot_nominal'/'r1_length',
    a "generator"-backed topology's own 'l'/'rs'/'cox'/...) -- these are
    real schematic tokens with no entry of their own in `parameters` (their
    value comes from a group's base*factor, a pure import of another
    block's own resolved value/stored metric, a formulas entry's own
    arithmetic expr, or (generator) a project generator module's own
    fit_electrical_params() -- see analog_designer.sim.run_sim.
    resolve_derived_params/resolve_import_params/resolve_import_metrics/
    resolve_formulas/resolve_generator_params), so check()'s used-but-not-
    declared check must not flag them as missing."""
    names = {
        name
        for group in (derived_cfg or {}).get("width_groups", [])
        for name in group.get("members", {})
    }
    names.update((derived_cfg or {}).get("import_params", {}))
    names.update((derived_cfg or {}).get("import_metrics", {}))
    names.update((derived_cfg or {}).get("formulas", {}))
    names.update((derived_cfg or {}).get("generator", {}))
    names.update((derived_cfg or {}).get("constants", {}))
    return names


def referenced_in_derived(derived_cfg):
    """Free-parameter names consumed as a width_groups entry's own `base`/
    member `factor`, or any name a formulas entry's own `expr` references --
    these never appear as literal '<name>' tokens in the schematic
    themselves (only the entry's own derived output name does), so check()'s
    declared-but-unused check must count this as real usage too, not just
    literal schematic tokens. import_params'/import_metrics' own `from`/
    `source`/`test`/`metric` are NOT included here -- those name a DIFFERENT
    block's own sub_blocks instance/parameter/test, not anything in THIS
    block's own param_defs, so they're validated separately (see
    derived_errors()'s own sub_blocks-aware check) -- an import entry is a
    pure fetch, it never references one of THIS block's own free parameters
    at all. A formula referencing another derived name (width_groups/an
    import/an earlier formula) rather than a free parameter is harmless to
    include here too -- referenced is only ever subtracted from `declared`
    (real `parameters` entries), so a derived name that was never in
    `declared` to begin with is simply a no-op subtraction."""
    referenced = set()
    for group in (derived_cfg or {}).get("width_groups", []):
        referenced.add(group.get("base"))
        referenced.update(member.get("factor") for member in group.get("members", {}).values())
    for entry in (derived_cfg or {}).get("formulas", {}).values():
        referenced.update(_safe_formula_names(entry.get("expr")))  # bad expr reported by derived_errors(), not here
    referenced.discard(None)
    return referenced


def derived_errors(param_defs, derived_cfg, used, sub_blocks=None):
    """[(group_id, message), ...] -- typo/consistency problems in
    derived_parameters itself, independent of schema_errors() (which only
    looks at `parameters`): a `base`/`factor` that doesn't resolve to a real
    declared parameter, a derived name that collides with a declared
    parameter or with another entry's own derived name, a derived name no
    longer referenced in the schematic (stale entry), or a `factor` that
    isn't declared "integer": true (defeats the whole point of the scheme
    silently) -- this last check is specific to width_groups' own "factor"
    form (another parameter multiplies `base`).

    import_params/import_metrics entries are PURE fetches (see
    analog_designer.sim.run_sim.resolve_import_params/resolve_import_metrics)
    -- no factor/ratio math to validate, just that `from` names a declared
    sub_blocks instance (checked when sub_blocks is given; omit for a
    topology with no sub_blocks -- `from`/`source`/`test`/`metric` can't be
    checked at all without it, since they name something in a DIFFERENT
    block's own parameters/tests, not this one's) and that the required keys
    are present: import_params needs `source`; import_metrics needs `test`
    and `metric` (`stat` is optional, defaults to "typical" at resolve time).
    Unlike the old cross_block mechanism this replaced, there's no "self"
    sentinel to special-case -- a formula that wants one of THIS block's own
    free parameters just references it directly (see formulas' own check
    below), no import required for that at all.

    A width_groups/import_params/import_metrics entry's own derived name
    doesn't have to be a literal '<name>' schematic token itself to count as
    "used" here -- it's also legitimately consumed by being referenced from
    a `formulas` entry's own `expr` instead (e.g. "top"'s own rbot_nominal,
    folded into r1_length..r8_length rather than substituted into the
    schematic directly), so the "isn't referenced in the schematic"
    staleness check below treats `used` as (literal schematic tokens) UNION
    (every name any formulas entry's expr references)."""
    errors = []
    seen = {}
    used = used | {
        name
        for entry in (derived_cfg or {}).get("formulas", {}).values()
        for name in _safe_formula_names(entry.get("expr"))
    }
    for group in (derived_cfg or {}).get("width_groups", []):
        gid = group.get("id", "<no id>")
        base = group.get("base")
        if base not in param_defs:
            errors.append((gid, f"base {base!r} is not declared in parameters"))
        for name, member in group.get("members", {}).items():
            if name in param_defs:
                errors.append((gid, f"derived parameter {name!r} is also declared in parameters -- remove one"))
            if name in seen:
                errors.append((gid, f"derived parameter {name!r} is produced by both {seen[name]!r} and {gid!r}"))
            seen[name] = gid
            if name not in used:
                errors.append((gid, f"derived parameter {name!r} isn't referenced in the schematic -- stale group entry?"))
            factor = member.get("factor")
            if factor not in param_defs:
                errors.append((gid, f"factor {factor!r} (member {name!r}) is not declared in parameters"))
            elif not param_defs[factor].get("integer"):
                errors.append((gid, f"factor {factor!r} (member {name!r}) should declare \"integer\": true"))
    for name, entry in (derived_cfg or {}).get("import_params", {}).items():
        label = f"import_params.{name}"
        if name in param_defs:
            errors.append((label, f"derived parameter {name!r} is also declared in parameters -- remove one"))
        if name in seen:
            errors.append((label, f"derived parameter {name!r} is produced by both {seen[name]!r} and {label!r}"))
        seen[name] = label
        if name not in used:
            errors.append((label, f"derived parameter {name!r} isn't referenced in the schematic -- stale import_params entry?"))
        from_instance = entry.get("from")
        if sub_blocks is not None and from_instance not in sub_blocks:
            errors.append((label, f"from {from_instance!r} is not a declared sub_blocks instance"))
        if "source" not in entry:
            errors.append((label, "missing \"source\" (which parameter of the sub-block to pull)"))
    for name, entry in (derived_cfg or {}).get("import_metrics", {}).items():
        label = f"import_metrics.{name}"
        if name in param_defs:
            errors.append((label, f"derived parameter {name!r} is also declared in parameters -- remove one"))
        if name in seen:
            errors.append((label, f"derived parameter {name!r} is produced by both {seen[name]!r} and {label!r}"))
        seen[name] = label
        if name not in used:
            errors.append((label, f"derived parameter {name!r} isn't referenced in the schematic -- stale import_metrics entry?"))
        from_instance = entry.get("from")
        if sub_blocks is not None and from_instance not in sub_blocks:
            errors.append((label, f"from {from_instance!r} is not a declared sub_blocks instance"))
        missing = [k for k in ("test", "metric") if k not in entry]
        if missing:
            errors.append((label, f"missing {', '.join(missing)}"))
    for name, entry in (derived_cfg or {}).get("constants", {}).items():
        label = f"constants.{name}"
        if name in param_defs:
            errors.append((label, f"constant {name!r} is also declared in parameters -- remove one"))
        if name in seen:
            errors.append((label, f"constant {name!r} is produced by both {seen[name]!r} and {label!r}"))
        seen[name] = label
        if name not in used:
            errors.append((label, f"constant {name!r} isn't referenced by any formula/schematic token -- stale constants entry?"))
        if "value" not in entry:
            errors.append((label, 'missing "value"'))
    # Names a formulas entry's own expr is allowed to reference: declared
    # parameters plus every width_groups/import_params/import_metrics
    # derived name already validated above, growing with each formula's own
    # derived name as we go -- matching resolve_formulas()'s own sequential
    # dict-threading (a LATER formula may reference an EARLIER one's result,
    # never the reverse).
    resolvable = set(param_defs) | set(seen)
    for name, entry in (derived_cfg or {}).get("formulas", {}).items():
        label = f"formulas.{name}"
        if name in param_defs:
            errors.append((label, f"derived parameter {name!r} is also declared in parameters -- remove one"))
        if name in seen:
            errors.append((label, f"derived parameter {name!r} is produced by both {seen[name]!r} and {label!r}"))
        seen[name] = label
        if name not in used:
            errors.append((label, f"derived parameter {name!r} isn't referenced in the schematic -- stale formulas entry?"))
        expr = entry.get("expr")
        if expr is None:
            errors.append((label, "missing \"expr\""))
        else:
            try:
                expr_names = _formula_names(expr)
            except SyntaxError as exc:
                errors.append((label, f"expr {expr!r} doesn't parse: {exc}"))
            else:
                unknown = sorted(expr_names - resolvable)
                if unknown:
                    errors.append((
                        label,
                        f"expr references {unknown} -- not a declared parameter, and not an earlier "
                        f"width_groups/import_params/import_metrics/formulas derived name",
                    ))
        resolvable.add(name)
    # "generator" entries have no base/factor/from/expr to validate (their
    # value comes from a project generator module's own
    # fit_electrical_params(), not from anything expressible here) -- just
    # the same name-collision + staleness checks every other kind gets.
    for name, entry in (derived_cfg or {}).get("generator", {}).items():
        label = f"generator.{name}"
        if name in param_defs:
            errors.append((label, f"derived parameter {name!r} is also declared in parameters -- remove one"))
        if name in seen:
            errors.append((label, f"derived parameter {name!r} is produced by both {seen[name]!r} and {label!r}"))
        seen[name] = label
        if name not in used:
            errors.append((label, f"derived parameter {name!r} isn't referenced in the schematic -- stale generator entry?"))
    return errors


def block_ref_names(param_defs):
    """Every parameter declared "type": "block_ref" (e.g. "top"'s own
    "X1_variation") -- these are never literal '<name>' schematic tokens
    themselves (they select WHICH already-registered variation of a
    sub_blocks instance composes this one, purely at materialization time --
    see analog_designer.sim.run_sim.materialize_sub_blocks/BLOCK_REF_DEFAULT),
    so check()'s declared-but-unused check must not flag them as orphans."""
    return {name for name, pdef in param_defs.items() if pdef.get("type") == "block_ref"}


def block_ref_errors(param_defs, sub_blocks, config=None):
    """[(name, message), ...] -- typo/consistency problems specific to
    "block_ref" parameters: a name that doesn't follow the
    "<sub_blocks instance>_variation" convention materialize_sub_blocks()
    actually reads (params[f"{instance}_variation"]), a declared
    block/topology that disagrees with its own sub_blocks entry, or (when
    `config` is given) a block/topology that isn't a real one declared in
    config.json at all. sub_blocks is the topology's own `sub_blocks` --
    omit (None) to skip the instance-matching checks (e.g. no way to tell
    which instance a name refers to without it)."""
    errors = []
    for name, pdef in param_defs.items():
        if pdef.get("type") != "block_ref":
            continue
        if not name.endswith("_variation"):
            errors.append((
                name,
                'block_ref parameter names must end in "_variation" -- '
                'materialize_sub_blocks() reads params[f"{instance}_variation"]',
            ))
            continue
        instance = name[: -len("_variation")]
        if sub_blocks is not None:
            ref = sub_blocks.get(instance)
            if ref is None:
                errors.append((name, f"{instance!r} is not a declared sub_blocks instance"))
            elif pdef.get("block") != ref.get("block") or pdef.get("topology") != ref.get("topology"):
                errors.append((
                    name,
                    f"declares block/topology {pdef.get('block')!r}/{pdef.get('topology')!r}, "
                    f"but sub_blocks[{instance!r}] says {ref.get('block')!r}/{ref.get('topology')!r}",
                ))
        if config is not None:
            block = pdef.get("block")
            topology = pdef.get("topology")
            if block not in config.get("blocks", {}):
                errors.append((name, f"block {block!r} is not declared in config.json"))
            elif topology not in config["blocks"][block].get("topologies", {}):
                errors.append((name, f"topology {topology!r} is not declared for block {block!r}"))
    return errors


def structural_sub_block_errors(param_defs, structural_sub_blocks):
    """[(name, message), ...] -- typo/consistency problems specific to a
    "sub_block" tag ([instance, ...]) on an ordinary free parameter: it
    marks that parameter's resolved value as one that gets threaded down to
    a structural_sub_blocks instance's own call-site attribute line in the
    parent schematic (e.g. "top"'s own m1m2_width feeding half_qvco_cell's
    X1/X2, see sch/vco/vco_quadrature_lc.sch) -- unlike a `sub_blocks`
    instance (see block_ref_errors()), a structural_sub_blocks instance is
    NOT independently variant: no {instance}_variation selector, no
    registered variations of its own, its own schematic is never
    materialized separately -- it shares the SAME parameter-substitution
    pass as its parent, so a "sub_block"-tagged parameter is still a real,
    literal '<name>' token in the parent's own schematic text (on that
    instance's own call-site attribute lines) and stays subject to the
    ordinary used/orphan check above, unlike a block_ref parameter.
    structural_sub_blocks is the topology's own `structural_sub_blocks`
    (config.json) -- omit (None) for a topology with none, in which case
    any "sub_block" tag at all is itself the error."""
    errors = []
    for name, pdef in param_defs.items():
        tag = pdef.get("sub_block")
        if tag is None:
            continue
        if not isinstance(tag, list) or not tag or not all(isinstance(i, str) for i in tag):
            errors.append((name, '"sub_block" must be a non-empty list of instance name strings'))
            continue
        if not structural_sub_blocks:
            errors.append((name, f"sub_block={tag!r} but this topology declares no structural_sub_blocks"))
            continue
        unknown = [i for i in tag if i not in structural_sub_blocks]
        if unknown:
            errors.append((name, f"sub_block references {unknown!r}, not a declared structural_sub_blocks instance"))
    return errors


def check(param_defs, sch_text, derived_cfg=None, sub_blocks=None, config=None, structural_sub_blocks=None, generator_backed=False):
    """(orphans, missing, schema_errors) -- orphans/missing are sorted name
    lists, schema_errors is check() -> [(name, message), ...]. derived_cfg is
    a topology's `derived_parameters` (see derived_names/referenced_in_derived
    above for how it reshapes the orphan/missing checks) -- omit for a
    topology with none. sub_blocks is a hierarchical topology's own
    `sub_blocks` (see derived_errors()/block_ref_errors()) -- omit for a
    non-hierarchical one. config is the full config.json dict, used only to
    validate a block_ref's own block/topology actually exist -- omit to
    skip that one check. structural_sub_blocks is the topology's own
    `structural_sub_blocks` (see structural_sub_block_errors()) -- omit for
    a topology with none. generator_backed (True when block_cfg declares a
    "generator", e.g. inductor.spiral) exempts EVERY declared parameter from
    the orphan check -- a generator-backed topology's own `parameters` are
    free GEOMETRIC inputs consumed by the generator module's own Python code
    (geometry_from_params()), never literal '<name>' schematic tokens
    themselves (only its `derived_parameters.generator` names are -- see
    analog_designer.sim.run_sim.resolve_generator_params()), same reasoning
    as block_ref_names()'s own exemption for "type": "block_ref" params, just
    applying to the whole set at once instead of a per-parameter type tag."""
    used = used_params(sch_text)
    declared = set(param_defs)
    dnames = derived_names(derived_cfg)
    referenced = referenced_in_derived(derived_cfg)
    block_refs = block_ref_names(param_defs)
    generator_inputs = declared if generator_backed else set()
    orphans = sorted((declared - used) - referenced - block_refs - generator_inputs)
    missing = sorted((used - declared) - dnames)
    errors = (
        schema_errors(param_defs)
        + derived_errors(param_defs, derived_cfg, used, sub_blocks)
        + block_ref_errors(param_defs, sub_blocks, config)
        + structural_sub_block_errors(param_defs, structural_sub_blocks)
    )
    return orphans, missing, errors


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--project-root", default=None, help="project folder to operate on; defaults to the last-opened folder, else CWD")
    parser.add_argument("--block", default=None, help="block to operate on; defaults to the first declared in config.json")
    parser.add_argument("--topology", default=None, help="topology to operate on; defaults to the first declared for --block")
    args = parser.parse_args()

    workspace.open_folder(args.project_root, block=args.block, topology=args.topology)
    block_cfg = workspace.CONFIG["blocks"][workspace.BLOCK]["topologies"][workspace.TOPOLOGY]
    param_defs = block_cfg["parameters"]
    derived_cfg = block_cfg.get("derived_parameters", {})
    sub_blocks = block_cfg.get("sub_blocks")
    structural_sub_blocks = block_cfg.get("structural_sub_blocks")
    sch_text = (workspace.PROJECT_ROOT / "sch" / block_cfg["schematic"]).read_text(encoding="utf-8")

    orphans, missing, errors = check(
        param_defs, sch_text, derived_cfg, sub_blocks, workspace.CONFIG, structural_sub_blocks,
        generator_backed="generator" in block_cfg,
    )

    print(f"{workspace.BLOCK}/{workspace.TOPOLOGY}: {len(param_defs)} declared parameter(s), "
          f"{len(used_params(sch_text))} referenced in {block_cfg['schematic']}")

    if orphans:
        print(f"\ndeclared in config.json but not used in the schematic ({len(orphans)}) -- safe to delete:")
        for name in orphans:
            print(f"  - {name}")
    if missing:
        print(f"\nused in the schematic but not declared in config.json ({len(missing)}) -- run_sim.py will fail to netlist until these are added:")
        for name in missing:
            print(f"  - {name}")
    if errors:
        print(f"\nschema problems in declared parameters / derived_parameters ({len(errors)}):")
        for name, message in errors:
            print(f"  - {name}: {message}")

    if not (orphans or missing or errors):
        print("\nOK -- schematic and config.json agree, no schema problems found.")
        return

    sys.exit(1)


if __name__ == "__main__":
    main()
