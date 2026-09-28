"""Figure-of-merit: classifies a variation into named design profiles (per
a block's config.json `blocks.<block>.profiles`) and evaluates each
profile's own FoM formula against the variation's metrics from
results.jsonl. Generic infrastructure -- no per-block/per-test logic lives
here, unlike the testbench parsers under tb/<block>/.

Tests report raw values only -- there is no absolute per-test pass/fail
spec (deliberately removed: it duplicated and sometimes conflicted with
profile-level judgment). A profile has its own `constraints` dict (metric
slug -> {"minimum"?, "maximum"?}) which is simultaneously the gate for
"does this variation qualify" and the source of that profile's own
`<slug>_min`/`<slug>_max` normalization terms -- e.g.:
    "pow(vref_core_current_consumption/vref_core_current_consumption_max, -1)"
evaluated with a restricted AST walker (no attribute access, no
subscripting, no arbitrary calls) -- config.json is trusted local input, but
there's no reason to run a full eval() over it.

Variables available to a formula:
  <slug>                -- each metric's TYPICAL value, keyed by
                           slugify(metric description) -- ABSENT for a
                           Monte Carlo metric (mismatch/global-process
                           sweep, see tb/_shared/parser_common.mc_stats()),
                           which has no single representative run; a
                           formula referencing that <slug> bare raises
                           "unknown variable" rather than silently
                           standing in the sample mean
  <slug>_observed_min    -- that metric's own observed minimum across every
                           condition it ran at (corners/PVT sweep, or MC seeds)
  <slug>_observed_max    -- that metric's own observed maximum, same sweep
  <slug>_mean            -- sample mean, only present for a Monte Carlo metric
  <slug>_std             -- sample standard deviation, only present for a
                           Monte Carlo metric
  <slug>_min             -- this profile's own constraint minimum, if it declared one
  <slug>_max             -- this profile's own constraint maximum, if it declared one
"""
import ast
import operator
import re

_BINOPS = {
    ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
    ast.Div: operator.truediv, ast.Pow: operator.pow, ast.Mod: operator.mod,
}
_UNARYOPS = {ast.UAdd: operator.pos, ast.USub: operator.neg}
_FUNCS = {"abs": abs, "min": min, "max": max, "round": round, "pow": pow}


def slugify(text):
    return re.sub(r"[^a-zA-Z0-9]+", "_", text.strip().lower()).strip("_")


def metrics_to_variables(metrics):
    """<slug>: the metric's TYPICAL value (what figure_of_merit ratio
    formulas like value/value_max are built around) -- OMITTED when typical
    is None (a Monte Carlo metric with no single representative run, see
    tb/_shared/parser_common.mc_stats()) rather than standing in the sample
    mean, so a formula that bare-references it fails loudly ("unknown
    variable") instead of silently treating an average as if it were a
    nominal design point. <slug>_mean/<slug>_std are added instead when the
    metric provides them, for a formula that explicitly wants to score
    against the distribution rather than a single value.
    <slug>_observed_min/_observed_max (always present) is that metric's own
    measured range across every condition it ran at. Deliberately NOT
    <slug>_min/_max, which already means "this profile's OWN constraint
    bound" (see classify()) -- those are profile-specific, since there's no
    absolute per-test spec to derive a single global bound from anymore."""
    variables = {}
    for m in metrics:
        slug = slugify(m["metric"])
        if m["typical"] is not None:
            variables[slug] = float(m["typical"])
        if m.get("mean") is not None:
            variables[f"{slug}_mean"] = float(m["mean"])
        if m.get("std") is not None:
            variables[f"{slug}_std"] = float(m["std"])
        variables[f"{slug}_observed_min"] = float(m["min"])
        variables[f"{slug}_observed_max"] = float(m["max"])
    return variables


def _eval_node(node, variables):
    if isinstance(node, ast.Expression):
        return _eval_node(node.body, variables)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.Name):
        if node.id not in variables:
            raise KeyError(f"unknown variable {node.id!r} (no metric produced this name for this variation)")
        return variables[node.id]
    if isinstance(node, ast.BinOp) and type(node.op) in _BINOPS:
        return _BINOPS[type(node.op)](_eval_node(node.left, variables), _eval_node(node.right, variables))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARYOPS:
        return _UNARYOPS[type(node.op)](_eval_node(node.operand, variables))
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in _FUNCS:
        return _FUNCS[node.func.id](*(_eval_node(a, variables) for a in node.args))
    raise ValueError(f"unsupported expression syntax: {ast.dump(node)}")


def safe_eval(expr, variables):
    return _eval_node(ast.parse(expr, mode="eval"), variables)


_FOM_SUFFIXES = ((1e12, "T"), (1e9, "B"), (1e6, "M"), (1e3, "K"))


def format_fom(value):
    """Financial-market-style magnitude formatting for a FOM value (a
    dimensionless product-of-ratios score that can span many orders of
    magnitude for non-physical Monte Carlo designs -- observed range on
    this project's own data: ~1e-12 to ~4e5): abs(value) >= 1000 groups
    into K/M/B/T suffixes (thousand/million/billion/trillion, 3 significant
    figures), same idiom as market cap/revenue figures. value < 1 collapses
    to "<1" -- FOMs down there aren't meaningfully differentiated from each
    other (a profile's ratio/weight formula is built around values near or
    above 1, by construction), so precision there isn't useful and
    scientific notation for it is actively confusing. Keeping the scale in
    a genuinely meaningful range (mostly single-digit-to-thousands) is a
    profile-tuning responsibility (constraint thresholds/weights), not
    something this formatter compensates for.

    A non-real value (pow() of a negative ratio by a fractional weight
    returns a Python complex, not an error -- classify() doesn't currently
    catch this) shows as "N/A" rather than crashing or printing a
    confusing "(a+bj)" string; this is a formatting fallback only, not a
    fix for that underlying case."""
    if isinstance(value, complex):
        return "N/A"
    if value < 1:
        return "<1"
    for threshold, suffix in _FOM_SUFFIXES:
        if value >= threshold:
            return f"{value / threshold:.3g}{suffix}"
    return f"{value:.4g}"


def constraint_satisfied(slug, bounds, variables):
    """Whether this metric (via metrics_to_variables()'s
    <slug>_observed_min/_observed_max) satisfies one {"minimum"?,
    "maximum"?} bound -- checked against the metric's OWN observed range,
    not just its typical reading, so a "maximum" bound must hold for the
    worst value ever seen across every condition, not only the nominal
    one, and likewise a "minimum" bound against the worst-case minimum.
    Public -- the GUI reuses this to show per-metric pass/fail relative to
    whichever profile is selected, since there's no absolute spec to check
    against anymore."""
    if f"{slug}_observed_min" not in variables:
        return False  # no data for this metric yet -- counts against it, not skipped
    if "minimum" in bounds and variables[f"{slug}_observed_min"] < bounds["minimum"]:
        return False
    if "maximum" in bounds and variables[f"{slug}_observed_max"] > bounds["maximum"]:
        return False
    return True


def constraints_violated(constraints, variables, max_failures=0):
    """True the moment MORE THAN `max_failures` of `constraints` (one
    profile's own {slug: bounds} dict) are ALREADY known to be violated,
    given only the metrics on hand so far -- the early-exit counterpart to
    classify()'s own constraint_satisfied(), with the OPPOSITE missing-
    metric polarity. constraint_satisfied()'s "no data -- counts against
    it" is correct only for classify()'s post-hoc, every-test-already-ran
    scoring; here, a constraint whose metric simply hasn't been simulated
    YET this run (a test still later in a still-in-progress variation)
    must NOT count as a violation -- treating it as one would trigger
    analog_designer.sim.run_sim's own --skip-on-fail after the very first
    test, every time, regardless of which profile was actually targeted. A
    constraint is judged only once its own metric is present in
    `variables` -- at which point it's judged exactly like classify()
    would.

    max_failures=0 (the original, default behavior) still stops at the
    FIRST known violation -- run_variation() only needs a yes/no to decide
    whether to keep simulating. A positive max_failures tolerates that many
    already-violated constraints before returning True, for a Monte Carlo
    search that wants to keep a design candidate whose profile isn't a
    perfect, unanimous pass across every one of its constraints (e.g. one
    corner slightly out of spec on an otherwise-good design) -- still O(1)
    extra work over the max_failures=0 case: this only ever counts up to
    max_failures+1 violations before returning, never tallies the rest."""
    violated = 0
    for slug, bounds in constraints.items():
        if f"{slug}_observed_min" not in variables:
            continue
        if not constraint_satisfied(slug, bounds, variables):
            violated += 1
            if violated > max_failures:
                return True
    return False


def classify(block_cfg, metrics):
    """block_cfg: config.json blocks.<block> dict (profiles lives as a
    sibling of "topologies", not inside it). metrics: latest_results()
    rows for one variation. Returns one entry per profile declared in
    block_cfg["profiles"], in config declaration order -- NOT filtered to
    full matches, so callers can see how close an unmatched profile got.
        [{"profile": name, "description": ..., "constraints": {...},
          "n_satisfied": int, "n_failed": int, "n_missing": int,
          "n_constraints": int, "matched": bool,
          "fom": float|None, "fom_error": str|None}, ...]
    A missing metric counts as an unsatisfied constraint (not skipped) --
    but NOT as failed: n_failed counts only constraints whose metric was
    measured and is out of bounds, n_missing the ones with no data yet
    (a test not run, or skipped by --skip-on-fail), so
    n_satisfied + n_failed + n_missing == n_constraints.
    fom is attempted regardless of match status; a formula referencing a
    metric this variation lacks (or that isn't one of this profile's own
    constraints, so has no _min/_max) surfaces as fom_error."""
    base_variables = metrics_to_variables(metrics)
    results = []
    for name, profile in block_cfg.get("profiles", {}).items():
        constraints = profile.get("constraints", {})
        n_satisfied = sum(
            1 for slug, bounds in constraints.items()
            if constraint_satisfied(slug, bounds, base_variables)
        )
        n_constraints = len(constraints)
        n_missing = sum(1 for slug in constraints if f"{slug}_observed_min" not in base_variables)
        n_failed = n_constraints - n_satisfied - n_missing

        variables = dict(base_variables)
        for slug, bounds in constraints.items():
            if "minimum" in bounds:
                variables[f"{slug}_min"] = bounds["minimum"]
            if "maximum" in bounds:
                variables[f"{slug}_max"] = bounds["maximum"]

        fom, fom_error = None, None
        formula = profile.get("figure_of_merit")
        if formula:
            try:
                fom = safe_eval(formula, variables)
            except Exception as exc:
                fom_error = str(exc)

        results.append({
            "profile": name,
            "description": profile.get("description", ""),
            "constraints": constraints,
            "n_satisfied": n_satisfied,
            "n_failed": n_failed,
            "n_missing": n_missing,
            "n_constraints": n_constraints,
            "matched": n_satisfied == n_constraints,
            "fom": fom,
            "fom_error": fom_error,
        })
    return results
