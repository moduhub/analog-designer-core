"""Pure computation behind the Variations table's "Declusterize" tool: find
near-duplicate parameter-space clusters and, within each, flag every
variation but the single best-scoring one as a candidate to trim --
collapsing redundant Monte Carlo/genetic-search samples that differ almost
nowhere in their free parameters and just add noise to the design space
without adding information.

This module deliberately departs from the most literal reading of "average
percent difference across parameters, judged pair by pair" in three ways,
each justified in its own function's docstring below:
  - distance is normalized by each PARAMETER's own declared [min, max]
    range, not by the pair's own values (variation_vector());
  - the per-parameter differences are aggregated as an RMS percentage, not
    a plain mean or sum (distance_matrix_pct());
  - "similar" is the transitive closure of pairwise similarity (union-find
    over the threshold graph), not just each pair judged in isolation, so
    a chain of near-duplicates collapses to ONE surviving sample instead of
    leaving an inconsistent tangle of pairwise flags (decluster()).

Vectorized with numpy throughout: even a few thousand variations means
hundreds of thousands of pairs, and the whole point of a decluttering tool
is to run instantly enough to re-try at a different threshold without
feeling like a batch job -- see distance_matrix_pct()'s own docstring for
why it's a single BLAS matrix multiply, not a nested Python loop."""
import numpy as np

from analog_designer.sim.spice_value import parse_spice_value


def distance_param_names(param_defs):
    """Which of param_defs' own free parameters are usable in a parameter-
    space distance calculation, in param_defs' own (dict) order -- excludes:
    - "block_ref" parameters: these select WHICH sub-block variation a
      hierarchical block references, not a continuous quantity (see
      gen_variations.random_params()'s own "not a numeric quantity"
      comment on the same distinction) -- a geometric distance between two
      such choices is meaningless (0 for "same sub-block variation", but
      undefined/arbitrary for "different" -- two different sub-block
      variations could themselves be near-identical or wildly different).
      A hierarchical block's own near-duplicates among ITS free parameters
      are still fully declustered; its sub-blocks' near-duplicates are a
      separate run of this same tool, one level down, scoped to that
      sub-block.
    - parameters locked to a single value (min == max, e.g. a design
      freedom deliberately deferred by pinning min=max=1) -- every valid
      sample already agrees on these by construction, so they carry zero
      information for telling samples apart, and normalizing by a
      zero-width range would divide by zero."""
    names = []
    for name, pdef in param_defs.items():
        if pdef.get("type") == "block_ref":
            continue
        lo, hi = parse_spice_value(pdef["min"]), parse_spice_value(pdef["max"])
        if hi > lo:
            names.append(name)
    return names


def variation_vector(parameters, param_defs, names):
    """`parameters` (a variation's own {name: "value string"} dict, from
    sim/variations.jsonl) as a numpy array of [0, 1] positions within each
    of `names`' own declared [min, max] range, in `names`' order -- e.g.
    0.5 means "exactly midway between this parameter's own min and max",
    regardless of its unit or magnitude, which is what makes every axis
    directly comparable in distance_matrix_pct() below (a 5u swing on a
    0.3u-40u width parameter is a very different "how different are these"
    signal than the same 5u swing on a 2u-15u one). A parameter declared
    after this variation was created (an additive config.json schema
    change -- see [[param_schema_migration_blast_radius]]) falls back to
    its own declared "default", the same convention
    resolve_derived_params()/substitute_params() already use for a value
    this specific variation never explicitly chose. Not clipped to [0, 1]
    afterward -- a value that somehow landed outside its declared range
    (hand-edited variations.jsonl, a since-tightened min/max) should still
    contribute its real, if unusual, distance, not a silently truncated
    one."""
    out = np.empty(len(names), dtype=float)
    for i, name in enumerate(names):
        pdef = param_defs[name]
        lo, hi = parse_spice_value(pdef["min"]), parse_spice_value(pdef["max"])
        raw = parameters.get(name, pdef["default"])
        out[i] = (parse_spice_value(str(raw)) - lo) / (hi - lo)
    return out


def distance_matrix_pct(vectors):
    """(n, n) symmetric matrix of RMS percent distance between every pair
    of rows of `vectors` (stacked variation_vector() results, shape
    (n, d)):
        100 * sqrt(mean_i((a_i - b_i)^2))
    RMS, not a plain mean, so a handful of very different parameters can't
    be "diluted" into looking similar by many identical ones: two
    variations differing on only 1 of 20 parameters, but differing on it
    by the FULL declared range, should not read as "95% similar" just
    because the other 19 match exactly -- a mean of nineteen 0s and one
    100 is 5%, the RMS is ~22%, which is much closer to how a designer
    would actually judge "these two spent their whole search budget on one
    knob". Not a plain sum either, so the result stays a bounded 0-100%
    reading regardless of how many parameters happen to be in play (a
    topology with 40 free parameters shouldn't automatically read as
    "less similar" than one with 10, all else equal).

    Computed via the ||a-b||^2 = ||a||^2 + ||b||^2 - 2 a.b identity (one
    (n, d) x (d, n) BLAS matrix multiply) rather than a naive elementwise
    (n, n, d) broadcast subtraction: for a few thousand variations across
    a few dozen parameters, the naive form is a multi-gigabyte
    intermediate array (and creates it TWICE, once for the difference and
    once for the square); this is O(n^2) memory, not O(n^2 * d), and the
    matrix multiply itself is orders of magnitude faster than the
    elementwise path for any n worth worrying about. sq_dist is clipped to
    >= 0 before the sqrt: the diagonal (and any near-duplicate pair) is
    mathematically exactly 0, but the identity above computes it as a
    difference of two nearly-equal floats, which can land a hair negative
    and NaN the sqrt."""
    if vectors.shape[0] == 0:
        return np.zeros((0, 0))
    d = vectors.shape[1]
    if d == 0:
        return np.zeros((vectors.shape[0], vectors.shape[0]))
    sq_norms = (vectors ** 2).sum(axis=1)
    sq_dist = sq_norms[:, None] + sq_norms[None, :] - 2.0 * (vectors @ vectors.T)
    np.clip(sq_dist, 0.0, None, out=sq_dist)
    return 100.0 * np.sqrt(sq_dist / d)


class _UnionFind:
    """Textbook union-find (path compression + union by rank) over the
    integers [0, n) -- the standard O(n * alpha(n)) way to turn a list of
    pairwise edges into connected components without an explicit graph
    traversal, used by decluster() to group a threshold-distance edge list
    into clusters."""

    def __init__(self, n):
        self._parent = list(range(n))
        self._rank = [0] * n

    def find(self, i):
        while self._parent[i] != i:
            self._parent[i] = self._parent[self._parent[i]]  # path halving
            i = self._parent[i]
        return i

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self._rank[ra] < self._rank[rb]:
            ra, rb = rb, ra
        self._parent[rb] = ra
        if self._rank[ra] == self._rank[rb]:
            self._rank[ra] += 1


def decluster(names, dist_pct, metric_values, threshold_pct):
    """Groups `names` (parallel to `dist_pct`'s rows/columns and to
    `metric_values`) into clusters wherever a CHAIN of pairwise distances
    <= threshold_pct connects them (union-find over the threshold graph's
    edges) -- the actual "declustering" this tool is named for. A chain
    A~B~C (A-B and B-C each within threshold, but not necessarily A-C)
    forms ONE cluster with ONE best-metric survivor, not two independently
    -judged pairs that could each nominate a different loser and leave the
    reader to work out which ones would actually still be near each other
    after only a pairwise trim.

    Within each cluster of 2+ members, every member except the highest-
    metric_values one becomes a candidate, paired against that cluster's
    OWN survivor -- the sample that actually stays after trimming, not
    just "whichever neighbor happens to be closest" -- since that's the
    comparison that matters for deciding whether to trim it.
    metric_values must already be oriented so higher = better (a "lower is
    better" metric's values negated before calling this -- see
    decluster_view.py's own use of a criterion's `descending` flag).

    A singleton cluster (nothing within threshold_pct of it, directly or
    transitively) produces no candidate -- there's nothing to declutter
    there, and it never appears in the result at all.

    Trimming a candidate never changes any SURVIVOR's own distance to any
    other survivor, so this result stays valid across any number of trims
    at the same threshold/metric -- no need to re-run after each one,
    only when the threshold or metric itself changes.

    Returns clusters sorted most-candidates-first:
        [{"kept": name, "kept_metric": float,
          "candidates": [{"name", "metric", "distance_pct"}, ...]}, ...]
    each cluster's candidates sorted worst-metric-first (the clearest
    eliminations at the top). O(n^2) here, scanning the whole distance
    matrix for edges -- already paid for by distance_matrix_pct(), and
    n^2 comparisons on an in-memory array is cheap; reuses that matrix
    rather than recomputing anything."""
    n = len(names)
    uf = _UnionFind(n)
    ii, jj = np.nonzero(np.triu(dist_pct <= threshold_pct, k=1))
    for i, j in zip(ii.tolist(), jj.tolist()):
        uf.union(i, j)

    groups = {}
    for i in range(n):
        groups.setdefault(uf.find(i), []).append(i)

    clusters = []
    for members in groups.values():
        if len(members) < 2:
            continue
        best = max(members, key=lambda i: metric_values[i])
        candidates = sorted(
            (
                {"name": names[i], "metric": metric_values[i], "distance_pct": float(dist_pct[i, best])}
                for i in members if i != best
            ),
            key=lambda c: c["metric"],
        )
        clusters.append({"kept": names[best], "kept_metric": metric_values[best], "candidates": candidates})
    clusters.sort(key=lambda c: len(c["candidates"]), reverse=True)
    return clusters
