"""Unit tests for analog_designer/results/decluster.py -- pure numpy, no
docker/GUI/results.jsonl involved."""
import unittest

import numpy as np

from analog_designer.results.decluster import (
    decluster, distance_matrix_pct, distance_param_names, variation_vector,
)


def _pdef(lo, hi, default=None, integer=False, ptype=None):
    d = {"min": lo, "max": hi, "default": default if default is not None else lo}
    if integer:
        d["integer"] = True
    if ptype:
        d["type"] = ptype
    return d


class DistanceParamNamesTests(unittest.TestCase):
    def test_excludes_block_ref(self):
        defs = {"w": _pdef("1u", "2u"), "sub_variation": _pdef("0", "0", ptype="block_ref")}
        self.assertEqual(distance_param_names(defs), ["w"])

    def test_excludes_locked_min_equals_max(self):
        defs = {"w": _pdef("1u", "2u"), "m8_factor": _pdef("1", "1")}
        self.assertEqual(distance_param_names(defs), ["w"])

    def test_keeps_declaration_order(self):
        defs = {"b": _pdef("0", "1"), "a": _pdef("0", "1")}
        self.assertEqual(distance_param_names(defs), ["b", "a"])


class VariationVectorTests(unittest.TestCase):
    def test_midpoint_is_half(self):
        defs = {"w": _pdef("0u", "10u")}
        v = variation_vector({"w": "5u"}, defs, ["w"])
        self.assertAlmostEqual(v[0], 0.5)

    def test_endpoints_are_0_and_1(self):
        defs = {"w": _pdef("0u", "10u")}
        self.assertAlmostEqual(variation_vector({"w": "0u"}, defs, ["w"])[0], 0.0)
        self.assertAlmostEqual(variation_vector({"w": "10u"}, defs, ["w"])[0], 1.0)

    def test_missing_key_falls_back_to_declared_default(self):
        defs = {"w": _pdef("0u", "10u", default="5u")}
        v = variation_vector({}, defs, ["w"])
        self.assertAlmostEqual(v[0], 0.5)

    def test_out_of_range_value_is_not_clipped(self):
        """A hand-edited or since-tightened range shouldn't silently hide
        real distance behind a clip to [0, 1]."""
        defs = {"w": _pdef("0u", "10u")}
        v = variation_vector({"w": "20u"}, defs, ["w"])
        self.assertAlmostEqual(v[0], 2.0)


class DistanceMatrixPctTests(unittest.TestCase):
    def test_identical_vectors_are_zero_distance(self):
        vectors = np.array([[0.2, 0.5], [0.2, 0.5]])
        d = distance_matrix_pct(vectors)
        self.assertAlmostEqual(d[0, 1], 0.0)
        self.assertAlmostEqual(d[1, 0], 0.0)

    def test_matches_hand_computed_rms(self):
        # One of two axes differs by the full range (1.0), the other matches.
        vectors = np.array([[0.0, 0.5], [1.0, 0.5]])
        d = distance_matrix_pct(vectors)
        expected = 100.0 * ((1.0 ** 2 + 0.0 ** 2) / 2) ** 0.5
        self.assertAlmostEqual(d[0, 1], expected)

    def test_rms_is_not_diluted_by_many_identical_parameters(self):
        """19 identical axes + 1 maximally different one should read
        clearly above a plain mean (5%) -- RMS keeps the outlier visible."""
        a = np.zeros(20)
        b = np.zeros(20)
        b[0] = 1.0
        d = distance_matrix_pct(np.array([a, b]))
        self.assertGreater(d[0, 1], 20.0)  # well above the 5% a plain mean would give

    def test_symmetric_and_zero_diagonal(self):
        vectors = np.array([[0.1, 0.9], [0.8, 0.2], [0.5, 0.5]])
        d = distance_matrix_pct(vectors)
        np.testing.assert_allclose(d, d.T)
        np.testing.assert_allclose(np.diag(d), 0.0, atol=1e-9)

    def test_empty_input(self):
        self.assertEqual(distance_matrix_pct(np.zeros((0, 3))).shape, (0, 0))

    def test_no_usable_parameters(self):
        d = distance_matrix_pct(np.zeros((3, 0)))
        self.assertEqual(d.shape, (3, 3))
        np.testing.assert_allclose(d, 0.0)


class DeclusterTests(unittest.TestCase):
    def _matrix(self, pairs, n):
        d = np.full((n, n), 100.0)
        np.fill_diagonal(d, 0.0)
        for i, j, value in pairs:
            d[i, j] = d[j, i] = value
        return d

    def test_isolated_sample_produces_no_cluster(self):
        names = ["a", "b"]
        d = self._matrix([], 2)
        clusters = decluster(names, d, [1.0, 2.0], threshold_pct=5.0)
        self.assertEqual(clusters, [])

    def test_close_pair_flags_the_lower_metric_one(self):
        names = ["a", "b"]
        d = self._matrix([(0, 1, 2.0)], 2)
        clusters = decluster(names, d, [10.0, 5.0], threshold_pct=5.0)
        self.assertEqual(len(clusters), 1)
        self.assertEqual(clusters[0]["kept"], "a")
        self.assertEqual([c["name"] for c in clusters[0]["candidates"]], ["b"])
        self.assertAlmostEqual(clusters[0]["candidates"][0]["distance_pct"], 2.0)

    def test_transitive_chain_forms_one_cluster_not_two(self):
        """A-B and B-C close, A-C NOT close on its own -- still one cluster
        with one survivor, not two separately-judged pairs."""
        names = ["a", "b", "c"]
        d = self._matrix([(0, 1, 2.0), (1, 2, 2.0), (0, 2, 50.0)], 3)
        clusters = decluster(names, d, [5.0, 1.0, 10.0], threshold_pct=5.0)
        self.assertEqual(len(clusters), 1)
        self.assertEqual(clusters[0]["kept"], "c")
        self.assertEqual({c["name"] for c in clusters[0]["candidates"]}, {"a", "b"})

    def test_candidates_compared_against_cluster_survivor_not_nearest_neighbor(self):
        """b is closer to c than to the actual survivor a -- distance_pct
        must still be measured to a (who actually stays), not to c."""
        names = ["a", "b", "c"]
        d = self._matrix([(0, 1, 4.0), (1, 2, 1.0), (0, 2, 4.5)], 3)
        clusters = decluster(names, d, [10.0, 5.0, 1.0], threshold_pct=5.0)
        self.assertEqual(clusters[0]["kept"], "a")
        b = next(c for c in clusters[0]["candidates"] if c["name"] == "b")
        self.assertAlmostEqual(b["distance_pct"], 4.0)

    def test_candidates_sorted_worst_metric_first(self):
        names = ["a", "b", "c"]
        d = self._matrix([(0, 1, 1.0), (0, 2, 1.0)], 3)
        clusters = decluster(names, d, [10.0, 3.0, 1.0], threshold_pct=5.0)
        self.assertEqual([c["name"] for c in clusters[0]["candidates"]], ["c", "b"])

    def test_clusters_sorted_most_candidates_first(self):
        names = ["a", "b", "c", "d", "e"]
        d = self._matrix([(0, 1, 1.0), (2, 3, 1.0), (2, 4, 1.0)], 5)
        clusters = decluster(names, d, [1, 1, 3, 1, 1], threshold_pct=5.0)
        self.assertEqual(len(clusters[0]["candidates"]), 2)
        self.assertEqual(len(clusters[1]["candidates"]), 1)

    def test_threshold_exactly_at_distance_counts_as_similar(self):
        names = ["a", "b"]
        d = self._matrix([(0, 1, 5.0)], 2)
        clusters = decluster(names, d, [1.0, 2.0], threshold_pct=5.0)
        self.assertEqual(len(clusters), 1)


if __name__ == "__main__":
    unittest.main()
