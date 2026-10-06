"""Unit tests for the model-extraction orchestrator -- pure Python, no docker/simulator needed.

Everything here runs against a STUB generator that has nothing to do with inductors or EM, which
is the point: the orchestrator must not know what a setup means, only that a generator declares
some and can combine their results.
"""
import json
import tempfile
import unittest
from pathlib import Path

from analog_designer.core import workspace
from analog_designer.sim import extract_model
from analog_designer.sim.run_sim import generator_cache_key


class _StubGenerator:
    """A generator whose 'device' is modelled from three bias points. No geometry, no mesh."""

    SETUPS = ("lo", "mid", "hi")

    @staticmethod
    def geometry_from_params(params):
        # Accepts the retired key spelling, like a generator mid-rename would.
        setup = params.get("setup", params.get("variant", "mid"))
        return {"w": float(params["w"]), "setup": str(setup)}

    @staticmethod
    def extraction_plan(params):
        base = {k: v for k, v in params.items() if k not in ("setup", "variant")}
        return [{"name": n, "params": dict(base, setup=n),
                 "required": n != "hi", "note": "watch out" if n == "hi" else None}
                for n in _StubGenerator.SETUPS]

    @staticmethod
    def combine_extraction(params, runs):
        missing = [s for s in ("lo", "mid") if s not in runs]
        if missing:
            raise ValueError(f"missing {missing}")
        return {"params": {"c": len(runs)}, "provenance": {"setups": sorted(runs)}}

    @staticmethod
    def installed_model(params):
        return None


class _TempProject(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)
        saved_root, saved_cfg = workspace.PROJECT_ROOT, getattr(workspace, "CONFIG", None)
        self.addCleanup(lambda: setattr(workspace, "PROJECT_ROOT", saved_root))
        self.addCleanup(lambda: setattr(workspace, "CONFIG", saved_cfg))
        workspace.PROJECT_ROOT = self.root
        (self.root / "sim").mkdir()
        self.cache_dir = self.root / "sim" / "_generator_cache" / "blk" / "topo"
        self.cache_dir.mkdir(parents=True)
        self.gen = _StubGenerator()

    def _write_cache(self, params):
        """Writes a cached result keyed exactly the way run_one_openems would."""
        resolved = self.gen.geometry_from_params(params)
        key = generator_cache_key(resolved)
        (self.cache_dir / f"{key}.json").write_text(
            json.dumps({"geometry": resolved, "q": [1.0]}), encoding="utf-8")
        return key

    def _write_legacy_cache(self, params):
        """A result cached BEFORE the key was renamed: its stored parameters still say 'variant',
        and its filename is the hash of that older dict."""
        legacy = {"w": float(params["w"]), "variant": params["variant"]}
        key = generator_cache_key(legacy)
        (self.cache_dir / f"{key}.json").write_text(
            json.dumps({"geometry": legacy, "q": [1.0]}), encoding="utf-8")
        return key


class PlanStatusTests(_TempProject):
    def test_nothing_run_yet(self):
        status = extract_model._plan_status(self.gen, {"w": "2"}, self.cache_dir)
        self.assertEqual([e["name"] for e, _r, _c in status], ["lo", "mid", "hi"])
        self.assertTrue(all(cache is None for _e, _r, cache in status))

    def test_a_cached_setup_is_detected(self):
        self._write_cache({"w": "2", "setup": "mid"})
        status = extract_model._plan_status(self.gen, {"w": "2"}, self.cache_dir)
        found = {e["name"]: cache is not None for e, _r, cache in status}
        self.assertEqual(found, {"lo": False, "mid": True, "hi": False})

    def test_a_different_device_does_not_count_as_done(self):
        self._write_cache({"w": "2", "setup": "mid"})
        status = extract_model._plan_status(self.gen, {"w": "9"}, self.cache_dir)
        self.assertTrue(all(cache is None for _e, _r, cache in status),
                        "a run of another device must not satisfy this device's plan")


class LegacyCacheMigrationTests(_TempProject):
    """The expensive failure this guards against: renaming a generator parameter changes the hash
    the cache is keyed by, so every result already on disk becomes unreachable and the campaign
    re-runs it. For the reference inductor campaign that is 60+ hours."""

    def test_legacy_entry_is_rekeyed_and_then_found(self):
        old_key = self._write_legacy_cache({"w": "2", "variant": "mid"})
        new_key = generator_cache_key(self.gen.geometry_from_params({"w": "2", "setup": "mid"}))
        self.assertNotEqual(old_key, new_key, "test is pointless unless the rename moves the hash")

        before = extract_model._plan_status(self.gen, {"w": "2"}, self.cache_dir)
        self.assertTrue(all(cache is None for _e, _r, cache in before),
                        "precondition: the legacy entry is invisible before migration")

        moved = extract_model._migrate_legacy_cache_keys(self.gen, self.cache_dir)
        self.assertEqual(moved, 1)
        self.assertFalse((self.cache_dir / f"{old_key}.json").exists())
        self.assertTrue((self.cache_dir / f"{new_key}.json").exists())

        after = extract_model._plan_status(self.gen, {"w": "2"}, self.cache_dir)
        found = {e["name"]: cache is not None for e, _r, cache in after}
        self.assertEqual(found, {"lo": False, "mid": True, "hi": False})

    def test_migrated_entry_keeps_its_payload(self):
        self._write_legacy_cache({"w": "2", "variant": "lo"})
        extract_model._migrate_legacy_cache_keys(self.gen, self.cache_dir)
        doc = json.loads(next(self.cache_dir.glob("*.json")).read_text(encoding="utf-8"))
        self.assertEqual(doc["q"], [1.0], "the measured data must survive a re-key untouched")
        self.assertEqual(doc["geometry"]["setup"], "lo")
        self.assertNotIn("variant", doc["geometry"])

    def test_sibling_artifacts_follow_the_rename(self):
        old_key = self._write_legacy_cache({"w": "2", "variant": "lo"})
        (self.cache_dir / f"{old_key}__field.png").write_bytes(b"png")
        extract_model._migrate_legacy_cache_keys(self.gen, self.cache_dir)
        new_key = generator_cache_key(self.gen.geometry_from_params({"w": "2", "setup": "lo"}))
        self.assertTrue((self.cache_dir / f"{new_key}__field.png").exists())
        self.assertFalse((self.cache_dir / f"{old_key}__field.png").exists())

    def test_it_is_idempotent(self):
        self._write_legacy_cache({"w": "2", "variant": "lo"})
        self.assertEqual(extract_model._migrate_legacy_cache_keys(self.gen, self.cache_dir), 1)
        self.assertEqual(extract_model._migrate_legacy_cache_keys(self.gen, self.cache_dir), 0)

    def test_an_already_current_entry_is_left_alone(self):
        key = self._write_cache({"w": "2", "setup": "mid"})
        self.assertEqual(extract_model._migrate_legacy_cache_keys(self.gen, self.cache_dir), 0)
        self.assertTrue((self.cache_dir / f"{key}.json").exists())

    def test_an_unreadable_entry_does_not_abort_the_migration(self):
        (self.cache_dir / "junk.json").write_text("{not json", encoding="utf-8")
        self._write_legacy_cache({"w": "2", "variant": "lo"})
        self.assertEqual(extract_model._migrate_legacy_cache_keys(self.gen, self.cache_dir), 1)

    def test_missing_cache_dir_is_not_an_error(self):
        self.assertEqual(
            extract_model._migrate_legacy_cache_keys(self.gen, self.root / "sim" / "nope"), 0)


class CombineGatingTests(_TempProject):
    def test_required_setups_are_what_gates_combining(self):
        self._write_cache({"w": "2", "setup": "lo"})
        status = extract_model._plan_status(self.gen, {"w": "2"}, self.cache_dir)
        missing = [e["name"] for e, _r, cache in status if cache is None and e["required"]]
        self.assertEqual(missing, ["mid"])
        with self.assertRaises(ValueError):
            self.gen.combine_extraction({"w": "2"}, {"lo": "x"})

    def test_optional_setups_do_not_gate(self):
        self._write_cache({"w": "2", "setup": "lo"})
        self._write_cache({"w": "2", "setup": "mid"})
        status = extract_model._plan_status(self.gen, {"w": "2"}, self.cache_dir)
        missing = [e["name"] for e, _r, cache in status if cache is None and e["required"]]
        self.assertEqual(missing, [], "'hi' is optional and must not block a model")
        built = self.gen.combine_extraction({"w": "2"}, {"lo": "x", "mid": "y"})
        self.assertEqual(built["params"], {"c": 2})


if __name__ == "__main__":
    unittest.main()
