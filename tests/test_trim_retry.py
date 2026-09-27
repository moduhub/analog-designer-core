"""A discard-on-fail checkpoint must survive another process holding
results.jsonl/variations.jsonl open (Windows refuses os.replace onto an
open file): run_sim._replace_with_retry() retries the swap, and
gen_variations._trim_all() hands a still-failing trim back for the next
checkpoint instead of raising out of the batch."""
import unittest
from unittest.mock import patch

from analog_designer.sim import gen_variations, run_sim


class ReplaceWithRetryTests(unittest.TestCase):
    def test_retries_until_the_file_is_released(self):
        calls = []

        def flaky_replace(src, dst):
            calls.append((src, dst))
            if len(calls) < 3:
                raise PermissionError(5, "Access is denied")

        with patch.object(run_sim.os, "replace", flaky_replace), patch.object(run_sim.time, "sleep"):
            run_sim._replace_with_retry("a.tmp", "a")
        self.assertEqual(len(calls), 3)

    def test_gives_up_after_the_timeout(self):
        def always_locked(src, dst):
            raise PermissionError(5, "Access is denied")

        with patch.object(run_sim.os, "replace", always_locked), patch.object(run_sim.time, "sleep"):
            with self.assertRaises(PermissionError):
                run_sim._replace_with_retry("a.tmp", "a", timeout_s=0)


class TrimAllTests(unittest.TestCase):
    def test_failed_trim_is_returned_for_a_later_retry(self):
        def trim(name):
            if name == "locked":
                raise PermissionError(5, "Access is denied")

        with patch.object(gen_variations, "trim_variation", trim):
            trimmed, failed = gen_variations._trim_all(["a", "locked", "b"])
        self.assertEqual(trimmed, ["a", "b"])
        self.assertEqual(failed, ["locked"])


if __name__ == "__main__":
    unittest.main()
