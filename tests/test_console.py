"""analog_designer/core/console.atomic_print: lines printed concurrently
from worker threads must never merge -- the GUI parses "@PROGRESS ..."
lines out of a job's stdout and loses any that don't start a line."""
import subprocess
import sys
import textwrap
import unittest
from pathlib import Path


class AtomicPrintTests(unittest.TestCase):
    def test_concurrent_progress_lines_stay_whole(self):
        code = textwrap.dedent("""
            import os, threading
            os.environ["ANALOG_DESIGNER_PROGRESS"] = "1"
            from analog_designer.sim import run_sim
            def work(i):
                for _ in range(1000):
                    run_sim.emit_progress_step(f"v{i}", "t", True)
                    run_sim.print("some log line from", i)  # run_sim's own (atomic) print
            threads = [threading.Thread(target=work, args=(i,)) for i in range(8)]
            [t.start() for t in threads]
            [t.join() for t in threads]
        """)
        out = subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True,
            cwd=Path(__file__).resolve().parents[1], check=True,
        ).stdout.splitlines()
        self.assertEqual(len(out), 16000)
        progress = [line for line in out if "@PROGRESS" in line]
        self.assertEqual(len(progress), 8000)
        self.assertTrue(all(line.startswith("@PROGRESS STEP ") and line.count("@") == 1 for line in progress))


if __name__ == "__main__":
    unittest.main()
