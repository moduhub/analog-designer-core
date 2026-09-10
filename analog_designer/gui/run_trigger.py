"""Runs an arbitrary python script as a subprocess on a background thread and
streams its output into a queue the main window polls. Deliberately shells
out instead of importing the target script -- run_sim.py/gen_variations.py/
manual_variation.py's functions are entangled with docker/argparse/sys.exit
side effects, so a subprocess boundary is the only clean way to reuse them
from the GUI. One shared instance runs whichever job is active -- they're
mutually exclusive anyway (same materialized schematic, same JSONL logs,
one docker container).

TOOL_ROOT is where this GUI application (the analog_designer/ package) is installed
-- NOT the project folder being worked on (that's analog_designer.core.workspace.PROJECT_ROOT,
resolved separately per job by each target script's own --project-root
handling). Every target script does `from analog_designer... import ...`, so all of them
are launched with `-m` from TOOL_ROOT -- a direct file-path invocation can't
resolve the `analog_designer` package."""
import os
import queue
import subprocess
import sys
import threading
from pathlib import Path

from analog_designer.gui.sleep_guard import allow_sleep, prevent_sleep

TOOL_ROOT = Path(__file__).resolve().parent.parent.parent
RUN_SIM_MODULE = "analog_designer.sim.run_sim"
MANUAL_VARIATION_MODULE = "analog_designer.sim.manual_variation"


class RunTrigger:
    def __init__(self, on_line, on_done):
        self.on_line = on_line
        self.on_done = on_done
        self._queue = queue.Queue()
        self._process = None
        self._thread = None
        self._container_id = None

    @property
    def running(self):
        return self._thread is not None and self._thread.is_alive()

    def start(self, argv):
        """argv: full command list, e.g. [sys.executable, "-m", RUN_SIM_MODULE, "--force"]."""
        if self.running:
            return
        self._container_id = None
        self._thread = threading.Thread(target=self._run, args=(argv,), daemon=True)
        self._thread.start()

    def cancel(self):
        """Best-effort stop of BOTH halves of a running job: the local
        Python subprocess (terminate(), as before) AND whichever docker
        container it started (see managed_container()'s own "@PROGRESS
        CONTAINER <id>" line, captured in _run() below) -- terminate() alone
        leaves a stuck ngspice run (e.g. a non-convergent testbench) running
        inside that container indefinitely, since a forcibly killed Windows
        process (TerminateProcess, what Popen.terminate() actually does
        there) never gets to run managed_container()'s own `finally: docker
        stop` cleanup, unlike a graceful exit. `docker stop` on an id that's
        already gone (job finished naturally right as cancel was clicked)
        just fails harmlessly -- output is discarded, not checked."""
        if self._container_id is not None:
            subprocess.run(["docker", "stop", self._container_id], capture_output=True, text=True)
        if self._process is not None and self._process.poll() is None:
            self._process.terminate()

    def _run(self, args):
        # bufsize=1 below only line-buffers *our* end of the pipe -- the
        # child process's own stdout defaults to fully block-buffered
        # (typically ~8KB) whenever it isn't a tty, which is exactly what a
        # redirected pipe is. Without PYTHONUNBUFFERED, every print() in
        # run_sim.py/manual_variation.py/etc. sits in the child's internal
        # buffer until it fills or the process exits, so the console looks
        # frozen and then dumps everything at once at the end -- not a
        # polling/queue issue on this side at all.
        # ANALOG_DESIGNER_PROGRESS=1 turns on run_sim.py's "@PROGRESS ..."
        # lines (see its emit_progress_total/emit_progress_step) -- only for
        # GUI-launched jobs, so a bare CLI invocation's output is unchanged.
        env = dict(os.environ, PYTHONUNBUFFERED="1", ANALOG_DESIGNER_PROGRESS="1")
        prevent_sleep()
        try:
            self._process = subprocess.Popen(
                args, cwd=str(TOOL_ROOT), stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, bufsize=1, env=env,
            )
            for line in self._process.stdout:
                line = line.rstrip("\n")
                # Consumed here, never forwarded to on_line/the console --
                # this is purely so cancel() (above) knows which container
                # to `docker stop`, not something a user needs to see.
                if line.startswith("@PROGRESS CONTAINER "):
                    self._container_id = line[len("@PROGRESS CONTAINER "):]
                    continue
                self._queue.put(line)
            returncode = self._process.wait()
        finally:
            allow_sleep()
        self._queue.put(None)  # sentinel: process finished
        self._queue.put(returncode)

    def poll(self):
        """Call periodically (e.g. via Tk.after) from the main thread. Drains
        available output lines and detects completion."""
        finished = False
        returncode = None
        try:
            while True:
                item = self._queue.get_nowait()
                if item is None:
                    finished = True
                    returncode = self._queue.get_nowait()
                    continue
                if finished:
                    continue
                self.on_line(item)
        except queue.Empty:
            pass
        if finished:
            self._process = None
            self._thread = None
            self.on_done(returncode)
