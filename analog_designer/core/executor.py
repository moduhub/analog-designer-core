"""Where simulator commands run: a fresh docker container of the project's
image (DockerExecutor, the original behavior) or directly on this machine
(HostExecutor -- e.g. when this tool, or the standalone runner exported from
it, runs inside the EDA image itself). Both take the same bash script
strings run_sim.py builds (`cd "<dir>" && ngspice -b ...`) and return a
subprocess.CompletedProcess, so every runner in run_sim.py is written once
for both. Paths inside those scripts come from workspace.exec_path(), the
one place host->container path mapping happens.

Pick one with run_sim.managed_executor(), which follows
workspace.execution_mode() (settings.json execution.mode, overridable per
process)."""
import os
import signal
import subprocess
import threading
from pathlib import Path

# Process groups of the host scripts running right now -- see
# kill_host_jobs_on_terminate().
_LIVE_GROUPS = set()
_LIVE_LOCK = threading.Lock()


def _kill_live_groups():
    with _LIVE_LOCK:
        groups = list(_LIVE_GROUPS)
    for pgid in groups:
        try:
            os.killpg(pgid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass


def kill_host_jobs_on_terminate():
    """Each host script runs in its own process group (so its timeout can
    kill the simulator it started), which also means killing this Python
    process alone -- the GUI's Cancel sends it SIGTERM -- would leave those
    simulators running. Docker mode has `docker stop` for this; here a
    SIGTERM handler kills every live group first. Main thread, POSIX only."""
    if os.name != "posix" or threading.current_thread() is not threading.main_thread():
        return
    previous = signal.getsignal(signal.SIGTERM)
    if getattr(previous, "_kills_host_jobs", False):
        return

    def handler(signum, frame):
        _kill_live_groups()
        signal.signal(signal.SIGTERM, signal.SIG_DFL)
        os.kill(os.getpid(), signal.SIGTERM)
    handler._kills_host_jobs = True
    signal.signal(signal.SIGTERM, handler)


def _timed_out(cmd, exc, timeout):
    """docker_exec()'s long-standing timeout contract: a stuck command
    becomes an ordinary CompletedProcess with the conventional 124 exit
    code, never an exception that would tear down a whole parallel batch."""
    out = exc.stdout or ""
    err = exc.stderr or ""
    if isinstance(out, bytes):
        out = out.decode(errors="replace")
    if isinstance(err, bytes):
        err = err.decode(errors="replace")
    return subprocess.CompletedProcess(cmd, 124, out, err + f"\n[command timed out after {timeout}s]")


class DockerExecutor:
    """Runs scripts in an already-started container (see
    run_sim.managed_executor()) via `docker exec ... bash -lc`, as the
    image's own user unless root=True. The image's Xvnc serves DISPLAY :1."""

    kind = "docker"
    display = ":1"

    def __init__(self, container_id):
        self.container_id = container_id

    def __str__(self):
        return self.container_id

    def run(self, script, timeout=120, root=False):
        cmd = ["docker", "exec"] + (["-u", "root"] if root else []) + [self.container_id, "bash", "-lc", script]
        try:
            return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        except subprocess.TimeoutExpired as exc:
            # Only kills the local `docker exec` client: the scripts
            # themselves wrap simulators in `timeout` so they die first.
            return _timed_out(cmd, exc, timeout)


class HostExecutor:
    """Runs scripts with this machine's own bash, in the environment this
    process has plus `env` (PDK_ROOT/PDK and tool overrides, see
    run_sim.managed_executor()). Each script gets its own process group so a
    timeout kills the simulator it started too, not just the shell. Needs a
    POSIX bash (Linux, macOS, WSL, or the EDA image itself)."""

    kind = "host"

    def __init__(self, env=None):
        self.env = dict(os.environ, **(env or {}))
        # xschem only netlists here (-x), which recent versions do without
        # an X server; reuse one if this machine has it (the EDA image's own
        # Xvnc on :1, say) for older versions that want one.
        self.display = self.env.get("DISPLAY") or None
        if not self.display:
            sockets = sorted(Path("/tmp/.X11-unix").glob("X*")) if Path("/tmp/.X11-unix").is_dir() else []
            self.display = ":" + sockets[0].name[1:] if sockets else None

    def __str__(self):
        return "host"

    def run(self, script, timeout=120, root=False):
        cmd = ["bash", "-c", script]
        posix = os.name == "posix"
        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=self.env,
            start_new_session=posix,
        )
        if posix:
            with _LIVE_LOCK:
                _LIVE_GROUPS.add(proc.pid)
        try:
            out, err = proc.communicate(timeout=timeout)
        except subprocess.TimeoutExpired as exc:
            if posix:
                os.killpg(proc.pid, signal.SIGKILL)
            else:
                proc.kill()
            out, err = proc.communicate()
            exc.stdout, exc.stderr = out, err
            return _timed_out(cmd, exc, timeout)
        finally:
            if posix:
                with _LIVE_LOCK:
                    _LIVE_GROUPS.discard(proc.pid)
        return subprocess.CompletedProcess(cmd, proc.returncode, out, err)
