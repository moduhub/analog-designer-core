"""Thread-safe print for modules whose worker threads all write to one
stdout: the built-in print() writes the text and the line ending as
separate writes, so two threads' lines can merge into one (measured: with 8
threads about a quarter of the lines came out joined). The GUI reads a
job's stdout line by line and only recognizes an "@PROGRESS ..." line at
the start of a line (see analog_designer/gui/progress_tracker.py), so a
merged line silently dropped a progress step, and that condition was
painted as failed at the end. Modules that print from worker threads bind
`print = console.atomic_print` at the top."""
import sys
import threading

_LOCK = threading.Lock()


def atomic_print(*args, sep=" ", end="\n", file=None, flush=False):
    stream = sys.stdout if file is None else file
    if stream is None:
        return
    text = sep.join(str(a) for a in args) + end
    with _LOCK:
        stream.write(text)
        if flush:
            stream.flush()
