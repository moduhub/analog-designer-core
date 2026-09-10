"""Keeps Windows from sleeping while a RunTrigger job is active.

Uses SetThreadExecutionState instead of a global power-plan change so the
machine can still sleep normally the rest of the time -- only the window
around an active run is covered. No-op on non-Windows platforms.
"""
import sys

if sys.platform == "win32":
    import ctypes

    _ES_CONTINUOUS = 0x80000000
    _ES_SYSTEM_REQUIRED = 0x00000001

    def prevent_sleep():
        ctypes.windll.kernel32.SetThreadExecutionState(_ES_CONTINUOUS | _ES_SYSTEM_REQUIRED)

    def allow_sleep():
        ctypes.windll.kernel32.SetThreadExecutionState(_ES_CONTINUOUS)
else:
    def prevent_sleep():
        pass

    def allow_sleep():
        pass
