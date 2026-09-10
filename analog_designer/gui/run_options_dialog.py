"""Force/skip-on-fail fields, shared by every action that can launch a
simulation job (Update, Create, and -- in pro -- several more variation-
generating actions). Each of those embeds add_force_skip_fields() directly
into its own confirmation/options dialog (create_variation_dialog.
ask_create_variation, ask_run_confirm below, plus pro's own dialogs)
instead of a separate always-visible "Run Options..." toolbar button --
force/skip used to be set there as disconnected prior setup, which read as
a confusing two-step flow; now the choice is made at the moment it actually
matters, pre-filled with the last-used value. ask_run_confirm() is the
fallback Toplevel for actions (Update, and in pro Train) that don't already
have a dialog of their own to embed the fields into."""
import tkinter as tk
from tkinter import ttk

from analog_designer.core import workspace

_OFF = "(off)"


def _center_on_parent(win, parent):
    win.update_idletasks()
    x = parent.winfo_rootx() + (parent.winfo_width() - win.winfo_width()) // 2
    y = parent.winfo_rooty() + (parent.winfo_height() - win.winfo_height()) // 2
    win.geometry(f"+{max(x, 0)}+{max(y, 0)}")


def add_force_skip_fields(parent, config, force, skip_on_fail_profile, row=0, skip_on_fail=True):
    """Grids a Force checkbox (+ a Skip-on-fail profile picker, unless
    skip_on_fail=False -- e.g. Train, where skip-on-fail doesn't apply) into
    `parent` starting at `row`. Returns (force_var, profile_var); profile_var
    is None when skip_on_fail=False. Pass its result to resolve_skip_profile()
    at OK time to turn profile_var back into a plain profile name or None.
    `config`'s profiles are read fresh by the caller, every call -- the
    active block's own declared profile names can change between one dialog
    open and the next, so a name picked for a previous block must never
    silently carry over."""
    force_var = tk.BooleanVar(value=force)
    ttk.Checkbutton(
        parent, text="--force (re-run tests even if a fresh result already exists)", variable=force_var,
    ).grid(row=row, column=0, columnspan=2, sticky="w", pady=(0, 4))

    if not skip_on_fail:
        return force_var, None

    ttk.Label(parent, text="Skip on fail:").grid(row=row + 1, column=0, sticky="w", padx=(0, 6))
    profiles = sorted(config.get("blocks", {}).get(workspace.BLOCK, {}).get("profiles", {}))
    profile_var = tk.StringVar(value=skip_on_fail_profile if skip_on_fail_profile in profiles else _OFF)
    ttk.Combobox(
        parent, textvariable=profile_var, values=[_OFF] + profiles, width=18, state="readonly",
    ).grid(row=row + 1, column=1, sticky="w")
    return force_var, profile_var


def resolve_skip_profile(profile_var):
    """profile_var (from add_force_skip_fields) -> a profile name, or None
    for "(off)"/skip_on_fail=False."""
    if profile_var is None:
        return None
    chosen = profile_var.get()
    return None if chosen == _OFF else chosen


def ask_run_confirm(parent, title, message, config, force, skip_on_fail_profile, skip_on_fail=True):
    """Small confirmation Toplevel for a job-launching action with no
    dialog of its own (Update, Train) -- shows `message` plus the same
    force/skip fields every other action's dialog embeds, pre-filled with
    the last-used values. {"force": bool, "skip_on_fail_profile": str|None}
    or None if cancelled, same convention as every ask_*() in this package."""
    result = {}
    win = tk.Toplevel(parent)
    win.title(title)
    win.resizable(False, False)
    win.transient(parent)

    body = ttk.Frame(win, padding=12)
    body.pack(fill="both", expand=True)

    ttk.Label(body, text=message, wraplength=320, justify="left").grid(
        row=0, column=0, columnspan=2, sticky="w", pady=(0, 8),
    )
    force_var, profile_var = add_force_skip_fields(
        body, config, force, skip_on_fail_profile, row=1, skip_on_fail=skip_on_fail,
    )

    def on_ok():
        result["value"] = {"force": force_var.get(), "skip_on_fail_profile": resolve_skip_profile(profile_var)}
        win.destroy()

    def on_cancel():
        win.destroy()

    buttons = ttk.Frame(body)
    buttons.grid(row=3, column=0, columnspan=2, sticky="ew", pady=(12, 0))
    ttk.Button(buttons, text="Cancel", command=on_cancel).pack(side="right")
    ttk.Button(buttons, text="OK", command=on_ok).pack(side="right", padx=(0, 8))

    win.protocol("WM_DELETE_WINDOW", on_cancel)
    _center_on_parent(win, parent)
    win.grab_set()
    win.wait_window()
    return result.get("value")
