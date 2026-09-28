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


def add_skip_fail_tolerance_fields(parent, max_failures, discard_on_fail, row=2, allow_discard=True):
    """Grids the two tolerance knobs for whichever --skip-on-fail profile
    add_force_skip_fields() already put in `parent` -- a SEPARATE function
    (not folded into add_force_skip_fields() itself) so every dialog that
    already embeds that one keeps working completely unchanged; only a
    caller that actually wants these (today: create_variation_dialog's own
    Monte Carlo-generating "Create variation" dialog -- the search
    workflow these tolerances exist for, and now every other job-launching
    dialog too: Update's ask_run_confirm, pro's Generate/Auto Combine/Vary)
    calls this too, right after it,
    into the SAME options area. `row` defaults to 2 -- right below
    add_force_skip_fields()'s own force/skip-profile rows (0 and 1) when
    both are called back to back starting at the SAME row=0.

    Left always-visible/enabled regardless of whether a profile is
    currently selected (no dynamic greying-out) -- resolve_skip_fail_tolerance()
    at OK time already resets an unpaired value to (0, False) rather than
    trusting whatever the widgets still show, mirroring run_sim.
    validate_skip_on_fail_tolerance()'s own CLI-side requirement. Returns
    (max_failures_var, discard_var); discard_var is None with
    allow_discard=False -- Update's ask_run_confirm, where every variation
    already exists and discarding would delete earlier work, not just a
    candidate this job generated."""
    ttk.Label(parent, text="Tolerance:").grid(row=row, column=0, sticky="w", padx=(0, 6), pady=(4, 0))
    tolerance_row = ttk.Frame(parent)
    tolerance_row.grid(row=row, column=1, columnspan=2, sticky="w", pady=(4, 0))
    max_failures_var = tk.IntVar(value=max_failures)
    ttk.Spinbox(tolerance_row, from_=0, to=99, textvariable=max_failures_var, width=4).pack(side="left")
    ttk.Label(tolerance_row, text="failed constraint(s) allowed before stopping").pack(side="left", padx=(4, 0))
    if not allow_discard:
        return max_failures_var, None
    discard_var = tk.BooleanVar(value=discard_on_fail)
    ttk.Checkbutton(
        parent, text="Discard the variation entirely once exceeded (instead of just skipping the rest of its tests)",
        variable=discard_var,
    ).grid(row=row + 1, column=0, columnspan=3, sticky="w")
    return max_failures_var, discard_var


def resolve_skip_fail_tolerance(profile_var, max_failures_var, discard_var):
    """(max_failures: int, discard: bool) from add_skip_fail_tolerance_fields()'s
    own vars -- reset to (0, False) whenever no profile is actually chosen
    (resolve_skip_profile(profile_var) is None), regardless of whatever the
    spinbox/checkbox still show (e.g. left over from a previously-chosen
    profile) -- mirrors run_sim.validate_skip_on_fail_tolerance()'s own
    requirement, but resets silently here instead of refusing outright:
    this is a GUI convenience default, not a place to block Create/Update
    over a stale widget value the person never touched this time."""
    if resolve_skip_profile(profile_var) is None:
        return 0, False
    return max_failures_var.get(), (discard_var.get() if discard_var is not None else False)


def ask_run_confirm(parent, title, message, config, force, skip_on_fail_profile, skip_on_fail=True,
                    skip_on_fail_max_failures=0, discard_on_fail=False):
    # discard_on_fail is accepted (App._run_option_kwargs() passes it to
    # every dialog) but never shown -- see the docstring below.
    """Small confirmation Toplevel for a job-launching action with no
    dialog of its own (Update, Train) -- shows `message` plus the same
    force/skip fields every other action's dialog embeds (and, unless
    skip_on_fail=False, the tolerance field too -- never discard: these
    actions run on variations that already exist), pre-filled with the
    last-used values. {"force": bool, "skip_on_fail_profile": str|None,
    "skip_on_fail_max_failures": int} or None if cancelled, same
    convention as every ask_*() in this package. No "discard_on_fail" key,
    so App._apply_run_options() keeps the last Create/Generate choice."""
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
    max_failures_var = (
        add_skip_fail_tolerance_fields(body, skip_on_fail_max_failures, False, row=3, allow_discard=False)[0]
        if skip_on_fail else None
    )

    def on_ok():
        max_failures = (
            resolve_skip_fail_tolerance(profile_var, max_failures_var, None)[0] if max_failures_var else 0
        )
        result["value"] = {
            "force": force_var.get(), "skip_on_fail_profile": resolve_skip_profile(profile_var),
            "skip_on_fail_max_failures": max_failures,
        }
        win.destroy()

    def on_cancel():
        win.destroy()

    buttons = ttk.Frame(body)
    buttons.grid(row=5, column=0, columnspan=2, sticky="ew", pady=(12, 0))
    ttk.Button(buttons, text="Cancel", command=on_cancel).pack(side="right")
    ttk.Button(buttons, text="OK", command=on_ok).pack(side="right", padx=(0, 8))

    win.protocol("WM_DELETE_WINDOW", on_cancel)
    _center_on_parent(win, parent)
    win.grab_set()
    win.wait_window()
    return result.get("value")
