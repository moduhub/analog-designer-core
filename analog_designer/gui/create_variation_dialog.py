"""Modal dialog for the toolbar's Create action in app.py: one Toplevel with
three tabs (Basic / From Parent / Monte Carlo), covering the core
distribution's ways to produce a new variation to simulate, plus the same
force/skip-on-fail fields every job-launching dialog in this package embeds
(see run_options_dialog.add_force_skip_fields) -- set at the moment of
creating, not as separate prior setup. Blocks (grab_set + wait_window)
until the user confirms or cancels, returning a dict tagged by "mode"
(plus "force"/"skip_on_fail_profile") or None on cancel -- app.py turns
that into a manual_variation.py/gen_variations.py argv and hands it to
RunTrigger.

Basic and From Parent share the exact same per-parameter grid
(_build_basic_tab) -- the only difference is where each field's initial
value comes from (config.json's own default vs. an existing "parent"
variation's own current values, re-picked via a combobox).

Pro's own copy of this module (analog_designer_pro.gui.create_variation_dialog)
adds a live "Predicted" panel below the notebook, backed by a trained
prediction model of its own -- omitted here since core ships no such
model-training or prediction code at all."""
import tkinter as tk
from tkinter import ttk
import random

from analog_designer.gui.run_options_dialog import (
    add_force_skip_fields, add_skip_fail_tolerance_fields, resolve_skip_fail_tolerance, resolve_skip_profile,
)
from analog_designer.results import data
from analog_designer.sim.run_sim import BLOCK_REF_DEFAULT

_TABS = ("basic", "from_parent", "monte_carlo")
_BASIC_COLUMNS = 3


def _center_on_parent(win, parent):
    win.update_idletasks()
    x = parent.winfo_rootx() + (parent.winfo_width() - win.winfo_width()) // 2
    y = parent.winfo_rooty() + (parent.winfo_height() - win.winfo_height()) // 2
    win.geometry(f"+{max(x, 0)}+{max(y, 0)}")


def _build_block_ref_row(tab, row, base_col, name, pdef, config, initial_values=None):
    """One "block_ref" parameter's row (see analog_designer.sim.run_sim.
    materialize_sub_blocks): a Combobox of every already-registered
    variation of that sub-block (plus the reserved "defaults" choice) --
    NOT a free-text Entry, a numeric value would never make sense here --
    plus a profile picker + "Sortear" button that draws ONE random matching
    variation (analog_designer.results.data.matching_variations()) and
    fills the Combobox with the result, so the user always sees the exact
    name that's about to be used, never a hidden pick. Returns the
    Combobox's own StringVar (what ends up in the "params" dict).
    initial_values, if given (a parent variation's own "parameters" dict --
    see the From Parent tab), seeds this field's starting value instead of
    the parameter's own config.json default."""
    block, topology = pdef["block"], pdef["topology"]
    choices = [BLOCK_REF_DEFAULT] + data.matching_variations(block, topology)
    default = (initial_values or {}).get(name, pdef.get("default", BLOCK_REF_DEFAULT))
    var = tk.StringVar(value=default)

    ttk.Label(tab, text=name).grid(
        row=row, column=base_col, sticky="w", padx=(0 if base_col == 0 else 16, 2), pady=2,
    )
    combo = ttk.Combobox(tab, textvariable=var, values=choices, width=24, state="readonly")
    combo.grid(row=row, column=base_col + 1, sticky="w", pady=2)

    profiles = sorted(config.get("blocks", {}).get(block, {}).get("profiles", {}))
    profile_var = tk.StringVar(value="any")
    profile_combo = ttk.Combobox(
        tab, textvariable=profile_var, values=["any"] + profiles, width=12, state="readonly",
    )
    profile_combo.grid(row=row, column=base_col + 2, sticky="w", padx=(6, 2), pady=2)

    def on_roll():
        profile_name = None if profile_var.get() == "any" else profile_var.get()
        candidates = data.matching_variations(block, topology, profile_name)
        if candidates:
            var.set(random.choice(candidates))

    ttk.Button(tab, text="Sortear", command=on_roll, width=8).grid(
        row=row, column=base_col + 3, sticky="w", padx=(2, 0), pady=2,
    )
    return var


def _build_basic_tab(tab, config, param_defs, initial_values=None):
    """One row per parameter, pre-filled with its config.json default (or,
    when initial_values is given -- a parent variation's own "parameters"
    dict, see the From Parent tab -- that value instead): an Entry (+ muted
    min-max hint) for a numeric parameter, a Combobox (+ profile-filtered
    random draw) for a "block_ref" one -- see _build_block_ref_row().
    Multi-column grid so many fields don't become one very tall single
    column. Returns {name: StringVar}."""
    names = list(param_defs)
    rows_per_col = -(-len(names) // _BASIC_COLUMNS)  # ceiling division
    vars_by_name = {}
    for i, name in enumerate(names):
        pdef = param_defs[name]
        col, row = divmod(i, rows_per_col)
        base_col = col * 4
        if pdef.get("type") == "block_ref":
            vars_by_name[name] = _build_block_ref_row(tab, row, base_col, name, pdef, config, initial_values=initial_values)
            continue
        default = (initial_values or {}).get(name, pdef["default"])
        var = tk.StringVar(value=default)
        vars_by_name[name] = var
        ttk.Label(tab, text=name).grid(
            row=row, column=base_col, sticky="w", padx=(0 if col == 0 else 16, 2), pady=2,
        )
        ttk.Entry(tab, textvariable=var, width=8).grid(row=row, column=base_col + 1, sticky="w", pady=2)
        ttk.Label(tab, text=f"({pdef['min']}-{pdef['max']})", foreground="#888888").grid(
            row=row, column=base_col + 2, sticky="w", padx=(2, 0), pady=2,
        )
    return vars_by_name


def _build_from_parent_tab(tab, config, param_defs, variation_names, variations_by_name, default_parent, on_change):
    """Same per-parameter grid as _build_basic_tab, pre-filled from a chosen
    "Parent:" variation's own current values instead of config.json
    defaults. Rebuilt (old field widgets destroyed, new ones built via
    _build_basic_tab) every time the parent changes -- there's no live
    rebind for an arbitrary new set of StringVars, so this just throws the
    grid away and remakes it. Returns a dict with a "vars" key kept current
    across rebuilds (read THIS, not a copy captured before a later rebuild)
    and "parent" (the parent-picker StringVar). on_change(vars_by_name)
    fires after every (re)build."""
    parent_default = default_parent if default_parent in variation_names else (variation_names[0] if variation_names else "")
    parent_var = tk.StringVar(value=parent_default)
    ttk.Label(tab, text="Parent:").grid(row=0, column=0, sticky="w")
    ttk.Combobox(
        tab, textvariable=parent_var, values=variation_names, state="readonly", width=28,
    ).grid(row=0, column=1, sticky="w")

    grid_frame = ttk.Frame(tab)
    grid_frame.grid(row=1, column=0, columnspan=4, sticky="nsew", pady=(8, 0))

    state = {"vars": {}, "parent": parent_var}

    def rebuild(*_args):
        for child in grid_frame.winfo_children():
            child.destroy()
        parent = variations_by_name.get(parent_var.get())
        initial_values = parent["parameters"] if parent else {}
        state["vars"] = _build_basic_tab(grid_frame, config, param_defs, initial_values=initial_values)
        on_change(state["vars"])

    parent_var.trace_add("write", rebuild)
    rebuild()
    return state


def _build_monte_carlo_tab(tab):
    n_var = tk.IntVar(value=5)
    mode_var = tk.StringVar(value="whole")
    spread_var = tk.DoubleVar(value=10.0)

    ttk.Label(tab, text="Count (N):").grid(row=0, column=0, sticky="w")
    ttk.Spinbox(tab, from_=1, to=200, textvariable=n_var, width=8).grid(row=0, column=1, sticky="w")

    ttk.Label(tab, text="Range:").grid(row=1, column=0, sticky="w", pady=(8, 0))
    mode_frame = ttk.Frame(tab)
    mode_frame.grid(row=1, column=1, sticky="w", pady=(8, 0))
    spread_entry = ttk.Entry(tab, textvariable=spread_var, width=8, state="disabled")

    def on_mode_change():
        spread_entry.configure(state="normal" if mode_var.get() == "spread" else "disabled")

    ttk.Radiobutton(
        mode_frame, text="Whole range", variable=mode_var, value="whole", command=on_mode_change,
    ).pack(side="left")
    ttk.Radiobutton(
        mode_frame, text="Defined spread", variable=mode_var, value="spread", command=on_mode_change,
    ).pack(side="left", padx=(8, 0))

    ttk.Label(tab, text="+/- % spread (if selected):").grid(row=2, column=0, sticky="w", pady=(4, 0))
    spread_entry.grid(row=2, column=1, sticky="w", pady=(4, 0))

    return {"count": n_var, "mode": mode_var, "spread": spread_var}


def ask_create_variation(
    parent, config, block, topology, variations, param_defs, force, skip_on_fail_profile,
    initial_tab="basic", default_parent=None, skip_on_fail_max_failures=0, discard_on_fail=False,
):
    """dict tagged by "mode" ("basic"|"from_parent"|"monte_carlo") with that
    mode's fields plus "force"/"skip_on_fail_profile"/
    "skip_on_fail_max_failures"/"discard_on_fail", or None if cancelled.
    config is the full config.json dict -- used both to look up
    a "block_ref" parameter's own declared profiles for its "Sortear"
    picker, and (via add_force_skip_fields) the block's skip-on-fail
    profiles. block/topology are accepted for interface parity with pro's
    own ask_create_variation (which uses them to resolve a trained model for
    its live prediction panel) but are otherwise unused here. variations is
    data.load_variations()'s own list, for the From Parent tab's picker.
    See module docstring."""
    result = {}
    win = tk.Toplevel(parent)
    win.title("Create variation")
    win.resizable(False, False)
    win.transient(parent)

    body = ttk.Frame(win, padding=12)
    body.pack(fill="both", expand=True)

    notebook = ttk.Notebook(body)
    notebook.pack(fill="both", expand=True)

    basic_tab = ttk.Frame(notebook, padding=8)
    from_parent_tab = ttk.Frame(notebook, padding=8)
    mc_tab = ttk.Frame(notebook, padding=8)
    notebook.add(basic_tab, text="Basic")
    notebook.add(from_parent_tab, text="From Parent")
    notebook.add(mc_tab, text="Monte Carlo")

    basic_vars = _build_basic_tab(basic_tab, config, param_defs)

    variation_names = [v["name"] for v in variations]
    variations_by_name = {v["name"]: v for v in variations}
    from_parent_state = _build_from_parent_tab(
        from_parent_tab, config, param_defs, variation_names, variations_by_name,
        default_parent=default_parent, on_change=lambda vars_by_name: None,
    )

    mc_vars = _build_monte_carlo_tab(mc_tab)

    options_frame = ttk.Frame(body)
    options_frame.pack(fill="x", pady=(12, 0))
    force_var, profile_var = add_force_skip_fields(options_frame, config, force, skip_on_fail_profile)
    max_failures_var, discard_var = add_skip_fail_tolerance_fields(
        options_frame, skip_on_fail_max_failures, discard_on_fail,
    )

    notebook.select(_TABS.index(initial_tab))

    def on_ok():
        tab = _TABS[notebook.index(notebook.select())]
        if tab == "basic":
            value = {"mode": "basic", "params": {n: v.get() for n, v in basic_vars.items()}}
        elif tab == "from_parent":
            value = {
                "mode": "from_parent", "base": from_parent_state["parent"].get(),
                "params": {n: v.get() for n, v in from_parent_state["vars"].items()},
            }
        else:
            spread = mc_vars["spread"].get() if mc_vars["mode"].get() == "spread" else None
            value = {"mode": "monte_carlo", "count": mc_vars["count"].get(), "spread": spread}
        value["force"] = force_var.get()
        value["skip_on_fail_profile"] = resolve_skip_profile(profile_var)
        value["skip_on_fail_max_failures"], value["discard_on_fail"] = resolve_skip_fail_tolerance(
            profile_var, max_failures_var, discard_var,
        )
        result["value"] = value
        win.destroy()

    def on_cancel():
        win.destroy()

    buttons = ttk.Frame(body)
    buttons.pack(fill="x", pady=(12, 0))
    ttk.Button(buttons, text="Cancel", command=on_cancel).pack(side="right")
    ttk.Button(buttons, text="Create", command=on_ok).pack(side="right", padx=(0, 8))

    win.protocol("WM_DELETE_WINDOW", on_cancel)
    _center_on_parent(win, parent)
    win.grab_set()
    win.wait_window()
    return result.get("value")
