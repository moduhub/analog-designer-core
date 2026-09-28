"""Standalone Parameters table for whichever variation is selected --
split out of VariationDetail (analog_designer/gui/variation_detail.py) so it
can sit below the Variations table (see analog_designer/gui/app.py's
left/right Panedwindow split) instead of stacked above Profiles/Metrics/
plots in the detail pane, freeing horizontal width for the plot view."""
import tkinter as tk
from tkinter import ttk

from analog_designer.results import data
from analog_designer.sim import run_sim

# "calc" is the leading equation-icon column (SolidWorks-style: a small
# marker next to a value that comes from a formula, not a free choice) --
# populated only for a row backed by a derived_parameters entry (width_groups/
# import_params/import_metrics/formulas), see _render_params(). "vary" is the sibling marker
# column driving on_vary (see __init__/_on_click) -- a free, non-block_ref
# parameter's own row gets a clickable pencil there, opening
# analog_designer.gui.vary_param_dialog scoped to just that parameter. Both
# stay leading/narrow (not appended after "description") so they're always
# visible without needing to scroll horizontally.
PARAM_COLUMNS = ("vary", "calc", "parameter", "value", "description")
_CALCULATED_MARKER = "ƒx"  # "ƒx" -- stands in for an equation-driven value, no icon asset needed
_VARY_MARKER = "✎"  # pencil -- "edit/vary this parameter", no icon asset needed, same convention as _CALCULATED_MARKER
_UNRESOLVED_VALUE = "—"  # em dash -- a calculated param whose dependency (e.g. a sub-block's own simulated metric) isn't available yet


class ParametersPanel(ttk.Frame):
    def __init__(self, master, on_vary=None, on_open_block_ref=None):
        super().__init__(master)
        self.on_vary = on_vary
        self.on_open_block_ref = on_open_block_ref
        # row iid -> (block, topology, variation_name) for every block_ref
        # row currently shown, whose value is a real registered variation
        # (not "defaults") -- see _render_params/_on_right_click. Rebuilt
        # from scratch on every show()/clear(), never mutated in place.
        self._block_refs = {}

        params_frame = ttk.LabelFrame(self, text="Parameters")
        params_frame.pack(fill="both", expand=True)
        self.params_tree = ttk.Treeview(
            params_frame, columns=PARAM_COLUMNS, show="headings", selectmode="browse",
        )
        for col in PARAM_COLUMNS:
            self.params_tree.heading(col, text=col)
            self.params_tree.column(col, width=110, anchor="w")
        self.params_tree.column("vary", width=32, anchor="center", stretch=False)
        self.params_tree.column("calc", width=32, anchor="center", stretch=False)
        self.params_tree.column("value", width=80)
        self.params_tree.column("description", width=420)
        params_vsb = ttk.Scrollbar(params_frame, orient="vertical", command=self.params_tree.yview)
        # description alone (420) plus the other 3 columns already add up to
        # more than this panel comfortably gets -- ttk.Treeview columns
        # never shrink below their own width to fit, so without this the
        # rightmost column(s) simply go invisible with no way back, even
        # maximized. See variations_table.py's own identical fix.
        params_hsb = ttk.Scrollbar(params_frame, orient="horizontal", command=self.params_tree.xview)
        self.params_tree.configure(yscrollcommand=params_vsb.set, xscrollcommand=params_hsb.set)
        self.params_tree.grid(row=0, column=0, sticky="nsew")
        params_vsb.grid(row=0, column=1, sticky="ns")
        params_hsb.grid(row=1, column=0, sticky="ew")
        params_frame.grid_rowconfigure(0, weight=1)
        params_frame.grid_columnconfigure(0, weight=1)
        # Blue, matching the equation-icon column -- "this value came from a
        # formula/derived_parameters entry, not a free choice".
        self.params_tree.tag_configure("calculated", foreground="#1a56db")
        # Green, distinct from the blue "calculated" marker -- "clickable
        # action", not "this value is derived".
        self.params_tree.tag_configure("varyable", foreground="#1a7f37")
        # Bound on the widget instance (checked before the Treeview class's
        # own <Button-1> binding, same reasoning as variations_table.py's
        # own _on_combine_click) so returning "break" here can suppress the
        # native click-to-select behavior when the click actually landed on
        # the vary marker -- a click there opens the dialog, it shouldn't
        # also just select the row.
        self.params_tree.bind("<Button-1>", self._on_click)
        self.params_tree.bind("<Button-3>", self._on_right_click)

    def _on_click(self, event):
        if self.on_vary is None:
            return None
        if self.params_tree.identify_region(event.x, event.y) != "cell":
            return None
        if self.params_tree.identify_column(event.x) != "#1":  # "vary" is PARAM_COLUMNS[0]
            return None
        row = self.params_tree.identify_row(event.y)
        if not row or "varyable" not in self.params_tree.item(row, "tags"):
            return None
        self.on_vary(self.params_tree.set(row, "parameter"))
        return "break"

    def _on_right_click(self, event):
        """Right-click on a block_ref row (e.g. "top"'s X1_variation/
        x2_variation) jumps straight to that sub-block variation's own
        results -- switching Block/Topology and selecting it there -- via
        on_open_block_ref(block, topology, variation_name), same wiring
        convention as on_vary. Only offered when this row's current value is
        a real registered variation: "defaults" (the reserved value meaning
        "config.json defaults, nothing materialized") has no results to jump
        to, so it's silently excluded from self._block_refs by
        _render_params rather than shown here with a dead menu entry."""
        if self.on_open_block_ref is None:
            return None
        row = self.params_tree.identify_row(event.y)
        ref = self._block_refs.get(row)
        if ref is None:
            return None
        self.params_tree.selection_set(row)
        block, topology, variation_name = ref
        menu = tk.Menu(self, tearoff=0)
        menu.add_command(
            label=f"Go to {variation_name} ({block}/{topology})",
            command=lambda: self.on_open_block_ref(block, topology, variation_name),
        )
        menu.tk_popup(event.x_root, event.y_root)
        return None

    def clear(self):
        self.params_tree.delete(*self.params_tree.get_children())
        self._block_refs = {}

    def show(self, variation_name):
        variations = {v["name"]: v for v in data.load_variations(all_topologies=True)}
        variation = variations.get(variation_name)
        config = data.load_config()
        block_cfg = config.get("blocks", {}).get(variation["block"], {}) if variation else {}
        topology_cfg = block_cfg.get("topologies", {}).get(variation["topology"], {}) if variation else {}
        self._render_params(variation["parameters"] if variation else {}, topology_cfg)

    def _render_params(self, free_params, topology_cfg):
        """Free parameters (this variation's own stored choices) followed by
        every CALCULATED one this topology declares (width_groups/
        import_params/import_metrics/formulas -- see
        run_sim.resolve_display_params()), marked with an equation icon so
        it reads clearly as "not a free choice" (SolidWorks marks an
        equation-driven dimension the same way). A calculated value
        resolve_display_params() couldn't compute yet (e.g. an
        import_metrics entry whose sub-block hasn't been simulated) still
        gets its own row -- name, icon and description --
        just with a placeholder instead of a number, so the panel explains
        what's missing instead of silently omitting it.

        Every free parameter also gets the "varyable" tag/marker (see
        __init__'s _on_click) UNLESS it's a "block_ref" (a sub-block
        variation name, not a number -- no [min,max] range, no slider/sweep
        makes sense) -- analog_designer.gui.vary_param_dialog is scoped to
        one plain numeric parameter at a time. A block_ref row instead gets
        registered in self._block_refs (see _on_right_click) when its value
        is a real variation name, not the reserved "defaults"."""
        self.params_tree.delete(*self.params_tree.get_children())
        self._block_refs = {}
        param_defs = topology_cfg.get("parameters", {})
        for name, value in free_params.items():
            pdef = param_defs.get(name, {})
            is_block_ref = pdef.get("type") == "block_ref"
            varyable = not is_block_ref
            row = self.params_tree.insert(
                "", "end", values=(_VARY_MARKER if varyable else "", "", name, value, pdef.get("description", "")),
                tags=("varyable",) if varyable else (),
            )
            if is_block_ref and value != "defaults":
                self._block_refs[row] = (pdef["block"], pdef["topology"], value)

        calc_descriptions = run_sim.calculated_param_descriptions(topology_cfg)
        if not calc_descriptions:
            return
        resolved = run_sim.resolve_display_params(topology_cfg, free_params)
        for name, description in calc_descriptions.items():
            value = resolved.get(name, _UNRESOLVED_VALUE)
            self.params_tree.insert(
                "", "end", values=("", _CALCULATED_MARKER, name, value, description), tags=("calculated",),
            )
