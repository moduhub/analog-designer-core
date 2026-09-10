"""Drill-down panel for a single variation: a selectable list of design
profiles it was scored against, a metrics table (separate from the profiles
list -- variations/profiles/tests are different granularities and don't
belong in one table), and the plot(s) for whichever metric row is selected
-- a parser can generate any number of named views for a test (0, 1, or
many, see analog_designer.results.data.plot_paths_for), shown here as
notebook tabs. Its own Parameters table lives in a separate sibling widget,
analog_designer.gui.parameters_panel.ParametersPanel -- split out so it can
sit below the Variations table instead of stacked above this panel's own
Profiles/Metrics/plots (see analog_designer/gui/app.py's left/right
Panedwindow split), freeing horizontal width for the plots.

Tests carry no absolute pass/fail spec anymore -- selecting a profile in
the Profiles list drives the metrics table's PASS/FAIL column, computed
relative to *that profile's own* constraints (analog_designer.results.fom.constraint_satisfied),
so the user can see which tests pass/fail for the profile they're looking
at. A metric that profile doesn't constrain shows a blank pass column, not
a judgment."""
import tkinter as tk
from tkinter import ttk

from analog_designer.results import fom
from analog_designer.results import data
from analog_designer.sim import run_sim
from analog_designer.gui import variations_table
from analog_designer.gui.zoom_image_view import ZoomableImageView

PROFILE_COLUMNS = ("profile", "score", "fom", "description")
METRIC_COLUMNS = ("test", "problems", "metric", "typical", "min", "max", "mean", "std", "unit", "duration", "pass", "stale")
# "problems" mirrors VariationsTable's own column (see
# analog_designer.gui.variations_table._problems_text) but per-TEST here
# instead of per-variation -- one data.diagnostics_for_variation(variation,
# test) call per distinct test name, not per metric row (several metric
# rows share the same test, same as "duration" above already does).
# "duration" is that TEST's own total simulate time (every one of its
# conditions, see run_sim.append_results's own duration_seconds docstring)
# -- the SAME value repeated on every metric row of one test, since
# duration is a per-test property but this table is one row per metric.
# Surfaced here (not just fed into the ETA estimate it was originally
# added for) so a variation already sitting in results.jsonl can be
# scanned for which of its own tests are the expensive ones, without
# having to re-run anything to find out.
def _format_sig(value, digits=4):
    """A metric's typical/min/max reading rounded to `digits` significant
    figures for the results table -- a raw simulator float (e.g.
    3235.6031249999996 um²) is noise past that precision for a skim, and
    "%g" already picks fixed vs. scientific notation sensibly across the
    huge magnitude range these metrics span (ppm/°C up to um²). None (e.g.
    a temperature-coefficient range with no run inside its bounds) shows
    blank rather than "None"."""
    if value is None:
        return ""
    return f"{value:.{digits}g}"


class VariationDetail(ttk.Frame):
    def __init__(self, master, problems_panel=None):
        """`problems_panel`: an analog_designer.gui.problems_panel.ProblemsPanel
        instance already built (and placed) elsewhere -- app.py hosts it as
        "Errors"/"Warnings" tabs sharing the same box as its own Console
        (see app.py's _build_console()), not as a section living here, so
        this panel only ever calls .show()/.clear() on it, never builds or
        packs it. None (e.g. a standalone smoke test) just skips those
        calls."""
        super().__init__(master)
        self.problems_panel = problems_panel
        self._variation_name = None
        self._variation_block = None
        self._metrics = []
        self._profiles = []  # last classify() result, sorted by compatibility

        header = ttk.Frame(self)
        header.pack(fill="x", pady=(0, 8))
        self.title_var = tk.StringVar(value="Select a variation to see details")
        ttk.Label(header, textvariable=self.title_var, font=("", 11, "bold")).pack(anchor="w")

        body = ttk.Frame(self)
        body.pack(fill="both", expand=True)

        profiles_frame = ttk.LabelFrame(body, text="Profiles (select one to see its pass/fail below)")
        profiles_frame.pack(fill="x", pady=(0, 8))
        self.profiles_tree = ttk.Treeview(
            profiles_frame, columns=PROFILE_COLUMNS, show="headings", selectmode="browse", height=3,
        )
        for col in PROFILE_COLUMNS:
            self.profiles_tree.heading(col, text=col)
            self.profiles_tree.column(col, width=90, anchor="w")
        self.profiles_tree.column("description", width=280)
        # Same clipped-columns-with-no-way-back problem ParametersPanel's own
        # params_tree fix addresses (see parameters_panel.py) -- profile+
        # score+fom+description (280) adds up to more than a narrow pane
        # reliably gets.
        profiles_hsb = ttk.Scrollbar(profiles_frame, orient="horizontal", command=self.profiles_tree.xview)
        self.profiles_tree.configure(xscrollcommand=profiles_hsb.set)
        self.profiles_tree.grid(row=0, column=0, sticky="ew")
        profiles_hsb.grid(row=1, column=0, sticky="ew")
        profiles_frame.grid_columnconfigure(0, weight=1)
        self.profiles_tree.tag_configure("error", foreground="#c0392b")
        self.profiles_tree.tag_configure("unmatched", foreground="#888888")
        self.profiles_tree.bind("<<TreeviewSelect>>", self._on_profile_select)

        metrics_frame = ttk.LabelFrame(body, text="Tests / metrics")
        metrics_frame.pack(fill="x", pady=(0, 8))
        self.metrics_tree = ttk.Treeview(
            metrics_frame, columns=METRIC_COLUMNS, show="headings", selectmode="browse", height=6,
        )
        for col in METRIC_COLUMNS:
            self.metrics_tree.heading(col, text=col)
            self.metrics_tree.column(col, width=90, anchor="w")
        self.metrics_tree.column("metric", width=180)
        self.metrics_tree.column("stale", width=40, anchor="center")
        self.metrics_tree.column("problems", width=60, anchor="center")
        # 11 columns (test/metric/typical/min/max/mean/std/unit/duration/
        # pass/stale) add up to >900px -- this is the "tabela de testes"
        # that was going invisible past the pane edge with no way to reach
        # it, even maximized; same fix as profiles_tree above.
        metrics_hsb = ttk.Scrollbar(metrics_frame, orient="horizontal", command=self.metrics_tree.xview)
        self.metrics_tree.configure(xscrollcommand=metrics_hsb.set)
        self.metrics_tree.grid(row=0, column=0, sticky="ew")
        metrics_hsb.grid(row=1, column=0, sticky="ew")
        metrics_frame.grid_columnconfigure(0, weight=1)
        self.metrics_tree.tag_configure("pass", foreground="#1a7f37")
        self.metrics_tree.tag_configure("fail", foreground="#c0392b")
        self.metrics_tree.tag_configure("stale", background="#fff3cd")
        self.metrics_tree.tag_configure("running", background="#cfe2ff")
        self.metrics_tree.bind("<<TreeviewSelect>>", self._on_metric_select)

        self.plot_notebook = ttk.Notebook(body)
        self.plot_notebook.pack(fill="both", expand=True)

    @property
    def variation_name(self):
        """Which variation this panel currently shows (None if none
        selected) -- app.py compares this against a batch job's own
        "currently running" variation (see mark_running_test() below) to
        decide whether live per-test highlighting applies here at all."""
        return self._variation_name

    def mark_running_test(self, test_name):
        """Highlights every metrics_tree row for `test_name` (one per
        metric, all sharing the same "test" column value) as
        currently-simulating, clearing any previous test's highlight --
        pass None to just clear. Driven by app.py from a batch job's own
        "@PROGRESS RUNNING <variation> <test> ..." lines, only when this
        panel's own self._variation_name is the variation that line is
        about (see the "test"/"stale" columns' own existing tag precedent
        this copies)."""
        for item in self.metrics_tree.get_children(""):
            tags = set(self.metrics_tree.item(item, "tags"))
            is_target = test_name is not None and self.metrics_tree.set(item, "test") == test_name
            if is_target and "running" not in tags:
                self.metrics_tree.item(item, tags=tuple(tags | {"running"}))
            elif not is_target and "running" in tags:
                self.metrics_tree.item(item, tags=tuple(tags - {"running"}))

    def clear(self):
        self._variation_name = None
        self._variation_block = None
        self.title_var.set("Select a variation to see details")
        self._metrics = []
        self._render_profiles([])
        self._render_plot(None)
        if self.problems_panel:
            self.problems_panel.clear()

    def show(self, variation_name):
        self._variation_name = variation_name
        self.title_var.set(variation_name)

        variations = {v["name"]: v for v in data.load_variations(all_topologies=True)}
        variation = variations.get(variation_name)
        self._variation_block = variation["block"] if variation else None

        config = data.load_config()
        block_cfg = config.get("blocks", {}).get(variation["block"], {}) if variation else {}

        self._metrics = [
            r for r in data.latest_results(data.load_results(all_topologies=True)) if r["variation"] == variation_name
        ]
        self._render_profiles(fom.classify(block_cfg, self._metrics))

        first_test = self._metrics[0]["test"] if self._metrics else None
        self._render_plot(first_test)
        if self.problems_panel:
            self.problems_panel.show(variation_name, first_test)

    def _render_profiles(self, profiles):
        self._profiles = sorted(profiles, key=lambda p: p["n_satisfied"], reverse=True)
        self.profiles_tree.delete(*self.profiles_tree.get_children())
        for p in self._profiles:
            fom_text = p["fom_error"] if p["fom_error"] else ("" if p["fom"] is None else fom.format_fom(p["fom"]))
            tag = "error" if p["fom_error"] else ("" if p["matched"] else "unmatched")
            score = f"{p['n_satisfied']}/{p['n_constraints']}"
            values = (p["profile"], score, fom_text, p["description"])
            self.profiles_tree.insert("", "end", iid=p["profile"], values=values, tags=(tag,) if tag else ())

        if self._profiles:
            self.profiles_tree.selection_set(self._profiles[0]["profile"])
            self._render_metrics(self._profiles[0])
        else:
            self._render_metrics(None)

    def _on_profile_select(self, _event):
        selection = self.profiles_tree.selection()
        if not selection:
            return
        profile = next((p for p in self._profiles if p["profile"] == selection[0]), None)
        self._render_metrics(profile)

    def _render_metrics(self, profile):
        self.metrics_tree.delete(*self.metrics_tree.get_children())
        constraints = profile["constraints"] if profile else {}
        variables = fom.metrics_to_variables(self._metrics)
        problems_by_test = {}
        for r in self._metrics:
            slug = fom.slugify(r["metric"])
            bounds = constraints.get(slug)
            if bounds is None:
                pass_text, tag = "", ()
            else:
                ok = fom.constraint_satisfied(slug, bounds, variables)
                pass_text, tag = ("PASS", "pass") if ok else ("FAIL", "fail")
            stale_text = "*" if r.get("stale") else ""
            duration = r.get("duration_seconds")
            duration_text = f"{duration:.1f}s" if duration is not None else ""
            if r["test"] not in problems_by_test:
                diagnostics = data.diagnostics_for_variation(self._variation_name, r["test"])
                problems_by_test[r["test"]] = variations_table._problems_text({
                    "errors": sum(1 for d in diagnostics if d["severity"] == "error"),
                    "warnings": sum(1 for d in diagnostics if d["severity"] == "warning"),
                })
            tags = ((tag,) if tag else ()) + (("stale",) if r.get("stale") else ())
            values = (
                r["test"], problems_by_test[r["test"]], r["metric"],
                _format_sig(r["typical"]), _format_sig(r["min"]), _format_sig(r["max"]),
                _format_sig(r.get("mean")), _format_sig(r.get("std")),
                r.get("unit", ""), duration_text, pass_text, stale_text,
            )
            self.metrics_tree.insert("", "end", values=values, tags=tags)

    def _on_metric_select(self, _event):
        selection = self.metrics_tree.selection()
        if not selection:
            return
        test_name = self.metrics_tree.set(selection[0], "test")
        self._render_plot(test_name)
        if self.problems_panel:
            self.problems_panel.show(self._variation_name, test_name)

    def _render_plot(self, test_name):
        for tab in self.plot_notebook.tabs():
            self.plot_notebook.forget(tab)
        if not test_name or not self._variation_name:
            return

        plots = data.plot_paths_for(self._variation_name, test_name)
        if not plots:
            # No PNG on disk yet -- run_test() stopped generating one
            # eagerly at simulation time (see analog_designer.sim.run_sim.generate_plot's
            # own docstring for why); this is the first time anyone's
            # looked, so materialize it now from the raw output the
            # simulation already left on disk, then look again.
            self._generate_plot_if_possible(test_name)
            plots = data.plot_paths_for(self._variation_name, test_name)
        if not plots:
            frame = ttk.Frame(self.plot_notebook)
            ttk.Label(frame, text="(no plot for this test)").pack()
            self.plot_notebook.add(frame, text="plot")
            return

        for label, path in plots:
            frame = ttk.Frame(self.plot_notebook)
            view = ZoomableImageView(frame)
            view.pack(fill="both", expand=True)
            view.set_image(path)
            self.plot_notebook.add(frame, text=label)

    def _generate_plot_if_possible(self, test_name):
        """Lazily materializes the plot for (self._variation_name, test_name)
        -- no docker, no re-simulation, just re-parsing already-existing raw
        output through the same parser (analog_designer.sim.run_sim.generate_plot).
        A no-op if the block/test can't be resolved (shouldn't happen -- show()
        always sets _variation_block from variations.jsonl) or the test has no
        output on disk yet (never simulated): plot_paths_for() then stays
        empty and _render_plot() falls back to its usual "(no plot for this
        test)" placeholder, same as before this existed."""
        if not self._variation_block:
            return
        config = data.load_config()
        test_cfg = config.get("tests", {}).get(self._variation_block, {}).get(test_name)
        if test_cfg is None:
            return
        run_sim.generate_plot(self._variation_name, test_name, test_cfg, config["defaults"])
