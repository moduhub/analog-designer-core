"""Entrypoint: python -m analog_designer.gui.app [PROJECT_ROOT] -- PROJECT_ROOT is the
project folder to open (needs a config.json); omit to reopen the
last-opened folder, else CWD. "Open Folder..." (File menu) switches to a
different project folder without restarting.

This is the CORE distribution's App: a classic menu bar (File: Open
Folder.../Docker Settings.../Reload -- Variation: Create.../Trim -- Run:
Update -- Data: Release.../Purge Stale Results.../Purge Plots...), a thin
always-visible context strip below it (Block/Topology selectors, Cancel,
status -- current state, not one-off commands, so it stays out of any menu),
a Notebook with just one tab -- "Variations" (a vertical split on the left
-- the variations table on top, one row per circuit variation with its
design-profile classification, and its selected row's Parameters table
below, analog_designer.gui.parameters_panel.ParametersPanel; drill-down on
the right, its own table of tests/metrics and their plots, analog_designer.
gui.variation_detail.VariationDetail -- a different granularity,
deliberately not merged into the variations table) -- and a scrolling log
console at the bottom that shows the active background job's output
(run_sim.py, gen_variations.py, manual_variation.py, update_variations.py),
alongside the Errors/Warnings tabs (analog_designer.gui.problems_panel.
ProblemsPanel).

No directed/randomized variation-generation research features, no
design-space scatter-plot viewer, no training, no IC layout generation --
those are all private "pro" features. The private analog-designer-pro tool
subclasses this App (analog_designer_pro.gui.app.App) rather than
re-wiring the shared machinery below from scratch: it overrides/extends
_build_menu(), _build_notebook_tabs() (adding its own extra tabs after
calling super()), and a few callbacks (_create_variation, for its own
richer create-variation dialog) to add its own actions on top, reusing
docker settings, RunTrigger, the variations-table refresh polling loop, and
the block/topology selectors as-is.

Force/skip-on-fail have no dedicated dialog of their own: Update and Create
each embed the same fields (analog_designer.gui.run_options_dialog.
add_force_skip_fields) directly into their own confirmation, pre-filled
with the last-used value -- see _start_update/_create_variation below.

Update targets whichever variation(s) are selected in the table (Shift-click
for a contiguous range, Ctrl-click to add one at a time, Ctrl+A for
everything) -- one row (or none) runs plain run_sim.py, falling back to the
config.json default-parameter variation when nothing is selected (run_sim.py's
only mode before named-variation support existed); two or more runs the
batch counterpart, analog_designer/sim/update_variations.py, after a quick count
confirmation. Either way, it re-checks the target(s)' tests against their
current definitions and only re-runs whichever are stale (or everything,
with --force).

Create opens analog_designer.gui.create_variation_dialog.ask_create_variation, a
tabbed dialog covering core's own ways to produce a new variation
(Basic manual entry / From Parent, Monte Carlo random sampling) -- see
_create_argv for how each tab's result maps to a script invocation.

Update/Create all share one RunTrigger instance and are mutually exclusive
-- they touch the same materialized schematic/JSONL logs/docker container.
self._active_job tracks which of them is running so _on_run_done can
report/re-enable correctly. Trim doesn't use the trigger (it's synchronous
local file I/O, see analog_designer.sim.run_sim.trim_variation) but is
still gated on trigger.running to avoid rewriting sim/variations.jsonl or
sim/results.jsonl while a background job is appending to them.

Release exports the selected variation as its block's presentable snapshot
for an outside evaluator -- materialized schematic under
release/xschem/sch/, a results writeup under release/doc/<block>/README.md
(see analog_designer.release.export.export_release) -- same synchronous,
trigger.running-gated shape as Trim, not routed through RunTrigger either,
though it does briefly open its own docker container (unlike Trim) to render
a schematic PNG for that writeup.
"""
import sys
import time
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from analog_designer.core import workspace
from analog_designer.release import export as release_export
from analog_designer.results import data
from analog_designer.sim import run_sim
from analog_designer.gui.create_variation_dialog import ask_create_variation
from analog_designer.gui.docker_settings_dialog import show_docker_settings
from analog_designer.gui.parameters_panel import ParametersPanel
from analog_designer.gui.problems_panel import ProblemsPanel
from analog_designer.gui.progress_tracker import ProgressTracker
from analog_designer.gui.run_options_dialog import ask_run_confirm
from analog_designer.gui.run_trigger import MANUAL_VARIATION_MODULE, RUN_SIM_MODULE, RunTrigger
from analog_designer.gui.variation_detail import VariationDetail
from analog_designer.gui.variations_table import VariationsTable

POLL_INTERVAL_MS = 200
# Progress bar segments (see _paint_progress_bar) -- simulated ok / failed
# (same red as the "finished WITH ERRORS" status line) / planned but never run.
PROGRESS_OK = "#3a9d5d"
PROGRESS_FAIL = "#c0392b"
PROGRESS_SKIP = "#a0a0a0"
PROGRESS_TROUGH = "#e6e6e6"
PROGRESS_BORDER = "#bcbcbc"
NEW_ROW_POLL_INTERVAL_MS = 1000  # deliberately much slower than POLL_INTERVAL_MS -- see App._poll_new_variations
_SLOW_ROW_THRESHOLD_S = 30  # elapsed time on an in-flight variation's row past which it's flagged "slow" (see VariationsTable.set_running) -- see App._refresh_running_rows


def _running_status_text(info):
    """(text, elapsed) for one App._running[variation] entry -- shared by
    _refresh_running_rows (the live 200ms loop) and _reapply_running_status
    (repaints after a reload() mid-batch) so the two can never show
    different text for the exact same in-flight variation."""
    elapsed = time.monotonic() - info["start_ts"]
    return f"⏳ {info['test_name']} {elapsed:.0f}s", elapsed


ALL_TOPOLOGIES = "Todas"  # Topology dropdown sentinel: widen the Variations table to every topology of the active block (they share tests/metrics) without changing workspace.TOPOLOGY itself, which stays the concrete topology Create/job-launches use -- see App._show_all_topologies.


class App(ttk.Frame):
    def __init__(self, master):
        super().__init__(master)
        self.pack(fill="both", expand=True)

        # Plain attributes, not tk.Variables -- nothing in the main window
        # binds to them live. These are just the LAST-used values: each
        # job-launching action's own dialog pre-fills its embedded
        # force/skip fields from these (see run_options_dialog.
        # add_force_skip_fields) and overwrites them with whatever the user
        # confirmed, rather than requiring separate prior setup.
        # skip_on_fail_profile is a profile NAME (config.json
        # blocks.<block>.profiles) or None/off -- see run_sim.run_variation's
        # own skip_on_fail_profile.
        self._run_force = False
        self._skip_on_fail_profile = None
        # Only create_variation_dialog's own Monte Carlo tab exposes these
        # today (see add_skip_fail_tolerance_fields) -- meaningless without
        # a profile, so both reset to (0, False) wherever _skip_on_fail_profile
        # itself gets reset for a block switch (see below, in this same
        # class's own block/topology-change handler).
        self._skip_on_fail_max_failures = 0
        self._discard_on_fail = False
        self.trigger = RunTrigger(on_line=self._log, on_done=self._on_run_done)
        self.selected_variation = None
        self._active_job = None  # "update" | "update_range" | "create" | None -- a pro subclass adds several more job kinds of its own
        self._variations_before_job = set()
        self._show_all_topologies = False  # Topology dropdown == ALL_TOPOLOGIES -- widens the table only, see ALL_TOPOLOGIES
        self._progress = ProgressTracker()  # replaced per job in _start_job -- idle (no PLAN lines) until a simulation job starts
        # {variation: {"test_name": "<test>", "start_ts": monotonic}} for
        # every variation the current batch job has *in flight* right now
        # (see _handle_progress_running/_refresh_running_rows) -- a dict, not
        # a single "current" variation, since several can genuinely be
        # running at once under workspace.cpu_budget() > 1. "test_name" is
        # the bare test, not the finer per-condition label RUNNING actually
        # carries (dozens of conditions per test) -- repainting a row that
        # often would reproduce the exact "flicker, not progress" problem
        # VariationsTable.set_running's own docstring warns about for a
        # global indicator. Popped once that variation's own "@PROGRESS DONE
        # ..." line arrives (see _handle_progress_done).
        self._running = {}
        # {variation: status_text} for every variation this batch job has
        # already finished (or that was still running when the job ended --
        # see _on_run_done) -- kept independent of the table's own rows so
        # reload()'s full delete+reinsert (App.reload -> VariationsTable.set_rows)
        # doesn't erase it; _reapply_running_status() re-paints it onto the
        # freshly rebuilt rows right after. This -- not any live indicator --
        # is what lets a "which sample took longer" comparison survive to
        # look at once the whole batch is done.
        self._last_run_status = {}

        self._build_menu()
        self._build_context_bar()
        self._build_progress_bar()
        self._build_console()

        self.notebook = ttk.Notebook(self)
        self.notebook.pack(fill="both", expand=True, padx=4, pady=4)
        self._build_notebook_tabs()

        self._refresh_block_topology_choices()
        self.reload()
        self.after(POLL_INTERVAL_MS, self._poll_run)
        self.after(NEW_ROW_POLL_INTERVAL_MS, self._poll_new_variations)

    def _build_menu(self):
        """Classic dropdown menu bar, grouped by what the action DOES rather
        than how often it's used -- room to grow each group independently
        instead of one ever-longer row of buttons. Block/Topology/Cancel/
        status stay OUT of this menu (see _build_context_bar): those are
        current state, not one-off commands, and Cancel in particular needs
        to stay clickable without opening a menu while a job is running.

        self._menus (keyed by group, not by individual label, so _set_busy
        can disable every action a running job would conflict with per menu
        without hardcoding each label twice) is exposed on self specifically
        so a pro subclass can call super()._build_menu() then append its own
        items (variation_menu.add_command("Generate...", ...), etc.) onto
        the very same Menu objects, instead of rebuilding the whole bar."""
        menubar = tk.Menu(self.master)
        self.master.config(menu=menubar)

        file_menu = tk.Menu(menubar, tearoff=False)
        file_menu.add_command(label="Open Folder...", command=self._open_folder)
        file_menu.add_command(label="Docker Settings...", command=self._open_docker_settings)
        file_menu.add_separator()
        file_menu.add_command(label="Reload", command=self.reload)
        menubar.add_cascade(label="File", menu=file_menu)

        variation_menu = tk.Menu(menubar, tearoff=False)
        variation_menu.add_command(label="Create...", command=self._create_variation)
        variation_menu.add_separator()
        variation_menu.add_command(label="Trim", command=self._trim_selected)
        menubar.add_cascade(label="Variation", menu=variation_menu)

        run_menu = tk.Menu(menubar, tearoff=False)
        run_menu.add_command(label="Update", command=self._start_update)
        menubar.add_cascade(label="Run", menu=run_menu)

        data_menu = tk.Menu(menubar, tearoff=False)
        data_menu.add_command(label="Release...", command=self._export_release)
        data_menu.add_separator()
        data_menu.add_command(label="Purge Stale Results...", command=self._purge_stale_selected)
        data_menu.add_command(label="Purge Plots...", command=self._purge_plots_selected)
        data_menu.add_command(label="Purge Raw Sim Outputs...", command=self._purge_aux_selected)
        menubar.add_cascade(label="Data", menu=data_menu)

        self._menus = {"variation": variation_menu, "run": run_menu, "data": data_menu}

    def _build_context_bar(self):
        """Thin strip, always visible, deliberately outside the menu bar:
        Block/Topology (which one is active right now) and Cancel/status
        (whether a job is running right now) are current state to glance
        at, not commands to go find in a dropdown."""
        bar = ttk.Frame(self)
        bar.pack(fill="x", padx=4, pady=4)

        ttk.Label(bar, text="Block:").pack(side="left")
        self.block_var = tk.StringVar()
        self.block_combo = ttk.Combobox(bar, textvariable=self.block_var, state="readonly", width=14)
        self.block_combo.pack(side="left", padx=(2, 0))
        self.block_combo.bind("<<ComboboxSelected>>", self._on_block_selected)

        ttk.Label(bar, text="Topology:").pack(side="left", padx=(8, 0))
        self.topology_var = tk.StringVar()
        self.topology_combo = ttk.Combobox(bar, textvariable=self.topology_var, state="readonly", width=14)
        self.topology_combo.pack(side="left", padx=(2, 0))
        self.topology_combo.bind("<<ComboboxSelected>>", self._on_topology_selected)

        self.cancel_button = ttk.Button(bar, text="Cancel", command=self.trigger.cancel, state="disabled")
        self.cancel_button.pack(side="left", padx=(16, 0))

        self.status_var = tk.StringVar(value="idle")
        self.status_label = ttk.Label(bar, textvariable=self.status_var)
        self.status_label.pack(side="right")

    def _build_progress_bar(self):
        """Three-color progress bar + "done/total (pct%) · N failed · N
        skipped · ETA mm:ss" label, driven entirely by the "@PROGRESS PLAN/
        STEP/TESTFAIL/SKIPPED" lines a simulation job prints (see
        analog_designer/gui/progress_tracker.py for how they become
        weighted ok/fail/skip fractions) -- _log() intercepts and consumes
        those lines before they ever reach the console below. A plain
        Canvas rather than ttk.Progressbar, which can only draw one fill
        color. A job that never emits a PLAN line just leaves this idle at
        empty/blank -- status_var (set in _start_job) is still the only
        indicator for those."""
        frame = ttk.Frame(self)
        frame.pack(side="top", fill="x", padx=4, pady=(0, 4))
        self.progress_canvas = tk.Canvas(frame, height=14, highlightthickness=0, background=PROGRESS_TROUGH)
        self.progress_canvas.pack(side="left", fill="x", expand=True)
        self.progress_canvas.bind("<Configure>", lambda _e: self._paint_progress_bar())
        self.progress_label_var = tk.StringVar(value="")
        ttk.Label(frame, textvariable=self.progress_label_var, width=48, anchor="e").pack(side="left", padx=(8, 0))
        # No separate "currently running" label here -- under
        # workspace.cpu_budget() > 1 several variations are genuinely in
        # flight at once, and one shared label can only ever show whichever
        # one printed last, which reads as flicker rather than progress. See
        # VariationsTable's own "status" column / set_running()/set_done()
        # instead: a live per-row elapsed-time readout, one per in-flight
        # variation, frozen at pass/fail once each one's own "@PROGRESS
        # DONE ..." line arrives -- that's what actually answers "which
        # sample is taking longer" and "is this one done yet".

    def _build_console(self):
        # side="bottom", packed before the (expand=True) notebook, so it
        # claims its natural height from the bottom of the window first --
        # otherwise the notebook's expand=True eats all remaining space and
        # the console never gets shown.
        #
        # "Errors"/"Warnings" (analog_designer.gui.problems_panel.ProblemsPanel)
        # share this SAME box as first-level tabs alongside "Console", rather
        # than living in their own separate area elsewhere -- ProblemsPanel
        # itself builds no Notebook of its own; it only hands back the
        # pieces (errors_frame/warnings_frame/toolbar) for placement here,
        # and reports count changes via on_counts_changed so the tab labels
        # ("Errors (3)") can be kept in sync without ProblemsPanel needing to
        # know about this Notebook at all.
        frame = ttk.LabelFrame(self, text="Log")
        frame.pack(side="bottom", fill="x", padx=4, pady=(0, 4))

        self.log_notebook = ttk.Notebook(frame)
        # ProblemsPanel's own constructor renders once immediately (an
        # empty state) and fires on_counts_changed right then, before the
        # tabs below even exist yet -- _update_problem_tab_labels no-ops
        # until this flips True, right after they're added.
        self._log_tabs_ready = False
        self.problems_panel = ProblemsPanel(
            frame, content_master=self.log_notebook, on_counts_changed=self._update_problem_tab_labels,
        )
        self.problems_panel.toolbar.pack(side="top", fill="x")
        self.log_notebook.pack(fill="both", expand=True)

        console_tab = ttk.Frame(self.log_notebook)
        self.console = tk.Text(console_tab, height=10, state="disabled", wrap="word")
        vsb = ttk.Scrollbar(console_tab, orient="vertical", command=self.console.yview)
        self.console.configure(yscrollcommand=vsb.set)
        self.console.pack(side="left", fill="both", expand=True)
        vsb.pack(side="right", fill="y")
        self.log_notebook.add(console_tab, text="Console")

        self.log_notebook.add(self.problems_panel.errors_frame, text="Errors (0)")
        self.log_notebook.add(self.problems_panel.warnings_frame, text="Warnings (0)")
        self._log_tabs_ready = True

    def _build_notebook_tabs(self):
        """Populates self.notebook (already created by __init__ before this
        runs) with just the "Variations" tab -- a vertical split on the left
        (the variations table on top, its selected row's Parameters table
        below, own vertical split) and the drill-down detail pane on the
        right (Profiles/Metrics/Plots). A pro subclass calls
        super()._build_notebook_tabs() then self.notebook.add(...) its own
        extra tabs (its design-space viewer, Training, Layout, ...) after this."""
        variations_tab = ttk.Frame(self.notebook)
        self.notebook.add(variations_tab, text="Variations")

        paned = ttk.Panedwindow(variations_tab, orient="horizontal")
        paned.pack(fill="both", expand=True)

        # Left column: the Variations table on top, its selected row's
        # Parameters table below (own vertical split) -- freeing the right
        # column (Profiles/Metrics/plots) for more horizontal width, since
        # it no longer also has to fit Parameters stacked above itself.
        left = ttk.Panedwindow(paned, orient="vertical")
        self.table = VariationsTable(left, on_select=self._on_select_variation)
        # on_vary=None: core ships no vary-dialog (that's PRO-only, backed
        # by analog_designer_pro.gui.vary_param_dialog -- a pro subclass
        # would need to build its own ParametersPanel with on_vary wired in
        # if it wants this row-level "vary just this parameter" action).
        self.params_panel = ParametersPanel(left, on_vary=None)
        left.add(self.table, weight=2)
        left.add(self.params_panel, weight=1)

        self.detail = VariationDetail(
            paned, problems_panel=self.problems_panel,
            on_sort=self.table.set_criterion, current_sort=lambda: self.table.criterion,
        )
        paned.add(left, weight=1)
        paned.add(self.detail, weight=2)

    def _update_problem_tab_labels(self, n_errors, n_warnings):
        if not self._log_tabs_ready:
            return
        self.log_notebook.tab(self.problems_panel.errors_frame, text=f"Errors ({n_errors})")
        self.log_notebook.tab(self.problems_panel.warnings_frame, text=f"Warnings ({n_warnings})")

    def _open_folder(self):
        if self.trigger.running:
            return
        chosen = filedialog.askdirectory(
            title="Open project folder", initialdir=str(workspace.PROJECT_ROOT),
        )
        if not chosen:
            return
        try:
            workspace.open_folder(chosen)
        except (FileNotFoundError, ValueError) as exc:
            messagebox.showerror("Open Folder", str(exc))
            return
        self._show_all_topologies = False
        self.selected_variation = None
        self.detail.clear()
        self.params_panel.clear()
        self.master.title(_window_title())
        self._refresh_block_topology_choices()
        self.reload()

    def _refresh_block_topology_choices(self):
        self.block_combo["values"] = list(workspace.CONFIG.get("blocks", {}).keys())
        self.block_var.set(workspace.BLOCK)
        self._refresh_topology_choices()

    def _refresh_topology_choices(self):
        topologies = list(workspace.CONFIG["blocks"][workspace.BLOCK].get("topologies", {}).keys())
        self.topology_combo["values"] = [ALL_TOPOLOGIES] + topologies
        self.topology_var.set(ALL_TOPOLOGIES if self._show_all_topologies else workspace.TOPOLOGY)

    def _on_block_selected(self, _event):
        if self.trigger.running:
            return
        new_block = self.block_var.get()
        topologies = list(workspace.CONFIG["blocks"][new_block].get("topologies", {}).keys())
        self._apply_block_topology(new_block, topologies[0])

    def _on_topology_selected(self, _event):
        if self.trigger.running:
            return
        choice = self.topology_var.get()
        if choice == ALL_TOPOLOGIES:
            self._show_all_topologies = True
            self.reload()
            return
        self._apply_block_topology(workspace.BLOCK, choice)

    def _apply_block_topology(self, block, topology):
        workspace.open_folder(str(workspace.PROJECT_ROOT), block=block, topology=topology)
        self._show_all_topologies = False
        self.selected_variation = None
        self.detail.clear()
        self.params_panel.clear()
        self.master.title(_window_title())
        self._refresh_block_topology_choices()
        # Profiles are declared per-block -- a profile chosen for the
        # PREVIOUS block may not exist for this one (or, worse,
        # coincidentally match a different, same-named profile there).
        # Silently forwarding a stale name to the next job would either
        # sys.exit() it (validate_skip_on_fail_profile) or, in the
        # coincidence case, quietly check the wrong constraints -- reset
        # instead so a switch never carries a hidden, block-mismatched
        # choice forward.
        if self._skip_on_fail_profile not in workspace.CONFIG["blocks"][workspace.BLOCK].get("profiles", {}):
            self._skip_on_fail_profile = None
            self._skip_on_fail_max_failures = 0
            self._discard_on_fail = False
        self.reload()

    def _open_docker_settings(self):
        # Global/machine-level settings, not project-scoped -- safe to open
        # even while a background job is running (it only affects
        # subsequent container launches, not the one already in flight).
        show_docker_settings(self)

    def reload(self):
        summaries = data.variation_summaries(all_topologies=self._show_all_topologies)
        self.table.set_rows(summaries)
        self._reapply_running_status()

    def _reapply_running_status(self):
        """set_rows() just rebuilt every row from scratch, wiping any
        "status" text/tag set_running()/set_done() had painted on -- reapply
        this batch's own bookkeeping (self._last_run_status, self._running)
        on top so a reload() (called after every job, and from Reload/folder
        switches) never erases the "which sample took how long" picture the
        user is looking at mid- or post-batch."""
        for name, text in self._last_run_status.items():
            self.table.set_done(name, text)
        for name, info in self._running.items():
            text, elapsed = _running_status_text(info)
            self.table.set_running(name, text, slow=elapsed >= _SLOW_ROW_THRESHOLD_S)

    def _poll_new_variations(self):
        """Runs forever from a single self.after() chain kicked off once in
        __init__ (same always-rescheduling shape as _poll_run -- guards the
        WORK below on self.trigger.running, not the rescheduling itself, so
        there's no separate start/stop bookkeeping to keep in sync with
        _start_job/_on_run_done). Deliberately its own, much slower cadence
        (NEW_ROW_POLL_INTERVAL_MS=1000 vs POLL_INTERVAL_MS=200): this only
        needs a brand-new variation's row to EXIST soon enough for the
        already-fast 200ms _refresh_running_rows/set_running loop to find
        and paint it -- that row's own live status text is still driven at
        200ms once it exists, so there's no benefit to polling for new rows
        any faster than this."""
        if self.trigger.running:
            self._add_new_variation_rows()
        self.after(NEW_ROW_POLL_INTERVAL_MS, self._poll_new_variations)

    def _add_new_variation_rows(self):
        """Closes the gap VariationsTable.set_running's own docstring
        documents: a Create job's brand-new variation has no row to paint a
        live status onto until this makes one exist. data.variation_summaries()
        is the same call reload() already makes (and add_new_rows() only
        inserts names not already present, so calling this every tick is
        cheap and idempotent) -- scoped by self._show_all_topologies the
        same way reload() is, so new rows respect whatever topology view the
        user is currently looking at, not necessarily the job's own
        workspace.TOPOLOGY."""
        summaries = data.variation_summaries(all_topologies=self._show_all_topologies)
        self.table.add_new_rows(summaries)

    def _on_select_variation(self, variation_name):
        self.selected_variation = variation_name
        self.detail.show(variation_name)
        self.params_panel.show(variation_name)

    def _set_busy(self, busy):
        state = "disabled" if busy else "normal"
        self._menus["run"].entryconfig("Update", state=state)
        self._menus["variation"].entryconfig("Create...", state=state)
        self._menus["variation"].entryconfig("Trim", state=state)
        self._menus["data"].entryconfig("Release...", state=state)
        self._menus["data"].entryconfig("Purge Stale Results...", state=state)
        self._menus["data"].entryconfig("Purge Plots...", state=state)
        self.block_combo.configure(state="disabled" if busy else "readonly")
        self.topology_combo.configure(state="disabled" if busy else "readonly")
        self.cancel_button.configure(state="normal" if busy else "disabled")

    def _start_job(self, job, argv, status):
        self.console.configure(state="normal")
        self.console.delete("1.0", "end")
        self.console.configure(state="disabled")
        self._active_job = job
        # snapshot of what's already registered, so _on_run_done can tell
        # exactly which variation(s) this job produced -- one for Update,
        # possibly many for a Create Monte Carlo batch.
        self._variations_before_job = {v["name"] for v in data.load_variations(all_topologies=True)}
        self._progress = ProgressTracker(cpu_budget=workspace.cpu_budget())
        self._paint_progress()
        self._clear_running()
        self._set_busy(True)
        self.status_label.configure(foreground="")
        self.status_var.set(status)
        self.trigger.start(argv)

    def _scope_args(self, topology=None):
        return [
            "--project-root", str(workspace.PROJECT_ROOT),
            "--block", workspace.BLOCK,
            "--topology", topology or workspace.TOPOLOGY,
        ]

    def _topology_of(self, variation_name):
        """The topology a given variation actually belongs to, falling back
        to the active workspace.TOPOLOGY when nothing is selected. Needed
        because in the "Todas" topology view (self._show_all_topologies)
        the selected variation may not belong to the active topology at
        all -- an action targeting it (Update) must use ITS topology, not
        the toolbar's, or it'd hit the wrong schematic/params."""
        if not variation_name:
            return workspace.TOPOLOGY
        row = next((v for v in data.load_variations(all_topologies=True) if v["name"] == variation_name), None)
        return row["topology"] if row else workspace.TOPOLOGY

    def _start_update(self):
        """One command, two underlying scripts, chosen by how many rows are
        selected in the Variations table (Shift-click for a contiguous
        range, Ctrl-click to add one at a time, Ctrl+A for everything -- see
        VariationsTable.selected_variations()): 2+ selected runs the batch
        counterpart (analog_designer/sim/update_variations.py --name ..., one per
        selected row, parallelized across workspace.cpu_budget()); 0 or 1
        selected runs plain run_sim.py against just that variation (or the
        config.json default-parameter variation if nothing is selected at
        all -- run_sim.py's own original, no-variations-registered-yet
        mode). Either way, always confirms first via ask_run_confirm --
        which is also where force/skip-on-fail get set for this run,
        pre-filled with the last-used value (see run_options_dialog.
        add_force_skip_fields)."""
        if self.trigger.running:
            return
        names = self.table.selected_variations()
        if len(names) >= 2:
            message = f"Update {len(names)} selected variation(s)?"
        else:
            message = f"Update {self.selected_variation or 'the default variation'}?"
        result = ask_run_confirm(self, "Update", message, data.load_config(), self._run_force, self._skip_on_fail_profile)
        if result is None:
            return
        self._run_force, self._skip_on_fail_profile = result["force"], result["skip_on_fail_profile"]

        if len(names) >= 2:
            argv = [sys.executable, "-m", "analog_designer.sim.update_variations"]
            for name in names:
                argv += ["--name", name]
            scope_args = self._scope_args()
            status = f"updating {len(names)} selected variation(s)..."
            job = "update_range"
        else:
            argv = [sys.executable, "-m", RUN_SIM_MODULE]
            if self.selected_variation:
                argv.append(self.selected_variation)
            scope_args = self._scope_args(topology=self._topology_of(self.selected_variation))
            status = f"updating {self.selected_variation or 'default variation'}..."
            job = "update"
        if self._run_force:
            argv.append("--force")
        if self._skip_on_fail_profile:
            argv += ["--skip-on-fail", self._skip_on_fail_profile]
        argv += scope_args
        self._start_job(job, argv, status)

    def _trim_selected(self):
        """Deletes one or more variations entirely (variations.jsonl/
        results.jsonl rows + sim/<name>/ dir) -- same Shift/Ctrl-click range
        selection as _start_update()'s own VariationsTable.selected_variations()
        (a Monte Carlo batch's worth of bad candidates at once, not just
        whichever single row happens to be the "selected" one for the
        detail/params panels)."""
        if self.trigger.running:
            return
        names = self.table.selected_variations()
        if not names:
            messagebox.showinfo("Trim", "Select one or more variations first.")
            return
        message = (
            f"Delete {names[0]} and all its simulation data? This cannot be undone."
            if len(names) == 1 else
            f"Delete {len(names)} selected variations and all their simulation data? This cannot be undone."
        )
        if not messagebox.askyesno("Trim variation(s)", message):
            return
        for name in names:
            run_sim.trim_variation(name)
        if self.selected_variation in names:
            self.selected_variation = None
            self.detail.clear()
            self.params_panel.clear()
        self.status_var.set(f"trimmed {len(names)} variation(s)" if len(names) > 1 else f"trimmed {names[0]}")
        self.reload()

    def _purge_stale_selected(self):
        """Data menu action: physically deletes results.jsonl rows (and
        their now-orphaned run dirs and plot PNGs) for whichever (variation, test) pairs
        are flagged stale among the selected rows -- see
        run_sim.purge_stale_results's own docstring for how that differs
        from Trim (which deletes a whole variation) and from just leaving
        stale rows in place (data.latest_results() already flags them at
        read time, this removes them outright)."""
        if self.trigger.running:
            return
        names = self.table.selected_variations()
        if not names:
            messagebox.showinfo("Purge Stale Results", "Select one or more variations first.")
            return
        if not messagebox.askyesno(
            "Purge Stale Results",
            f"Remove stale (outdated) results for {len(names)} selected variation(s)? "
            "Only results whose test definition has since changed are removed -- fresh results are untouched.",
        ):
            return
        n = run_sim.purge_stale_results(names)
        self.status_var.set(f"purged {n} stale result(s) for {len(names)} selected variation(s)")
        if self.selected_variation in names:
            self.detail.show(self.selected_variation)
        self.reload()

    def _purge_plots_selected(self):
        """Data menu action: deletes every cached plot PNG for the selected
        variations (all tests, not just stale ones) -- forces regeneration
        next time a plot is opened, useful when only a parser's plotting
        code changed (not tracked by definition_hash, so Purge Stale
        Results wouldn't catch it)."""
        if self.trigger.running:
            return
        names = self.table.selected_variations()
        if not names:
            messagebox.showinfo("Purge Plots", "Select one or more variations first.")
            return
        if not messagebox.askyesno(
            "Purge Plots",
            f"Delete cached plot images for {len(names)} selected variation(s)? "
            "They'll be regenerated automatically the next time you open a test's plot.",
        ):
            return
        n = run_sim.purge_plots(names)
        self.status_var.set(f"purged {n} cached plot(s) for {len(names)} selected variation(s)")
        if self.selected_variation in names:
            self.detail.show(self.selected_variation)
        self.reload()

    def _purge_aux_selected(self):
        """Data menu action: deletes leftover .raw files and side dumps
        (e.g. '_diag.data') from the selected variations' run dirs -- see
        run_sim.purge_aux_outputs. Results and plots are unaffected; new
        runs don't write these to the project folder at all anymore."""
        if self.trigger.running:
            return
        names = self.table.selected_variations()
        if not names:
            messagebox.showinfo("Purge Raw Sim Outputs", "Select one or more variations first.")
            return
        if not messagebox.askyesno(
            "Purge Raw Sim Outputs",
            f"Delete .raw files and debug data dumps for {len(names)} selected variation(s)? "
            "Results and plots are not affected.",
        ):
            return
        n, freed = run_sim.purge_aux_outputs(names)
        self.status_var.set(f"purged {n} raw output file(s), {freed / 1e6:.0f} MB, for {len(names)} selected variation(s)")

    def _export_release(self):
        """Exports the selected variation as its block's release pick --
        materialized schematic under release/xschem/sch/, results writeup
        under release/doc/<block>/README.md (see analog_designer.release.export).
        Synchronous, gate only on trigger.running (same pattern as
        _trim_selected above, not routed through RunTrigger) -- but unlike
        _trim_selected's pure local file I/O, export_release() does open one
        docker container for the call, to render each exported block's own
        schematic to a PNG via a real xschem invocation; that's a few extra
        seconds of GUI freeze per export, accepted deliberately (see
        export_release()'s own docstring) rather than routing this one step
        through RunTrigger's background-job machinery."""
        if self.trigger.running:
            return
        if not self.selected_variation:
            messagebox.showinfo("Release", "Select a variation first.")
            return
        name = self.selected_variation
        if not messagebox.askyesno(
            "Release",
            f"Export {name!r} as its block's release pick (release/xschem/sch/ + release/doc/)? "
            "This overwrites that block's existing release files -- and, for a hierarchical "
            "block, its sub-block dependencies' release schematics too. Renders a schematic "
            "PNG via a temporary docker container, so this will take a few seconds.",
        ):
            return
        try:
            result = release_export.export_release(str(workspace.PROJECT_ROOT), name)
        except SystemExit as exc:
            messagebox.showerror("Release", str(exc))
            return
        self.status_var.set(f"exported {result['block']} release: {name}")
        messagebox.showinfo(
            "Release",
            f"Exported {result['block']}/{result['topology']} variation {name} to "
            f"release/xschem/sch/ and release/doc/{result['block']}/.",
        )

    def _create_variation(self, initial_tab="basic"):
        if self.trigger.running:
            return
        config = data.load_config()
        param_defs = config["blocks"][workspace.BLOCK]["topologies"][workspace.TOPOLOGY]["parameters"]
        result = ask_create_variation(
            self, config, workspace.BLOCK, workspace.TOPOLOGY, data.load_variations(), param_defs,
            self._run_force, self._skip_on_fail_profile, initial_tab=initial_tab,
            default_parent=self.selected_variation,
            skip_on_fail_max_failures=self._skip_on_fail_max_failures, discard_on_fail=self._discard_on_fail,
        )
        if result is None:
            return
        self._run_force, self._skip_on_fail_profile = result["force"], result["skip_on_fail_profile"]
        self._skip_on_fail_max_failures = result["skip_on_fail_max_failures"]
        self._discard_on_fail = result["discard_on_fail"]
        argv, status = self._create_argv(result)
        if self._run_force:
            argv.append("--force")
        if self._skip_on_fail_profile:
            argv += ["--skip-on-fail", self._skip_on_fail_profile]
            if self._skip_on_fail_max_failures:
                argv += ["--skip-on-fail-max-failures", str(self._skip_on_fail_max_failures)]
            if self._discard_on_fail:
                argv.append("--discard-on-fail")
        argv += self._scope_args()
        self._start_job("create", argv, status)

    def _create_argv(self, result):
        mode = result["mode"]
        if mode == "basic":
            argv = [sys.executable, "-m", MANUAL_VARIATION_MODULE]
            for name, value in result["params"].items():
                argv += ["--param", f"{name}={value}"]
            return argv, "creating manual variation..."
        if mode == "from_parent":
            argv = [sys.executable, "-m", MANUAL_VARIATION_MODULE, "--base", result["base"]]
            for name, value in result["params"].items():
                argv += ["--param", f"{name}={value}"]
            return argv, f"creating variation from {result['base']}..."
        # mode == "monte_carlo"
        argv = [sys.executable, "-m", "analog_designer.sim.gen_variations", str(result["count"])]
        if result["spread"] is not None:
            argv += ["--spread", str(result["spread"])]
        return argv, f"generating {result['count']} random variation(s)..."

    def _on_run_done(self, returncode):
        job = self._active_job or "update"
        self._active_job = None
        self._set_busy(False)
        self._freeze_interrupted_running()
        # Clean exit: whatever's still unfilled was planned but never ran
        # (gray) -- the bar always ends at 100%. Cancel/crash: left as is.
        self._progress.finish(returncode == 0)
        self._paint_progress()
        self.reload()
        # A non-zero exit means SOMETHING went wrong -- anywhere from "one
        # simulation failed its spec" (routine, but still worth flagging) to
        # an outright crash. Whatever the console/table detail says, this
        # line is the one signal guaranteed to be visible without having to
        # go scroll/read either of those, so it needs its own color, not
        # just more text easy to skim past.
        if returncode == 0:
            self.status_var.set(f"{job} finished (exit {returncode})")
            self.status_label.configure(foreground="")
        else:
            self.status_var.set(f"⚠ {job} finished WITH ERRORS (exit {returncode}) -- see log")
            self.status_label.configure(foreground="#c0392b")

    def _freeze_interrupted_running(self):
        """Any variation still in self._running here never got its own
        "@PROGRESS DONE ..." line -- the job ended some other way (Cancel,
        a crash) before reaching it. Freeze it at its last known elapsed
        time with an "interrupted" marker (see _last_run_status/
        VariationsTable.set_done) instead of leaving it stuck on a
        live-ticking readout that would otherwise count up forever, since
        nothing is left running to drive its poll-tick updates
        meaningfully once the job itself is gone."""
        for name, info in self._running.items():
            elapsed = time.monotonic() - info["start_ts"]
            self._last_run_status[name] = f"⏹ {elapsed:.0f}s"  # stopped/interrupted, not pass or fail
        self._running = {}

    def _poll_run(self):
        self.trigger.poll()
        self._refresh_running_rows()
        if self._active_job is not None and self._progress.active:
            self.progress_label_var.set(self._progress.label())  # ETA counts down between steps too
        self.after(POLL_INTERVAL_MS, self._poll_run)

    def _log(self, line):
        if self._consume_progress_line(line):
            return
        # gen_variations.py's own _run_batch/_run_hierarchical_batch print a
        # leading blank line before each "=== variation N ===" header --
        # reads fine as a paragraph-style spacer in a scrolling terminal,
        # but under a real batch (Update Range routinely runs many
        # variations, often several at once under workspace.cpu_budget()>1
        # interleaving their output) it shows up here as a scatter of
        # meaningless blank rows in the Text widget instead, which is what
        # reads as a "weird" log. Dropped here (GUI-only) rather than at the
        # source, so the plain CLI's own terminal output -- where the blank
        # line actually helps -- is untouched.
        if not line.strip():
            return
        self.console.configure(state="normal")
        self.console.insert("end", line + "\n")
        self.console.see("end")
        self.console.configure(state="disabled")

    def _consume_progress_line(self, line):
        """"@PROGRESS PLAN/STEP/TESTFAIL/SKIPPED ..." (self._progress, see
        ProgressTracker) and "@PROGRESS RUNNING/DONE/TRIMMED ..." from a job's
        subprocess (see analog_designer/sim/run_sim.py's emit_progress_*
        functions) drive the progress bar/label and the Variations table's
        own per-row live status instead of the console -- True if `line`
        was one of these (caller skips logging it), False for every
        ordinary line."""
        if line.startswith("@PROGRESS RUNNING "):
            self._handle_progress_running(line[len("@PROGRESS RUNNING "):])
            return True
        if line.startswith("@PROGRESS DONE "):
            self._handle_progress_done(line[len("@PROGRESS DONE "):])
            return True
        if line.startswith("@PROGRESS TRIMMED "):
            self._handle_progress_trimmed(line[len("@PROGRESS TRIMMED "):].split())
            return True
        if self._progress.feed(line):
            self._paint_progress()
            return True
        return False

    def _paint_progress(self):
        self._paint_progress_bar()
        self.progress_label_var.set(self._progress.label())

    def _paint_progress_bar(self):
        """Green/red/gray segments stacked left to right, each sized by its
        share of the job's estimated time (ProgressTracker.fractions())."""
        canvas = self.progress_canvas
        canvas.delete("all")
        width, height = canvas.winfo_width(), canvas.winfo_height()
        x = 0.0
        for fraction, color in zip(self._progress.fractions(), (PROGRESS_OK, PROGRESS_FAIL, PROGRESS_SKIP)):
            if fraction > 0:
                canvas.create_rectangle(x, 0, x + fraction * width, height, fill=color, width=0)
                x += fraction * width
        canvas.create_rectangle(0, 0, width - 1, height - 1, outline=PROGRESS_BORDER)

    def _handle_progress_running(self, rest):
        """rest is "<variation> <test> <label>" (see run_sim.emit_progress_running)
        -- condition_label() values never contain a space, so a plain 3-way
        split recovers all three fields. Registers `variation` as in flight
        (self._running, keyed by name -- several can be in flight at once
        under workspace.cpu_budget() > 1, so this is a dict entry, not a
        single "current" pointer) the first time it's seen, and keeps its
        "test_name" entry current on every later call too -- the table's own
        status column shows "⏳ <test_name> <elapsed>s" (see
        _running_status_text/_refresh_running_rows), the bare test name
        rather than the finer per-condition `label` (RUNNING fires once per
        condition, dozens per test -- repainting the row that often would
        reproduce the exact "flicker, not progress" problem
        VariationsTable.set_running's own docstring warns about for a global
        indicator; a test name changes only a handful of times per
        variation, the right cadence for a text change in a narrow column).
        The CURRENT test is also still shown in the detail panel, if it
        happens to be displaying this same variation."""
        try:
            variation, test_name, label = rest.split(" ", 2)
        except ValueError:
            return
        info = self._running.setdefault(variation, {"start_ts": time.monotonic(), "test_name": test_name})
        info["test_name"] = test_name
        if self.detail.variation_name == variation:
            self.detail.mark_running_test(test_name)

    def _handle_progress_done(self, rest):
        """rest is "<variation> <ok|error> <elapsed_seconds>" (see
        run_sim.emit_progress_variation_done) -- the definitive "this one's
        finished" signal, since there's no reliable way to infer it purely
        from the absence of further RUNNING lines (especially with several
        variations interleaving). Pops `variation` out of self._running
        (stops its live ticking) and freezes its row at a final pass/fail +
        duration via self._last_run_status, which _reapply_running_status()
        keeps re-painting across every later reload() -- so the reader can
        still compare every sample's duration after the whole batch ends,
        not just while it's live. Also refreshes THIS variation's own
        profile/fom columns right now (_refresh_variation_row) -- and its
        detail/parameters panels too, if it happens to be the one currently
        selected -- so its spec match shows up the moment it finishes
        instead of needing a manual Reload mid-batch to see it."""
        try:
            variation, status, elapsed_str = rest.split(" ", 2)
            elapsed = float(elapsed_str)
        except ValueError:
            return
        self._running.pop(variation, None)
        self._progress.variation_done(variation)
        self._paint_progress()
        symbol = "✓" if status == "ok" else "✗"  # check / cross
        self._last_run_status[variation] = f"{symbol} {elapsed:.0f}s"
        self.table.set_done(variation, self._last_run_status[variation])
        self._refresh_variation_row(variation)
        if self.detail.variation_name == variation:
            self.detail.mark_running_test(None)
            self.detail.show(variation)
            self.params_panel.show(variation)

    def _handle_progress_trimmed(self, names):
        """A discard-on-fail checkpoint just deleted `names` (see
        run_sim.emit_progress_trimmed) -- drop their rows now instead of
        leaving them on screen until the batch's own end-of-job reload().
        A full reload() rather than a per-row delete: checkpoints are rare
        (one per checkpoint_size variations), and it's the same rebuild the
        end of the job does anyway."""
        for name in names:
            self._last_run_status.pop(name, None)
        self.reload()

    def _refresh_variation_row(self, variation):
        """Recomputes just `variation`'s own profile/fom/stale columns
        (data.variation_summaries(names=[variation]) -- cheap, skips every
        other variation's own fom.classify()) and repaints its row via
        VariationsTable.update_row(). A no-op if the variation isn't
        registered yet (shouldn't happen -- @PROGRESS DONE only fires for a
        variation run_sim.py already wrote to variations.jsonl) or doesn't
        belong to the current topology view (_show_all_topologies-scoped,
        same as reload())."""
        summaries = data.variation_summaries(all_topologies=self._show_all_topologies, names=[variation])
        if summaries:
            self.table.update_row(summaries[0])

    def _refresh_running_rows(self):
        """Called every _poll_run tick (200ms), independent of whether any
        new subprocess output arrived -- the whole point is that every
        in-flight variation's own elapsed count keeps rising even during a
        long silent stretch (e.g. a slow ngspice run), which is what
        actually distinguishes "still working" from "frozen", per row."""
        for variation, info in self._running.items():
            text, elapsed = _running_status_text(info)
            self.table.set_running(variation, text, slow=elapsed >= _SLOW_ROW_THRESHOLD_S)

    def _clear_running(self):
        self._running = {}
        self._last_run_status = {}
        self.table.clear_running()
        self.detail.mark_running_test(None)


def _window_title():
    return f"{workspace.PROJECT_ROOT.name} ({workspace.BLOCK}/{workspace.TOPOLOGY}) - analog-designer-core"


def main():
    project_root = sys.argv[1] if len(sys.argv) > 1 else None
    workspace.open_folder(project_root)

    root = tk.Tk()
    root.title(_window_title())
    root.geometry("1200x800")
    App(root)
    root.mainloop()


if __name__ == "__main__":
    main()
