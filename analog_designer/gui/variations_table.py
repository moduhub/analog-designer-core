"""Top-level table: one row per circuit variation (not per test/metric --
see variation_detail.py for that). Selecting a row drives
variation_detail.py; right-clicking a metric or profile there picks what
this table is sorted by (see set_criterion()), shown in its "sort" column.

selectmode="extended" gives native multi-select: plain click replaces the
selection, Shift+click extends a contiguous range from the last-clicked row
(what App._update_range runs Update against, via selected_variations() --
see there), Ctrl+click toggles one row at a time, Ctrl+A selects everything
(see _select_all). Ctrl-clicking a second row (with exactly one already
selected) instead fires on_combine_select(a, b) -- app.py uses that to open
the Combine dialog pre-filled with both, a shortcut for the same thing the
Combine... toolbar button does one field at a time. Deliberately NOT on
Shift+click (which used to trigger this) -- that gesture now means
range-select, the universal desktop convention, so Combine moved to
Ctrl+click instead. Pro's own design-space scatter-plot viewer has its own,
separate shift-click -> combine binding, unaffected by any of this."""
import tkinter as tk
from tkinter import ttk

from analog_designer.results import fom as fom_module

COLUMNS = ("variation", "sort", "status", "stale", "problems", "block", "topology", "metrics", "created")
# "sort" shows each row's value for whatever criterion the user picked from
# the detail panel's right-click menus (a metric's typical/min/max/mean/std,
# or a profile's fom / constraints passed) -- see set_criterion(). It
# replaces the fixed "profile"/"fom" columns, which only ever showed the
# FIRST matched profile and read as ambiguous with several profiles around.
# "problems" is a distinct-diagnostic-count readout (not raw occurrence
# count) sourced from analog_designer.results.data.variation_problem_counts
# -- e.g. "!1" for 5 identical ngspice warnings that all group into one
# distinct diagnostic key. Blank when the variation has none.
# "status" is a live, per-row readout of a batch job's own progress for that
# variation ("⏳ <test> Ns" while in flight -- the current test's own name,
# not just an elapsed counter, so a reader can tell WHAT a slow row is stuck
# on, not just THAT it's still going -- frozen at "pass/fail: N seconds" once
# done) -- see set_running()/set_done()/clear_running() below. Reuses this
# existing column rather than adding a dedicated "current test" one (tried
# first since it's the smaller change; revisit with a real column if the
# combined text ever reads as too cramped for this column's width).
# Deliberately per-row rather than one global "currently running" indicator:
# under workspace.cpu_budget() > 1, several variations are genuinely in
# flight at once, and a single indicator can only ever show one of them,
# jumping between rows as each one's own progress line happens to print --
# which reads as random flicker, not progress, and never answers "which one
# is actually taking longer" or "has this one finished yet".
# Columns whose displayed text doesn't sort correctly as a plain string --
# "sort" since its text is formatted ("<1"/"449K" foms, "%.4g" metrics,
# "3/5" passed counts), "metrics" because it's an int shown without
# zero-padding ("10" sorts before "9" as text). Sorted by the real
# underlying number instead, via _sort_values below.
NUMERIC_COLUMNS = {"sort", "metrics"}

SORT_HEADING = "sort (right-click a metric/profile)"


def criterion_label(criterion):
    """Heading text for a set_criterion() criterion, without the arrow."""
    if criterion["kind"] == "metric":
        return f"{criterion['metric']} ({criterion['field']})"
    return f"{criterion['profile']} ({'fom' if criterion['field'] == 'fom' else 'passed'})"


def criterion_value(criterion, s):
    """(sort_value, text) of summary `s` under `criterion`. sort_value None
    (missing metric/statistic, fom error, complex fom) always sorts last."""
    if criterion["kind"] == "metric":
        stats = s.get("metric_stats", {}).get((criterion["test"], criterion["metric"]))
        value = stats.get(criterion["field"]) if stats else None
        if not isinstance(value, (int, float)):
            return None, ""
        return value, f"{value:.4g} {stats.get('unit') or ''}".strip()
    profile = next((p for p in s.get("profiles", []) if p["profile"] == criterion["profile"]), None)
    if profile is None:
        return None, ""
    if criterion["field"] == "passed":
        # Ties (many variations at e.g. 6/7) broken by the profile's fom.
        fom_value = profile["fom"] if isinstance(profile["fom"], (int, float)) else float("-inf")
        return (profile["n_satisfied"], fom_value), f"{profile['n_satisfied']}/{profile['n_constraints']}"
    if profile["fom_error"]:
        return None, profile["fom_error"]
    value = profile["fom"]
    if not isinstance(value, (int, float)):
        # A complex fom (pow() of a negative ratio by a fractional weight)
        # renders "N/A" and can't be compared against a float for sorting.
        return None, "" if value is None else fom_module.format_fom(value)
    return value, fom_module.format_fom(value)


def _problems_text(problems):
    """"✗2 !5" for 2 distinct errors + 5 distinct warnings, "✗2" / "!5" when
    only one severity is present, "" when problems is falsy (no runs.jsonl
    yet, or none of its rows carry any diagnostic) -- same glyphs app.py
    already uses for a batch job's own ok/error/warning status text, so
    this reads consistently with that existing convention."""
    if not problems:
        return ""
    parts = []
    if problems.get("errors"):
        parts.append(f"✗{problems['errors']}")
    if problems.get("warnings"):
        parts.append(f"!{problems['warnings']}")
    return " ".join(parts)


class VariationsTable(ttk.Frame):
    def __init__(self, master, on_select, on_combine_select=None):
        super().__init__(master)
        self.on_select = on_select
        self.on_combine_select = on_combine_select

        # Total (+ how many of those are selected, directly relevant now
        # that Update/Purge Stale Results/Purge Plots all act on whatever's
        # selected here -- see App._start_update/_purge_stale_selected/
        # _purge_plots_selected) -- updated by _update_count_label(), called
        # from every place the row set or the selection changes.
        self.count_var = tk.StringVar(value="0 variations")
        ttk.Label(self, textvariable=self.count_var, foreground="#666666").grid(
            row=0, column=0, columnspan=2, sticky="w", pady=(0, 2),
        )

        self.tree = ttk.Treeview(self, columns=COLUMNS, show="headings", selectmode="extended")
        for col in COLUMNS:
            self.tree.heading(col, text=col, command=lambda c=col: self._sort_by(c))
            self.tree.column(col, width=110, anchor="w")
        self.tree.column("variation", width=180)
        self.tree.heading("sort", text=SORT_HEADING)
        self.tree.column("sort", width=190, anchor="e")
        self.tree.column("status", width=170, anchor="w")  # room for "⏳ <test> Ns", not just "Ns" -- a first guess, easy to retune
        self.tree.column("stale", width=40, anchor="center")
        self.tree.column("problems", width=60, anchor="center")

        vsb = ttk.Scrollbar(self, orient="vertical", command=self.tree.yview)
        # 9 columns at their declared widths add up to more than this pane
        # comfortably gets in the paned window's 50/50 split with the detail
        # panel (see app.py's Panedwindow.add(..., weight=1) for both) --
        # ttk.Treeview columns never shrink below their own width to fit,
        # so without this the rightmost column(s) (today: "created") simply
        # go invisible with no way back, even maximized -- a plain
        # horizontal scrollbar is the standard/permanent fix, same pattern
        # as the vertical one right above.
        hsb = ttk.Scrollbar(self, orient="horizontal", command=self.tree.xview)
        self.tree.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)
        self.tree.grid(row=1, column=0, sticky="nsew")
        vsb.grid(row=1, column=1, sticky="ns")
        hsb.grid(row=2, column=0, sticky="ew")
        self.grid_rowconfigure(1, weight=1)
        self.grid_columnconfigure(0, weight=1)

        self.tree.tag_configure("matched", foreground="#1a7f37")  # meets every constraint of the profile being sorted by
        self.tree.tag_configure("stale", background="#fff3cd")
        self.tree.tag_configure("running", background="#cfe2ff")  # cleared once that row's own "status" text freezes at pass/fail -- see set_done()
        self.tree.tag_configure("slow", foreground="#b35900")  # layered on top of "running" past App._SLOW_ROW_THRESHOLD_S -- flags a genuinely slow row, not just a numeric readout
        self.tree.bind("<<TreeviewSelect>>", self._on_select)
        # Bound on the widget instance, so it's dispatched before the
        # Treeview class's own <Button-1> binding (bindtags checks the
        # instance before the class) -- letting _on_combine_click return
        # "break" to suppress extended mode's native Ctrl+click toggle-select.
        self.tree.bind("<Control-Button-1>", self._on_combine_click)
        # Same idea for "select everything" -- extended mode has no built-in
        # binding for this, so it's added explicitly (keeps "update
        # everything" a one-keystroke operation, same convenience the old
        # From/To dialog's first/last defaults gave for free).
        self.tree.bind("<Control-a>", self._select_all)
        self.tree.bind("<Control-A>", self._select_all)

        self._sort_state = {}
        self._sort_values = {}
        self._summaries = {}  # iid -> summary, so set_criterion() can recompute "sort" without a reload
        self._criterion = None  # see set_criterion()

    def _build_row(self, s):
        """(values, tags, sort_entry) for one variation summary -- the exact
        row shape both set_rows() (full rebuild) and add_new_rows() (live,
        append-only insert during a running job) hand to tree.insert(), so a
        row painted mid-batch can never look different from what a later
        full reload() would have produced for the same summary."""
        sort_value, sort_text = criterion_value(self._criterion, s) if self._criterion else (None, "")
        values = (
            s["variation"], sort_text, "", "*" if s.get("has_stale") else "", _problems_text(s.get("problems")),
            s["block"], s["topology"], s["n_total"], s["created"],
        )
        tags = ("stale",) if s.get("has_stale") else ()
        if self._criterion and self._criterion["kind"] == "profile":
            profile = next((p for p in s.get("profiles", []) if p["profile"] == self._criterion["profile"]), None)
            if profile and profile["matched"]:
                tags += ("matched",)
        sort_entry = {"sort": sort_value, "metrics": s["n_total"]}
        return values, tags, sort_entry

    def set_criterion(self, criterion, descending=False):
        """Sort every row by `criterion` and show each row's value for it in
        the "sort" column -- driven by the detail panel's right-click menus
        (see VariationDetail). criterion is
            {"kind": "metric", "test": ..., "metric": ..., "field": "typical"|"min"|"max"|"mean"|"std"}
            {"kind": "profile", "profile": ..., "field": "fom"|"passed"}
        or None to clear it. Kept across reload()s (set_rows re-applies it);
        clicking the "sort" heading flips the direction."""
        self._criterion = criterion
        for iid, s in self._summaries.items():
            values, tags, sort_entry = self._build_row(s)
            live_tags = set(self.tree.item(iid, "tags")) & {"running", "slow"}
            status = self.tree.set(iid, "status")
            self.tree.item(iid, values=values, tags=tuple(set(tags) | live_tags))
            self.tree.set(iid, "status", status)
            self._sort_values[iid] = sort_entry
        if criterion is None:
            self.tree.heading("sort", text=SORT_HEADING)
            return
        self._sort_state["sort"] = descending
        self._sort_by("sort")

    @property
    def criterion(self):
        return self._criterion

    def _update_count_label(self):
        total = len(self.tree.get_children(""))
        selected = len(self.tree.selection())
        text = f"{total} variation{'s' if total != 1 else ''}"
        if selected:
            text += f" · {selected} selected"
        self.count_var.set(text)

    def set_rows(self, summaries):
        self.tree.delete(*self.tree.get_children())
        self._sort_values = {}
        self._summaries = {}
        for s in summaries:
            values, tags, sort_entry = self._build_row(s)
            iid = self.tree.insert("", "end", values=values, tags=tags)
            self._sort_values[iid] = sort_entry
            self._summaries[iid] = s
        if self._criterion is not None:
            # Keep the user's chosen order (and direction) across reloads.
            self._sort_state["sort"] = not self._sort_state.get("sort", True)
            self._sort_by("sort")
        self._update_count_label()

    def add_new_rows(self, summaries):
        """Inserts a row for every summary whose variation isn't already in
        the table -- called on a ~1s timer (App._poll_new_variations) while
        a Create/Generate job is running, so a brand-new variation gets a
        row to exist in well before the job finishes, closing the no-op gap
        set_running()'s own docstring describes ("a name with no matching
        row is simply a no-op"). Deliberately NOT set_rows(): that does a
        full delete+reinsert, which would reset scroll position and the
        user's current selection on every tick -- unacceptable at any
        polling cadence. Also deliberately does NOT touch any row that
        already exists (no re-classification/re-fom of previously-seen
        variations here) -- that stays set_running()/set_done()'s job while
        a job is in flight, and the next real reload()'s job once it ends;
        this method's only scope is "make a new variation exist as a row"."""
        existing = {self.tree.set(i, "variation") for i in self.tree.get_children("")}
        for s in summaries:
            if s["variation"] in existing:
                continue
            values, tags, sort_entry = self._build_row(s)
            iid = self.tree.insert("", "end", values=values, tags=tags)
            self._sort_values[iid] = sort_entry
            self._summaries[iid] = s
        self._update_count_label()

    def _sort_by(self, col):
        reverse = self._sort_state.get(col, False)
        if col in NUMERIC_COLUMNS:
            # Rows without a value always go last, whichever the direction.
            children = self.tree.get_children("")
            known = [k for k in children if self._sort_values.get(k, {}).get(col) is not None]
            missing = [k for k in children if self._sort_values.get(k, {}).get(col) is None]
            items = sorted(known, key=lambda k: self._sort_values[k][col], reverse=reverse) + missing
        else:
            items = [k for _, k in sorted(
                ((self.tree.set(k, col), k) for k in self.tree.get_children("")),
                key=lambda t: t[0], reverse=reverse,
            )]
        for pos, k in enumerate(items):
            self.tree.move(k, "", pos)
        self._sort_state[col] = not reverse
        if col == "sort" and self._criterion is not None:
            self.tree.heading("sort", text=f"{criterion_label(self._criterion)} {'↓' if reverse else '↑'}")

    def _on_select(self, _event):
        """Fires on every selection change, including a Shift-click range --
        still hands on_select() exactly one "primary" variation (unchanged
        contract, so the detail panel keeps working exactly as with the old
        single-select "browse" mode) via self.tree.focus(), Tk's own
        "last-interacted-with" row, which extended mode keeps correctly set
        even across a multi-row selection. Falls back to the first selected
        item on the rare chance focus is empty. For the FULL selection (e.g.
        what App._update_range runs Update against), see selected_variations()
        instead."""
        self._update_count_label()
        selection = self.tree.selection()
        if not selection:
            return
        focused = self.tree.focus()
        item = focused if focused in selection else selection[0]
        variation = self.tree.set(item, "variation")
        self.on_select(variation)

    def _on_combine_click(self, event):
        if self.on_combine_select is None:
            return None
        current = self.tree.selection()
        row = self.tree.identify_row(event.y)
        if not current or not row:
            return None
        a = self.tree.set(current[0], "variation")
        b = self.tree.set(row, "variation")
        if a == b:
            return None
        self.on_combine_select(a, b)
        return "break"

    def selected_variations(self):
        """Every currently-selected row's variation name, in the table's
        current (possibly sorted-by-column) order -- what App._update_range
        runs Update against directly, exactly as selected regardless of
        variations.jsonl's own creation order."""
        return [self.tree.set(i, "variation") for i in self.tree.selection()]

    def _select_all(self, _event):
        self.tree.selection_set(self.tree.get_children(""))
        return "break"

    def _find_row(self, name):
        for item in self.tree.get_children(""):
            if self.tree.set(item, "variation") == name:
                return item
        return None

    def update_row(self, s):
        """Refreshes one existing row's sort/stale/metrics columns
        (and their "matched"/"stale" tags) from a freshly
        computed summary -- "status" and any "running"/"slow" tag are left
        untouched, those stay set_running()/set_done()'s own job. Called
        right after a batch job's own per-variation "@PROGRESS DONE ..."
        line (see App._refresh_variation_row), so a variation's spec match
        shows up the moment IT finishes instead of only after the whole
        batch ends and reload() runs -- which used to mean a manual Reload
        mid-batch just to see this. A variation with no matching row is a
        no-op, same convention as set_running()/set_done()."""
        item = self._find_row(s["variation"])
        if item is None:
            return
        values, tags, sort_entry = self._build_row(s)
        current_status = self.tree.set(item, "status")
        live_tags = set(self.tree.item(item, "tags")) & {"running", "slow"}
        self.tree.item(item, values=values, tags=tuple(set(tags) | live_tags))
        self.tree.set(item, "status", current_status)
        self._sort_values[item] = sort_entry
        self._summaries[item] = s

    def set_running(self, name, text, slow=False):
        """Live "status" text for `name`'s own row (e.g. "running: 12s") plus
        its "running" highlight -- called every poll tick while that
        variation has a simulation in flight (see App._refresh_running_rows),
        so `text` keeps changing but only ITS OWN row does; every other row
        is untouched. A cheap, targeted per-row update (NOT set_rows()'s full
        delete+reinsert, which would drop scroll position/selection if
        called this often). A `name` with no matching row (e.g. a
        Create/Generate job's brand-new variation, not in the table until
        the job finishes) is simply a no-op. slow=True (past
        App._SLOW_ROW_THRESHOLD_S) layers the "slow" tag on top, flagging a
        row that's genuinely taking a while -- not just a number to notice,
        but a visual cue."""
        item = self._find_row(name)
        if item is None:
            return
        tags = set(self.tree.item(item, "tags"))
        tags.add("running")
        tags = (tags | {"slow"}) if slow else (tags - {"slow"})
        self.tree.item(item, tags=tuple(tags))
        self.tree.set(item, "status", text)

    def set_done(self, name, text):
        """Freezes `name`'s own row at its final "status" text (e.g. "pass:
        12s"/"error: 4s") and drops its "running"/"slow" highlight -- called
        once, when that variation's own "@PROGRESS DONE ..." line arrives
        (see analog_designer.sim.run_sim.emit_progress_variation_done). Left
        in place afterward (NOT cleared by a later reload() -- see
        App._reapply_running_status) so every row's final duration stays
        visible and comparable once the whole batch finishes, which is the
        actual point of tracking this per row instead of showing one
        transient global indicator."""
        item = self._find_row(name)
        if item is None:
            return
        tags = set(self.tree.item(item, "tags")) - {"running", "slow"}
        self.tree.item(item, tags=tuple(tags))
        self.tree.set(item, "status", text)

    def clear_running(self):
        """Blanks every row's "status" text and drops any "running"/"slow"
        highlight -- called once, at the start of a new batch job, so a
        previous job's per-row markers never linger into (and get confused
        with) the next one."""
        for item in self.tree.get_children(""):
            tags = set(self.tree.item(item, "tags")) - {"running", "slow"}
            self.tree.item(item, tags=tuple(tags))
            self.tree.set(item, "status", "")

    def select_variation(self, name):
        """Programmatically select a row (e.g. driven by a point click on
        pro's own design-space scatter-plot viewer) without re-firing
        on_select -- the caller already knows."""
        for item in self.tree.get_children(""):
            if self.tree.set(item, "variation") == name:
                self.tree.selection_set(item)
                self.tree.see(item)
                return
