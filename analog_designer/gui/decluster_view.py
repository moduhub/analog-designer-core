""""Declusterize" window: finds near-duplicate variations in parameter
space (analog_designer.results.decluster) and lets the user trim the
worse-scoring member of each near-duplicate cluster, with a side-by-side
compare against whichever sample would actually survive.

Opened from analog_designer/gui/app.py's "Variation > Declusterize..."
menu entry (no metric pre-chosen -- pick one from this window's own
Combobox) or from VariationDetail's "Declusterize by ..." right-click menu
entries (pre-seeded with that metric/profile and direction) -- see
App._open_decluster_window. A singleton Toplevel, same convention as pro's
own Inductor Generator popup: reopening just re-focuses (and, if a new
criterion was picked, re-seeds) the same window instead of stacking
duplicates.

Scoped to the CURRENT block/topology only (workspace.BLOCK/TOPOLOGY, same
as the Variations table's own default view) -- parameters aren't
comparable across topologies, the same constraint pro's Auto Combine
dialog already enforces for its own candidate pool. A hierarchical block's
own sub-blocks are declustered separately, one topology (and one open of
this window) at a time -- see analog_designer.results.decluster.
distance_param_names()'s own docstring for why "block_ref" parameters
can't be folded into one combined distance."""
import tkinter as tk
from tkinter import messagebox, ttk

import numpy as np

from analog_designer.core import workspace
from analog_designer.results import data, decluster
from analog_designer.gui import variations_table


def _negate(value):
    """Flip a variations_table.criterion_value() sort_value so "greater =
    more preferred" when the user asked to prefer the LOWEST value
    (descending=False) -- plain floats (typical/min/max/mean/std/fom)
    negate directly; the "passed"/"failed" tuples (a primary count plus
    fom-based tiebreakers, see criterion_value's own docstring) negate
    elementwise. This makes decluster.decluster()'s "kept" pick always
    exactly the variation that would sort to the TOP of the Variations
    table under this same (criterion, descending) -- the most intuitive
    anchor for "which one survives a trim": whichever one this exact sort
    would already put first."""
    if isinstance(value, tuple):
        return tuple(_negate(v) for v in value)
    return -value


def _all_criteria(summaries, block_cfg):
    """[(combo_label, criterion, descending), ...] for every metric and
    profile actually available across `summaries` -- the Declusterize
    window's own metric Combobox, built from the exact same criterion
    vocabulary (and shared metric_criteria()/profile_criteria() builders)
    as VariationsTable's sort menus and VariationDetail's right-click
    ones, so "declusterize by X" always means the same thing everywhere
    it's offered. Metrics are the union of every (test, metric) pair
    that's been measured on at least one variation (metric_stats keys),
    field choices ("typical"/"min"/"max"/"mean"/"std") limited to whichever
    fields ANY of them actually reports -- a Monte Carlo metric's mean/std
    show up even if only some corners/seeds have run so far."""
    fields_by_metric = {}
    for s in summaries:
        for (test, metric), stats in s.get("metric_stats", {}).items():
            fields = fields_by_metric.setdefault((test, metric), {"typical", "min", "max"})
            for field in ("mean", "std"):
                if stats.get(field) is not None:
                    fields.add(field)
    field_order = ["typical", "min", "max", "mean", "std"]

    entries = []
    for profile in block_cfg.get("profiles", {}):
        for label, criterion, descending in variations_table.profile_criteria(profile):
            entries.append((f"{profile}: {label}", criterion, descending))
    for (test, metric), fields in sorted(fields_by_metric.items()):
        ordered_fields = [f for f in field_order if f in fields]
        for label, criterion, descending in variations_table.metric_criteria(test, metric, ordered_fields):
            entries.append((f"{metric} ({test}): {label}", criterion, descending))
    return entries


PARAM_COLUMNS = ("candidate", "kept", "delta")
METRIC_COLUMNS = ("test", "metric", "candidate", "kept", "delta", "unit")


class DeclusterWindow(tk.Toplevel):
    def __init__(self, app, criterion=None, descending=True):
        """`app`: the main App window -- read for workspace.BLOCK/TOPOLOGY-
        scoped data (via analog_designer.results.data, re-read fresh on
        every _recompute() so a job finishing while this window sits open
        picks up new/changed results automatically), and called back into
        for trimming (app._trim_names(), the exact same confirm/delete/
        reload path Trim itself uses) and to check app.trigger.running
        before trimming (a job could start after this window opened)."""
        super().__init__(app)
        self.app = app
        self.title(f"Declusterize -- {workspace.BLOCK}/{workspace.TOPOLOGY}")
        self.geometry("980x720")

        self._criteria = []  # [(combo_label, criterion, descending), ...], rebuilt in _recompute()
        self._summaries_by_name = {}
        self._variations_by_name = {}
        self._vectors_by_name = {}
        self._param_names = []
        self._param_defs = {}
        self._candidate_kept = {}  # tree item id -> kept variation name, for the compare panel + Trim

        self._build_controls()
        self._build_results()
        self._build_compare()

        if criterion is not None:
            self.set_criterion(criterion, descending)
        else:
            self._recompute()  # populates self._criteria/self._criterion_combo before any selection exists

    # -- controls -----------------------------------------------------

    def _build_controls(self):
        bar = ttk.Frame(self, padding=8)
        bar.pack(fill="x")

        ttk.Label(bar, text="Declusterize by:").pack(side="left")
        self.criterion_var = tk.StringVar()
        self.criterion_combo = ttk.Combobox(bar, textvariable=self.criterion_var, state="readonly", width=42)
        self.criterion_combo.pack(side="left", padx=(4, 12))
        self.criterion_combo.bind("<<ComboboxSelected>>", lambda _e: self._recompute())

        ttk.Label(bar, text="Similarity threshold:").pack(side="left")
        self.similarity_var = tk.DoubleVar(value=95.0)
        similarity_spin = ttk.Spinbox(
            bar, from_=50.0, to=99.9, increment=0.5, textvariable=self.similarity_var, width=6,
        )
        similarity_spin.pack(side="left", padx=(4, 2))
        similarity_spin.bind("<Return>", lambda _e: self._recompute())
        ttk.Label(bar, text="%").pack(side="left")
        ttk.Button(bar, text="Run", command=self._recompute).pack(side="left", padx=(8, 0))

        # Similarity, not distance, is the control surface here -- matching
        # how the request for this tool was framed ("percentual de
        # similaridade limítrofe") -- but every function in
        # analog_designer.results.decluster works in DISTANCE percent
        # (0 = identical, 100 = opposite ends of every parameter's own
        # range), the more natural unit for the actual math/tests. The
        # conversion (distance = 100 - similarity) happens once, right
        # here, at the one point a human types a number in.
        self.summary_var = tk.StringVar(value="")
        ttk.Label(bar, textvariable=self.summary_var, foreground="#666666").pack(side="left", padx=(16, 0))

    # -- results tree ---------------------------------------------------

    def _build_results(self):
        frame = ttk.LabelFrame(self, text="Clusters (select a candidate row to compare it against its kept sample)")
        frame.pack(fill="both", expand=True, padx=8, pady=(0, 4))

        self.tree = ttk.Treeview(
            frame, columns=("metric", "distance"), show="tree headings", selectmode="extended", height=10,
        )
        self.tree.heading("#0", text="cluster / candidate")
        self.tree.column("#0", width=320, anchor="w")
        self.tree.heading("metric", text="metric value")
        self.tree.column("metric", width=160, anchor="w")
        self.tree.heading("distance", text="distance to kept")
        self.tree.column("distance", width=130, anchor="w")
        vsb = ttk.Scrollbar(frame, orient="vertical", command=self.tree.yview)
        hsb = ttk.Scrollbar(frame, orient="horizontal", command=self.tree.xview)
        self.tree.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)
        self.tree.grid(row=0, column=0, sticky="nsew")
        vsb.grid(row=0, column=1, sticky="ns")
        hsb.grid(row=1, column=0, sticky="ew")
        frame.grid_rowconfigure(0, weight=1)
        frame.grid_columnconfigure(0, weight=1)
        self.tree.tag_configure("cluster", foreground="#666666")
        self.tree.bind("<<TreeviewSelect>>", self._on_select)

        actions = ttk.Frame(self, padding=(8, 0, 8, 8))
        actions.pack(fill="x")
        ttk.Button(actions, text="Trim selected candidate(s)", command=self._trim_selected).pack(side="left")

    def _on_select(self, _event):
        """Reacts to a single candidate row -- via Tk's own "last-
        interacted-with" focus() when it's one of the selection (same
        convention VariationsTable._on_select uses), else the first
        selected candidate row, ignoring any cluster-header rows also
        selected. Selecting only cluster headers (no candidate at all)
        clears the compare panel rather than showing stale data."""
        item = self.tree.focus()
        if item not in self._candidate_kept:
            candidates = [i for i in self.tree.selection() if i in self._candidate_kept]
            item = candidates[0] if candidates else None
        if item is None:
            self._show_compare(None, None)
            return
        self._show_compare(self.tree.item(item, "text"), self._candidate_kept[item])

    def _trim_selected(self):
        names = [self.tree.item(i, "text") for i in self.tree.selection() if i in self._candidate_kept]
        if not names:
            messagebox.showinfo("Declusterize", "Select one or more candidate rows first (not a cluster header).")
            return
        if self.app.trigger.running:
            messagebox.showinfo("Declusterize", "A job is currently running -- try again once it finishes.")
            return
        if self.app._trim_names(names, noun="declustering candidate(s)"):
            self._recompute()

    # -- compare panel ----------------------------------------------------

    def _build_compare(self):
        frame = ttk.LabelFrame(self, text="Compare (candidate vs. the sample that would stay)")
        frame.pack(fill="both", expand=True, padx=8, pady=(0, 8))
        self.compare_title_var = tk.StringVar(value="Select a candidate above to compare it.")
        ttk.Label(frame, textvariable=self.compare_title_var).pack(anchor="w", padx=4, pady=(2, 4))

        body = ttk.Panedwindow(frame, orient="horizontal")
        body.pack(fill="both", expand=True)

        params_frame = ttk.Frame(body)
        ttk.Label(params_frame, text="Parameters (most different first)").pack(anchor="w")
        self.params_tree = self._build_diff_tree(params_frame, PARAM_COLUMNS, tree_heading="parameter")
        body.add(params_frame, weight=1)

        metrics_frame = ttk.Frame(body)
        ttk.Label(metrics_frame, text="Metrics").pack(anchor="w")
        self.metrics_tree = self._build_diff_tree(metrics_frame, METRIC_COLUMNS)
        body.add(metrics_frame, weight=1)

    def _build_diff_tree(self, parent, columns, tree_heading=None):
        """A small Treeview for one compare table. tree_heading (e.g.
        "parameter"): use Treeview's own hierarchical #0 column for each
        row's own identity, `columns` for the data fields alongside it --
        used for the parameters table, one row per parameter name. None: a
        plain flat table (show="headings" only) -- the metrics table has
        no single-column identity of its own, "test"+"metric" are already
        two of its own data columns."""
        show = "tree headings" if tree_heading else "headings"
        tree = ttk.Treeview(parent, columns=columns, show=show)
        if tree_heading:
            tree.heading("#0", text=tree_heading)
            tree.column("#0", width=150, anchor="w")
        for col in columns:
            tree.heading(col, text=col)
            tree.column(col, width=90, anchor="w")
        vsb = ttk.Scrollbar(parent, orient="vertical", command=tree.yview)
        tree.configure(yscrollcommand=vsb.set)
        tree.pack(side="left", fill="both", expand=True)
        vsb.pack(side="left", fill="y")
        return tree

    def _show_compare(self, candidate, kept):
        self.params_tree.delete(*self.params_tree.get_children())
        self.metrics_tree.delete(*self.metrics_tree.get_children())
        if candidate is None:
            self.compare_title_var.set("Select a candidate above to compare it.")
            return
        self.compare_title_var.set(f"{candidate}  vs.  {kept} (kept)")
        for name, cand_val, kept_val, delta_text in self._param_rows(candidate, kept):
            self.params_tree.insert("", "end", text=name, values=(cand_val, kept_val, delta_text))
        for row in self._metric_rows(candidate, kept):
            self.metrics_tree.insert("", "end", values=row)

    def _param_rows(self, candidate, kept):
        """(name, candidate_value, kept_value, delta_text) for every
        declared free parameter, sorted by delta% descending (the axes the
        two samples actually differ on, first) -- block_ref/locked
        parameters (excluded from the distance calc, see
        decluster.distance_param_names()) are still shown, for full
        transparency, with a "--" delta rather than a number."""
        cand_params = self._variations_by_name[candidate]["parameters"]
        kept_params = self._variations_by_name[kept]["parameters"]
        cand_vec = self._vectors_by_name.get(candidate)
        kept_vec = self._vectors_by_name.get(kept)
        index = {name: i for i, name in enumerate(self._param_names)}
        rows = []
        for name, pdef in self._param_defs.items():
            default = pdef.get("default", "")
            cand_text = str(cand_params.get(name, default))
            kept_text = str(kept_params.get(name, default))
            i = index.get(name)
            if i is not None and cand_vec is not None and kept_vec is not None:
                delta = abs(float(cand_vec[i]) - float(kept_vec[i])) * 100.0
                rows.append((delta, name, cand_text, kept_text, f"{delta:.2f}%"))
            else:
                rows.append((-1.0, name, cand_text, kept_text, "--"))
        rows.sort(key=lambda r: r[0], reverse=True)
        return [(name, c, k, d) for _, name, c, k, d in rows]

    def _metric_rows(self, candidate, kept):
        """(test, metric, candidate_typical, kept_typical, delta, unit) for
        every (test, metric) either sample has measured -- a metric only
        one of them has yet (e.g. a still-running batch) shows a blank on
        the other side rather than being left out."""
        cand_stats = self._summaries_by_name.get(candidate, {}).get("metric_stats", {})
        kept_stats = self._summaries_by_name.get(kept, {}).get("metric_stats", {})
        rows = []
        for test, metric in sorted(set(cand_stats) | set(kept_stats)):
            c, k = cand_stats.get((test, metric), {}), kept_stats.get((test, metric), {})
            cv, kv = c.get("typical"), k.get("typical")
            unit = c.get("unit") or k.get("unit") or ""
            delta = f"{cv - kv:+.4g}" if isinstance(cv, (int, float)) and isinstance(kv, (int, float)) else ""
            rows.append((
                test, metric,
                "" if cv is None else f"{cv:.4g}", "" if kv is None else f"{kv:.4g}",
                delta, unit,
            ))
        return rows

    # -- criterion / recompute -----------------------------------------

    def set_criterion(self, criterion, descending):
        """Re-seeds the metric Combobox from an externally-chosen criterion
        (VariationDetail's "Declusterize by ..." menu) and recomputes.
        Rebuilds self._criteria first (from CURRENT data) so the label this
        criterion maps to actually exists in the list even if the window
        was just created and never populated it yet."""
        self._recompute(select=(criterion, descending))

    def _recompute(self, select=None):
        """Reloads every variation of the current block/topology, ranks
        them by the chosen criterion, computes the parameter-space
        distance matrix, and re-runs decluster.decluster() at the current
        similarity threshold -- cheap enough (well under a second for
        thousands of variations, see decluster.distance_matrix_pct()'s own
        docstring) to re-run on every metric/threshold change rather than
        needing any incremental update path. `select`: (criterion,
        descending) to pre-select in the Combobox instead of whatever's
        already chosen there (used by set_criterion(); None keeps the
        current selection, or falls back to the first available one)."""
        config = data.load_config()
        block_cfg = config["blocks"][workspace.BLOCK]["topologies"][workspace.TOPOLOGY]
        self._param_defs = block_cfg.get("parameters", {})
        self._param_names = decluster.distance_param_names(self._param_defs)

        summaries = data.variation_summaries()
        self._summaries_by_name = {s["variation"]: s for s in summaries}
        self._variations_by_name = {v["name"]: v for v in data.load_variations()}

        self._criteria = _all_criteria(summaries, config["blocks"][workspace.BLOCK])
        self.criterion_combo["values"] = [label for label, _, _ in self._criteria]
        criterion, descending = self._resolve_selection(select)
        if criterion is None:
            self.summary_var.set("No metric or profile available yet -- run some tests first.")
            self._populate_tree([])
            return

        names, vectors, metric_values, excluded = [], [], [], 0
        for s in summaries:
            name = s["variation"]
            variation = self._variations_by_name.get(name)
            if variation is None:
                continue
            key, _ = variations_table.criterion_value(criterion, s)
            if key is None:
                excluded += 1
                continue
            names.append(name)
            vectors.append(decluster.variation_vector(variation["parameters"], self._param_defs, self._param_names))
            metric_values.append(key if descending else _negate(key))

        try:
            similarity_pct = self.similarity_var.get()
        except tk.TclError:
            # The Spinbox's free-text field can be typed into directly (not
            # just via the up/down arrows) -- e.g. cleared to empty, or
            # mid-edit when Enter lands -- which DoubleVar.get() can't
            # parse. Falls back to the last-known-good value rather than
            # raising out of this event handler.
            similarity_pct = 95.0
            self.similarity_var.set(similarity_pct)

        self._vectors_by_name = dict(zip(names, vectors))
        matrix = np.array(vectors) if vectors else np.zeros((0, len(self._param_names)))
        dist_pct = decluster.distance_matrix_pct(matrix)
        threshold_pct = 100.0 - similarity_pct
        clusters = decluster.decluster(names, dist_pct, metric_values, threshold_pct)

        n_candidates = sum(len(c["candidates"]) for c in clusters)
        label = variations_table.criterion_label(criterion)
        excluded_text = f", {excluded} excluded (no {label} data yet)" if excluded else ""
        self.summary_var.set(
            f"{len(names)} variation(s) considered{excluded_text} -- "
            f"{len(clusters)} cluster(s), {n_candidates} candidate(s) for elimination"
        )
        self._populate_tree(clusters, criterion)
        self._show_compare(None, None)

    def _resolve_selection(self, select):
        """(criterion, descending) to actually run with -- `select` if
        given, else whatever the Combobox already shows, else the first
        available entry. Also sets criterion_var/criterion_combo's own
        current selection to match, so the two never disagree."""
        if select is not None:
            criterion, descending = select
            label = variations_table.criterion_label(criterion)
            match = next(
                (e for e in self._criteria if e[1] == criterion and e[2] == descending),
                next((e for e in self._criteria if variations_table.criterion_label(e[1]) == label), None),
            )
            if match is not None:
                self.criterion_var.set(match[0])
                return match[1], match[2]
            # Not found among this run's own criteria (e.g. right-clicked a
            # metric with zero measured samples right now) -- still honor
            # the caller's explicit choice; _recompute()'s own per-variation
            # loop will just find nothing to rank and report 0 considered.
            self.criterion_var.set(f"{variations_table.criterion_label(criterion)} (no data yet)")
            return criterion, descending
        current = self.criterion_var.get()
        match = next((e for e in self._criteria if e[0] == current), None)
        if match is not None:
            return match[1], match[2]
        if self._criteria:
            self.criterion_var.set(self._criteria[0][0])
            return self._criteria[0][1], self._criteria[0][2]
        return None, True

    def _populate_tree(self, clusters, criterion=None):
        self.tree.delete(*self.tree.get_children())
        self._candidate_kept = {}
        for ci, cluster in enumerate(clusters):
            kept = cluster["kept"]
            _, kept_text = variations_table.criterion_value(criterion, self._summaries_by_name[kept]) if criterion else (None, "")
            parent = self.tree.insert(
                "", "end", text=f"Cluster {ci + 1} -- keep {kept} ({len(cluster['candidates'])} candidate(s))",
                values=(kept_text, ""), open=True, tags=("cluster",),
            )
            for cand in cluster["candidates"]:
                _, cand_text = variations_table.criterion_value(criterion, self._summaries_by_name[cand["name"]]) if criterion else (None, "")
                item = self.tree.insert(
                    parent, "end", text=cand["name"], values=(cand_text, f"{cand['distance_pct']:.2f}%"),
                )
                self._candidate_kept[item] = kept
