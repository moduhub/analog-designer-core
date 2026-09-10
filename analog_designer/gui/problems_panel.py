"""Problems panel: grouped, filterable view of simulator diagnostics
(errors/warnings) for a variation -- an IDE "Problems"-panel-style list,
grouped by distinct diagnostic (not one row per raw log line -- a single
recurring issue can print hundreds of times, see
analog_designer.sim.log_diagnostics's own docstring) with an occurrence
count column.

NOT a self-packing widget -- a content PROVIDER. app.py hosts "Errors"/
"Warnings" as first-level tabs of its own Notebook, right alongside
"Console" (see app.py's own _build_console()), sharing that one box
instead of this living in its own separate area -- so this class exposes
its pieces (`errors_frame`, `warnings_frame`, `toolbar`) for the caller to
place wherever it wants, rather than building a Notebook of its own the
way an earlier version did.

Data always comes from analog_designer.results.data.diagnostics_for_variation()
-- this widget never reads runs.jsonl or any log file directly."""
import tkinter as tk
from tkinter import ttk

from analog_designer.results import data

COLUMNS = ("count", "category", "message")


def _row_text(tree, item):
    """One diagnostic row as a single copy/paste-friendly line -- "Nx
    category: message", matching what's visually shown in the tree so a
    pasted line is self-explanatory without the table around it."""
    count, category, message = tree.item(item, "values")
    return f"{count}x {category}: {message}"


def _all_rows_text(tree):
    return "\n".join(_row_text(tree, item) for item in tree.get_children())


def _copy_to_clipboard(widget, text):
    if not text:
        return
    widget.clipboard_clear()
    widget.clipboard_append(text)


def _build_tree(master):
    tree = ttk.Treeview(master, columns=COLUMNS, show="headings", selectmode="browse", height=5)
    for col in COLUMNS:
        tree.heading(col, text=col)
    tree.column("count", width=50, anchor="center")
    tree.column("category", width=140, anchor="w")
    tree.column("message", width=420, anchor="w")
    vsb = ttk.Scrollbar(master, orient="vertical", command=tree.yview)
    tree.configure(yscrollcommand=vsb.set)
    tree.pack(side="left", fill="both", expand=True)
    vsb.pack(side="right", fill="y")

    # Right-click (Button-3: Windows/Linux; Button-2 covers some macOS/X11
    # trackpad-as-secondary-click setups) -- "Copy line" acts on whichever
    # row is under the cursor (selecting it first, so a copy never fires
    # against a stale/unrelated previous selection), "Copy all" ignores
    # selection entirely and always covers every row currently shown.
    menu = tk.Menu(tree, tearoff=False)

    def _popup(event):
        row = tree.identify_row(event.y)
        if row:
            tree.selection_set(row)
            menu.entryconfigure(0, state="normal")
        else:
            menu.entryconfigure(0, state="disabled")
        menu.tk_popup(event.x_root, event.y_root)

    def _copy_selected_line():
        selected = tree.selection()
        if selected:
            _copy_to_clipboard(tree, _row_text(tree, selected[0]))

    menu.add_command(label="Copy line", command=_copy_selected_line)
    menu.add_command(label="Copy all", command=lambda: _copy_to_clipboard(tree, _all_rows_text(tree)))
    tree.bind("<Button-3>", _popup)
    tree.bind("<Button-2>", _popup)
    # Ctrl+C on an already-selected row -- the common shortcut a user
    # reaches for before thinking to right-click.
    tree.bind("<Control-c>", lambda e: _copy_selected_line())

    return tree


class ProblemsPanel:
    def __init__(self, master, content_master=None, on_counts_changed=None):
        """`master`: parent for `toolbar`. `content_master`: parent for
        `errors_frame`/`warnings_frame` -- separate from `master` because
        Tk requires a Notebook tab's content widget to be an actual CHILD
        of that Notebook, so a caller hosting these frames as tabs (see
        app.py's _build_console()) must pass its Notebook here, not the
        LabelFrame the toolbar lives in; defaults to `master` for a caller
        that doesn't care (e.g. a standalone layout with no Notebook).
        Neither `master` nor `content_master` is itself packed anywhere by
        this class -- the caller places each piece (`toolbar`,
        `errors_frame`, `warnings_frame`) wherever it wants.

        `on_counts_changed(n_errors, n_warnings)`, if given, fires on every
        _render() (selection change, scope toggle, ...) so the caller can
        update its own tab labels (e.g. "Errors (3)") without this class
        needing to know about whatever Notebook hosts it -- including once,
        during this very constructor's own initial _render([]), before the
        caller has necessarily finished wiring its Notebook up; a caller
        relying on this callback needs to tolerate (or ignore) that first,
        pre-setup call."""
        content_master = content_master or master
        self._on_counts_changed = on_counts_changed
        self._variation_name = None
        self._test_name = None
        # "test" (default): scoped to whichever test is currently selected
        # in metrics_tree, matching what _refresh()'s caller just showed a
        # plot for -- "variation": every test's diagnostics, unfiltered.
        self._scope_var = tk.StringVar(value="test")

        self.toolbar = ttk.Frame(master)
        ttk.Label(self.toolbar, text="Show:").pack(side="left", padx=(0, 4))
        ttk.Radiobutton(
            self.toolbar, text="This test", variable=self._scope_var, value="test", command=self._refresh,
        ).pack(side="left")
        ttk.Radiobutton(
            self.toolbar, text="Whole variation", variable=self._scope_var, value="variation", command=self._refresh,
        ).pack(side="left")

        self.errors_frame = ttk.Frame(content_master)
        self.warnings_frame = ttk.Frame(content_master)
        self._trees = {
            "error": _build_tree(self.errors_frame),
            "warning": _build_tree(self.warnings_frame),
        }
        for tree in self._trees.values():
            tree.tag_configure("error", foreground="#c0392b")
            tree.tag_configure("warning", foreground="#b35900")

        self._render([])

    def clear(self):
        self._variation_name = None
        self._test_name = None
        self._render([])

    def show(self, variation_name, test_name):
        """Called from VariationDetail whenever the selected variation or
        the selected metric row's test changes -- test_name may be None
        (no metric row selected yet), which just means "This test" scope
        currently has nothing to narrow to; "Whole variation" still works."""
        self._variation_name = variation_name
        self._test_name = test_name
        self._refresh()

    def _refresh(self):
        if not self._variation_name:
            self._render([])
            return
        scoped_to_test = self._scope_var.get() == "test"
        if scoped_to_test and not self._test_name:
            self._render([])
            return
        test = self._test_name if scoped_to_test else None
        self._render(data.diagnostics_for_variation(self._variation_name, test))

    def _render(self, diagnostics):
        for tree in self._trees.values():
            tree.delete(*tree.get_children())
        for d in diagnostics:
            tree = self._trees[d["severity"]]
            tree.insert(
                "", "end", values=(d["count"], d["category"], d["message"]), tags=(d["severity"],),
            )
        n_errors = sum(1 for d in diagnostics if d["severity"] == "error")
        n_warnings = sum(1 for d in diagnostics if d["severity"] == "warning")
        if self._on_counts_changed:
            self._on_counts_changed(n_errors, n_warnings)
