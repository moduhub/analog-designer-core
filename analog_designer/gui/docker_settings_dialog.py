"""Modal "Docker Settings" dialog: edit the global, machine-level
analog_designer.core.settings values (container image, mount-path template, and how many
CPU cores this host budgets for simulation work) -- same modal Toplevel +
ttk.Entry/OK-Cancel shape as analog_designer.gui.schematic_editor.config_dialog.
Not project-scoped (these are facts about this machine, not any one
design), so the caller may show it regardless of whether a background job
is running.
"""
import tkinter as tk
from tkinter import messagebox, ttk

from analog_designer.core import settings


def _center_on_parent(win, parent):
    win.update_idletasks()
    x = parent.winfo_rootx() + (parent.winfo_width() - win.winfo_width()) // 2
    y = parent.winfo_rooty() + (parent.winfo_height() - win.winfo_height()) // 2
    win.geometry(f"+{max(x, 0)}+{max(y, 0)}")


def show_docker_settings(parent):
    """Blocks (grab_set + wait_window) until closed. Saves to
    ~/.mh-analog-designer/settings.json on OK; does nothing on Cancel."""
    current = settings.load()["container"]

    win = tk.Toplevel(parent)
    win.title("Docker Settings")
    win.resizable(False, False)
    win.transient(parent)

    body = ttk.Frame(win, padding=12)
    body.pack(fill="both", expand=True)

    image_var = tk.StringVar(value=current["image"])
    template_var = tk.StringVar(value=current["project_root_template"])
    cpu_budget_var = tk.StringVar(value=str(current["cpu_budget"]))

    grid = ttk.Frame(body)
    grid.pack(fill="x")

    ttk.Label(grid, text="Image").grid(row=0, column=0, sticky="w", pady=2)
    ttk.Entry(grid, textvariable=image_var, width=36).grid(row=0, column=1, sticky="w", pady=2)

    ttk.Label(grid, text="Container mount path template").grid(row=1, column=0, sticky="w", pady=2)
    ttk.Entry(grid, textvariable=template_var, width=36).grid(row=1, column=1, sticky="w", pady=2)
    ttk.Label(
        grid, text="{name} is replaced with the host project folder's name", foreground="#888888",
    ).grid(row=2, column=1, sticky="w")

    ttk.Label(grid, text="CPU budget (cores)").grid(row=3, column=0, sticky="w", pady=(10, 2))
    ttk.Entry(grid, textvariable=cpu_budget_var, width=8).grid(row=3, column=1, sticky="w", pady=(10, 2))
    ttk.Label(
        grid, text="Total cores this host spends on simulation work at once. Each job\n"
        "(a test condition, or a whole variation) requests as many cores as it\n"
        "needs and gets whatever's currently free -- many cheap ngspice/Xyce\n"
        "runs share this budget 1 core each, while a single expensive run can\n"
        "claim most of it when nothing else is competing. Size this to your\n"
        "machine's own core count (also its default), not RAM.",
        foreground="#888888", justify="left",
    ).grid(row=4, column=1, sticky="w")

    def on_ok():
        try:
            cpu_budget = int(cpu_budget_var.get())
            if cpu_budget < 1:
                raise ValueError
        except ValueError:
            messagebox.showerror("Docker Settings", "CPU budget must be a positive integer.")
            return
        if not image_var.get().strip():
            messagebox.showerror("Docker Settings", "Image must not be empty.")
            return
        if "{name}" not in template_var.get():
            messagebox.showerror("Docker Settings", "Container mount path template must contain {name}.")
            return
        settings.save({
            "container": {
                "image": image_var.get().strip(),
                "project_root_template": template_var.get().strip(),
                "cpu_budget": cpu_budget,
            },
        })
        win.destroy()

    def on_cancel():
        win.destroy()

    buttons = ttk.Frame(body)
    buttons.pack(fill="x", pady=(12, 0))
    ttk.Button(buttons, text="Cancel", command=on_cancel).pack(side="right")
    ttk.Button(buttons, text="Save", command=on_ok).pack(side="right", padx=(0, 8))

    win.protocol("WM_DELETE_WINDOW", on_cancel)
    _center_on_parent(win, parent)
    win.grab_set()
    win.wait_window()
