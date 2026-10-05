"""Modal "Simulation Settings" dialog: edit the global, machine-level
analog_designer.core.settings values -- where simulators run (a docker
container of the project's image, or this machine's own tools; see
analog_designer/core/executor.py), the docker image and mount-path
template, the host-mode PDK, and how many CPU cores this host budgets for
simulation work -- same modal Toplevel + ttk.Entry/OK-Cancel shape as
analog_designer.gui.schematic_editor.config_dialog. Not project-scoped
(these are facts about this machine, not any one design), so the caller may
show it regardless of whether a background job is running: a change applies
from the next job on.

(Module and function keep their old "docker settings" names so callers --
including the pro GUI -- don't break.)
"""
import os
import shutil
import tkinter as tk
from tkinter import messagebox, ttk

from analog_designer.core import settings

TITLE = "Simulation Settings"
_HINT = "#888888"


def _center_on_parent(win, parent):
    win.update_idletasks()
    x = parent.winfo_rootx() + (parent.winfo_width() - win.winfo_width()) // 2
    y = parent.winfo_rooty() + (parent.winfo_height() - win.winfo_height()) // 2
    win.geometry(f"+{max(x, 0)}+{max(y, 0)}")


def validate(mode, image, template, cpu_budget):
    """Error message for the first invalid value, else None -- a plain
    function so it's testable without a display."""
    try:
        if int(cpu_budget) < 1:
            raise ValueError
    except ValueError:
        return "CPU budget must be a positive integer."
    if mode not in settings.EXECUTION_MODES:
        return f"Unknown execution mode {mode!r}."
    if mode == "docker":
        if not image.strip():
            return "Image must not be empty."
        if "{name}" not in template:
            return "Container mount path template must contain {name}."
    elif shutil.which("bash") is None:
        return "Running on this host needs bash (Linux, macOS, WSL or the EDA image itself)."
    return None


def show_docker_settings(parent):
    """Blocks (grab_set + wait_window) until closed. Saves to
    ~/.mh-analog-designer/settings.json on Save; does nothing on Cancel."""
    current = settings.load()

    win = tk.Toplevel(parent)
    win.title(TITLE)
    win.resizable(False, False)
    win.transient(parent)

    body = ttk.Frame(win, padding=12)
    body.pack(fill="both", expand=True)

    mode_var = tk.StringVar(value=current["execution"]["mode"])
    image_var = tk.StringVar(value=current["container"]["image"])
    template_var = tk.StringVar(value=current["container"]["project_root_template"])
    pdk_root_var = tk.StringVar(value=current["host"].get("pdk_root", ""))
    pdk_var = tk.StringVar(value=current["host"].get("pdk", ""))
    cpu_budget_var = tk.StringVar(value=str(current["container"]["cpu_budget"]))

    ttk.Label(body, text="Run simulations").pack(anchor="w")
    modes = ttk.Frame(body)
    modes.pack(fill="x", pady=(2, 8))
    ttk.Radiobutton(modes, text="in a Docker container", value="docker", variable=mode_var).pack(side="left")
    ttk.Radiobutton(modes, text="on this host", value="host", variable=mode_var).pack(side="left", padx=(16, 0))

    docker_box = ttk.LabelFrame(body, text="Docker", padding=8)
    docker_box.pack(fill="x")
    ttk.Label(docker_box, text="Image").grid(row=0, column=0, sticky="w", pady=2)
    image_entry = ttk.Entry(docker_box, textvariable=image_var, width=36)
    image_entry.grid(row=0, column=1, sticky="w", pady=2)
    ttk.Label(docker_box, text="Container mount path template").grid(row=1, column=0, sticky="w", pady=2)
    template_entry = ttk.Entry(docker_box, textvariable=template_var, width=36)
    template_entry.grid(row=1, column=1, sticky="w", pady=2)
    ttk.Label(
        docker_box, text="{name} is replaced with the host project folder's name.\n"
        "A project's config.json container.image overrides the image.", foreground=_HINT, justify="left",
    ).grid(row=2, column=1, sticky="w")

    host_box = ttk.LabelFrame(body, text="Host", padding=8)
    host_box.pack(fill="x", pady=(8, 0))
    ttk.Label(host_box, text="PDK root").grid(row=0, column=0, sticky="w", pady=2)
    pdk_root_entry = ttk.Entry(host_box, textvariable=pdk_root_var, width=36)
    pdk_root_entry.grid(row=0, column=1, sticky="w", pady=2)
    ttk.Label(host_box, text="PDK").grid(row=1, column=0, sticky="w", pady=2)
    pdk_entry = ttk.Entry(host_box, textvariable=pdk_var, width=36)
    pdk_entry.grid(row=1, column=1, sticky="w", pady=2)
    env_note = ", ".join(f"${k}={os.environ[k]}" for k in ("PDK_ROOT", "PDK") if os.environ.get(k)) or "not set here"
    ttk.Label(
        host_box, text="Empty: $PDK_ROOT / $PDK from the environment (" + env_note + "),\n"
        "and the PDK name defaults to the tag of the project's container.image.\n"
        "xschem, ngspice and Xyce come from PATH ($XSCHEM/$NGSPICE/$XYCE override).",
        foreground=_HINT, justify="left",
    ).grid(row=2, column=1, sticky="w")

    def _sync_state(*_):
        docker = mode_var.get() == "docker"
        for entry in (image_entry, template_entry):
            entry.configure(state="normal" if docker else "disabled")
        for entry in (pdk_root_entry, pdk_entry):
            entry.configure(state="disabled" if docker else "normal")
    mode_var.trace_add("write", _sync_state)
    _sync_state()

    grid = ttk.Frame(body)
    grid.pack(fill="x", pady=(10, 0))
    ttk.Label(grid, text="CPU budget (cores)").grid(row=0, column=0, sticky="w", pady=2)
    ttk.Entry(grid, textvariable=cpu_budget_var, width=8).grid(row=0, column=1, sticky="w", pady=2)
    ttk.Label(
        grid, text="Total cores this host spends on simulation work at once. Each job\n"
        "(a test condition, or a whole variation) requests as many cores as it\n"
        "needs and gets whatever's currently free -- many cheap ngspice/Xyce\n"
        "runs share this budget 1 core each, while a single expensive run can\n"
        "claim most of it when nothing else is competing. Size this to your\n"
        "machine's own core count (also its default), not RAM.",
        foreground=_HINT, justify="left",
    ).grid(row=1, column=1, sticky="w")

    def on_ok():
        error = validate(mode_var.get(), image_var.get(), template_var.get(), cpu_budget_var.get())
        if error:
            messagebox.showerror(TITLE, error, parent=win)
            return
        updated = settings.load()
        updated["execution"]["mode"] = mode_var.get()
        updated["container"].update({
            "image": image_var.get().strip(),
            "project_root_template": template_var.get().strip(),
            "cpu_budget": int(cpu_budget_var.get()),
        })
        updated["host"].update({"pdk_root": pdk_root_var.get().strip(), "pdk": pdk_var.get().strip()})
        settings.save(updated)
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


show_simulation_settings = show_docker_settings
