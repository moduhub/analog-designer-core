"""Pan/zoom raster viewer for the layout PNG snapshots analog_designer.layout.gen_layout
produces (floorplan_snapshot.png, layout_snapshot.png, the xschem
schematic snapshot) -- replaces the old static ttk.Label display, which
showed a fixed image with no way to inspect fine detail (individual vias,
via_stack contact arrays -- confirmed sub-1um in this PDK, see
analog_designer.layout.gen_layout's own _SNAPSHOT_PX_PER_UM comment) beyond squinting at a
480px-wide thumbnail.

Uses Pillow (PIL.Image/ImageTk), not the bare tk.PhotoImage the rest of
this GUI otherwise gets away without needing (see layout_view.py's old
_load_photo docstring) -- tk.PhotoImage's own zoom()/subsample() are
integer-only nearest-neighbor scalers, not real resampling, so they can't
give a smooth zoom; Pillow's LANCZOS resize can. Pillow is already
available on this host outside the project's own docker-only
gdsfactory/klayout stack, so this doesn't add a new heavy dependency.

Zoom never goes below "fit the whole image in the canvas" (the initial
view) -- there's nothing useful below that for a layout snapshot, and it
keeps the pan-offset math simple (the visible region is always <= the
image's own size, no need to handle showing empty space around a
smaller-than-canvas image).
"""
import tkinter as tk
from tkinter import ttk

from PIL import Image, ImageTk

_ZOOM_STEP = 1.15
_MAX_ZOOM_OVER_FIT = 20.0


class ZoomableImageView(ttk.Frame):
    def __init__(self, master, popout=True):
        """popout=False for the view INSIDE a pop-out window itself (see
        _pop_out below) -- its own toolbar has no further "Pop out" button
        to open yet another copy of the same image, that would just nest
        identical windows with no benefit."""
        super().__init__(master)
        self._pil_image = None
        self._photo = None  # kept alive -- Tk drops a PhotoImage the moment nothing references it
        self._path = None  # kept so _pop_out() can reload the same image into a fresh view
        self._zoom = 1.0
        self._offset_x = 0.0
        self._offset_y = 0.0
        self._drag_start = None
        self._drag_offset_start = None

        self.canvas = tk.Canvas(self, background="#1a1a1a", highlightthickness=0)
        self.canvas.pack(fill="both", expand=True)
        self._empty_text_id = self.canvas.create_text(
            10, 10, anchor="nw", fill="#cccccc", text="(not generated yet)",
        )

        toolbar = ttk.Frame(self)
        toolbar.pack(side="bottom", fill="x")
        ttk.Button(toolbar, text="Fit", command=self.fit, width=6).pack(side="left", padx=(0, 4), pady=2)
        self.zoom_label_var = tk.StringVar(value="")
        ttk.Label(toolbar, textvariable=self.zoom_label_var, foreground="#888888").pack(side="left")
        if popout:
            ttk.Button(toolbar, text="Pop out", command=self._pop_out, width=8).pack(side="right", padx=(4, 0), pady=2)

        self.canvas.bind("<Configure>", self._on_resize)
        self.canvas.bind("<MouseWheel>", self._on_wheel)  # Windows/macOS
        self.canvas.bind("<Button-4>", self._on_wheel)  # X11 scroll up
        self.canvas.bind("<Button-5>", self._on_wheel)  # X11 scroll down
        self.canvas.bind("<ButtonPress-1>", self._on_press)
        self.canvas.bind("<B1-Motion>", self._on_drag)
        self.canvas.bind("<Double-Button-1>", lambda _event: self._pop_out())

    def clear(self, message="(not generated yet)"):
        self._pil_image = None
        self._photo = None
        self._path = None
        self.canvas.delete("all")
        self._empty_text_id = self.canvas.create_text(10, 10, anchor="nw", fill="#cccccc", text=message)
        self.zoom_label_var.set("")

    def set_image(self, path):
        """Loads path (a real PNG on disk -- never held in memory beyond
        this one load) and resets to a fit-to-canvas view."""
        try:
            self._pil_image = Image.open(path)
            self._pil_image.load()  # force the read now, while `path` is still known good
        except Exception:  # noqa: BLE001 -- a missing/corrupt snapshot shouldn't crash the GUI
            self.clear("(snapshot unavailable)")
            return
        self._path = path
        self.fit()

    def _pop_out(self):
        """Opens the SAME image, freshly loaded, in its own resizable
        Toplevel window at a larger default size -- lets someone inspect a
        plot/snapshot full-size without it fighting the embedding panel's
        own (often narrow) layout. Reloads from `self._path` rather than
        reusing self._pil_image/self._photo -- a Tk PhotoImage can't be
        shared across two Canvas widgets reliably, and the source PNG is
        cheap to re-read. A no-op (double-click/button click does nothing)
        when there's no image loaded yet -- nothing useful to pop out."""
        if self._path is None:
            return
        win = tk.Toplevel(self)
        win.title(str(self._path))
        win.geometry("900x700")
        view = ZoomableImageView(win, popout=False)
        view.pack(fill="both", expand=True)
        view.set_image(self._path)

    def _fit_zoom(self):
        if self._pil_image is None:
            return 1.0
        canvas_w = max(1, self.canvas.winfo_width())
        canvas_h = max(1, self.canvas.winfo_height())
        img_w, img_h = self._pil_image.size
        return min(canvas_w / img_w, canvas_h / img_h)

    def fit(self):
        if self._pil_image is None:
            return
        self._zoom = self._fit_zoom()
        self._offset_x = 0.0
        self._offset_y = 0.0
        self._redraw()

    def _on_resize(self, _event):
        if self._pil_image is None:
            return
        # Re-fitting on every resize (rather than preserving the current
        # zoom/pan) keeps this simple and matches what a user expects when
        # they resize the window mid-inspection -- clamping the existing
        # pan to the new canvas size would be extra bookkeeping for a
        # marginal benefit.
        self.fit()

    def _clamp_offset(self):
        img_w, img_h = self._pil_image.size
        canvas_w = max(1, self.canvas.winfo_width())
        canvas_h = max(1, self.canvas.winfo_height())
        vis_w = canvas_w / self._zoom
        vis_h = canvas_h / self._zoom
        self._offset_x = min(max(self._offset_x, 0.0), max(0.0, img_w - vis_w))
        self._offset_y = min(max(self._offset_y, 0.0), max(0.0, img_h - vis_h))

    def _on_wheel(self, event):
        if self._pil_image is None:
            return
        zooming_in = getattr(event, "delta", 0) > 0 or event.num == 4
        factor = _ZOOM_STEP if zooming_in else (1.0 / _ZOOM_STEP)
        fit_zoom = self._fit_zoom()
        old_zoom = self._zoom
        new_zoom = max(fit_zoom, min(old_zoom * factor, fit_zoom * _MAX_ZOOM_OVER_FIT))
        if new_zoom == old_zoom:
            return
        # Keep the image point under the cursor fixed across the zoom
        # change -- standard "zoom to cursor" feel, not just zoom-to-center.
        img_x = self._offset_x + event.x / old_zoom
        img_y = self._offset_y + event.y / old_zoom
        self._zoom = new_zoom
        self._offset_x = img_x - event.x / new_zoom
        self._offset_y = img_y - event.y / new_zoom
        self._clamp_offset()
        self._redraw()

    def _on_press(self, event):
        if self._pil_image is None:
            return
        self._drag_start = (event.x, event.y)
        self._drag_offset_start = (self._offset_x, self._offset_y)

    def _on_drag(self, event):
        if self._pil_image is None or self._drag_start is None:
            return
        dx = event.x - self._drag_start[0]
        dy = event.y - self._drag_start[1]
        self._offset_x = self._drag_offset_start[0] - dx / self._zoom
        self._offset_y = self._drag_offset_start[1] - dy / self._zoom
        self._clamp_offset()
        self._redraw()

    def _redraw(self):
        if self._pil_image is None:
            return
        canvas_w = max(1, self.canvas.winfo_width())
        canvas_h = max(1, self.canvas.winfo_height())
        img_w, img_h = self._pil_image.size

        vis_w = min(img_w, canvas_w / self._zoom)
        vis_h = min(img_h, canvas_h / self._zoom)
        x0, y0 = self._offset_x, self._offset_y
        crop = self._pil_image.crop((int(x0), int(y0), int(x0 + vis_w), int(y0 + vis_h)))
        if crop.width == 0 or crop.height == 0:
            return
        resized = crop.resize(
            (max(1, round(crop.width * self._zoom)), max(1, round(crop.height * self._zoom))), Image.LANCZOS,
        )
        self._photo = ImageTk.PhotoImage(resized)
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor="nw", image=self._photo)
        self.zoom_label_var.set(f"{self._zoom / self._fit_zoom():.1f}x")
