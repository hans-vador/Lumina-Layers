"""Art-directed relief editor - select part of an album cover and change its shape.

    cd ~/Lumina-Layers && .venv/bin/python scripts/relief_editor.py cover.jpg --width 150

Left: the cover on the relief grid with the selection overlaid (cyan) and the
SAM points (green = object, red = not this).  Right: the final relief as grey
(black = 0 mm, white = 4 mm) with the selection outline.

Selection
* Click object (SAM): left-click on the object, right-click / shift-click to mark
  what is NOT the object; every click refines the same selection.  SAM (Segment
  Anything, local, optional) must be fetched once with
  ``scripts/convert_relief.py --fetch-sam-model``; without it the brush works.
* Brush add / remove: paint the selection (size in mm on the plaque).

Shape edits (applied to the selection, geometry only - the Stack5 colours stay
the original image's):  Dome (hemisphere, apex at Height, Roundness 1 = sphere),
Extrude (flat plateau at Height), Raise / lower (signed Amount), Smooth edge
(blend across the outline over Feather), Reset (back to the Depth Anything relief).
Apply / Undo / Reset all.  "Use these edits" hands the edit script to the caller
(the Album Plaque Maker window passes it to the export); "Save edits..." writes
``<image>_edits.json`` for ``scripts/convert_relief.py --edits``.

The relief is a float32 height map in mm, clamped to 0..4 mm; the export
quantises it to 0.08 mm layers and keeps the 0.8 mm base and 0.4 mm colour shell.
"""
from __future__ import annotations

import argparse
import json
import os
import queue
import sys
import threading
import time
import traceback
from typing import Optional

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

import tkinter as tk  # noqa: E402
from tkinter import filedialog, messagebox, ttk  # noqa: E402

from PIL import Image, ImageTk  # noqa: E402

OPS_UI = {
    'Dome (hemisphere inside the selection)': 'dome',
    'Extrude (flat plateau at Height)': 'extrude',
    'Raise / lower (by Amount)': 'offset',
    'Smooth edge (blend across the outline)': 'feather',
    'Reset selection to the AI relief': 'reset',
}
SIZE_CHOICES = {'Auto (best score)': 'best', 'Small': 'small', 'Medium': 'medium', 'Large': 'large'}
CANVAS_MAX = 460
MASK_COLOUR = (0, 200, 255)
OUTLINE_COLOUR = (255, 140, 0)
DEFAULT_BRUSH_MM = 3.0


# --------------------------------------------------------------------------- helpers usable without a window
def synthetic_base(H: int = 120, W: int = 160, seed: int = 0) -> dict:
    """A small stand-in for core.relief.art_direct.relief_for_editor (tests / --selftest):
    a ramp with a raised disc, no Depth Anything needed."""
    from core.relief.art_direct import original_mm_from_field
    from core.relief.depth_processing import DepthParams, process_depth
    from core.relief.provider import GeometryResult
    from core.relief.relief_stack import ReliefDims
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:H, 0:W]
    depth = (yy / max(H - 1, 1)).astype(np.float32)
    disc = (xx - 0.6 * W) ** 2 + (yy - 0.5 * H) ** 2 < (0.2 * min(H, W)) ** 2
    depth[disc] += 0.3
    geom = GeometryResult(depth=depth, source='synthetic')
    dims, params = ReliefDims(), DepthParams()
    plaque = np.ones((H, W), bool)
    field = process_depth(geom, (H, W), dims, params, plaque_mask=plaque)
    rgb = np.stack([(xx * 255 / max(W - 1, 1)), (yy * 255 / max(H - 1, 1)), np.full((H, W), 120.0)], -1)
    rgb[disc] = (230, 230, 230)
    rgb = np.clip(rgb + rng.normal(0, 3, rgb.shape), 0, 255).astype(np.uint8)
    return {'field': field, 'geom': geom, 'plaque_mask': plaque, 'grid_hw': (H, W), 'pixel_mm': 0.1,
            'rgb_grid': rgb, 'image_hw': (H, W), 'original_mm': original_mm_from_field(field),
            'relief_max_mm': dims.relief_layers * dims.layer_h, 'dims': dims, 'params': params,
            'geometry': 'synthetic', 'depth_map': None}


class DemoMaskProvider:
    """Stand-in mask provider (tests / --selftest): a disc around the first
    positive point, in three sizes."""
    name = 'demo'

    def __init__(self, radius_frac: float = 0.15):
        self.radius_frac = float(radius_frac)
        self.image_hw = None

    def available(self):
        return True, 'demo mask provider (disc around the click)'

    def set_image(self, rgb):
        self.image_hw = (int(rgb.shape[0]), int(rgb.shape[1]))

    def predict(self, points_xy, labels):
        from core.relief.segment import MaskProposal
        H, W = self.image_hw
        pos = [p for p, l in zip(points_xy, labels) if int(l) == 1] or list(points_xy)
        cx, cy = float(pos[0][0]), float(pos[0][1])
        yy, xx = np.mgrid[0:H, 0:W]
        r0 = self.radius_frac * min(H, W)
        masks = np.stack([(xx - cx) ** 2 + (yy - cy) ** 2 <= (r0 * f) ** 2 for f in (0.7, 1.0, 1.4)])
        for p, l in zip(points_xy, labels):
            if int(l) == 0:
                masks &= ~((xx - p[0]) ** 2 + (yy - p[1]) ** 2 <= (0.3 * r0) ** 2)[None]
        return MaskProposal(masks=masks, scores=np.array([0.6, 0.9, 0.7]), points=list(points_xy), labels=list(labels),
                            source=self.name)


def _fit_scale(H: int, W: int, box: int = CANVAS_MAX) -> float:
    return min(box / max(W, 1), box / max(H, 1), 1.0) if max(H, W) > box else min(box / max(W, 1), box / max(H, 1))


# --------------------------------------------------------------------------- the window
class ReliefEditor(tk.Toplevel):
    """Toplevel editor.  ``on_done(script, geometry_result)`` is called from
    "Use these edits"; ``base`` (relief_for_editor output) skips the depth
    estimation; ``mask_provider`` overrides SAM (tests)."""

    def __init__(self, master, image_path: str, width_mm: float, *, dims=None, params=None, depth_source: str = 'ai',
                 depth_map: Optional[str] = None, provider_options: Optional[dict] = None,
                 sam_weights: Optional[str] = None, on_done=None, initial_script=None, base: Optional[dict] = None,
                 mask_provider=None, title: Optional[str] = None):
        super().__init__(master)
        from core.relief.depth_processing import DepthParams
        from core.relief.relief_stack import ReliefDims
        self.image_path = os.path.abspath(image_path) if image_path else ''
        self.width_mm = float(width_mm)
        self.dims = dims or ReliefDims()
        self.params = params or DepthParams()
        self.depth_source = depth_source
        self.depth_map = depth_map
        self.provider_options = dict(provider_options or {})
        self.sam_weights = sam_weights
        self.on_done = on_done
        self.initial_script = initial_script
        self.title(title or f"Art-direct relief - {os.path.basename(self.image_path) or 'synthetic'}")
        self.q: queue.Queue = queue.Queue()
        self.base: Optional[dict] = None
        self.session = None
        self.mask: Optional[np.ndarray] = None
        self.points: list[tuple[float, float, int]] = []          # image-pixel coordinates
        self.proposal = None
        self.mask_provider = mask_provider
        self.sam_ready = False
        self.sam_message = ''
        self.result_script = None
        self._stroke_last = None
        self._render_pending = False
        self._photo_left = self._photo_right = None
        self._img_disp = None
        self.scale = 1.0
        self.disp_hw = (1, 1)
        self._build()
        self.protocol('WM_DELETE_WINDOW', self.cancel)
        if base is not None:
            self._base_ready(base)
        else:
            self._start_base()
        self.after(120, self._poll)

    # ------------------------------------------------------------------ layout
    def _build(self):
        pad = {'padx': 6, 'pady': 3}
        f = ttk.Frame(self, padding=8)
        f.grid(sticky='nsew')
        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)
        self.status = tk.StringVar(value='Estimating the relief ...')
        ttk.Label(f, textvariable=self.status, wraplength=1100).grid(row=0, column=0, columnspan=3, sticky='w', **pad)

        ttk.Label(f, text='Cover + selection (cyan).  Left-click = object, right/shift-click = not this, or paint.',
                  foreground='#555').grid(row=1, column=0, sticky='w', **pad)
        ttk.Label(f, text='Final relief height: black = 0 mm, white = relief max.', foreground='#555').grid(
            row=1, column=1, sticky='w', **pad)
        self.cv_left = tk.Canvas(f, width=CANVAS_MAX, height=CANVAS_MAX, bg='#222', highlightthickness=0, cursor='crosshair')
        self.cv_left.grid(row=2, column=0, sticky='n', **pad)
        self.cv_right = tk.Canvas(f, width=CANVAS_MAX, height=CANVAS_MAX, bg='#222', highlightthickness=0)
        self.cv_right.grid(row=2, column=1, sticky='n', **pad)
        self.cv_left.bind('<Button-1>', self._on_left_down)
        self.cv_left.bind('<Shift-Button-1>', self._on_negative_click)
        self.cv_left.bind('<Button-2>', self._on_negative_click)
        self.cv_left.bind('<Button-3>', self._on_negative_click)
        self.cv_left.bind('<B1-Motion>', self._on_drag)
        self.cv_left.bind('<ButtonRelease-1>', self._on_release)
        self.cv_left.bind('<Motion>', self._on_motion)
        self.cv_left.bind('<Leave>', lambda e: self.cv_left.delete('cursor'))

        ctl = ttk.Frame(f)
        ctl.grid(row=1, column=2, rowspan=2, sticky='ns', **pad)

        sel = ttk.LabelFrame(ctl, text='Selection', padding=(8, 4))
        sel.pack(fill='x')
        self.tool = tk.StringVar(value='click')
        ttk.Radiobutton(sel, text='Click object (SAM)', variable=self.tool, value='click').grid(row=0, column=0, columnspan=2, sticky='w')
        ttk.Radiobutton(sel, text='Brush add', variable=self.tool, value='add').grid(row=1, column=0, sticky='w')
        ttk.Radiobutton(sel, text='Brush remove', variable=self.tool, value='sub').grid(row=1, column=1, sticky='w')
        ttk.Label(sel, text='Brush size (mm)').grid(row=2, column=0, sticky='w', pady=(4, 0))
        self.brush_mm = tk.StringVar(value=f'{DEFAULT_BRUSH_MM:g}')
        ttk.Spinbox(sel, from_=0.3, to=30, increment=0.5, textvariable=self.brush_mm, width=6).grid(row=2, column=1, sticky='w', pady=(4, 0))
        ttk.Label(sel, text='SAM result size').grid(row=3, column=0, sticky='w', pady=(4, 0))
        self.size_choice = ttk.Combobox(sel, values=list(SIZE_CHOICES), state='readonly', width=16)
        self.size_choice.set(next(iter(SIZE_CHOICES)))
        self.size_choice.grid(row=3, column=1, sticky='w', pady=(4, 0))
        self.size_choice.bind('<<ComboboxSelected>>', lambda *_: self._apply_proposal())
        row = ttk.Frame(sel)
        row.grid(row=4, column=0, columnspan=2, sticky='w', pady=(6, 0))
        ttk.Button(row, text='Undo click', command=self.undo_click).pack(side='left')
        ttk.Button(row, text='Clear', command=self.clear_selection).pack(side='left', padx=4)
        ttk.Button(row, text='Fill holes', command=self.fill_holes).pack(side='left')
        self.sam_status = ttk.Label(sel, text='', foreground='#666', wraplength=280)
        self.sam_status.grid(row=5, column=0, columnspan=2, sticky='w', pady=(6, 0))
        self.sel_info = ttk.Label(sel, text='no selection', wraplength=280)
        self.sel_info.grid(row=6, column=0, columnspan=2, sticky='w', pady=(4, 0))

        shp = ttk.LabelFrame(ctl, text='Shape of the selection', padding=(8, 4))
        shp.pack(fill='x', pady=(8, 0))
        ttk.Label(shp, text='Operation').grid(row=0, column=0, sticky='w')
        self.op_choice = ttk.Combobox(shp, values=list(OPS_UI), state='readonly', width=34)
        self.op_choice.set(next(iter(OPS_UI)))
        self.op_choice.grid(row=0, column=1, columnspan=3, sticky='w', padx=(4, 0))
        self.op_choice.bind('<<ComboboxSelected>>', lambda *_: self._on_op_change())
        ttk.Label(shp, text='Height (mm)').grid(row=1, column=0, sticky='w', pady=(6, 0))
        self.height_mm = tk.StringVar(value='4.0')
        self.height_entry = ttk.Entry(shp, textvariable=self.height_mm, width=7)
        self.height_entry.grid(row=1, column=1, sticky='w', padx=(4, 10), pady=(6, 0))
        ttk.Label(shp, text='Dome roundness').grid(row=1, column=2, sticky='w', pady=(6, 0))
        self.roundness = tk.StringVar(value='1.0')
        self.round_entry = ttk.Entry(shp, textvariable=self.roundness, width=7)
        self.round_entry.grid(row=1, column=3, sticky='w', padx=(4, 0), pady=(6, 0))
        ttk.Label(shp, text='Edge feather (mm)').grid(row=2, column=0, sticky='w', pady=(6, 0))
        self.feather_mm = tk.StringVar(value='0.5')
        self.feather_entry = ttk.Entry(shp, textvariable=self.feather_mm, width=7)
        self.feather_entry.grid(row=2, column=1, sticky='w', padx=(4, 10), pady=(6, 0))
        ttk.Label(shp, text='Raise / lower (mm)').grid(row=2, column=2, sticky='w', pady=(6, 0))
        self.delta_mm = tk.StringVar(value='1.0')
        self.delta_entry = ttk.Entry(shp, textvariable=self.delta_mm, width=7)
        self.delta_entry.grid(row=2, column=3, sticky='w', padx=(4, 0), pady=(6, 0))
        row = ttk.Frame(shp)
        row.grid(row=3, column=0, columnspan=4, sticky='w', pady=(8, 0))
        self.apply_btn = ttk.Button(row, text='Apply', command=self.apply_edit)
        self.apply_btn.pack(side='left')
        self.undo_btn = ttk.Button(row, text='Undo', command=self.undo_edit)
        self.undo_btn.pack(side='left', padx=4)
        ttk.Button(row, text='Reset all', command=self.reset_all).pack(side='left')
        self.edit_info = ttk.Label(shp, text='', wraplength=300)
        self.edit_info.grid(row=4, column=0, columnspan=4, sticky='w', pady=(6, 0))
        ttk.Label(shp, text='Heights are above the flat base (0..relief max); the 0.8 mm base and the 0.4 mm '
                            'colour shell are added at export.  Colours always come from the cover.',
                  foreground='#666', wraplength=300).grid(row=5, column=0, columnspan=4, sticky='w', pady=(6, 0))
        self._on_op_change()

        bottom = ttk.Frame(f)
        bottom.grid(row=3, column=0, columnspan=3, sticky='ew', **pad)
        ttk.Button(bottom, text='Save edits...', command=self.save_edits).pack(side='left')
        ttk.Button(bottom, text='Load edits...', command=self.load_edits).pack(side='left', padx=6)
        self.done_btn = ttk.Button(bottom, text='Use these edits', command=self.done, state='disabled')
        self.done_btn.pack(side='right')
        ttk.Button(bottom, text='Cancel', command=self.cancel).pack(side='right', padx=6)

    def _on_op_change(self):
        op = OPS_UI[self.op_choice.get()]
        state = lambda on: 'normal' if on else 'disabled'  # noqa: E731
        self.height_entry.configure(state=state(op in ('dome', 'extrude')))
        self.round_entry.configure(state=state(op == 'dome'))
        self.delta_entry.configure(state=state(op == 'offset'))
        self.feather_entry.configure(state=state(op != 'reset'))

    # ------------------------------------------------------------------ background work
    def _start_base(self):
        def work():
            try:
                from core.relief.art_direct import relief_for_editor
                geometry = 'imported-depth' if self.depth_source == 'file' else 'depth-anything'
                opts = dict(self.provider_options)
                if geometry == 'depth-anything':
                    opts.setdefault('device', 'cpu')
                    opts.setdefault('allow_download', False)
                base = relief_for_editor(self.image_path, self.width_mm, dims=self.dims, params=self.params,
                                         geometry=geometry, depth_map=self.depth_map, provider_options=opts)
                self.q.put(('base', base))
            except Exception as exc:  # noqa: BLE001 - shown in the window
                traceback.print_exc()
                self.q.put(('error', f'{type(exc).__name__}: {exc}'))
        threading.Thread(target=work, daemon=True).start()

    def _start_sam(self):
        rgb_full = None
        try:
            if self.image_path and os.path.isfile(self.image_path):
                rgb_full = np.asarray(Image.open(self.image_path).convert('RGB'))
        except Exception:  # noqa: BLE001
            rgb_full = None
        if rgb_full is None:
            rgb_full = self.base['rgb_grid']
        self.sam_image_hw = (int(rgb_full.shape[0]), int(rgb_full.shape[1]))

        def work():
            try:
                prov = self.mask_provider
                if prov is None:
                    from core.relief.segment import SamMaskProvider
                    prov = SamMaskProvider(weights=self.sam_weights, allow_download=False)
                ok, msg = prov.available()
                if not ok:
                    self.q.put(('sam_unavailable', msg))
                    return
                self.q.put(('sam_loading', msg))
                prov.set_image(rgb_full)
                self.mask_provider = prov
                self.q.put(('sam_ready', msg))
            except Exception as exc:  # noqa: BLE001
                traceback.print_exc()
                self.q.put(('sam_unavailable', f'{type(exc).__name__}: {exc}'))
        threading.Thread(target=work, daemon=True).start()

    def _poll(self):
        try:
            while True:
                kind, payload = self.q.get_nowait()
                if kind == 'base':
                    self._base_ready(payload)
                elif kind == 'error':
                    self.status.set('FAILED: ' + payload)
                    messagebox.showerror('Art-direct relief', payload, parent=self)
                elif kind == 'sam_loading':
                    self.sam_status.configure(text='SAM: loading the model and reading the cover ...')
                elif kind == 'sam_ready':
                    self.sam_ready = True
                    self.sam_message = payload
                    self.sam_status.configure(text='SAM ready - click the object. ' + payload)
                elif kind == 'sam_unavailable':
                    self.sam_ready = False
                    self.sam_message = payload
                    self.sam_status.configure(text='Click-to-select is off: ' + payload)
                    if self.tool.get() == 'click':
                        self.tool.set('add')
        except queue.Empty:
            pass
        if self.winfo_exists():
            self.after(120, self._poll)

    def _base_ready(self, base: dict):
        from core.relief.art_direct import EditSession
        self.base = base
        H, W = base['grid_hw']
        self.session = EditSession(base['original_mm'], base['pixel_mm'], base['relief_max_mm'])
        self.mask = np.zeros((H, W), bool)
        self.scale = _fit_scale(H, W)
        dw, dh = max(1, int(round(W * self.scale))), max(1, int(round(H * self.scale)))
        self.disp_hw = (dh, dw)
        self._img_disp = Image.fromarray(np.ascontiguousarray(base['rgb_grid'][..., :3])).resize((dw, dh), Image.Resampling.LANCZOS)
        self.cv_left.configure(width=dw, height=dh)
        self.cv_right.configure(width=dw, height=dh)
        st = base['field'].report.get('stats', {})
        self.status.set(f"Relief grid {W} x {H} px (0.1 mm/px, {self.width_mm:g} mm wide); AI relief "
                        f"{st.get('relief_mm_min', 0):.2f}..{st.get('relief_mm_max', 0):.2f} mm; edits clamp to "
                        f"0..{base['relief_max_mm']:g} mm.  Select the object, choose a shape, Apply.")
        self.done_btn.configure(state='normal')
        if self.initial_script is not None:
            try:
                from core.relief.art_direct import EditScript
                self.session.load_script(EditScript.coerce(self.initial_script))
            except Exception as exc:  # noqa: BLE001
                messagebox.showwarning('Art-direct relief', f'Could not replay the previous edits: {exc}', parent=self)
        self._render()
        self._update_edit_info()
        self._start_sam()

    # ------------------------------------------------------------------ coordinates
    def _to_grid(self, ex: float, ey: float) -> tuple[float, float]:
        return (ex + 0.5) / self.scale - 0.5, (ey + 0.5) / self.scale - 0.5

    def _in_grid(self, gx: float, gy: float) -> bool:
        H, W = self.base['grid_hw']
        return -0.5 <= gx < W - 0.5 and -0.5 <= gy < H - 0.5

    def _brush_px(self) -> float:
        try:
            mm = float(self.brush_mm.get())
        except ValueError:
            mm = DEFAULT_BRUSH_MM
        return max(0.5, mm / 2.0 / self.base['pixel_mm'])

    # ------------------------------------------------------------------ mouse
    def _on_left_down(self, ev):
        if self.base is None:
            return
        gx, gy = self._to_grid(ev.x, ev.y)
        if not self._in_grid(gx, gy):
            return
        if self.tool.get() == 'click':
            self._sam_click(gx, gy, 1)
        else:
            self._stroke_last = (gx, gy)
            self._paint(gx, gy)

    def _on_negative_click(self, ev):
        if self.base is None or self.tool.get() != 'click':
            return
        gx, gy = self._to_grid(ev.x, ev.y)
        if self._in_grid(gx, gy):
            self._sam_click(gx, gy, 0)
        return 'break'

    def _on_drag(self, ev):
        if self.base is None or self.tool.get() == 'click':
            return
        gx, gy = self._to_grid(ev.x, ev.y)
        self._paint(gx, gy, from_pt=self._stroke_last)
        self._stroke_last = (gx, gy)

    def _on_release(self, _ev):
        self._stroke_last = None
        if self.base is not None:
            self._update_sel_info()

    def _on_motion(self, ev):
        if self.base is None or self.tool.get() == 'click':
            self.cv_left.delete('cursor')
            return
        r = self._brush_px() * self.scale
        self.cv_left.delete('cursor')
        self.cv_left.create_oval(ev.x - r, ev.y - r, ev.x + r, ev.y + r, outline='#ffffff', tags='cursor')

    # ------------------------------------------------------------------ selection
    def _paint(self, gx: float, gy: float, from_pt=None):
        from core.relief.segment import paint_disc, paint_stroke
        add = self.tool.get() == 'add'
        r = self._brush_px()
        if from_pt is not None:
            paint_stroke(self.mask, from_pt, (gx, gy), r, add)
        else:
            paint_disc(self.mask, gx, gy, r, add)
        self.mask &= self.base['plaque_mask']
        self._schedule_render()

    def _sam_click(self, gx: float, gy: float, label: int):
        if not self.sam_ready or self.mask_provider is None:
            messagebox.showinfo('Art-direct relief', 'Click-to-select is not available:\n\n' + (self.sam_message or 'SAM is still loading.')
                                + '\n\nUse the brush instead.', parent=self)
            return
        from core.relief.art_direct import grid_to_image_xy
        ix, iy = grid_to_image_xy(gx, gy, self.base['grid_hw'], self.sam_image_hw)
        self.points.append((ix, iy, int(label)))
        try:
            self.proposal = self.mask_provider.predict([(p[0], p[1]) for p in self.points], [p[2] for p in self.points])
        except Exception as exc:  # noqa: BLE001
            self.points.pop()
            messagebox.showerror('Art-direct relief', f'SAM failed: {exc}', parent=self)
            return
        self._apply_proposal()

    def _apply_proposal(self):
        if self.proposal is None or self.base is None:
            return
        from core.relief.art_direct import resample_mask
        which = SIZE_CHOICES[self.size_choice.get()]
        m = self.proposal.choose(which)
        self.mask = resample_mask(m, self.base['grid_hw']) & self.base['plaque_mask']
        self._update_sel_info()
        self._schedule_render()

    def undo_click(self):
        if not self.points:
            return
        self.points.pop()
        if not self.points:
            self.proposal = None
            self._update_sel_info()
            self._schedule_render()
            return
        try:
            self.proposal = self.mask_provider.predict([(p[0], p[1]) for p in self.points], [p[2] for p in self.points])
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror('Art-direct relief', f'SAM failed: {exc}', parent=self)
            return
        self._apply_proposal()

    def clear_selection(self):
        if self.base is None:
            return
        self.points = []
        self.proposal = None
        self.mask[:] = False
        self._update_sel_info()
        self._schedule_render()

    def fill_holes(self):
        if self.base is None or not self.mask.any():
            return
        from core.relief.segment import fill_holes
        self.mask = fill_holes(self.mask) & self.base['plaque_mask']
        self._update_sel_info()
        self._schedule_render()

    def _update_sel_info(self):
        n = int(self.mask.sum()) if self.mask is not None else 0
        if n == 0:
            self.sel_info.configure(text='no selection')
            return
        px = self.base['pixel_mm']
        cur = self.session.current[self.mask]
        self.sel_info.configure(text=f'selection: {n} px = {n * px * px:.0f} mm2; relief inside now '
                                     f'{cur.min():.2f}..{cur.max():.2f} mm; {len(self.points)} SAM point(s)')

    # ------------------------------------------------------------------ edits
    def _read_float(self, var: tk.StringVar, name: str, lo: float, hi: float) -> float:
        try:
            v = float(var.get())
        except ValueError:
            raise ValueError(f'{name} must be a number') from None
        if not lo <= v <= hi:
            raise ValueError(f'{name} must be between {lo:g} and {hi:g}')
        return v

    def current_op(self) -> dict:
        """The op dict the controls describe (mask = a copy of the selection)."""
        op = OPS_UI[self.op_choice.get()]
        rmax = self.base['relief_max_mm']
        d: dict = {'op': op, 'mask': self.mask.copy()}
        if op in ('dome', 'extrude'):
            d['height_mm'] = self._read_float(self.height_mm, 'Height', 0.0, rmax)
        if op == 'dome':
            d['roundness'] = self._read_float(self.roundness, 'Dome roundness', 0.2, 3.0)
        if op == 'offset':
            d['delta_mm'] = self._read_float(self.delta_mm, 'Raise / lower', -rmax, rmax)
        if op in ('dome', 'extrude', 'offset'):
            d['feather_mm'] = self._read_float(self.feather_mm, 'Edge feather', 0.0, 50.0)
        if op == 'feather':
            d['width_mm'] = self._read_float(self.feather_mm, 'Edge feather', 0.05, 50.0)
        return d

    def apply_edit(self) -> Optional[dict]:
        if self.base is None:
            return None
        if self.mask is None or not self.mask.any():
            messagebox.showinfo('Art-direct relief', 'Select something first (click the object or paint it).', parent=self)
            return None
        try:
            op = self.current_op()
        except ValueError as exc:
            messagebox.showwarning('Art-direct relief', str(exc), parent=self)
            return None
        info = self.session.apply(op)
        if info.get('noop'):
            self.edit_info.configure(text='That edit changed nothing (already at that height?).')
        self._update_edit_info()
        self._update_sel_info()
        self._schedule_render()
        return info

    def undo_edit(self):
        if self.session is not None and self.session.undo():
            self._update_edit_info()
            self._update_sel_info()
            self._schedule_render()

    def reset_all(self):
        if self.session is None:
            return
        self.session.reset()
        self._update_edit_info()
        self._update_sel_info()
        self._schedule_render()

    def _update_edit_info(self):
        if self.session is None:
            return
        n = len(self.session.ops)
        cur = self.session.current
        lines = [f'{n} edit(s) applied; relief now {cur.min():.2f}..{cur.max():.2f} mm; '
                 f'{int(self.session.changed().sum())} px differ from the AI relief']
        if n:
            from core.relief.art_direct import _op_summary
            lines.append('last: ' + _op_summary(self.session.ops[-1]))
        self.edit_info.configure(text='\n'.join(lines))
        self.undo_btn.configure(state='normal' if n else 'disabled')

    # ------------------------------------------------------------------ script in / out
    def build_script(self):
        meta = {'image': self.image_path, 'width_mm': self.width_mm, 'base_mm': float(self.dims.base_mm),
                'depth': {'source': self.base.get('geometry'), 'depth_map': self.base.get('depth_map'),
                          'params': self.params.to_dict()}}
        return self.session.to_script(**meta)

    def save_edits(self):
        if self.session is None:
            return
        stem = os.path.splitext(os.path.basename(self.image_path or 'relief'))[0]
        path = filedialog.asksaveasfilename(parent=self, title='Save relief edits', defaultextension='.json',
                                            initialfile=f'{stem}_edits.json', filetypes=[('Relief edits', '*.json')])
        if path:
            self.build_script().save(path)
            self.status.set(f'Saved {len(self.session.ops)} edit(s) to {path}')

    def load_edits(self):
        if self.session is None:
            return
        path = filedialog.askopenfilename(parent=self, title='Load relief edits', filetypes=[('Relief edits', '*.json')])
        if not path:
            return
        try:
            from core.relief.art_direct import EditScript
            self.session.load_script(EditScript.load(path))
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror('Art-direct relief', f'Could not load {path}:\n{exc}', parent=self)
            return
        self._update_edit_info()
        self._update_sel_info()
        self._schedule_render()

    def done(self):
        if self.session is None:
            return
        self.result_script = self.build_script()
        if self.on_done is not None:
            try:
                self.on_done(self.result_script, self.base.get('geom'))
            except Exception as exc:  # noqa: BLE001
                traceback.print_exc()
                messagebox.showerror('Art-direct relief', f'{type(exc).__name__}: {exc}', parent=self)
                return
        self.destroy()

    def cancel(self):
        self.result_script = None
        self.destroy()

    # ------------------------------------------------------------------ rendering
    def _schedule_render(self):
        if not self._render_pending:
            self._render_pending = True
            self.after(25, self._render)

    def _render(self):
        self._render_pending = False
        if self.base is None or not self.winfo_exists():
            return
        dh, dw = self.disp_hw
        m_disp = np.asarray(Image.fromarray(self.mask.astype(np.uint8) * 255, 'L').resize((dw, dh), Image.Resampling.NEAREST)) > 127
        # left: cover + cyan selection
        left = np.asarray(self._img_disp, dtype=np.float32).copy()
        if m_disp.any():
            col = np.array(MASK_COLOUR, np.float32)
            left[m_disp] = left[m_disp] * 0.55 + col * 0.45
        self._photo_left = ImageTk.PhotoImage(Image.fromarray(left.astype(np.uint8)))
        self.cv_left.delete('img')
        self.cv_left.create_image(0, 0, anchor='nw', image=self._photo_left, tags='img')
        self.cv_left.tag_lower('img')
        self.cv_left.delete('pt')
        if self.points and self.base is not None:
            from core.relief.art_direct import image_to_grid_xy
            for ix, iy, lab in self.points:
                gx, gy = image_to_grid_xy(ix, iy, self.base['grid_hw'], self.sam_image_hw)
                x, y = (gx + 0.5) * self.scale, (gy + 0.5) * self.scale
                self.cv_left.create_oval(x - 5, y - 5, x + 5, y + 5, fill='#30e030' if lab else '#ff4040',
                                         outline='black', tags='pt')
        # right: grey relief + outline
        g = np.clip(self.session.current / max(self.base['relief_max_mm'], 1e-6), 0, 1)
        grey = Image.fromarray((g * 255).astype(np.uint8), 'L').resize((dw, dh), Image.Resampling.BILINEAR)
        right = np.stack([np.asarray(grey)] * 3, -1).astype(np.uint8)
        pl = np.asarray(Image.fromarray(self.base['plaque_mask'].astype(np.uint8) * 255, 'L').resize((dw, dh), Image.Resampling.NEAREST)) > 127
        right[~pl] = (40, 40, 60)
        if m_disp.any():
            from scipy import ndimage
            outline = m_disp & ~ndimage.binary_erosion(m_disp)
            right[outline] = OUTLINE_COLOUR
        self._photo_right = ImageTk.PhotoImage(Image.fromarray(right))
        self.cv_right.delete('all')
        self.cv_right.create_image(0, 0, anchor='nw', image=self._photo_right)


# --------------------------------------------------------------------------- standalone
def _selftest() -> int:
    root = tk.Tk()
    root.withdraw()
    base = synthetic_base()
    ed = ReliefEditor(root, image_path='', width_mm=16.0, base=base, mask_provider=DemoMaskProvider(), title='selftest')
    ed.update()
    for _ in range(50):                      # wait for the (threaded) demo provider
        ed._poll()
        ed.update()
        if ed.sam_ready:
            break
        time.sleep(0.05)
    assert ed.sam_ready, 'demo mask provider did not become ready'
    H, W = base['grid_hw']
    ed._sam_click(0.6 * W, 0.5 * H, 1)
    assert ed.mask.any(), 'SAM click produced no selection'
    n_click = int(ed.mask.sum())
    ed.size_choice.set('Large')
    ed._apply_proposal()
    assert int(ed.mask.sum()) > n_click
    ed.op_choice.set('Extrude (flat plateau at Height)')
    ed._on_op_change()
    ed.height_mm.set('2.0')
    ed.feather_mm.set('0')
    info = ed.apply_edit()
    assert info and not info.get('noop') and len(ed.session.ops) == 1
    assert np.allclose(ed.session.current[ed.mask], 2.0)
    ed.undo_edit()
    assert len(ed.session.ops) == 0 and np.array_equal(ed.session.current, ed.session.original)
    ed.tool.set('add')
    ed.clear_selection()
    ed._paint(0.3 * W, 0.3 * H)
    ed._paint(0.35 * W, 0.3 * H, from_pt=(0.3 * W, 0.3 * H))
    assert ed.mask.any()
    ed.op_choice.set('Dome (hemisphere inside the selection)')
    ed._on_op_change()
    ed.height_mm.set('3.5')
    ed.apply_edit()
    script = ed.build_script()
    assert len(script) == 1 and script.to_dict()['ops'][0]['op'] == 'dome'
    ed._render()
    ed.update()
    ed.destroy()
    root.destroy()
    print('selftest ok')
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('image', nargs='?', help='album cover image')
    ap.add_argument('--width', type=float, default=150.0, help='plaque width in mm (default 150)')
    ap.add_argument('--relief-mm', type=float, default=4.0)
    ap.add_argument('--base-mm', type=float, default=0.8)
    ap.add_argument('--depth-map', default=None, help='grey depth map instead of Depth Anything')
    ap.add_argument('--sam-weights', default=None, help='SAM checkpoint (HF id or local dir)')
    ap.add_argument('--edits', default=None, help='replay an existing edits JSON first')
    ap.add_argument('--out', default=None, help='where "Use these edits" writes the JSON (default <image>_edits.json)')
    ap.add_argument('--selftest', action='store_true')
    ap.add_argument('--screenshot', default=None, help='QA: capture the window to this PNG once the relief is ready, then quit')
    ap.add_argument('--demo-mask', action='store_true', help='QA: with --screenshot, select a disc in the centre and dome it')
    args = ap.parse_args(argv)
    if args.selftest:
        return _selftest()
    if not args.image:
        ap.error('an image is required')
    from core.relief.relief_stack import ReliefDims
    out = args.out or os.path.splitext(os.path.abspath(args.image))[0] + '_edits.json'
    root = tk.Tk()
    root.withdraw()

    def on_done(script, _geom):
        path = script.save(out)
        print(f'[EDITOR] {len(script)} edit(s) saved to {path}')
        print('[EDITOR] export with:')
        print(f'  .venv/bin/python scripts/convert_relief.py "{os.path.abspath(args.image)}" --geometry '
              + ('imported-depth --depth-map "%s"' % args.depth_map if args.depth_map else 'depth-anything')
              + f' --width {args.width:g} --edits "{path}"')

    ed = ReliefEditor(root, args.image, args.width, dims=ReliefDims(base_mm=args.base_mm, relief_mm=args.relief_mm),
                      depth_source='file' if args.depth_map else 'ai', depth_map=args.depth_map,
                      sam_weights=args.sam_weights, on_done=on_done, initial_script=args.edits)
    ed.lift()
    ed.attributes('-topmost', True)
    ed.after(800, lambda: ed.attributes('-topmost', False))
    ed.bind('<Destroy>', lambda e: root.after(50, root.quit) if e.widget is ed else None)
    if args.screenshot:
        def shot():
            if ed.base is None:
                ed.after(500, shot)
                return
            if args.demo_mask:
                from core.relief.segment import paint_disc
                H, W = ed.base['grid_hw']
                paint_disc(ed.mask, 0.545 * W, 0.62 * H, 0.125 * W, True)
                ed._update_sel_info()
                ed.height_mm.set('4.0')
                ed.apply_edit()
                ed._render()
            ed.update()
            region = f'{ed.winfo_rootx()},{ed.winfo_rooty()},{ed.winfo_width()},{ed.winfo_height()}'
            import subprocess
            subprocess.run(['screencapture', '-x', '-R', region, args.screenshot], check=False)
            print('screenshot', args.screenshot)
            ed.destroy()
        ed.after(1500, shot)
    root.mainloop()
    return 0


if __name__ == '__main__':
    sys.exit(main())
