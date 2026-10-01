"""Band Studio: the interactive Band-mode (HueForge-style) editor tab.

HueForge is an editor, not a batch converter - its value is that you drag a
swap slider and immediately see what the print will look like.  ``core/band``
already had the whole algorithm but only a CLI, so this module supplies the
missing half: a live editor over :class:`core.band.session.BandSession`.

Control names follow HueForge's own vocabulary (Color Core, Describe, Negative,
Full Range, Border) so its documentation and tutorials read across directly.

Use ``scripts/band_studio.py`` to launch this tab on its own.
"""
from __future__ import annotations

import os
import traceback
from typing import Optional

import gradio as gr

from core.band.optics import (DEFAULT_K_OPAQUE, Filament, hex_to_rgb01, load_filament_library,
                              srgb_to_linear)
from core.band.schedule import MAX_SLOTS
from core.band.session import LED_PRESETS, BandSession

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_LIBRARY = os.path.join(REPO, 'assets', 'filaments_user_measured.json')
DEFAULT_OUT = os.path.join(REPO, 'output', 'band_studio')


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _library_rows(library: dict, filt: str = '', sort: str = 'library order') -> list[list]:
    fils = list(library.values())
    q = (filt or '').strip().lower()
    if q:
        fils = [f for f in fils if q in f.name.lower() or q in f.hex.lower()]
    if sort == 'name':
        fils.sort(key=lambda f: f.name.lower())
    elif sort == 'td_mm':
        fils.sort(key=lambda f: float(f.td_mm))
    elif sort == 'lightness':
        fils.sort(key=lambda f: float(srgb_to_linear(hex_to_rgb01(f.hex)).mean()))
    return [[f.name, f.hex, round(float(f.td_mm), 3), f.td_source] for f in fils]


def _df_rows(df) -> list[list]:
    """Gradio hands the table back as a DataFrame or a list of lists."""
    if df is None:
        return []
    if hasattr(df, 'values'):
        return df.values.tolist()
    return list(df)


def _commit_df(library: dict, df) -> dict:
    """Fold TD edits made in the table back into the library (in place).

    The table can be filtered or sorted, so it is only ever a view: rows that
    are not currently shown keep whatever the library already holds.
    """
    for r in _df_rows(df):
        if len(r) < 3 or r[0] not in library or r[2] is None:
            continue
        try:
            td = float(r[2])
        except (TypeError, ValueError):
            continue
        f = library[r[0]]
        if td > 0 and abs(td - f.td_mm) > 1e-9:
            library[r[0]] = Filament.from_hex(f.name, f.hex, td, 'edited',
                                              f.settings_id, f.filament_id)
    return library


def _swatches_html(session: Optional[BandSession]) -> str:
    """Colour Core strip: the band stack, bottom (base) at the left."""
    if session is None:
        return "<div style='opacity:.6'>Load an image to start.</div>"
    shares = session.surface_shares()
    cells = []
    for name, n in zip(session.filament_names, session.layer_counts):
        f = session.library[name]
        lum = float(srgb_to_linear(hex_to_rgb01(f.hex)).mean())
        fg = '#000' if lum > 0.25 else '#fff'
        cells.append(
            f"<div style='flex:1;min-width:96px;background:{f.hex};color:{fg};"
            f"padding:8px 6px;border-radius:6px;font-size:11px;line-height:1.35;"
            f"border:1px solid rgba(128,128,128,.45)'>"
            f"<b>{name}</b><br>{n} layers · TD {f.td_mm:g}<br>"
            f"surface {shares.get(name, 0) * 100:.1f}%</div>")
    return ("<div style='display:flex;gap:6px;align-items:stretch'>" + ''.join(cells) + "</div>"
            "<div style='opacity:.6;font-size:11px;margin-top:4px'>base (printed first) → top "
            "(printed last). HueForge convention: darkest at the base.</div>")


def _stats_md(session: Optional[BandSession]) -> str:
    if session is None:
        return ''
    s = session.stats()
    lo, hi = s['height_range_mm']
    # The base band is meant to be covered - it is the dark ground the
    # translucent bands are veiled over, so 0% surface there is correct.
    # Only a band above it that never surfaces is a wasted spool and swap.
    dead = [n for i, (n, v) in enumerate(s['surface_shares'].items())
            if i > 0 and v < 0.001]
    warn = ''
    if dead:
        verb = 'never reaches' if len(dead) == 1 else 'never reach'
        warn = (f"\n\n⚠️ **{', '.join(dead)}** {verb} the surface — "
                f"you'd load the spool and swap for nothing. Drop it, or raise the relief.")
    if session.lighting.startswith('back'):
        import numpy as np
        t = session.transmitted(full=False)
        m = session._mask_small
        if float(t[m].mean() if m.any() else 0.0) < 0.02:
            base = session.filament_names[0]
            warn += (f"\n\n🔦 Back-lit render is dark because the base band "
                     f"(**{base}**, {session.layer_counts[0]} layers) blocks the light. "
                     f"A lithophane needs a translucent base — try White.")
    return (f"**{s['layers']} layers · {s['total_height_mm']} mm · {s['swaps']} swaps** · "
            f"relief {lo}–{hi} mm{warn}")


# --------------------------------------------------------------------------- #
# tab
# --------------------------------------------------------------------------- #
def create_band_tab_content(lang: str = 'en', library_path: str = DEFAULT_LIBRARY,
                            initial_image: Optional[str] = None) -> dict:
    library = load_filament_library(library_path)
    names = list(library)
    default_stack = [n for n in ('Black', 'Klein Blue', 'Lavender Purple', 'White') if n in library]
    while len(default_stack) < 4:
        default_stack.append(names[len(default_stack)])

    state = gr.State(None)

    gr.Markdown("## Band Studio — filament painting\n"
                "Colour comes from plate-wide swaps at fixed heights: one solid mesh, "
                "a handful of tool changes, no purge tower.")

    with gr.Row():
        # ---------------- left: image + live preview ----------------------- #
        with gr.Column(scale=3):
            image_in = gr.Image(label="Image", type='filepath', height=220,
                                value=initial_image)
            with gr.Row():
                width_mm = gr.Slider(40, 250, value=150, step=1, label="Width (mm)")
                remesh = gr.Button("Reload image", variant='secondary', scale=0)
            preview = gr.Image(label="Live preview (front lit)", height=430)
            stats_md = gr.Markdown()
            core_html = gr.HTML()

        # ---------------- right: the controls ------------------------------ #
        with gr.Column(scale=3):
            with gr.Tabs():
                with gr.Tab("Core"):
                    n_bands = gr.Slider(2, MAX_SLOTS, value=len(default_stack), step=1,
                                        label="Filaments (bands)")
                    band_names, band_counts, band_rows = [], [], []
                    for i in range(MAX_SLOTS):
                        with gr.Row(visible=i < len(default_stack)) as row:
                            band_names.append(gr.Dropdown(
                                names, value=default_stack[i] if i < len(default_stack) else names[0],
                                label=f"{'Base' if i == 0 else f'Band {i}'}", scale=3))
                            band_counts.append(gr.Slider(
                                1, 20, value=9 if i == 0 else 4, step=1, label="layers", scale=2))
                        band_rows.append(row)
                    auto = gr.Button("Auto-pick layer counts", variant='secondary')

                with gr.Tab("Image"):
                    negative = gr.Checkbox(False, label="Negative (flip dark ↔ light)")
                    full_range = gr.Checkbox(False, label="Full Range (stretch luminance)")
                    gr.Markdown("**Tone curve** — maps image lightness to relief height.")
                    q_lo = gr.Slider(-0.5, 0.5, value=-0.13, step=0.01, label="Black point (q_lo)")
                    q_hi = gr.Slider(0.5, 1.5, value=1.06, step=0.01, label="White point (q_hi)")
                    gamma = gr.Slider(0.3, 3.0, value=1.55, step=0.05, label="Gamma")
                    smooth = gr.Slider(0, 30, value=10, step=1, label="Smoothing (bilateral σ)")

                with gr.Tab("Shape"):
                    first_layer = gr.Slider(0.08, 0.32, value=0.16, step=0.04,
                                            label="First layer (mm)")
                    layer_h = gr.Slider(0.04, 0.20, value=0.08, step=0.01, label="Layer height (mm)")
                    engrave = gr.Checkbox(True, label="Engrave base band (relief in the darks)")
                    k_opaque = gr.Slider(2.0, 10.0, value=DEFAULT_K_OPAQUE, step=0.5,
                                         label="Opacity steepness (k)")
                    gr.Markdown("**Border**")
                    border_w = gr.Slider(0, 10, value=0, step=0.5, label="Border width (mm)")
                    border_d = gr.Slider(0, 2, value=0, step=0.04, label="Border depth (mm)")

                with gr.Tab("Light"):
                    blending = gr.Radio(
                        ['Filament Painting (front lit)', 'Lithophane (back lit)'],
                        value='Filament Painting (front lit)', label="Blending Type")
                    led = gr.Dropdown(list(LED_PRESETS), value='Natural White 4000K',
                                      label="LED light type (back lit only)")
                    intensity = gr.Slider(0.2, 6.0, value=2.5, step=0.1,
                                          label="Light intensity (back lit only)")

                with gr.Tab("Spools"):
                    gr.Markdown("Edit **TD** (transmission distance, mm) in place — "
                                "low = opaque, high = translucent. "
                                "Measure real values with `scripts/td_wedge.py`.")
                    with gr.Row():
                        lib_filter = gr.Textbox(label="Filter", placeholder="name or hex",
                                                scale=2)
                        lib_sort = gr.Dropdown(['library order', 'name', 'td_mm', 'lightness'],
                                               value='library order', label="Sort", scale=2)
                    lib_df = gr.Dataframe(
                        value=_library_rows(library), headers=['name', 'hex', 'td_mm', 'source'],
                        datatype=['str', 'str', 'number', 'str'], interactive=True,
                        label="Filament library", max_height=300)
                    with gr.Row():
                        new_name = gr.Textbox(label="New filament", placeholder="name", scale=2)
                        new_hex = gr.Textbox(label="Hex", placeholder="#RRGGBB", scale=1)
                        new_td = gr.Number(value=1.0, label="TD mm", scale=1)
                        add_fil = gr.Button("Add", scale=0)
                    with gr.Row():
                        lib_path = gr.Textbox(library_path, label="Library file", scale=3)
                        save_lib = gr.Button("Save library", variant='secondary', scale=1)
                    lib_log = gr.Markdown()

            describe = gr.Textbox(label="Describe (slicer instructions)", lines=9)
            with gr.Row():
                out_dir = gr.Textbox(DEFAULT_OUT, label="Output folder", scale=3)
                export = gr.Button("Export 3MF", variant='primary', scale=1)
                export_stl = gr.Button("Export STL", scale=1)
            export_log = gr.Markdown()

    # ----------------------------------------------------------------- wiring
    CONTROLS = ([image_in, width_mm, n_bands] + band_names + band_counts +
                [negative, full_range, q_lo, q_hi, gamma, smooth,
                 first_layer, layer_h, engrave, k_opaque, border_w, border_d,
                 blending, led, intensity, lib_df])
    OUTPUTS = [state, preview, describe, stats_md, core_html]

    def _sync(session, args, reload_image=False):
        (path, w, nb) = args[0], float(args[1]), int(args[2])
        nm = list(args[3:3 + MAX_SLOTS])
        ct = list(args[3 + MAX_SLOTS:3 + 2 * MAX_SLOTS])
        (neg, fr, qlo, qhi, gam, sig, fl, lh, eng, kop, bw, bd,
         blend, led_name, inten, df) = args[3 + 2 * MAX_SLOTS:]
        if not path:
            return None, None, '', '', _swatches_html(None)

        lib = _commit_df(library, df)

        picked, counts = [], []
        for i in range(nb):                    # de-duplicate: BandSchedule requires unique names
            n = nm[i]
            if n in picked:
                n = next((c for c in lib if c not in picked), n)
            picked.append(n)
            counts.append(max(1, int(ct[i])))

        fresh = (session is None or reload_image or session.image_path != path
                 or abs(session.width_mm - w) > 1e-9 or abs(session.smooth_sigma - sig) > 1e-9)
        if fresh:
            session = BandSession.load(path, lib, picked, counts, width_mm=w, smooth_sigma=sig)
        session.library = lib
        session.filament_names, session.layer_counts = picked, counts
        session.width_mm = w
        session.negative, session.full_range = bool(neg), bool(fr)
        session.tone = type(session.tone)(float(qlo), float(qhi), float(gam))
        session.first_layer_mm, session.layer_h = float(fl), float(lh)
        session.base_min_layers = min(5, counts[0])
        session.engrave, session.k_opaque = bool(eng), float(kop)
        session.border_width_mm, session.border_depth_mm = float(bw), float(bd)
        session.lighting = 'back' if 'back' in str(blend).lower() else 'front'
        session.led_hex = LED_PRESETS.get(led_name, session.led_hex)
        session.light_intensity = float(inten)
        return (session, session.preview(), session.describe(),
                _stats_md(session), _swatches_html(session))

    def on_change(session, *args):
        try:
            return _sync(session, args)
        except Exception as exc:
            return session, None, f"{type(exc).__name__}: {exc}", f"⚠️ {exc}", _swatches_html(session)

    def on_reload(session, *args):
        try:
            return _sync(session, args, reload_image=True)
        except Exception as exc:
            return session, None, f"{type(exc).__name__}: {exc}", f"⚠️ {exc}", _swatches_html(session)

    for c in CONTROLS:
        ev = c.change if hasattr(c, 'change') else None
        if ev:
            ev(on_change, [state] + CONTROLS, OUTPUTS, show_progress='minimal')
    remesh.click(on_reload, [state] + CONTROLS, OUTPUTS)

    def on_n_bands(nb):
        return [gr.update(visible=i < int(nb)) for i in range(MAX_SLOTS)]
    n_bands.change(on_n_bands, n_bands, band_rows)

    def on_auto(session, *args):
        """Optimise per-band thicknesses for the current filament order."""
        keep = [gr.update() for _ in band_counts]
        try:
            session, *rest = _sync(session, args)
            if session is None:
                return (session, *rest, *keep)
            counts = session.auto_thickness()
            session.layer_counts = list(counts)
            return (session, session.preview(), session.describe(),
                    _stats_md(session), _swatches_html(session),
                    *[gr.update(value=int(c)) for c in counts],
                    *keep[len(counts):])
        except Exception as exc:
            return (session, None, f"{type(exc).__name__}: {exc}", f"⚠️ {exc}",
                    _swatches_html(session), *keep)
    auto.click(on_auto, [state] + CONTROLS, OUTPUTS + band_counts)

    def on_export(session, folder):
        if session is None:
            return "Load an image first."
        try:
            res = session.export(folder or DEFAULT_OUT)
            return (f"✅ **{res.get('threemf', '?')}**\n\n"
                    f"Preview: `{res.get('preview_png', '?')}`")
        except Exception:
            return f"```\n{traceback.format_exc()[-1500:]}\n```"
    export.click(on_export, [state, out_dir], export_log)

    # ---- filament library management --------------------------------------
    def on_lib_view(df, filt, srt):
        _commit_df(library, df)
        return gr.update(value=_library_rows(library, filt, srt))
    lib_filter.change(on_lib_view, [lib_df, lib_filter, lib_sort], lib_df)
    lib_sort.change(on_lib_view, [lib_df, lib_filter, lib_sort], lib_df)

    def on_add(df, filt, srt, name, hexv, td):
        _commit_df(library, df)
        name = (name or '').strip()
        hexv = (hexv or '').strip()
        if not name:
            return gr.update(), *[gr.update() for _ in band_names], "Give the filament a name."
        if name in library:
            return gr.update(), *[gr.update() for _ in band_names], f"**{name}** already exists."
        try:
            library[name] = Filament.from_hex(name, hexv, float(td or 1.0), 'user')
        except Exception as exc:
            return gr.update(), *[gr.update() for _ in band_names], f"⚠️ {exc}"
        choices = list(library)
        return (gr.update(value=_library_rows(library, filt, srt)),
                *[gr.update(choices=choices) for _ in band_names],
                f"Added **{name}** ({hexv}, TD {float(td or 1.0):g}). "
                f"Save the library to keep it.")
    add_fil.click(on_add, [lib_df, lib_filter, lib_sort, new_name, new_hex, new_td],
                  [lib_df] + band_names + [lib_log])

    def on_save_lib(df, path):
        _commit_df(library, df)
        try:
            from core.band.optics import save_filament_library
            p = save_filament_library(path, library)
            return f"✅ Saved {len(library)} filaments to `{p}`"
        except Exception as exc:
            return f"⚠️ {type(exc).__name__}: {exc}"
    save_lib.click(on_save_lib, [lib_df, lib_path], lib_log)

    def on_export_stl(session, folder):
        """HueForge's native output: STL + Describe.txt."""
        if session is None:
            return "Load an image first."
        try:
            r = session.export_stl(folder or DEFAULT_OUT)
            return (f"✅ **{r['stl']}**\n\n{r['triangles']:,} triangles, "
                    f"watertight={r['watertight']}\n\nSwaps: `{r['describe_txt']}`")
        except Exception:
            return f"```\n{traceback.format_exc()[-1500:]}\n```"
    export_stl.click(on_export_stl, [state, out_dir], export_log)

    # Exposed so a host app can fire the first render on page load
    # (e.g. when an image was supplied on the command line).
    return {'state': state, 'preview': preview, 'describe': describe,
            'refresh': (on_change, [state] + CONTROLS, OUTPUTS)}
