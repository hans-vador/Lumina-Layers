"""Deterministic Graduation-reference tone relief; one material per Z layer.

The tone fit is empirical, not HueForge source code or a calibrated colour model.
Geometry is independent of TD: opaque white cannot erase image detail in mapping.

Any cover, any spools: auto_tone spreads the cover's brightness over the whole
relief (the reference cover already spanned black..white), and choose_palette
tries every combination of the library's spools and every swap plan against
that contrast-stretched cover.
"""
from pathlib import Path
import hashlib
import itertools
import json
import zipfile
import re
import uuid
import numpy as np
from PIL import Image, ImageOps
from scipy import ndimage
from core.band.heightfield_mesh import heightfield_to_trimesh
from core.band.optics import (DEFAULT_K_OPAQUE, Filament, linear_rgb_to_lab_d65, linear_to_srgb,
                              load_filament_library, luminance_y, srgb_to_lab_d65, srgb_to_linear)
from core.band.writer3mf import write_band_3mf

REPO = Path(__file__).resolve().parents[2]
DEFAULT_FILAMENTS_JSON = REPO / 'assets' / 'filaments_user_measured.json'
# Bambu's own X2D 0.08 mm process, inheritance resolved (the file Stack5 builds on).
SYSTEM_PROCESS_JSON = REPO / 'assets' / 'x2d_process_0.08mm_high_quality_system.json'
# Lists that say which variant belongs to which nozzle: always Bambu's, never ours.
VARIANT_STRUCTURE_KEYS = ('print_extruder_id', 'print_extruder_variant')
LAYER_H = .08
# Narrowest raised feature kept, in mm.  Bambu Studio's support check ignores
# lower-layer islands narrower than one extrusion (~.42 mm), so a hairline
# ridge sitting on an equally thin ridge is reported as a floating region.
# Nothing that thin prints as a distinct feature from a .4 mm nozzle anyway.
MIN_FEATURE_MM = .45
# The reference file's own slots and swaps (heights for the thin-floor layout).
GRADUATION_FILAMENTS = [('Black', '#000000', .15), ('Purple', '#800080', 2.5),
                        ('Pink Red', '#F55A74', 4.5), ('White', '#FFFFFF', .1)]
GRADUATION_SWAPS = [(.92, 2, '#800080'), (1.08, 3, '#F55A74'), (1.32, 4, '#FFFFFF')]

# Fit to registered source/mesh samples from Kanye_-_Graduation.3mf.
FLOOR = .481055573
AMPLITUDE = 1.82374436
GAMMA = .975045487
WEIGHTS = np.array([.299, .587, .114])

# Auto tone.  The rule above maps brightness 0..1 onto ~23 colour layers, and
# Graduation's cover fills that range (16+ layers each hold >1% of it).  A dark
# cover does not: to_hell_with_it's 99th brightness percentile is .37, so 72% of
# it landed in the lowest five layers and printed as plain Black.  Auto levels
# (these percentiles -> 0..1) give every cover the reference's spread.
# Contrast > 0 blends in contrast-limited histogram equalisation.  It is off by
# default: equalisation hands extra layers to whatever covers the most pixels,
# which on Rodeo is the flat dark background (55% of the cover at brightness
# ~.18) - lifted from layer 5 to 8, it printed brown instead of black.
TONE_PERCENTILES = (.5, 99.5)
EQUALIZE_CLIP = 3.          # no brightness bin gets more than 3x the mean share
DEFAULT_CONTRAST = 0.
# Auto palette: up to one AMS (nozzle 1 feeds four slots), like the reference's
# four spools; one more spool must lower the mean colour error by at least this.
AUTO_MAX_SPOOLS = 4
MIN_GAIN_DE = .5


def brightness(rgb):
    return np.asarray(rgb, dtype=np.float64) / 255 @ WEIGHTS


def height_from_tone(t):
    return FLOOR + AMPLITUDE * np.clip(t, 0, 1)**GAMMA


def tone_height(rgb):
    """The reference's own rule: raw brightness -> height."""
    return height_from_tone(brightness(rgb))


def auto_tone(rgb, contrast=DEFAULT_CONTRAST):
    """Brightness 0..1 that uses the whole relief for this cover.

    Auto levels stretch the TONE_PERCENTILES of brightness to 0..1; then
    contrast (0..1) blends in contrast-limited histogram equalisation, so
    brightness ranges holding many pixels (a night sky, a face) get more layers.
    Monotone in brightness, like the reference: brighter is never lower, so
    edges and detail keep their shape.  Returns (tone, (lo, hi)).
    """
    if not 0 <= contrast <= 1:
        raise ValueError('contrast must be 0..1')
    q = brightness(rgb)
    lo, hi = (float(v) for v in np.percentile(q, TONE_PERCENTILES))
    s = np.clip((q - lo) / max(hi - lo, 1e-6), 0, 1)
    hist = np.bincount(np.minimum((s * 256).astype(np.int64), 255).ravel(), minlength=256).astype(np.float64)
    cap = EQUALIZE_CLIP * hist.mean()
    hist = np.minimum(hist, cap) + np.clip(hist - cap, 0, None).sum() / 256
    cdf = np.concatenate([[0.], np.cumsum(hist)]) / hist.sum()
    equalized = np.interp(s, np.linspace(0, 1, 257), cdf)
    return (1 - contrast) * s + contrast * equalized, (lo, hi)


def toned_target(rgb, tone):
    """The cover re-lit to the given tone: what the colours are fitted to.

    Each pixel's linear RGB is scaled to the luminance of a grey of that tone,
    which keeps its hue and saturation (a navy night sky becomes a visible
    blue, not grey); colours pushed past white are desaturated toward white at
    the same luminance.  Returns linear RGB (..., 3).
    """
    lin = srgb_to_linear(np.asarray(rgb, dtype=np.float64) / 255)
    y = luminance_y(lin)
    yt = srgb_to_linear(np.clip(tone, 0, 1))
    out = np.where((y > 1e-4)[..., None], lin * (yt / np.maximum(y, 1e-4))[..., None], yt[..., None])
    peak = out.max(-1)
    k = np.where(peak > 1, (1 - yt) / np.maximum(peak - yt, 1e-9), 1.)
    return np.clip(yt[..., None] + (out - yt[..., None]) * k[..., None], 0, 1)


def column_layers(h, base_top):
    """Colour layers the slicer prints in each column: layer j (top base_top + j*.08)
    is printed when the surface passes its mid-height."""
    return np.clip(np.floor((np.asarray(h) - base_top) / LAYER_H + .5), 0, None).astype(np.int64)


def remove_thin_features(h, pitch_mm, min_feature_mm=MIN_FEATURE_MM):
    """Grey-level opening: every raised feature narrower than min_feature_mm is
    lowered to its surroundings, at every height at once.

    For a height field, a flat k x k opening makes each layer's cross-section a
    union of k x k squares, so every island on every layer is at least
    (k-1)*pitch wide on the interpolated mesh.  It only lowers, never raises,
    so it cannot create an overhang.  Returns (h, window_px); window 0 = off.
    """
    if not min_feature_mm or min_feature_mm <= 0:
        return h, 0
    k = int(np.ceil(float(min_feature_mm) / float(pitch_mm) - 1e-9)) + 1
    return ndimage.grey_opening(h, size=(k, k), mode='nearest'), k


def complete_two_nozzle_process(cfg, system=None):
    """Give every process list one value per X2D nozzle variant, in place.

    The X2D has two nozzles (direct drive + Bowden) and six process variants.
    The band template was captured from single-nozzle projects, so its speed /
    acceleration lists hold one value.  Our preset has a custom name and no
    parent, so Bambu Studio uses it verbatim: as soon as a spool is assigned to
    the second nozzle - which happens with a 5th spool, because nozzle 1's AMS
    has four slots - there is no process data for that nozzle and slicing fails.

    Rule (same as Stack5's _shaped_like): a value we deliberately changed from
    Bambu's applies to every variant; anything we left alone takes Bambu's own
    per-variant list, so the Bowden nozzle keeps Bowden speeds.  Keys missing
    from the project are filled from the system preset.  Returns changed keys.
    """
    sysp = (system or json.loads(SYSTEM_PROCESS_JSON.read_text()))['config']
    changed = []
    for k, sv in sysp.items():
        cur = cfg.get(k)
        if isinstance(sv, list):
            sv = [str(x) for x in sv]
            if k in VARIANT_STRUCTURE_KEYS or cur is None:
                new = sv
            elif isinstance(cur, list) and len(cur) == len(sv):
                continue
            else:
                ours = str(cur[0] if isinstance(cur, list) and cur else cur if cur is not None else sv[0])
                new = sv if ours == sv[0] else [ours] * len(sv)
        elif cur is None:
            new = str(sv)
        else:
            continue
        if cfg.get(k) != new:
            cfg[k] = new
            changed.append(k)
    return changed


def palette_filaments(names, library, order='auto'):
    """Library filaments for ``names``, bottom (printed first) -> top.

    order 'auto' sorts dark -> light: the tone rule puts light pixels on tall
    columns, so the lightest spool has to be the last one printed.
    """
    names = [str(n).strip() for n in names if str(n).strip()]
    if not 2 <= len(names) <= 5:
        raise ValueError(f'choose 2 to 5 filaments (got {len(names)})')
    if len(set(names)) != len(names):
        raise ValueError('each filament can be used once')
    missing = [n for n in names if n not in library]
    if missing:
        raise KeyError(f'not in the filament library: {missing}')
    fils = [library[n] for n in names]
    if order == 'auto':
        fils.sort(key=lambda f: f.luminance)
    elif order != 'fixed':
        raise ValueError("order must be 'auto' or 'fixed'")
    return fils


def _stack_colour(n_layers, starts, fils, k_opaque):
    """Linear RGB seen from above: an opaque base, then band b covering colour
    layers starts[b-1] .. starts[b]-1, each a Beer-Lambert veil over what is
    below.  n_layers (P,), starts (M, B-1) -> (M, P, 3)."""
    cols = np.asarray(n_layers)[None, :]
    starts = np.asarray(starts)
    colour = np.broadcast_to(fils[0].rgb_lin, (starts.shape[0], cols.shape[1], 3)).copy()
    for b in range(1, len(fils)):
        lo = starts[:, b - 1][:, None]
        hi = starts[:, b][:, None] - 1 if b < len(fils) - 1 else np.iinfo(np.int64).max
        count = np.clip(np.minimum(cols, hi) - lo + 1, 0, None)
        a = (1 - np.exp(-k_opaque * count * LAYER_H / max(float(fils[b].td_mm), 1e-6)))[..., None]
        colour = colour * (1 - a) + fils[b].rgb_lin * a
    return colour


def _layer_targets(lab, n, top):
    """Pixel count and mean CIELAB per layer count: all a swap plan's
    least-squares error depends on (plus a constant within-layer spread)."""
    w = np.bincount(n, minlength=top + 1).astype(np.float64)
    mean = np.stack([np.bincount(n, weights=lab[:, c], minlength=top + 1) for c in range(3)], 1)
    return w, mean / np.maximum(w, 1)[:, None]


def _best_plan(fils, w, mean, k):
    """Exhaustive search over every swap plan for fils; returns (starts, cost, plans)."""
    top = len(w) - 1
    B = len(fils)
    plans = np.array(list(itertools.combinations(range(1, top + 1), B - 1)), dtype=np.int64).reshape(-1, B - 1)
    layers = np.arange(top + 1)
    best, best_cost = None, np.inf
    for chunk in np.array_split(plans, max(1, len(plans) // 4096)):
        pred = linear_rgb_to_lab_d65(_stack_colour(layers, chunk, fils, k))
        cost = ((pred - mean) ** 2).sum(-1) @ w
        i = int(np.argmin(cost))
        if cost[i] < best_cost:
            best, best_cost = chunk[i], float(cost[i])
    return best, best_cost, len(plans)


def _fit_report(fils, starts, lab, n, h, base_top, k, plans_searched):
    layers = np.arange(int(n.max()) + 1)
    colour_by_layer = _stack_colour(layers, np.asarray(starts)[None], fils, k)[0]   # (top+1, 3) linear
    de = np.linalg.norm(lab - linear_rgb_to_lab_d65(colour_by_layer)[n], axis=1)
    band = np.searchsorted(starts, n, side='right')                                 # 0 = base
    return {
        'starts': [int(v) for v in starts],
        'swaps': [(round(base_top + int(v) * LAYER_H, 6), b + 1, fils[b].hex)
                  for b, v in enumerate(starts, 1)],
        'mean_dE': round(float(de.mean()), 3),
        'p90_dE': round(float(np.percentile(de, 90)), 3),
        'surface_share': {f.name: round(float((band == b).mean()), 4) for b, f in enumerate(fils)},
        'plans_searched': int(plans_searched),
        'colour_by_layer': colour_by_layer,
        'layers': column_layers(h, base_top),
    }


def _target_lab(rgb, target_lab):
    if target_lab is not None:
        return np.asarray(target_lab, dtype=np.float64).reshape(-1, 3)
    return srgb_to_lab_d65(np.asarray(rgb, dtype=np.float64).reshape(-1, 3) / 255.)


def fit_swaps(rgb, h, fils, base_top, k_opaque=None, target_lab=None):
    """Where each chosen filament should start so the print matches the image.

    The geometry is fixed by the tone rule, so every column's height - and
    therefore its layer count - is already decided; only the swap layers are
    free.  Columns with the same layer count print the same colour, so the
    least-squares error of a swap plan only needs, per layer count, how many
    pixels land there and their mean CIELAB colour.  That makes an exhaustive
    search over every plan exact and cheap (~10k plans x ~25 layer counts).
    target_lab (pixels, 3) replaces the image's own colours, e.g. toned_target.
    """
    k = DEFAULT_K_OPAQUE if k_opaque is None else float(k_opaque)
    n = column_layers(h, base_top).ravel()
    top = int(n.max())
    B = len(fils)
    if top < B - 1:
        raise ValueError(f'the image only spans {top} colour layers; too flat for {B} filaments')
    lab = _target_lab(rgb, target_lab)
    w, mean = _layer_targets(lab, n, top)
    best, _, count = _best_plan(fils, w, mean, k)
    return _fit_report(fils, best, lab, n, h, base_top, k, count)


def choose_palette(rgb, h, library, base_top, min_spools=2, max_spools=AUTO_MAX_SPOOLS,
                   k_opaque=None, target_lab=None, min_gain_dE=MIN_GAIN_DE):
    """Pick the spools as well as the swaps: every combination of min..max
    library spools (stacked dark -> light, as the tone rule needs) gets its
    own exhaustive swap search, and the least-squares best wins.

    A larger palette always fits at least about as well, so it is only taken
    when it lowers the mean colour error by min_gain_dE per extra spool.
    Returns fit_swaps' report plus 'filaments' and 'by_size' (best per count).
    """
    k = DEFAULT_K_OPAQUE if k_opaque is None else float(k_opaque)
    if not 2 <= min_spools <= max_spools <= 5:
        raise ValueError('spool counts must satisfy 2 <= min <= max <= 5')
    fls = sorted(library.values(), key=lambda f: (f.luminance, f.name))
    if len(fls) < min_spools:
        raise ValueError(f'the library has {len(fls)} spools; need at least {min_spools}')
    n = column_layers(h, base_top).ravel()
    top = int(n.max())
    lab = _target_lab(rgb, target_lab)
    w, mean = _layer_targets(lab, n, top)
    by_size, searched, palettes = {}, 0, 0
    for size in range(min_spools, min(max_spools, len(fls), top + 1) + 1):
        best = (np.inf, None, None)
        for combo in itertools.combinations(fls, size):
            starts, cost, count = _best_plan(list(combo), w, mean, k)
            searched += count
            palettes += 1
            if cost < best[0]:
                best = (cost, list(combo), starts)
        by_size[size] = _fit_report(best[1], best[2], lab, n, h, base_top, k, 0) | {'filaments': best[1]}
    if not by_size:
        raise ValueError(f'the image only spans {top} colour layers; too flat for {min_spools} filaments')
    chosen = min(by_size)
    for size in sorted(by_size)[1:]:
        if by_size[size]['mean_dE'] <= by_size[chosen]['mean_dE'] - min_gain_dE * (size - chosen):
            chosen = size
    fit = dict(by_size[chosen], plans_searched=searched, palettes_searched=palettes)
    fit['by_size'] = {s: {'filaments': [f.name for f in r['filaments']],
                          'swaps_mm': [z for z, _, _ in r['swaps']], 'mean_dE': r['mean_dE']}
                      for s, r in by_size.items()}
    return fit


def convert_reference_band(image_path, out_dir, width_mm=200., backing_mm=1., pitch_mm=.2,
                           palette='auto', filaments_json=None, order='auto', k_opaque=None,
                           min_feature_mm=MIN_FEATURE_MM, tone=None, contrast=DEFAULT_CONTRAST,
                           min_spools=2, max_spools=AUTO_MAX_SPOOLS):
    """Use the successful Graduation tone/swap profile on a source image.

    .56 mm translates the original art and swaps together, accommodating a
    1 mm base of .2 mm layers without changing any optical band thickness.
    A zero backing request keeps the reference's original thin black floor.

    palette 'auto' (or None) picks min..max_spools spools from the library and
    their swap layers together (choose_palette).  A list of 2-5 library names
    uses exactly those and fits only the swaps (fit_swaps).  Both score the
    Beer-Lambert colour of each layer count (from each spool's TD) against the
    cover.  'graduation' prints the reference's own four slot colours at its
    own swap heights (height preview only).

    tone 'auto' (default for spools) stretches the cover's brightness over the
    whole relief (auto_tone) and fits colours to the re-lit cover
    (toned_target); 'reference' (default for 'graduation') is the raw rule.

    pitch_mm .2 (was .15): features under min_feature_mm are removed anyway, and
    .15 made a 200 mm plaque 7.1M triangles (~560 MB of model XML), heavy enough
    to strain Bambu Studio; .2 is ~4M and was the pitch validated in the slicer.
    min_feature_mm removes raised features too thin to print (and that Bambu
    Studio would report as floating regions); 0 keeps the raw tone rule.
    """
    if not 10 <= width_mm <= 240 or not .08 <= pitch_mm <= .5:
        raise ValueError('width must be 10..240 mm and pitch .08..0.5 mm')
    if backing_mm not in (0, 1):
        raise ValueError('backing_mm must be 0 (reference) or 1 (five .2 mm layers)')
    shift = .56 if backing_mm else 0.
    image = ImageOps.exif_transpose(Image.open(image_path)).convert('RGB')
    if image.width != image.height:
        raise ValueError('Graduation reference profile requires a square image; crop explicitly first')
    n = int(round(width_mm / pitch_mm)) + 1
    pitch = width_mm / (n-1)
    # Artwork fills the plaque; no raised frame or cropped edge pixels.
    rgb = np.asarray(image.resize((n,n), Image.Resampling.BILINEAR))
    palette = 'auto' if palette is None else palette
    if isinstance(palette, str):
        palette_mode = {'auto': 'auto', 'graduation': 'graduation-reference'}.get(palette)
        if palette_mode is None:
            raise ValueError("palette must be 'auto', 'graduation' or a list of 2-5 filament names")
    else:
        palette_mode = 'custom'
    tone = tone or ('reference' if palette_mode == 'graduation-reference' else 'auto')
    if tone == 'auto':
        t, levels = auto_tone(rgb, contrast)
        target = toned_target(rgb, t)
    elif tone == 'reference':
        t, levels, target = brightness(rgb), None, None
    else:
        raise ValueError("tone must be 'auto' or 'reference'")
    raw = height_from_tone(t) + shift
    h, window_px = remove_thin_features(raw, pitch, min_feature_mm)
    mesh = heightfield_to_trimesh(h, pitch)
    out = Path(out_dir); out.mkdir(parents=True,exist_ok=True)
    stem = Path(image_path).stem + '_reference_band'
    fit = None
    if palette_mode == 'graduation-reference':
        # These are reference slot colours, not claims about spool transmission.
        filaments = [Filament.from_hex(name, hx, td) for name, hx, td in GRADUATION_FILAMENTS]
        swaps = [(round(z+shift,6),slot,hx) for z,slot,hx in GRADUATION_SWAPS]
    else:
        library = load_filament_library(str(filaments_json or DEFAULT_FILAMENTS_JSON))
        target_lab = None if target is None else linear_rgb_to_lab_d65(target.reshape(-1, 3))
        # backing: five .2 mm layers end at 1.0; thin floor: .2 first layer
        base_top = 1. if backing_mm else .2
        if palette_mode == 'auto':
            fit = choose_palette(rgb, h, library, base_top, min_spools, max_spools, k_opaque, target_lab)
            filaments = fit['filaments']
        else:
            filaments = palette_filaments(palette, library, order)
            fit = fit_swaps(rgb, h, filaments, base_top, k_opaque, target_lab)
        swaps = fit['swaps']
    if target is not None:
        Image.fromarray(np.uint8(np.round(linear_to_srgb(target)*255))).save(out / (stem+'_target.png'))
    preview = out / (stem+'_height.png')
    # Explicit height preview, not an uncalibrated promise of printed colours.
    Image.fromarray(np.uint8(np.clip((h-shift-FLOOR)/AMPLITUDE,0,1)*255)).save(preview)
    thumbnail = preview
    if fit is not None:
        # With real spools and their TDs the colours ARE modelled, so show them.
        thumbnail = out / (stem+'_predicted.png')
        lin = fit['colour_by_layer'][fit['layers']]
        Image.fromarray(np.uint8(np.round(np.clip(linear_to_srgb(lin), 0, 1)*255))).save(thumbnail)
    scripts = json.loads((Path(__file__).resolve().parents[2]/'assets/x2d_machine_gcodes.json').read_text())['config']
    overrides = dict(scripts, initial_layer_print_height='0.2', layer_height='0.08',
                     print_settings_id='Reference Band 0.08mm X2D',
                     top_surface_pattern='monotonic', bottom_surface_pattern='monotonic',
                     internal_solid_infill_pattern='monotonic', brim_type='no_brim',
                     nozzle_temperature=['220']*len(filaments),
                     nozzle_temperature_initial_layer=['220']*len(filaments))
    target = out / (stem+'.3mf')
    write_band_3mf(mesh,str(target),filaments,swaps,title=stem,
                   description=('Graduation reference tone profile; height preview only' if fit is None else
                                'Graduation-style reference layers; spool colours fitted to the cover'),size_mm=width_mm,
                   print_overrides=overrides,thumbnail_png=str(thumbnail))
    # Add variable backing layers and persist overrides in Studio's preset metadata.
    with zipfile.ZipFile(target) as z:
        members = {name:z.read(name) for name in z.namelist()}
    cfg=json.loads(members['Metadata/project_settings.config'])
    complete_two_nozzle_process(cfg)
    cfg['inherits_group'][0]=''
    cfg['different_settings_to_system'][0]=';'.join(sorted(set(cfg['different_settings_to_system'][0].split(';')) | (set(overrides)-set(scripts))))
    cfg['different_settings_to_system'][-1]=';'.join(sorted(set(scripts)))
    members['Metadata/project_settings.config']=json.dumps(cfg,indent=2).encode()
    if backing_mm:
        members['Metadata/layer_config_ranges.xml']=b'<?xml version="1.0"?><objects><object id="1"><range min_z="0" max_z="1"><option opt_key="layer_height">0.2</option></range></object></objects>'
    # The shared writer creates random UUIDs and a creation date. Normalize
    # those too, so identical inputs/settings produce identical 3MF bytes.
    ids = {}
    def stable_id(match):
        old = match.group().lower()
        if old not in ids:
            ids[old] = str(uuid.uuid5(uuid.NAMESPACE_URL,
                hashlib.sha256(h.tobytes()).hexdigest() + ':' + str(len(ids)))).encode()
        return ids[old]
    for name,data in list(members.items()):
        if name.endswith(('.model','.config','.xml')):
            data = re.sub(rb'[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}',stable_id,data)
            data = re.sub(rb' <metadata name="CreationDate">[^<]*</metadata>\n',b'',data)
            members[name] = data
    with zipfile.ZipFile(target,'w',compression=zipfile.ZIP_DEFLATED) as z:
        for name,data in members.items():
            info=zipfile.ZipInfo(name,date_time=(2020,1,1,0,0,0));info.compress_type=zipfile.ZIP_DEFLATED
            z.writestr(info,data)
    height_path=out/(stem+'_height.npy');np.save(height_path,h)
    report={'engine':'graduation-reference-tone-v3-auto','source':str(Path(image_path).resolve()),
            'source_sha256':hashlib.sha256(Path(image_path).read_bytes()).hexdigest(),
            'height_sha256':hashlib.sha256(h.tobytes()).hexdigest(),
            'tone':{'mode':tone,'contrast':float(contrast) if tone=='auto' else None,
                    'brightness_levels':levels,'floor':FLOOR,'amplitude':AMPLITUDE,'gamma':GAMMA,
                    'rgb_weights':WEIGHTS.tolist()},
            'backing_mm':backing_mm,'height_translation_mm':shift,'pitch_mm':pitch,
            'border_mm':0,
            'min_feature_mm':float(min_feature_mm or 0),'thin_feature_window_px':window_px,
            'thin_feature_lowered':{'pixel_fraction':round(float((raw-h>1e-9).mean()),4),
                                    'mean_mm':round(float((raw-h).mean()),4),'max_mm':round(float((raw-h).max()),4)},
            'art_height_range_mm':[float(h.min()),float(h.max())],
            'swap_entries':swaps,'filaments':[f.name for f in filaments],
            'palette_mode':palette_mode,
            'layer_share':[round(float(v),4) for v in np.bincount(column_layers(h,1. if backing_mm else .2).ravel())/h.size],
            'filament_td_mm':{f.name:float(f.td_mm) for f in filaments},
            'watertight':bool(mesh.is_watertight),'winding_consistent':bool(mesh.is_winding_consistent),
            'preview_kind':'height, not predicted print colour',
            'note':'Reference-derived geometry; physical colour parity needs comparable spools. No TD-based height matching.'}
    if fit is not None:
        report['colour_fit'] = {k: fit[k] for k in ('starts', 'mean_dE', 'p90_dE', 'surface_share',
                                                     'plans_searched', 'palettes_searched', 'by_size')
                                if k in fit}
        report['colour_fit']['against'] = ('contrast-stretched cover (_target.png)' if tone == 'auto'
                                           else 'cover as is')
        report['preview_kind'] = 'predicted print colour from spool TDs (Beer-Lambert); height map saved too'
        report['note'] = ('Reference geometry; ' + ('spools and ' if palette_mode == 'auto' else '')
                          + 'swap layers fitted by CIELAB least squares over every plan. '
                          'Colour accuracy depends on the library TDs.')
    recipe=out/(stem+'.json');recipe.write_text(json.dumps(report,indent=2)+'\n')
    return {'threemf':str(target),'preview_png':str(thumbnail),'recipe_json':str(recipe),'stats':report}
