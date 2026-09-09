# Relief Stack5 — full-colour 3D album covers

Relief Stack5 is an **add-on beside flat Stack5**. Flat Stack5 (`core/stack5/`,
`scripts/lumina_album.py`) is untouched and still prints face down. Relief Stack5
(`core/relief/`, `scripts/convert_relief.py`) keeps Stack5 in charge of colour and
lets a depth model shape the plastic under it.

```
original album image
   ├─ core.relief.color_stage  → Stack5 colour recipe, 5 layers per pixel   (unchanged Stack5)
   └─ geometry provider        → depth (+ normals, mask)
          └─ depth_processing  → align · normalise · invert · smooth · quantise · slope limit
                └─ relief_stack → 0.8 mm base + 0–4 mm relief + reversed 5-layer shell (face up)
                      └─ mesh → one closed box-union per filament (no dilation, no overlaps)
                            └─ export3mf → Lumina Bambu writer + Stack5 post-processor (X2D preset, flush, tower)
                                  └─ validate → the written 3MF is rasterised back and compared voxel by voxel
```

**Colour never comes from a model.** Depth Anything, Wonder3D and normal maps only
produce geometry. The material matrix for a given image + palette is byte-identical
to flat Stack5's (`stats.material_matrix_sha256`), whatever depth map is used.

## Dimensions (defaults)

| part | mm | layers (0.08 mm) |
|---|---|---|
| flat base | 0.8 | 10 |
| relief above the base | 0 … 4.0 | 0 … 50 |
| Stack5 colour shell | 0.4 | 5 |
| **total** | **1.2 … 5.2** | **15 … 65** |

`relief_steps = round(normalized_depth · 4.0 / 0.08)` (half-up), clamped to 0…50. Every
height in the model is a whole number of layers. With `--step-layers N` the result is
also clamped to the largest multiple of N inside the budget (48 for N = 3).

Physical order, bottom to top: base (backing filament) → relief body (backing filament)
→ `stack[4]` → `stack[3]` → `stack[2]` → `stack[1]` → `stack[0]` (visible). The backing
filament is the palette slot the Stack5 LUT was synthesised over — the colour of every
stack depends on it, so the body cannot be another material.

## CLI

```bash
# grey depth map: black = low, white = high
.venv/bin/python scripts/convert_relief.py album.png --depth-map album_depth.png --output album_relief.3mf

# same, with the palette fixed and cliffs limited to 8 layers between neighbours
.venv/bin/python scripts/convert_relief.py album.png --depth-map d.png --palette 'Black,White,Red,Oak,Beige' \
    --relief-mm 4.0 --base-mm 0.8 --layer-height 0.08 --color-layers 5 --max-neighbor-step 8

# 60 × 60 mm calibration relief (dome + 0–4 mm step band)
.venv/bin/python scripts/convert_relief.py --calibration --out output/relief
```

`python convert_relief.py …` at the repo root runs the same script. The desktop window
(`~/Downloads/BambuAlbums/Album Plaque Maker.command`, i.e. `scripts/lumina_album_ui.py`) has a
**Print style → Relief** switch with the same controls. Its default depth source is
**Estimate from the cover (Depth Anything V2)**, so no depth map has to be supplied; the
alternative takes a chosen depth map for a single image or finds `<image name>_depth.png`
next to each image for a batch.

Key options (`--help` lists all):

| option | default | meaning |
|---|---|---|
| `--geometry` | `imported-depth` | `imported-depth`, `depth-anything`, `wonder3d` |
| `--depth-map` | – | grey depth image / `.npy`; alpha < 10 = background |
| `--relief-mm` / `--base-mm` / `--layer-height` / `--color-layers` | 4.0 / 0.8 / 0.08 / 5 | vertical budget |
| `--invert-depth` | off | treat white as low |
| `--depth-normalize` | `percentile` | `none` takes the map as already 0…1 (headroom kept; values outside are clipped with a warning) |
| `--depth-low-percentile` / `--depth-high-percentile` | 1 / 99 | robust range |
| `--depth-smoothing` | 3 px | bilateral sigma_space on the 0.1 mm grid (edge preserving); 0 = off |
| `--median-px` | 0 | odd median kernel against speckle |
| `--min-feature-px` | 4 | closing + opening: no ridge / slot thinner than 0.4 mm |
| `--max-neighbor-step` | unlimited | max layers between 4-neighbours; higher side is ramped down |
| `--no-protect-silhouette` | – | also ramp the object/background boundary (default: keep it a vertical wall) |
| `--flatten-background` + `--foreground-mask` + `--background-level` | – | flat background from a mask |
| `--step-layers` | 1 | relief granularity (2 = 0.16 mm steps, fewer triangles) |
| `--width`, `--palette`, `--backing`, `--metric`, `--quantize`, `--advisory`, … | as Stack5 | colour stage (unchanged) |
| `--verify` | `auto` | rasterise the written 3MF and compare (`auto` = up to 250 k grid pixels) |
| `--edits` / `--edit-clean-px` | – / 4 | replay an art-direction script (see below); min-feature kernel for the edited region |
| `--fetch-sam-model` / `--sam-weights` | – / `facebook/sam-vit-base` | download the SAM checkpoint for click-to-select (explicit) / which one |

Outputs next to the 3MF: `_preview.png` (Stack5 colour shaded by the relief),
`_depth.png` (quantised steps), `_recipe.json`, `_ams.txt`, `_lut.npz`, and `_edits.json` when
art-direction edits were applied.

## Art-directed relief (select an object, change its shape)

Depth Anything gives a plausible relief, not necessarily the one you want: on
*Currents* the ball is only slightly raised above the ground plane. Art direction
lets you pick part of the cover and reshape it **without touching the colours**.

```
Depth Anything V2 (whole cover) -> normal relief processing -> steps (0.08 mm layers)
    -> ORIGINAL float32 height map in mm (steps x 0.08)
    -> mask edits: dome | extrude | raise/lower | smooth edge | reset      (core.relief.heightfield)
    -> clamp 0..4 mm -> quantise to layers -> min-feature cleanup in the edited region
    -> voxels / mesh / 3MF exactly as before (0.8 mm base + relief + 0.4 mm colour shell)
```

* **Masks** live on the relief grid (the cover resampled NEAREST to 10 px/mm, the same
  grid Stack5 matches colours on). They come from **SAM click-to-select**
  (`core.relief.segment.SamMaskProvider`, Hugging Face `transformers` `SamModel`, local,
  optional, lazily loaded) or from the **brush** (add / remove, size in mm). SAM proposes
  three masks per prompt (small / medium / large); positive and negative clicks refine one
  selection. Nothing is downloaded during normal runs: without the checkpoint the editor
  says so and the brush works.
* **Operations** (`core/relief/heightfield.py`, all deterministic, empty mask = no-op):
  `dome` - elliptical hemisphere, apex at *Height* above the flat base, rim on the continued
  background (`(1 - u**n) ** (1/n)`, `n = 2 / roundness`; roundness 1 = hemisphere, 0.5 =
  flat top, 2 = cone; `u` = normalised radius from the centroid to the outline, exact for an
  ellipse); `extrude` - flat plateau at *Height*; `offset` - signed *Raise / lower*;
  `feather` - blend across the outline over a band of *Feather* mm (pixels farther than half
  the band are untouched); `reset` - back to the Depth Anything values. *Edge feather* on
  dome / extrude / offset blends the new shape into the current relief inside the mask, so
  nothing outside the selection ever moves. Everything is clamped to 0..4 mm.
* **Scripts**: the edits are recorded as `<image>_edits.json` (`relief-edits/1`: ops with
  masks as base64 PNG, the authoring grid, the depth settings). The pipeline replays them
  (`convert_album_relief(height_edits=...)`, CLI `--edits`), saves a copy beside the 3MF and
  records `art_direction` in the recipe. A script authored at another width is resampled
  (warning). With no script the pipeline never enters this code, so unedited Relief output
  is byte-identical (tested).
* **Printability**: after quantisation the pipeline's min-feature rule (grey closing +
  opening, `clean_px` = 4 = 0.4 mm, `--edit-clean-px`) runs on the changed region only, so a
  1-px SAM tendril cannot become a 4 mm wall. Slope limiting is not re-applied to edits
  (an extruded plateau is an intentional cliff; every column is still solid to the bed).
* **Colour**: the Stack5 recipe is computed from the original RGB pixels before any
  geometry; `material_matrix_sha256` is identical with and without edits (tested).

### Editor

```bash
# from the Album Plaque Maker window: Print style = Relief -> "Art-direct relief..." -> Use these edits -> Generate
# standalone:
.venv/bin/python scripts/relief_editor.py ~/Downloads/BambuAlbums/tools/albumart/currents.jpg --width 150
# export the saved edits (the editor prints this command):
.venv/bin/python scripts/convert_relief.py currents.jpg --geometry depth-anything --width 150 --edits currents_edits.json
```

Left canvas: cover + selection (cyan) + SAM points; right canvas: the final relief as
grey (black = 0 mm, white = 4 mm) with the selection outline. Controls: tool (Click object
/ Brush add / Brush remove), brush size, SAM result size, Undo click / Clear / Fill holes;
Operation, Height (mm), Dome roundness, Edge feather (mm), Raise / lower (mm), Apply / Undo /
Reset all; Save edits... / Load edits... / Use these edits. The window reuses the depth
estimate for the export, so Generate does not run Depth Anything again.

### SAM checkpoint (optional, explicit download)

```bash
.venv/bin/python scripts/convert_relief.py --check-deps            # reports depth_anything and sam
.venv/bin/python scripts/convert_relief.py --fetch-sam-model       # facebook/sam-vit-base, ~375 MB, once
.venv/bin/python scripts/convert_relief.py --fetch-sam-model --sam-weights Zigeng/SlimSAM-uniform-77   # 39 MB light model
```

`torch` + `transformers` are the same packages Depth Anything already uses. The image
embedding is computed once per cover (a few seconds on CPU for ViT-B); each click then runs
only the mask decoder.

### Future AI object providers

`core/relief/object_provider.py` defines the interface: a provider receives an
`ObjectCrop` (RGBA crop + mask + bbox on the grid) and returns an `ObjectGeometry` (front
depth map and/or a mesh); `composite_object` renders the mesh orthographically (Wonder3D
rasteriser), normalises the front depth to a 0..1 profile and writes it into the relief
through the same primitive the dome uses (apex at *Height*, rim on the background,
feathered edge). The `object` op carries the profile in the edit script. Stable Fast 3D is
**not** integrated: `StableFast3DProvider.available()` returns `(False, message)`.

## Slope, support and printability

Every column is solid from the bed to its shell, so there are no overhangs by
construction: a silhouette is a fully supported vertical wall. What the cleanup
removes is what a 0.4 mm nozzle cannot print or what noise would add:

* `--min-feature-px 4`: grey closing then opening on the step map — 1–3 px spikes,
  ridges, pits and slots are removed, plateaus keep their exact position (reflected
  anchors, no 1-px shift).
* `--max-neighbor-step N`: the exact lower envelope under the step map whose slope
  never exceeds N layers per pixel (`h ← min(h, min_neighbour + N)` to the fixed point).
  Only the higher side is lowered. Pixels outside the protected region (foreground
  mask, and always the plaque cut-out) neither move nor constrain, so faces keep
  their outline while accidental cliffs inside a region become ramps.
* Bilateral smoothing runs on nearest-filled data (Neumann-like padding at the mask
  and replicate padding at the image border); afterwards the smoothed range is mapped
  back onto the pre-smoothing range, so ramps keep their extremes and an authored map
  keeps its headroom.
* With a foreground mask, depth is nearest-filled outside the object before it is
  resampled, so background depth never bleeds into the silhouette rim. Protected
  silhouettes are limited on both sides separately: cliffs inside the object and
  inside the background are ramped, the outline itself stays vertical.
* A flat depth map (no range) gives relief 0 everywhere; invert / flatten are skipped.

## Geometry providers

| name | input | convention |
|---|---|---|
| `imported-depth` | `--depth-map` grey image (8/16-bit PNG, TIFF, WebP, `.npy`, `.npz`) | black = low, white = high (`--invert-depth` flips) |
| `depth-anything` | the original image | Depth Anything V2 relative inverse depth: larger = closer |
| `wonder3d` | `--mesh` (obj/ply/glb/stl) or `--depth-map` rendered front depth | orthographic front view, larger = closer |

All return a `GeometryResult(depth (H,W) float32, normals (H,W,3) | None, mask (H,W) | None,
depth_convention)`. `depth_convention` is `higher-is-closer` (default) or `higher-is-farther`
(z-buffer distance); `GeometryResult.raised()` always yields "larger = raised".

### Depth Anything V2 (optional)

`pip install torch torchvision transformers` (installed in `.venv` on 6 Sep 2026). Weights are never
downloaded unless `--allow-download` is given; otherwise they must be in the Hugging Face
cache or passed with `--weights /dir`; `scripts/convert_relief.py --fetch-depth-model` downloads it
once (done on 6 Sep 2026, so the GUI's AI depth works offline). Default checkpoint
`depth-anything/Depth-Anything-V2-Small-hf` (~100 MB). Missing packages or weights raise
`MissingDependencyError` with the install command (`--check-deps` reports the status).
The prediction is resampled to the original image size, then onto the Stack5 grid.

### Wonder3D (optional, adapter only)

Wonder3D makes six RGB views + normal maps and reconstructs a textured mesh. Only the
**geometry** is used: the mesh is rendered as an orthographic depth map from the original
front view and aligned with the artwork.

* Accepted mesh formats: anything trimesh reads (`.obj`, `.ply`, `.glb/.gltf`, `.stl`, `.off`);
  scenes are flattened.
* Coordinate system: Wonder3D's object frame is centred at the origin, roughly unit size;
  its input view is the front view, camera on +Z looking down −Z with +Y up. Defaults
  `--front-axis +z --up-axis +y`; change them if your reconstruction exported another frame.
  The mesh is rotated so front → +Z, up → +Y and projected orthographically (image x = X,
  image y = −Y, depth = Z, max z-buffer, back faces hidden).
* Alignment: Wonder3D crops/recentres the object, so `--fit mask` (default when
  `--foreground-mask` is given) maps the mesh's XY bounding box onto the mask's bounding box;
  `--fit image` maps it onto the whole frame (use when the cover *is* the square input).
* Foreground: rendered pixels form the mask; the background gets the far value and can be
  flattened with `--flatten-background`.
* Or skip rendering: `--depth-map front_depth.png --depth-map-convention higher-is-farther`
  for a z-buffer rendered elsewhere.

### Normals (extension point, Milestone 4)

`core/relief/normal_integration.py`: `dz/dx = -nx/nz`, `dz/dy = -ny/nz`, Poisson integration
in the DCT domain (Neumann boundaries, exact for integrable fields), and
`fuse_depth_normals` minimising
`depth_weight·|z−d|² + normal_weight·|∇z−g|² + smoothness_weight·|∇²z|²` in closed form.
CLI: `--normals map.png [--normals-y-up] --normal-weight 1 --depth-weight 0.1`. Normal maps
are image-space (x right, y down, z out); OpenGL y-up maps need `--normals-y-up`.

## Validation

`tests/test_relief_stack5.py` covers: flat map → uniform relief; gradient → monotonic 0…50;
whole-layer heights within 0…50; five colour layers per pixel with `stack[0]` on top and
`stack[4]` deepest; backing below every shell voxel; 10-layer base; 65-layer / 5.2 mm maximum;
mesh volume = voxel volume, no gaps/overlaps (rasterised back from the 3MF); recipe SHA-256
unchanged across depth maps **and equal to flat Stack5's**; byte-identical reruns; valid
3MF XML/zip; provider errors; Wonder3D render; normal integration; calibration model; CLI.

## First physical print (calibration)

```bash
.venv/bin/python scripts/convert_relief.py --calibration --out output/relief
```

60 × 60 mm, palette Black/White/Red/Oak/Beige (Beige is the backing / body): a Beige
background at 1.2 mm, an eleven-step band along the bottom from 0 to 4.0 mm relief
(0.4 mm per step — check with calipers), an Oak dome to 5.2 mm total with Black eyes
(+0.24 mm) and a Red mouth (−0.32 mm). Print face up on the smooth plate with the same
filaments and the Stack5 process the 3MF carries: 0.08 mm layers incl. first layer,
1 wall, 100 % zig-zag, no top/bottom shells, prime tower on, no brim. Expected
thickness 1.20 mm background / 5.20 mm dome centre.

### First-layer adhesion (shared with flat Stack5)

The first layer stays one 0.08 mm colour layer (the flat Stack5 viewing layer depends on
it), so adhesion comes from the process/filament values every Stack5 and Relief 3MF now
carries: first-layer speed 18 mm/s, first-layer infill 20 mm/s, elephant-foot compensation
0, part and aux fan off on the first layer, bed 65 °C (smooth and textured PEI), nozzle
225 °C on the first layer (220 afterwards). Filament keys are listed in each filament's
`different_settings_to_system`, otherwise Bambu Studio silently reverts them. Already
generated 3MFs can be patched in place with
`~/Downloads/BambuAlbums/tools/stack5_patch_settings.py <dir or file>` (no re-slicing of
geometry; the old config is saved beside the file). Also clean the plate (dish soap / IPA)
and run bed levelling before a plaque; a 3 mm brim is the remaining option if a plate still
lifts, but on face-down flat Stack5 it attaches to the visible edge, so it is not on by default.

## Limitations

* Flat Stack5's meshing style is kept: each filament is a union of closed axis-aligned boxes
  (exact volume, outward winding, no gaps or overlaps between parts) with T-junctions where
  neighbouring boxes differ in height, so the meshes are not trimesh-"watertight" in the
  edge-manifold sense; that is exactly what flat Stack5 produces and Bambu Studio slices.
  The rasterised verification (`--verify`, on by default up to 2.5 M grid pixels) is the proof.
* Relief spreads the five colour layers over up to 51 heights, so there are ~10× more
  tool changes than flat Stack5 (the prime tower is planned over every layer and still
  must fit beside the plaque; `--flush-scale` / `--width` if it does not) and 3–10× more triangles.
  `--step-layers 2` and stronger `--min-feature-px` reduce both.
* `--color-layers` must be 5 (the Stack5 LUT is 5⁵ stacks) and `--layer-height` 0.08 (LUT,
  filament TDs and the process preset are calibrated for it).
* Depth Anything *Metric* checkpoints predict metres (larger = farther); the provider flips
  them automatically when "metric" is in the model id (`depth_convention=` overrides).
* Depth maps of a different aspect ratio are stretched onto the image grid (warning recorded).
* Wonder3D is not installed; its adapter is tested with synthetic inputs only (a Wonder3D mesh
  from a real run should be checked with `--fit mask`). Depth Anything V2 Small and SAM ViT-B
  are cached locally (Sep 2026) and exercised by the tests when present.
