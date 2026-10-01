# Face-up Stack5 / SUNLU PLA+ validation

Stack5 now supports a flat, face-up plaque. The 0.20 mm first layer belongs to
its backing; the five optical layers above it are all 0.08 mm, or experimentally
0.04 mm. The palette search, LUT, mesh, recipe and purge order use the same
physical thicknesses. The native mirrored face-down mesh is rotated about Y,
with the backing and optical shell scaled independently. Existing face-down
CLI/API behavior remains available. The desktop album UI defaults to face-up.
Face-up supports single-sided plaques. Backing layers default to 0.2 mm after
the 0.20 mm first layer; the optical layers retain their selected 0.04/0.08 mm
height. Requested backing thickness snaps to the nearest complete backing
schedule: the default requested 1.0 mm is exactly five 0.20 mm layers.
An explicit 1.6 mm is eight 0.20 mm layers. Reducing the backing does not increase
the current five-layer optical limit; backing thickness and shading depth are independent.
Height-range modifiers end exactly at that boundary. `--backing-layer 0`
retains uniform thin backing layers for comparison. The recipe records both
the requested spacer thickness and `stats.actual_backing_mm`.

The photograph shows a washed-out background, but does not by itself establish
filament transmission distance (TD). The saved great_chaos_itunes recipe uses
0.08 mm layers, a white backing, and Black/White values fitted from Bambu
measurements. It may not be the exact photographed job. These values are not a
SUNLU PLA+ calibration. Reducing thickness changes opacity, but cannot guarantee
accurate blending when the filament model is wrong. Surface texture, lighting,
flow and temperature can also affect appearance. Face-up exposes the deposited
top surface rather than the build-plate texture.

## Start with the swatches

Generated files live in `output/stack5_sunlu_calibration/`. Each strip is 48 x
10 mm, plus its purge tower. Use the same SUNLU black and white as the album.
The top-left notch identifies the zero-white end. From left to right, patches
contain 0, 1, 2, 3, 4 and 5 white layers over black. Each has a 0.8 mm black base
and a five-layer shell; unused white layers are replaced with black underneath.

- `sunlu_white_on_black_0.08mm.3mf`: white thicknesses 0, .08, .16, .24, .32, .40 mm.
- `sunlu_white_on_black_0.04mm.3mf`: white thicknesses 0, .04, .08, .12, .16, .20 mm.

Bambu Studio 02.08.02.61 CLI accepted both projects. Actual exported G-code Z
heights were verified: first layer .16, followed by .08 or .04 increments.
The X2D 0.4 mm profile's .08 minimum is left unchanged. Slicer acceptance does
not establish reliable physical extrusion at .04 mm; that option is experimental.
The 3MF files are editable projects, not printer-ready G-code deliveries.

View the printed strips from above under the same diffuse lighting, alongside
a gray reference. Use fixed exposure/white balance when photographing them.
The useful question is whether one .04 or .08 white layer gives a gray rather
than already appearing white. Actual measurements can then inform white TD
via `--td 'White=VALUE'`, followed by a repeat swatch/album check. Do not fit TD
from the existing uncalibrated room photograph or assume a thinner layer fixes it.

## Generate an album

```sh
.venv/bin/python scripts/lumina_album.py \
  --image /path/to/cover.jpg --orientation face-up \
  --first-layer 0.20 --layer-height 0.08 \
  --palette Black,White --backing Black \
  --out output/stack5_faceup
```

Use `--layer-height 0.04` for the experimental version and a separate output
folder for comparison. The UI offers both flat orientations and a thickness
selector. Measured face-down LUTs are rejected for face-up jobs; they must not
silently be reused as a face-up calibration. Synthetic previews retain the
current approximate filament values until measured calibration is supplied.

## Machine script correction

The cached album template carried A1 machine scripts despite an X2D label.
Stack5 postprocessing now replaces its machine G-code fields with the resolved
X2D scripts from the installed Bambu Studio 02.08.02.61 BBL profile. Both `inherits` and `include` dependencies must be resolved: the X2D-specific
startup, end, layer-change, timelapse and tool-change scripts live in includes.
The earlier inheritance-only snapshot incorrectly captured a generic startup;
old exports containing that snapshot must be repaired and re-sliced. Regenerate
the corrected snapshot with `scripts/refresh_x2d_machine_gcodes.py`. The snapshot
and provenance are in `assets/x2d_machine_gcodes.json`; printer overrides are
listed in the project so Studio preserves them. Process and filament settings
remain separately managed, including the existing 205 C temperature choice.

## Verification

Regression coverage checks face-up optical thickness, reversed print order,
backing thickness, mesh Z boundaries, slicer profile heights, and preservation
of the legacy face-down and relief paths. Swatches use prescribed material
geometry independent of palette matching. Full physical color calibration and
HueForge feature parity are not claimed by this change.

## Stop at the best exposed color

Face-up Stack5 now searches all material recipes from zero to five color layers,
independently for each quantized image color. It does not merely shorten the
initial full-stack recipe. Pink and blue may occupy the same printed layer at
neighboring positions; darker blue may choose a different underlying stack or
height. All candidates use the actual selected backing in the optical model.

The matcher finds the lowest predicted color distance, then accepts candidates
within one distance unit under the selected metric. Within that small tolerance,
it prefers fewer material changes, fewer distinct materials, then fewer layers.
Thus a simple blue column beats a complicated mixture with a negligible predicted
advantage. The tolerance is in the configured metric, not necessarily CIE delta E.
Repeated layers of one translucent filament remain available: depositing that
filament once is not assumed to produce its opaque color.

Small-region cleanup now copies complete selected recipes rather than editing
layers separately. Preview colors and stopping heights follow the cleaned recipe.
Spatial cleanup can worsen local color error to remove tiny regions; the recipe
reports the actual final error and cleanup counts rather than promising an
unconditional no-regression in every pixel.

A stopped column has air only above its last occupied layer. It has no internal
gaps, cannot restart higher up, and always rests on the continuous backing.
This creates a shallow stepped surface (at most five optical layers above the
backing), rather than a flat five-layer slab. The exported geometry, preview,
material statistics and purge plan use the stopped columns. Empty top layers
are omitted from print-layer estimates. The recipe records counts of pixels
using 0..5 color layers and the fraction of optical material removed; this is
not a prediction of total filament or time saved because backing and purging
still contribute. A `_stop_layers.npy` file stores the actual 0..5 height map.

The desktop checkbox “Face up: stop each area at its best colour match” controls
this behavior. CLI/API defaults enable it for face-up only. Use
`--no-early-stop` or `early_stop=False` to retain the full shell for comparison.
Face-down and relief behavior are unchanged. Mesh dilation is rejected with
early stopping because it can deposit material into neighboring stopped areas.

Palette selection still uses the existing full-stack palette scorer; after palette
selection, pixel matching searches every depth. This is not a shared, single-color
filament-swap schedule. HueForge's [Mesh Color Match](https://shop.thehueforge.com/pages/mesh-color-match)
uses its user-defined Mesh Core; this Stack5 path permits several materials at
the same Z height as requested. Optical values still require calibration for the
actual spools. A candidate unavailable within five layers cannot be made opaque
by labelling it “direct color.” The 0.04 mm process remains experimental.

## Monotonic fill

New exports use monotonic internal solid infill and monotonic top/bottom surface
patterns. The exporter sets top solid depth to cover the planned print layers:
Bambu otherwise treats even 100% fill as sparse infill, which does not accept
monotonic. This changes fill toolpaths, not the material meshes or stop heights.
Existing 3MFs must be regenerated and re-sliced to receive these settings.

## Source outline protection and opaque white

Stack5 detects adjacent source pixels separated by at least 18 CIELAB units.
Both sides retain source colours (rounded to 16 levels per channel to bound
search memory), are rematched with stronger lightness weight, and are protected
from small-region cleanup. It preserves existing boundaries, not artificial
black strokes. Features below the nozzle width can still be lost during slicing.

The default White now uses effective TD 0.10 mm, based on the user's observation
that one 0.08 mm SUNLU PLA+ layer looks white over other colours. Under the
current k=7 model that is 99.63% opacity. This is an observation-based estimate,
not a measured calibration. Existing exports/LUTs must be regenerated.
