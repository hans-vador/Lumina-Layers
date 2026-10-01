# Graduation reference layers

This path recreates the successful reference's **geometry strategy**: an
unquantized brightness heightmap, a single closed solid, and global swaps -
for any cover, with spools chosen automatically.
It is separate from the multi-material-per-layer Stack5 matcher. No learned
model, dithering or TD matching changes the geometry; the only cleanup removes
raised features too thin to print (see "Thin features" below).

Choose **Reference layers (face up, Graduation style)** in the album UI, or:

```sh
.venv/bin/python scripts/reference_band.py --image /path/to/cover.png --out output/reference --width 200
```

Use a square source. The default does everything automatically, for any cover:

1. **Contrast** (`--tone auto`). The reference cover already spans black to
   white, so its brightness rule used ~20 of the ~23 colour layers. A dark cover
   does not: `to_hell_with_it` only reaches brightness .37, so 72% of it landed in
   the lowest five layers and printed as plain Black. Auto levels (0.5th/99.5th
   brightness percentiles -> 0..1) spread every cover over the whole relief.
   The mapping stays linear in brightness, like the reference, so edges and
   detail keep their shape. `--contrast` (0..1, default 0) blends in
   contrast-limited histogram equalisation; it is off by default because it
   gives extra layers to whatever covers the most pixels: Rodeo's flat dark
   background (55% of the cover) was lifted into the brown layers. The colours are then fitted to the
   cover re-lit to that tone (`_target.png`: each pixel scaled to the new
   brightness, keeping its hue).
   Filaments are never repeated (each spool prints one band, dark -> light).
   Allowing repeats was tested (beam search over per-layer sequences): it
   lowered the colour error by at most ~2 dE, added 2-3 swaps, and on Rodeo
   alternated Beige/Coffee Brown single layers into a speckled grey background.
2. **Spools** (`--palette auto`, UI "Auto"). Every combination of 2-4 spools from
   the library (one AMS; `--max-spools 5` or "Always 5 spools" for five),
   stacked dark -> light, gets an exhaustive swap search. For each layer count
   the Beer-Lambert colour of the stack (from each spool's TD) is compared with
   the mean CIELAB colour of the pixels printed at that height, least squares.
   An extra spool is only used if it lowers the mean error by at least 0.5 dE.
   With 12 spools that is 781 palettes and ~0.93M swap plans (~10 s).

Check: given Graduation's own four colours (and a translucent white, TD 3 -
assumed, not measured), the swap search returns .84 / 1.08 / 1.40 mm against the
creator's .92 / 1.08 / 1.32 (thin-floor heights), each within one layer.

The export carries a **predicted colour** preview, the target, the height map
and a per-size table of the best palettes in the JSON.

## Your own spools

Tick 2–5 spools in the album UI ("Pick spools" mode), or pass
`--palette 'Black,Klein Blue,Beige'`. Only the swap layers are searched then.
`--order fixed` keeps your order instead of dark -> light.

## Graduation colours

`--palette graduation` prints the reference's own slot colours
(#000000, #800080, #F55A74, #FFFFFF) at its own swap heights with the raw tone
rule; the preview is then labelled **height**, since those are not measured spools.

Because height comes from brightness alone, pixels of equal brightness print
the same colour whatever their hue — so fidelity is bounded by the palette. An
opaque spool (e.g. White at TD 0.1) jumps straight to its own colour; tints need
a translucent light spool (e.g. Beige, TD 4) above a coloured one. Pink title
text on a purple sky of the same brightness disappears, as it does in HueForge.

## Thin features

`min_feature_mm` (default 0.45, `--min-feature 0` for the raw rule) applies a
grey-level opening so every raised feature on every layer is at least that
wide. Bambu Studio's support check ignores lower-layer islands narrower than
one extrusion, so hairline ridges were reported as "floating regions" (e.g.
`to_hell_with_it` at 0.15 mm pitch: 6,620 such islands; after cleanup Bambu
reports no warning). It only lowers features, never raises them; on that cover
the mean change is 0.018 mm.

Defaults: 1 mm backing (five .20 mm layers), then .08 mm layers. The original
artwork and swaps translate together by .56 mm. New swaps are at 1.48, 1.64,
and 1.88 mm. This preserves band thickness, rather than stretching the image
or adding an opaque cap. `--backing 0` retains the reference thin black floor
and original .92, 1.08, 1.32 mm swaps. The artwork fills the plaque edge to edge; there is no raised frame. Nozzle temperature is 220 C, matching the tested online project;
the verified installed X2D startup scripts supply machine heating behavior.

The tone rule is z = .481055573 + 1.82374436 × q^.975045487, where
q = .299 R + .587 G + .114 B for sRGB channels normalized to 0..1.
These constants are an empirical fit to the provided reference, not the
HueForge algorithm. Geometry and full exported 3MF bytes are repeatable for
identical inputs and settings in the same software environment.

Validation (raw tone rule, `min_feature_mm=0`) against a separately seeded
100,000-point sample of the original interior: mean absolute height error .01216 mm; 96.41% within .04 mm. This is
an aligned tone-rule comparison, not a claim of mesh identity. The source in
the online file differs from the user's Graduation.jpg near the copyright /
advisory mark. Use the embedded source for a controlled comparison.

```sh
.venv/bin/python scripts/validate_reference_band.py \
  --reference /path/to/Kanye_-_Graduation.3mf \
  --image /path/to/embedded-source.webp --out output/reference-validation.json
```

A 200 mm export at .20 mm mesh pitch was sliced successfully in Bambu Studio;
G-code confirms backing heights .2/.4/.6/.8/1.0 and all three swap heights.
Generated surfaces are watertight with consistent winding. Physical print
parity remains to be tested. No print was sent to the printer.
