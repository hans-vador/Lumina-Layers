# Region layers

A separate option next to Reference layers (`core/band/regions.py`; Reference
layers itself is unchanged). The relief is Reference layers' exactly — same
tone, height rule, thin-feature cleanup and mesh — but the cover is split into
up to three large colour regions and each region gets its **own spool
sequence**, from one shared palette of at most four spools.

Reference layers prints one spool per layer across the whole plate, so two
places at the same height always print the same colour: Graduation's pink
title on the equally bright purple sky vanishes, and In Rainbows' red and blue
rows print alike. Region layers keeps the depth and detail and adds hue where
the cover needs it.

Choose **Region layers (face up)** in the album UI, or:

```sh
.venv/bin/python scripts/region_band.py --image /path/to/cover.png --out output/regions --width 200
```

## Method

1. **Start**: k-means (fixed seed) on the hue (CIELAB a*, b*) of the
   contrast-stretched cover.
2. **Fit**: every combination of library spools, dark -> light, and every swap
   plan is scored for every region at once (Beer-Lambert stack colours from
   the spools' TDs, least squares against the pixels printed at each height).
   The palette is the <= 4 spools (one AMS) whose best per-region plans fit
   best. All regions share the darkest spool, which also prints the backing.
3. **Reassign**: each pixel joins the region whose sequence predicts its colour
   best (errors blurred over 0.5 mm; a pixel only moves if the other region is
   2% better), then a 1 mm majority filter, islands under 2 mm² join their
   neighbour, and corner-only contacts are removed. Repeat until stable.
4. **How many regions**: 1, 2 and 3 are all tried. Score = mean squared colour
   error × (1 + 0.5% per layer that holds two spools + 5% per extra region);
   the lowest wins. One region prints exactly like Reference layers.

5. **Accents**: a whole-plaque score can't see small things — Blond's green
   hair is 0.8% of the plaque. After the regions are chosen, two kinds of area
   (30 mm² to 4% of the plaque each, up to 6 regions in all) are tried as extra
   regions:
   - **colour**: clearly coloured (C* >= 15) but printed in the wrong colour
     family — the hair printed brown;
   - **neutral**: grey / black-and-white (C* <= 8) but printed in a colour —
     the barcode and advisory label, whose hairline white gaps and letters are
     lowered by the thin-feature cleanup to mid height, which is Coffee Brown.
     These are grown 2 mm into the neutral pixels around them (so a label is one
     area) and may only use neutral spools (Black, White), scored so that adding
     colour to a grey costs 3× a lightness error.

   Neutral areas go first. An accent is kept when its own pixels get at least 4
   closer and the rest of the cover gets no more than 1% worse; later accents
   never split an earlier one. With ticked spools, the assignment that prints
   every one of them is found by dynamic programming over which spools are
   covered (trying every combination took minutes at six regions).

## Colour scoring

Main regions use plain CIELAB least squares, as Reference layers does. Accent
regions are scored **colour-first**: Stack5's hue-first metric, weighted harder
on hue (chroma above C* 4 counts as colour; a MORE vivid version of the right
hue costs only 20% of the usual chroma penalty), blended in from plain CIELAB
between C* 10 and 20 so greys and near-blacks are never judged by hue.

Why both: plain CIELAB prints Blond's dark olive hair (Lab 38 −18 16) Coffee
Brown, because brown matches its darkness (28 vs 31 for a thin Green veil).
Colour-first alone went the other way on greys — Rodeo's grey-green background
turned Pink and to_hell_with_it's shadows Red — so it is only used where the
cover is clearly coloured and only for accents.

With your ticked spools, every one of them is printed by at least one region.

## Printing

**One object**: the plaque (the base spool — the darkest, which also prints the
backing) plus **one modifier per other spool**, named after it: the union of
that spool's bands in every region, as exact prisms (region outline × the
band's layer range, starting at the bottom of the band's first layer so the
slicer's mid-layer sample lands inside it). Regions are disjoint and a spool is
one band per region, so modifiers never overlap, and they end 0.1 mm above the
relief — the whole print selects and moves as one piece. Layers where regions
use different spools need tool changes, so a prime tower is sized and placed
with Stack5's flush model (`core/stack5/flush.py`), the plaque at the left bed
edge.

## Results (your 12 spools, 200 mm)

| Cover | Regions | Colour error dE | Tool changes |
|---|---|---|---|
| Blond (Black, White, Coffee Brown, Oak, Green ticked) | 1 + 4 accents: hair Green, barcode + label Black/White, skin patches | 12.4 | 28 (5 h 10 min) |
| In Rainbows | 2 (red/orange rows vs the rest) | 20.9 (Reference 27.1) | 12 |
| Graduation | 3 (upper sky + bear outline + title, lower sky, bear/cream) | 21.6 (Reference 25.0) | 11 |
| to_hell_with_it | 2 (warm face/porch, cool house/fence) | 11.5 | 12 |
| Rodeo | 1 — prints like Reference layers | 14.5 | 0 |

(The rows other than Blond are from before accents; accents add a few small regions and tool
changes, e.g. Graduation +1 accent / 15 changes, In Rainbows +2 / 18.)
All slice in Bambu Studio without warnings; G-code layer by layer
matches the planned spools per region. Physical prints not yet tested.
