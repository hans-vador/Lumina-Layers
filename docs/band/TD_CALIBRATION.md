# TD / per-layer opacity calibration from Lumina's measured Bambu LUTs

Inputs: `lut-npy预设/bambulab/Bambulab&PLA&4色&RYBW&红-蓝-黄-白.npy` (1024 stacks, slots W/Red/Y/B, row i -> base-4 digits MSB first, digit[0] = viewing layer; **SET4**) and
`lut-npy预设/bambulab/Bambulab&PLA&8色+4色.npz` (rows 0..2737 = 8-colour "smart" session, rgb = `Bambulab&PLA&8色-New.npy`, stacks = `assets/smart_8color_stacks.npy` reversed; rows 2738..3751 = SET4 remapped; **SET8** = the first block).
All stacks are 5 x 0.08 mm over a 1.6 mm white backing, printed face-down (calibration.py `BACKING_MM = 1.6`).
Model (as requested): colour after m layers of X over backing B, in linear sRGB: `c = c_X*(1-(1-o)^m) + B*(1-o)^m`; `TD = -k*t/ln(1-o)`, k = 7 (optics.DEFAULT_K_OPAQUE), t = 0.08 mm.
Scripts: scratchpad `calib2.py / calib3.py / calib4.py` (numpy + scipy.optimize.least_squares, `.venv`).

## Data-quality findings (verify before trusting any single number)
1. **SET8 pure rows are NOT measurements.** Rows `[i]*5` of the 8-colour LUT equal the Bambu nominal hex codes exactly (W FFFFFF, C 0086D6, M EC008C, Y F4EE2A, K 000000, Red C12E1F, DeepBlue 0A2989, Green 00AE42). They were excluded from every fit. SET4 pure rows are measured (Red^5 = 207,71,69; Y^5 = 246,244,80; B^5 = 25,33,128; W^5 = 255,255,255 normalised).
2. **The two sessions disagree** (122 stacks measured in both): dE76 mean 11.7, median 11.4, p95 20.0, max 30.9. SET8 is brighter / more saturated (Red W^4: SET4 205,85,61 vs SET8 236,95,59; DeepBlue W^4: 86,118,181 vs 103,143,221). Any model fitted to one session is off by ~10 dE on the other.
3. **A scalar per-layer opacity is a poor physical model of these prints.** Each pigment is nearly transparent in its own channel and opaque in the absorbed channels (Red: o_R 0.16 vs o_G/o_B 0.86/0.94; Yellow: o_R/o_G 0.17/0.26 vs o_B 0.85; Cyan: o_B 0.25 vs o_R 0.54). Consequences: one red or yellow layer over white already looks "fully" red/yellow (the absorbed channels saturate at once) while red over black is a dark maroon (red light passes through). No scalar o reproduces both. Fit quality over ALL measured stacks with nominal hex colours: scalar best-fit dE mean 24.6 (SET8) / 31.4 (SET4); per-channel o dE 13.4 / 17.2; per-channel with free colours 11.2 / 6.2. -> `td_rgb_mm` (per-channel TD) is provided alongside `td_mm` and is strongly recommended for the Stack5 synthetic LUT.
4. **Viewing layer (squished first layer on the bed)** is slightly MORE translucent than inner layers: fitting o_view = r * o_inner gives r = 0.74 (SET8) / 0.81-0.82 (SET4); per-filament split fits give the same direction for every colour (e.g. DeepBlue 0.72 vs 0.94, Cyan 0.42 vs 0.56, Red 0.61 vs 0.69). The fit improvement is marginal (dE 24.6 -> 24.1), so the data only weakly supports it: use TD_view ~ 1.25 x TD_inner if the builder distinguishes layer 0.

## White (from W^m X^(5-m) veils, channels where the X substack is dark)
| session | rows | o_W per 0.08 mm (LS) | median per-row | TD (k=7) | per-row o by m |
|---|---|---|---|---|---|
| SET4 | 12 (over Red, Y, DeepBlue) | 0.057 | 0.060 | 9.1-9.6 mm | Red .063/.044/.059/.048, Y .093/.098/.087/.061, DB .047/.045/.073/.042 |
| SET8 | 2 (W^4 C, W^4 M) + all-stack fit | 0.146 / 0.080-0.116 | 0.146 | 3.5-6.7 mm | C .107, M .185 |
Chosen: **TD_W = 5.0 mm (o = 0.106)**, measured band 3.5-9.5. Per-channel (all-stack fits): o_W = 0.03-0.10 / 0.06-0.14 / 0.06-0.15 (R/G/B) -> white is a little more transparent to red light; `td_rgb_mm` = [5.8, 4.5, 4.2].

## Colours - scalar o fitted over X^m W^(5-m) (+ X^m K^(5-m) where present)
Free c_X (as specified) and c_X fixed to nominal hex. "per-m" = implied single-layer o from each m (it drifts downward with m: the absorbed channels saturate first).
| X | session | free-c: c_X(sRGB), o, TD, dE(fit pts) | nominal-c: o, TD, dE | per-m implied o (m=1..) |
|---|---|---|---|---|
| Red | SET4 | (211,73,0) 0.972 0.16 23.1 | 0.933 0.21 12.3 | 1.0 1.0 .47 .55 .94 |
| Red | SET8 | (235,82,56) 0.982 0.14 3.4 | 0.844 0.30 19.1 | .98 .83 .99 .97 |
| Y | SET4 | (233,235,78) 0.999 0.08 3.7 | 0.951 0.19 11.3 | 1.0 1.0 .99 .29 .19 |
| Y | SET8 | (251,248,69) 0.970 0.16 1.3 | 0.906 0.24 6.1 | .97 .94 .85 .97 |
| DeepBlue | SET4 | (0,0,154) 0.839 0.31 25.1 | 0.823 0.32 9.1 | .84 .78 .61 .59 .94 |
| DeepBlue | SET8 | (0,78,207) 0.820 0.33 5.9 | 0.632 0.56 34.2 | .79 .81 .99 .97 |
| Green | SET8 | (122,202,76) 0.758 0.39 11.4 | 0.559 0.68 22.3 | .73 .62 .59 .97 |
| C | SET8 | (73,155,218) 0.672 0.50 6.1 | 0.561 0.68 9.4 | .72 .61 .53 .37 (K-backing .29) |
| M | SET8 | (208,45,150) 0.592 0.62 21.0 | 0.566 0.67 22.9 | .60 .50 .39 .32 (K .21) |
| K | SET8 | single K under j whites (86,89,96)/(110,117,127)/(156,161,165) -> o_K = 0.96-1.0 for o_W 0.06-0.15 | | |

## Colours - scalar and per-channel o fitted over ALL measured stacks, colours fixed to nominal hex (= what the LUT builder will do)
| X | scalar o SET4 / SET8 | scalar TD SET4 / SET8 | per-channel o (R,G,B) SET8 [SET4] | per-channel TD (R,G,B) SET8 |
|---|---|---|---|---|
| K | - / 1.00 | - / <=0.10 | .83 .91 .85 | 0.32 0.24 0.30 |
| W | 0.00* / 0.08 | - / 6.7 | .09 .14 .15 [.03 .06 .08] | 5.8 3.6 3.4 |
| Red | 0.644 / 0.702 | 0.54 / 0.46 | .16 .86 .94 [.30 .89 .98] | 3.3 0.29 0.20 |
| Y | 0.214 / 0.326 | 2.33 / 1.42 | .17 .26 .85 [.11 .18 .90] | 3.1 1.9 0.30 |
| DeepBlue | 0.900 / 0.804 | 0.24 / 0.34 | .80 .68 .23 [.92 .77 .38] | 0.35 0.49 2.1 |
| Green | - / 0.554 | - / 0.69 | .44 .27 .73 | 0.96 1.76 0.43 |
| C | - / 0.468 | - / 0.89 | .54 .39 .25 | 0.73 1.13 1.98 |
| M | - / 0.255 | - / 1.90 | .06 .48 .39 | 9.4 0.85 1.12 |
(*scalar SET4 fit pushes o_W to 0 because a scalar white cannot darken the red channel the way the measured veils do; the direct veil estimate above (0.06) is the usable number.)

## Consistency check 4-colour vs 8+4 npz (Red / Y / DeepBlue / W)
Stack tables: the npz's second block is the 4-colour file remapped (1014 rows, identical values, 1015 common keys including pure W). Physical consistency between sessions: dE 11.7 mean (see above). Scalar all-stack o agrees within 0.06-0.11 (Red .64/.70, Y .21/.33, DeepBlue .90/.80); white differs 2.5x (0.06 vs 0.15) - probably a different white spool and/or exposure; SET8's value matches HueForge's Bambu Jade White (TD 4-5).

## Evaluation of candidate TD tables over all measured stacks (nominal hex, dE76 mean / median / p95)
| table | SET4 (1024) | SET8 (2730) |
|---|---|---|
| seed guesses in assets/filaments_user.json (K .2 W 4 Red .9 Green .9 Y 3 Blue .8) | 29.3 / 28.1 / 44.7 | 27.1 / 25.7 / 50.1 |
| X-over-W convergence TDs (Red .16 Y .12 DB .33 Green .39 C .5 M .62) | 38.3 / 39.9 / 75.9 | 33.8 / 32.8 / 60.9 |
| **FINAL scalar td_mm** (below) | 27.0 / 25.4 / 46.3 | 25.5 / 24.0 / 47.5 |
| **FINAL per-channel td_rgb_mm** (below) | 17.7 / 17.8 / 28.7 | 13.7 / 12.2 / 27.5 |
Session disagreement (~11 dE) is the floor for any single table; per-channel TDs roughly halve the scalar error.

## Mapping to the user's 12 spools -> assets/filaments_user_measured.json
| spool | hex | source | td_mm | o/layer | td_rgb_mm (R,G,B) | td_over_white_mm | basis |
|---|---|---|---|---|---|---|---|
| Black | #000000 | lumina_measured | 0.15 | 0.976 | .15 .15 .15 | 0.10 | K: single layer under whites o >= 0.96 |
| White | #FFFFFF | lumina_measured | 5.0 | 0.106 | 5.8 4.5 4.2 | - | veils: SET4 0.06 (TD 9.3), SET8 0.10-0.15 (TD 3.5-6.7) |
| Coffee Brown | #6E4F3A | estimate | 0.25 | 0.894 | .6 .25 .2 | 0.2 | between K (0.15) and Red (0.5) |
| Klein Blue | #1E44BE | lumina_proxy | 0.30 | 0.845 | .35 .49 2.1 | 0.4 | Deep Blue proxy (o .80/.90); Klein is lighter -> possibly 0.3-0.5 |
| Red | #C12E1F | lumina_measured | 0.50 | 0.673 | 3.3 .29 .20 | 0.25 | identical SKU, o .644/.702 |
| Green | #00AE42 | lumina_measured | 0.70 | 0.551 | .96 1.76 .43 | 0.68 | identical SKU, o .554 |
| Sunny Orange | #FF9016 | estimate | 1.0 | 0.428 | 3.2 .6 .25 | 0.3 | geometric mean Red/Yellow; R passes, B absorbed |
| Oak | #B08A5C | estimate | 1.6 | 0.295 | 2.5 1.4 .7 | 1.0 | mid-tone tan, white base + brown pigment |
| Lavender Purple | #A78BD4 | estimate | 2.5 | 0.201 | 2.2 1.6 4.0 | 1.5 | pastel from M (1.9)/C (0.89) diluted to white (5.0) |
| Pink | #F5A3B7 | estimate | 4.5 | 0.117 | 6.0 2.5 3.5 | 3.0 | near-white + little magenta |
| Beige | #E8DCC5 | estimate | 4.0 | 0.131 | 5.0 4.5 3.0 | 3.0 | near-white + little yellow/brown |
| Vivid Yellow | #F4E62A | lumina_measured | 1.8 | 0.267 | 3.1 1.9 .30 | 0.20 | Bambu Yellow o .214/.326; only B absorbed |
`td_mm` = scalar all-stack fit (least-bad for synthesising a full LUT with nominal hex); `td_over_white_mm` = how fast the colour reaches its full look on a white base (use for "X on top of white" predictions if only a scalar is available); `td_rgb_mm` = per-channel Beer-Lambert TDs (recommended). Extra keys are ignored by `core.band.optics.load_filament_library`.
