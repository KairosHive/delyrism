# symbol_vectorizer

Turns a folder of hand-drawn symbols (phone photos or scans, ink on paper) into vector symbols, then joins them with their descriptors into the space JSON that live Delyrism builds from.

The package is self-contained. It depends on numpy, scipy, networkx, pillow and opencv-contrib-python (the contrib build supplies the thinning used for centre lines), and on nothing in `delyrism`, so it can move to the live repository as is.

```bash
python -m symbol_vectorizer all new-symbols descriptors.csv -o symbols_out
```

```bash
python -m symbol_vectorizer vectorize new-symbols -o symbols_out
```

```bash
python -m symbol_vectorizer space symbols_out descriptors.csv -o symbols_out/space.json --name workshop
```

## What it does to each drawing

1. **Paper flattening.** The image is divided by a wide median-filtered copy of itself, which takes shading, vignetting and paper tint to white.
2. **Ink detection.** Each pixel scores by Lab chroma (coloured markers) and by darkness (black ink). Hysteresis keeps faint stroke edges connected to confident ink. Ruled lines, printed text and bleed-through from the back of the sheet stay grey and faint, so they drop out. Components touching the photo border and specks smaller than a pen dot are removed.
3. **Crop** to the ink with a 6 % margin.
4. **Palette.** K-means in Lab on confident ink, with a distance that weighs hue above lightness and saturation, since one marker varies in pressure and overlap and keeps its hue. Near colours merge. Anti-aliased edge pixels take the colour of the nearest confident ink pixel.
5. **Outlines.** Contours per colour, with holes, fitted with cubic Béziers (Schneider's algorithm, split at corners). The dominant colour fills the whole silhouette and the other colours sit on top, so adjacent colours meet without seams.
6. **Centre lines.** Each colour is thinned to one pixel and traced as a graph. Thinning spurs are pruned, and branch points closer than a pen width merge into one junction. Chains that continue in a straight line through a junction join into one stroke, so a "+" gives two strokes and a line through a loop stays one stroke. Strokes start at their upper-left end and are ordered top to bottom. Regions thicker than 1.8 pen widths (a coloured-in triangle) become fills.

## Outputs

Per symbol in `<out>/<name>/`

| file | content |
|---|---|
| `<name>.svg` | filled outlines in the drawing's colours |
| `<name>_lines.svg` | centre-line strokes with the measured pen width, plus fills |
| `<name>.png` | the cropped drawing on a transparent background |
| `<name>.json` | geometry in the normalised frame |

Per folder

| file | content |
|---|---|
| `symbols.json` | all symbol records, plus any files that failed |
| `preview.png` | one row per symbol with the source crop, the outlines and the centre lines (red dot = stroke start), all rasterised from the vector data |
| `space.json` | written by the `space` and `all` commands |

## Descriptors

The `space` step accepts any of these formats.

- JSON in the `structures/*.json` form, `{"DESTINY": ["fate", "path"]}`, or `[{"symbol": ..., "descriptors": [...]}]`
- CSV or TSV with one `symbol,descriptor` pair per row, or `symbol,"fate; path; ..."`
- a folder of sidecar files such as `destiny.txt`, one descriptor per line or comma-separated

Names match image file names case-insensitively, ignoring spaces, hyphens and underscores. Symbols without descriptors and descriptors without images are listed under `missing`.

## space.json

```json
{
  "schema": "delyrism.space/1",
  "name": "workshop",
  "frame": "strokes and fills: ink box centred on (0, 0), longest side spans [-1, 1], y up",
  "structure": {"DESTINY": ["fate", "path"], "MEMORY": ["trace", "return"]},
  "symbols": [{
    "id": "DESTINY",
    "image_id": "destiny",
    "descriptors": ["fate", "path"],
    "files": {"svg": "destiny/destiny.svg", "lines_svg": "destiny/destiny_lines.svg", "png": "destiny/destiny.png"},
    "aspect": 1.33,
    "pen_width": 0.048,
    "palette": [{"hex": "#ffa845", "rgb": [255, 168, 69], "share": 0.88}],
    "strokes": [{"kind": "line", "color": 0, "width": 0.047, "closed": false, "length": 1.16,
                 "points": [[0.467, 0.607], [0.459, 0.601]]}],
    "fills": [{"color": 1, "rings": [[[-0.9, -0.5], [-0.8, -0.52]]]}]
  }],
  "missing": {"descriptors": [], "images": []}
}
```

`structure` is the `{symbol: [descriptors]}` mapping that `SymbolSpace(symbols_to_descriptors=...)` takes. Stroke points are spaced 0.01 apart along the stroke, so a renderer can draw them at constant speed or index them by arc length. `color` indexes `palette`. `kind` is `line` or `dot`. File paths are relative to `space.json`.

## Tuning

| flag | default | raise it when |
|---|---|---|
| `--chroma-hi` | 0.25 | paper texture or a coloured background is picked up as ink |
| `--dark-hi` | 0.40 | bleed-through or pencil guide lines are picked up |
| `--merge-de` | 22 | one marker splits into two palette colours |
| `--fill-factor` | 1.8 | thick marker lines are classed as fills |
| `--fit-error` | 0.8 px | the SVGs are larger than needed |
| `--work-size` | 2000 px | fine detail is lost (slower) |

Lower the first two for pale pencil drawings.

## Choice of method

Classical tracing runs here because these drawings are clean marker strokes on paper, where the job is segmentation and stroke structure. The main alternatives, as of late 2026, fall into four groups.

**Outline tracers.** Potrace (Selinger 2003) fits Bézier outlines to a binary bitmap and stays the reference for black-and-white tracing (Inkscape's Trace Bitmap uses it). VTracer (visioncortex, Rust, `pip install vtracer`) adds colour clustering and stacked layers. Adobe Illustrator's Image Trace and the commercial Vectorizer.AI give the best results for logos and scans. All of these trace the outline of the ink, so a marker line becomes a thin filled shape with two sides, and none of them gives the pen's path.

**Centre-line tracers.** AutoTrace's centerline mode and skeleton tracing (thinning, then graph walking, as here) give one path per pen stroke. That path is the form needed for animating a symbol being drawn, for stroke counts and lengths, and for comparing symbols with movement.

**Learned line-art vectorizers.** Deep Sketch Vectorization via Implicit Surface Extraction (Yan et al., SIGGRAPH 2024) predicts an unsigned distance field and extracts centre lines with dual contouring. It handles rough, overdrawn pencil sketches that thinning breaks on. Earlier work includes the virtual-sketching framework (Mo et al. 2021), keypoint-driven line vectorization (Puhachov et al. 2021) and Deep Vectorization of Technical Drawings (Egiazarian et al., ECCV 2020). LineDrawer (2025, in the SketchKit library) extracts strokes from dense line art. These models pay off for messy sketches with many overlapping pencil passes.

**Image-to-SVG generators.** StarVector (CVPR 2025) and OmniSVG (2025, built on Qwen2.5-VL) write SVG code from an image with a multimodal language model. Earlier optimisation methods include DiffVG, LIVE (CVPR 2022) and Im2Vec. The generators produce compact, editable icon-style SVGs. They redraw the image from what they recognise, so a hand-drawn symbol comes back regularised, and the redrawing can change the gesture that makes each symbol distinct.

A learned centre-line model such as Yan et al. 2024 is the upgrade path if the workshop drawings turn out to be sketchy pencil with many overdrawn passes. The module keeps segmentation and tracing in separate files, so another tracer can replace `trace.py` while the segmentation and JSON stay the same.

## Limitations

- Ink drawn in the same grey as the paper's printed lines or bleed-through falls below the darkness threshold. Coloured markers and black ink separate cleanly.
- Two parallel strokes closer than about a pen width merge into one thick stroke in the centre-line trace. The outline SVG keeps them as drawn.
- Stroke direction and order follow a fixed rule (upper-left start, top to bottom). They describe the finished shape and carry no record of how the drawer moved.
- A photo with a strong shadow edge across the sheet can leave a dark band that reads as ink.
