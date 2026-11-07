# DELYRISM

> Context-aware symbolic archetype explorer — watch meanings shift and relationships rewire when you give the engine a sentence, a sound, an image, or a blend of two contexts.

<p align="center">
  <img src="https://github.com/user-attachments/assets/76752b6c-6893-4eb8-b8c1-5d2670e7e5a0" width="720" alt="Meaning space with context shift arrows">
</p>

A FastAPI + Next.js stack. Slider tweaks are instant; backend memoises embeddings, UMAP layouts, and shift matrices so context changes don't trigger full recomputes. The previous Streamlit prototype lives on the [`old`](https://github.com/KairosHive/delyrism/tree/old) branch.

## Quickstart

```bash
conda create -n delyrism python=3.10 -y && conda activate delyrism
pip install -r requirements.txt -r web/backend/requirements.txt
cd web/frontend && npm install && cd ../..

# two terminals
uvicorn app.main:app --reload --port 8000 --app-dir web/backend
NEXT_PUBLIC_API_BASE=http://localhost:8000 npm --prefix web/frontend run dev
# open http://localhost:3000
```

### Local realtime stack (Windows · NVIDIA GPU · TouchDesigner)

The `realtime` branch streams the live Δ-graph over OSC into a TouchDesigner
patch (`td/delyrism_particles.toe`: GPU particles, the Lakota symbol mosaic,
a spoken voice with live effects).  Everything is set up from this repo:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/setup.ps1   # uv envs (.venv, .venv-tts), Node, npm, TD agent skills
powershell -ExecutionPolicy Bypass -File scripts/run.ps1     # backend :8000 · frontend :3000 · TTS :8010 · TouchDesigner
```

Only TouchDesigner 2025.x is installed by hand.  Models (Hugging Face
embedders, Chatterbox-Turbo, MediaPipe) download on first use; the voice
references are in `web/tts/voices/` (regenerate with `web/tts/make_voices.py`),
the symbol assets in `td/assets/lakota/`.  TDMCP (TouchDesigner ↔ agent bridge)
is embedded in the patch; `.mcp.json` points Claude Code at it.

For Cloudflare-backed embedders + LLMs, set `CLOUDFLARE_ACCOUNT_ID` and `CLOUDFLARE_API_TOKEN`. Or pick the local `sentence-transformer` backend in the sidebar for fully-offline use. Railway deployment: [`RAILWAY_SETUP.md`](RAILWAY_SETUP.md).

## What it does

Each Explorer panel is a different lens on the same shifted descriptor cloud:

| Panel | What it surfaces |
|---|---|
| **2D Meaning Space** | UMAP / t-SNE / PCA layout · symbol clusters · context-shift arrows · optional pull-intensity heatmap |
| **Rankings** | top archetypes for the current context (cosine coherence ⊕ personalized PageRank) |
| **Ambiguity** | dispersion / leakage / soft entropy per symbol — which archetypes blend, which stay sharp |
| **Descriptor attention** | softmax attention per archetype's descriptors under the current context |
| **Subgraph** | top-K archetypes + their top descriptors as a force-directed graph |
| **Within / between Δ heatmap** | descriptor-pair or symbol-pair cosine change (after − before) |
| **Δ-graph** | the headline relational view — strongest descriptor pairs whose similarity moved most |
| **Contextual transformations** | who migrated archetypes + per-archetype identity cards (before vs after top descriptors) |

## Context sources

Stack any combination. They all feed the same `v_ctx`:

- **Sentence** — free-text prompt in the Context Prompt card.
- **Symbol weights** — manually bias toward archetypes via sliders in the sidebar.
- **Image** — drop/paste an image; a Cloudflare vision LLM reads it as a short paragraph, then your text embedder encodes that paragraph (works with any text backend, no CLIP needed).
- **Audio** — upload or record (requires the CLAP embedder backend).
- **Alchemist mode** — two sentences A and B with a morph slider; the engine blends them server-side and every panel updates live as you drag.

## Shift strategies

The cloud rewriting is configurable. From the sidebar:

- **`gate`** — additive pull toward `v_ctx`, with per-descriptor gating (`relu` / `cos` / `softmax` / `uniform`). Fast, deterministic, default.
- **`pooling`** — convex blend (`avg`) or element-wise (`max` / `min`) toward `v_ctx`.
- **`reembed`** — re-encode each descriptor through the embedder with the context sentence prepended. Slowest, most semantically rich.
- **`hybrid`** — linear blend of `gate` and `reembed`.

Each strategy produces a different `D′` and every Δ panel updates accordingly. Rankings / Attention / Subgraph use the original `D` + `v_ctx` (they're a spotlight on a fixed landscape, not a landscape rewrite).

## Live mode — context × query, animated Δ-graph

A real-time tab built on the same Δ-graph as the Explorer. Two sentences, both continuously editable:

- **Context** rewrites the descriptor cloud. Every edit (debounced ~180 ms) recomputes the Δ-graph server-side and the new edges flow into the running layout — nodes keep their positions, so the graph morphs instead of re-seeding.
- **Query** selects the overlap. Every node of the current graph is scored by cosine to the query vector; the score becomes geometry. The query has live sources besides text, switchable on the fly in the query card (which ones appear depends on the backend the space was built with):
  - **text** — a typed sentence, embedded with the space's text tower.
  - **mic** — a sliding window of microphone audio (binary frames: kind byte, sample rate, 16-bit PCM) through the space's audio tower. `clap` or `peav` backend.
  - **camera** — the latest webcam frame through the CLIP image tower. `clip` backend (open-clip ViT-B/32).
  - **video** — a sliding window of webcam frames (window seconds × model fps, capped at 48 frames) through the PE-AV video tower. `peav` backend.
  - **audio + video** — the same window of sound and frames through PE-AV's joint audio-visual tower. `peav` backend.
  - **pose**: the dancer defines the query with their movement. MediaPipe tracks in the browser on the GPU (`@mediapipe/tasks-vision` through WebGL, since MediaPipe's Python build has no GPU path on Windows), with a choice of tracker: full body (33 points), hands (21 per hand, with fingers, no body needed), face (478 points, drawn as contours) or body + hands. Only the landmarks go to the server, which keeps the running trajectory, smooths it with a One Euro filter and draws it as moving landmarks with fading trails behind the chosen joints. The drawing comes back as a live preview. On each send tick the drawing is embedded. A `peav` space gets the last video-window seconds as a clip through the video tower, scored in the video head, and a `clip` space gets the current pose with its trails as a still. While nothing is in frame the last query holds. The last 10 s export as an mp4 of moving landmarks, and the last 30 s of landmarks export as an npz.
  - **movement → text**: the same tracking (body, hands, or body + hands) feeds `delyrism/motion.py`, which measures movement qualities over the last few seconds and writes the salient ones as one layered description that names no body part: Laban's effort action or state ("thrusting, decisive, explosive", "dreamy, buoyant, weightless"), the qualities behind it ("sudden, forceful, pinpoint, staccato"), the movement in space ("opening wide, rising, travelling through space") and its timing ("pulsing at about 120 beats per minute, off-balance"). A new description is generated at the chosen refresh rate and replaces the query outright. Before scoring, the server subtracts the mean embedding of reference descriptions spanning the effort space (the `contrast` slider), which removes what all movement descriptions share. On CLIP over the Four Elements this takes the rank correlation of node relevance between contrasting movements from 0.79 to 0.43. Each quality is also measured against the session's own range (the `session-relative` slider, with `↺ new session` to start over). With a fixed calibration, a real dancer sat at one end of several qualities for most windows (always fluid, rarely rhythmic), so the same words came back. Re-centred on the median and spread of the last two minutes, consecutive windows of a dance reel gave a different subgraph 67% of the time, up from 38%, on a par with typed queries. The live graph starts from the 30 strongest Δ edges (about 16 nodes on the Four Elements), where the same reel gave a different subgraph in 83% of windows. Stillness is always judged on the fixed scale. The text goes through the space's text tower, so this mode works with every backend. The panel shows the description, when it last reached the query, a bar per quality with its two poles, and the trends (opening, rising, travelling, turning, tempo).
  Sound and pictures highlight the same semantic space as words. Two smoothings are sliders: a blend of successive media *vectors* before scoring, and a blend of the per-node *activations* after scoring (the latter calms the springs without blurring the embedding).

**PE-AV** (`peav`, [facebook/pe-av-small](https://huggingface.co/facebook/pe-av-small), Apache-2.0, 0.8 B parameters, 1024-d) is Meta's Perception Encoder Audio-Visual model: one checkpoint with an audio tower, a video tower (ViT-PE-core-L at 336 px, any number of frames) and a joint audio-visual tower, each aligned with its own text head. A `peav` space keeps one descriptor matrix per head and scores an audio / video / joint query inside the head it came from (`SymbolSpace.D_heads`, `using_head`). The loader in `delyrism/peav.py` ties the shared towers by hand because transformers ≥ 5.22 fails on the checkpoint's tied weights. It needs `transformers>=5.22` and wants a GPU: on an RTX 4090 an 8-frame window embeds in ~0.4 s (≈55 ms per frame), audio in ~0.2 s; on CPU a frame costs ~0.6 s.

Transport is a WebSocket (`/realtime/graph/{space_id}`): the client sends `context`, `query` and `params` messages, the server answers with `graph` (nodes, edges, relevance) or `relevance` payloads. Shaping knobs: query focus (relevance exponent), contraction, damping, overlap cut, plus the Explorer's Δ parameters (top |Δ| edges, β, sign, shift strategy). The overlap subgraph is also listed as text, grouped by archetype.

The companion module `delyrism/traversal.py` (`DeltaWalker`) is a transport-agnostic walker over the same Δ-field — it steps from descriptor to descriptor following the context pull and assembles the trail into a new symbol. It is exposed on `/realtime/ws/{space_id}` and usable from plain Python, but has no UI in this tab yet.

The pose pipeline lives in `delyrism/pose.py` and runs without the server. `PoseTracker`, `HandTracker`, `FaceTracker` and `BodyHandsTracker` wrap MediaPipe's landmarkers (on the CPU, for video files), and `LandmarkFilter` smooths their output. `PoseTrajectory` stores the landmark time series, with missed detections kept as NaN rows so trails break where the dancer leaves the frame. `render_frame`, `render_clip` and `write_video` turn a trajectory into an image, a (T, H, W, 3) clip or an mp4.

`delyrism/motion.py` reads movement qualities from a trajectory in body-relative units (torso lengths, or hand lengths for hands). The dimensions follow Laban's Effort factors and Shape: energy, Time (sustained to sudden, from burstiness and oscillation rate), Weight (light to strong, from acceleration), Space (indirect to direct, from path curvature per stroke), Flow (bound to free, from spectral arc length and held stops), shape (reach from the body centre), level (raised arms, crouch), symmetry, rhythm with tempo, plus travel, approach, turning and speed trends. Hands add openness, finger articulation and the hands drawing together or apart. `SessionNorm` re-expresses each quality against its recent history in the session (`--relative 0` on the command line keeps the fixed scale). From the shell:

```bash
python -m delyrism.pose dance.mp4 --tracker hands --start 30 --end 60 --fit body --trace 1.5 --npz dance.npz
python -m delyrism.motion dance.mp4 --tracker body --window 3 --every 1 --values
```

MediaPipe's models (pose 9 MB, hand 8 MB, face 4 MB) download on first use, into `~/.cache/delyrism/mediapipe` for Python and into the browser cache for the live tab.

## Graphemes: drawing motion in one orthogonal basis

The `delyrism.graphemes` module holds this work. `delyrism/graphemes/glyphs.py` gathers symbols as pen motion. Omniglot provides 1,623 characters from 50 alphabets, each drawn by 20 people with the pen's timing (about 45 MB, fetched once into `~/.cache/delyrism/glyphs`). Symbol fonts add alchemical, planetary, zodiac, I Ching, religious, runic, Ogham, Phoenician, Old Italic, Gothic and Old Turkic signs (430 in all), thinned to centre lines, traced into strokes and timed with the two-thirds power law of human drawing. The Four Elements, Planets and Chinese Zodiac structures map to their canonical glyphs (`glyphs.structure_drawings`). Structures without a glyph enter through one drawing of each symbol.

`delyrism/graphemes/basis.py` turns a drawing into a 3D path at 60 Hz (pen-down strokes on the drawing plane, pen lifts above it), cuts it into fragments, and expresses each fragment's velocity in an orthonormal wavelet packet basis. The wavelet family and packet tree are the joint best basis (Coifman and Wickerhauser) over 60,000 fragments from Omniglot's 30 training alphabets, the orthonormal basis that writes them with the lowest entropy, which selects Symlet-6. Its six strongest atoms are the graphemes: a straight stroke, an S-bend, a bow and three finer waves (named by the path each draws across a stroke), each oriented by the phase of its coefficient, plus three atoms for the lifts. Fragments are cut at pen changes, then wherever the six graphemes rebuild each piece best: a dynamic program minimises the motion they miss plus a fixed cost per cut (`segment_optimal`, `CUT_COST`). Every fragment is normalised to the same length before projection, so the graphemes are scale-free and a fine wiggle splits into small S-bends and bows. With this cut, six graphemes rebuild 96% of drawings from the 20 unseen alphabets and 97% of the font symbols within 2% of the symbol's size. The basis itself was learned on fragments cut at speed minima and sharp turns (`method="speed"`).

Orthonormality splits a fragment's energy exactly across atoms, so a motion becomes a distribution over grapheme tokens (atom, direction, and position on a 3×3 grid of the symbol), and a symbol is the distribution of its glyph or of one drawing of it. `symbol_weights` turns a motion's distribution into weights over any set of symbols, with a temperature that sets how sharply they concentrate. A rough redraw of an Element, Planet or Chinese Zodiac glyph (rotated up to 20°, stretched up to 15%, noisy, re-timed) gets the right symbol 95 to 99% of the time. Between different people drawing the same unseen character, with one reference drawing per symbol, the right symbol comes first 44% of the time among 20 (chance 5%).

```python
from delyrism.graphemes import glyphs, default_basis
B = default_basis()                                  # delyrism/graphemes/data/grapheme_basis.npz, 6 graphemes
planets = glyphs.structure_drawings("planets")
profiles = {s: B.profile([d]) for s, d in planets.items()}
p = B.encode(xyz)                                    # (T, 3) motion at 60 Hz, z > 0 while the pen is up
weights = B.symbol_weights(p, profiles)              # {"VENUS": 0.71, "MARS": 0.12, ...}
```

```bash
python -m delyrism.graphemes gui                     # draw and watch the graphemes live (opens a local page)
python -m delyrism.graphemes symbols planets         # each symbol's strongest grapheme tokens
python -m delyrism.graphemes learn                   # relearn the basis from Omniglot
```

### From IMU or pose to symbol weights

`delyrism/graphemes/bridge.py` turns a movement into a hand path and reads it against a set of symbols through the graphemes. A pose joint (a wrist) is projected onto the body's front plane. A wrist IMU, or any fused quaternion, is read as a pointer, with gravity-referenced elevation and yaw as the path. Windows of fixed arc length slide along the path, each written in the wavelet basis, which gives P(grapheme | motion) and an ordered grapheme sequence. Ten visual symbols (circle, cross, X, triangle up, triangle down, square, spiral, wave, zigzag, infinity) are drawn procedurally in their common ways, and a motion's weight on each comes from aligning grapheme sequences (DTW). `add_symbol` adds any symbol from a drawing, and `add_symbol_path` adds a recorded movement as one more way of drawing a symbol.

`delyrism/graphemes/datasets.py` fetches the test data: UTD-MHAD (a wrist IMU and a Kinect skeleton recorded together while 8 people draw an X, circles and a triangle in the air), EMOKINE (one dancer in a 17-sensor Xsens suit, CC BY 4.0, only the motion CSVs are read out of the archive), and AIST++ 3D dance keypoints, read with an unpickler that only rebuilds numpy arrays. On UTD-MHAD, the Kinect hand path names the drawn shape among circle, X and triangle 94% of the time, and the right symbol comes first among all ten 43% of the time. Circle and spiral, and X and infinity, are close as air motions. With one person's recordings added as templates, the other seven people's gestures come first 86% of the time among ten from pose and 54% from the wrist IMU. On EMOKINE dance, the forearm-quaternion stream and the hand-position stream agree on the top symbol in 48% of 3 s windows (30% by chance).

`invented_symbols(n)` generates an alphabet nobody has seen (strokes built from lines, arcs, hooks, loops, zigzags and waves, chosen from 150 candidates to lie far apart in grapheme space). Six graphemes rebuild all 12 of the default set within 2% of their size (median 0.4%). Simple symbols sit close to most movement, so `calibrate(paths)` learns each symbol's distance mean and spread over ordinary movement, and `weights` then standardises distances (distinct mode, the default once calibrated; `mode="faithful"` keeps the nearest symbol). On UTD-MHAD's 21 arm actions this raises the number of different winning symbols from 4 to 7 and the share of the most frequent one falls from 56% to 20%. `glyphs.image_drawing(path)` turns a drawing image (photo, scan or export) into a symbol through the same thinning and tracing.

The GUI (`python -m delyrism.graphemes gui`, options `--drawings FOLDER`, `--out FILE`) writes one self-contained page and opens it in the browser. Its drawing pad cuts each stroke where the six graphemes fit best and shows the grapheme weights as you draw, per fragment and for the whole drawing, by grapheme and direction. The page also draws each grapheme as an animated stroke, mixes them, and rebuilds every image in the drawings folder fragment by fragment (the bundled samples by default).

## Story generator

A separate tab. Weaves Δ-graph motifs into micro-fiction with:

- 15 tone presets (Pynchon, Borges, Calvino, Tarkovsky, García Márquez, Kafkaesque, etc.)
- 6 output forms (prose / short-story / poem / myth / incantation / vignette)
- Anchor archetype, motif source + density, language (EN / FR / ES), tense, POV
- Reads the same Δ-graph params you set in Explorer — motifs match what you see.

## Archetype Builder

A tab that talks to the Egregore companion service to compose new symbol sets from PDFs, images, or raw text.

## Programmatic API

The engine works standalone, without the web stack:

```python
from delyrism import SymbolSpace, TextEmbedder, context_delta_graph

embedder = TextEmbedder(backend="qwen3")
symbols = {"Fire": ["passion", "energy"], "Water": ["calm", "flow"]}
space = SymbolSpace(symbols, embedder, descriptor_threshold=0.2)

proposals = space.propose(sentence="intense transformation", topk=3, tau=0.3)
G = context_delta_graph(space, sentence="healing journey", top_abs_edges=20)
```

Full reference: [`FUNCTIONS_README.md`](FUNCTIONS_README.md).

## Symbol presets

Ten archetypal systems ship under [`delyrism/structures/`](delyrism/structures/) — Elements, Chakras, Jungian, Lakota, Mayan, Chinese Zodiac, Planets, Musical Modes, Sacred Architecture, Seasons of Life. Each is a `{symbol: [descriptors…]}` JSON, freely editable.

## Documentation

- [`web/README.md`](web/README.md) — frontend + backend architecture, dev setup
- [`RAILWAY_SETUP.md`](RAILWAY_SETUP.md) — single-service Docker deployment
- [`FUNCTIONS_README.md`](FUNCTIONS_README.md) — engine API reference
- [`delyrism/legacy/app.py`](delyrism/legacy/app.py) — original Streamlit prototype, kept for reference

Every sidebar control has a `?` tooltip — the parameter reference lives in the UI, not in this README.

## Use cases

- **Mythopoetic writing** — generate story seeds from Δ-graph motifs in 15 literary registers
- **Cultural / semiotic studies** — compare how cultural contexts shift symbolic meaning; build archetypes from historical or image corpora
- **Psychological exploration** — map Jungian-style archetypes; measure ambiguity and overlap; trace themes across audio transcripts
- **Music & audio analysis** — CLAP joint embeddings trace sound → archetype associations

## License

MIT — see [`LICENSE`](LICENSE).

## Citation

```
@software{delyrism2026,
  title  = {Delyrism: Context-Aware Symbolic Archetype Explorer},
  author = {Bellemare, Antoine and contributors},
  year   = {2026},
  url    = {https://github.com/KairosHive/delyrism}
}
```

Built on Transformers, NetworkX, UMAP, CLAP, Plotly, Next.js. Inspired by archetypal psychology (Jung), distributional semantics, graph-based knowledge representation, and mythopoetic traditions.
