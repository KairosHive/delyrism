# Delyrism live API (package `delyrism`, v0.7)

How a host application — the do-you-dream installation, or anything else —
runs live delyrism as a component: install the package, start the server with
a profile, build a space in one call, then drive one WebSocket per screen.

## Install

```bash
# torch first (unpinned in the package): pick the build you want, e.g. CUDA 12.8
uv pip install torch --index-url https://download.pytorch.org/whl/cu128
# from a checkout (editable) …
uv pip install -e "../delyrism[server,live,vectorize]"
# … or from git
uv pip install "delyrism[server,live,vectorize] @ git+https://github.com/KairosHive/delyrism@realtime"
```

| extra | brings | for |
|---|---|---|
| (core) | numpy, scipy, scikit-learn, networkx, matplotlib, pandas, torch>=2.6, transformers>=5.22, **sentence-transformers>=5**, … | the engine |
| `server` | fastapi, uvicorn, pydantic, umap-learn | `python -m delyrism.server` |
| `live` | opencv-contrib-python, mediapipe | pose drawing, movement description |
| `vectorize` | opencv-contrib-python | `symbol_vectorizer` |
| `topology` | ripser | Topology tab |
| `media` | librosa, open-clip, timm, torchvision, sentence-transformers>=6.2, … | audio / image / video embedders (CLAP, CLIP, PE-AV, EmbeddingGemma 2) |
| `all` | everything | |

The package ships `delyrism/structures/*.json` (Lakota and the other symbol
sets) as package data:
`json.loads(importlib.resources.files("delyrism").joinpath("structures/lakota.json").read_text())`.

The default sentence-transformer model is
`sentence-transformers/all-mpnet-base-v2` (768-d), downloaded from Hugging Face
on first use (pre-fetch with
`python -c "from sentence_transformers import SentenceTransformer as S; S('sentence-transformers/all-mpnet-base-v2')"`).

## Run

```bash
python -m delyrism.server --host 127.0.0.1 --port 8000 --profile live_profile.json --warmup
#   --osc-host 127.0.0.1 --osc-port 7000 --osc-hz 60   (override the profile)
#   --tts-url http://127.0.0.1:8010                     (the TTS service; default)
#   --static DIR                                        (serve a frontend at /)
#   --max-embedders 1                                   (keep one model loaded: see Embedders)
delyrism-server …                                       # same, as a console script
```

Env equivalents: `DELYRISM_HOST`, `DELYRISM_PORT`, `DELYRISM_PROFILE`, plus
`DELYRISM_OSC_HOST|PORT|HZ`, `DELYRISM_TTS_URL`, `DELYRISM_SPEAK_DELAY`,
`DELYRISM_SPEAK_GAP` (read by the OSC and speech modules at import).
In-process: `from delyrism.server import create_app` →
`create_app(profile=…, static_dir=…, warm=…)` returns a FastAPI app.

**Health.** `GET /healthz` → `{"ok": true, "ready": bool, "warming": bool, "error": str|null, "uptime": s}`.
`ok` = the process answers; `ready` = the `--warmup` load of the profile's
embedder is done (always true without `--warmup`). Warm-up of
all-mpnet-base-v2 on this machine: ~40 s cold, then a Lakota space builds in
0.2 s.

## Profile

The server-wide defaults (`delyrism/server/profile.py`). Set at start
(`--profile`), or at run time with `POST /live/profile` (full profile) or as
the `profile` field of `/live/setup` (partial, merged). Every live socket
opened afterwards starts from it; a socket can still override per session
with the usual messages.

```json
{
  "embedder": {"backend": "sentence-transformer", "model": null},
  "descriptor_threshold": 0.2,
  "palette": "AuroraPop",
  "global_mode": "descriptor",          // "none" = raw geometry
  "global_pcs": 1,                       // the "PCA = 1": top principal direction removed
  "graph": {"top_abs_edges": 30},        // overrides of the live graph params (see below)
  "shape": {"focus": 2.0},               // layout: focus, contraction, damping, stiffness, repulsion, unlinked,
                                         // query_fade_tau (s, default 1.5: see Query presence)
  "pose": {"tracker": "body"},           // body | hands | face | both; trace, fit, joints, smooth
  "audio_smoothing": null, "rel_smoothing": null,
  "motion": {"auto": true, "refresh": 3.0, "window": 3.0, "level": "full", "contrast": 0.7, "relative": 1.0},
  "transcript": {"tau": 1.5, "hz": 20},  // latent smoothing of spoken-text queries (below)
  "lakota": false,                       // Lakota background matching (/delyrism/lakota* OSC, sim "lakota"); off by default
  "osc": {"enabled": true, "host": "127.0.0.1", "port": 7000, "hz": 60},
  "speech": {"enabled": true, "url": "http://127.0.0.1:8010", "media": true, "media_kinds": ["motion"],
             "delay": 1.0, "min_gap": 4.0, "mood": true}
}
```

Live graph params (`graph`) and their defaults: `strategy "gate"`, `beta 1.2`,
`gate "relu"`, `tau 0.3`, `within_symbol_softmax true`, `gamma 0.5`,
`pool_type "avg"`, `pool_w 0.7`, `membership_alpha 0`, `top_abs_edges 30`,
`min_abs_delta 0.01`, `within_symbol false`, `connected_only true`,
`sign_filter "up"`, `sym_filter null`.

Speech speaks the most activated word after a typed or spoken query settles
(`delay` s after the last change), and for the kinds listed in `media_kinds`
("pose", "motion", "speech") when the top node changes instead (never cutting
the previous word off: at least `min_gap` s apart). Needs the
TTS service at `url`; without it nothing is spoken and nothing fails.

## HTTP

| method | path | body → result |
|---|---|---|
| POST | `/live/setup` | `{"symbols": {NAME: [descriptor, …]}, "profile": {…partial…}?}` → `{space_id, symbols, descriptors, owners: {descriptor: symbol}, embedding_dim, color_map: {symbol: "#hex"}, global_mode: {mode, pcs}, modalities: ["text", …], profile}`. **Idempotent**: same symbols + profile → same `space_id` (cache hit, ~ms). Builds with the profile's embedder and applies its global mode. 400 if no symbol has descriptors, 502 if the embedder fails. |
| GET | `/live/profile` | → the current profile |
| POST | `/live/profile` | full profile → the applied profile (422 if invalid) |
| GET | `/spaces/{space_id}` | → 200 `{space_id, symbols, descriptors, owners, embedding_dim, color_map, modalities}` · 404 unknown (e.g. after a server restart: call `/live/setup` again) |
| GET | `/spaces/{space_id}/map` | → `{descriptors, owners, symbols, color_map, n, dim, D, xy, layout}`: every descriptor's vector (`D`: base64 float32 LE, row-major n × dim) and its place on the space's 2D map (`xy`; `layout` "umap", or "pca" when UMAP is unavailable). What a session report needs (`delyrism.session.analyse`) |
| GET | `/spaces/presets/{name}` | → `{name, symbols}` from the shipped structures (`lakota`, `elements`, …) |
| GET | `/realtime/osc` · POST | OSC status / `{enabled, host, port, hz, scale}`. The status has `owner: {space_id, symbols, context, layout_source, …}` for the socket TouchDesigner currently shows (null when none); TD also gets `/delyrism/space s space_id` with the labels (on owner change and about 1/s). |
| GET | `/realtime/speech` · POST | speech status / settings (as in the profile) |

Spaces live in memory: a server restart forgets them (the socket then closes
with code **4404** — re-run `/live/setup` and reconnect).

## WebSocket `/realtime/graph/{space_id}`

One socket per interactive screen. The socket that most recently sent an
input owns the OSC output (TouchDesigner follows it).

**On open** the server sends
`{"type": "ready", "params": {…graph params…}, "shape": {…}, "pose_opts": {…}, "symbols": [...], "motion": {…profile.motion…}, "pose_capable": bool, "modalities": ["text", …], …}`
or `{"type": "error", "detail": "unknown space_id …"}` and closes with 4404.

### Client → server (JSON text frames)

| message | effect |
|---|---|
| `{"type": "context", "text": "…"}` | the dream: rebuilds the Δ-graph (send on every settled edit, ~200 ms debounce is plenty) |
| `{"type": "query", "text": "…"}` | text query → relevance; speech after `delay` |
| `{"type": "transcript", "text": "…", "tau"?: 1.5, "hz"?: 20, "source"?: "typed"}` | spoken text, e.g. the live transcription of the last ~30 s, re-sent as it grows. **Latent smoothing**: each text sets a target embedding and the query vector glides toward it (normalised exponential approach, time constant `tau` s), re-scored `hz` times a second, so relevance moves continuously instead of jumping at each update. `relevance.kind == "speech"`, `relevance.query` = the text. Empty text holds the current query. `tau`/`hz` override the profile for this socket |
| `{"type": "query_clear"}` | no query (relevance off) |
| `{"type": "sim", "hz": 30}` | subscribe to the layout frames below (0 = off) |
| `{"type": "snapshot", "id"?: any, "format"?: "list" \| "b64", "vectors"?: true, "context_rel"?: false}` | one recording sample, answered at once with a `snapshot` message (below). `vectors: false` leaves the vectors out (a lighter sample); `context_rel: true` adds `context_rel_all`. Read-only: does not claim the OSC output |
| `{"type": "pose_params", "tracker": "body", …}` | pose options (defaults from the profile) |
| `{"type": "motion_tick", "window": 3, "contrast": 0.7, "relative": 1}` | describe the last `window` s now (not needed when `profile.motion.auto`) |
| `{"type": "motion_reset"}` | forget the session's movement range (new dancer) |
| `{"type": "pose_reset"}` | clear the trajectory |
| `{"type": "params", "params": {…}}` · `{"type": "shape", "shape": {…}}` | per-socket overrides (graph params include `habituation`, `habituation_tau`: see binary frames) |

### Client → server (binary frames)

Media queries (the embedder must handle the modality: `modalities` in the
`/live/setup`, `/spaces/{id}` and `ready` responses):

```
0x01 audio   uint32 LE sample rate + int16 LE PCM mono   (any rate; ≥ 1000 samples)
0x02 image   one JPEG
0x03 video   uint8 n + n × (uint32 LE length + JPEG)      a window of frames, oldest first
0x04 av      uint32 LE audio length + audio payload + frames payload   (PE-AV only)
```

A media embedding that is still running when a `query`, `query_clear` or
`transcript` arrives is dropped when it finishes (it does not bring the old
query back).  Each becomes the query (`relevance.kind` = `audio` | `image` | `video` | `av`),
blended with the previous media vector by `audio_smoothing`.

**Habituation** (graph params, so `profile.graph` or `{"type": "params"}`;
off by default): a camera that always sees the same room keeps lighting the
same few descriptors, because every frame shares the room (and the modality's
own offset); typed and spoken queries drift toward the same few words too.
Each socket keeps a running mean `mu` of the raw query vectors of each kind
and scores

```
b      = beta_kind · min(1, seconds since mu started / habituation_tau)    (warm-up: first frames are not cancelled)
scored = unit(v − b · mu)                                                  (then the usual audio_smoothing EMA)
mu    ← mu + (1 − exp(−dt / habituation_tau)) · (v − mu)                   (after scoring; dt = wall-clock s since the last vector of that kind)
```

| param | default | |
|---|---|---|
| `habituation` | `0` | a map `{"text", "speech", "motion", "image", "video", "audio", "av", "pose": beta}` (missing kinds 0), or a number = beta for image and video only. Clipped to 0…1 |
| `habituation_tau` | `60` | time constant of the running mean, s |

Which mean a query uses:

| key | vectors |
|---|---|
| `text` | each `query` message, and each `transcript` sent with `"source": "typed"` |
| `speech` | each other `transcript` target |
| `motion` | each movement description (after its `contrast` baseline) |
| `image` · `video` · `audio` · `av` · `pose` | each media / pose vector |

A transcript's target is habituated first and the glide then moves toward the
habituated target; `query_kind` stays `"speech"` for every transcript.  The
mean is tracked for every key even while its beta is 0 (so switching it on
starts warm), updates only when a new vector of that key arrives (dt =
wall-clock time since the previous one, so after a long pause the mean is
close to the previous query), lives as long as the socket, and survives
`query_clear` and mode switches.  Each `query` updates the `text` mean, so a
client that sends a query on every keystroke habituates to its partial words:
send on settled edits.  Changing these params applies from the next vector; it
does not rebuild the graph.  At beta 1 a perfectly static input leaves only
noise once the mean has caught up; a vector that cancels completely falls back
to the raw one.


Landmarks from the browser tracker (`web/frontend/lib/poseTracker.ts`
`packLandmarks`, run in `trackWorker.ts`), one message per tracked frame:

```
byte 0      6                      (kind: landmarks)
byte 1      uint8  tracker id      0 body · 1 hands · 2 face · 3 both
bytes 2-4   pad
bytes 5-8   float32 LE aspect      video width / height
bytes 9-16  float64 LE t           seconds, any monotonic clock of the client
bytes 17-20 uint32 LE rows         0 when nothing was found
then        rows × 7 float32 LE    x y z visibility wx wy wz (MediaPipe normalised + world), NaN = missing
            body 33 rows · hands 42 · face 478 · both 75
```

**Movement mode** = stream these frames; with `profile.motion.auto` the server
describes the movement every `refresh` s while frames arrive (stop sending
them to leave movement mode, then send a `query` or `query_clear`). Each
description arrives as a `motion` message and becomes the query
(`relevance.kind == "motion"`, `relevance.query` = the words).

### Server → client

| message | when | fields |
|---|---|---|
| `graph` | after each context change | `nodes: [{id, symbol, color, pull, …}]` (descriptors), `edges: [{source, target, delta, …}]` |
| `relevance` | after each query / graph change, and `hz`/s while a spoken query glides | `relevance: {descriptor: 0…1}`, `active`, `query` (the text, the transcript or the movement words), `kind` ("text" · "speech" · "motion" · …) |
| `motion` | each movement description | `text` (the words, "" when nothing moved or nobody is in frame), `level`, the qualities |
| `pose` | per tracked batch | `jpeg` (base64 drawing of the dancer with trails), `present` |
| `spoken` | a word was spoken | `word`, `duration`, `mood`, `mood_probs` |
| `sim` | at the subscribed rate while anything moves, 4 Hz once settled | see below |
| `snapshot` | in answer to `snapshot` | see below |
| `error` | | `detail` |

### Query presence

The server's layout does not switch a query on and off; it fades it.  A
presence `p` (0…1) eases toward 1 while the socket has a query and toward 0
after `query_clear`, with time constant `shape.query_fade_tau` (s, default 1.5;
0 = switch at once; per socket with `{"type": "shape"}` or from
`profile.shape`).  Everything the query drives is blended by `p`:

- node `rel` and edge `overlap` (sim frames, OSC) are scaled by `p`;
- spring rest lengths / stiffness and the radial pull blend from the no-query
  layout (p = 0) to the query layout (p = 1);
- the Lakota background weights blend from node size to relevance.

After `query_clear` the last relevance stays and fades out; once `p < 0.01` it
is dropped and `query_active` turns false (~7 s at tau 1.5).  `query_active`
(sim frames, OSC `/frame`, snapshot) = `p ≥ 0.01`; `query_presence` (sim
frames, snapshot) and OSC `/delyrism/query/presence f` carry `p` itself, to
blend by instead of switching on `query_active`.

Switching kinds keeps the latent smoothing too: a media vector blends
(`audio_smoothing`) from the current query whatever its kind, and a transcript
glides from it; after `query_clear`, a new query blends / glides from the
cleared one while it is still fading out, else starts at its own vector and
fades in by `p`.  Only PE-AV queries of different heads (audio / video / av)
start fresh, since their vectors live in different spaces.

### `snapshot` — a recording sample

The layout and the vectors it was built and scored from, all from the same
instant (the vectors are those taken when the layout last changed, not newer
ones still being scored):

```jsonc
{"type": "snapshot", "id": <echo>, "t": 1762540000.123,      // server epoch seconds
 "graph_seq": 4, "space_id": "…", "symbols": ["Fire", …],
 "context": "a storm at sea",           // the context the shown graph was built from
 "context_vec": [...] | null,           // its embedding (the Δ-graph's sentence vector), null without a context
 "context_pending": "…" | null,         // a newer context whose graph is still being built
 "query_active": true,
 "query_presence": 0.83,                // the layout's query presence (see Query presence)
 "query_kind": "text" | "speech" | "audio" | "image" | "video" | "av" | "motion" | "pose" | null,
 "query_text": "lightning" | null,      // the text / transcript / movement words; "(camera)" etc. for media
 "query_vec": [...] | null,             // the vector the relevance was scored with (after the media EMA / speech glide)
 "query_target": [...] | null,          // the latest raw embedding: = query_vec for text, the transcript's target
                                        // while the glide moves toward it, the newest media vector before habituation and the EMA
 "rel_all": [...] | null,               // the scored query's cosine with every descriptor of the space (space.descriptors
                                        // order, the shifted matrix the graph is scored in): a session recording's row (below)
 "context_rel_all": [...] | null,       // only with "context_rel": true - the context's cosine with every descriptor (raw D)
 "habituation": {"kind": "image", "beta_eff": 0.5} | null,   // the mean key and warm-up-scaled beta of the latest query vector
 "nodes": {…the sim frame's node columns…, "relevance": [...]},   // relevance = the scored 0…1 activation;
                                        // rel = the same after the layout's contrast curve; alive=false = fading out
 "edges": {…the sim frame's edge columns…},                       // delta = signed Δ cosine (3 decimals)
 "embedder": {"backend": "embeddinggemma2", "model": "google/embeddinggemma-2", "dim": 768},
 "vec_format": "list"}
```

Vectors are unit-length float lists rounded to 6 decimals, or with
`"format": "b64"` base64 of little-endian float32 (~40% smaller: about 15 KB
instead of 26 KB for an 11-node graph with 768-d vectors).

### Session reports (`delyrism.session`)

`delyrism/session.py` turns a recording of a live run into the panels of a
session report, without the server: `analyse(raw, D, XY, colors)` with
`raw = {descriptors, owners, symbols, t, kind, query, rel, events, duration,
name}` (`rel`: one row per moment with the query's cosine over every
descriptor, `null` without a query - a snapshot's `rel_all`; events of type
`context` (with `rel`: `context_rel_all`), `query`, `heard`, `spoken`,
`motion`, `graph` (with `edges: [[u, v, delta], …]`)), `D` and `XY` from
`/spaces/{id}/map`.  The result holds `summary`, `layout`, `trajectory`
(attention path, dwells, context path), `territory` (time in the top 5 per
descriptor, reach, never reached, symbol shares and their stream per minute),
`topology` (groups and loops per sliding window, their changes, Δ-graph edges
born and dying), `phases` (phases with their words and symbol, recurrence
matrix, returns), `words`, `archetypes` and `fingerprint`.  Loops need the
`topology` extra (ripser); without it they count 0.

### `sim` frames — the TouchDesigner view

The server's own layout (the one streamed to TouchDesigner over OSC), as
columnar JSON; ~2.6 KB for a 16-node graph, ~26/s at `hz: 30`.

```json
{"type": "sim", "seq": 412, "t": 13.71, "graph_seq": 3, "query_active": true, "query_presence": 1.0, "awake": true,
 "symbols": ["BAG", "CLOUDS", "DRAGONFLY", …],          // the space's symbol order (TD: Pal0…Pal9 by this index)
 "nodes": {"id": ["clouds", …],                          // descriptor label (nodes are descriptors)
           "uid": [17, …],                               // stable per descriptor, survives graph changes
           "sym": [1, …],                                // index into symbols = the owner symbol
           "x": [-0.192, …], "y": [0.537, …],            // output units ≈ −1…1, +y up, origin centre
           "rel": [0.0, …],                              // query relevance 0…1, × query_presence
           "size": [0.4, …],                             // Σ|Δ| of the node's edges, 0…1
           "fade": [1.0, …], "alive": [true, …],         // appear / disappear envelope; false = fading out
           "color": ["#ffd166", …]},                     // the space palette's colour (TD may use its own)
 "edges": {"src": [0, …], "dst": [3, …],                 // row indices into nodes
           "weight": [0.8, …], "delta": [0.12, …], "sign": [1, …],   // |Δ| 0…1, signed Δ cosine, +1/−1
           "overlap": [0.3, …], "fade": [1.0, …], "alive": [true, …]},
 "lakota": {"names": ["BAG", …], "activation": [0.1, …]}}   // the Lakota background distribution; null unless profile.lakota
```

Row order is stable between frames; it changes only when a faded row is
dropped (then `id`/`uid` tell you which is which).

## Embedders

`profile.embedder` = `{"backend": ID, "model": null}`; `GET /backends` lists the
ids with their modalities.  Every space reports `modalities` (`["text", …]`).

| backend | modalities | notes |
|---|---|---|
| `sentence-transformer` | text | default, all-mpnet-base-v2 (768-d) |
| `embeddinggemma2` | text, audio, image, video | `google/embeddinggemma-2` (740M, 768-d, bf16, ~1.5 GB). Descriptors and text queries both use the model's `SentenceSimilarity` prompt (`"task: sentence similarity \| query: …"`); set `default_instruction` to another prompt name of the model (`Document`, `Clustering`, …) to change it. Media take no prompt. Audio is resampled to mono 16 kHz. Needs the `media` extra (torchvision, sentence-transformers>=6.2). |
| `clap` · `clip` · `peav` | text + audio · image · audio, video, av | see `GET /backends` |
| `qwen3` · `qwen2` · `cloudflare-*` | text | |

**Memory.** Embedders stay loaded once built (one per distinct config).
`--max-embedders N` (env `DELYRISM_MAX_EMBEDDERS`, default 0 = no limit) caps
them: building another evicts the least recently used embedder together with
every space built on it (`GET /spaces/{id}` → 404 afterwards; call
`/live/setup` again to rebuild).  A socket still open on an evicted space keeps
working, and the model is freed (and its CUDA memory returned) when the last
such socket closes.  To switch backends without ever holding two models, close
the graph sockets of the old space first, then call `/live/setup` with the new
embedder.  `GET /spaces/cache/stats` → `{embedders, max_embedders, spaces, results}`.

## symbol_vectorizer

```python
from symbol_vectorizer import vectorize_array, vectorize_image, build_space, Params

p = Params(); p.segment.work_size = 600            # the working resolution (default in Params)
r = vectorize_array(rgb_uint8, p, name="feather")  # in memory, nothing written
r["svg"]        # str  filled outlines in the drawing's colours
r["lines_svg"]  # str  centre-line strokes (+ fills)
r["png"]        # bytes  the cropped drawing on a transparent background
r["record"]     # dict  what <name>.json holds: strokes, fills, palette, stats (frame: ink box, [-1, 1], y up)
r["preview"]    # RGB uint8 (360 × 1088) three-panel check image
# ValueError("… no ink found …") on an empty page

vectorize_image("photo.jpg", "out_dir", p, name="feather")   # writes out_dir/feather/{feather.svg,_lines.svg,.png,.json}
```

Speed here: ~0.9 s at work_size 600, ~1.7 s at 900 (CPU), so a live preview
runs at about 1 Hz at 600 px.
