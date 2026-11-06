# ImageBind IMU audit for delyrism (Live tab query source)

Date: 2026-11-06. Hardware: RTX 4090 Laptop 16 GB, driver 580. Everything below was actually run;
logs are in `imu_probe.log`, `imu_probe2.log`, `imu_probe_results.json` next to this file.

## 1. Install (what succeeded)

```sh
mkdir -p /home/philipp/projects/kairos/imagebind-audit/.checkpoints
cd /home/philipp/projects/kairos/imagebind-audit
git clone --depth 1 https://github.com/facebookresearch/ImageBind.git ImageBind
curl -L -C - -o .checkpoints/imagebind_huge.pth https://dl.fbaipublicfiles.com/imagebind/imagebind_huge.pth   # 4.58 GB, ~90 s at 50 MB/s
/home/philipp/.local/bin/uv venv --python /usr/bin/python3 .venv                       # Python 3.12.3
/home/philipp/.local/bin/uv pip install --python .venv/bin/python --index-url https://pypi.org/simple \
    torch==2.15.0 torchvision==0.29.2 torchaudio                                        # -> torch 2.15.0+cu132, torchaudio 2.11.1, CUDA works
/home/philipp/.local/bin/uv pip install --python .venv/bin/python --index-url https://pypi.org/simple \
    "pytorchvideo @ git+https://github.com/facebookresearch/pytorchvideo.git@6cdc929315aab1b5674b6dcf73b16ec99147735f" \
    timm ftfy regex einops iopath "numpy>=1.19" types-regex                             # timm 1.0.31, pytorchvideo 0.1.5
/home/philipp/.local/bin/uv pip install --python .venv/bin/python --no-deps -e ./ImageBind
```

Patch needed (only one): `imagebind/data.py` imports `pkg_resources`, which no longer exists in a
setuptools>=81 / py3.12 venv. Replaced the BPE-path lookup with `os.path`:

```diff
-import pkg_resources
+import os
 ...
 def return_bpe_path():
-    return pkg_resources.resource_filename("imagebind", "bpe/bpe_simple_vocab_16e6.txt.gz")
+    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "bpe", "bpe_simple_vocab_16e6.txt.gz")
```

No pytorchvideo / torchaudio breakage was hit on the text + IMU path (the pytorchvideo and torchaudio
imports are only exercised by the video/audio loaders, which we don't use). `imagebind_huge(pretrained=True)`
loads the checkpoint with `torch.load(..., weights_only=True)` from the *relative* path
`.checkpoints/imagebind_huge.pth`, so scripts must run from the audit dir (or the backend must pass its own
path by constructing the model and calling `load_state_dict` itself).

Licence: checkpoint and code are CC-BY-NC-SA 4.0 (setup.py classifier) — non-commercial.

## 2. IMU input convention (from the source and the paper)

Source (`imagebind/models/imagebind_model.py`, `multimodal_preprocessors.py::IMUPreprocessor`):

- Input tensor `(B, 6, 2000)` float. `IMUPreprocessor(img_size=[6, 2000], kernel_size=8)`: the signal is
  `unfold`-ed along time with kernel 8 / stride 8 → 250 patches of 6×8 = 48 values, each projected by
  `nn.Linear(48, 512, bias=False)` + LayerNorm, then a CLS token + learnable positional embedding
  (251 positions), a 6-block / 8-head / 512-d transformer (18.9 M params), head = LayerNorm → CLS →
  `Linear(512, 1024)`, post-processor = L2 `Normalize` then a *fixed* logit scale of 5.0
  (`LearnableLogitScaling(logit_scale_init=5.0, learnable=False)`). So the raw model output for IMU has
  norm 5; text output has norm = learned scale. Re-normalise before cosine (the probe does).
- **No normalisation of the raw samples anywhere in the code** (no mean/std, no clipping, no resampling).
  The linear patch stem sees the raw units.
- The positional embedding is tied to exactly 250 patches, so the input must be exactly 2000 samples
  (shorter/longer → shape error). Pad or tile to 2000.
- **Channel order, units and sampling rate are not stated anywhere in the repo** (grep for imu in
  README, model_card, data.py: only "IMU using the Ego4D dataset"). There is no IMU loader in `data.py`.

Paper (Girdhar et al., CVPR 2023, arXiv 2305.05665; read via ar5iv HTML and pdftotext of the PDF):

- §3.2: "We extract the IMU signal consisting of accelerometer and gyroscope measurements across the X, Y,
  and Z axes. We use 5 second clips resulting in 2K time step IMU readings which are projected using a
  1D convolution with a kernel size of 8."  App. B.1: "we sample fixed length clips of 5 seconds ...
  For each clip, we get a 6×2000 dimensional input". App. A.1: "For IMU we use a 6 × T tensor".
- **Sampling rate: not stated.** 2000 samples / 5 s implies **400 Hz**, not 200 Hz. The Ego4D IMU docs
  (https://ego4d-data.org/docs/data/imu/) also do not state a rate, but their sample rows have
  `canonical_timestamp_ms` ≈ 0, 4.975, 9.950 → ~200 Hz. So either the ImageBind loader resampled/concatenated,
  or the GoPro gyro stream is at 400 Hz and the accel was upsampled to it — **unverified**; both "200 Hz"
  and "400 Hz" conventions were tested empirically below.
- **Channel order: not stated** in the paper. The Ego4D CSV column order is
  `gyro_x, gyro_y, gyro_z, accl_x, accl_y, accl_z` (gyro first!), whereas the paper lists
  "accelerometer and gyroscope". Which order the training loader used is **unverified**; both were tested.
- **Units: not stated** in either source (Ego4D docs say only "No calibration of the IMU took place").
  GoPro GPMF streams are nominally m/s² (ACCL) and rad/s (GYRO); gravity is included in ACCL.
  Both m/s² vs g and rad/s vs deg/s were tested.
- Training: (video, IMU) pairs only from Ego4D (510 k train clips, head-mounted GoPro), temperature 0.2,
  stochastic depth 0.7, 8 epochs. Text↔IMU alignment is *emergent* (never trained on text-IMU pairs).
  Reported zero-shot Ego4D scenario classification (108 classes): 25.0 % top-1 (random 0.9 %).

Summary: the only hard facts are shape `(B, 6, 2000)`, acc+gyro × xyz, 5 s clips, no normalisation.
Rate (200 vs 400 Hz), channel order and units are unverified.

## 3. Probe results (`imu_probe.py`, `imu_probe2.py`)

Model: imagebind_huge, fp32, CUDA.

| metric | value |
|---|---|
| model load (disk → cuda, incl. 4.6 GB state dict) | 14.2 s |
| GPU allocated after load / peak | 4651 MB / 4835 MB (whole model, all 6 towers) |
| trunk params | vision 629.7 M, text 302.3 M, audio 85.1 M, imu 18.9 M |
| IMU embed latency, batch 1, after warm-up | median 4.53 ms, p90 4.81 ms |
| IMU embed latency, batch 10 | 22.6 ms |
| text embed, 50 strings (one batch) | 482 ms (~10 ms/string) |

Synthetic clips (channels ax, ay, az, gx, gy, gz; gravity on +z): still (g + noise), walking (2 Hz, ±2.5 m/s²,
gyro wobble), running (3 Hz, ±8 m/s²), shaking (8 Hz, ±15 m/s², 4 rad/s), rotation (slow gyro sweep,
gravity vector rotates accordingly).

### 3a. Text × IMU cosine, the "paper-literal" convention (accel-first, m/s², rad/s, 2000 samples @ 200 Hz)

```
                       still   walking   running   shaking  rotation
a person walking      -0.010    -0.000     0.053     0.114    -0.048
running               -0.019    -0.010     0.014     0.012     0.023
shaking a phone       -0.048    -0.018    -0.017     0.060     0.049
sitting still          0.045     0.046     0.062     0.000     0.034
riding a bike         -0.031    -0.015     0.038     0.080    -0.005
climbing stairs       -0.042    -0.024    -0.023     0.141    -0.040
best per clip: still→sitting still, walking→sitting still, running→sitting still, shaking→climbing stairs, rotation→shaking a phone
imu×imu: still·walking 0.97, still·running 0.83, shaking vs others -0.07..-0.39
```

At 400 Hz the picture is the same (same argmaxes). Text-IMU cosines are tiny (|cos| ≤ 0.14) and
walking/running collapse onto "still" in embedding space (cos 0.97).

### 3b. Convention sweep (argmax prompt per clip; correct hits in bold)

| fs | accel | gyro | order | still | walking | running | shaking | rotation | imu off-diag mean |
|---|---|---|---|---|---|---|---|---|---|
| 200 | m/s² | rad/s | accel-first | **sitting still** | sitting still | sitting still | climbing stairs | shaking a phone | 0.30 |
| 400 | m/s² | rad/s | accel-first | **sitting still** | sitting still | sitting still | climbing stairs | shaking a phone | 0.28 |
| 200 | g | rad/s | accel-first | **sitting still** | **a person walking** | a person walking | a person walking | riding a bike | 0.45 |
| 400 | g | rad/s | accel-first | **sitting still** | shaking a phone | a person walking | a person walking | riding a bike | 0.29 |
| 200 | m/s² | deg/s | accel-first | a person walking | **a person walking** | a person walking | a person walking | riding a bike | 0.44 |
| 200 | m/s² | rad/s | gyro-first | riding a bike | riding a bike | riding a bike | a person walking | riding a bike | 0.64 |
| 400 | m/s² | rad/s | gyro-first | riding a bike | riding a bike | a person walking | a person walking | riding a bike | 0.78 |
| 200 | g | rad/s | gyro-first | riding a bike | riding a bike | riding a bike | climbing stairs | riding a bike | 0.59 |
| 200 | g | deg/s | gyro-first | a person walking | riding a bike | riding a bike | riding a bike | riding a bike | 0.24 |

Best hit rate is 2/5 (accel-first, g, rad/s @ 200 Hz), max cosine 0.23. Gyro-first makes all clips
near-identical (still·walking 0.995, still·rotation 0.94) and everything maps to "riding a bike" — this
looks like the *wrong* order, which weakly supports accel-first for the trained model. The all-zero clip
embeds to "riding a bike" (cos 0.126), i.e. "riding a bike" is the model's prior for low-information input.

### 3c. Ego4D-style prompts (`#C C walks on the road`, "a video of a person running", ...), top-3

accel-first, g units: still → "a video of someone standing still" 0.087; walking → "#C C sits on a chair"
0.191; running → "#C C walks on the road" 0.167 / "a video of a person walking" 0.166; shaking → "#C C sits on
a chair" 0.137; rotation → "driving a car" 0.213 / "#C C rides a bicycle" 0.194.
With m/s² units: still/walking/running all → sitting prompts; shaking → "playing basketball" 0.144;
rotation → "driving a car" 0.074. Prompt rankings for still vs walking are 0.98-correlated in m/s² (not
distinguishable), 0.68 in g.

Conclusion on alignment: the model clearly *responds* to motion (shaking and rotation produce distinct
rankings, and "rotation" → driving/bike/moving-car is plausibly right: a head-mounted camera in a vehicle
experiences slow gyro sweeps), but with synthetic phone-like signals the text argmax is right at most 2/5
times and the margins are 0.02–0.1 cosine. Two caveats make this a lower bound: (a) my synthetic clips
are hand-held-phone-like, while the encoder was trained on head-mounted GoPro IMU (walking = vertical
head bob + yaw, not pocket swing); (b) the convention is unverified, and the wrong one costs a lot (see
gyro-first rows). The paper's own number — 25 % top-1 over 108 Ego4D scenarios — says that even on real
in-distribution data the alignment is coarse.

### 3d. Windows shorter than 2000 samples

cosine of the embedding of a truncated clip (zero-padded to 2000, or tiled/repeated to 2000) vs the full-clip
embedding; and whether the best text prompt is preserved:

| rate | window | zero-pad cos→full (still / walk / run / shake / rot) | tiled cos→full | best-prompt preserved pad / tiled |
|---|---|---|---|---|
| 200 Hz | 1 s (200) | 0.71 / 0.57 / -0.03 / 0.14 / 0.07 | 1.00 / 0.995 / 0.98 / 0.997 / 0.54 | 1/5 / 4/5 |
| 200 Hz | 2 s (400) | 0.76 / 0.68 / 0.24 / 0.24 / 0.23 | 1.00 / 0.98 / 0.99 / 0.999 / 0.35 | 0/5 / 4/5 |
| 200 Hz | 5 s (1000) | 0.82 / 0.81 / 0.73 / 0.75 / -0.01 | 1.00 / 1.00 / 0.99 / 1.00 / 0.70 | 1/5 / 5/5 |
| 400 Hz | 1 s (400) | 0.76 / 0.68 / 0.39 / 0.27 / 0.09 | 1.00 / 0.98 / 0.97 / 0.998 / 0.22 | 1/5 / 4/5 |
| 400 Hz | 2 s (800) | 0.81 / 0.76 / 0.80 / 0.42 / 0.14 | 1.00 / 0.99 / 0.99 / 1.00 / 0.32 | 1/5 / 5/5 |

**Zero-padding is unusable** (the zero region is itself a strong "signal": a half-zero clip moves as far
as 0.7 cosine away, and even 5 s of 10 drops to 0.73–0.82). **Tiling the window to 2000 samples is nearly
lossless for periodic/stationary motion** (≥ 0.97 at 1 s for still/walk/run/shake) and only degrades for
non-stationary content (the slow 0.2 Hz rotation sweep, whose period is longer than the window). So
sliding windows of 1–2 s, tiled to 2000, are usable.

### 3e. 60 Hz capture (browser) upsampled

Linear interpolation of a 60 Hz capture to the model rate gives cos 0.996–1.000 to native 200 Hz sampling
(0.93–0.97 for walking/running at 400 Hz). Feeding 60 Hz samples *as if* they were model-rate (2000 samples
= 33 s) shifts walking/running by 0.08–0.11 and shaking by 0.37 — do not do that; resample.

## 4. Recommendation

**Is IMU→text alignment good enough for a live query?** Marginal. Honest reading of the data:

- The IMU tower is tiny and fast (4.5 ms/clip, 19 M params) and the whole model fits comfortably
  (4.8 GB, but note the full checkpoint carries the 630 M vision tower you don't need; it still loads in 14 s).
- Text ↔ IMU cosines live in ±0.2, argmax over six activity prompts is right ≤ 2/5 on synthetic
  clips, and still/walking are nearly indistinguishable under the literal (m/s²) convention. For
  delyrism this means: the IMU query vector will *move* with motion (shaking, rotating, stillness are
  clearly separated in the IMU embedding itself; rankings over descriptors do change), but the direction
  it moves in the shared space is only loosely semantic. Expect "motion energy modulates the overlap"
  rather than "walking lights up the walking archetype". The paper's 25 % top-1 on real Ego4D says the
  ceiling is coarse even in-distribution; phone IMU is out of distribution (head-mounted GoPro).
- Compared with the existing CLAP / CLIP / PE-AV sources, this is a weaker, noisier modality. It is worth
  adding only as an *expressive* control (the user shakes/turns/walks and the overlap breathes), not as a
  recogniser. If the goal is "the graph knows I'm walking", a 20-line hand-crafted classifier (RMS of
  accel band-pass 1–3 Hz, gyro RMS, step frequency) feeding a *text* query ("walking", "shaking") through
  the space's own embedder will be far more reliable and needs no 4.6 GB model.

**If you do add it:**

- Input convention: use channels **[accel x,y,z, gyro x,y,z]**, accel **including gravity**, gyro in
  **rad/s**. Unit for accel: expose both and default to **g** (÷9.81): it gave the most differentiated
  prompt rankings and the only correct walking hit; the literal m/s² convention collapses still/walk/run.
  Mark it unverified in the code. Do not use gyro-first.
- Rate: assume **400 Hz = 2000 samples / 5 s** (paper-literal); 200 Hz behaved essentially the same in
  all tests, so the choice is not critical. Resample from the browser's ~60 Hz by linear interpolation
  (lossless in the tests); never feed 60 Hz samples as raw model-rate samples.
- Window / update rate: keep a ring buffer of the last **2 s** of resampled samples (800 @ 400 Hz), tile it
  to 2000, embed every **250–500 ms**. At 4.5 ms per embed the GPU cost is nil; the EMA smoothing already
  in `realtime.py` (`audio_smoothing`) covers the rest. 1 s windows also work for periodic motion. Avoid
  zero-padding entirely.
- Browser: `DeviceMotionEvent.accelerationIncludingGravity` (m/s², device axes) + `rotationRate`
  (**deg/s** — convert to rad/s client- or server-side) at ~60 Hz (iOS needs `DeviceMotionEvent.requestPermission()`
  from a user gesture and HTTPS). Send raw 60 Hz samples with the rate; resample server-side (numpy
  `np.interp`), so the binary format mirrors the audio one.
- delyrism plug-in, following the PE-AV pattern (`delyrism/peav.py`, `web/backend/app/routes/realtime.py`):
  - new backend id `imagebind` in `TextEmbedder` (`delyrism/delyrism.py`) and `_ENGINE_BACKENDS`
    (`web/backend/app/engine_cache.py`); `embed_text` = ImageBind text tower (1024-d; 10 ms/string, so
    embedding a descriptor set is fine), `imu_capable = True`, `embed_imu(samples_6xN, rate)` → resample →
    tile/crop to 2000 → `model({ModalityType.IMU: x})` → L2-normalise (strip the ×5 logit scale).
    Build the model with `imagebind_model.imagebind_huge(pretrained=False)` + `load_state_dict(torch.load(path, weights_only=True))`
    with an explicit checkpoint path (the library hard-codes a cwd-relative `.checkpoints/`). The text
    tower is a 77-token CLIP-style tokenizer: descriptors longer than ~60 words get truncated.
    Memory: `del model.modality_trunks['vision']` (and audio/depth/thermal) after loading drops ~2.5 GB.
  - `realtime.py`: `_BIN_KINDS[5] = ("imu", _imu_vec, 4 + 4 * 6 * 30)` with payload
    `uint32 LE rate` + `float32 LE (6, N)` planar (same layout as the tensor); `_LABELS["imu"] = "(motion)"`;
    `_head_for_kind` unchanged (ImageBind has a single joint space, no heads).
  - Frontend: a `liveMotion.ts` beside `liveAudio.ts` / `liveCamera.ts` that subscribes to
    `devicemotion`, keeps a 2 s ring buffer, and posts a kind-5 frame every 250–500 ms.
  - Licence: ImageBind weights are CC-BY-NC-SA; fine for a research/art project, flag it in the README
    next to the other backends.

Everything that failed or is unverified: no failures in install/run beyond the `pkg_resources` patch.
Unverified: sampling rate (200 vs 400 Hz), channel order, units — none are documented by Meta or Ego4D;
the recommendation above is the empirically least-bad choice from synthetic data, not ground truth.
