"use client";
// Landmark tracking in the browser, on the GPU.
//
// MediaPipe's Python build on Windows has no GPU path, so the live pose and
// movement queries track here: the tasks-vision WASM runtime with the GPU
// delegate (WebGL) reads the <video> element directly, and only landmarks
// travel to the server (the frames come from a worker, see trackWorker.ts,
// or from the <video> element where that is unavailable).  Rows follow delyrism/pose.py — x y z visibility
// wx wy wz per landmark, NaN where nothing was found:
//   body 33 · hands 42 (the subject's left hand, then right) · face 478 ·
//   both 75 (body, then left hand, then right hand)
// Falls back to the CPU delegate (WASM) when WebGL is unavailable.

import type {
  FaceLandmarker, HandLandmarker, HandLandmarkerResult, Landmark, NormalizedLandmark, PoseLandmarker,
} from "@mediapipe/tasks-vision";

const VERSION = "1.1.1";                      // matches package.json
const WASM = `https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@${VERSION}/wasm`;
const MODELS = {
  pose: "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task",
  hand: "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task",
  face: "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task",
};

export type TrackerKind = "body" | "hands" | "face" | "both";
export const TRACKER_ROWS: Record<TrackerKind, number> = { body: 33, hands: 42, face: 478, both: 75 };
const TRACKER_ID: Record<TrackerKind, number> = { body: 0, hands: 1, face: 2, both: 3 };
const COLS = 7;
const HAND = 21;

export interface Tracker {
  kind: TrackerKind;
  delegate: "GPU" | "CPU";
  /** rows × 7 landmarks for this video frame, or null when nothing was found */
  detect: (video: TexImageSource, tMs: number) => Float32Array | null;
  close: () => void;
}

let fileset: Promise<any> | null = null;
async function vision() {
  const mod = await import("@mediapipe/tasks-vision");
  if (!fileset) fileset = mod.FilesetResolver.forVisionTasks(WASM);
  return { mod, fs: await fileset };
}

/** Build a landmarker on the GPU, or on the CPU when the GPU delegate fails. */
async function onBest<T>(create: (delegate: "GPU" | "CPU") => Promise<T>): Promise<[T, "GPU" | "CPU"]> {
  try {
    return [await create("GPU"), "GPU"];
  } catch (e) {
    console.warn("[poseTracker] GPU delegate unavailable, using CPU:", e);
    return [await create("CPU"), "CPU"];
  }
}

function fill(out: Float32Array, row0: number, lm: NormalizedLandmark[], world?: Landmark[], visible1 = false) {
  for (let i = 0; i < lm.length; i++) {
    const o = (row0 + i) * COLS, p = lm[i], w = world?.[i];
    out[o] = p.x; out[o + 1] = p.y; out[o + 2] = p.z;
    out[o + 3] = visible1 ? 1 : (p.visibility ?? 1);
    out[o + 4] = w ? w.x : NaN; out[o + 5] = w ? w.y : NaN; out[o + 6] = w ? w.z : NaN;
  }
}

/** Hands as [side, landmarks, world] — side is the subject's.  On an
 *  unmirrored camera the model's "Left" is the subject's left (checked
 *  against the body model in delyrism/pose.py). */
function sides(res: HandLandmarkerResult): ["left" | "right", NormalizedLandmark[], Landmark[] | undefined, number][] {
  const out = res.landmarks.map((lm, i) => {
    const c = res.handedness?.[i]?.[0];
    return [c?.categoryName === "Left" ? "left" : "right", lm, res.worldLandmarks?.[i], c?.score ?? 1] as
      ["left" | "right", NormalizedLandmark[], Landmark[] | undefined, number];
  });
  if (out.length === 2 && out[0][0] === out[1][0]) {          // same call twice: the surer keeps it
    const loser = out[0][3] >= out[1][3] ? 1 : 0;
    out[loser][0] = out[1 - loser][0] === "left" ? "right" : "left";
  }
  return out;
}

export async function createTracker(kind: TrackerKind): Promise<Tracker> {
  const { mod, fs } = await vision();
  const base = (path: string, delegate: "GPU" | "CPU") => ({ modelAssetPath: path, delegate });
  let pose: PoseLandmarker | null = null, hand: HandLandmarker | null = null, face: FaceLandmarker | null = null;
  let delegate: "GPU" | "CPU" = "GPU";

  if (kind === "body" || kind === "both") {
    [pose, delegate] = await onBest((d) => mod.PoseLandmarker.createFromOptions(fs, {
      baseOptions: base(MODELS.pose, d), runningMode: "VIDEO", numPoses: 1,
    }));
  }
  if (kind === "hands" || kind === "both") {
    let d: "GPU" | "CPU";
    [hand, d] = await onBest((dd) => mod.HandLandmarker.createFromOptions(fs, {
      baseOptions: base(MODELS.hand, dd), runningMode: "VIDEO", numHands: 2,
    }));
    if (d === "CPU") delegate = "CPU";
  }
  if (kind === "face") {
    [face, delegate] = await onBest((d) => mod.FaceLandmarker.createFromOptions(fs, {
      baseOptions: base(MODELS.face, d), runningMode: "VIDEO", numFaces: 1,
    }));
  }

  const n = TRACKER_ROWS[kind];
  let lastMs = -1;
  const detect = (video: TexImageSource, tMs: number): Float32Array | null => {
    const ms = Math.max(Math.round(tMs), lastMs + 1);       // VIDEO mode wants strictly increasing time
    lastMs = ms;
    const out = new Float32Array(n * COLS).fill(NaN);
    let found = false;

    if (face) {
      const r = face.detectForVideo(video, ms);
      if (r.faceLandmarks?.[0]) { fill(out, 0, r.faceLandmarks[0].slice(0, n), undefined, true); found = true; }
      return found ? out : null;
    }
    let body: NormalizedLandmark[] | undefined;
    if (pose) {
      const r = pose.detectForVideo(video, ms);
      body = r.landmarks?.[0];
      if (body) { fill(out, 0, body, r.worldLandmarks?.[0]); found = true; }
    }
    if (hand) {
      let hands = sides(hand.detectForVideo(video, ms));
      const off = kind === "both" ? 33 : 0;
      if (kind === "both" && body) {
        // pin each hand to the nearest visible body wrist; drop hands farther
        // than 0.6 torso from both (someone else's)
        const wr: Record<string, NormalizedLandmark> = {};
        if ((body[15].visibility ?? 1) >= 0.3) wr.left = body[15];
        if ((body[16].visibility ?? 1) >= 0.3) wr.right = body[16];
        const dist = (a: NormalizedLandmark, s: string) => (wr[s] ? Math.hypot(a.x - wr[s].x, a.y - wr[s].y) : Infinity);
        if (Object.keys(wr).length) {
          const torso = (Math.hypot(body[11].x - body[23].x, body[11].y - body[23].y) +
                         Math.hypot(body[12].x - body[24].x, body[12].y - body[24].y)) / 2;
          const gate = Math.max(0.05, 0.6 * torso);
          if (hands.length === 2 && wr.left && wr.right) {
            const straight = dist(hands[0][1][0], "left") + dist(hands[1][1][0], "right");
            const crossed = dist(hands[0][1][0], "right") + dist(hands[1][1][0], "left");
            hands[0][0] = straight <= crossed ? "left" : "right";
            hands[1][0] = straight <= crossed ? "right" : "left";
          } else {
            hands = hands.map((h) => [dist(h[1][0], "left") <= dist(h[1][0], "right") ? "left" : "right", h[1], h[2], h[3]]);
          }
          hands = hands.filter((h) => dist(h[1][0], h[0]) <= gate);
        }
      }
      for (const [side, lm, world] of hands) {
        fill(out, off + (side === "left" ? 0 : HAND), lm, world, true);
        found = true;
      }
    }
    return found ? out : null;
  };

  return {
    kind, delegate, detect,
    close: () => { pose?.close(); hand?.close(); face?.close(); },
  };
}

/** The binary payload for the live socket (kind 6, see realtime.py _parse_landmarks). */
export function packLandmarks(kind: TrackerKind, aspect: number, tSec: number, rows: Float32Array | null): ArrayBuffer {
  const n = rows ? rows.length / COLS : 0;
  const buf = new ArrayBuffer(20 + n * COLS * 4);
  const dv = new DataView(buf);
  dv.setUint8(0, TRACKER_ID[kind]);
  dv.setFloat32(4, aspect, true);
  dv.setFloat64(8, tSec, true);
  dv.setUint32(16, n, true);
  if (rows) new Float32Array(buf, 20, rows.length).set(rows);
  return buf;
}
