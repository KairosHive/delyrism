"""
datasets.py — public movement recordings for the grapheme bridge
==================================================================

Three open sources, fetched once into ~/.cache/delyrism/motion:

  UTD-MHAD   27 actions, 8 people × 4 trials, a wrist IMU (accelerometer +
             gyroscope, 50 Hz) and a Kinect skeleton (20 joints, 30 Hz)
             recorded together.  Actions 8-11 draw shapes in the air
             (an X, a circle clockwise, a circle counter-clockwise, a
             triangle): ground truth for both bridges.  ~20 MB.
             Chen, Jafari & Kehtarnavaz 2015.
  EMOKINE    one dancer, 9 choreographies × 7 renditions (explanation +
             six emotions), a 17-sensor Xsens suit.  Only the per-segment
             CSVs (angular velocity, acceleration, orientation, position)
             are read, straight out of the 1.7 GB archive with HTTP range
             requests.  CC BY 4.0.  Christensen et al. 2024.
  AIST++     3D dance keypoints (17 COCO joints, 60 fps) for 1,408
             sequences over 10 genres, from multi-view video.  A chosen
             subset is read out of the 834 MB archive the same way.
             Li et al. 2021.
"""
from __future__ import annotations

import io
import os
import re
import zipfile
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

CACHE = os.path.join(os.path.expanduser("~"), ".cache", "delyrism", "motion")
UTD_URL = "https://personal.utdallas.edu/~kehtar/UTD-MAD/{part}.zip"
EMOKINE_URL = "https://zenodo.org/records/7821844/files/EmokineDataset_v1.0.zip?download=1"
AIST_URL = "https://github.com/google/aistplusplus_dataset/releases/download/v1.0/keypoints3d.zip"

UTD_ACTIONS = ["swipe left", "swipe right", "wave", "clap", "throw", "arm cross", "basketball shoot",
               "draw x", "draw circle cw", "draw circle ccw", "draw triangle", "bowling", "boxing",
               "baseball swing", "tennis swing", "arm curl", "tennis serve", "push", "knock", "catch",
               "pickup throw", "jog", "walk", "sit to stand", "stand to sit", "lunge", "squat"]
# Kinect v1 joints in UTD-MHAD skeleton files (20)
KINECT = ["hip_center", "spine", "shoulder_center", "head", "shoulder_left", "elbow_left", "wrist_left",
          "hand_left", "shoulder_right", "elbow_right", "wrist_right", "hand_right", "hip_left", "knee_left",
          "ankle_left", "foot_left", "hip_right", "knee_right", "ankle_right", "foot_right"]
COCO = ["nose", "left_eye", "right_eye", "left_ear", "right_ear", "left_shoulder", "right_shoulder",
        "left_elbow", "right_elbow", "left_wrist", "right_wrist", "left_hip", "right_hip", "left_knee",
        "right_knee", "left_ankle", "right_ankle"]


# ----------------------------------------------------------------------------
#  reading members out of a remote zip with HTTP range requests
# ----------------------------------------------------------------------------

class _HttpFile(io.RawIOBase):
    """A seekable read-only file over HTTP range requests (block cached)."""

    BLOCK = 1 << 20

    def __init__(self, url: str):
        import requests
        self.url, self._s = url, requests.Session()
        r = self._s.head(url, allow_redirects=True, timeout=60)
        self.url = r.url
        self.size = int(r.headers["Content-Length"])
        self.pos = 0
        self._cache: Dict[int, bytes] = {}

    def readable(self): return True
    def seekable(self): return True
    def tell(self): return self.pos

    def seek(self, off, whence=0):
        self.pos = off if whence == 0 else self.pos + off if whence == 1 else self.size + off
        return self.pos

    def _block(self, i: int) -> bytes:
        if i not in self._cache:
            a, b = i * self.BLOCK, min(self.size, (i + 1) * self.BLOCK) - 1
            r = self._s.get(self.url, headers={"Range": f"bytes={a}-{b}"}, timeout=120)
            r.raise_for_status()
            self._cache[i] = r.content
            if len(self._cache) > 64:
                self._cache.pop(next(iter(self._cache)))
        return self._cache[i]

    def read(self, n=-1):
        if n is None or n < 0:
            n = self.size - self.pos
        n = max(0, min(n, self.size - self.pos))
        out = bytearray()
        while n > 0:
            i, o = divmod(self.pos, self.BLOCK)
            chunk = self._block(i)[o:o + n]
            out += chunk
            self.pos += len(chunk)
            n -= len(chunk)
        return bytes(out)

    def readinto(self, b):
        data = self.read(len(b))
        b[:len(data)] = data
        return len(data)


def fetch_members(url: str, dest: str, want) -> List[str]:
    """Extract the members of a remote zip whose name satisfies `want`
    into dest (skipping ones already there); returns their local paths."""
    os.makedirs(dest, exist_ok=True)
    z = zipfile.ZipFile(_HttpFile(url))
    out = []
    for info in z.infolist():
        if info.is_dir() or not want(info.filename):
            continue
        path = os.path.join(dest, info.filename.replace("/", os.sep))
        if not os.path.exists(path):
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with z.open(info) as src, open(path, "wb") as dst:
                dst.write(src.read())
        out.append(path)
    return out


def list_members(url: str) -> List[Tuple[str, int]]:
    z = zipfile.ZipFile(_HttpFile(url))
    return [(i.filename, i.file_size) for i in z.infolist()]


# ----------------------------------------------------------------------------
#  UTD-MHAD
# ----------------------------------------------------------------------------

def _utd_dir() -> str:
    root = os.path.join(CACHE, "utd_mhad")
    for part in ("Inertial", "Skeleton"):
        if not os.path.isdir(os.path.join(root, part)):
            import urllib.request
            os.makedirs(root, exist_ok=True)
            zp = os.path.join(root, part + ".zip")
            if not os.path.exists(zp):
                urllib.request.urlretrieve(UTD_URL.format(part=part), zp)
            zipfile.ZipFile(zp).extractall(root)
    return root


def load_utd(actions: Optional[Iterable[int]] = None) -> List[dict]:
    """UTD-MHAD trials: {action (1-27), name, subject, trial, imu (T, 6)
    accel g + gyro deg/s at 50 Hz, skeleton (T, 20, 3) metres at 30 Hz}."""
    from scipy.io import loadmat
    root = _utd_dir()
    keep = set(actions) if actions else None
    out = []
    for dirpath, _, files in os.walk(root):
        for fn in files:
            m = re.match(r"a(\d+)_s(\d+)_t(\d+)_inertial\.mat$", fn)
            if not m:
                continue
            a, s, t = (int(x) for x in m.groups())
            if keep and a not in keep:
                continue
            imu = loadmat(os.path.join(dirpath, fn))["d_iner"].astype(np.float64)
            skp = None
            for d2, _, f2 in os.walk(root):
                if f"a{a}_s{s}_t{t}_skeleton.mat" in f2:
                    skp = os.path.join(d2, f"a{a}_s{s}_t{t}_skeleton.mat")
                    break
            sk = loadmat(skp)["d_skel"].astype(np.float64).transpose(2, 0, 1) if skp else None
            out.append(dict(action=a, name=UTD_ACTIONS[a - 1], subject=s, trial=t, imu=imu, skeleton=sk))
    return sorted(out, key=lambda r: (r["action"], r["subject"], r["trial"]))


# ----------------------------------------------------------------------------
#  EMOKINE (Xsens dance)
# ----------------------------------------------------------------------------

def emokine_files(features=("angularVelocity", "acceleration", "orientation", "position"),
                  emotions=("joy", "sad"), sequences=range(1, 10)) -> List[str]:
    """Fetch the per-feature CSVs of the chosen sequences (only those files,
    about 1.8 MB each)."""
    dest = os.path.join(CACHE, "emokine")
    seqs = {f"CSV_seq{q}_{e}" for q in sequences for e in emotions}
    feats = {f + ".csv" for f in features}
    return fetch_members(EMOKINE_URL, dest,
                         lambda n: "/CSV/" in n and n.split("/")[-2] in seqs and n.split("/")[-1] in feats)


def load_emokine(seq: int, emotion: str, fps: float = 60.0) -> Dict[str, object]:
    """One EMOKINE take: {segments, fps, angularVelocity (T, S, 3),
    acceleration (T, S, 3), orientation (T, S, 4) quaternion w x y z
    (segment to world, z up), position (T, S, 3) metres}, resampled to fps."""
    import pandas as pd
    files = emokine_files(emotions=(emotion,), sequences=(seq,))
    out: Dict[str, object] = {}
    for p in files:
        feat = os.path.basename(p)[:-4]
        df = pd.read_csv(p)
        ms = df["ms"].to_numpy(np.float64)
        cols = [c for c in df.columns if c not in ("frame_idx", "ms")]
        segs = list(dict.fromkeys(re.sub(r"_[^_]+$", "", c) for c in cols))
        X = df[cols].to_numpy(np.float64).reshape(len(df), len(segs), len(cols) // len(segs))
        t = (ms - ms[0]) / 1000.0
        g = np.arange(0, t[-1], 1.0 / fps)
        out[feat] = np.stack([np.stack([np.interp(g, t, X[:, s_, d]) for d in range(X.shape[2])], 1)
                              for s_ in range(len(segs))], 1)
        out["segments"] = segs
        out["source_fps"] = float(1.0 / np.median(np.diff(t)))
    out["fps"] = fps
    return out


# ----------------------------------------------------------------------------
#  AIST++
# ----------------------------------------------------------------------------

AIST_GENRES = {"gBR": "break", "gPO": "pop", "gLO": "lock", "gMH": "middle hip-hop", "gLH": "LA hip-hop",
               "gHO": "house", "gWA": "waack", "gKR": "krump", "gJS": "street jazz", "gJB": "ballet jazz"}


class _NumpyOnlyUnpickler:
    """Unpickle AIST++ annotation files while refusing anything but plain
    containers and numpy arrays (a downloaded pickle can otherwise run code)."""

    ALLOWED = {("numpy.core.multiarray", "_reconstruct"), ("numpy._core.multiarray", "_reconstruct"),
               ("numpy", "ndarray"), ("numpy", "dtype"), ("numpy.core.multiarray", "scalar"),
               ("numpy._core.multiarray", "scalar")}

    def __init__(self, f):
        import pickle

        allowed = self.ALLOWED

        class _U(pickle.Unpickler):
            def find_class(self, module, name):
                if (module, name) not in allowed:
                    raise pickle.UnpicklingError(f"refusing {module}.{name}")
                return super().find_class(module, name)

        self._u = _U(f)

    def load(self):
        return self._u.load()


def aist_keypoints(per_genre: int = 6, seed: int = 0) -> Dict[str, np.ndarray]:
    """{sequence name: (T, 17, 3) keypoints (cm, 60 fps)} for a few
    sequences of each genre, read out of the remote archive."""
    dest = os.path.join(CACHE, "aist")
    names = [n for n, _ in list_members(AIST_URL) if n.endswith(".pkl")]
    rng = np.random.default_rng(seed)
    pick = []
    for g in AIST_GENRES:
        gn = sorted(n for n in names if os.path.basename(n).startswith(g) and "sBM" in n)   # basic dances
        pick += [gn[i] for i in rng.choice(len(gn), min(per_genre, len(gn)), replace=False)]
    paths = fetch_members(AIST_URL, dest, lambda n: n in set(pick))
    out = {}
    for p in paths:
        with open(p, "rb") as f:
            d = _NumpyOnlyUnpickler(f).load()
        k = d.get("keypoints3d_optim", d.get("keypoints3d"))
        if k is not None:
            out[os.path.basename(p)[:-4]] = np.asarray(k, np.float64)
    return out
