# =========================
# Persistent-homology utils
# =========================
import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations
from typing import Dict, Tuple
import pandas as pd

# ---- helpers
def _ensure_2d(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    if X.ndim != 2:
        raise ValueError("Embeddings must be a 2D array [n_points, n_dims].")
    # (re)normalize rows just in case
    norms = np.linalg.norm(X, axis=1, keepdims=True) + 1e-12
    return X / norms

def _pairwise_dists(X: np.ndarray, metric: str = "euclidean") -> np.ndarray:
    X = _ensure_2d(X)
    if metric == "euclidean":
        diff = X[:, None, :] - X[None, :, :]
        return np.sqrt(np.sum(diff * diff, axis=2))
    elif metric == "cosine":
        S = X @ X.T
        S = np.clip(S, -1.0, 1.0)
        return 1.0 - S
    else:
        raise ValueError("metric must be 'euclidean' or 'cosine'")

# ---- H0 via MST (fast single-linkage lifetimes)
class _UF:
    def __init__(self, n):
        self.p = list(range(n)); self.r = [0]*n
    def f(self, x):
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]; x = self.p[x]
        return x
    def u(self, a, b):
        ra, rb = self.f(a), self.f(b)
        if ra == rb: return False
        if self.r[ra] < self.r[rb]: self.p[ra] = rb
        elif self.r[ra] > self.r[rb]: self.p[rb] = ra
        else: self.p[rb] = ra; self.r[ra] += 1
        return True

def h0_bar_lengths_from_mst(X: np.ndarray, metric: str = "euclidean") -> np.ndarray:
    D = _pairwise_dists(X, metric=metric)
    n = len(X)
    edges = [(D[i, j], i, j) for i in range(n) for j in range(i+1, n)]
    edges.sort(key=lambda t: t[0])
    uf = _UF(n)
    lengths = []
    for w, i, j in edges:
        if uf.u(i, j):
            lengths.append(float(w))
            if len(lengths) == n-1:
                break
    return np.array(lengths)

def plot_h0_barcode_from_mst(lengths: np.ndarray, title: str):
    order = np.argsort(lengths)
    plt.figure(figsize=(8, 4))
    for k, idx in enumerate(order):
        L = float(lengths[idx])
        plt.hlines(k, 0.0, L)
    plt.xlabel("Scale (ε)")
    plt.ylabel("Component index (sorted)")
    plt.title(title)
    plt.tight_layout()
    plt.show()

# ---- Rips persistent homology (ripser -> fallback)
def _vr_filtration_simplices_up_to3(D: np.ndarray):
    n = len(D)
    simplices = []
    # 0-simplices
    for i in range(n):
        simplices.append((0.0, 0, (i,)))
    # 1-simplices
    for i, j in combinations(range(n), 2):
        simplices.append((float(D[i, j]), 1, (i, j)))
    # 2-simplices (triangles)
    def tri_birth(i, j, k):
        return max(float(D[i, j]), float(D[i, k]), float(D[j, k]))
    for i, j, k in combinations(range(n), 3):
        simplices.append((tri_birth(i, j, k), 2, (i, j, k)))
    # 3-simplices (tetrahedra)
    def tetra_birth(i, j, k, l):
        idx = [i, j, k, l]
        m = 0.0
        for a, b in combinations(idx, 2):
            m = max(m, float(D[a, b]))
        return m
    for i, j, k, l in combinations(range(n), 4):
        simplices.append((tetra_birth(i, j, k, l), 3, (i, j, k, l)))
    simplices.sort(key=lambda t: (t[0], t[1]))
    return simplices

def _boundary_faces(simplex):
    s = list(simplex)
    return [tuple(s[:k] + s[k+1:]) for k in range(len(s))]

def _persistent_pairs_fallback(D: np.ndarray, max_return_dim=2):
    simplices = _vr_filtration_simplices_up_to3(D)
    m = len(simplices)
    index = {simplices[i][2]: i for i in range(m)}
    boundary = [set() for _ in range(m)]
    births = [simplices[i][0] for i in range(m)]
    for j, (_, dim, verts) in enumerate(simplices):
        if dim == 0: 
            continue
        for face in _boundary_faces(verts):
            boundary[j].add(index[face])

    low = {}
    paired = [False]*m
    dgms = {d: [] for d in range(max_return_dim+1)}

    def low_row(j):
        return max(boundary[j]) if boundary[j] else None

    for j in range(m):
        if simplices[j][1] == 0:
            continue
        lr = low_row(j)
        while lr is not None and lr in low:
            boundary[j] ^= boundary[low[lr]]
            lr = low_row(j)
        if lr is not None:
            i = lr
            birth = births[i]
            death = births[j]
            dim_class = simplices[i][1]
            if dim_class <= max_return_dim:
                dgms[dim_class].append((birth, death))
            low[lr] = j
            paired[i] = True; paired[j] = True

    for idx, (b, dim, _) in enumerate(simplices):
        if not paired[idx] and dim <= max_return_dim:
            dgms[dim].append((b, np.inf))

    for d in dgms:
        arr = np.array(dgms[d], dtype=float) if dgms[d] else np.zeros((0, 2))
        dgms[d] = arr
    return dgms

def compute_persistent_homology(X: np.ndarray, metric: str = "euclidean", maxdim: int = 2):
    X = _ensure_2d(X)
    try:
        from ripser import ripser
        out = ripser(X, maxdim=maxdim, metric=metric)
        dgms = out["dgms"]
        return {0: dgms[0], 1: dgms[1], 2: dgms[2] if maxdim >= 2 else np.zeros((0, 2))}, "ripser"
    except Exception:
        D = _pairwise_dists(X, metric=metric)
        return _persistent_pairs_fallback(D, max_return_dim=maxdim), "fallback"

# ---- plotting: PCA / PD / persistence image
def pca_2d(X: np.ndarray) -> np.ndarray:
    X = _ensure_2d(X)
    Xc = X - X.mean(0, keepdims=True)
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    return Xc @ Vt[:2].T

def plot_pca_scatter(X: np.ndarray, title: str):
    Z = pca_2d(X)
    plt.figure(figsize=(5, 5))
    plt.scatter(Z[:, 0], Z[:, 1], s=14)
    plt.title(title)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.tight_layout()
    plt.show()

def plot_persistence_diagram(dgms: Dict[int, np.ndarray], title: str):
    vmax = 0.0
    for arr in dgms.values():
        if arr.size:
            finite = np.where(np.isfinite(arr[:, 1]), arr[:, 1], arr[:, 0] + 0.1)
            vmax = max(vmax, float(np.max(finite)))
    vmax = max(vmax, 1.0)
    plt.figure(figsize=(7, 6))
    for h in sorted(dgms.keys()):
        arr = dgms[h]
        if arr.size:
            y = np.where(np.isfinite(arr[:, 1]), arr[:, 1], vmax)
            plt.scatter(arr[:, 0], y, s=18, label=f"H{h}")
    plt.plot([0, vmax], [0, vmax], linestyle="--", linewidth=1)
    plt.xlabel("Birth")
    plt.ylabel("Death (∞ truncated)")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.show()

def _persistence_image(dgm: np.ndarray, res: int = 30) -> np.ndarray:
    if dgm.size == 0:
        return np.zeros((res, res))
    pts = np.copy(dgm)
    pts[:, 1] = pts[:, 1] - pts[:, 0]  # persistence
    finite = np.isfinite(pts[:, 1])
    pts = pts[finite]
    if pts.size == 0:
        return np.zeros((res, res))
    xb = np.linspace(0, float(np.max(pts[:, 0]) + 1e-9), res + 1)
    yb = np.linspace(0, float(np.max(pts[:, 1]) + 1e-9), res + 1)
    img = np.zeros((res, res))
    for x, y in pts:
        xi = np.searchsorted(xb, x) - 1
        yi = np.searchsorted(yb, y) - 1
        if 0 <= xi < res and 0 <= yi < res:
            img[yi, xi] += 1.0
    return img

def plot_persistence_image(dgm: np.ndarray, title: str, res: int = 30):
    img = _persistence_image(dgm, res=res)
    plt.figure(figsize=(5, 4))
    plt.imshow(img, origin="lower", aspect="auto")
    plt.title(title)
    plt.tight_layout()
    plt.show()

# ---- numeric summaries per symbol
# --- 1) Summaries: now includes H2 as well ---
def _summarize_diagrams(symbol: str, dgms: Dict[int, np.ndarray], thr: float = 0.02) -> Dict[str, float]:
    row = {"symbol": symbol}

    # H0 (cohesion / outlierness)
    H0 = dgms.get(0, np.zeros((0, 2)))
    if H0.size and np.isfinite(H0[:,1]).any():
        finite0 = H0[np.isfinite(H0[:,1])]
        pers0 = finite0[:,1] - finite0[:,0]
        row.update({"H0_median": float(np.median(pers0)), "H0_max": float(np.max(pers0))})
    else:
        row.update({"H0_median": 0.0, "H0_max": 0.0})

    # H1 (loopiness)
    H1 = dgms.get(1, np.zeros((0, 2)))
    if H1.size and np.isfinite(H1[:,1]).any():
        finite1 = H1[np.isfinite(H1[:,1])]
        pers1 = finite1[:,1] - finite1[:,0]
        row.update({
            "H1_sum": float(np.sum(pers1)),
            "H1_max": float(np.max(pers1)),
            "H1_count_gt_thr": int(np.sum(pers1 > thr))
        })
    else:
        row.update({"H1_sum": 0.0, "H1_max": 0.0, "H1_count_gt_thr": 0})

    # H2 (voids/cavities)
    H2 = dgms.get(2, np.zeros((0, 2)))
    if H2.size and np.isfinite(H2[:,1]).any():
        finite2 = H2[np.isfinite(H2[:,1])]
        pers2 = finite2[:,1] - finite2[:,0]
        row.update({
            "H2_sum": float(np.sum(pers2)),
            "H2_max": float(np.max(pers2)),
            "H2_count_gt_thr": int(np.sum(pers2 > thr))
        })
    else:
        row.update({"H2_sum": 0.0, "H2_max": 0.0, "H2_count_gt_thr": 0})

    return row

def reduce_pcs(X, n_pcs=None):
    X = np.asarray(X, float)
    if n_pcs is None:  # full embeddings
        return X
    Xc = X - X.mean(0, keepdims=True)
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    Z = Xc @ Vt[:n_pcs].T
    # re-normalize rows to keep cosine≈euclid behavior
    Z /= (np.linalg.norm(Z, axis=1, keepdims=True) + 1e-12)
    return Z

# =========================
# 1) Per-symbol plots API
# =========================
# --- 2) Per-symbol plots: add optional H2 image ---
def plot_symbolwise_PH(
    symbol_to_embeddings: Dict[str, np.ndarray],
    metric: str = "euclidean",
    maxdim: int = 2,
    persistence_image_res: int = 30,
    show_h2_image: bool = True
):
    """
    For each symbol:
      - PCA scatter
      - H0 barcode (MST)
      - Persistence diagram (H0..Hmaxdim)
      - Persistence image for H1 (if present)
      - [NEW] Persistence image for H2 if show_h2_image=True and H2 present
    """
    summaries = []
    for symbol, X in symbol_to_embeddings.items():
        X = _ensure_2d(X)
        X = reduce_pcs(X, n_pcs=10)
        plot_pca_scatter(X, f"{symbol} — PCA scatter")
        lengths = h0_bar_lengths_from_mst(X, metric=metric)
        plot_h0_barcode_from_mst(lengths, f"{symbol} — H0 barcode via MST")
        dgms, src = compute_persistent_homology(X, metric=metric, maxdim=maxdim)
        plot_persistence_diagram(dgms, f"{symbol} — Persistence diagram (source: {src})")
        if maxdim >= 1 and dgms[1].size:
            plot_persistence_image(dgms[1], f"{symbol} — Persistence image (H1)", res=persistence_image_res)
        if show_h2_image and maxdim >= 2 and dgms[2].size:
            plot_persistence_image(dgms[2], f"{symbol} — Persistence image (H2)", res=persistence_image_res)
        summaries.append(_summarize_diagrams(symbol, dgms))
    return summaries


# =========================
# 2) Cross-symbol comparison
# =========================
# --- 3) Cross-symbol comparison: add H2 bars & a H1-vs-H2 map ---
def plot_comparative_PH(
    symbol_to_embeddings: Dict[str, np.ndarray],
    metric: str = "euclidean",
    thr: float = 0.02
):
    """
    Produces cross-symbol charts:
      - H0_median (cohesion)
      - H0_max (outlierness)
      - H1_sum (loopiness) + H1_count_gt_thr
      - [NEW] H2_sum (voids) + H2_count_gt_thr
      - 2D map: Cohesion vs Loopiness
      - [NEW] 2D map: H1 (loopiness) vs H2 (voidiness)
    """

    rows = []
    for sym, X in symbol_to_embeddings.items():
        X = _ensure_2d(X)
        X = reduce_pcs(X, n_pcs=10)
        dgms, _ = compute_persistent_homology(X, metric=metric, maxdim=2)
        rows.append(_summarize_diagrams(sym, dgms, thr=thr))
    df = pd.DataFrame(rows)
    symbols = df["symbol"].values

    # H0 cohesion
    plt.figure(); plt.bar(symbols, df["H0_median"].values)
    plt.ylabel("H0 median persistence (cohesion)"); plt.title("Cohesion by symbol (lower=tighter)")
    plt.xticks(rotation=20); plt.tight_layout(); plt.show()

    # H0 outlierness
    plt.figure(); plt.bar(symbols, df["H0_max"].values)
    plt.ylabel("H0 max persistence (outlierness)"); plt.title("Outlierness by symbol")
    plt.xticks(rotation=20); plt.tight_layout(); plt.show()

    # H1 loopiness
    plt.figure(); plt.bar(symbols, df["H1_sum"].values)
    plt.ylabel("H1 sum of persistences"); plt.title("Loopiness by symbol (H1 total)")
    plt.xticks(rotation=20); plt.tight_layout(); plt.show()

    plt.figure(); plt.bar(symbols, df["H1_count_gt_thr"].values)
    plt.ylabel(f"H1 count (> {thr})"); plt.title("Number of 'significant' H1 loops")
    plt.xticks(rotation=20); plt.tight_layout(); plt.show()

    # [NEW] H2 voidiness
    plt.figure(); plt.bar(symbols, df["H2_sum"].values)
    plt.ylabel("H2 sum of persistences"); plt.title("Voidiness by symbol (H2 total)")
    plt.xticks(rotation=20); plt.tight_layout(); plt.show()

    plt.figure(); plt.bar(symbols, df["H2_count_gt_thr"].values)
    plt.ylabel(f"H2 count (> {thr})"); plt.title("Number of 'significant' H2 voids")
    plt.xticks(rotation=20); plt.tight_layout(); plt.show()

    # 2D maps
    plt.figure()
    x = df["H0_median"].values; y = df["H1_sum"].values
    plt.scatter(x, y)
    for i, s in enumerate(symbols):
        plt.annotate(s, (x[i], y[i]), xytext=(5, 5), textcoords="offset points")
    plt.xlabel("Cohesion (H0 median) — lower is tighter")
    plt.ylabel("Loopiness (H1 sum)")
    plt.title("Symbol map: Cohesion vs Loopiness (H1)")
    plt.tight_layout(); plt.show()

    plt.figure()
    x = df["H1_sum"].values; y = df["H2_sum"].values
    plt.scatter(x, y)
    for i, s in enumerate(symbols):
        plt.annotate(s, (x[i], y[i]), xytext=(5, 5), textcoords="offset points")
    plt.xlabel("Loopiness (H1 sum)")
    plt.ylabel("Voidiness (H2 sum)")
    plt.title("Symbol map: H1 vs H2")
    plt.tight_layout(); plt.show()

    return df


# =========================
# Convenience switch
# =========================
def plot_symbols(
    symbol_to_embeddings: Dict[str, np.ndarray],
    mode: str = "per-symbol",
    metric: str = "euclidean",
    **kwargs
):
    """
    mode='per-symbol' -> calls plot_symbolwise_PH(...)
    mode='compare'    -> calls plot_comparative_PH(...)
    """
    if mode == "per-symbol":
        return plot_symbolwise_PH(symbol_to_embeddings, metric=metric, **kwargs)
    elif mode == "compare":
        return plot_comparative_PH(symbol_to_embeddings, metric=metric, **kwargs)
    else:
        raise ValueError("mode must be 'per-symbol' or 'compare'")


# ============================
# Topology Report (printable)
# ============================


# ---------- basic helpers ----------
def _row_norm(X): return X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)

def _sum_finite(dgm: np.ndarray) -> float:
    if dgm.size == 0: return 0.0
    mask = np.isfinite(dgm[:,1])
    return float(np.sum(dgm[mask,1] - dgm[mask,0])) if mask.any() else 0.0

def _ph(X, metric="euclidean", maxdim=2):
    from ripser import ripser
    return ripser(_row_norm(np.asarray(X, float)), maxdim=maxdim, metric=metric)["dgms"]

def ph_summary(X, metric="euclidean"):
    dgms = _ph(X, metric=metric, maxdim=2)
    H0, H1, H2 = dgms[0], dgms[1], dgms[2]
    if H0.size and np.isfinite(H0[:,1]).any():
        p0 = H0[np.isfinite(H0[:,1])]
        coh = float(np.median(p0[:,1]-p0[:,0]))   # lower=tighter
        out = float(np.max(p0[:,1]-p0[:,0]))
    else:
        coh, out = 0.0, 0.0
    return {
        "H0_cohesion": coh,       # median H0 persistence (lower = tighter)
        "H0_outlier":  out,       # max H0 persistence
        "H1_sum":      _sum_finite(H1),
        "H2_sum":      _sum_finite(H2),
    }

# ---------- intrinsic topology score ----------
def intrinsic_topology_scores(embeddings_dict, metric="euclidean"):
    rows=[]
    for s,X in embeddings_dict.items():
        m = ph_summary(X, metric=metric)
        rows.append({"symbol":s, **m})
    df = pd.DataFrame(rows)
    # z-scores and composite
    for col in ["H1_sum","H2_sum","H0_cohesion"]:
        df[col+"_z"] = (df[col]-df[col].mean())/(df[col].std()+1e-12)
    df["TopoScore"] = df["H1_sum_z"] + df["H2_sum_z"] - df["H0_cohesion_z"]
    return df.sort_values("TopoScore", ascending=False)

# ---------- pair synergy (shared topology) ----------
def _pair_synergy(XA, XB, metric="euclidean") -> tuple[float,float]:
    from ripser import ripser
    XA = _row_norm(XA); XB = _row_norm(XB)
    X  = np.vstack([XA, XB])
    outU = ripser(X, maxdim=2, metric=metric)
    s1U, s2U = _sum_finite(outU["dgms"][1]), _sum_finite(outU["dgms"][2])
    # remove cross edges via precomputed distances
    diff = X[:,None,:]-X[None,:,:]
    D = np.sqrt(np.sum(diff*diff, axis=2))
    nA = len(XA); big = 1e9
    for i in range(len(X)):
        for j in range(i+1,len(X)):
            if (i<nA) != (j<nA):
                D[i,j]=D[j,i]=big
    outNC = ripser(D, maxdim=2, metric="precomputed")
    s1NC, s2NC = _sum_finite(outNC["dgms"][1]), _sum_finite(outNC["dgms"][2])
    return (s1U - s1NC, s2U - s2NC)   # synergy_H1, synergy_H2

def synergy_centrality(embeddings_dict, metric="euclidean"):
    import itertools
    keys = list(embeddings_dict.keys())
    acc = {k: {"synergy_H1":0.0,"synergy_H2":0.0} for k in keys}
    pair_rows=[]
    for a,b in itertools.combinations(keys,2):
        h1,h2 = _pair_synergy(embeddings_dict[a], embeddings_dict[b], metric=metric)
        acc[a]["synergy_H1"] += h1; acc[b]["synergy_H1"] += h1
        acc[a]["synergy_H2"] += h2; acc[b]["synergy_H2"] += h2
        pair_rows.append({"pair":f"{a}–{b}","synergy_H1":h1,"synergy_H2":h2})
    sym = pd.DataFrame([{"symbol":k, **v} for k,v in acc.items()])
    for col in ["synergy_H1","synergy_H2"]:
        sym[col+"_z"] = (sym[col]-sym[col].mean())/(sym[col].std()+1e-12)
    sym["SharedScore"] = sym["synergy_H1_z"] + sym["synergy_H2_z"]
    return sym.sort_values("SharedScore", ascending=False), pd.DataFrame(pair_rows).sort_values("synergy_H1", ascending=False)

# ---------- PCA -> Varimax rotated basis ----------
def pca(X, k):
    X = np.asarray(X, float)
    mu = X.mean(0)
    Xc = X - mu
    U,S,Vt = np.linalg.svd(Xc, full_matrices=False)
    L = Vt[:k].T     # (d,k) loadings
    return mu, L

def varimax(Phi, gamma=1.0, q=50, tol=1e-6):
    p,k = Phi.shape
    R = np.eye(k); d=0
    for _ in range(q):
        Λ = Phi @ R
        u,s,vh = np.linalg.svd(Phi.T @ (Λ**3 - (gamma/p)*Λ @ np.diag(np.sum(Λ**2,axis=0))))
        R = u @ vh
        if abs(s.sum()-d) < tol: break
        d = s.sum()
    return Phi @ R, R

def build_global_rotated_basis(embeddings_dict, n_components=20):
    mats = [np.asarray(X,float) for X in embeddings_dict.values()]
    Xall = _row_norm(np.vstack(mats))
    mu, L = pca(Xall, n_components)
    Lr, R = varimax(L)
    return mu, Lr  # mean and rotated loadings (d,k)

def project_scores(X, mu, Lr):
    X = _row_norm(np.asarray(X,float))
    return (X - mu) @ Lr        # (n,k)

def reconstruct_from_scores(Zr, mu, Lr):
    return _row_norm(mu + Zr @ Lr.T)

# ---------- component amputation ----------
def amputate_components_and_measure(embeddings_dict, mu, Lr, metric="euclidean", comps=None):
    k = Lr.shape[1]
    if comps is None: comps = list(range(k))
    baselines = {s: ph_summary(X, metric=metric) for s,X in embeddings_dict.items()}
    results = {}
    for s,X in embeddings_dict.items():
        Z = project_scores(X, mu, Lr)
        base = baselines[s]
        rows=[]
        for c in comps:
            Zcut = Z.copy(); Zcut[:,c]=0.0
            Xcut = reconstruct_from_scores(Zcut, mu, Lr)
            m = ph_summary(Xcut, metric=metric)
            rows.append({
                "component": c+1,
                "ΔH1_sum": base["H1_sum"] - m["H1_sum"],
                "ΔH2_sum": base["H2_sum"] - m["H2_sum"],
                "ΔH0_cohesion": base["H0_cohesion"] - m["H0_cohesion"],
            })
        results[s] = pd.DataFrame(rows).sort_values("ΔH1_sum", ascending=False)
    return results

# ---------- top words on rotated components ----------
def top_words_on_component(X, words, mu, Lr, comp_index, topn=6):
    Z = project_scores(X, mu, Lr)              # (n,k)
    s = Z[:, comp_index]                        # scores on that component
    idx_pos = np.argsort(s)[-topn:][::-1]
    idx_neg = np.argsort(s)[:topn]
    return [words[i] for i in idx_pos], [words[i] for i in idx_neg]

# ---------- word-level: leave-one-out & cycle participation ----------
def loo_word_impact(X, words, metric="euclidean"):
    from ripser import ripser
    base = _ph(X, metric=metric, maxdim=2)
    baseH1, baseH2 = _sum_finite(base[1]), _sum_finite(base[2])
    rows=[]
    for i in range(len(X)):
        Xi = np.delete(X, i, axis=0)
        dg = ripser(_row_norm(Xi), maxdim=2, metric=metric)["dgms"]
        rows.append({
            "word": words[i],
            "ΔH1_sum": baseH1 - _sum_finite(dg[1]),
            "ΔH2_sum": baseH2 - _sum_finite(dg[2]),
        })
    return pd.DataFrame(rows).sort_values(["ΔH1_sum","ΔH2_sum"], ascending=False)

def cycle_participation_weights(X, words, metric="euclidean", top_k1=6, top_k2=4):
    from ripser import ripser
    out = ripser(_row_norm(X), maxdim=2, metric=metric, do_cocycles=True)
    H1, H2 = out["dgms"][1], out["dgms"][2]
    C1 = out.get("cocycles",[[],[],[]])[1] if "cocycles" in out else []
    C2 = out.get("cocycles",[[],[],[]])[2] if "cocycles" in out else []
    def score_by_vertices(H, C, topk, dim):
        if H.size==0: return np.zeros(len(words))
        pers = np.where(np.isfinite(H[:,1]), H[:,1]-H[:,0], 0.0)
        order = np.argsort(pers)[::-1][:topk]
        w = np.zeros(len(words))
        for idx in order:
            cyc = C[idx] if idx < len(C) else []
            verts=set()
            if dim==1:
                for i,j,_ in cyc: verts.update([int(i),int(j)])
            else:
                for row in cyc:
                    i,j,k = map(int, row[:3]); verts.update([i,j,k])
            for v in verts: w[v] += pers[idx]
        return w
    w1 = score_by_vertices(H1, C1, top_k1, dim=1)
    w2 = score_by_vertices(H2, C2, top_k2, dim=2)
    df = pd.DataFrame({"word": words, "H1_weight": w1, "H2_weight": w2, "Total": w1+w2})
    return df.sort_values("Total", ascending=False)

# ---------- main runner ----------
def run_topology_report(
    embeddings_dict: dict[str, np.ndarray],
    symbols_to_words: dict[str, list[str]],
    metric: str = "euclidean",
    n_components: int = 20,
    top_components_show: int = 3,
    top_words_show: int = 6,
    top_words_loo: int = 8,
    save_csv_dir: str | None = None
):
    print("\n==================== TOPOLOGY REPORT ====================\n")

    # 1) Intrinsic (per-symbol)
    print("1) Intrinsic topology per symbol")
    intr = intrinsic_topology_scores(embeddings_dict, metric=metric)
    print(intr[["symbol","H0_cohesion","H1_sum","H2_sum","TopoScore"]].to_string(index=False))
    if save_csv_dir is not None:
        intr.to_csv(f"{save_csv_dir}/intrinsic_topology.csv", index=False)

    # 2) Shared (synergy)
    print("\n2) Shared topology (synergy centrality)")
    sym_rank, pair_tbl = synergy_centrality(embeddings_dict, metric=metric)
    print(sym_rank[["symbol","synergy_H1","synergy_H2","SharedScore"]].to_string(index=False))
    print("\n   Top pairs by H1 synergy:")
    print(pair_tbl.head(10).to_string(index=False))
    if save_csv_dir is not None:
        sym_rank.to_csv(f"{save_csv_dir}/synergy_symbols.csv", index=False)
        pair_tbl.to_csv(f"{save_csv_dir}/synergy_pairs.csv", index=False)

    # 3) Global rotated basis
    print("\n3) Building global rotated basis (PCA -> Varimax) for component analysis...")
    mu, Lr = build_global_rotated_basis(embeddings_dict, n_components=n_components)
    print(f"   Learned {Lr.shape[1]} rotated components.")

    # 4) Component amputation: which axes cause topology?
    print("\n4) Component importance per symbol (Δ = baseline − after amputation)")
    comp_results = amputate_components_and_measure(embeddings_dict, mu, Lr, metric=metric)
    for s in embeddings_dict.keys():
        df = comp_results[s]
        topH1 = df.sort_values("ΔH1_sum", ascending=False).head(top_components_show)
        topH2 = df.sort_values("ΔH2_sum", ascending=False).head(top_components_show)
        # annotate with top± words on each component
        rows=[]
        for comp in sorted(set(topH1["component"]).union(set(topH2["component"]))):
            pos, neg = top_words_on_component(embeddings_dict[s], symbols_to_words[s], mu, Lr, comp-1, topn=top_words_show)
            r = df[df["component"]==comp].iloc[0].to_dict()
            rows.append({
                "component": comp,
                "ΔH1_sum": r["ΔH1_sum"],
                "ΔH2_sum": r["ΔH2_sum"],
                "RC+ words": ", ".join(pos),
                "RC− words": ", ".join(neg),
            })
        out = pd.DataFrame(rows).sort_values(["ΔH1_sum","ΔH2_sum"], ascending=False)
        print(f"\n   {s} — top components driving topology")
        print(out.to_string(index=False))
        if save_csv_dir is not None:
            out.to_csv(f"{save_csv_dir}/components_{s}.csv", index=False)

    # 5) Word-level: leave-one-out & cycle participation
    print("\n5) Word-level catalysts (per symbol)")
    for s in embeddings_dict.keys():
        print(f"\n   {s} — leave-one-out (how much a word supports H1/H2)")
        loo = loo_word_impact(embeddings_dict[s], symbols_to_words[s], metric=metric)
        print(loo.head(top_words_loo).to_string(index=False))
        print(f"\n   {s} — cycle participation weights (credits from top cocycles)")
        cpw = cycle_participation_weights(embeddings_dict[s], symbols_to_words[s], metric=metric)
        print(cpw.head(top_words_loo).to_string(index=False))
        if save_csv_dir is not None:
            loo.to_csv(f"{save_csv_dir}/loo_{s}.csv", index=False)
            cpw.to_csv(f"{save_csv_dir}/cycle_participation_{s}.csv", index=False)

    # 6) Short interpretive guide
    print("\n----------------------------------------------------------")
    print("How to read this:")
    print("• H0_cohesion: lower = tighter cluster (descriptors agree semantically).")
    print("• H1_sum: loopiness; H2_sum: voidiness. Bigger ⇒ richer topology.")
    print("• Synergy: positive when loops/voids require cross-symbol interactions.")
    print("• ΔH1_sum / ΔH2_sum: if positive, the component *causes* that topology.")
    print("• RC+ / RC− words: what that component means (opposite ends of the axis).")
    print("• LOO Δ: removing that word collapses loops/voids ⇒ structurally critical.")
    print("• Cycle weights: words that sit on many persistent cycles (witnesses).")
    print("==========================================================\n")




def intrinsic_topology_scores(embeddings_dict, metric="euclidean"):
    from ripser import ripser
    def sum_finite(dgm):
        if dgm.size==0: return 0.0
        m = np.isfinite(dgm[:,1])
        return float(np.sum(dgm[m,1]-dgm[m,0])) if m.any() else 0.0

    rows = []
    for s, X in embeddings_dict.items():
        out = ripser(X, maxdim=2, metric=metric)
        H0, H1, H2 = out["dgms"][0], out["dgms"][1], out["dgms"][2]
        # cohesion from H0
        if H0.size and np.isfinite(H0[:,1]).any():
            p0 = H0[np.isfinite(H0[:,1])]
            p0 = p0[:,1]-p0[:,0]
            h0_median = float(np.median(p0))
        else:
            h0_median = 0.0
        rows.append({
            "symbol": s,
            "H0_cohesion": h0_median,      # lower = tighter
            "H1_sum":      sum_finite(H1), # loopiness
            "H2_sum":      sum_finite(H2)  # voidiness
        })
    df = pd.DataFrame(rows)
    # optional composite (z-scored): more loops/voids, tighter cluster (invert H0)
    for col in ["H1_sum","H2_sum","H0_cohesion"]:
        df[col+"_z"] = (df[col]-df[col].mean())/(df[col].std()+1e-12)
    df["TopoScore"] = df["H1_sum_z"] + df["H2_sum_z"] - df["H0_cohesion_z"]
    return df.sort_values("TopoScore", ascending=False)


import itertools

def pair_synergy(A, B, metric="euclidean"):
    from ripser import ripser
    import numpy as np
    XA = A/ (np.linalg.norm(A,axis=1,keepdims=True)+1e-12)
    XB = B/ (np.linalg.norm(B,axis=1,keepdims=True)+1e-12)
    X  = np.vstack([XA, XB])
    # union PH
    out = ripser(X, maxdim=2, metric=metric)
    H1u, H2u = out["dgms"][1], out["dgms"][2]
    def sumf(dgm):
        if dgm.size==0: return 0.0
        m = np.isfinite(dgm[:,1])
        return float(np.sum(dgm[m,1]-dgm[m,0])) if m.any() else 0.0
    s1u, s2u = sumf(H1u), sumf(H2u)
    # remove cross edges
    D = np.sqrt(((X[:,None,:]-X[None,:,:])**2).sum(-1))
    nA = len(XA); big = 1e9
    for i in range(len(X)):
        for j in range(i+1,len(X)):
            if (i<nA) != (j<nA): D[i,j]=D[j,i]=big
    out2 = ripser(D, maxdim=2, metric="precomputed")
    s1nc, s2nc = sumf(out2["dgms"][1]), sumf(out2["dgms"][2])
    return s1u - s1nc, s2u - s2nc  # synergy_H1, synergy_H2

# def synergy_centrality(embeddings_dict, metric="euclidean"):
#     keys = list(embeddings_dict.keys())
#     S = {k: {"synergy_H1":0.0,"synergy_H2":0.0} for k in keys}
#     for a,b in itertools.combinations(keys,2):
#         h1,h2 = pair_synergy(embeddings_dict[a], embeddings_dict[b], metric=metric)
#         S[a]["synergy_H1"] += h1; S[b]["synergy_H1"] += h1
#         S[a]["synergy_H2"] += h2; S[b]["synergy_H2"] += h2
#     df = pd.DataFrame([{"symbol":k, **v} for k,v in S.items()])
#     # composite “SharedScore”: normalize and sum
#     for col in ["synergy_H1","synergy_H2"]:
#         df[col+"_z"] = (df[col]-df[col].mean())/(df[col].std()+1e-12)
#     df["SharedScore"] = df["synergy_H1_z"] + df["synergy_H2_z"]
#     return df.sort_values("SharedScore", ascending=False)


def top_components_table(results_for_symbol_df, rotated_scores, words, k=3):
    """
    results_for_symbol_df: from amputate_components_and_measure()[symbol]
    rotated_scores: Zr from your rotated_component_view (n_words x n_comps)
    """

    df = results_for_symbol_df.copy()
    df["rank_H1"] = df["ΔH1_sum"].rank(ascending=False, method="first")
    df["rank_H2"] = df["ΔH2_sum"].rank(ascending=False, method="first")
    topH1 = df.sort_values("ΔH1_sum", ascending=False).head(k)
    topH2 = df.sort_values("ΔH2_sum", ascending=False).head(k)

    def top_words_for_comp(c, topn=6):
        s = rotated_scores[:, c-1]  # component indices are 1-based in your df
        idx_pos = np.argsort(s)[-topn:][::-1]
        idx_neg = np.argsort(s)[:topn]
        return [words[i] for i in idx_pos], [words[i] for i in idx_neg]

    rows = []
    for _,r in pd.concat([topH1, topH2]).drop_duplicates(subset=["component"]).iterrows():
        pos, neg = top_words_for_comp(int(r["component"]))
        rows.append({
            "component": int(r["component"]),
            "ΔH1_sum": float(r["ΔH1_sum"]),
            "ΔH2_sum": float(r["ΔH2_sum"]),
            "RC+ words": pos,
            "RC− words": neg
        })
    return pd.DataFrame(rows).sort_values(["ΔH1_sum","ΔH2_sum"], ascending=False)

def loo_word_impact(X, words, metric="euclidean"):
    from ripser import ripser
    def sumf(dgm):
        if dgm.size==0: return 0.0
        m = np.isfinite(dgm[:,1])
        return float(np.sum(dgm[m,1]-dgm[m,0])) if m.any() else 0.0
    out = ripser(X, maxdim=2, metric=metric)
    base_H1, base_H2 = sumf(out["dgms"][1]), sumf(out["dgms"][2])

    impacts = []
    for i in range(len(X)):
        Xi = np.delete(X, i, axis=0)
        outi = ripser(Xi, maxdim=2, metric=metric)
        h1i, h2i = sumf(outi["dgms"][1]), sumf(outi["dgms"][2])
        impacts.append({
            "word": words[i],
            "ΔH1_sum": base_H1 - h1i,
            "ΔH2_sum": base_H2 - h2i
        })

    return pd.DataFrame(impacts).sort_values(["ΔH1_sum","ΔH2_sum"], ascending=False)


def cycle_participation_weights(X, words, metric="euclidean", top_k1=6, top_k2=4):
    from ripser import ripser
    out = ripser(X, maxdim=2, metric=metric, do_cocycles=True)
    H1, H2 = out["dgms"][1], out["dgms"][2]
    C1 = out.get("cocycles",[[],[],[]])[1] if "cocycles" in out else []
    C2 = out.get("cocycles",[[],[],[]])[2] if "cocycles" in out else []
    
    def score_by_vertices(H, C, topk, dim):
        if H.size==0: return np.zeros(len(words))
        pers = np.where(np.isfinite(H[:,1]), H[:,1]-H[:,0], 0.0)
        order = np.argsort(pers)[::-1][:topk]
        w = np.zeros(len(words))
        for idx in order:
            cyc = C[idx] if idx < len(C) else []
            if dim==1:
                verts = set([int(i) for (i,_,_) in cyc] + [int(j) for (_,j,_) in cyc])
            else:
                verts = set()
                for row in cyc:
                    i,j,k = map(int, row[:3]); verts.update([i,j,k])
            for v in verts: w[v] += pers[idx]
        return w
    w1 = score_by_vertices(H1, C1, top_k1, dim=1)
    w2 = score_by_vertices(H2, C2, top_k2, dim=2)
    df = pd.DataFrame({"word": words, "H1_weight": w1, "H2_weight": w2,
                       "Total": w1+w2}).sort_values("Total", ascending=False)
    return df

import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations
from typing import Dict, Tuple, List

# ---- Distance helpers (reuse your metric choice)
def _pairwise_dists(X, metric="euclidean"):
    X = np.asarray(X, float)
    if metric == "euclidean":
        diff = X[:, None, :] - X[None, :, :]
        return np.sqrt(np.sum(diff*diff, axis=2))
    elif metric == "cosine":
        X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
        S = np.clip(X @ X.T, -1.0, 1.0)
        return 1.0 - S
    else:
        raise ValueError("metric must be 'euclidean' or 'cosine'")

def _sum_finite_persistence(dgm):
    if dgm.size == 0: return 0.0
    finite = np.isfinite(dgm[:,1])
    if not finite.any(): return 0.0
    pers = dgm[finite,1] - dgm[finite,0]
    return float(np.sum(pers))

# ---- Build union + labels for a pair of symbols
def _union_with_labels(A: np.ndarray, B: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    A = np.asarray(A, float); B = np.asarray(B, float)
    X = np.vstack([A, B])
    labels = np.zeros(len(X), dtype=int)
    labels[len(A):] = 1
    # row-normalize (defensive)
    X /= (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
    return X, labels

# ---- PH on union and union-without-cross edges (precomputed distances)
def intersymbol_synergy(A: np.ndarray, B: np.ndarray, metric="euclidean"):
    from ripser import ripser
    X, lbl = _union_with_labels(A, B)

    # Union PH (point cloud)
    out = ripser(X, maxdim=2, metric=metric)
    H1u, H2u = out["dgms"][1], out["dgms"][2]
    sumH1_u = _sum_finite_persistence(H1u)
    sumH2_u = _sum_finite_persistence(H2u)

    # Union with cross edges disabled (distance matrix)
    D = _pairwise_dists(X, metric=metric)
    big = 1e9
    for i in range(len(X)):
        for j in range(i+1, len(X)):
            if lbl[i] != lbl[j]:
                D[i,j] = D[j,i] = big
    out_nc = ripser(D, maxdim=2, metric="precomputed")
    H1nc, H2nc = out_nc["dgms"][1], out_nc["dgms"][2]
    sumH1_nc = _sum_finite_persistence(H1nc)
    sumH2_nc = _sum_finite_persistence(H2nc)

    return {
        "sumH1_union": sumH1_u,
        "sumH1_no_cross": sumH1_nc,
        "synergy_H1": sumH1_u - sumH1_nc,
        "sumH2_union": sumH2_u,
        "sumH2_no_cross": sumH2_nc,
        "synergy_H2": sumH2_u - sumH2_nc
    }

# ---- Cycle mixing statistics on A∪B
def _cycle_vertex_sets(cocycle, dim=1):
    verts = set()
    if dim == 1:
        for i,j,c in cocycle:
            verts.add(int(i)); verts.add(int(j))
    elif dim == 2:
        for i,j,k,c in cocycle:
            verts.add(int(i)); verts.add(int(j)); verts.add(int(k))
    return verts

def _mixing_entropy(label_counts: List[int]) -> float:
    # Shannon entropy in bits for 2 labels (cap at 1.0 for two-class case)
    total = sum(label_counts)
    if total == 0: return 0.0
    import math
    H = 0.0
    for c in label_counts:
        if c > 0:
            p = c / total
            H -= p * math.log(p, 2)
    return float(H)

def cycle_mixing_stats(A: np.ndarray, B: np.ndarray, metric="euclidean", k1=5, k2=3):
    """
    Returns top-k H1/H2 cycles' (persistence, mixing_entropy, cross_fraction).
    """
    from ripser import ripser
    X, lbl = _union_with_labels(A, B)
    out = ripser(X, maxdim=2, metric=metric, do_cocycles=True)
    H1, H2 = out["dgms"][1], out["dgms"][2]
    coc1 = out.get("cocycles", [[],[],[]])[1] if "cocycles" in out else []
    coc2 = out.get("cocycles", [[],[],[]])[2] if "cocycles" in out else []

    stats_H1 = []
    if H1.size:
        pers = np.where(np.isfinite(H1[:,1]), H1[:,1]-H1[:,0], 0.0)
        order = np.argsort(pers)[::-1][:k1]
        for idx in order:
            p = float(pers[idx])
            cyc = coc1[idx] if idx < len(coc1) else []
            # vertex participation
            verts = _cycle_vertex_sets(cyc, dim=1)
            la = sum(lbl[v]==0 for v in verts)
            lb = sum(lbl[v]==1 for v in verts)
            Hmix = _mixing_entropy([la, lb])
            # cross-edge fraction
            cross_e = 0; total_e = 0
            for i,j,c in cyc:
                i,j = int(i),int(j); total_e += 1
                if lbl[i] != lbl[j]: cross_e += 1
            frac = (cross_e/total_e) if total_e>0 else 0.0
            stats_H1.append({"pers": p, "mix_entropy": Hmix, "cross_frac": frac})

    stats_H2 = []
    if H2.size:
        pers = np.where(np.isfinite(H2[:,1]), H2[:,1]-H2[:,0], 0.0)
        order = np.argsort(pers)[::-1][:k2]
        for idx in order:
            p = float(pers[idx])
            cyc = coc2[idx] if idx < len(coc2) else []
            verts = _cycle_vertex_sets(cyc, dim=2)
            la = sum(lbl[v]==0 for v in verts)
            lb = sum(lbl[v]==1 for v in verts)
            Hmix = _mixing_entropy([la, lb])
            # triangles spanning both labels
            cross_t = 0; total_t = 0
            for row in cyc:
                i,j,k = map(int, row[:3]); total_t += 1
                labs = {lbl[i], lbl[j], lbl[k]}
                if len(labs) >= 2: cross_t += 1
            frac = (cross_t/total_t) if total_t>0 else 0.0
            stats_H2.append({"pers": p, "mix_entropy": Hmix, "cross_frac": frac})

    return stats_H1, stats_H2


# ---- Convenience wrapper over all unordered pairs in your dict
def quantify_inter_symbol_PH(embeddings_dict: Dict[str, np.ndarray], metric="euclidean"):
    
    rows = []
    keys = list(embeddings_dict.keys())
    for (sa, sb) in itertools.combinations(keys, 2):
        A = embeddings_dict[sa]; B = embeddings_dict[sb]
        syn = intersymbol_synergy(A, B, metric=metric)
        rows.append({"pair": f"{sa}–{sb}", **syn})
    df = pd.DataFrame(rows).sort_values("synergy_H1", ascending=False)
    return df

def plot_cycle_mixing_scatter(A, B, title, metric="euclidean"):
    H1c, H2c = cycle_mixing_stats(A, B, metric=metric, k1=6, k2=4)
    if H1c:
        x = [c["mix_entropy"] for c in H1c]; y = [c["pers"] for c in H1c]
        plt.figure(); plt.scatter(x, y)
        plt.xlabel("H1 mixing entropy (bits)"); plt.ylabel("H1 persistence")
        plt.title(f"{title} — cycle mixing (H1)"); plt.tight_layout(); plt.show()
    if H2c:
        x = [c["mix_entropy"] for c in H2c]; y = [c["pers"] for c in H2c]
        plt.figure(); plt.scatter(x, y)
        plt.xlabel("H2 mixing entropy (bits)"); plt.ylabel("H2 persistence")
        plt.title(f"{title} — cycle mixing (H2)"); plt.tight_layout(); plt.show()


import numpy as np
from typing import List, Dict, Any

def _ph_with_cocycles(X, metric="euclidean"):
    from ripser import ripser
    return ripser(X, maxdim=2, metric=metric, do_cocycles=True)

def _finite_persistence(dgm):
    if dgm.size == 0: 
        return np.zeros((0,)), np.arange(0)
    pers = np.where(np.isfinite(dgm[:,1]), dgm[:,1]-dgm[:,0], 0.0)
    return pers, np.argsort(pers)[::-1]

def _h1_vertices_from_cocycle(cocycle):
    V=set()
    for i,j,c in cocycle: V.update([int(i), int(j)])
    return sorted(V)

def _h2_vertices_from_cocycle(cocycle):
    V=set()
    for row in cocycle:
        i,j,k = map(int, row[:3]); V.update([i,j,k])
    return sorted(V)

def extract_symbol_cycles(symbol: str,
                          X: np.ndarray,
                          words: List[str],
                          metric: str = "euclidean",
                          top_k1: int = 5,
                          top_k2: int = 3) -> Dict[str, List[Dict[str, Any]]]:
    """
    Returns dict with lists of top H1 and H2 cycles; each item has:
      birth, death, persistence, vertex_ids, word_list
    """
    out = _ph_with_cocycles(X, metric=metric)
    H1, H2 = out["dgms"][1], out["dgms"][2]
    coc1 = out.get("cocycles",[[],[],[]])[1] if "cocycles" in out else []
    coc2 = out.get("cocycles",[[],[],[]])[2] if "cocycles" in out else []

    res = {"H1": [], "H2": []}

    pers1, order1 = _finite_persistence(H1)
    for idx in order1[:top_k1]:
        cyc = coc1[idx] if idx < len(coc1) else []
        V = _h1_vertices_from_cocycle(cyc)
        res["H1"].append({
            "birth": float(H1[idx,0]), "death": float(H1[idx,1]),
            "persistence": float(pers1[idx]),
            "vertex_ids": V,
            "words": [words[i] for i in V]
        })

    pers2, order2 = _finite_persistence(H2)
    for idx in order2[:top_k2]:
        cyc = coc2[idx] if idx < len(coc2) else []
        V = _h2_vertices_from_cocycle(cyc)
        res["H2"].append({
            "birth": float(H2[idx,0]), "death": float(H2[idx,1]),
            "persistence": float(pers2[idx]),
            "vertex_ids": V,
            "words": [words[i] for i in V]
        })
    return res


from typing import Tuple

def _union_X_labels_words(A, wordsA, B, wordsB):
    A = np.asarray(A, float); B = np.asarray(B, float)
    X = np.vstack([A, B])
    labels = np.array([0]*len(A) + [1]*len(B), int)
    words  = list(wordsA) + list(wordsB)
    X /= (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
    return X, labels, words

def _mix_label_of_vertices(V, labels):
    labs = labels[V]
    if np.all(labs==0): return "pure_A"
    if np.all(labs==1): return "pure_B"
    return "mixed"

def extract_pair_cycles(
    A: np.ndarray, wordsA: List[str],
    B: np.ndarray, wordsB: List[str],
    metric="euclidean", top_k1=6, top_k2=4
):
    X, lbl, words = _union_X_labels_words(A, wordsA, B, wordsB)
    out = _ph_with_cocycles(X, metric=metric)
    H1, H2 = out["dgms"][1], out["dgms"][2]
    coc1 = out.get("cocycles",[[],[],[]])[1] if "cocycles" in out else []
    coc2 = out.get("cocycles",[[],[],[]])[2] if "cocycles" in out else []

    res = {"H1": [], "H2": []}
    pers1, order1 = _finite_persistence(H1)
    for idx in order1[:top_k1]:
        cyc = coc1[idx] if idx < len(coc1) else []
        V = _h1_vertices_from_cocycle(cyc)
        tag = _mix_label_of_vertices(np.array(V, int), lbl)
        res["H1"].append({
            "birth": float(H1[idx,0]), "death": float(H1[idx,1]),
            "persistence": float(pers1[idx]),
            "mix": tag,
            "vertex_ids": V,
            "words": [words[i] + (" [A]" if lbl[i]==0 else " [B]") for i in V]
        })

    pers2, order2 = _finite_persistence(H2)
    for idx in order2[:top_k2]:
        cyc = coc2[idx] if idx < len(coc2) else []
        V = _h2_vertices_from_cocycle(cyc)
        tag = _mix_label_of_vertices(np.array(V, int), lbl)
        res["H2"].append({
            "birth": float(H2[idx,0]), "death": float(H2[idx,1]),
            "persistence": float(pers2[idx]),
            "mix": tag,
            "vertex_ids": V,
            "words": [words[i] + (" [A]" if lbl[i]==0 else " [B]") for i in V]
        })
    return res

import matplotlib.pyplot as plt

def _pca2d(X):
    Xc = X - X.mean(0, keepdims=True)
    U,S,Vt = np.linalg.svd(Xc, full_matrices=False)
    return Xc @ Vt[:2].T

def plot_labeled_cycle(X, words, vertex_ids, title="Cycle"):
    Z = _pca2d(X)
    V = np.array(vertex_ids, int)
    plt.figure(figsize=(6,5))
    # light background points
    plt.scatter(Z[:,0], Z[:,1], s=14, alpha=0.2)
    # highlight cycle vertices
    plt.scatter(Z[V,0], Z[V,1], s=32)
    # label them
    for i in V:
        plt.annotate(words[i], (Z[i,0], Z[i,1]), textcoords="offset points", xytext=(5,5))
    plt.axis("equal"); plt.title(title); plt.tight_layout(); plt.show()