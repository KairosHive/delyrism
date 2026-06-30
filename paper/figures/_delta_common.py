"""Shared Δ-readout machinery for the application figures.

CALIBRATED READOUT.  The 30-probe oblique battery (fig_delta_probe_atlas.PROBES)
serves as a fixed REFERENCE: the context-generic mean fingerprint and the
per-symbol within-symbol statistics (mean, std) are estimated once on the
reference battery, then applied unchanged to NEW contexts.  This makes every
application figure a deployment of the same calibrated instrument rather than
a freshly re-normalized analysis.

Conventions (same as fig_delta_probe_atlas):
  Δ = D1 D1ᵀ − D0 D0ᵀ  (diag zeroed),  fingerprint = upper triangle of Δ
  res(fp)  = fp − reference mean fingerprint   (context-specific rewiring)
  ws_z(fp) = per-symbol z-score of the within-symbol mean Δ  (de-biased
             symbol response — removes the LIGHTNING/THUNDER magnitude bias)
"""
from __future__ import annotations

import numpy as np

SHIFT_KW = dict(
    strategy="gate", gate="relu", beta=1.2, tau=0.3,
    within_symbol_softmax=False, gamma=0.5,
    pool_type="avg", pool_w=0.7, membership_alpha=0.0,
)


class DeltaReadout:
    """Δ fingerprints + reference-battery calibration for a SymbolSpace."""

    def __init__(self, space, shift_kw=None):
        self.space = space
        self.kw = dict(shift_kw or SHIFT_KW)
        self.descs = space.descriptors
        self.syms = list(space.symbols)
        self.sidx = {s: i for i, s in enumerate(self.syms)}
        owner = space.owner
        self.iu = np.triu_indices(len(self.descs), k=1)
        self.oi = np.array([self.sidx[owner[self.descs[i]]] for i in self.iu[0]])
        self.oj = np.array([self.sidx[owner[self.descs[j]]] for j in self.iu[1]])
        self.same = self.oi == self.oj
        self.C0 = space.D @ space.D.T
        self._ws_masks = [self.same & (self.oi == s) for s in range(len(self.syms))]
        self.mean_fp = None
        self.ws_mu = None
        self.ws_sd = None

    # ── raw fingerprint ──────────────────────────────────────────────────────
    def fp(self, sentence: str) -> np.ndarray:
        D1 = self.space.make_shifted_matrix(sentence=sentence, **self.kw)
        Dl = D1 @ D1.T - self.C0
        np.fill_diagonal(Dl, 0.0)
        return Dl[self.iu]

    def ws(self, raw: np.ndarray) -> np.ndarray:
        """Within-symbol mean Δ per symbol (raw, biased)."""
        return np.array([raw[m].mean() if m.any() else np.nan
                         for m in self._ws_masks])

    # ── calibration on a reference battery ──────────────────────────────────
    def fit_reference(self, phrases) -> np.ndarray:
        """Estimate mean fingerprint + per-symbol WS stats; returns raw fps."""
        R = np.vstack([self.fp(p) for p in phrases])
        self.mean_fp = R.mean(axis=0)
        W = np.vstack([self.ws(r) for r in R])
        self.ws_mu = np.nanmean(W, axis=0)
        self.ws_sd = np.nanstd(W, axis=0) + 1e-12
        return R

    # ── calibrated readouts for new contexts ────────────────────────────────
    def res(self, raw: np.ndarray) -> np.ndarray:
        """Context-specific residual fingerprint."""
        return raw - self.mean_fp

    def ws_z(self, raw: np.ndarray) -> np.ndarray:
        """De-biased per-symbol response (z vs reference battery)."""
        return (self.ws(raw) - self.ws_mu) / self.ws_sd

    def sym_map(self, rvec: np.ndarray) -> np.ndarray:
        """Symbol×symbol mean map of a fingerprint-shaped vector."""
        n = len(self.syms)
        tot = np.zeros((n, n))
        cnt = np.zeros((n, n))
        np.add.at(tot, (self.oi, self.oj), rvec)
        np.add.at(cnt, (self.oi, self.oj), 1.0)
        tot = tot + tot.T
        cnt = cnt + cnt.T
        with np.errstate(invalid="ignore", divide="ignore"):
            M = np.where(cnt > 0, tot / cnt, np.nan)
        np.fill_diagonal(M, np.nan)
        return M


def cos(u: np.ndarray, v: np.ndarray) -> float:
    return float(u @ v / (np.linalg.norm(u) * np.linalg.norm(v) + 1e-12))


def norm_entropy(z: np.ndarray, tau: float = 1.0) -> float:
    """Entropy of softmax(z/tau), normalized to [0, 1] by log(len)."""
    z = np.asarray(z, dtype=float)
    p = np.exp((z - z.max()) / tau)
    p /= p.sum()
    return float(-(p * np.log(p + 1e-12)).sum() / np.log(len(z)))
