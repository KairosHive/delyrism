"""Shared setup for paper-figure scripts.

Provides:
  • build_space()        — construct + disk-cache the SymbolSpace
  • OUTDIR, FIGDIR       — output paths
  • set_paper_style()    — publication matplotlib style
  • save_fig(fig, stem)  — save PDF + PNG into OUTDIR
  • Contexts re-exported from lakota_descriptors

Pickle cache key: (sha1 of descriptors dict, embedder backend, model name).
Delete paper/figures/.cache/ to force a rebuild.
"""
from __future__ import annotations

import hashlib
import json
import os
import pickle
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np

# Windows consoles default to cp1252; force UTF-8 so Unicode in log lines
# (arrows, en-dashes, Δ, α, etc.) doesn't crash the script.
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except (AttributeError, OSError):
    pass

# Make the project root importable when scripts are run as `python paper/figures/figXX.py`
_HERE = Path(__file__).resolve().parent
_PROJECT_ROOT = _HERE.parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from lakota_descriptors import SYMBOLS_TO_DESCRIPTORS, CONTEXTS, CONTEXT_LABELS  # noqa: E402

# Output directories
OUTDIR = _HERE.parent / "v2" / "figures"
OUTDIR.mkdir(parents=True, exist_ok=True)
CACHEDIR = _HERE / ".cache"
CACHEDIR.mkdir(exist_ok=True)


def _cache_key(symbols_to_descriptors: dict, backend: str, model: str | None, pooling: str) -> str:
    payload = json.dumps(
        {"d": symbols_to_descriptors, "b": backend, "m": model, "p": pooling},
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha1(payload).hexdigest()[:16]


def build_space(
    *,
    backend: str = "qwen3",
    model: Optional[str] = None,
    pooling: str = "eos",
    descriptors: Optional[dict] = None,
    use_cache: bool = True,
    verbose: bool = True,
):
    """Construct a SymbolSpace (or load from cache).

    Returns
    -------
    SymbolSpace
        With `.D`, `.symbols`, `.symbol_to_idx`, `.symbol_centroids` set.
    """
    from delyrism.delyrism import SymbolSpace, TextEmbedder

    descriptors = descriptors or SYMBOLS_TO_DESCRIPTORS
    key = _cache_key(descriptors, backend, model, pooling)
    cache_path = CACHEDIR / f"space_{key}.pkl"

    if use_cache and cache_path.exists():
        if verbose:
            print(f"[setup] loading cached space from {cache_path.name}")
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    if verbose:
        print(f"[setup] building space: backend={backend} model={model} pooling={pooling}")
        print(f"[setup]   {len(descriptors)} symbols, "
              f"{sum(len(v) for v in descriptors.values())} descriptors total")
    t0 = time.time()
    embedder = TextEmbedder(backend=backend, model=model, pooling=pooling)
    space = SymbolSpace(symbols_to_descriptors=descriptors, embedder=embedder)
    if verbose:
        print(f"[setup]   built in {time.time() - t0:.1f}s")

    if use_cache:
        with open(cache_path, "wb") as f:
            pickle.dump(space, f)
        if verbose:
            print(f"[setup]   cached to {cache_path.name}")

    return space


# ─── matplotlib style ────────────────────────────────────────────────────────

def set_paper_style():
    """Publication-style matplotlib defaults — serif, tight, 300 DPI on save."""
    import matplotlib
    matplotlib.use("Agg")  # headless-safe; comment out for interactive use
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "font.size": 9,
        "axes.labelsize": 9,
        "axes.titlesize": 10,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "axes.linewidth": 0.7,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "lines.linewidth": 1.2,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.04,
        "figure.dpi": 110,
    })


def save_fig(fig, stem: str, *, also_png: bool = True, verbose: bool = True) -> Path:
    """Save figure as PDF (paper-ready) and optionally PNG (for preview).

    Returns the PDF path.
    """
    pdf = OUTDIR / f"{stem}.pdf"
    fig.savefig(pdf)
    if also_png:
        fig.savefig(OUTDIR / f"{stem}.png", dpi=200)
    if verbose:
        print(f"[setup] saved {pdf.name}" + (" (+png)" if also_png else ""))
    return pdf


# Re-exports so figure scripts can `from _setup import *`
__all__ = [
    "build_space", "set_paper_style", "save_fig",
    "OUTDIR", "CACHEDIR",
    "SYMBOLS_TO_DESCRIPTORS", "CONTEXTS", "CONTEXT_LABELS",
    "np",
]
