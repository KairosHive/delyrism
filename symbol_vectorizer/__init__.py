"""
symbol_vectorizer — hand-drawn symbols (photos or scans) to vector symbols,
and symbols + descriptors to the space JSON for live Delyrism.

    python -m symbol_vectorizer vectorize new-symbols -o symbols_out
    python -m symbol_vectorizer space symbols_out descriptors.json -o space.json

Standalone: depends on numpy, scipy, networkx, pillow and
opencv-contrib-python (for thinning), and on nothing in delyrism.
"""
from .core import Params, TraceParams, vectorize_array, vectorize_folder, vectorize_image
from .segment import SegmentParams, segment
from .space import build_space, load_descriptors

__all__ = ["Params", "SegmentParams", "TraceParams", "segment", "vectorize_array", "vectorize_image",
           "vectorize_folder", "build_space", "load_descriptors"]
