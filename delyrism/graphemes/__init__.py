"""
delyrism.graphemes — symbols and movement in one orthogonal basis of motion
===========================================================================

Any symbol, drawn, and any movement, from pose estimation or an IMU, is cut
into fragments and written in six scale-free motion graphemes (an
orthonormal Symlet-6 wavelet packet basis learned on Omniglot).  The
grapheme distribution of a movement then weighs any set of symbols,
without language.

  basis      GraphemeBasis, motion paths, optimal cutting, encoding
  glyphs     symbols as pen motion: Omniglot, symbol fonts, drawing images
  bridge     pose / IMU → path → graphemes → symbol weights; test alphabets
  datasets   UTD-MHAD, EMOKINE and AIST++ loaders
  gui        a local page to draw and watch the graphemes

    from delyrism.graphemes import default_basis, SymbolBridge
    B = default_basis()
    br = SymbolBridge()                       # ten visual symbols
    br.weights(path_2d)                       # {"circle": 0.71, ...}

    python -m delyrism.graphemes gui          # draw and see the graphemes live
"""
from .basis import (CUT_COST, DEFAULT_PATH, FS, L, GraphemeBasis, default_basis, fragment_signal, fragments,
                    motion, segment_optimal)
from .bridge import (SYMBOLS10, SymbolBridge, invented_symbols, path_from_imu, path_from_pose, path_from_quat,
                     symbol_drawings)
from .glyphs import Drawing, font_drawings, image_drawing, load_omniglot, structure_drawings

__all__ = ["GraphemeBasis", "default_basis", "motion", "fragments", "fragment_signal", "segment_optimal",
           "CUT_COST", "DEFAULT_PATH", "FS", "L", "SymbolBridge", "SYMBOLS10", "invented_symbols",
           "path_from_pose", "path_from_imu", "path_from_quat", "symbol_drawings", "Drawing",
           "load_omniglot", "font_drawings", "structure_drawings", "image_drawing"]
