"""Driver — run every paper figure script in order.

The space is cached on disk by `_setup.build_space`, so only the first script
pays the embedding cost; subsequent scripts hit the cache.

Usage:
    python paper/figures/make_all.py
    python paper/figures/make_all.py --skip fig05  # skip the cross-modal figure
"""
from __future__ import annotations

import argparse
import importlib
import sys
import time
from pathlib import Path

# Make the directory importable
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

FIGURES = [
    "fig_v1_umap",
    "fig_v1_ambiguity",
    "fig_v1_attention",
    "fig_v1_ppr",
    "fig_v1_starplots",
    "fig01_delta_graph",
    "fig02_topology",
    "fig03_phase_morphing",
    "fig04_catalysts",
    "fig05_crossmodal",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip", nargs="*", default=[],
                    help="Module names to skip (e.g. fig05_crossmodal).")
    ap.add_argument("--only", nargs="*", default=None,
                    help="Run only these (overrides default order).")
    ap.add_argument("--continue-on-error", action="store_true",
                    help="Keep going if one script fails.")
    args = ap.parse_args()

    to_run = args.only if args.only else FIGURES
    to_run = [m for m in to_run if m not in args.skip]

    total_t0 = time.time()
    for mod_name in to_run:
        print(f"\n{'='*72}\n>>> {mod_name}\n{'='*72}")
        t0 = time.time()
        try:
            mod = importlib.import_module(mod_name)
            # Reset sys.argv so each module's argparse sees no extra flags
            saved_argv = sys.argv
            sys.argv = [mod_name]
            try:
                mod.main()
            finally:
                sys.argv = saved_argv
        except SystemExit as e:
            print(f"!!! {mod_name} exited: {e}")
            if not args.continue_on_error:
                raise
        except Exception as e:
            print(f"!!! {mod_name} failed: {e!r}")
            if not args.continue_on_error:
                raise
        print(f"--- {mod_name} done in {time.time() - t0:.1f}s")

    print(f"\n[make_all] all done in {time.time() - total_t0:.1f}s")


if __name__ == "__main__":
    main()
