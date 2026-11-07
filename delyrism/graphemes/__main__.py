"""python -m delyrism.graphemes {gui, learn, symbols}"""
import argparse
import sys

from . import basis, gui


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "gui":
        ap = argparse.ArgumentParser(prog="python -m delyrism.graphemes gui",
                                     description="Build and open the grapheme page.")
        ap.add_argument("--drawings", help="folder of drawing images to show (default: the bundled samples)")
        ap.add_argument("--out", help="where to write the page (default: ~/.cache/delyrism/graphemes.html)")
        ap.add_argument("--no-open", action="store_true", help="write the page without opening a browser")
        a = ap.parse_args(argv[1:])
        gui.run(a.out, a.drawings, not a.no_open)
        return
    basis.main(argv)


if __name__ == "__main__":
    main()
