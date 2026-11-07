"""delyrism.server — the FastAPI backend: engine routes, the live Δ-graph
socket, OSC to TouchDesigner, the speech client and the live profile.

    python -m delyrism.server --port 8000 --profile live.json --warmup
    from delyrism.server import create_app

See docs/LIVE_API.md.
"""


def __getattr__(name):
    # lazy, so `import delyrism.server.profile` doesn't build the default app
    if name in ("create_app", "app", "warmup"):
        from . import main
        return getattr(main, name)
    raise AttributeError(name)
