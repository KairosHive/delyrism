"""Run the Delyrism API.

    python -m delyrism.server [--host 127.0.0.1] [--port 8000]
                              [--profile live.json] [--warmup]
                              [--osc-host 127.0.0.1] [--osc-port 7000] [--osc-hz 60]
                              [--tts-url http://127.0.0.1:8010] [--static DIR]
                              [--max-embedders N]

Also installed as the `delyrism-server` command.  Flags override the profile.
GET /healthz -> {"ok": true, "ready": ...} for process managers.
"""
from __future__ import annotations

import argparse
import os
import sys


def main(argv=None) -> None:
    # a console or pipe on Windows may be cp1252: never let a log line crash the server
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    ap = argparse.ArgumentParser(prog="delyrism-server", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default=os.environ.get("DELYRISM_HOST", "127.0.0.1"))
    ap.add_argument("--port", type=int, default=int(os.environ.get("DELYRISM_PORT", "8000")))
    ap.add_argument("--profile", default=os.environ.get("DELYRISM_PROFILE"),
                    help="LiveProfile JSON: live defaults, embedder, global mode, OSC, speech")
    ap.add_argument("--warmup", action="store_true",
                    help="load the profile's embedder at start (/healthz ready=false until done)")
    ap.add_argument("--osc-host")
    ap.add_argument("--osc-port", type=int)
    ap.add_argument("--osc-hz", type=float)
    ap.add_argument("--tts-url", help="TTS service for the spoken word (default http://127.0.0.1:8010)")
    ap.add_argument("--static", help="serve this static frontend at /")
    ap.add_argument("--max-embedders", type=int,
                    help="keep at most N embedders loaded; loading another evicts the least recently used "
                         "with its spaces (env DELYRISM_MAX_EMBEDDERS, default 0 = no limit)")
    ap.add_argument("--log-level", default="info")
    a = ap.parse_args(argv)

    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    if a.max_embedders is not None:
        os.environ["DELYRISM_MAX_EMBEDDERS"] = str(max(0, a.max_embedders))
    import uvicorn
    from . import profile as live_profile
    from .main import create_app

    prof = live_profile.load(a.profile) if a.profile else live_profile.current()
    osc = prof.osc.model_dump() if prof.osc else {}
    osc.update({k: v for k, v in (("host", a.osc_host), ("port", a.osc_port), ("hz", a.osc_hz)) if v is not None})
    speech = prof.speech.model_dump() if prof.speech else {}
    if a.tts_url:
        speech["url"] = a.tts_url
    prof = live_profile.LiveProfile.model_validate({**prof.model_dump(), "osc": osc or None, "speech": speech or None})

    app = create_app(profile=prof, static_dir=a.static, warm=a.warmup)
    uvicorn.run(app, host=a.host, port=a.port, log_level=a.log_level, ws_ping_interval=20, ws_ping_timeout=20)


if __name__ == "__main__":
    main()
