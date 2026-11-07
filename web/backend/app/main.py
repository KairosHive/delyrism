"""Compatibility shim — the FastAPI app is `delyrism.server.main:app`.

    python -m delyrism.server --port 8000          (preferred)
    uvicorn app.main:app --port 8000 --app-dir web/backend
"""
import os
import sys

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if _ROOT not in sys.path:          # repo checkout without `pip install -e .`
    sys.path.insert(0, _ROOT)

from delyrism.server.main import app  # noqa: E402,F401
