"""Shared fixtures: the real FastAPI app with the hashing stand-in embedder
(delyrism/server/testing.py), so the suite needs no network and no models."""
from __future__ import annotations

import os

os.environ["DELYRISM_TEST_EMBEDDER"] = "1"      # before the server modules load
os.environ.setdefault("DELYRISM_OSC", "0")
os.environ.setdefault("DELYRISM_SPEAK", "0")

import pytest
from fastapi.testclient import TestClient

from delyrism.server import engine_cache
from delyrism.server.main import create_app

PRESET = "elements"
CONTEXT = "flooding spirits dancing around floating suns"


@pytest.fixture(scope="session")
def client() -> TestClient:
    with TestClient(create_app(static_dir=None)) as c:
        yield c


def build_body(symbols: dict) -> dict:
    return {"symbols": symbols, "embedder": {"backend": "cloudflare-bge-m3"},
            "descriptor_threshold": 0.2, "palette": "AuroraPop"}


@pytest.fixture(scope="session")
def symbols(client) -> dict:
    r = client.get(f"/spaces/presets/{PRESET}")
    assert r.status_code == 200, r.text
    return r.json()["symbols"]


@pytest.fixture(scope="session")
def space(client, symbols) -> dict:
    """The /spaces response for the Four Elements preset."""
    r = client.post("/spaces", json=build_body(symbols))
    assert r.status_code == 200, r.text
    return r.json()


@pytest.fixture()
def forget_spaces():
    """Simulate a server restart: drop every built space (and its results)."""
    def _forget():
        for sid in list(engine_cache._space_cache):
            engine_cache.invalidate_space(sid)
    return _forget
