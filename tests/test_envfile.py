"""The repo's `.env` (delyrism/envfile.py): parsing, precedence over the
process environment, and the credential lookup that reads it."""
from __future__ import annotations

import os

import pytest

from delyrism import envfile
from delyrism.server import symbolgen


@pytest.fixture(autouse=True)
def restore_environ():
    """envfile.load() writes os.environ directly, outside monkeypatch's
    bookkeeping.  Restore the whole environment after each test, so a dummy
    CLOUDFLARE_API_TOKEN cannot reach later tests (the session story route
    calls Cloudflare whenever a token is set)."""
    saved = dict(os.environ)
    yield
    os.environ.clear()
    os.environ.update(saved)


def test_parse_handles_comments_quotes_and_export():
    text = "\n".join([
        "# a comment",
        "",
        "PLAIN=abc",
        "export EXPORTED=def",
        'DOUBLE="has # hash"',
        "SINGLE='x y'",
        "INLINE=val # trailing note",
        "EMPTY=",
        "no equals sign",
        "BAD-KEY=1",
    ])
    assert envfile.parse(text) == {
        "PLAIN": "abc", "EXPORTED": "def", "DOUBLE": "has # hash",
        "SINGLE": "x y", "INLINE": "val", "EMPTY": "",
    }


def test_load_sets_missing_and_keeps_existing(tmp_path, monkeypatch):
    f = tmp_path / ".env"
    # PowerShell 5's Set-Content -Encoding utf8 writes a BOM
    f.write_text("﻿DLY_A=from-file\nDLY_B=from-file\nDLY_C=\n", encoding="utf-8")
    monkeypatch.delenv("DLY_A", raising=False)
    monkeypatch.setenv("DLY_B", "from-process")
    monkeypatch.delenv("DLY_C", raising=False)
    assert envfile.load(f) == ["DLY_A"]
    assert envfile.os.environ["DLY_A"] == "from-file"
    assert envfile.os.environ["DLY_B"] == "from-process"
    assert "DLY_C" not in envfile.os.environ


def test_load_missing_file_sets_nothing(tmp_path):
    assert envfile.load(tmp_path / "absent.env") == []


def test_default_path_follows_override(tmp_path, monkeypatch):
    monkeypatch.setenv("DELYRISM_ENV_FILE", str(tmp_path / "x.env"))
    assert envfile.default_path() == tmp_path / "x.env"
    monkeypatch.delenv("DELYRISM_ENV_FILE")
    assert envfile.default_path() == envfile.REPO_ROOT / ".env"


def test_symbolgen_secret_reads_env_file(tmp_path, monkeypatch):
    f = tmp_path / ".env"
    f.write_text("CLOUDFLARE_API_TOKEN=dummy-token\n", encoding="utf-8")
    monkeypatch.setenv("DELYRISM_ENV_FILE", str(f))
    monkeypatch.delenv("CLOUDFLARE_API_TOKEN", raising=False)
    assert symbolgen.secret("CLOUDFLARE_API_TOKEN") == "dummy-token"
