"""Per-project secrets: a `.env` file at the repo root.

API credentials (CLOUDFLARE_ACCOUNT_ID, CLOUDFLARE_API_TOKEN, GEMINI_API_KEY, …)
live in `<repo>/.env`, which git and docker ignore.  Keeping them out of the
Windows user environment matters: every program on the machine inherits that
environment, and other projects' tools (wrangler, for one) then pick up this
project's Cloudflare token in place of their own login.

`load()` runs when the `delyrism` package is imported.  It copies the file's
KEY=VALUE pairs into `os.environ` without overriding variables the process
already has, so a deployment's own environment (Railway, `docker run -e`)
wins.  Values are never printed or logged.

    DELYRISM_ENV_FILE   read this file in place of <repo>/.env
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Optional, Union

REPO_ROOT = Path(__file__).resolve().parent.parent


def default_path() -> Path:
    override = os.environ.get("DELYRISM_ENV_FILE", "").strip()
    return Path(override).expanduser() if override else REPO_ROOT / ".env"


def parse(text: str) -> Dict[str, str]:
    """KEY=VALUE lines.  Blank lines and `#` comments are skipped, an
    `export ` prefix is allowed, matching single or double quotes around a
    value are removed, and an unquoted value ends at ` #`."""
    out: Dict[str, str] = {}
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export "):].lstrip()
        key, sep, val = line.partition("=")
        key = key.strip()
        if not sep or not key or not key.replace("_", "").isalnum():
            continue
        val = val.strip()
        if len(val) >= 2 and val[0] == val[-1] and val[0] in "\"'":
            val = val[1:-1]
        else:
            cut = val.find(" #")
            if cut >= 0:
                val = val[:cut].rstrip()
        out[key] = val
    return out


def load(path: Optional[Union[str, Path]] = None, override: bool = False) -> List[str]:
    """Copy the `.env` pairs into os.environ.  Returns the names it set,
    never the values.  A missing or unreadable file sets nothing."""
    p = Path(path).expanduser() if path else default_path()
    try:
        text = p.read_text(encoding="utf-8-sig")  # PowerShell 5 writes a BOM
    except OSError:
        return []
    names: List[str] = []
    for key, val in parse(text).items():
        if val and (override or not os.environ.get(key, "").strip()):
            os.environ[key] = val
            names.append(key)
    return names
