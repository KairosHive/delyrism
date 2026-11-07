"""Live session setup — one call from symbols to a ready live space.

    GET  /live/profile          the current LiveProfile (profile.py)
    POST /live/profile          replace it (also pushes its OSC / speech settings)
    POST /live/setup            {symbols, profile?} → build (or reuse) the space with
                                the profile's embedder, apply its global mode → space info

/live/setup is idempotent: the same symbols and profile return the same
space_id (the engine cache keys spaces on symbols + embedder + threshold), so
a host can call it whenever it (re)activates a session — e.g. after the live
socket closed with 4404 because this server restarted.  `profile`, when
given, is merged into the current profile and becomes current, since live
sockets read their defaults from it.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from .. import engine_cache
from .. import profile as live_profile
from ..schemas import SymbolMap
from ..util import color_map_to_hex

router = APIRouter(prefix="/live", tags=["live"])


class SetupRequest(BaseModel):
    symbols: SymbolMap
    profile: Optional[Dict[str, Any]] = None     # partial LiveProfile, merged into the current one


@router.get("/profile")
def get_profile() -> Dict[str, Any]:
    return live_profile.current().model_dump()


@router.post("/profile")
def set_profile(body: Dict[str, Any]) -> Dict[str, Any]:
    try:
        return live_profile.apply(body).model_dump()
    except Exception as e:
        raise HTTPException(status_code=422, detail=str(e))


def _merge(base: Dict[str, Any], patch: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(base)
    for k, v in patch.items():
        out[k] = _merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


@router.post("/setup")
def setup(req: SetupRequest) -> Dict[str, Any]:
    symbols = {str(k): [str(d) for d in v if str(d).strip()] for k, v in req.symbols.items()}
    symbols = {k: v for k, v in symbols.items() if v}
    if not symbols:
        raise HTTPException(status_code=400, detail="symbols may not be empty (each symbol needs descriptors)")
    prof = live_profile.current()
    if req.profile:
        try:
            prof = live_profile.apply(_merge(prof.model_dump(), req.profile))
        except Exception as e:
            raise HTTPException(status_code=422, detail=f"profile: {e}")
    try:
        space_id, space = engine_cache.get_or_build_space(
            symbols=symbols,
            embedder_cfg=prof.embedder.model_dump(),
            descriptor_threshold=prof.descriptor_threshold,
            contextual_embeddings=prof.contextual_embeddings,
            palette=prof.palette,
        )
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"could not build space with embedder '{prof.embedder.backend}': {e}")
    info = space.global_mode_info()
    if (info.get("mode"), info.get("pcs")) != (prof.global_mode, prof.global_pcs):
        space.set_global_mode(prof.global_mode, prof.global_pcs)
        engine_cache.invalidate_results(space_id)
        info = space.global_mode_info()
    return {
        "space_id": space_id,
        "symbols": list(space.symbols),
        "descriptors": list(space.descriptors),
        "owners": dict(space.owner),
        "embedding_dim": int(space.D.shape[1]),
        "color_map": color_map_to_hex(space.get_symbol_color_dict(palette=prof.palette)),
        "global_mode": {"mode": info.get("mode"), "pcs": info.get("pcs")},
        "profile": prof.model_dump(),
    }
