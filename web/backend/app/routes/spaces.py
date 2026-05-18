"""Space lifecycle: list presets, create / inspect a SymbolSpace."""
from __future__ import annotations
from fastapi import APIRouter, HTTPException

from ..schemas import SpaceConfig, SpaceCreateResponse
from .. import engine_cache, presets
from ..util import color_map_to_hex

router = APIRouter(prefix="/spaces", tags=["spaces"])


@router.get("/presets")
def get_presets() -> dict:
    """List the JSON presets under delyrism/structures/."""
    return {"presets": presets.list_presets()}


@router.get("/presets/{name}")
def get_preset(name: str) -> dict:
    try:
        return {"name": name, "symbols": presets.load_preset(name)}
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.post("", response_model=SpaceCreateResponse)
def create_space(cfg: SpaceConfig) -> SpaceCreateResponse:
    """Build (or reuse a cached) SymbolSpace. Returns a stable space_id used by
    every other endpoint to refer back to this instance."""
    if not cfg.symbols:
        raise HTTPException(status_code=400, detail="symbols may not be empty")

    space_id, space = engine_cache.get_or_build_space(
        symbols=cfg.symbols,
        embedder_cfg=cfg.embedder.model_dump(),
        descriptor_threshold=cfg.descriptor_threshold,
        contextual_embeddings=cfg.contextual_embeddings,
        palette=cfg.palette,
    )
    return SpaceCreateResponse(
        space_id=space_id,
        symbols=space.symbols,
        descriptors=space.descriptors,
        owners=space.owner,
        embedding_dim=int(space.D.shape[1]),
        color_map=color_map_to_hex(space.get_symbol_color_dict(palette=cfg.palette)),
    )


@router.get("/{space_id}")
def get_space_info(space_id: str, palette: str = "AuroraPop") -> dict:
    space = engine_cache.get_space(space_id)
    if space is None:
        raise HTTPException(status_code=404, detail="unknown space_id")
    return {
        "space_id": space_id,
        "symbols": space.symbols,
        "descriptors": space.descriptors,
        "owners": space.owner,
        "embedding_dim": int(space.D.shape[1]),
        "color_map": color_map_to_hex(space.get_symbol_color_dict(palette=palette)),
    }


@router.get("/cache/stats")
def stats() -> dict:
    return engine_cache.cache_stats()
