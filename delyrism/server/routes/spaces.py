"""Space lifecycle: list presets, create / inspect a SymbolSpace."""
from __future__ import annotations
from fastapi import APIRouter, HTTPException

from ..schemas import SpaceConfig, SpaceCreateResponse
from .. import engine_cache, presets
from ..util import color_map_to_hex

router = APIRouter(prefix="/spaces", tags=["spaces"])


@router.get("/presets")
def get_presets() -> dict:
    """List the JSON presets under delyrism/structures/ (each with its drawings: GET /spaces/presets/{name})."""
    return {"presets": presets.list_presets()}


@router.get("/presets/{name}")
def get_preset(name: str) -> dict:
    try:
        return {"name": name, "symbols": presets.load_preset(name), "art": presets.load_art(name)}
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.post("", response_model=SpaceCreateResponse)
def create_space(cfg: SpaceConfig) -> SpaceCreateResponse:
    """Build (or reuse a cached) SymbolSpace. Returns a stable space_id used by
    every other endpoint to refer back to this instance."""
    if not cfg.symbols:
        raise HTTPException(status_code=400, detail="symbols may not be empty")

    try:
        space_id, space = engine_cache.get_or_build_space(
            symbols=cfg.symbols,
            embedder_cfg=cfg.embedder.model_dump(),
            descriptor_threshold=cfg.descriptor_threshold,
            contextual_embeddings=cfg.contextual_embeddings,
            palette=cfg.palette,
        )
    except Exception as e:  # embedder / model failures → readable 502, not a bare 500
        # A raw 500 escapes the CORS middleware, so the browser reports it as a
        # CORS error and the real cause (e.g. a 401 from Cloudflare) is hidden.
        raise HTTPException(
            status_code=502,
            detail=f"could not build space with embedder '{cfg.embedder.backend}': {e}",
        )
    return SpaceCreateResponse(
        space_id=space_id,
        symbols=space.symbols,
        descriptors=space.descriptors,
        owners=space.owner,
        embedding_dim=int(space.D.shape[1]),
        color_map=color_map_to_hex(space.get_symbol_color_dict(palette=cfg.palette)),
        modalities=engine_cache.modalities(space.embedder),
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
        "modalities": engine_cache.modalities(space.embedder),
    }


@router.get("/{space_id}/map")
def get_space_map(space_id: str, palette: str = "AuroraPop") -> dict:
    """The space for a session report (delyrism/session.py): every descriptor's vector and its
    place on the 2D map (UMAP when it fits, else the space's PCA), with owners and colours."""
    import base64
    import numpy as np
    space = engine_cache.get_space(space_id)
    if space is None:
        raise HTTPException(status_code=404, detail="unknown space_id")
    D = np.asarray(space.D, dtype="<f4")
    try:
        _model, XY, _ = engine_cache.get_umap_layout(space)
        layout = "umap"
    except Exception:
        XY, layout = space._pca.transform(space.D)[:, :2], "pca"
    XY = np.asarray(XY, dtype=float)[:, :2]
    return {
        "space_id": space_id, "symbols": list(space.symbols), "descriptors": list(space.descriptors),
        "owners": dict(space.owner), "color_map": color_map_to_hex(space.get_symbol_color_dict(palette=palette)),
        "dim": int(D.shape[1]), "n": int(D.shape[0]),
        "D": base64.b64encode(np.ascontiguousarray(D).tobytes()).decode("ascii"),   # float32 LE, row-major n x dim
        "xy": [[round(float(x), 5), round(float(y), 5)] for x, y in XY], "layout": layout,
    }


@router.get("/cache/stats")
def stats() -> dict:
    return engine_cache.cache_stats()
