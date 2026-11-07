"""Story generator — wraps Cloudflare Workers AI (default) and the delta-graph
motif extractor used by the existing Streamlit app."""
from __future__ import annotations
import os
import numpy as np
import requests
from typing import List, Optional, Sequence, Tuple
from fastapi import APIRouter, HTTPException

from ..schemas import StoryRequest, StoryResponse, DeltaGraphRequest
from .. import engine_cache
from ..tone_presets import build_tone_extras, build_form_directive
from delyrism import context_delta_graph
from .topology import (
    _get_ph as _topology_get_ph,
    _symbol_embeddings_from as _topology_symbol_embeddings_from,
    walk_h1_cycle as _topology_walk_h1_cycle,
    _row_norm as _topology_row_norm,
)

router = APIRouter(prefix="/story", tags=["story"])

# Reasonable presets — same identifier strings the Streamlit app uses
CLOUDFLARE_MODELS = {
    "Llama 3.3 70B Fast": "@cf/meta/llama-3.3-70b-instruct-fp8-fast",
    "Llama 4 Scout 17B": "@cf/meta/llama-4-scout-17b-16e-instruct",
    "Mistral Small 3.1 24B": "@cf/mistralai/mistral-small-3.1-24b-instruct",
}


@router.get("/models")
def list_models() -> dict:
    return {"cloudflare": CLOUDFLARE_MODELS}


def _top_motifs(G, *, k_nodes: int = 10, positive_only: bool = True) -> List[str]:
    if G is None or G.number_of_edges() == 0:
        return []
    edges = []
    for u, v, d in G.edges(data=True):
        delta = float(d.get("delta", 0.0))
        if positive_only and delta <= 0:
            continue
        edges.append((u, v, abs(delta)))
    edges.sort(key=lambda x: x[2], reverse=True)
    picked: List[str] = []
    seen: set = set()
    for u, v, _ in edges:
        for n in (u, v):
            if n not in seen:
                seen.add(n)
                picked.append(n)
                if len(picked) >= k_nodes:
                    return picked
    return picked


def _resolve_anchor(space, req: StoryRequest) -> Optional[str]:
    """Translate the frontend's anchor field into an actual symbol name.
    None / ""  → no anchor
    "auto"     → top-ranked symbol from propose() under the current context
    "EARTH"…   → that explicit symbol (validated against space.symbols)
    """
    a = (req.anchor_archetype or "").strip()
    if not a:
        return None
    if a == "auto":
        rows = space.propose(
            weights=req.weights,
            sentence=req.sentence,
            topk=1, tau=0.3, lam=0.6, alpha=0.85, use_ppr=True,
        )
        return rows[0][0] if rows else None
    if a in space.symbol_to_idx:
        return a
    return None


def _top_attention_motifs(space, *, sentence, weights, k: int, anchor: Optional[str]) -> List[str]:
    """Pick motifs by top descriptor-attention under the current context.

    If `anchor` is set: rank that symbol's descriptors by attention and take
    the top k.  Otherwise: spread across the top-3 ranked symbols, taking
    the strongest descriptors from each.  Different signal than the Δ-graph
    motifs (which surface descriptors whose pair-relations CHANGED most) —
    here we want the descriptors most strongly aligned with the context.
    """
    if anchor and anchor in space.symbol_to_idx:
        _, att = space.conditioned_symbol(anchor, weights=weights, sentence=sentence, tau=0.3)
        return [d for d, _ in sorted(att.items(), key=lambda kv: kv[1], reverse=True)[:k]]
    rows = space.propose(
        weights=weights, sentence=sentence,
        topk=min(3, len(space.symbols)),
        tau=0.3, lam=0.6, alpha=0.85, use_ppr=True,
    )
    out: List[str] = []
    seen: set = set()
    per_sym = max(2, (k + len(rows) - 1) // max(1, len(rows)))
    for sym, _, _, _ in rows:
        _, att = space.conditioned_symbol(sym, weights=weights, sentence=sentence, tau=0.3)
        for d, _ in sorted(att.items(), key=lambda kv: kv[1], reverse=True)[:per_sym]:
            if d not in seen and len(out) < k:
                out.append(d)
                seen.add(d)
        if len(out) >= k:
            break
    return out[:k]


# ─────────────────── topology-driven motif sources ───────────────────────

def _shift_params_from_req(req: StoryRequest) -> dict:
    """Standard shift parameter pack derived from the request.  Honors the
    user's Δ-graph sidebar settings (carried in delta_params) so the
    transformation / cycle stories reflect the same context-shifted cloud
    the rest of Explorer is showing."""
    dp = req.delta_params or DeltaGraphRequest(space_id=req.space_id)
    return {
        "weights": req.weights,
        "sentence": req.sentence,
        "strategy": dp.strategy,
        "beta": dp.beta,
        "gate": dp.gate,
        "tau": dp.tau,
        "within_symbol_softmax": dp.within_symbol_softmax,
        "gamma": dp.gamma,
        "prompt_template": dp.prompt_template,
        "pool_type": dp.pool_type,
        "pool_w": dp.pool_w,
        "membership_alpha": dp.membership_alpha,
    }


def _transformation_motifs(
    space, req: StoryRequest, target: Optional[str], density: int,
) -> Tuple[List[str], str, str]:
    """Return (motifs, prompt_context_block, resolved_target).

    When `target` is None, auto-picks the most-transformed archetype
    (max number of new/faded descriptors entering/leaving its top-K).
    Mode (emergence / fading / becoming) drives which slice of the
    identity card becomes the motifs and shapes the prompt narration.
    """
    shift_params = _shift_params_from_req(req)
    D_after = engine_cache.get_or_compute_shifted_matrix(req.space_id, space, shift_params)

    symbols = list(space.symbols)
    cents = np.stack([space.symbol_centroids[s] for s in symbols])
    cents = cents / (np.linalg.norm(cents, axis=1, keepdims=True) + 1e-9)
    sims_before = space.D @ cents.T
    sims_after_to_orig = D_after @ cents.T  # only used for the auto-pick metric

    descriptor_names = list(space.descriptors)
    owners = [space.owner.get(d, "") for d in descriptor_names]

    K_topk = 8  # how many in the before/after lists to compute deltas from

    def _identity_card(sym: str):
        s_idx = symbols.index(sym)
        # before: top-K against original centroid (any descriptor)
        before_order = np.argsort(-sims_before[:, s_idx])
        seen_b: set = set()
        before_list: list = []
        for j in before_order:
            idx = int(j)
            name = descriptor_names[idx]
            if name in seen_b:
                continue
            seen_b.add(name)
            before_list.append((name, owners[idx]))
            if len(before_list) >= K_topk:
                break
        # after: top-K against shifted centroid (mean of sym's descriptors in D_after)
        home_indices = [i for i, o in enumerate(owners) if o == sym]
        if home_indices:
            c_after = D_after[home_indices].mean(axis=0)
            n = float(np.linalg.norm(c_after))
            c_after_unit = c_after / n if n > 1e-9 else cents[s_idx]
        else:
            c_after_unit = cents[s_idx]
        sims_after_shifted = D_after @ c_after_unit
        after_order = np.argsort(-sims_after_shifted)
        seen_a: set = set()
        after_list: list = []
        for j in after_order:
            idx = int(j)
            name = descriptor_names[idx]
            if name in seen_a:
                continue
            seen_a.add(name)
            after_list.append((name, owners[idx]))
            if len(after_list) >= K_topk:
                break
        before_names = {n for n, _ in before_list}
        after_names = {n for n, _ in after_list}
        emerged = [(n, o) for n, o in after_list if n not in before_names]
        faded   = [(n, o) for n, o in before_list if n not in after_names]
        return before_list, after_list, emerged, faded

    # Auto-pick target if not provided
    if not target or target not in space.symbol_to_idx:
        best_sym, best_score = None, -1
        for sym in symbols:
            _, _, em, fd = _identity_card(sym)
            score = len(em) + len(fd)
            if score > best_score:
                best_score = score
                best_sym = sym
        target = best_sym or (symbols[0] if symbols else "")

    before_list, after_list, emerged, faded = _identity_card(target)

    # Pick motifs based on mode
    mode = req.transformation_mode
    if mode == "emergence":
        chosen = [n for n, _ in emerged]
    elif mode == "fading":
        chosen = [n for n, _ in faded]
    else:  # becoming
        # interleave to give both sides representation
        chosen = []
        for pair in zip(emerged + [(None, None)] * len(faded), faded + [(None, None)] * len(emerged)):
            for n, _ in pair:
                if n and n not in chosen:
                    chosen.append(n)
        # if both lists are short, pad from after_list
        for n, _ in after_list:
            if len(chosen) >= density:
                break
            if n not in chosen:
                chosen.append(n)

    chosen = chosen[:density]

    # Build the prompt context — narration of what's changing in this archetype.
    def _fmt(items, lim=6):
        return ", ".join(f"{n} ({o})" if o and o != target else n for n, o in items[:lim]) or "—"

    if mode == "emergence":
        context_block = (
            f"The archetype {target} is taking on new aspects under this context.  "
            f"Newly drawn in: {_fmt(emerged)}.  "
            f"Tell a story of {target} becoming a vessel for these new presences."
        )
    elif mode == "fading":
        context_block = (
            f"The archetype {target} is shedding old aspects under this context.  "
            f"Falling away: {_fmt(faded)}.  "
            f"Tell a story of {target} releasing what it once held."
        )
    else:  # becoming
        context_block = (
            f"The archetype {target} is transforming.  "
            f"Fading: {_fmt(faded)}.  "
            f"Emerging: {_fmt(emerged)}.  "
            f"Tell a story of {target} crossing from its old form into its new one — "
            f"name what is left behind and what is taking its place."
        )

    return chosen, context_block, target


def _cycle_motifs(
    space, req: StoryRequest, target: Optional[str], density: int,
) -> Tuple[List[str], str, str]:
    """Return (motifs, prompt_context_block, resolved_target).

    When `target` is None, auto-picks the archetype with the highest
    persistence of a top H1 (or H2, per req.cycle_dim) feature.  The
    chosen cycle's ordered vertex words become the motifs IN ORDER —
    the prompt asks the LLM to trace them as the story's spine.
    """
    shift_params = _shift_params_from_req(req)
    D_after = engine_cache.get_or_compute_shifted_matrix(req.space_id, space, shift_params)
    sym_emb = _topology_symbol_embeddings_from(space, D_after)
    if not sym_emb:
        return [], "", target or ""

    dim_idx = 1 if req.cycle_dim == "h1" else 2

    def _top_cycle_for(sym: str):
        if sym not in sym_emb:
            return None  # (persistence, cycle_words, cocycle_raw)
        X = sym_emb[sym]
        out = _topology_get_ph(req.space_id, sym, X, maxdim=2, thresh=1.0, cocycles=True)
        H = out["dgms"][dim_idx]
        coc = out.get("cocycles", [[], [], []])[dim_idx] if "cocycles" in out else []
        if H.size == 0:
            return None
        pers = np.where(np.isfinite(H[:, 1]), H[:, 1] - H[:, 0], 0.0)
        if pers.max() <= 0:
            return None
        idx = int(np.argmax(pers))
        words = list(space.symbols_to_descriptors[sym])[: X.shape[0]]
        cyc_raw = coc[idx] if idx < len(coc) else []
        if dim_idx == 1:
            ordered = _topology_walk_h1_cycle(cyc_raw)
        else:
            # H2 — unordered vertex set
            verts = set()
            for row in cyc_raw:
                verts.update([int(row[0]), int(row[1]), int(row[2])])
            ordered = sorted(verts)
        cycle_words = [words[v] for v in ordered if v < len(words)]
        return float(pers[idx]), cycle_words

    # Auto-pick target = archetype with max top-cycle persistence
    if not target or target not in space.symbol_to_idx:
        best_sym, best = None, -1.0
        for sym in sym_emb.keys():
            result = _top_cycle_for(sym)
            if result and result[0] > best:
                best = result[0]
                best_sym = sym
        target = best_sym or (next(iter(sym_emb.keys())) if sym_emb else "")

    result = _top_cycle_for(target) if target else None
    if result is None:
        # No cycle for the requested dim — fall back to top-attention so the
        # story doesn't fail.  Empty cycle = blank motifs and a fallback prompt.
        return [], (
            f"No persistent {req.cycle_dim.upper()} {'loop' if dim_idx == 1 else 'void'} found in {target}.  "
            f"Tell a contemplative story about {target}'s inner stillness."
        ), target

    persistence, cycle_words = result
    motifs = cycle_words[:density]
    if dim_idx == 1:
        trail = " → ".join(cycle_words) + (f" → {cycle_words[0]}" if cycle_words else "")
        context_block = (
            f"A persistent semantic loop in {target}: {trail}.  "
            f"Use this loop as the narrative spine — let each word be a beat in the story, "
            f"and let the ending echo back to where it began.  Persistence={persistence:.3f}."
        )
    else:
        members = ", ".join(cycle_words)
        context_block = (
            f"A persistent void in {target} ({req.cycle_dim.upper()}), surrounded by: {members}.  "
            f"Tell a story that circles the absence at the heart of these words — what is NOT there shapes everything that is.  "
            f"Persistence={persistence:.3f}."
        )

    return motifs, context_block, target


def _build_prompt(
    *,
    context_sentence: Optional[str],
    motifs: Sequence[str],
    tone: str,
    pov: str,
    tense: str,
    target_words: int,
    language: str,
    form: str = "prose",
    anchor: Optional[str] = None,
    source_block: Optional[str] = None,
) -> List[dict]:
    lang_code = {"English": "en", "Français": "fr", "Español": "es"}.get(language, "en")
    sys_by_lang = {
        "en": (
            "You are a mythopoetic dream narrator. Write vivid, concise micro-fiction. "
            "No analysis or lists—just one cohesive paragraph. Evoke images, not exposition. "
            "Always write in English."
        ),
        "fr": (
            "Tu es un narrateur onirique et mythopoétique. Rédige une micro-fiction vive et concise. "
            "Aucune analyse ni liste—un seul paragraphe. Évoque des images, pas de l'exposition. "
            "Écris toujours en français."
        ),
        "es": (
            "Eres un narrador onírico y mitopoético. Escribe microficción vívida y concisa. "
            "Sin análisis ni listas—un solo párrafo. Evoca imágenes, no exposición. "
            "Escribe siempre en español."
        ),
    }
    pov_loc = {"en": {"first": "first person", "third": "third person"},
               "fr": {"first": "à la première personne", "third": "à la troisième personne"},
               "es": {"first": "en primera persona", "third": "en tercera persona"}}[lang_code].get(pov, pov)
    tense_loc = {"en": {"present": "present", "past": "past", "future": "future"},
                 "fr": {"present": "au présent", "past": "au passé", "future": "au futur"},
                 "es": {"present": "en presente", "past": "en pasado", "future": "en futuro"}}[lang_code].get(tense, tense)
    low, high = max(60, target_words - 40), target_words + 40
    if lang_code == "en":
        style = (f"tone={tone}; POV={pov_loc}; tense={tense_loc} tense; "
                 f"length≈{low}–{high} words; avoid clichés; end with a resonant image.")
    elif lang_code == "fr":
        style = (f"ton={tone} ; PDV={pov_loc} ; temps={tense_loc} ; "
                 f"longueur≈{low}–{high} mots ; évite les clichés ; termine sur une image marquante.")
    else:
        style = (f"tono={tone}; punto de vista={pov_loc}; tiempo verbal={tense_loc}; "
                 f"longitud≈{low}–{high} palabras; evita los clichés; termina con una imagen sugerente.")
    labels = {"en": ("Context", "Motifs to weave (use several explicitly)"),
              "fr": ("Contexte", "Motifs à tisser (utilise-les explicitement)"),
              "es": ("Contexto", "Motivos a entretejer (úsalos explícitamente)")}[lang_code]
    ctx_line = f"{labels[0]}: {context_sentence.strip()}" if (context_sentence and context_sentence.strip()) else f"{labels[0]}: (—)"
    motif_line = f"{labels[1]}: " + (", ".join(motifs[:12]) if motifs else "—")

    # Append per-tone style directives, avoid-lists and lexicons.  Without
    # these the LLM only sees a bare label like "tone=pynchon" and falls
    # back to its own stale caricature of the author — that's why every
    # Pynchon story was starting with "As she navigates the labyrinthine".
    extras = build_tone_extras(tone, lang_code)
    form_line = build_form_directive(form, lang_code)
    anchor_line = ""
    if anchor:
        if lang_code == "fr":
            anchor_line = f"Archétype d'ancrage : {anchor} — laisse cette présence centrer le texte."
        elif lang_code == "es":
            anchor_line = f"Arquetipo de anclaje: {anchor} — deja que esta presencia centre el texto."
        else:
            anchor_line = f"Anchor archetype: {anchor} — let this presence center the piece."

    # Order: shape (form) first, then style register (tone), then anchor.
    parts = []
    if form_line:
        parts.append(form_line)
    if extras:
        parts.extend(extras)
    if anchor_line:
        parts.append(anchor_line)
    extras_block = ("\n" + "\n".join(parts)) if parts else ""

    # Optional source-specific narration block — e.g. transformation arc
    # or cycle-loop walking instructions.  Placed BEFORE the motif list so
    # the LLM reads the structural directive first.
    source_section = f"\n\n{source_block.strip()}" if (source_block and source_block.strip()) else ""

    return [
        {"role": "system", "content": sys_by_lang[lang_code]},
        {"role": "user", "content": f"{ctx_line}{source_section}\n{motif_line}\nConstraints: {style}{extras_block}"},
    ]


def _call_cloudflare(messages, *, model, max_tokens, temperature, top_p) -> str:
    account_id = os.environ.get("CLOUDFLARE_ACCOUNT_ID")
    api_token = os.environ.get("CLOUDFLARE_API_TOKEN")
    if not account_id or not api_token:
        raise HTTPException(
            status_code=500,
            detail="CLOUDFLARE_ACCOUNT_ID / CLOUDFLARE_API_TOKEN env vars not set",
        )
    gateway = os.environ.get("CLOUDFLARE_GATEWAY_ID")
    if gateway:
        url = f"https://gateway.ai.cloudflare.com/v1/{account_id}/{gateway}/workers-ai/{model}"
    else:
        url = f"https://api.cloudflare.com/client/v4/accounts/{account_id}/ai/run/{model}"
    headers = {"Authorization": f"Bearer {api_token}", "Content-Type": "application/json",
               "cf-aig-skip-cache": "true"}
    if gateway:
        gtok = os.environ.get("CLOUDFLARE_GATEWAY_TOKEN")
        if gtok:
            headers["cf-aig-authorization"] = f"Bearer {gtok}"
    payload = {"messages": messages, "max_tokens": max_tokens,
               "temperature": temperature, "top_p": top_p}
    try:
        r = requests.post(url, json=payload, headers=headers, timeout=60)
        r.raise_for_status()
        data = r.json()
    except requests.RequestException as e:
        raise HTTPException(status_code=502, detail=f"Cloudflare request failed: {e}")
    if not data.get("success", False):
        raise HTTPException(status_code=502, detail=f"Cloudflare error: {data.get('errors')}")
    return (data.get("result", {}).get("response") or "").strip()


@router.post("/generate", response_model=StoryResponse)
def generate(req: StoryRequest) -> StoryResponse:
    space = engine_cache.get_space(req.space_id)
    if space is None:
        raise HTTPException(status_code=404, detail="unknown space_id")

    # 0. resolve anchor symbol (None / "auto" / explicit name).
    anchor = _resolve_anchor(space, req)

    # 1. extract motifs according to chosen source.
    density = max(2, min(30, int(req.motif_density or 12)))
    source = req.motif_source

    delta_motifs: List[str] = []
    att_motifs: List[str] = []
    source_block: Optional[str] = None
    auto_target: Optional[str] = None

    # ── topology sources: transformation / cycle ──
    if source == "transformation":
        # If anchor is set (auto-resolved or explicit), use it; else auto-pick
        # by transformation magnitude.
        target = anchor  # already resolved (None / explicit symbol)
        motifs_list, ctx_block, resolved = _transformation_motifs(space, req, target, density)
        if anchor is None:
            auto_target = resolved
        source_block = ctx_block
        motifs = motifs_list
    elif source == "cycle":
        target = anchor
        motifs_list, ctx_block, resolved = _cycle_motifs(space, req, target, density)
        if anchor is None:
            auto_target = resolved
        source_block = ctx_block
        motifs = motifs_list
    else:
        motifs = None  # legacy sources will compute below

    legacy_sources = source in ("delta-graph", "top-attention", "mixed")

    if legacy_sources and source in ("delta-graph", "mixed"):
        # Use the Δ-graph the user sees in Explorer (delta_params from
        # frontend) — fall back to defaults if absent.
        dparams = req.delta_params or DeltaGraphRequest(space_id=req.space_id)
        G = context_delta_graph(
            space,
            sentence=req.sentence,
            weights=req.weights,
            strategy=dparams.strategy,
            beta=dparams.beta,
            gate=dparams.gate,
            tau=dparams.tau,
            within_symbol_softmax=dparams.within_symbol_softmax,
            gamma=dparams.gamma,
            prompt_template=dparams.prompt_template,
            top_abs_edges=max(dparams.top_abs_edges, density * 2),
            sym_filter=dparams.sym_filter,
            min_abs_delta=dparams.min_abs_delta,
            within_symbol=dparams.within_symbol,
            only_symbol=dparams.only_symbol,
            connected_only=dparams.connected_only,
            pool_type=dparams.pool_type,
            pool_w=dparams.pool_w,
            membership_alpha=dparams.membership_alpha,
        )
        delta_motifs = _top_motifs(
            G,
            k_nodes=density if source == "delta-graph" else density,
            positive_only=req.positive_delta_only,
        )

    if legacy_sources and source in ("top-attention", "mixed"):
        att_motifs = _top_attention_motifs(
            space,
            sentence=req.sentence,
            weights=req.weights,
            k=density,
            anchor=anchor,
        )

    if legacy_sources:
        if source == "delta-graph":
            motifs = delta_motifs[:density]
        elif source == "top-attention":
            motifs = att_motifs[:density]
        else:  # mixed — interleave for variety, dedupe
            motifs = []
            seen: set = set()
            for a, b in zip(delta_motifs, att_motifs):
                for w in (a, b):
                    if w and w not in seen:
                        motifs.append(w)
                        seen.add(w)
                    if len(motifs) >= density:
                        break
                if len(motifs) >= density:
                    break
            # top up from whichever pool still has words
            for w in delta_motifs + att_motifs:
                if len(motifs) >= density:
                    break
                if w not in seen:
                    motifs.append(w)
                    seen.add(w)

    # 2. build prompt (form + anchor + tone all plumbed in).  When a
    # topology source resolved a target, prefer it as the prompt-level
    # anchor so the LLM sees the actual archetype the story is about.
    effective_anchor = anchor or auto_target
    messages = _build_prompt(
        context_sentence=req.sentence,
        motifs=motifs or [],
        tone=req.tone,
        pov=req.pov,
        tense=req.tense,
        target_words=req.length_words,
        language=req.language,
        form=req.form,
        anchor=effective_anchor,
        source_block=source_block,
    )

    # 3. call provider
    if req.provider == "cloudflare":
        story = _call_cloudflare(
            messages,
            model=req.model,
            max_tokens=int(req.length_words * 2.5),
            temperature=req.temperature,
            top_p=req.top_p,
        )
    else:
        raise HTTPException(status_code=400, detail="local provider not yet wired in API server; use Cloudflare")
    return StoryResponse(story=story, motifs=motifs or [], model=req.model, auto_target=auto_target)
