"""Story generator — wraps Cloudflare Workers AI (default) and the delta-graph
motif extractor used by the existing Streamlit app."""
from __future__ import annotations
import os
import requests
from typing import List, Optional, Sequence
from fastapi import APIRouter, HTTPException

from ..schemas import StoryRequest, StoryResponse, DeltaGraphRequest
from .. import engine_cache
from ..tone_presets import build_tone_extras, build_form_directive
from delyrism import context_delta_graph

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

    return [
        {"role": "system", "content": sys_by_lang[lang_code]},
        {"role": "user", "content": f"{ctx_line}\n{motif_line}\nConstraints: {style}{extras_block}"},
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

    if source in ("delta-graph", "mixed"):
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

    if source in ("top-attention", "mixed"):
        att_motifs = _top_attention_motifs(
            space,
            sentence=req.sentence,
            weights=req.weights,
            k=density,
            anchor=anchor,
        )

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

    # 2. build prompt (form + anchor + tone all plumbed in)
    messages = _build_prompt(
        context_sentence=req.sentence,
        motifs=motifs,
        tone=req.tone,
        pov=req.pov,
        tense=req.tense,
        target_words=req.length_words,
        language=req.language,
        form=req.form,
        anchor=anchor,
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
    return StoryResponse(story=story, motifs=motifs, model=req.model)
