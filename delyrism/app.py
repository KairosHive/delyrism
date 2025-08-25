# app.py — Archetype Explorer (Streamlit, final)
# -------------------------------------------------------------------
# GUI for SymbolSpace with:
#  • Meaning Space (context shift)
#  • Ranked Archetypes (proposal)
#  • Descriptor Attention
#  • Contextual Subgraph (new network view)
#  • Context Δ Graph (new delta-edges plot)
#  • JSON load/save
# Notes:
#  - We no-op pyplot.show/sci for Streamlit
#  - We wrap plotting calls and return figures
#  - Requires your added helpers in delyrism.py:
#       context_delta_graph, plot_delta_graph,
#       plot_contextual_subgraph_colored (and lighten_color inside the module)
# Run:  streamlit run app.py
# -------------------------------------------------------------------


from __future__ import annotations
import io
import json
from typing import Dict, List, Optional

from typing import Sequence
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
 
import streamlit as st
import numpy as np

try:
    import librosa
except Exception:
    librosa = None

try:
    from streamlit_mic_recorder import mic_recorder
except Exception:
    mic_recorder = None
    
# --- Matplotlib setup for Streamlit ---
import matplotlib
matplotlib.use("Agg")  # render off-screen
import matplotlib.pyplot as plt
plt.show = lambda *a, **k: None   # silence interactive popups
plt.sci  = lambda *a, **k: None   # avoid sci() errors

# Optional deps
try:
    import networkx as nx  # used by delta graph; required for those panels
except Exception:
    nx = None

import hashlib, json  # add at top if not present

# ---- import your core code (from your package/module) ----
from delyrism import (
    SymbolSpace,
    TextEmbedder,
    context_delta_graph,
    plot_delta_graph,
    plot_contextual_subgraph_colored,
    plot_ambiguity_metrics
)

if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

import os
os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")

try:
    import torch._dynamo as _dynamo
    _dynamo.config.enabled = False
    _dynamo.config.suppress_errors = True
    _dynamo.reset()
except Exception:
    pass

primaryColor = "#3498db"
# =============================
# Helpers
# =============================

def _freeze_weights(w: Optional[Dict[str, float]]):
    if not w: 
        return []
    return sorted((str(k), float(v)) for k, v in w.items())

def _delta_key(
    sentence, weights, shift_mode, beta, gate, tau_gate, within_symbol_softmax,
    gamma, pool_type, pool_w, top_abs_edges, sym_filter_sel, within_symbol,
    connected_only, membership_alpha, descriptor_threshold, embedder_fp, symbols_key
) -> str:
    payload = {
        "sentence": (sentence or "").strip(),
        "weights": _freeze_weights(weights),
        "shift_mode": shift_mode,
        "beta": float(beta),
        "gate": gate,
        "tau_gate": float(tau_gate if tau_gate is not None else 0.5),
        "within_symbol_softmax": bool(within_symbol_softmax),
        "gamma": float(gamma),
        "pool_type": pool_type,
        "pool_w": float(pool_w),
        "top_abs_edges": int(top_abs_edges),
        "sym_filter_sel": list(sym_filter_sel or []),
        "within_symbol": bool(within_symbol),
        "connected_only": bool(connected_only),
        "membership_alpha": float(membership_alpha),
        "descriptor_threshold": float(descriptor_threshold),
        "embedder_fp": embedder_fp,
        "symbols_key": symbols_key,
    }
    import hashlib, json as _json
    return hashlib.sha1(_json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()



TONE_PRESETS = {
    "pynchon": {
        "en": {
            "directives": (
                "Style: long, braided sentences with occasional sudden fragments; paranoid, satirical undertone; "
                "dense technical and historical vocabulary; quick zooms from street-level detail to systems theory. "
                "Favor appositives, parentheticals, sly asides in em-dashes, and conspiratorial hints. "
                "Weave motifs as if signals in a noisy network—interference, entropy, logistics, bureaucracy. "
                "Let humor flicker dryly; never explain the joke."
            ),
            "avoid": "Avoid tidy morals, flat exposition, and contemporary internet slang.",
            "lexicon": ["entropy", "carrier wave", "paper trail", "solder", "ledger", "detritus", "ballistic", "telemetry", "archive"]
        },
        "fr": {
            "directives": (
                "Style : phrases longues et tressées, apartés ironiques ; humour sec ; vocabulaire technique et historique. "
                "Glisse de la micro-sensation au système global ; sous-texte paranoïaque, satirique."
            ),
            "avoid": "Évite les explications plates et l’argot web contemporain.",
            "lexicon": ["entropie", "onde porteuse", "registre", "soudure", "archives"]
        },
        "es": {
            "directives": (
                "Estilo: frases largas entretejidas, apartes irónicos; humor seco; léxico técnico e histórico. "
                "Saltos de lo microscópico a lo sistémico; subtexto paranoico y satírico."
            ),
            "avoid": "Evita moralejas claras y jerga de internet moderna.",
            "lexicon": ["entropía", "onda portadora", "archivo", "soldadura", "bitácora"]
        }
    },

    "blake": {
        "en": {
            "directives": (
                "Style: prophetic lyricism with visionary imagery; elevated diction; choral cadences; "
                "antinomies (Innocence/Experience, Fire/Water); occasional archaic turns; "
                "capitalized abstract Nouns as presences. Employ parallelism and anaphora."
            ),
            "avoid": "Avoid modern bureaucratic phrasing and pop-culture references.",
            "lexicon": ["Tyger", "Albion", "Urizen", "Eternity", "Lambent", "Firmament", "Vesture", "Anvil"]
        },
        "fr": {
            "directives": (
                "Style : lyrisme prophétique, images visionnaires ; diction élevée ; parallélismes et anaphores ; "
                "antinomies ; Noms abstraits capitalisés comme Présences."
            ),
            "avoid": "Évite le jargon administratif et les références pop.",
            "lexicon": ["Urizen", "Albion", "Firmament", "Vesture", "Éternité"]
        },
        "es": {
            "directives": (
                "Estilo: lirismo profético, imaginería visionaria; dicción elevada; paralelismos y anáforas; "
                "antinomias; Sustantivos abstractos en mayúscula como Presencias."
            ),
            "avoid": "Evita jerga administrativa y referencias pop.",
            "lexicon": ["Urizen", "Albion", "Firmamento", "Vestidura", "Eternidad"]
        }
    },

    # two extra “complex & rich” options
    "mystic-baroque": {
        "en": {
            "directives": (
                "Style: ornate, clause-rich sentences; sensuous concretes; theological shimmer; "
                "use asyndeton and periodic build-ups; switch between close tactile detail and cosmic scales."
            ),
            "avoid": "Avoid minimalism and corporate clichés.",
            "lexicon": ["thurible", "vellum", "nave", "coruscation", "meridian", "throne", "censorial"]
        },
        "fr": {
            "directives": "Style : baroque mystique ; phrases amples ; concret sensuel ; élans cosmiques.",
            "avoid": "Évite le minimalisme et le jargon d’entreprise.",
            "lexicon": ["encensoir", "vélin", "nef", "coruscation", "méridien"]
        },
        "es": {
            "directives": "Estilo: barroco místico; frases amplias; concreción sensual; amplitud cósmica.",
            "avoid": "Evita minimalismo y clichés corporativos.",
            "lexicon": ["incensario", "vitela", "nave", "coruscación", "meridiano"]
        }
    },

    "gnostic-techno": {
        "en": {
            "directives": (
                "Style: luminous cyber-gnostic register; terse sentences braided with sudden liturgical bursts; "
                "mix semiconductor jargon with apocryphal reverence. Let signal and revelation mirror each other."
            ),
            "avoid": "Avoid comic-book technobabble and over-explaining.",
            "lexicon": ["lattice", "gate", "angelic protocol", "firmware", "pleroma", "daemon", "checksum"]
        },
        "fr": {
            "directives": "Style : cyber-gnostique lumineux ; phrases brèves entrecoupées d’élans liturgiques.",
            "avoid": "Évite le techno-jargon caricatural.",
            "lexicon": ["trame", "passerelle", "pleroma", "daemon", "somme de contrôle"]
        },
        "es": {
            "directives": "Estilo: ciber-gnóstico luminoso; frases breves con irrupciones litúrgicas.",
            "avoid": "Evita tecnicismos caricaturescos.",
            "lexicon": ["retícula", "compuerta", "pleroma", "daemon", "suma de verificación"]
        }
    },
}

SIMPLE_TONE_EXTRAS = {
    "dreamy": {
        "en": "soft focus, hypnagogic transitions, sensory synesthesia, ellipsis of motives, light anaphora.",
        "fr": "flou doux, transitions hypnagogiques, synesthésie sensorielle, ellipses de motifs.",
        "es": "foco suave, transiciones hipnagógicas, sinestesia sensorial, elipsis de motivos.",
    },
    "eerie": {
        "en": "quiet dread, negative space, mundane objects made numinous, withheld explanations, sparse adjectives.",
        "fr": "crainte silencieuse, vides, banal devenu numineux, explications retenues.",
        "es": "temor silencioso, espacios vacíos, lo banal hecho numinoso, explicaciones retenidas.",
    },
}


def build_gemma_prompt(
    *,
    context_sentence: str | None,
    motifs: Sequence[str],
    tone: str = "dreamy",
    pov: str = "first",
    tense: str = "present",
    target_words: tuple[int, int] = (120, 180),
    language: str = "English",
) -> list[dict]:
    """
    Build a chat-style prompt with localized constraints (EN/FR/ES),
    enriching `tone` via TONE_PRESETS or SIMPLE_TONE_EXTRAS when available.
    """
    # --- language maps ---
    lang_code = {
        "English": "en", "Français": "fr", "French": "fr",
        "Español": "es", "Spanish": "es",
        "en": "en", "fr": "fr", "es": "es",
    }.get(language, "en")

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

    # localized labels and style words
    tone_map = {
        "en": {"dreamy": "dreamy", "eerie": "eerie", "warm": "warm"},
        "fr": {"dreamy": "rêveur", "eerie": "étrange", "warm": "chaleureux"},
        "es": {"dreamy": "soñador", "eerie": "inquietante", "warm": "cálido"},
    }
    pov_map = {
        "en": {"first": "first person", "third": "third person"},
        "fr": {"first": "à la première personne", "third": "à la troisième personne"},
        "es": {"first": "en primera persona", "third": "en tercera persona"},
    }
    tense_map = {
        "en": {"present": "present tense", "past": "past tense"},
        "fr": {"present": "au présent", "past": "au passé"},
        "es": {"present": "en presente", "past": "en pasado"},
    }
    labels = {
        "en": {"context": "Context", "motifs": "Motifs to weave (use several explicitly)"},
        "fr": {"context": "Contexte", "motifs": "Motifs à tisser (utiliser plusieurs explicitement)"},
        "es": {"context": "Contexto", "motifs": "Motivos a entretejer (usa varios explícitamente)"},
    }

    # pick localized words for the basic constraints line
    tone_loc  = tone_map[lang_code].get(tone, tone)   # falls back to raw tone (e.g., 'pynchon')
    pov_loc   = pov_map[lang_code].get(pov, pov)
    tense_loc = tense_map[lang_code].get(tense, tense)

    # base constraints line (kept compact so models follow it)
    if lang_code == "en":
        style = (
            f"tone={tone_loc}; POV={pov_loc}; tense={tense_loc}; "
            f"length≈{target_words[0]}–{target_words[1]} words; avoid clichés; end with a resonant image."
        )
    elif lang_code == "fr":
        style = (
            f"ton={tone_loc} ; PDV={pov_loc} ; temps={tense_loc} ; "
            f"longueur≈{target_words[0]}–{target_words[1]} mots ; évite les clichés ; "
            "termine sur une image marquante."
        )
    else:  # es
        style = (
            f"tono={tone_loc}; punto de vista={pov_loc}; tiempo verbal={tense_loc}; "
            f"longitud≈{target_words[0]}–{target_words[1]} palabras; evita los clichés; "
            "termina con una imagen sugerente."
        )

    # ── enrich tone via presets or simple extras ───────────────────────────
    extra_style = ""
    preset = TONE_PRESETS.get(tone)
    if preset:
        loc = preset.get(lang_code, preset.get("en", {}))
        directives = loc.get("directives", "")
        avoid = loc.get("avoid", "")
        lex = loc.get("lexicon", [])
        if directives:
            extra_style += f"\nStyle directives: {directives}"
        if avoid:
            extra_style += f"\nAvoid: {avoid}"
        if lex:
            extra_style += f"\nSuggested lexicon: {', '.join(map(str, lex[:10]))}"
    else:
        # not a preset: try to augment simple tones like 'dreamy', 'eerie'
        simple_extra = SIMPLE_TONE_EXTRAS.get(tone, {})
        extra_text = simple_extra.get(lang_code) or simple_extra.get("en")
        if extra_text:
            extra_style += f"\nStyle directives: {extra_text}"

    # labels + content
    ctx_lab = labels[lang_code]["context"]
    motifs_lab = labels[lang_code]["motifs"]
    ctx_line = f"{ctx_lab}: {context_sentence.strip()}" if (context_sentence and context_sentence.strip()) else f"{ctx_lab}: (—)"
    motif_line = f"{motifs_lab}: " + (", ".join(map(str, motifs[:12])) if motifs else "—")

    # final chat payload
    messages = [
        {"role": "system", "content": sys_by_lang[lang_code]},
        {"role": "user", "content": f"{ctx_line}\n{motif_line}\nConstraints: {style}{extra_style}"},
    ]
    return messages




import transformers as tf

@st.cache_resource(show_spinner=False)
def load_gemma(model_id: str, use_8bit: bool=False, force_gpu: bool=False):
    """
    Works with:
      • Gemma 2 chat (e.g., google/gemma-2-2b-it)
      • Gemma 3 1B chat (google/gemma-3-1b-it)
      • Gemma 3n E2B chat (google/gemma-3n-e2b-it)  -> needs transformers >= 4.50.0
    Returns (tok_or_proc, model).
    """
    kw = dict(low_cpu_mem_usage=True)

    if torch.cuda.is_available():
        kw["torch_dtype"] = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        kw["attn_implementation"] = "eager"
        kw["device_map"] = {"": 0} if (force_gpu and not use_8bit) else "auto"

    if use_8bit:
        try:
            from transformers import BitsAndBytesConfig
            kw["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
            kw["device_map"] = "auto"
            kw["offload_folder"] = "offload"
        except Exception:
            st.warning("bitsandbytes not installed; using full precision.")

    mid = model_id.strip().lower()

    # ---- Gemma 3n (multimodal; we use text-only path via processor) ----
    if "gemma-3n" in mid:
        # version guard
        try:
            from packaging.version import Version
            if Version(tf.__version__) < Version("4.50.0"):
                st.error(f"Gemma 3n needs transformers>=4.50.0 (found {tf.__version__}). "
                         "Upgrade with: pip install -U transformers accelerate safetensors")
                raise RuntimeError("transformers too old for Gemma 3n")
        except Exception:
            # If packaging not available, still try and show version
            st.info(f"transformers version: {getattr(tf, '__version__', 'unknown')}")

        try:
            from transformers import AutoProcessor, Gemma3nForConditionalGeneration
        except Exception as e:
            st.error("Could not import Gemma3nForConditionalGeneration. "
                     "Please upgrade transformers (>=4.50.0).")
            raise

        proc = tf.AutoProcessor.from_pretrained(model_id)
        mdl  = tf.Gemma3nForConditionalGeneration.from_pretrained(model_id, **kw).eval()
        return proc, mdl

    # ---- Gemma 3 (1B, text-only) ----
    if "gemma-3-" in mid or mid.endswith("gemma-3"):
        try:
            from transformers import Gemma3ForCausalLM
            mdl = tf.Gemma3ForCausalLM.from_pretrained(model_id, **kw).eval()
        except Exception:
            # Fallback keeps things running on slightly older transformers
            mdl = tf.AutoModelForCausalLM.from_pretrained(model_id, **kw).eval()
        tok = tf.AutoTokenizer.from_pretrained(model_id, use_fast=True)
        return tok, mdl

    # ---- Gemma 2 (baseline that worked for you) ----
    tok = tf.AutoTokenizer.from_pretrained(model_id, use_fast=True)
    mdl = tf.AutoModelForCausalLM.from_pretrained(model_id, **kw).eval()
    return tok, mdl


def generate_with_gemma(tok_or_proc, mdl, messages, *, max_new_tokens=180, temperature=0.8, top_p=0.9, repetition_penalty=1.05):
    """
    Compatible with:
      • tokenizer + causal LM (Gemma 2 / Gemma 3 1B)
      • processor + Gemma3nForConditionalGeneration (text-only)
    """
    def is_processor(x):
        name = x.__class__.__name__.lower()
        return ("processor" in name) or hasattr(x, "image_processor")

    # normalize messages for processor vs tokenizer
    normalized = []
    for m in messages:
        if is_processor(tok_or_proc):
            content = m.get("content", "")
            if isinstance(content, str):
                content = [{"type": "text", "text": content}]
            normalized.append({"role": m.get("role", "user"), "content": content})
        else:
            normalized.append(m)

    # render to tensors
    try:
        if is_processor(tok_or_proc):
            inputs = tok_or_proc.apply_chat_template(
                normalized, add_generation_prompt=True, return_tensors="pt"
            )
        else:
            prompt = tok_or_proc.apply_chat_template(normalized, tokenize=False, add_generation_prompt=True)
            inputs = tok_or_proc(prompt, return_tensors="pt")
    except Exception as err:
        # fallback for tokenizers without system role support
        if not is_processor(tok_or_proc) and "system role" in str(err).lower():
            sys_text = "\n".join(m["content"] for m in messages if m.get("role") == "system").strip()
            user_msgs = [m for m in messages if m.get("role") == "user"]
            user_text = user_msgs[0]["content"] if user_msgs else ""
            fused = ("[System guidelines]\n" + sys_text + "\n\n" + user_text).strip() if sys_text else user_text
            inputs = tok_or_proc(fused, return_tensors="pt")
        else:
            raise

    # move to model device
    inputs = {k: v.to(mdl.device) for k, v in inputs.items()}
    input_len = inputs["input_ids"].shape[1]

    # eos/pad + decoder for text
    if is_processor(tok_or_proc):
        eos_id = getattr(mdl.config, "eos_token_id", None)
        pad_id = getattr(mdl.config, "pad_token_id", eos_id)
        tok_for_decode = getattr(tok_or_proc, "tokenizer", None)
    else:
        if tok_or_proc.pad_token_id is None:
            tok_or_proc.pad_token_id = tok_or_proc.eos_token_id
        eos_id = tok_or_proc.eos_token_id
        pad_id = tok_or_proc.pad_token_id
        tok_for_decode = tok_or_proc

    from contextlib import nullcontext
    sdp_ctx = nullcontext()
    if torch.cuda.is_available():
        sdp_ctx = torch.backends.cuda.sdp_kernel(enable_flash=False, enable_mem_efficient=False, enable_math=True)

    with torch.no_grad(), sdp_ctx:
        out = mdl.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            eos_token_id=eos_id,
            pad_token_id=pad_id,
            use_cache=True,
        )

    gen = out[0, input_len:] if isinstance(out, torch.Tensor) else out.sequences[0, input_len:]
    return tok_for_decode.decode(gen, skip_special_tokens=True).strip() if tok_for_decode else ""



# ===== Δ-graph → motifs =====
def top_motifs_from_delta_graph(G, *, k_nodes=10, positive_only=True):
    """
    Pick salient descriptors from the Δ graph.
    - Rank edges by |Δ| (or Δ>0 if positive_only).
    - Return top unique node labels (descriptor names).
    """
    if G is None or G.number_of_edges() == 0:
        return []

    # sort edges
    edges = []
    for u, v, d in G.edges(data=True):
        delta = float(d.get("delta", 0.0))
        if positive_only and delta <= 0:
            continue
        edges.append((u, v, abs(delta)))
    edges.sort(key=lambda x: x[2], reverse=True)

    picked = []
    seen = set()
    for u, v, _ in edges:
        for n in (u, v):
            if n not in seen:
                seen.add(n)
                picked.append(n)
                if len(picked) >= k_nodes:
                    return picked
    return picked

def _default_symbols_map() -> Dict[str, List[str]]:
    return {
        "EARTH": [
            "ground",
            "soil",
            "minerals",
            "crystal lattice",
            "plate tectonics",
            "sedimentation",
            "geomagnetism",
            "gravity",
            "fertility",
            "body",
            "stability",
            "structure",
            "homeostasis",
            "roots",
            "mycorrhizae",
            "nourishment",
            "boundaries",
            "patience",
            "pentacle",
            "muladhara",
        ],
        "WATER": [
            "river",
            "ocean",
            "universal solvent",
            "cohesion",
            "surface tension",
            "capillarity",
            "osmosis",
            "hydrologic cycle",
            "blood",
            "lymph",
            "emotion",
            "cleansing",
            "reflection",
            "adaptability",
            "tides",
            "lunar influence",
            "baptism",
            "flow state",
            "viscosity",
            "purification",
        ],
        "FIRE": [
            "combustion",
            "ignition",
            "exothermy",
            "heat transfer",
            "radiation",
            "photons",
            "plasma",
            "volcano",
            "lightning",
            "metabolism",
            "fever",
            "calcination",
            "transformation",
            "passion",
            "will",
            "forge",
            "entropy increase",
            "catalysis",
            "solar radiation",
            "hearth",
        ],
        "AIR": [
            "breath",
            "oxygenation",
            "wind",
            "turbulence",
            "convection",
            "pressure gradients",
            "diffusion",
            "sound wave",
            "speech",
            "communication",
            "intellect",
            "clarity",
            "birds",
            "pollen dispersal",
            "volatility",
            "fragrance",
            "prana",
            "qi (chi)",
            "inspiration",
            "changeability",
        ],
        "ETHER": [
            "space",
            "akasha",
            "subtle body",
            "sahasrara",
            "emptiness",
            "silence",
            "awareness",
            "consciousness",
            "nonlocality",
            "resonance",
            "coherence",
            "electromagnetism",
            "spacetime",
            "quantum vacuum",
            "zero-point",
            "information",
            "connectivity",
            "signal",
            "harmony",
            "presence",
        ],
    }



def _embedder_key(e: TextEmbedder) -> str:
    b = getattr(e, "backend_type", "unknown")
    m = getattr(e, "model_name", None)
    p = getattr(e, "pooling", None)
    d = getattr(e, "dim", None)
    inst = getattr(e, "default_instruction", None)
    ctx  = getattr(e, "default_context", None)  # may be "Distributed" sentinel, a string, or None

    base = f"{b}|{m}|{p}|{d}"
    extra = json.dumps({"inst": inst or "", "ctx": ctx or ""}, ensure_ascii=False, sort_keys=True)
    h = hashlib.sha1(extra.encode("utf-8")).hexdigest()[:10]
    return f"{base}|{h}"

def _symbols_map_key(symbols_map: Dict[str, List[str]]) -> str:
    return json.dumps(symbols_map, sort_keys=True, ensure_ascii=False)

def _load_symbols_map(txt: str | None) -> Dict[str, List[str]]:
    if not txt:
        return _default_symbols_map()
    try:
        data = json.loads(txt)
        assert isinstance(data, dict)
        return {str(k): [str(x) for x in v] for k, v in data.items()}
    except Exception as e:
        st.error(f"Failed to parse JSON: {e}")
        return _default_symbols_map()


@st.cache_resource(show_spinner=False)
def get_embedder(backend: str, model: Optional[str], pooling: str) -> TextEmbedder:
    return TextEmbedder(backend=backend, model=model or None, pooling=pooling)


@st.cache_resource(show_spinner=False)
def build_space(
    symbols_map_key: str,
    descriptor_threshold: float,
    embedder_fingerprint: str
) -> SymbolSpace:
    # Rebuild the actual objects inside the cached function
    # by reading from the global session state or by passing the real objects back in.
    # Easiest: pass the real objects through st.session_state.
    e = st.session_state["_current_embedder"]
    smap = st.session_state["_current_symbols_map"]
    return SymbolSpace(symbols_to_descriptors=smap, embedder=e, descriptor_threshold=descriptor_threshold)


def fig_from_callable(callable_fn, *args, **kwargs):
    """Call a plotting function that draws with pyplot; return the current figure."""
    fig = plt.figure()
    plt.close(fig)
    callable_fn(*args, **kwargs)
    return plt.gcf()

def focus_to_tau(focus: float, tau_min: float = 0.01, tau_max: float = 0.2) -> float:
    # focus=0 -> tau_max (soft); focus=1 -> tau_min (sharp)
    return tau_max - focus * (tau_max - tau_min)
# =============================
# UI
# =============================

st.set_page_config(page_title="Archetype Explorer", layout="wide")
st.title("🧭 DELYRISM - Archetype Explorer ")

with st.sidebar:
    
    st.markdown("""
    <style>
      /* --- Existing custom headers (Data / Embeddings / Context) --- */
      .sb-one summary {
        background-color: #2a1536 !important;
        border: 1px solid #4b2560 !important;
        border-radius: 8px !important;
        padding: 6px 10px !important;
        color: #fff !important;
      }
      .sb-two summary {
        background-color: #0f1a2b !important;
        border: 1px solid #1b2b44 !important;
        border-radius: 8px !important;
        padding: 6px 10px !important;
        color: #fff !important;
      }
      .sb-three summary {
        background-color: #14403d !important;
        border: 1px solid #23736e !important;
        border-radius: 8px !important;
        padding: 6px 10px !important;
        color: #fff !important;
      }

    </style>
    """, unsafe_allow_html=True)


    # --- Data ----------------------------------------------------
    # --- Data ----------------------------------------------------
    st.markdown('<div class="sb-one">', unsafe_allow_html=True)
    with st.expander("Data", expanded=True):
        # 1) File upload (takes precedence if present)
        uploaded = st.file_uploader("Upload symbols→descriptors JSON", type=["json"])
        symbols_map: Dict[str, List[str]]

        if uploaded is not None:
            try:
                raw = uploaded.read().decode("utf-8")
                symbols_map = _load_symbols_map(raw)  # uses your str->dict helper
                st.success("Loaded JSON from file.")
            except Exception as e:
                st.error(f"Failed to read uploaded file: {e}")
                symbols_map = _default_symbols_map()
        else:
            # 2) Text area fallback when no file uploaded
            symbols_txt = st.text_area(
                "…or paste JSON here",
                value=json.dumps(_default_symbols_map(), ensure_ascii=False, indent=2),
                height=210,
                help="Format: {symbol: [descriptor, ...]}",
            )
            symbols_map = _load_symbols_map(symbols_txt)
    st.markdown('</div>', unsafe_allow_html=True)

    # --- Embeddings ----------------------------------------------
    st.markdown('<div class="sb-two">', unsafe_allow_html=True)
    # --- Embeddings (sidebar) ---
    with st.expander("Embeddings", expanded=True):
        # ADD (audio): include audioclip as a backend option
        backend_help = (
            "Which encoder turns your inputs into vectors.\n"
            "- Sentence-Transformer: good general text embeddings (e.g., all-mpnet-base-v2).\n"
            "- Qwen2/Qwen3 Embedding: strong multilingual; uses token pooling (EOS by default).\n"
            "- AudioCLIP / CLAP: enable AUDIO → vector (and text, for AudioCLIP). Use only if you need audio.\n"
            "Changing backend re-embeds descriptors and context; results, dims, and speed can change."
        )
        # backend = st.selectbox("Embedding backend", [...], help=backend_help)
        backend = st.selectbox("Backend", ["qwen3", "qwen2", "sentence-transformer", "clap"], index=0, help=backend_help)
        hf_model_help = (
            "Hugging Face repo ID for the embedding model (e.g., 'sentence-transformers/all-mpnet-base-v2', "
            "'Qwen/Qwen2-Embedding', 'Qwen/Qwen3-Embedding-0.6B').\n"
            "Pick an *embedding* model (not a causal LM) to get fixed-size vectors.\n"
            "Changing this will re-embed everything (dimension, quality, and speed can differ)."
        )
        model = st.text_input("HF model override (optional)", help=hf_model_help)
        pooling_help = (
            "How token embeddings are collapsed into one vector per text.\n"
            "- eos (default): last *non-padding* token. Length-safe and works well for Qwen-style encoders.\n"
            "- mean: mask-aware average over all real tokens. Very robust for sentence embeddings.\n"
            "- cls: first token ([CLS]). Best when the model was trained to use CLS (e.g., BERT family).\n"
            "- last: final position regardless of padding/truncation. Can be brittle—avoid unless you need it.\n"
            "All pooled vectors are L2-normalized. Keep the same setting when comparing runs."
        )
        pooling = st.selectbox("Pooling", ["eos", "mean", "cls", "last"], index=0, help=pooling_help)
        embedder = get_embedder(backend, model or None, pooling)

        if backend == "qwen3":  # (use `in ("qwen2","qwen3")` if you want both)
            st.markdown("**Qwen prompting (instruction + context)**")

            instr_help = (
                "A global instruction prepended to every encode. Keep it stable across runs to make embeddings comparable.\n"
                "Example: 'Instruction: Encode archetypal descriptors for retrieval and clustering.'"
            )
            qwen_instruction = st.text_area(
                "Instruction (applied to all encodes)", height=80, help=instr_help,
                placeholder="Instruction: Encode archetypal descriptors for retrieval and clustering."
            )

            ctx_mode_help = (
                "Choose how to provide `Context:` for Qwen:\n"
                "• None — no context added\n"
                "• Global string — one context string for all descriptors (good for dataset-wide framing)\n"
                "• Per-descriptor owner — uses each descriptor's owning symbol as its context"
            )
            qwen_ctx_mode = st.radio(
                "Context mode", ["None", "Global string", "Per-descriptor owner"],
                index=0, horizontal=False, help=ctx_mode_help
            )

            qwen_ctx_global = ""
            if qwen_ctx_mode == "Global string":
                qwen_ctx_global = st.text_input(
                    "Global context string",
                    help="Will be used for every descriptor as `Context:`. Example: 'Domain=archetypes; Audience=practitioners.'",
                    placeholder="Domain=archetypes; Audience=practitioners."
                )

            # Apply to the embedder (these affect how `encode()` templates inputs)
            embedder.default_instruction = (qwen_instruction or None)
            if   qwen_ctx_mode == "Global string" and qwen_ctx_global.strip():
                embedder.default_context = qwen_ctx_global.strip()
            elif qwen_ctx_mode == "Per-descriptor owner":
                # sentinel used by your SymbolSpace.__post_init__ to pass per-descriptor contexts
                embedder.default_context = "Distributed"
            else:
                embedder.default_context = None
        # (optional) tiny hint when audioclip is selected
        if backend == "clap":
            st.caption("CLAP enabled. Upload or record short audio clip in the Context panel to drive the analysis.")

    # --- Context (panel-colored sliders) --------------------------
    st.markdown('<div class="sb-three panel-context">', unsafe_allow_html=True)
    with st.expander("Context", expanded=True):
        sentence = st.text_area(
            "Context prompt",
            placeholder="e.g., A ceremony by the river focusing on transformation and healing",
            height=90,
            key="ctx_sentence",
        )

        # ADD (audio): optional audio context
        # ADD (audio): optional audio context (upload or record)
        audio_ctx_vec = None
        with st.popover("Or use an audio context"):
            src = st.radio("Source", ["Upload", "Record"], horizontal=True)

            max_secs = st.slider("Use up to (seconds)", 2, 30, 12, key="audio_secs")

            if src == "Upload":
                audio_file = st.file_uploader("Upload audio (wav/mp3/m4a)", type=["wav", "mp3", "m4a"])
                if audio_file is not None:
                    if backend not in ("audioclip", "clap"):
                        st.warning("Switch backend to 'audioclip' to use audio context.")
                    elif librosa is None:
                        st.error("librosa not installed. `pip install librosa` to enable audio loading.")
                    else:
                        try:
                            # mono @48k, center-trim to max_secs
                            y, sr = librosa.load(audio_file, sr=48000, mono=True)
                            if len(y) > sr * max_secs:
                                start = (len(y) - sr * max_secs) // 2
                                y = y[start:start + sr * max_secs]
                            audio_ctx_vec = embedder.embed_audio_array(y, sr)
                            st.session_state["audio_ctx_vec"] = audio_ctx_vec
                            st.session_state["audio_backend"] = backend
                            st.session_state["audio_fp"] = float(np.linalg.norm(audio_ctx_vec))
                            st.success(f"Audio context ready (||v||={st.session_state['audio_fp']:.3f}).")
                            st.success("Audio context ready (upload). It will override the text context.")
                        except Exception as e:
                            st.error(f"Audio read/encode failed: {e}")

            else:  # Record
                if mic_recorder is None:
                    st.error("`streamlit-mic-recorder` is not installed. Run: pip install streamlit-mic-recorder")
                else:
                    st.caption("Click to start/stop. Your mic stays in the browser; only the WAV is sent.")
                    # one-shot recorder; returns dict with 'bytes' for the wav
                    rec = mic_recorder(
                        start_prompt="🎙️ Start recording",
                        stop_prompt="⏹ Stop",
                        just_once=False,            # allow recording again
                        use_container_width=True,
                        format="wav",
                        key="mic_widget",          # stable key
                    )

                    if rec and rec.get("bytes"):
                        if backend not in ("audioclip", "clap"):
                            st.warning("Switch backend to 'audioclip' to use audio context.")
                        elif librosa is None:
                            st.error("librosa not installed. `pip install librosa` to enable audio loading.")
                        else:
                            try:
                                import io as _io
                                # Load from bytes as file-like, resample to 48k mono
                                y, sr = librosa.load(_io.BytesIO(rec["bytes"]), sr=48000, mono=True)
                                # center-trim to user limit
                                if len(y) > sr * max_secs:
                                    start = (len(y) - sr * max_secs) // 2
                                    y = y[start:start + sr * max_secs]
                                audio_ctx_vec = embedder.embed_audio_array(y, sr)
                                st.session_state["audio_ctx_vec"] = audio_ctx_vec
                                st.session_state["audio_backend"] = backend
                                st.session_state["audio_fp"] = float(np.linalg.norm(audio_ctx_vec))
                                st.success(f"Audio context ready (||v||={st.session_state['audio_fp']:.3f}).")
                                st.audio(rec["bytes"])  # quick playback
                                st.success("Audio context ready (mic). It will override the text context.")
                            except Exception as e:
                                st.error(f"Mic audio encode failed: {e}")

        if "audio_fp" in st.session_state:
            st.caption(f"Current audio vector norm: {st.session_state['audio_fp']:.3f}")

        st.caption("Select which symbols are in the context and give them weights.")
        sym_preview = list(symbols_map.keys())

        # selection
        default_ctx = st.session_state.get("ctx_chosen", sym_preview[:2])
        context_symbols_help = (
            "Optional symbol priors to steer the context.\n\n"
            "What it feeds: we build v_ctx = normalize(  Σᵢ wᵢ·centroid(symbolᵢ)  +  sentence_vector ).\n"
            "Only positive weights are used; they are internally normalized.\n\n"
            "Where it matters:\n"
            "1) Descriptor shifting (all strategies): v_ctx sets the direction of the shift/tilt.\n"
            "2) Proposals/ranking: contributes to the coherence/attention term; with PPR on, weights also seed PR personalization.\n"
            "3) Graph visuals: marked as ‘context’ in the big symbol/descriptor graph (highlighted nodes).\n\n"
            "Tips:\n"
            "- Prefer a few focused symbols with moderate weights over many tiny weights.\n"
            "- Heavy weights can dominate the sentence; lower them if text isn’t showing effect.\n"
            "- Set to none (or all zeros) to rely purely on the sentence context."
        )
        ctx_chosen = st.multiselect(
            "Context symbols",
            options=sym_preview,
            default=[s for s in default_ctx if s in sym_preview],
            key="ctx_chosen",
            help=context_symbols_help
        )

        # weights (these sliders inherit .panel-context color)
        ctx_weights: Dict[str, float] = {}
        for s in ctx_chosen:
            ctx_weights[s] = st.slider(
                f"Weight: {s}",
                0.0, 1.0,
                value=st.session_state.get(f"w_{s}", 0.5),
                step=0.05,
                key=f"w_{s}",
            )
    st.markdown('</div>', unsafe_allow_html=True)

    # --- Shift settings (panel-colored sliders) -------------------
    st.markdown('<div class="panel-shift">', unsafe_allow_html=True)
    with st.expander("Semantic Map", expanded=True):
        st.markdown("**Map display**")
        with_hulls = st.checkbox("Draw convex hulls", True)
        inc_cent_help = "Show one star per symbol at its mean descriptor position."
        include_centroids = st.checkbox("Include centroids (stars)", True, help=inc_cent_help)

        norm_cent_help = (
            "Scale each centroid to unit length BEFORE projection.\n"
            "On → star positions reflect direction (angle) only, ignoring vector length effects "
            "from cluster tightness/size. Off → magnitude can tug centroids in the layout."
        )
        normalize_centroids = st.checkbox("Normalize centroids (unit-length)", False, help=norm_cent_help)

        
        show_arrow = st.checkbox("Show arrows", True)
        
    st.markdown('</div>', unsafe_allow_html=True)

    # --- Ranking (panel-colored sliders) --------------------------
    st.markdown('<div class="panel-ranking">', unsafe_allow_html=True)
    with st.expander("Ranking (proposal)", expanded=True):
        tau_help = "Lower τ = sharper attention; higher τ = broader attention."
        tau = st.slider("Softmax temperature (τ)", 0.01, 2.0, 0.3, 0.01, help=tau_help)

        # PageRank damping (α)
        alpha_help = (
            "Affects HOW PR is computed (inside the graph walk), not the blend.\n"
            "α = chance to follow graph edges each step; (1−α) = jump back to the context prior.\n"
            "Higher α (e.g., 0.9) → trust graph structure more; lower α (e.g., 0.6) → trust the context prior more."
        )
        alpha = st.slider("PageRank damping (α)", 0.10, 0.99, 0.8, 0.01, help=alpha_help)

        # Blend λ (PR vs. attention)
        lam_help = (
            "Affects HOW scores are combined AFTER PR is computed.\n"
            "Final score = (1−λ)·PR + λ·attention/coherence.\n"
            "λ = 0 → PR only; λ = 1 → attention only. λ does not change PR itself; it just mixes it.\n"
            "If PPR is off, the PR term is zero, so λ≈1 has no PR to blend."
        )
        lam = st.slider("Blend λ (PR-graph vs Softmax-attention)", 0.0, 1.0, 0.6, 0.01, help=lam_help)

        ppr_help = "Include graph-aware Personalized PageRank in the score blend."
        use_ppr = st.checkbox("Use Personalized PageRank", True, help=ppr_help)

    st.markdown('</div>', unsafe_allow_html=True)

    # --- Contextual Subgraph (panel-colored sliders) --------------
    st.markdown('<div class="panel-subgraph">', unsafe_allow_html=True)
    with st.expander("Contextual Subgraph (network)", expanded=True):
        topk_symbols_help = "How many highest-scoring symbols to show for this context."
        ctx_topk_symbols = st.slider("Top symbols", 1, 12, 3, help=topk_symbols_help)

        topk_desc_help = "How many of each symbol’s descriptors to keep (highest weighted)."
        ctx_topk_desc = st.slider("Top descriptors / symbol", 1, 12, 3, help=topk_desc_help)

        ctx_method_help = "ppr = graph-aware PageRank; softmax = direct context→descriptor attention."
        ctx_method = st.selectbox("Scoring method", ["ppr", "softmax"], index=0, help=ctx_method_help)

        focus_help = "Right = sharper focus (lower τ). Left = broader context (higher τ)."
        ctx_focus = st.slider("Context Focus (sharper → right)", 0.0, 1.0, 0.95, 0.01, help=focus_help)

        alpha_help = "PageRank damping: higher α = more graph influence; lower α = more context."
        ctx_alpha = st.slider("α (subgraph PageRank)", 0.50, 0.99, 0.85, 0.01, help=alpha_help)

        norm_help = "Remove centrality prior: show PR(context) minus PR(uniform). (PPR only)"
        ctx_normalize = st.checkbox("Normalize by baseline PR (remove centrality)", True, help=norm_help)

        thr_help = "Cosine cutoff for descriptor–descriptor edges. Higher = sparser/cleaner; lower = denser/noisier."
        descriptor_threshold = st.slider("Descriptor edge threshold (cosine)", 0.0, 0.9, 0.1, 0.02, help=thr_help)
    st.markdown('</div>', unsafe_allow_html=True)

    # --- Δ Graph (panel-colored sliders) --------------------------
    st.markdown('<div class="panel-delta">', unsafe_allow_html=True)
    with st.expander("Δ Graph", expanded=True):
        st.markdown("**Shift settings**")
        # add "pooling" to the strategy list
        strategy_help = """
        How to inject context into descriptors:
         - gate: tilt existing vectors toward the context (fast, geometry-preserving)
         - reembed: re-encode “sentence + descriptor” (expressive, heavier)
         - hybrid: blend gate & reembed (γ controls the mix)
         - pooling: average with the context (or max/min), smoother group motion
        """
        shift_mode = st.selectbox("Strategy", ["gate", "reembed", "hybrid", "pooling"],
                                index=0, help=strategy_help)

        if shift_mode in ('reembed', 'hybrid'):
            st.caption("Reembed and hybrid modes only take into account textual context")

        # --- pooling controls (visible only when selected) ---
        if shift_mode == "pooling":
            pool_type = st.selectbox("Pooling type", ["avg", "max", "min"], index=0, help="Element-wise pool of descriptors with context vector")
            if pool_type == 'avg':
                pool_w = st.slider("Pooling weight w (avg mode)", 0.0, 1.0, 0.7, 0.05, help="Weight of context for pool_type='avg'")
            else:
                pool_w = 0.7
            st.caption("Pooling blends each descriptor with the context vector directly (no β/gate). Try max for aggressive context, or avg with w≈0.7–0.9.")

        else:
            # still define defaults so downstream calls don't KeyError
            pool_type, pool_w = "avg", 0.7
        if shift_mode == 'gate':
            help_text_alpha = """
            **Membership (α)** controls how much each descriptor moves with its symbol.
            - α=0: per-descriptor shift only (fine detail, spikier Δ).
            - α=1: all descriptors follow the symbol centroid (smooth, group-level Δ).
            On the Δ graph, higher α makes within-symbol changes rise/fall together, highlighting coherent shifts.
            """

            membership_alpha = st.slider("Membership α (desc vs centroid)", 0.0, 1.0, 0.0, 0.05, help=help_text_alpha)
            gate_help = """
            Per-descriptor gate g_i (used by 'gate' and 'hybrid'):
            - relu: max(0, cos) — only aligned descriptors move
            - cos: signed cos — aligned move toward, anti-aligned move away
            - softmax: softmax(cos/τ) — sharp focus on the most relevant descriptors
            - uniform: 1 — uniform tilt toward the context
            """
            gate = st.selectbox(
                "Gate",
                ["relu", "cos", "softmax", "uniform"],
                index=0,
                disabled=(shift_mode == "pooling"),
                help=gate_help
            )

            if gate == "softmax":
                tau_gate = st.slider(
                    "Softmax temperature (τ)",
                    0.01, 2.0, 0.3, 0.01,
                    help="Lower = sharper (focus on a few descriptors). Higher = broader."
                )
            else:
                tau_gate = None  # not used by other gates
            softmax_help = """
            If gate='softmax': normalize per symbol instead of globally.
            - ON: each symbol's descriptors compete only with each other (fairer, comparable emphasis).
            - OFF: all descriptors compete together (strong global winners).
            """
            within_symbol_softmax = st.checkbox(
                "Softmax within symbol (if gate=softmax)",
                True,
                # NOTE: Enable this for both 'gate' *and* 'hybrid' (hybrid uses the gate too).
                disabled=not (shift_mode in ("gate", "hybrid") and gate == "softmax"),
                help=softmax_help
            )

        else:
            membership_alpha = 0.0
            gate = 'relu'
            within_symbol_softmax = True
            tau_gate = None
        # existing controls (unchanged)
        if shift_mode == "hybrid":
            gamma = st.slider("Hybrid blend γ (0=gate, 1=reembed)", 0.0, 1.0, 0.5, 0.05, disabled=(shift_mode != "hybrid"))
        else:
            gamma = 0.5

        beta = st.slider("Shift strength β", 0.0, 2.0, 1.2, 0.05, disabled=(shift_mode == "pooling"))  # β not used by pooling
        
        
        st.markdown("**Graph network settings**")
        within_symbol = st.checkbox("Within-symbol pairs only", False)
        sym_filter_sel = st.multiselect("Or restrict to symbols", sym_preview)
        top_abs_edges = st.slider("Top |Δ| edges", 2, 100, 10, 1)
        connected_only = st.checkbox("Connected nodes only", True)
        
    st.markdown('</div>', unsafe_allow_html=True)

# after you compute 'symbols_map' and 'embedder'
st.session_state["_current_symbols_map"] = symbols_map
st.session_state["_current_embedder"] = embedder

space = build_space(
    _symbols_map_key(symbols_map),
    descriptor_threshold,
    _embedder_key(embedder)  # ← includes instruction/context
)
space.set_context_vec(audio_ctx_vec)  # None clears; vector overrides text everywhere
# Clear stale audio when backend changes away from an audio-capable model
if st.session_state.get("audio_backend") not in ("audioclip", "clap") and "audio_ctx_vec" in st.session_state:
    st.session_state.pop("audio_ctx_vec", None)
    st.session_state.pop("audio_fp", None)

audio_vec = st.session_state.get("audio_ctx_vec")

# Only set context if we actually have an audio vector AND the backend supports audio
if backend in ("audioclip", "clap") and audio_vec is not None:
    space.set_context_vec(audio_vec)
else:
    space.set_context_vec(None)

tab_explore, tab_story = st.tabs(["Explorer", "Story Generator"])

with tab_explore:
    # =============================
    # Row 1: Meaning Space (left) | Rankings & Attention (right)
    # =============================
    colL, colR = st.columns([1.15, 1])
    color_map = space.get_symbol_color_dict("AuroraPop")
    with colL:
        st.subheader("Meaning Space (2D)")
        reducer = st.selectbox("Reducer", ["umap", "tsne","pca"], index=0)
        if show_arrow is True:
            arrow_scale = 0.5
        if show_arrow is False:
            arrow_scale = 0
        fig_ms = fig_from_callable(
            space.plot_map_shift,
            weights=ctx_weights if ctx_weights else None,
            sentence=sentence if sentence else None,
            method=reducer,
            with_hulls=with_hulls,
            include_centroids=include_centroids,
            normalize_centroids=normalize_centroids,
            figsize=(6.8, 5.4),
            title="Context shift on descriptor map",
            arrow_scale=arrow_scale,
            arrow_alpha=0.65,
            gate=gate,
            tau=tau,
            beta=beta,
            membership_alpha=membership_alpha,
            within_symbol_softmax=within_symbol_softmax,
            color_dict=color_map
        )
        st.pyplot(fig_ms, clear_figure=True)

        st.divider()
        st.subheader("Ambiguity Metrics")

        sort_opt = st.selectbox("Sort by", ["dispersion", "leakage", "entropy", "none"], index=0)
        # color_map = getattr(space, "get_symbol_color_dict", lambda: None)()
        
        
        fig_amb = plot_ambiguity_metrics(space, sort_by=sort_opt, color_dict=color_map, figsize=(7.5, 4))
        st.pyplot(fig_amb, clear_figure=True)


    with colR:

        
        st.subheader("Descriptor Attention")
        sym = st.selectbox("Symbol", list(space.symbols))
        if sym:
            try:
                fig_att = fig_from_callable(
                    space.plot_attention,
                    sym,
                    weights=ctx_weights if ctx_weights else None,
                    sentence=sentence if sentence else None,
                    tau=tau,
                    top_n=8,
                    figsize=(6, 3.6),
                )
                st.pyplot(fig_att, clear_figure=True)
            except Exception as e:
                st.warning(f"Attention plot failed: {e}")
        
        st.subheader("Top Symbols for Context")
        try:
            print('-----------------------------------')
            print(ctx_weights, sentence)
            print('-----------------------------------')
            preds = space.propose(
                weights=ctx_weights if ctx_weights else None,
                sentence=sentence if sentence else None,
                tau=tau,
                lam=lam,
                alpha=alpha,
                topk=len(space.symbols),
                use_ppr=use_ppr,
            )

            if preds:
                # Exclude symbols explicitly in the context weights
                exclude = {k.lower() for k in (ctx_weights or {}).keys()}
                
                # (optional) also exclude symbols literally mentioned in the sentence
                ctx_words = {w.strip(".,;:!?()[]{}\"'").lower() for w in (sentence or "").split()}
                preds = [p for p in preds if (p[0].lower() not in exclude and p[0].lower() not in ctx_words)]

                if not preds:
                    st.info("All top symbols are part of the context.")
                else:
                    labels = [p[0] for p in preds]
                    scores = np.array([p[1] for p in preds])

                    norm = (scores - scores.min()) / (scores.max() - scores.min() + 1e-8)
                    cmap = plt.cm.coolwarm
                    colors = [cmap(n) for n in norm]

                    fig_rank, ax = plt.subplots(figsize=(6, 4))
                    bars = ax.barh(
                        range(len(scores))[::-1],
                        scores[::-1],
                        color=colors[::-1],
                        edgecolor='gray',
                        linewidth=1.2
                    )
                    ax.set_yticks(range(len(labels))[::-1])
                    ax.set_yticklabels(labels[::-1])
                    vmin, vmax = scores.min(), scores.max()
                    ax.set_xlim(vmin - 0.01, vmax + 0.01)
                    ax.set_xlabel("Score")
                    ax.set_title("Symbol prediction for context (excluding context symbols)")

                    for bar, v in zip(bars, scores[::-1]):
                        ax.add_patch(plt.Rectangle(
                            (bar.get_x(), bar.get_y()), bar.get_width(), bar.get_height(),
                            color='white', alpha=0.08, zorder=0
                        ))

                    fig_rank.tight_layout()
                    st.pyplot(fig_rank, clear_figure=True)
            else:
                st.info("No predictions yet.")
        except Exception as e:
            st.warning(f"Ranking failed: {e}")

    st.divider()

    # =============================
    # Row 2: Contextual Subgraph (network) | Heatmaps
    # =============================

    colN, colH = st.columns([1.1, 1])

    with colN:
        st.subheader("Network View — Contextual Subgraph")
        try:
            tau_subgraph = focus_to_tau(ctx_focus)
            print('Tau subgraph', tau_subgraph)
            # Build a stable global color palette once
            cmap = plt.cm.tab20
            global_color_map = {s: cmap(i / max(1, len(space.symbols)-1)) for i, s in enumerate(space.symbols)}
            fig_ctxnet = fig_from_callable(
                plot_contextual_subgraph_colored,
                space,
                context_sentence=sentence or "",
                topk_symbols=ctx_topk_symbols,
                topk_desc=ctx_topk_desc,
                method=ctx_method,
                alpha=ctx_alpha,
                tau=tau_subgraph,
                normalize=ctx_normalize,
                global_color_map=color_map
            )
            st.pyplot(fig_ctxnet, clear_figure=True)
        except Exception as e:
            st.warning(f"Contextual subgraph failed: {e}")

    with colH:
        st.subheader("Within-Symbol Associative Increase (Δ)")
        sym2 = st.selectbox("Symbol for heatmaps", list(space.symbols), key="heat_sym")
        if sym2:
            try:
                simdict = space.descriptor_similarity_matrices(
                    weights=ctx_weights if ctx_weights else None,
                    sentence=sentence if sentence else None,
                    strategy=shift_mode,
                    beta=beta,
                    gate=gate,
                    tau=(tau_gate if tau_gate is not None else 0.5),  # pick a default if not softmax
                    within_symbol_softmax=within_symbol_softmax,
                    gamma=gamma,
                    pool_type=pool_type,
                    pool_w=pool_w,
                    order_by_attention=True,
                    membership_alpha=membership_alpha,
                )


                from functools import partial
                fig_heat = fig_from_callable(
                    space.plot_symbol_similarity_heatmaps,
                    simdict,
                    sym2,
                    vmax=1.0,
                    figsize=(12, 3.8),
                )
                st.pyplot(fig_heat, clear_figure=True)
            except Exception as e:
                st.warning(f"Heatmaps failed: {e}")

    st.divider()

    # =============================
    # Row 3: Context Δ Graph
    # =============================

    st.subheader("Graph of strongest edge changes within network")
    if nx is None:
        st.info("networkx not installed — delta graph requires networkx.")
    else:
        try:
            # cdict = getattr(space, "get_symbol_color_dict", None)
            # color_map = cdict() if callable(cdict) else None

            sym_filter_arg = sym_filter_sel if sym_filter_sel else None

            G = context_delta_graph(
                space,
                sentence=sentence if sentence else None,
                weights=ctx_weights if ctx_weights else None,
                strategy=shift_mode,
                beta=beta,
                gate=gate,
                tau=(tau_gate if tau_gate is not None else 0.5),
                within_symbol_softmax=within_symbol_softmax,
                gamma=gamma,
                pool_type=pool_type,
                pool_w=pool_w,
                top_abs_edges=top_abs_edges,
                sym_filter=sym_filter_arg,
                within_symbol=within_symbol,
                connected_only=connected_only,
                membership_alpha=membership_alpha,
            )

            key_dg = _delta_key(
                sentence, ctx_weights, shift_mode, beta, gate, (tau_gate if tau_gate is not None else 0.5),
                within_symbol_softmax, gamma, pool_type, pool_w, top_abs_edges, sym_filter_sel,
                within_symbol, connected_only, membership_alpha, descriptor_threshold,
                _embedder_key(embedder), _symbols_map_key(symbols_map),
            )
            st.session_state["delta_graph"] = G
            st.session_state["delta_graph_key"] = key_dg

            fig_delta = fig_from_callable(
                plot_delta_graph,
                G,
                title="Context Δ graph",
                color_dict=color_map,
                figsize=(8.0, 3.0),        # ⬅️ smaller figure
                node_size_base=130,        # ⬇️ shrink nodes
                node_size_scale=700.0,
                edge_width_min=0.4,        # ⬇️ thinner edges
                edge_width_max=3.0,
            )
            st.pyplot(fig_delta, clear_figure=True, use_container_width=False)  # prevent auto-stretch
        except Exception as e:
            st.warning(f"Δ graph failed: {e}")

    st.divider()

    # =============================
    # Save / Export
    # =============================

    st.subheader("📦 Export / Save")
    colA, colB = st.columns(2)
    with colA:
        buf = io.StringIO()
        json.dump(symbols_map, buf, ensure_ascii=False, indent=2)
        st.download_button(
            label="Download symbols.json",
            data=buf.getvalue().encode("utf-8"),
            file_name="symbols.json",
            mime="application/json",
        )
    with colB:
        st.caption("(Coming soon) Export PNGs/CSVs for figures and descriptor weights.")

    st.caption("Tip: steer the landscape with the prompt/weights; adjust β/τ/α/λ; use contextual subgraph + Δ graph to inspect structural changes.")

with tab_story:
    st.subheader("Dream-like Story (Gemma)")

    # --- Controls live inside a FORM so nothing "executes" until submit ---
    with st.form("story_form", clear_on_submit=False):
        # Model controls
        GEMMA_MODEL_PRESETS = {
            "Gemma 2 (2B-IT)":  "google/gemma-2-2b-it",
            "Gemma 3 (1B-IT)":  "google/gemma-3-1b-it",   # text-only chat
            "Gemma 3n (E2B-IT)": "google/gemma-3n-e2b-it", # multimodal (we use text-only here)
        }

        # ... inside the story_form() block, replace the single text_input with:
        preset = st.selectbox(
            "Model preset",
            list(GEMMA_MODEL_PRESETS.keys()),
            index=0,
            help="Quick-switch between Gemma 2, Gemma 3 1B, and Gemma 3n."
        )
        
        default_model_id = GEMMA_MODEL_PRESETS[preset]
        model_id = st.text_input(
            "Hugging Face repo (override if you want)",
            value=default_model_id,
            help="Use a chat-tuned repo ID. You must accept the model license on Hugging Face first."
        )
        use_8bit = st.checkbox("Load in 8-bit (less VRAM, slightly lower quality)", False)

        # Narrative controls
        story_len_words = st.slider("Target length (words)", 60, 300, 140, 10)
        language = st.selectbox("Language", ["English", "Français", "Español"], index=0)  # NEW
        temperature = st.slider("Temperature", 0.1, 1.8, 0.85, 0.05)
        top_p = st.slider("Top-p", 0.1, 1.0, 0.9, 0.05)
        pos_only = st.checkbox("Use only strengthening (Δ > 0) edges", True)
        pov = st.selectbox("POV", ["first","third"], index=0)
        tense = st.selectbox("Tense", ["present","past"], index=0)
        tone = st.selectbox(
            "Tone",
            ["dreamy","eerie","warm","pynchon","blake","mystic-baroque","gnostic-techno"],
            index=0
        )

        submit = st.form_submit_button("🚀 Generate story", type="primary", use_container_width=True)

    # --- Only generate when the button was pressed ---
    if submit:
        with st.spinner("Generating..."):
            # Build Δ graph only on demand
            sym_filter_arg = sym_filter_sel if sym_filter_sel else None
            tau_gate_eff = tau_gate if (gate == "softmax" and tau_gate is not None) else 0.5


            key_story = _delta_key(
                sentence, ctx_weights, shift_mode, beta, gate, tau_gate_eff,
                within_symbol_softmax, gamma, pool_type, pool_w, top_abs_edges, sym_filter_sel,
                within_symbol, connected_only, membership_alpha, descriptor_threshold,
                _embedder_key(embedder), _symbols_map_key(symbols_map),
            )

            G_story = None
            if st.session_state.get("delta_graph_key") == key_story and "delta_graph" in st.session_state:
                G_story = st.session_state["delta_graph"]
            else:
                G_story = context_delta_graph(
                    space,
                    sentence=sentence or None,
                    weights=ctx_weights or None,
                    strategy=shift_mode,
                    beta=beta,
                    gate=gate,
                    tau=tau_gate_eff,
                    within_symbol_softmax=within_symbol_softmax,
                    gamma=gamma,
                    pool_type=pool_type,
                    pool_w=pool_w,
                    top_abs_edges=top_abs_edges,
                    sym_filter=sym_filter_sel if sym_filter_sel else None,
                    within_symbol=within_symbol,
                    connected_only=connected_only,          # <-- same as Explorer now
                    membership_alpha=membership_alpha,
                )
                st.session_state["delta_graph"] = G_story
                st.session_state["delta_graph_key"] = key_story
            
            motifs = top_motifs_from_delta_graph(G_story, k_nodes=12, positive_only=pos_only)

            # Free VRAM from embedder, then lazy-load Gemma
            import gc, torch as _torch
            try:
                getattr(space.embedder, "to", lambda *_: None)("cpu")
            except Exception:
                pass
            gc.collect()
            if _torch.cuda.is_available():
                _torch.cuda.empty_cache()

            force_gpu = st.checkbox("Force GPU (no offload)", False)
            tok, mdl = load_gemma(model_id, use_8bit=use_8bit, force_gpu=force_gpu)  

            with st.expander("⚙️ Inference device map (debug)"):
                lines = [f"torch.cuda.is_available(): {torch.cuda.is_available()}"]
                if torch.cuda.is_available():
                    lines.append(f"CUDA device: {torch.cuda.get_device_name(0)}")
                dm = getattr(mdl, "hf_device_map", None)
                lines.append(f"hf_device_map: {dm if dm else '(none)'}")
                try:
                    first_param_dev = next(mdl.parameters()).device
                    lines.append(f"first parameter device: {first_param_dev}")
                except StopIteration:
                    pass
                st.code("\n".join(lines), language="text")
   
            messages = build_gemma_prompt(
                context_sentence=sentence or "",
                motifs=motifs,
                tone=tone, pov=pov, tense=tense,
                target_words=(story_len_words-30, story_len_words+30),
                language=language,   # NEW
            )
            story = generate_with_gemma(
                tok, mdl, messages,
                max_new_tokens=story_len_words + 60,
                temperature=temperature,
                top_p=top_p,
                repetition_penalty=1.07
            )

            # Persist for display on future reruns without regenerating
            st.session_state["story_text"] = story
            st.session_state["story_motifs"] = motifs

    # --- Display last generated story (or a hint) ---
    if st.session_state.get("story_text"):
        st.text_area("Story", st.session_state["story_text"], height=260)
        st.download_button("Download story.txt", st.session_state["story_text"], file_name="story.txt")
        m = st.session_state.get("story_motifs") or []
        st.caption("Motifs used: " + (", ".join(m) if m else "—"))
        if st.button("🧹 Clear story"):
            st.session_state.pop("story_text", None)
            st.session_state.pop("story_motifs", None)
    else:
        st.info("Set your context and parameters, then click **Generate story**.")
