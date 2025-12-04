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
from typing import Dict, List, Optional, Sequence

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
import threading

# Global lock for pyplot to prevent race conditions between users
_pyplot_lock = threading.Lock()

plt.show = lambda *a, **k: None   # silence interactive popups
plt.sci  = lambda *a, **k: None   # avoid sci() errors

# Optional deps
try:
    import networkx as nx  # used by delta graph; required for those panels
except Exception:
    nx = None

try:
    import umap
    _HAS_UMAP = True
except ImportError:
    _HAS_UMAP = False

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

# CPU Optimization for Container/Cloud Environments
if not torch.cuda.is_available():
    # Limit PyTorch threads to avoid CPU contention/throttling in shared vCPUs
    torch.set_num_threads(min(4, os.cpu_count() or 1))

try:
    import torch._dynamo as _dynamo
    _dynamo.config.enabled = False
    _dynamo.config.suppress_errors = True
    _dynamo.reset()
except Exception:
    pass

# === Multimodal miner & IO helpers ===
from multimodal_archetype_miner import ArchetypeMiner, MMItem

# Image & OpenCLIP
try:
    import open_clip
    from PIL import Image
except Exception:
    open_clip = None
    Image = None

import tempfile, pathlib, uuid

# --- Fragment compatibility ---
try:
    from streamlit import fragment
except ImportError:
    try:
        from streamlit import experimental_fragment as fragment
    except ImportError:
        # Fallback: no-op decorator if not supported
        def fragment(func):
            return func

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
import requests

# ===== Cloudflare Workers AI =====
CLOUDFLARE_MODELS = {
    "Llama 3.1 (8B-Instruct)": "@cf/meta/llama-3.1-8b-instruct",
    "Llama 3.2 (3B-Instruct)": "@cf/meta/llama-3.2-3b-instruct",
    "Llama 3.2 (1B-Instruct)": "@cf/meta/llama-3.2-1b-instruct",
    "Mistral (7B-Instruct)": "@cf/mistral/mistral-7b-instruct-v0.1",
    "Qwen 1.5 (7B-Chat)": "@cf/qwen/qwen1.5-7b-chat-awq",
    "Qwen 1.5 (1.8B-Chat)": "@cf/qwen/qwen1.5-1.8b-chat",
    "Qwen 1.5 (0.5B-Chat)": "@cf/qwen/qwen1.5-0.5b-chat",
    "Gemma (7B-IT-LoRA)": "@cf/google/gemma-7b-it-lora",
}

def generate_with_cloudflare(
    messages: list,
    *,
    model: str = "@cf/meta/llama-3.1-8b-instruct",
    account_id: str = None,
    api_token: str = None,
    max_tokens: int = 256,
    temperature: float = 0.8,
    top_p: float = 0.9,
) -> str:
    """
    Call Cloudflare Workers AI for text generation.
    Requires CLOUDFLARE_ACCOUNT_ID and CLOUDFLARE_API_TOKEN env vars or Streamlit secrets.
    """
    # Get credentials from env or secrets
    account_id = account_id or os.environ.get("CLOUDFLARE_ACCOUNT_ID") or st.secrets.get("CLOUDFLARE_ACCOUNT_ID", "")
    api_token = api_token or os.environ.get("CLOUDFLARE_API_TOKEN") or st.secrets.get("CLOUDFLARE_API_TOKEN", "")
    
    if not account_id or not api_token:
        raise ValueError(
            "Cloudflare credentials not found. Set CLOUDFLARE_ACCOUNT_ID and CLOUDFLARE_API_TOKEN "
            "as environment variables or in Streamlit secrets."
        )
    
    url = f"https://api.cloudflare.com/client/v4/accounts/{account_id}/ai/run/{model}"
    
    # Convert messages to Cloudflare format (same as OpenAI style)
    cf_messages = []
    for m in messages:
        role = m.get("role", "user")
        content = m.get("content", "")
        # Cloudflare doesn't support system role for all models - merge into user
        if role == "system":
            cf_messages.append({"role": "system", "content": content})
        else:
            cf_messages.append({"role": role, "content": content})
    
    payload = {
        "messages": cf_messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
    }
    
    headers = {
        "Authorization": f"Bearer {api_token}",
        "Content-Type": "application/json",
    }
    
    try:
        response = requests.post(url, json=payload, headers=headers, timeout=60)
        response.raise_for_status()
        data = response.json()
        
        if not data.get("success", False):
            errors = data.get("errors", [])
            raise RuntimeError(f"Cloudflare API error: {errors}")
        
        return data.get("result", {}).get("response", "").strip()
    
    except requests.exceptions.Timeout:
        raise RuntimeError("Cloudflare API timeout. Try a smaller model or shorter output.")
    except requests.exceptions.RequestException as e:
        raise RuntimeError(f"Cloudflare API request failed: {e}")


@st.cache_resource(show_spinner=False)
def load_gemma(model_id: str, use_8bit: bool=False, force_gpu: bool=False):
    """
    Works with:
      • Qwen2.5-Instruct models (ungated, recommended)
      • TinyLlama-Chat (ungated)
      • SmolLM2-Instruct (ungated)
      • Gemma 2/3 chat (⚠️ gated - requires HF login)
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
                normalized, add_generation_prompt=True, return_tensors="pt", tokenize=True
            )
            # Some processors return a string or tensor, not a dict
            if isinstance(inputs, str):
                # It returned a string prompt, tokenize it manually
                inputs = tok_or_proc(text=inputs, return_tensors="pt")
            elif isinstance(inputs, torch.Tensor):
                # It returned just input_ids tensor
                inputs = {"input_ids": inputs}
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

    # move to model device - handle dict, BatchEncoding, and tensor inputs
    if isinstance(inputs, torch.Tensor):
        # Pure tensor case
        inputs = inputs.to(mdl.device)
        input_len = inputs.shape[1]
        inputs = {"input_ids": inputs}
    elif hasattr(inputs, "to") and hasattr(inputs, "input_ids"):
        # BatchEncoding or similar dict-like with .to() method
        inputs = inputs.to(mdl.device)
        input_len = inputs["input_ids"].shape[1]
        # Convert to plain dict for generate()
        inputs = {k: v for k, v in inputs.items() if isinstance(v, torch.Tensor)}
    elif isinstance(inputs, dict):
        # Plain dict
        inputs = {k: v.to(mdl.device) if hasattr(v, "to") else v for k, v in inputs.items()}
        input_len = inputs["input_ids"].shape[1]
    else:
        raise TypeError(f"Unexpected inputs type: {type(inputs)}")

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
    embedder_fingerprint: str,
    version: str = "v1"
) -> SymbolSpace:
    # Rebuild the actual objects inside the cached function
    # by reading from the global session state or by passing the real objects back in.
    # Easiest: pass the real objects through st.session_state.
    e = st.session_state["_current_embedder"]
    smap = st.session_state["_current_symbols_map"]
    return SymbolSpace(symbols_to_descriptors=smap, embedder=e, descriptor_threshold=descriptor_threshold)


def fig_from_callable(callable_fn, *args, **kwargs):
    """
    Call a plotting function that draws with pyplot; return the *live* current figure.
    We temporarily no-op plt.close() so internal helpers can't detach artists.
    """
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_agg import FigureCanvasAgg

    # Acquire lock to ensure thread safety for global pyplot state
    with _pyplot_lock:
        # Avoid RuntimeWarning: More than 20 figures have been opened
        plt.close('all')

        # Keep original close; temporarily block closes inside callable_fn
        _orig_close = plt.close
        try:
            plt.close = lambda *a, **k: None  # prevent premature figure teardown

            # Ensure there is a current figure before drawing
            _ = plt.figure()
            callable_fn(*args, **kwargs)

            fig = plt.gcf()
            
            # Apply dark theme styling (transparent background, white text)
            fig.patch.set_alpha(0.0)
            for ax in fig.axes:
                ax.patch.set_alpha(0.0)
                ax.xaxis.label.set_color('white')
                ax.yaxis.label.set_color('white')
                ax.title.set_color('white')
                ax.tick_params(axis='x', colors='white')
                ax.tick_params(axis='y', colors='white')
                for spine in ax.spines.values():
                    spine.set_visible(False)
                if ax.get_legend():
                    legend = ax.get_legend()
                    legend.get_frame().set_alpha(0.0)
                    legend.get_frame().set_linewidth(0.0)
                    plt.setp(legend.get_texts(), color='white')
                    if legend.get_title():
                        legend.get_title().set_color('white')

            # Make sure the figure has a canvas so Streamlit can savefig
            if fig.canvas is None:
                FigureCanvasAgg(fig)
            return fig
        finally:
            # Restore normal behavior
            plt.close = _orig_close


def focus_to_tau(focus: float, tau_min: float = 0.01, tau_max: float = 0.2) -> float:
    # focus=0 -> tau_max (soft); focus=1 -> tau_min (sharp)
    return tau_max - focus * (tau_max - tau_min)


def _ensure_upload_dir() -> str:
    """Per-session upload dir for images/audio/text."""
    if "_upload_dir" not in st.session_state:
        d = tempfile.mkdtemp(prefix="delyrism_uploads_")
        st.session_state["_upload_dir"] = d
    return st.session_state["_upload_dir"]

def _save_upload(file, subdir: str) -> str:
    """Persist an uploaded file; return local path."""
    base = _ensure_upload_dir()
    sd = pathlib.Path(base) / subdir
    sd.mkdir(parents=True, exist_ok=True)
    ext = pathlib.Path(file.name).suffix or ".bin"
    out = sd / f"{uuid.uuid4().hex}{ext}"
    out.write_bytes(file.read())
    return str(out)

@st.cache_resource(show_spinner=False)
def _load_openclip(model_name="ViT-B-32", pretrained="laion2b_s34b_b79k"):
    if open_clip is None or Image is None:
        raise RuntimeError("open-clip-torch and pillow are required for image embeddings.")
    model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
    model.eval()
    if torch.cuda.is_available():
        model = model.to("cuda")
    tokenizer = open_clip.get_tokenizer(model_name)
    return model, preprocess, tokenizer

class _ImageAdapter:
    def __init__(self, model_name="ViT-B-32", pretrained="laion2b_s34b_b79k"):
        self.model, self.preprocess, _ = _load_openclip(model_name, pretrained)
        self.device = next(self.model.parameters()).device

    @torch.no_grad()
    def encode_images(self, paths):
        ims = []
        for p in paths:
            im = Image.open(p).convert("RGB")
            ims.append(self.preprocess(im))
        x = torch.stack(ims).to(self.device)
        feats = self.model.encode_image(x)
        feats = feats / feats.norm(dim=-1, keepdim=True)
        return feats.detach().cpu().numpy()

class _AudioAdapter:
    """Uses your existing embedder if it exposes embed_audio_array(y, sr). Falls back to CLAP via HF if needed."""
    def __init__(self, text_embedder):
        self.e = text_embedder
        self.has_embed_audio = hasattr(text_embedder, "embed_audio_array")
        # Optional HF fallback (kept lazy to avoid import cost)
        self._hf = None

    def _hf_init(self):
        if self._hf is None:
            import transformers as _tf
            self._hf = _tf.pipeline("feature-extraction", model="laion/clap-htsat-unfused", trust_remote_code=True)
        return self._hf

    def _embed_with_embedder(self, wav_paths):
        vecs = []
        import soundfile as sf
        import numpy as np
        for p in wav_paths:
            y, sr = sf.read(p, always_2d=False)
            if isinstance(y, np.ndarray) and y.ndim > 1:
                y = y.mean(axis=1)  # mono
            # Resample to 48k if your embedder expects that; librosa is already in your deps:
            if librosa is not None and sr != 48000:
                y = librosa.resample(y.astype(np.float32), orig_sr=sr, target_sr=48000)
                sr = 48000
            v = self.e.embed_audio_array(y, sr)
            vecs.append(v)
        import numpy as np
        X = np.stack(vecs, 0)
        X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)
        return X

    def _embed_with_hf(self, wav_paths):
        # Minimal CLAP feature extractor; average-pool frames, L2-normalize.
        pipe = self._hf_init()
        import numpy as np
        outs = []
        for p in wav_paths:
            feat = pipe(p)  # [T, D] or [1, T, D]
            arr = np.array(feat)
            arr = arr.squeeze(0) if arr.ndim == 3 else arr
            v = arr.mean(axis=0)
            v = v / (np.linalg.norm(v) + 1e-8)
            outs.append(v)
        return np.stack(outs, 0)

    def embed_audio_files(self, paths):
        if self.has_embed_audio:
            return self._embed_with_embedder(paths)
        return self._embed_with_hf(paths)

class _TextAdapter:
    def __init__(self, text_embedder): self.e = text_embedder
    def encode(self, texts): return self.e.encode(texts)

# =============================
# UI
# =============================

st.set_page_config(
    page_title="Archetype Explorer", 
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'About': "Delyrism - Archetype Explorer"
    }
)

# Force dark theme via CSS (for users without dark mode set)
st.markdown("""
    <style>
    /* Force dark background for light mode users */
    .stApp, [data-testid="stAppViewContainer"], [data-testid="stHeader"] {
        background-color: #0e1117 !important;
    }
    [data-testid="stSidebar"], [data-testid="stSidebarContent"] {
        background-color: #262730 !important;
    }
    /* Ensure text is readable */
    .stApp, .stApp p, .stApp span, .stApp label, .stApp div {
        color: #fafafa !important;
    }
    /* Fix selectbox and input backgrounds */
    [data-testid="stSelectbox"] > div > div,
    .stTextArea textarea,
    .stTextInput input {
        background-color: #262730 !important;
        color: #fafafa !important;
    }
    </style>
""", unsafe_allow_html=True)
st.markdown("""
    <style>
    @keyframes glow-pulse {
        0%, 100% {
            text-shadow: 
                0 0 5px rgba(161, 196, 253, 0.15),
                0 0 10px rgba(161, 196, 253, 0.1);
        }
        50% {
            text-shadow: 
                0 0 8px rgba(161, 196, 253, 0.25),
                0 0 15px rgba(161, 196, 253, 0.15),
                0 0 25px rgba(194, 166, 254, 0.1);
        }
    }
    @keyframes gradient-shift {
        0%, 100% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
    }
    @keyframes drip-fall {
        0% { transform: translateY(0); opacity: 0.7; }
        100% { transform: translateY(60px); opacity: 0; }
    }
    @keyframes drip-spawn {
        0%, 100% { opacity: 0; }
        10%, 40% { opacity: 0.6; }
    }
    .delyrism-header {
        text-align: center;
        padding: 1rem 0 2rem 0;
        margin-bottom: 1rem;
        position: relative;
        overflow: visible;
    }
    .delyrism-title-container {
        position: relative;
        display: inline-block;
    }
    .delyrism-title {
        font-size: 4.5rem;
        font-weight: 200;
        letter-spacing: 0.15em;
        background: linear-gradient(120deg, #ffffff 0%, #a1c4fd 50%, #c2a6fe 100%);
        background-size: 200% 200%;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin: 0;
        line-height: 1.1;
        animation: glow-pulse 4s ease-in-out infinite, gradient-shift 8s ease infinite;
        filter: drop-shadow(0 0 4px rgba(161, 196, 253, 0.15));
        position: relative;
    }
    .drip-container {
        position: absolute;
        top: 100%;
        left: 0;
        right: 0;
        height: 60px;
        pointer-events: none;
        overflow: hidden;
    }
    .drip {
        position: absolute;
        font-size: 1.2rem;
        font-weight: 200;
        background: linear-gradient(180deg, #a1c4fd 0%, transparent 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        animation: drip-fall 3s ease-in infinite, drip-spawn 3s ease-in-out infinite;
        opacity: 0;
    }
    .drip:nth-child(1) { left: 8%; animation-delay: 0s; }
    .drip:nth-child(2) { left: 22%; animation-delay: 0.7s; }
    .drip:nth-child(3) { left: 35%; animation-delay: 1.4s; }
    .drip:nth-child(4) { left: 48%; animation-delay: 2.1s; }
    .drip:nth-child(5) { left: 62%; animation-delay: 0.3s; }
    .drip:nth-child(6) { left: 75%; animation-delay: 1.8s; }
    .drip:nth-child(7) { left: 88%; animation-delay: 1.1s; }
    .drip:nth-child(8) { left: 15%; animation-delay: 2.5s; }
    .drip:nth-child(9) { left: 55%; animation-delay: 0.9s; }
    .drip:nth-child(10) { left: 82%; animation-delay: 2.3s; }
    .delyrism-subtitle {
        font-size: 1.1rem;
        color: #8899a6;
        letter-spacing: 0.4em;
        text-transform: uppercase;
        margin-top: 0.2rem;
    }
    
    /* Sidebar Expander Styling with Markers */
    
    /* 1. Data */
    div[data-testid="stVerticalBlock"] > div:has(span#section-data) + div details > summary {
        background-color: rgba(52, 152, 219, 0.15) !important;
        border-left: 3px solid #3498db !important;
        color: #3498db !important;
    }
    div[data-testid="stVerticalBlock"] > div:has(span#section-data) + div details > summary:hover {
        color: #3498db !important;
    }
    div[data-testid="stVerticalBlock"] > div:has(span#section-data) + div details > summary svg {
        fill: #3498db !important;
    }

    /* 2. Embeddings */
    div[data-testid="stVerticalBlock"] > div:has(span#section-embeddings) + div details > summary {
        background-color: rgba(155, 89, 182, 0.15) !important;
        border-left: 3px solid #9b59b6 !important;
        color: #9b59b6 !important;
    }
    div[data-testid="stVerticalBlock"] > div:has(span#section-embeddings) + div details > summary:hover {
        color: #9b59b6 !important;
    }
    div[data-testid="stVerticalBlock"] > div:has(span#section-embeddings) + div details > summary svg {
        fill: #9b59b6 !important;
    }

    /* 3. Context */
    div[data-testid="stVerticalBlock"] > div:has(span#section-context) + div details > summary {
        background-color: rgba(46, 204, 113, 0.15) !important;
        border-left: 3px solid #2ecc71 !important;
        color: #2ecc71 !important;
    }
    div[data-testid="stVerticalBlock"] > div:has(span#section-context) + div details > summary:hover {
        color: #2ecc71 !important;
    }
    div[data-testid="stVerticalBlock"] > div:has(span#section-context) + div details > summary svg {
        fill: #2ecc71 !important;
    }

    /* 4. Semantic Map */
    div[data-testid="stVerticalBlock"] > div:has(span#section-map) + div details > summary {
        background-color: rgba(241, 196, 15, 0.15) !important;
        border-left: 3px solid #f1c40f !important;
        color: #f1c40f !important;
    }
    div[data-testid="stVerticalBlock"] > div:has(span#section-map) + div details > summary:hover {
        color: #f1c40f !important;
    }
    div[data-testid="stVerticalBlock"] > div:has(span#section-map) + div details > summary svg {
        fill: #f1c40f !important;
    }

    /* 5. Ranking */
    div[data-testid="stVerticalBlock"] > div:has(span#section-ranking) + div details > summary {
        background-color: rgba(230, 126, 34, 0.15) !important;
        border-left: 3px solid #e67e22 !important;
        color: #e67e22 !important;
    }
    div[data-testid="stVerticalBlock"] > div:has(span#section-ranking) + div details > summary:hover {
        color: #e67e22 !important;
    }
    div[data-testid="stVerticalBlock"] > div:has(span#section-ranking) + div details > summary svg {
        fill: #e67e22 !important;
    }

    /* 6. Contextual Subgraph */
    div[data-testid="stVerticalBlock"] > div:has(span#section-subgraph) + div details > summary {
        background-color: rgba(231, 76, 60, 0.15) !important;
        border-left: 3px solid #e74c3c !important;
        color: #e74c3c !important;
    }
    div[data-testid="stVerticalBlock"] > div:has(span#section-subgraph) + div details > summary:hover {
        color: #e74c3c !important;
    }
    div[data-testid="stVerticalBlock"] > div:has(span#section-subgraph) + div details > summary svg {
        fill: #e74c3c !important;
    }

    /* 7. Delta Graph */
    div[data-testid="stVerticalBlock"] > div:has(span#section-delta) + div details > summary {
        background-color: rgba(26, 188, 156, 0.15) !important;
        border-left: 3px solid #1abc9c !important;
        color: #1abc9c !important;
    }
    div[data-testid="stVerticalBlock"] > div:has(span#section-delta) + div details > summary:hover {
        color: #1abc9c !important;
    }
    div[data-testid="stVerticalBlock"] > div:has(span#section-delta) + div details > summary svg {
        fill: #1abc9c !important;
    }

    /* Hide the marker containers to prevent layout spacing issues */
    div[data-testid="stVerticalBlock"] > div:has(span[id^="section-"]) {
        display: none;
    }
    
    .console-label {
        font-size: 0.65rem;
        letter-spacing: 0.15em;
        text-transform: uppercase;
        color: rgba(161, 196, 253, 0.6);
        margin-bottom: 0.5rem;
        font-weight: 500;
    }
    
    .console-label.blue {
        color: #3498db !important;
    }
    
    .console-label.green {
        color: #2ecc71 !important;
    }
    </style>
    <div class="delyrism-header">
        <div class="delyrism-title-container">
            <div class="delyrism-title">DELYRISM</div>
            <div class="drip-container">
                <span class="drip">ᛞ</span>
                <span class="drip">ᛖ</span>
                <span class="drip">ᛚ</span>
                <span class="drip">ʏ</span>
                <span class="drip">ᚱ</span>
                <span class="drip">ɪ</span>
                <span class="drip">ꜱ</span>
                <span class="drip">ᛗ</span>
                <span class="drip">◌</span>
                <span class="drip">∿</span>
            </div>
        </div>
        <div class="delyrism-subtitle">🧭 Archetype Explorer</div>
    </div>
""", unsafe_allow_html=True)

# ========== MAIN CONSOLE: Symbolic Structure & Context ==========
# Scan for structure files
structures_dir = pathlib.Path(__file__).parent / "structures"
structure_files = sorted([f.name for f in structures_dir.glob("*.json")]) if structures_dir.exists() else []

# Nice display names for structure files
STRUCTURE_DISPLAY_NAMES = {
    "elements.json": "🜂 Four Elements",
    "planets.json": "☉ Celestial Planets",
    "jungian.json": "🜏 Jungian Archetypes",
    "lakota.json": "🪶 Lakota Spirits",
    "chakras.json": "◉ Chakra System",
    "chinese_zodiac.json": "🐉 Chinese Zodiac",
    "mayan.json": "𐊗 Mayan Calendar",
    "musical.json": "♪ Musical Modes",
    "architecture.json": "🏛 Sacred Architecture",
    "seasons_life.json": "🌱 Seasons of Life",
}

def get_display_name(filename: str) -> str:
    """Convert filename to nice display name."""
    if filename in STRUCTURE_DISPLAY_NAMES:
        return STRUCTURE_DISPLAY_NAMES[filename]
    # Fallback: convert filename to title case
    name = filename.replace(".json", "").replace("_", " ").title()
    return f"📄 {name}"

def get_filename_from_display(display_name: str) -> str:
    """Convert display name back to filename."""
    for fname, dname in STRUCTURE_DISPLAY_NAMES.items():
        if dname == display_name:
            return fname
    # Fallback: try to reconstruct filename
    name = display_name.lstrip("📄 ").lower().replace(" ", "_") + ".json"
    return name

# Build options with nice names
structure_display_opts = ["✦ Custom"] + [get_display_name(f) for f in structure_files]

# Add EGREGORE option if imported
if "_egregore_imported" in st.session_state:
    structure_display_opts.insert(1, "⚗️ EGREGORE (Mined)")

# Callback to load preset into session state
def on_structure_change():
    sel = st.session_state.get("structure_select")
    if sel and sel != "✦ Custom":
        # Handle EGREGORE special case
        if sel == "⚗️ EGREGORE (Mined)" and "_egregore_structure" in st.session_state:
            st.session_state["symbol_json_text"] = json.dumps(
                st.session_state["_egregore_structure"], indent=2, ensure_ascii=False
            )
        else:
            filename = get_filename_from_display(sel)
            p = structures_dir / filename
            if p.exists():
                st.session_state["symbol_json_text"] = p.read_text(encoding="utf-8")

# Initialize JSON text in session state if missing
if "symbol_json_text" not in st.session_state:
    def_path = structures_dir / "elements.json"
    if def_path.exists():
        st.session_state["symbol_json_text"] = def_path.read_text(encoding="utf-8")
    else:
        st.session_state["symbol_json_text"] = json.dumps(_default_symbols_map(), indent=2, ensure_ascii=False)

# Check for pending archetypes from Corpus Miner "Use in Explorer" button
if "_pending_archetypes" in st.session_state:
    st.session_state["symbol_json_text"] = st.session_state.pop("_pending_archetypes")
    st.toast("✅ Archetypes loaded into Explorer!", icon="🎯")

# Main console layout with styled containers
console_col1, console_col2 = st.columns([1, 2])

with console_col1:
    with st.container(border=True):
        st.markdown('''
            <style>
            [data-testid="stVerticalBlockBorderWrapper"]:has(.console-label.blue) {
                border-color: rgba(52, 152, 219, 0.5) !important;
                background: linear-gradient(135deg, rgba(52, 152, 219, 0.1) 0%, rgba(15, 18, 28, 0.95) 100%) !important;
            }
            </style>
            <div class="console-label blue">⬡ SYMBOLIC STRUCTURE</div>
        ''', unsafe_allow_html=True)
        
        # Determine default index - prefer EGREGORE if imported
        if "_egregore_imported" in st.session_state and "⚗️ EGREGORE (Mined)" in structure_display_opts:
            def_idx = structure_display_opts.index("⚗️ EGREGORE (Mined)")
        else:
            try:
                def_idx = structure_display_opts.index(get_display_name("elements.json"))
            except ValueError:
                def_idx = 0
        
        st.selectbox(
            "Structure",
            structure_display_opts,
            index=def_idx,
            key="structure_select",
            on_change=on_structure_change,
            label_visibility="collapsed"
        )

with console_col2:
    with st.container(border=True):
        st.markdown('''
            <style>
            [data-testid="stVerticalBlockBorderWrapper"]:has(.console-label.green) {
                border-color: rgba(46, 204, 113, 0.5) !important;
                background: linear-gradient(135deg, rgba(46, 204, 113, 0.1) 0%, rgba(15, 18, 28, 0.95) 100%) !important;
            }
            </style>
            <div class="console-label green">◈ CONTEXT PROMPT</div>
        ''', unsafe_allow_html=True)
        sentence = st.text_area(
            "Context",
            value="Flooding spirits dancing around floating suns",
            placeholder="e.g., A ceremony by the river focusing on transformation and healing",
            height=68,
            key="ctx_sentence",
            label_visibility="collapsed"
        )

# Load symbols map
symbols_map = _load_symbols_map(st.session_state.get("symbol_json_text", "{}"))

st.session_state.setdefault("mm_items", [])  # list[MMItem-like dicts]

with st.sidebar:
    
    # --- Symbolic Structure ----------------------------------------------------
    st.markdown('<span id="section-data"></span>', unsafe_allow_html=True)
    with st.expander("Symbolic Structure", expanded=False):
        st.caption("Import/export & edit archetype structures")
        
        # File uploader
        uploaded = st.file_uploader("Import JSON", type=["json"], key="json_uploader")
        if uploaded is not None:
            content = uploaded.getvalue().decode("utf-8")
            u_hash = hashlib.md5(content.encode("utf-8")).hexdigest()
            # Only update if this is a new upload (hash check)
            if st.session_state.get("last_upload_hash") != u_hash:
                st.session_state["symbol_json_text"] = content
                st.session_state["last_upload_hash"] = u_hash
                st.rerun()
        
        # JSON Editor
        st.text_area(
            "JSON Editor",
            key="symbol_json_text",
            height=150,
            help="Edit the archetype structure directly"
        )
        
        # Export button
        st.download_button(
            "📥 Export JSON",
            data=st.session_state.get("symbol_json_text", "{}"),
            file_name="archetypes.json",
            mime="application/json",
            use_container_width=True
        )

    # --- Context Options --------------------------
    # Ensure backend/embedder are defined for audio checks (they are selected in Embeddings below)
    backend = st.session_state.get("backend_selection", "qwen3")
    model_val = st.session_state.get("embedding_model", "")
    pooling_val = st.session_state.get("embedding_pooling", "eos")
    embedder = get_embedder(backend, model_val or None, pooling_val)

    st.markdown('<span id="section-context"></span>', unsafe_allow_html=True)
    with st.expander("Context Options", expanded=False):
        st.caption("Audio context & advanced options")
        
        # ADD (audio): optional audio context (upload or record)
        audio_ctx_vec = None
        with st.popover("🎤 Audio Context"):
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
                        use_container_width='stretch',
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
        default_ctx = st.session_state.get("ctx_chosen", [])
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
            
        # --- Alchemist Mode (Context B) ---
        st.divider()
        enable_alchemist = st.checkbox("⚗️ Enable Alchemist Mode (Context B)", False, help="Mix two contexts together.")
        
        ctx_b_sentence = ""
        ctx_b_weights = {}
        
        if enable_alchemist:
            st.markdown("#### Context B (Secondary)")
            ctx_b_sentence = st.text_area(
                "Context B prompt",
                value="",
                placeholder="e.g., A chaotic storm of entropy and decay",
                height=70,
                key="ctx_b_sentence",
            )
            
            ctx_b_chosen = st.multiselect(
                "Context B symbols",
                options=sym_preview,
                default=[],
                key="ctx_b_chosen",
            )
            
            for s in ctx_b_chosen:
                ctx_b_weights[s] = st.slider(
                    f"Weight B: {s}",
                    0.0, 1.0,
                    value=0.5,
                    step=0.05,
                    key=f"wb_{s}",
                )
    # st.markdown('</div>', unsafe_allow_html=True)
    # --- Embeddings ----------------------------------------------
    # --- Embedding Model (sidebar) ---
    st.markdown('<span id="section-embeddings"></span>', unsafe_allow_html=True)
    with st.expander("Embedding Model", expanded=False):
        # ADD (audio): include audioclip as a backend option
        backend_help = (
            "Which encoder turns your inputs into vectors.\n"
            "- Sentence-Transformer: good general text embeddings (e.g., all-mpnet-base-v2).\n"
            "- Qwen2/Qwen3 Embedding: strong multilingual; uses token pooling (EOS by default).\n"
            "- AudioCLIP / CLAP: enable AUDIO → vector (and text, for AudioCLIP). Use only if you need audio.\n"
            "Changing backend re-embeds descriptors and context; results, dims, and speed can change."
        )
        
        def on_backend_change():
            # Auto-calibrate Beta (Shift Strength) based on model sensitivity
            b = st.session_state.backend_selection
            if b in ["sentence-transformer", "qwen2"]:
                st.session_state["beta_slider"] = 0.5  # Lower beta for high-similarity models
            else:
                st.session_state["beta_slider"] = 1.2  # Higher beta for Qwen3/others

        backend = st.selectbox(
            "Backend", 
            ["qwen3", "qwen2", "sentence-transformer", "clap"], 
            index=0, 
            help=backend_help, 
            key="backend_selection",
            on_change=on_backend_change
        )
        if backend in ["qwen3", "qwen2"]:
            st.warning("⚠️ Qwen models require ~2.5GB RAM. If on a free/starter cloud instance, this may crash or run very slowly. Use 'sentence-transformer' for speed.")
        hf_model_help = (
            "Hugging Face repo ID for the embedding model (e.g., 'sentence-transformers/all-mpnet-base-v2', "
            "'Qwen/Qwen2-Embedding', 'Qwen/Qwen3-Embedding-0.6B').\n"
            "Pick an *embedding* model (not a causal LM) to get fixed-size vectors.\n"
            "Changing this will re-embed everything (dimension, quality, and speed can differ)."
        )
        model = st.text_input("HF model override (optional)", help=hf_model_help, key="embedding_model")
        pooling_help = (
            "How token embeddings are collapsed into one vector per text.\n"
            "- eos (default): last *non-padding* token. Length-safe and works well for Qwen-style encoders.\n"
            "- mean: mask-aware average over all real tokens. Very robust for sentence embeddings.\n"
            "- cls: first token ([CLS]). Best when the model was trained to use CLS (e.g., BERT family).\n"
            "- last: final position regardless of padding/truncation. Can be brittle—avoid unless you need it.\n"
            "All pooled vectors are L2-normalized. Keep the same setting when comparing runs.\n\n"
            "⚠️ Pooling only applies to Qwen models. SentenceTransformer/CLAP use their own built-in pooling."
        )
        pooling_disabled = backend in ["sentence-transformer", "clap"]
        pooling_index = 1 if pooling_disabled else 0  # default to 'mean' label when disabled
        pooling = st.selectbox(
            "Pooling", ["eos", "mean", "cls", "last"], 
            index=pooling_index, 
            help=pooling_help, 
            key="embedding_pooling",
            disabled=pooling_disabled
        )
        if pooling_disabled:
            st.caption(f"ℹ️ {backend} uses its own internal pooling — this setting is ignored.")
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

    # --- Shift settings (panel-colored sliders) -------------------
    st.markdown('<span id="section-map"></span>', unsafe_allow_html=True)
    with st.expander("Semantic Map", expanded=False):
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
        
    # st.markdown('</div>', unsafe_allow_html=True)

    # --- Ranking (panel-colored sliders) --------------------------
    st.markdown('<span id="section-ranking"></span>', unsafe_allow_html=True)
    with st.expander("Ranking (proposal)", expanded=False):
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

    # st.markdown('</div>', unsafe_allow_html=True)

    # --- Contextual Subgraph (panel-colored sliders) --------------
    st.markdown('<span id="section-subgraph"></span>', unsafe_allow_html=True)
    with st.expander("Contextual Subgraph (network)", expanded=False):
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
    # st.markdown('</div>', unsafe_allow_html=True)

    # --- Δ Graph (panel-colored sliders) --------------------------
    st.markdown('<span id="section-delta"></span>', unsafe_allow_html=True)
    with st.expander("Δ Graph", expanded=False):
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

        beta = st.slider("Shift strength β", 0.0, 2.0, 1.2, 0.05, disabled=(shift_mode == "pooling"), key="beta_slider")  # β not used by pooling
        
        
        st.markdown("**Graph network settings**")
        within_symbol = st.checkbox("Within-symbol pairs only", False)
        sym_filter_sel = st.multiselect("Or restrict to symbols", sym_preview)
        top_abs_edges = st.slider("Top |Δ| edges", 2, 100, 10, 1)
        min_abs_delta = st.slider("Min |Δ| threshold", 0.0001, 0.1, 0.01, 0.0005, 
                                   format="%.4f",
                                   help="Lower = more edges appear. Raise if graph too cluttered. Try 0.001 or lower if no graph appears.")
        connected_only = st.checkbox("Connected nodes only", True)
        
    # st.markdown('</div>', unsafe_allow_html=True)

# after you compute 'symbols_map' and 'embedder'
st.session_state["_current_symbols_map"] = symbols_map
st.session_state["_current_embedder"] = embedder

space = build_space(
    _symbols_map_key(symbols_map),
    descriptor_threshold,
    _embedder_key(embedder),  # ← includes instruction/context
    "v2_blind_spot"
)
# Store space in session state for fragments to access
st.session_state["_current_space"] = space

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

tab_explore, tab_story, tab_mine = st.tabs(["Explorer", "Story Generator", "Corpus Miner"])

with tab_explore:
    color_map = space.get_symbol_color_dict("AuroraPop")

    # =============================
    # Alchemist / Explorer Fragment
    # =============================
    @fragment
    def render_explorer_view(
        space, 
        ctx_weights, sentence, 
        ctx_b_weights, ctx_b_sentence, enable_alchemist,
        tau, lam, alpha, use_ppr,
        show_arrow, with_hulls, include_centroids, normalize_centroids,
        gate, beta, membership_alpha, within_symbol_softmax,
        ctx_focus, ctx_topk_symbols, ctx_topk_desc, ctx_method, ctx_alpha, ctx_normalize,
        shift_mode, pool_type, pool_w, gamma,
        audio_vec,
        mix_t, mix_mode, fluid_mode
    ):
        # --- 1. Compute Mixed Context Vector ---
        # If Alchemist mode is ON, we blend A and B.
        # If OFF, we just use A (which is passed as `ctx_weights`, `sentence`).
        
        final_vec = None
        
        # Pre-declare variables for Fluid Mode
        v_a_fluid = None
        v_b_fluid = None
        
        if enable_alchemist:
            # Compute vectors for A and B
            # We must ensure we don't use a stale override from a previous fragment run.
            
            # Vector A
            if audio_vec is not None:
                v_a = audio_vec.copy()
            else:
                space.set_context_vec(None)
                v_a = space.ctx_vec(weights=ctx_weights, sentence=sentence)

            # Vector B
            space.set_context_vec(None)
            v_b = space.ctx_vec(weights=ctx_b_weights, sentence=ctx_b_sentence)
            
            # Store for Fluid Mode
            v_a_fluid = v_a
            v_b_fluid = v_b
            
            # Blend
            if mix_mode == "Morph (A→B)":
                # Linear interpolation: (1-t)A + tB
                v_mix = (1.0 - mix_t) * v_a + mix_t * v_b
            elif mix_mode == "Infuse (A+tB)":
                # Additive: A + tB
                v_mix = v_a + mix_t * v_b
            elif mix_mode == "Mask (A-tB)":
                # Subtractive: A - tB
                v_mix = v_a - mix_t * v_b
            else:
                v_mix = v_a

            # Normalize result
            n = np.linalg.norm(v_mix)
            if n > 1e-9:
                final_vec = v_mix / n
            else:
                final_vec = v_mix # zero
                
            # Apply to space as override
            space.set_context_vec(final_vec)
            
            # For plotting functions below, we pass None for weights/sentence 
            # because the vector is already set in the space override.
            p_weights = None
            p_sentence = None
            
            # FORCE "gate" strategy if Alchemist is active, because "hybrid" requires text re-embedding
            # which we cannot do with a pure vector mix.
            if shift_mode in ("hybrid", "reembed"):
                shift_mode = "gate"
            
        else:
            # Standard mode: restore original context (Audio or None)
            space.set_context_vec(audio_vec)
            p_weights = ctx_weights
            p_sentence = sentence

        # =============================
        # Row 1: Meaning Space | Ambiguity Metrics
        # =============================
        c1, c2 = st.columns(2)

        with c1:
            st.subheader("Meaning Space (2D)")
            reducer = st.selectbox("Reducer", ["umap", "tsne","pca"], index=0)
            
            if fluid_mode and enable_alchemist:
                # --- FLUID MODE (Altair) ---
                import altair as alt
                import pandas as pd
                
                with st.spinner("Computing fluid interpolation..."):
                    # 1. Compute Shifted Matrices for A and B
                    # We need to temporarily set the context vector to A, then B
                    
                    # State A
                    space.set_context_vec(v_a_fluid)
                    D_a = space.make_shifted_matrix(
                        weights=None, sentence=None,
                        strategy=shift_mode, beta=beta, gate=gate, tau=tau,
                        within_symbol_softmax=within_symbol_softmax, gamma=gamma,
                        pool_type=pool_type, pool_w=pool_w, membership_alpha=membership_alpha
                    )
                    
                    # State B
                    space.set_context_vec(v_b_fluid)
                    D_b = space.make_shifted_matrix(
                        weights=None, sentence=None,
                        strategy=shift_mode, beta=beta, gate=gate, tau=tau,
                        within_symbol_softmax=within_symbol_softmax, gamma=gamma,
                        pool_type=pool_type, pool_w=pool_w, membership_alpha=membership_alpha
                    )
                    
                    # Restore mixed vector
                    space.set_context_vec(final_vec)
                    
                    # 2. Project to 2D
                    # We must fit on the BASE space (space.D) to ensure a common coordinate system
                    # Note: space.reduce_2d fits on space.D. We need to access the fitted reducer.
                    # But space.reduce_2d returns the transformed array, it doesn't return the reducer object easily unless we hack it.
                    # Actually, space._pca is stored. But UMAP/t-SNE are not stored in the class instance persistently in a way we can reuse easily 
                    # unless we re-fit or modify delyrism.py.
                    # However, for PCA, we can use space._pca.
                    # For UMAP, we have to fit a new one on space.D.
                    
                    # Use cached reducer from SymbolSpace
                    red, _, _, _ = space.get_cached_reducer_and_projection(
                        method=reducer, n_neighbors=15, 
                        include_centroids=include_centroids, normalize_centroids=normalize_centroids
                    )
                    
                    if reducer == "tsne":
                        st.warning("t-SNE does not support fluid interpolation well (no transform). Using PCA fallback.")
                        red, _, _, _ = space.get_cached_reducer_and_projection(method="pca")
                        
                    XY_a = red.transform(D_a)
                    XY_b = red.transform(D_b)
                    
                    # 3. Build DataFrame
                    # We need symbol labels and colors
                    # space.descriptors is the list of descriptors
                    # space.owner maps descriptor -> symbol
                    
                    data = []
                    for i, desc in enumerate(space.descriptors):
                        sym = space.owner[desc]
                        data.append({
                            "desc": desc,
                            "symbol": sym,
                            "xA": XY_a[i, 0], "yA": XY_a[i, 1],
                            "xB": XY_b[i, 0], "yB": XY_b[i, 1],
                            "color": color_map.get(sym, "#333333")
                        })
                    df_fluid = pd.DataFrame(data)
                    
                    # 4. Altair Chart
                    slider = alt.binding_range(min=0, max=1, step=0.01, name='Mix (t): ')
                    t_param = alt.param(bind=slider, value=mix_t, name='t')
                    
                    # Interpolation expression
                    # x = xA * (1-t) + xB * t
                    
                    chart = alt.Chart(df_fluid).mark_circle(size=60).encode(
                        x=alt.X('x:Q', axis=None),
                        y=alt.Y('y:Q', axis=None),
                        color=alt.Color('color:N', scale=None),
                        tooltip=['symbol', 'desc']
                    ).transform_calculate(
                        x = f"datum.xA * (1 - t) + datum.xB * t",
                        y = f"datum.yA * (1 - t) + datum.yB * t"
                    ).add_params(
                        t_param
                    ).properties(
                        width=600,
                        height=450,
                        title="Fluid Context Morph (Client-Side)"
                    ).configure_view(
                        stroke=None
                    ).interactive()
                    
                    st.altair_chart(chart, use_container_width=True)
                    st.caption("💡 Drag the slider **below the chart** (or use the sidebar) to morph.")

            else:
                # --- STANDARD MODE ---
                # Visualization style options
                with st.expander("🎨 Visualization Style", expanded=False):
                    viz_style = st.radio(
                        "Render Engine",
                        ["Modern (Plotly)", "Classic (Matplotlib)"],
                        horizontal=True,
                        key="viz_style_map"
                    )
                    
                    vcol1, vcol2, vcol3 = st.columns(3)
                    with vcol1:
                        point_size = st.slider("Point Size", 4, 20, 8, key="viz_point_size")
                        show_labels = st.checkbox("Show Labels", False, key="viz_show_labels")
                    with vcol2:
                        line_width = st.slider("Line Width", 0.5, 3.0, 1.0, 0.5, key="viz_line_width")
                        smooth_arrows = st.checkbox("Smooth Arrows", True, key="viz_smooth_arrows")
                    with vcol3:
                        region_style = st.selectbox(
                            "Region Style",
                            ["None", "Density Glow", "Soft Contours", "Gradient Fields"],
                            index=3,
                            key="viz_region_style"
                        )
                        region_opacity = st.slider("Region Opacity", 0.1, 0.6, 0.2, 0.05, key="viz_region_opacity")
                
                if show_arrow is True:
                    arrow_scale = 0.5
                if show_arrow is False:
                    arrow_scale = 0
                
                if viz_style == "Modern (Plotly)":
                    # --- MODERN PLOTLY MODE ---
                    try:
                        import plotly.graph_objects as go
                        import pandas as pd
                        from scipy.stats import gaussian_kde
                        from scipy.ndimage import gaussian_filter
                        
                        # Get projections from space
                        reducer_obj, Z_fit, X_fit, slices = space.get_cached_reducer_and_projection(
                            method=reducer, n_neighbors=15,
                            include_centroids=include_centroids, normalize_centroids=normalize_centroids
                        )
                        
                        # Compute shifted matrix
                        D_ctx = space.make_shifted_matrix(
                            weights=p_weights, sentence=p_sentence,
                            strategy=shift_mode, beta=beta, gate=gate, tau=tau,
                            within_symbol_softmax=within_symbol_softmax, gamma=gamma,
                            pool_type=pool_type, pool_w=pool_w, membership_alpha=membership_alpha
                        )
                        
                        # Transform shifted descriptors
                        if reducer != "tsne":
                            Z_ctx = reducer_obj.transform(D_ctx)
                        else:
                            Z_ctx = Z_fit[:len(space.descriptors)]  # fallback
                        
                        # Build original positions using slices from get_cached_reducer_and_projection
                        # slices maps symbol -> (start, end) in the DESCRIPTOR part of Z_fit
                        # Z_fit layout: [sym1_descs, (sym1_centroid), sym2_descs, (sym2_centroid), ...]
                        # We need to map from slices indices to Z_fit indices, accounting for centroids
                        
                        # First, build a mapping from global descriptor index to Z_fit position
                        Z_orig = np.zeros((len(space.descriptors), 2), dtype=np.float32)
                        centroid_positions = {}  # symbol -> (x, y) position of centroid in Z_fit
                        
                        z_cursor = 0
                        for s in space.symbols:
                            idx = space.symbol_to_idx[s]
                            n_desc = len(idx)
                            if n_desc > 0:
                                # Extract positions from Z_fit
                                Z_sym = Z_fit[z_cursor:z_cursor + n_desc]
                                # Assign to global positions
                                Z_orig[idx] = Z_sym
                                z_cursor += n_desc
                                
                                if include_centroids:
                                    # The centroid position is right after the descriptors
                                    centroid_positions[s] = Z_fit[z_cursor]
                                    z_cursor += 1
                        
                        # Create Plotly figure
                        fig = go.Figure()
                        
                        # Compute tight bounds from ALL visible data
                        # Get only the actually-used descriptor positions (non-zero check as safety)
                        used_mask = np.any(Z_orig != 0, axis=1) | (Z_orig.sum(axis=1) != 0)  # True if row has any non-zero
                        # Actually safer: use the positions we know are filled
                        all_positions = []
                        for s in space.symbols:
                            idx = space.symbol_to_idx[s]
                            if len(idx) > 0:
                                all_positions.append(Z_orig[idx])
                                if include_centroids and s in centroid_positions:
                                    all_positions.append(centroid_positions[s].reshape(1, 2))
                        
                        if all_positions:
                            all_pos = np.vstack(all_positions)
                            x_data_min, x_data_max = float(all_pos[:, 0].min()), float(all_pos[:, 0].max())
                            y_data_min, y_data_max = float(all_pos[:, 1].min()), float(all_pos[:, 1].max())
                        else:
                            x_data_min, x_data_max = 0.0, 1.0
                            y_data_min, y_data_max = 0.0, 1.0
                        
                        # Also include shifted positions if arrows are shown
                        if arrow_scale > 0 and all_positions:
                            all_ctx_positions = []
                            for s in space.symbols:
                                idx = space.symbol_to_idx[s]
                                if len(idx) > 0:
                                    all_ctx_positions.append(Z_ctx[idx])
                            if all_ctx_positions:
                                all_ctx = np.vstack(all_ctx_positions)
                                x_data_min = min(x_data_min, float(all_ctx[:, 0].min()))
                                x_data_max = max(x_data_max, float(all_ctx[:, 0].max()))
                                y_data_min = min(y_data_min, float(all_ctx[:, 1].min()))
                                y_data_max = max(y_data_max, float(all_ctx[:, 1].max()))
                        
                        # Add padding (5% for tighter zoom)
                        x_range = x_data_max - x_data_min
                        y_range = y_data_max - y_data_min
                        x_pad = x_range * 0.05 if x_range > 1e-6 else 0.5
                        y_pad = y_range * 0.05 if y_range > 1e-6 else 0.5
                        
                        x_plot_min = float(x_data_min - x_pad)
                        x_plot_max = float(x_data_max + x_pad)
                        y_plot_min = float(y_data_min - y_pad)
                        y_plot_max = float(y_data_max + y_pad)
                        
                        # Add region visualization FIRST (so it's behind points)
                        if region_style != "None":
                            # Create grid for density estimation - USE CENTERED BOUNDS
                            grid_size = 60
                            xx = np.linspace(x_plot_min, x_plot_max, grid_size)
                            yy = np.linspace(y_plot_min, y_plot_max, grid_size)
                            XX, YY = np.meshgrid(xx, yy)
                            grid_points = np.vstack([XX.ravel(), YY.ravel()])
                            
                            for s in space.symbols:
                                idx = space.symbol_to_idx[s]
                                if len(idx) < 3:
                                    continue
                                
                                color = color_map.get(s, "#888888")
                                points = Z_orig[idx]
                                
                                try:
                                    # Compute kernel density
                                    kde = gaussian_kde(points.T, bw_method=0.3)
                                    Z_density = kde(grid_points).reshape(XX.shape)
                                    
                                    # Smooth and normalize
                                    Z_density = gaussian_filter(Z_density, sigma=1.5)
                                    Z_density = Z_density / Z_density.max() if Z_density.max() > 0 else Z_density
                                    
                                    if region_style == "Density Glow":
                                        # Single filled contour with glow effect
                                        fig.add_trace(go.Contour(
                                            x=xx, y=yy, z=Z_density,
                                            showscale=False,
                                            contours=dict(
                                                start=0.15,
                                                end=0.9,
                                                size=0.25,
                                                coloring='fill'
                                            ),
                                            colorscale=[
                                                [0, f'rgba(0,0,0,0)'],
                                                [0.3, color.replace(')', f',{region_opacity * 0.3})').replace('rgb', 'rgba') if 'rgb' in color else f'rgba(128,128,128,{region_opacity * 0.3})'],
                                                [0.6, color.replace(')', f',{region_opacity * 0.6})').replace('rgb', 'rgba') if 'rgb' in color else f'rgba(128,128,128,{region_opacity * 0.6})'],
                                                [1, color.replace(')', f',{region_opacity})').replace('rgb', 'rgba') if 'rgb' in color else f'rgba(128,128,128,{region_opacity})']
                                            ],
                                            line=dict(width=0),
                                            hoverinfo='skip',
                                            showlegend=False
                                        ))
                                    
                                    elif region_style == "Soft Contours":
                                        # Line contours only
                                        fig.add_trace(go.Contour(
                                            x=xx, y=yy, z=Z_density,
                                            showscale=False,
                                            contours=dict(
                                                start=0.2,
                                                end=0.8,
                                                size=0.3,
                                                coloring='lines'
                                            ),
                                            line=dict(width=1.5, color=color),
                                            opacity=region_opacity * 2,
                                            hoverinfo='skip',
                                            showlegend=False
                                        ))
                                    
                                    elif region_style == "Gradient Fields":
                                        # Heatmap style with transparency
                                        # Convert hex to rgba
                                        import matplotlib.colors as mcolors
                                        try:
                                            rgb = mcolors.to_rgb(color)
                                        except:
                                            rgb = (0.5, 0.5, 0.5)
                                        
                                        # Create custom colorscale
                                        colorscale = [
                                            [0, 'rgba(0,0,0,0)'],
                                            [0.5, f'rgba({int(rgb[0]*255)},{int(rgb[1]*255)},{int(rgb[2]*255)},{region_opacity * 0.5})'],
                                            [1, f'rgba({int(rgb[0]*255)},{int(rgb[1]*255)},{int(rgb[2]*255)},{region_opacity})']
                                        ]
                                        
                                        fig.add_trace(go.Heatmap(
                                            x=xx, y=yy, z=Z_density,
                                            showscale=False,
                                            colorscale=colorscale,
                                            hoverinfo='skip',
                                            showlegend=False,
                                            zmin=0.1,
                                            zmax=1.0
                                        ))
                                        
                                except Exception:
                                    pass  # Skip density if KDE fails
                        
                        # Add traces per symbol (points on top)
                        for s in space.symbols:
                            idx = space.symbol_to_idx[s]
                            if not idx:
                                continue
                            
                            color = color_map.get(s, "#888888")
                            descs = [space.descriptors[i] for i in idx]
                            
                            # Original points
                            fig.add_trace(go.Scatter(
                                x=Z_orig[idx, 0],
                                y=Z_orig[idx, 1],
                                mode='markers+text' if show_labels else 'markers',
                                marker=dict(
                                    size=point_size,
                                    color=color,
                                    opacity=0.85,
                                    line=dict(width=0.5, color='rgba(255,255,255,0.3)')
                                ),
                                text=descs if show_labels else None,
                                textposition='top center',
                                textfont=dict(size=8, color='rgba(255,255,255,0.7)'),
                                name=s,
                                legendgroup=s,
                                hovertemplate=f'<b>{s}</b><br>%{{text}}<extra></extra>' if show_labels else f'<b>{s}</b><extra></extra>',
                            ))
                            
                            # Arrows showing shift
                            if arrow_scale > 0:
                                for i, desc_idx in enumerate(idx):
                                    x0, y0 = Z_orig[desc_idx]
                                    x1, y1 = Z_ctx[desc_idx]
                                    dx = (x1 - x0) * arrow_scale
                                    dy = (y1 - y0) * arrow_scale
                                    
                                    if smooth_arrows:
                                        # Smooth bezier-like arrow using line with gradient
                                        fig.add_trace(go.Scatter(
                                            x=[x0, x0 + dx],
                                            y=[y0, y0 + dy],
                                            mode='lines',
                                            line=dict(
                                                width=line_width,
                                                color=color,
                                            ),
                                            opacity=0.5,
                                            showlegend=False,
                                            hoverinfo='skip'
                                        ))
                                    else:
                                        # Arrow annotation
                                        fig.add_annotation(
                                            x=x0 + dx, y=y0 + dy,
                                            ax=x0, ay=y0,
                                            xref='x', yref='y',
                                            axref='x', ayref='y',
                                            showarrow=True,
                                            arrowhead=2,
                                            arrowsize=0.8,
                                            arrowwidth=line_width,
                                            arrowcolor=color,
                                            opacity=0.5
                                        )
                            
                            # Centroid star
                            if include_centroids and s in centroid_positions:
                                c_orig = centroid_positions[s]
                                c_ctx = Z_ctx[idx].mean(0)
                                fig.add_trace(go.Scatter(
                                    x=[float(c_orig[0])],
                                    y=[float(c_orig[1])],
                                    mode='markers',
                                    marker=dict(
                                        size=14,
                                        symbol='star',
                                        color=color,
                                        line=dict(width=1, color='white')
                                    ),
                                    showlegend=False,
                                    legendgroup=s,
                                    hovertemplate=f'<b>{s} centroid</b><extra></extra>'
                                ))
                                if arrow_scale > 0:
                                    fig.add_trace(go.Scatter(
                                        x=[float(c_orig[0]), float(c_ctx[0])],
                                        y=[float(c_orig[1]), float(c_ctx[1])],
                                        mode='lines',
                                        line=dict(width=2, color=color, dash='dot'),
                                        opacity=0.6,
                                        showlegend=False,
                                        hoverinfo='skip'
                                    ))
                        
                        # Layout styling
                        axis_labels = {
                            "umap": ("UMAP 1", "UMAP 2"),
                            "tsne": ("t-SNE 1", "t-SNE 2"),
                            "pca": ("PCA 1", "PCA 2")
                        }
                        xl, yl = axis_labels.get(reducer, ("Dim 1", "Dim 2"))
                        
                        fig.update_layout(
                            xaxis=dict(
                                title=xl,
                                showgrid=True,
                                gridcolor='rgba(255,255,255,0.06)',
                                gridwidth=1,
                                zeroline=False,
                                color='rgba(255,255,255,0.5)',
                                tickfont=dict(color='rgba(255,255,255,0.4)', size=10),
                                range=[x_plot_min, x_plot_max],
                                autorange=False,
                            ),
                            yaxis=dict(
                                title=yl,
                                showgrid=True,
                                gridcolor='rgba(255,255,255,0.06)',
                                gridwidth=1,
                                zeroline=False,
                                color='rgba(255,255,255,0.5)',
                                tickfont=dict(color='rgba(255,255,255,0.4)', size=10),
                                range=[y_plot_min, y_plot_max],
                                autorange=False,
                            ),
                            plot_bgcolor='rgba(0,0,0,0)',
                            paper_bgcolor='rgba(0,0,0,0)',
                            legend=dict(
                                orientation='h',
                                yanchor='bottom',
                                y=1.02,
                                xanchor='left',
                                x=0,
                                font=dict(size=13, color='rgba(255,255,255,0.95)'),
                                bgcolor='rgba(0,0,0,0)'
                            ),
                            margin=dict(l=50, r=20, t=40, b=50),
                            height=480,
                            hovermode='closest'
                        )
                        
                        st.plotly_chart(fig, use_container_width=True, config={
                            'displayModeBar': True,
                            'displaylogo': False,
                            'modeBarButtonsToRemove': ['lasso2d', 'select2d']
                        })
                        
                    except Exception as e:
                        st.error(f"Plotly visualization error: {e}")
                        import traceback
                        st.code(traceback.format_exc())
                
                else:
                    # --- CLASSIC MATPLOTLIB MODE ---
                    try:
                        fig_ms = fig_from_callable(
                            space.plot_map_shift,
                            weights=p_weights,
                            sentence=p_sentence,
                            method=reducer,
                            with_hulls=with_hulls,
                            include_centroids=include_centroids,
                            normalize_centroids=normalize_centroids,
                            figsize=(6.8, 4.5),
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
                        
                        # Apply dark theme
                        fig_ms.patch.set_alpha(0.0)
                        for ax in fig_ms.axes:
                            ax.patch.set_alpha(0.0)
                            ax.xaxis.label.set_color('white')
                            ax.yaxis.label.set_color('white')
                            ax.title.set_color('white')
                            ax.tick_params(axis='x', colors='white')
                            ax.tick_params(axis='y', colors='white')
                            for spine in ax.spines.values():
                                spine.set_color((1, 1, 1, 0.2))
                            if ax.get_legend():
                                leg = ax.get_legend()
                                leg.get_frame().set_alpha(0.0)
                                plt.setp(leg.get_texts(), color='white')
                        
                        st.pyplot(fig_ms, clear_figure=True)
                    except Exception as e:
                        st.error(f"Map plot error: {e}")

        with c2:
            st.subheader("Ambiguity Metrics")
            sort_opt = st.selectbox("Sort by", ["dispersion", "leakage", "entropy", "none"], index=0, key="amb_sort")
            fig_amb = plot_ambiguity_metrics(space, sort_by=sort_opt, color_dict=color_map, figsize=(7.5, 4.0))
            
            # Apply dark theme styling manually
            fig_amb.patch.set_alpha(0.0)
            for ax in fig_amb.axes:
                ax.patch.set_alpha(0.0)
                ax.xaxis.label.set_color('white')
                ax.yaxis.label.set_color('white')
                ax.title.set_color('white')
                ax.tick_params(axis='x', colors='white')
                ax.tick_params(axis='y', colors='white')
                for spine in ax.spines.values():
                    spine.set_visible(False)
                if ax.get_legend():
                    legend = ax.get_legend()
                    legend.get_frame().set_alpha(0.0)
                    legend.get_frame().set_linewidth(0.0)
                    plt.setp(legend.get_texts(), color='white')
                    if legend.get_title():
                        legend.get_title().set_color('white')
                    
            st.pyplot(fig_amb, clear_figure=True)
            plt.close(fig_amb)

        st.divider()

        # =============================
        # Row 2: Descriptor Attention | Top Symbols
        # =============================
        c3, c4 = st.columns(2)

        with c3:
            st.subheader("Descriptor Attention")
            att_blind_spot = st.checkbox("Blind Spots", False, key="att_blind_spot", help="Show least attended descriptors")
            
            # Container for the plot to appear ABOVE the selectbox
            att_container = st.container()
            
            sym = st.selectbox("Symbol", list(space.symbols), key="att_sym_select")
                
            if sym:
                try:
                    fig_att = fig_from_callable(
                        space.plot_attention,
                        sym,
                        weights=p_weights,
                        sentence=p_sentence,
                        tau=tau,
                        top_n=8,
                        figsize=(6, 4.0),
                        blind_spot=att_blind_spot
                    )
                    with att_container:
                        st.pyplot(fig_att, clear_figure=True)
                except Exception as e:
                    st.warning(f"Attention plot failed: {e}")

        with c4:
            st.subheader("Top Symbols for Context")
            rank_blind_spot = st.checkbox("Show Anti-Matches (Blind Spots)", False, key="rank_blind_spot", help="Show symbols most distant from the context")
            try:
                preds = space.propose(
                    weights=p_weights,
                    sentence=p_sentence,
                    tau=tau,
                    lam=lam,
                    alpha=alpha,
                    topk=len(space.symbols),
                    use_ppr=use_ppr,
                    blind_spot=rank_blind_spot
                )

                if preds:
                    # Exclude symbols explicitly in the context weights (only if using direct weights)
                    exclude = set()
                    if p_weights:
                        exclude = {k.lower() for k in p_weights.keys()}
                    
                    # (optional) also exclude symbols literally mentioned in the sentence
                    ctx_words = set()
                    if p_sentence:
                        ctx_words = {w.strip(".,;:!?()[]{}\"'").lower() for w in p_sentence.split()}
                        
                    preds = [p for p in preds if (p[0].lower() not in exclude and p[0].lower() not in ctx_words)]

                    if not preds:
                        st.info("All top symbols are part of the context.")
                    else:
                        labels = [p[0] for p in preds]
                        scores = np.array([p[1] for p in preds])

                        norm = (scores - scores.min()) / (scores.max() - scores.min() + 1e-8)
                        cmap = plt.cm.coolwarm
                        colors = [cmap(n) for n in norm]

                        fig_rank, ax = plt.subplots(figsize=(6, 4.0))
                        # Apply dark theme styling manually since we don't use fig_from_callable here
                        fig_rank.patch.set_alpha(0.0)
                        ax.patch.set_alpha(0.0)
                        ax.xaxis.label.set_color('white')
                        ax.yaxis.label.set_color('white')
                        ax.title.set_color('white')
                        ax.tick_params(axis='x', colors='white')
                        ax.tick_params(axis='y', colors='white')
                        for spine in ax.spines.values():
                            spine.set_visible(False)

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
                        title_suffix = " (Anti-Matches)" if rank_blind_spot else ""
                        ax.set_title(f"Symbol prediction for context{title_suffix}")

                        for bar, v in zip(bars, scores[::-1]):
                            ax.add_patch(plt.Rectangle(
                                (bar.get_x(), bar.get_y()), bar.get_width(), bar.get_height(),
                                color='white', alpha=0.08, zorder=0
                            ))

                        fig_rank.tight_layout()
                        st.pyplot(fig_rank, clear_figure=True)
                        plt.close(fig_rank)
                else:
                    st.info("No predictions yet.")
            except Exception as e:
                st.warning(f"Ranking failed: {e}")

        st.divider()

        # =============================
        # Row 3: Contextual Subgraph (network) | Heatmaps
        # =============================

        colN, colH = st.columns([1.1, 1])

        with colN:
            st.subheader("Network View — Contextual Subgraph")
            try:
                tau_subgraph = focus_to_tau(ctx_focus)
                # Build a stable global color palette once
                cmap = plt.cm.tab20
                # global_color_map = {s: cmap(i / max(1, len(space.symbols)-1)) for i, s in enumerate(space.symbols)}
                
                # If Alchemist mode is on, p_sentence is None, so we provide a label
                subgraph_label = p_sentence or ("(Alchemist Mix)" if enable_alchemist else "")
                
                fig_ctxnet = fig_from_callable(
                    plot_contextual_subgraph_colored,
                    space,
                    context_sentence=subgraph_label,
                    topk_symbols=ctx_topk_symbols,
                    topk_desc=ctx_topk_desc,
                    method=ctx_method,
                    alpha=ctx_alpha,
                    tau=tau_subgraph,
                    normalize=ctx_normalize,
                    global_color_map=color_map,
                    figsize=(7, 5)
                )
                st.pyplot(fig_ctxnet, clear_figure=True)
            except Exception as e:
                st.warning(f"Contextual subgraph failed: {e}")

        with colH:
            st.subheader("Within-Symbol Associative Increase (Δ)")
            
            # Container for the plot to appear ABOVE the selectbox
            plot_container = st.container()
            
            sym2 = st.selectbox("Symbol for heatmaps", list(space.symbols), key="heat_sym")
            
            if sym2:
                try:
                    simdict = space.descriptor_similarity_matrices(
                        weights=p_weights,
                        sentence=p_sentence,
                        strategy=shift_mode,
                        beta=beta,
                        gate=gate,
                        tau=(tau if tau is not None else 0.5),  # pick a default if not softmax
                        within_symbol_softmax=within_symbol_softmax,
                        gamma=gamma,
                        pool_type=pool_type,
                        pool_w=pool_w,
                        order_by_attention=True,
                        membership_alpha=membership_alpha,
                    )


                    # New Strategy: Interactive Altair Heatmap
                    import pandas as pd
                    import altair as alt

                    data = simdict[sym2]
                    matrix = data["S_delta"]
                    labels = data["descriptors"]

                    # Convert to long format for Altair
                    df_heat = pd.DataFrame(matrix, index=labels, columns=labels)
                    df_heat.index.name = "Row"
                    df_heat.columns.name = "Col"
                    df_long = df_heat.stack().reset_index(name="Delta")

                    # Determine color domain excluding diagonal (self-correlations often skew the range)
                    mask_nd = df_long['Row'] != df_long['Col']
                    if mask_nd.any():
                        min_d = df_long.loc[mask_nd, "Delta"].min()
                        max_d = df_long.loc[mask_nd, "Delta"].max()
                    else:
                        min_d, max_d = df_long["Delta"].min(), df_long["Delta"].max()

                    # Custom palette: Blue -> Green -> Yellow -> Orange -> Red
                    if max_d - min_d < 1e-9:
                        scale = alt.Scale(domain=[min_d, max_d], range=['#3498DB', '#3498DB'])
                    else:
                        step = (max_d - min_d) / 4.0
                        dom = [min_d, min_d + step, min_d + 2*step, min_d + 3*step, max_d]
                        # Colors: Blue, Green, Yellow, Orange, Red
                        scale = alt.Scale(domain=dom, range=['#3498DB', '#2ECC71', '#FFD700', '#FF9F1C', '#E74C3C'])

                    chart = alt.Chart(df_long).mark_rect().encode(
                        x=alt.X('Col', sort=labels, title=None, axis=alt.Axis(labelAngle=-90, labelFontSize=10, labelColor='white', titleColor='white')),
                        # Force all labels to appear by disabling overlap checks
                        y=alt.Y('Row', sort=labels, title=None, axis=alt.Axis(labelFontSize=10, labelOverlap=False, labelColor='white', titleColor='white')),
                        color=alt.Color('Delta', scale=scale, title="Δ", legend=alt.Legend(titleColor='white', labelColor='white')),
                        tooltip=['Row', 'Col', alt.Tooltip('Delta', format='.4f')]
                    ).properties(
                        title=alt.TitleParams(text=f"Δ After-Before for {sym2}", color='white'),
                        width=550,
                        height=550
                    ).configure_axis(
                        grid=False,
                        domainColor='white',
                        tickColor='white'
                    ).configure_view(
                        strokeWidth=0
                    ).configure(
                        background='transparent'
                    )

                    with plot_container:
                        st.altair_chart(chart, use_container_width=False)
                except Exception as e:
                    st.warning(f"Heatmaps failed: {e}")

    # --- Alchemist Mixing Controls (Outside Fragment) ---
    mix_t = 0.5
    mix_mode = "Morph (A→B)"
    fluid_mode = False
    
    if enable_alchemist:
        with st.sidebar:
            st.markdown("---")
            st.markdown("#### ⚗️ Mixing Desk")
            mix_t = st.slider("Interpolation (t)", 0.0, 1.0, 0.5, 0.01, key="mix_t")
            mix_mode = st.selectbox("Operation", ["Morph (A→B)", "Infuse (A+tB)", "Mask (A-tB)"], key="mix_mode")
            fluid_mode = st.checkbox("Fluid Mode (Experimental)", False, help="Use Altair for 60fps client-side interpolation. May be slower to load initially.")

    # --- Call the fragment ---
    render_explorer_view(
        space, 
        ctx_weights, sentence, 
        ctx_b_weights, ctx_b_sentence, enable_alchemist,
        tau, lam, alpha, use_ppr,
        show_arrow, with_hulls, include_centroids, normalize_centroids,
        gate, beta, membership_alpha, within_symbol_softmax,
        ctx_focus, ctx_topk_symbols, ctx_topk_desc, ctx_method, ctx_alpha, ctx_normalize,
        shift_mode, pool_type, pool_w, gamma,
        audio_vec if (backend in ("audioclip", "clap")) else None,
        mix_t, mix_mode, fluid_mode
    )

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
                min_abs_delta=min_abs_delta,
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
                figsize=(7.0, 2.0),        # ⬅️ smaller figure
                node_size_base=130,        # ⬇️ shrink nodes
                node_size_scale=700.0,
                edge_width_min=0.4,        # ⬅️ thinner edges
                edge_width_max=3.0,
            )
            if G.number_of_edges() == 0:
                st.info("📊 No edges in Δ graph. Try: lower 'Min |Δ| threshold', increase 'Shift strength β', add a context sentence, or uncheck 'Within-symbol pairs only'.")
            else:
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
    # =============================
    # Story Generator Fragment
    # =============================
    # Store context params in session state so the fragment can access them
    # without triggering a full app rerun
    st.session_state["_story_ctx"] = {
        "sentence": sentence,
        "ctx_weights": ctx_weights,
        "shift_mode": shift_mode,
        "beta": beta,
        "gate": gate,
        "tau_gate": tau_gate,
        "within_symbol_softmax": within_symbol_softmax,
        "gamma": gamma,
        "pool_type": pool_type,
        "pool_w": pool_w,
        "top_abs_edges": top_abs_edges,
        "sym_filter_sel": sym_filter_sel,
        "within_symbol": within_symbol,
        "connected_only": connected_only,
        "membership_alpha": membership_alpha,
        "descriptor_threshold": descriptor_threshold,
        "min_abs_delta": min_abs_delta,
        "embedder_key": _embedder_key(embedder),
        "symbols_key": _symbols_map_key(symbols_map),
    }
    
    @fragment
    def render_story_generator():
        """Isolated fragment for story generation - won't trigger Explorer recomputation."""
        # Pull context from session state (set by main app before fragment runs)
        ctx = st.session_state.get("_story_ctx", {})
        _sentence = ctx.get("sentence", "")
        _ctx_weights = ctx.get("ctx_weights", {})
        _shift_mode = ctx.get("shift_mode", "gate")
        _beta = ctx.get("beta", 1.2)
        _gate = ctx.get("gate", "relu")
        _tau_gate = ctx.get("tau_gate", 0.5)
        _within_symbol_softmax = ctx.get("within_symbol_softmax", True)
        _gamma = ctx.get("gamma", 0.5)
        _pool_type = ctx.get("pool_type", "avg")
        _pool_w = ctx.get("pool_w", 0.7)
        _top_abs_edges = ctx.get("top_abs_edges", 10)
        _sym_filter_sel = ctx.get("sym_filter_sel", [])
        _within_symbol = ctx.get("within_symbol", False)
        _connected_only = ctx.get("connected_only", True)
        _membership_alpha = ctx.get("membership_alpha", 0.0)
        _descriptor_threshold = ctx.get("descriptor_threshold", 0.1)
        _min_abs_delta = ctx.get("min_abs_delta", 0.01)
        _embedder_key = ctx.get("embedder_key", "")
        _symbols_key = ctx.get("symbols_key", "")
        
        # Get the space from session state
        _space = st.session_state.get("_current_space")
        if _space is None:
            st.warning("Space not initialized. Please configure settings in the sidebar first.")
            return
        
        st.markdown("### ✨ Generative Storytelling Engine")
        st.caption("Derive short story from the current context and Δ-graph motifs.")

        # --- Controls live inside a FORM so nothing "executes" until submit ---
        with st.form("story_form", clear_on_submit=False):
            
            # --- Top: Model Settings (Hidden by default for ergonomics) ---
            with st.expander("🧠 Model Configuration", expanded=False):
                # Backend toggle: Local vs Cloudflare
                inference_backend = st.radio(
                    "Inference Backend",
                    ["☁️ Cloudflare Workers AI (fast, free tier)", "💻 Local (HuggingFace models)"],
                    index=0,
                    horizontal=True,
                    help="Cloudflare: fast cloud inference, no GPU needed. Local: runs on your machine/server."
                )
                use_cloudflare = "Cloudflare" in inference_backend
                
                if use_cloudflare:
                    st.info("💡 Cloudflare Workers AI provides free inference. Set `CLOUDFLARE_ACCOUNT_ID` and `CLOUDFLARE_API_TOKEN` in environment or Streamlit secrets.")
                    c_cf1, c_cf2 = st.columns([3, 1])
                    with c_cf1:
                        cf_preset = st.selectbox("Cloudflare Model", list(CLOUDFLARE_MODELS.keys()), index=0)
                    with c_cf2:
                        st.write("")  # spacer
                    cf_model_id = CLOUDFLARE_MODELS[cf_preset]
                    st.caption(f"Model: `{cf_model_id}`")
                    # Cloudflare doesn't need 8-bit or local settings
                    use_8bit = False
                    model_id = cf_model_id
                else:
                    c_mod1, c_mod2 = st.columns([3, 1])
                    GEMMA_MODEL_PRESETS = {
                        "Qwen2.5 (0.5B-Instruct)": "Qwen/Qwen2.5-0.5B-Instruct",
                        "Qwen2.5 (1.5B-Instruct)": "Qwen/Qwen2.5-1.5B-Instruct",
                        "TinyLlama (1.1B-Chat)": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
                        "SmolLM2 (360M-Instruct)": "HuggingFaceTB/SmolLM2-360M-Instruct",
                        "─── Gated (need HF login) ───": None,
                        "Gemma 2 (2B-IT)":  "google/gemma-2-2b-it",
                        "Gemma 3 (1B-IT)":  "google/gemma-3-1b-it",
                        "Gemma 3n (E2B-IT)": "google/gemma-3n-e2b-it",
                    }
                    with c_mod1:
                        preset = st.selectbox("Model preset", list(GEMMA_MODEL_PRESETS.keys()), index=0)
                    with c_mod2:
                        st.write("") # spacer
                        st.write("") 
                        use_8bit = st.checkbox("8-bit Quant.", False, help="Lowers VRAM usage.")
                    
                    default_model_id = GEMMA_MODEL_PRESETS[preset] or ""
                    if GEMMA_MODEL_PRESETS[preset] is None:
                        st.warning("⚠️ Select a model above — this is just a separator.")
                    model_id = st.text_input("Repo ID", value=default_model_id, help="Hugging Face repo ID")

            # --- Middle: Creative Controls ---
            c_left, c_right = st.columns(2)
            
            with c_left:
                st.markdown("#### 📜 Narrative Structure")
                language = st.selectbox("Language", ["English", "Français", "Español"], index=0)
                
                c_l1, c_l2 = st.columns(2)
                with c_l1:
                    pov = st.selectbox("POV", ["first", "third"], index=0)
                with c_l2:
                    tense = st.selectbox("Tense", ["present", "past"], index=0)
                
                story_len_words = st.slider("Length (words)", 80, 500, 180, 20)

            with c_right:
                st.markdown("#### 🎨 Atmosphere & Chaos")
                tone = st.selectbox(
                    "Tone Style",
                    ["dreamy", "eerie", "warm", "pynchon", "blake", "mystic-baroque", "gnostic-techno"],
                    index=0
                )
                
                c_r1, c_r2 = st.columns(2)
                with c_r1:
                    temperature = st.slider("Temp (Creativity)", 0.1, 1.8, 0.85, 0.05)
                with c_r2:
                    top_p = st.slider("Top-p (Focus)", 0.1, 1.0, 0.9, 0.05)
                
                pos_only = st.checkbox("Positive Δ edges only", True, help="Only use strengthening connections as motifs.")

            st.markdown("---")
            submit = st.form_submit_button("🔮 Generate Story", type="primary", width='stretch')

        # --- Only generate when the button was pressed ---
        if submit:
            with st.spinner("Generating..."):
                # Build Δ graph only on demand using fragment-local variables
                _sym_filter_arg = _sym_filter_sel if _sym_filter_sel else None
                _tau_gate_eff = _tau_gate if (_gate == "softmax" and _tau_gate is not None) else 0.5

                key_story = _delta_key(
                    _sentence, _ctx_weights, _shift_mode, _beta, _gate, _tau_gate_eff,
                    _within_symbol_softmax, _gamma, _pool_type, _pool_w, _top_abs_edges, _sym_filter_sel,
                    _within_symbol, _connected_only, _membership_alpha, _descriptor_threshold,
                    _embedder_key, _symbols_key,
                )

                G_story = None
                if st.session_state.get("delta_graph_key") == key_story and "delta_graph" in st.session_state:
                    G_story = st.session_state["delta_graph"]
                else:
                    G_story = context_delta_graph(
                        _space,
                        sentence=_sentence or None,
                        weights=_ctx_weights or None,
                        strategy=_shift_mode,
                        beta=_beta,
                        gate=_gate,
                        tau=_tau_gate_eff,
                        within_symbol_softmax=_within_symbol_softmax,
                        gamma=_gamma,
                        pool_type=_pool_type,
                        pool_w=_pool_w,
                        top_abs_edges=_top_abs_edges,
                        min_abs_delta=_min_abs_delta,
                        sym_filter=_sym_filter_sel if _sym_filter_sel else None,
                        within_symbol=_within_symbol,
                        connected_only=_connected_only,
                        membership_alpha=_membership_alpha,
                    )
                    st.session_state["delta_graph"] = G_story
                    st.session_state["delta_graph_key"] = key_story
                
                motifs = top_motifs_from_delta_graph(G_story, k_nodes=12, positive_only=pos_only)

                # Build prompt (same for both backends)
                messages = build_gemma_prompt(
                    context_sentence=_sentence or "",
                    motifs=motifs,
                    tone=tone, pov=pov, tense=tense,
                    target_words=(story_len_words-20, story_len_words+20),
                    language=language,
                )
                # Words → tokens: ~1.5 tokens per word on average, plus buffer for completion
                estimated_tokens = int(story_len_words * 1.6) + 80

                # === Cloudflare Workers AI ===
                if use_cloudflare:
                    try:
                        story = generate_with_cloudflare(
                            messages,
                            model=model_id,
                            max_tokens=estimated_tokens,
                            temperature=temperature,
                            top_p=top_p,
                        )
                        st.success(f"✅ Generated via Cloudflare (`{model_id}`)")
                    except Exception as e:
                        st.error(f"Cloudflare API error: {e}")
                        story = ""
                
                # === Local HuggingFace model ===
                else:
                    # Free VRAM from embedder, then lazy-load model
                    import gc, torch as _torch
                    try:
                        getattr(_space.embedder, "to", lambda *_: None)("cpu")
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
           
                    story = generate_with_gemma(
                        tok, mdl, messages,
                        max_new_tokens=estimated_tokens,
                        temperature=temperature,
                        top_p=top_p,
                        repetition_penalty=1.05
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
    
    # Call the fragment
    render_story_generator()

with tab_mine:
    st.subheader("⚗️ Egregore — Archetype Builder")
    
    # Mode selector
    miner_mode = st.radio(
        "Builder Mode",
        ["🚀 Real-time Builder (Recommended)", "📋 Classic Streamlit Builder"],
        horizontal=True,
        help="Real-time Builder uses WebSocket for instant feedback. Classic mode runs in Streamlit (slower, blocks UI)."
    )
    
    if miner_mode == "🚀 Real-time Builder (Recommended)":
        # ========== EGREGORE IFRAME EMBED ==========
        st.markdown("""
        <style>
        .egregore-container {
            border: 1px solid rgba(194, 166, 254, 0.3);
            border-radius: 12px;
            overflow: hidden;
            background: linear-gradient(135deg, rgba(194, 166, 254, 0.05) 0%, rgba(10, 10, 15, 0.95) 100%);
        }
        .egregore-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 0.5rem 1rem;
            background: rgba(194, 166, 254, 0.1);
            border-bottom: 1px solid rgba(194, 166, 254, 0.2);
        }
        .egregore-status {
            display: flex;
            align-items: center;
            gap: 0.5rem;
            font-size: 0.75rem;
            color: #a1a1aa;
        }
        .status-dot {
            width: 8px;
            height: 8px;
            border-radius: 50%;
            background: #ef4444;
        }
        .status-dot.connected {
            background: #22c55e;
            box-shadow: 0 0 8px #22c55e;
        }
        </style>
        """, unsafe_allow_html=True)
        
        # Get Egregore URL from environment or default to localhost
        # On Railway: set EGREGORE_URL to the Egregore service's public domain
        # Locally: defaults to localhost:8765
        egregore_url = os.environ.get("EGREGORE_URL", "http://localhost:8765")
        
        # Ensure URL has protocol
        if not egregore_url.startswith("http"):
            egregore_url = f"https://{egregore_url}"
        
        col_iframe, col_actions = st.columns([4, 1])
        
        # Detect if running on Railway (separate services = no file sharing)
        is_railway = os.environ.get("RAILWAY_ENVIRONMENT") or os.environ.get("EGREGORE_URL")
        
        with col_actions:
            st.markdown("#### Actions")
            
            # Button to open in new tab
            st.markdown(f'<a href="{egregore_url}" target="_blank"><button style="width:100%; padding:0.5rem; margin-bottom:0.5rem; cursor:pointer;">🔗 Open Full Screen</button></a>', unsafe_allow_html=True)
            
            st.divider()
            
            st.markdown("##### 📥 Import Archetypes")
            
            if is_railway:
                # Railway mode: use manual paste (file sharing not possible between services)
                st.caption("Copy JSON from Egregore, paste below:")
                
                manual_json = st.text_area(
                    "Paste archetypes JSON",
                    height=120,
                    placeholder='{"ARCHETYPE": ["desc1", ...], ...}',
                    key="railway_egregore_json"
                )
                if st.button("📥 Import", type="primary", use_container_width=True, key="btn_railway_import"):
                    if manual_json.strip():
                        try:
                            arch_data = json.loads(manual_json)
                            if isinstance(arch_data, dict) and len(arch_data) > 0:
                                st.session_state["_pending_archetypes"] = json.dumps(arch_data, indent=2, ensure_ascii=False)
                                st.session_state["_egregore_structure"] = arch_data
                                st.session_state["_egregore_imported"] = True
                                st.success(f"✅ Imported {len(arch_data)} archetypes!")
                                st.rerun()
                            else:
                                st.error("Empty or invalid JSON")
                        except json.JSONDecodeError as e:
                            st.error(f"Invalid JSON: {e}")
                    else:
                        st.warning("Paste JSON first")
            else:
                # Local mode: use file-based transfer
                st.caption("Click 'Open in Explorer' in Egregore, then:")
                
                egregore_export_path = pathlib.Path(__file__).parent / ".egregore_export.json"
                
                if st.button("📥 Import Archetypes", type="primary", use_container_width=True, key="btn_import_egregore"):
                    if egregore_export_path.exists():
                        try:
                            arch_data = json.loads(egregore_export_path.read_text(encoding="utf-8"))
                            if isinstance(arch_data, dict) and len(arch_data) > 0:
                                st.session_state["_pending_archetypes"] = json.dumps(arch_data, indent=2, ensure_ascii=False)
                                st.session_state["_egregore_structure"] = arch_data
                                st.session_state["_egregore_imported"] = True
                                egregore_export_path.unlink()
                                st.success(f"✅ Imported {len(arch_data)} archetypes!")
                                st.toast("Archetypes loaded! Switch to Explorer tab.", icon="🎯")
                                st.rerun()
                            else:
                                st.error("Empty or invalid archetypes file")
                        except Exception as e:
                            st.error(f"Failed to import: {e}")
                    else:
                        st.warning("No archetypes found. Click 'Open in Explorer' in Egregore first.")
                
                # Show file status
                if egregore_export_path.exists():
                    try:
                        preview = json.loads(egregore_export_path.read_text(encoding="utf-8"))
                        st.success(f"✅ {len(preview)} archetypes ready!")
                    except:
                        pass
                
                # Manual fallback for local too
                with st.expander("📋 Manual paste", expanded=False):
                    manual_json = st.text_area(
                        "Paste JSON",
                        height=100,
                        placeholder='{"ARCHETYPE": ["desc1", ...], ...}',
                        key="manual_egregore_json"
                    )
                    if st.button("Import", key="btn_manual_import"):
                        if manual_json.strip():
                            try:
                                arch_data = json.loads(manual_json)
                                if isinstance(arch_data, dict):
                                    st.session_state["_pending_archetypes"] = json.dumps(arch_data, indent=2, ensure_ascii=False)
                                    st.session_state["_egregore_structure"] = arch_data
                                    st.session_state["_egregore_imported"] = True
                                    st.success(f"✅ Imported {len(arch_data)} archetypes!")
                                    st.rerun()
                            except json.JSONDecodeError as e:
                                st.error(f"Invalid JSON: {e}")
            
            # Show if previously imported
            if st.session_state.get("_egregore_imported"):
                st.info("✅ EGREGORE loaded in Explorer")
            
            st.divider()
            st.caption("Egregore server:")
            st.code("uvicorn miner_server:app --port 8765", language="bash")
        
        with col_iframe:
            import streamlit.components.v1 as components
            
            # Embed the Egregore iframe
            components.iframe(egregore_url, height=850, scrolling=False)
    
    else:
        # ========== CLASSIC STREAMLIT MINER ==========
        st.info("💡 For real-time progress monitoring, use the Real-time Builder mode above.")
        
        # Initialize enhanced miner state
        st.session_state.setdefault("enhanced_miner_corpus", {"pdfs": [], "folders": [], "texts": [], "images": []})
        st.session_state.setdefault("miner_embeddings", None)
        st.session_state.setdefault("miner_results", None)
        
        # Import enhanced miner
        try:
            from enhanced_miner import (
                EnhancedArchetypeMiner, LLMArchetypeRefiner, PDFExtractor, 
                TextChunker, FolderScanner, MinerCorpus, VisionDescriber
            )
            _HAS_ENHANCED_MINER = True
        except ImportError as e:
            _HAS_ENHANCED_MINER = False
            st.warning(f"Enhanced miner not available: {e}")
        
        # Check for PDF support
        try:
            import fitz
            _HAS_PYMUPDF = True
        except ImportError:
            _HAS_PYMUPDF = False
        try:
            import pdfplumber
            _HAS_PDFPLUMBER = True
        except ImportError:
            _HAS_PDFPLUMBER = False
        
        _HAS_PDF = _HAS_PYMUPDF or _HAS_PDFPLUMBER

        st.markdown("""
        **Ingest corpora** from PDFs, image folders, or raw text → **cluster by embedding similarity** → 
        **derive archetypes** with LLM refinement.
        """)
        
        # ========== LEFT: Data Ingestion ==========
        col_ingest, col_preview = st.columns([1, 1])
        
        with col_ingest:
            st.markdown("### 📥 Add Data Sources")
            
            # --- PDF Upload ---
            with st.expander("📄 PDF Documents", expanded=True):
                if not _HAS_PDF:
                    st.warning("PDF support requires `pip install pymupdf` or `pip install pdfplumber`")
                else:
                    pdf_files = st.file_uploader(
                        "Upload PDFs", 
                        type=["pdf"], 
                        accept_multiple_files=True,
                        key="pdf_uploader"
                    )
                    
                    st.markdown("**Chunking Strategy**")
                    chunk_strategy = st.selectbox(
                        "How to split text",
                        ["paragraph", "sentence", "sliding", "semantic"],
                        index=0,
                        help=(
                            "• **Paragraph**: Split on double newlines, merge small chunks\n"
                            "• **Sentence**: Group N sentences with optional overlap\n"
                            "• **Sliding**: Fixed character window with stride\n"
                            "• **Semantic**: Merge while embedding similarity is high (uses your embedder)"
                        )
                    )
                
                # Strategy-specific params
                if chunk_strategy == "paragraph":
                    c1, c2 = st.columns(2)
                    chunk_min = c1.number_input("Min chars", 30, 500, 50, key="p_min")
                    chunk_max = c2.number_input("Max chars", 200, 3000, 1000, key="p_max")
                    chunk_kwargs = {"min_length": chunk_min, "max_length": chunk_max}
                    
                elif chunk_strategy == "sentence":
                    c1, c2, c3 = st.columns(3)
                    sents_per = c1.number_input("Sentences/chunk", 2, 20, 5, key="s_per")
                    overlap = c2.number_input("Overlap sentences", 0, 5, 1, key="s_overlap")
                    min_len = c3.number_input("Min chars", 30, 300, 50, key="s_min")
                    chunk_kwargs = {"sentences_per_chunk": sents_per, "overlap_sentences": overlap, "min_length": min_len}
                    
                elif chunk_strategy == "sliding":
                    c1, c2, c3 = st.columns(3)
                    window = c1.number_input("Window size", 128, 2048, 512, key="sl_win")
                    stride = c2.number_input("Stride", 64, 1024, 256, key="sl_stride")
                    min_len = c3.number_input("Min chars", 30, 300, 50, key="sl_min")
                    chunk_kwargs = {"window_size": window, "stride": stride, "min_length": min_len}
                    
                else:  # semantic
                    c1, c2, c3 = st.columns(3)
                    sim_thresh = c1.slider("Similarity threshold", 0.3, 0.95, 0.7, 0.05, key="sem_thresh")
                    min_len = c2.number_input("Min chars", 30, 300, 50, key="sem_min")
                    max_len = c3.number_input("Max chars", 500, 3000, 1500, key="sem_max")
                    chunk_kwargs = {"similarity_threshold": sim_thresh, "min_length": min_len, "max_length": max_len}
                
                if pdf_files and st.button("📄 Process PDFs", key="btn_process_pdf"):
                    with st.spinner("Extracting and chunking PDFs..."):
                        for pdf_file in pdf_files:
                            # Save to temp
                            pdf_path = _save_upload(pdf_file, "pdf")
                            
                            try:
                                pages = PDFExtractor.extract_text(pdf_path)
                                total_chunks = 0
                                
                                for page_data in pages:
                                    text = page_data["text"]
                                    
                                    if chunk_strategy == "paragraph":
                                        chunks = TextChunker.chunk_paragraphs(text, **chunk_kwargs)
                                    elif chunk_strategy == "sentence":
                                        chunks = TextChunker.chunk_sentences(text, **chunk_kwargs)
                                    elif chunk_strategy == "sliding":
                                        chunks = TextChunker.chunk_sliding_window(text, **chunk_kwargs)
                                    else:
                                        # Semantic requires embedder
                                        def _embed_fn(texts):
                                            return embedder.encode(texts)
                                        chunks = TextChunker.chunk_semantic(text, embedder=_embed_fn, **chunk_kwargs)
                                    
                                    for i, chunk_text in enumerate(chunks):
                                        st.session_state["enhanced_miner_corpus"]["texts"].append({
                                            "id": f"{pdf_file.name}_p{page_data['page']}_c{i}",
                                            "text": chunk_text,
                                            "source": pdf_file.name,
                                            "page": page_data["page"],
                                            "strategy": chunk_strategy
                                        })
                                        total_chunks += 1
                                
                                st.session_state["enhanced_miner_corpus"]["pdfs"].append({
                                    "name": pdf_file.name,
                                    "chunks": total_chunks,
                                    "pages": len(pages)
                                })
                                
                            except Exception as e:
                                st.error(f"Failed to process {pdf_file.name}: {e}")
                    
                    st.success(f"Processed {len(pdf_files)} PDF(s)")
                    st.rerun()
        
        # --- Image Folder ---
        with st.expander("🖼️ Image Folder", expanded=False):
            st.caption("Enter a local folder path containing images")
            
            img_folder = st.text_input(
                "Folder path",
                placeholder="C:/path/to/images or /home/user/images",
                key="img_folder_path"
            )
            
            c1, c2 = st.columns(2)
            recursive = c1.checkbox("Scan subfolders", True, key="img_recursive")
            max_images = c2.number_input("Max images", 10, 2000, 500, key="img_max")
            
            # Vision description option
            describe_images = st.checkbox(
                "🔮 Describe images with Vision LLM",
                False,
                key="img_describe",
                help="Use Cloudflare Vision LLM to generate symbolic descriptions for each image. This enables better concept extraction from image-only clusters."
            )
            
            if img_folder and st.button("🖼️ Scan Folder", key="btn_scan_folder"):
                folder_path = pathlib.Path(img_folder)
                if not folder_path.exists():
                    st.error(f"Folder not found: {img_folder}")
                else:
                    with st.spinner("Scanning for images..."):
                        try:
                            images = FolderScanner.scan_images(folder_path, recursive=recursive, max_files=max_images)
                            
                            # If vision description enabled, describe images
                            descriptions = {}
                            if describe_images and images:
                                try:
                                    vision = VisionDescriber(
                                        cloudflare_account_id=CLOUDFLARE_ACCOUNT_ID,
                                        cloudflare_api_token=CLOUDFLARE_API_TOKEN
                                    )
                                    
                                    progress_bar = st.progress(0, text="Describing images with Vision LLM...")
                                    
                                    def update_progress(current, total):
                                        progress_bar.progress(current / total, text=f"Describing image {current}/{total}...")
                                    
                                    descriptions = vision.describe_batch(images, progress_callback=update_progress)
                                    progress_bar.empty()
                                    st.success(f"Generated {len(descriptions)} image descriptions")
                                    
                                except Exception as e:
                                    st.warning(f"Vision description failed: {e}. Continuing without descriptions.")
                            
                            for img in images:
                                img_data = {
                                    "id": img.id,
                                    "path": img.path,
                                    "filename": img.filename,
                                    "source": str(folder_path)
                                }
                                # Include description if available
                                if img.path in descriptions:
                                    img_data["description"] = descriptions[img.path]
                                    
                                st.session_state["enhanced_miner_corpus"]["images"].append(img_data)
                            
                            st.session_state["enhanced_miner_corpus"]["folders"].append({
                                "path": str(folder_path),
                                "count": len(images)
                            })
                            
                            st.success(f"Found {len(images)} images")
                            st.rerun()
                            
                        except Exception as e:
                            st.error(f"Scan failed: {e}")
        
        # --- Raw Text ---
        with st.expander("📝 Raw Text", expanded=False):
            raw_text = st.text_area(
                "Paste text content",
                height=150,
                placeholder="Paste articles, notes, or any text corpus here...",
                key="raw_text_input"
            )
            
            text_source = st.text_input("Source label", "manual_input", key="text_source")
            auto_chunk = st.checkbox("Auto-chunk (paragraphs)", True, key="text_auto_chunk")
            
            if raw_text and st.button("📝 Add Text", key="btn_add_text"):
                if auto_chunk:
                    chunks = TextChunker.chunk_paragraphs(raw_text, min_length=50, max_length=1000)
                else:
                    chunks = [raw_text]
                
                for i, chunk in enumerate(chunks):
                    st.session_state["enhanced_miner_corpus"]["texts"].append({
                        "id": f"{text_source}_{i}",
                        "text": chunk,
                        "source": text_source,
                        "strategy": "paragraph" if auto_chunk else "full"
                    })
                
                st.success(f"Added {len(chunks)} text chunk(s)")
                st.rerun()
        
        # --- Legacy item upload (keep for backwards compat) ---
        with st.expander("📦 Legacy Item Upload", expanded=False):
            st.caption("Original item-by-item upload (for pairing modalities)")
            
            txts = st.text_area("Text items (id:::text per line)", height=80, key="legacy_txt")
            if st.button("Add legacy text", key="btn_legacy_txt"):
                for line in [l.strip() for l in txts.splitlines() if l.strip()]:
                    if ":::" in line:
                        i, t = line.split(":::", 1)
                        item_id, text = i.strip(), t.strip()
                    else:
                        item_id = uuid.uuid4().hex[:8]
                        text = line
                st.session_state["mm_items"].append({"id": item_id, "text": text})
                st.rerun()
    
        # ========== RIGHT: Corpus Preview ==========
        with col_preview:
            st.markdown("### 📊 Corpus Summary")
            
            corpus = st.session_state["enhanced_miner_corpus"]
            n_texts = len(corpus["texts"])
            n_images = len(corpus["images"])
            n_pdfs = len(corpus["pdfs"])
            n_folders = len(corpus["folders"])
            
            # Stats cards
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Text Chunks", n_texts)
            c2.metric("Images", n_images)
            c3.metric("PDFs", n_pdfs)
            c4.metric("Folders", n_folders)
            
            if n_texts + n_images == 0:
                st.info("No data added yet. Use the panels on the left to add PDFs, images, or text.")
            else:
                # Show sources
                with st.expander(f"📄 PDF Sources ({n_pdfs})", expanded=False):
                    for pdf in corpus["pdfs"]:
                        st.write(f"• **{pdf['name']}** — {pdf['pages']} pages, {pdf['chunks']} chunks")
                
                with st.expander(f"🖼️ Image Folders ({n_folders})", expanded=False):
                    for folder in corpus["folders"]:
                        st.write(f"• **{folder['path']}** — {folder['count']} images")
                
                with st.expander(f"📝 Text Chunks Preview ({n_texts})", expanded=False):
                    for i, txt in enumerate(corpus["texts"][:20]):
                        preview = txt["text"][:100] + "..." if len(txt["text"]) > 100 else txt["text"]
                        st.caption(f"**{txt['id']}** ({txt.get('source', '?')}): {preview}")
                    if n_texts > 20:
                        st.caption(f"... and {n_texts - 20} more")
                
                with st.expander(f"🖼️ Images Preview ({n_images})", expanded=False):
                    # Show how many have descriptions
                    n_described = sum(1 for img in corpus["images"] if img.get("description"))
                    if n_described > 0:
                        st.caption(f"🔮 {n_described}/{n_images} images have vision descriptions")
                    
                    # Show thumbnail grid
                    cols = st.columns(5)
                    for i, img in enumerate(corpus["images"][:15]):
                        with cols[i % 5]:
                            try:
                                caption = img["filename"][:15]
                                if img.get("description"):
                                    caption += " 🔮"  # Indicate has description
                                st.image(img["path"], caption=caption, use_container_width=True)
                                if img.get("description"):
                                    st.caption(img["description"][:80] + "..." if len(img.get("description", "")) > 80 else img.get("description", ""))
                            except:
                                st.caption(img["filename"][:20])
                    if n_images > 15:
                        st.caption(f"... and {n_images - 15} more")
                
                # Clear button
                if st.button("🗑️ Clear All Data", key="btn_clear_corpus"):
                    st.session_state["enhanced_miner_corpus"] = {"pdfs": [], "folders": [], "texts": [], "images": []}
                    st.session_state["miner_embeddings"] = None
                    st.session_state["miner_results"] = None
                    st.rerun()
        
        st.divider()
        
        # ========== Mining Configuration ==========
        st.markdown("### ⚙️ Mining Configuration")
        
        col_algo, col_llm = st.columns(2)
        
        with col_algo:
            st.markdown("**Clustering Parameters**")
            c1, c2, c3 = st.columns(3)
            k_neighbors = c1.slider("k-NN neighbors", 3, 50, 10, key="mine_k")
            min_cluster = c2.slider("Min cluster size", 2, 30, 3, key="mine_min")
            resolution = c3.slider("Resolution", 0.1, 3.0, 1.0, 0.1, key="mine_res", 
                                   help="Higher = more clusters")
        
        with col_llm:
            st.markdown("**LLM Refinement**")
            use_llm = st.checkbox("Use LLM for archetype naming", True, key="mine_use_llm",
                                  help="Uses Cloudflare Workers AI to generate proper archetype names and descriptors")
            
            if use_llm:
                llm_model = st.selectbox(
                    "LLM Model",
                    list(CLOUDFLARE_MODELS.keys()),
                    index=0,
                    key="mine_llm_model"
                )
                st.caption(f"Model: `{CLOUDFLARE_MODELS[llm_model]}`")
        
        # ========== Run Mining ==========
        st.markdown("---")
        
        corpus = st.session_state["enhanced_miner_corpus"]
        can_mine = len(corpus["texts"]) + len(corpus["images"]) >= 5
        
        if not can_mine:
            st.warning("Add at least 5 items (text chunks or images) to start mining.")
        
        col_run, col_viz = st.columns([2, 1])
        
        with col_run:
            if st.button("🔬 Run Archetype Mining", type="primary", disabled=not can_mine, use_container_width=True):
                with st.spinner("Mining archetypes from corpus..."):
                    try:
                        # Build text adapter
                        class _MinerTextAdapter:
                            def __init__(self, emb):
                                self.emb = emb
                            def encode(self, texts):
                                return self.emb.encode(texts)
                        
                        text_adapter = _MinerTextAdapter(embedder)
                        
                        # Build image adapter if we have images
                        image_adapter = None
                        if corpus["images"] and Image is not None and open_clip is not None:
                            image_adapter = _ImageAdapter(model_name="ViT-B-32", pretrained="laion2b_s34b_b79k")
                        
                        # Build LLM refiner
                        llm_refiner = None
                        if use_llm:
                            llm_refiner = LLMArchetypeRefiner(
                                backend="cloudflare",
                                model=CLOUDFLARE_MODELS[llm_model]
                            )
                        
                        # Create enhanced miner
                        miner = EnhancedArchetypeMiner(
                            text_encoder=text_adapter,
                            image_encoder=image_adapter,
                            llm_refiner=llm_refiner
                        )
                        
                        # Add text chunks - track count
                        text_count = 0
                        for txt in corpus["texts"]:
                            added = miner.add_text(txt["text"], source=txt.get("source", "unknown"), chunk=False)
                            text_count += added
                        
                        # Add images (they're already scanned)
                        img_count = 0
                        for img in corpus["images"]:
                            miner.corpus.images.append(type('ImageItem', (), {
                                'id': img['id'],
                                'path': img['path'],
                                'source_folder': img.get('source', ''),
                                'filename': img['filename'],
                                'description': img.get('description'),  # Include vision LLM description
                                'meta': {}
                            })())
                            img_count += 1
                        
                        st.info(f"📊 Corpus: {text_count} text chunks, {img_count} images")
                        
                        # Progress bar for mining
                        progress_bar = st.progress(0, text="Starting mining...")
                        
                        def mining_progress(stage: str, progress: float):
                            progress_bar.progress(progress, text=f"🔄 {stage}...")
                        
                        # Capture debug output
                        import io
                        import sys
                        debug_buffer = io.StringIO()
                        old_stdout = sys.stdout
                        sys.stdout = debug_buffer
                        
                        # Run mining with debug and progress enabled
                        archetypes = miner.run(
                            k_neighbors=k_neighbors,
                            min_cluster_size=min_cluster,
                            resolution=resolution,
                            use_llm=use_llm,
                            debug=True,
                            progress_callback=mining_progress
                        )
                        
                        # Restore stdout and get debug output
                        sys.stdout = old_stdout
                        debug_output = debug_buffer.getvalue()
                        
                        progress_bar.progress(1.0, text="✅ Mining complete!")
                        
                        # Store debug output in session for display after rerun
                        st.session_state["miner_debug_log"] = debug_output
                        
                        # Check if we got results
                        if not archetypes:
                            st.warning("No archetypes discovered. Try:\n- Lowering 'Min cluster size'\n- Adding more text/images\n- Increasing 'Resolution'")
                        else:
                            # Store results
                            st.session_state["miner_results"] = archetypes
                            
                            # Get embeddings for visualization
                            ids, embs, types = miner.get_embeddings_for_viz()
                            st.session_state["miner_embeddings"] = {
                                "ids": ids,
                                "embeddings": embs,
                                "types": types
                            }
                            
                            st.success(f"✅ Discovered {len(archetypes)} archetypes!")
                            
                            # Force rerun to show results
                            st.rerun()
                        
                    except Exception as e:
                        st.error(f"Mining failed: {e}")
                        import traceback
                        st.code(traceback.format_exc())
        
        with col_viz:
            if st.button("📈 Visualize Embeddings", disabled=st.session_state.get("miner_embeddings") is None):
                st.session_state["show_miner_viz"] = True
        
        # ========== Results Display ==========
        # Show debug log from last mining run
        if st.session_state.get("miner_debug_log"):
            with st.expander("🔍 Mining Debug Log", expanded=False):
                st.code(st.session_state["miner_debug_log"], language="text")
        
        if st.session_state.get("miner_results"):
            st.markdown("### 🏛️ Discovered Archetypes")
            
            archetypes = st.session_state["miner_results"]
            st.success(f"Discovered **{len(archetypes)}** archetypes from your corpus!")
            
            # Raw JSON preview
            with st.expander("📋 Raw JSON Result", expanded=False):
                st.json(archetypes)
            
            # Display archetypes in a nice grid
            cols = st.columns(2)
            for i, (name, descriptors) in enumerate(archetypes.items()):
                with cols[i % 2]:
                    with st.container(border=True):
                        st.markdown(f"### {name}")
                        st.caption(f"{len(descriptors)} descriptors")
                        st.write(", ".join(descriptors[:12]))
                        if len(descriptors) > 12:
                            with st.expander("See all"):
                                st.write(", ".join(descriptors))
            
            # Actions
            st.markdown("---")
            col_a, col_b, col_c = st.columns(3)
            
            with col_a:
                if st.button("👉 Use in Explorer", type="primary", use_container_width=True):
                    # Store the archetypes to be loaded on next rerun (before widget renders)
                    st.session_state["_pending_archetypes"] = json.dumps(archetypes, indent=2, ensure_ascii=False)
                    st.rerun()
            
            with col_b:
                st.download_button(
                    "💾 Download JSON",
                    data=json.dumps(archetypes, indent=2, ensure_ascii=False).encode("utf-8"),
                    file_name="archetypes.json",
                    mime="application/json",
                    use_container_width=True
                )
            
            with col_c:
                # Editable JSON
                if st.button("✏️ Edit JSON", use_container_width=True):
                    st.session_state["show_edit_archetypes"] = True
            
            if st.session_state.get("show_edit_archetypes"):
                edited = st.text_area(
                    "Edit archetypes JSON",
                    value=json.dumps(archetypes, indent=2, ensure_ascii=False),
                    height=300,
                    key="edit_archetypes_json"
                )
                if st.button("Save edits"):
                    try:
                        st.session_state["miner_results"] = json.loads(edited)
                        st.success("Saved!")
                        st.rerun()
                    except json.JSONDecodeError as e:
                        st.error(f"Invalid JSON: {e}")
        
        # ========== Embedding Visualization ==========
        if st.session_state.get("show_miner_viz") and st.session_state.get("miner_embeddings"):
            st.markdown("### 📊 Embedding Space Visualization")
            
            emb_data = st.session_state["miner_embeddings"]
            embeddings = emb_data["embeddings"]
            ids = emb_data["ids"]
            types = emb_data["types"]
            
            if embeddings.shape[0] > 2:
                viz_method = st.selectbox("Projection", ["UMAP", "t-SNE", "PCA"], key="miner_viz_method")
                
                with st.spinner(f"Computing {viz_method} projection..."):
                    try:
                        if viz_method == "PCA":
                            from sklearn.decomposition import PCA
                            reducer = PCA(n_components=2)
                            coords = reducer.fit_transform(embeddings)
                        elif viz_method == "t-SNE":
                            from sklearn.manifold import TSNE
                            reducer = TSNE(n_components=2, perplexity=min(30, len(ids)-1), random_state=42)
                            coords = reducer.fit_transform(embeddings)
                        else:  # UMAP
                            if _HAS_UMAP:
                                reducer = umap.UMAP(n_components=2, random_state=42)
                                coords = reducer.fit_transform(embeddings)
                            else:
                                st.warning("UMAP not installed, falling back to PCA")
                                from sklearn.decomposition import PCA
                                coords = PCA(n_components=2).fit_transform(embeddings)
                        
                        # Plot with matplotlib
                        fig, ax = plt.subplots(figsize=(10, 8))
                        fig.patch.set_alpha(0.0)
                        ax.patch.set_alpha(0.0)
                        
                        # Color by type
                        colors = {"text": "#3498db", "image": "#e74c3c"}
                        for t in set(types):
                            mask = [i for i, tp in enumerate(types) if tp == t]
                            ax.scatter(
                                coords[mask, 0], coords[mask, 1],
                                c=colors.get(t, "#95a5a6"),
                                label=t,
                                alpha=0.6,
                                s=30
                            )
                        
                        ax.legend()
                        ax.set_title(f"{viz_method} Projection of Corpus Embeddings", color="white")
                        ax.tick_params(colors="white")
                        for spine in ax.spines.values():
                            spine.set_visible(False)
                        
                        st.pyplot(fig)
                        plt.close(fig)
                        
                    except Exception as e:
                        st.error(f"Visualization failed: {e}")
            else:
                st.info("Need more items for meaningful visualization.")
        
        # Divider before legacy section
        st.divider()
        with st.expander("🔧 Legacy Miner (Original)", expanded=False):
            st.caption("The original item-by-item miner is still available below.")
            
            # --- Legacy Mining hyperparams ---
            c1, c2, c3 = st.columns(3)
            k_legacy = c1.slider("k-NN (legacy)", 4, 32, 12, key="k_legacy")
            min_cluster_legacy = c2.slider("Min cluster (legacy)", 4, 64, 8, key="min_legacy")
            top_desc_legacy = c3.slider("Top desc (legacy)", 4, 24, 12, key="top_legacy")
            canon_legacy = st.selectbox("Canonical space (legacy)", ["text","image","audio"], index=0, key="canon_legacy")

            if st.button("⚙️ Run legacy miner", disabled=not st.session_state["mm_items"]):
                with st.spinner("Mining (legacy)..."):
                    mm_items = []
                    for row in st.session_state["mm_items"]:
                        mm_items.append(MMItem(
                            id=row.get("id") or uuid.uuid4().hex[:8],
                            text=(row.get("text") or None),
                            image_path=(row.get("image_path") or None),
                            audio_path=(row.get("audio_path") or None),
                            meta=None
                        ))

                    text_adapter = _TextAdapter(embedder)
                    image_adapter = _ImageAdapter(model_name="ViT-B-32", pretrained="laion2b_s34b_b79k") if (Image is not None and open_clip is not None) else None
                    audio_adapter = _AudioAdapter(embedder)

                    miner = ArchetypeMiner(
                        text_encoder=text_adapter,
                        image_encoder=image_adapter if image_adapter else _ImageAdapter,
                        audio_encoder=audio_adapter,
                        canon=canon_legacy,
                        ridge_lambda=1e-3
                    )
                    for it in mm_items: 
                        miner.add(it)

                    try:
                        symbols_json = miner.run(k=int(k_legacy), min_cluster=int(min_cluster_legacy), top_desc=int(top_desc_legacy))
                    except Exception as e:
                        st.error(f"Mining failed: {e}")
                        symbols_json = {}

                if symbols_json:
                    st.success(f"Discovered {len(symbols_json)} symbols (legacy).")
                    for s, descs in symbols_json.items():
                        st.markdown(f"**{s}** — {', '.join(descs)}")

