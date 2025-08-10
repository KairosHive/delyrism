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

# ---- import your core code (from your package/module) ----
from delyrism import (
    SymbolSpace,
    TextEmbedder,
    context_delta_graph,
    plot_delta_graph,
    plot_contextual_subgraph_colored,
    plot_ambiguity_metrics
)

primaryColor = "#3498db"
# =============================
# Helpers
# =============================

def _default_symbols_map() -> Dict[str, List[str]]:
    return {
        "CLOUDS": [
            "spirit world",
            "transcendence",
            "wisdom",
            "portal",
            "spiritual connection",
        ],
        "EARTH": ["ground", "soil", "people", "life", "physical world"],
        "WATER": ["river", "emotion", "flow", "cleansing", "change"],
        "DRAGONFLY": ["messenger", "transformation", "threshold", "quick movement", "reflection"],
        "HOUSE": ["home", "shelter", "family", "boundary", "gathering"],
    }

def _embedder_key(e: TextEmbedder) -> str:
    # include all params that change embeddings
    b = getattr(e, "backend_type", "unknown")
    m = getattr(e, "model_name", None)
    p = getattr(e, "pooling", None)
    d = getattr(e, "dim", None)
    return f"{b}|{m}|{p}|{d}"

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
        backend = st.selectbox("Backend", ["qwen3", "qwen2", "original", "clap"], index=0)
        model = st.text_input("HF model override (optional)")
        pooling = st.selectbox("Pooling", ["eos", "mean", "cls", "last"], index=0)
        embedder = get_embedder(backend, model or None, pooling)

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
        ctx_chosen = st.multiselect(
            "Context symbols",
            options=sym_preview,
            default=[s for s in default_ctx if s in sym_preview],
            key="ctx_chosen",
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
        include_centroids = st.checkbox("Include centroids (stars)", True)
        normalize_centroids = st.checkbox("Normalize centroids (unit-length)", False)
        st.markdown("**Shift settings**")
        beta = st.slider("Shift strength β", 0.0, 2.0, 0.6, 0.05)
        gate = st.selectbox("Gate", ["relu", "cos", "softmax", "uniform"], index=0)
        within_symbol_softmax = st.checkbox("Softmax within symbol (if gate=softmax)", True)
        membership_alpha = st.slider("Membership α (desc vs centroid)", 0.0, 1.0, 0.0, 0.05)
        show_arrow = st.checkbox("Show arrows", True)
        
    st.markdown('</div>', unsafe_allow_html=True)

    # --- Ranking (panel-colored sliders) --------------------------
    st.markdown('<div class="panel-ranking">', unsafe_allow_html=True)
    with st.expander("Ranking (proposal)", expanded=True):
        tau = st.slider("Softmax temperature (τ)", 0.01, 2.0, 0.3, 0.01)
        alpha = st.slider("PageRank damping (α)", 0.10, 0.99, 0.8, 0.01)
        lam = st.slider("Blend λ (PR-graph vs Softmax-attention)", 0.0, 1.0, 0.6, 0.01)
        use_ppr = st.checkbox("Use Personalized PageRank", True)
    st.markdown('</div>', unsafe_allow_html=True)

    # --- Contextual Subgraph (panel-colored sliders) --------------
    st.markdown('<div class="panel-subgraph">', unsafe_allow_html=True)
    with st.expander("Contextual Subgraph (network)", expanded=True):
        ctx_topk_symbols = st.slider("Top symbols", 1, 12, 3)
        ctx_topk_desc = st.slider("Top descriptors / symbol", 1, 12, 3)
        ctx_method = st.selectbox("Scoring method", ["ppr", "softmax"], index=0)
        ctx_focus = st.slider("Context Focus (sharper → right)", 0.0, 1.0, 0.6, 0.01)
        ctx_alpha = st.slider("α (subgraph PageRank)", 0.50, 0.99, 0.85, 0.01)
        ctx_normalize = st.checkbox("Normalize by baseline PR (remove centrality)", True)
        descriptor_threshold = st.slider("Descriptor edge threshold (cosine)", 0.0, 0.9, 0.7, 0.02)
    st.markdown('</div>', unsafe_allow_html=True)

    # --- Δ Graph (panel-colored sliders) --------------------------
    st.markdown('<div class="panel-delta">', unsafe_allow_html=True)
    with st.expander("Δ Graph", expanded=True):
        within_symbol = st.checkbox("Within-symbol pairs only", False)
        sym_filter_sel = st.multiselect("Or restrict to symbols", sym_preview)
        top_abs_edges = st.slider("Top |Δ| edges", 2, 100, 10, 1)
        connected_only = st.checkbox("Connected nodes only", True)
    st.markdown('</div>', unsafe_allow_html=True)

# after you compute 'symbols_map' and 'embedder'
st.session_state["_current_symbols_map"] = symbols_map
st.session_state["_current_embedder"] = embedder

space = build_space(
    _symbols_map_key(symbols_map),          # changes when the JSON changes
    descriptor_threshold,                   # changes when slider changes
    _embedder_key(embedder)                 # changes when backend/model/pooling/dim changes
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
# =============================
# Row 1: Meaning Space (left) | Rankings & Attention (right)
# =============================
colL, colR = st.columns([1.15, 1])

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
    )
    st.pyplot(fig_ms, clear_figure=True)

    st.divider()
    st.subheader("Ambiguity Metrics")

    sort_opt = st.selectbox("Sort by", ["dispersion", "leakage", "entropy", "none"], index=0)
    color_map = getattr(space, "get_symbol_color_dict", lambda: None)()
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
            global_color_map=global_color_map,
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
                beta=beta,
                gate=gate,
                tau=tau,
                order_by_attention=True,
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
        cdict = getattr(space, "get_symbol_color_dict", None)
        color_map = cdict() if callable(cdict) else None
        sym_filter_arg = sym_filter_sel if sym_filter_sel else None

        G = context_delta_graph(
            space,
            sentence=sentence if sentence else None,
            weights=ctx_weights if ctx_weights else None,
            beta=beta,
            gate=gate,
            tau=tau,
            top_abs_edges=top_abs_edges,
            sym_filter=sym_filter_arg,
            within_symbol=within_symbol,
            connected_only=connected_only,

        )

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
