# symbols_latent_context_weighted_sentence.py
# ------------------------------------------------------------
# Context-conditioned Lakota-symbol space with
#   • weighted symbol proximities
#   • sentence / free-text context (hybrid)
#   • personalised PageRank graph diffusion
#   • ambiguity metrics + rich plots
# ------------------------------------------------------------

from __future__ import annotations
import math, warnings, itertools
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx
import numpy as np
import networkx as nx
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from typing import Optional, List

# ---------- optional back-ends ----------
_HAS_ST = _HAS_SCI = _HAS_UMAP = False
try:
    from sentence_transformers import SentenceTransformer          # type: ignore
    _HAS_ST = True
except Exception:
    pass
try:
    from scipy.spatial import ConvexHull                            # type: ignore
    _HAS_SCI = True
except Exception:
    pass
try:
    import umap                                                    # type: ignore
    _HAS_UMAP = True
except Exception:
    pass

# ---------- embedding backend ----------
from transformers import AutoTokenizer, AutoModel

import math, warnings
import numpy as np
from typing import List

from transformers import AutoTokenizer, AutoModel

# ---------- embedding backend ----------
from transformers import AutoTokenizer, AutoModel
import numpy as np
import math, warnings
from typing import List, Optional

# ---------- utilities ----------
def l2_normalize(X, axis=1, eps=1e-9):
    n = np.linalg.norm(X, axis=axis, keepdims=True) + eps
    return X / n

def softmax(x, tau=1.0):
    z = (x - np.max(x)) / max(tau, 1e-6)
    e = np.exp(z)
    return e / np.sum(e)

def entropy(p, eps=1e-12):
    p = np.clip(p, eps, 1.0)
    return float(-np.sum(p * np.log(p)))

def minmax_scale(d: Dict[str, float]):
    if not d:
        return {}
    v = np.array(list(d.values()))
    lo, hi = float(v.min()), float(v.max())
    if hi - lo < 1e-12:
        return {k: 0.5 for k in d}
    return {k: (val - lo) / (hi - lo) for k, val in d.items()}

# ---------- utilities ----------
def l2_normalize(X, axis=1, eps=1e-9):
    n = np.linalg.norm(X, axis=axis, keepdims=True) + eps
    return X / n


def _l2norm_torch(x, eps=1e-9):
    return torch.nn.functional.normalize(x, p=2, dim=1)

def make_symbol_palette(symbols, name="Nord"):
    base = CHIC_PALETTES[name]
    print('BASE', base)

    # extend if you have more symbols than base colors (cycles: normal, slightly lighter, slightly darker)
    layers = [0, +1, -1]
    if len(base) < len(symbols):
        colors = []
        while len(colors) < len(symbols):
            for s in base:
                for step in layers:
                    colors.append(_vary_luma(s, step))
                    if len(colors) >= len(symbols):
                        break
                if len(colors) >= len(symbols):
                    break
    else:
        colors = base[0:len(symbols)]
    return {s: colors[i] for i, s in enumerate(symbols)}

CHIC_PALETTES = {
    "Nord": ["#8FBCBB","#88C0D0","#81A1C1","#5E81AC","#BF616A","#D08770","#EBCB8B","#A3BE8C","#B48EAD","#4C566A","#3B4252","#2E3440"],
    "TeaHouse": ["#2D3142","#4F5D75","#BFC0C0","#EF8354","#2A9D8F","#E9C46A","#264653","#A26769","#7B6D8D","#C9ADA7"],
    "Jewel": ["#355070","#6D597A","#B56576","#E56B6F","#EAAC8B","#2A9D8F","#264653","#E9C46A","#F4A261","#E76F51"],
    "NeonEarth": ["#1F271B","#A6A57A","#FFE45E","#FF6392","#7AE582","#2DE1FC","#FF9F1C","#5F0F40","#227C9D","#17C3B2"],
    "MonoPop": ["#0B132B","#1C2541","#3A506B","#5BC0BE","#6FFFE9","#E0FBFC","#F17300","#F71735","#A1A1A1"],
    "OkabeIto": ["#E69F00","#56B4E9","#009E73","#F0E442","#0072B2","#D55E00","#CC79A7","#000000"],
    "AuroraPop": ["#FF6F59","#FFD166","#06D6A0","#118AB2","#073B4C",
                    "#EF476F","#8D99AE","#8338EC","#3A86FF","#F77F00",
                    "#2A9D8F","#E63946"],
    "NeonBodega": ["#F15BB5","#00BBF9","#00F5D4","#B9FBC0","#FEE440",
                "#9B5DE5","#F72585","#4CC9F0","#3A0CA3","#FF9F1C",
                "#2EC4B6","#90BE6D"]
                    }
    


import matplotlib.colors as mcolors

def _vary_luma(hex_color, step=0):  # subtle lighten/darken
    r,g,b = mcolors.to_rgb(hex_color)
    if step > 0:
        r,g,b = r+(1-r)*0.12*step, g+(1-g)*0.12*step, b+(1-b)*0.12*step
    elif step < 0:
        r,g,b = r*(1-0.12*abs(step)), g*(1-0.12*abs(step)), b*(1-0.12*abs(step))
    return (min(max(r,0),1), min(max(g,0),1), min(max(b,0),1), 1.0)


class TextEmbedder:
    """
    Flexible text embedder supporting:
      - Original SentenceTransformer backend
      - Qwen2 / Qwen3 embedding models (EOS pooling by default)
    Also supports optional instruction + context prompting.
    Fallback: hashing-based projection.
    """
    def __init__(self, backend: str = "original", model: Optional[str] = None, dim_fallback: int = 384,
                 device: Optional[str] = None, pooling: str = "eos",
                 default_instruction: Optional[str] = None, default_context: Optional[str] = None):
        self.backend_type = backend.lower()
        self.model_name = model
        self.dim = dim_fallback
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.pooling = pooling.lower()
        self.default_instruction = default_instruction
        self.default_context = default_context

        self._backend = None
        self._tokenizer = None
        self._proj = None
        self.audio_capable = False        # <-- ADD

        if self.backend_type == "sentence-transformer":
            self._init_original(model or "sentence-transformers/all-mpnet-base-v2")
        elif self.backend_type == "qwen2":
            self._init_qwen(model or "Qwen/Qwen2-Embedding")
        elif self.backend_type == "qwen3":
            self._init_qwen(model or "Qwen/Qwen3-Embedding-0.6B")
        elif self.backend_type == "audioclip":           # <-- ADD
            self._init_audioclip(model)                  # <-- ADD
        elif self.backend_type == "clap":
            self._init_clap(model or "laion/clap-htsat-fused")

        else:
            raise ValueError(f"Unknown backend '{backend}'. Choose: original | qwen2 | qwen3 | audioclip")

        if self._backend is None and self.backend_type != "audioclip":  # audioclip uses separate model var
            self._init_fallback()

    # ---------- init helpers ----------
    def _init_clap(self, model_name: str):
        try:
            from transformers import ClapProcessor, ClapModel
            self.clap_processor = ClapProcessor.from_pretrained(model_name)
            self._clap = ClapModel.from_pretrained(model_name).to(self.device)
            self._clap.eval()
            with torch.no_grad():
                dummy = np.zeros(48000, dtype=np.float32)  # 1s silence @48k
                ain = self.clap_processor(audios=[dummy], sampling_rate=48000, return_tensors="pt")
                ain = {k: v.to(self.device) for k, v in ain.items()}
                a = self._clap.get_audio_features(**ain)
                self.dim = int(a.shape[-1])
            self.audio_capable = True
            self._ac_sr = 48000
            print(f"[Embedder] CLAP loaded: {model_name} (dim={self.dim})")
        except Exception as e:
            raise RuntimeError(f"CLAP load failed: {e}")

    def _init_audioclip(self, model_name: Optional[str] = None):
        """
        AudioCLIP via open_clip text tower + tiny audio projector.
        No torchaudio, no HF dependency.
        """
        try:
            import open_clip
            # TEXT PATH — CLIP text encoder
            self.oc_model, _, self.oc_preprocess = open_clip.create_model_and_transforms(
                "ViT-B-32", pretrained="laion2b_s34b_b79k", device=self.device
            )
            self.oc_model.eval()
            self.oc_tokenizer = open_clip.get_tokenizer("ViT-B-32")

            # match CLIP text embedding dim
            self.dim = int(self.oc_model.text_projection.shape[1]) if hasattr(self.oc_model, "text_projection") else 512

            # AUDIO PATH — simple conv projector
            self.audio_proj = nn.Sequential(
                nn.Conv1d(1, 64, 5, 2, 2), nn.ReLU(),
                nn.Conv1d(64, 128, 5, 2, 2), nn.ReLU(),
                nn.AdaptiveAvgPool1d(1), nn.Flatten(),
                nn.Linear(128, self.dim),
            ).to(self.device).eval()

            self.audio_capable = True
            self._ac_sr = 48000
            self._ac_len = self._ac_sr * 10  # seconds

            print(f"[Embedder] AudioCLIP (open_clip) ready, dim={self.dim}.")

        except Exception as e:
            raise RuntimeError(f"AudioCLIP init failed (open_clip): {e}")

    # --- replace your embed_audio_array fallback with this (no torchaudio) ---
    @torch.no_grad()
    def embed_audio_array(self, wave: np.ndarray, sr: int) -> np.ndarray:
        if self.backend_type == "audioclip":
            # resample to 48k if needed
            if sr != self._ac_sr:
                t_old = np.linspace(0, 1, len(wave), endpoint=False)
                t_new = np.linspace(0, 1, int(len(wave) * (self._ac_sr / sr)), endpoint=False)
                wave = np.interp(t_new, t_old, wave).astype(np.float32)
                sr = self._ac_sr
            # center trim or pad to fixed length
            if len(wave) > self._ac_len:
                s = (len(wave) - self._ac_len) // 2
                wave = wave[s:s+self._ac_len]
            elif len(wave) < self._ac_len:
                pad = self._ac_len - len(wave)
                wave = np.pad(wave, (pad//2, pad - pad//2))
            x = torch.from_numpy(wave).float().to(self.device)[None, None, :]  # (1,1,T)
            z = self.audio_proj(x)[0]
            z = z / (z.norm() + 1e-8)
            return z.detach().cpu().float().numpy()

        if self.backend_type == "clap" and hasattr(self, "_clap"):
            ain = self.clap_processor(audios=[wave], sampling_rate=sr, return_tensors="pt")
            ain = {k: v.to(self.device) for k, v in ain.items()}
            z = self._clap.get_audio_features(**ain)[0]
            z = z / (z.norm() + 1e-8)
            return z.detach().cpu().float().numpy()

        raise RuntimeError("embed_audio_array called but this backend has no audio path.")

    def _init_original(self, model_name):
        try:
            from sentence_transformers import SentenceTransformer
            self._backend = SentenceTransformer(model_name)
            self.dim = int(self._backend.get_sentence_embedding_dimension())
            print(f"[Embedder] SentenceTransformer loaded ({self.dim}-d).")
        except Exception as e:
            warnings.warn(f"SentenceTransformer load failed: {e}")
            self._backend = None

    def _init_qwen(self, model_name):
        try:
            print(f"[Embedder] Loading Qwen model: {model_name}")
            self._tokenizer = AutoTokenizer.from_pretrained(model_name)
            self._backend = AutoModel.from_pretrained(model_name, trust_remote_code=True).to(self.device)
            self._backend.eval()
            # probe dim with EOS pooling
            with torch.no_grad():
                toks = self._tokenizer("probe", return_tensors="pt").to(self.device)
                out = self._backend(**toks)
                pooled = self._pool(out.last_hidden_state, toks["attention_mask"])
                self.dim = int(pooled.shape[1])
            print(f"[Embedder] Qwen loaded ({self.dim}-d), pooling={self.pooling}.")
        except Exception as e:
            warnings.warn(f"Qwen load failed: {e}")
            self._backend = None
            self._tokenizer = None

    def _init_fallback(self):
        rng = np.random.default_rng(42)
        self._proj = rng.normal(0, 1 / math.sqrt(self.dim), size=(7000, self.dim))
        print(f"[Embedder] Using hashing fallback ({self.dim}-d).")

    # ---------- pooling ----------
    def _pool(self, last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        last_hidden_state: [B, T, H]
        attention_mask   : [B, T] with 1 for real tokens, 0 for pad
        """
        if self.pooling == "mean":
            # mean over non-padding tokens
            mask = attention_mask.unsqueeze(-1)  # [B, T, 1]
            sum_hidden = (last_hidden_state * mask).sum(dim=1)
            lengths = mask.sum(dim=1).clamp(min=1)
            pooled = sum_hidden / lengths
            return _l2norm_torch(pooled)

        if self.pooling == "cls":
            # first token (works if model prepends a special token)
            pooled = last_hidden_state[:, 0, :]
            return _l2norm_torch(pooled)

        # "last": last timestep regardless of padding (usually not what you want)
        if self.pooling == "last":
            pooled = last_hidden_state[:, -1, :]
            return _l2norm_torch(pooled)

        # default: "eos" — last *non-pad* token
        idxs = attention_mask.sum(dim=1) - 1  # [B]
        idxs = idxs.clamp(min=0)
        bsz = last_hidden_state.size(0)
        pooled = last_hidden_state[torch.arange(bsz, device=last_hidden_state.device), idxs, :]
        return _l2norm_torch(pooled)

    # ---------- input templating ----------
    def _apply_prompt_template(self, texts: List[str],
                            instruction: Optional[str],
                            context: Optional[str]) -> List[str]:
        inst = instruction if instruction is not None else self.default_instruction
        ctx  = context if context is not None else self.default_context

        # Nothing to add? Return unchanged.
        if inst is None and ctx is None:
            return texts

        # Allow a single global context (str) or per-text contexts (list/tuple/ndarray)
        ctx_is_seq = isinstance(ctx, (list, tuple, np.ndarray))
        if ctx_is_seq:
            if len(ctx) != len(texts):
                raise ValueError(f"Context list length ({len(ctx)}) must match texts ({len(texts)})")

        templated = []
        for i, t in enumerate(texts):
            parts = []
            if inst:
                parts.append(str(inst).strip())                # e.g., "Instruction: …"
            if ctx is not None:
                parts.append(f"Context: {ctx[i] if ctx_is_seq else ctx}")
            parts.append(f"Text: {t}")
            templated.append("\n".join(parts))
        return templated


    # ---------- encoding ----------
    def encode(
        self,
        texts: List[str],
        instruction: Optional[str] = None,
        context: Optional[str] = None,
    ):
        # --- Qwen path (unchanged) ---
        if self._backend is not None and self._tokenizer is not None and self.backend_type in ("qwen2","qwen3"):
            inputs = self._apply_prompt_template(texts, instruction, context)
            with torch.no_grad():
                toks = self._tokenizer(inputs, padding=True, truncation=True, return_tensors="pt").to(self.device)
                out = self._backend(**toks)
                pooled = self._pool(out.last_hidden_state, toks["attention_mask"])
                return pooled.cpu().numpy().astype(np.float32)

        # --- SentenceTransformer (unchanged) ---
        if self.backend_type == "original" and self._backend is not None:
            return np.asarray(self._backend.encode(texts, normalize_embeddings=True), dtype=np.float32)

        # --- NEW: AudioCLIP text path (open_clip) ---
        if self.backend_type == "audioclip":
            toks = self.oc_tokenizer(texts).to(self.device)   # list[str] -> token ids
            with torch.no_grad():
                z = self.oc_model.encode_text(toks)           # [B, D]
                z = z / (z.norm(dim=1, keepdim=True) + 1e-8)
            return z.detach().cpu().numpy().astype(np.float32)

        # inside TextEmbedder.encode(...)
        if self.backend_type == "clap" and hasattr(self, "_clap"):
            with torch.no_grad():
                tin = self.clap_processor(text=texts, return_tensors="pt", padding=True)
                tin = {k: v.to(self.device) for k, v in tin.items()}
                z = self._clap.get_text_features(**tin)  # [B, D]
                z = z / (z.norm(dim=1, keepdim=True) + 1e-8)
            return z.detach().cpu().numpy().astype(np.float32)


        # --- hashing fallback (last resort) ---
        vocab = 7000
        def vec(t):
            v = np.zeros(vocab, np.float32)
            for tok in t.lower().split():
                v[hash(tok) % vocab] += 1
            return v
        if self._proj is None:
            # safety: create projection if not already set
            self._init_fallback()
        M = np.stack([vec(t) for t in texts]) @ self._proj
        M = M / (np.linalg.norm(M, axis=1, keepdims=True) + 1e-9)
        return M.astype(np.float32)



# ---------- core container ----------
@dataclass
class SymbolSpace:
    symbols_to_descriptors: Dict[str, List[str]]
    embedder: TextEmbedder
    descriptor_threshold: float = 0.2
    contextual_embeddings: bool = False   # <---- NEW FLAG
    

    # -------- init --------
    def __post_init__(self):
        self.symbols = list(self.symbols_to_descriptors)
        self.descriptors = [d for lst in self.symbols_to_descriptors.values() for d in lst]
        self.owner = {d: s for s, lst in self.symbols_to_descriptors.items() for d in lst}
        symbol_membership = np.array([self.owner[d] for d in self.descriptors])
        if self.embedder.default_context == "Distributed":
            self.D = l2_normalize(self.embedder.encode(self.descriptors, context=symbol_membership), axis=1)
        else:
            self.D = l2_normalize(self.embedder.encode(self.descriptors), axis=1)
        self.desc_idx = {d: i for i, d in enumerate(self.descriptors)}
        self.symbol_to_idx = {s: [self.desc_idx[d] for d in ds]
                              for s, ds in self.symbols_to_descriptors.items()}
        self.symbol_centroids = {
            s: self.D[idx].mean(0)
            for s, idx in self.symbol_to_idx.items()
        }


        self.G = self._build_graph()
        self._nbrs = NearestNeighbors(metric="cosine",
                                      n_neighbors=min(10, len(self.descriptors) - 1)
                                     ).fit(self.D)
        self._pca = PCA(n_components=2, random_state=42).fit(self.D)
        self.context_override: Optional[np.ndarray] = None   # <-- ADD

    # -------- graph --------
    def _build_graph(self):
        G = nx.Graph()
        for s in self.symbols:
            G.add_node(f"S:{s}")
            for d in self.symbols_to_descriptors[s]:
                G.add_edge(f"S:{s}", f"D:{d}", weight=1.0)
        if self.descriptor_threshold > 0:
            C = self.D @ self.D.T
            for i, j in itertools.combinations(range(len(self.descriptors)), 2):
                if C[i, j] > self.descriptor_threshold:
                    G.add_edge(f"D:{self.descriptors[i]}",
                               f"D:{self.descriptors[j]}",
                               weight=float(C[i, j]))
        return G

    # in class SymbolSpace

    def _pool_with_context(self, D, vctx, *, pool_type="avg", pool_w=0.7):
        """
        Pool each descriptor vector with the context vector.
        pool_type: "avg" | "max" | "min"
        pool_w: weight for context when pool_type='avg'  (0..1)
        """
        if np.linalg.norm(vctx) < 1e-8:
            return D.copy()

        if pool_type == "avg":
            # weighted average: (1-w)*desc + w*ctx
            D_ctx = ((1.0 - float(pool_w)) * D) + (float(pool_w) * vctx[None, :])
        elif pool_type == "max":
            D_ctx = np.maximum(D, vctx[None, :])
        elif pool_type == "min":
            D_ctx = np.minimum(D, vctx[None, :])
        else:
            raise ValueError(f"Unknown pool_type '{pool_type}'")
        return l2_normalize(D_ctx, axis=1)

    # -------- context helpers --------
    def set_context_vec(self, vec: Optional[np.ndarray]) -> None:
        if vec is None:
            self.context_override = None
        else:
            v = vec.astype(np.float32)
            v /= (np.linalg.norm(v) + 1e-9)
            self.context_override = v

    def _sent_vec(self, text: Optional[str], ignore_override=False):
        # <-- CHANGED: honor audio override first
        if not ignore_override and self.context_override is not None:
            return self.context_override.copy()
        if not text or not text.strip():
            return np.zeros(self.embedder.dim, np.float32)
        v = self.embedder.encode([text])[0]
        return v / (np.linalg.norm(v) + 1e-9)

    def _weight_vec(self, w: Optional[Dict[str, float]]):
        if not w:
            return np.zeros(self.embedder.dim, np.float32)
        parts = [(wt, self.symbol_centroids[s]) for s, wt in w.items()
                 if wt > 0 and s in self.symbol_centroids]
        if not parts:
            return np.zeros(self.embedder.dim, np.float32)
        ws, vecs = zip(*parts)
        v = (np.stack(vecs) * np.array(ws)[:, None]).sum(0) / (sum(ws) + 1e-9)
        return v / (np.linalg.norm(v) + 1e-9)

    def ctx_vec(self, weights=None, sentence=None, ignore_override=False):
        v = self._weight_vec(weights) + self._sent_vec(sentence, ignore_override=ignore_override)
        n = np.linalg.norm(v)
        return v if n == 0 else v / n

    # -------- conditioned embedding --------

    def _attention(self, symbol, vctx, tau, sentence=None):
        D, descriptors = self._get_descriptor_vectors(symbol, sentence)
        scores = D @ vctx
        a = softmax(scores, tau)
        return a, descriptors


    def conditioned_symbol(self, symbol, weights=None, sentence=None, tau=0.3):
        """
        Compute the context-conditioned embedding and descriptor attention for a given symbol.

        This method returns a contextually weighted centroid for the symbol, based on:
          - Optional symbol weights (e.g., from user or prior context)
          - Optional free-text sentence context
          - Softmax attention over the symbol's descriptors, conditioned on the context vector

        If no context is provided (weights and sentence are both None or empty), the plain centroid
        of the symbol's descriptors is returned, and attention is uniform over descriptors.

        Parameters
        ----------
        symbol : str
            The symbol for which to compute the conditioned embedding.
        weights : dict, optional
            Optional dictionary mapping symbols to weights (float). Used to build a context vector.
        sentence : str, optional
            Optional free-text context sentence. Used to build a context vector.
        tau : float, default=0.3
            Softmax temperature for attention over descriptors.

        Returns
        -------
        e : np.ndarray
            The context-conditioned embedding vector for the symbol (L2-normalized).
        attn_dict : dict
            Dictionary mapping each descriptor (str) to its attention weight (float) under the context.

        Notes
        -----
        - If both weights and sentence are provided, both are used to form the context vector.
        - If the context vector is zero (no context), the plain centroid and uniform attention are returned.
        - The attention weights sum to 1 over the symbol's descriptors.
        """
        vctx = self.ctx_vec(weights, sentence)
        if np.linalg.norm(vctx) < 1e-8:
            k = len(self.symbol_to_idx[symbol])
            return self.symbol_centroids[symbol], \
                {d: 1.0 / k for d in self.symbols_to_descriptors[symbol]}
        a, ds = self._attention(symbol, vctx, tau, sentence=sentence)
        # Use D as in _get_descriptor_vectors:
        D, _ = self._get_descriptor_vectors(symbol, sentence)
        e = (D * a[:, None]).sum(0)
        return e / (np.linalg.norm(e) + 1e-9), {d: float(ai) for d, ai in zip(ds, a)}


    # -------- proposals --------
    def _ppr(self, weights, alpha):
        if not weights:
            return {s: 0.0 for s in self.symbols}
        pers = {n: 0.0 for n in self.G.nodes}
        tot = sum(v for v in weights.values() if v > 0)
        for s, v in weights.items():
            if v > 0 and f"S:{s}" in self.G:
                pers[f"S:{s}"] = v / tot
        pr = nx.pagerank(self.G, alpha=alpha, personalization=pers, weight="weight")
        return {n[2:]: float(v) for n, v in pr.items() if n.startswith("S:")}

    def _ppr_with_sentence(self, sentence: str, alpha: float = 0.85, tau: float = 0.2):
        """
        Personalized PageRank: inject probability mass at descriptor nodes based on 
        their semantic similarity to the input sentence. The closer the descriptor to 
        the sentence, the higher its initial mass.
        """
        # 1. Embed the sentence
        sent_vec = self.embedder.encode([sentence])[0]
        sent_vec = sent_vec / (np.linalg.norm(sent_vec) + 1e-9)

        # 2. Compute cosine similarity to every descriptor
        sims = self.D @ sent_vec
        # 3. Softmax for normalized, temperature-controlled attention over descriptors
        desc_weights = softmax(sims, tau=tau)
        
        pers = {n: 0.0 for n in self.G.nodes}
        for i, d in enumerate(self.descriptors):
            pers[f"D:{d}"] = desc_weights[i]

        # 4. Run PPR with personalization over descriptors
        pr = nx.pagerank(self.G, alpha=alpha, personalization=pers, weight="weight")

        # Return just the scores for symbols (could also return for descriptors if needed)
        return {n[2:]: float(v) for n, v in pr.items() if n.startswith("S:")}


    def _ppr_general(self, weights=None, sentence=None, alpha=0.85, tau=0.2, lam=0.5):
        # Get personalization for symbols (if any)
        pers_syms = {n: 0.0 for n in self.G.nodes}
        if weights:
            tot = sum(v for v in weights.values() if v > 0)
            for s, v in weights.items():
                if v > 0 and f"S:{s}" in self.G:
                    pers_syms[f"S:{s}"] = v / tot
        # Get personalization for descriptors (if sentence)
        pers_descs = {n: 0.0 for n in self.G.nodes}
        if sentence:
            sent_vec = self.embedder.encode([sentence])[0]
            sent_vec = sent_vec / (np.linalg.norm(sent_vec) + 1e-9)
            sims = self.D @ sent_vec
            desc_weights = softmax(sims, tau=tau)
            for i, d in enumerate(self.descriptors):
                pers_descs[f"D:{d}"] = desc_weights[i]
        # Blend both (lam controls mixture)
        pers = {n: lam * pers_syms[n] + (1-lam) * pers_descs[n] for n in self.G.nodes}
        pr = nx.pagerank(self.G, alpha=alpha, personalization=pers, weight="weight")
        return {n[2:]: float(v) for n, v in pr.items() if n.startswith("S:")}

    
    def get_symbol_color_dict(self, palette="Nord"):
        return make_symbol_palette(self.symbols, name=palette)

        
    def propose(self, weights=None, sentence=None,
                topk=5, tau=0.3, lam=0.6, alpha=0.85, use_ppr=True, blind_spot=False):
        vctx = self.ctx_vec(weights, sentence)
        pr = {s: 0.0 for s in self.symbols}
        if use_ppr:
            if weights is not None and sentence is not None:
                pr = self._ppr_general(weights, sentence, alpha, tau, lam)
            elif sentence is not None and weights is None:
                pr = self._ppr_with_sentence(sentence, alpha, tau)
            elif weights is not None and sentence is None:
                pr = self._ppr(weights, alpha)
        
        coh = {}
        for s in self.symbols:
            es, _ = self.conditioned_symbol(s, weights, sentence, tau)
            coh[s] = float(es @ vctx) if np.linalg.norm(vctx) > 0 else 0.0
        coh_n = minmax_scale(coh)
        pr_n = minmax_scale(pr)
        score = {s: lam * coh_n[s] + (1 - lam) * pr_n.get(s, 0.0) for s in self.symbols}
        
        # Sort by score (descending for normal, ascending for blind_spot)
        ranked = sorted(self.symbols, key=lambda s: score[s], reverse=(not blind_spot))[:topk]
        return [(s, score[s], coh_n[s], pr_n.get(s, 0.0)) for s in ranked]

    def _get_descriptor_vectors(self, symbol, sentence=None):
        """
        Return descriptor embeddings for a symbol.
        If contextual_embeddings=True and sentence provided, embed descriptors in context.
        """
        descriptors = self.symbols_to_descriptors[symbol]
        if self.contextual_embeddings and sentence:
            # Example template; you can adjust for better results
            texts = [f"{sentence}, {desc}" for desc in descriptors]
            D = l2_normalize(self.embedder.encode(texts), axis=1)
        else:
            idx = self.symbol_to_idx[symbol]
            D = self.D[idx]
        return D, descriptors

    # in class SymbolSpace
    def make_shifted_matrix(
        self,
        *,
        weights=None,
        sentence=None,
        strategy: str = "gate",
        beta: float = 0.6,
        gate: str = "relu",
        tau: float = 0.5,
        within_symbol_softmax: bool = False,
        gamma: float = 0.5,
        prompt_template: str = "{sent}, {desc}",
        pool_type: str = "avg",
        pool_w: float = 0.7,
        # NEW ▼
        membership_alpha: float = 0.0,
    ):
        ...
        vctx = self.ctx_vec(weights=weights, sentence=sentence)
        # --- pooling ---
        if strategy == "pooling":
            Dtmp = self._pool_with_context(self.D, vctx, pool_type=pool_type, pool_w=pool_w)
        # --- gate / hybrid ---
        elif strategy in ("gate", "hybrid"):
            sims = self.D @ vctx
            if gate == "relu":
                g = np.maximum(0.0, sims)
            elif gate == "cos":
                g = sims
            elif gate == "softmax":
                if within_symbol_softmax:
                    g = np.zeros_like(sims)
                    for s in self.symbols:
                        idx = self.symbol_to_idx[s]
                        if len(idx) > 0:
                            g[idx] = softmax(sims[idx], tau=tau)
                else:
                    g = softmax(sims, tau=tau)
            elif gate == "uniform":
                g = np.ones_like(sims)
            else:
                raise ValueError(f"Unknown gate '{gate}'")
            D_gate = l2_normalize(self.D + (beta * g[:, None]) * vctx[None, :], axis=1)

            if strategy == "gate":
                Dtmp = D_gate
            else:
                # hybrid: blend gate + reembed, then norm
                desc_texts = [prompt_template.format(sent=sentence, desc=d) for d in self.descriptors]
                D_re = l2_normalize(self.embedder.encode(desc_texts), axis=1)
                D_mix = (1.0 - float(gamma)) * D_gate + float(gamma) * D_re
                Dtmp = l2_normalize(D_mix, axis=1)
        # --- reembed ---
        elif strategy == "reembed":
            desc_texts = [prompt_template.format(sent=sentence, desc=d) for d in self.descriptors]
            Dtmp = l2_normalize(self.embedder.encode(desc_texts), axis=1)
        else:
            raise ValueError(f"Unknown strategy '{strategy}'")

        # NEW: apply membership blend so downstream (heatmaps/Δ-graph) see the same α effect
        if float(membership_alpha) > 0.0:
            return self._apply_membership_blend(self.D, Dtmp, float(membership_alpha))
        return Dtmp


    # -------- metrics --------
    def dispersion(self, symbol):
        idx = self.symbol_to_idx[symbol]
        if len(idx) < 2:
            return 0.0
        S = cosine_similarity(self.D[idx])
        return float(np.mean(1 - S[np.triu_indices(len(idx), 1)]))

    def leakage(self, symbol, k=10):
        idx = self.symbol_to_idx[symbol]
        if not idx:
            return 0.0
        k = min(k, len(self.descriptors) - 1)
        _, nbrs = self._nbrs.kneighbors(self.D[idx], n_neighbors=k + 1)
        leaks = []
        for r, row in enumerate(nbrs):
            row = [j for j in row if j != idx[r]][:k]
            labs = [self.owner[self.descriptors[j]] for j in row]
            leaks.append(sum(1 for lab in labs if lab != symbol) / k)
        return float(np.mean(leaks))

    def soft_entropy(self, symbol, tau=0.5):
        idx = self.symbol_to_idx[symbol]
        if not idx:
            return 0.0
        cent = np.stack(list(self.symbol_centroids.values()))
        ents = []
        for i in idx:
            p = softmax(self.D[i] @ cent.T, tau)
            ents.append(entropy(p))
        return float(np.mean(ents))

    def reduce_2d(self, method: str = "auto", random_state: int = 42):
        """Return 2-D projection of every descriptor embedding."""
        X = self.D
        if method == "auto" and _HAS_UMAP:
            return (
                umap.UMAP(
                    n_neighbors=20,
                    min_dist=0.1,
                    metric="cosine",
                    random_state=random_state,
                )
                .fit_transform(X)
                .astype(np.float32)
            )
        if method == "tsne":
            return (
                TSNE(
                    n_components=2,
                    metric="cosine",
                    random_state=random_state,
                    init="random",
                    perplexity=20,
                )
                .fit_transform(X)
                .astype(np.float32)
            )
        return PCA(n_components=2, random_state=random_state).fit_transform(X)

    def top_symbols_and_descriptors(self, sentence, method="ppr", topk_symbols=3, topk_desc=3, alpha=0.85, tau=0.02):
        if method == "ppr":
            return recommend_symbols_with_ppr_descriptors(self, sentence, topk=topk_symbols, alpha=alpha, tau=tau, n_best=topk_desc)
        else:
            return recommend_symbols_with_softmax_attention(self, sentence, topk=topk_symbols, tau=tau, n_best=topk_desc)

    def recommend_symbols_with_ppr_descriptors(self, sentence, topk=2, alpha=0.85, tau=0.02, n_best=3):
        # 1) Build descriptor personalization from sentence (or override)
        sent_vec = self.ctx_vec(sentence=sentence)
        
        pers = {f"D:{d}": float(w) for d, w in zip(self.descriptors, softmax(self.D @ sent_vec, tau=tau))}
        # 2) Run a single PageRank over the whole graph
        pr_all = nx.pagerank(self.G, alpha=alpha, personalization=pers, weight="weight")
        # 3) Symbols ranked by PR
        pr_syms = {n[2:]: v for n, v in pr_all.items() if n.startswith("S:")}
        top_syms = sorted(pr_syms.items(), key=lambda x: x[1], reverse=True)[:topk]
        # 4) Top descriptors per symbol by PR
        results = []
        for sym, score in top_syms:
            descs = self.symbols_to_descriptors[sym]
            desc_scores = sorted([(d, pr_all.get(f"D:{d}", 0.0)) for d in descs],
                                 key=lambda x: x[1], reverse=True)[:n_best]
            results.append({
                "symbol": sym,
                "score": score,
                "best_descriptors": [d for d, _ in desc_scores],
                "descriptor_scores": [s for _, s in desc_scores]
            })
        return results

    def recommend_symbols_with_softmax_attention(self, context_sentence, topk=2, tau=0.02, n_best=3, score_method="entropy",
                                                    lam=None, topN=5):
        """
        Get top-k symbols and their most attended descriptor, all ranked by attention
        (using direct softmax attention between context and descriptors).
        """
        # Compute the context vector from the sentence
        ctx_vec = self.ctx_vec(sentence=context_sentence)
        symbol_scores = {}
        symbol_desc_att = {}

        for sym in self.symbols:
            e_sym, desc_attention = self.conditioned_symbol(sym, sentence=context_sentence, tau=tau)
            vals = np.fromiter(desc_attention.values(), dtype=float)
            k = len(vals)

            if k == 0:
                score = 0.0
            elif score_method == "max":
                score = float(vals.max())
            elif score_method == "excess":
                # k-invariant: 0 for uniform (1/k), 1 for one-hot
                score = float((vals.max() - 1.0/k) / (1.0 - 1.0/k + 1e-12))
            elif score_method == "mean":
                score = float(vals.mean())
            elif score_method == "sum_topN":
                score = float(np.sort(vals)[-topN:].sum())
            elif score_method == "mean_topN":
                score = float(np.sort(vals)[-topN:].mean())
            elif score_method == "entropy":
                p = np.clip(vals, 1e-12, 1.0)
                H = float(-(p * np.log(p)).sum())
                score = 1.0 - H / np.log(k)            # 0 uniform, 1 one-hot
            elif score_method == "coh":
                score = float(e_sym @ ctx_vec) if np.linalg.norm(ctx_vec) > 0 else 0.0
            elif score_method == "blend":
                # blend coherence with k-invariant attention sharpness
                att = (vals.max() - 1.0/k) / (1.0 - 1.0/k + 1e-12)
                coh = float(e_sym @ ctx_vec) if np.linalg.norm(ctx_vec) > 0 else 0.0
                lam = 0.5 if lam is None else float(lam)
                score = lam * coh + (1 - lam) * att
            else:
                raise ValueError(f"Unknown score_method '{score_method}'")

            symbol_scores[sym] = score
            sorted_descs = sorted(desc_attention.items(), key=lambda x: x[1], reverse=True)[:n_best]
            symbol_desc_att[sym] = sorted_descs

        # Rank symbols by score (max attention)
        top_syms = sorted(symbol_scores.items(), key=lambda x: x[1], reverse=True)[:topk]

        results = []
        for sym, sym_score in top_syms:
            desc_scores_sorted = symbol_desc_att[sym]
            results.append({
                "symbol": sym,
                "score": sym_score,
                "best_descriptors": [d for d, _ in desc_scores_sorted],
                "descriptor_scores": [s for _, s in desc_scores_sorted]
            })
        return results

    def descriptor_shifted_embeddings(self, *, weights=None, sentence=None,
                                  beta: float = 0.6, gate: str = "relu", tau: float = 0.5):
        """
        Return a context-shifted descriptor matrix D_ctx from original D.
        Shift rule: D_i' = norm( D_i + beta * g_i * v_ctx ), where g_i gates the shift.
        - v_ctx: normalized context vector from (weights + sentence)
        - g_i: gating per descriptor; options:
            "relu"   -> max(0, cos(D_i, v_ctx))
            "cos"    -> raw cosine(D_i, v_ctx) (can be negative: two-way shift)
            "softmax"-> softmax over cosine similarities (temperature tau)
            "uniform"-> 1 for all descriptors (global tilt)
        """
        vctx = self.ctx_vec(weights=weights, sentence=sentence)
        if np.linalg.norm(vctx) < 1e-8:
            return self.D.copy()

        sims = self.D @ vctx  # cosine because rows of D are L2-normalized
        if gate == "relu":
            g = np.maximum(0.0, sims)
        elif gate == "cos":
            g = sims
        elif gate == "softmax":
            g = softmax(sims, tau=tau)
        elif gate == "uniform":
            g = np.ones_like(sims)
        else:
            raise ValueError(f"Unknown gate '{gate}'")

        D_ctx = self.D + (beta * g[:, None]) * vctx[None, :]
        return l2_normalize(D_ctx, axis=1)

    def descriptor_similarity_matrices(
        self,
        *,
        weights=None,
        sentence=None,
        strategy: str = "gate",
        beta: float = 0.6,
        gate: str = "relu",
        tau: float = 0.5,
        within_symbol_softmax: bool = False,
        order_by_attention: bool = True,
        gamma: float = 0.5,
        prompt_template: str = "In this context: {sent}. Descriptor: {desc}",
        pool_type: str = "avg",
        pool_w: float = 0.7,
        # NEW ▼
        membership_alpha: float = 0.0,
    ):
        D_ctx = self.make_shifted_matrix(
            weights=weights, sentence=sentence,
            strategy=strategy,
            beta=beta, gate=gate, tau=tau, within_symbol_softmax=within_symbol_softmax,
            gamma=gamma, prompt_template=prompt_template,
            pool_type=pool_type, pool_w=pool_w,
            # NEW ▼
            membership_alpha=membership_alpha,
        )

        out = {}
        for s in self.symbols:
            idx = self.symbol_to_idx[s]
            desc_names = self.symbols_to_descriptors[s]

            D_before = self.D[idx]
            D_after  = D_ctx[idx]

            if order_by_attention and (weights is not None or sentence or self.context_override is not None):
                _, att = self.conditioned_symbol(s, weights=weights, sentence=sentence, tau=tau)
                scores = [att.get(d, 0.0) for d in desc_names]
                order = np.argsort(scores)[::-1]
                D_before = D_before[order]
                D_after  = D_after[order]
                desc_names = [desc_names[i] for i in order]

            S_before = cosine_similarity(D_before)
            S_after  = cosine_similarity(D_after)
            S_delta  = S_after - S_before

            out[s] = {
                "descriptors": desc_names,
                "S_before": S_before,
                "S_after": S_after,
                "S_delta": S_delta,
            }
        return out



    # ------------------------------------------------------------------
    #                       P L O T T I N G
    # ------------------------------------------------------------------
    def plot_map(self,
             method="umap",
             n_neighbors=15,
             with_hulls=True,
             include_centroids=True,
             normalize_centroids=False,
             figsize=(8, 6),
             title="Descriptor map"):
        """
        Minimal, original-style descriptor map.

        - Points fed to reducer are ordered per symbol:
        [descriptors of S] then [centroid of S] (if include_centroids), then next symbol.
        - Centroids (if included) are NOT normalized, matching the original behavior.
        """

        # --- choose reducer ---
        if method == "umap" and _HAS_UMAP:
            reducer = umap.UMAP(
                n_neighbors=n_neighbors, min_dist=0.1,
                metric="cosine", random_state=42
            )
            xlbl, ylbl = "UMAP 1", "UMAP 2"
        elif method == "tsne":
            reducer = TSNE(
                n_components=2, metric="cosine",
                random_state=42, init="random", perplexity=30
            )
            xlbl, ylbl = "t-SNE 1", "t-SNE 2"
        else:
            reducer = PCA(n_components=2, random_state=42)
            xlbl, ylbl = "PCA 1", "PCA 2"

        # --- stack points in legacy order ---
        all_points = []
        for s in self.symbols:
            idx = self.symbol_to_idx[s]
            all_points.append(self.D[idx])  # descriptors
            if include_centroids:
                c = self.D[idx].mean(0)      # centroid (not L2-normalized)
                if normalize_centroids:
                    c = c / (np.linalg.norm(c) + 1e-9)
                all_points.append(c[None, :])

        X = np.concatenate(all_points, axis=0)
        Z = reducer.fit_transform(X)

        # --- colors per symbol ---
        colors = plt.cm.tab20(np.linspace(0, 1, len(self.symbols)))
        color_dict = {s: colors[i] for i, s in enumerate(self.symbols)}

        # --- plot following the same indexing pattern ---
        plt.figure(figsize=figsize)
        cidx = 0
        for s in self.symbols:
            idx = self.symbol_to_idx[s]
            Zi = Z[cidx : cidx + len(idx)]            # descriptors
            plt.scatter(Zi[:, 0], Zi[:, 1], s=40, alpha=0.85, color=color_dict[s], label=s)
            cidx += len(idx)

            # centroid marker (immediately after descriptors in Z)
            if include_centroids:
                cz = Z[cidx]
                plt.scatter(cz[0], cz[1], s=220, color=color_dict[s],
                            edgecolor="black", marker="*", alpha=0.95, zorder=10)
                cidx += 1

            # convex hull over descriptors
            if with_hulls and _HAS_SCI and len(idx) >= 3:
                try:
                    hull = ConvexHull(Zi)
                    poly = Zi[hull.vertices]
                    closed = np.vstack([poly, poly[0]])
                    plt.fill(closed[:, 0], closed[:, 1], alpha=0.15, color=color_dict[s])
                    plt.plot(closed[:, 0], closed[:, 1], color=color_dict[s], lw=2, alpha=0.7)
                except Exception:
                    pass

        plt.xlabel(xlbl); plt.ylabel(ylbl)
        plt.title(title)
        plt.legend(frameon=False, ncol=2)
        plt.tight_layout()
        plt.show()


    def _apply_membership_blend(self, D0: np.ndarray, Dtmp: np.ndarray, alpha: float) -> np.ndarray:
        """
        Membership blend:
        Δ_i = (1-α) * (Dtmp_i - D0_i) + α * Δ_centroid(symbol_i)
        and return L2-normalized matrix.
        """
        alpha = float(alpha)
        if alpha <= 1e-8:
            return l2_normalize(Dtmp, axis=1)

        D_out = D0.copy()
        for s, idx in self.symbol_to_idx.items():
            if not idx:
                continue
            c0 = D0[idx].mean(0)
            c1 = Dtmp[idx].mean(0)
            delta_sym = c1 - c0                       # centroid shift for this symbol
            # per-row blend: keep each descriptor's own move + a share of the centroid move
            D_out[idx] = D0[idx] + (1.0 - alpha) * (Dtmp[idx] - D0[idx]) + alpha * delta_sym[None, :]
        return l2_normalize(D_out, axis=1)


    # --- in class SymbolSpace ---

    def get_cached_reducer_and_projection(self, method="umap", n_neighbors=15, include_centroids=True, normalize_centroids=False):
        """
        Returns (reducer, Z_fit, X_fit, slices) for the BASE state.
        Caches the result to avoid re-fitting UMAP/PCA.
        """
        if not hasattr(self, "_cached_projections"):
            self._cached_projections = {}
            
        cache_key = (method, n_neighbors, include_centroids, normalize_centroids)
        
        if cache_key in self._cached_projections:
            return self._cached_projections[cache_key]
            
        # --- Build X_fit ---
        blocks, slices, cursor = [], {}, 0
        for s in self.symbols:
            idx = self.symbol_to_idx[s]
            Xi = self.D[idx]
            blocks.append(Xi)
            slices[s] = (cursor, cursor + len(idx))
            cursor += len(idx)
            if include_centroids and len(idx) > 0:
                c = self.D[idx].mean(0)
                if normalize_centroids:
                    c = c / (np.linalg.norm(c) + 1e-9)
                blocks.append(c[None, :])
                cursor += 1

        X_fit = np.concatenate(blocks, axis=0)
        
        # --- Fit Reducer ---
        if method == "umap" and _HAS_UMAP:
            reducer = umap.UMAP(
                n_neighbors=n_neighbors, min_dist=0.1,
                metric="cosine", random_state=42
            )
        elif method == "tsne":
            reducer = TSNE(n_components=2, random_state=42, perplexity=30, init="pca")
        else:
            reducer = PCA(n_components=2, random_state=42)
            
        if method == "tsne":
            Z_fit = reducer.fit_transform(X_fit)
        else:
            reducer.fit(X_fit)
            Z_fit = reducer.transform(X_fit)
            
        result = (reducer, Z_fit, X_fit, slices)
        
        # Only cache if reusable (t-SNE usually isn't for transform)
        if method != "tsne":
            self._cached_projections[cache_key] = result
            
        return result

    def plot_map_shift(
        self,
        *,
        weights=None,
        sentence=None,
        method="umap",
        n_neighbors=15,
        with_hulls=True,
        include_centroids=True,
        normalize_centroids=False,
        figsize=(9, 7),
        title="Context shift on descriptor map",
        arrow_scale=1.0,
        arrow_alpha=0.65,
        gate="relu",
        tau=0.5,
        beta=0.6,
        membership_alpha=0.0,      # existing: 0 = pure descriptor shift, 1 = pure symbol-centroid shift
        within_symbol_softmax=False, # NEW: if gate=='softmax', apply softmax per symbol (temperature=tau),
        color_dict=None
    ):
        """
        Show original descriptor positions and their context-shifted positions, with arrows.

        membership_alpha in [0,1] blends two HD shift components per descriptor i in symbol s:
        Δ_i = (1 - a) * [g_i * v_ctx] + a * [Δ_centroid(s)]
        where Δ_centroid(s) = mean(D_ctx[s]) - mean(D[s]) is the symbol centroid shift,
        g_i is the descriptor gate (relu/cos/softmax/uniform), and v_ctx is the normalized context.

        When membership_alpha=0.0, behavior matches the earlier descriptor-only shift.

        If gate == 'softmax' and within_symbol_softmax == True, the softmax is computed
        **within each symbol** over its descriptors using temperature `tau`. Otherwise,
        the softmax is computed globally over all descriptors.
        """

        # ---------- 1) Build context + gates ----------
        D = self.D
        vctx = self.ctx_vec(weights=weights, sentence=sentence)
        use_ctx = np.linalg.norm(vctx) > 1e-8

        if not use_ctx:
            return self.plot_map(
                method=method,
                n_neighbors=n_neighbors,
                with_hulls=with_hulls,
                include_centroids=include_centroids,
                normalize_centroids=normalize_centroids,
                figsize=figsize,
                title=title + " (no context)"
            )

        sims = D @ vctx  # cosine sims (rows of D are L2-normalized)

        # ---- gating g_i (only change is the per-symbol softmax branch) ----
        if gate == "relu":
            g = np.maximum(0.0, sims)
        elif gate == "cos":
            g = sims
        elif gate == "softmax":
            if within_symbol_softmax:
                g = np.zeros_like(sims)
                for s in self.symbols:
                    idx = self.symbol_to_idx[s]
                    if len(idx) > 0:
                        g[idx] = softmax(sims[idx], tau=tau)  # softmax INSIDE each symbol
            else:
                g = softmax(sims, tau=tau)  # global softmax across all descriptors
        elif gate == "uniform":
            g = np.ones_like(sims)
        else:
            raise ValueError(f"Unknown gate '{gate}'")

        # ---------- 2) Descriptor-only shifted matrix (for centroid deltas) ----------
        D_ctx_desc = l2_normalize(D + (beta * g[:, None]) * vctx[None, :], axis=1)

        # ---------- 3) Symbol-level centroid shifts ----------
        delta_sym = {}
        for s in self.symbols:
            idx = self.symbol_to_idx[s]
            if not idx:
                continue
            c0 = D[idx].mean(0)
            c1 = D_ctx_desc[idx].mean(0)
            delta_sym[s] = c1 - c0

        # ---------- 4) Final membership-blended shifted descriptors ----------
        D_ctx = D.copy()
        for s in self.symbols:
            idx = self.symbol_to_idx[s]
            if not idx:
                continue
            desc_term = (beta * g[idx][:, None]) * vctx[None, :]
            sym_term  = delta_sym[s][None, :].repeat(len(idx), axis=0)
            delta = (1.0 - membership_alpha) * desc_term + membership_alpha * sym_term
            D_ctx[idx] = D[idx] + delta
        D_ctx = l2_normalize(D_ctx, axis=1)

        # ---------- 5) Choose reducer (mirror plot_map) ----------
        reducer, Z_fit, X_fit, slices = self.get_cached_reducer_and_projection(
            method=method, n_neighbors=n_neighbors,
            include_centroids=include_centroids, normalize_centroids=normalize_centroids
        )
        
        if method == "umap":
            xlbl, ylbl = "UMAP 1", "UMAP 2"
        elif method == "tsne":
            xlbl, ylbl = "t-SNE 1", "t-SNE 2"
        else:
            xlbl, ylbl = "PCA 1", "PCA 2"

        # ---------- 6) Transform Shifted Descriptors ----------
        if method == "tsne":
             # t-SNE fallback: we can't transform, so we just use Z_fit (no shift shown)
             Z_ctx = np.zeros((len(self.descriptors), 2))
             cursor = 0
             for s in self.symbols:
                 idx = self.symbol_to_idx[s]
                 Z_ctx[idx] = Z_fit[cursor : cursor + len(idx)]
                 cursor += len(idx)
                 if include_centroids and len(idx) > 0:
                     cursor += 1
        else:
            Z_ctx = reducer.transform(D_ctx)

        Z_orig = np.zeros((len(self.descriptors), 2), dtype=np.float32)
        # Re-build slices to map Z_fit back to Z_orig
        cursor = 0
        for s in self.symbols:
            idx = self.symbol_to_idx[s]
            # descriptors
            Z_orig[idx] = Z_fit[cursor : cursor + len(idx)]
            cursor += len(idx)
            # centroid
            if include_centroids and len(idx) > 0:
                cursor += 1

        Zc_orig, Zc_ctx = {}, {}
        if include_centroids:
            for s in self.symbols:
                a, b = slices[s]
                idx = self.symbol_to_idx[s]
                if len(idx) == 0:
                    continue
                Zc_orig[s] = Z_fit[b]
            for s in self.symbols:
                idx = self.symbol_to_idx[s]
                if idx:
                    Zc_ctx[s] = Z_ctx[idx].mean(0)
        else:
            for s in self.symbols:
                idx = self.symbol_to_idx[s]
                if idx:
                    Zc_orig[s] = Z_orig[idx].mean(0)
                    Zc_ctx[s]  = Z_ctx[idx].mean(0)

        # ---------- 7) Plot ----------
        if color_dict is None:
            colors = plt.cm.tab20(np.linspace(0, 1, len(self.symbols)))
            color_dict = {s: colors[i] for i, s in enumerate(self.symbols)}

        plt.figure(figsize=figsize)
        for s in self.symbols:
            idx = self.symbol_to_idx[s]
            if not idx:
                continue

            Zi  = Z_orig[idx]
            Zi2 = Z_ctx[idx]

            plt.scatter(Zi[:, 0], Zi[:, 1], s=18, alpha=0.85, color=color_dict[s], label=s)

            dx = (Zi2[:, 0] - Zi[:, 0]) * arrow_scale
            dy = (Zi2[:, 1] - Zi[:, 1]) * arrow_scale
            plt.quiver(
                Zi[:, 0], Zi[:, 1], dx, dy,
                angles="xy", scale_units="xy", scale=1,
                width=0.003, alpha=arrow_alpha, color=color_dict[s]
            )

            if with_hulls and _HAS_SCI and len(idx) >= 3:
                try:
                    hull = ConvexHull(Zi)
                    poly = Zi[hull.vertices]
                    closed = np.vstack([poly, poly[0]])
                    plt.fill(closed[:, 0], closed[:, 1], alpha=0.12, color=color_dict[s])
                    plt.plot(closed[:, 0], closed[:, 1], color=color_dict[s], lw=1.7, alpha=0.6)
                except Exception:
                    pass

            if include_centroids and (s in Zc_orig) and (s in Zc_ctx):
                c0 = Zc_orig[s]; c1 = Zc_ctx[s]
                plt.scatter(c0[0], c0[1], s=140, marker="*", color=color_dict[s],
                            edgecolor="black", zorder=5)
                plt.scatter(c1[0], c1[1], s=140, marker="*", color=color_dict[s],
                            edgecolor="black", zorder=5, alpha=0.55)
                plt.plot([c0[0], c1[0]], [c0[1], c1[1]], '-', color=color_dict[s], alpha=0.6, linewidth=2)

        plt.xlabel(xlbl); plt.ylabel(ylbl)
        plt.title(title + (f"  (membership α={membership_alpha:.2f})" if membership_alpha else ""))
        plt.legend(frameon=False, ncol=2)
        plt.tight_layout()
        plt.show()







    def plot_symbol_similarity_heatmaps(self, simdict, symbol, vmax=1.0, figsize=(13,4), only_delta=True):
        """
        simdict: output of descriptor_similarity_matrices(...)
        Plots S_before | S_after | S_delta for the chosen symbol.
        """
        import matplotlib.pyplot as plt
        try:
            import seaborn as sns
            use_sns = True
        except Exception:
            use_sns = False

        data = simdict[symbol]
        labels = data["descriptors"]
        S0, S1, Sd = data["S_before"], data["S_after"], data["S_delta"]

        if only_delta is False:
            fig, axs = plt.subplots(1, 3, figsize=figsize)
            sd_min_without_diagonal = Sd[np.triu_indices(len(Sd), 1)].min()
            mats = [("Before", S0, (-1, vmax)), ("After", S1, (-1, vmax)), ("Δ Meaning Connection", Sd, (sd_min_without_diagonal, Sd.max()))]

            for ax, (title, M, vr) in zip(axs, mats):
                if use_sns:
                    import seaborn as sns
                    im = sns.heatmap(M, vmin=vr[0], vmax=vr[1], cmap="coolwarm", xticklabels=labels, yticklabels=labels, ax=ax)
                else:
                    im = ax.imshow(M, vmin=vr[0], vmax=vr[1], cmap="coolwarm")
                    ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels, rotation=90)
                    ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels)
                ax.set_title(title)

        if only_delta is True:
            # divide by three the figure width
            # figsize = (int(figsize[0] / 3), figsize[1])
            fig, ax = plt.subplots(figsize=figsize)
            sd_min_without_diagonal = Sd[np.triu_indices(len(Sd), 1)].min()
            if use_sns:
                import seaborn as sns
                # square=True ensures cells are square, cbar_kws shrinks the colorbar to fit better
                im = sns.heatmap(Sd, vmin=sd_min_without_diagonal, vmax=Sd.max(), cmap="coolwarm", 
                                 xticklabels=labels, yticklabels=labels, ax=ax, square=True, cbar_kws={"shrink": 0.7})
                # Much smaller font size for ticks
                ax.tick_params(axis='both', which='major', labelsize=4)
                plt.xticks(rotation=90, fontsize=4)
                plt.yticks(rotation=0, fontsize=4)
            else:
                im = ax.imshow(Sd, vmin=sd_min_without_diagonal, vmax=Sd.max(), cmap="coolwarm")
                ax.set_xticks(range(len(labels)))
                ax.set_xticklabels(labels, rotation=90, fontsize=4)
                ax.set_yticks(range(len(labels)))
                ax.set_yticklabels(labels, fontsize=4)
            ax.set_title(f"Δ After-Before for {symbol}", fontsize=9)
        plt.tight_layout()
        plt.show()

    def plot_attention(self, symbol: str, *,
                        weights=None, sentence=None,
                        tau=0.3, top_n=6, figsize=(6, 4),
                        title: Optional[str] = None, normalize=True, blind_spot=False):
        """Horizontal bar plot of top-N descriptor attention weights (coolwarm style)."""
        _, w = self.conditioned_symbol(symbol, weights, sentence, tau)
        
        # Sort descending for normal, ascending for blind_spot
        top = sorted(w.items(), key=lambda kv: kv[1], reverse=(not blind_spot))[:top_n]
        if not top:
            return


        labels, vals = zip(*top)
        vals = np.array(vals)
        # Normalize for colormap
        norm = (vals - vals.min()) / (vals.max() - vals.min() + 1e-8)
        cmap = plt.cm.coolwarm
        colors = [cmap(n) for n in norm]

        plt.figure(figsize=figsize)
        bars = plt.barh(range(len(vals))[::-1], vals[::-1], color=colors[::-1], edgecolor='gray', linewidth=1.2)
        plt.yticks(range(len(labels))[::-1], labels[::-1])
        vmin, vmax = vals.min(), vals.max()
        if normalize:
            plt.xlim(vmin - 0.01, vmax + 0.01)
        else:
            plt.xlim(0, 1)
        plt.xlabel("Attention weight")
        
        default_title = f"{symbol} – {'blind spots' if blind_spot else 'descriptor attention'}"
        plt.title(title or default_title)
        # Add value labels and subtle shading
        for bar, v in zip(bars, vals[::-1]):
            plt.gca().add_patch(
                plt.Rectangle(
                    (bar.get_x(), bar.get_y()), bar.get_width(), bar.get_height(),
                    color='white', alpha=0.08, zorder=0
                )
            )
            
        plt.tight_layout()
        plt.show()

    def plot_symbol_predictions(self, weights=None, sentence=None, tau=0.5, lam=0.6, alpha=0.85, topk=None, figsize=(7,4), title=None, use_ppr=True):
        """
        Plot symbol prediction scores for current context (horizontal bar chart).
        """

        preds = self.propose(weights=weights, sentence=sentence, tau=tau, lam=lam, alpha=alpha, topk=len(self.symbols) if topk is None else topk, use_ppr=use_ppr)
        symbols, scores, _, _ = zip(*preds)
        colors = plt.cm.tab20(np.linspace(0, 1, len(self.symbols)))
        color_dict = {s: colors[i] for i, s in enumerate(self.symbols)}
        bar_colors = [color_dict[s] for s in symbols]
        
        plt.figure(figsize=figsize)
        plt.barh(range(len(symbols))[::-1], scores[::-1], color=bar_colors[::-1])
        plt.yticks(range(len(symbols))[::-1], symbols[::-1])
        plt.xlabel("Prediction score")
        plt.title(title or "Symbol prediction for context")
        plt.tight_layout()
        plt.show()


    def plot_weight_sweep(self, symbol: str,
                            sweep: List[Dict[str, float]],
                            tau=0.5, figsize=(6, 5),
                            title="Trajectory in weight sweep"):
        """2-D PCA trajectory of the symbol under a list of weight dictionaries."""
        traj = [self.conditioned_symbol(symbol, w, None, tau)[0] for w in sweep]
        Z_traj = self._pca.transform(np.stack(traj))
        Z_desc = self._pca.transform(self.D)

        plt.figure(figsize=figsize)
        plt.scatter(Z_desc[:, 0], Z_desc[:, 1],
                    s=8, alpha=0.08, color="gray")
        plt.plot(Z_traj[:, 0], Z_traj[:, 1], "-o",
                    linewidth=2, markersize=5, color="tab:red")
        for i, (x, y) in enumerate(Z_traj):
            plt.text(x, y, str(i), fontsize=8)
        plt.title(title)
        plt.tight_layout()
        plt.show()




    def plot_graph(self, node_size=600, font_size=10, context_symbols=None,
                pagerank_scores=None, figsize=(11, 9), title="Symbol/Descriptor Graph", seed=42):

        import matplotlib.patches as mpatches
        import numpy as np
        import matplotlib.pyplot as plt
        import networkx as nx

        G = self.G
        # Build color map: assign a color per symbol
        from matplotlib.cm import get_cmap
        cmap = get_cmap("tab20")
        symbol_list = self.symbols
        color_dict = {s: cmap(i / max(1, len(symbol_list)-1)) for i, s in enumerate(symbol_list)}

        pos = nx.spring_layout(G, seed=seed, k=0.6)
        labels = {}

        # For legend later
        legend_handles = [mpatches.Patch(color=color_dict[s], label=s) for s in symbol_list]

        # 1. Plot symbol nodes
        for node in G.nodes():
            if node.startswith("S:"):
                sname = node[2:]
                color = color_dict[sname]
                is_context = context_symbols and sname in context_symbols
                size = node_size * (0.6 + 3.5 * pagerank_scores.get(sname, 0)) if pagerank_scores else node_size * 1.2
                edgecolor = "black" if is_context else "gray"
                linewidth = 2.7 if is_context else 1.0
                alpha = 1.0 if is_context else 0.7
                # Draw context symbol nodes with strong border and size
                nx.draw_networkx_nodes(
                    G, pos,
                    nodelist=[node],
                    node_color=[color],
                    node_size=[size],
                    alpha=alpha,
                    edgecolors=edgecolor,
                    linewidths=linewidth,
                )
                labels[node] = sname

        # 2. Plot descriptor nodes
        for node in G.nodes():
            if node.startswith("D:"):
                dname = node[2:]
                sname = self.owner[dname]
                color = color_dict[sname]
                nx.draw_networkx_nodes(
                    G, pos,
                    nodelist=[node],
                    node_color=[color],
                    node_size=[node_size * 0.68],
                    alpha=0.45,
                    edgecolors="gray",
                    linewidths=0.7,
                )
                labels[node] = dname

        # 3. Edges and labels
        nx.draw_networkx_edges(G, pos, alpha=0.26, width=1.2)
        nx.draw_networkx_labels(G, pos, labels, font_size=font_size, font_weight="bold")

        # 4. Custom legend (with context symbol example if relevant)
        if context_symbols:
            # Add a patch with thick black border to show context highlight
            legend_handles.append(
                mpatches.Patch(facecolor=(1,1,0.8,1), edgecolor="black", linewidth=2.2, label="Context symbol (highlighted)")
            )
        plt.legend(handles=legend_handles, bbox_to_anchor=(1.04, 1), loc="upper left", borderaxespad=0, fontsize=font_size)

        plt.axis("off")
        plt.title(title)
        plt.tight_layout()
        plt.show()

    
def plot_descriptor_attention_heatmap(space, symbol, contexts, context_labels=None, tau=0.1):
    """
    Plot heatmap of descriptor attention weights for a symbol across multiple contexts.
    :param space: SymbolSpace object
    :param symbol: symbol to plot (string)
    :param contexts: list of dicts: each dict with optional keys 'sentence', 'weights'
    :param context_labels: list of str, optional
    :param tau: float, softmax temperature
    """
    descriptors = space.symbols_to_descriptors[symbol]
    att_matrix = []
    for ctx in contexts:
        _, attn = space.conditioned_symbol(symbol, weights=ctx.get("weights"), sentence=ctx.get("sentence"), tau=tau)
        att_matrix.append([attn.get(d, 0.0) for d in descriptors])
    att_matrix = np.array(att_matrix).T  # shape: [descriptors, contexts]

    if context_labels is None:
        context_labels = [f"C{i+1}" for i in range(len(contexts))]

    plt.figure(figsize=(1.8 * len(contexts) + 3, 0.5 * len(descriptors) + 3))
    sns.heatmap(att_matrix, annot=True, cmap="YlGnBu", 
                yticklabels=descriptors, xticklabels=context_labels,
                cbar_kws={'label': 'Attention weight'})
    plt.title(f"Descriptor Attention for '{symbol}' across Contexts")
    plt.xlabel("Context")
    plt.ylabel("Descriptor")
    plt.tight_layout()
    plt.show()

def plot_meaning_landscape(
    space,
    method="umap",
    grid_res=220,
    sigma=None,              # Let sigma adapt to centroid spread if None
    symbols=None,
    figsize=(12, 10),
    colormap="coolwarm"
):
    import matplotlib.colors as mcolors
    from sklearn.preprocessing import StandardScaler

    # 1. Reduce descriptors+centroids to 2D (UMAP/t-SNE = nonlinear)
    if symbols is None:
        symbols = space.symbols
    idxs = [i for s in symbols for i in space.symbol_to_idx[s]]
    desc_2d = space.reduce_2d(method=method)[idxs]
    # Project centroids
    centroids_2d = []
    for s in symbols:
        idx = space.symbol_to_idx[s]
        centroid_vec = space.D[idx].mean(0)
        # UMAP/t-SNE: find centroid in reduced space
        centroid_2d = desc_2d[ [j-idxs[0] for j in idx] ].mean(0)
        centroids_2d.append(centroid_2d)
    centroids_2d = np.stack(centroids_2d)

    # (Optional) Normalize (whiten) 2D space for better separation
    scaler = StandardScaler()
    all_points = np.vstack([desc_2d, centroids_2d])
    all_points_scaled = scaler.fit_transform(all_points)
    desc_2d_scaled = all_points_scaled[:len(desc_2d)]
    centroids_2d_scaled = all_points_scaled[len(desc_2d):]

    # Choose sigma adaptively (default: 0.2 × median inter-centroid distance)
    if sigma is None:
        dists = []
        for i in range(len(centroids_2d_scaled)):
            for j in range(i+1, len(centroids_2d_scaled)):
                dists.append(np.linalg.norm(centroids_2d_scaled[i] - centroids_2d_scaled[j]))
        med_dist = np.median(dists) if dists else 0.2
        sigma = 0.2 * med_dist

    # 2. Make grid over the scaled 2D space
    xlim = (desc_2d_scaled[:,0].min()-0.6, desc_2d_scaled[:,0].max()+0.6)
    ylim = (desc_2d_scaled[:,1].min()-0.6, desc_2d_scaled[:,1].max()+0.6)
    xx, yy = np.meshgrid(
        np.linspace(*xlim, grid_res), np.linspace(*ylim, grid_res)
    )
    pos = np.stack([xx.ravel(), yy.ravel()], axis=1)

    # 3. Compute semantic "potential" (energy landscape): sum of negative sharp Gaussians at each centroid
    potential = np.zeros(len(pos))
    for mu in centroids_2d_scaled:
        dist2 = ((pos - mu)**2).sum(1)
        potential -= np.exp(-dist2 / (2 * sigma ** 2))
    potential = potential.reshape(xx.shape)

    # 4. Plot surface + centroids + descriptors
    plt.figure(figsize=figsize)
    cs = plt.contourf(xx, yy, potential, 60, cmap=colormap, alpha=0.92)
    plt.colorbar(cs, label="Semantic Potential (lower = attractor)")

    # Colors per symbol
    color_list = plt.cm.tab20(np.linspace(0, 1, len(symbols)))
    # Plot centroids
    for i, s in enumerate(symbols):
        plt.scatter(
            *centroids_2d_scaled[i],
            s=300,
            marker='*',
            edgecolor='k',
            color=color_list[i],
            label=s,
            zorder=6,
            linewidths=1.3
        )
    # Plot descriptors (faded)
    plt.scatter(desc_2d_scaled[:,0], desc_2d_scaled[:,1], s=26, alpha=0.11, color='k', label='Descriptors', zorder=1)
    # set axis to (-1, 1) in both dimensions
    plt.xlim(-1, 1)
    plt.ylim(-1, 1)
    plt.title("Meaning Landscape for Symbols (Semantic Energy Surface, Maximized Granularity)")
    plt.legend(loc="upper right", bbox_to_anchor=(1.24, 1.03))
    plt.tight_layout()
    plt.show()



import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

def plot_meaning_energy_3d(space, method="pca", grid_res=220, sigma=0.14, figsize=(12, 8)):
    # 1. 2D reduction
    desc_2d = space.reduce_2d(method=method)
    # 2. Get centroids
    centroids_2d = []
    for s in space.symbols:
        idx = space.symbol_to_idx[s]
        centroid_vec = space.D[idx].mean(0)
        centroid_2d = space._pca.transform([centroid_vec])[0] if method == "pca" else desc_2d[idx].mean(0)
        centroids_2d.append(centroid_2d)
    centroids_2d = np.stack(centroids_2d)
    colors = plt.cm.tab20(np.linspace(0, 1, len(space.symbols)))
    symbols = space.symbols

    # 3. Create grid over embedding space
    xlim = (desc_2d[:, 0].min() - 0.2, desc_2d[:, 0].max() + 0.2)
    ylim = (desc_2d[:, 1].min() - 0.2, desc_2d[:, 1].max() + 0.2)
    xx, yy = np.meshgrid(np.linspace(*xlim, grid_res), np.linspace(*ylim, grid_res))
    pos = np.stack([xx, yy], axis=-1)  # shape [grid_res, grid_res, 2]

    # 4. Energy surface: sum of negative Gaussians
    E = np.zeros(xx.shape)
    for mu, c in zip(centroids_2d, colors):
        dx = xx - mu[0]
        dy = yy - mu[1]
        E -= np.exp(-(dx**2 + dy**2) / (2 * sigma**2))
    # Optionally, add a constant to bring the maximum up to 0 (just for color mapping)
    E = E - E.max()

    # 5. Plot 3D surface
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(xx, yy, E, cmap='coolwarm', alpha=0.93, linewidth=0, antialiased=True)

    # Overlay centroids
    for i, (mu, sym) in enumerate(zip(centroids_2d, symbols)):
        ax.scatter(mu[0], mu[1], np.min(E), color=colors[i], marker="*", s=170, label=sym, edgecolor='k')

    ax.set_title("3D Meaning Energy Landscape for Symbols")
    ax.set_xlabel("Embedding dim 1")
    ax.set_ylabel("Embedding dim 2")
    ax.set_zlabel("Semantic Potential (lower=attractor)")
    fig.colorbar(surf, shrink=0.6, aspect=10, label="Semantic Potential")
    ax.legend(loc="upper left", bbox_to_anchor=(1, 1))
    plt.tight_layout()
    plt.show()


def plot_symbol_context_trajectory(space, symbol, sentences, tau=0.3, method="pca", figsize=(7,5), title=None):
    """
    Plot the trajectory of the context-conditioned embedding for a symbol
    as the context sentence varies. Points are projected to 2D via PCA/UMAP.
    """
    # Get conditioned embeddings for each context
    embeddings = []
    for sent in sentences:
        emb, _ = space.conditioned_symbol(symbol, sentence=sent, tau=tau)
        embeddings.append(emb)
    embeddings = np.stack(embeddings)
    
    # Project to 2D
    all_desc = space.D  # all descriptor embeddings
    reducer = space._pca if method == "pca" else PCA(n_components=2, random_state=42).fit(all_desc)
    traj_2d = reducer.transform(embeddings)
    desc_2d = reducer.transform(all_desc)
    
    # Plot all descriptors faintly
    plt.figure(figsize=figsize)
    plt.scatter(desc_2d[:,0], desc_2d[:,1], s=12, alpha=0.11, color="gray")
    
    # Plot trajectory
    plt.plot(traj_2d[:,0], traj_2d[:,1], '-o', color="crimson", linewidth=2.5, label=f"{symbol} (contextual centroid)")
    for i, (x, y) in enumerate(traj_2d):
        plt.text(x, y, str(i+1), fontsize=10, weight='bold', color="crimson")
    plt.title(title or f"Trajectory of '{symbol}' meaning across contexts")
    plt.xlabel("Latent dim 1")
    plt.ylabel("Latent dim 2")
    plt.legend()
    plt.tight_layout()
    plt.show()



def plot_symbol_centroid_shifts(space, symbol, sentences, tau=0.3, method="pca", figsize=(8,6), title=None, context_labels=None):
    """
    Plot 2D descriptor map with original centroid and context-conditioned centroids
    for a symbol, showing movement across contexts.
    """
    # Get all descriptor embeddings (projected to 2D)
    all_desc = space.D
    reducer = space._pca if method == "pca" else PCA(n_components=2, random_state=42).fit(all_desc)
    desc_2d = reducer.transform(all_desc)
    # Get indices for the symbol's descriptors
    idx = space.symbol_to_idx[symbol]
    desc_2d_sym = desc_2d[idx]
    descriptors = space.symbols_to_descriptors[symbol]

    # Original centroid
    orig_centroid = desc_2d_sym.mean(0)

    # Context-conditioned centroids
    centroids = []
    for sent in sentences:
        emb, _ = space.conditioned_symbol(symbol, sentence=sent, tau=tau)
        centroids.append(reducer.transform([emb])[0])
    centroids = np.stack(centroids)
    
    # Plot descriptors for all symbols (faded)
    plt.figure(figsize=figsize)
    plt.scatter(desc_2d[:,0], desc_2d[:,1], s=12, alpha=0.08, color="gray", label="Descriptors")
    # add names for descriptors
    for i, (x, y) in enumerate(desc_2d_sym):
        plt.text(x, y, descriptors[i], fontsize=8, color="black", alpha=0.7, ha='center', va='center')
        
    # be sure there is no legend for descriptors
    plt.legend().remove()
    # Plot descriptors for this symbol (highlighted)
    plt.scatter(desc_2d_sym[:,0], desc_2d_sym[:,1], s=46, alpha=0.45, color="orange", label=f"{symbol} descriptors")

    # Plot original centroid
    plt.scatter(orig_centroid[0], orig_centroid[1], s=170, marker="*", color="black", edgecolor="orange", label="Original centroid", zorder=5)
    # Plot shifted centroids and arrows
    plt.scatter(centroids[:,0], centroids[:,1], s=110, marker="*", color="crimson", label="Contextual centroid(s)", zorder=6)
    for i in range(len(centroids)):
        plt.arrow(orig_centroid[0], orig_centroid[1],
                  centroids[i,0] - orig_centroid[0], centroids[i,1] - orig_centroid[1],
                  color="darkred", width=0.002, head_width=0.01, length_includes_head=True, alpha=0.85, zorder=3)
        # Optionally label
        lbl = context_labels[i] if context_labels and i < len(context_labels) else str(i+1)
        plt.text(centroids[i,0], centroids[i,1], lbl, fontsize=8, color="crimson", weight="bold", zorder=7)

    xlim = (desc_2d[:,0].min() - 0.01, desc_2d[:,0].max() + 0.01)
    ylim = (desc_2d[:,1].min() - 0.01, desc_2d[:,1].max() + 0.01)
    plt.xlim(xlim)
    plt.ylim(ylim)
    plt.title(title or f"Centroid shift for '{symbol}' across contexts")
    plt.xlabel("Latent dim 1")
    plt.ylabel("Latent dim 2")
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_contexts_and_symbol_centroid(space, symbol, sentences, tau=0.3, method="pca", figsize=(7,5), context_labels=None):
    all_desc = space.D
    reducer = space._pca if method == "pca" else PCA(n_components=2, random_state=42).fit(all_desc)
    desc_2d = reducer.transform(all_desc)
    idx = space.symbol_to_idx[symbol]
    desc_2d_sym = desc_2d[idx]
    orig_centroid = desc_2d_sym.mean(0)

    # Context-conditioned centroids and context vectors
    centroids = []
    context_vecs = []
    for sent in sentences:
        emb, _ = space.conditioned_symbol(symbol, sentence=sent, tau=tau)
        centroids.append(reducer.transform([emb])[0])
        cvec = space.embedder.encode([sent])[0]
        context_vecs.append(reducer.transform([cvec])[0])
    centroids = np.stack(centroids)
    context_vecs = np.stack(context_vecs)

    plt.figure(figsize=figsize)
    plt.scatter(desc_2d[:,0], desc_2d[:,1], s=12, alpha=0.11, color="gray")
    plt.scatter(desc_2d_sym[:,0], desc_2d_sym[:,1], s=46, alpha=0.45, color="orange", label=f"{symbol} descriptors")
    plt.scatter(orig_centroid[0], orig_centroid[1], s=170, marker="*", color="black", edgecolor="orange", label="Original centroid", zorder=5)
    plt.scatter(centroids[:,0], centroids[:,1], s=110, marker="*", color="crimson", label="Contextual centroid(s)", zorder=6)
    plt.scatter(context_vecs[:,0], context_vecs[:,1], s=70, marker="P", color="royalblue", label="Context vector(s)", zorder=6)
    for i in range(len(centroids)):
        plt.arrow(orig_centroid[0], orig_centroid[1],
                  centroids[i,0] - orig_centroid[0], centroids[i,1] - orig_centroid[1],
                  color="crimson", width=0.01, head_width=0.04, length_includes_head=True, alpha=0.85, zorder=3)
        lbl = context_labels[i] if context_labels and i < len(context_labels) else str(i+1)
        plt.text(centroids[i,0], centroids[i,1], lbl, fontsize=10, color="crimson", weight="bold", zorder=7)
        plt.text(context_vecs[i,0], context_vecs[i,1], lbl, fontsize=9, color="royalblue", style='italic', zorder=8)
    plt.title(f"Centroid/context vectors for '{symbol}'")
    plt.xlabel("Latent dim 1")
    plt.ylabel("Latent dim 2")
    plt.legend()
    plt.tight_layout()
    plt.show()

def project_to_centroid_context_plane(centroid, context, vectors):
    """
    Project vectors onto the 2D plane spanned by the centroid and context.
    centroid: [D,] np.array
    context:  [D,] np.array
    vectors:  [N, D] np.array
    Returns: [N, 2] np.array
    """
    # 1. Make orthonormal basis
    v1 = (context - centroid)
    v1 = v1 / (np.linalg.norm(v1) + 1e-9)
    # Pick any vector not colinear for second axis
    tmp = np.random.randn(*v1.shape)
    v2 = tmp - v1 * np.dot(tmp, v1)
    v2 = v2 / (np.linalg.norm(v2) + 1e-9)
    # Project all vectors onto [v1, v2] basis
    projections = np.stack([np.dot(vec - centroid, v1) for vec in vectors])
    projections2 = np.stack([np.dot(vec - centroid, v2) for vec in vectors])
    return np.stack([projections, projections2], axis=1)

def plot_centroid_context_plane(space, symbol, sentences, tau=0.3, figsize=(7,5), context_labels=None):
    # --- Get original centroid and context vectors ---
    orig_cent = space.symbol_centroids[symbol]
    context_vecs = np.stack([space.embedder.encode([sent])[0] for sent in sentences])
    centroids = np.stack([space.conditioned_symbol(symbol, sentence=sent, tau=tau)[0] for sent in sentences])
    # Project all points onto the 2D plane (first context for orientation)
    Z_centroids = project_to_centroid_context_plane(orig_cent, context_vecs[0], centroids)
    Z_contexts = project_to_centroid_context_plane(orig_cent, context_vecs[0], context_vecs)
    Z_orig = np.zeros(2)
    Z_context0 = project_to_centroid_context_plane(orig_cent, context_vecs[0], [context_vecs[0]])[0]
    # --- Plot ---
    plt.figure(figsize=figsize)
    plt.scatter(Z_contexts[:,0], Z_contexts[:,1], s=80, marker="P", color="royalblue", label="Context vector(s)", zorder=6)
    plt.scatter(Z_centroids[:,0], Z_centroids[:,1], s=110, marker="*", color="crimson", label="Contextual centroid(s)", zorder=7)
    plt.scatter([Z_orig[0]], [Z_orig[1]], s=170, marker="*", color="black", edgecolor="orange", label="Original centroid", zorder=5)
    # draw arrows and labels
    for i in range(len(centroids)):
        plt.arrow(Z_orig[0], Z_orig[1],
                  Z_centroids[i,0], Z_centroids[i,1],
                  color="crimson", width=0.004, head_width=0.016, length_includes_head=True, alpha=0.8, zorder=4)
        # draw line to context too
        plt.plot([Z_orig[0], Z_contexts[i,0]], [Z_orig[1], Z_contexts[i,1]], '--', color="gray", alpha=0.6, linewidth=1.2)
        lbl = context_labels[i] if context_labels and i < len(context_labels) else str(i+1)
        plt.text(Z_centroids[i,0], Z_centroids[i,1], lbl, fontsize=10, color="crimson", weight="bold", zorder=8)
        plt.text(Z_contexts[i,0], Z_contexts[i,1], lbl, fontsize=9, color="royalblue", style='italic', zorder=9)
    plt.title(f"Context-conditioned centroid shifts for '{symbol}' (true 2D plane)")
    plt.xlabel("axis 1 (centroid→context dir)")
    plt.ylabel("axis 2 (orthogonal)")
    plt.legend()
    plt.tight_layout()
    plt.show()


def context_delta_graph(
    space, *,
    sentence=None, weights=None,
    strategy: str = "gate",
    beta=0.6, gate="relu", tau=0.2,
    within_symbol_softmax: bool = False,
    gamma: float = 0.5,
    prompt_template: str = "In this context: {sent}. Descriptor: {desc}",
    top_abs_edges=150, sym_filter=None,
    min_abs_delta=0.02, within_symbol=True, only_symbol=None, connected_only=False,
    pool_type: str = "avg",
    pool_w: float = 0.7,
    # NEW ▼
    membership_alpha: float = 0.0,
):
    D0 = space.D
    D1 = space.make_shifted_matrix(
        weights=weights, sentence=sentence,
        strategy=strategy,
        beta=beta, gate=gate, tau=tau,
        within_symbol_softmax=within_symbol_softmax,
        gamma=gamma, prompt_template=prompt_template,
        pool_type=pool_type, pool_w=pool_w,
        # NEW ▼
        membership_alpha=membership_alpha,
    )

    # --- choose which descriptors to keep ---
    if only_symbol is not None:
        sym_filter = [only_symbol]  # override

    if sym_filter:
        keep_idx = []
        for s in sym_filter:
            keep_idx.extend(space.symbol_to_idx[s])
        keep_idx = sorted(set(keep_idx))
    else:
        keep_idx = list(range(len(space.descriptors)))

    D0 = D0[keep_idx]
    D1 = D1[keep_idx]
    names  = [space.descriptors[i] for i in keep_idx]
    owners = {space.descriptors[i]: space.owner[space.descriptors[i]] for i in keep_idx}

    # --- deltas ---
    C0 = D0 @ D0.T
    C1 = D1 @ D1.T
    Delta = C1 - C0
    np.fill_diagonal(Delta, 0.0)

    # --- optionally mask cross-symbol pairs ---
    if within_symbol:
        sym_of_row = np.array([owners[n] for n in names])
        same = (sym_of_row[:, None] == sym_of_row[None, :])
        Delta = Delta * same.astype(Delta.dtype)

    # --- pick strongest |Δ| edges (upper triangle only) ---
    tri = np.triu_indices_from(Delta, k=1)
    vals = Delta[tri]
    order = np.argsort(np.abs(vals))[::-1]

    G = nx.Graph()
    for n in names:
        G.add_node(n, kind="desc", symbol=owners[n])

    kept = 0
    for k in order:
        if kept >= top_abs_edges:
            break
        d = float(vals[k])
        if abs(d) < min_abs_delta:
            continue
        i, j = tri[0][k], tri[1][k]
        if d == 0.0:
            continue
        G.add_edge(names[i], names[j],
                   delta=d,
                   sign=("up" if d > 0 else "down"),
                   abs_delta=abs(d))
        kept += 1

    # --- remove isolated nodes if requested ---
    if connected_only:
        G.remove_nodes_from(list(nx.isolates(G)))

    return G



def plot_delta_graph(
    G,
    node_alpha=0.9,
    min_edge_alpha=0.15,
    title="Context Δ graph",
    color_dict=None,            # pass space.get_symbol_color_dict() for consistency
    *,
    # node sizing
    node_size_base=220,
    node_size_scale=1200.0,
    normalize_node_sizes=True,  
    # edge sizing
    edge_width_min=0.6,         
    edge_width_max=6.0,         
    normalize_edge_widths=True,  
    figsize=(6, 4),           
    ax=None                   
):
    import matplotlib.pyplot as plt, networkx as nx
    from matplotlib.patches import Patch
    import numpy as np

    # Symbols present
    syms_present = sorted({G.nodes[n]["symbol"] for n in G.nodes()})

    # Colors
    if color_dict is None:
        base = plt.cm.tab20(np.linspace(0, 1, max(1, len(syms_present))))
        cdict = {s: base[i] for i, s in enumerate(syms_present)}
    else:
        cdict = {s: color_dict.get(s, (0.5, 0.5, 0.5, 1.0)) for s in syms_present}

    pos = nx.spring_layout(G, seed=42, k=0.6)

    # ---- node sizes from total |Δ| of incident edges ----
    abs_sum = {n: 0.0 for n in G.nodes()}
    for u, v, d in G.edges(data=True):
        a = float(d.get("abs_delta", 0.0))
        abs_sum[u] += a
        abs_sum[v] += a

    if normalize_node_sizes and len(abs_sum) > 0:
        arr = np.array(list(abs_sum.values()), dtype=float)
        lo, hi = arr.min(), arr.max()
        if hi - lo < 1e-12:
            norm = np.ones_like(arr) * 0.5
        else:
            norm = (arr - lo) / (hi - lo)
        node_sizes = [node_size_base + node_size_scale * n for n in norm]
        node_sizes = dict(zip(G.nodes(), node_sizes))
    else:
        node_sizes = {n: node_size_base + node_size_scale * abs_sum[n] for n in G.nodes()}

    nx.draw_networkx_nodes(
        G, pos,
        nodelist=list(G.nodes()),
        node_color=[cdict[G.nodes[n]["symbol"]] for n in G.nodes()],
        edgecolors="black", linewidths=0.3,
        alpha=node_alpha,
        node_size=[node_sizes[n] for n in G.nodes()]
    )

    # ---- edge widths from |Δ|, min–max normalized ----
    edges = list(G.edges(data=True))
    abs_d = np.array([float(d.get("abs_delta", 0.0)) for _, _, d in edges], dtype=float)
    if normalize_edge_widths and len(abs_d) > 0:
        lo, hi = abs_d.min(), abs_d.max()
        if hi - lo < 1e-12:
            nabs = np.ones_like(abs_d) * 0.5
        else:
            nabs = (abs_d - lo) / (hi - lo)
        widths = edge_width_min + nabs * (edge_width_max - edge_width_min)
    else:
        widths = np.maximum(edge_width_min, abs_d * (edge_width_max - edge_width_min))

    width_map = { (u, v): w for (u, v, _), w in zip(edges, widths) }
    width_map.update({ (v, u): w for (u, v, _), w in zip(edges, widths) })  # undirected

    up = [(u, v) for u, v, d in edges if d["delta"] > 0]
    dn = [(u, v) for u, v, d in edges if d["delta"] < 0]
    w_up = [width_map[(u, v)] for u, v in up]
    w_dn = [width_map[(u, v)] for u, v in dn]

    nx.draw_networkx_edges(G, pos, edgelist=up, edge_color="darkred",  width=w_up, alpha=0.75)
    nx.draw_networkx_edges(G, pos, edgelist=dn, edge_color="tab:blue", width=w_dn, alpha=0.55)

    nx.draw_networkx_labels(G, pos, font_size=8, font_color="white")

    handles = [Patch(facecolor=cdict[s], edgecolor='black', label=s) for s in syms_present]
    if handles:
        plt.legend(handles=handles, title="Symbols", loc='best', fontsize=8)

    plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    plt.show()


def plot_contextual_subgraph_colored(
    space,
    context_sentence,
    topk_symbols=3,
    topk_desc=3,
    method="ppr",
    alpha=0.85,
    tau=0.1,
    normalize=False,
    global_color_map=None,  # pass global palette here
    figsize=(8, 6)
):
    """
    Plot contextual subgraph for a given sentence using a fixed global color palette.

    normalize=True will subtract baseline PPR with uniform personalization
    to remove centrality prior.
    """

    if method == "ppr":
        # 1. Contextual PPR
        results_raw = space.recommend_symbols_with_ppr_descriptors(
            context_sentence, topk=None, alpha=alpha, tau=tau, n_best=topk_desc
        )

        # 2. Baseline PPR with uniform personalization
        uniform_weights = {f"S:{s}": 1.0 / len(space.symbols) for s in space.symbols}
        baseline_scores = nx.pagerank(
            space.G, alpha=alpha, personalization=uniform_weights, weight="weight"
        )

        # 3. Remove centrality prior if normalize=True
        norm_results = []
        for r in results_raw:
            sym_node = f"S:{r['symbol']}"
            if normalize:
                norm_score = r['score'] - baseline_scores.get(sym_node, 0.0)
            else:
                norm_score = r['score']

            norm_results.append({
                "symbol": r['symbol'],
                "score": norm_score,
                "best_descriptors": r['best_descriptors']
            })

        # 4. Sort and take top k
        results = sorted(norm_results, key=lambda x: x['score'], reverse=True)[:topk_symbols]

        for r in results:
            print(f"{r['symbol']}: Δscore={r['score']:.6f} | {r['best_descriptors']}")

    elif method == "softmax":
        results = space.recommend_symbols_with_softmax_attention(
            context_sentence, topk=topk_symbols, tau=tau, n_best=topk_desc
        )


    # --- Build subgraph ---
    symbols = [r['symbol'] for r in results]
    symbol_to_desc = {r['symbol']: r['best_descriptors'] for r in results}
    descriptors = [d for r in results for d in r['best_descriptors']]

    # Use global color map if provided, else fall back to local mapping
    if global_color_map is None:
        cmap = plt.cm.tab20
        color_map = {s: cmap(i / max(1, len(space.symbols)-1)) for i, s in enumerate(space.symbols)}
    else:
        color_map = global_color_map

    nodes = [f"S:{s}" for s in symbols] + [f"D:{d}" for d in descriptors]
    subG = space.G.subgraph(nodes).copy()
    pos = nx.spring_layout(subG, seed=42)

    plt.figure(figsize=figsize)

    # Symbol nodes
    for s in symbols:
        nx.draw_networkx_nodes(subG, pos,
                               nodelist=[f"S:{s}"],
                               node_color=[color_map[s]],
                               node_shape="s", node_size=750)

    # Descriptor nodes (lightened color of symbol)
    for s in symbols:
        desc_nodes = [f"D:{d}" for d in symbol_to_desc[s]]
        nx.draw_networkx_nodes(subG, pos,
                               nodelist=desc_nodes,
                               node_color=[lighten_color(color_map[s], amount=0.52)]*len(desc_nodes),
                               node_shape="o", node_size=430, alpha=0.9)

    scale_factor = 5
    edge_widths = [subG[u][v].get("weight", 1.0) * scale_factor for u, v in subG.edges()]
    nx.draw_networkx_edges(subG, pos, width=edge_widths, alpha=0.4, edge_color="gray")


    labels = {n: n[2:] for n in subG}
    nx.draw_networkx_labels(subG, pos, labels, font_size=13, font_weight="bold", font_color="white")

    plt.title(f"Contextual Subgraph for: '{context_sentence[:60]}...'")
    plt.axis("off")
    plt.tight_layout()
    plt.show()

# Also make your other helper robust:
def lighten_color(color, amount=0.5):
    r, g, b, a = _rgba_tuple(color)
    r, g, b = r + (1 - r) * amount, g + (1 - g) * amount, b + (1 - b) * amount
    return (min(1, r), min(1, g), min(1, b), a)
    
def _rgba_tuple(c):
        """Robustly convert any Matplotlib color (hex, name, RGB/RGBA tuple) to (r,g,b,a)."""
        r, g, b, a = mcolors.to_rgba(c)
        return float(r), float(g), float(b), float(a)

def plot_ambiguity_metrics(
    space,
    *,
    sort_by: str = "dispersion",   # "dispersion" | "leakage" | "entropy" | "none"
    rescale: tuple[float, float] = (0.1, 0.9),
    figsize=(7, 4),
    color_dict: dict[str, tuple] | None = None,
    return_table: bool = False
):
    """
    Compute and plot ambiguity metrics per symbol:
      - dispersion(space.dispersion)
      - leakage(space.leakage)
      - entropy(space.soft_entropy)

    Returns a Matplotlib Figure (and optionally a metrics table dict).
    Designed to be Streamlit-safe (no plt.show()).
    """
    import numpy as np
    import matplotlib.pyplot as plt

    symbols = list(space.symbols)
    if not symbols:
        fig = plt.figure(figsize=figsize)
        plt.text(0.5, 0.5, "No symbols found", ha="center", va="center")
        plt.axis("off")
        return (fig, {}) if return_table else fig

    # --- compute raw metrics ---
    disp = np.array([space.dispersion(s) for s in symbols], dtype=float)
    leak = np.array([space.leakage(s) for s in symbols], dtype=float)
    entr = np.array([space.soft_entropy(s) for s in symbols], dtype=float)

    # --- pick sort order ---
    sort_key = {
        "dispersion": disp,
        "leakage":    leak,
        "entropy":    entr,
        "none":       np.arange(len(symbols))
    }.get(sort_by.lower(), disp)

    order = np.argsort(sort_key)[::-1] if sort_by.lower() != "none" else np.arange(len(symbols))

    symbols_sorted = [symbols[i] for i in order]
    disp_sorted = disp[order]
    leak_sorted = leak[order]
    entr_sorted = entr[order]

    # --- min-max normalize then rescale to improve visibility ---
    def _safe_minmax(x):
        x = np.asarray(x, float)
        lo, hi = float(np.min(x)), float(np.max(x))
        if hi - lo < 1e-12:
            return np.ones_like(x) * 0.5
        return (x - lo) / (hi - lo)

    a, b = rescale
    disp_n = a + (b - a) * _safe_minmax(disp_sorted)
    leak_n = a + (b - a) * _safe_minmax(leak_sorted)
    entr_n = a + (b - a) * _safe_minmax(entr_sorted)

    # --- colors (stable by symbol) ---
    if color_dict is None:
        base = plt.cm.tab20(np.linspace(0, 1, max(1, len(space.symbols))))
        color_dict = {s: base[i] for i, s in enumerate(space.symbols)}

    colors_sorted = [color_dict[s] for s in symbols_sorted]

    

    def _darken(c, amount=0.35):
        """
        Darken color by 'amount' in [0,1]; 0=no change, 1=black.
        Works for hex strings, named colors, RGB/RGBA tuples.
        """
        r, g, b, a = _rgba_tuple(c)
        r, g, b = r * (1 - amount), g * (1 - amount), b * (1 - amount)
        return (max(0, r), max(0, g), max(0, b), a)

    def _lighten(c, amount=0.35):
        """
        Lighten color by 'amount' in [0,1]; 0=no change, 1=white.
        """
        r, g, b, a = _rgba_tuple(c)
        r, g, b = r + (1 - r) * amount, g + (1 - g) * amount, b + (1 - b) * amount
        return (min(1, r), min(1, g), min(1, b), a)

    

    # --- plot ---
    fig, ax = plt.subplots(figsize=figsize)
    x = np.arange(len(symbols_sorted))
    w = 0.28

    ax.bar(x - w, disp_n, width=w, color=[_darken(c) for c in colors_sorted], label="Dispersion")
    ax.bar(x,       leak_n, width=w, color=colors_sorted,                      label="Leakage")
    ax.bar(x + w,   entr_n, width=w, color=[_lighten(c) for c in colors_sorted], alpha=0.7, label="Entropy")

    ax.set_xticks(x)
    ax.set_xticklabels(symbols_sorted, rotation=90, fontsize=9)
    ax.set_ylabel("Metric value (normalized)")
    title_map = {"dispersion": "Sorted by Dispersion", "leakage": "Sorted by Leakage",
                 "entropy": "Sorted by Entropy", "none": "Original order"}
    ax.set_title(f"Symbol Ambiguity Metrics — {title_map.get(sort_by.lower(), 'Sorted by Dispersion')}")
    ax.set_ylim(0, 1.05)
    ax.legend()
    fig.tight_layout()

    table = {
        "symbol": symbols_sorted,
        "dispersion": disp_sorted.tolist(),
        "leakage": leak_sorted.tolist(),
        "entropy": entr_sorted.tolist(),
        "dispersion_norm": disp_n.tolist(),
        "leakage_norm": leak_n.tolist(),
        "entropy_norm": entr_n.tolist(),
    }
    return (fig, table) if return_table else fig
