# multimodal_archetype_miner.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np
import networkx as nx
import re
from collections import Counter, defaultdict

@dataclass
class MMItem:
    id: str
    text: Optional[str] = None
    image_path: Optional[str] = None
    audio_path: Optional[str] = None
    meta: Optional[dict] = None

# ---- util ----
def l2n(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x, axis=1, keepdims=True) + 1e-8
    return x / n

def simple_noun_phrases(s: str) -> List[str]:
    # super lightweight: lowercase, keep 1-3 word noun-ish chunks
    s = s.lower()
    toks = re.findall(r"[a-zA-ZÀ-ÿ0-9']+", s)
    phrases = []
    for i in range(len(toks)):
        for L in (1,2,3):
            if i+L <= len(toks):
                ph = " ".join(toks[i:i+L])
                if len(ph) >= 3:
                    phrases.append(ph)
    return phrases

# ---- main miner ----
class ArchetypeMiner:
    def __init__(self,
                 text_encoder,       # .encode(list[str]) -> [N,D]
                 image_encoder,      # .encode_images(list[path]) -> [N,D]
                 audio_encoder,      # .embed_audio_files(list[path]) -> [N,D]
                 canon: str = "text",
                 ridge_lambda: float = 1e-3,
                 seed_align_pairs: Optional[List[Tuple[str,str]]] = None # (item_id_text, item_id_image) etc.
                 ):
        self.text_enc = text_encoder
        self.img_enc  = image_encoder
        self.aud_enc  = audio_encoder
        self.canon = canon
        self.ridge_lambda = ridge_lambda
        self.items: Dict[str, MMItem] = {}
        self.seed_align_pairs = seed_align_pairs or []

    def add(self, item: MMItem):
        self.items[item.id] = item

    # ---- embeddings ----
    def _embed_all(self):
        ids = list(self.items.keys())
        text_ids, img_ids, aud_ids = [], [], []
        texts, imgs, auds = [], [], []
        for i in ids:
            it = self.items[i]
            if it.text:       text_ids.append(i); texts.append(it.text)
            if it.image_path: img_ids.append(i);  imgs.append(it.image_path)
            if it.audio_path: aud_ids.append(i);  auds.append(it.audio_path)

        E = {}
        if texts:
            E["text"] = (text_ids, l2n(self.text_enc.encode(texts)))
        else: E["text"] = ([], np.zeros((0, 1)))
        if imgs:
            E["image"] = (img_ids, l2n(self.img_enc.encode_images(imgs)))
        else: E["image"] = ([], np.zeros((0, 1)))
        if auds:
            E["audio"] = (aud_ids, l2n(self.aud_enc.embed_audio_files(auds)))
        else: E["audio"] = ([], np.zeros((0, 1)))
        return E

    def _learn_ridge(self, X: np.ndarray, Y: np.ndarray, lam: float) -> np.ndarray:
        XtX = X.T @ X
        return np.linalg.solve(XtX + lam*np.eye(XtX.shape[0]), X.T @ Y)

    def _align(self, E: Dict[str, Tuple[List[str], np.ndarray]]) -> Dict[str, Tuple[List[str], np.ndarray]]:
        # Align each modality to the canonical one using paired ids (same item id present in both).
        canon_ids, canon_vecs = E.get(self.canon, ([], np.zeros((0,1))))
        out = {self.canon: (canon_ids, canon_vecs)}
        for m,(ids,V) in E.items():
            if m == self.canon or V.shape[0]==0: 
                if m!=self.canon: out[m]=(ids,V)
                continue
            commons = sorted(set(ids) & set(canon_ids))
            if len(commons) >= 8:
                Xi = np.stack([V[ids.index(cid)] for cid in commons], 0)
                Yi = np.stack([canon_vecs[canon_ids.index(cid)] for cid in commons], 0)
                W = self._learn_ridge(Xi, Yi, self.ridge_lambda)
                V = l2n(V @ W)
            out[m] = (ids, V)
        return out

    def _fuse_early(self, E: Dict[str, Tuple[List[str], np.ndarray]]) -> Tuple[List[str], np.ndarray]:
        # Intersect ids present in at least one modality; fill missing with zeros then L2 normalize row-wise
        all_ids = sorted(set(sum([ids for ids,_ in E.values()], [])))
        mats = []
        for m in ("text","image","audio"):
            ids, V = E[m]
            if V.size == 0:
                mats.append(np.zeros((len(all_ids), 0)))
                continue
            idx = {i:p for p,i in enumerate(ids)}
            X = np.zeros((len(all_ids), V.shape[1]))
            for r,i in enumerate(all_ids):
                if i in idx: X[r] = V[idx[i]]
            mats.append(X)
        Z = np.concatenate(mats, axis=1)
        Z = l2n(Z)
        return all_ids, Z

    # ---- build similarity graph and cluster ----
    def build_graph(self, ids: List[str], Z: np.ndarray, k: int=12) -> nx.Graph:
        S = Z @ Z.T
        np.fill_diagonal(S, 0.0)
        kk = min(k, max(1, S.shape[0]-1))
        A = np.zeros_like(S)
        idx = np.argpartition(-S, kk, axis=1)[:, :kk]
        rows = np.arange(S.shape[0])[:, None]
        A[rows, idx] = S[rows, idx]
        A = np.maximum(A, A.T)
        G = nx.Graph()
        for i,u in enumerate(ids): G.add_node(u)
        nz = np.argwhere(A>0)
        for i,j in nz:
            if i<j: G.add_edge(ids[i], ids[j], w=float(A[i,j]))
        return G

    def cluster(self, G: nx.Graph, min_size: int=8) -> Dict[int, List[str]]:
        # Leiden/Louvain if available; fallback to connected components on kNN graph
        try:
            import igraph as ig, leidenalg as la  # optional
            g = ig.Graph(n=G.number_of_nodes())
            id2idx = {n:i for i,n in enumerate(G.nodes())}
            g.add_edges([(id2idx[u], id2idx[v]) for u,v in G.edges()])
            part = la.find_partition(g, la.RBConfigurationVertexPartition, weights="weight")
            comms = defaultdict(list)
            for node, cid in enumerate(part.membership):
                if len(part.parts[cid])>=min_size:
                    comms[cid].append(list(G.nodes())[node])
            return dict(comms)
        except Exception:
            # Fallback: connected components, then drop small ones
            comms = {}
            for c_id, comp in enumerate(nx.connected_components(G)):
                comp = list(comp)
                if len(comp) >= min_size:
                    comms[c_id] = comp
            return comms

    # ---- descriptor mining ----
    def mine_descriptors(self, cluster_ids: List[str], vocab_top: int=400, top_k_per_cluster: int=12) -> List[str]:
        # Build corpus-wide counts + cluster counts from text & captions (if you have image/audio tags, add them here).
        all_txt = []
        for i in cluster_ids:
            t = (self.items[i].text or "").strip()
            if t: all_txt.extend(simple_noun_phrases(t))
        # global stats
        all_corpus = []
        for it in self.items.values():
            if it.text:
                all_corpus.extend(simple_noun_phrases(it.text))
        C_global = Counter(all_corpus)
        C_cluster = Counter(all_txt)
        # vocab pre-filter: frequent but not stop-ish
        cand = [w for w,_ in C_global.most_common(vocab_top) if len(w)>2]
        # PMI + IDF
        N = max(1, sum(C_global.values()))
        Nc = max(1, sum(C_cluster.values()))
        desc_scores = []
        for w in cand:
            p_w  = C_global[w]/N
            p_wc = C_cluster[w]/Nc if C_cluster[w]>0 else 1e-8
            pmi  = np.log((p_wc+1e-9)/(p_w+1e-9))
            idf  = np.log(N/(1+C_global[w]))
            score = 1.0*pmi + 0.2*idf
            desc_scores.append((w, score))
        desc_scores.sort(key=lambda x: x[1], reverse=True)
        return [w for w,_ in desc_scores[:top_k_per_cluster]]

    # ---- end-to-end ----
    def run(self, k: int=12, min_cluster: int=8, top_desc: int=12) -> Dict[str, List[str]]:
        E = self._embed_all()
        E = self._align(E)
        ids, Z = self._fuse_early(E)
        G = self.build_graph(ids, Z, k=k)
        comms = self.cluster(G, min_size=min_cluster)
        symbols = {}
        for cid, members in comms.items():
            # name proposal: top two descriptors concatenated (edit later in UI)
            descs = self.mine_descriptors(members, top_k_per_cluster=top_desc)
            name = " / ".join(descs[:2]).upper() if descs else f"CLUSTER_{cid}"
            symbols[name] = descs
        return symbols
