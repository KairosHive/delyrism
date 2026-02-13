"""
td_realtime.py — Real-time Delta Graph Export for TouchDesigner
================================================================

This module provides real-time computation and export of delta subgraphs
for use in TouchDesigner performance applications.

COMMUNICATION AVENUES:
----------------------
1. OSC (Open Sound Control) — UDP-based, low latency, TouchDesigner native
2. WebSocket — Bidirectional, web-compatible, real-time parameter control
3. JSON File — Simple polling, file-watch compatible
4. Shared Memory — Ultra-low latency for same-machine (numpy memmap)
5. ZeroMQ — High-performance pub/sub pattern

USAGE:
------
    from delyrism.td_realtime import DeltaGraphServer
    from delyrism.delyrism import SymbolSpace, TextEmbedder
    
    # Create space
    embedder = TextEmbedder(backend="cloudflare", model="@cf/baai/bge-base-en-v1.5")
    space = SymbolSpace(symbols_to_descriptors={...}, embedder=embedder)
    
    # Launch server (pick your protocol)
    server = DeltaGraphServer(space, protocol="osc", port=7000)
    server.start()
    
    # Update context in real-time
    server.update_context("fire burning in the darkness")
"""

from __future__ import annotations
import json
import time
import threading
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Callable
from pathlib import Path
import struct

# Core delyrism imports (delayed to avoid circular)
def _import_delyrism():
    from . import delyrism as dly
    return dly


# ============================================================================
#  DATA STRUCTURES FOR EXPORT
# ============================================================================

@dataclass
class DeltaGraphData:
    """Container for all graph data needed by TouchDesigner."""
    
    # Timing
    timestamp: float = 0.0
    frame_id: int = 0
    
    # Context
    context_sentence: str = ""
    
    # Nodes: list of dicts
    nodes: List[Dict[str, Any]] = field(default_factory=list)
    # Each node: {id, label, symbol, x, y, size, r, g, b, a, degree, delta_sum}
    
    # Edges: list of dicts
    edges: List[Dict[str, Any]] = field(default_factory=list)
    # Each edge: {source, target, delta, abs_delta, sign, width, r, g, b, a}
    
    # Symbol summary
    symbols: List[Dict[str, Any]] = field(default_factory=list)
    # Each symbol: {name, node_count, total_delta, centroid_x, centroid_y, r, g, b}
    
    # Metrics
    metrics: Dict[str, float] = field(default_factory=dict)
    # graph_density, avg_delta, max_delta, total_edges, total_nodes
    
    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp,
            "frame_id": self.frame_id,
            "context": self.context_sentence,
            "nodes": self.nodes,
            "edges": self.edges,
            "symbols": self.symbols,
            "metrics": self.metrics,
        }
    
    def to_json(self, indent=None) -> str:
        return json.dumps(self.to_dict(), indent=indent)
    
    def to_binary(self) -> bytes:
        """Compact binary format for ultra-low-latency streaming."""
        # Header: frame_id (4), timestamp (8), n_nodes (4), n_edges (4)
        header = struct.pack('<IdII', self.frame_id, self.timestamp, 
                             len(self.nodes), len(self.edges))
        
        # Nodes: per-node -> x(f), y(f), size(f), r(f), g(f), b(f), delta_sum(f) = 28 bytes
        node_data = b''.join(
            struct.pack('<7f', n.get('x', 0), n.get('y', 0), n.get('size', 1),
                        n.get('r', 1), n.get('g', 1), n.get('b', 1), n.get('delta_sum', 0))
            for n in self.nodes
        )
        
        # Edges: per-edge -> src_idx(H), tgt_idx(H), delta(f), width(f) = 12 bytes
        node_id_map = {n['id']: i for i, n in enumerate(self.nodes)}
        edge_data = b''.join(
            struct.pack('<HHff', 
                        node_id_map.get(e['source'], 0),
                        node_id_map.get(e['target'], 0),
                        e.get('delta', 0), e.get('width', 1))
            for e in self.edges
        )
        
        return header + node_data + edge_data


# ============================================================================
#  PARAMETER CONTROLLER
# ============================================================================

@dataclass
class DeltaParams:
    """All parameters that influence delta graph computation."""
    
    # Context
    sentence: str = ""
    weights: Optional[Dict[str, float]] = None
    
    # Strategy
    strategy: str = "gate"           # gate | hybrid | reembed | pooling
    beta: float = 1.2                # context blend strength (shift strength β)
    gate: str = "relu"               # relu | cos | softmax | uniform
    tau: float = 0.2                 # softmax temperature
    
    # Advanced
    within_symbol_softmax: bool = True   # softmax within symbol if gate=softmax
    gamma: float = 0.5               # hybrid blend ratio
    prompt_template: str = "In this context: {sent}. Descriptor: {desc}"
    pool_type: str = "avg"
    pool_w: float = 0.7
    membership_alpha: float = 0.0
    
    # Graph filtering
    top_abs_edges: int = 10          # top |Δ| edges
    min_abs_delta: float = 0.005      # min |Δ| threshold
    within_symbol: bool = False      # within-symbol pairs only
    sym_filter: Optional[List[str]] = None
    only_symbol: Optional[str] = None
    connected_only: bool = True      # connected nodes only
    
    # Layout
    layout_seed: int = 42
    layout_k: float = 0.6            # spring layout spacing
    
    # Visual
    palette: str = "Nord"
    node_size_base: float = 220.0
    node_size_scale: float = 1200.0
    edge_width_min: float = 0.6
    edge_width_max: float = 6.0
    
    def to_dict(self) -> dict:
        return {
            "sentence": self.sentence,
            "weights": self.weights,
            "strategy": self.strategy,
            "beta": self.beta,
            "gate": self.gate,
            "tau": self.tau,
            "within_symbol_softmax": self.within_symbol_softmax,
            "gamma": self.gamma,
            "prompt_template": self.prompt_template,
            "pool_type": self.pool_type,
            "pool_w": self.pool_w,
            "membership_alpha": self.membership_alpha,
            "top_abs_edges": self.top_abs_edges,
            "min_abs_delta": self.min_abs_delta,
            "within_symbol": self.within_symbol,
            "sym_filter": self.sym_filter,
            "only_symbol": self.only_symbol,
            "connected_only": self.connected_only,
            "layout_seed": self.layout_seed,
            "layout_k": self.layout_k,
            "palette": self.palette,
            "node_size_base": self.node_size_base,
            "node_size_scale": self.node_size_scale,
            "edge_width_min": self.edge_width_min,
            "edge_width_max": self.edge_width_max,
        }
    
    @classmethod
    def from_dict(cls, d: dict) -> 'DeltaParams':
        return cls(**{k: v for k, v in d.items() if hasattr(cls, k) or k in cls.__dataclass_fields__})


# ============================================================================
#  CORE COMPUTATION ENGINE
# ============================================================================

class DeltaGraphEngine:
    """
    Core engine for computing delta graphs with caching and incremental updates.
    """
    
    def __init__(self, space, cache_embeddings: bool = True, deterministic: bool = True,
                 layout_cache_path: Optional[str] = None):
        self.space = space
        self.cache_embeddings = cache_embeddings
        self.deterministic = deterministic
        self.layout_cache_path = Path(layout_cache_path) if layout_cache_path else None
        self._layout_cache: Dict[str, np.ndarray] = {}
        self._sentence_cache: Dict[str, np.ndarray] = {}  # Cache for sentence embeddings
        self._last_params_hash: Optional[int] = None
        self._last_graph = None
        self._frame_id = 0
        
        if deterministic:
            self._set_deterministic()
        
        # Load persistent layout cache if exists
        if self.layout_cache_path and self.layout_cache_path.exists():
            try:
                import pickle
                with open(self.layout_cache_path, 'rb') as f:
                    self._layout_cache = pickle.load(f)
                print(f"[Engine] Loaded {len(self._layout_cache)} cached layouts")
            except Exception as e:
                print(f"[Engine] Could not load layout cache: {e}")
    
    def save_layout_cache(self):
        """Save layout cache to disk for persistence across restarts."""
        if self.layout_cache_path:
            import pickle
            with open(self.layout_cache_path, 'wb') as f:
                pickle.dump(self._layout_cache, f)
            print(f"[Engine] Saved {len(self._layout_cache)} layouts to {self.layout_cache_path}")
    
    def _set_deterministic(self):
        """Set random seeds for reproducibility."""
        import random
        random.seed(42)
        np.random.seed(42)
        try:
            import torch
            torch.manual_seed(42)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(42)
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False
        except ImportError:
            pass
        
    def compute(self, params: DeltaParams) -> DeltaGraphData:
        """Compute delta graph for given parameters."""
        import networkx as nx
        dly = _import_delyrism()
        
        self._frame_id += 1
        t0 = time.time()
        
        # Build the graph using existing context_delta_graph
        G = dly.context_delta_graph(
            self.space,
            sentence=params.sentence,
            weights=params.weights,
            strategy=params.strategy,
            beta=params.beta,
            gate=params.gate,
            tau=params.tau,
            within_symbol_softmax=params.within_symbol_softmax,
            gamma=params.gamma,
            prompt_template=params.prompt_template,
            top_abs_edges=params.top_abs_edges,
            sym_filter=params.sym_filter,
            min_abs_delta=params.min_abs_delta,
            within_symbol=params.within_symbol,
            only_symbol=params.only_symbol,
            connected_only=params.connected_only,
            pool_type=params.pool_type,
            pool_w=params.pool_w,
            membership_alpha=params.membership_alpha,
        )
        
        # Layout: use stable node-based hash (not edge-based) for better caching
        # Sort nodes for deterministic ordering
        sorted_nodes = tuple(sorted(G.nodes()))
        graph_key = (sorted_nodes, params.layout_seed)
        
        if graph_key in self._layout_cache:
            pos = self._layout_cache[graph_key]
        else:
            # Use fixed seed and iteration count for determinism
            pos = nx.spring_layout(G, seed=params.layout_seed, k=params.layout_k, iterations=50)
            self._layout_cache[graph_key] = pos
            # Limit cache size
            if len(self._layout_cache) > 100:
                self._layout_cache.pop(next(iter(self._layout_cache)))
        
        # Colors
        color_dict = self.space.get_symbol_color_dict(palette=params.palette)
        
        # Convert colors to RGBA tuples
        import matplotlib.colors as mcolors
        def to_rgba(c):
            return mcolors.to_rgba(c)
        
        color_rgba = {s: to_rgba(c) for s, c in color_dict.items()}
        
        # Node sizing based on incident edge delta
        abs_sum = {n: 0.0 for n in G.nodes()}
        for u, v, d in G.edges(data=True):
            abs_sum[u] += d.get('abs_delta', 0)
            abs_sum[v] += d.get('abs_delta', 0)
        
        max_sum = max(abs_sum.values()) if abs_sum else 1.0
        max_sum = max(max_sum, 1e-9)
        
        # Build node list
        nodes = []
        for n in G.nodes():
            sym = G.nodes[n].get('symbol', 'unknown')
            x, y = pos.get(n, (0, 0))
            r, g, b, a = color_rgba.get(sym, (0.5, 0.5, 0.5, 1.0))
            size = params.node_size_base + params.node_size_scale * (abs_sum[n] / max_sum)
            nodes.append({
                'id': n,
                'label': n,
                'symbol': sym,
                'x': float(x),
                'y': float(y),
                'size': float(size),
                'r': float(r),
                'g': float(g),
                'b': float(b),
                'a': float(a),
                'degree': G.degree(n),
                'delta_sum': float(abs_sum[n]),
            })
        
        # Edge widths normalized
        edges_data = list(G.edges(data=True))
        abs_deltas = np.array([d.get('abs_delta', 0) for _, _, d in edges_data])
        if len(abs_deltas) > 0 and abs_deltas.max() > abs_deltas.min():
            widths = params.edge_width_min + (params.edge_width_max - params.edge_width_min) * \
                     (abs_deltas - abs_deltas.min()) / (abs_deltas.max() - abs_deltas.min())
        else:
            widths = np.full(len(abs_deltas), params.edge_width_min)
        
        # Build edge list
        edges = []
        for i, (u, v, d) in enumerate(edges_data):
            delta = d.get('delta', 0)
            sign = d.get('sign', 'up')
            # Color: red for positive delta, blue for negative
            if delta > 0:
                r, g, b, a = 0.55, 0.15, 0.15, 0.75
            else:
                r, g, b, a = 0.15, 0.35, 0.65, 0.55
            edges.append({
                'source': u,
                'target': v,
                'delta': float(delta),
                'abs_delta': float(abs(delta)),
                'sign': sign,
                'width': float(widths[i]),
                'r': r, 'g': g, 'b': b, 'a': a,
            })
        
        # Symbol summary
        symbols_present = sorted({G.nodes[n]['symbol'] for n in G.nodes()})
        symbol_summary = []
        for sym in symbols_present:
            sym_nodes = [n for n in G.nodes() if G.nodes[n]['symbol'] == sym]
            total_delta = sum(abs_sum[n] for n in sym_nodes)
            centroid_x = np.mean([pos[n][0] for n in sym_nodes]) if sym_nodes else 0
            centroid_y = np.mean([pos[n][1] for n in sym_nodes]) if sym_nodes else 0
            r, g, b, a = color_rgba.get(sym, (0.5, 0.5, 0.5, 1.0))
            symbol_summary.append({
                'name': sym,
                'node_count': len(sym_nodes),
                'total_delta': float(total_delta),
                'centroid_x': float(centroid_x),
                'centroid_y': float(centroid_y),
                'r': float(r), 'g': float(g), 'b': float(b),
            })
        
        # Metrics
        metrics = {
            'total_nodes': len(nodes),
            'total_edges': len(edges),
            'graph_density': nx.density(G) if len(G) > 1 else 0,
            'avg_delta': float(np.mean(abs_deltas)) if len(abs_deltas) > 0 else 0,
            'max_delta': float(np.max(abs_deltas)) if len(abs_deltas) > 0 else 0,
            'compute_time_ms': (time.time() - t0) * 1000,
        }
        
        return DeltaGraphData(
            timestamp=time.time(),
            frame_id=self._frame_id,
            context_sentence=params.sentence,
            nodes=nodes,
            edges=edges,
            symbols=symbol_summary,
            metrics=metrics,
        )


# ============================================================================
#  PROTOCOL HANDLERS
# ============================================================================

class OSCTransport:
    """OSC (Open Sound Control) transport for TouchDesigner."""
    
    def __init__(self, host: str = "127.0.0.1", port: int = 7000):
        self.host = host
        self.port = port
        self._client = None
        self._has_osc = False
        
    def start(self):
        try:
            from pythonosc import udp_client
            self._client = udp_client.SimpleUDPClient(self.host, self.port)
            self._has_osc = True
            print(f"[OSC] Sending to {self.host}:{self.port} (python-osc)")
        except ImportError:
            import socket
            self._socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self._has_osc = False
            print(f"[OSC] Sending to {self.host}:{self.port} (raw UDP - install python-osc for proper OSC)")
    
    def stop(self):
        if hasattr(self, '_socket') and self._socket:
            self._socket.close()
        self._client = None
    
    def send(self, data: DeltaGraphData):
        if self._has_osc and self._client:
            self._send_osc(data)
        else:
            self._send_json(data)
    
    def _send_json(self, data: DeltaGraphData):
        """Fallback: send JSON over UDP."""
        try:
            msg = data.to_json().encode('utf-8')
            self._socket.sendto(msg, (self.host, self.port))
        except Exception as e:
            print(f"[OSC] Send error: {e}")
    
    def _send_osc(self, data: DeltaGraphData):
        """Send proper OSC messages."""
        c = self._client
        
        # Frame info
        c.send_message("/delyrism/frame", [data.frame_id, float(data.timestamp)])
        c.send_message("/delyrism/context", data.context_sentence)
        
        # Counts (so TD knows how many to expect)
        c.send_message("/delyrism/count/nodes", len(data.nodes))
        c.send_message("/delyrism/count/edges", len(data.edges))
        
        # Metrics
        c.send_message("/delyrism/metrics", [
            float(data.metrics.get('total_nodes', 0)),
            float(data.metrics.get('total_edges', 0)),
            float(data.metrics.get('avg_delta', 0)),
            float(data.metrics.get('max_delta', 0)),
            float(data.metrics.get('compute_time_ms', 0)),
        ])
        
        # Nodes: /delyrism/node/<index> [name, symbol, x, y, size, r, g, b]
        for i, n in enumerate(data.nodes):
            c.send_message(f"/delyrism/node/{i}", [
                n['label'],           # node name (descriptor)
                n['symbol'],          # symbol family
                float(n['x']),        # position x
                float(n['y']),        # position y
                float(n['size']),     # node size
                float(n['r']),        # color r
                float(n['g']),        # color g
                float(n['b']),        # color b
            ])
        
        # Edges: /delyrism/edge/<index> [source_name, target_name, delta, width]
        for i, e in enumerate(data.edges):
            c.send_message(f"/delyrism/edge/{i}", [
                e['source'],          # source node name
                e['target'],          # target node name
                float(e['delta']),    # delta value (signed)
                float(e['width']),    # edge width
            ])
        
        # Symbol summaries: /delyrism/symbol/<index> [name, node_count, centroid_x, centroid_y, r, g, b]
        for i, s in enumerate(data.symbols):
            c.send_message(f"/delyrism/symbol/{i}", [
                s['name'],
                int(s['node_count']),
                float(s['centroid_x']),
                float(s['centroid_y']),
                float(s['r']),
                float(s['g']),
                float(s['b']),
            ])


class WebSocketTransport:
    """
    WebSocket server for bidirectional real-time communication.
    TouchDesigner can connect via WebSocket DAT.
    """
    
    def __init__(self, host: str = "0.0.0.0", port: int = 8765):
        self.host = host
        self.port = port
        self._server = None
        self._clients = set()
        self._thread = None
        self._running = False
        self._param_callback: Optional[Callable[[dict], None]] = None
        
    def set_param_callback(self, callback: Callable[[dict], None]):
        """Register callback for when parameters are received from TD."""
        self._param_callback = callback
    
    def start(self):
        try:
            import asyncio
            import websockets
        except ImportError:
            print("[WebSocket] ERROR: pip install websockets")
            return
        
        self._running = True
        self._thread = threading.Thread(target=self._run_server, daemon=True)
        self._thread.start()
        print(f"[WebSocket] Server started on ws://{self.host}:{self.port}")
    
    def _run_server(self):
        import asyncio
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        async def handler(websocket, path):
            self._clients.add(websocket)
            try:
                async for message in websocket:
                    # Handle incoming parameter updates from TD
                    try:
                        params = json.loads(message)
                        if self._param_callback:
                            self._param_callback(params)
                    except json.JSONDecodeError:
                        pass
            finally:
                self._clients.discard(websocket)
        
        async def main():
            import websockets
            async with websockets.serve(handler, self.host, self.port):
                while self._running:
                    await asyncio.sleep(0.1)
        
        loop.run_until_complete(main())
    
    def stop(self):
        self._running = False
    
    def send(self, data: DeltaGraphData):
        if not self._clients:
            return
        
        import asyncio
        
        msg = data.to_json()
        
        async def broadcast():
            dead = set()
            for client in self._clients:
                try:
                    await client.send(msg)
                except:
                    dead.add(client)
            self._clients -= dead
        
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                asyncio.ensure_future(broadcast())
            else:
                loop.run_until_complete(broadcast())
        except:
            pass


class JSONFileTransport:
    """
    Simple file-based export. TouchDesigner can watch the file.
    Good for debugging and low-frequency updates.
    """
    
    def __init__(self, filepath: str = "delta_graph.json", 
                 binary_path: Optional[str] = None):
        self.filepath = Path(filepath)
        self.binary_path = Path(binary_path) if binary_path else None
        
    def start(self):
        print(f"[JSONFile] Writing to {self.filepath}")
    
    def stop(self):
        pass
    
    def send(self, data: DeltaGraphData):
        # Write JSON
        self.filepath.write_text(data.to_json(indent=2))
        
        # Optionally write binary for faster parsing
        if self.binary_path:
            self.binary_path.write_bytes(data.to_binary())


class SharedMemoryTransport:
    """
    Shared memory for ultra-low-latency same-machine communication.
    Uses numpy memmap for direct memory access from TouchDesigner Python.
    """
    
    def __init__(self, name: str = "delyrism_delta", 
                 max_nodes: int = 500, max_edges: int = 500):
        self.name = name
        self.max_nodes = max_nodes
        self.max_edges = max_edges
        self._mmap_path = Path(f"./{name}.mmap")
        self._meta_path = Path(f"./{name}_meta.json")
        self._mmap = None
        
        # Memory layout:
        # Header: [frame_id, timestamp, n_nodes, n_edges] = 4 floats
        # Nodes: [x, y, size, r, g, b, delta_sum] × max_nodes = 7 × max_nodes floats
        # Edges: [src, tgt, delta, width] × max_edges = 4 × max_edges floats
        self._header_size = 4
        self._node_stride = 7
        self._edge_stride = 4
        self._total_floats = (self._header_size + 
                              self._node_stride * max_nodes + 
                              self._edge_stride * max_edges)
        
    def start(self):
        self._mmap = np.memmap(str(self._mmap_path), dtype=np.float32, 
                               mode='w+', shape=(self._total_floats,))
        self._mmap[:] = 0
        
        # Write metadata for TD to read
        meta = {
            "path": str(self._mmap_path.absolute()),
            "dtype": "float32",
            "header_size": self._header_size,
            "node_stride": self._node_stride,
            "edge_stride": self._edge_stride,
            "max_nodes": self.max_nodes,
            "max_edges": self.max_edges,
            "total_floats": self._total_floats,
        }
        self._meta_path.write_text(json.dumps(meta, indent=2))
        print(f"[SharedMem] Memory-mapped file: {self._mmap_path}")
        
    def stop(self):
        if self._mmap is not None:
            del self._mmap
            self._mmap = None
    
    def send(self, data: DeltaGraphData):
        if self._mmap is None:
            return
        
        # Header
        self._mmap[0] = data.frame_id
        self._mmap[1] = data.timestamp
        self._mmap[2] = len(data.nodes)
        self._mmap[3] = len(data.edges)
        
        # Nodes
        offset = self._header_size
        for i, n in enumerate(data.nodes[:self.max_nodes]):
            base = offset + i * self._node_stride
            self._mmap[base:base+7] = [
                n['x'], n['y'], n['size'],
                n['r'], n['g'], n['b'], n['delta_sum']
            ]
        
        # Edges
        offset = self._header_size + self._node_stride * self.max_nodes
        node_id_map = {n['id']: i for i, n in enumerate(data.nodes)}
        for i, e in enumerate(data.edges[:self.max_edges]):
            base = offset + i * self._edge_stride
            self._mmap[base:base+4] = [
                node_id_map.get(e['source'], 0),
                node_id_map.get(e['target'], 0),
                e['delta'], e['width']
            ]
        
        # Flush
        self._mmap.flush()


class ZeroMQTransport:
    """
    ZeroMQ pub/sub for high-performance distributed streaming.
    Supports multiple subscribers and topics.
    """
    
    def __init__(self, endpoint: str = "tcp://*:5555"):
        self.endpoint = endpoint
        self._socket = None
        self._context = None
        
    def start(self):
        try:
            import zmq
        except ImportError:
            print("[ZeroMQ] ERROR: pip install pyzmq")
            return
        
        self._context = zmq.Context()
        self._socket = self._context.socket(zmq.PUB)
        self._socket.bind(self.endpoint)
        print(f"[ZeroMQ] Publishing on {self.endpoint}")
    
    def stop(self):
        if self._socket:
            self._socket.close()
        if self._context:
            self._context.term()
    
    def send(self, data: DeltaGraphData):
        if not self._socket:
            return
        
        # Send as multipart: topic + JSON
        self._socket.send_multipart([
            b"delyrism.delta",
            data.to_json().encode('utf-8')
        ])
        
        # Also send binary for performance-critical applications
        self._socket.send_multipart([
            b"delyrism.delta.bin",
            data.to_binary()
        ])


# ============================================================================
#  MAIN SERVER CLASS
# ============================================================================

class DeltaGraphServer:
    """
    Main server for real-time delta graph streaming to TouchDesigner.
    
    Example:
        server = DeltaGraphServer(space, protocol="osc", port=7000)
        server.start()
        
        # In your main loop or callback:
        server.update_context("fire burning bright")
        server.set_param("beta", 0.8)
        
        # Or run with auto-refresh:
        server.run_loop(fps=30)
    """
    
    PROTOCOLS = {
        "osc": OSCTransport,
        "websocket": WebSocketTransport,
        "json": JSONFileTransport,
        "sharedmem": SharedMemoryTransport,
        "zmq": ZeroMQTransport,
    }
    
    def __init__(
        self,
        space,
        protocol: str = "osc",
        host: str = "127.0.0.1",
        port: int = 7000,
        layout_cache_path: str = ".delyrism_layout_cache.pkl",
        **transport_kwargs
    ):
        self.space = space
        self.engine = DeltaGraphEngine(space, layout_cache_path=layout_cache_path)
        self.params = DeltaParams()
        self._lock = threading.Lock()
        self._running = False
        self._callbacks: List[Callable[[DeltaGraphData], None]] = []
        
        # Create transport
        if protocol not in self.PROTOCOLS:
            raise ValueError(f"Unknown protocol '{protocol}'. Use: {list(self.PROTOCOLS.keys())}")
        
        TransportClass = self.PROTOCOLS[protocol]
        
        if protocol == "osc":
            self.transport = TransportClass(host=host, port=port)
        elif protocol == "websocket":
            self.transport = TransportClass(host=host, port=port)
            self.transport.set_param_callback(self._handle_incoming_params)
        elif protocol == "json":
            filepath = transport_kwargs.get("filepath", "delta_graph.json")
            self.transport = TransportClass(filepath=filepath)
        elif protocol == "sharedmem":
            name = transport_kwargs.get("name", "delyrism_delta")
            self.transport = TransportClass(name=name)
        elif protocol == "zmq":
            endpoint = transport_kwargs.get("endpoint", f"tcp://*:{port}")
            self.transport = TransportClass(endpoint=endpoint)
    
    def _handle_incoming_params(self, params_dict: dict):
        """Handle parameter updates from TouchDesigner (WebSocket)."""
        with self._lock:
            for key, value in params_dict.items():
                if hasattr(self.params, key):
                    setattr(self.params, key, value)
    
    def start(self):
        """Start the transport."""
        self.transport.start()
        self._running = True
    
    def stop(self):
        """Stop the server and save caches."""
        self._running = False
        self.transport.stop()
        self.engine.save_layout_cache()
    
    def add_callback(self, callback: Callable[[DeltaGraphData], None]):
        """Add a callback to receive computed data."""
        self._callbacks.append(callback)
    
    def update_context(self, sentence: str):
        """Update the context sentence and recompute."""
        with self._lock:
            self.params.sentence = sentence
        self._compute_and_send()
    
    def set_param(self, key: str, value):
        """Set a single parameter."""
        with self._lock:
            if hasattr(self.params, key):
                setattr(self.params, key, value)
    
    def set_params(self, **kwargs):
        """Set multiple parameters at once."""
        with self._lock:
            for key, value in kwargs.items():
                if hasattr(self.params, key):
                    setattr(self.params, key, value)
    
    def get_params(self) -> dict:
        """Get current parameters as dict."""
        return self.params.to_dict()
    
    def _compute_and_send(self):
        """Compute delta graph and send to transport."""
        with self._lock:
            params = DeltaParams(**self.params.to_dict())
        
        data = self.engine.compute(params)
        
        # Send to transport
        self.transport.send(data)
        
        # Call registered callbacks
        for cb in self._callbacks:
            try:
                cb(data)
            except Exception as e:
                print(f"[Callback Error] {e}")
        
        return data
    
    def compute_once(self) -> DeltaGraphData:
        """Compute and return data without sending."""
        with self._lock:
            params = DeltaParams(**self.params.to_dict())
        return self.engine.compute(params)
    
    def run_loop(self, fps: float = 30.0, duration: Optional[float] = None):
        """
        Run continuous update loop at specified FPS.
        Blocks until duration expires or stop() is called.
        """
        interval = 1.0 / fps
        start = time.time()
        
        while self._running:
            t0 = time.time()
            
            self._compute_and_send()
            
            # Check duration
            if duration and (time.time() - start) >= duration:
                break
            
            # Sleep to maintain FPS
            elapsed = time.time() - t0
            if elapsed < interval:
                time.sleep(interval - elapsed)
    
    def run_async(self, fps: float = 30.0):
        """Run update loop in background thread."""
        self._running = True
        thread = threading.Thread(target=self.run_loop, args=(fps,), daemon=True)
        thread.start()
        return thread


# ============================================================================
#  CONVENIENCE FUNCTIONS
# ============================================================================

def export_delta_graph_json(
    space,
    sentence: str,
    output_path: str = "delta_graph.json",
    **params
) -> DeltaGraphData:
    """
    One-shot export of delta graph to JSON file.
    
    Args:
        space: SymbolSpace instance
        sentence: Context sentence
        output_path: Output JSON file path
        **params: Any DeltaParams fields
    
    Returns:
        DeltaGraphData
    """
    engine = DeltaGraphEngine(space)
    p = DeltaParams(sentence=sentence, **params)
    data = engine.compute(p)
    Path(output_path).write_text(data.to_json(indent=2))
    return data


def create_osc_sender(space, host="127.0.0.1", port=7000) -> DeltaGraphServer:
    """Create a simple OSC sender for TouchDesigner."""
    server = DeltaGraphServer(space, protocol="osc", host=host, port=port)
    server.start()
    return server


def create_websocket_server(space, host="0.0.0.0", port=8765) -> DeltaGraphServer:
    """Create a WebSocket server with bidirectional control."""
    server = DeltaGraphServer(space, protocol="websocket", host=host, port=port)
    server.start()
    return server


# ============================================================================
#  TOUCHDESIGNER HELPER CODE (copy into TD Python DAT)
# ============================================================================

TD_RECEIVER_CODE = '''
# ============================================================================
# TOUCHDESIGNER RECEIVER CODE (paste into Execute DAT or Script CHOP)
# ============================================================================

# --- For WebSocket DAT ---
# Set this as the onReceiveText callback:
def onReceiveText(dat, rowIndex, message, bytes):
    import json
    data = json.loads(message)
    
    # Store in TABLE DAT
    nodes_table = op('nodes_table')
    edges_table = op('edges_table')
    
    # Clear and populate nodes
    nodes_table.clear()
    nodes_table.appendRow(['id', 'symbol', 'x', 'y', 'size', 'r', 'g', 'b', 'delta_sum'])
    for n in data['nodes']:
        nodes_table.appendRow([
            n['id'], n['symbol'], n['x'], n['y'], n['size'],
            n['r'], n['g'], n['b'], n['delta_sum']
        ])
    
    # Clear and populate edges
    edges_table.clear()
    edges_table.appendRow(['source', 'target', 'delta', 'width', 'r', 'g', 'b'])
    for e in data['edges']:
        edges_table.appendRow([
            e['source'], e['target'], e['delta'], e['width'],
            e['r'], e['g'], e['b']
        ])
    
    # Store metrics in CHOP
    metrics_chop = op('metrics_chop')
    for k, v in data['metrics'].items():
        # custom parameter or channel
        pass

# --- For OSC In CHOP ---
# Use callbacks to route /delyrism/* messages

# --- For Shared Memory ---
import numpy as np
import json

def read_delta_from_sharedmem():
    meta = json.load(open('delyrism_delta_meta.json'))
    data = np.memmap(meta['path'], dtype='float32', mode='r')
    
    frame_id = int(data[0])
    timestamp = data[1]
    n_nodes = int(data[2])
    n_edges = int(data[3])
    
    # Parse nodes
    offset = meta['header_size']
    nodes = []
    for i in range(n_nodes):
        base = offset + i * meta['node_stride']
        nodes.append({
            'x': data[base], 'y': data[base+1], 'size': data[base+2],
            'r': data[base+3], 'g': data[base+4], 'b': data[base+5],
            'delta_sum': data[base+6]
        })
    
    return {'frame_id': frame_id, 'nodes': nodes, 'n_nodes': n_nodes}
'''


# ============================================================================
#  CLI ENTRY POINT
# ============================================================================

def main():
    """Command-line interface for testing."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Delta Graph Server for TouchDesigner")
    parser.add_argument("--protocol", choices=["osc", "websocket", "json", "sharedmem", "zmq"],
                        default="osc", help="Communication protocol")
    parser.add_argument("--host", default="127.0.0.1", help="Host address")
    parser.add_argument("--port", type=int, default=7000, help="Port number")
    parser.add_argument("--fps", type=float, default=30.0, help="Update rate")
    parser.add_argument("--structure", default="elements", help="Symbol structure to load")
    parser.add_argument("--backend", default="sentence-transformer", 
                        choices=["sentence-transformer", "qwen2", "qwen3", "cloudflare"],
                        help="Embedding backend")
    parser.add_argument("--pooling", default="eos", choices=["eos", "mean", "cls"],
                        help="Pooling strategy for transformer models")
    args = parser.parse_args()
    
    # Load a default structure
    from pathlib import Path
    import json as _json
    
    structures_dir = Path(__file__).parent / "structures"
    structure_file = structures_dir / f"{args.structure}.json"
    
    if not structure_file.exists():
        print(f"Structure file not found: {structure_file}")
        return
    
    symbols_to_descriptors = _json.loads(structure_file.read_text())
    
    # Create embedder and space
    dly = _import_delyrism()
    embedder = dly.TextEmbedder(backend=args.backend, pooling=args.pooling)
    space = dly.SymbolSpace(symbols_to_descriptors=symbols_to_descriptors, embedder=embedder)
    
    # Create server
    server = DeltaGraphServer(space, protocol=args.protocol, host=args.host, port=args.port)
    
    print(f"\nDelta Graph Server")
    print(f"Protocol: {args.protocol}")
    print(f"Endpoint: {args.host}:{args.port}")
    print(f"FPS: {args.fps}")
    print(f"Structure: {args.structure}")
    print(f"Backend: {args.backend} (pooling: {args.pooling})")
    print(f"Params: beta={server.params.beta}, tau={server.params.tau}, top_edges={server.params.top_abs_edges}")
    print("\nPress Ctrl+C to stop\n")
    
    server.start()
    
    # Demo: cycle through some contexts
    contexts = [
        "fire burning in the night",
        "water flowing gently",
        "earth solid and stable",
        "air moving freely",
    ]
    
    try:
        i = 0
        while True:
            ctx = contexts[i % len(contexts)]
            print(f"\n{'='*60}")
            print(f"[Frame {i+1}] Context: \"{ctx}\"")
            print(f"{'='*60}")
            
            # DEBUG: Check sentence embedding consistency
            sent_emb = space.embedder.encode([ctx])[0]
            emb_hash = hash(sent_emb.tobytes()) % 100000
            print(f"  [DEBUG] Sentence embedding hash: {emb_hash}")
            print(f"  [DEBUG] space.D hash: {hash(space.D.tobytes()) % 100000}")
            
            server.update_context(ctx)
            data = server.compute_once()
            
            # Group nodes by symbol
            by_symbol = {}
            for n in data.nodes:
                sym = n['symbol']
                if sym not in by_symbol:
                    by_symbol[sym] = []
                by_symbol[sym].append(n['label'])
            
            print(f"\nNodes ({data.metrics['total_nodes']}):")
            for sym, nodes in sorted(by_symbol.items()):
                print(f"  [{sym}] {', '.join(nodes)}")
            
            print(f"\nEdges ({data.metrics['total_edges']}):")
            for e in data.edges[:10]:  # Show first 10 edges
                sign = "+" if e['delta'] > 0 else "-"
                print(f"  {e['source']} <--> {e['target']}  (Δ={sign}{abs(e['delta']):.3f})")
            if len(data.edges) > 10:
                print(f"  ... and {len(data.edges) - 10} more edges")
            
            print(f"\nCompute time: {data.metrics['compute_time_ms']:.1f}ms")
            time.sleep(2.0)
            i += 1
    except KeyboardInterrupt:
        print("\nStopping...")
        server.stop()


if __name__ == "__main__":
    main()
