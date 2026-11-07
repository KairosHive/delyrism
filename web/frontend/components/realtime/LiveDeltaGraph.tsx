"use client";
import * as React from "react";
import { useMeasure } from "@/lib/useMeasure";
import { useLive } from "@/lib/liveGraph";
import { SpringSim, type SimNode, type SimLink } from "@/lib/springSim";
import { SymbolLegend } from "@/components/explorer/SymbolLegend";

/**
 * The Explorer's Δ-graph, kept alive.  Same marks (node radius = Σ|Δ|,
 * edge width = |Δ|, teal strengthens / red weakens) drawn on a canvas, but
 * laid out by `SpringSim` — symmetric springs + repulsion, central
 * forces, and ONE viscous damping term on node velocity.  The query sets
 * the springs' rest lengths directly:
 *
 *   overlap(e) = sqrt(rel(u) · rel(v))          rel = sharpened query relevance
 *   rest(e)    = lerp(long, short, overlap)     short = 60 px, long grows with contraction
 *
 * and a radial spring pulls relevant nodes inward.  Node objects persist
 * across graph updates so context edits morph the layout instead of
 * re-seeding it; the simulation sleeps once it has settled.
 */

type VNode = SimNode & { label: string; color: string; symbol: string; weight: number; pull: number };
type VLink = SimLink & { value: number; sign: "up" | "down"; delta: number; overlap: number };

const BASE_DIST = 80;
const SHORT_DIST = 60;   // minimum spring length: related nodes stay close, never on top of each other

export function LiveDeltaGraph() {
  const [wrapRef, w] = useMeasure();
  const canvasRef = React.useRef<HTMLCanvasElement>(null);
  const graphSeq = useLive((s) => s.graphSeq);
  const graph = useLive((s) => s.graph);
  const queryActive = useLive((s) => s.queryActive);
  const hasContext = !!graph?.context?.trim();

  const simRef = React.useRef<SpringSim>(new SpringSim());
  const nodesRef = React.useRef<VNode[]>([]);
  const linksRef = React.useRef<VLink[]>([]);
  const edgeNorm = React.useRef({ lo: 0, range: 1 });
  const camRef = React.useRef({ x: 0, y: 0, k: 1 });   // auto-fit camera, eased
  const dragRef = React.useRef<{ node: VNode; dx: number; dy: number } | null>(null);
  const hoverRef = React.useRef<VNode | null>(null);

  const W = Math.max(320, w - 24);
  const H = Math.max(560, Math.min(860, Math.round((w || 1000) * 0.5)));

  // ---- relevance → rest lengths (instant; the physics does the rest) ---------
  const applyRest = React.useCallback(() => {
    const st = useLive.getState();
    const nodes = nodesRef.current, links = linksRef.current;
    const g = st.focus, active = st.queryActive;
    for (const n of nodes) {
      const r = st.relevance[n.id];
      n.rel = r === undefined ? 0 : Math.pow(Math.max(0, Math.min(1, r)), g);
    }
    const long = BASE_DIST + 140 * st.contraction;
    for (const l of links) {
      const a = nodes[l.a].rel, b = nodes[l.b].rel;
      l.overlap = Math.sqrt(Math.max(0, a * b));
      l.rest = active ? long + (SHORT_DIST - long) * l.overlap : BASE_DIST;
      l.k = active ? 0.6 + 0.8 * l.overlap : 1;
    }
    const sim = simRef.current;
    sim.params.radialOn = active;
    sim.wake();
  }, []);

  // ---- ingest graph updates: reuse node objects by id -----------------------
  React.useEffect(() => {
    if (!graph) return;
    const prev = new Map(nodesRef.current.map((n) => [n.id, n]));
    const weight = new Map<string, number>();
    for (const e of graph.edges) {
      weight.set(e.source, (weight.get(e.source) ?? 0) + e.abs_delta);
      weight.set(e.target, (weight.get(e.target) ?? 0) + e.abs_delta);
    }
    const wv = Array.from(weight.values());
    const wLo = wv.length ? Math.min(...wv) : 0, wHi = wv.length ? Math.max(...wv) : 1;
    const wR = wHi - wLo > 1e-12 ? wHi - wLo : 1;
    const ev = graph.edges.map((e) => e.abs_delta);
    const eLo = ev.length ? Math.min(...ev) : 0, eHi = ev.length ? Math.max(...ev) : 1;
    edgeNorm.current = { lo: eLo, range: eHi - eLo > 1e-12 ? eHi - eLo : 1 };

    const bySym = new Map<string, { x: number; y: number; n: number }>();
    for (const n of nodesRef.current) {
      const a = bySym.get(n.symbol) ?? { x: 0, y: 0, n: 0 };
      a.x += n.x; a.y += n.y; a.n += 1; bySym.set(n.symbol, a);
    }
    const golden = Math.PI * (3 - Math.sqrt(5));
    const nodes: VNode[] = graph.nodes.map((n, i) => {
      const old = prev.get(n.id);
      const wgt = ((weight.get(n.id) ?? 0) - wLo) / wR;
      if (old) { old.weight = wgt; old.pull = n.pull; old.color = n.color; return old; }
      const a = bySym.get(n.symbol);
      const jitter = () => (Math.random() - 0.5) * 30;
      // new nodes: beside their archetype siblings, else a phyllotaxis seed
      const r = 12 * Math.sqrt(i + 1), th = i * golden;
      return {
        id: n.id, label: n.id, color: n.color, symbol: n.symbol, weight: wgt, pull: n.pull,
        rel: 0, fixed: false, vx: 0, vy: 0,
        x: a && a.n ? a.x / a.n + jitter() : r * Math.cos(th),
        y: a && a.n ? a.y / a.n + jitter() : r * Math.sin(th),
      };
    });
    const index = new Map(nodes.map((n, i) => [n.id, i]));
    const links: VLink[] = [];
    for (const e of graph.edges) {
      const a = index.get(e.source), b = index.get(e.target);
      if (a === undefined || b === undefined) continue;
      links.push({ a, b, rest: BASE_DIST, k: 1, value: e.abs_delta, sign: e.sign, delta: e.delta, overlap: 1 });
    }
    nodesRef.current = nodes;
    linksRef.current = links;
    const sim = simRef.current;
    sim.nodes = nodes; sim.links = links;
    applyRest();
  }, [graphSeq, graph, applyRest]);

  const relevanceSeq = useLive((s) => s.relevanceSeq);
  const focus = useLive((s) => s.focus);
  const contraction = useLive((s) => s.contraction);
  React.useEffect(() => { applyRest(); }, [relevanceSeq, focus, contraction, queryActive, applyRest]);

  // physics params from the knobs
  const damping = useLive((s) => s.damping);
  const stiffness = useLive((s) => s.stiffness);
  const unlinked = useLive((s) => s.unlinked);
  const repulsion = useLive((s) => s.repulsion);
  React.useEffect(() => {
    const p = simRef.current.params;
    p.damping = damping; p.stiffness = stiffness; p.unlinked = unlinked; p.repulsion = repulsion;
    simRef.current.wake();
  }, [damping, stiffness, unlinked, repulsion]);

  // ---- step + draw loop ------------------------------------------------------
  React.useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const dpr = Math.min(2, window.devicePixelRatio || 1);
    canvas.width = Math.round(W * dpr); canvas.height = Math.round(H * dpr);
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    let raf = 0;
    let last = performance.now();
    let lastSync = 0, fps = 60;
    const draw = () => {
      raf = requestAnimationFrame(draw);
      const now = performance.now();
      const frameMs = now - last;
      const dt = Math.min(0.05, frameMs / 1000);
      last = now;
      if (frameMs > 0) fps += (1000 / frameMs - fps) * 0.05;
      const sim = simRef.current;
      sim.step(dt);
      const nodes = nodesRef.current, links = linksRef.current;

      // Stream what this canvas shows to the server, which forwards it over OSC
      // (TouchDesigner) — so TD gets exactly this motion, slow settling and drags
      // included.  ≤60 Hz while moving, a 0.4 s heartbeat while asleep.
      const send = useLive.getState()._send;
      if (send && nodes.length && now - lastSync >= (sim.awake ? 16 : 400)) {
        const ids = new Array<string>(nodes.length);
        const xy = new Array<number>(nodes.length * 2), v = new Array<number>(nodes.length * 2);
        for (let i = 0; i < nodes.length; i++) {
          const n = nodes[i];
          ids[i] = n.id;
          xy[2 * i] = Math.round(n.x * 10) / 10; xy[2 * i + 1] = Math.round(n.y * 10) / 10;
          v[2 * i] = Math.round(n.vx * 10) / 10; v[2 * i + 1] = Math.round(n.vy * 10) / 10;
        }
        send({ type: "layout", ids, xy, v, awake: sim.awake, fps: Math.round(fps) });
        lastSync = now;
      }
      const active = useLive.getState().queryActive;

      // camera: fit bounds with margin, eased
      let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
      for (const n of nodes) { minX = Math.min(minX, n.x); maxX = Math.max(maxX, n.x); minY = Math.min(minY, n.y); maxY = Math.max(maxY, n.y); }
      const cam = camRef.current;
      if (nodes.length) {
        const bw = Math.max(80, maxX - minX + 120), bh = Math.max(80, maxY - minY + 120);
        const kT = Math.min(2.2, Math.min(W / bw, H / bh));
        const cxT = (minX + maxX) / 2, cyT = (minY + maxY) / 2;
        cam.k += (kT - cam.k) * 0.06; cam.x += (cxT - cam.x) * 0.06; cam.y += (cyT - cam.y) * 0.06;
      }
      const sx = (x: number) => W / 2 + (x - cam.x) * cam.k;
      const sy = (y: number) => H / 2 + (y - cam.y) * cam.k;
      const k = cam.k;

      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      ctx.clearRect(0, 0, W, H);
      const { lo, range } = edgeNorm.current;

      // edges
      ctx.lineCap = "round";
      for (const l of links) {
        const A = nodes[l.a], B = nodes[l.b];
        const a = active ? 0.12 + 0.85 * l.overlap : 1;
        ctx.strokeStyle = l.sign === "up" ? `rgba(38,161,149,${(0.78 * a).toFixed(3)})` : `rgba(191,97,106,${(0.72 * a).toFixed(3)})`;
        ctx.lineWidth = (0.6 + ((l.value - lo) / range) * 5.4) * (active ? 0.5 + 0.7 * l.overlap : 1) * k;
        ctx.beginPath(); ctx.moveTo(sx(A.x), sy(A.y)); ctx.lineTo(sx(B.x), sy(B.y)); ctx.stroke();
      }
      // nodes
      for (const n of nodes) {
        const x = sx(n.x), y = sy(n.y);
        const rel = active ? n.rel : 1;
        const r = ((3 + n.weight * 10) + (active ? 4 * n.rel : 0)) * k + 1.5;
        ctx.globalAlpha = active ? 0.22 + 0.78 * rel : 1;
        if (active && n.rel > 0.45) { ctx.shadowColor = n.color; ctx.shadowBlur = 22 * (n.rel - 0.45) / 0.55; }
        ctx.beginPath(); ctx.arc(x, y, r, 0, Math.PI * 2);
        ctx.fillStyle = n.color; ctx.fill();
        ctx.shadowBlur = 0;
        ctx.lineWidth = hoverRef.current === n ? 1.6 : 0.6;
        ctx.strokeStyle = hoverRef.current === n ? "rgba(255,255,255,0.9)" : "rgba(0,0,0,0.5)"; ctx.stroke();
        const showLabel = k > 0.55 || (active && n.rel > 0.6);
        if (showLabel) {
          const fs = (12 + n.weight * 4 + (active ? 3 * n.rel : 0)) * Math.min(1.2, Math.max(0.7, k));
          ctx.font = `${active && n.rel > 0.6 ? 700 : 600} ${fs}px Inter, system-ui`;
          ctx.textAlign = "center"; ctx.textBaseline = "top";
          ctx.fillStyle = "rgba(0,0,0,0.7)"; ctx.fillText(n.label, x + 0.6, y + r + 2.6);
          ctx.fillStyle = "#f2f6fb"; ctx.fillText(n.label, x, y + r + 2);
        }
        ctx.globalAlpha = 1;
      }
    };
    raf = requestAnimationFrame(draw);
    return () => cancelAnimationFrame(raf);
  }, [W, H]);

  // ---- pointer: hover + drag -------------------------------------------------
  const pick = (px: number, py: number): VNode | null => {
    const cam = camRef.current;
    let best: VNode | null = null, bd = Infinity;
    for (const n of nodesRef.current) {
      const x = W / 2 + (n.x - cam.x) * cam.k, y = H / 2 + (n.y - cam.y) * cam.k;
      const hit = (3 + n.weight * 10) * cam.k + 8;
      const d = (x - px) ** 2 + (y - py) ** 2;
      if (d < hit * hit && d < bd) { best = n; bd = d; }
    }
    return best;
  };
  const toWorld = (px: number, py: number) => {
    const cam = camRef.current;
    return [cam.x + (px - W / 2) / cam.k, cam.y + (py - H / 2) / cam.k] as const;
  };
  const onDown = (e: React.PointerEvent<HTMLCanvasElement>) => {
    const rect = e.currentTarget.getBoundingClientRect();
    const px = e.clientX - rect.left, py = e.clientY - rect.top;
    const n = pick(px, py);
    if (!n) return;
    const [wx, wy] = toWorld(px, py);
    n.fixed = true;
    dragRef.current = { node: n, dx: n.x - wx, dy: n.y - wy };
    e.currentTarget.setPointerCapture(e.pointerId);
    simRef.current.wake();
  };
  const onMove = (e: React.PointerEvent<HTMLCanvasElement>) => {
    const rect = e.currentTarget.getBoundingClientRect();
    const px = e.clientX - rect.left, py = e.clientY - rect.top;
    const d = dragRef.current;
    if (d) {
      const [wx, wy] = toWorld(px, py);
      d.node.x = wx + d.dx; d.node.y = wy + d.dy;
      simRef.current.wake();
      return;
    }
    hoverRef.current = pick(px, py);
  };
  const onUp = (e: React.PointerEvent<HTMLCanvasElement>) => {
    const d = dragRef.current;
    if (d) { d.node.fixed = false; d.node.vx = 0; d.node.vy = 0; dragRef.current = null; simRef.current.wake(); }
    try { e.currentTarget.releasePointerCapture(e.pointerId); } catch {}
  };

  return (
    <div ref={wrapRef} className="panel-tight overflow-hidden">
      <div className="mb-2 flex flex-wrap items-center justify-between gap-3">
        <div className="section-title">Context Δ graph · live</div>
        <SymbolLegend />
        <div className="flex items-center gap-2 text-[10px]">
          <span className="pill"><span className="mr-1 inline-block h-1.5 w-3 rounded-full bg-accent-400" /> ↑ strengthens</span>
          <span className="pill"><span className="mr-1 inline-block h-1.5 w-3 rounded-full bg-danger" /> ↓ weakens</span>
          {graph && <span className="pill">{graph.edges.length} edges · {graph.nodes.length} nodes</span>}
          <span className="pill text-ink-400">{queryActive ? "edge length = query overlap" : "size = Σ|Δ| · width = |Δ|"}</span>
        </div>
      </div>

      <div className="relative" style={{ height: H }}>
        <canvas
          ref={canvasRef}
          style={{ width: W, height: H, cursor: "grab", touchAction: "none" }}
          onPointerDown={onDown}
          onPointerMove={onMove}
          onPointerUp={onUp}
          onPointerCancel={onUp}
          onPointerLeave={() => { hoverRef.current = null; }}
        />
        {!hasContext && (
          <div className="pointer-events-none absolute inset-0 flex items-center justify-center text-sm text-ink-400">
            type a context above — the Δ-graph appears and keeps following your edits
          </div>
        )}
      </div>
    </div>
  );
}
