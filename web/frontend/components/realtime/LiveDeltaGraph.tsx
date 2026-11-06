"use client";
import * as React from "react";
import dynamic from "next/dynamic";
import { useMeasure } from "@/lib/useMeasure";
import { useLive } from "@/lib/liveGraph";
import { SymbolLegend } from "@/components/explorer/SymbolLegend";

const ForceGraph2D = dynamic(() => import("../plots/ForceGraph2DInner"), { ssr: false }) as any;

/**
 * The Explorer's Δ-graph, kept alive.  Same marks (node radius = Σ|Δ|,
 * edge width = |Δ|, teal strengthens / red weakens) rendered by the same
 * force-graph — but the simulation never cools (alphaDecay 0), velocity
 * damping is the "spring" damping, and the QUERY continuously rewrites the
 * spring rest lengths:
 *
 *   overlap(e) = sqrt(rel(u) · rel(v))          rel = sharpened query relevance
 *   rest(e)    = lerp(long, short, overlap)     short ≈ 36 px, long grows with contraction
 *
 * plus a radial force that pulls relevant nodes toward the centre and lets
 * the rest drift out.  Node objects are reused across graph updates so the
 * context edits morph the layout instead of re-seeding it.
 */

type GNode = { id: string; label: string; color: string; symbol: string; weight: number; pull: number; rel: number; relFrom: number; relT: number; x?: number; y?: number; vx?: number; vy?: number };
type GLink = { source: any; target: any; value: number; sign: "up" | "down"; delta: number; overlap: number };

const BASE_DIST = 80;
const SHORT_DIST = 34;

export function LiveDeltaGraph() {
  const [wrapRef, w] = useMeasure();
  const fgRef = React.useRef<any>(null);
  const graphSeq = useLive((s) => s.graphSeq);
  const graph = useLive((s) => s.graph);
  const damping = useLive((s) => s.damping);
  const queryActive = useLive((s) => s.queryActive);
  const hasContext = !!graph?.context?.trim();

  // persistent graphData — mutate in place, reuse node objects by id
  const dataRef = React.useRef<{ nodes: GNode[]; links: GLink[] }>({ nodes: [], links: [] });
  const [data, setData] = React.useState(dataRef.current);
  const edgeNorm = React.useRef({ lo: 0, range: 1 });

  React.useEffect(() => {
    if (!graph) return;
    const prev = new Map(dataRef.current.nodes.map((n) => [n.id, n]));
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

    // new nodes spawn near the centroid of their archetype siblings (or the
    // graph centre) so they ease in instead of flying from the periphery
    const bySym = new Map<string, { x: number; y: number; n: number }>();
    for (const n of dataRef.current.nodes) {
      if (n.x === undefined || n.y === undefined) continue;
      const a = bySym.get(n.symbol) ?? { x: 0, y: 0, n: 0 };
      a.x += n.x; a.y += n.y; a.n += 1; bySym.set(n.symbol, a);
    }
    const nodes: GNode[] = graph.nodes.map((n) => {
      const old = prev.get(n.id);
      const wgt = ((weight.get(n.id) ?? 0) - wLo) / wR;
      if (old) { old.weight = wgt; old.pull = n.pull; old.color = n.color; return old; }
      const a = bySym.get(n.symbol);
      const jitter = () => (Math.random() - 0.5) * 30;
      return {
        id: n.id, label: n.id, color: n.color, symbol: n.symbol, weight: wgt, pull: n.pull,
        rel: 0, relFrom: 0, relT: 0,
        x: a && a.n ? a.x / a.n + jitter() : jitter(),
        y: a && a.n ? a.y / a.n + jitter() : jitter(),
      };
    });
    const links: GLink[] = graph.edges.map((e) => ({
      source: e.source, target: e.target, value: e.abs_delta, sign: e.sign, delta: e.delta, overlap: 1,
    }));
    dataRef.current = { nodes, links };
    setData({ nodes, links });
  }, [graphSeq, graph]);

  // ---- forces: temperature-controlled springs + rigid-mode damping --------
  // d3's alpha is pinned at 1 (no decay); instead every force strength is
  // scaled by our own temperature T: 1 while a relevance tween runs, then an
  // exponential fade to exactly 0, so the graph is truly at rest between
  // changes.  A separate "rigid" force removes the net rotation and net
  // translation of the velocity field each tick — charge/link/radial are all
  // rotation-invariant, so without this any asymmetric impulse leaves the
  // whole layout spinning with nothing to stop it.
  const radialRef = React.useRef<any>(null);
  const rigidRef = React.useRef<any>(null);
  const tempRef = React.useRef({ T: 1, hotUntil: 0 });

  const applyForces = React.useCallback((T: number) => {
    const fg = fgRef.current;
    if (!fg) return;
    const st = useLive.getState();
    const active = st.queryActive;
    const contraction = st.contraction;
    const long = BASE_DIST + 140 * contraction;
    try {
      const link = fg.d3Force("link");
      const charge = fg.d3Force("charge");
      if (charge) charge.strength(-160 * T);
      if (link) {
        link
          .distance((l: GLink) => (active ? long + (SHORT_DIST - long) * l.overlap : BASE_DIST))
          .strength((l: GLink) => (active ? 0.18 + 0.4 * l.overlap : 0.35) * T);
      }
      if (!radialRef.current) radialRef.current = makeRelevanceRadial();
      radialRef.current.configure(active, (0.6 + 0.6 * contraction) * T);
      if (fg.d3Force("radial") !== radialRef.current) fg.d3Force("radial", radialRef.current);
      if (!rigidRef.current) rigidRef.current = makeRigidDamper();
      // registered last so it sees the velocities every other force produced
      if (fg.d3Force("rigid") !== rigidRef.current) fg.d3Force("rigid", rigidRef.current);
    } catch (err) {
      console.warn("[live] applyForces failed", err);
    }
  }, []);

  /** Keep the layout hot for `ms` (new data, drag, slider) — the envelope fades after. */
  const wake = React.useCallback((ms = 600) => {
    tempRef.current.hotUntil = Math.max(tempRef.current.hotUntil, performance.now() + ms);
    fgRef.current?.d3ReheatSimulation?.();
  }, []);

  // first mount / data swap: install forces (new link arrays need the distance fn re-bound)
  React.useEffect(() => { applyForces(1); wake(1200); }, [data, applyForces, wake]);

  // animation: tween rel → relT over `transition` seconds (ease-in-out); the
  // temperature stays 1 while rest lengths move, then fades with τ = 0.7 s.
  const lastActive = React.useRef(false);
  React.useEffect(() => {
    let raf = 0;
    let lastSeq = -1;
    let tweenStart = 0;
    let tweening = false;
    let lastT = -1;
    const FADE_TAU = 700;
    const ease = (t: number) => (t < 0.5 ? 2 * t * t : 1 - Math.pow(-2 * t + 2, 2) / 2);
    const tick = () => {
      raf = requestAnimationFrame(tick);
      const st = useLive.getState();
      const { nodes, links } = dataRef.current;
      const now = performance.now();
      if (st.relevanceSeq !== lastSeq) {
        lastSeq = st.relevanceSeq;
        const g = st.focus;
        for (const n of nodes) {
          const r = st.relevance[n.id];
          n.relFrom = n.rel;
          n.relT = r === undefined ? 0 : Math.pow(Math.max(0, Math.min(1, r)), g);
        }
        tweenStart = now;
        tweening = true;
      }
      if (st.queryActive !== lastActive.current) {
        lastActive.current = st.queryActive;
        tempRef.current.hotUntil = Math.max(tempRef.current.hotUntil, now + 400);
        if (!tweening) { tweenStart = now; tweening = true; }
      }
      let restMoved = false;
      if (tweening) {
        const dur = Math.max(100, st.transition * 1000);
        const u = Math.min(1, (now - tweenStart) / dur);
        const k = ease(u);
        for (const n of nodes) n.rel = n.relFrom + (n.relT - n.relFrom) * k;
        for (const l of links) {
          const a = typeof l.source === "object" ? (l.source as GNode).rel : 0;
          const b = typeof l.target === "object" ? (l.target as GNode).rel : 0;
          l.overlap = Math.sqrt(Math.max(0, a * b));
        }
        restMoved = true;
        // hot while moving, plus a short tail so the springs can reach the new lengths
        tempRef.current.hotUntil = Math.max(tempRef.current.hotUntil, now + 500);
        if (u >= 1) tweening = false;
      }
      // temperature envelope
      const env = tempRef.current;
      let T = now < env.hotUntil ? 1 : Math.exp(-(now - env.hotUntil) / FADE_TAU);
      if (T < 0.01) T = 0;
      env.T = T;
      if (restMoved || Math.abs(T - lastT) > 1e-3) {
        lastT = T;
        applyForces(T);
        if (T > 0) fgRef.current?.d3ReheatSimulation?.();
      }
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [applyForces]);

  // re-sharpen when focus changes
  const focus = useLive((s) => s.focus);
  React.useEffect(() => { useLive.setState((s) => ({ relevanceSeq: s.relevanceSeq + 1 })); }, [focus]);
  const contraction = useLive((s) => s.contraction);
  React.useEffect(() => { wake(1500); }, [contraction, wake]);

  const height = Math.max(560, Math.min(860, Math.round((w || 1000) * 0.5)));
  const { lo: eLo, range: eRange } = edgeNorm.current;

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

      <div className="relative" style={{ height }}>
        {w > 0 && (
          <ForceGraph2D
            fwdRef={fgRef}
            graphData={data}
            width={w - 24}
            height={height}
            backgroundColor="rgba(0,0,0,0)"
            cooldownTicks={Infinity}
            cooldownTime={Infinity}
            d3AlphaDecay={0}
            d3AlphaMin={0}
            d3VelocityDecay={damping}
            warmupTicks={0}
            enableNodeDrag={true}
            onNodeDrag={() => wake(400)}
            onNodeDragEnd={() => wake(1500)}
            linkColor={(l: GLink) => {
              const a = queryActive ? 0.12 + 0.85 * l.overlap : 1;
              return l.sign === "up" ? `rgba(38,161,149,${(0.78 * a).toFixed(3)})` : `rgba(191,97,106,${(0.72 * a).toFixed(3)})`;
            }}
            linkWidth={(l: GLink) => (0.6 + ((l.value - eLo) / eRange) * 5.4) * (queryActive ? 0.5 + 0.7 * l.overlap : 1)}
            nodeRelSize={1}
            nodeVal={(n: GNode) => 4 + n.weight * 24}
            nodeCanvasObject={(node: GNode, ctx: CanvasRenderingContext2D, scale: number) => {
              const x = node.x ?? 0, y = node.y ?? 0;
              const rel = queryActive ? node.rel : 1;
              const r = (3 + node.weight * 10) + (queryActive ? 4 * node.rel : 0) + 1.5 / scale;
              ctx.globalAlpha = queryActive ? 0.22 + 0.78 * rel : 1;
              if (queryActive && node.rel > 0.45) {
                ctx.shadowColor = node.color; ctx.shadowBlur = 22 * (node.rel - 0.45) / 0.55;
              }
              ctx.beginPath(); ctx.arc(x, y, r, 0, 2 * Math.PI, false);
              ctx.fillStyle = node.color; ctx.fill();
              ctx.shadowBlur = 0;
              ctx.lineWidth = 0.6; ctx.strokeStyle = "rgba(0,0,0,0.5)"; ctx.stroke();
              const showLabel = scale > 0.55 || (queryActive && node.rel > 0.6);
              if (showLabel) {
                const fs = (12 + node.weight * 4 + (queryActive ? 3 * node.rel : 0)) / scale;
                ctx.font = `${queryActive && node.rel > 0.6 ? 700 : 600} ${fs}px Inter, system-ui`;
                ctx.textAlign = "center"; ctx.textBaseline = "top";
                ctx.fillStyle = "rgba(0,0,0,0.7)";
                ctx.fillText(node.label, x + 0.6 / scale, y + r + 2.6 / scale);
                ctx.fillStyle = "#f2f6fb";
                ctx.fillText(node.label, x, y + r + 2 / scale);
              }
              ctx.globalAlpha = 1;
            }}
            nodePointerAreaPaint={(node: GNode, color: string, ctx: CanvasRenderingContext2D) => {
              ctx.fillStyle = color;
              ctx.beginPath(); ctx.arc(node.x ?? 0, node.y ?? 0, 6 + node.weight * 10, 0, 2 * Math.PI); ctx.fill();
            }}
            linkHoverPrecision={6}
          />
        )}
        {!hasContext && (
          <div className="pointer-events-none absolute inset-0 flex items-center justify-center text-sm text-ink-400">
            type a context above — the Δ-graph appears and keeps following your edits
          </div>
        )}
      </div>
    </div>
  );
}

/** Radial spring: relevant nodes are pulled toward the centre, irrelevant
 *  ones toward a wide ring — the query literally pulls its subgraph in.
 *  One instance lives for the component's lifetime; `configure` updates it.
 *  `strength` already carries the temperature. */
function makeRelevanceRadial() {
  let nodes: GNode[] = [];
  let active = false;
  let strength = 1;
  const force: any = () => {
    if (strength <= 0) return;
    for (const n of nodes) {
      const x = n.x ?? 0, y = n.y ?? 0;
      const r = Math.hypot(x, y) || 1e-6;
      const target = active ? 40 + (1 - n.rel) * 240 : 0;
      const k = (active ? 0.03 : 0.02) * strength;
      const pull = (target - r) * k;
      n.vx = (n.vx ?? 0) + (x / r) * pull;
      n.vy = (n.vy ?? 0) + (y / r) * pull;
    }
  };
  force.initialize = (n: GNode[]) => { nodes = n; };
  force.configure = (a: boolean, s: number) => { active = a; strength = s; };
  return force;
}

/** Rigid-mode damper.  Subtracts the mean velocity (net translation) and
 *  the rigid rotation about the centroid (net angular momentum) from every
 *  node's velocity, each tick.  The remaining motion is purely internal —
 *  expansion / contraction of the springs — which is the only motion that
 *  carries information.  Fixed (dragged) nodes are left alone. */
function makeRigidDamper() {
  let nodes: GNode[] = [];
  const force: any = () => {
    const n = nodes.length;
    if (n < 2) return;
    let cx = 0, cy = 0, mvx = 0, mvy = 0;
    for (const p of nodes) { cx += p.x ?? 0; cy += p.y ?? 0; mvx += p.vx ?? 0; mvy += p.vy ?? 0; }
    cx /= n; cy /= n; mvx /= n; mvy /= n;
    let L = 0, I = 0;
    for (const p of nodes) {
      const rx = (p.x ?? 0) - cx, ry = (p.y ?? 0) - cy;
      const vx = (p.vx ?? 0) - mvx, vy = (p.vy ?? 0) - mvy;
      L += rx * vy - ry * vx;
      I += rx * rx + ry * ry;
    }
    const w = I > 1e-9 ? L / I : 0;
    for (const p of nodes) {
      if ((p as any).fx !== undefined && (p as any).fx !== null) continue;
      const rx = (p.x ?? 0) - cx, ry = (p.y ?? 0) - cy;
      p.vx = (p.vx ?? 0) - mvx - (-ry * w);
      p.vy = (p.vy ?? 0) - mvy - (rx * w);
    }
  };
  force.initialize = (n: GNode[]) => { nodes = n; };
  return force;
}
