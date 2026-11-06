"use client";
import * as React from "react";
import { useMeasure } from "@/lib/useMeasure";
import { useWalker, ringAnchors, ringWeights, type Candidate } from "@/lib/realtime";

/**
 * The field view: every descriptor as a dot on the 2D layout, the live Δ
 * edges the context is lighting up, the walker's trail, and the candidates
 * it is weighing next.  All drawing happens in a requestAnimationFrame loop
 * that reads the latest frame straight from the store — React never
 * re-renders this component per frame.
 *
 *   field  — the pointer IS the context: a Gaussian kernel over the dots
 *   ring   — archetypes sit on a ring; the pointer blends their centroids
 *   choose — no pointer context; click a candidate to step
 *
 * Clicking a candidate steps to it in every mode.
 */
export function TraversalCanvas() {
  const [wrapRef, w] = useMeasure();
  const canvasRef = React.useRef<HTMLCanvasElement>(null);
  const hoverRef = React.useRef<number | null>(null);
  const mode = useWalker((s) => s.mode);
  const status = useWalker((s) => s.status);
  const hasLayout = useWalker((s) => !!s.layout);

  const W = Math.max(320, w - 2);
  const H = Math.max(420, Math.min(720, Math.round(W * 0.64)));
  const PAD = 28;

  // unit square ↔ canvas
  const toXY = React.useCallback((u: number, v: number): [number, number] =>
    [PAD + u * (W - 2 * PAD), PAD + (1 - v) * (H - 2 * PAD)], [W, H]);
  const toUV = React.useCallback((x: number, y: number): [number, number] =>
    [(x - PAD) / (W - 2 * PAD), 1 - (y - PAD) / (H - 2 * PAD)], [W, H]);

  // ---- render loop ----------------------------------------------------------
  React.useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const dpr = Math.min(2, window.devicePixelRatio || 1);
    canvas.width = Math.round(W * dpr);
    canvas.height = Math.round(H * dpr);
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    let heatSmooth: Float32Array | null = null;
    let raf = 0;
    const t0 = performance.now();

    const draw = () => {
      raf = requestAnimationFrame(draw);
      const st = useWalker.getState();
      const lay = st.layout;
      const f = st.frame;
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      ctx.clearRect(0, 0, W, H);
      if (!lay) return;
      const t = (performance.now() - t0) / 1000;
      const nodes = lay.nodes;
      const N = nodes.length;
      if (!heatSmooth || heatSmooth.length !== N) heatSmooth = new Float32Array(N);
      const target = f?.heat && f.heat.length === N ? f.heat : null;
      for (let i = 0; i < N; i++) {
        const tg = target ? target[i] : 0;
        heatSmooth[i] += (tg - heatSmooth[i]) * 0.18;
      }
      const sign = f?.params?.sign ?? "up";
      const edgeRGB = sign === "down" ? "191,97,106" : "59,189,176";

      // positions (ring mode moves nothing — dots stay, anchors are overlaid)
      const pos: [number, number][] = new Array(N);
      for (let i = 0; i < N; i++) pos[i] = toXY(nodes[i].x, nodes[i].y);

      // ---- hot edges
      if (f?.hot_edges?.length) {
        const vmax = f.hot_edges[0][2] || 1;
        ctx.lineCap = "round";
        for (const [i, j, v] of f.hot_edges) {
          const k = v / vmax;
          ctx.strokeStyle = `rgba(${edgeRGB},${(0.06 + 0.5 * k).toFixed(3)})`;
          ctx.lineWidth = 0.4 + 2.2 * k;
          ctx.beginPath(); ctx.moveTo(pos[i][0], pos[i][1]); ctx.lineTo(pos[j][0], pos[j][1]); ctx.stroke();
        }
      }

      // ---- all descriptors
      for (let i = 0; i < N; i++) {
        const h = heatSmooth[i];
        const r = 1.5 + 3.4 * h;
        ctx.globalAlpha = 0.22 + 0.78 * h;
        ctx.fillStyle = nodes[i].color;
        ctx.beginPath(); ctx.arc(pos[i][0], pos[i][1], r, 0, Math.PI * 2); ctx.fill();
      }
      ctx.globalAlpha = 1;

      // ---- ring anchors
      if (st.mode === "ring") {
        const anchors = ringAnchors(lay.symbols);
        const lp = st.localPointer;
        for (const s of lay.symbols) {
          const [ax, ay] = toXY(anchors[s][0], anchors[s][1]);
          const wgt = st.anchorWeights[s] ?? 0;
          const col = lay.color_map[s] ?? "#888";
          if (lp && wgt > 0.01) {
            const [px, py] = toXY(lp[0], lp[1]);
            ctx.strokeStyle = col; ctx.globalAlpha = 0.15 + 0.6 * wgt; ctx.lineWidth = 0.8 + 3 * wgt;
            ctx.beginPath(); ctx.moveTo(px, py); ctx.lineTo(ax, ay); ctx.stroke();
            ctx.globalAlpha = 1;
          }
          ctx.fillStyle = col; ctx.globalAlpha = 0.25 + 0.75 * Math.min(1, wgt * 1.5);
          ctx.beginPath(); ctx.arc(ax, ay, 9 + 10 * wgt, 0, Math.PI * 2); ctx.fill();
          ctx.globalAlpha = 1;
          ctx.strokeStyle = "rgba(255,255,255,0.65)"; ctx.lineWidth = 1;
          ctx.beginPath(); ctx.arc(ax, ay, 9 + 10 * wgt, 0, Math.PI * 2); ctx.stroke();
          ctx.font = "600 11px Inter, system-ui"; ctx.textAlign = "center"; ctx.textBaseline = "middle";
          ctx.fillStyle = "#f2f6fb";
          ctx.fillText(s, ax, ay - 16 - 10 * wgt);
        }
      }

      // ---- cursor halo (field)
      if (st.mode === "field") {
        const p = st.localPointer ?? f?.ctx_point ?? null;
        if (p) {
          const [px, py] = toXY(p[0], p[1]);
          const R = st.bandwidth * (W - 2 * PAD) * 1.6;
          const g = ctx.createRadialGradient(px, py, 0, px, py, R);
          g.addColorStop(0, "rgba(255,255,255,0.10)");
          g.addColorStop(0.6, "rgba(59,189,176,0.05)");
          g.addColorStop(1, "rgba(59,189,176,0)");
          ctx.fillStyle = g; ctx.beginPath(); ctx.arc(px, py, R, 0, Math.PI * 2); ctx.fill();
          ctx.strokeStyle = "rgba(255,255,255,0.35)"; ctx.lineWidth = 1;
          ctx.beginPath(); ctx.arc(px, py, 4, 0, Math.PI * 2); ctx.stroke();
        }
      }

      // ---- path
      const path = f?.path ?? [];
      if (path.length > 1) {
        for (let k = 1; k < path.length; k++) {
          const a = pos[path[k - 1]], b = pos[path[k]];
          const q = k / (path.length - 1);
          ctx.strokeStyle = `rgba(242,246,251,${(0.25 + 0.7 * q).toFixed(3)})`;
          ctx.lineWidth = 1.2 + 2.2 * q;
          ctx.beginPath(); ctx.moveTo(a[0], a[1]); ctx.lineTo(b[0], b[1]); ctx.stroke();
        }
      }
      for (let k = 0; k < path.length; k++) {
        const i = path[k];
        const [x, y] = pos[i];
        const last = k === path.length - 1;
        const r = last ? 7 + 1.5 * Math.sin(t * 5) : 5;
        if (last) {
          ctx.shadowColor = nodes[i].color; ctx.shadowBlur = 18;
        }
        ctx.fillStyle = nodes[i].color;
        ctx.beginPath(); ctx.arc(x, y, r, 0, Math.PI * 2); ctx.fill();
        ctx.shadowBlur = 0;
        ctx.strokeStyle = "rgba(255,255,255,0.9)"; ctx.lineWidth = last ? 1.6 : 1;
        ctx.beginPath(); ctx.arc(x, y, r, 0, Math.PI * 2); ctx.stroke();
        label(ctx, nodes[i].id, x, y + r + 3, last ? 12 : 10.5, 1);
      }

      // ---- candidates
      const cands: Candidate[] = f?.candidates ?? [];
      const cur = path.length ? pos[path[path.length - 1]] : null;
      const hov = hoverRef.current;
      for (const c of cands) {
        const [x, y] = pos[c.i];
        const p = c.p;
        const hovered = hov === c.i;
        if (cur) {
          ctx.setLineDash([3, 4]);
          ctx.strokeStyle = `rgba(255,255,255,${(0.12 + 0.55 * p).toFixed(3)})`;
          ctx.lineWidth = 0.6 + 2.4 * p;
          ctx.beginPath(); ctx.moveTo(cur[0], cur[1]); ctx.lineTo(x, y); ctx.stroke();
          ctx.setLineDash([]);
        }
        const pulse = cur ? 0 : 1.5 * Math.sin(t * 3 + c.i);
        const r = 4 + 9 * p + pulse + (hovered ? 3 : 0);
        ctx.strokeStyle = hovered ? "#ffffff" : nodes[c.i].color;
        ctx.lineWidth = hovered ? 2 : 1.4;
        ctx.beginPath(); ctx.arc(x, y, r, 0, Math.PI * 2); ctx.stroke();
        ctx.fillStyle = nodes[c.i].color; ctx.globalAlpha = 0.18 + 0.4 * p;
        ctx.beginPath(); ctx.arc(x, y, r, 0, Math.PI * 2); ctx.fill();
        ctx.globalAlpha = 1;
        label(ctx, `${c.id} · ${Math.round(p * 100)}%`, x, y - r - 13, 10.5, hovered ? 1 : 0.85);
      }

      // ---- hints
      if (!f?.has_context && st.mode !== "choose") {
        hint(ctx, W, H, st.mode === "field"
          ? "move over the field — the pointer is the context"
          : "move between the archetypes to blend them");
      } else if (st.mode === "choose" && !f?.has_context) {
        hint(ctx, W, H, "choose mode uses the sentence context — type one in the controls");
      }
    };
    raf = requestAnimationFrame(draw);
    return () => cancelAnimationFrame(raf);
  }, [W, H, toXY]);

  // ---- pointer -------------------------------------------------------------
  const onMove = (e: React.PointerEvent<HTMLCanvasElement>) => {
    const rect = e.currentTarget.getBoundingClientRect();
    const [u, v] = toUV(e.clientX - rect.left, e.clientY - rect.top);
    const st = useWalker.getState();
    // hover over a candidate?
    const f = st.frame; const lay = st.layout;
    let hit: number | null = null;
    if (f && lay) {
      let best = 14 * 14;
      for (const c of f.candidates) {
        const n = lay.nodes[c.i];
        const [x, y] = toXY(n.x, n.y);
        const d2 = (x - (e.clientX - rect.left)) ** 2 + (y - (e.clientY - rect.top)) ** 2;
        if (d2 < best) { best = d2; hit = c.i; }
      }
    }
    hoverRef.current = hit;
    if (st.mode === "field") st.pointer(Math.min(1, Math.max(0, u)), Math.min(1, Math.max(0, v)));
    else if (st.mode === "ring" && lay) {
      useWalker.setState({ localPointer: [u, v] });
      st.anchors(ringWeights(ringAnchors(lay.symbols), u, v, 0.11 + st.bandwidth));
    }
  };
  const onClick = () => {
    const hit = hoverRef.current;
    if (hit !== null) useWalker.getState().step(hit);
  };

  return (
    <div ref={wrapRef} className="panel-tight overflow-hidden">
      <div className="mb-2 flex flex-wrap items-center justify-between gap-2">
        <div className="section-title">Δ-field traversal</div>
        <div className="flex items-center gap-2 text-[10px]">
          <span className="pill"><span className="mr-1 inline-block h-1.5 w-3 rounded-full bg-accent-400" /> live Δ edges</span>
          <span className="pill"><span className="mr-1 inline-block h-2 w-2 rounded-full border border-white/80" /> candidates · ring = p</span>
          <span className="pill text-ink-400">dot brightness = context pull</span>
        </div>
      </div>
      <div className="relative" style={{ height: H }}>
        <canvas
          ref={canvasRef}
          style={{ width: W, height: H, cursor: mode === "choose" ? "pointer" : "crosshair", touchAction: "none" }}
          onPointerMove={onMove}
          onPointerDown={onMove}
          onClick={onClick}
          onPointerLeave={() => { hoverRef.current = null; }}
        />
        {!hasLayout && (
          <div className="absolute inset-0 flex items-center justify-center text-sm text-ink-400">
            {status === "connecting" ? "connecting to the walker…" : status === "error" ? "socket error — is the backend running?" : "waiting for layout…"}
          </div>
        )}
      </div>
    </div>
  );
}

function label(ctx: CanvasRenderingContext2D, text: string, x: number, y: number, size: number, alpha: number) {
  ctx.font = `600 ${size}px Inter, system-ui`;
  ctx.textAlign = "center"; ctx.textBaseline = "top";
  ctx.globalAlpha = alpha;
  ctx.fillStyle = "rgba(0,0,0,0.75)"; ctx.fillText(text, x + 0.7, y + 0.7);
  ctx.fillStyle = "#f2f6fb"; ctx.fillText(text, x, y);
  ctx.globalAlpha = 1;
}

function hint(ctx: CanvasRenderingContext2D, W: number, H: number, text: string) {
  ctx.font = "500 12px Inter, system-ui"; ctx.textAlign = "center"; ctx.textBaseline = "bottom";
  ctx.fillStyle = "rgba(159,173,193,0.9)";
  ctx.fillText(text, W / 2, H - 10);
}
