"use client";
import * as React from "react";
import { useMeasure } from "@/lib/useMeasure";
import { useWalker, type WalkerSymbol } from "@/lib/realtime";
import { drawSigil } from "./sigil";

/** The symbol under construction as a live radial glyph, plus a gallery of
 *  sealed symbols as mini sigils. */
export function SymbolSigil() {
  const [wrapRef, w] = useMeasure();
  const canvasRef = React.useRef<HTMLCanvasElement>(null);
  const size = Math.max(240, Math.min(380, Math.round((w || 320) - 24)));
  const sealed = useWalker((s) => s.frame?.sealed ?? EMPTY);
  const hasFrame = useWalker((s) => !!s.frame);

  React.useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const dpr = Math.min(2, window.devicePixelRatio || 1);
    canvas.width = size * dpr; canvas.height = size * dpr;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    let raf = 0;
    const t0 = performance.now();
    const loop = () => {
      raf = requestAnimationFrame(loop);
      const st = useWalker.getState();
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      ctx.clearRect(0, 0, size, size);
      const sym = st.frame?.symbol ?? null;
      const n = sym?.descriptors.length ?? 0;
      drawSigil(ctx, size, sym, {
        slots: st.params?.max_len ?? 8,
        labels: true,
        time: (performance.now() - t0) / 1000,
        colorMap: st.layout?.color_map ?? {},
        subtitle: sym && n ? `coh ${sym.coherence.toFixed(2)} · fit ${sym.context_fit.toFixed(2)}` : undefined,
      });
    };
    raf = requestAnimationFrame(loop);
    return () => cancelAnimationFrame(raf);
  }, [size]);

  return (
    <div ref={wrapRef} className="panel-tight">
      <div className="mb-1 flex items-center justify-between">
        <div className="section-title">Emerging symbol</div>
        <span className="pill text-ink-400">chord = cosine · ring = origin</span>
      </div>
      <div className="flex justify-center">
        <canvas ref={canvasRef} style={{ width: size, height: size }} />
      </div>
      {hasFrame && (
        <div className="mt-2 border-t border-ink-700/50 pt-2">
          <div className="mb-1.5 flex items-center justify-between">
            <span className="sub-title">sealed</span>
            <span className="text-[10px] text-ink-500">{sealed.length ? `${sealed.length} symbol${sealed.length > 1 ? "s" : ""}` : "none yet — seal the walk or let it reach max length"}</span>
          </div>
          <div className="flex flex-wrap gap-2">
            {sealed.map((s) => <MiniSigil key={s.index ?? s.name} sym={s} />)}
          </div>
        </div>
      )}
    </div>
  );
}

const EMPTY: WalkerSymbol[] = [];

export function MiniSigil({ sym, size = 72 }: { sym: WalkerSymbol; size?: number }) {
  const ref = React.useRef<HTMLCanvasElement>(null);
  const colorMap = useWalker((s) => s.layout?.color_map ?? {});
  const slots = useWalker((s) => s.params?.max_len ?? 8);
  React.useEffect(() => {
    const c = ref.current; if (!c) return;
    const dpr = Math.min(2, window.devicePixelRatio || 1);
    c.width = size * dpr; c.height = size * dpr;
    const ctx = c.getContext("2d"); if (!ctx) return;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, size, size);
    drawSigil(ctx, size, sym, { slots: Math.max(slots, sym.descriptors.length), labels: false, colorMap });
  }, [sym, size, colorMap, slots]);
  return (
    <div className="flex flex-col items-center gap-0.5" title={sym.descriptors.join(" · ")}>
      <canvas ref={ref} style={{ width: size, height: size }} className="rounded-lg bg-ink-950/40" />
      <span className="max-w-[80px] truncate text-[10px] text-ink-300">{sym.name}</span>
    </div>
  );
}
