// Shared canvas renderer for the symbol sigil — the emerging symbol drawn as
// a radial glyph.  Each descriptor occupies a slot on a ring (slot count =
// max_len, so the glyph fills clockwise as the walk proceeds); chords
// between slots carry the base cosine between the two descriptors, so the
// *shape* of the glyph is the internal structure of the new symbol.  A
// composition donut in the centre shows which archetypes it borrows from.

import type { WalkerSymbol } from "@/lib/realtime";

export interface SigilOpts {
  slots: number;
  labels?: boolean;
  time?: number;           // seconds, for the pulse on the newest node
  colorMap: Record<string, string>;
  title?: string;
  subtitle?: string;
}

export function drawSigil(ctx: CanvasRenderingContext2D, size: number, sym: WalkerSymbol | null, o: SigilOpts) {
  const cx = size / 2, cy = size / 2;
  const R = size * (o.labels ? 0.33 : 0.38);
  const n = sym?.descriptors.length ?? 0;
  const slots = Math.max(3, o.slots, n);

  // ambient slot ring
  ctx.strokeStyle = "rgba(255,255,255,0.07)"; ctx.lineWidth = 1;
  ctx.beginPath(); ctx.arc(cx, cy, R, 0, Math.PI * 2); ctx.stroke();
  for (let k = 0; k < slots; k++) {
    const a = -Math.PI / 2 + (2 * Math.PI * k) / slots;
    ctx.fillStyle = "rgba(255,255,255,0.10)";
    ctx.beginPath(); ctx.arc(cx + R * Math.cos(a), cy + R * Math.sin(a), 1.5, 0, Math.PI * 2); ctx.fill();
  }
  if (!sym || n === 0) {
    if (o.labels) {
      ctx.font = "500 11px Inter, system-ui"; ctx.textAlign = "center"; ctx.textBaseline = "middle";
      ctx.fillStyle = "rgba(159,173,193,0.8)";
      ctx.fillText("no symbol yet", cx, cy);
    }
    return;
  }

  const gains = sym.gains;
  const gmax = Math.max(1e-6, ...gains.map((g) => Math.abs(g)));
  const pts: [number, number][] = [];
  for (let k = 0; k < n; k++) {
    const a = -Math.PI / 2 + (2 * Math.PI * k) / slots;
    const r = R * (0.78 + 0.3 * Math.min(1, Math.abs(gains[k]) / gmax));
    pts.push([cx + r * Math.cos(a), cy + r * Math.sin(a)]);
  }

  // chords — pairwise cosine
  const P = sym.pairs;
  for (let i = 0; i < n; i++) for (let j = i + 1; j < n; j++) {
    const s = P?.[i]?.[j] ?? 0;
    const k = Math.max(0, Math.min(1, (s - 0.05) / 0.55));
    if (k <= 0) continue;
    const consecutive = j === i + 1;
    ctx.strokeStyle = consecutive ? `rgba(242,246,251,${(0.35 + 0.55 * k).toFixed(3)})` : `rgba(161,196,253,${(0.08 + 0.5 * k).toFixed(3)})`;
    ctx.lineWidth = (consecutive ? 1.2 : 0.5) + 2 * k;
    ctx.beginPath(); ctx.moveTo(pts[i][0], pts[i][1]); ctx.lineTo(pts[j][0], pts[j][1]); ctx.stroke();
  }

  // nodes
  for (let k = 0; k < n; k++) {
    const [x, y] = pts[k];
    const col = o.colorMap[sym.owners[k]] ?? "#888";
    const last = k === n - 1;
    const pulse = last && o.time !== undefined ? 1.2 * Math.sin(o.time * 5) : 0;
    const r = (o.labels ? 5 : 3) + (last ? 1.5 : 0) + pulse;
    if (last && o.labels) { ctx.shadowColor = col; ctx.shadowBlur = 14; }
    ctx.fillStyle = col; ctx.beginPath(); ctx.arc(x, y, r, 0, Math.PI * 2); ctx.fill();
    ctx.shadowBlur = 0;
    ctx.strokeStyle = "rgba(255,255,255,0.85)"; ctx.lineWidth = 1;
    ctx.beginPath(); ctx.arc(x, y, r, 0, Math.PI * 2); ctx.stroke();
    if (o.labels) {
      const a = -Math.PI / 2 + (2 * Math.PI * k) / slots;
      const lx = cx + (R * 1.18) * Math.cos(a), ly = cy + (R * 1.18) * Math.sin(a);
      ctx.font = `${last ? 600 : 500} 10.5px Inter, system-ui`;
      ctx.textAlign = Math.cos(a) > 0.3 ? "left" : Math.cos(a) < -0.3 ? "right" : "center";
      ctx.textBaseline = Math.sin(a) > 0.3 ? "top" : Math.sin(a) < -0.3 ? "bottom" : "middle";
      ctx.fillStyle = last ? "#f2f6fb" : "rgba(202,212,224,0.9)";
      ctx.fillText(sym.descriptors[k], lx, ly);
    }
  }

  // composition donut
  const r0 = size * (o.labels ? 0.11 : 0.13), thick = o.labels ? 7 : 4;
  let a0 = -Math.PI / 2;
  for (const [s, share] of Object.entries(sym.composition)) {
    const a1 = a0 + 2 * Math.PI * share;
    ctx.strokeStyle = o.colorMap[s] ?? "#888"; ctx.lineWidth = thick;
    ctx.beginPath(); ctx.arc(cx, cy, r0, a0 + 0.03, a1 - 0.03); ctx.stroke();
    a0 = a1;
  }
  if (o.labels) {
    const name = o.title ?? sym.name;
    let fs = 13;
    ctx.font = `600 ${fs}px Inter, system-ui`;
    while (ctx.measureText(name).width > r0 * 1.7 && fs > 8) { fs -= 1; ctx.font = `600 ${fs}px Inter, system-ui`; }
    ctx.textAlign = "center"; ctx.textBaseline = "middle"; ctx.fillStyle = "#f2f6fb";
    ctx.fillText(name, cx, cy - (o.subtitle ? 6 : 0));
    if (o.subtitle) {
      ctx.font = "500 9px JetBrains Mono, ui-monospace, monospace"; ctx.fillStyle = "rgba(159,173,193,0.95)";
      ctx.fillText(o.subtitle, cx, cy + 8);
    }
  }
}
