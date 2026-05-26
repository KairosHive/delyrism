// Build a downloadable .zip bundle of the current Explorer session.
//
// Contents:
//   manifest.json   — full sidebar state + timestamp (reproducibility)
//   README.md       — humans: how to read the bundle
//   data/*.json     — raw response of every panel
//   figures/*.html  — standalone, double-click-to-open interactive plots
//                     (Plotly + force-graph loaded from CDN)
//
// All data is pulled from the TanStack Query cache so the export captures
// the exact state the user is currently looking at — no extra round-trips.

import JSZip from "jszip";
import { QueryClient } from "@tanstack/react-query";
import {
  ProposeResponse,
  AmbiguityResponse,
  Reduce2DResponse,
  ShiftResponse,
  DeltaGraphResponse,
  SubgraphResponse,
  AttentionResponse,
  SimilarityResponse,
  SymbolSimilarityResponse,
  TransformationsResponse,
} from "./api";
import { SidebarState } from "./store";

const PLOTLY_CDN = "https://cdn.plot.ly/plotly-2.35.2.min.js";
const FORCE_GRAPH_CDN = "https://unpkg.com/force-graph@1.43";

// ─────────────────── helpers ───────────────────

function fmtTs(d: Date = new Date()): string {
  const pad = (n: number) => String(n).padStart(2, "0");
  return (
    `${d.getFullYear()}${pad(d.getMonth() + 1)}${pad(d.getDate())}-` +
    `${pad(d.getHours())}${pad(d.getMinutes())}${pad(d.getSeconds())}`
  );
}

function findCachedByPrefix<T>(qc: QueryClient, prefix: string): T | undefined {
  // Most query keys start with a string; find the freshest match by prefix.
  const all = qc.getQueryCache().getAll();
  for (const q of all) {
    const k = q.queryKey;
    if (Array.isArray(k) && k[0] === prefix && q.state.data) {
      return q.state.data as T;
    }
  }
  return undefined;
}

function findAllCachedByPrefix<T>(qc: QueryClient, prefix: string): { key: any[]; data: T }[] {
  const out: { key: any[]; data: T }[] = [];
  for (const q of qc.getQueryCache().getAll()) {
    const k = q.queryKey;
    if (Array.isArray(k) && k[0] === prefix && q.state.data) {
      out.push({ key: k, data: q.state.data as T });
    }
  }
  return out;
}

function safeFilename(s: string): string {
  return s.replace(/[^a-z0-9-_]/gi, "_").slice(0, 60);
}

function html(title: string, body: string, head = ""): string {
  return `<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>${escapeHtml(title)}</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
  body { margin: 0; background: #0b0f1a; color: #cad4e0; font-family: Inter, system-ui, sans-serif; }
  .wrap { max-width: 1100px; margin: 0 auto; padding: 24px; }
  h1 { font-weight: 300; font-size: 1.4rem; letter-spacing: 0.06em; margin: 0; }
  .sub { color: #6e7e95; font-size: 0.8rem; margin-top: 4px; }
  .chart { min-height: 600px; }
</style>
${head}
</head>
<body>
<div class="wrap">
${body}
</div>
</body>
</html>`;
}

function escapeHtml(s: string): string {
  return s
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#39;");
}

function plotlyShell(title: string, traces: any, layout: any): string {
  const cfg = { displaylogo: false, responsive: true };
  // Plotly charts use a full-bleed layout — header at top, plot fills the rest.
  // Plotly itself handles its own canvas sizing via responsive:true.
  return `<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>${escapeHtml(title)}</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
  html, body { margin: 0; padding: 0; height: 100%;
               background: #0b0f1a; color: #cad4e0;
               font-family: Inter, system-ui, sans-serif; overflow: hidden; }
  .wrap { display: flex; flex-direction: column; height: 100vh; }
  .hdr { padding: 16px 24px; flex-shrink: 0;
         border-bottom: 1px solid rgba(255,255,255,0.06); }
  .hdr h1 { margin: 0; font-weight: 300; font-size: 1.2rem; letter-spacing: 0.06em; }
  #chart { flex: 1; min-height: 0; }
</style>
<script src="${PLOTLY_CDN}"></script>
</head>
<body>
<div class="wrap">
  <div class="hdr"><h1>${escapeHtml(title)}</h1></div>
  <div id="chart"></div>
</div>
<script>
const traces = ${JSON.stringify(traces)};
const layout = ${JSON.stringify(layout)};
Plotly.newPlot('chart', traces, layout, ${JSON.stringify(cfg)});
window.addEventListener('resize', () => Plotly.Plots.resize('chart'));
</script>
</body>
</html>`;
}

function forceGraphShell(title: string, data: any, opts: { kind: "delta" | "subgraph" }): string {
  // Subgraph nodes carry score; symbol nodes are bigger than descriptor nodes.
  // Δ-graph nodes are uniform descriptor dots — labels at all zoom levels.
  // Either way: full-bleed canvas, labels rendered next to each dot, auto
  // zoomToFit after physics settle.
  const isSubgraph = opts.kind === "subgraph";
  return `<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>${escapeHtml(title)}</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
  html, body { margin: 0; padding: 0; height: 100%; background: #0b0f1a;
               color: #cad4e0; font-family: Inter, system-ui, sans-serif;
               overflow: hidden; }
  .wrap { display: flex; flex-direction: column; height: 100vh; }
  .hdr { padding: 16px 24px; flex-shrink: 0;
         border-bottom: 1px solid rgba(255,255,255,0.06); }
  .hdr h1 { margin: 0; font-weight: 300; font-size: 1.2rem; letter-spacing: 0.06em; }
  .sub { color: #6e7e95; font-size: 0.75rem; margin-top: 4px; }
  #graph { flex: 1; min-height: 0; position: relative; }
</style>
<script src="${FORCE_GRAPH_CDN}"></script>
</head>
<body>
<div class="wrap">
  <div class="hdr">
    <h1>${escapeHtml(title)}</h1>
    <div class="sub">drag to pan · scroll to zoom · drag nodes to pin · click empty space to reset</div>
  </div>
  <div id="graph"></div>
</div>
<script>
const data = ${JSON.stringify(data)};

// Per-node base radius (visual).  Subgraph distinguishes symbol vs
// descriptor by their score; Δ-graph uses a uniform smaller radius.
const baseR = ${isSubgraph}
  ? (n => (n && n.score && n.score >= 3) ? 9 : 4.5)
  : (n => 4);

const el = document.getElementById('graph');
const g = ForceGraph()(el)
  .graphData(data)
  .backgroundColor('#0b0f1a')
  .nodeId('id')
  .nodeRelSize(1)                     // we draw nodes ourselves
  .linkColor(l => l.color || 'rgba(202,212,224,0.35)')
  .linkWidth(l => Math.max(0.5, (l.weight != null ? l.weight : (l.abs_delta != null ? l.abs_delta : 0.5)) * 1.6))
  .nodeCanvasObjectMode(() => 'replace')
  .nodeCanvasObject((node, ctx, globalScale) => {
    const r = baseR(node);
    // dot
    ctx.beginPath();
    ctx.fillStyle = node.color || '#88c0d0';
    ctx.arc(node.x, node.y, r, 0, 2 * Math.PI);
    ctx.fill();
    ctx.strokeStyle = 'rgba(0,0,0,0.4)';
    ctx.lineWidth = 0.5 / globalScale;
    ctx.stroke();
    // label
    const fontSize = Math.max(9, 11 / globalScale);
    ctx.font = (node.score && node.score >= 3 ? '600 ' : '') + fontSize + 'px Inter, system-ui, sans-serif';
    ctx.fillStyle = '#dbe2ee';
    ctx.textAlign = 'left';
    ctx.textBaseline = 'middle';
    // strip leading "S:" / "D:" prefix the engine uses for subgraph nodes
    const label = String(node.id || '').replace(/^[SD]:/, '');
    ctx.fillText(label, node.x + r + 3 / globalScale, node.y);
  })
  .nodePointerAreaPaint((node, color, ctx) => {
    const r = baseR(node) + 4;
    ctx.beginPath();
    ctx.fillStyle = color;
    ctx.arc(node.x, node.y, r, 0, 2 * Math.PI);
    ctx.fill();
  })
  .cooldownTime(2500)
  .onEngineStop(() => g.zoomToFit(600, 60));

// Resize with the window
window.addEventListener('resize', () => g.width(el.clientWidth).height(el.clientHeight));
g.width(el.clientWidth).height(el.clientHeight);
</script>
</body>
</html>`;
}

// ─────────────────── individual exporters ───────────────────

const SPECTRAL = [
  [0.0, "#3a86ff"],
  [0.25, "#06d6a0"],
  [0.5, "#ffd166"],
  [0.75, "#f77f00"],
  [1.0, "#ef476f"],
];

const PLOTLY_DARK_LAYOUT = {
  paper_bgcolor: "rgba(0,0,0,0)",
  plot_bgcolor: "rgba(0,0,0,0)",
  font: { color: "#cad4e0", family: "Inter, system-ui" },
  margin: { l: 60, r: 24, t: 24, b: 60 },
};

function buildMeaningSpaceHtml(
  shift: ShiftResponse | undefined,
  reduce: Reduce2DResponse | undefined,
  colorMap: Record<string, string>,
): string | null {
  if (!shift && !reduce) return null;
  type Pt = { x: number; y: number; label: string; symbol: string; kind: "descriptor" | "centroid" };
  const pts: Pt[] = [];
  if (shift) {
    for (const a of shift.arrows) pts.push({ x: a.x0, y: a.y0, label: a.descriptor, symbol: a.symbol, kind: "descriptor" });
    for (const c of shift.centroids) pts.push({ x: c.x, y: c.y, label: c.symbol, symbol: c.symbol, kind: "centroid" });
  } else if (reduce) {
    for (const p of reduce.points) pts.push({ x: p.x, y: p.y, label: p.label, symbol: p.symbol, kind: p.kind as any });
  }
  const groups = new Map<string, Pt[]>();
  for (const p of pts) {
    const arr = groups.get(p.symbol) ?? [];
    arr.push(p);
    groups.set(p.symbol, arr);
  }
  const traces: any[] = [];
  for (const [sym, items] of groups.entries()) {
    const color = colorMap[sym] ?? "#88c0d0";
    const descs = items.filter((i) => i.kind === "descriptor");
    const cents = items.filter((i) => i.kind === "centroid");
    if (descs.length)
      traces.push({
        x: descs.map((p) => p.x), y: descs.map((p) => p.y), text: descs.map((p) => p.label),
        type: "scatter", mode: "markers", name: sym,
        marker: { color, size: 9, opacity: 0.88 },
        hovertemplate: `<b>${sym}</b> · %{text}<extra></extra>`,
      });
    if (cents.length)
      traces.push({
        x: cents.map((p) => p.x), y: cents.map((p) => p.y), text: cents.map((p) => p.label),
        type: "scatter", mode: "markers+text",
        marker: { color, size: 18, symbol: "star", line: { color: "white", width: 1.5 } },
        textposition: "top center", showlegend: false, hoverinfo: "skip",
      });
  }
  const annotations: any[] = [];
  if (shift) {
    for (const a of shift.arrows) {
      const d = Math.hypot(a.x1 - a.x0, a.y1 - a.y0);
      if (d < 1e-3) continue;
      annotations.push({
        x: a.x1, y: a.y1, ax: a.x0, ay: a.y0,
        xref: "x", yref: "y", axref: "x", ayref: "y",
        showarrow: true, arrowhead: 2, arrowsize: 0.9, arrowwidth: 1,
        arrowcolor: colorMap[a.symbol] ?? "#888", opacity: 0.55, standoff: 3,
      });
    }
  }
  return plotlyShell("Meaning Space", traces, {
    ...PLOTLY_DARK_LAYOUT,
    autosize: true,
    xaxis: { showgrid: true, gridcolor: "rgba(255,255,255,0.04)", zeroline: false, showticklabels: false },
    yaxis: { showgrid: true, gridcolor: "rgba(255,255,255,0.04)", zeroline: false, showticklabels: false },
    legend: { orientation: "h", y: -0.05 },
    annotations,
  });
}

function buildAmbiguityHtml(a: AmbiguityResponse, colorMap: Record<string, string>): string {
  const syms = a.rows.map((r) => r.symbol);
  const traces = [
    {
      type: "bar", name: "dispersion", x: syms, y: a.rows.map((r) => r.dispersion),
      marker: { color: syms.map((s) => colorMap[s] ?? "#88c0d0") },
    },
    { type: "bar", name: "leakage", x: syms, y: a.rows.map((r) => r.leakage), marker: { color: "#d08770" } },
    { type: "bar", name: "entropy", x: syms, y: a.rows.map((r) => r.entropy), marker: { color: "#c2a6fe" } },
  ];
  return plotlyShell("Ambiguity", traces, {
    ...PLOTLY_DARK_LAYOUT,
    barmode: "group",
    autosize: true,
    xaxis: { tickangle: -25 },
    yaxis: { title: "value" },
    legend: { orientation: "h", y: -0.15 },
  });
}

function buildAttentionHtml(
  perSymbol: { symbol: string; data: AttentionResponse }[],
  colorMap: Record<string, string>,
): string {
  if (!perSymbol.length) return html("Attention", "<p>No data — open the Attention panel for each symbol first.</p>");
  const traces = perSymbol.map((p, i) => {
    const pairs = p.data.descriptors.map((d, j) => ({ d, w: p.data.weights[j] })).sort((x, y) => y.w - x.w);
    return {
      type: "bar", orientation: "h",
      x: pairs.map((q) => q.w),
      y: pairs.map((q) => q.d),
      name: p.symbol,
      visible: i === 0 ? true : "legendonly",
      marker: { color: colorMap[p.symbol] ?? "#88c0d0" },
    };
  });
  return plotlyShell("Descriptor attention", traces, {
    ...PLOTLY_DARK_LAYOUT,
    autosize: true,
    showlegend: true,
    legend: { orientation: "h", y: -0.1 },
    xaxis: { title: "attention weight" },
    yaxis: { autorange: "reversed" },
  });
}

function buildSimilarityWithinHtml(
  perSymbol: { symbol: string; data: SimilarityResponse }[],
): string {
  if (!perSymbol.length) return html("Similarity (within-symbol)", "<p>No data — open the Within Δ heatmap for each symbol first.</p>");
  const traces = perSymbol.map((p, i) => {
    const n = p.data.descriptors.length;
    const masked = p.data.delta.map((row, ii) => row.map((v, jj) => (ii === jj ? null : v)));
    return {
      type: "heatmap", z: masked,
      x: p.data.descriptors, y: p.data.descriptors,
      colorscale: SPECTRAL,
      name: p.symbol,
      visible: i === 0 ? true : false,
      colorbar: { title: { text: "Δ" } },
      hovertemplate: "%{y} ↔ %{x}<br>Δ = %{z:.3f}<extra></extra>",
    };
  });
  // Dropdown to switch symbol
  const buttons = perSymbol.map((p, i) => ({
    method: "update", label: p.symbol,
    args: [{ visible: perSymbol.map((_, j) => j === i) }],
  }));
  return plotlyShell("Similarity (within-symbol)", traces, {
    ...PLOTLY_DARK_LAYOUT,
    autosize: true,
    xaxis: { tickangle: -55 },
    yaxis: { autorange: "reversed" },
    updatemenus: [{ buttons, x: 0, y: 1.12, xanchor: "left", yanchor: "top" }],
  });
}

function buildSimilarityBetweenHtml(data: SymbolSimilarityResponse): string {
  const n = data.symbols.length;
  const masked = data.delta.map((row, i) => row.map((v, j) => (i === j ? null : v)));
  return plotlyShell("Similarity (between-symbol)", [{
    type: "heatmap", z: masked, x: data.symbols, y: data.symbols,
    colorscale: SPECTRAL,
    colorbar: { title: { text: "Δ" } },
    hovertemplate: "%{y} ↔ %{x}<br>Δ = %{z:.3f}<extra></extra>",
  }], {
    ...PLOTLY_DARK_LAYOUT,
    autosize: true,
    yaxis: { autorange: "reversed" },
  });
}

function buildDeltaGraphHtml(d: DeltaGraphResponse): string {
  const nodes = d.nodes.map((n) => ({ id: n.id, color: n.color, symbol: n.symbol }));
  const links = d.edges.map((e) => ({
    source: e.source, target: e.target,
    weight: e.abs_delta,
    color: e.sign === "up" ? "rgba(95,207,196,0.55)" : "rgba(208,135,112,0.55)",
  }));
  return forceGraphShell("Δ-graph", { nodes, links }, { kind: "delta" });
}

function buildSubgraphHtml(d: SubgraphResponse): string {
  const nodes = d.nodes.map((n) => ({
    id: n.id, color: n.color, symbol: n.symbol,
    score: n.kind === "symbol" ? 5 : 1,
  }));
  const links = d.edges.map((e) => ({
    source: e.source, target: e.target,
    weight: e.weight,
  }));
  return forceGraphShell("Contextual subgraph", { nodes, links }, { kind: "subgraph" });
}

function buildTransformationsHtml(
  t: TransformationsResponse,
  colorMap: Record<string, string>,
): string {
  const migRows = t.migrations.map((m) => {
    const fromC = colorMap[m.from_archetype] ?? "#888";
    const toC = colorMap[m.to_archetype] ?? "#888";
    return `<tr>
      <td style="color:${fromC}; font-weight:500">${escapeHtml(m.descriptor)}</td>
      <td><span style="color:${fromC}; border:1px solid ${fromC}55; padding:2px 6px; border-radius:4px; font-size:11px">${escapeHtml(m.from_archetype)}</span> → <span style="color:${toC}; border:1px solid ${toC}; padding:2px 6px; border-radius:4px; font-size:11px">${escapeHtml(m.to_archetype)}</span></td>
      <td style="font-family:monospace; text-align:right; color:#9fadc1">+${m.score.toFixed(2)}</td>
    </tr>`;
  }).join("\n");

  const cards = t.identities.map((card) => {
    const hC = colorMap[card.symbol] ?? "#888";
    const emerged = new Set(card.emerged);
    const faded = new Set(card.faded);
    const li = (e: { descriptor: string; owner: string; score: number }, isFaded: boolean, isEmerged: boolean) => {
      const c = colorMap[e.owner] ?? "#cbd";
      const fade = isFaded ? "color:#4d5a6f; text-decoration:line-through" : `color:${c}`;
      const foreign = e.owner && e.owner !== card.symbol && !isFaded
        ? ` <span style="font-size:9px; border:1px solid ${c}55; color:${c}; padding:1px 4px; border-radius:3px; text-transform:uppercase; letter-spacing:0.5px">${escapeHtml(e.owner)}</span>`
        : "";
      const dot = isEmerged ? `<span style="display:inline-block;width:6px;height:6px;border-radius:50%;background:#3bbdb0;margin-right:4px"></span>` : "";
      return `<li style="display:flex;align-items:center;gap:4px;margin:2px 0;font-size:12px">${dot}<span style="${fade}">${escapeHtml(e.descriptor)}</span>${foreign}<span style="margin-left:auto;font-family:monospace;font-size:10px;color:#6e7e95">${e.score.toFixed(2)}</span></li>`;
    };
    return `<div style="border:1px solid ${hC}55; background:rgba(16,19,28,0.5); padding:10px 12px; border-radius:8px; flex:1 1 280px; min-width:0">
      <div style="color:${hC}; font-size:12px; font-weight:600; letter-spacing:1px; text-transform:uppercase; margin-bottom:8px">${escapeHtml(card.symbol)}</div>
      <div style="display:grid; grid-template-columns:1fr 1fr; gap:12px">
        <div>
          <div style="font-size:9px; color:#6e7e95; letter-spacing:1px; text-transform:uppercase; margin-bottom:4px">Originally</div>
          <ul style="list-style:none; padding:0; margin:0">${card.before.map((e) => li(e, faded.has(e.descriptor), false)).join("")}</ul>
        </div>
        <div>
          <div style="font-size:9px; color:#6e7e95; letter-spacing:1px; text-transform:uppercase; margin-bottom:4px">Under context</div>
          <ul style="list-style:none; padding:0; margin:0">${card.after.map((e) => li(e, false, emerged.has(e.descriptor))).join("")}</ul>
        </div>
      </div>
      ${(card.emerged.length || card.faded.length) ? `
        <div style="margin-top:8px;padding-top:6px;border-top:1px solid rgba(255,255,255,0.06);font-size:10px">
          ${card.emerged.length ? `<div style="color:#9fadc1"><span style="color:#5fcfc4">+ new:</span> ${card.emerged.map(escapeHtml).join(", ")}</div>` : ""}
          ${card.faded.length ? `<div style="color:#6e7e95"><span style="color:#d08770">− faded:</span> ${card.faded.map(escapeHtml).join(", ")}</div>` : ""}
        </div>` : ""}
    </div>`;
  }).join("\n");

  return html("Contextual transformations", `
<h1>Contextual transformations</h1>
<div class="sub">who switched archetypes, and what each archetype looks like now</div>

<h2 style="font-weight:400; font-size:0.95rem; margin-top:24px; color:#9fadc1; text-transform:uppercase; letter-spacing:1px">Who switched archetypes</h2>
${t.migrations.length === 0
  ? `<p style="color:#6e7e95; font-size:13px">No archetype switches — every descriptor's nearest archetype stayed the same.</p>`
  : `<table style="border-collapse:collapse; width:100%; font-size:12px">
       <thead><tr style="color:#6e7e95; text-align:left"><th style="padding:6px">descriptor</th><th style="padding:6px">migration</th><th style="padding:6px; text-align:right">score</th></tr></thead>
       <tbody>${migRows}</tbody>
     </table>`}

<h2 style="font-weight:400; font-size:0.95rem; margin-top:32px; color:#9fadc1; text-transform:uppercase; letter-spacing:1px">What each archetype looks like now</h2>
<div style="display:flex; flex-wrap:wrap; gap:12px">${cards}</div>
`);
}

// ─────────────────── manifest + README ───────────────────

function buildManifest(state: SidebarState, present: Record<string, boolean>): string {
  const m = {
    exported_at: new Date().toISOString(),
    version: 1,
    space_id: state.spaceId,
    preset: state.presetName,
    symbol_map_json: state.symbolMapJson,
    embedder: {
      backend: state.embedderBackend,
      model: state.embedderModel || null,
      pooling: state.embedderPooling,
      qwen_instruction: state.qwenInstruction || null,
      qwen_context_mode: state.qwenContextMode,
      qwen_global_context: state.qwenGlobalContext || null,
    },
    context: {
      sentence: state.contextSentence || null,
      symbol_weights: state.symbolWeights,
      selected_context_symbols: state.selectedContextSymbols,
      alchemist_mode: state.alchemistMode,
      alchemist_sentence_b: state.contextSentenceB || null,
      alchemist_blend: state.alchemistBlend,
      audio_active: state.audioActive,
      image_active: state.imageActive,
      image_description: state.imageDescription || null,
    },
    shift_params: {
      strategy: state.strategy,
      gate: state.gate,
      beta: state.beta,
      gamma: state.gamma,
      pool_type: state.poolType,
      pool_w: state.poolW,
      membership_alpha: state.membershipAlpha,
      shift_tau: state.shiftTau,
      within_symbol_softmax: state.withinSymbolSoftmax,
    },
    ranking_params: {
      tau: state.tau,
      alpha: state.alpha,
      lambda: state.lambda,
      use_ppr: state.usePPR,
      blind_spot: state.blindSpot,
      topk: state.topk,
    },
    descriptor_threshold: state.descriptorThreshold,
    figures_included: present,
  };
  return JSON.stringify(m, null, 2);
}

function buildReadme(state: SidebarState, present: Record<string, boolean>): string {
  return `# Delyrism session export

Exported **${new Date().toISOString()}**
Preset: **${state.presetName ?? "—"}** · Embedder: **${state.embedderBackend}**
Context: ${state.contextSentence ? `"${state.contextSentence}"` : "(see manifest)"}

## Layout

\`\`\`
manifest.json     — full sidebar state + version (the reproducibility record)
data/             — raw JSON responses from every panel that had data
figures/          — standalone interactive HTML, double-click to open
\`\`\`

The HTML figures load Plotly and force-graph from CDN, so they need an
internet connection on first open.  Everything else is self-contained.

## Figures present in this bundle

${Object.entries(present).map(([k, v]) => `- ${v ? "✅" : "❌"} ${k}`).join("\n")}

If a figure is missing, the corresponding panel hadn't loaded data in
the app when you exported.  Visit the panel in the Explorer and export
again to capture it.

## Reproducing

The \`manifest.json\` contains every input that produced this session.
Paste \`symbol_map_json\` into the Symbolic Structure card, the
\`context.sentence\` into the Context Prompt, set the matching embedder
backend, and press Build.  All the data files should match.
`;
}

// ─────────────────── public entrypoint ───────────────────

export async function exportSession(qc: QueryClient, state: SidebarState): Promise<Blob> {
  const zip = new JSZip();
  const colorMap = state.colorMap;

  // pull from cache
  const reduce = findCachedByPrefix<Reduce2DResponse>(qc, "reduce2d");
  const shift = findCachedByPrefix<ShiftResponse>(qc, "shift");
  const rankings = findCachedByPrefix<ProposeResponse>(qc, "propose");
  const ambiguity = findCachedByPrefix<AmbiguityResponse>(qc, "ambiguity");
  const deltaGraph = findCachedByPrefix<DeltaGraphResponse>(qc, "delta-graph");
  const subgraph = findCachedByPrefix<SubgraphResponse>(qc, "subgraph");
  const transformations = findCachedByPrefix<TransformationsResponse>(qc, "transformations");
  const simBetween = findCachedByPrefix<SymbolSimilarityResponse>(qc, "similarity-symbols");

  // similarity (within) and attention may have multiple entries — one per symbol
  const simWithinAll = findAllCachedByPrefix<SimilarityResponse>(qc, "similarity")
    .filter((e) => e.data && "symbol" in e.data && e.data.descriptors) // exclude the symbols-variant
    .map((e) => ({ symbol: e.data.symbol, data: e.data }));
  const attentionAll = findAllCachedByPrefix<AttentionResponse>(qc, "attention")
    .map((e) => ({ symbol: (e.key[2] as string) ?? (e.data as any).symbol ?? "?", data: e.data }));

  const present: Record<string, boolean> = {
    "meaning-space.html": !!(shift || reduce),
    "rankings.json": !!rankings,
    "ambiguity.html": !!ambiguity,
    "attention.html": attentionAll.length > 0,
    "similarity-within.html": simWithinAll.length > 0,
    "similarity-between.html": !!simBetween,
    "delta-graph.html": !!deltaGraph,
    "subgraph.html": !!subgraph,
    "transformations.html": !!transformations,
  };

  zip.file("manifest.json", buildManifest(state, present));
  zip.file("README.md", buildReadme(state, present));

  // ─── data ───
  const data = zip.folder("data")!;
  if (rankings) data.file("rankings.json", JSON.stringify(rankings, null, 2));
  if (ambiguity) data.file("ambiguity.json", JSON.stringify(ambiguity, null, 2));
  if (shift) data.file("shift-arrows.json", JSON.stringify(shift, null, 2));
  if (deltaGraph) data.file("delta-graph.json", JSON.stringify(deltaGraph, null, 2));
  if (subgraph) data.file("subgraph.json", JSON.stringify(subgraph, null, 2));
  if (transformations) data.file("transformations.json", JSON.stringify(transformations, null, 2));
  if (simBetween) data.file("similarity-between.json", JSON.stringify(simBetween, null, 2));
  for (const s of simWithinAll) {
    data.file(`similarity-within-${safeFilename(s.symbol)}.json`, JSON.stringify(s.data, null, 2));
  }
  for (const a of attentionAll) {
    data.file(`attention-${safeFilename(a.symbol)}.json`, JSON.stringify(a.data, null, 2));
  }

  // ─── figures ───
  const figs = zip.folder("figures")!;
  const ms = buildMeaningSpaceHtml(shift, reduce, colorMap);
  if (ms) figs.file("meaning-space.html", ms);
  if (ambiguity) figs.file("ambiguity.html", buildAmbiguityHtml(ambiguity, colorMap));
  if (attentionAll.length) figs.file("attention.html", buildAttentionHtml(attentionAll, colorMap));
  if (simWithinAll.length) figs.file("similarity-within.html", buildSimilarityWithinHtml(simWithinAll));
  if (simBetween) figs.file("similarity-between.html", buildSimilarityBetweenHtml(simBetween));
  if (deltaGraph) figs.file("delta-graph.html", buildDeltaGraphHtml(deltaGraph));
  if (subgraph) figs.file("subgraph.html", buildSubgraphHtml(subgraph));
  if (transformations) figs.file("transformations.html", buildTransformationsHtml(transformations, colorMap));

  return await zip.generateAsync({ type: "blob", compression: "DEFLATE", compressionOptions: { level: 6 } });
}

export function triggerDownload(blob: Blob, filename: string): void {
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

export function exportFilename(state: SidebarState): string {
  const preset = state.presetName ?? "session";
  return `delyrism-${safeFilename(preset)}-${fmtTs()}.zip`;
}
