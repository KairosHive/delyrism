"use client";
import * as React from "react";
import { useShallow } from "zustand/react/shallow";
import { api, apiUrl, type SessionDone, type SessionPoint, type SessionResult } from "@/lib/api";
import { useLive } from "@/lib/liveGraph";
import { useSidebar } from "@/lib/store";

/**
 * The result of a Live session (delyrism/session.py), read along one shared timeline: the
 * session strip at the top holds the phases, the events and the returns, and its playhead (drag
 * it, or ▶ replay) moves through every view at once — the path on the map, the archetype
 * stream, the recurrence matrix, the topology and the words.
 */

const mmss = (s: number) => `${Math.floor(Math.max(0, s) / 60)}:${String(Math.floor(Math.max(0, s) % 60)).padStart(2, "0")}`;
const T0 = [95, 207, 196], T1 = [194, 166, 254];           // the path's colour, first moment → last
const timeColor = (u: number) => `rgb(${T0.map((x, i) => Math.round(x + (T1[i] - x) * Math.min(1, Math.max(0, u)))).join(",")})`;

interface Ctx { r: SessionResult; t: number; setT: (t: number) => void; dur: number; colorOf: Record<string, string>;
                symbols: string[]; ownerOf: Record<string, string>; at: SessionPoint | null }
const Cursor = React.createContext<Ctx | null>(null);
const useCursor = () => React.useContext(Cursor)!;

function pointAt(path: SessionPoint[], t: number): SessionPoint | null {
  if (!path.length) return null;
  let lo = 0, hi = path.length - 1;
  while (lo < hi) { const mid = (lo + hi + 1) >> 1; if (path[mid].t <= t) lo = mid; else hi = mid - 1; }
  return path[lo];
}

// ============================================================================ panel
export function SessionPanel() {
  const done = useLive((s) => s.sessionResult);
  const close = useLive((s) => s.closeSessionResult);
  if (!done) return null;
  return <Panel key={done.id} done={done} close={close} />;
}

function Panel({ done, close }: { done: SessionDone; close: () => void }) {
  const r = done.result;
  const dur = Math.max(r.summary.duration, 1);
  const [t, setT] = React.useState(dur);
  const [playing, setPlaying] = React.useState(false);
  const colorOf = React.useMemo(() => Object.fromEntries(r.symbols.map((s) => [s.id, s.color])), [r]);
  const ownerOf = React.useMemo(() => Object.fromEntries(r.layout.map((d) => [d.id, d.symbol])), [r]);
  const path = r.trajectory?.path ?? [];
  const at = pointAt(path, t);

  // replay: the whole session in about 12 s (never faster than 1 s per second of session)
  React.useEffect(() => {
    if (!playing) return;
    let raf = 0, last = performance.now();
    const rate = Math.max(1, dur / 12);
    const step = (now: number) => {
      const dt = (now - last) / 1000; last = now;
      setT((x) => {
        const nx = x + dt * rate;
        if (nx >= dur) { setPlaying(false); return dur; }
        return nx;
      });
      raf = requestAnimationFrame(step);
    };
    raf = requestAnimationFrame(step);
    return () => cancelAnimationFrame(raf);
  }, [playing, dur]);
  const play = () => { if (t >= dur - 0.01) setT(0); setPlaying((p) => !p); };

  const ctx: Ctx = { r, t, setT: (x) => { setPlaying(false); setT(Math.min(dur, Math.max(0, x))); }, dur, colorOf,
                     symbols: r.symbols.map((s) => s.id), ownerOf, at };
  return (
    <Cursor.Provider value={ctx}>
      <div className="panel mt-4 overflow-hidden p-0" data-testid="session-panel">
        <Header done={done} close={close} />
        {r.empty ? (
          <p className="px-5 pb-6 text-sm text-ink-300">No query was active long enough to draw a path. Start a session, then type, speak, play or move.</p>
        ) : (
          <div className="space-y-6 px-5 pb-6">
            <Cards />
            <Strip playing={playing} play={play} />
            <div className="grid grid-cols-1 gap-6 xl:grid-cols-[minmax(0,1.35fr),minmax(0,1fr)]">
              <Block n="A" title="Trajectory" hint="where the query's attention sat on the map of the space">
                <HeroMap />
              </Block>
              <div className="space-y-6">
                <Block n="B" title="Territory" hint="how its time was shared between the archetypes, and what it never reached">
                  <Territory />
                </Block>
                <Block n="D" title="Returns" hint="each moment against every other: bright where two moments resemble each other">
                  <Recurrence />
                </Block>
              </div>
            </div>
            <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
              <Block n="C" title="Topology over time" hint="how scattered its attention was, and when it circled">
                <Topology />
              </Block>
              <Block n="E" title="Words" hint="everything typed, heard, spoken and described, by phase">
                <Words done={done} />
              </Block>
            </div>
            <div className="grid grid-cols-1 gap-6 lg:grid-cols-[minmax(0,1.4fr),minmax(0,1fr)]">
              <Block n="G" title="Session archetypes" hint="the words it visited, clustered into the archetypes it formed; colours show where each word came from">
                <Archetypes />
              </Block>
              <Block n="H" title="Fingerprint" hint="the session in one glyph, next to the earlier ones from this browser">
                <Fingerprint done={done} />
              </Block>
            </div>
          </div>
        )}
      </div>
    </Cursor.Provider>
  );
}

function Header({ done, close }: { done: SessionDone; close: () => void }) {
  const r = done.result;
  return (
    <div className="flex flex-wrap items-center gap-x-4 gap-y-2 border-b border-ink-700/50 bg-gradient-to-r from-accent-900/30 via-transparent to-[#c2a6fe]/10 px-5 py-4">
      <div>
        <div className="text-[10px] uppercase tracking-[0.3em] text-accent-300">Session</div>
        <div className="text-xl font-light text-ink-50">{r.summary.name || "untitled"}</div>
      </div>
      <div className="ml-auto flex items-center gap-2 text-xs">
        <a className="btn" href={apiUrl(`/realtime/session/${done.id}/raw`)} download>↓ raw data</a>
        <button className="btn" onClick={close} aria-label="close the session result">✕</button>
      </div>
      {done.saved && <div className="w-full font-mono text-[10px] text-ink-500">saved on this machine · {done.saved}</div>}
    </div>
  );
}

function Block({ n, title, hint, children }: { n: string; title: string; hint: string; children: React.ReactNode }) {
  return (
    <section className="space-y-3">
      <header className="flex items-baseline gap-2">
        <span className="flex h-5 w-5 shrink-0 items-center justify-center rounded-full border border-ink-600 text-[10px] font-semibold text-ink-300">{n}</span>
        <h3 className="shrink-0 text-sm font-medium text-ink-100">{title}</h3>
        <span className="truncate text-[11px] text-ink-500">{hint}</span>
      </header>
      {children}
    </section>
  );
}

// ============================================================================ overview
function Cards() {
  const { r, colorOf } = useCursor();
  const share = r.territory?.symbol_share ?? {};
  const lead = Object.entries(share).sort((a, b) => b[1] - a[1])[0];
  const nRet = r.phases?.returns.length ?? 0;
  const cards: [string, React.ReactNode, string][] = [
    ["duration", mmss(r.summary.duration), `${r.summary.snapshots} moments · ${r.summary.kinds.join(", ") || "no query"}`],
    ["reached", `${Math.round((r.territory?.reached ?? 0) * 100)}%`, "of the network, in the top 5"],
    ["leaned toward", <span key="l" style={{ color: lead ? colorOf[lead[0]] : undefined }}>{lead?.[0] ?? "—"}</span>, lead ? `${Math.round(lead[1] * 100)}% of the time` : ""],
    ["phases", String(r.phases?.phases.length ?? 0), `${nRet} return${nRet === 1 ? "" : "s"}`],
    ["travelled", `${(r.trajectory?.distance ?? 0).toFixed(1)}`, "map widths"],
  ];
  return (
    <div className="grid grid-cols-2 gap-2 pt-4 sm:grid-cols-3 lg:grid-cols-5">
      {cards.map(([k, v, sub]) => (
        <div key={k} className="rounded-lg border border-ink-700/60 bg-ink-900/50 px-3 py-2">
          <div className="text-[10px] uppercase tracking-[0.18em] text-ink-500">{k}</div>
          <div className="truncate text-2xl font-light leading-tight text-ink-50">{v}</div>
          <div className="truncate text-[11px] text-ink-400">{sub}</div>
        </div>
      ))}
    </div>
  );
}

const EVENT_ICON: Record<string, string> = { query: "⌕", heard: "◌", spoken: "♪", motion: "∿", context: "◆" };
const EVENT_LABEL: Record<string, string> = { query: "typed", heard: "heard", spoken: "spoken", motion: "movement", context: "context" };

function Strip({ playing, play }: { playing: boolean; play: () => void }) {
  const { r, t, setT, dur, colorOf, at } = useCursor();
  const ref = React.useRef<SVGSVGElement>(null);
  const W = 1000, X = (s: number) => (s / dur) * W;
  const fromEvent = (e: React.PointerEvent) => {
    const b = ref.current!.getBoundingClientRect();
    setT(((e.clientX - b.left) / b.width) * dur);
  };
  const phases = r.phases?.phases ?? [];
  const words = r.words ?? [];
  return (
    <div className="rounded-xl border border-ink-700/60 bg-ink-950/50 p-3">
      <div className="mb-2 flex flex-wrap items-center gap-3 text-xs">
        <button className="btn min-w-[88px]" onClick={play}>{playing ? "❚❚ pause" : "▶ replay"}</button>
        <span className="font-mono text-ink-100">{mmss(t)}</span><span className="-ml-2 font-mono text-ink-500">/ {mmss(dur)}</span>
        {at && (
          <span className="flex min-w-0 flex-1 items-center gap-2 text-ink-200">
            <span className="h-2.5 w-2.5 shrink-0 rounded-full" style={{ background: colorOf[r.symbols[at.s ?? 0]?.id], boxShadow: `0 0 8px ${colorOf[r.symbols[at.s ?? 0]?.id]}` }} />
            <span className="truncate">{(at.top ?? []).join(" · ")}</span>
            {at.q && <span className="hidden truncate text-ink-500 md:inline">← “{at.q}”</span>}
          </span>
        )}
      </div>
      <svg ref={ref} viewBox={`0 0 ${W} 74`} className="h-20 w-full cursor-ew-resize touch-none select-none" preserveAspectRatio="none"
           onPointerDown={(e) => { (e.currentTarget as Element).setPointerCapture?.(e.pointerId); fromEvent(e); }}
           onPointerMove={(e) => { if (e.buttons) fromEvent(e); }}>
        {phases.map((p, i) => {
          const x0 = X(p.t0), x1 = i + 1 < phases.length ? X(phases[i + 1].t0) : W;
          return <g key={i}>
            <rect x={x0} y={18} width={Math.max(1, x1 - x0 - 3)} height={26} rx={6} fill={colorOf[p.symbol]} fillOpacity={0.88} />
            <text x={x0 + 8} y={35} fontSize={11} fill="#070912" fontWeight={600}>{x1 - x0 > 70 ? p.words.slice(0, x1 - x0 > 170 ? 2 : 1).join(" · ") : ""}</text>
            <title>{`${mmss(p.t0)}–${mmss(p.t1)} · ${p.symbol} · ${p.words.join(", ")}`}</title>
          </g>;
        })}
        {(r.phases?.returns ?? []).map((x, i) => {
          const a = X(x.to), b = X(x.t), h = Math.min(14, 5 + (b - a) / 30);
          return <path key={i} d={`M${a},16 Q${(a + b) / 2},${16 - h * 1.7} ${b},16`} fill="none" stroke="#ebcb8b" strokeOpacity={0.8} strokeWidth={1.3} strokeDasharray="4 3">
            <title>{`${mmss(x.t)} came back to ${mmss(x.to)}`}</title></path>;
        })}
        {words.map((w, i) => (
          <text key={i} x={X(w.t)} y={61} fontSize={11} textAnchor="middle" fill={w.type === "context" ? "#f5f7fa" : "#9fadc1"} fillOpacity={0.85}>
            {EVENT_ICON[w.type] ?? "·"}<title>{`${mmss(w.t)} ${EVENT_LABEL[w.type] ?? w.type}: ${w.text}`}</title></text>
        ))}
        <line x1={0} x2={W} y1={69} y2={69} stroke="#2a3142" strokeWidth={2} />
        <rect x={0} y={68} width={X(t)} height={2} fill="#5fcfc4" />
        <line x1={X(t)} x2={X(t)} y1={8} y2={72} stroke="#f5f7fa" strokeWidth={1.5} />
        <circle cx={X(t)} cy={69} r={4.5} fill="#f5f7fa" />
      </svg>
      <div className="mt-1 flex flex-wrap gap-x-3 text-[10px] text-ink-500">
        <span>bands · phases</span><span><span className="text-warning">⌒</span> returns</span>
        {Object.entries(EVENT_ICON).map(([k, v]) => <span key={k}>{v} {EVENT_LABEL[k]}</span>)}
        <span className="ml-auto">drag to move through the session</span>
      </div>
    </div>
  );
}

// ============================================================================ A · map
function smoothSegments(pts: [number, number][]): string[] {
  // one Catmull-Rom piece per segment, so each can carry its own colour
  const out: string[] = [];
  for (let i = 0; i + 1 < pts.length; i++) {
    const p0 = pts[Math.max(0, i - 1)], p1 = pts[i], p2 = pts[i + 1], p3 = pts[Math.min(pts.length - 1, i + 2)];
    const c1 = [p1[0] + (p2[0] - p0[0]) / 6, p1[1] + (p2[1] - p0[1]) / 6];
    const c2 = [p2[0] - (p3[0] - p1[0]) / 6, p2[1] - (p3[1] - p1[1]) / 6];
    out.push(`M${p1[0].toFixed(2)},${p1[1].toFixed(2)} C${c1[0].toFixed(2)},${c1[1].toFixed(2)} ${c2[0].toFixed(2)},${c2[1].toFixed(2)} ${p2[0].toFixed(2)},${p2[1].toFixed(2)}`);
  }
  return out;
}

const mapXY = (x: number, y: number): [number, number] => [5 + x * 90, 95 - y * 90];

function HeroMap() {
  const { r, t, dur, at } = useCursor();
  const tr = r.trajectory!;
  const visits = r.territory?.visits ?? {};
  const path = React.useMemo(() => tr.path.filter((_, i, a) => a.length <= 500 || i % Math.ceil(a.length / 500) === 0), [tr]);
  const segs = React.useMemo(() => smoothSegments(path.map((p) => mapXY(p.x, p.y))), [path]);
  const regions = React.useMemo(() => {
    const out = r.symbols.map((s) => {
      const pts = r.layout.filter((d) => d.symbol === s.id);
      const x = pts.reduce((a, d) => a + d.x, 0) / Math.max(1, pts.length), y = pts.reduce((a, d) => a + d.y, 0) / Math.max(1, pts.length);
      return { id: s.id, color: s.color, xy: mapXY(x, y) as [number, number] };
    });
    // keep the archetype names apart: push a label down while it sits on one already placed
    const clash = (a: (typeof out)[number], b: (typeof out)[number]) =>
      Math.abs(a.xy[0] - b.xy[0]) < 3 + (a.id.length + b.id.length) * 1.1 && Math.abs(a.xy[1] - b.xy[1]) < 4.5;
    for (let i = 0; i < out.length; i++)
      for (let guard = 0; guard < 12 && out.slice(0, i).some((o) => clash(o, out[i])); guard++)
        out[i].xy = [out[i].xy[0], Math.min(97, out[i].xy[1] + 4.5)];
    return out;
  }, [r]);
  const dwells = React.useMemo(() => [...tr.dwells].sort((a, b) => (b.t1 - b.t0) - (a.t1 - a.t0)).slice(0, 5), [tr]);
  const now = at ? mapXY(at.x, at.y) : null;
  return (
    <div>
      <div className="relative">
        <svg viewBox="0 0 100 100" className="w-full rounded-xl bg-[radial-gradient(ellipse_at_center,_#10131c_0%,_#070912_85%)]" style={{ aspectRatio: "1 / 1" }}>
          <defs>
            <filter id="sess-blur" x="-50%" y="-50%" width="200%" height="200%"><feGaussianBlur stdDeviation="2.6" /></filter>
            <filter id="sess-glow" x="-100%" y="-100%" width="300%" height="300%"><feGaussianBlur stdDeviation="1.2" /></filter>
          </defs>
          {/* where attention spent its time: a soft glow per visited word, in its archetype's colour */}
          <g filter="url(#sess-blur)">
            {r.layout.map((d) => {
              const v = visits[d.id]; if (!v) return null;
              const [x, y] = mapXY(d.x, d.y);
              return <circle key={d.id} cx={x} cy={y} r={2.5 + Math.sqrt(v) * 10} fill={d.color} fillOpacity={0.3} />;
            })}
          </g>
          {regions.map((g) => (
            <text key={g.id} x={g.xy[0]} y={g.xy[1]} fontSize={3.6} textAnchor="middle" fill={g.color} fillOpacity={0.45}
                  letterSpacing={0.7} fontWeight={600}>{g.id.toUpperCase()}</text>
          ))}
          {r.layout.map((d) => {
            const [x, y] = mapXY(d.x, d.y);
            const v = visits[d.id] ?? 0;
            return <circle key={d.id} cx={x} cy={y} r={v ? 0.7 + Math.sqrt(v) * 1.6 : 0.45} fill={d.color} fillOpacity={v ? 0.95 : 0.22}>
              <title>{`${d.id} · ${d.symbol}${v ? ` · in the top 5 for ${Math.round(v * 100)}% of the session` : " · never reached"}`}</title></circle>;
          })}
          {/* the path: the part already travelled is bright, the rest a faint ghost */}
          {segs.map((d, i) => {
            const pt = path[i + 1].t, passed = pt <= t;
            return <path key={i} d={d} fill="none" stroke={timeColor(pt / dur)} strokeWidth={passed ? 0.8 : 0.35}
                         strokeOpacity={passed ? 0.92 : 0.12} strokeLinecap="round" />;
          })}
          {tr.context_path.map((c, i) => {
            const [x, y] = mapXY(c.x, c.y);
            return <g key={`c${i}`} opacity={c.t <= t ? 1 : 0.35}>
              <rect x={x - 1.4} y={y - 1.4} width={2.8} height={2.8} transform={`rotate(45 ${x} ${y})`} fill="#070912" stroke="#f5f7fa" strokeWidth={0.4} />
              <text x={x} y={y + 0.8} fontSize={2.1} textAnchor="middle" fill="#f5f7fa">{i + 1}</text>
              <title>{`context ${i + 1} at ${mmss(c.t)}: ${c.text}`}</title>
            </g>;
          })}
          {dwells.map((d, i) => {
            const [x, y] = mapXY(d.x, d.y);
            return <g key={`d${i}`} opacity={d.t0 <= t ? 1 : 0.3}>
              <circle cx={x} cy={y} r={1.8 + Math.min(4, (d.t1 - d.t0) / 12)} fill="none" stroke="#f5f7fa" strokeOpacity={0.6} strokeWidth={0.28} strokeDasharray="0.8 0.6" />
              <text x={x + 2.6} y={y - 1.8} fontSize={2.4} fill="#f5f7fa" fillOpacity={0.85}>{d.words[0]}</text>
              <title>{`lingered ${mmss(d.t0)}–${mmss(d.t1)}: ${d.words.join(", ")}`}</title>
            </g>;
          })}
          {path.length > 0 && (() => {
            const [x, y] = mapXY(path[0].x, path[0].y);
            return <g><circle cx={x} cy={y} r={1.1} fill={timeColor(0)} /><text x={x} y={y + 3.8} fontSize={2.2} textAnchor="middle" fill={timeColor(0)}>start</text></g>;
          })()}
          {now && <g>
            <circle cx={now[0]} cy={now[1]} r={3.4} fill={timeColor(t / dur)} fillOpacity={0.4} filter="url(#sess-glow)" />
            <circle cx={now[0]} cy={now[1]} r={1.3} fill="#f5f7fa" />
          </g>}
        </svg>
        {at && now && (
          <div className="pointer-events-none absolute max-w-[46%] rounded-md border border-ink-700/70 bg-ink-950/85 px-2 py-1 text-[11px] leading-snug text-ink-100 shadow-soft backdrop-blur"
               style={{ left: `${Math.min(68, now[0] + 2)}%`, top: `${Math.min(88, Math.max(2, now[1] - 9))}%` }}>
            <div className="font-mono text-[10px] text-ink-400">{mmss(at.t)}</div>
            {(at.top ?? []).join(" · ")}
          </div>
        )}
      </div>
      <div className="mt-2 flex flex-wrap items-center gap-x-4 gap-y-1 text-[10px] text-ink-500">
        <span className="flex items-center gap-1.5"><span className="h-1 w-8 rounded-full" style={{ background: `linear-gradient(90deg, ${timeColor(0)}, ${timeColor(1)})` }} /> path, first moment → last</span>
        <span>glow · where attention spent its time</span>
        <span>◇ contexts, in order</span>
        <span>◌ where it lingered</span>
      </div>
    </div>
  );
}

// ============================================================================ B · territory
function Territory() {
  const { r, t, dur, colorOf, symbols } = useCursor();
  const terr = r.territory!;
  const path = r.trajectory?.path ?? [];
  const [showNever, setShowNever] = React.useState(false);
  // which archetype led, over time: shares of the leading archetype in a sliding tenth of the session
  const stream = React.useMemo(() => {
    const n = 80, win = Math.max(dur / 10, 2);
    return Array.from({ length: n + 1 }, (_, k) => {
      const tc = (k / n) * dur;
      const near = path.filter((p) => Math.abs(p.t - tc) <= win / 2);
      const c = symbols.map((_, j) => near.filter((p) => p.s === j).length);
      const tot = c.reduce((a, b) => a + b, 0) || 1;
      return c.map((x) => x / tot);
    });
  }, [path, dur, symbols]);
  const W = 300, H = 64;
  const areas = symbols.map((_, j) => {
    const top = stream.map((v, k) => [(k / (stream.length - 1)) * W, H - v.slice(0, j + 1).reduce((a, b) => a + b, 0) * H]);
    const bot = stream.map((v, k) => [(k / (stream.length - 1)) * W, H - v.slice(0, j).reduce((a, b) => a + b, 0) * H]).reverse();
    return [...top, ...bot].map(([x, y], i) => `${i ? "L" : "M"}${x.toFixed(1)},${y.toFixed(1)}`).join(" ") + "Z";
  });
  const ownerOfNever = (w: string) => r.layout.find((d) => d.id === w)?.symbol ?? "";
  return (
    <div className="space-y-3">
      <div className="space-y-1.5">
        {symbols.map((s) => {
          const v = terr.symbol_share[s] ?? 0;
          return (
            <div key={s} className="flex items-center gap-2 text-xs">
              <span className="w-28 truncate" style={{ color: colorOf[s] }}>{s}</span>
              <div className="h-2.5 flex-1 overflow-hidden rounded-full bg-ink-800/80">
                <div className="h-full rounded-full" style={{ width: `${v * 100}%`, background: colorOf[s], boxShadow: v ? `0 0 10px ${colorOf[s]}66` : undefined }} />
              </div>
              <span className="w-9 text-right font-mono text-ink-400">{Math.round(v * 100)}%</span>
            </div>
          );
        })}
      </div>
      <svg viewBox={`0 0 ${W} ${H + 12}`} className="w-full overflow-visible">
        {areas.map((d, j) => <path key={j} d={d} fill={colorOf[symbols[j]]} fillOpacity={0.8} />)}
        <line x1={(t / dur) * W} x2={(t / dur) * W} y1={0} y2={H} stroke="#f5f7fa" strokeWidth={1.2} />
        <text x={0} y={H + 10} fontSize={8} fill="#6e7e95">0:00</text>
        <text x={W} y={H + 10} fontSize={8} fill="#6e7e95" textAnchor="end">{mmss(dur)}</text>
        <text x={W / 2} y={H + 10} fontSize={8} fill="#6e7e95" textAnchor="middle">which archetype led, over time</text>
      </svg>
      {terr.never.length > 0 && (
        <div className="text-xs">
          <button className="text-ink-300 hover:text-ink-100" onClick={() => setShowNever(!showNever)}>
            {showNever ? "▾" : "▸"} never reached · {terr.never.length} of {r.layout.length} words
          </button>
          {showNever && (
            <div className="mt-2 flex flex-wrap gap-1">
              {terr.never.map((w) => (
                <span key={w} className="rounded-full border px-2 py-0.5 text-[11px] text-ink-300" style={{ borderColor: `${colorOf[ownerOfNever(w)] ?? "#4d5a6f"}66` }}>{w}</span>
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  );
}

// ============================================================================ D · recurrence
function Recurrence() {
  const { r, t, dur, setT } = useCursor();
  const ph = r.phases!;
  const ref = React.useRef<HTMLCanvasElement>(null);
  const tb = ph.recurrence.t;
  React.useEffect(() => {
    const c = ref.current; if (!c) return;
    const M = ph.recurrence.M, n = M.length; if (!n) return;
    c.width = n; c.height = n;
    const ctx = c.getContext("2d")!; const img = ctx.createImageData(n, n);
    const stops = [[7, 9, 18], [27, 103, 95], [95, 207, 196], [194, 166, 254], [245, 247, 250]];
    for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) {
      const u = Math.max(0, Math.min(1, (M[i][j] + 1) / 2)) ** 1.6 * (stops.length - 1);
      const a = Math.floor(Math.min(u, stops.length - 2)), f = u - a;
      const col = stops[a].map((x, k) => x + (stops[a + 1][k] - x) * f);
      const k = ((n - 1 - i) * n + j) * 4;
      img.data[k] = col[0]; img.data[k + 1] = col[1]; img.data[k + 2] = col[2]; img.data[k + 3] = 255;
    }
    ctx.putImageData(img, 0, 0);
  }, [ph]);
  const u = (s: number) => (tb.length > 1 ? Math.min(1, Math.max(0, (s - tb[0]) / Math.max(1e-9, tb[tb.length - 1] - tb[0]))) : 0);
  return (
    <div className="flex items-start gap-4">
      <div className="w-44 shrink-0">
        <div className="relative cursor-crosshair" onPointerDown={(e) => { const b = e.currentTarget.getBoundingClientRect(); setT(((e.clientX - b.left) / b.width) * dur); }}>
          <canvas ref={ref} className="aspect-square w-full rounded-lg" style={{ imageRendering: "pixelated" }} />
          <svg viewBox="0 0 100 100" className="pointer-events-none absolute inset-0 h-full w-full">
            <line x1={u(t) * 100} x2={u(t) * 100} y1={0} y2={100} stroke="#f5f7fa" strokeOpacity={0.7} strokeWidth={0.6} />
            <line y1={100 - u(t) * 100} y2={100 - u(t) * 100} x1={0} x2={100} stroke="#f5f7fa" strokeOpacity={0.7} strokeWidth={0.6} />
            {ph.returns.map((x, i) => <circle key={i} cx={u(x.to) * 100} cy={100 - u(x.t) * 100} r={3.2} fill="none" stroke="#ebcb8b" strokeWidth={1} />)}
          </svg>
        </div>
        <div className="mt-1 flex justify-between text-[9px] text-ink-500"><span>0:00</span><span>time →</span><span>{mmss(dur)}</span></div>
      </div>
      <div className="min-w-0 space-y-2 text-xs text-ink-300">
        <p className="text-[11px] leading-relaxed text-ink-400">The diagonal is each moment against itself. A bright square is a phase that held; a bright spot away from the diagonal (<span className="text-warning">○</span>) is a return.</p>
        {ph.returns.length ? ph.returns.map((x, i) => (
          <button key={i} className="block text-left hover:text-ink-100" onClick={() => setT(x.t)}>
            <span className="text-warning">○</span> at <span className="font-mono">{mmss(x.t)}</span> it came back to <span className="font-mono">{mmss(x.to)}</span>
          </button>
        )) : <div className="text-ink-500">No returns: the session kept moving on.</div>}
      </div>
    </div>
  );
}

// ============================================================================ C · topology
function StepChart({ label, color, values, max, t, dur, note }: { label: string; color: string; values: { t: number; v: number }[];
                                                                   max: number; t: number; dur: number; note: string }) {
  const W = 300, H = 46;
  const X = (s: number) => (s / dur) * W, Y = (v: number) => H - 4 - (v / Math.max(1, max)) * (H - 12);
  const d = values.map((p, i) => (i ? `H${X(p.t).toFixed(1)}V${Y(p.v).toFixed(1)}` : `M${X(p.t).toFixed(1)},${Y(p.v).toFixed(1)}`)).join(" ");
  const cur = [...values].reverse().find((p) => p.t <= t) ?? values[0];
  return (
    <div>
      <div className="flex items-baseline gap-2 text-[11px]">
        <span className="shrink-0" style={{ color }}>{label}</span>
        <span className="truncate text-ink-500">{note}</span>
        <span className="ml-auto shrink-0 text-ink-300">{cur ? `${cur.v} at ${mmss(t)}` : ""}</span>
      </div>
      <svg viewBox={`0 0 ${W} ${H}`} className="w-full">
        <line x1={0} x2={W} y1={H - 4} y2={H - 4} stroke="#2a3142" />
        {values.length > 1 && <path d={`${d} H${W}`} fill="none" stroke={color} strokeWidth={1.6} />}
        <line x1={X(t)} x2={X(t)} y1={0} y2={H} stroke="#f5f7fa" strokeOpacity={0.6} />
      </svg>
    </div>
  );
}

function Topology() {
  const { r, t, dur, setT } = useCursor();
  const tp = r.topology!;
  const mid = tp.windows.map((w) => ({ t: (w.t0 + w.t1) / 2, groups: w.groups, loops: w.loops }));
  const gmax = Math.max(1, ...mid.map((m) => m.groups)), lmax = Math.max(1, ...mid.map((m) => m.loops));
  const ge = tp.graph_events;
  const bmax = Math.max(1, ...ge.map((g) => Math.max(g.born, g.died)));
  if (mid.length < 2) return <div className="text-xs text-ink-500">The session is too short to follow its shape over time.</div>;
  return (
    <div className="space-y-3">
      <StepChart label="separate groups of words in its attention" color="#5fcfc4" values={mid.map((m) => ({ t: m.t, v: m.groups }))} max={gmax} t={t} dur={dur}
                 note="1 = one focus · more = scattered" />
      <StepChart label="loops it closed" color="#c2a6fe" values={mid.map((m) => ({ t: m.t, v: m.loops }))} max={lmax} t={t} dur={dur}
                 note="circling among related words" />
      {ge.length > 0 && (
        <div>
          <div className="text-[11px] text-ink-300">Δ-graph at each context change <span className="text-ink-500">· edges <span className="text-accent-300">born ↑</span> and <span className="text-warmth">dying ↓</span></span></div>
          <svg viewBox="0 0 300 40" className="w-full">
            <line x1={0} x2={300} y1={20} y2={20} stroke="#2a3142" />
            {ge.map((g, i) => {
              const x = (g.t / dur) * 300;
              return <g key={i} className="cursor-pointer" onClick={() => setT(g.t)}>
                <rect x={x - 3} y={20 - (g.born / bmax) * 18} width={6} height={(g.born / bmax) * 18} rx={1.5} fill="#5fcfc4" />
                <rect x={x - 3} y={20} width={6} height={(g.died / bmax) * 18} rx={1.5} fill="#d08770" />
                <title>{`${mmss(g.t)}: +${g.born} / −${g.died} edges${g.top_born.length ? ` · new: ${g.top_born.join(", ")}` : ""}`}</title>
              </g>;
            })}
            <line x1={(t / dur) * 300} x2={(t / dur) * 300} y1={0} y2={40} stroke="#f5f7fa" strokeOpacity={0.6} />
          </svg>
        </div>
      )}
      {tp.events.length > 0 && (
        <ul className="max-h-24 space-y-0.5 overflow-auto text-[11px] text-ink-400">
          {tp.events.map((e, i) => (
            <li key={i}><button className="hover:text-ink-100" onClick={() => setT(e.t)}><span className="font-mono text-ink-500">{mmss(e.t)}</span> {e.text}</button></li>
          ))}
        </ul>
      )}
    </div>
  );
}

// ============================================================================ E · words
function Words({ done }: { done: SessionDone }) {
  const { r, t, setT, colorOf } = useCursor();
  const words = r.words ?? [];
  const phases = r.phases?.phases ?? [];
  const [story, setStory] = React.useState<string | null>(null);
  const [busy, setBusy] = React.useState(false);
  const [err, setErr] = React.useState<string | null>(null);
  const current = [...words].reverse().find((w) => w.t <= t);
  async function weave() {
    setBusy(true); setErr(null);
    try { setStory((await api.post<{ story: string }>(`/realtime/session/${done.id}/story`, {})).story); }
    catch (e: any) { setErr(String(e?.message ?? e)); }
    finally { setBusy(false); }
  }
  const groups = phases.length
    ? phases.map((p, i) => ({ p, items: words.filter((w) => (i === 0 || w.t >= p.t0) && (i + 1 >= phases.length || w.t < phases[i + 1].t0)) }))
    : [{ p: null as null | (typeof phases)[number], items: words }];
  return (
    <div className="space-y-3">
      <div className="max-h-64 space-y-2 overflow-auto pr-1">
        {words.length === 0 && <div className="text-xs text-ink-500">No words in this session.</div>}
        {groups.map((g, gi) => g.items.length > 0 && (
          <div key={gi} className="border-l-2 pl-3" style={{ borderColor: g.p ? colorOf[g.p.symbol] : "#2a3142" }}>
            {g.p && <div className="mb-0.5 text-[10px] uppercase tracking-[0.16em]" style={{ color: colorOf[g.p.symbol] }}>{g.p.symbol} · {g.p.words.slice(0, 3).join(", ")}</div>}
            {g.items.map((w, i) => (
              <button key={i} onClick={() => setT(w.t)}
                      className={`flex w-full items-baseline gap-2 rounded px-1 text-left text-xs transition ${w === current ? "bg-ink-800/80 text-ink-50" : "text-ink-300 hover:text-ink-100"}`}>
                <span className="w-9 shrink-0 font-mono text-[10px] text-ink-500">{mmss(w.t)}</span>
                <span className="w-3 shrink-0 text-ink-500" title={EVENT_LABEL[w.type]}>{EVENT_ICON[w.type] ?? "·"}</span>
                <span className={w.type === "context" ? "italic" : ""}>{w.text}</span>
              </button>
            ))}
          </div>
        ))}
      </div>
      <button className="btn text-xs" onClick={weave} disabled={busy}>{busy ? "weaving…" : "✶ weave a text from the phases"}</button>
      {err && <div className="text-xs text-danger">{err}</div>}
      {story && <blockquote className="whitespace-pre-wrap rounded-lg border-l-2 border-[#c2a6fe] bg-ink-950/60 p-4 text-[14px] italic leading-relaxed text-ink-100">{story}</blockquote>}
    </div>
  );
}

// ============================================================================ G · archetypes
function Archetypes() {
  const { r, colorOf, ownerOf } = useCursor();
  const a = r.archetypes ?? {};
  const set = useSidebar((s) => s.set);
  const [used, setUsed] = React.useState(false);
  const names = Object.keys(a);
  if (!names.length) return <div className="text-xs text-ink-500">Too few words were visited to form archetypes.</div>;
  const json = JSON.stringify(a, null, 2);
  return (
    <div className="space-y-3">
      <div className="grid grid-cols-1 gap-2 sm:grid-cols-2">
        {names.map((n) => {
          const counts: Record<string, number> = {};
          a[n].forEach((w) => { counts[ownerOf[w]] = (counts[ownerOf[w]] ?? 0) + 1; });
          const dom = Object.entries(counts).sort((x, y) => y[1] - x[1])[0]?.[0];
          return (
            <div key={n} className="rounded-lg border border-ink-700/60 bg-ink-900/40 p-2.5" style={{ boxShadow: `inset 3px 0 0 ${colorOf[dom] ?? "#4d5a6f"}` }}>
              <div className="mb-1.5 text-sm font-medium tracking-wide text-ink-50">{n}</div>
              <div className="flex flex-wrap gap-1">
                {a[n].map((w) => (
                  <span key={w} className="rounded-full px-2 py-0.5 text-[11px] text-ink-100" title={`from ${ownerOf[w]}`}
                        style={{ background: `${colorOf[ownerOf[w]] ?? "#4d5a6f"}33`, border: `1px solid ${colorOf[ownerOf[w]] ?? "#4d5a6f"}77` }}>{w}</span>
                ))}
              </div>
            </div>
          );
        })}
      </div>
      <div className="flex flex-wrap items-center gap-3 text-xs">
        <button className="btn" onClick={() => { set("symbolMapJson", json); set("presetName", null); setUsed(true); }}>use as the symbolic structure</button>
        <a className="text-accent-300 hover:underline" download="session-archetypes.json" href={`data:application/json;charset=utf-8,${encodeURIComponent(json)}`}>↓ JSON</a>
        {used && <span className="text-ink-400">Loaded into the sidebar. Press Build space to explore it.</span>}
      </div>
    </div>
  );
}

// ============================================================================ H · fingerprint
const GALLERY = "delyrism-session-gallery";
interface GalleryItem { id: string; name: string; at: number; f: NonNullable<SessionResult["fingerprint"]>; colors: string[] }

function Glyph({ f, colors, size = 120, uid }: { f: GalleryItem["f"]; colors: string[]; size?: number; uid: string }) {
  const n = f.symbols.length, c = 50;
  const turns = 1 + f.recurrence * 4;
  const spiral = Array.from({ length: 90 }, (_, k) => {
    const u = k / 89, ang = u * turns * Math.PI * 2, rr = 2.5 + u * 10;
    return `${(c + Math.cos(ang) * rr).toFixed(2)},${(c + Math.sin(ang) * rr).toFixed(2)}`;
  }).join(" ");
  const fid = `fg-${uid.replace(/[^a-zA-Z0-9]/g, "")}`;
  return (
    <svg viewBox="0 0 100 100" width={size} height={size}>
      <defs><filter id={fid} x="-50%" y="-50%" width="200%" height="200%"><feGaussianBlur stdDeviation="2.2" /></filter></defs>
      <circle cx={c} cy={c} r={14 + f.diversity * 30} fill="none" stroke="#f5f7fa" strokeOpacity={0.22} strokeWidth={0.6} strokeDasharray="1.5 1.5" />
      {[0, 1].map((layer) => (
        <g key={layer} filter={layer ? undefined : `url(#${fid})`} opacity={layer ? 0.95 : 0.6}>
          {f.symbol_share.map((v, i) => {
            const a = (i / n) * Math.PI * 2 - Math.PI / 2, w = (Math.PI / n) * 0.75, L = 10 + Math.sqrt(v) * 36;
            const p = (ang: number, rr: number) => `${(c + Math.cos(ang) * rr).toFixed(2)},${(c + Math.sin(ang) * rr).toFixed(2)}`;
            return <path key={i} d={`M${c},${c} Q${p(a - w, L * 0.75)} ${p(a, L)} Q${p(a + w, L * 0.75)} ${c},${c}`} fill={colors[i]} fillOpacity={0.12 + v * 0.85} />;
          })}
        </g>
      ))}
      <polyline points={spiral} fill="none" stroke="#f5f7fa" strokeOpacity={0.7} strokeWidth={0.55} strokeLinecap="round" />
      {Array.from({ length: f.phases }, (_, k) => {
        const a = (k / Math.max(1, f.phases)) * Math.PI * 2 - Math.PI / 2;
        return <circle key={k} cx={c + Math.cos(a) * 47} cy={c + Math.sin(a) * 47} r={1.3} fill="#f5f7fa" fillOpacity={0.8} />;
      })}
    </svg>
  );
}

function Fingerprint({ done }: { done: SessionDone }) {
  const { colorOf } = useCursor();
  const f = done.result.fingerprint!;
  const colors = f.symbols.map((s) => colorOf[s] ?? "#888");
  const [gallery, setGallery] = React.useState<GalleryItem[]>([]);
  React.useEffect(() => {
    let g: GalleryItem[] = [];
    try { g = JSON.parse(localStorage.getItem(GALLERY) || "[]"); } catch {}
    if (!g.some((x) => x.id === done.id)) {
      g = [{ id: done.id, name: done.result.summary.name || "", at: Date.now(), f, colors }, ...g].slice(0, 24);
      try { localStorage.setItem(GALLERY, JSON.stringify(g)); } catch {}
    }
    setGallery(g);
  }, [done.id]);  // eslint-disable-line react-hooks/exhaustive-deps
  const stat = (k: string, v: string, help: string) => (
    <div title={help}><div className="text-[10px] uppercase tracking-[0.14em] text-ink-500">{k}</div><div className="font-mono text-sm text-ink-100">{v}</div></div>
  );
  return (
    <div className="space-y-3">
      <div className="flex items-center gap-4">
        <Glyph f={f} colors={colors} size={148} uid={done.id} />
        <div className="grid grid-cols-2 gap-x-4 gap-y-2">
          {stat("diversity", `${Math.round(f.diversity * 100)}%`, "how evenly the words in the top 5 were shared (ring)")}
          {stat("returns", `${Math.round(f.recurrence * 100)}%`, "how often it came back to an earlier state (spiral turns)")}
          {stat("drift", f.drift.toFixed(3), "map widths per second")}
          {stat("phases", String(f.phases), "dots around the rim")}
        </div>
      </div>
      <div className="text-[10px] text-ink-500">petals · time on each archetype · ring · diversity · spiral · returns · dots · phases</div>
      {gallery.length > 1 && (
        <div>
          <div className="mb-1 text-[11px] text-ink-400">earlier sessions in this browser</div>
          <div className="flex flex-wrap gap-2">
            {gallery.filter((g) => g.id !== done.id).slice(0, 8).map((g) => (
              <div key={g.id} className="text-center text-[10px] text-ink-500" title={`${g.name || g.id} · ${new Date(g.at).toLocaleString()}`}>
                <Glyph f={g.f} colors={g.colors} size={52} uid={g.id} />
                <div className="w-14 truncate">{g.name || new Date(g.at).toLocaleDateString()}</div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

// ============================================================================ the bar
/** Start / end a session, with its clock. */
export function SessionBar() {
  const { state, started, start, stop, status } = useLive(useShallow((s) => ({
    state: s.sessionState, started: s.sessionStartedAt, start: s.startSession, stop: s.stopSession, status: s.status,
  })));
  const [name, setName] = React.useState("");
  const [, tick] = React.useState(0);
  React.useEffect(() => {
    if (state !== "recording") return;
    const id = setInterval(() => tick((x) => x + 1), 500);
    return () => clearInterval(id);
  }, [state]);
  const elapsed = started ? (performance.now() - started) / 1000 : 0;
  return (
    <div className={`flex flex-wrap items-center gap-3 rounded-xl border px-3 py-2 text-xs transition ${state === "recording" ? "border-danger/50 bg-danger/5" : "border-ink-700/60 bg-ink-900/40"}`}>
      <span className="font-semibold uppercase tracking-[0.18em] text-ink-300">Session</span>
      {state === "idle" ? (
        <>
          <input className="input-base w-48 py-1 text-xs" placeholder="name (optional)" value={name} onChange={(e) => setName(e.target.value)} />
          <button className="btn" onClick={() => start(name)} disabled={status !== "ready"}>● Start session</button>
          <span className="text-ink-500">records every second until you end it; the result opens below the graph</span>
        </>
      ) : state === "recording" ? (
        <>
          <span className="flex items-center gap-1.5 text-danger"><span className="h-2 w-2 animate-pulse rounded-full bg-danger" /> recording</span>
          <span className="font-mono text-base text-ink-50">{mmss(elapsed)}</span>
          <button className="btn border-danger/60 text-danger" onClick={stop}>■ End session</button>
        </>
      ) : (
        <span className="flex items-center gap-2 text-ink-300"><span className="h-2 w-2 animate-pulse rounded-full bg-accent-300" /> reading the session…</span>
      )}
    </div>
  );
}
