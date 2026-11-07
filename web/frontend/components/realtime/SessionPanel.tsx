"use client";
import * as React from "react";
import { api, apiUrl, type SessionDone, type SessionResult } from "@/lib/api";
import { useShallow } from "zustand/react/shallow";
import { useLive } from "@/lib/liveGraph";
import { useSidebar } from "@/lib/store";

/**
 * The result of a Live session (delyrism/session.py): where the query went on the space's map,
 * what it covered, how its structure changed, its phases and returns, its words, the archetypes
 * it formed and its fingerprint, with the raw data to download.
 */

const START = [59, 189, 176];      // #3bbdb0 — the path's first moments
const END = [194, 166, 254];       // #c2a6fe — its last
const lerp = (a: number[], b: number[], u: number) => `rgb(${a.map((x, i) => Math.round(x + (b[i] - x) * u)).join(",")})`;
const mmss = (s: number) => `${Math.floor(s / 60)}:${String(Math.floor(s % 60)).padStart(2, "0")}`;

export function SessionPanel() {
  const done = useLive((s) => s.sessionResult);
  const close = useLive((s) => s.closeSessionResult);
  if (!done) return null;
  const r = done.result;
  const colorOf = Object.fromEntries(r.symbols.map((s) => [s.id, s.color]));
  return (
    <div className="panel mt-4 space-y-5 p-4" data-testid="session-panel">
      <div className="flex flex-wrap items-baseline gap-x-4 gap-y-1">
        <div className="section-title">Session{r.summary.name ? ` · ${r.summary.name}` : ""}</div>
        <span className="font-mono text-xs text-ink-400">
          {mmss(r.summary.duration)} · {r.summary.snapshots} moments · query on {Math.round(r.summary.active_share * 100)}%
          {r.summary.kinds.length ? ` · ${r.summary.kinds.join(", ")}` : ""}
        </span>
        <span className="ml-auto flex items-center gap-3 text-xs">
          <a className="text-accent-300 hover:underline" href={apiUrl(`/realtime/session/${done.id}/raw`)} download>↓ raw data</a>
          <button className="text-ink-400 hover:text-ink-100" onClick={close}>close ✕</button>
        </span>
      </div>
      {done.saved && <div className="-mt-3 font-mono text-[11px] text-ink-500">saved on the server: {done.saved}</div>}
      {r.empty ? (
        <p className="text-sm text-ink-300">No query was active long enough to read a path. Start a session, then type, speak, play or move.</p>
      ) : (
        <>
          <div className="grid grid-cols-1 gap-5 lg:grid-cols-[1.25fr,1fr]">
            <Section label="A · Trajectory" hint="where the query's attention sat on the space's map, second by second">
              <TrajectoryMap r={r} />
            </Section>
            <Section label="B · Territory" hint="what the session reached, and what it never touched">
              <Territory r={r} colorOf={colorOf} />
            </Section>
          </div>
          <div className="grid grid-cols-1 gap-5 lg:grid-cols-2">
            <Section label="C · Topology over time" hint="separate groups the attended words form, and the loops they close, in sliding windows">
              <Topology r={r} />
            </Section>
            <Section label="D · Phases and returns" hint="where the session shifted, and when it came back to an earlier state">
              <Phases r={r} colorOf={colorOf} />
            </Section>
          </div>
          <div className="grid grid-cols-1 gap-5 lg:grid-cols-[1.2fr,1fr,0.9fr]">
            <Section label="E · Words" hint="what was typed, heard, spoken and described, in order">
              <Words done={done} />
            </Section>
            <Section label="G · Session archetypes" hint="the visited words clustered into the archetypes this session formed">
              <Archetypes r={r} />
            </Section>
            <Section label="H · Fingerprint" hint="time on each archetype (petals), diversity (ring), returns (spiral)">
              <Fingerprint done={done} colorOf={colorOf} />
            </Section>
          </div>
        </>
      )}
    </div>
  );
}

function Section({ label, hint, children }: { label: string; hint: string; children: React.ReactNode }) {
  return (
    <div className="space-y-2">
      <div>
        <div className="text-[11px] font-semibold uppercase tracking-[0.18em] text-ink-200">{label}</div>
        <div className="text-[11px] text-ink-500">{hint}</div>
      </div>
      {children}
    </div>
  );
}

// ---------------------------------------------------------------- A
function TrajectoryMap({ r }: { r: SessionResult }) {
  const tr = r.trajectory!;
  const visits = r.territory?.visits ?? {};
  const P = (x: number, y: number) => [6 + x * 88, 94 - y * 88];
  const path = tr.path.filter((_, i, a) => a.length <= 600 || i % Math.ceil(a.length / 600) === 0);
  const tMax = Math.max(r.summary.duration, 1);
  return (
    <svg viewBox="0 0 100 100" className="w-full rounded-md bg-ink-950/60" style={{ aspectRatio: "1 / 1" }}>
      {r.layout.map((d) => {
        const [x, y] = P(d.x, d.y);
        const v = visits[d.id] ?? 0;
        return <circle key={d.id} cx={x} cy={y} r={0.55 + Math.sqrt(v) * 2.2} fill={d.color} opacity={v ? 0.85 : 0.18}>
          <title>{`${d.id} · ${d.symbol}${v ? ` · in the top 5 for ${Math.round(v * 100)}% of the session` : " · never reached"}`}</title>
        </circle>;
      })}
      {path.slice(1).map((p, i) => {
        const [x0, y0] = P(path[i].x, path[i].y), [x1, y1] = P(p.x, p.y);
        return <line key={i} x1={x0} y1={y0} x2={x1} y2={y1} stroke={lerp(START, END, p.t / tMax)} strokeWidth={0.55} strokeOpacity={0.85} strokeLinecap="round" />;
      })}
      {tr.context_path.length > 1 && (
        <polyline points={tr.context_path.map((c) => P(c.x, c.y).join(",")).join(" ")} fill="none"
                  stroke="#e9e7e2" strokeOpacity={0.35} strokeWidth={0.4} strokeDasharray="1.2 1.2" />
      )}
      {tr.context_path.map((c, i) => {
        const [x, y] = P(c.x, c.y);
        return <g key={`c${i}`}><rect x={x - 1.1} y={y - 1.1} width={2.2} height={2.2} transform={`rotate(45 ${x} ${y})`} fill="#e9e7e2" fillOpacity={0.8}>
          <title>{`context at ${mmss(c.t)}: ${c.text}`}</title></rect></g>;
      })}
      {tr.dwells.map((d, i) => {
        const [x, y] = P(d.x, d.y);
        return <g key={`d${i}`}>
          <circle cx={x} cy={y} r={1.2 + Math.min(4, (d.t1 - d.t0) / 15)} fill="none" stroke="#e9e7e2" strokeOpacity={0.6} strokeWidth={0.3} />
          <text x={x + 2.2} y={y - 1.4} fontSize={2.3} fill="#e9e7e2" fillOpacity={0.85}>{d.words[0]}</text>
          <title>{`dwelt ${mmss(d.t0)}–${mmss(d.t1)}: ${d.words.join(", ")}`}</title>
        </g>;
      })}
      {path.length > 0 && (() => {
        const [x0, y0] = P(path[0].x, path[0].y), [x1, y1] = P(path[path.length - 1].x, path[path.length - 1].y);
        return <g><circle cx={x0} cy={y0} r={1.1} fill={lerp(START, END, 0)} /><circle cx={x1} cy={y1} r={1.1} fill={lerp(START, END, 1)} /></g>;
      })()}
      <text x={6} y={4.2} fontSize={2.4} fill="#9a9892">{`start ● → ● end · path ${tr.distance.toFixed(1)} map widths · ◆ context`}</text>
    </svg>
  );
}

// ---------------------------------------------------------------- B
function Territory({ r, colorOf }: { r: SessionResult; colorOf: Record<string, string> }) {
  const t = r.territory!;
  const syms = r.symbols.map((s) => s.id);
  const [all, setAll] = React.useState(false);
  return (
    <div className="space-y-3 text-xs">
      <div className="text-ink-200"><span className="text-2xl font-light text-ink-50">{Math.round(t.reached * 100)}%</span> of the network reached the top 5 at least once</div>
      <div className="space-y-1">
        {syms.map((s) => (
          <div key={s} className="flex items-center gap-2">
            <span className="w-28 truncate text-ink-300">{s}</span>
            <div className="h-2 flex-1 overflow-hidden rounded-full bg-ink-800">
              <div className="h-full rounded-full" style={{ width: `${(t.symbol_share[s] ?? 0) * 100}%`, background: colorOf[s] }} />
            </div>
            <span className="w-9 text-right font-mono text-ink-400">{Math.round((t.symbol_share[s] ?? 0) * 100)}%</span>
          </div>
        ))}
      </div>
      {t.stream.length > 1 && (
        <svg viewBox={`0 0 ${t.stream.length * 10} 24`} className="w-full" preserveAspectRatio="none" style={{ height: 34 }}>
          {t.stream.map((m, i) => {
            let y = 0;
            return <g key={i}>{m.shares.map((v, j) => {
              const h = v * 24; const el = <rect key={j} x={i * 10 + 0.5} y={y} width={9} height={h} fill={colorOf[syms[j]]}><title>{`${mmss(m.t0)}: ${syms[j]} ${Math.round(v * 100)}%`}</title></rect>;
              y += h; return el;
            })}</g>;
          })}
        </svg>
      )}
      {t.never.length > 0 && (
        <div>
          <div className="mb-1 text-ink-400">never reached ({t.never.length})</div>
          <div className="flex flex-wrap gap-1">
            {(all ? t.never : t.never.slice(0, 24)).map((w) => <span key={w} className="rounded border border-ink-700 px-1.5 py-0.5 text-ink-400">{w}</span>)}
            {t.never.length > 24 && <button className="text-accent-300" onClick={() => setAll(!all)}>{all ? "fewer" : `+ ${t.never.length - 24} more`}</button>}
          </div>
        </div>
      )}
    </div>
  );
}

// ---------------------------------------------------------------- C
function Topology({ r }: { r: SessionResult }) {
  const tp = r.topology!;
  const w = tp.windows;
  const tMax = Math.max(r.summary.duration, 1);
  const gMax = Math.max(2, ...w.map((x) => x.groups), ...w.map((x) => x.loops));
  const X = (t: number) => 4 + (t / tMax) * 92, Y = (v: number) => 34 - (v / gMax) * 28;
  const line = (k: "groups" | "loops") => w.map((x) => `${X((x.t0 + x.t1) / 2)},${Y(x[k])}`).join(" ");
  return (
    <div className="space-y-2 text-xs">
      {w.length > 1 ? (
        <svg viewBox="0 0 100 40" className="w-full rounded-md bg-ink-950/60">
          <polyline points={line("groups")} fill="none" stroke="#3bbdb0" strokeWidth={0.7} />
          <polyline points={line("loops")} fill="none" stroke="#c2a6fe" strokeWidth={0.7} strokeDasharray="1.5 1" />
          {tp.events.map((e, i) => <line key={i} x1={X(e.t)} x2={X(e.t)} y1={5} y2={35} stroke="#e9e7e2" strokeOpacity={0.25} strokeWidth={0.3}><title>{`${mmss(e.t)}: ${e.text}`}</title></line>)}
          <text x={4} y={4} fontSize={2.6} fill="#3bbdb0">groups</text>
          <text x={16} y={4} fontSize={2.6} fill="#c2a6fe">loops</text>
          <text x={96} y={39} fontSize={2.4} fill="#9a9892" textAnchor="end">{mmss(tMax)}</text>
        </svg>
      ) : <div className="text-ink-500">Too short for sliding windows.</div>}
      <ul className="max-h-28 space-y-0.5 overflow-auto text-ink-300">
        {tp.events.map((e, i) => <li key={i}><span className="font-mono text-ink-500">{mmss(e.t)}</span> {e.text}</li>)}
      </ul>
      {tp.graph_events.length > 0 && (
        <div>
          <div className="mb-1 text-ink-400">Δ-graph at each context change</div>
          <ul className="max-h-28 space-y-1 overflow-auto">
            {tp.graph_events.map((g, i) => (
              <li key={i} className="text-ink-300">
                <span className="font-mono text-ink-500">{mmss(g.t)}</span> <span className="text-accent-300">+{g.born}</span> / <span className="text-warning">−{g.died}</span> edges ({g.kept} kept)
                {g.top_born.length > 0 && <span className="text-ink-500"> · new: {g.top_born.join(", ")}</span>}
              </li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
}

// ---------------------------------------------------------------- D
function Phases({ r, colorOf }: { r: SessionResult; colorOf: Record<string, string> }) {
  const ph = r.phases!;
  const tMax = Math.max(r.summary.duration, 1);
  const ref = React.useRef<HTMLCanvasElement>(null);
  React.useEffect(() => {
    const c = ref.current; if (!c) return;
    const M = ph.recurrence.M, n = M.length; if (!n) return;
    c.width = n; c.height = n;
    const ctx = c.getContext("2d")!; const img = ctx.createImageData(n, n);
    for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) {
      const u = Math.max(0, Math.min(1, (M[i][j] + 1) / 2)) ** 2;         // correlation −1…1 → 0…1
      const k = ((n - 1 - i) * n + j) * 4;
      img.data[k] = 20 + u * 174; img.data[k + 1] = 20 + u * 146; img.data[k + 2] = 30 + u * 224; img.data[k + 3] = 255;
    }
    ctx.putImageData(img, 0, 0);
  }, [ph]);
  return (
    <div className="space-y-2 text-xs">
      <div className="flex h-9 overflow-hidden rounded-md">
        {ph.phases.map((p, i) => (
          <div key={i} className="flex min-w-0 items-center px-1.5 text-[10px] text-ink-950"
               style={{ width: `${((p.t1 - p.t0 + 1) / tMax) * 100}%`, background: colorOf[p.symbol] }}
               title={`${mmss(p.t0)}–${mmss(p.t1)} · ${p.symbol} · ${p.words.join(", ")}`}>
            <span className="truncate font-medium">{p.words[0]}</span>
          </div>
        ))}
      </div>
      <ol className="space-y-0.5 text-ink-300">
        {ph.phases.map((p, i) => <li key={i}><span className="font-mono text-ink-500">{mmss(p.t0)}</span> <span style={{ color: colorOf[p.symbol] }}>{p.symbol}</span> · {p.words.join(", ")}</li>)}
      </ol>
      <div className="flex items-start gap-3">
        <canvas ref={ref} className="h-36 w-36 shrink-0 rounded-md" style={{ imageRendering: "pixelated" }}
                title="recurrence: how alike each moment's relevance is to each other moment (time runs right and up)" />
        <div className="space-y-1 text-ink-300">
          <div className="text-ink-400">returns</div>
          {ph.returns.length ? ph.returns.map((x, i) => <div key={i}><span className="font-mono text-ink-500">{mmss(x.t)}</span> came back to {mmss(x.to)} <span className="text-ink-500">(r {x.r})</span></div>)
            : <div className="text-ink-500">none: the session kept moving on</div>}
        </div>
      </div>
    </div>
  );
}

// ---------------------------------------------------------------- E
function Words({ done }: { done: SessionDone }) {
  const words = done.result.words ?? [];
  const [story, setStory] = React.useState<string | null>(null);
  const [busy, setBusy] = React.useState(false);
  const [err, setErr] = React.useState<string | null>(null);
  const icon: Record<string, string> = { query: "⌕", heard: "◌", spoken: "♪", motion: "∿", context: "◆" };
  async function weave() {
    setBusy(true); setErr(null);
    try { setStory((await api.post<{ story: string }>(`/realtime/session/${done.id}/story`, {})).story); }
    catch (e: any) { setErr(String(e?.message ?? e)); }
    finally { setBusy(false); }
  }
  return (
    <div className="space-y-2 text-xs">
      <ul className="max-h-44 space-y-0.5 overflow-auto text-ink-300">
        {words.length ? words.map((w, i) => (
          <li key={i}><span className="font-mono text-ink-500">{mmss(w.t)}</span> <span className="text-ink-500" title={w.type}>{icon[w.type] ?? "·"}</span> {w.text}</li>
        )) : <li className="text-ink-500">no words in this session</li>}
      </ul>
      <button className="btn text-[11px]" onClick={weave} disabled={busy}>{busy ? "weaving…" : "✶ weave a text from the phases"}</button>
      {err && <div className="text-danger">{err}</div>}
      {story && <p className="whitespace-pre-wrap rounded-md bg-ink-950/50 p-3 text-[13px] leading-relaxed text-ink-100">{story}</p>}
    </div>
  );
}

// ---------------------------------------------------------------- G
function Archetypes({ r }: { r: SessionResult }) {
  const a = r.archetypes ?? {};
  const set = useSidebar((s) => s.set);
  const [used, setUsed] = React.useState(false);
  const names = Object.keys(a);
  if (!names.length) return <div className="text-xs text-ink-500">Too few words visited to form archetypes.</div>;
  const json = JSON.stringify(a, null, 2);
  return (
    <div className="space-y-2 text-xs">
      {names.map((n) => (
        <div key={n} className="rounded-md border border-ink-700/70 px-2 py-1.5">
          <div className="font-semibold tracking-wide text-ink-100">{n}</div>
          <div className="text-ink-400">{a[n].join(", ")}</div>
        </div>
      ))}
      <div className="flex gap-3">
        <button className="text-accent-300 hover:underline" onClick={() => { set("symbolMapJson", json); set("presetName", null); setUsed(true); }}>
          use as the symbolic structure
        </button>
        <a className="text-accent-300 hover:underline" download="session-archetypes.json"
           href={`data:application/json;charset=utf-8,${encodeURIComponent(json)}`}>↓ JSON</a>
      </div>
      {used && <div className="text-ink-400">Loaded into the sidebar. Press Build space to explore it.</div>}
    </div>
  );
}

// ---------------------------------------------------------------- H
const GALLERY = "delyrism-session-gallery";
interface GalleryItem { id: string; name: string; at: number; f: NonNullable<SessionResult["fingerprint"]>; colors: string[] }

function Glyph({ f, colors, size = 120 }: { f: GalleryItem["f"]; colors: string[]; size?: number }) {
  const n = f.symbols.length;
  const c = 50;
  const petals = f.symbol_share.map((v, i) => {
    const a = (i / n) * Math.PI * 2 - Math.PI / 2, w = Math.PI / n * 0.8, L = 12 + Math.sqrt(v) * 34;
    const p = (ang: number, rr: number) => `${c + Math.cos(ang) * rr},${c + Math.sin(ang) * rr}`;
    return <path key={i} d={`M${c},${c} Q${p(a - w, L * 0.7)} ${p(a, L)} Q${p(a + w, L * 0.7)} ${c},${c}`} fill={colors[i]} fillOpacity={0.15 + v * 0.8} />;
  });
  const turns = 1 + f.recurrence * 4;
  const spiral = Array.from({ length: 80 }, (_, k) => {
    const u = k / 79, ang = u * turns * Math.PI * 2, rr = 3 + u * 9;
    return `${c + Math.cos(ang) * rr},${c + Math.sin(ang) * rr}`;
  }).join(" ");
  return (
    <svg viewBox="0 0 100 100" width={size} height={size}>
      <circle cx={c} cy={c} r={14 + f.diversity * 32} fill="none" stroke="#e9e7e2" strokeOpacity={0.25} strokeWidth={0.6} />
      {petals}
      <polyline points={spiral} fill="none" stroke="#e9e7e2" strokeOpacity={0.6} strokeWidth={0.5} />
      {Array.from({ length: f.phases }, (_, k) => {
        const a = (k / Math.max(1, f.phases)) * Math.PI * 2;
        return <circle key={k} cx={c + Math.cos(a) * 47} cy={c + Math.sin(a) * 47} r={1.1} fill="#e9e7e2" fillOpacity={0.7} />;
      })}
    </svg>
  );
}

function Fingerprint({ done, colorOf }: { done: SessionDone; colorOf: Record<string, string> }) {
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
  return (
    <div className="space-y-2 text-xs">
      <div className="flex items-center gap-3">
        <Glyph f={f} colors={colors} />
        <div className="space-y-0.5 text-ink-300">
          <div>diversity <span className="font-mono text-ink-100">{Math.round(f.diversity * 100)}%</span></div>
          <div>returns <span className="font-mono text-ink-100">{Math.round(f.recurrence * 100)}%</span></div>
          <div>drift <span className="font-mono text-ink-100">{f.drift.toFixed(3)}</span>/s</div>
          <div>{f.phases} phase{f.phases === 1 ? "" : "s"} · {mmss(f.duration)}</div>
        </div>
      </div>
      {gallery.length > 1 && (
        <div>
          <div className="mb-1 text-ink-400">earlier sessions in this browser</div>
          <div className="flex flex-wrap gap-2">
            {gallery.filter((g) => g.id !== done.id).slice(0, 8).map((g) => (
              <div key={g.id} className="text-center text-[10px] text-ink-500" title={`${g.name || g.id} · ${new Date(g.at).toLocaleString()}`}>
                <Glyph f={g.f} colors={g.colors} size={48} />
                <div className="w-12 truncate">{g.name || new Date(g.at).toLocaleDateString()}</div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

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
    <div className="flex flex-wrap items-center gap-3 rounded-lg border border-ink-700/60 bg-ink-900/40 px-3 py-2 text-xs">
      <span className="font-semibold uppercase tracking-[0.18em] text-ink-300">Session</span>
      {state === "idle" ? (
        <>
          <input className="input-base w-48 py-1 text-xs" placeholder="name (optional)" value={name} onChange={(e) => setName(e.target.value)} />
          <button className="btn" onClick={() => start(name)} disabled={status !== "ready"}>● Start session</button>
          <span className="text-ink-500">records every second until you end it; the result opens below</span>
        </>
      ) : state === "recording" ? (
        <>
          <span className="flex items-center gap-1.5 text-danger"><span className="h-2 w-2 animate-pulse rounded-full bg-danger" /> recording</span>
          <span className="font-mono text-ink-200">{mmss(elapsed)}</span>
          <button className="btn border-danger/60 text-danger" onClick={stop}>■ End session</button>
        </>
      ) : (
        <span className="text-ink-300">analysing the session…</span>
      )}
    </div>
  );
}
