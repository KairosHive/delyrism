"use client";
import * as React from "react";
import { useShallow } from "zustand/react/shallow";
import { useWalker } from "@/lib/realtime";
import { useSidebar } from "@/lib/store";

/** The recipe of the symbol under construction — ordered descriptors with
 *  their Δ gains, archetype composition, metrics — plus the step chooser
 *  and the bridge back into the structure JSON (so a generated symbol can
 *  be built into a space like any preset). */
export function SymbolRecipe() {
  const { symbol, candidates, hasContext, sealed } = useWalker(useShallow((s) => ({
    symbol: s.frame?.symbol ?? null,
    candidates: s.frame?.candidates ?? [],
    hasContext: s.frame?.has_context ?? false,
    sealed: s.frame?.sealed ?? [],
  })));
  const colorMap = useWalker((s) => s.layout?.color_map ?? {});
  const mode = useWalker((s) => s.mode);
  const { step, undo, reset, seal, dropSealed } = useWalker(useShallow((s) => ({
    step: s.step, undo: s.undo, reset: s.reset, seal: s.seal, dropSealed: s.dropSealed,
  })));

  const n = symbol?.descriptors.length ?? 0;
  const [name, setName] = React.useState("");
  const [touched, setTouched] = React.useState(false);
  // auto-name follows the walk until the user edits it; reset on new walk
  React.useEffect(() => { if (!touched) setName(symbol?.name ?? ""); }, [symbol?.name, touched]);
  React.useEffect(() => { if (n === 0) { setTouched(false); setName(""); } }, [n]);

  const [added, setAdded] = React.useState<string | null>(null);
  const addToStructure = useAddToStructure();
  const gmax = Math.max(1e-6, ...(symbol?.gains.map(Math.abs) ?? [0]));

  return (
    <div className="panel-tight space-y-3">
      <div className="flex items-center justify-between">
        <div className="section-title">Recipe</div>
        <span className="text-[10px] text-ink-500">{n ? `${n} descriptor${n > 1 ? "s" : ""}` : "empty"}</span>
      </div>

      {/* name + actions */}
      <div className="flex flex-wrap items-center gap-2">
        <input
          className="input-base max-w-[220px] font-mono"
          value={name}
          placeholder="symbol name"
          onChange={(e) => { setName(e.target.value); setTouched(true); }}
        />
        <button className="btn" onClick={() => seal(name.trim() || undefined)} disabled={n === 0} title="freeze this walk as a finished symbol and start a new one">seal ◉</button>
        <button className="btn" onClick={undo} disabled={n === 0}>undo</button>
        <button className="btn" onClick={reset} disabled={n === 0}>reset</button>
        <button
          className="btn-primary"
          disabled={n === 0}
          onClick={() => { const k = addToStructure(name.trim() || symbol!.name, symbol!.descriptors); setAdded(k); }}
          title="append to the Symbolic Structure JSON — press Rebuild space to embed it"
        >
          add to structure
        </button>
      </div>
      {added && <div className="text-[11px] text-accent-300">added <span className="font-mono">{added}</span> to the structure — press <em>Rebuild space</em> to make it a real archetype.</div>}

      {/* descriptors */}
      {n > 0 && symbol && (
        <ol className="space-y-1">
          {symbol.descriptors.map((d, k) => {
            const col = colorMap[symbol.owners[k]] ?? "#888";
            const g = symbol.gains[k];
            return (
              <li key={`${d}-${k}`} className="grid grid-cols-[18px,1fr,110px,56px] items-center gap-2 text-[12px]">
                <span className="font-mono text-[10px] text-ink-500">{k + 1}</span>
                <span className="flex items-center gap-1.5 truncate">
                  <span className="inline-block h-2 w-2 shrink-0 rounded-sm" style={{ background: col }} />
                  <span className="truncate text-ink-100">{d}</span>
                  <span className="truncate text-[10px] text-ink-500">{symbol.owners[k]}</span>
                </span>
                <span className="h-1.5 overflow-hidden rounded-full bg-ink-800">
                  <span className="block h-full rounded-full" style={{ width: `${Math.round(100 * Math.min(1, Math.abs(g) / gmax))}%`, background: g >= 0 ? "#3bbdb0" : "#bf616a" }} />
                </span>
                <span className="text-right font-mono text-[10px] text-ink-400">{k === 0 ? "start" : (g >= 0 ? "+" : "") + g.toFixed(3)}</span>
              </li>
            );
          })}
        </ol>
      )}

      {/* composition + metrics */}
      {n > 0 && symbol && (
        <div className="space-y-1.5">
          <div className="flex h-2 overflow-hidden rounded-full bg-ink-800">
            {Object.entries(symbol.composition).map(([s, share]) => (
              <span key={s} style={{ width: `${share * 100}%`, background: colorMap[s] ?? "#888" }} title={`${s} ${(share * 100).toFixed(0)}%`} />
            ))}
          </div>
          <div className="flex flex-wrap gap-x-3 gap-y-1 text-[10px] text-ink-400">
            <span>coherence <span className="font-mono text-ink-200">{symbol.coherence.toFixed(3)}</span></span>
            <span>context fit <span className="font-mono text-ink-200">{symbol.context_fit.toFixed(3)}</span></span>
            <span>nearest {symbol.nearest.map((x) => (
              <span key={x.symbol} className="ml-1 font-mono text-ink-200">{x.symbol} {x.sim.toFixed(2)}</span>
            ))}</span>
          </div>
        </div>
      )}

      {/* candidates — the progressive-choice interface */}
      <div>
        <div className="mb-1 flex items-center justify-between">
          <span className="sub-title">{n ? "next step" : "opening move"}</span>
          <span className="text-[10px] text-ink-500">{mode === "choose" ? "click to choose" : "click, or let the walker sample"}</span>
        </div>
        {candidates.length === 0 ? (
          <div className="text-[11px] text-ink-500">{hasContext ? "no candidates — all descriptors visited?" : "no context yet — the walker has nothing to follow"}</div>
        ) : (
          <div className="flex flex-wrap gap-1.5">
            {candidates.map((c) => (
              <button
                key={c.i}
                onClick={() => step(c.i)}
                className="group inline-flex items-center gap-1.5 rounded-md border px-2 py-1 text-[11px] transition hover:bg-ink-800"
                style={{ borderColor: `${colorMap[c.symbol] ?? "#888"}66` }}
                title={`${c.symbol} · Δ ${c.delta >= 0 ? "+" : ""}${c.delta.toFixed(3)} · score ${c.score.toFixed(3)}`}
              >
                <span className="inline-block h-2 w-2 rounded-sm" style={{ background: colorMap[c.symbol] ?? "#888" }} />
                <span className="text-ink-100">{c.id}</span>
                <span className="font-mono text-[10px] text-ink-400">{Math.round(c.p * 100)}%</span>
              </button>
            ))}
            <button className="btn px-2 py-1 text-[11px]" onClick={() => step()}>sample ⚄</button>
          </div>
        )}
      </div>

      {/* sealed symbols */}
      {sealed.length > 0 && (
        <div className="border-t border-ink-700/50 pt-2">
          <div className="sub-title mb-1">sealed symbols</div>
          <ul className="space-y-1">
            {sealed.map((s) => (
              <li key={s.index ?? s.name} className="flex items-center gap-2 text-[11px]">
                <span className="flex h-1.5 w-14 overflow-hidden rounded-full bg-ink-800">
                  {Object.entries(s.composition).map(([k, share]) => (
                    <span key={k} style={{ width: `${share * 100}%`, background: colorMap[k] ?? "#888" }} />
                  ))}
                </span>
                <span className="font-mono text-ink-100">{s.name}</span>
                <span className="truncate text-ink-500">{s.descriptors.join(" · ")}</span>
                <span className="ml-auto flex shrink-0 gap-1">
                  <button className="btn px-1.5 py-0.5 text-[10px]" onClick={() => setAdded(addToStructure(s.name, s.descriptors))}>add</button>
                  <button className="btn px-1.5 py-0.5 text-[10px]" onClick={() => dropSealed(s.index ?? -1)}>×</button>
                </span>
              </li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
}

/** Append `{name: descriptors}` to the editable structure JSON.  Returns the
 *  key actually used (uppercased, de-duplicated with a numeric suffix). */
function useAddToStructure() {
  const json = useSidebar((s) => s.symbolMapJson);
  const set = useSidebar((s) => s.set);
  return React.useCallback((name: string, descriptors: string[]) => {
    let map: Record<string, string[]> = {};
    try { map = JSON.parse(json || "{}"); } catch { map = {}; }
    const base = (name || "symbol").toUpperCase().replace(/\s+/g, "-");
    let key = base; let k = 2;
    while (key in map) key = `${base}-${k++}`;
    map[key] = [...descriptors];
    set("symbolMapJson", JSON.stringify(map, null, 2));
    return key;
  }, [json, set]);
}
