"use client";
import * as React from "react";
import { useQuery } from "@tanstack/react-query";
import { api, WordCatalystResponse, WordCatalystEntry } from "@/lib/api";
import { useSidebar } from "@/lib/store";
import { Skeleton } from "../ui/Skeleton";
import { useTopologyContext } from "./useTopologyContext";
import { ContextPill } from "./TopologyOverview";

/**
 * Word catalysts — which descriptors hold the topology together.
 *
 * For each word in a symbol, two scores:
 *   ΔH1 / ΔH2 — leave-one-out: how much loop/void mass collapses if you
 *               remove this word from the cloud
 *   cycle weight — vertex-credit from the top persistent cocycles
 *
 * High composite = a load-bearing descriptor.  Remove it and the
 * archetype's internal structure disintegrates.
 *
 * Rendered as a ranked horizontal bar chart in pure HTML so we can
 * decorate each row with the three sub-scores.
 */
export function TopologyCatalysts() {
  const sid = useSidebar((s) => s.spaceId);
  const symbols = useSidebar((s) => s.symbols);
  const colorMap = useSidebar((s) => s.colorMap);
  const [symbol, setSymbol] = React.useState<string>("");

  React.useEffect(() => {
    if (!symbol && symbols.length) setSymbol(symbols[0]);
  }, [symbols, symbol]);

  const ctx = useTopologyContext();
  const q = useQuery({
    enabled: !!sid && !!symbol,
    queryKey: ["topo-catalysts", sid, symbol, ...ctx.keyTail],
    queryFn: () =>
      api.post<WordCatalystResponse>("/topology/catalysts", { space_id: sid, symbol, ...ctx.payload }),
  });

  const accent = colorMap[symbol] ?? "#88c0d0";

  return (
    <div className="space-y-3">
    {ctx.active && <ContextPill summary={ctx.summary} />}
    <div className="panel-tight">
      <div className="mb-3 flex items-center justify-between gap-3">
        <div>
          <div className="section-title">Word catalysts · {symbol}</div>
          <div className="text-[11px] text-ink-400">
            removing these descriptors collapses the most topology
          </div>
        </div>
        <select
          className="select-base !w-auto !min-w-[160px]"
          value={symbol}
          onChange={(e) => setSymbol(e.target.value)}
        >
          {symbols.map((s) => <option key={s} value={s}>{s}</option>)}
        </select>
      </div>

      {q.isPending && (
        <>
          <Skeleton lines={2} className="mb-3" />
          <Skeleton lines={8} />
        </>
      )}
      {q.error && <div className="text-sm text-danger">{(q.error as Error).message}</div>}
      {q.data && <CatalystList data={q.data} accent={accent} />}
    </div>
    </div>
  );
}

function CatalystList({ data, accent }: { data: WordCatalystResponse; accent: string }) {
  const entries = data.entries.filter((e) => e.composite > 0);
  if (entries.length === 0) {
    return (
      <div className="rounded-md border border-ink-700/40 bg-ink-900/30 p-3 text-[11px] text-ink-400">
        No descriptor has measurable topological impact in this symbol — its cloud is too small
        or too topologically trivial.
      </div>
    );
  }
  const maxComp = Math.max(...entries.map((e) => e.composite));
  const maxDH1 = Math.max(...data.entries.map((e) => Math.abs(e.delta_h1)), 1e-6);
  const maxDH2 = Math.max(...data.entries.map((e) => Math.abs(e.delta_h2)), 1e-6);
  const maxCW  = Math.max(...data.entries.map((e) => e.cycle_weight), 1e-6);

  return (
    <>
      <div className="mb-3 rounded-md border border-ink-700/40 bg-ink-900/30 p-2 text-[10px] text-ink-400">
        baseline H1 = <span className="font-mono text-ink-200">{data.h1_baseline.toFixed(3)}</span>
        {" · "}
        baseline H2 = <span className="font-mono text-ink-200">{data.h2_baseline.toFixed(3)}</span>
        {" · "}
        rows show the marginal collapse from removing each word
      </div>
      <div className="space-y-1">
        <div className="grid grid-cols-[1.4fr,1fr,1fr,1fr,3rem] items-center gap-2 px-1.5 pb-1 text-[10px] uppercase tracking-wider text-ink-500">
          <div>word</div>
          <div>ΔH1</div>
          <div>ΔH2</div>
          <div>cycle wt</div>
          <div className="text-right">composite</div>
        </div>
        {entries.slice(0, 20).map((e, i) => (
          <CatalystRow
            key={e.word + i}
            entry={e}
            accent={accent}
            relComposite={e.composite / maxComp}
            maxDH1={maxDH1}
            maxDH2={maxDH2}
            maxCW={maxCW}
          />
        ))}
      </div>
      <div className="mt-3 border-t border-ink-700/40 pt-2 text-[10px] leading-relaxed text-ink-400">
        <span className="text-ink-200">ΔH1 / ΔH2</span>: how much H1/H2 mass disappears when you delete this word.
        Positive = the word <em>supports</em> loops/voids.
        <br />
        <span className="text-ink-200">cycle wt</span>: vertex-credit from the top persistent cocycles. Words sitting
        on many persistent cycles get high credit.
      </div>
    </>
  );
}

function CatalystRow({
  entry, accent, relComposite, maxDH1, maxDH2, maxCW,
}: {
  entry: WordCatalystEntry;
  accent: string;
  relComposite: number;
  maxDH1: number;
  maxDH2: number;
  maxCW: number;
}) {
  return (
    <div
      className="grid grid-cols-[1.4fr,1fr,1fr,1fr,3rem] items-center gap-2 rounded-md px-1.5 py-1 text-[11px] hover:bg-ink-800/30"
      style={{
        background: relComposite > 0.6 ? `${accent}0d` : undefined,
      }}
    >
      <div className="flex items-center gap-2 truncate">
        <div className="h-1 w-1 shrink-0 rounded-full" style={{ background: accent, opacity: 0.6 + relComposite * 0.4 }} />
        <span className="truncate text-ink-100" style={{ fontWeight: relComposite > 0.5 ? 500 : 400 }}>{entry.word}</span>
      </div>
      <MicroBar value={entry.delta_h1} max={maxDH1} color="#3bbdb0" />
      <MicroBar value={entry.delta_h2} max={maxDH2} color="#d08770" />
      <MicroBar value={entry.cycle_weight} max={maxCW} color="#c2a6fe" />
      <div className="text-right font-mono text-ink-100" style={{ opacity: 0.6 + relComposite * 0.4 }}>
        {entry.composite.toFixed(3)}
      </div>
    </div>
  );
}

function MicroBar({ value, max, color }: { value: number; max: number; color: string }) {
  // Centred bar that extends left for negative values, right for positive.
  const norm = value / max; // in [-1, 1]
  const pct = Math.min(50, Math.abs(norm) * 50);
  const positive = norm >= 0;
  return (
    <div className="flex items-center gap-1.5">
      <div className="relative h-1.5 flex-1 overflow-hidden rounded-full bg-ink-800">
        <div className="absolute left-1/2 top-0 h-full w-px bg-ink-600/60" />
        <div
          className="absolute top-0 h-full rounded"
          style={{
            background: color,
            opacity: 0.85,
            left: positive ? "50%" : `${50 - pct}%`,
            width: `${pct}%`,
          }}
        />
      </div>
      <span className="w-10 shrink-0 text-right font-mono text-[9px] tabular-nums text-ink-400">
        {value > 0 ? "+" : ""}{value.toFixed(2)}
      </span>
    </div>
  );
}
