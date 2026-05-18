"use client";
import * as React from "react";
import { useRankings } from "@/lib/hooks";
import { useSidebar } from "@/lib/store";

type Metric = "score" | "coherence" | "pagerank";

const METRIC_LABEL: Record<Metric, string> = {
  score: "composite",
  coherence: "coherence",
  pagerank: "PageRank",
};

const METRIC_DESC: Record<Metric, string> = {
  score: "λ · coherence + (1 − λ) · PageRank",
  coherence: "direct cosine to context (normalized)",
  pagerank: "graph-diffusion centrality (normalized)",
};

/**
 * HTML/CSS bar list — easier to read than overlapping Plotly bars and avoids
 * the unintuitive "click a legend entry to toggle a series" pattern.  Each row
 * shows the primary metric as a colored bar with the other two as small
 * sub-values, so all three numbers are visible at once.
 */
export function Rankings() {
  const colorMap = useSidebar((s) => s.colorMap);
  const select = useSidebar((s) => s.set);
  const selected = useSidebar((s) => s.selectedSymbol);
  const r = useRankings();

  const [metric, setMetric] = React.useState<Metric>("score");

  if (r.isPending && !r.data) return <PanelStub label="Ranking symbols…" />;
  if (r.error) return <div className="panel-pad text-sm text-danger">{(r.error as Error).message}</div>;
  if (!r.data) return null;

  const rows = [...r.data.rows].sort((a, b) => b[metric] - a[metric]);
  const maxVal = Math.max(0.0001, ...rows.map((row) => row[metric]));

  return (
    <div className="panel-tight">
      <div className="mb-3 flex flex-wrap items-center justify-between gap-2">
        <div>
          <div className="section-title">Ranked archetypes</div>
          <div className="text-[11px] text-ink-400">{rows.length} archetypes · {METRIC_DESC[metric]}</div>
        </div>
        <div className="flex rounded-md border border-ink-700 bg-ink-900 p-0.5 text-[11px]">
          {(["score", "coherence", "pagerank"] as Metric[]).map((m) => (
            <button
              key={m}
              onClick={() => setMetric(m)}
              className={`rounded px-2.5 py-1 transition ${
                metric === m
                  ? "bg-accent-600/30 text-accent-200"
                  : "text-ink-300 hover:bg-ink-800 hover:text-ink-100"
              }`}
            >
              {METRIC_LABEL[m]}
            </button>
          ))}
        </div>
      </div>

      <ul className="space-y-1.5">
        {rows.map((row) => {
          const color = colorMap[row.symbol] ?? "#888";
          const pct = (row[metric] / maxVal) * 100;
          const isSelected = selected === row.symbol;
          return (
            <li key={row.symbol}>
              <button
                onClick={() => select("selectedSymbol", isSelected ? null : row.symbol)}
                className={`group flex w-full items-center gap-3 rounded-md px-2 py-1.5 text-left transition
                  ${isSelected ? "bg-ink-800" : "hover:bg-ink-800/50"}`}
              >
                <span className="w-20 shrink-0 truncate text-[13px] font-medium" style={{ color }}>
                  {row.symbol}
                </span>
                <span className="relative h-3 flex-1 overflow-hidden rounded-sm bg-ink-800/80">
                  <span
                    className="absolute inset-y-0 left-0 rounded-sm transition-[width] duration-300 ease-out"
                    style={{
                      width: `${pct}%`,
                      background: `linear-gradient(90deg, ${color}, ${color}cc)`,
                      boxShadow: `0 0 0 1px ${color}33`,
                    }}
                  />
                </span>
                <span className="w-10 shrink-0 text-right font-mono text-[11px] text-ink-100">
                  {row[metric].toFixed(2)}
                </span>
                {/* the two non-active metrics inline */}
                <span className="hidden w-28 shrink-0 text-right font-mono text-[10px] text-ink-400 xl:inline">
                  {metric !== "coherence" && <>coh {row.coherence.toFixed(2)} · </>}
                  {metric !== "pagerank" && <>pr {row.pagerank.toFixed(2)}</>}
                  {metric === "coherence" && <>score {row.score.toFixed(2)} · pr {row.pagerank.toFixed(2)}</>}
                </span>
              </button>
            </li>
          );
        })}
      </ul>

      {selected && (
        <div className="mt-3 text-[11px] text-ink-400">
          drilled in: <span className="font-medium text-accent-300">{selected}</span>
          <button className="ml-2 underline hover:text-ink-100" onClick={() => select("selectedSymbol", null)}>clear</button>
        </div>
      )}
    </div>
  );
}

function PanelStub({ label }: { label: string }) {
  return (
    <div className="panel-pad text-sm text-ink-300">
      <span className="mr-2 inline-block h-2 w-2 animate-pulse rounded-full bg-accent-400" />
      {label}
    </div>
  );
}
