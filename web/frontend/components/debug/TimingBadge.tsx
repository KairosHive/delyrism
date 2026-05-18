"use client";
import * as React from "react";
import { getTimings, subscribeTimings, Timing } from "@/lib/api";

/**
 * Floating perf badge — shows the slowest backend route from the most recent
 * second of activity.  Click to expand a per-route breakdown.  This is the
 * fastest way to see "what to optimize next" without opening DevTools.
 */
export function TimingBadge() {
  const [, force] = React.useReducer((n) => n + 1, 0);
  const [open, setOpen] = React.useState(false);
  React.useEffect(() => subscribeTimings(force), []);

  const all = Object.entries(getTimings()) as [string, Timing][];
  const recent = all.filter(([, t]) => Date.now() - t.at < 4000);
  if (!recent.length) return null;

  const slowest = recent.reduce((a, b) => (a[1].serverMs > b[1].serverMs ? a : b));
  const color = slowest[1].serverMs > 1500
    ? "border-danger/60 text-danger"
    : slowest[1].serverMs > 500
    ? "border-warning/60 text-warning"
    : "border-accent-500/60 text-accent-300";

  return (
    <div className="fixed bottom-4 right-4 z-50 select-none">
      <button
        onClick={() => setOpen((o) => !o)}
        className={`flex items-center gap-2 rounded-lg border bg-ink-900/95 px-3 py-1.5 text-[11px] font-mono shadow-soft backdrop-blur ${color}`}
        title="click to expand all route timings"
      >
        <span className="h-1.5 w-1.5 rounded-full bg-current" />
        slowest: {slowest[0]} · {slowest[1].serverMs.toFixed(0)}ms
      </button>
      {open && (
        <div className="mt-2 w-72 rounded-lg border border-ink-700 bg-ink-900/95 p-2.5 text-[11px] font-mono text-ink-200 shadow-soft backdrop-blur">
          <div className="mb-1 flex items-center justify-between text-ink-400">
            <span>route</span>
            <span>server · total</span>
          </div>
          {all
            .sort((a, b) => b[1].at - a[1].at)
            .map(([p, t]) => (
              <div key={p} className="flex items-center justify-between gap-3 py-0.5">
                <span className="truncate">{p}</span>
                <span className="shrink-0">
                  <span className={t.serverMs > 500 ? "text-warning" : "text-ink-200"}>
                    {t.serverMs.toFixed(0)}
                  </span>
                  <span className="text-ink-500"> · {t.totalMs.toFixed(0)}ms</span>
                </span>
              </div>
            ))}
        </div>
      )}
    </div>
  );
}
