"use client";
import * as React from "react";
import { useSidebar } from "@/lib/store";

/** Compact horizontal legend mapping each symbol to its palette color. */
export function SymbolLegend() {
  const symbols = useSidebar((s) => s.symbols);
  const colorMap = useSidebar((s) => s.colorMap);
  if (!symbols.length) return null;
  return (
    <div className="flex flex-wrap items-center gap-x-3 gap-y-1.5 text-[11px] text-ink-200">
      {symbols.map((s) => (
        <span key={s} className="inline-flex items-center gap-1.5">
          <span
            className="inline-block h-2.5 w-2.5 rounded-sm"
            style={{ background: colorMap[s] ?? "#888", boxShadow: `0 0 0 1px ${(colorMap[s] ?? "#888")}55` }}
          />
          <span className="font-medium tracking-tight">{s}</span>
        </span>
      ))}
    </div>
  );
}
