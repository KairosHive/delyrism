"use client";
import * as React from "react";

/**
 * Lightweight loading placeholder used in place of "computing…" text.
 *
 * Shows a soft animated pulse — quieter than a spinner, more honest than
 * a static label.  Pass `lines` for tabular/text panels or `height` for
 * chart panels.
 */
export function Skeleton({
  lines, height, className = "",
}: {
  lines?: number;
  height?: number | string;
  className?: string;
}) {
  if (height != null) {
    return (
      <div
        className={`animate-pulse rounded-lg bg-ink-800/40 ${className}`}
        style={{ height: typeof height === "number" ? `${height}px` : height }}
      />
    );
  }
  const count = Math.max(1, lines ?? 3);
  return (
    <div className={`space-y-2 ${className}`}>
      {Array.from({ length: count }).map((_, i) => (
        <div
          key={i}
          className="h-3 animate-pulse rounded bg-ink-800/40"
          style={{ width: `${60 + ((i * 13) % 35)}%` }}
        />
      ))}
    </div>
  );
}
