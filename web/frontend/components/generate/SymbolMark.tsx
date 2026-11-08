"use client";
import * as React from "react";
import { useSidebar } from "@/lib/store";
import { artOf } from "@/lib/symbolSets";
import { Glyph } from "./Glyph";

/** A symbol's mark wherever its name is listed: its drawing tinted with its colour when the set
 *  has one (generated on the Structures tab, or shipped with a preset), else a colour dot. */
export function SymbolMark({ symbol, color, size = 20, dot = "rounded-sm", className = "" }: {
  symbol: string; color: string; size?: number; dot?: string; className?: string;
}) {
  const art = useSidebar((s) => s.symbolArt);
  const a = artOf(art, symbol);
  if (a?.svg) {
    return <Glyph svg={a.svg} color={color} title={a.glyph} className={className} style={{ width: size, height: size }} />;
  }
  const d = Math.max(8, Math.round(size * 0.5));
  return (
    <span
      className={`inline-block shrink-0 ${dot} ${className}`}
      style={{ width: d, height: d, background: color, boxShadow: `0 0 0 1px ${color}55` }}
    />
  );
}
