"use client";
import * as React from "react";
import { svgUrl } from "@/lib/symbolSets";

/** A vector drawing (black ink on transparent) tinted with a colour: the SVG is a CSS mask over
 *  a filled box, as do-you-dream shows its symbols. */
export function Glyph({ svg, color, className = "", title, style }: {
  svg: string; color: string; className?: string; title?: string; style?: React.CSSProperties;
}) {
  const url = React.useMemo(() => svgUrl(svg), [svg]);
  const mask = `url("${url}") center / contain no-repeat`;
  return (
    <span
      className={`inline-block shrink-0 ${className}`}
      title={title}
      style={{ ...style, background: color, WebkitMask: mask, mask }}
      aria-hidden
    />
  );
}
