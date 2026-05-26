"use client";
import dynamic from "next/dynamic";
import * as React from "react";

// Plotly is heavy — load only on the client.
const RawPlot = dynamic(() => import("react-plotly.js"), { ssr: false }) as any;

/**
 * Drop-in `<Plot>` that wraps react-plotly's Plot in a `ResizeObserver`-
 * equipped div.  Plotly's own `useResizeHandler` only listens to window
 * resize events; it misses container-level changes (subview swaps inside
 * a tab, mobile orientation, drawer animations, sidebar collapse, panel
 * reflow on data load).
 *
 * Why this matters:
 *   - On desktop, switching topology subviews left charts measured at
 *     the *previous* subview's dimensions → truncated panels.
 *   - On mobile, the cycle plots rendered then "disappeared" after ~1s:
 *     the layout settled to a smaller size after first paint and Plotly
 *     never re-measured.
 *
 * Fix: ResizeObserver on our own wrapping div.  Whenever the container's
 * width or height changes (defer through requestAnimationFrame so we run
 * AFTER the layout settles), synthesize a window `resize` event.  React-
 * plotly's `useResizeHandler` catches it and Plotly re-flows to the new
 * dimensions.  Cheap, framework-agnostic, fires on every change.
 */
export function Plot(props: any) {
  const wrapRef = React.useRef<HTMLDivElement | null>(null);

  React.useEffect(() => {
    if (typeof window === "undefined") return;
    const node = wrapRef.current;
    if (!node) return;

    let raf: number | null = null;
    const trigger = () => {
      if (raf != null) cancelAnimationFrame(raf);
      raf = requestAnimationFrame(() => {
        window.dispatchEvent(new Event("resize"));
      });
    };

    // ResizeObserver fires once on observe() so the initial measurement
    // is also covered — fixes the "first paint truncated" case.
    const ro = new ResizeObserver(trigger);
    ro.observe(node);
    return () => {
      ro.disconnect();
      if (raf != null) cancelAnimationFrame(raf);
    };
  }, []);

  return (
    <div ref={wrapRef} style={{ width: "100%", minHeight: 0 }}>
      <RawPlot {...props} />
    </div>
  );
}
