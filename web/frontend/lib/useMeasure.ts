"use client";
import * as React from "react";

/**
 * Callback-ref based size measurement.  Unlike `useRef + useLayoutEffect([])`,
 * this fires whenever the underlying element is mounted or replaced, so it
 * works when the measured node lives inside a conditionally-rendered branch
 * (e.g. only shown after data arrives).
 */
export function useMeasure(): [(node: HTMLElement | null) => void, number, number] {
  const [size, setSize] = React.useState<{ w: number; h: number }>({ w: 0, h: 0 });
  const obs = React.useRef<ResizeObserver | null>(null);

  const ref = React.useCallback((node: HTMLElement | null) => {
    obs.current?.disconnect();
    if (!node) return;
    const rect = node.getBoundingClientRect();
    setSize({ w: rect.width, h: rect.height });
    obs.current = new ResizeObserver(([e]) => {
      setSize({ w: e.contentRect.width, h: e.contentRect.height });
    });
    obs.current.observe(node);
  }, []);

  return [ref, size.w, size.h];
}
