"use client";
import * as React from "react";

/**
 * A small "?" icon next to a control label.  On hover/focus it pops a short
 * description into a tooltip.
 *
 * Implementation note: tooltips use `position: fixed` (with coords computed
 * from the icon's `getBoundingClientRect()`) so they can escape the panel's
 * `overflow: hidden` + `contain: paint` clipping.  An absolutely-positioned
 * tooltip would be clipped to the panel's borders, making half of them
 * disappear.
 */
export function HelpTip({ text, side = "right" }: { text: string; side?: "right" | "left" | "top" | "bottom" }) {
  const ref = React.useRef<HTMLButtonElement>(null);
  const [open, setOpen] = React.useState(false);
  const [pos, setPos] = React.useState<{ top: number; left: number } | null>(null);
  const W = 240;

  const show = React.useCallback(() => {
    if (!ref.current) return;
    const r = ref.current.getBoundingClientRect();
    let top = r.bottom + 6;
    let left = r.left + r.width / 2 - W / 2;
    // clamp horizontally so the tip stays on screen
    const margin = 8;
    if (left < margin) left = margin;
    const max = window.innerWidth - W - margin;
    if (left > max) left = max;
    // flip above if too close to viewport bottom
    if (top + 80 > window.innerHeight) top = r.top - 6 - 80;
    setPos({ top, left });
    setOpen(true);
  }, []);
  const hide = React.useCallback(() => setOpen(false), []);

  return (
    <>
      <button
        ref={ref}
        type="button"
        aria-label="help"
        className="ml-1 inline-flex h-3.5 w-3.5 shrink-0 items-center justify-center rounded-full border border-ink-600 text-[9px] font-medium text-ink-400 transition
                   hover:border-accent-500 hover:text-accent-300 focus:border-accent-500 focus:text-accent-300 focus:outline-none"
        onMouseEnter={show}
        onMouseLeave={hide}
        onFocus={show}
        onBlur={hide}
        onClick={(e) => { e.preventDefault(); open ? hide() : show(); }}
      >
        ?
      </button>
      {open && pos && (
        <div
          role="tooltip"
          className="pointer-events-none fixed z-[100] rounded-md border border-ink-700 bg-ink-900 px-2.5 py-1.5
                     text-[10.5px] font-normal leading-snug text-ink-200 shadow-soft"
          style={{ top: pos.top, left: pos.left, width: W }}
        >
          {text}
        </div>
      )}
    </>
  );
}
