"use client";
import * as React from "react";

interface Props {
  title: string;
  icon?: React.ReactNode;
  /** A hex color for the accent strip / hover; matches Streamlit section palette. */
  color?: string;
  defaultOpen?: boolean;
  children: React.ReactNode;
}

export function Section({ title, icon, color, defaultOpen = true, children }: Props) {
  const [open, setOpen] = React.useState(defaultOpen);
  const accent = color ?? "#88C0D0";
  return (
    <div
      className="mb-2.5 overflow-hidden rounded-lg border bg-ink-900/40 transition-shadow"
      style={{
        borderColor: `${accent}33`,
        boxShadow: open ? `inset 3px 0 0 0 ${accent}` : `inset 3px 0 0 0 ${accent}88`,
      }}
    >
      <button
        type="button"
        onClick={() => setOpen((o) => !o)}
        className="flex w-full items-center justify-between px-3.5 py-2.5 text-left transition hover:bg-ink-800/40"
        style={{ color: accent }}
        aria-expanded={open}
      >
        <span className="flex items-center gap-2">
          {icon && <span aria-hidden>{icon}</span>}
          <span className="text-[13px] font-medium tracking-tight">{title}</span>
        </span>
        <svg
          className={`h-3.5 w-3.5 transition-transform ${open ? "rotate-180" : ""}`}
          viewBox="0 0 20 20"
          fill="currentColor"
          style={{ opacity: 0.7 }}
        >
          <path
            fillRule="evenodd"
            d="M5.23 7.21a.75.75 0 011.06.02L10 11.06l3.71-3.83a.75.75 0 011.08 1.04l-4.25 4.39a.75.75 0 01-1.08 0L5.21 8.27a.75.75 0 01.02-1.06z"
            clipRule="evenodd"
          />
        </svg>
      </button>
      {open && (
        <div
          className="space-y-2.5 border-t px-3.5 py-3.5"
          style={{ borderColor: `${accent}22` }}
        >
          {children}
        </div>
      )}
    </div>
  );
}
