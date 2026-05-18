"use client";
import * as React from "react";
import { HelpTip } from "./HelpTip";

export function Toggle({
  label,
  value,
  onChange,
  hint,
  help,
  disabled,
}: {
  label: string;
  value: boolean;
  onChange: (v: boolean) => void;
  hint?: string;
  /** Tooltip text — shows a "?" icon next to the label. */
  help?: string;
  disabled?: boolean;
}) {
  return (
    <label className={`flex items-center justify-between gap-2 ${disabled ? "opacity-50" : ""}`}>
      <span>
        <div className="flex items-center text-sm text-ink-100">
          {label}
          {help && <HelpTip text={help} />}
        </div>
        {hint && <div className="text-[10px] text-ink-400">{hint}</div>}
      </span>
      <button
        type="button"
        role="switch"
        aria-checked={value}
        disabled={disabled}
        onClick={() => onChange(!value)}
        className={`relative inline-flex h-5 w-9 shrink-0 cursor-pointer items-center rounded-full transition
          ${value ? "bg-accent-500" : "bg-ink-700"}`}
      >
        <span
          className={`inline-block h-4 w-4 transform rounded-full bg-white shadow transition
            ${value ? "translate-x-4" : "translate-x-0.5"}`}
        />
      </button>
    </label>
  );
}
