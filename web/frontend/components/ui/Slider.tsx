"use client";
import * as React from "react";
import { HelpTip } from "./HelpTip";

interface Props {
  label?: string;
  value: number;
  min: number;
  max: number;
  step?: number;
  onChange: (v: number) => void;
  format?: (v: number) => string;
  hint?: string;
  /** Tooltip text — shows a "?" icon next to the label. */
  help?: string;
  disabled?: boolean;
}

export function Slider({ label, value, min, max, step = 0.01, onChange, format, hint, help, disabled }: Props) {
  return (
    <div className={`space-y-1 ${disabled ? "opacity-50" : ""}`}>
      {label && (
        <div className="flex items-baseline justify-between">
          <span className="flex items-center label-sm">
            {label}
            {help && <HelpTip text={help} />}
          </span>
          <span className="font-mono text-xs text-ink-200">
            {format ? format(value) : value.toFixed(step >= 1 ? 0 : 2)}
          </span>
        </div>
      )}
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        disabled={disabled}
        onChange={(e) => onChange(parseFloat(e.target.value))}
        className="w-full accent-accent-400"
      />
      {hint && <p className="text-[10px] text-ink-400">{hint}</p>}
    </div>
  );
}
