"use client";
import * as React from "react";

interface Props {
  label?: string;
  value: number;
  min: number;
  max: number;
  step?: number;
  onChange: (v: number) => void;
  format?: (v: number) => string;
  hint?: string;
  disabled?: boolean;
}

export function Slider({ label, value, min, max, step = 0.01, onChange, format, hint, disabled }: Props) {
  return (
    <div className={`space-y-1 ${disabled ? "opacity-50" : ""}`}>
      {label && (
        <div className="flex items-baseline justify-between">
          <span className="label-sm">{label}</span>
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
