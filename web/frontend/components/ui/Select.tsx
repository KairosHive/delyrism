"use client";
import * as React from "react";

interface Opt { value: string; label: string; }

export function Select({
  label,
  value,
  onChange,
  options,
  disabled,
}: {
  label?: string;
  value: string;
  onChange: (v: string) => void;
  options: Opt[];
  disabled?: boolean;
}) {
  return (
    <div className="space-y-1">
      {label && <div className="label-sm">{label}</div>}
      <select
        className="select-base"
        value={value}
        onChange={(e) => onChange(e.target.value)}
        disabled={disabled}
      >
        {options.map((o) => (
          <option key={o.value} value={o.value}>
            {o.label}
          </option>
        ))}
      </select>
    </div>
  );
}
