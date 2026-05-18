"use client";
import * as React from "react";
import { HelpTip } from "./HelpTip";

interface Opt { value: string; label: string; }

export function Select({
  label,
  value,
  onChange,
  options,
  disabled,
  help,
}: {
  label?: string;
  value: string;
  onChange: (v: string) => void;
  options: Opt[];
  disabled?: boolean;
  /** Tooltip text — shows a "?" icon next to the label. */
  help?: string;
}) {
  return (
    <div className="space-y-1">
      {label && (
        <div className="flex items-center label-sm">
          {label}
          {help && <HelpTip text={help} />}
        </div>
      )}
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
