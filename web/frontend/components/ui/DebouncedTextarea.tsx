"use client";
import * as React from "react";

/**
 * Textarea that keeps typing local-only until the user pauses, then pushes
 * the value upward via `onChange`.  Without this, every keystroke into the
 * Context Prompt fires across Zustand → ten subscribing panels re-render →
 * a handful of API calls go out, and the textarea feels laggy under load.
 *
 * - `onChange` is called debounced (`delay` ms, default 280)
 * - immediate flush on blur, so Tab-to-next-control updates instantly
 * - external changes to `value` (preset switches, override clears, etc.)
 *   re-sync down via effect — local state never gets stuck
 */
interface Props extends Omit<React.TextareaHTMLAttributes<HTMLTextAreaElement>, "onChange" | "value"> {
  value: string;
  onChange: (next: string) => void;
  delay?: number;
}

export function DebouncedTextarea({ value, onChange, delay = 280, onBlur, ...rest }: Props) {
  const [local, setLocal] = React.useState(value);
  const timerRef = React.useRef<number | null>(null);
  const onChangeRef = React.useRef(onChange);
  React.useEffect(() => {
    onChangeRef.current = onChange;
  }, [onChange]);

  // Resync down whenever the external value moves (preset pick, morphing
  // clear, programmatic reset).
  React.useEffect(() => {
    setLocal(value);
  }, [value]);

  React.useEffect(() => {
    return () => {
      if (timerRef.current) clearTimeout(timerRef.current);
    };
  }, []);

  function handleChange(e: React.ChangeEvent<HTMLTextAreaElement>) {
    const v = e.target.value;
    setLocal(v);
    if (timerRef.current) clearTimeout(timerRef.current);
    timerRef.current = window.setTimeout(() => {
      onChangeRef.current(v);
      timerRef.current = null;
    }, delay);
  }

  function handleBlur(e: React.FocusEvent<HTMLTextAreaElement>) {
    if (timerRef.current) {
      clearTimeout(timerRef.current);
      timerRef.current = null;
    }
    if (local !== value) onChangeRef.current(local);
    onBlur?.(e);
  }

  return <textarea {...rest} value={local} onChange={handleChange} onBlur={handleBlur} />;
}
