"use client";
import * as React from "react";
import { useQuery } from "@tanstack/react-query";
import { api } from "@/lib/api";
import { useSidebar } from "@/lib/store";
import { SECTION_COLORS, STRUCTURE_DISPLAY } from "@/lib/theme";

interface PresetsResp { presets: string[]; }
interface PresetResp { name: string; symbols: Record<string, string[]>; }

/**
 * Two prominent cards at the top of the main area:
 *   • SYMBOLIC STRUCTURE  — preset dropdown (the primary entry point)
 *   • CONTEXT PROMPT      — large textarea for the context sentence
 *
 * Both are bound to the same Zustand store as the sidebar, so they stay in
 * sync if the user uses the advanced controls.
 */
export function Console() {
  const presetName = useSidebar((s) => s.presetName);
  const set = useSidebar((s) => s.set);
  const sentence = useSidebar((s) => s.contextSentence);
  const audioActive = useSidebar((s) => s.audioActive);

  const presets = useQuery({
    queryKey: ["presets"],
    queryFn: () => api.get<PresetsResp>("/spaces/presets"),
  });

  async function pick(name: string) {
    set("presetName", name);
    const p = await api.get<PresetResp>(`/spaces/presets/${name}`);
    set("symbolMapJson", JSON.stringify(p.symbols, null, 2));
  }

  return (
    <div className="mx-auto grid w-full max-w-5xl grid-cols-1 gap-4 md:grid-cols-2">
      <div
        className="console-card"
        data-accent
        style={{ boxShadow: `0 0 0 1px ${SECTION_COLORS.symbolic}33, 0 8px 28px -16px ${SECTION_COLORS.symbolic}` }}
      >
        <div className="console-label" style={{ color: SECTION_COLORS.symbolic }}>
          <span style={{ width: 8, height: 8, borderRadius: 999, background: SECTION_COLORS.symbolic }} />
          Symbolic Structure
        </div>
        <div className="relative">
          <select
            className="w-full appearance-none rounded-lg border border-ink-700 bg-ink-900/70 px-4 py-3 text-base text-ink-100
                       transition focus:border-accent-500 focus:outline-none focus:ring-1 focus:ring-accent-500/40"
            value={presetName ?? ""}
            onChange={(e) => pick(e.target.value)}
          >
            {!presets.data && <option>Loading presets…</option>}
            {presets.data?.presets.map((p) => (
              <option key={p} value={p}>
                {STRUCTURE_DISPLAY[p] ?? p}
              </option>
            ))}
          </select>
          <svg className="pointer-events-none absolute right-3 top-1/2 h-4 w-4 -translate-y-1/2 text-ink-400" viewBox="0 0 20 20" fill="currentColor">
            <path fillRule="evenodd" d="M5.23 7.21a.75.75 0 011.06.02L10 11.06l3.71-3.83a.75.75 0 011.08 1.04l-4.25 4.39a.75.75 0 01-1.08 0L5.21 8.27a.75.75 0 01.02-1.06z" clipRule="evenodd" />
          </svg>
        </div>
      </div>

      <div
        className="console-card"
        data-accent
        style={{ boxShadow: `0 0 0 1px ${SECTION_COLORS.context}33, 0 8px 28px -16px ${SECTION_COLORS.context}` }}
      >
        <div className="flex items-center justify-between gap-2">
          <div className="console-label" style={{ color: SECTION_COLORS.context }}>
            <span style={{ width: 8, height: 8, borderRadius: 999, background: SECTION_COLORS.context }} />
            Context Prompt
          </div>
          {audioActive && (
            <span className="pill !text-[10px] border-accent-500/60 bg-accent-600/15 text-accent-200">
              ♪ audio override — text is ignored
            </span>
          )}
        </div>
        <textarea
          className={`h-[68px] w-full resize-none rounded-lg border border-ink-700 bg-ink-900/70 px-4 py-3 text-base text-ink-100
                     placeholder:text-ink-500 focus:border-accent-500 focus:outline-none focus:ring-1 focus:ring-accent-500/40
                     ${audioActive ? "opacity-50" : ""}`}
          placeholder="e.g. flooding spirits dancing around floating suns…"
          value={sentence}
          onChange={(e) => set("contextSentence", e.target.value)}
          disabled={audioActive}
        />
      </div>
    </div>
  );
}
