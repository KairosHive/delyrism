"use client";
import * as React from "react";
import { useQuery } from "@tanstack/react-query";
import { api } from "@/lib/api";
import { useSidebar } from "@/lib/store";
import { SECTION_COLORS, STRUCTURE_DISPLAY } from "@/lib/theme";
import { DebouncedTextarea } from "@/components/ui/DebouncedTextarea";

interface PresetsResp { presets: string[]; }
interface PresetResp { name: string; symbols: Record<string, string[]>; }

/**
 * Two prominent cards at the top of the main area:
 *   • SYMBOLIC STRUCTURE  — preset dropdown (the primary entry point)
 *   • CONTEXT PROMPT      — sentence textarea; expands into the alchemist
 *                            morph panel (A | slider | B) when enabled
 */
export function Console() {
  const presetName = useSidebar((s) => s.presetName);
  const set = useSidebar((s) => s.set);
  const sentence = useSidebar((s) => s.contextSentence);
  const sentenceB = useSidebar((s) => s.contextSentenceB);
  const alchemistMode = useSidebar((s) => s.alchemistMode);
  const alchemistBlend = useSidebar((s) => s.alchemistBlend);
  const alchemistActive = useSidebar((s) => s.alchemistActive);
  const audioActive = useSidebar((s) => s.audioActive);
  const imageActive = useSidebar((s) => s.imageActive);
  const spaceId = useSidebar((s) => s.spaceId);
  const otherOverride = audioActive || imageActive;

  const presets = useQuery({
    queryKey: ["presets"],
    queryFn: () => api.get<PresetsResp>("/spaces/presets"),
  });

  async function pick(name: string) {
    set("presetName", name);
    const p = await api.get<PresetResp>(`/spaces/presets/${name}`);
    set("symbolMapJson", JSON.stringify(p.symbols, null, 2));
  }

  // ─── alchemist blend effect ─────────────────────────────────────────────
  // Whenever A / B / blend / mode changes (and a space exists), debounce-push
  // the new override to the backend.  Single-flight encode cache on the
  // server makes repeated calls with the same sentences cheap — only the
  // lerp changes on each slider tick.  Bump the local nonce so dependent
  // queries refetch.
  React.useEffect(() => {
    if (!spaceId) return;

    // Turning alchemist OFF — clear the override if we owned it.
    if (!alchemistMode) {
      if (alchemistActive) {
        api.post("/context/set-override", { space_id: spaceId, vector: null })
          .catch(() => {})
          .finally(() => {
            set("alchemistActive", false);
            set("alchemistNonce", Date.now());
          });
      }
      return;
    }

    // Need at least one filled sentence to do anything useful.
    if (!sentence.trim() && !sentenceB.trim()) {
      if (alchemistActive) {
        api.post("/context/set-override", { space_id: spaceId, vector: null })
          .catch(() => {})
          .finally(() => {
            set("alchemistActive", false);
            set("alchemistNonce", Date.now());
          });
      }
      return;
    }

    // Debounce so dragging the slider doesn't spam the API.
    const t = setTimeout(() => {
      api.post<{ ok: boolean; active: boolean }>("/context/set-alchemist-blend", {
        space_id: spaceId,
        sentence_a: sentence,
        sentence_b: sentenceB,
        blend: alchemistBlend,
      })
        .then((r) => {
          set("alchemistActive", !!r.active);
          set("alchemistNonce", Date.now());
          // Mutually exclusive with audio/image — taking the override slot
          // means those just got replaced server-side; sync UI state.
          const st = useSidebar.getState();
          if (st.audioActive) { set("audioActive", false); set("audioNonce", Date.now()); }
          if (st.imageActive) {
            if (st.imageThumbnail) URL.revokeObjectURL(st.imageThumbnail);
            set("imageActive", false);
            set("imageDescription", "");
            set("imageThumbnail", null);
            set("imageNonce", Date.now());
          }
        })
        .catch(() => {
          /* surfaced elsewhere in the UI on real failures */
        });
    }, 180);
    return () => clearTimeout(t);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [spaceId, alchemistMode, sentence, sentenceB, alchemistBlend]);

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
            {alchemistMode ? "Context Prompt · Alchemist mode" : "Context Prompt"}
          </div>
          <div className="flex items-center gap-1.5">
            {audioActive && (
              <span className="pill !text-[10px] border-accent-500/60 bg-accent-600/15 text-accent-200">
                ♪ audio override
              </span>
            )}
            {imageActive && (
              <span className="pill !text-[10px] border-accent-500/60 bg-accent-600/15 text-accent-200">
                🖼 image override
              </span>
            )}
            <button
              type="button"
              className={`pill !text-[10px] transition ${
                alchemistMode
                  ? "border-accent-500/60 bg-accent-600/25 text-accent-100"
                  : "hover:border-ink-500 hover:text-ink-100"
              }`}
              onClick={() => set("alchemistMode", !alchemistMode)}
              title="Blend two contexts A↔B with a single slider — every panel updates live"
            >
              {alchemistMode ? "● alchemist on" : "+ alchemist"}
            </button>
          </div>
        </div>

        {!alchemistMode ? (
          <DebouncedTextarea
            className={`h-[68px] w-full resize-none rounded-lg border border-ink-700 bg-ink-900/70 px-4 py-3 text-base text-ink-100
                       placeholder:text-ink-500 focus:border-accent-500 focus:outline-none focus:ring-1 focus:ring-accent-500/40
                       ${otherOverride ? "opacity-50" : ""}`}
            placeholder="e.g. flooding spirits dancing around floating suns…"
            value={sentence}
            onChange={(v) => set("contextSentence", v)}
            disabled={otherOverride}
          />
        ) : (
          <AlchemistMorph
            sentence={sentence}
            sentenceB={sentenceB}
            blend={alchemistBlend}
            onA={(v) => set("contextSentence", v)}
            onB={(v) => set("contextSentenceB", v)}
            onBlend={(v) => set("alchemistBlend", v)}
          />
        )}
      </div>
    </div>
  );
}

function AlchemistMorph({
  sentence, sentenceB, blend, onA, onB, onBlend,
}: {
  sentence: string;
  sentenceB: string;
  blend: number;
  onA: (v: string) => void;
  onB: (v: string) => void;
  onBlend: (v: number) => void;
}) {
  // Percentage shown on the slider thumb — handy reference when sweeping.
  const pct = Math.round(blend * 100);
  return (
    <div className="space-y-2">
      <DebouncedTextarea
        className="h-[56px] w-full resize-none rounded-lg border border-ink-700 bg-ink-900/70 px-3 py-2 text-sm text-ink-100
                   placeholder:text-ink-500 focus:border-accent-500 focus:outline-none focus:ring-1 focus:ring-accent-500/40"
        placeholder="context A — e.g. a quiet grief"
        value={sentence}
        onChange={onA}
      />

      <div className="flex items-center gap-2 px-0.5">
        <span className="text-[10px] font-medium uppercase tracking-wider text-ink-400">A</span>
        <input
          type="range"
          min={0}
          max={100}
          step={1}
          value={pct}
          onChange={(e) => onBlend(Number(e.target.value) / 100)}
          className="alchemist-slider w-full"
          aria-label="blend A to B"
        />
        <span className="text-[10px] font-medium uppercase tracking-wider text-ink-400">B</span>
        <span className="ml-1 w-10 text-right font-mono text-[10px] text-ink-300">
          {pct < 10 ? "0" : ""}{pct}%
        </span>
      </div>

      <DebouncedTextarea
        className="h-[56px] w-full resize-none rounded-lg border border-ink-700 bg-ink-900/70 px-3 py-2 text-sm text-ink-100
                   placeholder:text-ink-500 focus:border-accent-500 focus:outline-none focus:ring-1 focus:ring-accent-500/40"
        placeholder="context B — e.g. a slow rage"
        value={sentenceB}
        onChange={onB}
      />
    </div>
  );
}
