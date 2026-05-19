"use client";
import * as React from "react";
import { Section } from "../ui/Section";
import { Slider } from "../ui/Slider";
import { Toggle } from "../ui/Toggle";
import { HelpTip } from "../ui/HelpTip";
import { useSidebar } from "@/lib/store";
import { SECTION_COLORS } from "@/lib/theme";
import { AudioContext } from "./AudioContext";
import { ImageContext } from "./ImageContext";

export function ContextOptions() {
  const symbols = useSidebar((s) => s.symbols);
  const sentence = useSidebar((s) => s.contextSentence);
  const selected = useSidebar((s) => s.selectedContextSymbols);
  const weights = useSidebar((s) => s.symbolWeights);
  const alch = useSidebar((s) => s.alchemistMode);
  const sentenceB = useSidebar((s) => s.contextSentenceB);
  const set = useSidebar((s) => s.set);
  const setWeight = useSidebar((s) => s.setWeight);

  function toggleSymbol(sym: string) {
    if (selected.includes(sym)) {
      set("selectedContextSymbols", selected.filter((x) => x !== sym));
    } else {
      set("selectedContextSymbols", [...selected, sym]);
      if (!(sym in weights)) setWeight(sym, 0.5);
    }
  }

  return (
    <Section title="Context Options" color={SECTION_COLORS.context} defaultOpen={false}>
      <div className="space-y-1">
        <div className="flex items-center label-sm">
          Context sentence
          <HelpTip text="Free-text prompt that defines the semantic context. The engine encodes this into a vector and uses it as the conditioning signal for every panel — rankings, attention, Δ-graph, etc. Same field as the big Context Prompt card in the main area." />
        </div>
        <textarea
          className="input-base h-16"
          placeholder="(or type in the main Context Prompt card)"
          value={sentence}
          onChange={(e) => set("contextSentence", e.target.value)}
        />
      </div>

      <div className="space-y-2">
        <div className="flex items-center label-sm">
          Symbol weights
          <HelpTip text="Manually bias the context toward specific archetypes. Click a symbol chip to add it, then slide its weight. The engine mixes these into the context vector alongside the sentence. Useful when you want 'fire + a touch of water' rather than guessing the right sentence." />
        </div>
        <div className="flex flex-wrap gap-1.5">
          {symbols.map((s) => {
            const active = selected.includes(s);
            return (
              <button
                key={s}
                onClick={() => toggleSymbol(s)}
                className={`pill !text-[11px] ${
                  active
                    ? "border-accent-500 bg-accent-600/30 text-ink-50"
                    : "hover:border-ink-600 hover:bg-ink-800"
                }`}
              >
                {s}
              </button>
            );
          })}
        </div>
        {selected.map((s) => (
          <Slider
            key={s}
            label={s}
            value={weights[s] ?? 0}
            min={0}
            max={1}
            step={0.05}
            onChange={(v) => setWeight(s, v)}
          />
        ))}
      </div>

      <div className="space-y-2 border-t border-ink-700/60 pt-3">
        <AudioContext />
        <ImageContext />
      </div>

      <div className="border-t border-ink-700/60 pt-3 space-y-2">
        <Toggle
          label="Alchemist mode (Context A → B)"
          help="Enable a second context (B) for side-by-side comparison. The main panels show context A; alchemist controls let you blend A and B to study trajectories between two semantic positions."
          value={alch}
          onChange={(v) => set("alchemistMode", v)}
        />
        {alch && (
          <textarea
            className="input-base h-16"
            placeholder="Context B sentence"
            value={sentenceB}
            onChange={(e) => set("contextSentenceB", e.target.value)}
          />
        )}
      </div>
    </Section>
  );
}
