"use client";
import * as React from "react";
import { Section } from "../ui/Section";
import { Slider } from "../ui/Slider";
import { Toggle } from "../ui/Toggle";
import { useSidebar } from "@/lib/store";
import { SECTION_COLORS } from "@/lib/theme";

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
        <div className="label-sm">Context sentence</div>
        <textarea
          className="input-base h-16"
          placeholder="(or type in the main Context Prompt card)"
          value={sentence}
          onChange={(e) => set("contextSentence", e.target.value)}
        />
      </div>

      <div className="space-y-2">
        <div className="label-sm">Symbol weights</div>
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

      <div className="border-t border-ink-700/60 pt-3 space-y-2">
        <Toggle
          label="Alchemist mode (Context A → B)"
          hint="Compare two contexts side by side"
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
