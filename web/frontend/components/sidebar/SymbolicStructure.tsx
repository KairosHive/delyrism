"use client";
import * as React from "react";
import { useQuery } from "@tanstack/react-query";
import { api } from "@/lib/api";
import { Section } from "../ui/Section";
import { HelpTip } from "../ui/HelpTip";
import { useSidebar } from "@/lib/store";
import { SECTION_COLORS, STRUCTURE_DISPLAY } from "@/lib/theme";
import { SymbolGenerator } from "./SymbolGenerator";
import { LOCAL_PREFIX, deleteSet, isLocal, localId, resolvePreset, useSavedSets } from "@/lib/symbolSets";

interface PresetsResp { presets: string[]; }

export function SymbolicStructure() {
  const presetName = useSidebar((s) => s.presetName);
  const symbolMapJson = useSidebar((s) => s.symbolMapJson);
  const set = useSidebar((s) => s.set);

  const presets = useQuery({
    queryKey: ["presets"],
    queryFn: () => api.get<PresetsResp>("/spaces/presets"),
  });
  const saved = useSavedSets();

  React.useEffect(() => {
    if (!presetName || symbolMapJson) return;
    resolvePreset(presetName)
      .then((symbols) => set("symbolMapJson", JSON.stringify(symbols, null, 2)))
      .catch(() => {});
  }, [presetName, symbolMapJson, set]);

  async function loadPreset(name: string) {
    set("presetName", name);
    const symbols = await resolvePreset(name);
    set("symbolMapJson", JSON.stringify(symbols, null, 2));
  }

  function removeSaved() {
    if (!isLocal(presetName)) return;
    const s = saved.find((x) => x.id === localId(presetName!));
    if (!s || !confirm(`Delete the saved set “${s.name}”?`)) return;
    deleteSet(s.id);
    set("presetName", null);
  }

  async function loadFile(file: File) {
    const text = await file.text();
    try {
      JSON.parse(text);
      set("symbolMapJson", text);
      set("presetName", null);
    } catch {
      alert("Not valid JSON");
    }
  }

  function downloadJson() {
    const blob = new Blob([symbolMapJson], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `${isLocal(presetName) ? localId(presetName!) : presetName || "symbols"}.json`;
    a.click();
    URL.revokeObjectURL(url);
  }

  return (
    <Section title="Symbolic Structure" color={SECTION_COLORS.symbolic} defaultOpen={false}>
      <div className="grid grid-cols-2 gap-2">
        <div className="space-y-1">
          <div className="flex items-center label-sm">
            Preset
            <HelpTip text="A built-in {symbol: [descriptors]} map. Switch to load a different archetypal system. Mine your own via the Archetype Builder tab." />
          </div>
          <select
            className="select-base"
            value={presetName ?? "(custom)"}
            onChange={(e) => loadPreset(e.target.value)}
            disabled={!presets.data}
          >
            {!presets.data && <option>Loading…</option>}
            {presets.data?.presets.map((p) => (
              <option key={p} value={p}>
                {STRUCTURE_DISPLAY[p] ?? p}
              </option>
            ))}
            {saved.length > 0 && (
              <optgroup label="my sets">
                {saved.map((s) => (
                  <option key={s.id} value={LOCAL_PREFIX + s.id}>✦  {s.name}</option>
                ))}
              </optgroup>
            )}
            {!presetName && <option value="(custom)">(custom)</option>}
          </select>
        </div>
        <div className="flex items-end gap-2">
          <label className="btn flex-1 cursor-pointer">
            Upload JSON
            <input
              type="file"
              accept="application/json,.json"
              hidden
              onChange={(e) => {
                const f = e.target.files?.[0];
                if (f) loadFile(f);
              }}
            />
          </label>
          <button className="btn" onClick={downloadJson} title="Download the current symbol map as JSON">
            ↓
          </button>
          {isLocal(presetName) && (
            <button className="btn" onClick={removeSaved} title="Delete this saved set from the browser">
              ✕
            </button>
          )}
          <HelpTip text="Upload your own symbol map (JSON of {symbol: [descriptors...]}). Download exports the currently-loaded map — useful for backups or sharing." />
        </div>
      </div>

      <details>
        <summary className="cursor-pointer text-xs text-ink-300 hover:text-ink-100">
          Edit JSON directly
        </summary>
        <textarea
          className="input-base mt-2 h-40 font-mono text-xs"
          value={symbolMapJson}
          onChange={(e) => set("symbolMapJson", e.target.value)}
        />
      </details>

      <SymbolGenerator />
    </Section>
  );
}
