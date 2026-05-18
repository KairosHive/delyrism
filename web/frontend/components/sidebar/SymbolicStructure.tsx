"use client";
import * as React from "react";
import { useQuery } from "@tanstack/react-query";
import { api } from "@/lib/api";
import { Section } from "../ui/Section";
import { useSidebar } from "@/lib/store";
import { SECTION_COLORS, STRUCTURE_DISPLAY } from "@/lib/theme";

interface PresetsResp { presets: string[]; }
interface PresetResp { name: string; symbols: Record<string, string[]>; }

export function SymbolicStructure() {
  const presetName = useSidebar((s) => s.presetName);
  const symbolMapJson = useSidebar((s) => s.symbolMapJson);
  const set = useSidebar((s) => s.set);

  const presets = useQuery({
    queryKey: ["presets"],
    queryFn: () => api.get<PresetsResp>("/spaces/presets"),
  });

  React.useEffect(() => {
    if (!presetName || symbolMapJson) return;
    api
      .get<PresetResp>(`/spaces/presets/${presetName}`)
      .then((p) => set("symbolMapJson", JSON.stringify(p.symbols, null, 2)))
      .catch(() => {});
  }, [presetName, symbolMapJson, set]);

  async function loadPreset(name: string) {
    set("presetName", name);
    const p = await api.get<PresetResp>(`/spaces/presets/${name}`);
    set("symbolMapJson", JSON.stringify(p.symbols, null, 2));
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
    a.download = `${presetName || "symbols"}.json`;
    a.click();
    URL.revokeObjectURL(url);
  }

  return (
    <Section title="Symbolic Structure" color={SECTION_COLORS.symbolic} defaultOpen={false}>
      <div className="grid grid-cols-2 gap-2">
        <div className="space-y-1">
          <div className="label-sm">Preset</div>
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
          <button className="btn" onClick={downloadJson} title="Download current map">
            ↓
          </button>
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
    </Section>
  );
}
