"use client";
import * as React from "react";
import { useShallow } from "zustand/react/shallow";
import { Slider } from "@/components/ui/Slider";
import { Select } from "@/components/ui/Select";
import { Toggle } from "@/components/ui/Toggle";
import { useWalker } from "@/lib/realtime";
import { useSidebar } from "@/lib/store";

/** Progressive sliders + the sentence context.  Every change streams to the
 *  walker immediately (coalesced at ~30 Hz) so dragging a slider while the
 *  walk is playing reshapes it live. */
export function WalkerControls() {
  const p = useWalker((s) => s.params);
  const { setParams, tempo, setTempo, bandwidth, setBandwidth, text, textWeight, setText, pointerWeight, setPointerWeight, mode, clearContext } =
    useWalker(useShallow((s) => ({
      setParams: s.setParams, tempo: s.tempo, setTempo: s.setTempo, bandwidth: s.bandwidth, setBandwidth: s.setBandwidth,
      text: s.text, textWeight: s.textWeight, setText: s.setText, pointerWeight: s.pointerWeight, setPointerWeight: s.setPointerWeight,
      mode: s.mode, clearContext: s.clearContext,
    })));
  const explorerSentence = useSidebar((s) => s.contextSentence);
  const [draft, setDraft] = React.useState(text);
  React.useEffect(() => { setDraft(text); }, [text]);

  if (!p) return <div className="panel-tight text-sm text-ink-400">controls appear once the walker is connected</div>;

  return (
    <div className="panel-tight space-y-4">
      <div className="section-title">Walk controls</div>

      {/* sentence context */}
      <div className="space-y-1.5">
        <div className="label-sm">sentence context {mode === "choose" ? "(primary in choose mode)" : "(blended with the pointer)"}</div>
        <form className="flex gap-2" onSubmit={(e) => { e.preventDefault(); setText(draft); }}>
          <input className="input-base" value={draft} placeholder="e.g. a slow fire under ice" onChange={(e) => setDraft(e.target.value)} />
          <button className="btn shrink-0" type="submit">set</button>
          {explorerSentence.trim() && explorerSentence.trim() !== text && (
            <button className="btn shrink-0" type="button" onClick={() => { setDraft(explorerSentence); setText(explorerSentence); }} title="use the Explorer's context prompt">
              ⤓ explorer
            </button>
          )}
          <button className="btn shrink-0" type="button" onClick={() => { setDraft(""); clearContext(); }} title="drop every context source">clear</button>
        </form>
        <div className="grid grid-cols-2 gap-3">
          <Slider label="sentence weight" value={textWeight} min={0} max={2} step={0.05} onChange={(v) => setText(text, v)} help="How much the sentence counts in the blended context vector." />
          <Slider label="pointer weight" value={pointerWeight} min={0} max={2} step={0.05} onChange={setPointerWeight} help="How much the pointer (field kernel / ring blend) counts." disabled={mode === "choose"} />
        </div>
      </div>

      <div className="grid grid-cols-2 gap-x-4 gap-y-3 md:grid-cols-3">
        <Slider label="tempo" value={tempo} min={0.25} max={8} step={0.25} format={(v) => `${v.toFixed(2)} /s`} onChange={setTempo} help="Steps per second while playing." />
        <Slider label="temperature" value={p.temperature} min={0} max={1} step={0.01} onChange={(v) => setParams({ temperature: v })} help="0 = always take the best candidate; higher = sample more freely among the top-K." />
        <Slider label="memory" value={p.memory} min={0} max={1} step={0.01} onChange={(v) => setParams({ memory: v })} help="How much earlier nodes in the walk still pull. 0 = only the current node; 1 = the whole path equally." />
        <Slider label="diversity" value={p.diversity} min={0} max={1} step={0.01} onChange={(v) => setParams({ diversity: v })} help="Penalty for staying inside the archetype you are already in — pushes the symbol to cross archetypes." />
        <Slider label="coherence" value={p.coherence} min={0} max={1} step={0.01} onChange={(v) => setParams({ coherence: v })} help="Weight of plain cosine similarity in the score — keeps the walk semantically continuous when Δ is weak." />
        <Slider label="self weight" value={p.self_weight} min={0} max={1} step={0.01} onChange={(v) => setParams({ self_weight: v })} help="Autopoiesis: blend the symbol-so-far into the context, so the walk reshapes its own field." />
        <Slider label="β shift" value={p.beta} min={0} max={3} step={0.05} onChange={(v) => setParams({ beta: v })} help="Context shift strength (same β as the Explorer's Δ-graph)." />
        <Slider label="max length" value={p.max_len} min={3} max={16} step={1} onChange={(v) => setParams({ max_len: v })} help="Seal the symbol when the walk reaches this many descriptors." />
        <Slider label="candidates" value={p.n_candidates} min={2} max={12} step={1} onChange={(v) => setParams({ n_candidates: v })} help="How many next-step options are weighed (and shown)." />
        <Slider label="kernel width" value={bandwidth} min={0.02} max={0.3} step={0.005} onChange={setBandwidth} help="Radius of the pointer's influence over the layout (field mode), and softness of the ring blend." disabled={mode === "choose"} />
        <Select label="Δ sign" value={p.sign} onChange={(v) => setParams({ sign: v as any })} options={[
          { value: "up", label: "↑ strengthening" }, { value: "down", label: "↓ weakening" }, { value: "abs", label: "|Δ| either" },
        ]} help="Follow pairs the context pulls together, pushes apart, or either." />
        <Select label="gate" value={p.gate} onChange={(v) => setParams({ gate: v as any })} options={[
          { value: "relu", label: "relu" }, { value: "cos", label: "cos" }, { value: "softmax", label: "softmax" }, { value: "uniform", label: "uniform" },
        ]} help="Per-descriptor gating of the context pull." />
        <div className="flex items-end pb-1">
          <Toggle label="auto-seal" value={p.auto_seal} onChange={(v) => setParams({ auto_seal: v })} hint="start a new symbol at max length" />
        </div>
      </div>
    </div>
  );
}
