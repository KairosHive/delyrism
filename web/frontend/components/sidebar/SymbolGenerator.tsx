"use client";
import * as React from "react";
import { useQuery } from "@tanstack/react-query";
import { api, SymbolMap } from "@/lib/api";
import { useSidebar } from "@/lib/store";
import { HelpTip } from "../ui/HelpTip";
import { LOCAL_PREFIX, getSet, isLocal, localId, saveSet } from "@/lib/symbolSets";

/**
 * Symbol sets written by a language model (server: /generate, delyrism/server/symbolgen.py),
 * the same principle as do-you-dream's GENERATE page without the drawing step.
 *
 *   describe the set  →  "write names"  →  N names (editable, or paste your own)
 *   names + theme     →  "generate"     →  ~10 descriptors a symbol, into the sidebar's map
 *
 * The result replaces the current map or joins it; "save" keeps it in this browser as a
 * selectable set.  Glyphs (how to draw each symbol) are kept with a saved set.
 */
interface GenInfo { llm: string; llm_note: string; n_descriptors: number; max_names: number; missing: string[]; available: boolean; }
interface NamesResp { names: string[]; title: string; llm: string; }
interface GenResp {
  symbols: SymbolMap;
  entries: { name: string; descriptors: string[]; glyph: string; llm: string }[];
  errors: { name: string; error: string }[];
  skipped: string[];
  llm: string;
}

const KEY = "delyrism.generator";

function parseNames(text: string): string[] {
  const out: string[] = [];
  for (const line of text.split(/\r?\n/)) {
    const n = line.replace(/^\s*(?:[-*•·]+|\d+[.)]|\(\d+\))\s*/, "").split(/\s+/).join(" ").trim()
      .replace(/^[ ,;]+|[ ,;]+$/g, "").slice(0, 40);
    if (n && !out.some((o) => o.toLowerCase() === n.toLowerCase())) out.push(n);
  }
  return out;
}

function currentMap(json: string): SymbolMap {
  try {
    const m = JSON.parse(json);
    return m && typeof m === "object" && !Array.isArray(m) ? m : {};
  } catch { return {}; }
}

export function SymbolGenerator() {
  const set = useSidebar((s) => s.set);
  const presetName = useSidebar((s) => s.presetName);
  const symbolMapJson = useSidebar((s) => s.symbolMapJson);

  const info = useQuery({ queryKey: ["generate-info"], queryFn: () => api.get<GenInfo>("/generate"), staleTime: 60_000 });

  const [description, setDescription] = React.useState("");
  const [count, setCount] = React.useState(8);
  const [names, setNames] = React.useState("");
  const [theme, setTheme] = React.useState("");
  const [mode, setMode] = React.useState<"replace" | "add">("replace");
  const [busy, setBusy] = React.useState<null | "names" | "symbols">(null);
  const [status, setStatus] = React.useState("");
  const [error, setError] = React.useState("");
  const [saveName, setSaveName] = React.useState("");
  const glyphs = React.useRef<Record<string, string>>({});

  // what was typed survives a reload
  React.useEffect(() => {
    try {
      const st = JSON.parse(localStorage.getItem(KEY) || "{}");
      if (st.description) setDescription(st.description);
      if (st.count) setCount(st.count);
      if (st.names) setNames(st.names);
      if (st.theme) setTheme(st.theme);
      if (st.mode) setMode(st.mode);
    } catch {}
  }, []);
  React.useEffect(() => {
    try { localStorage.setItem(KEY, JSON.stringify({ description, count, names, theme, mode })); } catch {}
  }, [description, count, names, theme, mode]);

  const list = parseNames(names);
  const current = currentMap(symbolMapJson);
  const have = new Set(Object.keys(current).map((k) => k.toLowerCase()));
  const fresh = mode === "add" ? list.filter((n) => !have.has(n.toLowerCase())) : list;
  const missing = info.data?.missing ?? [];
  const ready = !!info.data?.available && !busy;

  async function writeNames() {
    if (!description.trim()) return;
    setBusy("names"); setError(""); setStatus("writing names…");
    try {
      const existing = mode === "add" ? Object.keys(current) : [];
      const r = await api.post<NamesResp>("/generate/names", { description, n: count, existing });
      setNames((prev) => (mode === "add" && prev.trim() ? prev.trimEnd() + "\n" : "") + r.names.join("\n"));
      if (!theme.trim()) setTheme(description.trim());
      if (!saveName.trim() && r.title) setSaveName(r.title);
      setStatus(`${r.names.length} names by ${r.llm.split("/").pop()} — edit them, then generate`);
    } catch (e: any) {
      setError(e.message); setStatus("");
    } finally { setBusy(null); }
  }

  async function generate() {
    if (!fresh.length) return;
    setBusy("symbols"); setError("");
    setStatus(`writing ${fresh.length} symbol${fresh.length === 1 ? "" : "s"}, ~${Math.ceil(fresh.length / 4) * 4} s…`);
    try {
      const existing = mode === "add" ? current : {};
      const r = await api.post<GenResp>("/generate", { names: fresh, theme, existing });
      const map: SymbolMap = mode === "add" ? { ...current, ...r.symbols } : r.symbols;
      const g = mode === "add" ? { ...glyphs.current } : {};
      for (const e of r.entries) g[e.name.toUpperCase()] = e.glyph;
      glyphs.current = g;
      set("symbolMapJson", JSON.stringify(map, null, 2));
      // a generated map is a custom map until it is saved (an "add" into a saved set edits that set)
      if (mode === "add" && isLocal(presetName)) {
        const s = getSet(localId(presetName!));
        if (s) saveSet({ id: s.id, name: s.name, symbols: map, glyphs: { ...s.glyphs, ...g }, theme: s.theme || theme });
      } else {
        set("presetName", null);
      }
      const made = r.entries.length;
      const parts = [`${made} symbol${made === 1 ? "" : "s"} by ${r.llm.split("/").pop()}`];
      if (r.skipped.length) parts.push(`${r.skipped.length} already in the set`);
      if (r.errors.length) parts.push(`${r.errors.length} failed: ${r.errors.map((x) => x.name).join(", ")}`);
      setStatus(parts.join(" · ") + (mode === "add" && isLocal(presetName) ? " · saved" : " · now in the sidebar — build the space, or save the set"));
      setNames(r.errors.map((x) => x.name).join("\n"));
    } catch (e: any) {
      setError(e.message); setStatus("");
    } finally { setBusy(null); }
  }

  function save() {
    const symbols = currentMap(symbolMapJson);
    if (!Object.keys(symbols).length) { setError("the symbol map is empty"); return; }
    const name = saveName.trim() || theme.trim() || description.trim() || "untitled";
    const s = saveSet({ name, theme, symbols, glyphs: glyphs.current });
    set("presetName", LOCAL_PREFIX + s.id);
    setStatus(`saved as “${s.name}” — it is in the preset list now`);
    setSaveName("");
  }

  const nameCount = (() => {
    if (!list.length) return "one name per line, or write them from a description";
    const p = [`${list.length} name${list.length === 1 ? "" : "s"}`];
    if (mode === "add" && list.length > fresh.length) p.push(`${list.length - fresh.length} already in the set`);
    return p.join(" · ");
  })();

  return (
    <div className="space-y-2 border-t border-ink-700/60 pt-2.5">
      <div className="flex items-center label-sm">
        Generate a set
        <HelpTip text="A language model writes a symbol set: describe it and it names the symbols, or paste your own names; then it writes ~10 descriptors a symbol (the name first) and how to draw it. The result goes into the map above. Save it to pick it like a preset." />
        {info.data && (
          <span className="ml-auto text-[10px] text-ink-400" title={info.data.llm_note || info.data.llm}>
            {info.data.llm.split("/").pop()}
          </span>
        )}
      </div>

      {missing.length > 0 && (
        <div className="text-xs text-amber-300">not set on the server: {missing.join(", ")}</div>
      )}

      <div className="flex gap-1.5">
        <input
          className="input-base flex-1 text-xs"
          placeholder="describe the set — e.g. plants, the Greek pantheon…"
          value={description}
          onChange={(e) => setDescription(e.target.value)}
          onKeyDown={(e) => { if (e.key === "Enter") { e.preventDefault(); writeNames(); } }}
          disabled={!!busy}
        />
        <input
          type="number" min={1} max={info.data?.max_names ?? 40}
          className="input-base w-14 text-xs"
          value={count}
          onChange={(e) => setCount(Math.max(1, Math.min(info.data?.max_names ?? 40, Number(e.target.value) || 1)))}
          title="how many symbols to name"
          disabled={!!busy}
        />
        <button className="btn whitespace-nowrap" onClick={writeNames} disabled={!ready || !description.trim()}>
          {busy === "names" ? "…" : "write names"}
        </button>
      </div>

      <textarea
        className="input-base h-24 font-mono text-xs"
        placeholder={"Oak\nRose\nFern\n…"}
        value={names}
        onChange={(e) => setNames(e.target.value)}
        onKeyDown={(e) => { if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) { e.preventDefault(); generate(); } }}
        disabled={!!busy}
      />
      <div className="text-[10px] text-ink-400">{nameCount}</div>

      <div className="flex gap-1.5">
        <input
          className="input-base flex-1 text-xs"
          placeholder="theme (optional) — reads every symbol through it"
          value={theme}
          onChange={(e) => setTheme(e.target.value)}
          disabled={!!busy}
        />
        <select className="select-base w-auto text-xs" value={mode} onChange={(e) => setMode(e.target.value as any)} disabled={!!busy}
                title="replace the current map, or add the new symbols to it (their descriptors then share ground with it)">
          <option value="replace">new set</option>
          <option value="add">add to current</option>
        </select>
        <button className="btn btn-primary whitespace-nowrap" onClick={generate} disabled={!ready || !fresh.length}
                title="Ctrl+Enter in the list">
          {busy === "symbols" ? "…" : "generate"}
        </button>
      </div>

      <div className="flex gap-1.5">
        <input
          className="input-base flex-1 text-xs"
          placeholder="save the current map as a set named…"
          value={saveName}
          onChange={(e) => setSaveName(e.target.value)}
          onKeyDown={(e) => { if (e.key === "Enter") { e.preventDefault(); save(); } }}
        />
        <button className="btn" onClick={save} disabled={!!busy || !Object.keys(current).length}>save</button>
      </div>

      {status && <div className="text-xs text-accent-300">{status}</div>}
      {error && <div className="text-xs text-red-300">{error}</div>}
    </div>
  );
}
