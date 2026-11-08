"use client";
import * as React from "react";
import { useQuery } from "@tanstack/react-query";
import { api, SymbolMap } from "@/lib/api";
import { useSidebar } from "@/lib/store";
import { HelpTip } from "@/components/ui/HelpTip";
import { Glyph } from "./Glyph";
import {
  ArtMap, LOCAL_PREFIX, SymbolArt, deleteSet, fileToSet, getSet, isLocal, localId, saveSet, setToFile, slug,
  useSavedSets,
} from "@/lib/symbolSets";
import { STRUCTURE_DISPLAY } from "@/lib/theme";

/**
 * STRUCTURES: the current symbol set with its drawings and descriptors, to look at, organize and
 * edit; and the generator that writes and draws new ones, the same pipeline as do-you-dream's
 * GENERATE page (server: /generate, delyrism/server/symbolgen.py).
 *
 *   describe the set  →  write names  →  N names (editable), or paste your own
 *   names + theme     →  generate     →  per name: descriptors + glyph (LLM), then a drawing
 *                                        (FLUX.2), cleaned and vectorized; a few at a time
 *
 * The set is the sidebar's symbol map (every tab reads it) plus `symbolArt` for the drawings.
 * Cards can be reordered by dragging, renamed, their descriptors edited, redrawn, rewritten or
 * removed; "save" keeps the set in this browser, where it appears under "my sets" in the preset
 * lists (and a selected saved set takes every edit at once).  The server keeps nothing.
 */
interface GenInfo {
  llm: string; llm_note: string; n_descriptors: number; max_names: number; missing: string[]; available: boolean;
  image_models: { id: string; label: string; hint: string }[]; default_model: string; drawing: boolean;
}
interface NamesResp { names: string[]; title: string; llm: string; }
interface EntryResp { name: string; descriptors: string[]; glyph: string; llm: string; }
interface DrawResp {
  photo: string; svg: string; lines_svg: string; geometry: unknown; seed: number; model: string;
  image_prompt: string; glyph: string; at: string;
}

type Step = "writing" | "drawing" | "error";
interface Job { name: string; key: string; step: Step; error?: string; descriptors?: string[]; glyph?: string; }

const KEY = "delyrism.generate";
const WORKERS = 3;
// do-you-dream's palette: earthy tones that hold up on an off-black ground, one per symbol
const PALETTE = ["#E2A653", "#C4613F", "#93A66B", "#E3D6BC", "#B5793F", "#6E9C8F", "#A85A6A", "#8494AA",
                 "#D6977B", "#617F4A", "#D2C27A", "#457D88", "#8E6A4C", "#9A8BAA", "#9C4632", "#B7BE9A"];

function parseNames(text: string): string[] {
  const out: string[] = [];
  for (const line of text.split(/\r?\n/)) {
    const n = line.replace(/^\s*(?:[-*•·]+|\d+[.)]|\(\d+\))\s*/, "").split(/\s+/).join(" ").trim()
      .replace(/^[ ,;]+|[ ,;]+$/g, "").slice(0, 40);
    if (n && !out.some((o) => o.toLowerCase() === n.toLowerCase())) out.push(n);
  }
  return out;
}
function parseMap(json: string): SymbolMap {
  try {
    const m = JSON.parse(json);
    return m && typeof m === "object" && !Array.isArray(m) ? m : {};
  } catch { return {}; }
}
const keyOf = (name: string) => name.trim().toUpperCase();
const cleanDescriptors = (text: string) =>
  Array.from(new Set(text.split(/[\n,;]+/).map((d) => d.trim().toLowerCase().replace(/\s+/g, " ")).filter(Boolean)));

export function StructuresTab() {
  const set = useSidebar((s) => s.set);
  const presetName = useSidebar((s) => s.presetName);
  const symbolMapJson = useSidebar((s) => s.symbolMapJson);
  const art = useSidebar((s) => s.symbolArt);
  const colorMap = useSidebar((s) => s.colorMap);
  const saved = useSavedSets();

  const info = useQuery({ queryKey: ["generate-info"], queryFn: () => api.get<GenInfo>("/generate"), staleTime: 60_000 });

  const [description, setDescription] = React.useState("");
  const [count, setCount] = React.useState(8);
  const [names, setNames] = React.useState("");
  const [theme, setTheme] = React.useState("");
  const [mode, setMode] = React.useState<"replace" | "add">("add");
  const [model, setModel] = React.useState<string>("");
  const [draw, setDraw] = React.useState(true);
  const [view, setView] = React.useState<"vector" | "ink">("vector");
  const [naming, setNaming] = React.useState(false);
  const [jobs, setJobs] = React.useState<Record<string, Job>>({});
  const [status, setStatus] = React.useState("");
  const [error, setError] = React.useState("");
  const [saveName, setSaveName] = React.useState("");
  const [editing, setEditing] = React.useState<string | null>(null);
  const [dragging, setDragging] = React.useState<string | null>(null);
  const fileInput = React.useRef<HTMLInputElement>(null);
  const themeTouched = React.useRef(false);      // the theme was typed by hand: "write names" leaves it

  // what was typed survives a reload
  React.useEffect(() => {
    try {
      const st = JSON.parse(localStorage.getItem(KEY) || "{}");
      if (st.description) setDescription(st.description);
      if (st.count) setCount(st.count);
      if (st.names) setNames(st.names);
      if (st.theme) setTheme(st.theme);
      if (st.mode) setMode(st.mode);
      if (st.model) setModel(st.model);
      if (typeof st.draw === "boolean") setDraw(st.draw);
      if (st.view) setView(st.view);
    } catch {}
  }, []);
  React.useEffect(() => {
    try { localStorage.setItem(KEY, JSON.stringify({ description, count, names, theme, mode, model, draw, view })); } catch {}
  }, [description, count, names, theme, mode, model, draw, view]);
  React.useEffect(() => {
    if (info.data && !info.data.image_models.some((m) => m.id === model)) setModel(info.data.default_model);
  }, [info.data, model]);

  const map = parseMap(symbolMapJson);
  const keys = Object.keys(map);
  const list = parseNames(names);
  const have = new Set(keys.map(keyOf));
  const running = Object.values(jobs).filter((j) => j.step !== "error");
  const fresh = (mode === "add" ? list.filter((n) => !have.has(keyOf(n))) : list)
    .filter((n) => !jobs[keyOf(n)] || jobs[keyOf(n)].step === "error");
  const missing = info.data?.missing ?? [];
  const canDraw = !!info.data?.drawing;
  const ready = !!info.data?.available;
  const current = isLocal(presetName) ? getSet(localId(presetName!)) : undefined;
  const undrawn = keys.filter((k) => !art[k]?.svg && !(jobs[k] && jobs[k].step !== "error"));
  const color = (k: string) => colorMap[k] ?? PALETTE[keys.indexOf(k) % PALETTE.length] ?? "#888";
  const setTitle = current ? `✦ ${current.name}` : presetName ? (STRUCTURE_DISPLAY[presetName] ?? presetName) : "a custom set";

  // ---------------------------------------------------------------- the map and the art
  /** Apply a change to the symbol map and its art; a saved set that is selected is updated in place. */
  function commit(fn: (m: SymbolMap, a: ArtMap) => void, note?: string) {
    const st = useSidebar.getState();
    const m = parseMap(st.symbolMapJson);
    const a = { ...st.symbolArt };
    fn(m, a);
    set("symbolMapJson", JSON.stringify(m, null, 2));
    set("symbolArt", a);
    if (isLocal(st.presetName)) {
      const s = getSet(localId(st.presetName!));
      if (s) {
        try { saveSet({ id: s.id, name: s.name, symbols: m, art: a, theme: s.theme || theme }); }
        catch (e: any) { setError(e.message); }
      }
    } else if (st.presetName) {
      set("presetName", null);           // a shipped preset, changed: now a custom map (save keeps it)
    }
    if (note) setStatus(note);
  }
  const setJob = (key: string, j: Job | null) =>
    setJobs((prev) => { const n = { ...prev }; if (j) n[key] = j; else delete n[key]; return n; });

  // ---------------------------------------------------------------- 1. names
  async function writeNames() {
    if (!description.trim() || naming) return;
    setNaming(true); setError(""); setStatus("writing names…");
    try {
      const existing = mode === "add" ? keys : [];
      const r = await api.post<NamesResp>("/generate/names", { description, n: count, existing });
      // the list is replaced: names typed since the last run that are not in the set yet are kept in front
      setNames((prev) => {
        const kept = parseNames(prev).filter((x) => !have.has(keyOf(x)) && !r.names.some((y) => keyOf(y) === keyOf(x)));
        return kept.concat(r.names).join("\n");
      });
      // the theme follows the description unless it was typed by hand for this set
      if (!themeTouched.current || !theme.trim()) { setTheme(description.trim()); themeTouched.current = false; }
      if (!saveName.trim() && !current && r.title) setSaveName(r.title);
      setStatus(`${r.names.length} names by ${r.llm.split("/").pop()} — edit them, then generate`);
    } catch (e: any) {
      setError(e.message); setStatus("");
    } finally { setNaming(false); }
  }

  // ---------------------------------------------------------------- 2 + 3. entries and drawings, a few at a time
  async function generate() {
    if (!fresh.length || naming) return;
    setError("");
    const batch = fresh.slice();
    const others = () => Object.keys(parseMap(useSidebar.getState().symbolMapJson)).concat(batch);
    if (mode === "replace") {
      set("presetName", null);
      commit((m, a) => { for (const k of Object.keys(m)) { delete m[k]; delete a[k]; } });
    }
    setNames(list.filter((n) => !batch.includes(n)).join("\n"));
    const queue = batch.slice();
    let made = 0, failed = 0;
    const worker = async () => {
      for (;;) {
        const name = queue.shift();
        if (!name) return;
        const key = keyOf(name);
        setJob(key, { name, key, step: "writing" });
        try {
          const e = await api.post<EntryResp>("/generate/entry", { name, theme, others: others() });
          commit((m, a) => { m[key] = e.descriptors; a[key] = { ...(a[key] ?? {}), glyph: e.glyph, llm: e.llm }; });
          if (draw && canDraw) {
            setJob(key, { name, key, step: "drawing", descriptors: e.descriptors, glyph: e.glyph });
            const d = await api.post<DrawResp>("/generate/draw", { name, descriptors: e.descriptors, glyph: e.glyph, model });
            commit((m, a) => { a[key] = { ...(a[key] ?? {}), ...artOf(d) }; });
          }
          made++;
          setJob(key, null);
        } catch (err: any) {
          failed++;
          setJob(key, { name, key, step: "error", error: err.message });
        }
      }
    };
    await Promise.all(Array.from({ length: Math.min(WORKERS, batch.length) }, worker));
    const parts = [`${made} symbol${made === 1 ? "" : "s"} made`];
    if (failed) parts.push(`${failed} failed (retry from the card)`);
    parts.push(isLocal(useSidebar.getState().presetName) ? "saved" : "now in the sidebar — build the space, or save the set");
    setStatus(parts.join(" · "));
  }
  const artOf = (d: DrawResp): SymbolArt => ({
    svg: d.svg, lines_svg: d.lines_svg, photo: d.photo, geometry: d.geometry, seed: d.seed, model: d.model,
    image_prompt: d.image_prompt, glyph: d.glyph, at: d.at,
  });

  // ---------------------------------------------------------------- a symbol again
  /** redraw: a new drawing from the glyph (new seed); rewrite: new descriptors and glyph, then redraw;
   *  draw: a first drawing for a symbol that has none (a preset's, a pasted one). */
  async function redo(key: string, kind: "redraw" | "rewrite" | "draw") {
    if (jobs[key] && jobs[key].step !== "error") return;
    const name = key.charAt(0) + key.slice(1).toLowerCase();
    setError("");
    try {
      let descriptors = map[key] ?? [];
      let glyph = art[key]?.glyph ?? "";
      if (kind === "rewrite") {
        setJob(key, { name, key, step: "writing" });
        const e = await api.post<EntryResp>("/generate/entry", { name, theme, others: keys });
        descriptors = e.descriptors; glyph = e.glyph;
        commit((m, a) => { m[key] = descriptors; a[key] = { ...(a[key] ?? {}), glyph, llm: e.llm }; });
      }
      if (canDraw) {
        setJob(key, { name, key, step: "drawing", descriptors, glyph });
        const d = await api.post<DrawResp>("/generate/draw", { name, descriptors, glyph, model });
        commit((m, a) => { a[key] = { ...(a[key] ?? {}), ...artOf(d) }; });
      }
      setJob(key, null);
    } catch (err: any) {
      setJob(key, { name, key, step: "error", error: err.message });
    }
  }
  /** Draw every symbol that has no drawing yet (a shipped preset without art, a pasted map). */
  async function drawAll() {
    const todo = undrawn.slice();
    const worker = async () => { for (;;) { const k = todo.shift(); if (!k) return; await redo(k, "draw"); } };
    await Promise.all(Array.from({ length: Math.min(WORKERS, todo.length) }, worker));
  }
  function remove(key: string) {
    if (!confirm(`Remove ${key} from the set?`)) return;
    setJob(key, null);
    commit((m, a) => { delete m[key]; delete a[key]; }, `${key} removed`);
  }
  function retry(key: string) {
    const j = jobs[key];
    if (!j) return;
    setJob(key, null);
    if (map[key]) redo(key, art[key]?.svg ? "redraw" : "draw");
    else setNames((prev) => (prev.trim() ? prev.trimEnd() + "\n" : "") + j.name);
  }

  // ---------------------------------------------------------------- editing: rename, descriptors, order
  function rename(key: string, name: string, descriptors: string[]) {
    const nk = keyOf(name);
    if (!nk) { setError("a symbol needs a name"); return; }
    if (nk !== key && map[nk]) { setError(`${nk} is already in the set`); return; }
    if (!descriptors.length) { setError("a symbol needs at least one descriptor"); return; }
    commit((m, a) => {
      const entries = Object.entries(m).map(([k, v]) => [k === key ? nk : k, k === key ? descriptors : v] as const);
      for (const k of Object.keys(m)) delete m[k];
      for (const [k, v] of entries) m[k] = v as string[];
      if (nk !== key) { a[nk] = a[key] ?? {}; delete a[key]; }
    }, nk !== key ? `${key} is now ${nk}` : `${key} edited`);
    setEditing(null);
  }
  function move(key: string, before: string | null) {
    if (key === before) return;
    commit((m) => {
      const order = Object.keys(m).filter((k) => k !== key);
      const i = before ? order.indexOf(before) : order.length;
      order.splice(i < 0 ? order.length : i, 0, key);
      const copy = { ...m };
      for (const k of Object.keys(m)) delete m[k];
      for (const k of order) m[k] = copy[k];
    });
  }

  // ---------------------------------------------------------------- the set as a whole
  function save() {
    if (!keys.length) { setError("the symbol map is empty"); return; }
    const name = saveName.trim() || current?.name || theme.trim() || description.trim() || "untitled";
    try {
      const s = saveSet({ id: current?.id, name, theme: theme || current?.theme || "", symbols: map, art });
      set("presetName", LOCAL_PREFIX + s.id);
      setStatus(`saved as “${s.name}” — it is in the preset lists now`);
      setSaveName("");
    } catch (e: any) { setError(e.message); }
  }
  function saveCopy() {
    if (!keys.length) return;
    const name = saveName.trim() || theme.trim() || `${current?.name || setTitle.replace(/^[^\w]+/, "")} copy`;
    try {
      const s = saveSet({ name, theme: theme || current?.theme || "", symbols: map, art });
      set("presetName", LOCAL_PREFIX + s.id);
      setStatus(`saved a copy as “${s.name}”`);
      setSaveName("");
    } catch (e: any) { setError(e.message); }
  }
  /** An empty custom map to generate or paste into; whatever was selected stays as it is. */
  function startEmpty() {
    if (running.length) return;
    set("presetName", null);
    set("symbolMapJson", JSON.stringify({}, null, 2));
    set("symbolArt", {});
    setMode("add");
    setSaveName("");
    setTheme("");
    themeTouched.current = false;
    setStatus("an empty set — generate or paste symbols into it, then save it under a name");
  }
  function removeSaved() {
    if (!current || !confirm(`Delete the saved set “${current.name}”?`)) return;
    deleteSet(current.id);
    set("presetName", null);
    setStatus(`“${current.name}” deleted (the symbols stay in the sidebar until you pick another set)`);
  }
  function exportFile() {
    const name = current?.name || saveName.trim() || (presetName && STRUCTURE_DISPLAY[presetName]?.replace(/^\S+\s+/, "")) || theme.trim() || "symbols";
    const blob = new Blob([JSON.stringify(setToFile({ name, theme: theme || current?.theme || "", symbols: map, art }), null, 1)],
                          { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url; a.download = `${slug(name)}.delyrism-set.json`; a.click();
    URL.revokeObjectURL(url);
  }
  async function importFile(file: File) {
    try {
      const f = fileToSet(await file.text());
      set("presetName", null);
      set("symbolMapJson", JSON.stringify(f.symbols, null, 2));
      set("symbolArt", f.art);
      if (f.theme) setTheme(f.theme);
      if (f.name) setSaveName(f.name);
      setStatus(`${Object.keys(f.symbols).length} symbols from ${file.name} — save to keep them`);
    } catch (e: any) { setError(`not a symbol set: ${e.message}`); }
  }

  // ---------------------------------------------------------------- render
  const nameCount = (() => {
    if (!list.length) return "one name per line, or write them from a description";
    const p = [`${list.length} name${list.length === 1 ? "" : "s"}`];
    if (mode === "add" && list.length > fresh.length) p.push(`${list.length - fresh.length} already in the set`);
    return p.join(" · ");
  })();
  const modelInfo = info.data?.image_models.find((m) => m.id === model);
  const pending = Object.values(jobs).filter((j) => !map[j.key]);     // not in the map yet: writing, or failed before the entry

  return (
    <div className="grid grid-cols-1 gap-4 xl:grid-cols-[1fr,360px]">
      {/* ---------------- the symbols */}
      <div className="panel p-0 overflow-hidden xl:order-1">
        <div className="flex flex-wrap items-center justify-between gap-2 border-b border-ink-700/60 px-4 py-2.5">
          <div className="section-title">
            {setTitle}
            <span className="ml-2 font-normal text-ink-500">
              {keys.length ? `${keys.length} symbol${keys.length === 1 ? "" : "s"}` : "no symbols yet"}
              {running.length ? ` · ${running.length} on the way` : ""}
            </span>
          </div>
          <div className="flex items-center gap-1">
            {canDraw && undrawn.length > 0 && !running.length && (
              <button type="button" className="btn !py-0.5 text-[11px]" onClick={drawAll}
                      title="draw every symbol that has no drawing, from its name and descriptors">
                draw {undrawn.length === keys.length ? "all" : `${undrawn.length} undrawn`}
              </button>
            )}
            {(["vector", "ink"] as const).map((v) => (
              <button key={v} type="button"
                      className={`pill !text-[10px] ${v === view ? "border-accent-500/60 bg-accent-600/25 text-accent-100" : "hover:border-ink-500"}`}
                      onClick={() => setView(v)}>{v}</button>
            ))}
            <HelpTip text="The current set, as the sidebar's map holds it. Drag a card to reorder, edit a name or its descriptors, redraw (a new drawing from the same glyph), rewrite (new descriptors and glyph, then a drawing) or remove. A shipped preset becomes a custom map once changed; save keeps it as a set of yours." />
          </div>
        </div>
        {!keys.length && !pending.length ? (
          <div className="p-6 text-sm text-ink-300">
            No symbols. Pick a structure in the sidebar, or describe a set and write its names on the right.
          </div>
        ) : (
          <div className="grid grid-cols-2 gap-3 p-4 sm:grid-cols-3 lg:grid-cols-4 2xl:grid-cols-5"
               onDragOver={(e) => { if (dragging) e.preventDefault(); }}
               onDrop={(e) => { e.preventDefault(); if (dragging) move(dragging, null); setDragging(null); }}>
            {keys.map((k) => (
              <Card key={k} name={k} descriptors={map[k]} art={art[k]} color={color(k)} job={jobs[k]} view={view}
                    canDraw={canDraw} editing={editing === k}
                    onEdit={() => { setEditing(editing === k ? null : k); setError(""); }}
                    onSave={(n, d) => rename(k, n, d)}
                    onRedo={(kind) => redo(k, kind)} onRemove={() => remove(k)} onRetry={() => retry(k)}
                    dragging={dragging === k}
                    onDragStart={() => setDragging(k)} onDragEnd={() => setDragging(null)}
                    onDropBefore={() => { if (dragging) move(dragging, k); setDragging(null); }} />
            ))}
            {pending.map((j) => (
              <Card key={"pending-" + j.key} name={j.name} descriptors={j.descriptors ?? []} color="#666" job={j} view={view}
                    canDraw={canDraw} pending onRetry={() => retry(j.key)} onRemove={() => setJob(j.key, null)} />
            ))}
          </div>
        )}
      </div>

      {/* ---------------- the generator and the set */}
      <div className="space-y-3 xl:order-2">
        <div className="panel-pad space-y-2.5">
          <div className="flex items-center label-sm">
            Generate symbols
            <HelpTip text="A language model writes the symbols: describe a set and it names them, or paste your own names. Then per name it writes ~10 descriptors (the name first) and a glyph (how to draw it in a few basic shapes), and FLUX.2 draws the glyph as a black marker sign, cleaned and vectorized. 'add' puts them into the current set (their descriptors share ground with it); 'new set' starts from nothing." />
            {info.data && (
              <span className="ml-auto text-[10px] text-ink-400" title={info.data.llm_note || info.data.llm}>
                {info.data.llm.split("/").pop()}
              </span>
            )}
          </div>
          {missing.length > 0 && <div className="text-xs text-amber-300">not set on the server: {missing.join(", ")}</div>}
          {info.data && ready && !canDraw && (
            <div className="text-xs text-amber-300">the server cannot draw (no vectorizer): symbols are written without drawings</div>
          )}

          <div className="flex gap-1.5">
            <input
              className="input-base flex-1 text-xs"
              placeholder="describe a set — e.g. plants, the Greek pantheon…"
              value={description}
              onChange={(e) => setDescription(e.target.value)}
              onKeyDown={(e) => { if (e.key === "Enter") { e.preventDefault(); writeNames(); } }}
              disabled={naming}
            />
            <input
              type="number" min={1} max={info.data?.max_names ?? 40}
              className="input-base w-14 text-xs"
              value={count}
              onChange={(e) => setCount(Math.max(1, Math.min(info.data?.max_names ?? 40, Number(e.target.value) || 1)))}
              title="how many symbols to name"
              disabled={naming}
            />
            <button className="btn whitespace-nowrap" onClick={writeNames} disabled={!ready || naming || !description.trim()}>
              {naming ? "…" : "write names"}
            </button>
          </div>

          <textarea
            className="input-base h-32 font-mono text-xs"
            placeholder={"Oak\nRose\nFern\n…"}
            value={names}
            onChange={(e) => setNames(e.target.value)}
            onKeyDown={(e) => { if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) { e.preventDefault(); generate(); } }}
          />
          <div className="text-[10px] text-ink-400">{nameCount}</div>

          <input
            className="input-base text-xs"
            placeholder="theme (optional) — every symbol is read through it"
            title="the descriptors and glyph of every symbol are written through this; 'write names' fills it with the description unless you typed it"
            value={theme}
            onChange={(e) => { setTheme(e.target.value); themeTouched.current = true; }}
          />

          <div className="space-y-1">
            <div className="grid grid-cols-2 gap-1 rounded-md border border-ink-700 p-0.5">
              {([["add", "into the current set"], ["replace", "a new set"]] as const).map(([m, label]) => (
                <button key={m} type="button"
                        className={`rounded px-2 py-1 text-xs transition ${mode === m ? "bg-accent-600/30 text-accent-100" : "text-ink-300 hover:bg-ink-800 hover:text-ink-100"}`}
                        onClick={() => setMode(m)}
                        title={m === "add" ? "the new symbols join the current set; their descriptors share ground with it; names already there are skipped"
                                           : "the current set is left as it is; the new symbols start a set of their own, which you can save under a new name"}>
                  {label}
                </button>
              ))}
            </div>
            <div className="text-[10px] text-ink-500">
              {mode === "add"
                ? <>the symbols join <span className="text-ink-300">{setTitle}</span>{current ? " (saved as they come)" : ""}</>
                : <>{current ? `“${current.name}” stays as it is; ` : ""}the symbols start a new set — save it under a name afterwards</>}
            </div>
          </div>
          <label className="flex items-center gap-1.5 text-xs text-ink-300" title={canDraw ? "draw each symbol with FLUX.2 and vectorize it" : "the server cannot draw"}>
            <input type="checkbox" checked={draw && canDraw} disabled={!canDraw} onChange={(e) => setDraw(e.target.checked)} />
            draw each symbol
          </label>
          {canDraw && info.data && (
            <div className="flex flex-wrap items-center gap-1">
              {info.data.image_models.map((m) => (
                <button key={m.id} type="button" title={m.hint}
                        className={`pill !text-[10px] ${m.id === model ? "border-accent-500/60 bg-accent-600/25 text-accent-100" : "hover:border-ink-500"}`}
                        onClick={() => setModel(m.id)}>
                  {m.label}
                </button>
              ))}
              <span className="ml-1 text-[10px] text-ink-500">{modelInfo?.hint}</span>
            </div>
          )}

          <button className="btn-primary w-full" onClick={generate} disabled={!ready || naming || !fresh.length} title="Ctrl+Enter in the list">
            {running.length ? `generating… ${running.length} on the way` : `generate${fresh.length ? ` ${fresh.length}` : ""}`}
          </button>
          {status && <div className="text-xs text-accent-300">{status}</div>}
          {error && <div className="text-xs text-red-300">{error}</div>}
        </div>

        <div className="panel-pad space-y-2">
          <div className="flex items-center label-sm">
            The set
            <HelpTip text="Save keeps the set in this browser; it then appears under 'my sets' in the preset lists, and every change here updates it in place. Export writes a file with the symbols, their glyphs and their vector drawings; the sidebar's Upload JSON and the import here read it back (do-you-dream can read it too)." />
          </div>
          <div className="text-xs text-ink-300">
            {current ? <>selected: <span className="text-ink-100">✦ {current.name}</span>{current.theme ? <span className="text-ink-500"> · {current.theme}</span> : null}</>
                     : presetName ? <>a shipped structure: <span className="text-ink-100">{setTitle}</span> — changes make it a custom map; save keeps it</>
                     : "a custom map — save it to keep it"}
          </div>
          <div className="flex gap-1.5">
            <input
              className="input-base flex-1 text-xs"
              placeholder={current ? `a name for a new set, or to rename “${current.name}”…` : "a name for the set…"}
              value={saveName}
              onChange={(e) => setSaveName(e.target.value)}
              onKeyDown={(e) => { if (e.key === "Enter") { e.preventDefault(); current ? saveCopy() : save(); } }}
            />
            {current ? (
              <>
                <button className="btn whitespace-nowrap" onClick={saveCopy} disabled={!keys.length}
                        title="a new saved set with these symbols, under the name typed (or “… copy”); the current one stays as it is">
                  save as new set
                </button>
                <button className="btn" onClick={save} disabled={!keys.length} title={`update “${current.name}” (and rename it, if a name is typed)`}>save</button>
              </>
            ) : (
              <button className="btn whitespace-nowrap" onClick={save} disabled={!keys.length}>save as set</button>
            )}
          </div>
          <div className="flex flex-wrap gap-1.5">
            <button className="btn text-xs" onClick={startEmpty} title="an empty set to generate or paste into; the selected set is left as it is">new empty set</button>
            <button className="btn text-xs" onClick={exportFile} disabled={!keys.length}>export file</button>
            <button className="btn text-xs" onClick={() => fileInput.current?.click()}>import file</button>
            <input ref={fileInput} type="file" accept="application/json,.json" hidden
                   onChange={(e) => { const f = e.target.files?.[0]; if (f) importFile(f); e.target.value = ""; }} />
            {current && <button className="btn text-xs" onClick={removeSaved}>delete set</button>}
          </div>
          {saved.length > 0 && (
            <div className="text-[10px] text-ink-500">
              {saved.length} saved set{saved.length === 1 ? "" : "s"} in this browser: {saved.map((s) => s.name).join(", ")}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

const STEP: Record<Step, string> = { writing: "writing descriptors", drawing: "drawing", error: "failed" };

function Card({ name, descriptors, art, color, job, view, canDraw, pending, editing, onEdit, onSave, onRedo, onRemove, onRetry,
                dragging, onDragStart, onDragEnd, onDropBefore }: {
  name: string; descriptors: string[]; art?: SymbolArt; color: string; job?: Job; view: "vector" | "ink";
  canDraw: boolean; pending?: boolean; editing?: boolean;
  onEdit?: () => void; onSave?: (name: string, descriptors: string[]) => void;
  onRedo?: (kind: "redraw" | "rewrite" | "draw") => void; onRemove: () => void; onRetry: () => void;
  dragging?: boolean; onDragStart?: () => void; onDragEnd?: () => void; onDropBefore?: () => void;
}) {
  const running = job && job.step !== "error";
  const failed = job?.step === "error";
  const svg = art?.svg;
  const [over, setOver] = React.useState(false);
  return (
    <article
      className={`flex flex-col gap-1.5 rounded-lg border p-2 transition ${
        failed ? "border-red-400/40" : pending ? "border-dashed border-ink-700" : editing ? "border-accent-500/60" : "border-ink-700/60"
      } ${dragging ? "opacity-40" : ""} ${over ? "ring-2 ring-accent-500/60" : ""}`}
      style={{ ["--c" as any]: color }}
      draggable={!pending && !running && !editing}
      onDragStart={(e) => { e.dataTransfer.effectAllowed = "move"; onDragStart?.(); }}
      onDragEnd={() => { setOver(false); onDragEnd?.(); }}
      onDragOver={(e) => { if (onDropBefore) { e.preventDefault(); e.stopPropagation(); setOver(true); } }}
      onDragLeave={() => setOver(false)}
      onDrop={(e) => { e.preventDefault(); e.stopPropagation(); setOver(false); onDropBefore?.(); }}
    >
      <div className="relative grid aspect-square place-items-center rounded-md bg-ink-900/60" title={art?.glyph}>
        {view === "ink" && art?.photo ? (
          <img src={art.photo} alt={name} className="h-full w-full rounded-md object-contain" loading="lazy" draggable={false} />
        ) : svg ? (
          <Glyph svg={svg} color={color} className="h-[72%] w-[72%]" />
        ) : (
          <span className="text-[10px] uppercase tracking-wider text-ink-500">{running ? "" : "no drawing"}</span>
        )}
        {running && (
          <div className="absolute inset-0 flex items-center justify-center gap-2 rounded-md bg-ink-950/70 text-[10px] uppercase tracking-wider text-ink-300">
            <span className="h-1.5 w-1.5 animate-pulse rounded-full bg-accent-400" />
            {STEP[job!.step]}
          </div>
        )}
        {failed && (
          <div className="absolute inset-0 flex items-center justify-center rounded-md bg-ink-950/80 p-2 text-center text-[11px] leading-snug text-red-300">
            {job!.error}
          </div>
        )}
      </div>
      {editing && onSave ? (
        <Editor name={name} descriptors={descriptors} onSave={onSave} onCancel={() => onEdit?.()} />
      ) : (
        <>
          <div className="flex items-center gap-1.5 text-sm text-ink-100">
            <span className="inline-block h-2 w-2 shrink-0 rounded-full" style={{ background: color }} />
            <span className="truncate" title={name}>{name}</span>
          </div>
          <div className="line-clamp-3 text-[11px] leading-snug text-ink-400" title={descriptors.join(", ")}>
            {descriptors.join(", ")}
          </div>
          {running && job?.glyph && <div className="line-clamp-2 text-[10px] italic text-ink-500">{job.glyph}</div>}
        </>
      )}
      {!editing && (
        <div className="mt-auto flex flex-wrap gap-1">
          {failed ? (
            <>
              <Act onClick={onRetry}>retry</Act>
              <Act onClick={onRemove}>dismiss</Act>
            </>
          ) : running || pending ? null : (
            <>
              {onEdit && <Act onClick={onEdit} title="name and descriptors">edit</Act>}
              {canDraw && onRedo && (svg
                ? <Act onClick={() => onRedo("redraw")} title="a new drawing from the same glyph">redraw</Act>
                : <Act onClick={() => onRedo("draw")} title="draw this symbol from its name and descriptors">draw</Act>)}
              {onRedo && <Act onClick={() => onRedo("rewrite")} title="new descriptors and glyph, then a new drawing">rewrite</Act>}
              <Act onClick={onRemove} title="remove from the set">remove</Act>
            </>
          )}
        </div>
      )}
    </article>
  );
}

function Editor({ name, descriptors, onSave, onCancel }: {
  name: string; descriptors: string[]; onSave: (name: string, descriptors: string[]) => void; onCancel: () => void;
}) {
  const [n, setN] = React.useState(name);
  const [d, setD] = React.useState(descriptors.join("\n"));
  const submit = () => onSave(n, cleanDescriptors(d));
  return (
    <div className="space-y-1.5">
      <input className="input-base !px-2 !py-1 text-xs" value={n} onChange={(e) => setN(e.target.value)} autoFocus
             onKeyDown={(e) => { if (e.key === "Enter") { e.preventDefault(); submit(); } if (e.key === "Escape") onCancel(); }} />
      <textarea className="input-base h-28 !px-2 !py-1 font-mono text-[11px] leading-snug" value={d} onChange={(e) => setD(e.target.value)}
                placeholder="one descriptor per line" title="one descriptor per line (commas work too); the first is usually the name"
                onKeyDown={(e) => { if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) { e.preventDefault(); submit(); } if (e.key === "Escape") onCancel(); }} />
      <div className="flex gap-1">
        <Act onClick={submit} title="Ctrl+Enter">save</Act>
        <Act onClick={onCancel} title="Escape">cancel</Act>
      </div>
    </div>
  );
}

const Act = ({ children, onClick, title }: { children: React.ReactNode; onClick: () => void; title?: string }) => (
  <button type="button" className="rounded px-1.5 py-0.5 text-[10px] text-ink-400 transition hover:bg-ink-800 hover:text-ink-100" onClick={onClick} title={title}>
    {children}
  </button>
);
