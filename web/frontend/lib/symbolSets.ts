// Symbol sets made on the site (generated or pasted) and kept in this browser, next to the
// shipped presets.  A saved set is selected like a preset: `presetName` = "local:<id>".
// `resolvePreset` gives the symbols (and drawings) of either kind, so every picker goes through it.
import * as React from "react";
import { api, SymbolMap } from "./api";

const KEY = "delyrism.symbolSets";
export const LOCAL_PREFIX = "local:";

/** What the generator made for one symbol: how to draw it, and the drawing (server: /generate/draw). */
export interface SymbolArt {
  glyph?: string;          // the drawing instruction, in a few basic shapes
  llm?: string;            // who wrote the descriptors and glyph
  svg?: string;            // filled outlines, framed square around the ink (black on transparent)
  lines_svg?: string;      // centre-line strokes, same frame
  photo?: string;          // data URL: the cleaned ink on white, small
  geometry?: unknown;      // the vectorizer's record (strokes, fills; ink box centred on 0, y up)
  seed?: number;
  model?: string;          // the image model
  image_prompt?: string;
  at?: string;             // ISO
}
export type ArtMap = Record<string, SymbolArt>;

export interface SavedSet {
  id: string;
  name: string;
  theme: string;
  symbols: SymbolMap;
  art: ArtMap;
  created: string;   // ISO
  edited: string;    // ISO
}

const listeners = new Set<() => void>();
let cache: SavedSet[] | null = null;

function read(): SavedSet[] {
  if (cache) return cache;
  try {
    const raw = typeof window === "undefined" ? null : localStorage.getItem(KEY);
    const arr = raw ? JSON.parse(raw) : [];
    cache = Array.isArray(arr)
      ? arr.filter((s) => s && s.id && s.symbols).map((s) => ({ theme: "", ...s, art: s.art ?? glyphsToArt(s.glyphs) }))
      : [];
  } catch {
    cache = [];
  }
  return cache!;
}
// sets saved by the first version kept only the glyphs
const glyphsToArt = (g: Record<string, string> | undefined): ArtMap =>
  Object.fromEntries(Object.entries(g ?? {}).map(([k, glyph]) => [k, { glyph }]));

export class StorageFull extends Error {}

function write(sets: SavedSet[]) {
  const before = cache;
  cache = sets;
  try {
    localStorage.setItem(KEY, JSON.stringify(sets));
  } catch (e) {
    cache = before;
    throw new StorageFull("this browser's storage is full: delete a saved set, or export this one as a file");
  }
  listeners.forEach((fn) => fn());
}

export function listSets(): SavedSet[] { return read(); }
export function getSet(id: string): SavedSet | undefined { return read().find((s) => s.id === id); }
export function isLocal(presetName: string | null | undefined): boolean {
  return !!presetName && presetName.startsWith(LOCAL_PREFIX);
}
export function localId(presetName: string): string { return presetName.slice(LOCAL_PREFIX.length); }

export const slug = (s: string) => s.toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/^-+|-+$/g, "") || "untitled";

/** Save a set under a name; an `id` updates that set in place.  Returns the set. */
export function saveSet(input: { id?: string; name: string; theme?: string; symbols: SymbolMap; art?: ArtMap }): SavedSet {
  const sets = read().slice();
  const now = new Date().toISOString();
  const name = input.name.trim() || "untitled";
  const i = input.id ? sets.findIndex((s) => s.id === input.id) : -1;
  if (i >= 0) {
    const s = { ...sets[i], name, theme: input.theme ?? sets[i].theme, symbols: input.symbols,
                art: input.art ?? sets[i].art, edited: now };
    sets[i] = s;
    write(sets);
    return s;
  }
  let id = `${now.slice(0, 10)}-${slug(name)}`;
  for (let k = 2; sets.some((s) => s.id === id); k++) id = `${now.slice(0, 10)}-${slug(name)}-${k}`;
  const s: SavedSet = { id, name, theme: input.theme ?? "", symbols: input.symbols, art: input.art ?? {}, created: now, edited: now };
  sets.unshift(s);
  write(sets);
  return s;
}

export function deleteSet(id: string) { write(read().filter((s) => s.id !== id)); }

/** The symbols and drawings of a preset of either kind (a shipped JSON, or a set saved here). */
export async function resolvePreset(presetName: string): Promise<{ symbols: SymbolMap; art: ArtMap }> {
  if (isLocal(presetName)) {
    const s = getSet(localId(presetName));
    if (!s) throw new Error("saved set not found");
    return { symbols: s.symbols, art: s.art };
  }
  const p = await api.get<{ name: string; symbols: SymbolMap; art?: ArtMap }>(`/spaces/presets/${presetName}`);
  return { symbols: p.symbols, art: p.art ?? {} };
}

/** Display name of any preset id (shipped: the caller's table; saved: its name). */
export function displaySet(presetName: string | null | undefined): string | null {
  if (!isLocal(presetName)) return null;
  return getSet(localId(presetName!))?.name ?? "(deleted set)";
}

const subscribe = (fn: () => void) => { listeners.add(fn); return () => { listeners.delete(fn); }; };
const EMPTY: SavedSet[] = [];

/** The saved sets, re-rendering on every change (SSR sees none). */
export function useSavedSets(): SavedSet[] {
  return React.useSyncExternalStore(subscribe, read, () => EMPTY);
}

/** An SVG string as a data URL, for CSS masks (the drawing tinted with the symbol's colour). */
export function svgUrl(svg: string): string {
  return "data:image/svg+xml;base64," + btoa(unescape(encodeURIComponent(svg)));
}

/** The drawing with its ink in a colour, as a data URL: for canvases and plots, where a CSS mask
 *  cannot tint it (the vectorizer writes the ink as fill="#000000"). */
const tinted = new Map<string, string>();
export function tintedSvgUrl(svg: string, color: string): string {
  const k = color + "|" + svg.length + "|" + svg.slice(0, 160);
  let u = tinted.get(k);
  if (!u) {
    u = svgUrl(svg.replace(/fill="#000000"/g, `fill="${color}"`).replace(/stroke="#000000"/g, `stroke="${color}"`));
    if (tinted.size > 500) tinted.clear();
    tinted.set(k, u);
  }
  return u;
}

/** A loaded <img> of a tinted drawing, for canvas drawing; null until it has loaded. */
const images = new Map<string, HTMLImageElement>();
export function tintedImage(svg: string, color: string): HTMLImageElement | null {
  const url = tintedSvgUrl(svg, color);
  let im = images.get(url);
  if (!im) {
    im = new Image();
    im.src = url;
    if (images.size > 500) images.clear();
    images.set(url, im);
  }
  return im.complete && im.naturalWidth > 0 ? im : null;
}

/** The art of a symbol, by name in any case. */
export function artOf(art: ArtMap, symbol: string): SymbolArt | undefined {
  return art[symbol] ?? art[symbol.toUpperCase()];
}

// ------------------------------------------------------------------ files: a whole set as JSON
/** The file a set exports to: symbols in the space's shape, plus how each one was made and drawn.
 *  `symbols` alone is what the sidebar's Upload JSON reads; do-you-dream can read the rest. */
export interface SetFile {
  format: "delyrism-symbol-set";
  version: 1;
  name: string;
  theme: string;
  symbols: SymbolMap;
  art: ArtMap;
  exported: string;
}

export function setToFile(s: { name: string; theme: string; symbols: SymbolMap; art: ArtMap }): SetFile {
  return { format: "delyrism-symbol-set", version: 1, name: s.name, theme: s.theme, symbols: s.symbols,
           art: s.art, exported: new Date().toISOString() };
}

/** Read a set file, or a bare {symbol: [descriptors]} map. */
export function fileToSet(text: string): { name: string; theme: string; symbols: SymbolMap; art: ArtMap } {
  const j = JSON.parse(text);
  if (j && j.format === "delyrism-symbol-set" && j.symbols) {
    return { name: String(j.name || ""), theme: String(j.theme || ""), symbols: j.symbols, art: j.art ?? {} };
  }
  const map = j?.symbols && !Array.isArray(j.symbols) && typeof j.symbols === "object" ? j.symbols : j;
  if (!map || typeof map !== "object" || Array.isArray(map)) throw new Error("not a symbol map");
  for (const [k, v] of Object.entries(map)) {
    if (!Array.isArray(v)) throw new Error(`"${k}" has no descriptor list`);
  }
  return { name: "", theme: "", symbols: map as SymbolMap, art: {} };
}
