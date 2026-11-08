// Symbol sets made on the site (generated or pasted) and kept in this browser, next to the
// shipped presets.  A saved set is selected like a preset: `presetName` = "local:<id>".
// `resolvePreset` gives the symbols of either kind, so every preset picker goes through it.
import * as React from "react";
import { api, SymbolMap } from "./api";

const KEY = "delyrism.symbolSets";
export const LOCAL_PREFIX = "local:";

export interface SavedSet {
  id: string;
  name: string;
  theme: string;
  symbols: SymbolMap;
  /** how each symbol is drawn, in a few basic shapes (from the generator; empty for pasted sets) */
  glyphs: Record<string, string>;
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
    cache = Array.isArray(arr) ? arr.filter((s) => s && s.id && s.symbols) : [];
  } catch {
    cache = [];
  }
  return cache!;
}

function write(sets: SavedSet[]) {
  cache = sets;
  try { localStorage.setItem(KEY, JSON.stringify(sets)); } catch {}
  listeners.forEach((fn) => fn());
}

export function listSets(): SavedSet[] { return read(); }
export function getSet(id: string): SavedSet | undefined { return read().find((s) => s.id === id); }
export function isLocal(presetName: string | null | undefined): boolean {
  return !!presetName && presetName.startsWith(LOCAL_PREFIX);
}
export function localId(presetName: string): string { return presetName.slice(LOCAL_PREFIX.length); }

const slug = (s: string) => s.toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/^-+|-+$/g, "") || "untitled";

/** Save a set under a name; an `id` updates that set in place.  Returns the set. */
export function saveSet(input: { id?: string; name: string; theme?: string; symbols: SymbolMap; glyphs?: Record<string, string> }): SavedSet {
  const sets = read().slice();
  const now = new Date().toISOString();
  const name = input.name.trim() || "untitled";
  const i = input.id ? sets.findIndex((s) => s.id === input.id) : -1;
  if (i >= 0) {
    const s = { ...sets[i], name, theme: input.theme ?? sets[i].theme, symbols: input.symbols,
                glyphs: input.glyphs ?? sets[i].glyphs, edited: now };
    sets[i] = s;
    write(sets);
    return s;
  }
  let id = `${now.slice(0, 10)}-${slug(name)}`;
  for (let k = 2; sets.some((s) => s.id === id); k++) id = `${now.slice(0, 10)}-${slug(name)}-${k}`;
  const s: SavedSet = { id, name, theme: input.theme ?? "", symbols: input.symbols, glyphs: input.glyphs ?? {}, created: now, edited: now };
  sets.unshift(s);
  write(sets);
  return s;
}

export function deleteSet(id: string) { write(read().filter((s) => s.id !== id)); }

/** The symbols of a preset of either kind (a shipped JSON, or a set saved here). */
export async function resolvePreset(presetName: string): Promise<SymbolMap> {
  if (isLocal(presetName)) {
    const s = getSet(localId(presetName));
    if (!s) throw new Error("saved set not found");
    return s.symbols;
  }
  const p = await api.get<{ name: string; symbols: SymbolMap }>(`/spaces/presets/${presetName}`);
  return p.symbols;
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
