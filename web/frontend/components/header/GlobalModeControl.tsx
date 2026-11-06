"use client";
import * as React from "react";
import { HelpTip } from "../ui/HelpTip";
import { useSidebar } from "@/lib/store";
import { useLive } from "@/lib/liveGraph";
import { api, GlobalModeResponse } from "@/lib/api";

/**
 * Global-mode switch next to the Build button: removes the descriptor offset
 * (and optionally the top principal directions of the descriptor cloud)
 * from the geometry the Δ readout works in.
 *
 * The setting lives on the cached space server-side, so every Δ readout on
 * that space follows it (Explorer Δ graph, Live graph, story motifs).  The
 * sync effect re-applies it after every (re)build.
 */
export function GlobalModeControl() {
  const sid = useSidebar((s) => s.spaceId);
  const mode = useSidebar((s) => s.globalMode);
  const pcs = useSidebar((s) => s.globalModePcs);
  const buildNonce = useSidebar((s) => s.buildNonce);
  const set = useSidebar((s) => s.set);
  const [info, setInfo] = React.useState<GlobalModeResponse | null>(null);
  const [busy, setBusy] = React.useState(false);
  const [error, setError] = React.useState<string | null>(null);
  const on = mode === "descriptor";

  React.useEffect(() => {
    if (!sid) {
      setInfo(null);
      return;
    }
    let cancelled = false;
    const wantPcs = on ? pcs : 0;
    setBusy(true);
    setError(null);
    (async () => {
      try {
        const cur = await api.get<GlobalModeResponse>(`/global-mode/${encodeURIComponent(sid)}`);
        if (cancelled) return;
        if (cur.mode === mode && cur.pcs === wantPcs) {
          setInfo(cur);
          return;
        }
        const next = await api.post<GlobalModeResponse>("/global-mode", { space_id: sid, mode, pcs: wantPcs });
        if (cancelled) return;
        setInfo(next);
        // Δ queries key on this nonce, so they refetch only once the server
        // has switched geometry.
        useSidebar.setState((s) => ({ globalModeNonce: s.globalModeNonce + 1 }));
        // The live graph recomputes on any params message.
        const live = useLive.getState();
        if (live.status === "ready" && live._send) live.setParams({});
      } catch (e: any) {
        if (!cancelled) setError(String(e?.message ?? e));
      } finally {
        if (!cancelled) setBusy(false);
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [sid, mode, pcs, on, buildNonce]);

  const shared = info ? `${Math.round(info.shared_energy * 100)}%` : "most";
  const help =
    `Every descriptor shares ${shared} of its vector with all the others, so each context lifts almost ` +
    `every relation at once and the Δ graph keeps returning the same edges. This switch removes that ` +
    `shared offset from the descriptors and the context before the shift. The offset is fitted when the ` +
    `space is built. The slider also removes the strongest directions of the descriptor cloud ` +
    `(0 removes the offset alone, 1 gave the best grounding on the Lakota probes). ` +
    `Applies to the Δ graph in Explorer and Live and to story motifs.`;

  return (
    <div className="flex flex-wrap items-center gap-x-3 gap-y-1 text-xs text-ink-300">
      <label className="flex cursor-pointer items-center gap-2">
        <button
          type="button"
          role="switch"
          aria-checked={on}
          aria-label="Remove global mode from Δ"
          onClick={() => set("globalMode", on ? "none" : "descriptor")}
          className={`relative inline-flex h-4 w-7 shrink-0 items-center rounded-full transition
            ${on ? "bg-accent-500" : "bg-ink-700"}`}
        >
          <span
            className={`inline-block h-3 w-3 transform rounded-full bg-white shadow transition
              ${on ? "translate-x-3.5" : "translate-x-0.5"}`}
          />
        </button>
        <span className="flex items-center">
          Remove global mode
          <HelpTip text={help} />
        </span>
      </label>

      {on && (
        <label className="flex items-center gap-1.5" title="Principal directions of the descriptor cloud also removed">
          <span className="text-ink-400">PCs</span>
          <input
            type="range"
            min={0}
            max={info?.max_pcs ?? 5}
            step={1}
            value={pcs}
            onChange={(e) => set("globalModePcs", Math.round(Number(e.target.value)))}
            className="h-1 w-16 cursor-pointer accent-accent-500"
            aria-label="Principal directions removed"
          />
          <span className="w-3 font-mono">{pcs}</span>
        </label>
      )}

      {info && (
        <span
          className="font-mono text-[11px] text-ink-400"
          title="Mean descriptor–descriptor cosine, raw → in the geometry Δ uses now"
        >
          cos {info.mean_cos_raw.toFixed(2)} → {info.mean_cos_active.toFixed(2)}
          {busy && " …"}
        </span>
      )}
      {error && <span className="text-[11px] text-danger">{error}</span>}
    </div>
  );
}
