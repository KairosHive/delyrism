"use client";
import * as React from "react";
import { useQueryClient } from "@tanstack/react-query";
import { useSidebar } from "@/lib/store";
import { exportSession, exportFilename, triggerDownload } from "@/lib/exportSession";

/**
 * Download the current Explorer session as a .zip — manifest + raw JSON for
 * every panel + standalone interactive HTML figures (Plotly + force-graph
 * from CDN).  Lives next to the Build button so users find it without a
 * hunt; greyed out until a space exists.
 */
export function ExportButton() {
  const qc = useQueryClient();
  const spaceId = useSidebar((s) => s.spaceId);
  const [busy, setBusy] = React.useState(false);
  const [err, setErr] = React.useState<string | null>(null);

  async function onExport() {
    if (busy || !spaceId) return;
    setBusy(true);
    setErr(null);
    try {
      const state = useSidebar.getState();
      const blob = await exportSession(qc, state);
      triggerDownload(blob, exportFilename(state));
    } catch (e: any) {
      setErr(e?.message ?? "export failed");
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="flex flex-col items-end gap-1">
      <button
        className="btn !text-xs"
        onClick={onExport}
        disabled={busy || !spaceId}
        title={
          spaceId
            ? "Download a .zip with the current state, raw JSON data, and standalone interactive HTML plots."
            : "Build a space first."
        }
      >
        {busy ? (
          <>
            <span className="h-1.5 w-1.5 animate-pulse rounded-full bg-accent-400" />
            packing…
          </>
        ) : (
          <>↓ Export session</>
        )}
      </button>
      {err && <div className="text-[10px] text-danger">{err}</div>}
    </div>
  );
}
