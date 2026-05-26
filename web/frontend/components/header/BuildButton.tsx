"use client";
import * as React from "react";
import { useMutation } from "@tanstack/react-query";
import { api, SpaceConfig, SpaceCreateResponse } from "@/lib/api";
import { useSidebar } from "@/lib/store";
import { ExportButton } from "./ExportButton";

/**
 * Build / rebuild the SymbolSpace.  Lives directly under the Console cards
 * (archetype dropdown + context prompt) so new users find it without having
 * to discover the sidebar drawer.
 *
 * Logic mirrors what used to live at the bottom of Sidebar.tsx — same
 * payload, same store updates.  Sidebar advanced controls still affect the
 * resulting space (embedder backend, descriptor threshold, etc.); we just
 * moved the trigger.
 */
export function BuildButton() {
  const json = useSidebar((s) => s.symbolMapJson);
  const backend = useSidebar((s) => s.embedderBackend);
  const model = useSidebar((s) => s.embedderModel);
  const pooling = useSidebar((s) => s.embedderPooling);
  const instr = useSidebar((s) => s.qwenInstruction);
  const ctxMode = useSidebar((s) => s.qwenContextMode);
  const ctxText = useSidebar((s) => s.qwenGlobalContext);
  const dthr = useSidebar((s) => s.descriptorThreshold);
  const spaceId = useSidebar((s) => s.spaceId);
  const setBulk = useSidebar((s) => s.setBulk);

  const build = useMutation({
    mutationFn: async () => {
      const symbols = JSON.parse(json);
      const body: SpaceConfig = {
        symbols,
        embedder: {
          backend,
          model: model.trim() || null,
          pooling,
          default_instruction: instr || null,
          default_context: ctxMode === "global" ? ctxText : ctxMode === "per-descriptor" ? "Distributed" : null,
        },
        descriptor_threshold: dthr,
        contextual_embeddings: false,
        palette: "AuroraPop",
      };
      return api.post<SpaceCreateResponse>("/spaces", body);
    },
    onSuccess: (data) => {
      setBulk({
        spaceId: data.space_id,
        symbols: data.symbols,
        colorMap: data.color_map,
      });
    },
  });

  return (
    <div className="mx-auto mt-3 flex w-full max-w-5xl flex-col items-stretch gap-2 sm:flex-row sm:items-center sm:justify-between">
      <div className="flex flex-col gap-1">
        <button
          className="btn-primary w-full sm:w-auto sm:min-w-[200px]"
          onClick={() => build.mutate()}
          disabled={build.isPending || !json}
        >
          {build.isPending ? "Building space…" : spaceId ? "Rebuild space" : "Build space"}
        </button>
        {build.isError && (
          <div className="text-xs text-danger">{(build.error as Error).message}</div>
        )}
      </div>
      <div className="flex items-center gap-3">
        {spaceId && !build.isPending && (
          <div className="flex items-center gap-2 text-[11px] text-ink-400">
            <span className="h-1.5 w-1.5 rounded-full bg-accent-400" />
            <span className="font-mono">space {spaceId.slice(0, 8)}…</span>
          </div>
        )}
        <ExportButton />
      </div>
    </div>
  );
}
