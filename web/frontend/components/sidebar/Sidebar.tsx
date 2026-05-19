"use client";
import * as React from "react";
import { useMutation } from "@tanstack/react-query";
import { api, SpaceConfig, SpaceCreateResponse } from "@/lib/api";
import { useSidebar } from "@/lib/store";
import { SymbolicStructure } from "./SymbolicStructure";
import { ContextOptions } from "./ContextOptions";
import { EmbeddingModel } from "./EmbeddingModel";
import { SemanticMap } from "./SemanticMap";
import { RankingSection } from "./RankingSection";
import { SubgraphSection } from "./SubgraphSection";
import { DeltaSection } from "./DeltaSection";

/**
 * Sidebar — orchestrates all config panels and "Build space" action.
 *
 * Building a space is an explicit step (it's expensive — embeds every
 * descriptor).  Slider/toggle changes after that update visualizations only,
 * cheaply, without rebuilding the space.
 */
export function Sidebar() {
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
    <aside className="flex h-full flex-col">
      <div className="border-b border-ink-700/60 p-4">
        <div className="flex items-center gap-2.5">
          {/* eslint-disable-next-line @next/next/no-img-element — static file
              served by FastAPI in prod, by Next's public/ in dev.  No need
              for next/image's optimizer here (static export + small asset). */}
          <img
            src="/delyrism-logo.png"
            alt="delyrism"
            className="h-8 w-8 rounded-md object-contain"
          />
          <div>
            <div className="font-display text-lg leading-tight">delyrism</div>
            <div className="text-[10px] uppercase tracking-widest text-ink-400">
              symbolic explorer
            </div>
          </div>
        </div>
      </div>

      <div className="flex-1 overflow-y-auto p-3">
        <SymbolicStructure />
        <ContextOptions />
        <EmbeddingModel />
        <SemanticMap />
        <RankingSection />
        <SubgraphSection />
        <DeltaSection />
      </div>

      <div className="border-t border-ink-700/60 p-3">
        <button
          className="btn-primary w-full"
          onClick={() => build.mutate()}
          disabled={build.isPending || !json}
        >
          {build.isPending ? "Building space…" : spaceId ? "Rebuild space" : "Build space"}
        </button>
        {spaceId && (
          <div className="mt-2 flex items-center gap-2 text-[11px] text-ink-400">
            <span className="h-1.5 w-1.5 rounded-full bg-accent-400" />
            <span className="font-mono">space {spaceId.slice(0, 8)}…</span>
          </div>
        )}
        {build.isError && (
          <div className="mt-2 text-xs text-danger">{(build.error as Error).message}</div>
        )}
      </div>
    </aside>
  );
}
