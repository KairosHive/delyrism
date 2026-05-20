"use client";
import * as React from "react";
import { SymbolicStructure } from "./SymbolicStructure";
import { ContextOptions } from "./ContextOptions";
import { EmbeddingModel } from "./EmbeddingModel";
import { SemanticMap } from "./SemanticMap";
import { RankingSection } from "./RankingSection";
import { SubgraphSection } from "./SubgraphSection";
import { DeltaSection } from "./DeltaSection";

/**
 * Sidebar — config panels for the symbolic space.
 *
 * The Build/Rebuild trigger lives in the main page now (under the Console
 * cards), since first-time users couldn't find it tucked at the bottom of
 * a scrollable drawer.  Everything else — embedder backend, descriptor
 * threshold, ranking knobs, etc. — still lives here.
 */
export function Sidebar() {
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
    </aside>
  );
}
