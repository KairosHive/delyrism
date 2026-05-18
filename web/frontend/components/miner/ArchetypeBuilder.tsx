"use client";
import * as React from "react";
import { useQuery } from "@tanstack/react-query";
import { api } from "@/lib/api";
import { useSidebar } from "@/lib/store";

interface MinerInfo { url: string; available: boolean; }

export function ArchetypeBuilder() {
  const set = useSidebar((s) => s.set);
  const info = useQuery({ queryKey: ["miner"], queryFn: () => api.get<MinerInfo>("/miner") });
  const [imported, setImported] = React.useState<string | null>(null);

  async function importFromText(text: string) {
    try {
      const parsed = JSON.parse(text);
      const map = parsed.symbols ?? parsed;
      set("symbolMapJson", JSON.stringify(map, null, 2));
      set("presetName", null);
      setImported(`Loaded ${Object.keys(map).length} symbols into the sidebar.`);
    } catch (e: any) {
      setImported(`Could not parse JSON: ${e.message}`);
    }
  }

  return (
    <div className="grid grid-cols-1 gap-4 xl:grid-cols-[1.5fr,1fr]">
      <div className="panel p-0 overflow-hidden">
        <div className="flex items-center justify-between border-b border-ink-700/60 px-4 py-2.5">
          <div className="section-title">Egregore · real-time miner</div>
          {info.data && (
            <a
              className="text-xs text-accent-300 hover:underline"
              href={info.data.url}
              target="_blank"
              rel="noreferrer"
            >open ↗</a>
          )}
        </div>
        {info.data ? (
          <iframe
            src={info.data.url}
            className="h-[640px] w-full"
            sandbox="allow-scripts allow-same-origin allow-forms allow-popups"
          />
        ) : (
          <div className="p-6 text-sm text-ink-300">Connecting to miner…</div>
        )}
      </div>
      <div className="space-y-3">
        <div className="panel-pad space-y-2">
          <div className="section-title">Import archetypes</div>
          <p className="text-xs text-ink-300">
            Paste an Egregore JSON export (or any <code className="font-mono">{`{symbol: [descriptors...]}`}</code>) below
            and it will replace the current symbol map in the sidebar.
          </p>
          <textarea
            className="input-base h-40 font-mono text-xs"
            placeholder='{ "ORACLE": ["liminal", "voice", "threshold"], ... }'
            onPaste={(e) => {
              const t = e.clipboardData.getData("text");
              setTimeout(() => importFromText(t), 0);
            }}
            onChange={(e) => importFromText(e.target.value)}
          />
          {imported && (
            <div className="text-xs text-accent-300">{imported}</div>
          )}
        </div>
        <div className="panel-pad text-xs text-ink-300 space-y-2">
          <div className="section-title text-ink-100">Notes</div>
          <p>
            Egregore is the existing FastAPI miner (lives at{" "}
            <code className="font-mono text-accent-300">{info.data?.url ?? "EGREGORE_URL"}</code>).
            Real-time mining, clustering and LLM refinement happen there; once you have a
            satisfying archetype set, export the JSON and paste it here to drop straight into
            the Explorer.
          </p>
        </div>
      </div>
    </div>
  );
}
