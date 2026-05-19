"use client";
import * as React from "react";
import { Sidebar } from "@/components/sidebar/Sidebar";
import { Explorer } from "@/components/explorer/Explorer";
import { StoryGenerator } from "@/components/story/StoryGenerator";
import { ArchetypeBuilder } from "@/components/builder/ArchetypeBuilder";
import { Title } from "@/components/header/Title";
import { Console } from "@/components/header/Console";
import { TimingBadge } from "@/components/debug/TimingBadge";
import { useSidebar } from "@/lib/store";

type TabId = "explorer" | "story" | "builder";

const TABS: { id: TabId; label: string; hint: string }[] = [
  { id: "explorer", label: "Explorer",          hint: "context-conditioned embeddings" },
  { id: "story",    label: "Story Generator",   hint: "weave motifs into micro-fiction" },
  { id: "builder",  label: "Archetype Builder", hint: "compose new symbol sets" },
];

export default function Home() {
  const [tab, setTab] = React.useState<TabId>("explorer");
  const spaceId = useSidebar((s) => s.spaceId);

  // Plotly's autosize measures the container at first paint — which is
  // often BEFORE the panels have settled their final dimensions on first
  // build, leaving charts at the wrong size until the next render.  We
  // nudge every chart on the page to re-measure by firing a few window
  // resize events after the space appears.  Plotly listens (each Plot has
  // `useResizeHandler`), so this snaps every panel to its true size.
  React.useEffect(() => {
    if (!spaceId) return;
    const timers = [50, 250, 800].map((d) =>
      setTimeout(() => window.dispatchEvent(new Event("resize")), d),
    );
    return () => timers.forEach(clearTimeout);
  }, [spaceId]);

  // Same trick on tab change — panels in the inactive tab don't lay out
  // until the user actually visits them, so first paint can come up at
  // the wrong size.  One nudge after a short delay snaps everything.
  React.useEffect(() => {
    const t = setTimeout(() => window.dispatchEvent(new Event("resize")), 80);
    return () => clearTimeout(t);
  }, [tab]);

  return (
    <div className="grid h-screen grid-cols-[300px,1fr]">
      <aside className="border-r border-ink-700/60 bg-ink-900/40 backdrop-blur-md">
        <Sidebar />
      </aside>

      <main className="flex h-full flex-col overflow-y-auto">
        <Title />

        <div className="px-8 pb-2">
          <Console />
        </div>

        <Tabs tab={tab} setTab={setTab} />

        <section className="flex-1 px-8 pb-10 pt-6">
          {tab === "explorer" && <Explorer />}
          {tab === "story" && <StoryGenerator />}
          {tab === "builder" && <ArchetypeBuilder />}
        </section>
      </main>
      <TimingBadge />
    </div>
  );
}

function Tabs({ tab, setTab }: { tab: TabId; setTab: (t: TabId) => void }) {
  return (
    <div className="mx-auto mt-2 w-full max-w-5xl border-b border-ink-700/50 px-1">
      <nav className="flex items-center gap-1">
        {TABS.map((t) => {
          const active = t.id === tab;
          return (
            <button
              key={t.id}
              onClick={() => setTab(t.id)}
              className={`relative px-4 py-2.5 text-sm transition
                ${active ? "text-ink-50" : "text-ink-400 hover:text-ink-100"}`}
            >
              {t.label}
              <span className="ml-2 hidden text-[10px] uppercase tracking-wider text-ink-500 xl:inline">
                · {t.hint}
              </span>
              {active && (
                <span className="absolute bottom-[-1px] left-3 right-3 h-[2px] rounded-full bg-gradient-to-r from-accent-300 via-accent-400 to-warmth" />
              )}
            </button>
          );
        })}
      </nav>
    </div>
  );
}
