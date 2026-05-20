"use client";
import * as React from "react";
import { Sidebar } from "@/components/sidebar/Sidebar";
import { Explorer } from "@/components/explorer/Explorer";
import { StoryGenerator } from "@/components/story/StoryGenerator";
import { ArchetypeBuilder } from "@/components/builder/ArchetypeBuilder";
import { Title } from "@/components/header/Title";
import { Console } from "@/components/header/Console";
import { BuildButton } from "@/components/header/BuildButton";
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
  const [sidebarOpen, setSidebarOpen] = React.useState(false);
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

  // Auto-close the mobile drawer on tab change so it doesn't cover the new view.
  React.useEffect(() => { setSidebarOpen(false); }, [tab]);

  return (
    <div className="md:grid md:h-screen md:grid-cols-[300px,1fr]">
      {/* Sidebar:
            • md+    → static column, part of the grid
            • mobile → fixed off-canvas drawer with backdrop, toggled by ☰ */}
      <aside
        className={`fixed inset-y-0 left-0 z-40 w-[86vw] max-w-[340px] transform border-r border-ink-700/60
                    bg-ink-900/95 backdrop-blur-md transition-transform duration-200
                    md:static md:z-0 md:w-auto md:max-w-none md:translate-x-0 md:bg-ink-900/40
                    ${sidebarOpen ? "translate-x-0" : "-translate-x-full"}`}
      >
        <Sidebar />
      </aside>

      {/* Backdrop — only rendered on mobile when the drawer is open */}
      {sidebarOpen && (
        <button
          aria-label="close sidebar"
          className="fixed inset-0 z-30 bg-black/60 backdrop-blur-[2px] md:hidden"
          onClick={() => setSidebarOpen(false)}
        />
      )}

      <main className="flex min-h-screen flex-col overflow-y-auto md:h-full md:min-h-0">
        <MobileTopBar onMenu={() => setSidebarOpen(true)} />

        <Title />

        <div className="px-3 pb-2 md:px-8">
          <Console />
          <BuildButton />
        </div>

        <Tabs tab={tab} setTab={setTab} />

        <section className="flex-1 px-3 pb-10 pt-4 md:px-8 md:pt-6">
          {tab === "explorer" && <Explorer />}
          {tab === "story" && <StoryGenerator />}
          {tab === "builder" && <ArchetypeBuilder />}
        </section>
      </main>
      <TimingBadge />
    </div>
  );
}

/** Slim sticky top bar — only visible below md.  Holds the hamburger that
 *  opens the sidebar drawer and shows the active space pill, so the user
 *  always has access to controls without a giant header eating screen
 *  real estate. */
function MobileTopBar({ onMenu }: { onMenu: () => void }) {
  const spaceId = useSidebar((s) => s.spaceId);
  return (
    <div className="sticky top-0 z-20 flex items-center justify-between gap-2 border-b border-ink-700/50
                    bg-ink-950/85 px-3 py-2 backdrop-blur-md md:hidden">
      <button
        aria-label="open sidebar"
        onClick={onMenu}
        className="inline-flex h-9 w-9 items-center justify-center rounded-md border border-ink-700 bg-ink-900/80 text-ink-100 active:bg-ink-800"
      >
        {/* eslint-disable-next-line @next/next/no-img-element */}
        <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round">
          <path d="M3 6h18M3 12h18M3 18h18" />
        </svg>
      </button>
      <div className="flex items-center gap-2 text-[11px] text-ink-400">
        {spaceId ? (
          <>
            <span className="h-1.5 w-1.5 rounded-full bg-accent-400" />
            <span className="font-mono">space {spaceId.slice(0, 6)}…</span>
          </>
        ) : (
          <span className="text-ink-500">no space yet — open menu to build</span>
        )}
      </div>
    </div>
  );
}

function Tabs({ tab, setTab }: { tab: TabId; setTab: (t: TabId) => void }) {
  return (
    <div className="mx-auto mt-2 w-full max-w-5xl border-b border-ink-700/50 px-1">
      {/* overflow-x-auto + whitespace-nowrap so tab labels scroll horizontally
          on phones instead of wrapping / squishing */}
      <nav className="flex items-center gap-1 overflow-x-auto whitespace-nowrap">
        {TABS.map((t) => {
          const active = t.id === tab;
          return (
            <button
              key={t.id}
              onClick={() => setTab(t.id)}
              className={`relative shrink-0 px-3 py-2.5 text-sm transition md:px-4
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
