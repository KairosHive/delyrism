"use client";
import * as React from "react";
import { keepPreviousData, useQuery } from "@tanstack/react-query";
import {
  api,
  TransformationsResponse,
  MigrationEntry,
  ArchetypeIdentityCard,
  IdentityEntry,
} from "@/lib/api";
import { useSidebar, buildContextWeights } from "@/lib/store";
import { Skeleton } from "../ui/Skeleton";

/**
 * Contextual transformations — two concrete views on top of the same shift:
 *
 *   ▸ Migrations: descriptors whose nearest archetype flipped under context
 *     ("ash: FIRE → WATER under quiet grief").  A narrative ranking.
 *
 *   ▸ Identity cards: per-archetype before/after top-K descriptors.  Shows
 *     what each archetype "becomes" in the current context — including
 *     foreign descriptors that drifted in.
 *
 * Replaces the abstract spectrum as the primary "context-effect" view.
 * Descriptors are coloured by their HOME archetype so migrants visually
 * stand out wherever they show up.
 */
export function ContextualTransformations() {
  const sid = useSidebar((s) => s.spaceId);
  const colorMap = useSidebar((s) => s.colorMap);
  const sentence = useSidebar((s) => s.contextSentence);
  const weights = useSidebar(buildContextWeights);
  const strategy = useSidebar((s) => s.strategy);
  const beta = useSidebar((s) => s.beta);
  const gate = useSidebar((s) => s.gate);
  const tau = useSidebar((s) => s.shiftTau);
  const wss = useSidebar((s) => s.withinSymbolSoftmax);
  const gamma = useSidebar((s) => s.gamma);
  const poolType = useSidebar((s) => s.poolType);
  const poolW = useSidebar((s) => s.poolW);
  const mAlpha = useSidebar((s) => s.membershipAlpha);

  const audioActive = useSidebar((s) => s.audioActive);
  const imageActive = useSidebar((s) => s.imageActive);
  const morphActive = useSidebar((s) => s.morphActive);
  const audioNonce = useSidebar((s) => s.audioNonce);
  const imageNonce = useSidebar((s) => s.imageNonce);
  const morphNonce = useSidebar((s) => s.morphNonce);

  const hasCtx = !!sentence.trim() || !!weights || audioActive || imageActive || morphActive;

  const q = useQuery({
    enabled: !!sid && hasCtx,
    placeholderData: keepPreviousData,
    queryKey: [
      "transformations", sid, sentence, weights, strategy, beta, gate, tau, wss, gamma,
      poolType, poolW, mAlpha, audioNonce, imageNonce, morphNonce,
    ],
    queryFn: () =>
      api.post<TransformationsResponse>("/transformations", {
        space_id: sid,
        sentence: sentence.trim() || null,
        weights,
        strategy, beta, gate, tau,
        within_symbol_softmax: wss,
        gamma,
        pool_type: poolType,
        pool_w: poolW,
        membership_alpha: mAlpha,
        topk: 6,
      }),
  });

  return (
    <div className="panel-tight">
      <div className="mb-2 flex items-baseline justify-between gap-3">
        <div>
          <div className="section-title flex items-center gap-2">
            Contextual transformations
            {q.isFetching && q.data && (
              <span className="inline-flex items-center gap-1 text-[10px] text-ink-400">
                <span className="h-1.5 w-1.5 animate-pulse rounded-full bg-accent-400" />
                refreshing
              </span>
            )}
          </div>
          <div className="text-[11px] text-ink-400">
            who switched archetypes, and what each archetype looks like now
          </div>
        </div>
      </div>

      {!hasCtx && (
        <div className="p-6 text-sm text-ink-300">
          Add a context (sentence, weights, audio, image, or morphing blend) to see what changes.
        </div>
      )}
      {hasCtx && q.isPending && !q.data && (
        <div className="space-y-3 py-2">
          <Skeleton lines={3} />
          <div className="grid grid-cols-1 gap-3 md:grid-cols-3">
            <Skeleton height={180} />
            <Skeleton height={180} />
            <Skeleton height={180} />
          </div>
        </div>
      )}
      {q.data && <Content data={q.data} colorMap={colorMap} />}
    </div>
  );
}

function Content({
  data, colorMap,
}: {
  data: TransformationsResponse;
  colorMap: Record<string, string>;
}) {
  return (
    <div className="space-y-4">
      <MigrationsBlock migrations={data.migrations} colorMap={colorMap} />
      <IdentityGrid identities={data.identities} colorMap={colorMap} />
    </div>
  );
}

// ─── Migrations: who switched archetypes ────────────────────────────────────

function MigrationsBlock({
  migrations, colorMap,
}: {
  migrations: MigrationEntry[];
  colorMap: Record<string, string>;
}) {
  const [expanded, setExpanded] = React.useState(false);
  const TRUNCATED = 5;
  const showAll = expanded || migrations.length <= TRUNCATED;
  const shown = showAll ? migrations : migrations.slice(0, TRUNCATED);

  return (
    <div>
      <div className="mb-1.5 flex items-baseline justify-between">
        <div className="text-[10px] uppercase tracking-widest text-ink-400">
          Who switched archetypes
        </div>
        {migrations.length > TRUNCATED && (
          <button
            className="text-[10px] text-accent-300 hover:text-accent-200"
            onClick={() => setExpanded(!expanded)}
          >
            {expanded ? "show fewer" : `show all ${migrations.length}`}
          </button>
        )}
      </div>

      {migrations.length === 0 ? (
        <div className="rounded-md border border-ink-700/60 bg-ink-900/30 p-3 text-[11px] text-ink-500">
          no archetype switches — every descriptor's nearest archetype stayed the same
        </div>
      ) : (
        <div className="space-y-0.5 rounded-md border border-ink-700/60 bg-ink-900/30 p-2">
          {shown.map((m, i) => (
            <MigrationRow key={i} m={m} colorMap={colorMap} />
          ))}
        </div>
      )}
    </div>
  );
}

function MigrationRow({ m, colorMap }: { m: MigrationEntry; colorMap: Record<string, string> }) {
  const fromC = colorMap[m.from_archetype] ?? "#888";
  const toC = colorMap[m.to_archetype] ?? "#888";
  return (
    <div
      className="grid grid-cols-[1.2fr,auto,1.2fr,auto] items-center gap-2 rounded px-1.5 py-1 text-[11px] hover:bg-ink-800/40"
      title={
        `${m.descriptor}\n` +
        `from ${m.from_archetype}: ${m.sim_before_from.toFixed(2)} → ${m.sim_after_from.toFixed(2)}\n` +
        `to   ${m.to_archetype}: ${m.sim_before_to.toFixed(2)} → ${m.sim_after_to.toFixed(2)}`
      }
    >
      <div className="truncate">
        <span style={{ color: fromC }} className="font-medium">{m.descriptor}</span>
      </div>
      <div className="text-ink-500">·</div>
      <div className="flex items-center gap-1.5 truncate">
        <ArchPill name={m.from_archetype} color={fromC} muted />
        <span className="text-ink-400">→</span>
        <ArchPill name={m.to_archetype} color={toC} />
      </div>
      <div className="font-mono text-[10px] tabular-nums text-ink-400">+{m.score.toFixed(2)}</div>
    </div>
  );
}

function ArchPill({
  name, color, muted = false,
}: {
  name: string;
  color: string;
  muted?: boolean;
}) {
  return (
    <span
      className="rounded-md border px-1.5 py-0.5 text-[10px] font-medium"
      style={{
        color,
        borderColor: color + (muted ? "44" : "70"),
        background: color + (muted ? "0c" : "18"),
      }}
    >
      {name}
    </span>
  );
}

// ─── Identity cards: what each archetype looks like now ─────────────────────

function IdentityGrid({
  identities, colorMap,
}: {
  identities: ArchetypeIdentityCard[];
  colorMap: Record<string, string>;
}) {
  return (
    <div>
      <div
        className="mb-1.5 flex items-center gap-1.5 text-[10px] uppercase tracking-widest text-ink-400"
        title={
          "Both columns rank ALL descriptors (any archetype) by similarity to this archetype's centroid.\n" +
          "  originally = ranked against the ORIGINAL centroid (no context).\n" +
          "  under context = ranked against the SHIFTED centroid after context applied.\n" +
          "Foreign descriptors that sit naturally near this archetype's centroid appear in BOTH columns.\n" +
          "The '+ new' chip flags only descriptors that genuinely entered the top-K because of context."
        }
      >
        What each archetype looks like now
        <span className="text-ink-600">ⓘ</span>
      </div>
      <div className="grid grid-cols-1 gap-3 md:grid-cols-2 xl:grid-cols-3">
        {identities.map((card) => (
          <IdentityCard key={card.symbol} card={card} colorMap={colorMap} />
        ))}
      </div>
    </div>
  );
}

function IdentityCard({
  card, colorMap,
}: {
  card: ArchetypeIdentityCard;
  colorMap: Record<string, string>;
}) {
  const headerColor = colorMap[card.symbol] ?? "#888";
  const emergedSet = new Set(card.emerged);
  const fadedSet = new Set(card.faded);

  return (
    <div
      className="rounded-lg border bg-ink-900/40 p-2.5"
      style={{ borderColor: headerColor + "55" }}
    >
      <div
        className="mb-2 text-[11px] font-semibold uppercase tracking-wider"
        style={{ color: headerColor }}
      >
        {card.symbol}
      </div>

      <div className="grid grid-cols-2 gap-3 text-[11px]">
        <div>
          <div className="mb-1 text-[9px] uppercase tracking-widest text-ink-500">
            originally
          </div>
          <ul className="space-y-0.5">
            {card.before.map((e) => (
              <DescriptorLine
                key={e.descriptor}
                entry={e}
                cardArchetype={card.symbol}
                colorMap={colorMap}
                strike={fadedSet.has(e.descriptor)}
              />
            ))}
          </ul>
        </div>
        <div>
          <div className="mb-1 text-[9px] uppercase tracking-widest text-ink-500">
            under context
          </div>
          <ul className="space-y-0.5">
            {card.after.map((e) => (
              <DescriptorLine
                key={e.descriptor}
                entry={e}
                cardArchetype={card.symbol}
                colorMap={colorMap}
                emerged={emergedSet.has(e.descriptor)}
              />
            ))}
          </ul>
        </div>
      </div>

      {(card.emerged.length > 0 || card.faded.length > 0) && (
        <div className="mt-2 space-y-0.5 border-t border-ink-700/40 pt-1.5 text-[10px]">
          {card.emerged.length > 0 && (
            <div className="text-ink-300">
              <span className="text-accent-300">+ new:</span>{" "}
              {card.emerged.map((d, i) => (
                <React.Fragment key={d}>
                  {i > 0 && <span className="text-ink-600">, </span>}
                  <span>{d}</span>
                </React.Fragment>
              ))}
            </div>
          )}
          {card.faded.length > 0 && (
            <div className="text-ink-400">
              <span className="text-warmth">− faded:</span>{" "}
              {card.faded.map((d, i) => (
                <React.Fragment key={d}>
                  {i > 0 && <span className="text-ink-600">, </span>}
                  <span>{d}</span>
                </React.Fragment>
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  );
}

function DescriptorLine({
  entry, cardArchetype, colorMap, strike = false, emerged = false,
}: {
  entry: IdentityEntry;
  cardArchetype: string;
  colorMap: Record<string, string>;
  strike?: boolean;
  emerged?: boolean;
}) {
  // Colour by HOME archetype.  When a foreign descriptor shows up in
  // another card, its colour gives it away — and we tag it with a
  // small "← HOME" pill so the relationship is unambiguous.
  const c = colorMap[entry.owner] ?? "#cbd";
  const isForeign = entry.owner && entry.owner !== cardArchetype;
  return (
    <li className="flex items-center gap-1 leading-tight">
      {emerged && (
        <span
          className="inline-block h-1.5 w-1.5 shrink-0 rounded-full bg-accent-400"
          title="new top descriptor under this context"
        />
      )}
      <span
        className={`truncate ${strike ? "text-ink-500 line-through" : ""}`}
        style={{ color: strike ? undefined : c }}
      >
        {entry.descriptor}
      </span>
      {isForeign && !strike && (
        <span
          className="shrink-0 rounded border px-1 text-[8px] font-medium uppercase tracking-wider"
          style={{
            color: c,
            borderColor: c + "55",
            background: c + "0d",
          }}
          title={`home archetype: ${entry.owner}`}
        >
          {entry.owner}
        </span>
      )}
      <span className="ml-auto shrink-0 font-mono text-[9px] tabular-nums text-ink-500">
        {entry.score.toFixed(2)}
      </span>
    </li>
  );
}
