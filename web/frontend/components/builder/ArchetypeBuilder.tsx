"use client";
import * as React from "react";
import { useSidebar } from "@/lib/store";

/**
 * The Builder tab invites to Egregore, the archetype builder that lives at its own address:
 * its logo at the centre of Egregore's own spiral of orbiting particles (five rings, purple to
 * cyan, alternating directions, comet trails), as on Egregore's empty state.  A pasted Egregore
 * export still drops straight into the sidebar.
 */
const EGREGORE_URL = process.env.NEXT_PUBLIC_EGREGORE_URL || "https://egregore.kairos-hive.org";
const SIZE = 340;

interface Dot { key: string; r: number; size: number; color: string; glow: string; opacity: number; blur: number;
                speed: number; delay: number; dir: "normal" | "reverse"; angle: number }

/** The logo's own palette, ring by ring from the centre out (sampled from egregore-logo.png):
 *  pale cyan light, sky blue, deep blue, then the gold and bronze of its outline.
 *  [hue, saturation, trail lightness, lead lightness] */
const RINGS: [number, number, number, number][] = [
  [186, 70, 74, 86],   // #8ce2ec
  [199, 68, 58, 72],   // #48aedd
  [216, 52, 52, 66],   // #3461a3, lifted to show on the dark panel
  [45, 55, 70, 82],    // #d9c791
  [34, 32, 52, 66],    // #967a57
];

/** Egregore's spiral (miner_ui.html initSpiralAnimation), scaled to SIZE, in the logo's colours. */
function orbitDots(): Dot[] {
  const dots: Dot[] = [];
  const layers = RINGS.length, trail = 8, scale = SIZE / 250;
  for (let layer = 0; layer < layers; layer++) {
    const r = (25 + layer * 22) * scale + 34;            // leave the logo room
    const n = 5 + layer;
    const speed = 7 + layer * 3;
    const dir = layer % 2 === 0 ? "normal" : "reverse";
    const [hue, sat, light, lead] = RINGS[layer];
    const color = `hsl(${hue}, ${sat}%, ${light}%)`;
    const base = 3 + (layers - layer) * 0.7;
    for (let i = 0; i < n; i++) {
      const angle = (i / n) * 360;
      for (let t = trail - 1; t >= 0; t--) {
        const p = t / trail;
        dots.push({
          key: `${layer}-${i}-${t}`, r, size: base * (1 - p * 0.6), color, glow: `0 0 ${4 + (1 - p) * 8}px ${color}`,
          opacity: Math.pow(1 - p, 1.5) * 0.85, blur: p * 3, speed,
          delay: ((-angle - (dir === "normal" ? 1 : -1) * t * 4) / 360) * speed, dir, angle: angle - t * 4,
        });
      }
      dots.push({
        key: `${layer}-${i}-lead`, r, size: base, color: `hsl(${hue}, ${Math.min(95, sat + 15)}%, ${lead}%)`,
        glow: `0 0 ${8 + layer}px ${color}, 0 0 ${15 + layer * 2}px ${color}`, opacity: 1, blur: 0,
        speed, delay: (-angle / 360) * speed, dir, angle,
      });
    }
  }
  return dots;
}

export function ArchetypeBuilder() {
  const dots = React.useMemo(orbitDots, []);
  return (
    <div className="space-y-4">
      <div className="panel relative overflow-hidden px-6 py-10">
        <div className="mx-auto flex max-w-3xl flex-col items-center gap-8 md:flex-row md:gap-12">
          <div className="relative shrink-0" style={{ width: SIZE, height: SIZE, maxWidth: "80vw", maxHeight: "80vw" }} aria-hidden>
            <div className="egregore-halo absolute inset-0 rounded-full" />
            {dots.map((d) => (
              <span
                key={d.key}
                className="egregore-orbit"
                style={{
                  width: d.size, height: d.size, marginLeft: -d.size / 2, marginTop: -d.size / 2,
                  background: d.color, boxShadow: d.glow, opacity: d.opacity, filter: d.blur ? `blur(${d.blur}px)` : undefined,
                  ["--r" as any]: `${d.r}px`, ["--speed" as any]: `${d.speed}s`, ["--delay" as any]: `${d.delay}s`,
                  ["--dir" as any]: d.dir, ["--angle" as any]: d.angle,
                }}
              />
            ))}
            {/* eslint-disable-next-line @next/next/no-img-element */}
            <img
              src="/egregore-logo.png"
              alt=""
              className="egregore-logo absolute left-[27%] top-[27%] h-[46%] w-[46%] object-contain"
            />
          </div>
          <div className="space-y-4 text-center md:text-left">
            <div className="text-[11px] uppercase tracking-[0.3em] text-accent-300">Egregore</div>
            <h2 className="text-2xl font-light leading-snug text-ink-50 md:text-3xl">Grow your own archetypes</h2>
            <p className="text-sm leading-relaxed text-ink-300">
              Egregore composes archetype sets from your texts, images and documents. It clusters what they
              share, names the archetypes and refines their descriptors with a language model. Export a set
              and it becomes a symbolic structure to explore here.
            </p>
            <a
              href={EGREGORE_URL}
              target="_blank"
              rel="noreferrer"
              className="btn-primary inline-flex items-center gap-2 px-6 py-2.5"
            >
              Open Egregore <span aria-hidden>↗</span>
            </a>
            <div className="font-mono text-[11px] text-ink-500">{EGREGORE_URL.replace(/^https?:\/\//, "")}</div>
          </div>
        </div>
      </div>
      <ImportExport />
    </div>
  );
}

/** Paste an Egregore export (or any {symbol: [descriptors]}) into the sidebar's symbol map. */
function ImportExport() {
  const set = useSidebar((s) => s.set);
  const [msg, setMsg] = React.useState<string | null>(null);
  function load(text: string) {
    try {
      const parsed = JSON.parse(text);
      const map = parsed.symbols ?? parsed;
      set("symbolMapJson", JSON.stringify(map, null, 2));
      set("presetName", null);
      setMsg(`Loaded ${Object.keys(map).length} symbols into the sidebar.`);
    } catch (e: any) {
      setMsg(`Could not parse JSON: ${e.message}`);
    }
  }
  return (
    <details className="panel-pad text-xs text-ink-300">
      <summary className="cursor-pointer select-none text-ink-200">Already have an Egregore export? Paste it here</summary>
      <textarea
        className="input-base mt-3 h-32 font-mono text-xs"
        placeholder='{ "ORACLE": ["liminal", "voice", "threshold"], ... }'
        onPaste={(e) => { const t = e.clipboardData.getData("text"); setTimeout(() => load(t), 0); }}
        onChange={(e) => load(e.target.value)}
      />
      {msg && <div className="mt-2 text-accent-300">{msg}</div>}
    </details>
  );
}
