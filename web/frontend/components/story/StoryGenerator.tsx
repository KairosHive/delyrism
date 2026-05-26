"use client";
import * as React from "react";
import { useMutation, useQuery } from "@tanstack/react-query";
import { api, StoryResponse } from "@/lib/api";
import { useSidebar, buildContextWeights } from "@/lib/store";
import { deltaGraphPayload } from "@/lib/hooks";
import { Section } from "../ui/Section";
import { Slider } from "../ui/Slider";
import { Select } from "../ui/Select";
import { Toggle } from "../ui/Toggle";

// In alphabetical order so the dropdown is scannable. `dreamy` first as a
// neutral default; the rest are named literary registers.
const TONES = [
  "dreamy",
  "angela-carter",
  "blake",
  "borges",
  "calvino",
  "cosmic-horror",
  "garcia-marquez",
  "gnostic-techno",
  "homeric",
  "kafkaesque",
  "murakami",
  "mystic-baroque",
  "psalmic",
  "pynchon",
  "tarkovsky",
];

// Distinct accent colour per section so the story sidebar reads as four
// clearly-separated cards rather than one long uniform list.  Each colour
// pairs with a small icon for redundant signalling.
const STORY_COLORS = {
  model:      "#9b59b6", // purple — the machine
  narrative:  "#3498db", // blue  — text / structure
  motifs:     "#e67e22", // orange — symbolic anchors
  atmosphere: "#bf616a", // red    — mood / heat
} as const;

const FORMS: { value: string; label: string }[] = [
  { value: "prose",       label: "Prose (one paragraph)" },
  { value: "short-story", label: "Short story (2–4 paragraphs)" },
  { value: "poem",        label: "Poem (free verse)" },
  { value: "myth",        label: "Myth (cosmic / etiological)" },
  { value: "incantation", label: "Incantation (ritual / 2nd-person)" },
  { value: "vignette",    label: "Vignette (single held scene)" },
];

interface ModelsResp { cloudflare: Record<string, string>; }

export function StoryGenerator() {
  // ---- read everything from the global store so it survives tab switches ----
  const spaceId = useSidebar((s) => s.spaceId);
  const sentence = useSidebar((s) => s.contextSentence);
  const weights = useSidebar(buildContextWeights);

  const model = useSidebar((s) => s.storyModel);
  const tone = useSidebar((s) => s.storyTone);
  const language = useSidebar((s) => s.storyLanguage);
  const pov = useSidebar((s) => s.storyPov);
  const tense = useSidebar((s) => s.storyTense);
  const form = useSidebar((s) => s.storyForm);
  const length = useSidebar((s) => s.storyLengthWords);
  const temperature = useSidebar((s) => s.storyTemperature);
  const topP = useSidebar((s) => s.storyTopP);
  const positiveOnly = useSidebar((s) => s.storyPositiveOnly);
  const anchor = useSidebar((s) => s.storyAnchor);
  const motifDensity = useSidebar((s) => s.storyMotifDensity);
  const motifSource = useSidebar((s) => s.storyMotifSource);
  const transformationMode = useSidebar((s) => s.storyTransformationMode);
  const cycleDim = useSidebar((s) => s.storyCycleDim);
  const symbols = useSidebar((s) => s.symbols);
  const storyResult = useSidebar((s) => s.storyResult);
  const storyError = useSidebar((s) => s.storyError);
  const set = useSidebar((s) => s.set);

  const models = useQuery({ queryKey: ["story-models"], queryFn: () => api.get<ModelsResp>("/story/models") });

  const story = useMutation({
    mutationFn: async () => {
      // Pull the CURRENT sidebar Δ-graph state and pass it through.  This
      // is what guarantees the motifs surfaced under the story match the
      // Δ-graph the user actually sees in the Explorer tab — without this,
      // the backend re-derives a default-parameter Δ-graph internally and
      // the motif words drift from the visible graph.
      const delta_params = deltaGraphPayload();
      return api.post<StoryResponse>("/story/generate", {
        space_id: spaceId,
        sentence: sentence.trim() || null,
        weights,
        provider: "cloudflare",
        model,
        tone,
        language,
        pov,
        tense,
        form,
        length_words: length,
        temperature,
        top_p: topP,
        positive_delta_only: positiveOnly,
        anchor_archetype: anchor || null,
        motif_density: motifDensity,
        motif_source: motifSource,
        transformation_mode: transformationMode,
        cycle_dim: cycleDim,
        delta_params,
      });
    },
    onSuccess: (data) => {
      set("storyResult", { story: data.story, motifs: data.motifs, model: data.model, auto_target: data.auto_target ?? null });
      set("storyError", null);
    },
    onError: (err: Error) => {
      set("storyError", err.message);
    },
  });

  if (!spaceId) {
    return (
      <div className="panel-pad text-sm text-ink-300">Build a space first to generate stories.</div>
    );
  }

  return (
    <div className="grid grid-cols-1 gap-4 xl:grid-cols-[1fr,1.5fr]">
      <div className="space-y-3">
        <Section title="Model" defaultOpen color={STORY_COLORS.model} icon="🧠">
          <Select
            label="Provider · model"
            value={model}
            onChange={(v) => set("storyModel", v)}
            options={Object.entries(models.data?.cloudflare ?? {}).map(([label, value]) => ({ label, value }))}
            help="Which Cloudflare Workers AI model writes the story. Llama 3.3 70B is the most reliable; smaller models are faster but more uneven."
          />
        </Section>

        <Section title="Narrative" defaultOpen color={STORY_COLORS.narrative} icon="📖">
          <Select label="Language" value={language} onChange={(v) => set("storyLanguage", v as any)}
            options={[
              { value: "English", label: "English" },
              { value: "Français", label: "Français" },
              { value: "Español", label: "Español" },
            ]}
            help="Output language. The prompt and style directives are localized for each — the model writes natively in the chosen language." />
          <Select label="POV" value={pov} onChange={(v) => set("storyPov", v as any)}
            options={[{ value: "first", label: "First person" }, { value: "third", label: "Third person" }]}
            help="Narrative perspective. First-person feels more intimate; third gives more authorial distance." />
          <Select label="Tense" value={tense} onChange={(v) => set("storyTense", v as any)}
            options={[
              { value: "present", label: "Present" },
              { value: "past", label: "Past" },
              { value: "future", label: "Future" },
            ]}
            help="Present tense reads immediate / dreamlike; past tense reads mythic / recounted; future tense reads prophetic / oracular — the events are still to come." />
          <Slider label="Length (words)" value={length} min={80} max={500} step={10}
            onChange={(v) => set("storyLengthWords", Math.round(v))}
            help="Target word count. The prompt asks for length≈[low, high] = target ± 40 words. Longer stories get more motifs woven in but also drift more." />
          <Select label="Form" value={form} onChange={(v) => set("storyForm", v as any)}
            options={FORMS}
            help="Output shape — separate from Tone (register/style). Prose = one paragraph. Short story = 2–4 paragraphs with a turn. Poem = line breaks and stanzas. Myth = cosmic/etiological. Incantation = ritual repetition with 2nd-person address. Vignette = a single held scene." />
        </Section>

        <Section title="Anchor & Motifs" defaultOpen color={STORY_COLORS.motifs} icon="⚓">
          <Select label="Anchor archetype" value={anchor} onChange={(v) => set("storyAnchor", v)}
            options={[
              { value: "",     label: "— none —" },
              { value: "auto", label: "(auto) top-ranked under current context" },
              ...symbols.map((s) => ({ value: s, label: s })),
            ]}
            help="Pin one archetype as the story's center. 'auto' picks whichever symbol scored highest in the Ranked Archetypes panel for the current context. A specific symbol forces the story to embody it regardless of ranking. None = no anchor line in the prompt." />
          <Slider label="Motif density" value={motifDensity} min={4} max={24} step={1}
            onChange={(v) => set("storyMotifDensity", Math.round(v))}
            help="How many motif words the prompt explicitly asks the model to weave in. More = richer texture but harder for the LLM to integrate naturally. 8–14 is the sweet spot." />
          <Select label="Motif source" value={motifSource} onChange={(v) => set("storyMotifSource", v as any)}
            options={[
              { value: "delta-graph",     label: "Δ-graph nodes (relational shift)" },
              { value: "top-attention",   label: "Top-attention descriptors (sharp focus)" },
              { value: "mixed",           label: "Mixed (Δ + attention)" },
              { value: "transformation",  label: "Contextual transformation (becoming)" },
              { value: "cycle",           label: "Cycle journey (loop as story spine)" },
            ]}
            help="Where motif words come from. Δ-graph / top-attention / mixed surface descriptor words by various criteria. The two topology-driven sources structure the story around archetypal motion: transformation tells the arc of an archetype's drift under context; cycle traces a persistent loop in semantic space as a closed narrative spine." />

          {motifSource === "transformation" && (
            <div className="rounded-md border border-warmth/40 bg-warmth/10 p-2 space-y-1">
              <div className="text-[10px] uppercase tracking-widest text-warmth">
                Transformation mode
              </div>
              <Select label="Mode" value={transformationMode} onChange={(v) => set("storyTransformationMode", v as any)}
                options={[
                  { value: "becoming",  label: "Becoming (both, before → after)" },
                  { value: "emergence", label: "Emergence (what enters the archetype)" },
                  { value: "fading",    label: "Fading (what leaves the archetype)" },
                ]}
                help="Which slice of the archetype's identity-card drift drives the story." />
              <p className="text-[10px] leading-snug text-ink-400">
                {anchor
                  ? <>Target: <span className="text-warmth">{anchor === "auto" ? "(top-ranked)" : anchor}</span></>
                  : <><span className="text-warmth">Anchor is none</span> → backend auto-picks the most-transformed archetype (max |emerged|+|faded|).</>}
              </p>
            </div>
          )}

          {motifSource === "cycle" && (
            <div className="rounded-md border border-accent-500/40 bg-accent-600/10 p-2 space-y-1">
              <div className="text-[10px] uppercase tracking-widest text-accent-300">
                Cycle journey
              </div>
              <Select label="Dimension" value={cycleDim} onChange={(v) => set("storyCycleDim", v as any)}
                options={[
                  { value: "h1", label: "H1 — semantic loop (closed path of words)" },
                  { value: "h2", label: "H2 — void (circle the absence)" },
                ]}
                help="H1 loops give an ordered word-trail to walk. H2 voids give an unordered vertex set surrounding an absence the story circles around." />
              <p className="text-[10px] leading-snug text-ink-400">
                {anchor
                  ? <>Target: <span className="text-accent-300">{anchor === "auto" ? "(top-ranked)" : anchor}</span></>
                  : <><span className="text-accent-300">Anchor is none</span> → backend auto-picks the archetype with the most-persistent {cycleDim.toUpperCase()} cycle.</>}
              </p>
            </div>
          )}
        </Section>

        <Section title="Atmosphere" defaultOpen color={STORY_COLORS.atmosphere} icon="🌫">
          <Select label="Tone" value={tone} onChange={(v) => set("storyTone", v)}
            options={TONES.map((t) => ({ value: t, label: t }))}
            help="Style register. Plain tones (dreamy/eerie/warm) just adjust adjectives; the named-author tones (pynchon/blake/mystic-baroque/gnostic-techno) inject specific style directives, lexicons, and avoid-lists into the prompt." />
          <Slider label="Temperature" value={temperature} min={0.1} max={1.8} step={0.05}
            onChange={(v) => set("storyTemperature", v)}
            help="LLM randomness. Lower (~0.3) = predictable, on-topic. Higher (~1.2) = surprising, sometimes incoherent. 0.7–0.9 is the sweet spot for mythopoetic prose." />
          <Slider label="Top-p" value={topP} min={0.1} max={1} step={0.05}
            onChange={(v) => set("storyTopP", v)}
            help="Nucleus sampling cutoff. Restricts the model to the top tokens whose probabilities sum to p. Lower = more conservative word choice; 0.9 is the standard." />
          <Toggle label="Positive Δ edges only" value={positiveOnly}
            onChange={(v) => set("storyPositiveOnly", v)}
            help="When extracting motif words from the Δ-graph, use only edges that STRENGTHENED under context. Off = also include weakened edges (less coherent motifs, but more variety)." />
        </Section>

        <button
          className="btn-primary w-full"
          onClick={() => story.mutate()}
          disabled={story.isPending}
        >
          {story.isPending ? "Spinning a tale…" : "Generate story"}
        </button>

        <p className="text-[10px] text-ink-400 leading-relaxed">
          Motifs are extracted from the Δ-graph using the current Δ Graph sidebar settings
          (strategy, β, gate, sign filter, etc.).  Change those to steer which words feed
          the prompt.
        </p>
      </div>

      <div className="panel-pad min-h-[400px] space-y-3">
        <div className="flex items-center justify-between gap-3">
          <div className="flex items-center gap-2">
            <div className="section-title">Story</div>
            {storyResult?.auto_target && (
              <span
                className="pill !text-[10px] border-accent-500/60 bg-accent-600/15 text-accent-200"
                title="Anchor was none — backend auto-picked this archetype for the chosen topology source."
              >
                auto · {storyResult.auto_target}
              </span>
            )}
          </div>
          {storyResult?.motifs?.length ? (
            <div className="flex max-w-[60%] flex-wrap items-center gap-1 text-[10px] text-ink-300">
              motifs:
              {storyResult.motifs.slice(0, 12).map((m) => (
                <span className="pill" key={m}>{m}</span>
              ))}
            </div>
          ) : null}
        </div>
        {storyError && !story.isPending && (
          <div className="text-sm text-danger">{storyError}</div>
        )}
        {!storyResult && !story.isPending && !storyError && (
          <p className="text-sm text-ink-400">
            Press <span className="font-mono">Generate story</span> to weave a micro-fiction
            from the current Δ-graph.
          </p>
        )}
        {story.isPending && (
          <div className="flex items-center gap-2 text-sm text-ink-300">
            <span className="h-2 w-2 animate-pulse rounded-full bg-accent-400" />
            spinning a tale…
          </div>
        )}
        {storyResult && (
          <div className="whitespace-pre-wrap text-[15px] leading-relaxed text-ink-100">
            {storyResult.story}
          </div>
        )}
      </div>
    </div>
  );
}
