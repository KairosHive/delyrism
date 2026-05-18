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

const TONES = [
  "dreamy", "eerie", "warm",
  "pynchon", "blake", "mystic-baroque", "gnostic-techno",
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
  const length = useSidebar((s) => s.storyLengthWords);
  const temperature = useSidebar((s) => s.storyTemperature);
  const topP = useSidebar((s) => s.storyTopP);
  const positiveOnly = useSidebar((s) => s.storyPositiveOnly);
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
        length_words: length,
        temperature,
        top_p: topP,
        positive_delta_only: positiveOnly,
        delta_params,
      });
    },
    onSuccess: (data) => {
      set("storyResult", { story: data.story, motifs: data.motifs, model: data.model });
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
        <Section title="Model" defaultOpen>
          <Select
            label="Provider · model"
            value={model}
            onChange={(v) => set("storyModel", v)}
            options={Object.entries(models.data?.cloudflare ?? {}).map(([label, value]) => ({ label, value }))}
            help="Which Cloudflare Workers AI model writes the story. Llama 3.3 70B is the most reliable; smaller models are faster but more uneven."
          />
        </Section>

        <Section title="Narrative" defaultOpen>
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
            options={[{ value: "present", label: "Present" }, { value: "past", label: "Past" }]}
            help="Present tense reads immediate / dreamlike; past tense reads mythic / recounted." />
          <Slider label="Length (words)" value={length} min={80} max={500} step={10}
            onChange={(v) => set("storyLengthWords", Math.round(v))}
            help="Target word count. The prompt asks for length≈[low, high] = target ± 40 words. Longer stories get more motifs woven in but also drift more." />
        </Section>

        <Section title="Atmosphere" defaultOpen>
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
        <div className="flex items-center justify-between">
          <div className="section-title">Story</div>
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
