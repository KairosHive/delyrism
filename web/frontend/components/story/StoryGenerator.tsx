"use client";
import * as React from "react";
import { useMutation, useQuery } from "@tanstack/react-query";
import { api, StoryResponse } from "@/lib/api";
import { useSidebar, buildContextWeights } from "@/lib/store";
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
  const spaceId = useSidebar((s) => s.spaceId);
  const sentence = useSidebar((s) => s.contextSentence);
  const weights = useSidebar(buildContextWeights);

  const models = useQuery({ queryKey: ["story-models"], queryFn: () => api.get<ModelsResp>("/story/models") });

  const [model, setModel] = React.useState("@cf/meta/llama-3.3-70b-instruct-fp8-fast");
  const [tone, setTone] = React.useState("dreamy");
  const [language, setLanguage] = React.useState("English");
  const [pov, setPov] = React.useState<"first" | "third">("third");
  const [tense, setTense] = React.useState<"present" | "past">("present");
  const [length, setLength] = React.useState(180);
  const [temperature, setTemperature] = React.useState(0.85);
  const [topP, setTopP] = React.useState(0.9);
  const [positiveOnly, setPositiveOnly] = React.useState(true);

  const story = useMutation({
    mutationFn: () =>
      api.post<StoryResponse>("/story/generate", {
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
      }),
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
            onChange={setModel}
            options={Object.entries(models.data?.cloudflare ?? {}).map(([label, value]) => ({ label, value }))}
          />
        </Section>

        <Section title="Narrative" defaultOpen>
          <Select label="Language" value={language} onChange={setLanguage}
            options={[
              { value: "English", label: "English" },
              { value: "Français", label: "Français" },
              { value: "Español", label: "Español" },
            ]} />
          <Select label="POV" value={pov} onChange={(v) => setPov(v as any)}
            options={[{ value: "first", label: "First person" }, { value: "third", label: "Third person" }]} />
          <Select label="Tense" value={tense} onChange={(v) => setTense(v as any)}
            options={[{ value: "present", label: "Present" }, { value: "past", label: "Past" }]} />
          <Slider label="Length (words)" value={length} min={80} max={500} step={10}
            onChange={(v) => setLength(Math.round(v))} />
        </Section>

        <Section title="Atmosphere" defaultOpen>
          <Select label="Tone" value={tone} onChange={setTone}
            options={TONES.map((t) => ({ value: t, label: t }))} />
          <Slider label="Temperature" value={temperature} min={0.1} max={1.8} step={0.05}
            onChange={setTemperature} />
          <Slider label="Top-p" value={topP} min={0.1} max={1} step={0.05}
            onChange={setTopP} />
          <Toggle label="Positive Δ edges only" value={positiveOnly}
            onChange={setPositiveOnly} hint="extract motifs only from strengthening edges" />
        </Section>

        <button
          className="btn-primary w-full"
          onClick={() => story.mutate()}
          disabled={story.isPending}
        >
          {story.isPending ? "Spinning a tale…" : "Generate story"}
        </button>
      </div>

      <div className="panel-pad min-h-[400px] space-y-3">
        <div className="flex items-center justify-between">
          <div className="section-title">Story</div>
          {story.data?.motifs?.length ? (
            <div className="flex flex-wrap items-center gap-1 text-[10px] text-ink-300">
              motifs:
              {story.data.motifs.slice(0, 10).map((m) => (
                <span className="pill" key={m}>{m}</span>
              ))}
            </div>
          ) : null}
        </div>
        {story.isError && (
          <div className="text-sm text-danger">{(story.error as Error).message}</div>
        )}
        {!story.data && !story.isPending && (
          <p className="text-sm text-ink-400">Press <span className="font-mono">Generate story</span> to weave a micro-fiction from the current Δ-graph.</p>
        )}
        {story.data && (
          <div className="whitespace-pre-wrap text-[15px] leading-relaxed text-ink-100">
            {story.data.story}
          </div>
        )}
      </div>
    </div>
  );
}
