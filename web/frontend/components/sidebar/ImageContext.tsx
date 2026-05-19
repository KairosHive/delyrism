"use client";
import * as React from "react";
import { api } from "@/lib/api";
import { useSidebar } from "@/lib/store";

/**
 * Image context override (vision-LLM shim path).
 *
 * Flow:
 *   1) user drops, pastes, or browses an image
 *   2) POST /context/encode-image — backend asks a Cloudflare vision LLM
 *      to render the image as a short symbolic paragraph, then embeds that
 *      text via the space's text embedder, returning the vector + the
 *      description the LLM produced
 *   3) POST /context/set-override with the vector
 *   4) UI shows the LLM's reading next to a thumbnail so the user can see
 *      what the engine actually "saw" in their image
 *
 * Works with ANY text-embedder backend (BGE-M3, Qwen3, sentence-transformer)
 * — the image is converted to text inside the same embedding space as the
 * descriptors, so no backend rebuild is required.
 *
 * Image and audio override the same single backend slot
 * (`SymbolSpace.context_override`).  Setting an image clears any active
 * audio override and vice versa — surfaced visually so the user isn't
 * surprised.
 */
export function ImageContext() {
  const spaceId = useSidebar((s) => s.spaceId);
  const imageActive = useSidebar((s) => s.imageActive);
  const description = useSidebar((s) => s.imageDescription);
  const thumbnail = useSidebar((s) => s.imageThumbnail);
  const audioActive = useSidebar((s) => s.audioActive);
  const set = useSidebar((s) => s.set);

  const [status, setStatus] = React.useState<"idle" | "encoding" | "ok" | "error">("idle");
  const [error, setError] = React.useState<string | null>(null);
  const [dragOver, setDragOver] = React.useState(false);

  async function uploadFile(file: File) {
    if (!spaceId) return;
    if (!file.type.startsWith("image/")) {
      setStatus("error");
      setError("not an image file");
      return;
    }

    // generate thumbnail object-URL — replace any prior one
    if (thumbnail) URL.revokeObjectURL(thumbnail);
    const objUrl = URL.createObjectURL(file);
    set("imageThumbnail", objUrl);

    setStatus("encoding");
    setError(null);
    try {
      const fd = new FormData();
      fd.append("space_id", spaceId);
      fd.append("file", file);
      const enc = await api.upload<{ vector: number[]; dim: number; description: string; model: string }>(
        "/context/encode-image",
        fd,
      );
      await api.post("/context/set-override", { space_id: spaceId, vector: enc.vector });
      // if audio was on, the single backend slot just got replaced — sync UI
      if (audioActive) {
        set("audioActive", false);
        set("audioNonce", Date.now());
      }
      set("imageDescription", enc.description);
      set("imageActive", true);
      set("imageNonce", Date.now());
      setStatus("ok");
    } catch (e: any) {
      setStatus("error");
      setError(e?.message ?? "encoding failed");
    }
  }

  async function clearImage() {
    if (!spaceId) return;
    try {
      await api.post("/context/set-override", { space_id: spaceId, vector: null });
    } catch {}
    if (thumbnail) URL.revokeObjectURL(thumbnail);
    set("imageThumbnail", null);
    set("imageDescription", "");
    set("imageActive", false);
    set("imageNonce", Date.now());
    setStatus("idle");
    setError(null);
  }

  // Browser-wide paste: catches Ctrl+V from clipboard (e.g. screenshot tools)
  // even when no part of this component has keyboard focus.
  React.useEffect(() => {
    function onPaste(e: ClipboardEvent) {
      if (!spaceId) return;
      const items = e.clipboardData ? Array.from(e.clipboardData.items) : [];
      const item = items.find((it) => it.type.startsWith("image/"));
      if (!item) return;
      const file = item.getAsFile();
      if (file) {
        e.preventDefault();
        uploadFile(file);
      }
    }
    window.addEventListener("paste", onPaste);
    return () => window.removeEventListener("paste", onPaste);
  }, [spaceId]); // eslint-disable-line react-hooks/exhaustive-deps

  function onDrop(e: React.DragEvent<HTMLLabelElement>) {
    e.preventDefault();
    setDragOver(false);
    const file = e.dataTransfer.files?.[0];
    if (file) uploadFile(file);
  }

  return (
    <div className="space-y-2 rounded-md border border-ink-700/60 bg-ink-900/40 p-2.5">
      <div className="flex items-center justify-between">
        <div className="sub-title">Image context</div>
        {imageActive && (
          <span className="pill !text-[10px] border-accent-500/60 bg-accent-600/20 text-accent-200">● override active</span>
        )}
      </div>

      {!spaceId ? (
        <p className="text-[11px] text-ink-400">Build the space first.</p>
      ) : (
        <>
          {/* drop / browse zone */}
          <label
            htmlFor="image-context-file"
            onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
            onDragLeave={() => setDragOver(false)}
            onDrop={onDrop}
            className={`flex cursor-pointer flex-col items-center justify-center gap-1 rounded-md border-2 border-dashed
              px-3 py-5 text-center transition
              ${dragOver
                ? "border-accent-400 bg-accent-600/10 text-accent-200"
                : "border-ink-700 bg-ink-900/30 text-ink-300 hover:border-ink-600 hover:text-ink-100"}`}
          >
            <span className="text-base leading-none">🖼</span>
            <span className="text-[11px] leading-snug">
              <span className="font-medium">drop image</span>{" "}
              <span className="text-ink-400">— or click to browse, paste with</span>{" "}
              <kbd className="rounded border border-ink-700 bg-ink-800 px-1 text-[10px]">Ctrl+V</kbd>
            </span>
            <input
              id="image-context-file"
              type="file"
              accept="image/png,image/jpeg,image/webp,image/gif,image/*"
              hidden
              onChange={(e) => {
                const f = e.target.files?.[0];
                if (f) uploadFile(f);
                e.target.value = "";
              }}
            />
          </label>

          {/* status row */}
          {status === "encoding" && (
            <div className="flex items-center gap-2 text-[11px] text-ink-300">
              <span className="h-1.5 w-1.5 animate-pulse rounded-full bg-accent-400" />
              encoding · vision LLM reading…
            </div>
          )}
          {status === "error" && error && (
            <div className="text-[11px] text-danger">{error}</div>
          )}

          {/* thumbnail + description */}
          {(thumbnail || description) && (
            <div className="flex gap-2">
              {thumbnail && (
                /* eslint-disable-next-line @next/next/no-img-element */
                <img
                  src={thumbnail}
                  alt="image context preview"
                  className="h-20 w-20 shrink-0 rounded-md border border-ink-700 object-cover"
                />
              )}
              {description && (
                <div className="flex-1 rounded-md border border-ink-700/60 bg-ink-900/60 p-2 text-[11px] leading-snug text-ink-200">
                  <div className="mb-0.5 text-[9px] uppercase tracking-wider text-ink-400">
                    vision LLM reading
                  </div>
                  {description}
                </div>
              )}
            </div>
          )}

          {imageActive && (
            <button className="btn !text-xs" onClick={clearImage}>
              clear image context
            </button>
          )}

          <p className="text-[10px] leading-snug text-ink-400">
            The image is described by a Cloudflare vision LLM, then its description
            is embedded via the same model that built your symbol space — so the
            image context lands natively in the descriptors' embedding space, no
            backend rebuild required.
          </p>
        </>
      )}
    </div>
  );
}
