"""Perception Encoder Audio-Visual (facebook/pe-av-*) loader + thin embedding API.

PE-AV ships one checkpoint in which the text tower, the audio tower and the
video tower are stored once and *tied* into three places (text→audio-text /
video-text heads, audio+video encoders → the joint audio-visual encoder).
transformers 5.22 trips over that tying on load (a non-parameter buffer
`bands` in the tied regexes, and a timm classification head whose width is
only recorded as `label_names`), so we build the model from its config,
load the safetensors by hand and tie the parameters ourselves.

Embedding spaces (all L2-normalised here, all the same width):
  head "audio" : audio tower            ↔ text via get_text_audio_embeds
  head "video" : video tower            ↔ text via get_text_video_embeds
  head "av"    : joint audio-visual     ↔ text via get_text_audio_video_embeds
A text query in one head is NOT comparable with media in another head, which
is why the engine keeps a descriptor matrix per head.
"""
from __future__ import annotations

import glob
import os
import re
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch

PEAV_SR = 48_000
HEADS = ("av", "audio", "video")

_TIES = [
    # (target prefix regex, source prefix, buffers/params to leave alone)
    (r"^audio_model\.text_model\.", "text_model.", r"rotary_emb"),
    (r"^video_model\.text_model\.", "text_model.", r"rotary_emb"),
    (r"^audio_video_encoder\.embedder\.audio_encoder\.", "audio_model.audio_encoder.", r"rotary_emb"),
    (r"^audio_video_encoder\.embedder\.video_encoder\.", "video_model.video_encoder.", r"rotary_emb|\.rope\.pos_embed"),
]


def _snapshot_dir(model_name: str) -> str:
    from huggingface_hub import snapshot_download
    return snapshot_download(model_name)


def _set_module_tensor(model: torch.nn.Module, name: str, tensor: torch.Tensor, is_param: bool) -> None:
    mod_name, _, attr = name.rpartition(".")
    mod = model.get_submodule(mod_name) if mod_name else model
    if is_param:
        mod._parameters[attr] = tensor  # share the Parameter object (true tying)
    else:
        mod._buffers[attr] = tensor


def load_peav(model_name: str = "facebook/pe-av-small", device: str = "cpu"):
    """Return (model, processor) for a PE-AV checkpoint with shared towers tied."""
    from safetensors.torch import load_file
    from transformers import PeAudioVideoConfig, PeAudioVideoModel, PeAudioVideoProcessor

    snap = _snapshot_dir(model_name)
    cfg = PeAudioVideoConfig.from_pretrained(snap)
    # the timm vision head's width is only recorded as `label_names`
    for vcfg in (cfg.audio_video_config.video_config.vision_config,
                 cfg.video_config.video_config.vision_config):
        names = getattr(vcfg, "label_names", None)
        if names:
            vcfg.num_labels = len(names)

    # built on CPU (not meta) so init-computed buffers such as rotary tables exist
    model = PeAudioVideoModel(cfg)
    sd: Dict[str, torch.Tensor] = {}
    for f in sorted(glob.glob(os.path.join(snap, "*.safetensors"))):
        sd.update(load_file(f))
    missing, unexpected = model.load_state_dict(sd, strict=False, assign=True)

    # tie the shared towers parameter-by-parameter
    params = dict(model.named_parameters())
    buffers = dict(model.named_buffers())
    still_missing: List[str] = []
    for name in missing:
        src = None
        for pat, src_prefix, skip in _TIES:
            if re.match(pat, name) and not re.search(skip, name):
                src = re.sub(pat, src_prefix, name)
                break
        if src is not None and src in params:
            _set_module_tensor(model, name, params[src], True)
        elif src is not None and src in buffers:
            _set_module_tensor(model, name, buffers[src], False)
        else:
            still_missing.append(name)
    if still_missing:
        raise RuntimeError(f"PE-AV load: {len(still_missing)} tensors unresolved, e.g. {still_missing[:5]}")
    if unexpected:
        print(f"[peav] {len(unexpected)} unexpected checkpoint keys ignored, e.g. {list(unexpected)[:3]}")
    model = model.to(device).eval()
    proc = PeAudioVideoProcessor.from_pretrained(snap)
    return model, proc


class PeAV:
    """Small stateful wrapper: text (per head), audio, video and audio-video embeddings."""

    def __init__(self, model_name: str = "facebook/pe-av-small", device: str = "cpu"):
        self.model_name = model_name
        self.device = device
        self.model, self.proc = load_peav(model_name, device)
        with torch.inference_mode():
            z = self.embed_text(["a probe"], head="av")
        self.dim = int(z.shape[1])

    @staticmethod
    def _l2(z: torch.Tensor) -> np.ndarray:
        z = z / (z.norm(dim=-1, keepdim=True) + 1e-8)
        return z.detach().cpu().float().numpy().astype(np.float32)

    def _text_cls(self, ti) -> torch.Tensor:
        out = self.model.text_model(input_ids=ti["input_ids"], attention_mask=ti.get("attention_mask"),
                                    output_hidden_states=True, return_dict=True)
        return out.hidden_states[-1][:, 0]

    @torch.inference_mode()
    def embed_text(self, texts: Sequence[str], head: str = "av", batch_size: int = 32) -> np.ndarray:
        # transformers' own get_text_* helpers on the sub-models are unreliable
        # across 5.x, so go through the shared ModernBERT + the matching head.
        m = self.model
        proj = {"av": m.text_audio_video_head,
                "audio": m.audio_model.text_audio_head,
                "video": m.video_model.text_video_head}[head]
        out = []
        for i in range(0, len(texts), batch_size):
            ti = self.proc(text=list(texts[i:i + batch_size]), return_tensors="pt", padding=True, truncation=True)
            ti = {k: v.to(self.device) for k, v in ti.items()}
            out.append(self._l2(proj(self._text_cls(ti))))
        return np.concatenate(out, 0) if out else np.zeros((0, self.dim), np.float32)

    def _audio_inputs(self, wave: np.ndarray, sr: int):
        if sr != PEAV_SR and len(wave) > 1:
            n_new = int(round(len(wave) * PEAV_SR / sr))
            wave = np.interp(np.linspace(0, 1, n_new, endpoint=False),
                             np.linspace(0, 1, len(wave), endpoint=False), wave).astype(np.float32)
        ai = self.proc(audio=[np.asarray(wave, np.float32)], sampling_rate=PEAV_SR, return_tensors="pt", padding=True)
        return {k: v.to(self.device) for k, v in ai.items()}

    def _video_inputs(self, frames: Sequence[np.ndarray]):
        vi = self.proc(videos=[list(frames)], return_tensors="pt")
        return {k: v.to(self.device) for k, v in vi.items()}

    @torch.inference_mode()
    def embed_audio(self, wave: np.ndarray, sr: int) -> np.ndarray:
        ai = self._audio_inputs(wave, sr)
        am = self.model.audio_model
        ao = am.audio_encoder(input_values=ai["input_values"], padding_mask=ai.get("padding_mask"), return_dict=True)
        return self._l2(am.audio_head(ao.pooler_output))[0]

    @torch.inference_mode()
    def embed_video(self, frames: Sequence[np.ndarray]) -> np.ndarray:
        vi = self._video_inputs(frames)
        vm = self.model.video_model
        vo = vm.video_encoder(pixel_values_videos=vi["pixel_values_videos"],
                              padding_mask_videos=vi.get("padding_mask_videos"), return_dict=True)
        return self._l2(vm.video_head(vo.pooler_output))[0]

    @torch.inference_mode()
    def embed_audio_video(self, wave: np.ndarray, sr: int, frames: Sequence[np.ndarray]) -> np.ndarray:
        ai, vi = self._audio_inputs(wave, sr), self._video_inputs(frames)
        out = self.model.get_audio_video_embeds(ai["input_values"], vi["pixel_values_videos"],
                                                ai.get("padding_mask"), vi.get("padding_mask_videos"))
        return self._l2(out.audio_video_embeds)[0]
