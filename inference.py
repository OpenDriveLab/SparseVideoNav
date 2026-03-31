import math
from datetime import datetime
from pathlib import Path

import cv2
from einops import rearrange
import hydra
from moviepy.editor import ImageSequenceClip
import numpy as np
from omegaconf import DictConfig
import torch

from sparseVideoNav.vgm.modules.t5 import T5EncoderModel
from sparseVideoNav.vgm.modules.vae import WanVAE
from sparseVideoNav.vgm.modules.wan_flow_matching_unipc import FlowUniPCMultistepScheduler
from sparseVideoNav.svn_model import SVNModel


class SVNPipeline:
    """End-to-end inference pipeline for SparseVideoNav.

    Following diffusers conventions the pipeline holds the model, schedulers,
    and auxiliary components (VAE, text-encoder).  The full multi-step
    denoising loop lives in ``__call__``.
    """

    def __init__(
        self,
        model: SVNModel,
        t5: T5EncoderModel,
        wan_vae: WanVAE,
        video_scheduler: FlowUniPCMultistepScheduler,
        cfg: DictConfig,
    ):
        self.cfg = cfg
        self.device = torch.device(cfg.inference.device)
        self.model = model
        self.t5 = t5
        self.wan_vae = wan_vae
        self.video_scheduler = video_scheduler

    @classmethod
    def from_pretrained(cls, cfg: DictConfig) -> "SVNPipeline":
        device = torch.device(cfg.inference.device)
        backbone_cfg = cfg.model.backbone_config
        paths = _resolve_ckpt_paths(cfg)

        t5 = T5EncoderModel(
            text_len=backbone_cfg.t5.get("text_len", 512),
            dtype=torch.bfloat16,
            device=torch.device("cpu"),
            checkpoint_path=paths["t5_encoder"],
            tokenizer_path=paths["t5_tokenizer"],
        )

        wan_vae = WanVAE(
            z_dim=backbone_cfg.vae.get("z_dim", 16),
            vae_pth=paths["vae"],
            dtype=torch.float32,
            device=device,
        )

        svn_ckpt = Path(paths["svn_ckpt"])
        model = SVNModel.from_pretrained(str(svn_ckpt))
        model.to(device).eval()

        video_scheduler = FlowUniPCMultistepScheduler()

        return cls(
            model=model,
            t5=t5,
            wan_vae=wan_vae,
            video_scheduler=video_scheduler,
            cfg=cfg,
        )

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def __call__(
        self,
        video: str,
        text: str,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Run end-to-end video generation.

        Returns:
            clean_video:     np.ndarray (T, H, W, C) uint8
            video_with_traj: np.ndarray (T, H, W, C) uint8
        """
        inf_cfg = self.cfg.inference
        device = self.device

        # ---- 1. Preprocess the input video into reference / history latents ----
        latent_data = self._video_preprocess(video)
        if latent_data["history_latent"] is None:
            raise ValueError("Video too short: need history frames before reference frames")

        history_latent = latent_data["history_latent"]      # (C, T_hist, H, W)
        reference_latent = latent_data["reference_latent"]  # (C, T_ref,  H, W)

        C, T_hist, H, W = history_latent.shape
        T_ref = reference_latent.shape[1]

        history_latent = history_latent.unsqueeze(0).to(device)      # (1, C, T_hist, H, W)
        reference_latent = reference_latent.unsqueeze(0).to(device)  # (1, C, T_ref,  H, W)

        pos = 1 + T_hist
        ref_positions    = list(range(pos, pos + T_ref));  pos += T_ref
        future_positions = [pos + offset for offset in list(inf_cfg.future_chunk_offsets)]

        history_mask = torch.ones(1, T_hist, dtype=torch.bool, device=device)

        num_future   = len(future_positions)
        future_noise = torch.randn(1, C, num_future, H, W, device=device, dtype=reference_latent.dtype)
        video_latent = torch.cat([reference_latent, future_noise], dim=2)  # (1, C, T_ref+T_fut, H, W)

        future_gt_idx     = torch.tensor([ref_positions + future_positions], device=device, dtype=torch.long)
        ref_inject_frames = T_ref

        # ---- 2. Encode the text prompt ----
        self.t5.model.to(device)
        with torch.no_grad():
            prompt_embedding = self.t5([text], device)
        prompt_embedding = [emb.float().to(device) for emb in prompt_embedding]
        self.t5.model.cpu()
        torch.cuda.empty_cache()

        # ---- 3. Compress history latents ----
        with torch.no_grad():
            history_features = self.model.vgm.process_history_features(
                history_latent, history_mask, prompt_embedding
            )

        # ---- 4. Video denoising loop ----
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            denoised_latent = self._video_denoise(
                video_latent=video_latent,
                prompt_embedding=prompt_embedding,
                history_features=history_features,
                ref_inject_frames=ref_inject_frames,
                denoise_steps=inf_cfg.denoise_steps,
            )

        # ---- 5. Decode the latent video ----
        clean_video = self._decode_video(denoised_latent)

        return clean_video

    # ------------------------------------------------------------------
    # Denoising loop
    # ------------------------------------------------------------------

    def _video_denoise(
        self,
        video_latent: torch.Tensor,
        prompt_embedding: list,
        history_features: torch.Tensor | None,
        ref_inject_frames: int,
        denoise_steps: int,
    ) -> torch.Tensor:
        """Run the video flow-matching denoising loop.

        Returns:
            denoised_latent: (B, C, T, H, W).
        """
        B      = video_latent.shape[0]
        device = video_latent.device

        reference_latent = video_latent[:, :, :ref_inject_frames].clone()
        noisy_latent     = video_latent.clone()

        self.video_scheduler.set_timesteps(denoise_steps, device=device)

        patch_size  = getattr(self.model.vgm, "patch_size", [1, 2, 2])
        f, h, w     = video_latent.shape[2:]
        seq_len     = int(np.ceil((h * w) / (patch_size[1] * patch_size[2]) * f))
        model_dtype = next(self.model.vgm.parameters()).dtype

        for t in self.video_scheduler.timesteps:
            timestep_batch = torch.full((B,), t, device=device, dtype=torch.long)

            with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                noisy_list    = [lat.to(dtype=model_dtype) for lat in noisy_latent.unbind(0)]
                context_typed = [emb.to(dtype=model_dtype) for emb in prompt_embedding]

                model_pred = self.model(
                    hidden_states=noisy_list,
                    timestep=timestep_batch,
                    encoder_hidden_states=context_typed,
                    seq_len=seq_len,
                    history_features=history_features,
                )

            model_pred   = model_pred.to(dtype=model_dtype)
            noisy_latent = noisy_latent.to(dtype=model_dtype)
            noisy_latent = self.video_scheduler.step(
                model_output=model_pred,
                timestep=t,
                sample=noisy_latent,
            ).prev_sample

            noisy_latent[:, :, :ref_inject_frames] = reference_latent.to(dtype=model_dtype)

        return noisy_latent

    # ------------------------------------------------------------------
    # Post-processing
    # ------------------------------------------------------------------

    def _decode_video(self, generated_video_latent: torch.Tensor) -> np.ndarray:
        """Decode latents to RGB frames.

        Returns:
            clean_video: np.ndarray (T, H, W, C) uint8
        """
        latent = generated_video_latent
        if latent.ndim == 5 and latent.shape[0] == 1:
            latent = latent.squeeze(0)

        with torch.no_grad():
            decoded = self.wan_vae.decode([latent])[0]  # (C, T, H, W)

        decoded = rearrange(decoded, "c t h w -> t h w c")
        decoded = ((decoded + 1.0) * 127.5).clamp(0, 255)
        return decoded.cpu().numpy().astype(np.uint8)

    def _video_preprocess(self, video: str) -> dict:
        """Load a video, sample frames, and encode them with the VAE."""
        inf_cfg = self.cfg.inference

        cap = cv2.VideoCapture(video)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video}")

        original_fps    = cap.get(cv2.CAP_PROP_FPS)
        sample_interval = max(1, int(original_fps / inf_cfg.target_fps))

        all_frames, frame_idx = [], 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx % sample_interval == 0:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                h, w = frame_rgb.shape[:2]
                min_dim = min(h, w)
                top, left = (h - min_dim) // 2, (w - min_dim) // 2
                frame_cropped = frame_rgb[top:top + min_dim, left:left + min_dim]
                all_frames.append(cv2.resize(frame_cropped, (inf_cfg.target_size, inf_cfg.target_size)))
            frame_idx += 1
        cap.release()

        if not all_frames:
            raise ValueError("No frames extracted from video")

        dropped = len(all_frames) % 4
        if dropped:
            all_frames = all_frames[dropped:]

        num_chunks = len(all_frames) // 4
        if num_chunks < 2:
            raise ValueError("Video too short: need at least 2 chunks (8 frames) after alignment")
        history_frames   = all_frames[:-8] if len(all_frames) > 8 else []
        reference_frames = all_frames[-8:]

        def frames_to_latent(frames_list: list) -> torch.Tensor:
            frames_with_ctx = [frames_list[0]] + frames_list
            frames_np = np.stack(frames_with_ctx, axis=0).transpose(0, 3, 1, 2)
            frames_t  = torch.from_numpy(frames_np).float() / 127.5 - 1.0
            frames_t  = rearrange(frames_t, "t c h w -> c t h w").to(self.device)
            with torch.no_grad():
                latent = self.wan_vae.encode([frames_t])[0]
            return latent[:, 1:]

        ref_latent  = frames_to_latent(reference_frames)
        hist_latent = frames_to_latent(history_frames) if history_frames else None

        return {
            "history_latent":   hist_latent,
            "reference_latent": ref_latent,
            "history_chunks":   hist_latent.shape[1] if hist_latent is not None else 0,
            "ref_chunks":       ref_latent.shape[1],
        }


def _resolve_ckpt_paths(cfg: DictConfig) -> dict:
    """Resolve checkpoint paths from `ckpt_path` and optional overrides."""
    base = Path(cfg.ckpt_path)
    paths = dict(
        t5_tokenizer=str(base / "google/umt5-xxl"),
        t5_encoder=str(base / "models_t5_umt5-xxl-enc-bf16.pth"),
        vae=str(base / "Wan2.1_VAE.pth"),
        svn_ckpt=str(base / "svn_ckpt"),
    )
    if cfg.get("t5_ckpt_path"):
        t5_base = Path(cfg.t5_ckpt_path)
        paths["t5_tokenizer"] = str(t5_base / "google/umt5-xxl")
        paths["t5_encoder"] = str(t5_base / "models_t5_umt5-xxl-enc-bf16.pth")
    if cfg.get("vae_ckpt_path"):
        paths["vae"] = cfg.vae_ckpt_path
    if cfg.get("svn_ckpt_path"):
        paths["svn_ckpt"] = cfg.svn_ckpt_path
    return paths


@hydra.main(version_base="1.3", config_path="config", config_name="inference")
def main(cfg: DictConfig):
    if not cfg.get("video_path"):
        raise ValueError("video_path is required")
    if not cfg.get("prompt"):
        raise ValueError("prompt is required")

    pipeline = SVNPipeline.from_pretrained(cfg)

    clean_video = pipeline(
        video=cfg.video_path,
        text=cfg.prompt,
    )

    timestamp  = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(cfg.get("output_path") or "outputs") / f"{timestamp}_{Path(cfg.video_path).stem}"
    output_dir.mkdir(parents=True, exist_ok=True)

    target_fps = cfg.inference.target_fps
    clip = ImageSequenceClip(list(clean_video), fps=target_fps)
    clip.write_videofile(str(output_dir / "predicted_video.mp4"), codec="libx264", logger=None, fps=target_fps)

    print(f"Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
