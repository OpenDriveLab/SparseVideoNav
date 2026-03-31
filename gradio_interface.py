"""Gradio interface for end-to-end SparseVideoNav inference."""

import argparse
import tempfile
import traceback
from pathlib import Path

import gradio as gr
import numpy as np
from moviepy.editor import ImageSequenceClip
from omegaconf import OmegaConf

from inference import SVNPipeline

# Global state shared by the UI callbacks.
_pipeline: SVNPipeline | None = None
_cfg: OmegaConf | None = None


# Model loading
def load_pipeline(
    config_path: str | Path,
    device: str = "cuda:0",
    ckpt_override: str | None = None,
    vae_override: str | None = None,
    t5_ckpt_override: str | None = None,
    t5_tok_override: str | None = None,
) -> None:
    """Load `SVNPipeline` from config, optionally overriding checkpoint paths."""
    global _pipeline, _cfg

    print("[SparseVideoNav] Loading config …", flush=True)
    cfg = OmegaConf.load(str(config_path))

    if ckpt_override:
        cfg.ckpt_path = ckpt_override
    if vae_override:
        cfg.model.backbone_config.vae.model_path = vae_override
    if t5_ckpt_override:
        cfg.model.backbone_config.t5.checkpoint_path = t5_ckpt_override
    if t5_tok_override:
        cfg.model.backbone_config.t5.tokenizer_path = t5_tok_override
    cfg.inference.device = device

    print("[SparseVideoNav] Building SVNPipeline …", flush=True)
    _pipeline = SVNPipeline.from_pretrained(cfg)
    _cfg = cfg
    print("[SparseVideoNav] Model ready.", flush=True)


# Core inference helpers
def _numpy_to_video(frames: np.ndarray, fps: int) -> str:
    """Write a `(T, H, W, C)` uint8 array to a temporary MP4 file."""
    tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    clip = ImageSequenceClip(list(frames), fps=fps)
    clip.write_videofile(tmp.name, codec="libx264", logger=None, fps=fps)
    return tmp.name


def run_prediction(
    video_path: str | None,
    prompt: str,
) -> str | None:
    """Run inference for the Gradio UI and return the output video path."""
    if _pipeline is None or _cfg is None:
        return None
    if video_path is None:
        return None
    if not prompt or not prompt.strip():
        return None

    _cfg.inference.denoise_steps = 4

    print(
        f"[SparseVideoNav] Inference  video={Path(video_path).name}  "
        f"prompt='{prompt.strip()}'",
        flush=True,
    )

    try:
        clean_video = _pipeline(
            video=video_path,
            text=prompt.strip(),
        )
    except Exception:
        traceback.print_exc()
        return None

    fps = int(_cfg.inference.target_fps)
    return _numpy_to_video(clean_video, fps)


# Gradio layout
def build_interface() -> gr.Blocks:
    """Build the Gradio interface."""

    with gr.Blocks(
        title="SparseVideoNav",
        theme=gr.themes.Soft(),
    ) as demo:
        gr.Markdown(
            """
            # SparseVideoNav — Navigation Prediction
            Upload a video and enter a navigation instruction to predict future frames.
            """
        )

        with gr.Row(equal_height=False):
            with gr.Column(scale=1):
                video_input = gr.Video(
                    label="Input Video",
                    sources=["upload"],
                    value="asset/demo.mp4",
                )
                prompt_input = gr.Textbox(
                    label="Navigation Prompt",
                    placeholder="e.g., 'walk forward', 'turn right', 'go to the kitchen'",
                    lines=2,
                    value="Walk to the brown sign ahead and stop."
                )

                run_btn = gr.Button("Run Prediction", variant="primary", size="lg")
                clear_btn = gr.ClearButton(
                    components=[video_input, prompt_input],
                    value="Clear",
                )

            with gr.Column(scale=1):
                video_output = gr.Video(label="Predicted Video", interactive=False)

        run_btn.click(
            fn=run_prediction,
            inputs=[video_input, prompt_input],
            outputs=[video_output],
        )
        clear_btn.add(components=[video_output])

    return demo


# CLI entry point
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SparseVideoNav Gradio Interface")
    parser.add_argument("--config", type=str, default=None,
                        help="Path to inference YAML config (default: config/inference.yaml)")
    parser.add_argument("--ckpt_path", type=str, default=None,
                        help="Override checkpoint path")
    parser.add_argument("--vae_model_path", type=str, default=None,
                        help="Override VAE model path")
    parser.add_argument("--t5_checkpoint_path", type=str, default=None,
                        help="Override T5 checkpoint path")
    parser.add_argument("--t5_tokenizer_path", type=str, default=None,
                        help="Override T5 tokenizer path")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--server_name", type=str, default="0.0.0.0")
    parser.add_argument("--share", action="store_true",
                        help="Generate a public Gradio share link")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    config_path = Path(args.config) if args.config else Path(__file__).parent / "config" / "inference.yaml"
    assert config_path.exists(), f"Config not found: {config_path}"

    load_pipeline(
        config_path=config_path,
        device=args.device,
        ckpt_override=args.ckpt_path,
        vae_override=args.vae_model_path,
        t5_ckpt_override=args.t5_checkpoint_path,
        t5_tok_override=args.t5_tokenizer_path,
    )

    demo = build_interface()
    demo.launch(
        server_name=args.server_name,
        server_port=args.port,
        share=args.share,
    )
