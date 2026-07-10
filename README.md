<p align="center">
  <img src="asset/caption.png" alt="SparseVideoNav Logo">
</p>

<p align="center">
<h1 align="center"><strong>SparseVideoNav: Sparse Video Generation Propels Real-World Beyond-the-View Vision-Language Navigation</strong></h1>
<p align="center">
    <a href='https://betray12138.github.io/resume/' target='_blank'>Hai Zhang</a><sup>*</sup>&emsp;
    <a href='https://github.com/stdcat' target='_blank'>Siqi Liang</a><sup>*</sup>&emsp;
    <a href='https://scholar.google.com/citations?user=ulZxvY0AAAAJ&hl=en' target='_blank'>Li Chen</a>&emsp;
    <a href='https://github.com/LiYuxian0306' target='_blank'>Yuxian Li</a>&emsp;
    <a href='https://github.com/PUFFERFISHTNT' target='_blank'>Yukuan Xu</a>&emsp;
    <a href='https://z-taylcr7.github.io/' target='_blank'>Yichao Zhong</a>&emsp;
    <a href='https://mars.hku.hk/people.html' target='_blank'>Fu Zhang</a><sup>†</sup>&emsp;
    <a href='https://lihongyang.info/' target='_blank'>Hongyang Li</a><sup>†</sup>&emsp;
    <br><br>
    <a href='https://hku.hk/' target='_blank'><img src="asset/HKU_logo.png" alt="Project Page" height="40"></a>
    <br>
    The University of Hong Kong&emsp;
    <br>
  </p>
</p>

<div id="top" align="center">

<div>
<a href="https://opendrivelab.com/SparseVideoNav" target="_blank"><img src="https://img.shields.io/badge/Project_Page-green" alt="Project Page"></a>
<a href="https://github.com/OpenDriveLab/SparseVideoNav"><img alt="Repo" src="https://img.shields.io/badge/github-repo-blue?logo=github"/></a>
<a href="https://arxiv.org/abs/2602.05827" target="_blank"><img src="https://img.shields.io/badge/arXiv-2602.05827-b31b1b" alt="arXiv"></a>
<a href="https://huggingface.co/datasets/OpenDriveLab/SparseVideoNav" target="_blank"><img src="https://img.shields.io/badge/Hugging_Face-FF9D00?logo=huggingface&logoColor=white" alt="arXiv"></a>
<a href="https://creativecommons.org/licenses/by-nc-sa/4.0" target="_blank"><img src="https://img.shields.io/badge/License-CC_%20_BY--NC--SA_4.0-blue.svg" alt="License"/></a>
</div>


</div>

## 📖 Introduction

SparseVideoNav introduces video generation models to real-world beyond-the-view vision-language navigation for the first time. It achieves sub-second trajectory inference with a sparse future spanning a 20-second horizon, yielding a remarkable 27× speed-up. Real-world zero-shot experiments show 2.5× higher success rate than state-of-the-art LLM baselines and mark the first realization in challenging night scenes.

**Developers**: [Hai Zhang](https://zhanghenryhai12138.github.io/) and [Siqi Liang](https://github.com/stdcat)

## 📢 News

> [!IMPORTANT]
> 🌟 Stay up to date at [opendrivelab.com](https://opendrivelab.com/#news)!

- 🎉 **2026-02-05**: [Project Page](https://opendrivelab.com/SparseVideoNav) is now available!
- 🎉 **2026-02-06**: [arXiv preprint](https://arxiv.org/abs/2602.05827) is now available!
- 🎉 **2026-03-31**: Inference code and model checkpoint released!
- 🎉 **2026-07-10**: [SparseVideoNav Dataset](https://huggingface.co/datasets/OpenDriveLab/SparseVideoNav) is now available!

## 📌 Table of Contents
- 📖 [Introduction](#-introduction)
- 📢 [News](#-news)
- 🔥 [Highlights](#highlights)
- 🔧 [Installation](#-installation)
- 📥 [Checkpoint](#-checkpoint)
- 🚀 [Usage](#-usage)
- 📝 [TODO List](#-todo-list)
- 📬 [Contact](#-contact)
- 📄 [License and Citation](#-license-and-citation)


## 🔥 Highlights

- We investigate beyond-the-view navigation tasks in the real world  by introducing video generation model to this field for the first time.
- We pioneer a paradigm shift from continuous to sparse video generation for longer prediction horizon.
- We achieve sub-second trajectory inference guided by a generated sparse future spanning a 20-second horizon. This yields a remarkable 27x speed-up compared to the unoptimized counterpart.
- We achieve the first realization of beyond-the-view navigation in challenging night scenes with a 17.5% success rate.

## 🔧 Installation

**Requirements**
- Linux (tested on Ubuntu)
- Python 3.10
- NVIDIA GPU with ≥ 16 GB VRAM
- [uv](https://docs.astral.sh/uv/) ≥ 0.7

```bash
# 1. Clone the repository
git clone https://github.com/OpenDriveLab/SparseVideoNav.git
cd SparseVideoNav

# 2. Create virtual environment and install dependencies
uv sync --all-groups

# 3. Activate the environment
source .venv/bin/activate
```

> `--all-groups` also installs `flash-attn`. Building it from source takes a few minutes on first install.

## 📥 Checkpoint

Download the SparseVideoNav pipeline checkpoint and place it under `models/SparseVideoNav-Models/`:

| Component | Download |
|-----------|----------|
| **SparseVideoNav pipeline checkpoint** | 🤗 [HuggingFace](https://huggingface.co/OpenDriveLab/SparseVideoNav_VGM) |

Expected directory layout after download:

```
models/SparseVideoNav-Models/
├── google/
│   └── umt5-xxl/
│       ├── special_tokens_map.json
│       ├── spiece.model
│       ├── tokenizer.json
│       └── tokenizer_config.json
├── models_t5_umt5-xxl-enc-bf16.pth
├── Wan2.1_VAE.pth
└── svn_ckpt/
    ├── config.json
    └── diffusion_pytorch_model.safetensors
```

If you place the checkpoint elsewhere, update `ckpt_path` in `config/inference.yaml` or override it on the command line.

## 🚀 Usage

### 1. Command-line inference

```bash
python inference.py video_path=/path/to/input.mp4 'prompt=turn right'
```

Results are written to `outputs/<timestamp>_<video_name>/`:
- `predicted_video.mp4` — generated future video

Key overrides:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `video_path` | — | Input video path (required) |
| `prompt` | — | Language instruction (required) |
| `output_path` | `outputs` | Root output directory |
| `ckpt_path` | `models/SparseVideoNav-Models` | Pipeline checkpoint directory |
| `inference.device` | `cuda:0` | Target device |
| `inference.denoise_steps` | `4` | Denoising steps (higher → better quality) |

Example with overrides:

```bash
python inference.py \
    video_path=/path/to/input.mp4 \
    'prompt=walk forward and turn left' \
    ckpt_path=/path/to/checkpoint \
    inference.device=cuda:0 \
    inference.denoise_steps=8
```

### 2. Gradio web demo

```bash
python gradio_interface.py
```

Opens a local demo at `http://0.0.0.0:7860`. Upload a video, enter a navigation instruction, and click **Run Prediction**.

Common options:

| Flag | Default | Description |
|------|---------|-------------|
| `--ckpt_path` | from config | Override checkpoint directory |
| `--device` | `cuda:0` | Target device |
| `--port` | `7860` | Server port |
| `--share` | `False` | Create a public Gradio share link |

### 3. Python API

```python
from omegaconf import OmegaConf
from inference import SVNPipeline

cfg = OmegaConf.load("config/inference.yaml")
cfg.ckpt_path = "/path/to/checkpoint"
cfg.inference.device = "cuda:0"

pipeline = SVNPipeline.from_pretrained(cfg)

# Returns np.ndarray (T, H, W, C) uint8
video = pipeline(video="/path/to/input.mp4", text="turn right")
```

For direct access to the latent-space model:

```python
from sparseVideoNav.svn_model import SVNModel

model = SVNModel.from_pretrained("/path/to/checkpoint/svn_ckpt")
```

## 📝 TODO List
- [x] SparseVideoNav Paper Release.
  - [x] arXiv preprint is now available!
- [x] SparseVideoNav Code Release.
  - [x] Inference code of distilled video generation model and model checkpoint.
  - [ ] Inference code of continuous action head and model checkpoint (Estimate 2026 Q3).
- [x] SparseVideoNav Dataset Release
  - [x] [~140h real-world VLN data](https://huggingface.co/datasets/OpenDriveLab/SparseVideoNav).

## 📬 Contact

For further inquiries or assistance, please contact [zhanghenryhai12138@gmail.com](mailto:zhanghenryhai12138@gmail.com) or [liangsiqi@connect.hku.hk](mailto:liangsiqi@connect.hku.hk)

## 📄 License and Citation

All the data and code within this repo are under [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/).

- Please consider citing our work if it helps your research.
```BibTeX
@article{zhang2026sparse,
  title={Sparse Video Generation Propels Real-World Beyond-the-View Vision-Language Navigation},
  author={Zhang, Hai and Liang, Siqi and Chen, Li and Li, Yuxian and Xu, Yukuan and Zhong, Yichao and Zhang, Fu and Li, Hongyang},
  journal={arXiv preprint arXiv:2602.05827},
  year={2026}
}
```
