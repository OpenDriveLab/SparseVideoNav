"""SparseVideoNav latent-space model with Diffusers-style checkpoint support.

The model wraps the WAN video backbone (VGM).

Following diffusers conventions, the ``forward`` method performs a **single
denoising step** for the video backbone.  The full multi-step denoising loop
is orchestrated by the companion ``SVNPipeline``.
"""

from typing import Optional

import torch
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.modeling_utils import ModelMixin

from sparseVideoNav.vgm.wan_model import WanModel


class SVNModel(ModelMixin, ConfigMixin):
    """SparseVideoNav latent-space model with Diffusers-style loading.

    Exposes a single-step prediction interface:

    * ``forward`` – one VGM backbone step (video flow-matching).
    """

    @register_to_config
    def __init__(
        self,
        # WAN backbone configuration.
        vgm_dim: int = 1536,
        vgm_ffn_dim: int = 8960,
        vgm_num_heads: int = 12,
        vgm_num_layers: int = 30,
        vgm_qformer_config: Optional[dict] = None,
        vgm_video_former_config: Optional[dict] = None,
    ):
        super().__init__()

        self.vgm = WanModel(
            dim=vgm_dim,
            ffn_dim=vgm_ffn_dim,
            num_heads=vgm_num_heads,
            num_layers=vgm_num_layers,
            qformer_config=vgm_qformer_config,
            video_former_config=vgm_video_former_config,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        encoder_hidden_states: list,
        seq_len: int,
        history_features: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Predict the velocity field for a single video denoising step.

        Args:
            hidden_states:         (B, C, T, H, W) noisy video latents, **or**
                                   ``List[Tensor(C, T, H, W)]`` (one per batch item).
            timestep:              (B,) current diffusion timestep.
            encoder_hidden_states: ``List[Tensor(L, D)]`` text embeddings per item.
            seq_len:               Maximum token-sequence length for the backbone.
            history_features:      Optional pre-computed history features.

        Returns:
            model_output: (B, C, T, H, W) predicted velocity / noise.
        """
        if isinstance(hidden_states, torch.Tensor) and hidden_states.ndim == 5:
            hidden_states_list = list(hidden_states.unbind(0))
        else:
            hidden_states_list = hidden_states

        output = self.vgm(
            hidden_states_list,
            t=timestep,
            context=encoder_hidden_states,
            seq_len=seq_len,
            history_features=history_features,
        )
        return torch.stack(output, dim=0)
