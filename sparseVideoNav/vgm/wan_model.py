# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
# Modified
import math

from diffusers.configuration_utils import ConfigMixin
from diffusers.configuration_utils import register_to_config
from diffusers.models.modeling_utils import ModelMixin
from einops import rearrange
import torch
import torch.cuda.amp as amp
import torch.nn as nn

from .modules.attention import flash_attention

__all__ = ["WanModel"]

T5_CONTEXT_TOKEN_NUMBER = 512
FIRST_LAST_FRAME_CONTEXT_TOKEN_NUMBER = 257 * 2


def sinusoidal_embedding_1d(dim, position):
    # preprocess
    assert dim % 2 == 0
    half = dim // 2
    position = position.type(torch.float64)

    # calculation
    sinusoid = torch.outer(
        position, torch.pow(10000, -torch.arange(half).to(position).div(half)))
    x = torch.cat([torch.cos(sinusoid), torch.sin(sinusoid)], dim=1)
    return x


@amp.autocast(enabled=False)
def rope_params(max_seq_len, dim, theta=10000):
    assert dim % 2 == 0
    freqs = torch.outer(
        torch.arange(max_seq_len),
        1.0 / torch.pow(theta,
                        torch.arange(0, dim, 2).to(torch.float64).div(dim)))
    freqs = torch.polar(torch.ones_like(freqs), freqs)
    return freqs


@amp.autocast(enabled=False)
def rope_apply(x, grid_sizes, freqs):
    n, c = x.size(2), x.size(3) // 2

    # split freqs
    freqs = freqs.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)

    # loop over samples
    output = []
    for i, (f, h, w) in enumerate(grid_sizes.tolist()):
        seq_len = f * h * w

        # precompute multipliers
        x_i = torch.view_as_complex(x[i, :seq_len].to(torch.float64).reshape(
            seq_len, n, -1, 2))
        freqs_i = torch.cat([
            freqs[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
            freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
            freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1)
        ],
                            dim=-1).reshape(seq_len, 1, -1)

        # apply rotary embedding
        x_i = torch.view_as_real(x_i * freqs_i).flatten(2)
        x_i = torch.cat([x_i, x[i, seq_len:]])

        # append to collection
        output.append(x_i)
    return torch.stack(output).float()


class WanRMSNorm(nn.Module):

    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        r"""
        Args:
            x(Tensor): Shape [B, L, C]
        """
        return self._norm(x.float()).type_as(x) * self.weight

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)


class WanLayerNorm(nn.LayerNorm):

    def __init__(self, dim, eps=1e-6, elementwise_affine=False):
        super().__init__(dim, elementwise_affine=elementwise_affine, eps=eps)

    def forward(self, x):
        r"""
        Args:
            x(Tensor): Shape [B, L, C]
        """
        return super().forward(x.float()).type_as(x)


class WanSelfAttention(nn.Module):

    def __init__(self,
                 dim,
                 num_heads,
                 window_size=(-1, -1),
                 qk_norm=True,
                 eps=1e-6):
        assert dim % num_heads == 0
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.eps = eps

        # Self-attention projections.
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()
        self.norm_k = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()

    def forward(self, x, seq_lens, grid_sizes, freqs):
        r"""
        Args:
            x(Tensor): Shape [B, L, num_heads, C / num_heads]
            seq_lens(Tensor): Shape [B]
            grid_sizes(Tensor): Shape [B, 3], the second dimension contains (F, H, W)
            freqs(Tensor): Rope freqs, shape [1024, C / num_heads / 2]
        """
        b, s, n, d = *x.shape[:2], self.num_heads, self.head_dim

        # Build query, key, and value tensors.
        def qkv_fn(x):
            q = self.norm_q(self.q(x)).view(b, s, n, d)
            k = self.norm_k(self.k(x)).view(b, s, n, d)
            v = self.v(x).view(b, s, n, d)
            return q, k, v

        q, k, v = qkv_fn(x)

        x = flash_attention(
            q=rope_apply(q, grid_sizes, freqs),
            k=rope_apply(k, grid_sizes, freqs),
            v=v,
            k_lens=seq_lens,
            window_size=self.window_size)

        # output
        x = x.flatten(2)
        x = self.o(x)
        return x


class WanT2VCrossAttention(WanSelfAttention):

    def forward(self, x, context, context_lens):
        r"""
        Args:
            x(Tensor): Shape [B, L1, C]
            context(Tensor): Shape [B, L2, C]
            context_lens(Tensor): Shape [B]
        """
        b, n, d = x.size(0), self.num_heads, self.head_dim

        # compute query, key, value
        q = self.norm_q(self.q(x)).view(b, -1, n, d)
        k = self.norm_k(self.k(context)).view(b, -1, n, d)
        v = self.v(context).view(b, -1, n, d)

        # compute attention
        x = flash_attention(q, k, v, k_lens=context_lens)

        # output
        x = x.flatten(2)
        x = self.o(x)
        return x


class WanI2VCrossAttention(WanSelfAttention):

    def __init__(self,
                 dim,
                 num_heads,
                 window_size=(-1, -1),
                 qk_norm=True,
                 eps=1e-6):
        super().__init__(dim, num_heads, window_size, qk_norm, eps)

        self.k_img = nn.Linear(dim, dim)
        self.v_img = nn.Linear(dim, dim)
        # self.alpha = nn.Parameter(torch.zeros((1, )))
        self.norm_k_img = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()

    def forward(self, x, context, context_lens):
        r"""
        Args:
            x(Tensor): Shape [B, L1, C]
            context(Tensor): Shape [B, L2, C]
            context_lens(Tensor): Shape [B]
        """
        image_context_length = context.shape[1] - T5_CONTEXT_TOKEN_NUMBER
        context_img = context[:, :image_context_length]
        context = context[:, image_context_length:]
        b, n, d = x.size(0), self.num_heads, self.head_dim

        # compute query, key, value
        q = self.norm_q(self.q(x)).view(b, -1, n, d)
        k = self.norm_k(self.k(context)).view(b, -1, n, d)
        v = self.v(context).view(b, -1, n, d)
        k_img = self.norm_k_img(self.k_img(context_img)).view(b, -1, n, d)
        v_img = self.v_img(context_img).view(b, -1, n, d)
        img_x = flash_attention(q, k_img, v_img, k_lens=None)
        # compute attention
        x = flash_attention(q, k, v, k_lens=context_lens)

        # output
        x = x.flatten(2)
        img_x = img_x.flatten(2)
        x = x + img_x
        x = self.o(x)
        return x


WAN_CROSSATTENTION_CLASSES = {
    "t2v_cross_attn": WanT2VCrossAttention,
    "i2v_cross_attn": WanI2VCrossAttention,
}


class WanAttentionBlock(nn.Module):

    def __init__(self,
                 cross_attn_type,
                 dim,
                 ffn_dim,
                 num_heads,
                 window_size=(-1, -1),
                 qk_norm=True,
                 cross_attn_norm=False,
                 eps=1e-6,
                 **kwargs):
        super().__init__()
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps

        # Core transformer layers.
        self.norm1 = WanLayerNorm(dim, eps)
        self.self_attn = WanSelfAttention(dim, num_heads, window_size, qk_norm,
                                          eps)
        self.norm3 = WanLayerNorm(
            dim, eps,
            elementwise_affine=True) if cross_attn_norm else nn.Identity()
        self.cross_attn = WAN_CROSSATTENTION_CLASSES[cross_attn_type](dim,
                                                                      num_heads,
                                                                      (-1, -1),
                                                                      qk_norm,
                                                                      eps)
        self.norm2 = WanLayerNorm(dim, eps)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim), nn.GELU(approximate="tanh"),
            nn.Linear(ffn_dim, dim))

        # Learnable modulation parameters.
        self.modulation = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)

    def forward(
        self,
        x,
        e,
        seq_lens,
        grid_sizes,
        freqs,
        context,
        context_lens,
        **kwargs,
    ):
        r"""
        Args:
            x(Tensor): Shape [B, L, C]
            e(Tensor): Shape [B, 6, C]
            seq_lens(Tensor): Shape [B], length of each sequence in batch
            grid_sizes(Tensor): Shape [B, 3], the second dimension contains (F, H, W)
            freqs(Tensor): Rope freqs, shape [1024, C / num_heads / 2]
        """
        assert e.dtype == torch.float32
        with amp.autocast(dtype=torch.float32):
            e = (self.modulation + e).chunk(6, dim=1)
        assert e[0].dtype == torch.float32

        # self-attention
        y = self.self_attn(
            self.norm1(x).float() * (1 + e[1]) + e[0], seq_lens, grid_sizes,
            freqs)
        with amp.autocast(dtype=torch.float32):
            x = x + y * e[2]

        # cross-attention & ffn function
        def cross_attn_ffn(x, context, context_lens, e):
            x = x + self.cross_attn(self.norm3(x), context, context_lens)
            y = self.ffn(self.norm2(x).float() * (1 + e[4]) + e[3])
            with amp.autocast(dtype=torch.float32):
                x = x + y * e[5]
            return x

        x = cross_attn_ffn(x, context, context_lens, e)
        return x


class WanHistoryAttentionBlock(WanAttentionBlock):

    def __init__(self,
                 cross_attn_type,
                 dim,
                 ffn_dim,
                 num_heads,
                 window_size=(-1, -1),
                 qk_norm=True,
                 cross_attn_norm=False,
                 eps=1e-6,
                 history_dim=None,
                 **kwargs):
        super().__init__(cross_attn_type, dim, ffn_dim, num_heads, window_size, qk_norm, cross_attn_norm, eps)

        if history_dim is None:
            history_dim = dim

        # History-conditioning layers.
        self.history_proj = nn.Sequential(
            nn.Linear(history_dim, history_dim),
            nn.GELU(),
            nn.Linear(history_dim, dim),
        )
        self.norm_history = WanLayerNorm(dim, eps, elementwise_affine=True)
        self.history_cross_attn = WAN_CROSSATTENTION_CLASSES[cross_attn_type](
            dim, num_heads, window_size, qk_norm, eps
        )

        nn.init.zeros_(self.history_cross_attn.o.weight)
        if self.history_cross_attn.o.bias is not None:
            nn.init.zeros_(self.history_cross_attn.o.bias)

    def forward(
        self,
        x,
        e,
        seq_lens,
        grid_sizes,
        freqs,
        context,
        context_lens,
        history_features=None,
        **kwargs,
    ):
        r"""
        Args:
            x(Tensor): Shape [B, L, C]
            e(Tensor): Shape [B, 6, C]
            seq_lens(Tensor): Shape [B]
            grid_sizes(Tensor): Shape [B, 3]
            freqs(Tensor): Rope freqs, shape [1024, C / num_heads / 2]
            history_features(Tensor): History features, shape [B, N, history_dim]
        """
        assert e.dtype == torch.float32
        with amp.autocast(dtype=torch.float32):
            e = (self.modulation + e).chunk(6, dim=1)
        assert e[0].dtype == torch.float32

        # self-attention
        y = self.self_attn(
            self.norm1(x).float() * (1 + e[1]) + e[0], seq_lens, grid_sizes, freqs
        )
        with amp.autocast(dtype=torch.float32):
            x = x + y * e[2]

        # cross-attention
        x = x + self.cross_attn(self.norm3(x), context, context_lens)

        # history cross-attention
        if history_features is not None:
            history_tokens = self.history_proj(history_features)
            history_context_lens = torch.full(
                (history_tokens.shape[0],), history_tokens.shape[1],
                dtype=torch.long, device=history_tokens.device
            )
            x = x + self.history_cross_attn(
                self.norm_history(x), history_tokens, history_context_lens
            )

        # ffn
        y = self.ffn(self.norm2(x).float() * (1 + e[4]) + e[3])
        with amp.autocast(dtype=torch.float32):
            x = x + y * e[5]

        return x


class Head(nn.Module):

    def __init__(self, dim, out_dim, patch_size, eps=1e-6):
        super().__init__()
        self.dim = dim
        self.out_dim = out_dim
        self.patch_size = patch_size
        self.eps = eps

        # Output projection layers.
        out_dim = math.prod(patch_size) * out_dim
        self.norm = WanLayerNorm(dim, eps)
        self.head = nn.Linear(dim, out_dim)

        # Learnable modulation parameters.
        self.modulation = nn.Parameter(torch.randn(1, 2, dim) / dim**0.5)

    def forward(self, x, e):
        r"""
        Args:
            x(Tensor): Shape [B, L1, C]
            e(Tensor): Shape [B, C]
        """
        assert e.dtype == torch.float32
        with amp.autocast(dtype=torch.float32):
            e = (self.modulation + e.unsqueeze(1)).chunk(2, dim=1)
            x = (self.head(self.norm(x) * (1 + e[1]) + e[0]))
        return x


class MLPProj(torch.nn.Module):

    def __init__(self, in_dim, out_dim, flf_pos_emb=False):
        super().__init__()

        self.proj = torch.nn.Sequential(
            torch.nn.LayerNorm(in_dim), torch.nn.Linear(in_dim, in_dim),
            torch.nn.GELU(), torch.nn.Linear(in_dim, out_dim),
            torch.nn.LayerNorm(out_dim))
        if flf_pos_emb:  # NOTE: we only use this for `flf2v`
            self.emb_pos = nn.Parameter(
                torch.zeros(1, FIRST_LAST_FRAME_CONTEXT_TOKEN_NUMBER, 1280))

    def forward(self, image_embeds):
        if hasattr(self, "emb_pos"):
            bs, n, d = image_embeds.shape
            image_embeds = image_embeds.view(-1, 2 * n, d)
            image_embeds = image_embeds + self.emb_pos
        clip_extra_context_tokens = self.proj(image_embeds)
        return clip_extra_context_tokens


class WanModel(ModelMixin, ConfigMixin):
    r"""
    Wan diffusion backbone for the SparseVideoNav t2v pipeline.

    This model always includes:
    - QFormer for temporal compression of history frames
    - VideoFormer for spatial compression of history features
    """

    ignore_for_config = [
        "patch_size", "cross_attn_norm", "qk_norm", "text_dim", "window_size"
    ]
    _no_split_modules = ["WanAttentionBlock", "WanHistoryAttentionBlock"]
    _supports_gradient_checkpointing = True

    @register_to_config
    def __init__(self,
                 model_type="t2v",
                 patch_size=(1, 2, 2),
                 text_len=512,
                 in_dim=16,
                 dim=2048,
                 ffn_dim=8192,
                 freq_dim=256,
                 text_dim=4096,
                 out_dim=16,
                 num_heads=16,
                 num_layers=32,
                 window_size=(-1, -1),
                 qk_norm=True,
                 cross_attn_norm=True,
                 eps=1e-6,
                 qformer_config=None,
                 video_former_config=None,
                 **extra_model_kwargs,
                ):
        r"""
        Args:
            model_type(str): Backbone variant. SparseVideoNav only supports 't2v'.
            patch_size(tuple): (t_patch, h_patch, w_patch)
            qformer_config(dict): QFormer configuration
            video_former_config(dict): VideoFormer configuration
        """
        super().__init__()

        if qformer_config is None:
            raise ValueError("qformer_config is required for SparseVideoNav")
        if video_former_config is None:
            raise ValueError("video_former_config is required for SparseVideoNav")

        self.model_type = model_type
        self.patch_size = patch_size
        self.text_len = text_len
        self.in_dim = in_dim
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.freq_dim = freq_dim
        self.text_dim = text_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps

        self.qformer_config = qformer_config
        self.video_former_config = video_former_config
        self.history_dim = qformer_config.get("dim", dim)

        self.patch_embedding = nn.Conv3d(
            in_dim, dim, kernel_size=patch_size, stride=patch_size)
        self.text_embedding = nn.Sequential(
            nn.Linear(text_dim, dim), nn.GELU(approximate="tanh"),
            nn.Linear(dim, dim))
        self.time_embedding = nn.Sequential(
            nn.Linear(freq_dim, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.time_projection = nn.Sequential(nn.SiLU(), nn.Linear(dim, dim * 6))

        self.qformer = self._init_qformer()
        self.video_former = self._init_video_former()

        self.history_pre_proj = nn.Linear(16, dim)
        self.history_post_proj = nn.Linear(dim, self.qformer_config["dim"])

        cross_attn_type = "t2v_cross_attn"
        blocks = []
        for _ in range(num_layers):
            block = WanHistoryAttentionBlock(
                cross_attn_type, dim, ffn_dim, num_heads,
                window_size=window_size, qk_norm=qk_norm,
                cross_attn_norm=cross_attn_norm, eps=eps,
                history_dim=self.history_dim
            )
            blocks.append(block)
        self.blocks = nn.ModuleList(blocks)

        self.head = Head(dim, out_dim, patch_size, eps)

        # buffers (don't use register_buffer otherwise dtype will be changed in to())
        assert (dim % num_heads) == 0 and (dim // num_heads) % 2 == 0
        d = dim // num_heads
        self.freqs = torch.cat([
            rope_params(1024, d - 4 * (d // 6)),
            rope_params(1024, 2 * (d // 6)),
            rope_params(1024, 2 * (d // 6))
        ],
                               dim=1)

        # initialize weights
        self.init_weights()

    def forward(
        self,
        x,
        t,
        context,
        seq_len,
        history_features=None,
        **extra_kwargs,
    ):
        r"""
        Args:
            x(List[Tensor]): List of input videos [C_in, F, H, W]
            t(Tensor): Timesteps [B]
            context(List[Tensor]): Text embeddings [L, C]
            seq_len(int): Max sequence length
            history_features(Tensor): Pre-computed history features [B, num_tokens, dim]
        """

        device = self.patch_embedding.weight.device
        if self.freqs.device != device:
            self.freqs = self.freqs.to(device)

        # embeddings
        x = [self.patch_embedding(u.unsqueeze(0)) for u in x]
        grid_sizes = torch.stack(
            [torch.tensor(u.shape[2:], dtype=torch.long) for u in x])
        x = [u.flatten(2).transpose(1, 2) for u in x]
        seq_lens = torch.tensor([u.size(1) for u in x], dtype=torch.long)
        assert seq_lens.max() <= seq_len
        x = torch.cat([
            torch.cat([u, u.new_zeros(1, seq_len - u.size(1), u.size(2))],
                      dim=1) for u in x
        ])

        with amp.autocast(dtype=torch.float32):
            e = self.time_embedding(
                sinusoidal_embedding_1d(self.freq_dim, t).float())
            e0 = self.time_projection(e).unflatten(1, (6, self.dim))
            assert e.dtype == torch.float32 and e0.dtype == torch.float32

        text_emb_dtype = next(self.text_embedding.parameters()).dtype
        context_tensor = torch.stack([
            torch.cat([u, u.new_zeros(self.text_len - u.size(0), u.size(1))])
            for u in context
        ]).to(dtype=text_emb_dtype)
        context = self.text_embedding(context_tensor)

        kwargs = {
            "e": e0,
            "seq_lens": seq_lens,
            "grid_sizes": grid_sizes,
            "freqs": self.freqs,
            "context": context,
            "context_lens": None,
            "history_features": history_features,
        }

        for block in self.blocks:
            x = block(x, **kwargs)

        x = self.head(x, e)
        x = self.unpatchify(x, grid_sizes)
        return [u.float() for u in x]

    def unpatchify(self, x, grid_sizes):
        r"""
        Args:
            x(List[Tensor]): Patchified features [L, C_out * prod(patch_size)]
            grid_sizes(Tensor): Grid dimensions [B, 3] (F_patches, H_patches, W_patches)
        """
        c = self.out_dim
        out = []
        for u, v in zip(x, grid_sizes.tolist(), strict=False):
            u = u[:math.prod(v)].view(*v, *self.patch_size, c)
            u = torch.einsum("fhwpqrc->cfphqwr", u)
            u = u.reshape(c, *[i * j for i, j in zip(v, self.patch_size, strict=False)])
            out.append(u)
        return out

    def init_weights(self):
        r"""
        Initialize model parameters using Xavier initialization.
        """

        # Initialize linear layers.
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        # Initialize input and conditioning embeddings.
        nn.init.xavier_uniform_(self.patch_embedding.weight.flatten(1))
        for m in self.text_embedding.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=.02)
        for m in self.time_embedding.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=.02)

        # Zero-initialize the final output layer.
        nn.init.zeros_(self.head.head.weight)

    def _init_qformer(self):
        from sparseVideoNav.vgm.modules.video_former import Q_Former_3D

        return Q_Former_3D(**self.qformer_config)

    def _init_video_former(self):
        from sparseVideoNav.vgm.modules.video_former import Video_Former_3D

        return Video_Former_3D(**self.video_former_config)

    def process_history_features(self, history_latents, history_mask, context_list=None):
        r"""
        Args:
            history_latents(Tensor): VAE latents [B, C=16, F, H, W]
            history_mask(Tensor): Valid frame mask [B, F]
            context_list(List[Tensor] or Tensor): Text embeddings
        """
        if history_latents is None:
            return None

        device = history_latents.device
        x = rearrange(history_latents, "b c f h w -> b f (h w) c")

        x = self.history_pre_proj(x)
        x = self.history_post_proj(x)

        language_features = None
        language_mask = None
        if context_list is not None:
            if isinstance(context_list, torch.Tensor):
                language_features = context_list
                language_mask = torch.ones(
                    context_list.shape[0], context_list.shape[1],
                    dtype=torch.bool, device=device
                )
            elif isinstance(context_list, list) and len(context_list) > 0:
                max_lang_len = max(ctx.size(0) for ctx in context_list)
                padded_lang_features = []
                lang_masks = []
                for ctx in context_list:
                    seq_len = ctx.size(0)
                    if seq_len < max_lang_len:
                        padding = ctx.new_zeros(max_lang_len - seq_len, ctx.size(1))
                        padded_ctx = torch.cat([ctx, padding], dim=0)
                    else:
                        padded_ctx = ctx
                    padded_lang_features.append(padded_ctx)

                    mask = torch.ones(max_lang_len, dtype=torch.bool, device=device)
                    if seq_len < max_lang_len:
                        mask[seq_len:] = False
                    lang_masks.append(mask)

                language_features = torch.stack(padded_lang_features, dim=0)
                language_mask = torch.stack(lang_masks, dim=0)

        x = self.qformer(
            x,
            mask=history_mask,
            language_feature=language_features,
            language_mask=language_mask
        )

        num_latents = x.shape[1]
        _, _, _, H, W = history_latents.shape
        spatial_tokens_per_frame = H * W
        num_frames, remainder = divmod(num_latents, spatial_tokens_per_frame)
        if remainder != 0:
            raise ValueError(
                "History features cannot be reshaped back to frame-major layout: "
                f"num_latents={num_latents}, spatial_tokens_per_frame={spatial_tokens_per_frame}"
            )

        x = rearrange(
            x,
            "b (f hw) d -> b f hw d",
            f=num_frames,
            hw=spatial_tokens_per_frame
        )
        x = self.video_former(
            x,
            language_feature=language_features,
            language_mask=language_mask
        )

        return x
