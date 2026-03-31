# Adapted from flamingo-mini:
# https://github.com/dhansmair/flamingo-mini

import math

from einops import rearrange
from einops import repeat
import torch
from torch import nn
import torch.nn.functional as F


class SquaredReLU(nn.Module):
    """Squared ReLU activation."""

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.pow(torch.relu(x), 2)


def feed_forward_layer(dim: int, mult: int = 4, activation: str = "gelu"):
    """Build a feed-forward block with the requested activation."""

    activations = dict(gelu=nn.GELU, sqrelu=SquaredReLU, relu=nn.ReLU)
    assert activation in activations, f"activation can only be one of {activations.keys()}"

    inner_dim = int(dim * mult)
    return nn.Sequential(
        nn.Linear(dim, inner_dim, bias=False),
        activations[activation](),
        nn.Linear(inner_dim, dim, bias=False),
    )

class Attention(nn.Module):
    def __init__(
            self,
            n_embd: int,
            n_head: int,
            attn_pdrop: float = 0.1,
            resid_pdrop: float = 0.1,
            causal: bool = False,
            bias=True,
        ):
        super().__init__()
        assert n_embd % n_head == 0
        # Per-head query, key, and value projections.
        self.key = nn.Linear(n_embd, n_embd)
        self.query = nn.Linear(n_embd, n_embd)
        self.value = nn.Linear(n_embd, n_embd)
        # Output projection.
        self.c_proj = nn.Linear(n_embd, n_embd, bias=bias)
        # Dropout layers.
        self.attn_dropout = nn.Dropout(attn_pdrop)
        self.resid_dropout = nn.Dropout(resid_pdrop)
        self.n_head = n_head
        self.n_embd = n_embd
        self.causal = causal

    def forward(self, x, context=None, custom_attn_mask=None):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)

        # Build attention projections for either self-attention or cross-attention.
        if context is not None:
            k = self.key(context).reshape(B, -1, self.n_head, C // self.n_head).transpose(1, 2).contiguous() # (B, nh, T, hs)
            q = self.query(x).reshape(B, T, self.n_head, C // self.n_head).transpose(1, 2).contiguous() # (B, nh, T, hs)
            v = self.value(context).reshape(B, -1, self.n_head, C // self.n_head).transpose(1, 2).contiguous() # (B, nh, T, hs)
        else:
            k = self.key(x).reshape(B, T, self.n_head, C // self.n_head).transpose(1, 2).contiguous() # (B, nh, T, hs)
            q = self.query(x).reshape(B, T, self.n_head, C // self.n_head).transpose(1, 2).contiguous() # (B, nh, T, hs)
            v = self.value(x).reshape(B, T, self.n_head, C // self.n_head).transpose(1, 2).contiguous() # (B, nh, T, hs)

        if custom_attn_mask is not None:
            custom_attn_mask = custom_attn_mask.unsqueeze(1).unsqueeze(2)  # [n_batch, 1, 1, n_features]
            custom_attn_mask = custom_attn_mask.expand(-1, self.n_head, x.shape[1], -1)  # [n_batch, heads, n_queries, n_features_latents]

        # Use PyTorch scaled dot-product attention.
        y = torch.nn.functional.scaled_dot_product_attention(
            q, k, v,
            attn_mask=custom_attn_mask,
            dropout_p=self.attn_dropout.p if self.training else 0,
            is_causal=self.causal
        )
        y = y.transpose(1, 2).contiguous().reshape(B, T, C) # re-assemble all head outputs side by side

        # Project back to the model dimension.
        y = self.resid_dropout(self.c_proj(y))
        return y

class PerceiverAttentionLayer(nn.Module):
    def __init__(self, dim: int, dim_head: int = 64, heads: int = 8, attn_dropout: float = 0.0):
        super().__init__()
        self.heads = heads
        self.dim_head = dim_head
        inner_dim = dim_head * heads

        self.norm_media = nn.LayerNorm(dim)
        self.norm_latents = nn.LayerNorm(dim)
        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_k = nn.Linear(dim, inner_dim, bias=False)
        self.to_v = nn.Linear(dim, inner_dim, bias=False)
        self.to_out = nn.Linear(inner_dim, dim, bias=False)
        self.attn_dropout = nn.Dropout(attn_dropout)

    def forward(self, features, latents, mask: torch.BoolTensor = None):
        n_batch, n_features, dim = features.shape
        n_queries = latents.shape[1]

        if mask is not None:
            features = features * mask.unsqueeze(-1)

        x = self.norm_media(features)
        latents = self.norm_latents(latents)

        q = self.to_q(latents)
        q = rearrange(q, "b q (h d) -> b h q d", h=self.heads)

        kv_input = torch.cat((x, latents), dim=-2)
        k = self.to_k(kv_input)
        v = self.to_v(kv_input)
        k = rearrange(k, "b f (h d) -> b h f d", h=self.heads)
        v = rearrange(v, "b f (h d) -> b h f d", h=self.heads)

        # Build the attention mask expected by scaled dot-product attention.
        attn_mask = None
        if mask is not None:
            features_mask = mask.unsqueeze(1).unsqueeze(2)  # [n_batch, 1, 1, n_features]
            latents_mask = torch.ones(n_batch, 1, 1, n_queries, device=mask.device, dtype=torch.bool)
            combined_mask = torch.cat([features_mask, latents_mask], dim=-1)  # [n_batch, 1, 1, n_features_latents]
            combined_mask = combined_mask.expand(-1, self.heads, n_queries, -1)  # [n_batch, heads, n_queries, n_features_latents]
            attn_mask = combined_mask

        # Apply attention over feature and latent tokens.
        scale = self.dim_head ** -0.5
        q = q * scale

        out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,  # Automatically handles numerical stability
            dropout_p=self.attn_dropout.p if self.training else 0.0,
            is_causal=False
        )

        out = rearrange(out, "b h q d -> b q (h d)")
        return self.to_out(out)

class FiLMLayer(nn.Module):
    """Generate FiLM modulation parameters from language features."""
    def __init__(self, language_dim, inner_dim, num_layers=2):
        super().__init__()
        self.language_dim = language_dim
        self.inner_dim = inner_dim

        layers = []
        input_dim = language_dim
        for i in range(num_layers - 1):
            layers.append(nn.Linear(input_dim, inner_dim))
            layers.append(nn.ReLU())
            input_dim = inner_dim

        # The final layer predicts gamma and beta.
        final_layer = nn.Linear(input_dim, inner_dim * 2)
        layers.append(final_layer)
        self.mlp = nn.Sequential(*layers)

        # Initialize the FiLM MLP near zero.
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize weights so the initial FiLM modulation is near zero."""
        for module in self.mlp.modules():
            if isinstance(module, nn.Linear):
                # Keep the initial modulation small.
                nn.init.normal_(module.weight, mean=0.0, std=0.01)
                # Start from zero bias.
                nn.init.zeros_(module.bias)

        # Make the last layer even smaller for stable initialization.
        last_layer = self.mlp[-1]
        nn.init.normal_(last_layer.weight, mean=0.0, std=0.001)  # Smaller standard deviation
        nn.init.zeros_(last_layer.bias)

    def forward(self, language_features, language_mask=None):
        """
        Args:
            language_features: [B, seq_len, dim] language features
            language_mask: [B, seq_len] boolean mask, True for valid tokens, False for padding
        """
        output = self.mlp(language_features)
        gamma, beta = output.chunk(2, dim=-1)

        # Keep FiLM scaling bounded near zero at initialization.
        gamma = F.tanh(gamma)

        if language_mask is not None:
            # Broadcast the token mask across the channel dimension.
            mask_expanded = language_mask.unsqueeze(-1).float()

            # Ignore padding tokens when pooling FiLM parameters.
            gamma_masked = gamma * mask_expanded
            beta_masked = beta * mask_expanded

            # Average only across valid tokens.
            valid_token_counts = language_mask.sum(dim=1, keepdim=True).float()  # [B, 1]
            valid_token_counts = torch.clamp(valid_token_counts, min=1.0)  # Avoid division by zero

            gamma_mean = gamma_masked.sum(dim=1, keepdim=True) / valid_token_counts.unsqueeze(-1)  # [B, 1, dim]
            beta_mean = beta_masked.sum(dim=1, keepdim=True) / valid_token_counts.unsqueeze(-1)   # [B, 1, dim]
        else:
            # Fall back to a simple mean when no mask is provided.
            gamma_mean = gamma.mean(dim=1, keepdim=True)
            beta_mean = beta.mean(dim=1, keepdim=True)

        return gamma_mean, beta_mean

    def modulate(self, x, gamma, beta):
        return x * (1 + gamma) + beta


class Video_Former_3D(nn.Module):
    """VideoFormer with spatial-temporal resampling and optional language FiLM."""

    def __init__(
            self,
            dim: int,
            depth: int,
            video_feature_dim: int,
            language_dim: int,
            dim_head: int = 64,
            heads: int = 8,
            num_latents: int = 256,
            num_frame: int = 64,
            ff_mult: int = 4,
            activation: str = "gelu",
            attn_pdrop: float = 0.1,
            resid_pdrop: float = 0.1,
            **extras
    ):
        super().__init__()

        self.dim = dim  # Internal and output dimension (768)
        self.num_queries = num_latents
        self.num_frame = num_frame
        self.condition_dim = video_feature_dim

        self.pre_proj = nn.Sequential(
            nn.Linear(video_feature_dim, dim * 2),
            nn.GELU(),
            nn.Linear(dim * 2, dim)
        )

        frame_seq_len = num_latents // num_frame
        self.latents = nn.Parameter(torch.randn(self.num_frame, frame_seq_len, dim)).requires_grad_(True)

        self.layers = nn.ModuleList([])
        self.film_layer = FiLMLayer(language_dim, dim)

        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        PerceiverAttentionLayer(dim=dim, dim_head=dim_head, heads=heads),
                        # Temporal self-attention over latent tokens.
                        Attention(
                            n_embd=dim,
                            n_head=heads,
                            attn_pdrop=attn_pdrop,
                            resid_pdrop=resid_pdrop,
                            causal=False  # Temporal relationship can be bidirectional
                        ),
                        nn.LayerNorm(dim),
                        feed_forward_layer(dim=dim, mult=ff_mult, activation=activation),
                    ]
                )
            )

        self.norm = nn.LayerNorm(dim)

    def forward(self, x_f: torch.Tensor, mask: torch.BoolTensor = None, language_feature: torch.Tensor = None, language_mask: torch.BoolTensor = None):
        """Resample visual features with optional language modulation."""
        assert x_f.ndim == 4
        batch_size, max_length, n_features, input_dim = x_f.shape

        # Compute FiLM parameters from language features when provided.
        if language_feature is not None:
            # Infer a mask if the caller does not provide one.
            if language_mask is None:
                language_mask = (language_feature != 0).any(dim=-1)  # [B, seq_len]

            gamma, beta = self.film_layer(language_feature, language_mask)

        x_f = self.pre_proj(x_f)

        if mask is not None:
            x_f = x_f * mask.unsqueeze(-1).unsqueeze(-1)

        # Prepare the spatial attention mask.
        spatial_mask = None
        if mask is not None:
            spatial_mask = repeat(mask, "b f -> (b f) c", c=n_features)

        # Flatten frames into a batch of per-frame token sequences.
        x_f = rearrange(x_f, "b T n d -> (b T) n d")

        # Repeat the learnable latents for each batch item and frame.
        x = repeat(self.latents, "T q d -> b T q d", b=batch_size)
        x = rearrange(x, "b T q d -> (b T) q d")

        for attn, temp_attn, lm, ffw in self.layers:
        # Spatial attention over tokens within each frame.
            x = x + attn(x_f, x, mask=spatial_mask)
        # Temporal attention across frames.
            x_rearranged = rearrange(x, "(b T) q d -> (b q) T d", b=batch_size)
            temp_output = temp_attn(x_rearranged)  # Direct self-attention, no mask
            x = x + rearrange(temp_output, "(b q) T d -> (b T) q d", b=batch_size)
            x = lm(x)
            if  language_feature is not None:
                x = rearrange(x, "(b T) q d -> b (T q) d", b=batch_size)
                x = self.film_layer.modulate(x, gamma, beta)
                x = rearrange(x, "b (T q) d -> (b T) q d", T=self.num_frame)
            x = x + ffw(x)
        # Restore the original batch layout.
        x = rearrange(x, "(b T) q d -> b (T q) d", b=batch_size)
        assert x.shape == torch.Size([batch_size, self.num_queries, self.dim])

        # Normalize the output tokens.
        x = self.norm(x)

        return x


class Q_Former_3D(nn.Module):
    """Q-Former style temporal compressor with optional FiLM modulation."""

    def __init__(
            self,
            dim: int,
            depth: int,
            video_feature_dim: int,
            language_dim: int,
            heads: int = 8,
            num_latents: int = 256,
            ff_mult: int = 4,
            activation: str = "gelu",
            attn_pdrop: float = 0.1,
            resid_pdrop: float = 0.1,
            max_frames: int = 512,
            **kwargs
    ):
        super().__init__()

        self.dim = dim
        self.num_queries = num_latents
        self.condition_dim = video_feature_dim

        self.latents = nn.Parameter(torch.randn(self.num_queries, dim)).requires_grad_(True)
        self.film_layer = FiLMLayer(language_dim, dim)

        position = torch.arange(max_frames).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2) * (-math.log(10000.0) / dim))

        pos_embedding = torch.zeros(max_frames, dim)
        pos_embedding[:, 0::2] = torch.sin(position * div_term)
        pos_embedding[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer("internal_pos_embedding", pos_embedding)
        self.pos_embed_proj = nn.Linear(dim, dim)

        self.layers = nn.ModuleList([])

        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        Attention(
                            n_embd=dim,
                            n_head=heads,
                            attn_pdrop=attn_pdrop,
                            resid_pdrop=resid_pdrop,
                            causal=False  # Temporal relationship can be bidirectional
                        ),
                        nn.LayerNorm(dim),
                        feed_forward_layer(dim=dim, mult=ff_mult, activation=activation),
                    ]
                )
            )

        self.norm = nn.LayerNorm(dim)

    def forward(self, x_f: torch.Tensor, mask: torch.BoolTensor = None,
                language_feature: torch.Tensor = None, language_mask: torch.BoolTensor = None):
        """Compress spatial-temporal features with query tokens."""
        assert x_f.ndim == 4
        batch_size, num_frames, n_features, input_dim = x_f.shape

        # Compute FiLM parameters from language features when provided.
        if language_feature is not None:
            # Infer a mask if the caller does not provide one.
            if language_mask is None:
                language_mask = (language_feature != 0).any(dim=-1)

            gamma, beta = self.film_layer(language_feature, language_mask)

        assert num_frames <= self.internal_pos_embedding.shape[0], \
            f"num_frames ({num_frames}) exceeds max_frames ({self.internal_pos_embedding.shape[0]})"

        frame_pos_emb = torch.flip(self.internal_pos_embedding[:num_frames], dims=[0])
        frame_pos_emb = self.pos_embed_proj(frame_pos_emb)
        frame_pos_emb = frame_pos_emb.unsqueeze(0).unsqueeze(2)
        x_f = x_f + frame_pos_emb

        # Apply the frame mask before flattening.
        if mask is not None:
            x_f = x_f * mask.unsqueeze(-1).unsqueeze(-1)

        # Flatten spatial-temporal tokens into a single context sequence.
        x_context = rearrange(x_f, "b f hw d -> b (f hw) d")

        # Repeat the learnable query latents across the batch.
        x = repeat(self.latents, "n d -> b n d", b=batch_size)

        # Build the flattened context mask.
        temporal_mask = None
        if mask is not None:
            temporal_mask = repeat(mask, "b f -> b (f hw)", hw=n_features)

        # Apply cross-attention and feed-forward layers.
        for _, (attn, lm, ffw) in enumerate(self.layers):
            # Attend from query latents into the flattened video context.
            x = x + attn(x, context=x_context, custom_attn_mask=temporal_mask)
            x = lm(x)

            if language_feature is not None:
                x = self.film_layer.modulate(x, gamma, beta)

            x = x + ffw(x)

        # Normalize the compressed latent sequence.
        x = self.norm(x)

        assert x.shape == torch.Size([batch_size, self.num_queries, self.dim])
        return x



if __name__ == "__main__":
    # Minimal smoke test.
    batch_size, channels, frames, height, width = 2, 512, 64, 4, 4
    x = torch.randn(batch_size, channels, frames, height, width)    # [B, C, F, H, W]
    lang_feature = torch.randn(batch_size, 77, 1024)

    # Example frame-validity mask.
    mask = torch.cat([
        torch.ones(batch_size, 32, dtype=torch.bool),
        torch.zeros(batch_size, 32, dtype=torch.bool)
    ], dim=1)

    processor = Video_Former_3D(
        dim=256,
        language_dim=1024,
        depth=8,
        video_feature_dim=x.shape[1],
        num_frame=frames  # Must match the input frame count.
    )

    # Reshape to `(B, F, HW, C)` before running the module.
    x = rearrange(x, "b c f h w -> b f (h w) c")

    output = processor(x, mask=mask, language_feature=lang_feature)

    # -----
    x = torch.randn(batch_size, channels, frames, height, width)    # [B, C, F, H, W]
    processor = Q_Former_3D(
        dim=256,
        language_dim=1024,
        depth=8,
        video_feature_dim=x.shape[1],
        num_frame=128   # Does not need to match the input frame count.
    )
    # Reshape to `(B, F, HW, C)` before running the module.
    x = rearrange(x, "b c f h w -> b f (h w) c")

    output = processor(x, mask=mask, language_feature=lang_feature)

    # Example parameter settings:
    """
    q-former: dim = 768; heads = 12; num_frame = 5; num_time_embeds = 512
    video-former: dim = 768; dim_head = 64; heads = 12; num_frame = 5; num_latents = 80; num_time_embeds = 512
    
    """
