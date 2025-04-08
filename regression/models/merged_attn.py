import torch
import torch.nn as nn
import torch.nn.functional as F
from attrdict import AttrDict
from torch.distributions.normal import Normal

class MERGED_ATTN(nn.Module):
    """Neural Process model that merges context and target points using only self-attention"""
    def __init__(self, x_dim=1, y_dim=1, embed_dim=128, ff_dim=512, num_layers=3, norm_type='layernorm', **kwargs):
        super().__init__()
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.embed_dim = embed_dim
        self.num_layers = num_layers

        # Embedding layers
        self.context_embedding_layer = nn.Linear(x_dim + y_dim, embed_dim)  # Input: (B, N_c, x_dim + y_dim) → (B, N_c, embed_dim)
        self.target_embedding_layer = nn.Linear(x_dim, embed_dim)           # Input: (B, N_t, x_dim) → (B, N_t, embed_dim)

        # Transformer layers
        self.layers = nn.ModuleList([
            SelfAttentionLayer(embed_dim, ff_dim, norm_type) for _ in range(num_layers)
        ])

        # Output layer
        self.decoder_output_layer = nn.Linear(embed_dim, y_dim * 2)         # Input: (B, N_t, embed_dim) → (B, N_t, 2 * y_dim)

    def forward(self, batch, reduce_ll=True):
        """
        Args:
            batch: AttrDict with keys 'xc', 'yc', 'xt', 'yt'
        Returns:
            outputs: AttrDict with keys 'loss', 'tar_ll'
        """
        # Shapes:
        # batch.xc: (B, N_c, x_dim)
        # batch.yc: (B, N_c, y_dim)
        # batch.xt: (B, N_t, x_dim)
        # batch.yt: (B, N_t, y_dim)

        n_context = batch.xc.shape[-2]  # N_c
        n_target = batch.xt.shape[-2]   # N_t

        xy_c = torch.cat([batch.xc, batch.yc], dim=-1)                      # (B, N_c, x_dim + y_dim)
        xy_c_emb = self.context_embedding_layer(xy_c)                       # (B, N_c, embed_dim)
        x_t_emb = self.target_embedding_layer(batch.xt)                     # (B, N_t, embed_dim)

        merged_emb = torch.cat([xy_c_emb, x_t_emb], dim=-2)                 # (B, N_c + N_t, embed_dim)

        for layer in self.layers:
            merged_emb = layer(merged_emb)                                  # (B, N_c + N_t, embed_dim)

        target_emb = merged_emb[:, -n_target:, :]                           # (B, N_t, embed_dim)

        output = self.decoder_output_layer(target_emb)                      # (B, N_t, 2 * y_dim)
        mean, log_std = torch.chunk(output, 2, dim=-1)                      # Each: (B, N_t, y_dim)
        scale = torch.exp(log_std) + 1e-6                                   # (B, N_t, y_dim)

        dist = Normal(mean, scale)
        if reduce_ll:
            tar_ll = dist.log_prob(batch.yt).sum(-1).mean()                     # (B,) → scalar
        else:
            tar_ll = dist.log_prob(batch.yt).sum(-1)
        loss = -tar_ll

        outputs = AttrDict()
        outputs.loss = loss
        outputs.tar_ll = tar_ll

        return outputs

    def predict(self, xc, yc, xt):
        """
        Args:
            xc: (B, N_c, x_dim)
            yc: (B, N_c, y_dim)
            xt: (B, N_t, x_dim)
        Returns:
            outputs: AttrDict with keys 'mean', 'scale', 'loc', 'log_scale'
        """
        self.eval()
        with torch.no_grad():
            n_context = xc.shape[-2]                                        # N_c
            n_target = xt.shape[-2]                                         # N_t

            xy_c = torch.cat([xc, yc], dim=-1)                              # (B, N_c, x_dim + y_dim)
            xy_c_emb = self.context_embedding_layer(xy_c)                   # (B, N_c, embed_dim)
            x_t_emb = self.target_embedding_layer(xt)                       # (B, N_t, embed_dim)

            merged_emb = torch.cat([xy_c_emb, x_t_emb], dim=-2)             # (B, N_c + N_t, embed_dim)

            for layer in self.layers:
                merged_emb = layer(merged_emb)                              # (B, N_c + N_t, embed_dim)

            target_emb = merged_emb[..., -n_target:, :]                     # (B, N_t, embed_dim)

            output = self.decoder_output_layer(target_emb)                  # (B, N_t, 2 * y_dim)
            mean, log_std = torch.chunk(output, 2, dim=-1)                  # Each: (B, N_t, y_dim)
            scale = torch.exp(log_std) + 1e-6                               # (B, N_t, y_dim)

        outputs = AttrDict()
        outputs.loc = mean                                                  # (B, N_t, y_dim)
        outputs.mean = mean
        outputs.scale = scale                                               # (B, N_t, y_dim)
        outputs.log_scale = torch.log(scale)                                # (B, N_t, y_dim)

        return outputs

    def sample(self, xc, yc, xt, num_samples=1):
        """
        Args:
            xc: (B, N_c, x_dim)
            yc: (B, N_c, y_dim)
            xt: (B, N_t, x_dim)
            num_samples: int
        Returns:
            samples: (num_samples, B, N_t, y_dim)
        """
        self.eval()
        with torch.no_grad():
            outputs = self.predict(xc, yc, xt)
            dist = Normal(outputs.loc, outputs.scale)
            samples = [dist.sample() for _ in range(num_samples)]          # num_samples × (B, N_t, y_dim)
            samples = torch.stack(samples, dim=0)                          # (num_samples, B, N_t, y_dim)

        return samples


class SelfAttentionLayer(nn.Module):
    """Simple transformer layer with only self-attention and feed-forward network"""
    def __init__(self, embed_dim, ff_dim, norm_type='layernorm'):
        super().__init__()
        self.self_attention = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=8, batch_first=True)
        
        self.up_proj = nn.Linear(embed_dim, ff_dim)                         # (B, N, embed_dim) → (B, N, ff_dim)
        self.down_proj = nn.Linear(ff_dim, embed_dim)                      # (B, N, ff_dim) → (B, N, embed_dim)

        if norm_type == 'layernorm':
            self.attn_norm = nn.LayerNorm(embed_dim)
            self.ff_norm = nn.LayerNorm(embed_dim)
        elif norm_type == 'rmsnorm':
            self.attn_norm = RMSNorm(embed_dim)
            self.ff_norm = RMSNorm(embed_dim)
        else:
            raise ValueError(f"Unsupported norm_type: {norm_type}")
    
    def forward(self, x):
        # x: (B, N, embed_dim)
        x_attn, _ = self.self_attention(x, x, x)                            # (B, N, embed_dim)
        x = self.attn_norm(x + x_attn)                                     # (B, N, embed_dim)

        x_ff = self.down_proj(F.gelu(self.up_proj(x)))                     # (B, N, embed_dim)
        x = self.ff_norm(x + x_ff)                                         # (B, N, embed_dim)

        return x


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization"""
    def __init__(self, embed_dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(embed_dim))                 # (embed_dim,)
        self.eps = eps
        
    def forward(self, x):
        # x: (B, N, embed_dim)
        norm = x.norm(2, dim=-1, keepdim=True)                             # (B, N, 1)
        return self.weight * x / (norm + self.eps)                         # (B, N, embed_dim)
