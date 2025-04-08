import torch
import torch.nn as nn
import torch.nn.functional as F
from attrdict import AttrDict
from torch.distributions.normal import Normal

class LAYERANP(nn.Module):
    """Attention Neural Process model with multiple Transformer layers"""
    def __init__(self, x_dim=1, y_dim=1, embed_dim=128, ff_dim=512, num_layers=3, norm_type='layernorm', **kwargs):
        super().__init__()
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.embed_dim = embed_dim
        self.num_layers = num_layers

        # Embedding layers
        self.encoder_embedding_layer = nn.Linear(x_dim + y_dim, embed_dim)
        self.decoder_embedding_layer = nn.Linear(x_dim, embed_dim)

        # Transformer layers
        self.layers = nn.ModuleList([
            TransformerLayer(embed_dim, ff_dim, norm_type) for _ in range(num_layers)
        ])

        # Output layer
        self.decoder_output_layer = nn.Linear(embed_dim, y_dim * 2)  # Mean and sigma

    def forward(self, batch):
        """
        Args:
            batch: AttrDict with keys 'xc', 'yc', 'xt', 'yt'
        Returns:
            outputs: AttrDict with keys 'loss', 'tar_ll'
        """
        # Process context points
        xy_c = torch.cat([batch.xc, batch.yc], dim=-1)                      # (N_c, d+1)
        xy_c_emb = self.encoder_embedding_layer(xy_c)                      # (N_c, emb_dim)

        # Process target points
        x_t_emb = self.decoder_embedding_layer(batch.xt)                   # (N_t, emb_dim)

        # Pass through Transformer layers
        for layer in self.layers:
            xy_c_emb, x_t_emb = layer(xy_c_emb, x_t_emb)

        # Output layer
        output = self.decoder_output_layer(x_t_emb)                        # (N_t, 2)
        mean, log_std = torch.chunk(output, 2, dim=-1)
        scale = torch.exp(log_std) + 1e-6  # Ensure positive scale

        # Compute log-likelihood
        dist = Normal(mean, scale)
        tar_ll = dist.log_prob(batch.yt).sum(-1).mean()
        loss = -tar_ll

        outputs = AttrDict()
        outputs.loss = loss
        outputs.tar_ll = tar_ll

        return outputs

    def predict(self, xc, yc, xt):
        """
        Args:
            xc: context x values (batch_size, n_context, x_dim)
            yc: context y values (batch_size, n_context, y_dim)
            xt: target x values (batch_size, n_target, x_dim)
        Returns:
            outputs: AttrDict with keys 'mean', 'scale', 'loc', 'log_scale'
        """
        self.eval()
        with torch.no_grad():
            # Process context points
            xy_c = torch.cat([xc, yc], dim=-1)
            xy_c_emb = self.encoder_embedding_layer(xy_c)

            # Process target points
            x_t_emb = self.decoder_embedding_layer(xt)

            # Pass through Transformer layers
            for layer in self.layers:
                xy_c_emb, x_t_emb = layer(xy_c_emb, x_t_emb)

            # Output layer
            output = self.decoder_output_layer(x_t_emb)
            mean, log_std = torch.chunk(output, 2, dim=-1)
            scale = torch.exp(log_std) + 1e-6

        outputs = AttrDict()
        outputs.loc = mean
        outputs.mean = mean
        outputs.scale = scale
        outputs.log_scale = torch.log(scale)

        return outputs

    def sample(self, xc, yc, xt, num_samples=1):
        """
        Args:
            xc: context x values 
            yc: context y values
            xt: target x values
            num_samples: number of samples to draw
        Returns:
            samples: (num_samples, batch_size, n_target, y_dim)
        """
        self.eval()
        with torch.no_grad():
            outputs = self.predict(xc, yc, xt)
            dist = Normal(outputs.loc, outputs.scale)
            samples = [dist.sample() for _ in range(num_samples)]
            samples = torch.stack(samples, dim=0)

        return samples


class TransformerLayer(nn.Module):
    """One layer of the Transformer with self-attention, cross-attention, and feed-forward network"""
    def __init__(self, embed_dim, ff_dim, norm_type='layernorm'):
        super().__init__()
        self.context_self_attention = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=1, batch_first=True)
        self.target_self_attention = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=1, batch_first=True)
        self.cross_attention = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=1, batch_first=True)

        self.up_proj = nn.Linear(embed_dim, ff_dim)
        self.down_proj = nn.Linear(ff_dim, embed_dim)

        # Normalization layers
        if norm_type == 'layernorm':
            self.context_norm = nn.LayerNorm(embed_dim)
            self.target_norm = nn.LayerNorm(embed_dim)
            self.cross_norm = nn.LayerNorm(embed_dim)
            self.ff_norm = nn.LayerNorm(embed_dim)
        elif norm_type == 'rmsnorm':
            self.context_norm = RMSNorm(embed_dim)
            self.target_norm = RMSNorm(embed_dim)
            self.cross_norm = RMSNorm(embed_dim)
            self.ff_norm = RMSNorm(embed_dim)
        else:
            raise ValueError(f"Unsupported norm_type: {norm_type}")

    def forward(self, xy_c_emb, x_t_emb):
        # Self-attention for context
        xy_c_emb_attn, _ = self.context_self_attention(xy_c_emb, xy_c_emb, xy_c_emb)
        xy_c_emb = self.context_norm(xy_c_emb + xy_c_emb_attn)

        # Self-attention for target
        x_t_emb_attn, _ = self.target_self_attention(x_t_emb, x_t_emb, x_t_emb)
        x_t_emb = self.target_norm(x_t_emb + x_t_emb_attn)

        # Cross-attention between target and context
        x_t_emb_cross, _ = self.cross_attention(x_t_emb, xy_c_emb, xy_c_emb)
        x_t_emb = self.cross_norm(x_t_emb + x_t_emb_cross)

        # Feed-forward network
        x_t_emb_ff = self.down_proj(F.gelu(self.up_proj(x_t_emb)))
        x_t_emb = self.ff_norm(x_t_emb + x_t_emb_ff)

        return xy_c_emb, x_t_emb


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization"""
    def __init__(self, embed_dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(embed_dim))
        self.eps = eps

    def forward(self, x):
        norm = x.norm(2, dim=-1, keepdim=True)
        return self.weight * x / (norm + self.eps)