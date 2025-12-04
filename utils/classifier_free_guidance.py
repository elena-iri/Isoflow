import os
import sys
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import anndata as ad
import matplotlib.pyplot as plt
import scanpy as sc
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scvi.distributions import NegativeBinomial

class EmpiricalDistribution(nn.Module):
    def __init__(self, data):
        super().__init__()
        self.register_buffer("data", data)
    def sample(self, n):
        idx = torch.randint(0, len(self.data), (n,), device=self.data.device)
        return self.data[idx]

class GaussianConditionalProbabilityPath:
    def __init__(self, p_data):
        self.p_data = p_data
    def sample_conditional_path(self, z, t):
        # Linear interpolation: t * z + (1-t) * noise
        return t * z + (1 - t) * torch.randn_like(z)

    def conditional_vector_field(self, x_1, x_0):
        return x_1 - x_0

class CellTypeConditioner(nn.Module):
    def __init__(self, n_types, latent_dim):
        super().__init__()
        self.embed = nn.Embedding(n_types, latent_dim)
    def forward(self, idx):
        return self.embed(idx)

class TimeEmbedder(nn.Module):
    def __init__(self, embed_dim=32):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim), nn.SiLU(),
            nn.Linear(embed_dim, embed_dim), nn.SiLU()
        )
        self.embed_dim = embed_dim
    def forward(self, t):
        half_dim = self.embed_dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device) * -emb)
        emb = t * emb[None, :]
        emb = torch.cat((torch.sin(emb), torch.cos(emb)), dim=1)
        return self.mlp(emb)
    
class ResNetBlock(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, dim)
        )
    def forward(self, x):
        return x + self.mlp(x)

class NeuralVectorField(nn.Module):
    def __init__(self, latent_dim, hidden_dim=256, n_resblocks=5, time_embed_dim=64):
        super().__init__()
        self.x_proj = nn.Linear(latent_dim, hidden_dim)
        self.c_proj = nn.Linear(latent_dim, hidden_dim)
        self.l_proj = nn.Linear(1, hidden_dim)
        self.time_embedder = TimeEmbedder(time_embed_dim)
        self.null_cond = nn.Parameter(torch.randn(1, latent_dim))

        input_dim = hidden_dim * 3 + time_embed_dim 
        self.resblocks = nn.ModuleList([
            ResNetBlock(input_dim, hidden_dim * 3) for _ in range(n_resblocks)
        ])
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_dim * 3 + time_embed_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, latent_dim)
        )

    def forward(self, x, c, t, l):
        xh = self.x_proj(x)
        ch = self.c_proj(c) 
        th = self.time_embedder(t)
        lh = self.l_proj(l) 
        h = torch.cat([xh, ch, lh, th], dim=-1)
        for block in self.resblocks:
            h = block(h)
        return self.output_layer(h)

# Euler Integration Class
class LearnedVectorFieldODE:
    def __init__(self, vf_model, conditioner, z_target_idx, l_target, guidance_scale=2.0):
        self.vf = vf_model
        self.c = conditioner(z_target_idx) # Embed the cell type indices
        self.l = l_target
        self.scale = guidance_scale
        self.c_null = self.vf.null_cond.expand(self.c.shape[0], -1)
    
    def drift(self, x, t):
        # Duplicate inputs for [Conditional, Unconditional] batching
        x_in = torch.cat([x, x], dim=0)
        t_in = torch.cat([t, t], dim=0)
        l_in = torch.cat([self.l, self.l], dim=0)
        
        # Stack: [Conditioned, Null]
        c_in = torch.cat([self.c, self.c_null], dim=0)
        
        # Forward Pass
        v_out = self.vf(x_in, c_in, t_in, l_in)
        v_cond, v_uncond = v_out.chunk(2, dim=0)
        
        # CFG Formula: v = v_uncond + s * (v_cond - v_uncond)
        return v_uncond + self.scale * (v_cond - v_uncond)


def plot_umap_scatter(ax, adata_subset, color_col, title, palette=None, s=10):
    sc.pl.umap(adata_subset, color=color_col, ax=ax, show=False, 
               title=title, frameon=True, s=s, palette=palette, legend_loc='on data')
    ax.set_xlabel("UMAP1")
    ax.set_ylabel("UMAP2")


def plot_overlap(ax, c_type, adata_merged):
    mask_r = (adata_merged.obs['dataset'] == 'Real') & (adata_merged.obs['cell_type'] == c_type)
    mask_g = (adata_merged.obs['dataset'] == 'Generated') & (adata_merged.obs['cell_type'] == c_type)
    
    umap_r = adata_merged[mask_r].obsm['X_umap']
    umap_g = adata_merged[mask_g].obsm['X_umap']
    
    ax.scatter(umap_r[:, 0], umap_r[:, 1], s=15, c='#377eb8', alpha=0.5, label='Real')
    ax.scatter(umap_g[:, 0], umap_g[:, 1], s=15, c='#e41a1c', alpha=0.6, label='Generated')
    
    ax.set_title(c_type)
    ax.set_xticks([])
    ax.set_yticks([])

