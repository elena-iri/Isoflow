#!/usr/bin/env python
# coding: utf-8

# # Training of the flow matching model

# In this script with classes and functions defining / training flow matching model.


# - Loading libraries

from abc import ABC, abstractmethod
from typing import Optional #, List, Type, Tuple, Dict
import math
import anndata as ad
#import numpy as np
from matplotlib import pyplot as plt
#import matplotlib.cm as cm
#from matplotlib.axes._axes import Axes
import torch
import torch.distributions as D
#from torch.func import vmap, jacrev
#from tqdm import tqdm
#import seaborn as sns
#from sklearn.datasets import make_moons, make_circles
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ### Defining the flow model classes

# - **Alpha** and **beta** classes

# We use $\alpha_t$ and $\beta_t$ to schedule the noise in the Gaussian probability paths.
# 
# These are two continuously differentiable, monotonic functions that follow:
# 
# $\alpha_0$ = $\beta_1$ = 0     and     $\alpha_1$ = $\beta_0$ = 1.
# 
# We have chosen a linear form with a simple derivative:
# 
# $\alpha_t$ = t ; $\alpha_t$' = 1
# 
# $\beta_t$ = 1 - t ; $\beta_t$' = -1
# 


# We want to go with Gaussian probability path, therefore we need to load functions for alpha and beta
class LinearAlpha():
    """Implements alpha_t = t"""

    def __call__(self, t: torch.Tensor) -> torch.Tensor:
        return t  # linear in time

    def dt(self, t: torch.Tensor) -> torch.Tensor:
        return torch.ones_like(t)  # derivative of t is 1


class LinearBeta():
    """Implements beta_t = 1 - t"""

    def __call__(self, t: torch.Tensor) -> torch.Tensor:
        return 1 - t

    def dt(self, t: torch.Tensor) -> torch.Tensor:
        return -torch.ones_like(t)  # derivative of 1 - t is -1



# - **Gaussian probability path** class



class GaussianConditionalProbabilityPath():
    def __init__(self, alpha, beta):
        self.alpha = alpha
        self.beta = beta

    def sample_conditional_path(self, z: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Samples from the conditional distribution p_t(x|z) = N(alpha_t * z, beta_t**2 * I_d)
        Args:
            - z: conditioning variable (num_samples, dim)
            - t: time (num_samples, 1)
        Returns:
            - x: samples from p_t(x|z), (num_samples, dim)
        """
        return self.alpha(t) * z + self.beta(t) * torch.randn_like(z)

    def conditional_vector_field(self, x: torch.Tensor, z: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Evaluates the conditional vector field u_t(x|z)
        Note: Only defined on t in [0,1)
        Args:
            - x: position variable (num_samples, dim)
            - z: conditioning variable (num_samples, dim)
            - t: time (num_samples, 1)
        Returns:
            - conditional_vector_field: conditional vector field (num_samples, dim)
        """ 
        alpha_t = self.alpha(t) # (num_samples, 1)
        beta_t = self.beta(t) # (num_samples, 1)
        dt_alpha_t = self.alpha.dt(t) # (num_samples, 1)
        dt_beta_t = self.beta.dt(t) # (num_samples, 1)

        return (dt_alpha_t - dt_beta_t / beta_t * alpha_t) * z + dt_beta_t / beta_t * x

    def conditional_score(self, x: torch.Tensor, z: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Evaluates the conditional score of p_t(x|z) = N(alpha_t * z, beta_t**2 * I_d)
        Note: Only defined on t in [0,1)
        Args:
            - x: position variable (num_samples, dim)
            - z: conditioning variable (num_samples, dim)
            - t: time (num_samples, 1)
        Returns:
        - conditional_score: conditional score (num_samples, dim)
        """ 
        alpha_t = self.alpha(t)
        beta_t = self.beta(t)
        return (z * alpha_t - x) / beta_t ** 2


# - Solver for sampling

class ODEFunc(nn.Module):
    def __init__(self, vf_model, z):
        super().__init__()
        self.vf_model = vf_model
        self.z = z # fixed conditioning 

    def forward(self, t, x): 
        batch_size = x.shape[0] 
     
        # Expand conditioning z to match batch 
        if self.z.shape[0] == 1: 
            z = self.z.expand(batch_size, -1) 
        else: 
            z = self.z 
     
        # Expand t to batch dimension for concatenation 
        if t.dim() == 0: 
            t_batch = t.expand(batch_size, 1)  # (batch, 1) 
        else: 
            t_batch = t.view(batch_size, -1)   # (batch, 1) if 1D 
     
        return self.vf_model(x, z, t_batch) 
      

# - **Time** embedder


class TimeEmbedder(nn.Module):
    def __init__(self, embed_dim=64, max_freq=1e4):
        super().__init__()
        self.embed_dim = embed_dim
        self.max_freq = max_freq
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim*2),
            nn.SiLU(),
            nn.Linear(embed_dim*2, embed_dim*2),
            nn.SiLU(),
            nn.Linear(embed_dim*2, embed_dim)
        )

    def forward(self, t):
        freqs = torch.exp(torch.linspace(0, math.log(self.max_freq), self.embed_dim // 2, device=t.device))
        args = t * freqs
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        return self.mlp(emb)


# - **ResNetBlock** class

# In[7]:


class ResNetBlock(nn.Module):
    def __init__(self, dim, hidden_dim=None):
        super().__init__()
        hidden_dim = hidden_dim or dim*2
        self.block = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, dim),
            nn.LayerNorm(dim)
        )

    def forward(self, x):
        return x + self.block(x)


# - **Neural vector field** class


class NeuralVectorField(nn.Module):
    def __init__(self, latent_dim, hidden_dim=256, n_resblocks=5, time_embed_dim=64):
        super().__init__()
        self.x_proj = nn.Linear(latent_dim, hidden_dim)
        self.z_proj = nn.Linear(latent_dim, hidden_dim)
        self.time_embedder = TimeEmbedder(time_embed_dim)

        self.resblocks = nn.ModuleList([
            ResNetBlock(hidden_dim*2 + time_embed_dim, hidden_dim*2) for _ in range(n_resblocks)
        ])
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_dim*2 + time_embed_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, latent_dim)
        )

    def forward(self, x, z, t):
        xh = self.x_proj(x)
        zh = self.z_proj(z)
        th = self.time_embedder(t)
        h = torch.cat([xh, zh, th], dim=-1)
        for block in self.resblocks:
            h = block(h)
        return self.output_layer(h)


# - Best **learned vector field**


# we want to save the best vector field:
class LearnedVectorFieldODE():
    def __init__(self, vf_model):
        self.vf_model = vf_model

    def drift_coefficient(self, x: torch.Tensor, t: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        # x, z: (batch_size, latent_dim)
        # t: (batch_size, 1)
        return self.vf_model(x, z, t)

