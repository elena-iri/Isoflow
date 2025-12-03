#!/usr/bin/env python
# coding: utf-8

# In[1]:


# this the cleaner version of the flow matching model
# import all packages and data
# the data comes from the encoder in 50 dimensional format
from abc import ABC, abstractmethod
from typing import Optional, List, Type, Tuple, Dict
import math
import anndata as ad
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.cm as cm
from matplotlib.axes._axes import Axes
import torch
import torch.distributions as D
from torch.func import vmap, jacrev
from tqdm import tqdm
import seaborn as sns
from sklearn.datasets import make_moons, make_circles
import torch.nn.functional as F
device = torch.device('cpu') #'cuda' if torch.cuda.is_available() else 'cpu')


# In[3]:


# Load the encoded data from the autoencoder
input_file_path = "/dtu/blackhole/1e/213566/data/datasets/pbmc3k/pbmc3k_train_with_latent.h5ad"
adata = ad.read_h5ad(input_file_path)

# Access latent representation
latent = adata.obsm["X_latent"]
# make it to a tensor and save in GPU
latent_tensor = torch.tensor(latent, dtype=torch.float32, device = device)
print("Shape of latent space:", latent.shape)
print(latent)


# In[4]:


n_hidden_dim = 256 #128
embedding_dim = 64 #32
max_freq_time= 1e3 #1e4


# In[5]:


# This is a way of encoding our data for empirical data
class EmpiricalDistribution(torch.nn.Module):
    def __init__(
        self,
        data: torch.Tensor,
        bandwidth: Optional[float] = None,
        compute_log_density: bool = True,
    ):
        super().__init__()
        assert data.dim() == 2, "data must be shape (N, D)"
        data = data.contiguous()

        self.register_buffer("data", data)   # (N, D)
        self.n = data.shape[0]
        self.data_dim = data.shape[1]        # <-- renamed attribute
        self.compute_log_density_flag = compute_log_density

        # Bandwidth estimation
        if bandwidth is None:
            std = torch.std(data, dim=0).mean().item()
            factor = (4.0 / (self.data_dim + 2.0)) ** (1.0 / (self.data_dim + 4.0))
            bw = factor * (self.n ** (-1.0 / (self.data_dim + 4.0))) * (std + 1e-6)
            self.bandwidth = torch.tensor(float(bw), device=self.data.device)
        else:
            self.bandwidth = torch.tensor(float(bandwidth), device=self.data.device)

        self._log_const = -0.5 * self.data_dim * math.log(2.0 * math.pi) - self.data_dim * torch.log(self.bandwidth).item()

    @property
    def dim(self):
        return self.data_dim
    def sample(self, num_samples: int) -> torch.Tensor:
        idx = torch.randint(0, self.n, (num_samples,), device=self.data.device)
        return self.data[idx]

    def log_density(self, x: torch.Tensor) -> torch.Tensor:
        if not self.compute_log_density_flag:
            raise RuntimeError("log_density disabled (compute_log_density=False).")

        assert x.dim() == 2 and x.shape[1] == self.data_dim

        x = x.to(self.data.device)
        x_norm2 = (x ** 2).sum(dim=1, keepdim=True)
        data_norm2 = (self.data ** 2).sum(dim=1).unsqueeze(0)
        cross = x @ self.data.t()
        d2 = x_norm2 + data_norm2 - 2.0 * cross

        sigma2 = (self.bandwidth ** 2).item()
        exponents = -0.5 * d2 / (sigma2 + 1e-12)
        lse = torch.logsumexp(exponents, dim=1, keepdim=True)

        log_prob = math.log(1.0 / self.n) + lse + self._log_const
        return log_prob


# In[6]:


# lets test if the empirical distribution class actually works
# the data has to be a torch tensor

dist = EmpiricalDistribution(latent_tensor)
samples = dist.sample(3)
logp = dist.log_density(samples)
print(logp)

# it seems to work


# In[7]:


# we have to have a class that can draw from a Gaussian distribution

class Gaussian(torch.nn.Module):
    """
    Multivariate Gaussian distribution
    """
    def __init__(self, mean: torch.Tensor, cov: torch.Tensor):
        """
        mean: shape (dim,)
        cov: shape (dim,dim)
        """
        super().__init__()
        self.register_buffer("mean", mean)
        self.register_buffer("cov", cov)

    @property
    def dim(self) -> int:
        return self.mean.shape[0]

    @property
    def distribution(self):
        return D.MultivariateNormal(self.mean, self.cov, validate_args=False)

    def sample(self, num_samples) -> torch.Tensor:
        return self.distribution.sample((num_samples,))

    def log_density(self, x: torch.Tensor):
        return self.distribution.log_prob(x).view(-1, 1)

    @classmethod
    def isotropic(cls, dim: int, std: float) -> "Gaussian":
        mean = torch.zeros(dim)
        cov = torch.eye(dim) * std ** 2
        return cls(mean, cov)


# In[8]:


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



# In[9]:


class GaussianConditionalProbabilityPath():
    def __init__(self, p_data, alpha, beta):
        self.p_data = p_data 
        p_simple = Gaussian.isotropic(p_data.dim, 1.0)
        self.alpha = alpha
        self.beta = beta

    def sample_conditioning_variable(self, num_samples: int) -> torch.Tensor:
        """
        Samples the conditioning variable z ~ p_data(x)
        Args:
            - num_samples: the number of samples
        Returns:
            - z: samples from p(z), (num_samples, dim)
        """
        return self.p_data.sample(num_samples)

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


# In[10]:


emp_dist = dist
alpha = LinearAlpha()
beta = LinearBeta()
path = GaussianConditionalProbabilityPath(
    p_data=emp_dist,
    alpha=alpha,
    beta=beta
)
print(path)


# In[11]:


# now that we were able to construct a Gaussian probability path, we have to be able to make a conditional vector field

class ConditionalVectorFieldODE():
    def __init__(self, path, z: torch.Tensor):
        """
        Args:
        - path: the ConditionalProbabilityPath object to which this vector field corresponds
        - z: the conditioning variable, (1, dim)
        """
        super().__init__()
        self.path = path
        self.z = z

    def drift_coefficient(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Returns the conditional vector field u_t(x|z)
        Args:
            - x: state at time t, shape (bs, dim)
            - t: time, shape (bs,.)
        Returns:
            - u_t(x|z): shape (batch_size, dim)
        """
        bs = x.shape[0]
        z = self.z.expand(bs, *self.z.shape[1:])
        return self.path.conditional_vector_field(x,z,t)


# In[12]:


# now we somehow want to model the marginal vector field from the conditonal vector field
# for that we will use eulers:
class EulerSimulator():
    def __init__(self, ode, z: torch.Tensor):
        self.ode = ode
        self.z = z

    def step(self, xt: torch.Tensor, t: torch.Tensor, h: float):

        # Expand z to match batch size
        if self.z.shape[0] == 1:
            z_exp = self.z.expand(xt.shape[0], -1)
        else:
            z_exp = self.z
        dx = self.ode.drift_coefficient(xt, t, z_exp)
        return xt + dx * h



# In[13]:


import math
import torch.nn as nn

class TimeEmbedder(nn.Module):
    def __init__(self, embed_dim=embedding_dim, max_freq=max_freq_time):
        super().__init__()
        self.embed_dim = embed_dim
        self.max_freq = max_freq
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.SiLU(),
            nn.Linear(embed_dim, embed_dim),
            nn.SiLU()
        )

    def forward(self, t):
        freqs = torch.exp(torch.linspace(0, math.log(self.max_freq), self.embed_dim // 2, device=t.device))
        args = t * freqs
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        return self.mlp(emb)

class ResNetBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim)
        )

    def forward(self, x):
        return x + self.block(x)

class NeuralVectorField(nn.Module):
    def __init__(self, latent_dim, hidden_dim=n_hidden_dim, n_resblocks=3, time_embed_dim=embedding_dim):
        super().__init__()
        self.x_proj = nn.Linear(latent_dim, hidden_dim)
        self.z_proj = nn.Linear(latent_dim, hidden_dim)
        self.time_embedder = TimeEmbedder(time_embed_dim)

        self.resblocks = nn.ModuleList([
            ResNetBlock(hidden_dim*2 + time_embed_dim) for _ in range(n_resblocks)
        ])
        self.output_layer = nn.Linear(hidden_dim*2 + time_embed_dim, latent_dim)

    def forward(self, x, z, t):
        xh = self.x_proj(x)
        zh = self.z_proj(z)
        th = self.time_embedder(t)
        h = torch.cat([xh, zh, th], dim=-1)
        for block in self.resblocks:
            h = block(h)
        return self.output_layer(h)



# In[14]:


device = "cpu" #"cuda" if torch.cuda.is_available() else "cpu"

batch_size = 500
num_epochs = 3000
learning_rate = 1e-4
latent_dim = latent_tensor.shape[1]  # e.g., 50

epoch_list=[]
loss_list=[]

vf_model = NeuralVectorField(latent_dim=latent_dim).to(device)
optimizer = torch.optim.AdamW(vf_model.parameters(), lr=learning_rate)

# Initialize GaussianConditionalProbabilityPath and ConditionalVectorFieldODE
path = GaussianConditionalProbabilityPath(emp_dist, alpha, beta)  # define alpha, beta
#cvf_ode = ConditionalVectorFieldODE(path, z=torch.zeros(1, latent_dim, device=device))

for epoch in range(num_epochs):
    # --- Sample conditioning variable z ---
    z = emp_dist.sample(batch_size).to(device)
    # lets try sampling from real points
    #indices = torch.randint(0, latent_tensor.shape[0], (batch_size,))
    #z = latent_tensor[indices].to(device)
    # --- Sample time ---
    t = torch.rand(batch_size, 1, device=device)

    # --- Sample x_t from conditional path ---
    with torch.no_grad():
        x = path.sample_conditional_path(z, t)
        u_target = path.conditional_vector_field(x, z, t)

    # --- Normalize target ---
    u_mean = u_target.mean(dim=0, keepdim=True)
    u_std = u_target.std(dim=0, keepdim=True) + 1e-6
    u_target_norm = (u_target - u_mean) / u_std

    # --- Forward pass ---
    v_pred = vf_model(x, z, t)

    # --- Loss ---
    loss = F.mse_loss(v_pred, u_target_norm)

    # --- Backprop ---
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(vf_model.parameters(), max_norm=1.0)
    optimizer.step()

    epoch_list.append(epoch)
    loss_list.append(loss)

    if epoch % 50 == 0:
        print(f"[{epoch}] Loss: {loss.item():.6f}")


# In[15]:


loss_list=[i.item() for i in loss_list]
plt.plot(epoch_list, loss_list)
plt.title('Training of flow')
plt.xlabel("Epoch")
plt.ylabel("Average loss")
plt.show()


# In[16]:


torch.save(vf_model.state_dict(), "vf_model_weights.pt")


# In[17]:


# we want to save the best vector field:
class LearnedVectorFieldODE():
    def __init__(self, vf_model):
        self.vf_model = vf_model

    def drift_coefficient(self, x: torch.Tensor, t: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        # x, z: (batch_size, latent_dim)
        # t: (batch_size, 1)
        return self.vf_model(x, z, t)


# In[18]:


# Wrap the trained neural network
learned_ode = LearnedVectorFieldODE(vf_model)

# Save the wrapper
torch.save(learned_ode, "learned_ode.pt")


# In[19]:


#device = "cuda" if torch.cuda.is_available() else "cpu"

# Number of samples and latent dimension
n_samples = 2110
latent_dim = latent_tensor.shape[1]

# Starting points (pure noise)
x = torch.randn(n_samples, latent_dim, device=device)

# Unconditional generation: use a single zero vector as z
z = torch.zeros(1, latent_dim, device=device)

# Wrap the trained neural network as an ODE
learned_ode = LearnedVectorFieldODE(vf_model)

# Euler simulator with unconditional z
simulator = EulerSimulator(learned_ode, z)

# Simulation parameters
t0, t1 = 0.0, 1.0
n_steps = 500
dt = (t1 - t0) / n_steps

# Store trajectory (optional, can skip intermediate steps if memory is tight)
trajectory = [x.clone()]
t = torch.full((n_samples, 1), t0, device=device)

# Euler integration
for _ in range(n_steps):
    x = simulator.step(x, t, dt)
    trajectory.append(x.clone())
    t = t + dt

# Final generated samples
generated_cells = trajectory[-1]
print(generated_cells.shape)  # (n_samples, latent_dim)

# Save the generated latent space
torch.save(generated_cells, "generated_latent.pt")
print(generated_cells)


# In[ ]:





# In[ ]:




