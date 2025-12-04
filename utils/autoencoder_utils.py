# Autoencoder script but not for training just for using to encode and decode

import anndata as ad
import torch
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from typing import List, Optional, Callable
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import beta


# -------------------------------
# Define MLP (like the one in CFGEN)
# -------------------------------

class MLP(nn.Module):
    def __init__(self, 
                 dims: List[int],
                 batch_norm: bool, 
                 dropout: bool, 
                 dropout_p: float, 
                 activation: Optional[Callable] = nn.ELU, 
                 final_activation: Optional[str] = None):
        super().__init__()
        self.dims = dims
        self.batch_norm = batch_norm
        self.activation = activation
        layers = []
        for i in range(len(dims[:-2])):
            block = [nn.Linear(dims[i], dims[i+1])]
            if batch_norm:
                block.append(nn.BatchNorm1d(dims[i+1]))
            block.append(activation())
            if dropout:
                block.append(nn.Dropout(dropout_p))
            layers.append(nn.Sequential(*block))
        layers.append(nn.Linear(dims[-2], dims[-1]))
        self.net = nn.Sequential(*layers)
        if final_activation == "tanh":
            self.final_activation = nn.Tanh()
        elif final_activation == "sigmoid":
            self.final_activation = nn.Sigmoid()
        else:
            self.final_activation = None

    def forward(self, x):
        x = self.net(x)
        return x if self.final_activation is None else self.final_activation(x)

    def forward(self, x):
        x = self.net(x)
        return x if self.final_activation is None else self.final_activation(x)

# -------------------------------
# NB Autoencoder
# -------------------------------
# Here we incorporate our MLP into a bigger class, so that we can train an autoencoder. We train it independetly of the 
# flow matching model.
class NB_Autoencoder(nn.Module):
    def __init__(self,
                 num_features: int,
                 latent_dim: int = 50,
                 hidden_dims: List[int] = [512, 256],
                 dropout_p: float = 0.1,
                 l2_reg: float = 1e-5,
                 kl_reg: float = 0):
        super().__init__()
        self.num_features = num_features
        self.latent_dim = latent_dim
        self.l2_reg = l2_reg
        self.kl_reg = kl_reg

        self.hidden_encoder = MLP(
        dims=[num_features, *hidden_dims, latent_dim],
        batch_norm=True,
        dropout=False,
        dropout_p=dropout_p
        )
        #self.latent_layer = nn.Linear(hidden_dims[-1], latent_dim)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)
        self.decoder = MLP(
            dims=[latent_dim, *hidden_dims[::-1], num_features],
            batch_norm=True,
            dropout=False,
            dropout_p=dropout_p
        )

        #self.log_theta = nn.Parameter(torch.randn(num_features) * 0.01)
        self.theta = torch.nn.Parameter(torch.randn(num_features), requires_grad=True)
    def forward(self, x, library_size = None):
        """ forward function that encodes and decodes"""
        z = self.hidden_encoder(x["X_norm"])
        
        #z = self.latent_layer(h)
        # Raw decoded logits
        logits = self.decoder(z)  
        
        # Softmax over genes → normalized probabilities
        gene_probs = F.softmax(logits, dim=1)

        if library_size is None:
            # Use average library size 1.0 if not provided
            # Sample size factors from your custom distribution
            lib = size_factor_distribution(adata, z.size(0))   # returns numpy array or list

            # Convert to torch tensor, match shape, move to correct device
            library_size = torch.tensor(lib, dtype=torch.float32, device=z.device).unsqueeze(1)

            #library_size = torch.ones(z.size(0), 1, device=z.device)
        # Library size of each cell (sum of counts)
        #library_size = x["X"].sum(1).unsqueeze(1).to(self.device)  
        
        # Scale probabilities by library size → mean parameter μ
        mu = gene_probs * library_size
 

        #theta = torch.exp(self.log_theta).unsqueeze(0).expand_as(mu)
        return {"z": z, "mu": mu, "theta": self.theta}

    def encode(self, x):
        """ decoding function"""
        z = self.hidden_encoder(x)
        return z

    def size_factor_distribution(self, adata_train, n_samples):
        """ same function as in block before """
        min_val = adata_train.obs['n_counts'].min() 
        max_val = adata_train.obs['n_counts'].max()
        mean = adata_train.obs['n_counts'].mean()
        std = adata_train.obs['n_counts'].std()
        
        # Beta distribution sampling
        m = (mean - min_val) / (max_val - min_val)
        v = (std**2) / ((max_val - min_val)**2)
        temp = m*(1-m)/v - 1
        a_beta = m * temp
        b_beta = (1-m) * temp
        
        samples_beta = beta.rvs(a_beta, b_beta, size=n_samples)
        samples_beta = samples_beta * (max_val - min_val) + min_val
    
        return samples_beta

    
        
    def decode(self, z, library_size=None):
        """
        Decode latent vectors z to NB parameters mu, theta.
        z: (batch, latent_dim)
        library_size: (batch, 1) sum of counts per cell; if None, use 1.0
        """
        logits = self.decoder(z)  # (batch, num_genes)
        gene_probs = F.softmax(logits, dim=1)  # softmax over genes
        # if library size isnt specified:
        if library_size is None:
            # Sample size factors from your custom distribution
            lib = size_factor_distribution(adata, z.size(0))   # returns numpy array or list

            # Convert to torch tensor, match shape, move to correct device
            library_size = torch.tensor(lib, dtype=torch.float32, device=z.device).unsqueeze(1)

            #library_size = torch.ones(z.size(0), 1, device=z.device)
    
        mu = gene_probs * library_size  # scale by library size
        #theta = torch.exp(self.log_theta).unsqueeze(0).expand_as(mu)
        return {"mu": mu, "theta": self.theta}
   

    def loss_function(self, x, outputs):
        """
        Compute loss using scvi NegativeBinomial.
        """
        mu = outputs["mu"]          # (batch, n_genes)
        theta = outputs["theta"]    # (batch, n_genes)
        z = outputs["z"]            # latent
    
        # scvi NegativeBinomial expects mu and theta
        nb_dist = NegativeBinomial(mu=mu, theta=torch.exp(self.theta))
        nll = -nb_dist.log_prob(x).sum(dim=1).mean()  # mean over batch
        
        # Optional regularization
        l2_loss = sum((p**2).sum() for p in self.parameters()) * self.l2_reg
        kl_loss = (z**2).mean() * self.kl_reg
    
        loss = nll + l2_loss + kl_loss
        return {"loss": loss, "nll": nll}

# function to for dataloader 
# dataloader for the 

import torch
from torch.utils.data import Dataset, DataLoader

class CountsDataset(Dataset):
    def __init__(self, X, y=None):
        """
        X: raw counts tensor (num_cells, num_genes)
        y: optional labels tensor (num_cells,)
        """
        if hasattr(X, "toarray"):
            X = X.toarray()
        self.X = torch.tensor(X, dtype=torch.float32)
        self.X_norm = torch.log1p(self.X)  # log1p = log(1 + x)
        self.y = torch.tensor(y, dtype=torch.long) if y is not None else None
        self.n_samples = self.X.shape[0]

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        sample = dict(
            X=self.X[idx],
            X_norm=self.X_norm[idx]
        )
        if self.y is not None:
            sample["y"] = self.y[idx]
        return sample



#Function to sample count data from generated distribution
# we dont sample anymore
