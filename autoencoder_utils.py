from typing import List, Optional, Callable
import torch.nn as nn
import torch.nn.functional as F
import torch

# -------------------------------
# Define MLP (like the one in CFGEN)
# -------------------------------
class MLP(nn.Module):
    def __init__(self, 
                 dims: List[int],
                 batch_norm: bool = True, 
                 dropout: bool = True, 
                 dropout_p: float = 0.1, 
                 activation: Optional[Callable] = nn.ELU, 
                 final_activation: Optional[str] = None):
        super().__init__()
        self.dims = dims
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
    
# -------------------------------
# Negative Binomial log-likelihood
# -------------------------------
def negative_binomial_log_likelihood(x, mu, theta, eps=1e-8):
    t1 = torch.lgamma(theta + eps) + torch.lgamma(x + 1.0) - torch.lgamma(x + theta + eps)
    t2 = (theta * (torch.log(theta + eps) - torch.log(mu + theta + eps))) + \
         (x * (torch.log(mu + eps) - torch.log(mu + theta + eps)))
    return t1 + t2

# -------------------------------
# NB Autoencoder
# -------------------------------
class NB_Autoencoder(nn.Module):
    def __init__(self,
                 num_features: int,
                 latent_dim: int = 50,
                 hidden_dims: List[int] = [512, 256],
                 dropout_p: float = 0.1,
                 l2_reg: float = 1e-5,
                 kl_reg: float = 1e-3):
        super().__init__()
        self.num_features = num_features
        self.latent_dim = latent_dim
        self.l2_reg = l2_reg
        self.kl_reg = kl_reg

        self.encoder = MLP(
            dims=[num_features, *hidden_dims, latent_dim],
            batch_norm=True,
            dropout=True,
            dropout_p=dropout_p
        )

        self.decoder = MLP(
            dims=[latent_dim, *hidden_dims[::-1], num_features],
            batch_norm=True,
            dropout=True,
            dropout_p=dropout_p
        )

        self.log_theta = nn.Parameter(torch.randn(num_features) * 0.01)

    def forward(self, x):
        z = self.encoder(x)
        mu = F.softplus(self.decoder(z))
        theta = torch.exp(self.log_theta).unsqueeze(0).expand_as(mu)
        return {"z": z, "mu": mu, "theta": theta}

    def loss_function(self, x, outputs):
        mu = outputs["mu"]
        theta = outputs["theta"]
        z = outputs["z"]
        nll = -negative_binomial_log_likelihood(x, mu, theta).sum(dim=1).mean()
        l2_loss = sum((p**2).sum() for p in self.parameters()) * self.l2_reg
        kl_loss = (z**2).mean() * self.kl_reg
        loss = nll + l2_loss + kl_loss
        return {"loss": loss, "nll": nll, "l2": l2_loss, "kl": kl_loss}