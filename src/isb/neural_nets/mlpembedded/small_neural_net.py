"""Neural network classes."""

import torch
import torch.nn.functional as F
import math


class DSPSmall(torch.nn.Module):

    def __init__(self, input_dim, device='cpu', sigmoid=False):
        super(DSPSmall, self).__init__()
        self.input_dim = input_dim
        self.sigmoid = sigmoid
        self.embedded_dim = 16
        if not isinstance(input_dim, int):
            input_dim = input_dim[0]
        self.mlp_1a = torch.nn.Sequential(torch.nn.Linear(input_dim, 16),
                                           torch.nn.LeakyReLU(),
                                           torch.nn.Linear(16, 32)).to(device).double()
        self.mlp_1b = torch.nn.Sequential(torch.nn.Linear(16, 16),
                                           torch.nn.LeakyReLU(),
                                           torch.nn.Linear(16, 32)).to(device).double()
        self.mlp_2 = torch.nn.Sequential(torch.nn.Linear(64, 128),
                                      torch.nn.LeakyReLU(),
                                      torch.nn.Linear(128, 128),
                                      torch.nn.LeakyReLU(),
                                      torch.nn.Linear(128, input_dim)).to(device).double()
        self.device = device

    def get_timestep_embedding(self, timesteps, embedding_dim=128):
        """
          Embedding from the Scrödinger bridge codebase.
          From Fairseq.
          Build sinusoidal embeddings.
          This matches the implementation in tensor2tensor, but differs slightly
          from the description in Section 3.5 of "Attention Is All You Need".
          https://github.com/pytorch/fairseq/blob/master/fairseq/modules/sinusoidal_positional_embedding.py
        """
        half_dim = embedding_dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float64, device=timesteps.device) * -emb)
        emb = emb.unsqueeze(0)
        emb = timesteps.float() * emb
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        if embedding_dim % 2 == 1:  # zero pad
            emb = F.pad(emb, [0,1])

        return emb

    def forward(self, x, t):
        if x.ndim == 1:
            x = x.unsqueeze(0)
        if t.ndim == 1:
            t = t.unsqueeze(1)
        t_embedded = self.get_timestep_embedding(t, self.embedded_dim)
        t_2 = self.mlp_1b.forward(t_embedded)
        x_1 = self.mlp_1a.forward(x)
        x_t = torch.cat([x_1, t_2], axis=-1)
        y = self.mlp_2.forward(x_t)
        if self.sigmoid:
            y = torch.nn.Sigmoid()(y)
        return y
