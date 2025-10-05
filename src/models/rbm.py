# src/models/rbm.py
"""
Lightweight RBM implementation (contrastive divergence)
This implementation is simple and educational; for production prefer optimized libraries.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class RBM(nn.Module):
    def __init__(self, n_vis, n_hid, k=1):
        super().__init__()
        self.n_vis = n_vis
        self.n_hid = n_hid
        self.k = k
        self.W = nn.Parameter(torch.randn(n_hid, n_vis) * 0.01)
        self.h_bias = nn.Parameter(torch.zeros(n_hid))
        self.v_bias = nn.Parameter(torch.zeros(n_vis))

    def sample_h(self, v):
        prob = torch.sigmoid(F.linear(v, self.W, self.h_bias))

        return prob, torch.bernoulli(prob)

    def sample_v(self, h):
       prob = torch.sigmoid(F.linear(h, self.W.t(), self.v_bias))
       return prob, torch.bernoulli(prob)

    def forward(self, v):
        vk = v
        for _ in range(self.k):
            ph, hk = self.sample_h(vk)
            pv, vk = self.sample_v(hk)
        return vk.detach()

    def free_energy(self, v):
        vbias_term = v.mv(self.v_bias)
        wx_b = F.linear(v, self.W, self.h_bias) 
        hidden_term = torch.sum(torch.log1p(torch.exp(wx_b)), dim=1)
        return -vbias_term - hidden_term
