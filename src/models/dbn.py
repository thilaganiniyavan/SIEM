# src/models/dbn.py
"""
Stack RBMs into a DBN-like encoder and build classifier.
We implement greedy layer-wise RBM pretraining, then initialize a feedforward network and fine-tune.
"""

import torch
import torch.nn as nn
from .rbm import RBM
import torch.nn.functional as F

class DBNEncoder(nn.Module):
    def __init__(self, sizes):
        super().__init__()
        layers = []
        for i in range(len(sizes)-1):
            layers.append(nn.Linear(sizes[i], sizes[i+1]))
            layers.append(nn.ReLU())
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

class DBNClassifier(nn.Module):
    def __init__(self, sizes):
        super().__init__()
        self.encoder = DBNEncoder(sizes)
        self.classifier = nn.Linear(sizes[-1], 1)

    def forward(self, x):
        z = self.encoder(x)
        return torch.sigmoid(self.classifier(z))

# Pretraining function (greedy)
def pretrain_rbms(X_tensor, layer_sizes, rbm_epochs=10, lr=0.01):
    """
    X_tensor: torch.Tensor of shape (N, D)
    layer_sizes: list e.g., [D, 64, 32]
    returns: list of trained RBMs
    """
    rbms = []
    input_data = X_tensor
    for i in range(len(layer_sizes)-1):
        rbm = RBM(n_vis=layer_sizes[i], n_hid=layer_sizes[i+1], k=1)
        opt = torch.optim.SGD(rbm.parameters(), lr=lr)
        for epoch in range(rbm_epochs):
            opt.zero_grad()
            vk = rbm(input_data)
            # approximate gradient via free energy difference
            loss = torch.mean(rbm.free_energy(input_data)) - torch.mean(rbm.free_energy(vk))
            loss.backward()
            opt.step()
        # get hidden activations as next input
        ph, _ = rbm.sample_h(input_data)
        input_data = ph.detach()
        rbms.append(rbm)
    return rbms

def transfer_weights(dbn_model, rbms):
    """
    Transfer RBM weights into DBN feedforward layers
    """
    ff_layers = [l for l in dbn_model.encoder.net if isinstance(l, nn.Linear)]
    for i, rbm in enumerate(rbms):
        W = rbm.W.data
        b_h = rbm.h_bias.data
        # rbm.W shape: (n_hid, n_vis) ; need to transpose for Linear(in=n_vis,out=n_hid)
        ff_layers[i].weight.data = W
        ff_layers[i].bias.data = b_h
