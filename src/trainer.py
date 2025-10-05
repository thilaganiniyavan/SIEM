# src/trainer.py
"""
Orchestrates:
- load features
- scale
- pretrain (RBM or AE)
- fine-tune classifier
- evaluate and save
"""

import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score
from .models.autoencoder import Autoencoder
from .models.dbn import DBNClassifier, pretrain_rbms, transfer_weights
from .utils import set_seed
import os

set_seed(42)

DATA_PATH = "data/features.csv"
MODEL_OUT = "data/dbn_model.pt"
SCALER_OUT = "data/scaler.npy"

def load_features(path=DATA_PATH):
    df = pd.read_csv(path)
    # numeric features (exclude host,label)
    X = df.drop(columns=["host","label"]).values.astype(float)
    y = df["label"].values
    return X, y

def train_with_autoencoder(X_train, y_train, X_val, y_val, epochs_pre=100, epochs_ft=200):
    input_dim = X_train.shape[1]
    ae = Autoencoder(input_dim, hidden1=64, hidden2=32)
    opt = torch.optim.Adam(ae.parameters(), lr=1e-3)
    Xtr_t = torch.tensor(X_train, dtype=torch.float32)
    for epoch in range(epochs_pre):
        opt.zero_grad()
        recon = ae(Xtr_t)
        loss = ((recon - Xtr_t)**2).mean()
        loss.backward()
        opt.step()
    class Classifier(nn.Module):
        def __init__(self, encoder):
            super().__init__()
            self.enc = encoder.enc
            self.fc = nn.Linear(32,1)
        def forward(self,x):
            z = self.enc(x)
            return torch.sigmoid(self.fc(z))
    clf = Classifier(ae)
    opt2 = torch.optim.Adam(clf.parameters(), lr=1e-3)
    bce = nn.BCELoss()
    Xtr_t = torch.tensor(X_train, dtype=torch.float32)
    ytr_t = torch.tensor(y_train.reshape(-1,1), dtype=torch.float32)
    for epoch in range(epochs_ft):
        opt2.zero_grad()
        p = clf(Xtr_t)
        loss = bce(p, ytr_t)
        loss.backward()
        opt2.step()
    with torch.no_grad():
        pval = clf(torch.tensor(X_val, dtype=torch.float32)).numpy().ravel()
    return clf, pval

def train_with_dbn(X_train, y_train, X_val, y_val, layer_sizes=[64, 32], rbm_epochs=20, ft_epochs=300):
    # Fix: prepend input dimension explicitly to layer_sizes for RBM pretraining
    input_dim = X_train.shape[1]
    sizes = [input_dim] + layer_sizes
    Xtr_t = torch.tensor(X_train, dtype=torch.float32)
    rbms = pretrain_rbms(Xtr_t, sizes, rbm_epochs=rbm_epochs)
    dbn = DBNClassifier(sizes)
    transfer_weights(dbn, rbms)
    opt = torch.optim.Adam(dbn.parameters(), lr=1e-3)
    bce = nn.BCELoss()
    Xtr_t = torch.tensor(X_train, dtype=torch.float32)
    ytr_t = torch.tensor(y_train.reshape(-1,1), dtype=torch.float32)
    for epoch in range(ft_epochs):
        opt.zero_grad()
        p = dbn(Xtr_t)
        loss = bce(p, ytr_t)
        loss.backward()
        opt.step()
    with torch.no_grad():
        pval = dbn(torch.tensor(X_val, dtype=torch.float32)).numpy().ravel()
    return dbn, pval

def main(use_dbn=False):
    X, y = load_features()
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    os.makedirs("../data", exist_ok=True)
    np.save(SCALER_OUT, scaler.scale_)
    Xtr, Xte, ytr, yte = train_test_split(Xs, y, test_size=0.25, random_state=42, stratify=y)
    if use_dbn:
        model, preds = train_with_dbn(Xtr, ytr, Xte, yte)
    else:
        model, preds = train_with_autoencoder(Xtr, ytr, Xte, yte)
    auc = roc_auc_score(yte, preds)
    acc = accuracy_score(yte, (preds>0.5).astype(int))
    prec = precision_score(yte, (preds>0.5).astype(int))
    rec = recall_score(yte, (preds>0.5).astype(int))
    print(f"[RESULT] AUC={auc:.3f} ACC={acc:.3f} PREC={prec:.3f} REC={rec:.3f}")
    torch.save(model.state_dict(), MODEL_OUT)
    print(f"[INFO] Model saved to {MODEL_OUT}")

if __name__ == "__main__":
    main(use_dbn=True)  # Set to True for DBN pretraining
    # main(use_dbn=False)  # Use this line to train with Autoencoder
