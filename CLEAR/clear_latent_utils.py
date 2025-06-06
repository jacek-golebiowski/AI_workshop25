import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import sklearn.linear_model as lm

class Encoder(nn.Sequential):
    def __init__(self, latent_dim=64):
        super().__init__(
            nn.Conv2d(1, 16, 4, 2, 1),   # 0
            nn.ReLU(),                    # 1
            nn.Conv2d(16, 32, 4, 2, 1),  # 2
            nn.ReLU(),                    # 3
            nn.Flatten(),                 # 4
            nn.Linear(32*32*32, latent_dim)  # 5
        )

class Decoder(nn.Module):
    def __init__(self, latent_dim=64):
        super().__init__()
        self.dec_lin = nn.Linear(latent_dim, 32*32*32)
        self.dec     = nn.Sequential(
            nn.ConvTranspose2d(32, 16, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, 4, 2, 1),
            nn.Sigmoid()
        )

    def forward(self, z):
        x = self.dec_lin(z).view(-1, 32, 32, 32)
        return self.dec(x)


def collect_latent_samples(img, encoder, decoder, anomaly_wrapper,
                           n_samples=500, sigma=0.1, device='cpu'):
    encoder.eval(); decoder.eval()
    with torch.no_grad():
        z0 = encoder(img.unsqueeze(0).to(device)).cpu().numpy().reshape(-1)
    D = z0.shape[0]
    Z = z0 + sigma * np.random.randn(n_samples, D)
    P = np.zeros(n_samples, dtype=float)
    for i in range(n_samples):
        z = torch.from_numpy(Z[i]).float().to(device).unsqueeze(0)
        with torch.no_grad():
            x_rec = decoder(z)
            logits = anomaly_wrapper(x_rec)
            probs = F.softmax(logits, dim=1).cpu().numpy()
        P[i] = probs[0,1]
    return z0, Z, P

def clear_on_latent(z0, Z, P, boundary=0.5, target_class=0):
    """
    Z: [n_samples, D], P: [n_samples]
    target_class: 0 (good) or 1 (anomaly)
    Returns: {dim_i: delta} – minimal perturbations in latent space to reach target_class
    """
    y = (P >= boundary).astype(int)
    unique = np.unique(y)
    if unique.size < 2:
        print(f"⚠️ W lokalnej próbce tylko klasa={unique[0]}; nie można trenować logistycznej regresji.")
        return {}

    clf = lm.LogisticRegression().fit(Z, y)
    w, b = clf.coef_[0], clf.intercept_[0]

    sign = 1 if target_class == 1 else -1
    deltas = {}
    for i in range(Z.shape[1]):
        if w[i] == 0:
            continue
        deltas[i] = float(-sign * (np.dot(w, z0) + b) / w[i])
    return deltas

def decode_counterfactual(z0, deltas, decoder, dims, device='cpu', scale=1.0):
    z_cf = z0.copy()
    for d in dims:
        z_cf[d] += deltas[d] * scale
    z_cf = torch.from_numpy(z_cf).float().to(device).unsqueeze(0)
    with torch.no_grad():
        img_cf = decoder(z_cf).cpu().squeeze(0)
    return img_cf