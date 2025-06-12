import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
        self.pool  = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.fc1   = nn.Linear(32 * 32 * 32, 64)
        self.dropout = nn.Dropout(0.5)
        self.fc2   = nn.Linear(64, 2)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 32 * 32 * 32)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)


class PytorchMVTECWrapper:
    backend = 'PYT'
    model_type = 'classifier'

    def __init__(self, model_path, pca=None, device='cpu'):
        self.device = device
        self.pca = pca  # <-- store PCA
        self.net = SimpleCNN().to(device)
        self.net.load_state_dict(torch.load(model_path, map_location=device))
        self.net.eval()

    def load_model(self):
        pass

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        logits = self.__call__(X)
        probs = torch.softmax(logits, dim=1)
        return probs.cpu().numpy()

    def __call__(self, X):
        if isinstance(X, np.ndarray):
            t = torch.from_numpy(X).float().to(self.device)
        else:
            t = X.float().to(self.device)

        if self.pca is not None:
            # PCA inverse transform before feeding model
            t_np = t.cpu().numpy()
            t_np_full = self.pca.inverse_transform(t_np)
            t = torch.from_numpy(t_np_full).float().to(self.device)

        if t.ndim == 2:
            n, f = t.shape
            side = int(f**0.5)
            t = t.view(n, 1, side, side)

        with torch.no_grad():
            out = self.net(t)
        return out
