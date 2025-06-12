import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
import os
from dotenv import load_dotenv
from mvtec_dataset import MVTecAnomalyDataset
from model import SimpleCNN
import numpy as np
import random

np.random.seed(42)
random.seed(42)
torch.manual_seed(42)

def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    load_dotenv()
    dataset_name = os.getenv("DATASET_NAME")
    if dataset_name is None:
        raise ValueError("DATASET_NAME not set")

    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x + 0.01 * torch.randn_like(x))
    ])

    data_root = f'../datasets/mvtec/{dataset_name}'
    train_ds = MVTecAnomalyDataset(data_root, phase='train', transform=transform)
    loader = DataLoader(train_ds, batch_size=32, shuffle=True)

    model = SimpleCNN().to(device)
    opt = optim.Adam(model.parameters(), lr=5e-4)
    crit = nn.CrossEntropyLoss()

    for epoch in range(1, 10):
        model.train()
        running_loss = 0.0
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            opt.zero_grad()
            out = model(imgs)
            loss = crit(out, labels)
            loss.backward()
            opt.step()
            running_loss += loss.item()

            with torch.no_grad():
                preds = torch.softmax(out, dim=1)
                mean_good = preds[:, 0].mean().item()
                mean_anom = preds[:, 1].mean().item()
                print(f"Batch preds avg: good={mean_good:.4f} anomaly={mean_anom:.4f}")

        avg = running_loss / len(loader)
        print(f"Epoch {epoch:02d} | loss: {avg:.4f}")

    os.makedirs("pthFiles", exist_ok=True)
    model_path = f"pthFiles/{dataset_name}_cnn.pth"
    torch.save(model.state_dict(), model_path)
    print(f"Saved model: {model_path}")

if __name__ == '__main__':
    train()
