import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from mvtec_dataset import MVTecAnomalyDataset
from model import SimpleCNN
from random import shuffle

from dotenv import load_dotenv
import os

load_dotenv()
dataset_name = os.getenv("DATASET_NAME")

if dataset_name is None:
    raise ValueError("❌ DATASET_NAME nie jest ustawione w .env!")

class BalancedDataset(Dataset):
    def __init__(self, data):
        self.data = data
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        return self.data[idx]

def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
    ])

    data_root = f'../datasets/mvtec/{dataset_name}'

    train_ds = MVTecAnomalyDataset(data_root, phase='train', transform=transform)
    good = [(x, y) for x, y in train_ds if y == 0]

    test_ds = MVTecAnomalyDataset(data_root, phase='test', transform=transform)
    anom = [(x, y) for x, y in test_ds if y == 1]

    shuffle(good)
    shuffle(anom)
    used_good = good[:len(anom)]
    balanced_samples = used_good + anom
    shuffle(balanced_samples)

    print(f"Using {len(used_good)} good and {len(anom)} anomaly samples")

    train_ds = BalancedDataset(balanced_samples)
    loader = DataLoader(train_ds, batch_size=32, shuffle=True)

    model = SimpleCNN().to(device)
    opt = optim.Adam(model.parameters(), lr=1e-3)
    crit = nn.CrossEntropyLoss()

    for epoch in range(1, 71):
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

        avg = running_loss / len(loader)
        print(f"Epoch {epoch:02d} | loss: {avg:.4f}")

    torch.save(model.state_dict(), f'pthFiles/{dataset_name}_cnn.pth')
    print(f"✅ Zapisano model: {dataset_name}_cnn.pth")

if __name__ == '__main__':
    train()