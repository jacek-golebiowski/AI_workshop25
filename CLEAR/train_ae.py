import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from mvtec_dataset import MVTecAnomalyDataset
from torch.nn.functional import mse_loss

from dotenv import load_dotenv
import os

load_dotenv()
dataset_name = os.getenv("DATASET_NAME")

if dataset_name is None:
    raise ValueError("❌ DATASET_NAME nie jest ustawione w .env!")

class AE(nn.Module):
    def __init__(self, latent_dim=64):
        super().__init__()
        # enkoder
        self.enc = nn.Sequential(
            nn.Conv2d(1,16,4,2,1), nn.ReLU(),
            nn.Conv2d(16,32,4,2,1), nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32*32*32, latent_dim)
        )
        # dekoder
        self.dec_lin = nn.Linear(latent_dim, 32*32*32)
        self.dec = nn.Sequential(
            nn.ConvTranspose2d(32,16,4,2,1), nn.ReLU(),
            nn.ConvTranspose2d(16,1,4,2,1), nn.Sigmoid()
        )
    def forward(self, x):
        z = self.enc(x)
        x1 = self.dec_lin(z).view(-1,32,32,32)
        return self.dec(x1), z

def train_ae():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    transform = transforms.Compose([transforms.Resize((128,128)), transforms.ToTensor()])
    ds = MVTecAnomalyDataset(f'../datasets/mvtec/{dataset_name}', phase='test', transform=transform)
    dl = DataLoader(ds, batch_size=32, shuffle=True)
    ae = AE(latent_dim=64).to(device)
    opt = optim.Adam(ae.parameters(), lr=1e-3)
    for epoch in range(1,31):
        ae.train()
        tot=0
        for x,_ in dl:
            x = x.to(device)
            opt.zero_grad()
            x_rec, _ = ae(x)
            loss = mse_loss(x_rec, x)
            loss.backward(); opt.step()
            tot += loss.item()
        print(f"Epoch {epoch} | MSE loss = {tot/len(dl):.6f}")
    torch.save(ae.enc.state_dict(), 'pthFiles/encoder.pth')
    torch.save({'dec_lin':ae.dec_lin.state_dict(),
                'dec':ae.dec.state_dict()}, 'pthFiles/decoder.pth')
    print("Saved encoder.pth and decoder.pth")

if __name__=='__main__':
    train_ae()