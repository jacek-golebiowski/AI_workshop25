import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import transforms
from mvtec_dataset import MVTecAnomalyDataset
from model import PytorchMVTECWrapper
from clear_latent_utils import Encoder, Decoder, collect_latent_samples, clear_on_latent, decode_counterfactual
from dotenv import load_dotenv

load_dotenv()
dataset_name = os.getenv("DATASET_NAME")
if dataset_name is None:
    raise ValueError("❌ DATASET_NAME nie jest ustawione w .env!")

def run_clear_image():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    output_dir = './generatedImages'
    os.makedirs(output_dir, exist_ok=True)

    enc = Encoder(latent_dim=64).to(device)
    dec = Decoder(latent_dim=64).to(device)
    enc.load_state_dict(torch.load('pthFiles/encoder.pth', map_location=device))
    ck = torch.load('pthFiles/decoder.pth', map_location=device)
    dec.dec_lin.load_state_dict(ck['dec_lin'])
    dec.dec.load_state_dict(ck['dec'])
    enc.eval(); dec.eval()

    wrapper = PytorchMVTECWrapper(f'pthFiles/{dataset_name}_cnn.pth', device=device)
    wrapper.net.eval()

    transform = transforms.Compose([transforms.Resize((128, 128)), transforms.ToTensor()])
    ds = MVTecAnomalyDataset(f'../datasets/mvtec/{dataset_name}', phase='test', transform=transform)
    loader = DataLoader(ds, batch_size=1, shuffle=False)

    good_img, anom_img = None, None
    for img, label in loader:
        if label.item() == 0 and good_img is None:
            good_img = (img.clone(), label)
        elif label.item() == 1 and anom_img is None:
            anom_img = (img.clone(), label)
        if good_img and anom_img:
            break

    if good_img is None or anom_img is None:
        print("❌ Could not find both good and anomaly images.")
        return

    print(f"📊 Image sums: good={torch.sum(good_img[0]):.4f}, anomaly={torch.sum(anom_img[0]):.4f}")

    for img, label in [good_img, anom_img]:
        img = img.to(device)
        orig_class = 'anomaly' if label.item() == 1 else 'good'
        print(f"\n=== {orig_class.upper()} ===")
        print(f"🧾 Image sum: {torch.sum(img):.4f}")

        with torch.no_grad():
            logits = wrapper(img)
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
        print(f"[ORIGINAL] good={probs[0]:.4f}, anomaly={probs[1]:.4f}")

        z0, Z, P = collect_latent_samples(img[0], enc, dec, wrapper, n_samples=8000, sigma=3.0, device=device)
        pred_classes = (P >= 0.5).astype(int)
        unique = np.unique(pred_classes)
        print(f"👀 Classes in latent space: {unique}")

        if len(unique) < 2:
            print(f"⚠️ Only class {unique[0]} — skipping CF.")
            continue

        target_class = 1 - label.item()
        direction = f"{orig_class}_to_{'good' if target_class == 0 else 'anomaly'}"
        deltas = clear_on_latent(z0, Z, P, boundary=0.5, target_class=target_class)

        if not deltas:
            print(f"❌ No delta found for {direction}")
            continue

        top_dims = sorted(deltas, key=lambda i: abs(deltas[i]))[:5]
        cf = decode_counterfactual(z0, deltas, dec, dims=top_dims, device=device, scale=0.2)

        with torch.no_grad():
            cf_input = cf.unsqueeze(0).to(device)
            cf_logits = wrapper(cf_input)
            cf_probs = torch.softmax(cf_logits, dim=1).cpu().numpy()[0]
        print(f"[CF] good={cf_probs[0]:.4f}, anomaly={cf_probs[1]:.4f}")

        diff = torch.abs(img[0, 0].cpu() - cf[0].cpu())
        diff = (diff - diff.min()) / (diff.max() - diff.min() + 1e-8)

        fig, ax = plt.subplots(1, 3, figsize=(9, 3))
        ax[0].imshow(img[0, 0].cpu(), cmap='gray'); ax[0].set_title('Original')
        ax[1].imshow(cf[0], cmap='gray'); ax[1].set_title('CLEAR CF')
        ax[2].imshow(diff, cmap='hot'); ax[2].set_title('Difference')
        for a in ax: a.axis('off')

        fname = f"CLEAR_{dataset_name}_orig-{direction}.png"
        plt.savefig(os.path.join(output_dir, fname), dpi=150)
        plt.close()
        print(f"✅ Saved: {fname}")

if __name__ == '__main__':
    run_clear_image()