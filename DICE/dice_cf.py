import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from dotenv import load_dotenv
from torch.utils.data import DataLoader
from torchvision import transforms
import dice_ml
from dice_ml import Model
from mvtec_dataset import MVTecAnomalyDataset
from model import PytorchMVTECWrapper
from sklearn.decomposition import PCA
from raiutils.exceptions import UserConfigValidationException
import random
import torch

np.random.seed(42)
random.seed(42)
torch.manual_seed(42)

def run_cf():
    print("⚙️ Starting DiCE counterfactual generation...")

    load_dotenv()
    dataset_name = os.getenv("DATASET_NAME")
    if dataset_name is None:
        raise ValueError("DATASET_NAME not set")

    print(f"📂 Using dataset: {dataset_name}")

    transform = transforms.Compose([
        transforms.Resize((128,128)),
        transforms.ToTensor()
    ])

    data_root = f'../datasets/mvtec/{dataset_name}'

    train_ds = MVTecAnomalyDataset(data_root, phase='train', transform=transform)
    train_loader = DataLoader(train_ds, batch_size=len(train_ds), shuffle=False)
    imgs_tr, labels_tr = next(iter(train_loader))
    X_tr = imgs_tr.numpy().reshape(imgs_tr.size(0), -1)
    print(f"🧮 Training data shape before PCA: {X_tr.shape}")

    n_components = 64
    pca = PCA(n_components=n_components, whiten=True)
    X_tr_pca = pca.fit_transform(X_tr)
    print(f"🧪 Training data shape after PCA: {X_tr_pca.shape}")

    continuous = [f'pc_{i}' for i in range(X_tr_pca.shape[1])]
    df_train = pd.DataFrame(X_tr_pca, columns=continuous)
    df_train['anomaly'] = labels_tr.numpy()
    print(f"🔢 Train set class counts:\n{df_train['anomaly'].value_counts()}")

    data = dice_ml.Data(
        dataframe=df_train,
        continuous_features=continuous,
        outcome_name='anomaly'
    )

    test_ds = MVTecAnomalyDataset(data_root, phase='test', transform=transform)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False)

    good_img, anom_img = None, None
    for img, label in test_loader:
        if label.item() == 0 and good_img is None:
            good_img = (img.clone(), label)
        elif label.item() == 1 and anom_img is None:
            anom_img = (img.clone(), label)
        if good_img and anom_img:
            break

    if good_img is None or anom_img is None:
        print("❌ Could not find both good and anomaly samples.")
        return

    wrapper = PytorchMVTECWrapper(f'pthFiles/{dataset_name}_cnn.pth', pca=pca)
    ml_model = Model(model=wrapper, backend='PYT', model_type='classifier')
    dice = dice_ml.Dice(data, ml_model, method='random')

    os.makedirs("generatedImages", exist_ok=True)

    for img, label in [good_img, anom_img]:
        X_query = img.numpy().reshape(1, -1)
        X_query_pca = pca.transform(X_query)
        df_query = pd.DataFrame(X_query_pca, columns=continuous)
        desired = int(1 - label.item())

        print(f"🚀 Generating CF for original class {label.item()} → desired {desired}")
        try:
            cf = dice.generate_counterfactuals(
                df_query,
                total_CFs=100,
                desired_class=desired,
                proximity_weight=0.0,
                diversity_weight=0.0,
                features_to_vary='all'
            )
        except UserConfigValidationException as e:
            print(f"❌ No counterfactuals found: {str(e)}")
            continue
        except Exception as e:
            print(f"❌ Unexpected error: {str(e)}")
            continue

        cf_df = cf.cf_examples_list[0].final_cfs_df
        cf_vals_pca = cf_df.iloc[0, :-1].to_numpy(dtype=np.float32)
        cf_vals = pca.inverse_transform(cf_vals_pca)
        img_cf = cf_vals.reshape((128,128))

        diff = np.abs(img[0,0].numpy() - img_cf)
        diff = (diff - diff.min()) / (diff.max() - diff.min() + 1e-8)

        fig, ax = plt.subplots(1, 3, figsize=(9, 3))
        ax[0].imshow(img[0,0].numpy(), cmap='gray')
        ax[0].set_title('Original')
        ax[1].imshow(img_cf, cmap='gray')
        ax[1].set_title('Counterfactual')
        ax[2].imshow(diff, cmap='hot')
        ax[2].set_title('Difference')
        for a in ax: a.axis('off')

        direction = f"{'good' if label.item()==0 else 'anomaly'}_to_{'anomaly' if label.item()==0 else 'good'}"
        fname = f"DiCE_{dataset_name}_{direction}.png"
        plt.savefig(os.path.join("generatedImages", fname), dpi=150)
        plt.close()
        print(f"✅ Saved: {fname}")

if __name__ == '__main__':
    run_cf()
