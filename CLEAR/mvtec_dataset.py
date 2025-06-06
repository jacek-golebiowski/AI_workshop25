import os
from PIL import Image
from torch.utils.data import Dataset

class MVTecAnomalyDataset(Dataset):
    def __init__(self, root_dir, phase='train', transform=None):
        """
        root_dir:   np. '../../datasets/mvtec/bottle'
        phase:      'train' albo 'test'
        transform:  torchvision.transforms
        """
        self.transform = transform
        self.images = []
        self.labels = []

        phase_dir = os.path.join(root_dir, phase)
        if phase == 'train':
            good_dir = os.path.join(phase_dir, 'good')
            if os.path.isdir(good_dir):
                for fn in os.listdir(good_dir):
                    if fn.startswith('.'):
                        continue
                    self.images.append(os.path.join(good_dir, fn))
                    self.labels.append(0)

            test_dir = os.path.join(root_dir, 'test')
            if os.path.isdir(test_dir):
                for sub in os.listdir(test_dir):
                    subdir = os.path.join(test_dir, sub)
                    if not os.path.isdir(subdir) or sub == 'good':
                        continue
                    for fn in os.listdir(subdir):
                        if fn.startswith('.'):
                            continue
                        self.images.append(os.path.join(subdir, fn))
                        self.labels.append(1)

        else:
            for sub in os.listdir(phase_dir):
                subdir = os.path.join(phase_dir, sub)
                if not os.path.isdir(subdir):
                    continue
                label = 0 if sub == 'good' else 1
                for fn in os.listdir(subdir):
                    if fn.startswith('.'):
                        continue
                    self.images.append(os.path.join(subdir, fn))
                    self.labels.append(label)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = Image.open(self.images[idx]).convert('L')
        if self.transform:
            img = self.transform(img)
        return img, self.labels[idx]