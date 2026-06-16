import csv
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
from PIL import Image
from cnn import EMOTIONS, IMAGENET_MEAN, IMAGENET_STD

IMG_SIZE = 224

def build_transforms(train: bool = True) -> transforms.Compose:
    if train:
        return transforms.Compose([
            transforms.Resize((IMG_SIZE + 16, IMG_SIZE + 16)),
            transforms.RandomCrop(IMG_SIZE),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05),
            transforms.RandomGrayscale(p=0.05),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            transforms.RandomErasing(p=0.25, scale=(0.02, 0.15)),
        ])
    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
 
class FER2013Dataset(Dataset):
    """
    FER2013 CSV dataset (Kaggle format).
 
    Parameters
    ----------
    csv_path : path to fer2013.csv
    split    : 'train' | 'val' | 'test'
    """
    SPLIT_MAP = {"train": "Training", "val": "PublicTest", "test": "PrivateTest"}
 
    def __init__(self, csv_path: str, split: str = "train", transform=None):
        self.transform = transform or build_transforms(split == "train")
        self.data: list = []
        target = self.SPLIT_MAP[split]
        with open(csv_path) as f:
            for row in csv.DictReader(f):
                if row["Usage"] != target:
                    continue
                label  = int(row["emotion"])
                pixels = np.array(row["pixels"].split(), dtype=np.uint8).reshape(48, 48)
                self.data.append((pixels, label))
 
    def __len__(self):
        return len(self.data)
 
    def __getitem__(self, idx):
        pixels, label = self.data[idx]
        img = Image.fromarray(pixels, "L").convert("RGB")
        return self.transform(img), label
 
    def class_weights(self) -> torch.Tensor:
        counts = torch.zeros(len(EMOTIONS))
        for _, l in self.data:
            counts[l] += 1
        w = 1.0 / counts.clamp(min=1)
        return w / w.sum()
    
class EmotionFolderDataset(Dataset):
    """
    Image-folder dataset.
    Expected layout:  root/Happy/*.jpg   root/Angry/*.png   etc.
    """
    def __init__(self, root: str, split: str = "train", transform=None):
        self.transform = transform or build_transforms(split == "train")
        self.samples: list = []
        for cls_dir in sorted(Path(root).iterdir()):
            if not cls_dir.is_dir():
                continue
            cls_name = cls_dir.name.capitalize()
            if cls_name not in EMOTIONS:
                continue
            label = EMOTIONS.index(cls_name)
            for p in cls_dir.glob("**/*"):
                if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
                    self.samples.append((p, label))
 
    def __len__(self):
        return len(self.samples)
 
    def __getitem__(self, idx):
        path, label = self.samples[idx]
        return self.transform(Image.open(path).convert("RGB")), label
    

def make_fer2013_loaders(csv_path: str, batch_size: int = 64,
                         num_workers: int = 4, use_sampler: bool = True):
    """Returns (train_loader, val_loader, test_loader)."""
    train_ds = FER2013Dataset(csv_path, "train")
    val_ds   = FER2013Dataset(csv_path, "val")
    test_ds  = FER2013Dataset(csv_path, "test")
 
    if use_sampler:
        cw      = train_ds.class_weights()
        sw      = torch.tensor([cw[l].item() for _, l in train_ds.data])
        sampler = WeightedRandomSampler(sw, len(sw), replacement=True)
        train_loader = DataLoader(train_ds, batch_size=batch_size,
                                  sampler=sampler, num_workers=num_workers)
    else:
        train_loader = DataLoader(train_ds, batch_size=batch_size,
                                  shuffle=True, num_workers=num_workers)
 
    val_loader  = DataLoader(val_ds,  batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, val_loader, test_loader
 
 
def make_folder_loaders(root: str, batch_size: int = 64,
                        num_workers: int = 4, val_split: float = 0.15):
    """Returns (train_loader, val_loader) for a custom image-folder dataset."""
    full  = EmotionFolderDataset(root, "train")
    n_val = int(len(full) * val_split)
    n_tr  = len(full) - n_val
    tr_ds, va_ds = torch.utils.data.random_split(
        full, [n_tr, n_val], generator=torch.Generator().manual_seed(42)
    )
    va_ds.dataset.transform = build_transforms(False)
    return (
        DataLoader(tr_ds, batch_size=batch_size, shuffle=True,  num_workers=num_workers),
        DataLoader(va_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers),
    )