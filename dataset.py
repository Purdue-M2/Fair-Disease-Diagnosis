"""Lung CT 3D Volume Dataset with gender metadata."""

import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset


class LungCTDataset(Dataset):
    """3D CT volume dataset for fair lung disease diagnosis.

    Each sample is a CT scan resampled to (target_depth, target_size, target_size),
    paired with a disease label and binary gender indicator.

    Args:
        data_dirs: List of (folder_path, label) tuples.
        target_depth: Number of slices to resample to (default: 64).
        target_size: Spatial resolution per slice (default: 256).
        is_train: If True, applies random cropping and augmentation.
    """

    def __init__(self, data_dirs, target_depth=64, target_size=256, is_train=True):
        self.samples = []
        self.target_depth = target_depth
        self.target_size = target_size
        self.is_train = is_train

        for folder_path, label in data_dirs:
            for gender in ["male", "female"]:
                gender_path = os.path.join(folder_path, gender)
                if not os.path.exists(gender_path):
                    continue
                gender_id = 1 if gender == "male" else 0
                for scan_name in os.listdir(gender_path):
                    if scan_name.startswith("._"):
                        continue
                    scan_path = os.path.join(gender_path, scan_name)
                    if os.path.isdir(scan_path):
                        self.samples.append((scan_path, label, gender_id))

        males = sum(1 for _, _, g in self.samples if g == 1)
        females = sum(1 for _, _, g in self.samples if g == 0)
        print(f"Loaded {len(self.samples)} samples (Male: {males}, Female: {females})")

    def _load_scan(self, scan_path):
        slice_files = []
        for f in os.listdir(scan_path):
            if f.startswith("._") or not f.endswith(".jpg"):
                continue
            slice_num = int(f.replace(".jpg", ""))
            slice_files.append((slice_num, f))
        slice_files.sort(key=lambda x: x[0])

        slices = []
        for _, fname in slice_files:
            img = Image.open(os.path.join(scan_path, fname)).convert("L")
            img = img.resize((self.target_size, self.target_size))
            slices.append(np.array(img, dtype=np.float32))

        if len(slices) == 0:
            return np.zeros((1, self.target_size, self.target_size), dtype=np.float32)
        return np.stack(slices, axis=0)

    def _remove_non_lung_slices(self, volume):
        D = volume.shape[0]
        if D <= 5:
            return volume
        start = int(D * 0.10)
        end = int(D * 0.90)
        if end <= start:
            return volume
        return volume[start:end]

    def _resize_depth(self, volume, target_depth):
        D = volume.shape[0]
        if D == 0:
            return np.zeros(
                (target_depth, volume.shape[1], volume.shape[2]), dtype=np.float32
            )
        indices = np.linspace(0, D - 1, target_depth).astype(int)
        return volume[indices]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        scan_path, label, gender = self.samples[idx]

        volume = self._load_scan(scan_path)
        volume = self._remove_non_lung_slices(volume)
        volume = self._resize_depth(volume, self.target_depth)
        volume = volume / 255.0

        if self.is_train:
            crop_size = 224
            h_start = np.random.randint(0, self.target_size - crop_size + 1)
            w_start = np.random.randint(0, self.target_size - crop_size + 1)
            volume = volume[:, h_start : h_start + crop_size, w_start : w_start + crop_size]

            if np.random.random() > 0.5:
                k = np.random.choice([1, 2, 3])
                volume = np.rot90(volume, k, axes=(1, 2)).copy()

            if np.random.random() > 0.5:
                volume = np.flip(volume, axis=2).copy()
        else:
            start = (self.target_size - 224) // 2
            volume = volume[:, start : start + 224, start : start + 224]

        import torch

        volume_tensor = torch.from_numpy(volume).unsqueeze(0).float()
        return volume_tensor, label, gender
