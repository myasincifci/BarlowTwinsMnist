import os
from typing import Tuple
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np

class TimeShiftDataset(Dataset):
    def __init__(self, path, transform=None, proximity:int=3) -> None:
        self.p = proximity
        self.transform = transform
        self.image_paths = sorted([os.path.join(path, p) for p in os.listdir(path)])

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index) -> Tuple[torch.Tensor, torch.Tensor]:
        # 1. get one element x
        image = Image.open(self.image_paths[index])

        # 2. sample element x' in the neighbourhood of x within proximity (between 1 and p where p is proximity)
        possbile_l = range(((index - self.p) if (index - self.p) >= 0 else 0), index)
        possbile_r = range(index + 1, ((index + self.p + 1) if (index + self.p + 1) <= len(self) else len(self)))
        index_d = np.random.choice(list(possbile_l) + list(possbile_r))
        image_d = Image.open(self.image_paths[index_d])

        if self.transform:
            image = self.transform(image)
            image_d = self.transform(image_d)

        # 3. return (x, x')
        return (image, image_d)


if __name__ == '__main__':
    dataset = TimeShiftDataset('./datasets/hand')