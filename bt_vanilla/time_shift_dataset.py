import os
from typing import Tuple
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np

import torchvision.transforms as T

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid

class HandDataset(Dataset):
    def __init__(self, path, transform=None, train=True) -> None:
        self.train = train
        self.transform = transform
        self.image_paths = sorted([os.path.join(path, p) for p in os.listdir(path)])

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index):

        # 1. get one element x
        image = Image.open(self.image_paths[index])

        cls = self.get_class(index)

        if self.transform:
            image = self.transform(image)

        # 3. return (x, x')
        return image, cls

    def get_class(self, image_no):
        # 0: scissors, 1: paper, 2: rock
        if image_no <= 122:
            return 0
        elif image_no <= 203:
            return 1
        elif image_no <= 292:
            return 2
        elif image_no <= 393:
            return 0
        elif image_no <= 493:
            return 1
        elif image_no <= 620:
            return 2
        elif image_no <= 722:
            return 0
        elif image_no <= 821:
            return 1
        elif image_no <= 932:
            return 2
        elif image_no <= 1010:
            return 0
        elif image_no <= 1123 :
            return 1
        elif image_no <= 1232:
            return 2
        # test
        elif image_no <= 1327:
            return 0
        elif image_no <= 1425:
            return 1
        elif image_no <= 1527:
            return 2
        elif image_no <= 1620:
            return 0
        elif image_no <= 1703:
            return 1
        else:
            return 2


if __name__ == '__main__':
    transform = T.Compose([
            T.Resize(128),
            T.ToTensor(),
            T.Grayscale()
        ])

    test_dataset = HandDataset('./datasets/hand', transform=transform, train=False)
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=5*10,
        shuffle=True,
        num_workers=1,
    )

    images, _, classes, _ = next(iter(test_dataloader))

    fig = plt.figure(figsize=(200,200))
    grid = ImageGrid(fig, 111, nrows_ncols=(10, 5), axes_pad=0.1)

    for ax, im, cls in zip(grid, images, classes):
        im = im.swapaxes(0,-1).swapaxes(0,1)
        cls = cls.item()
        ax.text(10.0, 15.0, cls, {'color': 'red'})
        ax.imshow(im)

    plt.show()