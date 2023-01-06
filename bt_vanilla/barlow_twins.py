from torch.utils.data import DataLoader 

from tqdm import tqdm

from torchvision import transforms as T

import torch
from torch import nn
import torchvision
from torchvision.datasets import MNIST

from lightly.data import LightlyDataset
from lightly.data import ImageCollateFunction, BaseCollateFunction
from lightly.models.modules import BarlowTwinsProjectionHead
from lightly.loss import BarlowTwinsLoss

from time_shift_dataset import HandDataset

import matplotlib.pyplot as plt

BATCHSIZE = 64

class BarlowTwins(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone
        self.projection_head = BarlowTwinsProjectionHead(512, 2048, 2048)

    def forward(self, x):
        x = self.backbone(x).flatten(start_dim=1)
        z = self.projection_head(x)
        return z


if __name__ == '__main__':
    resnet = torchvision.models.resnet34()
    resnet.conv1 = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(2, 2), padding=(3, 3), bias=False)
    backbone = nn.Sequential(*list(resnet.children())[:-1])
    model = BarlowTwins(backbone)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    transform = T.Compose([
            T.Resize(128),
            # T.ToTensor(),
            # T.Grayscale()
    ])

    hand_dataset = HandDataset('./datasets/hand')
    dataset = LightlyDataset.from_torch_dataset(hand_dataset, transform=transform)

    collate_fn = ImageCollateFunction(input_size=128, min_scale=0.7, hf_prob=0.0, kernel_size=0.01)

    dataloader = DataLoader(
        dataset,
        batch_size=BATCHSIZE,
        collate_fn=collate_fn,
        shuffle=True,
        drop_last=True,
        num_workers=8,
    )

    criterion = BarlowTwinsLoss(lambda_param=0.001)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    print("Starting Training")
    for epoch in range(100):
        total_loss = 0
        for (x0, x1), _, _ in tqdm(dataloader):
            x0 = x0.to(device)
            x1 = x1.to(device)
            z0 = model(x0)
            z1 = model(x1)
            loss = criterion(z0, z1)
            total_loss += loss.detach()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        avg_loss = total_loss / len(dataloader)
        print(f"epoch: {epoch:>02}, loss: {avg_loss:.5f}")

    torch.save(model, 'models/model_x.pth')
    torch.save(model.backbone, 'models/backbone_x.pth')