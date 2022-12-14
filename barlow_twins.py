from torch.utils.data import DataLoader 

from tqdm import tqdm

from custom_mnist import CustomMnist

import torch
from torch import nn
import torchvision

from lightly.data import LightlyDataset
from lightly.data import ImageCollateFunction, BaseCollateFunction
from lightly.models.modules import BarlowTwinsProjectionHead
from lightly.loss import BarlowTwinsLoss

import matplotlib.pyplot as plt

BATCHSIZE = 256

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
    resnet = torchvision.models.resnet18()
    resnet.conv1 = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(2, 2), padding=(3, 3), bias=False)
    backbone = nn.Sequential(*list(resnet.children())[:-1])
    model = BarlowTwins(backbone)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    # cifar10 = torchvision.datasets.CIFAR10("datasets/cifar10", download=True)
    # mnist = torchvision.datasets.MNIST("datasets/mnist", download=True)
    mnist = CustomMnist("datasets/mnist", download=True)

    mnist.data = mnist.data.unsqueeze(dim=1).repeat(1,3,1,1) # make images RGB, TODO: remove later

    dataset = LightlyDataset.from_torch_dataset(mnist)
    # or create a dataset from a folder containing images or videos:
    # dataset = LightlyDataset("path/to/folder")

    collate_fn = ImageCollateFunction(input_size=28, min_scale=0.7, hf_prob=0.0)

    dataloader = DataLoader(
        dataset,
        batch_size=BATCHSIZE,
        collate_fn=collate_fn,
        shuffle=True,
        drop_last=True,
        num_workers=8,
    )

    criterion = BarlowTwinsLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.06)

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