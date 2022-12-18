from math import floor
import numpy as np

import torch
import torch.nn as nn

import torchvision.transforms as T
from torchvision.models import ResNet, resnet34, ResNet34_Weights

from time_shift_dataset import TimeShiftDataset

from tqdm import tqdm

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid

# MODEL_NAME = 'backbone_time_shift_EP_30_PRX_60_LAMBDA_0.001' # ~15%
# MODEL_NAME = 'backbone_time_shift_EP_30_BS_64_PRX_40_LAMBDA_0.001' # ~5%
# MODEL_NAME = 'backbone_time_shift_EP_50_BS_64_PRX_30_LAMBDA_0.001' # ~4.5%
# MODEL_NAME = 'backbone_time_shift_EP_30_BS_80_PRX_30_LAMBDA_0.001' # ~4.3%
MODEL_NAME = 'backbone_time_shift_EP_30_BS_80_PRX_30_LAMBDA_0.001'

NUM_EPOCHS = 150

def plot_batch(batch):
    images, classes = batch
    n_cols = 5
    n_rows = floor(len(images) / 5)

    fig = plt.figure(figsize=(200,200))
    grid = ImageGrid(fig, 111, nrows_ncols=(n_rows, n_cols), axes_pad=0.1)

    for ax, im, cls in zip(grid, images, classes):
        im = im.swapaxes(0,-1).swapaxes(0,1)
        cls = cls.item()
        ax.text(10.0, 15.0, cls, {'color': 'red'})
        ax.imshow(im)

    plt.show()

class LinearEval(nn.Module):
    def __init__(self, backbone: nn.Module):
        super(LinearEval, self).__init__()
        self.backbone = backbone
        self.linear = nn.Linear(in_features=512, out_features=3, bias=True)

    def forward(self, x):
        x = self.backbone(x)
        x = torch.flatten(x, start_dim=1)
        x = self.linear(x)

        return x

class Baseline(nn.Module):
    def __init__(self):
        super(Baseline, self).__init__()
        resnet = resnet34(ResNet34_Weights.IMAGENET1K_V1)
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        self.backbone[0] = nn.Conv2d(1, 64, kernel_size=(3, 3), stride=(2, 2), padding=(3, 3), bias=False)
        self.linear = nn.Linear(in_features=512, out_features=3, bias=True)

    def forward(self, x):
        x = self.backbone(x)
        x = torch.flatten(x, start_dim=1)
        x = self.linear(x)

        return x

def test_model(model, testloader):

    wrongly_classified = 0
    for i, data in enumerate(testloader, 0):
        total = len(data[0])
        inputs, _, labels, _ = data
        inputs,labels = inputs.to(device), labels.to(device)

        with torch.no_grad():
            preds = model(inputs).argmax(dim=1)

        wrong = (total - (preds == labels).sum()).item()
        wrongly_classified += wrong

    return wrongly_classified / len(test_dataset)

def train(model, num_epochs,train_loader, test_loader, device):
    # loop over the dataset multiple times
    for epoch in range(num_epochs):
        running_loss = 0.0
        for i, data in enumerate(tqdm(train_loader), 0):
            inputs, _, labels, _ = data
            labels = nn.functional.one_hot(labels, num_classes=3).float()
            inputs, labels = inputs.to(device), labels.to(device)


            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f'Epoch: {epoch}, Loss: {running_loss:.4f}, Test error: {test_model(model, test_loader):.4f}')

    print('Finished Training')

if __name__ == '__main__':

    backbone: ResNet = torch.load(f'time_shift/models/{MODEL_NAME}.pth')
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(device)
    backbone.to(device)

    transform = T.Compose([
            T.Resize(128),
            T.ToTensor(),
            T.Grayscale()
        ])

    train_dataset = TimeShiftDataset('./datasets/hand', transform=transform, train=True)
    test_dataset = TimeShiftDataset('./datasets/hand', transform=transform, train=False)

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=True,
        num_workers=8,
    )

    test_dataloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=32,
        shuffle=True,
        num_workers=8,
    )

    eval_model = LinearEval(backbone=backbone).to(device)
    baseline = Baseline().to(device)

    model = eval_model

    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.SGD(model.linear.parameters(), lr=0.0005, momentum=0.9)
    # optimizer = torch.optim.SGD(lin_baseline.linear.parameters(), lr=0.001, momentum=0.8)

    # train(eval_model, NUM_EPOCHS, train_dataloader, test_dataloader, device)
    train(model, NUM_EPOCHS, train_dataloader, test_dataloader, device)