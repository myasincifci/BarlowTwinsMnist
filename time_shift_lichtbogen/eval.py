from math import floor
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as T
from models import (BaselineImagenet1K, BaselineRandomInit, LinearEval,
                    TimeShiftModel)
from mpl_toolkits.axes_grid1 import ImageGrid
from time_shift_dataset import TimeShiftDataset
from torchvision.models import ResNet, ResNet34_Weights, resnet34
from tqdm import tqdm

# MODEL_NAME = 'backbone_time_shift_EP_30_PRX_60_LAMBDA_0.001' # ~15%
# MODEL_NAME = 'backbone_time_shift_EP_30_BS_64_PRX_40_LAMBDA_0.001' # ~5%
# MODEL_NAME = 'backbone_time_shift_EP_50_BS_64_PRX_30_LAMBDA_0.001' # ~4.5%
# MODEL_NAME = 'backbone_time_shift_EP_30_BS_80_PRX_30_LAMBDA_0.001' # ~4.3%
MODEL_NAME = 'backbone_time_shift_EP_30_BS_80_PRX_30_LAMBDA_0.001'
NUM_EPOCHS = 30
device = "cuda" if torch.cuda.is_available() else "cpu"

def plot_batch(batch):
    images, _, classes, _ = batch
    n_cols = 5
    n_rows = floor(len(images) / 5)

    fig = plt.figure(figsize=(200,200))
    grid = ImageGrid(fig, 111, nrows_ncols=(n_rows, n_cols), axes_pad=0.1)

    for ax, im, cls in zip(grid, images, classes):
        im = im.swapaxes(0,-1).swapaxes(0,1)
        cls = cls.item()
        ax.text(10.0, 15.0, cls, {'color': 'red'})
        ax.imshow(im, cmap='gray')

    plt.show()

def test_model(model, test_dataset, testloader):

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

def train(model, num_epochs, train_dataset, test_dataset, train_loader, test_loader, device, criterion, optimizer):
    # loop over the dataset multiple times
    losses, errors = [], []
    for epoch in range(num_epochs):
        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
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

        test_error = test_model(model, test_dataset, test_loader)
        # print(f'Epoch: {epoch}, Loss: {running_loss:.4f}, Test error: {test_error:.4f}')
        losses.append(running_loss)
        errors.append(test_error)

    # print('Finished Training')
    return (losses, errors)

def eval(model: nn.Module, train_dataset, test_dataset, train_loader, test_loader) -> List[float]:
    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

    losses, errors = train(model, NUM_EPOCHS, train_dataset, test_dataset, train_loader, test_loader, device, criterion, optimizer)
    return losses, errors

if __name__ == '__main__':

    backbone: ResNet = torch.load(f'time_shift/models/{MODEL_NAME}.pth')
    print(device)
    backbone.to(device)

    transform = T.Compose([
            T.Resize(128),
            T.ToTensor(),
            T.Grayscale()
        ])

    train_dataset = TimeShiftDataset('./datasets/hand', transform=transform, train=True)
    test_dataset = TimeShiftDataset('./datasets/hand', transform=transform, train=False)

    lss_eval, errs_eval = [], []
    lss_bl, errs_bl = [], []
    for i in tqdm(range(1), leave=False):
        ss_indices = np.random.choice(len(train_dataset), 200, replace=False)
        ss_train = torch.utils.data.Subset(train_dataset, ss_indices)

        train_dataloader = torch.utils.data.DataLoader(
            ss_train, # train_dataset,
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
        baseline = BaselineImagenet1K().to(device)

        losses_eval, errors_eval = eval(eval_model, train_dataset, test_dataset, train_dataloader, test_dataloader)
        losses_bl, errors_bl = eval(baseline, train_dataset, test_dataset, train_dataloader, test_dataloader)

        lss_eval.append(losses_eval)
        errs_eval.append(errors_eval)
        lss_bl.append(losses_bl)
        errs_bl.append(errors_bl)

    errs_eval = np.array(errs_eval)
    errs_bl = np.array(errs_bl)

    plt.plot(np.arange(30), errs_eval.mean(axis=0), '-b', label='Tempo(ours)')
    plt.plot(np.arange(30), errs_bl.mean(axis=0), '-r', label='Baseline')
    plt.show()