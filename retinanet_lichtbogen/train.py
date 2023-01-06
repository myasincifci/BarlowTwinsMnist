import os

import torch
from torchvision.models import resnet50
from torchvision.models.detection import retinanet_resnet50_fpn, RetinaNet_ResNet50_FPN_Weights
from torchvision.models import ResNet50_Weights

from torch.utils.data import DataLoader
import torchvision.transforms as T

from tqdm import tqdm

from dataset import LichtbogenDataset

import matplotlib.pyplot as plt

NUM_EPOCHS = 50

def coll_fn(x):
    inputs, labels = zip(*x)

    inputs = torch.stack(inputs, dim=0)

    labels = [{'boxes':l['boxes'], 'labels':l['labels']} for l in labels]

    return (inputs, labels)

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    root = './retinanet_lichtbogen/lichtbogen_ds_coco'
    transform = T.Compose([
        T.Resize(round(1024/4)),
        T.ToTensor(),
    ])
    
    dataset = LichtbogenDataset(
        root=os.path.join(root, 'data'), 
        annotation=os.path.join(root, 'labels.json'), 
        transforms=transform
    )
    dataloader = DataLoader(dataset, batch_size=1, collate_fn=coll_fn, num_workers=8)

    weights = torch.load('./time_shift_lichtbogen/models/backbone_time_shift_lb_EP_30_BS_20_PRX_20_LAMBDA_0.001.pth')
    model = retinanet_resnet50_fpn(weights_backbone=None)

    # model = retinanet_resnet50_fpn(weights_backbone=ResNet50_Weights.IMAGENET1K_V1)

    resnet = resnet50()
    
    model.to(device=device)
    model.train()

    optimizer = torch.optim.SGD(model.parameters(), lr=0.005, weight_decay=0.0005)

    avg_losses = []
    for epoch in range(NUM_EPOCHS):
        epoch_losses = []
        for i, (images, targets) in enumerate(tqdm(dataloader)):
            images = images.to(device)

            for t in targets:
                t['boxes'] = t['boxes'].to(device)
                t['labels'] = t['labels'].to(device)

            loss_dict = model(images, targets) 

            losses = sum(loss for loss in loss_dict.values()) 
            epoch_losses.append(losses.item())
            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

        epoch_losses = torch.tensor(epoch_losses)
        avg_loss = epoch_losses.mean()
        avg_losses.append(avg_loss)
        print(avg_loss)

    torch.save(model.state_dict(), './retinanet_lichtbogen/models/model.pth')

    print(avg_losses)
    plt.plot(avg_losses)
    plt.show()

if __name__ == '__main__':
    main()