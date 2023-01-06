import torch
from torchvision import transforms as T
from torch.utils.data import DataLoader
from time_shift_dataset import TimeShiftDataset
from models import TimeShiftModel, TimeShiftModel50
from lightly.loss import BarlowTwinsLoss

from tqdm import tqdm

LR = 1e-3

NUM_EPOCHS = 30
BATCH_SIZE = 20
PROXIMITY = 20
LAMBDA_PARAM = 1e-3

SUFFIX = f'EP_{NUM_EPOCHS}_BS_{BATCH_SIZE}_PRX_{PROXIMITY}_LAMBDA_{LAMBDA_PARAM}'

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f'Using device: {device}.')

    transform = T.Compose([
        T.Resize(round(1024/4)),
        T.ToTensor(),
    ])

    dataset = TimeShiftDataset('./datasets/lichtbogen', transform=transform, proximity=PROXIMITY)
    dataset.image_paths = dataset.image_paths
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True, num_workers=2)

    model = TimeShiftModel50().to(device)
    criterion = BarlowTwinsLoss(lambda_param=LAMBDA_PARAM)
    optimizer = torch.optim.SGD(model.parameters(), lr=LR, weight_decay=0.001)

    for epoch in range(NUM_EPOCHS):
        losses = []
        for image, image_d in tqdm(dataloader):
            torch.cuda.empty_cache()
            
            image = image.to(device)
            z0 = model(image)
            image = image.to('cpu')
            torch.cuda.empty_cache()

            image_d = image_d.to(device)
            z1 = model(image_d)
            image_d = image_d.to('cpu')
            torch.cuda.empty_cache()

            loss = criterion(z0, z1)
            losses.append(loss.item())
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        
        avg_loss = torch.tensor(losses).mean()
        print(f"epoch: {epoch:>02}, loss: {avg_loss:.5f}")

    # torch.save(model.backbone, f'time_shift_lichtbogen/models/backbone_time_shift_lb_{SUFFIX}.pth')
    torch.save(model.backbone.state_dict(), f'time_shift_lichtbogen/models/backbone_time_shift_lb_{SUFFIX}.pth')
    print(f'backbone_time_shift_lb_{SUFFIX}')

if __name__ == '__main__':
    main()