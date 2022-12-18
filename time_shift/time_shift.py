import torch
from torchvision import transforms as T
from torch.utils.data import DataLoader
from time_shift_dataset import TimeShiftDataset
from model import TimeShiftModel
from lightly.loss import BarlowTwinsLoss

from tqdm import tqdm

LR = 1e-3

NUM_EPOCHS = 30
BATCH_SIZE = 80
PROXIMITY = 30
LAMBDA_PARAM = 1e-3

SUFFIX = f'EP_{NUM_EPOCHS}_BS_{BATCH_SIZE}_PRX_{PROXIMITY}_LAMBDA_{LAMBDA_PARAM}'

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f'Using device: {device}.')

    transform = T.Compose([
        T.Resize(128),
        T.ToTensor(),
        T.Grayscale()
    ])

    # dataset = TimeShiftDataset('video_short_half_res', transform, proximity=PROXIMITY)
    dataset = TimeShiftDataset('./datasets/hand', transform=transform, proximity=PROXIMITY)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True, num_workers=8)

    model = TimeShiftModel().to(device)
    criterion = BarlowTwinsLoss(lambda_param=LAMBDA_PARAM)
    optimizer = torch.optim.SGD(model.parameters(), lr=LR, weight_decay=0.001)

    for epoch in range(NUM_EPOCHS):
        losses = []
        for image, image_d, _, _ in tqdm(dataloader):
            image = image.to(device)
            image_d = image_d.to(device)
            
            z0 = model(image)
            z1 = model(image_d)
            loss = criterion(z0, z1)
            losses.append(loss.detach())
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        
        avg_loss = torch.tensor(losses).mean()
        print(f"epoch: {epoch:>02}, loss: {avg_loss:.5f}")

    # torch.save(model, f'time_shift/models/model_time_shift_{SUFFIX}.pth')
    torch.save(model.backbone, f'time_shift/models/backbone_time_shift_{SUFFIX}.pth')
    print(f'backbone_time_shift_{SUFFIX}')

if __name__ == '__main__':
    main()