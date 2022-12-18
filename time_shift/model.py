from lightly.models.modules import BarlowTwinsProjectionHead
from torchvision.models import resnet34, ResNet34_Weights
from torch import nn

class TimeShiftModel(nn.Module):
    def __init__(self) -> None:
        super(TimeShiftModel, self).__init__()
        resnet = resnet34(ResNet34_Weights.IMAGENET1K_V1)
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        self.backbone[0] = nn.Conv2d(1, 64, kernel_size=(3, 3), stride=(2, 2), padding=(3, 3), bias=False)
        self.projection_head = BarlowTwinsProjectionHead(512, 1024, 1024)

    def forward(self, x):
        x = self.backbone(x).flatten(start_dim=1)
        z = self.projection_head(x)
        return z