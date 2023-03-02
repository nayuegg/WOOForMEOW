#import modules
import torch.nn as nn
from torchvision import transforms
import pytorch_lightning as pl

from torchvision.models import resnet18


transform = transforms.Compose([
    transforms.ToTensor()
])

class Net(pl.LightningModule):

    def __init__(self):
        super().__init__()

        self.feature = resnet18(pretrained=True)
        self.fc = nn.Linear(1000,2)


    def forward(self, x):
        h = self.feature(x)
        h = self.fc(h)
        return h