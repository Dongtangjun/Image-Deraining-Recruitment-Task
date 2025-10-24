import torch.nn as nn
from torch.nn import functional as F

class BaselineNet(nn.Module):
    def __init__(self):
        super(BaselineNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 3, 1, 1)
        self.conv2 = nn.Conv2d(64, 64, 3, 1, 1)
        self.conv3 = nn.Conv2d(64, 64, 3, 1, 1)
        self.conv4 = nn.Conv2d(64, 64, 3, 1, 1)
        self.conv5 = nn.Conv2d(64, 64, 3, 1, 1)
        self.conv6 = nn.Conv2d(64, 64, 3, 1, 1)
        self.conv7 = nn.Conv2d(64, 64, 3, 1, 1)
        self.conv8 = nn.Conv2d(64, 3, 3, 1, 1)

        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(64)
        self.bn4 = nn.BatchNorm2d(64)
        self.bn5 = nn.BatchNorm2d(64)
        self.bn6 = nn.BatchNorm2d(64)
        self.bn7 = nn.BatchNorm2d(64)
        self.bn8 = nn.BatchNorm2d(3)

    def forward(self, x):
        Y = F.relu(self.bn1(self.conv1(x)))
        Y = F.relu(self.bn2(self.conv2(Y)))
        Y = F.relu(self.bn3(self.conv3(Y)))
        Y = F.relu(self.bn4(self.conv4(Y)))
        Y = F.relu(self.bn5(self.conv5(Y)))
        Y = F.relu(self.bn6(self.conv6(Y)))
        Y = F.relu(self.bn7(self.conv7(Y)))
        Y = self.bn8(self.conv8(Y))

        Y += x
        Y = F.relu(Y)

        return Y

