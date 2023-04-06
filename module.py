import torch.nn as nn
import torch.nn.functional as F
import torch

weight_PReLU = torch.FloatTensor([0.25]).cuda()


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # PReLU
        x = self.pool(F.prelu(self.conv1(x), weight=weight_PReLU))
        x = self.pool(F.prelu(self.conv2(x), weight=weight_PReLU))
        # ReLU
        # x = self.pool(F.relu(self.conv1(x)))
        # x = self.pool(F.relu(self.conv2(x)))
        # Sigmoid效果太差
        x = x.view(-1, 16 * 5 * 5)
        x = F.prelu(self.fc1(x), weight=weight_PReLU)
        x = F.prelu(self.fc2(x), weight=weight_PReLU)
        x = self.fc3(x)
        return x
