import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self, opt_class):
        super().__init__()
        self.conv11 = nn.Conv2d(3, 64, 3, padding=1)
        self.conv12 = nn.Conv2d(64, 64, 3, padding=1)

        self.conv21 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv22 = nn.Conv2d(128, 128, 3, padding=1)

        self.conv31 = nn.Conv2d(128, 256, 3, padding=1)
        self.conv32 = nn.Conv2d(256, 256, 3, padding=1)
        self.conv33 = nn.Conv2d(256, 256, 3, padding=1)

        self.conv41 = nn.Conv2d(256, 512, 3, padding=1)
        self.conv42 = nn.Conv2d(512, 512, 3, padding=1)
        self.conv43 = nn.Conv2d(512, 512, 3, padding=1)

        self.conv51 = nn.Conv2d(512, 512, 3, padding=1)
        self.conv52 = nn.Conv2d(512, 512, 3, padding=1)
        self.conv53 = nn.Conv2d(512, 512, 3, padding=1)

        self.pool = nn.MaxPool2d(2, 2)

        self.fc1 = nn.Linear(7*7*512, 4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.fc3 = nn.Linear(4096, opt_class)
    
    def forward(self, input):
        x = F.relu(self.conv11(input))
        x = F.relu(self.conv12(x))
        x = self.pool(x)
        x = F.relu(self.conv21(x))
        x = F.relu(self.conv22(x))
        x = self.pool(x)
        x = F.relu(self.conv31(x))
        x = F.relu(self.conv32(x))
        x = F.relu(self.conv33(x))
        x = self.pool(x)
        x = F.relu(self.conv41(x))
        x = F.relu(self.conv42(x))
        x = F.relu(self.conv43(x))
        x = self.pool(x)
        x = F.relu(self.conv51(x))
        x = F.relu(self.conv52(x))
        x = F.relu(self.conv53(x))
        x = self.pool(x)

        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x =  F.dropout(x, 0.5)
        x = F.relu(self.fc2(x))
        x =  F.dropout(x, 0.5)
        x = self.fc3(x)
        return x