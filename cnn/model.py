import torch.nn
import torch.nn as nn
import torch.nn.functional as F


class SimpleVGG16(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=3,
                               out_channels=64,
                               kernel_size=(3, 3),
                               stride=1,
                               padding="same")
        self.conv2 = nn.Conv2d(in_channels=64,
                               out_channels=64,
                               kernel_size=(3, 3),
                               stride=1,
                               padding="same")
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.batchnorm1 = nn.BatchNorm2d(64)
        self.drop = nn.Dropout(p=0.2)

        self.conv3 = nn.Conv2d(in_channels=64,
                               out_channels=256,
                               kernel_size=(3, 3),
                               stride=1,
                               padding="same")
        self.conv4 = nn.Conv2d(in_channels=256,
                               out_channels=256,
                               kernel_size=(3, 3),
                               stride=1,
                               padding="same")

        self.dense1 = nn.Linear(in_features=36864, out_features=1024)
        self.dense2 = nn.Linear(in_features=1024, out_features=1024)
        self.dense3 = nn.Linear(in_features=1024, out_features=256)
        self.batchnorm2 = nn.BatchNorm2d(256)

        self.finaldense = nn.Linear(in_features=256, out_features=7)

    def forward(self, x):
        x = self.drop(self.pool(self.batchnorm1(F.relu(self.conv2(F.relu(self.conv1(x)))))))
        x = self.drop(self.pool(self.batchnorm2(F.relu(self.conv4(F.relu(self.conv3(x)))))))

        x = nn.Flatten(1,-1)(x)
        x = self.dense1(x)
        x = self.dense2(x)
        x = self.dense3(x)
        x = nn.BatchNorm1d(256)(x)
        x = F.softmax(self.finaldense(x), dim=1)

        return x
