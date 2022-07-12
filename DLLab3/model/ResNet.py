import torch
import torch.nn as nn


class ResNet(nn.Module):
    def __init__(self):
        super(ResNet, self).__init__()
        self.relu = nn.ReLU()
        self.conv1 = nn.Conv2d(3, 64, (7, 7), stride=(2, 2), padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(3, stride=2)

        self.conv2_1 = nn.Conv2d(64, 64, (3, 3), stride=(1, 1), padding=1)
        self.bn2_1 = nn.BatchNorm2d(64)
        self.conv2_2 = nn.Conv2d(64, 64, (3, 3), stride=(1, 1), padding=1)
        self.bn2_2 = nn.BatchNorm2d(64)

        self.conv2_3 = nn.Conv2d(64, 64, (3, 3), stride=(1, 1), padding=1)
        self.bn2_3 = nn.BatchNorm2d(64)
        self.conv2_4 = nn.Conv2d(64, 64, (3, 3), stride=(1, 1), padding=1)
        self.bn2_4 = nn.BatchNorm2d(64)

        self.conv3_0 = nn.Conv2d(64, 128, (1, 1), stride=(2, 2))
        self.bn3_0 = nn.BatchNorm2d(128)
        self.conv3_1 = nn.Conv2d(64, 128, (3, 3), stride=(2, 2), padding=1)
        self.bn3_1 = nn.BatchNorm2d(128)
        self.conv3_2 = nn.Conv2d(128, 128, (3, 3), stride=(1, 1), padding=1)
        self.bn3_2 = nn.BatchNorm2d(128)

        self.conv3_3 = nn.Conv2d(128, 128, (3, 3), stride=(1, 1), padding=1)
        self.bn3_3 = nn.BatchNorm2d(128)
        self.conv3_4 = nn.Conv2d(128, 128, (3, 3), stride=(1, 1), padding=1)
        self.bn3_4 = nn.BatchNorm2d(128)

        self.conv4_0 = nn.Conv2d(128, 256, (1, 1), stride=(2, 2))
        self.bn4_0 = nn.BatchNorm2d(256)
        self.conv4_1 = nn.Conv2d(128, 256, (3, 3), stride=(2, 2), padding=1)
        self.bn4_1 = nn.BatchNorm2d(256)
        self.conv4_2 = nn.Conv2d(256, 256, (3, 3), stride=(1, 1), padding=1)
        self.bn4_2 = nn.BatchNorm2d(256)

        self.conv4_3 = nn.Conv2d(256, 256, (3, 3), stride=(1, 1), padding=1)
        self.bn4_3 = nn.BatchNorm2d(256)
        self.conv4_4 = nn.Conv2d(256, 256, (3, 3), stride=(1, 1), padding=1)
        self.bn4_4 = nn.BatchNorm2d(256)

        self.conv5_0 = nn.Conv2d(256, 512, (1, 1), stride=(2, 2))
        self.bn5_0 = nn.BatchNorm2d(512)
        self.conv5_1 = nn.Conv2d(256, 512, (3, 3), stride=(2, 2), padding=1)
        self.bn5_1 = nn.BatchNorm2d(512)
        self.conv5_2 = nn.Conv2d(512, 512, (3, 3), stride=(1, 1), padding=1)
        self.bn5_2 = nn.BatchNorm2d(512)

        self.conv5_3 = nn.Conv2d(512, 512, (3, 3), stride=(1, 1), padding=1)
        self.bn5_3 = nn.BatchNorm2d(512)
        self.conv5_4 = nn.Conv2d(512, 512, (3, 3), stride=(1, 1), padding=1)
        self.bn5_4 = nn.BatchNorm2d(512)

        self.avgpool = nn.AvgPool2d(7, 1)
        self.fc = nn.Linear(512, 12)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # conv1
        x = self.relu(self.bn1(self.conv1(x)))

        # conv2
        x = self.maxpool(x)
        t = self.relu(self.bn2_1(self.conv2_1(x)))
        x = self.relu(self.bn2_2(self.conv2_2(t) + x))

        t = self.relu(self.bn2_3(self.conv2_3(x)))
        x = self.relu(self.bn2_4(self.conv2_4(t) + x))

        # conv3
        t = self.relu(self.bn3_1(self.conv3_1(x)))
        x = self.bn3_0(self.conv3_0(x))
        x = self.relu(self.bn3_2(self.conv3_2(t) + x))

        t = self.relu(self.bn3_3(self.conv3_3(x)))
        x = self.relu(self.bn3_4(self.conv3_4(t) + x))

        # conv4
        t = self.relu(self.bn4_1(self.conv4_1(x)))
        x = self.bn4_0(self.conv4_0(x))
        x = self.relu(self.bn4_2(self.conv4_2(t) + x))

        t = self.relu(self.bn4_3(self.conv4_3(x)))
        x = self.relu(self.bn4_4(self.conv4_4(t) + x))

        # conv5
        t = self.relu(self.bn5_1(self.conv5_1(x)))
        x = self.bn5_0(self.conv5_0(x))
        x = self.relu(self.bn5_2(self.conv5_2(t) + x))

        t = self.relu(self.bn5_3(self.conv5_3(x)))
        x = self.relu(self.bn5_4(self.conv5_4(t) + x))

        x = self.avgpool(x)
        x = self.fc(x.view(x.size(0), -1))
        return self.softmax(x)
