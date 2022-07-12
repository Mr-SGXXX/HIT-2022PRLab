import torch.nn as nn


class VGG(nn.Module):
    def __init__(self):
        super(VGG, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(2, stride=2)
        self.conv3_64 = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=1)
        self.conv3_128 = nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=1)
        self.conv3_256_1 = nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=1)
        self.conv3_256_2 = nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=1)
        self.conv3_512_1 = nn.Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=1)
        self.conv3_512_2 = nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=1)
        self.conv3_512_3 = nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=1)
        self.conv3_512_4 = nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=1)
        self.fc_1 = nn.Linear(7 * 7 * 512, 4096)
        self.drop1 = nn.Dropout(0.5)
        self.fc_2 = nn.Linear(4096, 4096)
        self.drop2 = nn.Dropout(0.5)
        self.fc_3 = nn.Linear(4096, 12)
        # self.softmax = nn.Softmax(dim=1)

        # 初始化权值
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.relu(self.conv3_64(x))
        x = self.maxpool(x)
        x = self.relu(self.conv3_128(x))
        x = self.maxpool(x)
        x = self.relu(self.conv3_256_1(x))
        x = self.relu(self.conv3_256_2(x))
        x = self.maxpool(x)
        x = self.relu(self.conv3_512_1(x))
        x = self.relu(self.conv3_512_2(x))
        x = self.maxpool(x)
        x = self.relu(self.conv3_512_3(x))
        x = self.relu(self.conv3_512_4(x))
        x = self.maxpool(x).view(x.size(0), -1)
        x = self.drop1(self.relu(self.fc_1(x)))
        x = self.drop2(self.relu(self.fc_2(x)))
        x = self.fc_3(x)  # SoftMax好像没啥用，在这加ReLU反而会效果变差
        return x
        # return self.softmax(x)
