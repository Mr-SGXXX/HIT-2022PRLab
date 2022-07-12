import paddle.nn as nn


class MobileNetV2(nn.Layer):
    def __init__(self, num_classes, dropout_prob=0.2):
        super(MobileNetV2, self).__init__()

        # Network is created here, then will be unpacked into nn.sequential
        s1, s2 = 2, 2
        self.network_settings = [{'t': -1, 'c': 32, 'n': 1, 's': s1},
                                 {'t': 1, 'c': 16, 'n': 1, 's': 1},
                                 {'t': 6, 'c': 24, 'n': 2, 's': s2},
                                 {'t': 6, 'c': 32, 'n': 3, 's': 2},
                                 {'t': 6, 'c': 64, 'n': 4, 's': 2},
                                 {'t': 6, 'c': 96, 'n': 3, 's': 1},
                                 {'t': 6, 'c': 160, 'n': 3, 's': 2},
                                 {'t': 6, 'c': 320, 'n': 1, 's': 1},
                                 {'t': None, 'c': 1280, 'n': 1, 's': 1}]
        self.num_classes = num_classes
        # Feature Extraction part
        # Layer 0
        self.network = [conv2d_bn_relu6(3, int(self.network_settings[0]['c']),
                                        3, self.network_settings[0]['s'], dropout_prob)]

        # Layers from 1 to 7
        for i in range(1, 8):
            self.network.extend(
                inverted_residual_sequence(
                    int(self.network_settings[i - 1]['c']),
                    int(self.network_settings[i]['c']),
                    self.network_settings[i]['n'], self.network_settings[i]['t'],
                    3, self.network_settings[i]['s']))

        # Last layer before flattening
        self.network.append(
            conv2d_bn_relu6(int(self.network_settings[7]['c']),
                            int(self.network_settings[8]['c']), 1,
                            self.network_settings[8]['s'],
                            dropout_prob))

        # Classification part
        self.network.append(nn.Dropout2D(dropout_prob))
        self.network.append(nn.AvgPool2D((7, 7)))
        self.network.append(nn.Dropout2D(dropout_prob))
        self.network.append(nn.Conv2D(int(self.network_settings[8]['c']), self.num_classes, 1, bias_attr=None))

        self.network = nn.Sequential(*self.network)

    def forward(self, x):
        x = self.network(x)
        x = x.reshape([-1, self.num_classes])
        return x


class InvertedResidualBlock(nn.Layer):
    def __init__(self, in_channels, out_channels, expansion_factor=6, kernel_size=3, stride=1):
        super(InvertedResidualBlock, self).__init__()
        if stride != 1 and stride != 2:
            raise ValueError("Stride should be 1 or 2")
        self.block = nn.Sequential(
            nn.Conv2D(in_channels, in_channels * expansion_factor, 1, bias_attr=False),
            nn.BatchNorm2D(in_channels * expansion_factor),
            nn.ReLU6(),
            # depthwise conv
            nn.Conv2D(in_channels * expansion_factor, in_channels * expansion_factor,
                      kernel_size, stride, 1, groups=in_channels * expansion_factor, bias_attr=False),
            nn.BatchNorm2D(in_channels * expansion_factor),
            nn.ReLU6(),

            nn.Conv2D(in_channels * expansion_factor, out_channels, 1, bias_attr=False),
            nn.BatchNorm2D(out_channels))
        self.is_residual = True if stride == 1 else False
        self.is_conv_res = False if in_channels == out_channels else True
        if stride == 1 and self.is_conv_res:
            self.conv_res = nn.Sequential(nn.Conv2D(in_channels, out_channels, 1, bias_attr=False),
                                          nn.BatchNorm2D(out_channels))

    def forward(self, x):
        block = self.block(x)
        if self.is_residual:
            if self.is_conv_res:
                return self.conv_res(x) + block
            return x + block
        return block


def inverted_residual_sequence(in_channels, out_channels, num_units, expansion_factor=6,
                               kernel_size=3,
                               initial_stride=2):
    bottleneck_arr = [
        InvertedResidualBlock(in_channels, out_channels, expansion_factor, kernel_size,
                              initial_stride)]

    for i in range(num_units - 1):
        bottleneck_arr.append(
            InvertedResidualBlock(out_channels, out_channels, expansion_factor, kernel_size, 1))

    return bottleneck_arr


def conv2d_bn_relu6(in_channels, out_channels, kernel_size=3, stride=2, dropout_prob=0.0):
    # To preserve the equation of padding. (k=1 maps to pad 0, k=3 maps to pad 1, k=5 maps to pad 2, etc.)
    padding = (kernel_size + 1) // 2 - 1
    return nn.Sequential(
        nn.Conv2D(in_channels, out_channels, kernel_size, stride, padding, bias_attr=False),
        nn.BatchNorm2D(out_channels),
        # For efficiency, Dropout is placed before Relu.
        nn.Dropout2D(dropout_prob),
        # Assumption: Relu6 is used everywhere.
        nn.ReLU6()
    )
