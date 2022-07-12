import torch
import torch.nn as nn


class IMDN(nn.Module):
    def __init__(self, in_nc=3, nf=64, out_nc=3, upscale=4):
        super(IMDN, self).__init__()
        # self.fea_conv = nn.Sequential(
        #     nn.Conv2d(in_nc, nf, 3, 1, 1),
        #     nn.LeakyReLU(0.05),
        #     nn.Conv2d(nf, nf, 3, 1, 1)
        # )
        self.fea_conv = nn.Conv2d(in_nc, nf, 3, 1, 1)

        self.IMDB1 = IMDB(nf)
        self.IMDB2 = IMDB(nf)
        self.IMDB3 = IMDB(nf)
        self.IMDB4 = IMDB(nf)
        self.IMDB5 = IMDB(nf)
        self.IMDB6 = IMDB(nf)

        self.c = nn.Sequential(
            nn.Conv2d(6 * nf, nf, 1, 1, 0),
            nn.LeakyReLU(0.05)
        )
        self.LR_conv = nn.Conv2d(nf, nf, 3, 1, 1)

        self.upsample = nn.Sequential(
            nn.Conv2d(nf, out_nc * (upscale ** 2), 3, 1, 1),
            nn.PixelShuffle(upscale)
        )

    def forward(self, input):
        out_fea = self.fea_conv(input)
        out_B1 = self.IMDB1(out_fea)
        out_B2 = self.IMDB2(out_B1)
        out_B3 = self.IMDB3(out_B2)
        out_B4 = self.IMDB4(out_B3)
        out_B5 = self.IMDB5(out_B4)
        out_B6 = self.IMDB6(out_B5)

        out_B = self.c(torch.cat([out_B1, out_B2, out_B3, out_B4, out_B5, out_B6], dim=1))
        out_lr = self.LR_conv(out_B) + out_fea
        output = self.upsample(out_lr)
        return output


class IMDB(nn.Module):
    def __init__(self, in_channels, distillation_rate=0.25):
        super(IMDB, self).__init__()
        self.distilled_channels = int(in_channels * distillation_rate)
        self.remaining_channels = int(in_channels - self.distilled_channels)
        self.c1 = nn.Conv2d(in_channels, in_channels, 3, 1, padding=1, bias=True, dilation=1, groups=1)
        self.c2 = nn.Conv2d(self.remaining_channels, in_channels, 3, 1, padding=1, bias=True, dilation=1, groups=1)
        self.c3 = nn.Conv2d(self.remaining_channels, in_channels, 3, 1, padding=1, bias=True, dilation=1, groups=1)
        self.c4 = nn.Conv2d(self.remaining_channels, self.distilled_channels, 3, 1, padding=1, bias=True,
                            dilation=1,
                            groups=1)
        self.act = nn.LeakyReLU(negative_slope=0.05, inplace=True)
        self.c5 = nn.Conv2d(in_channels, in_channels, 1, 1, padding=0, bias=True, dilation=1, groups=1)
        self.cca = CCA(self.distilled_channels * 4)

    def forward(self, input):
        out_c1 = self.act(self.c1(input))
        distilled_c1, remaining_c1 = torch.split(out_c1, (self.distilled_channels, self.remaining_channels), dim=1)
        out_c2 = self.act(self.c2(remaining_c1))
        distilled_c2, remaining_c2 = torch.split(out_c2, (self.distilled_channels, self.remaining_channels), dim=1)
        out_c3 = self.act(self.c3(remaining_c2))
        distilled_c3, remaining_c3 = torch.split(out_c3, (self.distilled_channels, self.remaining_channels), dim=1)
        out_c4 = self.c4(remaining_c3)
        out = torch.cat([distilled_c1, distilled_c2, distilled_c3, out_c4], dim=1)
        out_fused = self.c5(self.cca(out)) + input
        return out_fused


class CCA(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CCA, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.contrast(x) + self.avg_pool(x)
        y = self.conv_du(y)
        return x * y

    def contrast(self, F):
        assert (F.dim() == 4)
        spatial_sum = F.sum(3, keepdim=True).sum(2, keepdim=True)
        F_mean = spatial_sum / (F.size(2) * F.size(3))
        F_variance = (F - F_mean).pow(2).sum(3, keepdim=True).sum(2, keepdim=True) / (F.size(2) * F.size(3))
        return F_variance.pow(0.5)
