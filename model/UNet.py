import torch
import torch.nn as nn

# for compatibilities issues with python 2.x
# use super(childClass, self).__init__()
# else use "super().__init__"
# but pytorch is not compatible with python 2.x

"""
Parameter Definition :
- in_channels = number of channels in the input image
- out_channels = number of channels produced by the convolution
- mid_channels = number of channels produced / as input
- kernel_size = size of the convolution window (filter size)
- padding = padding added to all four sides of image input
- bias = adds a learnable bias to the output (Default=True)
"""

# UNet Parts
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels

        # nn.Sequential is a sequential container of Modules
        self.double_conv = nn.Sequential(
            # nn.ReflectionPad2d(padding=(3 - 1) // 2),
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.PReLU(),
            # nn.ReflectionPad2d(padding=(3 - 1) // 2),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.PReLU(),
        )

    # x is data input
    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # self.up = nn.Upsample(size=in_channels, mode='bilinear')
        self.up = nn.ConvTranspose2d(in_channels, in_channels//2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        # self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diff_y = x2.size()[2] - x1.size()[2]
        diff_x = x2.size()[3] - x1.size()[3]

        x1 = nn.functional.pad(
            x1,
            [diff_x // 2, diff_x - diff_x // 2,
            diff_y // 2, diff_y - diff_y // 2]
        )

        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super().__init__()

        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down_1 = Down(64, 128)
        self.down_2 = Down(128, 256)
        self.down_3 = Down(256, 512)
        self.down_4 = Down(512, 1024)
        self.up_1 = Up(1024, 512)
        self.up_2 = Up(512, 256)
        self.up_3 = Up(256, 128)
        self.up_4 = Up(128, 64)
        self.outc = OutConv(64, n_classes) # -> default classes size is 2

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down_1(x1)
        x3 = self.down_2(x2)
        x4 = self.down_3(x3)
        x5 = self.down_4(x4)
        x = self.up_1(x5, x4)
        x = self.up_2(x, x3)
        x = self.up_3(x, x2)
        x = self.up_4(x, x1)

        # "logits" mean the raw outputs of the last layer in a neural network
        logits = self.outc(x)
        return logits


