from torch import nn

import torch

# encoding block
class encoding_block(nn.Module):
    """
    Convolutional batch norm block with relu activation (main block used in the encoding steps)
    """

    def __init__(
        self,
        in_size,
        out_size,
        kernel_size=3,
        padding=0,
        stride=1,
        dilation=1,
        batch_norm=True,
        dropout=False,
    ):
        super().__init__()

        if batch_norm:

            # reflection padding for same size output as input (reflection padding has shown better results than zero padding)
            layers = [
                nn.ReflectionPad2d(padding=(kernel_size - 1) // 2),
                nn.Conv2d(
                    in_size,
                    out_size,
                    kernel_size=kernel_size,
                    padding=padding,
                    stride=stride,
                    dilation=dilation,
                ),
                nn.PReLU(),
                nn.BatchNorm2d(out_size),
                nn.ReflectionPad2d(padding=(kernel_size - 1) // 2),
                nn.Conv2d(
                    out_size,
                    out_size,
                    kernel_size=kernel_size,
                    padding=padding,
                    stride=stride,
                    dilation=dilation,
                ),
                nn.PReLU(),
                nn.BatchNorm2d(out_size),
            ]

        else:
            layers = [
                nn.ReflectionPad2d(padding=(kernel_size - 1) // 2),
                nn.Conv2d(
                    in_size,
                    out_size,
                    kernel_size=kernel_size,
                    padding=padding,
                    stride=stride,
                    dilation=dilation,
                ),
                nn.PReLU(),
                nn.ReflectionPad2d(padding=(kernel_size - 1) // 2),
                nn.Conv2d(
                    out_size,
                    out_size,
                    kernel_size=kernel_size,
                    padding=padding,
                    stride=stride,
                    dilation=dilation,
                ),
                nn.PReLU(),
            ]

        if dropout:
            layers.append(nn.Dropout())

        self.encoding_block = nn.Sequential(*layers)

    def forward(self, input):

        output = self.encoding_block(input)

        return output


# decoding block
class decoding_block(nn.Module):
    def __init__(self, in_size, out_size, batch_norm=False, upsampling=True):
        super().__init__()

        if upsampling:
            self.up = nn.Sequential(
                nn.Upsample(mode="bilinear", scale_factor=2),
                nn.Conv2d(in_size, out_size, kernel_size=1),
            )

        else:
            self.up = nn.ConvTranspose2d(in_size, out_size, kernel_size=2, stride=2)

        self.conv = encoding_block(in_size, out_size, batch_norm=batch_norm)
        self.rlam = RLAM(out_size, out_size)

    def forward(self, input1, input2):

        output2 = self.up(input2)

        output1 = nn.functional.interpolate(input1, output2.size()[2:], mode="bilinear")

        output1 = self.rlam(output1)

        return self.conv(torch.cat([output1, output2], 1))

# merge multi-scale feature maps
class merge_multi_scale(nn.Module):
    def __init__(
        self,
        in_size,
        out_size,
        kernel_size=3,
        padding=0,
        stride=1,
        dilation=1,
    ):
        super().__init__()

        self.up = nn.Sequential(
            nn.Upsample(mode="bilinear", scale_factor=2),
            nn.Conv2d(in_size, out_size, kernel_size=1),
        )

        self.conv = nn.Sequential(
            nn.ReflectionPad2d(padding=(kernel_size - 1) // 2),
            nn.Conv2d(
                in_size,
                out_size,
                kernel_size=kernel_size,
                padding=padding,
                stride=stride,
                dilation=dilation,
            ),
            nn.PReLU(),
            nn.BatchNorm2d(out_size),
        )
    
    def forward(self, input1, input2):

        output2 = self.up(input2)

        output1 = nn.functional.interpolate(input1, output2.size()[2:], mode="bilinear")

        return self.conv(torch.cat([output1, output2], 1))

# CAM block
class CAM(nn.Module):
    def __init__(self, in_size, reduction=16):
        super().__init__()

        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_size, in_size // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(in_size // reduction, in_size)
        )
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        b, c, _, _ = x.size()

        avg_pool = self.avg_pool(x).view(b, c)
        max_pool = self.max_pool(x).view(b, c)

        # print(f"AVG_SHAPE: {avg_pool.shape}, MAX_SHAPE: {max_pool.shape}")

        avg_out = self.fc(avg_pool).view(b, c, 1, 1)
        max_out = self.fc(max_pool).view(b, c, 1, 1)
        # print(f"AVG_SHAPE: {avg_out.shape}, MAX_SHAPE: {max_out.shape}")
        out = avg_out + max_out
        return x * self.sigmoid(out)

# RLAM block
class RLAM(nn.Module):
    def __init__( 
        self,
        in_size,
        out_size,
    ):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(
                in_size*2,
                out_size,
                kernel_size=1,
                bias=True
            ),
            nn.PReLU(),
            nn.BatchNorm2d(out_size),
        )
        self.cam = CAM(in_size)
        self.vertical_pool = nn.AdaptiveAvgPool2d((1, None))
        self.horizontal_pool = nn.AdaptiveAvgPool2d((None, 1))
        self.conv_1d_v = nn.Conv2d(in_size, in_size, kernel_size=(1, 3), padding=(0, 1), bias=False)
        self.conv_1d_h = nn.Conv2d(in_size, in_size, kernel_size=(3, 1), padding=(1, 0), bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        _, _, h, w = x.size()

        cam = self.cam(x)

        v_pool = self.vertical_pool(cam)
        v_pool = self.conv_1d_v(v_pool)
        v_pool = nn.functional.interpolate(v_pool, (h, w))

        h_pool = self.horizontal_pool(cam)
        h_pool = self.conv_1d_h(h_pool)
        h_pool = nn.functional.interpolate(h_pool, (h, w))

        # print(f"v_size: {v_pool.shape} | h_size: {h_pool.shape}")

        merge = torch.cat([v_pool, h_pool], dim=1)
        # merge = v_pool + h_pool
        merge = self.conv(merge)

        out = cam * self.sigmoid(merge)
        
        return self.conv(torch.cat([x, out], dim=1))

class RLUNet(nn.Module):
    """
    Main RL-UNet architecture
    """

    def __init__(self, num_classes=1):
        super().__init__()

        # encoding
        self.conv1 = encoding_block(3, 64)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)

        self.conv2 = encoding_block(64, 128)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)

        self.conv3 = encoding_block(128, 256)
        self.maxpool3 = nn.MaxPool2d(kernel_size=2)

        self.conv4 = encoding_block(256, 512)
        self.maxpool4 = nn.MaxPool2d(kernel_size=2)

        # center
        self.center = encoding_block(512, 1024)

        # merging multi-scale feature
        self.mms3 = merge_multi_scale(512, 256)
        self.mms2 = merge_multi_scale(256, 128)
        self.mms1 = merge_multi_scale(128, 64)

        # Residual Linear Attention Module
        self.rlam4 = RLAM(512, 512)
        self.rlam3 = RLAM(256, 256)
        self.rlam2 = RLAM(128, 128)
        self.rlam1 = RLAM(64, 64)

        # decoding
        self.decode4 = decoding_block(1024, 512)
        self.decode3 = decoding_block(512, 256)
        self.decode2 = decoding_block(256, 128)
        self.decode1 = decoding_block(128, 64)

        # final
        self.final = nn.Conv2d(64, num_classes, kernel_size=1)

    def forward(self, input):

        # encoding
        conv1 = self.conv1(input)
        maxpool1 = self.maxpool1(conv1)

        conv2 = self.conv2(maxpool1)
        maxpool2 = self.maxpool2(conv2)

        conv3 = self.conv3(maxpool2)
        maxpool3 = self.maxpool3(conv3)

        conv4 = self.conv4(maxpool3)
        maxpool4 = self.maxpool4(conv4)

        # center
        center = self.center(maxpool4)

        # merging multi-scale feature
        merge3 = self.mms3(conv3, conv4)
        merge2 = self.mms2(conv2, merge3)
        merge1 = self.mms1(conv1, merge2)

        # decoding 
        decode4 = self.decode4(conv4, center)

        decode3 = self.decode3(merge3, decode4)

        decode2 = self.decode2(merge2, decode3)

        decode1 = self.decode1(merge1, decode2)

        # final
        final = nn.functional.interpolate(
            self.final(decode1), input.size()[2:], mode="bilinear"
        )

        return final