""" Full assembly of the parts to form the complete network """

from .unet_parts import *


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        factor = 2
        self.inc = (DoubleConv(n_channels, int(64 // factor)))
        self.down1 = (Down(int(64 // factor), int(128 // factor)))
        self.down2 = (Down(int(128 // factor), int(256 // factor)))
        self.down3 = (Down(int(256 // factor), int(512 // factor)))
        # factor = 2 if bilinear else 1
        self.down4 = (Down(int(512 // factor), int(1024 // factor)))
        conv_channels = int(1024 // factor) + 256
        self.down5 = (Down(conv_channels, conv_channels * 2))
        conv_channels *= 2
        self.down6 = (Down(conv_channels, conv_channels * 2))
        conv_channels *= 2
        self.up1 = (Up(conv_channels, int(conv_channels // 2), bilinear))
        conv_channels = int(conv_channels // 2)
        self.up2 = (Up(conv_channels, int(conv_channels // 2), bilinear))
        conv_channels = int(conv_channels // 2)
        self.outc = (OutConv(conv_channels, n_classes))

    def forward(self, x, encodings):
        x1 = self.inc(x)
        x2 = self.down1(x1) # 512
        x3 = self.down2(x2) # 256
        x4 = self.down3(x3) # 128
        x5 = self.down4(x4) # 64
        x5 = torch.cat([x5, encodings], dim=1)
        x6 = self.down5(x5) # 32
        x7 = self.down6(x6) # 16
        x = self.up1(x7, x6)
        x = self.up2(x, x5)
        logits = (self.outc(x))
        return logits

    def use_checkpointing(self):
        self.inc = torch.utils.checkpoint(self.inc)
        self.down1 = torch.utils.checkpoint(self.down1)
        self.down2 = torch.utils.checkpoint(self.down2)
        self.down3 = torch.utils.checkpoint(self.down3)
        self.down4 = torch.utils.checkpoint(self.down4)
        self.up1 = torch.utils.checkpoint(self.up1)
        self.up2 = torch.utils.checkpoint(self.up2)
        self.up3 = torch.utils.checkpoint(self.up3)
        self.up4 = torch.utils.checkpoint(self.up4)
        self.outc = torch.utils.checkpoint(self.outc)