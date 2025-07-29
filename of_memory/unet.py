""" Full assembly of the parts to form the complete network """

from .unet_parts import *


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        factor = 4 / 3
        self.inc = (DoubleConv(n_channels, int(64 // factor)))
        self.down1 = (Down(int(64 // factor), int(128 // factor)))
        self.down2 = (Down(int(128 // factor), int(256 // factor)))
        self.down3 = (Down(int(256 // factor), int(512 // factor)))
        # factor = 2 if bilinear else 1
        self.down4 = (Down(int(512 // factor), int(1024 // factor)))
        self.down5 = (Down(int(1024 // factor), int(2048 // factor)))
        self.down6 = (Down(int(2048 // factor), int(4096 // factor)))
        self.up1 = (Up(int(4096 // factor), int(2048 // factor), bilinear))
        self.up2 = (Up(int(2048 // factor), int(1024 // factor), bilinear))
        self.outc = (OutConv(int(1024 // factor), n_classes))

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x6 = self.down5(x5)
        x7 = self.down6(x6)
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