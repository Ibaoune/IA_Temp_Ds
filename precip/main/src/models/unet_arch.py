import torch
import torch.nn as nn
import torch.nn.functional as F

class UNet(nn.Module):
    def __init__(self, in_channels=15, out_channels=1):
        super(UNet, self).__init__()

        # --- ENCODER ---
        self.enc_conv1 = self.conv_block(in_channels, 64, k=2)
        self.enc_conv2 = self.conv_block(64, 128, k=2)
        self.enc_conv3 = self.conv_block(128, 256, k=2)
        
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # --- BOTTLENECK ---
        self.bottleneck_conv = self.conv_block(256, 512, k=2)

        # --- DECODER ---
        self.upconv3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec_conv3 = self.conv_block(512, 256, k=2)

        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec_conv2 = self.conv_block(256, 128, k=2)

        self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec_conv1 = self.conv_block(128, 64, k=2)

        self.head_upconv1 = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)
        self.head_conv1 = self.conv_block(64, 64, k=2)
        
        self.head_upconv2 = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)
        self.head_conv2 = self.conv_block(64, 64, k=2)

        self.final_output = nn.Conv2d(64, out_channels, kernel_size=1, stride=1)

    def conv_block(self, in_channels, out_channels, k=2):
        """
        Creates a block of TWO consecutive Convolution -> BatchNorm -> ReLU sequences.
        Uses padding='same' to preserve spatial dimensions during convolutions.
        """
        block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=k, stride=1, padding='same'),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=k, stride=1, padding='same'),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        return block

    @staticmethod
    def _match_size(upsampled, skip):
        """Pad or crop upsampled tensor to match skip-connection spatial dims."""
        diff_h = skip.size(2) - upsampled.size(2)
        diff_w = skip.size(3) - upsampled.size(3)
        upsampled = F.pad(upsampled, [diff_w // 2, diff_w - diff_w // 2,
                                       diff_h // 2, diff_h - diff_h // 2])
        return upsampled

    def forward(self, x):
        # Encoder
        c1 = self.enc_conv1(x)
        p1 = self.pool(c1)

        c2 = self.enc_conv2(p1)
        p2 = self.pool(c2)

        c3 = self.enc_conv3(p2)
        p3 = self.pool(c3)

        # Bottleneck
        bn = self.bottleneck_conv(p3)

        # Decoder (with concatenations)
        u3 = self.upconv3(bn)
        u3 = self._match_size(u3, c3)
        u3 = torch.cat([u3, c3], dim=1)
        d3 = self.dec_conv3(u3)

        u2 = self.upconv2(d3)
        u2 = self._match_size(u2, c2)
        u2 = torch.cat([u2, c2], dim=1)
        d2 = self.dec_conv2(u2)

        u1 = self.upconv1(d2)
        u1 = self._match_size(u1, c1)
        u1 = torch.cat([u1, c1], dim=1)
        d1 = self.dec_conv1(u1)

        # Extended Head
        h1_up = self.head_upconv1(d1)
        h1 = self.head_conv1(h1_up)

        h2_up = self.head_upconv2(h1)
        h2 = self.head_conv2(h2_up)

        # Final 1x1 Conv
        out = self.final_output(h2)

        return out