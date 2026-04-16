import torch
import torch.nn as nn
import torch.nn.functional as F


class UNet(nn.Module):
    """
    U-Net for statistical temperature downscaling.

    Compatible with:
    - Gaussian output: out_channels = 2  -> [mean, log_var]
    - MSE output     : out_channels = 1  -> [mean only]

    Input :
        (B, in_channels, H, W)

    Output :
        if out_channels == 2:
            (B, 2, H_out, W_out)
            channel 0 -> mean
            channel 1 -> log_var
        else:
            (B, 1, H_out, W_out)
    """

    def __init__(
        self,
        in_channels=15,
        out_channels=2,
        base_filters=64,
        upscale_factor=1,
    ):
        super().__init__()

        assert upscale_factor in (1, 2, 4, 8), "upscale_factor must be 1, 2, 4 or 8"

        self.out_channels = out_channels
        self.upscale_factor = upscale_factor
        f = base_filters

        # -------------------------
        # Encoder
        # -------------------------
        self.enc1 = self.conv_block(in_channels, f)
        self.enc2 = self.conv_block(f, f * 2)
        self.enc3 = self.conv_block(f * 2, f * 4)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # -------------------------
        # Bottleneck
        # -------------------------
        self.bottleneck = self.conv_block(f * 4, f * 8)

        # -------------------------
        # Decoder
        # -------------------------
        self.up3 = nn.ConvTranspose2d(f * 8, f * 4, kernel_size=2, stride=2)
        self.dec3 = self.conv_block(f * 8, f * 4)

        self.up2 = nn.ConvTranspose2d(f * 4, f * 2, kernel_size=2, stride=2)
        self.dec2 = self.conv_block(f * 4, f * 2)

        self.up1 = nn.ConvTranspose2d(f * 2, f, kernel_size=2, stride=2)
        self.dec1 = self.conv_block(f * 2, f)

        # -------------------------
        # Optional super-resolution head
        # -------------------------
        self.sr_head = self._build_sr_head(f, upscale_factor)

        # -------------------------
        # Output heads
        # -------------------------
        if out_channels == 2:
            self.head_mean = nn.Conv2d(f, 1, kernel_size=1)
            self.head_log_var = nn.Conv2d(f, 1, kernel_size=1)
        else:
            self.final_output = nn.Conv2d(f, out_channels, kernel_size=1)

    def conv_block(self, in_channels, out_channels, k=3):
        """
        Two consecutive Conv -> BatchNorm -> ReLU blocks.
        Uses 3x3 convolutions for better spatial feature extraction.
        """
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=k, stride=1, padding="same", bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_channels, out_channels, kernel_size=k, stride=1, padding="same", bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def _build_sr_head(self, channels, upscale_factor):
        """
        Optional learnable upscaling head.
        If upscale_factor == 1, it acts as identity.
        """
        if upscale_factor == 1:
            return nn.Identity()

        n_steps = {2: 1, 4: 2, 8: 3}[upscale_factor]
        layers = []

        for _ in range(n_steps):
            layers.extend([
                nn.ConvTranspose2d(channels, channels, kernel_size=2, stride=2),
                nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(channels),
                nn.ReLU(inplace=True),
            ])

        return nn.Sequential(*layers)

    @staticmethod
    def _match_size(upsampled, skip):
        """
        Pad/crop upsampled tensor to match skip-connection spatial size.
        """
        diff_h = skip.size(2) - upsampled.size(2)
        diff_w = skip.size(3) - upsampled.size(3)

        if diff_h != 0 or diff_w != 0:
            upsampled = F.pad(
                upsampled,
                [
                    diff_w // 2, diff_w - diff_w // 2,
                    diff_h // 2, diff_h - diff_h // 2
                ]
            )
        return upsampled

    def forward(self, x):
        # -------------------------
        # Encoder
        # -------------------------
        c1 = self.enc1(x)
        p1 = self.pool(c1)

        c2 = self.enc2(p1)
        p2 = self.pool(c2)

        c3 = self.enc3(p2)
        p3 = self.pool(c3)

        # -------------------------
        # Bottleneck
        # -------------------------
        bn = self.bottleneck(p3)

        # -------------------------
        # Decoder
        # -------------------------
        u3 = self.up3(bn)
        u3 = self._match_size(u3, c3)
        d3 = self.dec3(torch.cat([u3, c3], dim=1))

        u2 = self.up2(d3)
        u2 = self._match_size(u2, c2)
        d2 = self.dec2(torch.cat([u2, c2], dim=1))

        u1 = self.up1(d2)
        u1 = self._match_size(u1, c1)
        d1 = self.dec1(torch.cat([u1, c1], dim=1))

        # -------------------------
        # Optional SR head
        # -------------------------
        features = self.sr_head(d1)

        # -------------------------
        # Outputs
        # -------------------------
        if self.out_channels == 2:
            mean = self.head_mean(features)
            log_var = self.head_log_var(features).clamp(min=-10.0, max=10.0)
            out = torch.cat([mean, log_var], dim=1)
        else:
            out = self.final_output(features)

        return out