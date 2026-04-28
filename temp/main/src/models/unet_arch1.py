import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvNormReLU(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        norm_type="group",
        num_groups=32,
        dropout=0.0,
    ):
        super().__init__()

        if norm_type == "group":
            groups = min(num_groups, out_channels)
            norm = nn.GroupNorm(groups, out_channels)
        else:
            norm = nn.BatchNorm2d(out_channels)

        layers = [
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            norm,
            nn.ReLU(inplace=True),
        ]

        if dropout > 0:
            layers.append(nn.Dropout2d(dropout))

        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class DoubleConv(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        norm_type="group",
        num_groups=32,
        dropout=0.0,
    ):
        super().__init__()

        self.block = nn.Sequential(
            ConvNormReLU(
                in_channels,
                out_channels,
                norm_type=norm_type,
                num_groups=num_groups,
                dropout=dropout,
            ),
            ConvNormReLU(
                out_channels,
                out_channels,
                norm_type=norm_type,
                num_groups=num_groups,
                dropout=dropout,
            ),
        )

    def forward(self, x):
        return self.block(x)


class UpBlock(nn.Module):
    """
    Upsample + Conv.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        norm_type="group",
        num_groups=32,
        dropout=0.0,
    ):
        super().__init__()

        self.conv = ConvNormReLU(
            in_channels,
            out_channels,
            norm_type=norm_type,
            num_groups=num_groups,
            dropout=dropout,
        )

    def forward(self, x, target_size):
        x = F.interpolate(
            x,
            size=target_size,
            mode="bilinear",
            align_corners=False,
        )
        x = self.conv(x)
        return x


class UNet1(nn.Module):
    """
    Input:
        x : (B, C, H_low, W_low)

    Output:
        si out_channels = 1:
            (B, 1, H_high, W_high)

        si out_channels = 2:
            canal 0 : mean
            canal 1 : log_var
    """

    def __init__(
        self,
        in_channels,
        out_channels=1,
        base_filters=64,
        use_gaussian=False,
        norm_type="group",
        num_groups=32,
        dropout=0.0,
    ):
        super().__init__()

        self.out_channels = out_channels
        self.use_gaussian = use_gaussian

        f = base_filters

        # -------------------------
        # Encoder
        # -------------------------
        self.enc1 = DoubleConv(
            in_channels,
            f,
            norm_type=norm_type,
            num_groups=num_groups,
            dropout=dropout,
        )

        self.enc2 = DoubleConv(
            f,
            f * 2,
            norm_type=norm_type,
            num_groups=num_groups,
            dropout=dropout,
        )

        self.enc3 = DoubleConv(
            f * 2,
            f * 4,
            norm_type=norm_type,
            num_groups=num_groups,
            dropout=dropout,
        )

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # -------------------------
        # Bottleneck
        # -------------------------
        self.bottleneck = DoubleConv(
            f * 4,
            f * 8,
            norm_type=norm_type,
            num_groups=num_groups,
            dropout=dropout,
        )

        # -------------------------
        # Decoder : Upsample + Conv
        # -------------------------
        self.up3 = UpBlock(
            f * 8,
            f * 4,
            norm_type=norm_type,
            num_groups=num_groups,
            dropout=dropout,
        )
        self.dec3 = DoubleConv(
            f * 8,
            f * 4,
            norm_type=norm_type,
            num_groups=num_groups,
            dropout=dropout,
        )

        self.up2 = UpBlock(
            f * 4,
            f * 2,
            norm_type=norm_type,
            num_groups=num_groups,
            dropout=dropout,
        )
        self.dec2 = DoubleConv(
            f * 4,
            f * 2,
            norm_type=norm_type,
            num_groups=num_groups,
            dropout=dropout,
        )

        self.up1 = UpBlock(
            f * 2,
            f,
            norm_type=norm_type,
            num_groups=num_groups,
            dropout=dropout,
        )
        self.dec1 = DoubleConv(
            f * 2,
            f,
            norm_type=norm_type,
            num_groups=num_groups,
            dropout=dropout,
        )

        # -------------------------
        # High-resolution head
        # -------------------------
        self.hr_up = UpBlock(
            f,
            f,
            norm_type=norm_type,
            num_groups=num_groups,
            dropout=dropout,
        )

        self.hr_refine = nn.Sequential(
            DoubleConv(
                f,
                f,
                norm_type=norm_type,
                num_groups=num_groups,
                dropout=dropout,
            ),
            DoubleConv(
                f,
                f,
                norm_type=norm_type,
                num_groups=num_groups,
                dropout=dropout,
            ),
        )

        # -------------------------
        # Output heads
        # -------------------------
        if self.use_gaussian or out_channels == 2:
            self.head_mean = nn.Conv2d(f, 1, kernel_size=1)
            self.head_log_var = nn.Conv2d(f, 1, kernel_size=1)
        else:
            self.final_output = nn.Conv2d(f, out_channels, kernel_size=1)

    def forward(self, x, target_size):
        # Encoder
        c1 = self.enc1(x)
        p1 = self.pool(c1)

        c2 = self.enc2(p1)
        p2 = self.pool(c2)

        c3 = self.enc3(p2)
        p3 = self.pool(c3)

        # Bottleneck
        bn = self.bottleneck(p3)

        # Decoder
        u3 = self.up3(bn, target_size=c3.shape[-2:])
        d3 = self.dec3(torch.cat([u3, c3], dim=1))

        u2 = self.up2(d3, target_size=c2.shape[-2:])
        d2 = self.dec2(torch.cat([u2, c2], dim=1))

        u1 = self.up1(d2, target_size=c1.shape[-2:])
        d1 = self.dec1(torch.cat([u1, c1], dim=1))

        # Extension vers la grille haute résolution
        features = self.hr_up(d1, target_size=target_size)

        # Raffinement appris à haute résolution
        features = self.hr_refine(features)

        # Output
        if self.use_gaussian or self.out_channels == 2:
            mean = self.head_mean(features)
            log_var = self.head_log_var(features)
            log_var = torch.clamp(log_var, min=-10.0, max=10.0)

            out = torch.cat([mean, log_var], dim=1)
        else:
            out = self.final_output(features)

        return out