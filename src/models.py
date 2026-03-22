import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class ResidualBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.shortcut = nn.Identity() if in_ch == out_ch else nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.shortcut(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.relu(out + identity)
        return out


class UpBlock(nn.Module):
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int, conv_block=DoubleConv):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
        self.conv = conv_block(out_ch + skip_ch, out_ch)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = nn.functional.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)


class AttentionGate(nn.Module):
    def __init__(self, g_ch: int, x_ch: int, inter_ch: int):
        super().__init__()
        self.g_proj = nn.Sequential(nn.Conv2d(g_ch, inter_ch, 1, bias=False), nn.BatchNorm2d(inter_ch))
        self.x_proj = nn.Sequential(nn.Conv2d(x_ch, inter_ch, 1, bias=False), nn.BatchNorm2d(inter_ch))
        self.psi = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_ch, 1, 1, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, g: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        w = self.psi(self.g_proj(g) + self.x_proj(x))
        return x * w


class AttentionUpBlock(nn.Module):
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int, conv_block=DoubleConv):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
        self.attn = AttentionGate(g_ch=out_ch, x_ch=skip_ch, inter_ch=max(out_ch // 2, 1))
        self.conv = conv_block(out_ch + skip_ch, out_ch)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = nn.functional.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        skip = self.attn(x, skip)
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)


class ASPP(nn.Module):
    def __init__(self, channels: int, dilations=(1, 6, 12, 18)):
        super().__init__()
        self.branches = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(
                        channels,
                        channels,
                        kernel_size=1 if d == 1 else 3,
                        padding=0 if d == 1 else d,
                        dilation=d,
                        bias=False,
                    ),
                    nn.BatchNorm2d(channels),
                    nn.ReLU(inplace=True),
                )
                for d in dilations
            ]
        )
        self.project = nn.Sequential(
            nn.Conv2d(channels * len(dilations), channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = [branch(x) for branch in self.branches]
        return self.project(torch.cat(feats, dim=1))


class UNet(nn.Module):
    def __init__(self, in_channels: int = 3, out_channels: int = 1, base_channels: int = 32):
        super().__init__()
        c = base_channels
        self.e1 = DoubleConv(in_channels, c)
        self.e2 = DoubleConv(c, c * 2)
        self.e3 = DoubleConv(c * 2, c * 4)
        self.e4 = DoubleConv(c * 4, c * 8)
        self.b = DoubleConv(c * 8, c * 16)

        self.pool = nn.MaxPool2d(2)

        self.d4 = UpBlock(c * 16, c * 8, c * 8)
        self.d3 = UpBlock(c * 8, c * 4, c * 4)
        self.d2 = UpBlock(c * 4, c * 2, c * 2)
        self.d1 = UpBlock(c * 2, c, c)

        self.head = nn.Conv2d(c, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.e1(x)
        e2 = self.e2(self.pool(e1))
        e3 = self.e3(self.pool(e2))
        e4 = self.e4(self.pool(e3))
        b = self.b(self.pool(e4))

        d4 = self.d4(b, e4)
        d3 = self.d3(d4, e3)
        d2 = self.d2(d3, e2)
        d1 = self.d1(d2, e1)
        return self.head(d1)


class AttentionUNet(nn.Module):
    def __init__(self, in_channels: int = 3, out_channels: int = 1, base_channels: int = 32):
        super().__init__()
        c = base_channels
        self.e1 = DoubleConv(in_channels, c)
        self.e2 = DoubleConv(c, c * 2)
        self.e3 = DoubleConv(c * 2, c * 4)
        self.e4 = DoubleConv(c * 4, c * 8)
        self.b = DoubleConv(c * 8, c * 16)

        self.pool = nn.MaxPool2d(2)

        self.d4 = AttentionUpBlock(c * 16, c * 8, c * 8)
        self.d3 = AttentionUpBlock(c * 8, c * 4, c * 4)
        self.d2 = AttentionUpBlock(c * 4, c * 2, c * 2)
        self.d1 = AttentionUpBlock(c * 2, c, c)

        self.head = nn.Conv2d(c, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.e1(x)
        e2 = self.e2(self.pool(e1))
        e3 = self.e3(self.pool(e2))
        e4 = self.e4(self.pool(e3))
        b = self.b(self.pool(e4))

        d4 = self.d4(b, e4)
        d3 = self.d3(d4, e3)
        d2 = self.d2(d3, e2)
        d1 = self.d1(d2, e1)
        return self.head(d1)


class ResidualUNet(nn.Module):
    def __init__(self, in_channels: int = 3, out_channels: int = 1, base_channels: int = 32):
        super().__init__()
        c = base_channels
        self.e1 = ResidualBlock(in_channels, c)
        self.e2 = ResidualBlock(c, c * 2)
        self.e3 = ResidualBlock(c * 2, c * 4)
        self.e4 = ResidualBlock(c * 4, c * 8)
        self.b = ResidualBlock(c * 8, c * 16)

        self.pool = nn.MaxPool2d(2)

        self.d4 = UpBlock(c * 16, c * 8, c * 8, conv_block=ResidualBlock)
        self.d3 = UpBlock(c * 8, c * 4, c * 4, conv_block=ResidualBlock)
        self.d2 = UpBlock(c * 4, c * 2, c * 2, conv_block=ResidualBlock)
        self.d1 = UpBlock(c * 2, c, c, conv_block=ResidualBlock)

        self.head = nn.Conv2d(c, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.e1(x)
        e2 = self.e2(self.pool(e1))
        e3 = self.e3(self.pool(e2))
        e4 = self.e4(self.pool(e3))
        b = self.b(self.pool(e4))

        d4 = self.d4(b, e4)
        d3 = self.d3(d4, e3)
        d2 = self.d2(d3, e2)
        d1 = self.d1(d2, e1)
        return self.head(d1)


class AttentionASPPUNet(nn.Module):
    def __init__(self, in_channels: int = 3, out_channels: int = 1, base_channels: int = 32):
        super().__init__()
        c = base_channels
        self.e1 = DoubleConv(in_channels, c)
        self.e2 = DoubleConv(c, c * 2)
        self.e3 = DoubleConv(c * 2, c * 4)
        self.e4 = DoubleConv(c * 4, c * 8)
        self.b = nn.Sequential(DoubleConv(c * 8, c * 16), ASPP(c * 16))

        self.pool = nn.MaxPool2d(2)

        self.d4 = AttentionUpBlock(c * 16, c * 8, c * 8)
        self.d3 = AttentionUpBlock(c * 8, c * 4, c * 4)
        self.d2 = AttentionUpBlock(c * 4, c * 2, c * 2)
        self.d1 = AttentionUpBlock(c * 2, c, c)

        self.head = nn.Conv2d(c, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.e1(x)
        e2 = self.e2(self.pool(e1))
        e3 = self.e3(self.pool(e2))
        e4 = self.e4(self.pool(e3))
        b = self.b(self.pool(e4))

        d4 = self.d4(b, e4)
        d3 = self.d3(d4, e3)
        d2 = self.d2(d3, e2)
        d1 = self.d1(d2, e1)
        return self.head(d1)


def build_model(name: str, in_channels: int, out_channels: int, base_channels: int) -> nn.Module:
    name = name.lower()
    if name == "unet":
        return UNet(in_channels, out_channels, base_channels)
    if name == "attention_unet":
        return AttentionUNet(in_channels, out_channels, base_channels)
    if name == "residual_unet":
        return ResidualUNet(in_channels, out_channels, base_channels)
    if name == "attention_aspp_unet":
        return AttentionASPPUNet(in_channels, out_channels, base_channels)
    raise ValueError(f"Unsupported model name: {name}")
