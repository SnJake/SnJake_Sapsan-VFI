from __future__ import annotations

from typing import Tuple

import torch
from torch import nn
from torch.nn import functional as F


def conv3x3(in_ch: int, out_ch: int, stride: int = 1) -> nn.Module:
    return nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1)


def conv1x1(in_ch: int, out_ch: int) -> nn.Module:
    return nn.Conv2d(in_ch, out_ch, 1)


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__()
        self.conv = conv3x3(in_ch, out_ch, stride)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.conv(x))


class ResBlock(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.conv1 = conv3x3(ch, ch)
        self.act = nn.GELU()
        self.conv2 = conv3x3(ch, ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.act(self.conv1(x))
        out = self.conv2(out)
        return self.act(out + x)


def window_partition(x: torch.Tensor, window_size: int) -> Tuple[torch.Tensor, Tuple[int, int, int, int]]:
    b, c, h, w = x.shape
    pad_h = (window_size - (h % window_size)) % window_size
    pad_w = (window_size - (w % window_size)) % window_size
    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left
    if pad_h > 0 or pad_w > 0:
        x = F.pad(x, (pad_left, pad_right, pad_top, pad_bottom), mode="reflect")
    _, _, hp, wp = x.shape
    x = x.view(b, c, hp // window_size, window_size, wp // window_size, window_size)
    x = x.permute(0, 2, 4, 3, 5, 1).contiguous()
    windows = x.view(-1, window_size * window_size, c)
    return windows, (pad_left, pad_right, pad_top, pad_bottom)


def window_reverse(
    windows: torch.Tensor,
    window_size: int,
    original_hw: Tuple[int, int],
    pad: Tuple[int, int, int, int],
) -> torch.Tensor:
    pad_left, pad_right, pad_top, pad_bottom = pad
    h, w = original_hw
    hp = h + pad_top + pad_bottom
    wp = w + pad_left + pad_right
    num_windows = (hp // window_size) * (wp // window_size)
    b = int(windows.shape[0] // num_windows)
    x = windows.view(b, hp // window_size, wp // window_size, window_size, window_size, -1)
    x = x.permute(0, 5, 1, 3, 2, 4).contiguous()
    x = x.view(b, -1, hp, wp)
    if pad_top or pad_bottom or pad_left or pad_right:
        x = x[..., pad_top : hp - pad_bottom, pad_left : wp - pad_right]
    return x


class WindowAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, window_size: int):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        windows, pad = window_partition(x, self.window_size)
        qkv = self.qkv(windows)
        qkv = qkv.view(qkv.shape[0], qkv.shape[1], 3, self.num_heads, c // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = F.scaled_dot_product_attention(q, k, v)
        attn = attn.permute(0, 2, 1, 3).contiguous().view(windows.shape[0], windows.shape[1], c)
        out = self.proj(attn)
        out = window_reverse(out, self.window_size, (h, w), pad)
        return out


class TransformerFusionBlock(nn.Module):
    def __init__(self, dim: int, heads: int, window_size: int):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, dim)
        self.attn = WindowAttention(dim, heads, window_size)
        self.norm2 = nn.GroupNorm(8, dim)
        self.mlp = nn.Sequential(
            nn.Conv2d(dim, dim * 4, 1),
            nn.GELU(),
            nn.Conv2d(dim * 4, dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class UpBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv = nn.Sequential(
            conv3x3(in_ch, out_ch),
            nn.GELU(),
            conv3x3(out_ch, out_ch),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        return self.conv(x)


class DownBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv = nn.Sequential(
            conv3x3(in_ch, out_ch, stride=2),
            nn.GELU(),
            conv3x3(out_ch, out_ch),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)
