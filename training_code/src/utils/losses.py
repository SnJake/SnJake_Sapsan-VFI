from __future__ import annotations

from typing import Iterable, Optional, Sequence, Tuple

import torch
from torch import nn
from torch.nn import functional as F


def charbonnier_loss(x: torch.Tensor, y: torch.Tensor, eps: float = 1e-3) -> torch.Tensor:
    diff = x - y
    loss = torch.sqrt(diff * diff + eps * eps)
    return loss.mean()


def ssim(x: torch.Tensor, y: torch.Tensor, window_size: int = 11, C1: float = 0.01 ** 2, C2: float = 0.03 ** 2):
    pad = window_size // 2
    mu_x = F.avg_pool2d(x, window_size, stride=1, padding=pad)
    mu_y = F.avg_pool2d(y, window_size, stride=1, padding=pad)
    sigma_x = F.avg_pool2d(x * x, window_size, stride=1, padding=pad) - mu_x * mu_x
    sigma_y = F.avg_pool2d(y * y, window_size, stride=1, padding=pad) - mu_y * mu_y
    sigma_xy = F.avg_pool2d(x * y, window_size, stride=1, padding=pad) - mu_x * mu_y
    ssim_map = ((2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)) / ((mu_x * mu_x + mu_y * mu_y + C1) * (sigma_x + sigma_y + C2))
    return ssim_map.mean()


def flow_smoothness(flow: torch.Tensor) -> torch.Tensor:
    dx = torch.abs(flow[:, :, :, 1:] - flow[:, :, :, :-1])
    dy = torch.abs(flow[:, :, 1:, :] - flow[:, :, :-1, :])
    return dx.mean() + dy.mean()


def edge_aware_flow_smoothness(flow: torch.Tensor, image: torch.Tensor, alpha: float = 10.0) -> torch.Tensor:
    img_gray = image.mean(dim=1, keepdim=True)
    img_dx = torch.abs(img_gray[:, :, :, 1:] - img_gray[:, :, :, :-1])
    img_dy = torch.abs(img_gray[:, :, 1:, :] - img_gray[:, :, :-1, :])
    weight_x = torch.exp(-alpha * img_dx)
    weight_y = torch.exp(-alpha * img_dy)
    flow_dx = torch.abs(flow[:, :, :, 1:] - flow[:, :, :, :-1])
    flow_dy = torch.abs(flow[:, :, 1:, :] - flow[:, :, :-1, :])
    return (flow_dx * weight_x).mean() + (flow_dy * weight_y).mean()


def gradient_loss(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    x_dx = x[:, :, :, 1:] - x[:, :, :, :-1]
    y_dx = y[:, :, :, 1:] - y[:, :, :, :-1]
    x_dy = x[:, :, 1:, :] - x[:, :, :-1, :]
    y_dy = y[:, :, 1:, :] - y[:, :, :-1, :]
    return F.l1_loss(x_dx, y_dx) + F.l1_loss(x_dy, y_dy)


def laplacian_loss(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    kernel = torch.tensor(
        [[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]],
        device=x.device,
        dtype=x.dtype,
    ).view(1, 1, 3, 3)
    channels = x.shape[1]
    kernel = kernel.repeat(channels, 1, 1, 1)
    x_lap = F.conv2d(x, kernel, padding=1, groups=channels)
    y_lap = F.conv2d(y, kernel, padding=1, groups=channels)
    return F.l1_loss(x_lap, y_lap)


def _channel_stats(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    mean = x.mean(dim=(2, 3))
    std = x.std(dim=(2, 3), unbiased=False)
    return mean, std


def color_consistency_loss(x: torch.Tensor, y: torch.Tensor, use_std: bool = True) -> torch.Tensor:
    mean_x, std_x = _channel_stats(x)
    mean_y, std_y = _channel_stats(y)
    loss = F.l1_loss(mean_x, mean_y)
    if use_std:
        loss = loss + F.l1_loss(std_x, std_y)
    return loss


def brightness_consistency_loss(x: torch.Tensor, y: torch.Tensor, use_std: bool = True) -> torch.Tensor:
    if x.shape[1] != 3 or y.shape[1] != 3:
        raise ValueError("Brightness consistency expects RGB tensors with 3 channels")
    coeffs = torch.tensor([0.299, 0.587, 0.114], device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
    luma_x = (x * coeffs).sum(dim=1, keepdim=True)
    luma_y = (y * coeffs).sum(dim=1, keepdim=True)
    mean_x, std_x = _channel_stats(luma_x)
    mean_y, std_y = _channel_stats(luma_y)
    loss = F.l1_loss(mean_x, mean_y)
    if use_std:
        loss = loss + F.l1_loss(std_x, std_y)
    return loss


class VGGPerceptualLoss(nn.Module):
    def __init__(self, layer_ids: Sequence[int]):
        super().__init__()
        try:
            from torchvision.models import VGG16_Weights, vgg16
        except Exception as exc:
            raise RuntimeError("torchvision is required for perceptual loss") from exc

        vgg = vgg16(weights=VGG16_Weights.IMAGENET1K_FEATURES).features
        for param in vgg.parameters():
            param.requires_grad = False
        self.vgg = vgg.eval()
        self.layer_ids = tuple(int(x) for x in layer_ids)
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        if x.shape[1] != 3 or y.shape[1] != 3:
            raise ValueError("Perceptual loss expects RGB tensors with 3 channels")
        x = (x - self.mean) / self.std
        y = (y - self.mean) / self.std
        loss = 0.0
        feat_x = x
        feat_y = y
        max_layer = max(self.layer_ids)
        for idx, layer in enumerate(self.vgg):
            feat_x = layer(feat_x)
            feat_y = layer(feat_y)
            if idx in self.layer_ids:
                loss = loss + F.l1_loss(feat_x, feat_y)
            if idx >= max_layer:
                break
        return loss


class LPIPSLoss(nn.Module):
    def __init__(self, net: str = "vgg"):
        super().__init__()
        try:
            import lpips
        except Exception as exc:
            raise RuntimeError("lpips is required for LPIPS loss") from exc
        self.fn = lpips.LPIPS(net=net)
        for param in self.fn.parameters():
            param.requires_grad = False

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self.fn(x * 2.0 - 1.0, y * 2.0 - 1.0).mean()


def build_perceptual_loss(
    loss_cfg: dict,
    device: torch.device,
) -> Optional[nn.Module]:
    if float(loss_cfg.get("perceptual_weight", 0.0)) <= 0.0:
        return None
    loss_type = str(loss_cfg.get("perceptual_type", "vgg")).lower()
    if loss_type == "lpips":
        net = str(loss_cfg.get("lpips_net", "vgg"))
        loss = LPIPSLoss(net=net)
    else:
        layers = loss_cfg.get("perceptual_layers", [3, 8, 15])
        loss = VGGPerceptualLoss(layer_ids=layers)
    return loss.to(device)
