from __future__ import annotations

from typing import Dict, Tuple

import torch
from torch import nn
from torch.nn import functional as F

from .modules import ConvBlock, ResBlock, DownBlock, UpBlock, TransformerFusionBlock, conv3x3


def flow_warp(x: torch.Tensor, flow: torch.Tensor) -> torch.Tensor:
    b, c, h, w = x.shape
    grid_y, grid_x = torch.meshgrid(
        torch.arange(h, device=x.device),
        torch.arange(w, device=x.device),
        indexing="ij",
    )
    grid = torch.stack((grid_x, grid_y), dim=0).float()  # 2, H, W
    grid = grid.unsqueeze(0).repeat(b, 1, 1, 1)
    vgrid = grid + flow
    vgrid[:, 0, :, :] = 2.0 * vgrid[:, 0, :, :] / max(w - 1, 1) - 1.0
    vgrid[:, 1, :, :] = 2.0 * vgrid[:, 1, :, :] / max(h - 1, 1) - 1.0
    vgrid = vgrid.permute(0, 2, 3, 1)
    return F.grid_sample(x, vgrid, mode="bilinear", padding_mode="border", align_corners=True)


class FeaturePyramid(nn.Module):
    def __init__(self, in_ch: int, base_ch: int, num_scales: int):
        super().__init__()
        self.num_scales = num_scales
        self.stem = nn.Sequential(
            conv3x3(in_ch, base_ch),
            nn.GELU(),
            conv3x3(base_ch, base_ch),
            nn.GELU(),
        )
        downs = []
        ch = base_ch
        for _ in range(1, num_scales):
            downs.append(DownBlock(ch, ch * 2))
            ch *= 2
        self.downs = nn.ModuleList(downs)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        feats = []
        x = self.stem(x)
        feats.append(x)
        for down in self.downs:
            x = down(x)
            feats.append(x)
        return tuple(feats)


class FlowRefineBlock(nn.Module):
    def __init__(self, in_ch: int, hidden_ch: int, depth: int):
        super().__init__()
        layers = [ConvBlock(in_ch, hidden_ch)]
        for _ in range(depth):
            layers.append(ResBlock(hidden_ch))
        layers.append(conv3x3(hidden_ch, 4))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class FlowEstimator(nn.Module):
    def __init__(self, channels: Tuple[int, ...], depth: int):
        super().__init__()
        self.blocks = nn.ModuleList()
        for ch in channels:
            in_ch = ch * 2 + 4
            hidden = max(32, ch)
            self.blocks.append(FlowRefineBlock(in_ch, hidden, depth))

    def forward(self, feats0: Tuple[torch.Tensor, ...], feats1: Tuple[torch.Tensor, ...]) -> Tuple[torch.Tensor, torch.Tensor]:
        flow01 = None
        flow10 = None
        num_scales = len(feats0)
        for scale in reversed(range(num_scales)):
            f0 = feats0[scale]
            f1 = feats1[scale]
            if flow01 is None:
                flow01 = torch.zeros((f0.shape[0], 2, f0.shape[2], f0.shape[3]), device=f0.device)
                flow10 = torch.zeros_like(flow01)
            else:
                flow01 = F.interpolate(flow01, size=f0.shape[-2:], mode="bilinear", align_corners=False) * 2.0
                flow10 = F.interpolate(flow10, size=f0.shape[-2:], mode="bilinear", align_corners=False) * 2.0
            inp = torch.cat([f0, f1, flow01, flow10], dim=1)
            delta = self.blocks[scale](inp)
            flow01 = flow01 + delta[:, :2]
            flow10 = flow10 + delta[:, 2:4]
        return flow01, flow10


class FusionNet(nn.Module):
    def __init__(
        self,
        in_ch: int,
        base_ch: int,
        depth: int,
        use_attention: bool,
        attn_heads: int,
        attn_window: int,
        out_ch: int,
    ):
        super().__init__()
        self.in_conv = nn.Sequential(conv3x3(in_ch, base_ch), nn.GELU())
        self.down1 = DownBlock(base_ch, base_ch * 2)
        self.down2 = DownBlock(base_ch * 2, base_ch * 4)
        self.down3 = DownBlock(base_ch * 4, base_ch * 8)

        blocks = [ResBlock(base_ch * 8) for _ in range(depth)]
        if use_attention:
            blocks.append(TransformerFusionBlock(base_ch * 8, attn_heads, attn_window))
        self.bottleneck = nn.Sequential(*blocks)

        self.up2 = UpBlock(base_ch * 8, base_ch * 4)
        self.up1 = UpBlock(base_ch * 4, base_ch * 2)
        self.up0 = UpBlock(base_ch * 2, base_ch)
        self.out_conv = conv3x3(base_ch, out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x0 = self.in_conv(x)
        d1 = self.down1(x0)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        b = self.bottleneck(d3)
        u2 = self.up2(b) + d2
        u1 = self.up1(u2) + d1
        u0 = self.up0(u1) + x0
        return self.out_conv(u0)


class VFIModel(nn.Module):
    def __init__(self, cfg: Dict):
        super().__init__()
        base_ch = int(cfg.get("base_channels", 48))
        num_scales = int(cfg.get("num_scales", 4))
        flow_depth = int(cfg.get("flow_refine_depth", 2))
        fusion_depth = int(cfg.get("fusion_depth", 2))
        use_attention = bool(cfg.get("use_attention", True))
        attn_heads = int(cfg.get("attention_heads", 4))
        attn_window = int(cfg.get("attention_window", 8))
        use_occ = bool(cfg.get("use_occlusion_mask", True))
        fusion_out_ch = 6 if use_occ else 4
        self.use_occlusion = use_occ

        self.fpn = FeaturePyramid(3, base_ch, num_scales)
        channels = tuple(base_ch * (2**i) for i in range(num_scales))
        self.flow = FlowEstimator(channels, flow_depth)

        in_ch = 3 * 4 + 4 + 1
        self.fusion = FusionNet(
            in_ch=in_ch,
            base_ch=base_ch,
            depth=fusion_depth,
            use_attention=use_attention,
            attn_heads=attn_heads,
            attn_window=attn_window,
            out_ch=fusion_out_ch,
        )

    def forward(self, frame0: torch.Tensor, frame1: torch.Tensor, t: float = 0.5):
        feats0 = self.fpn(frame0)
        feats1 = self.fpn(frame1)
        flow01, flow10 = self.flow(feats0, feats1)

        t_map = torch.full((frame0.shape[0], 1, frame0.shape[2], frame0.shape[3]), t, device=frame0.device)
        flow_t0 = flow01 * t
        flow_t1 = flow10 * (1.0 - t)
        warp0 = flow_warp(frame0, flow_t0)
        warp1 = flow_warp(frame1, flow_t1)
        warp0_raw = warp0
        warp1_raw = warp1

        fusion_inp = torch.cat([frame0, frame1, warp0, warp1, flow01, flow10, t_map], dim=1)
        fusion_out = self.fusion(fusion_inp)
        residual = fusion_out[:, :3]
        alpha = torch.sigmoid(fusion_out[:, 3:4])
        occ0 = None
        occ1 = None
        if self.use_occlusion:
            occ = torch.sigmoid(fusion_out[:, 4:6])
            occ0 = occ[:, 0:1]
            occ1 = occ[:, 1:2]
            warp0 = warp0 * occ0
            warp1 = warp1 * occ1
        out = alpha * warp0 + (1.0 - alpha) * warp1 + residual
        out = out.clamp(0.0, 1.0)

        aux = {
            "flow01": flow01,
            "flow10": flow10,
            "alpha": alpha,
            "residual": residual,
            "warp0": warp0,
            "warp1": warp1,
            "warp0_raw": warp0_raw,
            "warp1_raw": warp1_raw,
            "occ0": occ0,
            "occ1": occ1,
        }
        return out, aux
