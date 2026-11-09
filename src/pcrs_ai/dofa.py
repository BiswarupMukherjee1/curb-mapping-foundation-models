"""DOFA ViT-B/16 encoder (Xiong et al., "Neural plasticity-inspired multimodal foundation model for Earth
observation", 2024). Architecture ported from github.com/zhu-xlab/DOFA (MIT licence) so that only the published
weights are loaded, with weights_only=True.

DOFA generates its patch-embedding kernel from the band wavelengths, so any set of bands can be encoded. It was
pretrained on Sentinel-1/2, NAIP (1 m RGB + NIR), Gaofen and EnMAP.
"""
from __future__ import annotations

from functools import partial
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

WEIGHTS_URL = "https://huggingface.co/earthflow/DOFA/resolve/main/DOFA_ViT_base_e100.pth"


def _sincos_1d(dim: int, pos: torch.Tensor) -> torch.Tensor:
    omega = torch.arange(dim // 2, dtype=torch.float32, device=pos.device) / (dim / 2.0)
    omega = 1.0 / 10000 ** omega
    out = torch.einsum("m,d->md", pos.reshape(-1).float(), omega)
    return torch.cat([torch.sin(out), torch.cos(out)], dim=1)


class _WeightGenerator(nn.Module):
    """Small transformer that turns wavelength embeddings into patch-embedding kernels and a bias."""

    def __init__(self, input_dim: int, output_dim: int, embed_dim: int, num_heads: int = 4, n_weight_tokens: int = 128):
        super().__init__()
        layer = nn.TransformerEncoderLayer(d_model=input_dim, nhead=num_heads, activation="gelu", norm_first=False,
                                           batch_first=False, dropout=0.0)
        self.transformer_encoder = nn.TransformerEncoder(layer, num_layers=1, enable_nested_tensor=False)
        self.fc_weight = nn.Linear(input_dim, output_dim)
        self.fc_bias = nn.Linear(input_dim, embed_dim)
        self.wt_num = n_weight_tokens
        self.weight_tokens = nn.Parameter(torch.zeros(n_weight_tokens, input_dim))
        self.bias_token = nn.Parameter(torch.zeros(1, input_dim))

    def forward(self, wave_emb: torch.Tensor):
        x = torch.cat([self.weight_tokens, wave_emb, self.bias_token], dim=0)
        out = self.transformer_encoder(x)
        weights = self.fc_weight(out[self.wt_num:-1] + wave_emb)
        bias = self.fc_bias(out[-1])
        return weights, bias


class _FCRes(nn.Module):
    def __init__(self, n: int = 128):
        super().__init__()
        self.w1, self.w2 = nn.Linear(n, n), nn.Linear(n, n)

    def forward(self, x):
        return x + F.relu(self.w2(F.relu(self.w1(x))))


class _DynamicPatchEmbed(nn.Module):
    def __init__(self, wv_planes: int = 128, kernel_size: int = 16, embed_dim: int = 768):
        super().__init__()
        self.kernel_size, self.wv_planes, self.embed_dim = kernel_size, wv_planes, embed_dim
        self.weight_generator = _WeightGenerator(wv_planes, kernel_size * kernel_size * embed_dim, embed_dim)
        self.fclayer = _FCRes(wv_planes)
        self.scaler = 0.01

    def forward(self, img: torch.Tensor, wavelengths_um: torch.Tensor) -> torch.Tensor:
        n = wavelengths_um.numel()
        w_emb = self.fclayer(_sincos_1d(self.wv_planes, wavelengths_um * 1000))
        weight, bias = self.weight_generator(w_emb)
        kernel = weight.view(n, self.kernel_size, self.kernel_size, self.embed_dim).permute(3, 0, 1, 2) * self.scaler
        out = F.conv2d(img, kernel, bias=bias.view(self.embed_dim) * self.scaler, stride=self.kernel_size, padding=1)
        return out.flatten(2).transpose(1, 2)


class DOFAViTBase(nn.Module):
    """ViT-B/16 with the DOFA dynamic patch embedding; returns patch tokens (B, 768, H/16, W/16)."""

    def __init__(self, img_size: int = 224, embed_dim: int = 768, depth: int = 12, num_heads: int = 12):
        super().__init__()
        from timm.models.vision_transformer import Block
        self.patch_embed = _DynamicPatchEmbed(128, 16, embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, (img_size // 16) ** 2 + 1, embed_dim), requires_grad=False)
        norm = partial(nn.LayerNorm, eps=1e-6)
        self.blocks = nn.ModuleList([Block(embed_dim, num_heads, 4.0, qkv_bias=True, norm_layer=norm) for _ in range(depth)])
        self.norm = norm(embed_dim)

    def forward(self, x: torch.Tensor, wavelengths_um: torch.Tensor) -> torch.Tensor:
        h, w = x.shape[-2] // 16, x.shape[-1] // 16
        t = self.patch_embed(x, wavelengths_um)
        pos = self.pos_embed[:, 1:, :]
        g = int(round(pos.shape[1] ** 0.5))
        pos = F.interpolate(pos.reshape(1, g, g, -1).permute(0, 3, 1, 2), size=(h, w), mode="bicubic",
                            align_corners=False).permute(0, 2, 3, 1).reshape(1, h * w, -1)
        t = t[:, : h * w] + pos
        cls = (self.cls_token + self.pos_embed[:, :1, :]).expand(t.shape[0], -1, -1)
        t = torch.cat([cls, t], 1)
        for blk in self.blocks:
            t = blk(t)
        t = self.norm(t[:, 1:])
        return t.transpose(1, 2).reshape(x.shape[0], -1, h, w)


def load_dofa_base(weights_path: Path) -> DOFAViTBase:
    model = DOFAViTBase()
    sd = torch.load(weights_path, map_location="cpu", weights_only=True)
    sd = sd.get("model", sd)
    skip = ("head.", "mask_token", "projector.")            # pretraining-only parameters
    missing, unexpected = model.load_state_dict({k: v for k, v in sd.items() if not k.startswith(skip)}, strict=False)
    if missing:
        raise RuntimeError(f"DOFA weights: missing keys {missing[:8]}")
    return model
