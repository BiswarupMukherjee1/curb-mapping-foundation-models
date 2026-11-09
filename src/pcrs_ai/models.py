"""Frozen foundation-model encoder + light trainable head for dense curb-edge prediction at 5 cm.

FrozenViT: DINOv3 (timm) with local fp16 weights, patch tokens of the last block -> (B, C, H/16, W/16).
EdgeHead : fuses upsampled ViT tokens with a small full-resolution RGB stem (needed because one 16 px token covers
           80 cm while the target tolerance is 10 cm). With use_fm=False the same head runs on RGB only, which
           isolates what the foundation model contributes.
"""
from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from .paths import MODELS, ROOT

# local fp16 copies of the DINOv3 weights (created by scripts/fetch_weights.py)
WEIGHTS_DIR = MODELS / "dinov3"
ENCODERS = {
    "dinov3_vitl16_sat": "vit_large_patch16_dinov3.sat493m",
    "dinov3_vitb16_web": "vit_base_patch16_dinov3.lvd1689m",
}


class FrozenViT(nn.Module):
    def __init__(self, name: str = "dinov3_vitl16_sat"):
        super().__init__()
        import timm
        from safetensors.torch import load_file
        arch = ENCODERS[name]
        local = WEIGHTS_DIR / f"{arch}.fp16.safetensors"
        self.model = timm.create_model(arch.split(".")[0], pretrained=False, num_classes=0)
        self.model.load_state_dict({k: v.float() for k, v in load_file(str(local)).items()}, strict=True)
        cfg = timm.models.get_pretrained_cfg(arch).to_dict()
        self.register_buffer("mean", torch.tensor(cfg["mean"]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(cfg["std"]).view(1, 3, 1, 1))
        self.dim = self.model.embed_dim
        self.n_blocks = len(self.model.blocks)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

    upsample: int = 1                                             # >1: encode a resized input (finer tokens)

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:          # x in [0, 1]
        x = x[:, :3]
        if self.upsample > 1:
            x = F.interpolate(x, scale_factor=self.upsample, mode="bilinear", align_corners=False)
        x = (x - self.mean) / self.std
        with torch.autocast("cuda", dtype=torch.float16):
            z = self.model.forward_intermediates(x, indices=[self.n_blocks - 1], norm=True, output_fmt="NCHW",
                                                 intermediates_only=True)[0]
        return z.float()


def conv(ci, co, s=1):
    return nn.Sequential(nn.Conv2d(ci, co, 3, s, 1, bias=False), nn.GroupNorm(8, co), nn.ReLU(inplace=True))


class EdgeHead(nn.Module):
    def __init__(self, fm_dim: int = 0, width: int = 64, n_out: int = 2):
        super().__init__()
        self.use_fm = fm_dim > 0
        self.s1 = conv(3, 32)                     # 1/1
        self.s2 = conv(32, width, 2)              # 1/2
        self.s4 = conv(width, width, 2)           # 1/4
        if self.use_fm:
            self.proj = nn.Sequential(nn.Conv2d(fm_dim, 128, 1), nn.GroupNorm(8, 128), nn.ReLU(inplace=True))
        self.f4 = conv(width + (128 if self.use_fm else 0), 96)
        self.f2 = conv(96 + width, 64)
        self.f1 = conv(64 + 32, 32)
        self.out = nn.Conv2d(32, n_out, 1)

    def forward(self, x: torch.Tensor, feats: torch.Tensor | None = None) -> torch.Tensor:
        x = x[:, :3]
        e1 = self.s1(x); e2 = self.s2(e1); e4 = self.s4(e2)
        h = e4
        if self.use_fm:
            f = F.interpolate(self.proj(feats), size=e4.shape[-2:], mode="bilinear", align_corners=False)
            h = torch.cat([e4, f], 1)
        h = self.f4(h)
        h = self.f2(torch.cat([F.interpolate(h, size=e2.shape[-2:], mode="bilinear", align_corners=False), e2], 1))
        h = self.f1(torch.cat([F.interpolate(h, size=e1.shape[-2:], mode="bilinear", align_corners=False), e1], 1))
        return self.out(h)


def n_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


def save_head(head: nn.Module, path: Path):
    torch.save(head.state_dict(), path)


class UNetR34(nn.Module):
    """Standard supervised baseline: U-Net with an ImageNet-pretrained ResNet-34 encoder, fine-tuned end to end.
    Same 2-channel output as EdgeHead; forward(x, feats=None) keeps the EdgeHead call signature."""

    def __init__(self, n_out: int = 2, pretrained: bool = True):
        super().__init__()
        import timm
        self.encoder = timm.create_model("resnet34.a1_in1k", pretrained=pretrained, features_only=True)
        ch = self.encoder.feature_info.channels()                  # [64, 64, 128, 256, 512] at 1/2 ... 1/32
        cfg = self.encoder.pretrained_cfg
        self.register_buffer("mean", torch.tensor(cfg["mean"]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(cfg["std"]).view(1, 3, 1, 1))
        self.dec4 = conv(ch[4] + ch[3], 256)
        self.dec3 = conv(256 + ch[2], 128)
        self.dec2 = conv(128 + ch[1], 64)
        self.dec1 = conv(64 + ch[0], 64)
        self.full = conv(64 + 3, 32)
        self.out = nn.Conv2d(32, n_out, 1)

    def forward(self, x: torch.Tensor, feats=None) -> torch.Tensor:
        x = x[:, :3]
        f = self.encoder((x - self.mean) / self.std)
        up = lambda a, b: F.interpolate(a, size=b.shape[-2:], mode="bilinear", align_corners=False)
        h = self.dec4(torch.cat([up(f[4], f[3]), f[3]], 1))
        h = self.dec3(torch.cat([up(h, f[2]), f[2]], 1))
        h = self.dec2(torch.cat([up(h, f[1]), f[1]], 1))
        h = self.dec1(torch.cat([up(h, f[0]), f[0]], 1))
        h = self.full(torch.cat([up(h, x), x], 1))
        return self.out(h)


class FrozenAnySat(nn.Module):
    """AnySat (Astruc et al., CVPR 2025), 'aerial' modality: RGB + NIR at 0.2 m, standardised per band.
    Input: (B, 4, H, W) in [0, 1] at 5 cm; resampled 4x to 0.2 m. Output: dense features (B, 1536, h, w),
    one vector per ~2.5 m cell (context only; far coarser than a curb)."""

    def __init__(self):
        super().__init__()
        self.model = torch.hub.load("gastruc/anysat", "anysat", pretrained=True, flash_attn=False, trust_repo=True)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.register_buffer("mean", torch.full((1, 4, 1, 1), 128.0))
        self.register_buffer("std", torch.full((1, 4, 1, 1), 50.0))
        self.dim = 1536

    def set_stats(self, mean, std):
        self.mean.copy_(torch.as_tensor(mean, dtype=torch.float32).view(1, 4, 1, 1))
        self.std.copy_(torch.as_tensor(std, dtype=torch.float32).view(1, 4, 1, 1))

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.avg_pool2d(x[:, :4] * 255.0, 4)
        x = (x - self.mean) / self.std
        f = self.model({"aerial": x}, patch_size=10, output="dense", output_modality="aerial")
        return f.permute(0, 3, 1, 2).float()


class FrozenDOFA(nn.Module):
    """DOFA ViT-B/16 (src/pcrs_ai/dofa.py): EO foundation model with a wavelength-conditioned patch embedding.
    Input: (B, 4, H, W) RGB + NIR in [0, 1] at 5 cm, standardised per band. Output: patch tokens (B, 768, H/16, W/16).
    Its finest pretraining imagery is 1 m NAIP, so 5 cm input is outside its training range."""

    WAVELENGTHS_UM = [0.665, 0.56, 0.49, 0.842]          # R, G, B, NIR band centres (approximate for an aerial camera)

    def __init__(self):
        super().__init__()
        from .dofa import load_dofa_base
        self.model = load_dofa_base(MODELS / "dofa" / "DOFA_ViT_base_e100.pth").eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.register_buffer("mean", torch.full((1, 4, 1, 1), 128.0))
        self.register_buffer("std", torch.full((1, 4, 1, 1), 50.0))
        self.register_buffer("waves", torch.tensor(self.WAVELENGTHS_UM))
        self.dim = 768

    def set_stats(self, mean, std):
        self.mean.copy_(torch.as_tensor(mean, dtype=torch.float32).view(1, 4, 1, 1))
        self.std.copy_(torch.as_tensor(std, dtype=torch.float32).view(1, 4, 1, 1))

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = (x[:, :4] * 255.0 - self.mean) / self.std
        with torch.autocast("cuda", dtype=torch.float16):
            return self.model(x, self.waves).float()


class FeatureStack(nn.Module):
    """Concatenate the features of several frozen encoders on the grid of the first one."""

    def __init__(self, encoders):
        super().__init__()
        self.encoders = nn.ModuleList(encoders)
        self.dim = sum(e.dim for e in encoders)

    @torch.no_grad()
    def forward(self, x):
        fs = [e(x) for e in self.encoders]
        size = fs[0].shape[-2:]
        return torch.cat([fs[0]] + [F.interpolate(f, size=size, mode="bilinear", align_corners=False) for f in fs[1:]], 1)


def needs_nir(encoder: str) -> bool:
    return "anysat" in encoder or "dofa" in encoder


def build(encoder: str, width: int = 64, upsample: int = 1, pretrained_unet: bool = True):
    """(frozen encoder or None, trainable head) for an experiment config.
    encoder: none | dinov3_vitl16_sat | dinov3_vitb16_web | anysat | dofa_b | any '+' combination | unet_r34"""
    if encoder == "unet_r34":
        return None, UNetR34(pretrained=pretrained_unet)
    if encoder == "none":
        return None, EdgeHead(fm_dim=0, width=width)
    parts = []
    for name in encoder.split("+"):
        if name == "anysat":
            parts.append(FrozenAnySat())
        elif name == "dofa_b":
            parts.append(FrozenDOFA())
        else:
            v = FrozenViT(name); v.upsample = upsample; parts.append(v)
    enc = parts[0] if len(parts) == 1 else FeatureStack(parts)
    return enc, EdgeHead(fm_dim=enc.dim, width=width)


def anysat_modules(enc):
    """Frozen encoders that need per-band standardisation statistics (AnySat, DOFA)."""
    if enc is None:
        return []
    mods = enc.encoders if isinstance(enc, FeatureStack) else [enc]
    return [e for e in mods if hasattr(e, "set_stats")]


def load_trained(run_dir, device: str = "cuda"):
    """Rebuild a trained model (frozen encoder + trained head) from its run folder."""
    import json
    from .paths import MODELS
    run_dir = Path(run_dir)
    cfg = json.loads((run_dir / "config.json").read_text())
    enc, head = build(cfg["encoder"], width=cfg.get("width", 64), upsample=cfg.get("upsample", 1), pretrained_unet=False)
    enc = enc.to(device) if enc is not None else None
    st = run_dir / "anysat_stats.json"
    for m in anysat_modules(enc):
        s = json.loads(st.read_text()); m.set_stats(s["mean"], s["std"])
    head = head.to(device)
    head.load_state_dict(torch.load(MODELS / f"head_{run_dir.name}.pt", map_location=device))
    head.eval()
    return cfg, enc, head
