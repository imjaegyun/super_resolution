# torchrun --nproc_per_node=2 lmsr_ddp.py  --root /home/user/im_ig/SR/super_resolution/LSDIR  --json /home/user/im_ig/SR/super_resolution/LSDIR/data/train.json  --batch 32  --epochs 10  --allow_bicubic_fallback  --save_dir .\checkpoints_wavelet


import os, math, json, argparse, random
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision.io import read_image
from torchvision.transforms.functional import convert_image_dtype
from torchvision import models
from PIL import Image

from diffusers import AutoencoderKL

# =========================
# 0) Utils & Wavelet Transform
# =========================
def set_seed(seed: int = 42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def imresize_bicubic(img: Image.Image, scale: int) -> Image.Image:
    w, h = img.size
    return img.resize((w // scale, h // scale), Image.BICUBIC)


class HaarWaveletTransform(nn.Module):
    """2D Haar Wavelet Transform for frequency decomposition"""
    def __init__(self):
        super().__init__()
        # Haar wavelet filters
        self.register_buffer('ll', torch.tensor([[0.5, 0.5], [0.5, 0.5]]))
        self.register_buffer('lh', torch.tensor([[0.5, 0.5], [-0.5, -0.5]]))
        self.register_buffer('hl', torch.tensor([[0.5, -0.5], [0.5, -0.5]]))
        self.register_buffer('hh', torch.tensor([[0.5, -0.5], [-0.5, 0.5]]))
    
    def forward(self, x):
        """
        x: [B, C, H, W]
        Returns: LL (approx), LH, HL, HH (details) each [B, C, H/2, W/2]
        """
        B, C, H, W = x.shape
        
        # Create filter banks [4, 1, 2, 2]
        filters = torch.stack([self.ll, self.lh, self.hl, self.hh]).unsqueeze(1)
        
        # Apply to each channel separately
        coeffs = []
        for c in range(C):
            x_c = x[:, c:c+1, :, :]  # [B, 1, H, W]
            coeff = F.conv2d(x_c, filters, stride=2, padding=0)  # [B, 4, H/2, W/2]
            coeffs.append(coeff)
        
        coeffs = torch.cat(coeffs, dim=1)  # [B, C*4, H/2, W/2]
        
        # Split into 4 components
        LL = coeffs[:, 0::4, :, :]  # [B, C, H/2, W/2]
        LH = coeffs[:, 1::4, :, :]
        HL = coeffs[:, 2::4, :, :]
        HH = coeffs[:, 3::4, :, :]
        
        return LL, LH, HL, HH


class InverseHaarWavelet(nn.Module):
    """Inverse 2D Haar Wavelet Transform"""
    def __init__(self):
        super().__init__()
        self.register_buffer('ll', torch.tensor([[1., 1.], [1., 1.]]))
        self.register_buffer('lh', torch.tensor([[1., 1.], [-1., -1.]]))
        self.register_buffer('hl', torch.tensor([[1., -1.], [1., -1.]]))
        self.register_buffer('hh', torch.tensor([[1., -1.], [-1., 1.]]))
    
    def forward(self, LL, LH, HL, HH):
        """
        Reconstruct from wavelet coefficients
        """
        B, C, H, W = LL.shape
        
        # Interleave coefficients
        coeffs = torch.stack([LL, LH, HL, HH], dim=2)  # [B, C, 4, H, W]
        coeffs = coeffs.view(B, C*4, H, W)
        
        # Create inverse filter banks
        filters = torch.stack([self.ll, self.lh, self.hl, self.hh]).unsqueeze(1) * 0.25
        
        # Reconstruct each channel
        recon = []
        for c in range(C):
            c_coeffs = coeffs[:, c*4:(c+1)*4, :, :]  # [B, 4, H, W]
            c_recon = F.conv_transpose2d(c_coeffs, filters, stride=2, padding=0)
            recon.append(c_recon[:, 0:1, :, :])  # Take first channel
        
        return torch.cat(recon, dim=1)  # [B, C, H*2, W*2]


# =========================
# 1) Dataset (동일)
# =========================
class LSDIRSRPairsFromJSON(Dataset):
    """LSDIR Dataset for Super-Resolution pairs"""
    def __init__(
        self,
        root: str,
        json_path: Optional[str],
        scale: int = 4,
        split: str = "train",
        to_tensor: bool = True,
        limit: Optional[int] = None,
        strict: bool = False,
        allow_bicubic_fallback: bool = False,
        hr_size: int = 256,
    ):
        assert scale in (2, 3, 4)
        assert split in ("train", "val")
        self.root = Path(root)
        self.scale = scale
        self.split = split
        self.to_tensor = to_tensor
        self.allow_bicubic_fallback = allow_bicubic_fallback
        self.hr_size = hr_size
        self.lr_size = hr_size // scale
        self.items = []

        if split == "train":
            if json_path is None:
                raise ValueError("train split requires --json path")
            self._build_train(json_path, limit, strict)
        else:
            self._build_val(limit, strict)

        if not self.items:
            raise RuntimeError("No valid (LR, HR) pairs found")

    def _build_train(self, json_path: str, limit: Optional[int], strict: bool):
        with open(json_path, "r", encoding="utf-8") as f:
            meta = json.load(f)

        miss_lr = miss_hr = 0
        found_first_lr = False
        
        for m in meta:
            # JSON 형식 감지: path_gt/path_lq vs path
            if "path_gt" in m and "path_lq" in m:
                # train_X2.json, train_X3.json, train_X4.json 형식
                hr_rel_path = m["path_gt"]
                lr_rel_path = m["path_lq"]
                has_lr_in_json = True
            elif "path" in m:
                # train.json 형식 (HR만)
                hr_rel_path = m["path"]
                lr_rel_path = None
                has_lr_in_json = False
            else:
                continue
            
            # HR 경로 찾기
            hr_candidates = [
                self.root / hr_rel_path,
                Path(hr_rel_path),  # 절대 경로일 수도
            ]
            
            hr_path = None
            for candidate in hr_candidates:
                if candidate.exists():
                    hr_path = candidate.resolve()
                    break
            
            if hr_path is None:
                miss_hr += 1
                continue

            # LR 경로 찾기
            lr_path = None
            
            if has_lr_in_json:
                # JSON에 LR 경로가 명시되어 있음
                lr_candidates = [
                    self.root / lr_rel_path,
                    Path(lr_rel_path),
                ]
                
                # 디버깅: 첫 번째 파일만 출력
                if not found_first_lr and miss_lr == 0:
                    print(f"[Debug] Trying LR paths for: {lr_rel_path}")
                    for idx, candidate in enumerate(lr_candidates):
                        exists = candidate.exists()
                        print(f"  [{idx}] {candidate} -> {'EXISTS' if exists else 'NOT FOUND'}")
                
                for candidate in lr_candidates:
                    if candidate.exists():
                        lr_path = candidate.resolve()
                        if not found_first_lr:
                            print(f"[Dataset] Using JSON LR paths: {lr_rel_path}")
                            found_first_lr = True
                        break
            else:
                # JSON에 HR만 있음 - LR 경로 추측
                bucket = hr_path.parent.name
                stem = hr_path.stem
                
                lr_candidates = [
                    # 패턴 1: HR과 같은 위치에 x4 suffix
                    hr_path.parent / f"{stem}x{self.scale}.png",
                    
                    # 패턴 2: 표준 X4 폴더
                    self.root / f"X{self.scale}" / "train" / bucket / f"{stem}x{self.scale}.png",
                    self.root / f"X{self.scale}" / "train" / bucket / f"{stem}.png",
                    
                    # 패턴 3: LR 폴더
                    self.root / "LR" / "train" / bucket / f"{stem}.png",
                    
                    # 패턴 4: 경로 문자열 치환
                    Path(str(hr_path).replace("HR", f"X{self.scale}")),
                    Path(str(hr_path).replace("HR", "LR")),
                ]
                
                for candidate in lr_candidates:
                    if candidate.exists():
                        lr_path = candidate.resolve()
                        if not found_first_lr:
                            print(f"[Dataset] Found LR pattern: {candidate}")
                            found_first_lr = True
                        break

            if lr_path is None:
                if self.allow_bicubic_fallback:
                    self.items.append((hr_path, hr_path, True))
                else:
                    miss_lr += 1
                    continue
            else:
                self.items.append((lr_path, hr_path, False))

            if limit and len(self.items) >= limit:
                break

        print(f"[Dataset] Total: {len(self.items)} pairs | Missing - HR:{miss_hr}, LR:{miss_lr}")

    def _build_val(self, limit: Optional[int], strict: bool):
        val_root = self.root / "val1"
        hr_root = val_root / "HR" / "val"
        lr_root = val_root / f"X{self.scale}"
        if not hr_root.exists() or not lr_root.exists():
            raise FileNotFoundError(f"Val structure not found: {hr_root} or {lr_root}")

        cnt = 0
        for hr_path in hr_root.rglob("*.png"):
            stem = hr_path.stem
            candidates = [
                lr_root / f"{stem}x{self.scale}.png",
                lr_root / f"{stem}.png",
            ]
            lr_path = next((c for c in candidates if c.exists()), None)
            if lr_path is None:
                if self.allow_bicubic_fallback:
                    self.items.append((hr_path, hr_path, True))
                else:
                    if strict: continue
                    else: continue
            else:
                self.items.append((lr_path, hr_path, False))
            cnt += 1
            if limit and cnt >= limit:
                break

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        lr_or_hr, hr_path, is_bicubic = self.items[idx]
        hr_pil = Image.open(hr_path).convert("RGB")

        if is_bicubic:
            lr_pil = imresize_bicubic(hr_pil, self.scale)
        else:
            lr_pil = Image.open(lr_or_hr).convert("RGB")

        if self.to_tensor:
            import torchvision.transforms.functional as TF
            hr_pil = TF.resize(hr_pil, [self.hr_size, self.hr_size])
            lr_pil = TF.resize(lr_pil, [self.lr_size, self.lr_size])
            lr = TF.to_tensor(lr_pil)
            hr = TF.to_tensor(hr_pil)
            return lr, hr, str(lr_or_hr if not is_bicubic else hr_path), str(hr_path)
        else:
            return lr_pil, hr_pil, str(lr_or_hr if not is_bicubic else hr_path), str(hr_path)


# =========================
# 2) Perceptual Loss
# =========================
class PerceptualLoss(nn.Module):
    def __init__(self, layers=None, device='cuda'):
        super().__init__()
        if layers is None:
            layers = ['relu2_2', 'relu3_4']
        
        vgg = models.vgg19(pretrained=True).features.to(device).eval()
        self.vgg = vgg
        self.layers = layers
        self.device = device
        
        self.layer_name_mapping = {
            'relu1_2': 3, 'relu2_2': 8, 'relu3_4': 17, 
            'relu4_4': 26, 'relu5_4': 35
        }
        
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)
        self.register_buffer('mean', mean)
        self.register_buffer('std', std)
        
        for param in self.vgg.parameters():
            param.requires_grad = False
    
    def normalize(self, x):
        return (x - self.mean) / self.std
    
    def extract_features(self, x):
        x = self.normalize(x)
        features = []
        for name, layer in self.vgg._modules.items():
            x = layer(x)
            if int(name) in [self.layer_name_mapping[l] for l in self.layers]:
                features.append(x)
        return features
    
    def forward(self, pred, target):
        pred_features = self.extract_features(pred)
        target_features = self.extract_features(target)
        
        loss = 0.0
        for pred_feat, target_feat in zip(pred_features, target_features):
            loss += F.mse_loss(pred_feat, target_feat)
        
        return loss / len(pred_features)


# =========================
# 3) Swin Transformer Blocks
# =========================
def window_partition(x, window_size):
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class WindowAttention(nn.Module):
    def __init__(self, dim, window_size, num_heads):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))
        attn = attn.softmax(dim=-1)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        return x


class SwinTransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, window_size=8, shift_size=0, mlp_ratio=4.0):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio

        num_groups = min(8, dim) if dim >= 8 else dim
        while dim % num_groups != 0:
            num_groups -= 1

        self.norm1 = nn.GroupNorm(num_groups, dim)
        self.attn = WindowAttention(dim, window_size, num_heads)
        self.norm2 = nn.GroupNorm(num_groups, dim)
        
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Linear(mlp_hidden_dim, dim)
        )

    def forward(self, x):
        B, C, H, W = x.shape
        shortcut = x

        x = self.norm1(x)
        x = x.permute(0, 2, 3, 1)

        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x

        x_windows = window_partition(shifted_x, self.window_size)
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)

        attn_windows = self.attn(x_windows)

        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)

        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x

        x = x.permute(0, 3, 1, 2)
        x = shortcut + x

        shortcut = x
        x = self.norm2(x)
        x = x.permute(0, 2, 3, 1)
        B, H, W, C = x.shape
        x = x.view(B, H * W, C)
        x = self.mlp(x)
        x = x.view(B, H, W, C).permute(0, 3, 1, 2)
        x = shortcut + x

        return x


class ResidualSwinTransformerGroup(nn.Module):
    def __init__(self, dim, depth, num_heads, window_size):
        super().__init__()
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(
                dim=dim,
                num_heads=num_heads,
                window_size=window_size,
                shift_size=0 if (i % 2 == 0) else window_size // 2
            )
            for i in range(depth)
        ])
        self.conv = nn.Conv2d(dim, dim, 3, 1, 1)

    def forward(self, x):
        shortcut = x
        for blk in self.blocks:
            x = blk(x)
        x = self.conv(x)
        return x + shortcut


# =========================
# 4) Wavelet-Enhanced Multi-scale Fusion
# =========================
class WaveletEnhancedFusion(nn.Module):
    """
    Multi-scale fusion enhanced with wavelet decomposition
    Preserves high-frequency details at multiple scales
    """
    def __init__(self, c=4, scale=4, depth=2, num_heads=4, window_size=8):
        super().__init__()
        self.scale = scale
        self.dwt = HaarWaveletTransform()
        self.idwt = InverseHaarWavelet()
        
        # Adaptive window size for small latents (after wavelet: H/2, W/2)
        # If HR=256, LR=64, latent=8, after wavelet=4 → need window_size <= 4
        adaptive_window = min(window_size, 4)
        
        # Process low-frequency (LL) component
        self.lr_process = ResidualSwinTransformerGroup(c, depth, num_heads, adaptive_window)
        
        # Process high-frequency components (LH, HL, HH)
        self.freq_process = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(c, c, 3, 1, 1),
                nn.GELU(),
                nn.Conv2d(c, c, 3, 1, 1)
            ) for _ in range(3)
        ])
        
        # Progressive upscaling
        num_stages = int(math.log2(scale))
        self.up_blocks = nn.ModuleList([
            ResidualSwinTransformerGroup(c, depth, num_heads, window_size)
            for _ in range(num_stages)
        ])
        
        # Frequency-aware fusion
        self.final_fuse = nn.Conv2d(c * (num_stages + 1 + 3), c, 1)
    
    def forward(self, z_lr):
        features = []
        
        # Wavelet decomposition
        LL, LH, HL, HH = self.dwt(z_lr)
        
        # Process low-frequency
        ll_feat = self.lr_process(LL)
        features.append(ll_feat)
        
        # Process high-frequency components
        for i, (freq_comp, freq_proc) in enumerate(zip([LH, HL, HH], self.freq_process)):
            freq_feat = freq_proc(freq_comp)
            features.append(freq_feat)
        
        # Progressive upscaling on low-frequency
        current = ll_feat
        for up_block in self.up_blocks:
            current = F.interpolate(current, scale_factor=2, mode='bilinear', align_corners=False)
            current = up_block(current)
            features.append(current)
        
        # Align all features to target size
        target_size = features[-1].shape[-2:]
        aligned_features = []
        for feat in features:
            if feat.shape[-2:] != target_size:
                feat = F.interpolate(feat, size=target_size, mode='bilinear', align_corners=False)
            aligned_features.append(feat)
        
        # Frequency-aware fusion
        fused = self.final_fuse(torch.cat(aligned_features, dim=1))
        return fused


# =========================
# 5) Frequency Bridge Decoder
# =========================
class FrequencyBridgeDecoder(nn.Module):
    """
    Bridge latent space to pixel space with frequency compensation
    Based on Frequency-Augmented Decoder approach
    """
    def __init__(self, latent_dim=4, hidden_dim=64):
        super().__init__()
        
        # High-frequency enhancement without changing spatial size
        self.hf_enhance = nn.Sequential(
            nn.Conv2d(latent_dim, hidden_dim, 3, 1, 1),
            nn.GELU(),
            nn.Conv2d(hidden_dim, hidden_dim, 3, 1, 1),
            nn.GELU(),
            nn.Conv2d(hidden_dim, latent_dim, 3, 1, 1)
        )
        
        # Low-frequency refinement
        self.lf_refine = nn.Conv2d(latent_dim, latent_dim, 3, 1, 1)
    
    def forward(self, z):
        """
        Enhance latent with frequency compensation before VAE decode
        Maintains spatial dimensions
        """
        # High-frequency compensation
        hf_comp = self.hf_enhance(z)
        
        # Low-frequency refinement
        lf_refined = self.lf_refine(z)
        
        # Combine (residual connection)
        z_enhanced = lf_refined + hf_comp
        
        return z_enhanced


# =========================
# 6) Wavelet-Enhanced UNet
# =========================
def timestep_embedding(timesteps, dim):
    half = dim // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(0, half, device=timesteps.device).float() / half)
    args = timesteps.float()[:, None] * freqs[None]
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
    return emb


class WaveletEnhancedUNet(nn.Module):
    """UNet with wavelet-enhanced frequency processing"""
    def __init__(self, zc=4, base=128, depth=2, num_heads=8, window_size=8):
        super().__init__()
        self.dwt = HaarWaveletTransform()
        
        self.in_conv = nn.Conv2d(zc, base, 3, 1, 1)
        self.cond_conv = nn.Conv2d(zc, base, 3, 1, 1)
        
        # Wavelet branch for high-frequency
        self.wavelet_branch = nn.Sequential(
            nn.Conv2d(zc * 3, base // 2, 3, 1, 1),  # LH, HL, HH
            nn.GELU(),
            nn.Conv2d(base // 2, base // 2, 3, 1, 1)
        )
        
        self.time = nn.Sequential(
            nn.Linear(128, base*2), 
            nn.SiLU(), 
            nn.Linear(base*2, base + base // 2)  # Match concatenated channel dimension
        )
        
        # Encoder
        self.d1 = ResidualSwinTransformerGroup(base + base // 2, depth, num_heads, window_size)
        self.down1 = nn.Conv2d(base + base // 2, base, 3, 2, 1)
        self.d2 = ResidualSwinTransformerGroup(base, depth, num_heads, window_size//2)
        self.down2 = nn.Conv2d(base, base, 3, 2, 1)
        
        # Bottleneck
        self.m = ResidualSwinTransformerGroup(base, depth*2, num_heads, window_size//4)
        
        # Decoder
        self.up1 = nn.ConvTranspose2d(base, base, 4, 2, 1)
        self.fuse1 = nn.Conv2d(base * 2, base, 1)
        self.u1 = ResidualSwinTransformerGroup(base, depth, num_heads, window_size//2)
        
        self.up2 = nn.ConvTranspose2d(base, base + base // 2, 4, 2, 1)
        self.fuse2 = nn.Conv2d((base + base // 2) * 2, base + base // 2, 1)
        self.u2 = ResidualSwinTransformerGroup(base + base // 2, depth, num_heads, window_size)
        
        # Output
        num_groups = min(8, base + base // 2)
        while (base + base // 2) % num_groups != 0:
            num_groups -= 1
        
        self.out = nn.Sequential(
            nn.GroupNorm(num_groups, base + base // 2),
            nn.SiLU(),
            nn.Conv2d(base + base // 2, zc, 3, 1, 1)
        )
    
    def forward(self, z_noisy, cond_fused, t_emb):
        # Wavelet decomposition for frequency awareness
        _, LH, HL, HH = self.dwt(z_noisy)
        wavelet_feat = self.wavelet_branch(torch.cat([LH, HL, HH], dim=1))
        
        # Upsample wavelet features to match spatial dimensions
        # DWT reduces H,W by 2, so we need to restore
        wavelet_feat = F.interpolate(wavelet_feat, size=z_noisy.shape[-2:], 
                                     mode='bilinear', align_corners=False)
        
        # Main path
        x = self.in_conv(z_noisy) + self.cond_conv(cond_fused)
        
        # Combine with wavelet features
        x = torch.cat([x, wavelet_feat], dim=1)
        
        # Time embedding
        t = self.time(t_emb)[:, :, None, None]
        x = x + t
        
        # Encoder
        x1 = self.d1(x)
        x = self.down1(x1)
        
        x2 = self.d2(x)
        x = self.down2(x2)
        
        # Bottleneck
        x = self.m(x)
        
        # Decoder
        x = self.up1(x)
        x = self.fuse1(torch.cat([x, x2], dim=1))
        x = self.u1(x)
        
        x = self.up2(x)
        x = self.fuse2(torch.cat([x, x1], dim=1))
        x = self.u2(x)
        
        return self.out(x)


# =========================
# 7) Gaussian Diffusion
# =========================
@dataclass
class DiffusionConfig:
    timesteps: int = 1000
    beta_start: float = 1e-4
    beta_end: float = 0.02
    schedule: str = "cosine"


class GaussianDiffusion:
    def __init__(self, cfg, device):
        self.T = cfg.timesteps
        
        if cfg.schedule == "cosine":
            steps = torch.arange(self.T + 1, dtype=torch.float32, device=device) / self.T
            alphas_cumprod = torch.cos((steps + 0.008) / 1.008 * math.pi / 2) ** 2
            alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
            betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
            betas = torch.clip(betas, 0.0001, 0.9999)
        else:
            betas = torch.linspace(cfg.beta_start, cfg.beta_end, self.T, device=device)
        
        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
        
        self.betas = betas
        self.alphas = alphas
        self.alphas_cumprod = alphas_cumprod
        self.alphas_cumprod_prev = alphas_cumprod_prev
        self.sqrt_ac = torch.sqrt(alphas_cumprod)
        self.sqrt_om = torch.sqrt(1 - alphas_cumprod)
        self.sqrt_recip_ac = torch.sqrt(1.0 / alphas_cumprod)
        self.sqrt_recip_ac_minus_one = torch.sqrt(1.0 / alphas_cumprod - 1)
        self.posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        
    def add_noise(self, x0, t, noise=None):
        if noise is None: 
            noise = torch.randn_like(x0)
        sqrt_ac = self.sqrt_ac[t][:, None, None, None]
        sqrt_om = self.sqrt_om[t][:, None, None, None]
        return sqrt_ac * x0 + sqrt_om * noise, noise
    
    def predict_start_from_noise(self, x_t, t, noise):
        sqrt_recip = self.sqrt_recip_ac[t][:, None, None, None]
        sqrt_recip_m1 = self.sqrt_recip_ac_minus_one[t][:, None, None, None]
        return sqrt_recip * x_t - sqrt_recip_m1 * noise
    
    def q_posterior_mean_variance(self, x_start, x_t, t):
        posterior_mean = (
            torch.sqrt(self.alphas_cumprod_prev[t])[:, None, None, None] * x_start +
            torch.sqrt(self.betas[t])[:, None, None, None] * x_t
        ) / torch.sqrt(self.alphas_cumprod[t])[:, None, None, None]
        
        posterior_variance = self.posterior_variance[t][:, None, None, None]
        return posterior_mean, posterior_variance


# =========================
# 8) Wavelet-Enhanced LMSR Model
# =========================
class LMSR_WaveletEnhanced(nn.Module):
    """
    LMSR with Wavelet-Enhanced Latent Space
    - Wavelet decomposition for frequency awareness
    - Frequency bridge decoder for high-freq compensation
    - Multi-scale wavelet fusion
    """
    def __init__(self, vae, scale=4, zc=4, base=128, depth=2, num_heads=8, window_size=8):
        super().__init__()
        self.vae = vae
        self.scale = scale
        self.fusion = WaveletEnhancedFusion(c=zc, scale=scale, depth=depth, 
                                           num_heads=num_heads//2, window_size=window_size)
        self.unet = WaveletEnhancedUNet(zc=zc, base=base, depth=depth, 
                                       num_heads=num_heads, window_size=window_size)
        self.freq_bridge = FrequencyBridgeDecoder(latent_dim=zc, hidden_dim=64)

    @torch.no_grad()
    def encode(self, x):
        posterior = self.vae.encode((x * 2 - 1))
        z = posterior.latent_dist.sample() * 0.18215
        return z

    @torch.no_grad()
    def decode(self, z, use_freq_bridge=True):
        if use_freq_bridge:
            z = self.freq_bridge(z)
        z = z / 0.18215
        x = self.vae.decode(z).sample
        x = (x + 1) / 2
        return x.clamp(0, 1)

    def forward(self, z_noisy_hr, z_lr, t_emb):
        cond = self.fusion(z_lr)
        
        if cond.shape[-2:] != z_noisy_hr.shape[-2:]:
            cond = F.interpolate(cond, size=z_noisy_hr.shape[-2:], 
                               mode="bilinear", align_corners=False)
        
        eps = self.unet(z_noisy_hr, cond, t_emb)
        return eps


# =========================
# 9) Wavelet Loss Manager
# =========================
class WaveletLossManager:
    def __init__(self, device, loss_weights=None):
        if loss_weights is None:
            loss_weights = {
                'diffusion': 1.0,
                'perceptual': 0.5,
                'pixel': 1.0,
                'wavelet': 0.3,  # Multi-scale wavelet loss
            }
        
        self.loss_weights = loss_weights
        self.perceptual_loss = None
        self.dwt = HaarWaveletTransform().to(device)
        
        if loss_weights.get('perceptual', 0) > 0:
            self.perceptual_loss = PerceptualLoss(device=device)
        
        self.step_counter = 0
        
    def wavelet_loss(self, pred_img, hr_img):
        """Multi-scale wavelet loss for frequency preservation"""
        _, pred_lh, pred_hl, pred_hh = self.dwt(pred_img)
        _, hr_lh, hr_hl, hr_hh = self.dwt(hr_img)
        
        loss = (F.l1_loss(pred_lh, hr_lh) + 
                F.l1_loss(pred_hl, hr_hl) + 
                F.l1_loss(pred_hh, hr_hh)) / 3.0
        return loss
        
    def compute_losses(self, eps_pred, noise_target, pred_img=None, hr_img=None):
        losses = {}
        
        # Diffusion loss
        losses['diffusion'] = F.mse_loss(eps_pred, noise_target)
        
        # Pixel-space losses
        if pred_img is not None and hr_img is not None:
            # Pixel L1
            if self.loss_weights.get('pixel', 0) > 0:
                losses['pixel'] = F.l1_loss(pred_img, hr_img)
            
            # Perceptual
            if (self.perceptual_loss is not None and 
                self.step_counter % 5 == 0):
                losses['perceptual'] = self.perceptual_loss(pred_img, hr_img)
            
            # Wavelet (frequency-domain)
            if self.loss_weights.get('wavelet', 0) > 0:
                losses['wavelet'] = self.wavelet_loss(pred_img, hr_img)
        
        total_loss = sum(self.loss_weights.get(k, 0) * v 
                        for k, v in losses.items() 
                        if k in self.loss_weights)
        losses['total'] = total_loss
        
        self.step_counter += 1
        return losses


# =========================
# 10) Training
# =========================
# =========================
# 10) Multi-GPU Setup (torchrun)
# =========================
def setup_distributed():
    """Initialize distributed training (called by torchrun)"""
    dist.init_process_group(backend="nccl")
    
    # torchrun sets these automatically
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    
    torch.cuda.set_device(local_rank)
    return rank, local_rank, world_size


def cleanup_distributed():
    """Cleanup distributed training"""
    if dist.is_initialized():
        dist.destroy_process_group()


def is_distributed():
    """Check if running in distributed mode"""
    return dist.is_available() and dist.is_initialized()


# =========================
# 11) Training (Multi-GPU)
# =========================
def build_vae(args, device):
    if args.vae_path:
        vae = AutoencoderKL.from_pretrained(args.vae_path).to(device)
    else:
        vae = AutoencoderKL.from_pretrained(args.vae_id).to(device)
    vae.requires_grad_(False)
    vae.eval()
    return vae


# =========================
# 11) Training (torchrun)
# =========================
def build_vae(args, device):
    if args.vae_path:
        vae = AutoencoderKL.from_pretrained(args.vae_path).to(device)
    else:
        vae = AutoencoderKL.from_pretrained(args.vae_id).to(device)
    vae.requires_grad_(False)
    vae.eval()
    return vae


def train(args):
    """Main training function (called by torchrun or single GPU)"""
    # Check if running with torchrun
    if "RANK" in os.environ:
        rank, local_rank, world_size = setup_distributed()
        is_ddp = True
    else:
        rank = 0
        local_rank = 0
        world_size = 1
        is_ddp = False
        if torch.cuda.is_available():
            torch.cuda.set_device(0)
    
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    set_seed(args.seed + rank)
    
    if rank == 0:
        if is_ddp:
            print(f"=" * 70)
            print(f"Multi-GPU Training with {world_size} GPUs (torchrun)")
            print(f"=" * 70)
        else:
            print(f"=" * 70)
            print(f"Single GPU Training")
            print(f"=" * 70)
        print(f"\nModel: LMSR with Wavelet-Enhanced Latent Space")
        print(f"  - Wavelet decomposition for frequency awareness")
        print(f"  - Frequency bridge decoder for high-freq compensation")
        print(f"  - Multi-scale wavelet loss")

    # Dataset
    ds = LSDIRSRPairsFromJSON(
        root=args.root, 
        json_path=args.json, 
        scale=args.scale,
        split=args.split, 
        to_tensor=True, 
        limit=args.limit, 
        strict=False,
        allow_bicubic_fallback=args.allow_bicubic_fallback,
        hr_size=args.hr_size
    )
    
    # DataLoader with DistributedSampler for multi-GPU
    if is_ddp:
        sampler = DistributedSampler(
            ds, 
            num_replicas=world_size, 
            rank=rank, 
            shuffle=True,
            seed=args.seed
        )
        dl = DataLoader(
            ds, 
            batch_size=args.batch, 
            sampler=sampler,
            num_workers=args.workers, 
            pin_memory=True,
            drop_last=True
        )
    else:
        dl = DataLoader(
            ds, 
            batch_size=args.batch, 
            shuffle=True,
            num_workers=args.workers, 
            pin_memory=True,
            drop_last=True
        )

    # Model
    vae = build_vae(args, device)
    model = LMSR_WaveletEnhanced(
        vae=vae, 
        scale=args.scale, 
        zc=4, 
        base=args.width,
        depth=args.depth,
        num_heads=args.num_heads,
        window_size=args.window_size
    ).to(device)
    
    # Wrap with DDP for multi-GPU
    if is_ddp:
        model = DDP(
            model, 
            device_ids=[local_rank], 
            output_device=local_rank,
            find_unused_parameters=True
        )
        model_module = model.module
    else:
        model_module = model
    
    if rank == 0:
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"  Trainable parameters: {total_params/1e6:.2f}M")
        print(f"  Batch size per GPU: {args.batch}")
        if is_ddp:
            print(f"  Effective batch size: {args.batch * world_size}")

    # Optimizer
    opt = torch.optim.AdamW(
        model.parameters(), 
        lr=args.lr, 
        weight_decay=0.01,
        betas=(0.9, 0.999)
    )
    
    # Scheduler
    total_steps = len(dl) * args.epochs
    warmup_steps = total_steps // 10
    
    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    
    # Diffusion
    diff = GaussianDiffusion(
        DiffusionConfig(
            timesteps=args.timesteps,
            schedule=args.schedule
        ), 
        device
    )
    
    # Loss manager
    loss_weights = {
        'diffusion': args.loss_diffusion,
        'perceptual': args.loss_perceptual,
        'pixel': args.loss_pixel,
        'wavelet': args.loss_wavelet,
    }
    loss_manager = WaveletLossManager(device, loss_weights)

    # Checkpoints (only rank 0 saves)
    if rank == 0:
        os.makedirs(args.save_dir, exist_ok=True)
        log_file = os.path.join(args.save_dir, "training_log.txt")
        with open(log_file, "w") as f:
            f.write("epoch,step,total_loss,diffusion,perceptual,pixel,wavelet,lr\n")

    global_step = 0
    best_loss = float('inf')
    
    # Training loop
    for epoch in range(args.epochs):
        if is_ddp:
            sampler.set_epoch(epoch)  # Shuffle differently each epoch
        
        model.train()
        running_losses = {
            'total': 0.0, 
            'diffusion': 0.0,
            'perceptual': 0.0, 
            'pixel': 0.0,
            'wavelet': 0.0
        }
        
        for i, (lr_img, hr_img, _, _) in enumerate(dl):
            lr_img = lr_img.to(device, non_blocking=True)
            hr_img = hr_img.to(device, non_blocking=True)

            with torch.no_grad():
                z_lr = model_module.encode(lr_img)
                z_hr = model_module.encode(hr_img)

            t = torch.randint(0, diff.T, (lr_img.size(0),), device=device, dtype=torch.long)
            z_noisy, noise = diff.add_noise(z_hr, t)
            t_emb = timestep_embedding(t, 128)

            eps_pred = model(z_noisy, z_lr, t_emb)
            
            # Decode for pixel-space losses
            pred_img = None
            if i % 5 == 0:
                with torch.no_grad():
                    x_start = diff.predict_start_from_noise(z_noisy, t, eps_pred)
                    pred_img = model_module.decode(x_start, use_freq_bridge=True)
            
            losses = loss_manager.compute_losses(eps_pred, noise, pred_img, hr_img)
            loss = losses['total']

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            scheduler.step()

            # Accumulate losses
            for k in running_losses.keys():
                loss_val = losses.get(k, torch.tensor(0.0, device=device))
                if is_ddp:
                    # Average across all GPUs
                    dist.all_reduce(loss_val, op=dist.ReduceOp.SUM)
                    loss_val = loss_val / world_size
                running_losses[k] += loss_val.item()
            
            global_step += 1
            
            # Logging (only rank 0)
            if rank == 0 and (i + 1) % args.log_interval == 0:
                n = args.log_interval
                current_lr = scheduler.get_last_lr()[0]
                
                log_msg = (
                    f"[{epoch+1}/{args.epochs}] step {i+1}/{len(dl)} | "
                    f"total={running_losses['total']/n:.4f} | "
                    f"diff={running_losses['diffusion']/n:.4f}"
                )
                
                if args.loss_perceptual > 0 and running_losses['perceptual'] > 0:
                    log_msg += f" | perc={running_losses['perceptual']/n:.4f}"
                if args.loss_pixel > 0:
                    log_msg += f" | pix={running_losses['pixel']/n:.4f}"
                if args.loss_wavelet > 0:
                    log_msg += f" | wav={running_losses['wavelet']/n:.4f}"
                
                log_msg += f" | lr={current_lr:.2e}"
                print(log_msg)
                
                with open(log_file, "a") as f:
                    f.write(
                        f"{epoch+1},{i+1},"
                        f"{running_losses['total']/n:.6f},"
                        f"{running_losses['diffusion']/n:.6f},"
                        f"{running_losses['perceptual']/n:.6f},"
                        f"{running_losses['pixel']/n:.6f},"
                        f"{running_losses['wavelet']/n:.6f},"
                        f"{current_lr:.6e}\n"
                    )
                
                avg_loss = running_losses['total'] / n
                if avg_loss < best_loss:
                    best_loss = avg_loss
                    torch.save({
                        "epoch": epoch + 1,
                        "step": global_step,
                        "model": model_module.state_dict(),
                        "optimizer": opt.state_dict(),
                        "scheduler": scheduler.state_dict(),
                        "best_loss": best_loss,
                        "args": vars(args)
                    }, os.path.join(args.save_dir, "lmsr_wavelet_best.pt"))
                
                for k in running_losses.keys():
                    running_losses[k] = 0.0
        
        # Save epoch checkpoint (only rank 0)
        if rank == 0:
            ckpt = {
                "epoch": epoch + 1,
                "step": global_step,
                "model": model_module.state_dict(),
                "optimizer": opt.state_dict(),
                "scheduler": scheduler.state_dict(),
                "best_loss": best_loss,
                "args": vars(args)
            }
            torch.save(ckpt, os.path.join(args.save_dir, f"lmsr_wavelet_epoch{epoch+1}.pt"))
            print(f"Epoch {epoch+1} checkpoint saved. Best loss: {best_loss:.4f}")
    
    if rank == 0:
        print("\nTraining completed!")
    
    # Cleanup
    if is_ddp:
        cleanup_distributed()


@torch.no_grad()
def sample_ddim(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vae = build_vae(args, device)
    model = LMSR_WaveletEnhanced(
        vae=vae, 
        scale=args.scale, 
        zc=4, 
        base=args.width,
        depth=args.depth,
        num_heads=args.num_heads,
        window_size=args.window_size
    ).to(device)
    
    sd = torch.load(args.ckpt, map_location="cpu")
    model.load_state_dict(sd["model"], strict=True)
    model.eval()

    im = read_image(args.input_lr)
    im = convert_image_dtype(im, torch.float32).unsqueeze(0).to(device)
    z_lr = model.encode(im)
    
    _, _, h_lr_pix, w_lr_pix = im.shape
    h_hr_lat = (h_lr_pix * args.scale) // 8
    w_hr_lat = (w_lr_pix * args.scale) // 8
    
    print(f"DDIM Sampling with {args.ddim_steps} steps")
    print(f"Using Frequency Bridge Decoder: {args.use_freq_bridge}")
    
    z = torch.randn((1, z_lr.size(1), h_hr_lat, w_hr_lat), device=device)

    diff = GaussianDiffusion(
        DiffusionConfig(
            timesteps=args.timesteps,
            schedule=args.schedule
        ), 
        device
    )
    
    ddim_steps = args.ddim_steps
    step_size = diff.T // ddim_steps
    timesteps = list(range(0, diff.T, step_size))[::-1]
    
    for i, t_step in enumerate(timesteps):
        t = torch.full((1,), t_step, device=device, dtype=torch.long)
        t_emb = timestep_embedding(t, 128)
        
        eps = model(z, z_lr, t_emb)
        pred_x0 = diff.predict_start_from_noise(z, t, eps)
        
        if i < len(timesteps) - 1:
            alpha_t = diff.alphas_cumprod[t_step]
            alpha_t_prev = diff.alphas_cumprod[timesteps[i + 1]]
            
            sigma = args.ddim_eta * torch.sqrt(
                (1 - alpha_t_prev) / (1 - alpha_t) * (1 - alpha_t / alpha_t_prev)
            )
            
            dir_xt = torch.sqrt(1 - alpha_t_prev - sigma**2) * eps
            z = torch.sqrt(alpha_t_prev) * pred_x0 + dir_xt
            
            if sigma > 0:
                noise = torch.randn_like(z)
                z = z + sigma * noise
        else:
            z = pred_x0
        
        if i % 10 == 0:
            print(f"DDIM step {i+1}/{len(timesteps)}")

    out = model.decode(z, use_freq_bridge=args.use_freq_bridge)
    out_img = (out.clamp(0, 1) * 255).byte().squeeze(0).cpu()
    
    from torchvision.utils import save_image
    os.makedirs(Path(args.out).parent, exist_ok=True)
    save_image(out_img.float() / 255.0, args.out)
    print(f"Output saved: {args.out}")


def parse():
    ap = argparse.ArgumentParser(
        description="LMSR with Wavelet-Enhanced Latent Space"
    )
    
    # Dataset
    ap.add_argument("--root", type=str, required=True)
    ap.add_argument("--json", type=str, default=None)
    ap.add_argument("--scale", type=int, default=4, choices=[2, 3, 4])
    ap.add_argument("--split", type=str, default="train", choices=["train", "val"])
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--allow_bicubic_fallback", action="store_true")
    
    # Training
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--width", type=int, default=128)
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--timesteps", type=int, default=1000)
    ap.add_argument("--schedule", type=str, default="cosine", choices=["linear", "cosine"])
    ap.add_argument("--log_interval", type=int, default=50)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--save_dir", type=str, default="./checkpoints_lmsr_wavelet")
    ap.add_argument("--hr_size", type=int, default=256)
    
    # Swin params
    ap.add_argument("--depth", type=int, default=2)
    ap.add_argument("--num_heads", type=int, default=8)
    ap.add_argument("--window_size", type=int, default=8)
    
    # Loss weights
    ap.add_argument("--loss_diffusion", type=float, default=1.0)
    ap.add_argument("--loss_perceptual", type=float, default=0.5)
    ap.add_argument("--loss_pixel", type=float, default=1.0)
    ap.add_argument("--loss_wavelet", type=float, default=0.3, help="Multi-scale wavelet loss")

    # VAE
    ap.add_argument("--vae_id", type=str, default="stabilityai/sd-vae-ft-mse")
    ap.add_argument("--vae_path", type=str, default="")

    # Sampling
    ap.add_argument("--sample", action="store_true")
    ap.add_argument("--ddim", action="store_true")
    ap.add_argument("--ddim_steps", type=int, default=50)
    ap.add_argument("--ddim_eta", type=float, default=0.0)
    ap.add_argument("--use_freq_bridge", action="store_true", default=True, 
                   help="Use frequency bridge decoder")
    ap.add_argument("--ckpt", type=str, default="")
    ap.add_argument("--input_lr", type=str, default="")
    ap.add_argument("--out", type=str, default="./sr_out.png")

    # Utilities
    ap.add_argument("--inspect_only", action="store_true")
    
    return ap.parse_args()


def main():
    args = parse()
    
    print("=" * 70)
    print("LMSR with Wavelet-Enhanced Latent Space")
    print("=" * 70)
    
    if args.inspect_only:
        print("\nDataset Inspection Mode")
        ds = LSDIRSRPairsFromJSON(
            root=args.root,
            json_path=args.json if args.split == "train" else None,
            scale=args.scale,
            split=args.split,
            limit=50,
            allow_bicubic_fallback=args.allow_bicubic_fallback,
            hr_size=args.hr_size
        )
        print(f"\nDataset: {len(ds)} pairs")
        return

    if args.sample:
        if not args.ckpt:
            raise ValueError("--ckpt required")
        if not args.input_lr:
            raise ValueError("--input_lr required")
        
        print(f"\nSampling with Wavelet Enhancement")
        print(f"  Frequency Bridge: {args.use_freq_bridge}")
        
        sample_ddim(args)
    
    else:
        print("\nTraining with Wavelet-Enhanced Latent Space")
        print(f"  Wavelet decomposition: Haar 2D DWT")
        print(f"  Multi-scale fusion: {args.scale}x upsampling")
        print(f"  Loss - Diffusion: {args.loss_diffusion}")
        print(f"  Loss - Perceptual: {args.loss_perceptual}")
        print(f"  Loss - Pixel: {args.loss_pixel}")
        print(f"  Loss - Wavelet: {args.loss_wavelet}")
        
        train(args)


if __name__ == "__main__":
    main()