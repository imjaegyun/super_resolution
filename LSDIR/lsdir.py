"""
Usage:
    torchrun --nproc_per_node=3 lsdir.py
"""

import os
import random
import glob
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torchvision import transforms
import math
from timm.layers import DropPath, to_2tuple, trunc_normal_
from torch.amp import autocast, GradScaler
import torchvision.models as models

# ================================
# Configuration
# ================================
class Config:
    # Paths - LSDIR Structure
    base_path = "/home/user/im_ig/SR/super_resolution/LSDIR"
    
    # Training paths
    hr_train_path = f"{base_path}/HR/train"
    lr_train_path = f"{base_path}/HR/train/train"
    
    # Validation paths
    val_base_path = f"{base_path}/HR/train/val1"
    
    pretrained_model = f"{base_path}/scripts/src/pretrained/HAT-L_SRx4_ImageNet-pretrain.pth"
    save_dir = "./experiments/HAT_LoRA_Balanced_MultiScale_LSDIR"
    
    # Multi-Scale Training
    train_scales = [2, 3, 4]
    val_scales = [2, 3, 4]
    target_scale = 4
    normalize_input_size = 48
    scale_balance_mode = 'cycle'
    scale_weights = [1, 1, 1]
    
    # Training
    hr_patch_size = 192
    batch_size = 8
    num_workers = 8
    num_epochs = 200
    
    # Optimizer
    lr = 2e-5
    weight_decay = 0.01
    betas = (0.9, 0.999)
    
    # Scheduler
    warmup_epochs = 5
    T_0 = 50
    T_mult = 1
    eta_min = 1e-7
    
    # Model - HAT-L
    img_size = 64
    embed_dim = 180
    depths = [6, 6, 6, 6, 6, 6]
    num_heads = [6, 6, 6, 6, 6, 6]
    window_size = 16
    compress_ratio = 3
    squeeze_factor = 30
    conv_scale = 0.01
    mlp_ratio = 2.0
    qkv_bias = True
    qk_scale = None
    drop_rate = 0.0
    attn_drop_rate = 0.0
    drop_path_rate = 0.1
    ape = False
    patch_norm = True
    use_checkpoint = True
    upscale = 4
    img_range = 1.0
    upsampler = 'pixelshuffle'
    resi_connection = '1conv'
    
    # LoRA Configuration
    use_lora = True
    lora_r = 8
    lora_alpha = 16
    lora_dropout = 0.1
    lora_targets = ['qkv', 'proj', 'mlp']
    
    # Enhancement Modules
    use_multi_scale = True
    use_cross_scale_attn = True
    use_frequency_aware = True
    use_adaptive_modulation = True
    use_enhanced_residual = True
    use_large_kernel_attn = True
    train_enhancement_modules = False
    
    # Advanced Training
    use_mixed_precision = False
    use_ema = True
    ema_decay = 0.999
    gradient_clip_norm = 0.5
    
    # Loss Configuration
    use_perceptual_loss = True
    perceptual_weight = 0.01
    use_multiscale_loss = True
    multiscale_weights = [1.0, 0.5, 0.25]
    
    # Data Augmentation
    use_mixup = True
    mixup_prob = 0.3
    mixup_alpha = 0.4
    use_cutmix = True
    cutmix_prob = 0.3
    cutmix_alpha = 1.0
    use_color_jitter = True
    color_jitter_prob = 0.5
    use_gaussian_noise = True
    noise_prob = 0.3
    noise_sigma = 25
    
    # Logging
    print_freq = 100
    save_freq = 1
    num_val_images = 1

config = Config()

# ================================
# EMA Implementation
# ================================
class ExponentialMovingAverage:
    def __init__(self, parameters, decay=0.999):
        parameters = list(parameters)
        self.shadow_params = [p.clone().detach() for p in parameters]
        self.collected_params = None
        self.decay = decay
        self.num_updates = 0
    
    def update(self, parameters=None):
        if parameters is not None:
            parameters = list(parameters)
        
        self.num_updates += 1
        decay = min(self.decay, (1 + self.num_updates) / (10 + self.num_updates))
        
        for s_param, param in zip(self.shadow_params, parameters):
            if param.requires_grad:
                s_param.sub_((1 - decay) * (s_param - param.data))
    
    def copy_to(self, parameters=None):
        if parameters is not None:
            parameters = list(parameters)
        
        for s_param, param in zip(self.shadow_params, parameters):
            param.data.copy_(s_param.data)
    
    def store(self, parameters=None):
        if parameters is not None:
            parameters = list(parameters)
        
        self.collected_params = [param.clone() for param in parameters]
    
    def restore(self, parameters=None):
        if self.collected_params is None:
            raise RuntimeError("No stored parameters to restore")
        
        if parameters is not None:
            parameters = list(parameters)
        
        for c_param, param in zip(self.collected_params, parameters):
            param.data.copy_(c_param.data)
        
        self.collected_params = None
    
    def state_dict(self):
        return {
            'shadow_params': self.shadow_params,
            'collected_params': self.collected_params,
            'decay': self.decay,
            'num_updates': self.num_updates
        }
    
    def load_state_dict(self, state_dict):
        self.shadow_params = state_dict['shadow_params']
        self.collected_params = state_dict['collected_params']
        self.decay = state_dict['decay']
        self.num_updates = state_dict['num_updates']

# ================================
# LoRA Implementation
# ================================
class LoRALayer(nn.Module):
    def __init__(self, in_features, out_features, rank=8, alpha=16, dropout=0.1):
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        
        self.lora_A = nn.Parameter(torch.zeros(in_features, rank))
        self.lora_B = nn.Parameter(torch.zeros(rank, out_features))
        self.lora_dropout = nn.Dropout(dropout)
        
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)
    
    def forward(self, x):
        lora_out = self.lora_dropout(x) @ self.lora_A @ self.lora_B
        return lora_out * self.scaling

class LinearWithLoRA(nn.Module):
    def __init__(self, linear_layer, rank=8, alpha=16, dropout=0.1, use_lora=True):
        super().__init__()
        self.linear = linear_layer
        self.use_lora = use_lora
        
        if use_lora:
            self.lora = LoRALayer(
                linear_layer.in_features,
                linear_layer.out_features,
                rank=rank,
                alpha=alpha,
                dropout=dropout
            )
            self.linear.weight.requires_grad = False
            if self.linear.bias is not None:
                self.linear.bias.requires_grad = False
    
    def forward(self, x):
        result = self.linear(x)
        if self.use_lora:
            result = result + self.lora(x)
        return result

# ================================
# Advanced Loss Functions
# ================================
class VGGPerceptualLoss(nn.Module):
    def __init__(self, device, local_rank=0):
        super().__init__()
        
        if local_rank == 0:
            vgg = models.vgg19(weights='DEFAULT').features
        
        if dist.is_initialized():
            dist.barrier()
        
        if local_rank != 0:
            vgg = models.vgg19(weights='DEFAULT').features
        
        self.slice1 = nn.Sequential(*list(vgg)[:8]).to(device)
        self.slice2 = nn.Sequential(*list(vgg)[8:17]).to(device)
        self.slice3 = nn.Sequential(*list(vgg)[17:26]).to(device)
        
        for param in self.parameters():
            param.requires_grad = False
        
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
    
    def forward(self, pred, target):
        pred = (pred - self.mean) / self.std
        target = (target - self.mean) / self.std
        
        pred_feat1 = self.slice1(pred)
        pred_feat2 = self.slice2(pred_feat1)
        pred_feat3 = self.slice3(pred_feat2)
        
        with torch.no_grad():
            target_feat1 = self.slice1(target)
            target_feat2 = self.slice2(target_feat1)
            target_feat3 = self.slice3(target_feat2)
        
        loss = F.l1_loss(pred_feat1, target_feat1) + \
               F.l1_loss(pred_feat2, target_feat2) + \
               F.l1_loss(pred_feat3, target_feat3)
        
        return loss / 3.0

class CombinedLoss(nn.Module):
    def __init__(self, use_perceptual=True, perceptual_weight=0.1,
                 use_multiscale=True, multiscale_weights=[1.0, 0.5, 0.25],
                 device=None, local_rank=0):
        super().__init__()
        self.use_perceptual = use_perceptual
        self.perceptual_weight = perceptual_weight
        self.use_multiscale = use_multiscale
        self.multiscale_weights = multiscale_weights
        
        if use_perceptual:
            self.perceptual_loss = VGGPerceptualLoss(device, local_rank)
    
    def forward(self, pred, target):
        if self.use_multiscale:
            l1_loss = 0
            for weight in self.multiscale_weights:
                if weight == 1.0:
                    l1_loss += F.l1_loss(pred, target)
                else:
                    h, w = int(target.size(2) * weight), int(target.size(3) * weight)
                    pred_scaled = F.interpolate(pred, size=(h, w), mode='bilinear', align_corners=False)
                    target_scaled = F.interpolate(target, size=(h, w), mode='bilinear', align_corners=False)
                    l1_loss += F.l1_loss(pred_scaled, target_scaled) * weight
            l1_loss /= sum(self.multiscale_weights)
        else:
            l1_loss = F.l1_loss(pred, target)
        
        if self.use_perceptual:
            perceptual = self.perceptual_loss(pred, target)
            total_loss = l1_loss + self.perceptual_weight * perceptual
            return total_loss, {'l1': l1_loss.item(), 'perceptual': perceptual.item()}
        else:
            return l1_loss, {'l1': l1_loss.item()}

# ================================
# Enhancement Modules
# ================================
class LargeKernelAttention(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv0 = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)
        self.conv_spatial = nn.Conv2d(dim, dim, 7, stride=1, padding=9, groups=dim, dilation=3)
        self.conv1 = nn.Conv2d(dim, dim, 1)

    def forward(self, x):
        u = x.clone()
        attn = self.conv0(x)
        attn = self.conv_spatial(attn)
        attn = self.conv1(attn)
        return u * attn

class MultiScaleFeatureFusion(nn.Module):
    def __init__(self, dim, scales=[1, 2, 4]):
        super().__init__()
        self.scales = scales
        self.convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(dim, dim, 3, 1, 1),
                nn.GELU(),
                nn.Conv2d(dim, dim, 3, 1, 1)
            ) for _ in scales
        ])
        self.fusion = nn.Conv2d(dim * len(scales), dim, 1)
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim * len(scales), dim * len(scales) // 4, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim * len(scales) // 4, dim * len(scales), 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        B, C, H, W = x.shape
        features = []
        
        for scale, conv in zip(self.scales, self.convs):
            if scale == 1:
                feat = conv(x)
            else:
                h, w = H // scale, W // scale
                scaled = F.interpolate(x, size=(h, w), mode='bilinear', align_corners=False)
                feat = conv(scaled)
                feat = F.interpolate(feat, size=(H, W), mode='bilinear', align_corners=False)
            features.append(feat)
        
        concat_feat = torch.cat(features, dim=1)
        gate = self.gate(concat_feat)
        gated_feat = concat_feat * gate
        out = self.fusion(gated_feat)
        return out

class CrossScaleAttention(nn.Module):
    def __init__(self, dim, num_heads=4):
        super().__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        
        self.qkv = nn.Conv2d(dim, dim * 3, 1, bias=False)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, 3, 1, 1, groups=dim * 3, bias=False)
        self.project_out = nn.Conv2d(dim, dim, 1, bias=False)

    def forward(self, x):
        b, c, h, w = x.shape
        
        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)
        
        q = q.reshape(b, self.num_heads, c // self.num_heads, h * w)
        k = k.reshape(b, self.num_heads, c // self.num_heads, h * w)
        v = v.reshape(b, self.num_heads, c // self.num_heads, h * w)

        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)
        out = out.reshape(b, c, h, w)
        out = self.project_out(out)
        return out

class FrequencyAwareEnhancement(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.complex_weight = nn.Parameter(torch.randn(dim, 2, dtype=torch.float32) * 0.02)
        self.conv = nn.Sequential(
            nn.Conv2d(dim * 2, dim, 1),
            nn.GELU(),
            nn.Conv2d(dim, dim, 1)
        )
        
    def forward(self, x):
        B, C, H, W = x.shape
        
        orig_dtype = x.dtype
        x_float = x.float()
        
        H_pad = H if H % 2 == 0 else H + 1
        W_pad = W if W % 2 == 0 else W + 1
        
        if H != H_pad or W != W_pad:
            x_float = F.pad(x_float, (0, W_pad - W, 0, H_pad - H), mode='reflect')
        
        x_freq = torch.fft.rfft2(x_float, norm='ortho')
        weight = torch.view_as_complex(self.complex_weight)
        x_freq_weighted = x_freq * weight.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
        x_spatial = torch.fft.irfft2(x_freq_weighted, s=(H_pad, W_pad), norm='ortho')
        
        if H != H_pad or W != W_pad:
            x_spatial = x_spatial[:, :, :H, :W]
        
        x_spatial = x_spatial.to(orig_dtype)
        
        combined = torch.cat([x, x_spatial], dim=1)
        out = self.conv(combined)
        
        return out + x

class AdaptiveFeatureModulation(nn.Module):
    def __init__(self, dim, reduction=16):
        super().__init__()
        self.channel_attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, dim // reduction, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim // reduction, dim, 1),
            nn.Sigmoid()
        )
        
        self.spatial_attn = nn.Sequential(
            nn.Conv2d(2, 1, 7, padding=3),
            nn.Sigmoid()
        )
        
        self.modulation = nn.Sequential(
            nn.Conv2d(dim, dim * 2, 1),
            nn.GELU(),
            nn.Conv2d(dim * 2, dim, 1)
        )
    
    def forward(self, x):
        ca = self.channel_attn(x)
        x_ca = x * ca
        
        avg_out = torch.mean(x_ca, dim=1, keepdim=True)
        max_out, _ = torch.max(x_ca, dim=1, keepdim=True)
        sa = self.spatial_attn(torch.cat([avg_out, max_out], dim=1))
        x_sa = x_ca * sa
        
        modulated = self.modulation(x_sa)
        
        return modulated + x

class EnhancedResidualBlock(nn.Module):
    def __init__(self, dim, num_blocks=3):
        super().__init__()
        self.blocks = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(dim * (i + 1), dim, 3, 1, 1),
                nn.GELU(),
                nn.Conv2d(dim, dim, 3, 1, 1)
            ) for i in range(num_blocks)
        ])
        self.fusion = nn.Conv2d(dim * (num_blocks + 1), dim, 1)
    
    def forward(self, x):
        features = [x]
        
        for block in self.blocks:
            feat_in = torch.cat(features, dim=1)
            feat_out = block(feat_in)
            features.append(feat_out)
        
        all_features = torch.cat(features, dim=1)
        out = self.fusion(all_features)
        
        return out + x

# ================================
# Advanced Data Augmentation
# ================================
class AdvancedAugmentation:
    def __init__(self, config):
        self.config = config
        self.color_jitter = transforms.ColorJitter(
            brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1
        )
    
    def mixup(self, hr1, lq1, hr2, lq2, alpha=0.4):
        lam = np.random.beta(alpha, alpha)
        hr_mixed = lam * hr1 + (1 - lam) * hr2
        lq_mixed = lam * lq1 + (1 - lam) * lq2
        return hr_mixed, lq_mixed
    
    def cutmix(self, hr1, lq1, hr2, lq2, alpha=1.0, scale=4):
        lam = np.random.beta(alpha, alpha)
        _, _, h, w = hr1.shape
        
        cut_rat = np.sqrt(1. - lam)
        cut_h = int(h * cut_rat)
        cut_w = int(w * cut_rat)
        
        cx = np.random.randint(h)
        cy = np.random.randint(w)
        
        bbx1 = np.clip(cx - cut_h // 2, 0, h)
        bby1 = np.clip(cy - cut_w // 2, 0, w)
        bbx2 = np.clip(cx + cut_h // 2, 0, h)
        bby2 = np.clip(cy + cut_w // 2, 0, w)
        
        hr_mixed = hr1.clone()
        lq_mixed = lq1.clone()
        hr_mixed[:, :, bbx1:bbx2, bby1:bby2] = hr2[:, :, bbx1:bbx2, bby1:bby2]
        lq_mixed[:, :, bbx1//scale:bbx2//scale, bby1//scale:bby2//scale] = \
            lq2[:, :, bbx1//scale:bbx2//scale, bby1//scale:bby2//scale]
        
        return hr_mixed, lq_mixed
    
    def add_gaussian_noise(self, img, sigma=25):
        noise = torch.randn_like(img) * (sigma / 255.0)
        return torch.clamp(img + noise, 0, 1)
    
    def apply_color_jitter(self, img):
        img_pil = transforms.ToPILImage()(img.squeeze(0))
        img_jittered = self.color_jitter(img_pil)
        return transforms.ToTensor()(img_jittered).unsqueeze(0)

# ================================
# ✅ Dataset for LSDIR structure
# ================================
class BalancedMultiScaleLSDirectDataset(Dataset):
    """
    ✅ VERIFIED LSDIR structure:
    Training:
      - HR: /HR/train/0001000/0000001.png
      - LR: /HR/train/train/0001000/0000001x2.png, x3, x4
    Validation:
      - HR: /HR/train/val1/HR/val/0000001.png
      - X2: /HR/train/val1/X2/val/0000001x2.png
      - X3: /HR/train/val1/X3/val/0000001x3.png
      - X4: /HR/train/val1/X4/val/0000001x4.png
    """
    def __init__(self, hr_path=None, lr_path=None, val_base_path=None, scales=[2, 3, 4], 
                 patch_size=192, normalize_size=48, is_train=True, 
                 config=None, balance_mode='cycle'):
        self.scales = scales
        self.patch_size = patch_size
        self.normalize_size = normalize_size
        self.is_train = is_train
        self.config = config
        self.balance_mode = balance_mode
        
        if is_train and config:
            self.aug = AdvancedAugmentation(config)
        
        self.image_data = []
        
        if is_train:
            if hr_path is None or lr_path is None:
                raise ValueError("hr_path and lr_path are required for training")
            self._load_training_data(hr_path, lr_path)
        else:
            if val_base_path is None:
                raise ValueError("val_base_path is required for validation")
            self._load_validation_data(val_base_path)
        
        if len(self.image_data) == 0:
            raise RuntimeError(
                f"❌ No images found!\n"
                f"   HR path: {hr_path if is_train else val_base_path}\n"
                f"   LR path: {lr_path if is_train else 'N/A'}\n"
                f"   Please check dataset structure!"
            )
        
        # Balanced scale assignment
        if is_train:
            self.scale_assignment = self._create_balanced_scale_assignment()
        
        mode_str = "training" if is_train else "validation"
        print(f"✓ Loaded {len(self.image_data)} {mode_str} image pairs")
        
        if is_train:
            scale_counts = {scale: sum(1 for _, lr_dict in self.image_data if scale in lr_dict) 
                           for scale in scales}
            for scale, count in scale_counts.items():
                print(f"  - x{scale}: {count} pairs available")
            
            print(f"\n✅ Balanced Scale Distribution (mode: {balance_mode}):")
            assignment_counts = {scale: 0 for scale in scales}
            for s in self.scale_assignment:
                assignment_counts[s] += 1
            
            for scale in scales:
                count = assignment_counts[scale]
                ratio = count / len(self.scale_assignment) * 100
                print(f"  → x{scale}: {count} samples ({ratio:.1f}%)")
    
    def _create_balanced_scale_assignment(self):
        """균형잡힌 스케일 할당"""
        n_samples = len(self.image_data)
        
        if self.balance_mode == 'cycle':
            assignment = []
            for i in range(n_samples):
                scale = self.scales[i % len(self.scales)]
                assignment.append(scale)
            return assignment
        
        elif self.balance_mode == 'shuffle':
            n_per_scale = n_samples // len(self.scales)
            remainder = n_samples % len(self.scales)
            
            assignment = []
            for idx, scale in enumerate(self.scales):
                count = n_per_scale + (1 if idx < remainder else 0)
                assignment.extend([scale] * count)
            
            random.shuffle(assignment)
            return assignment
        
        elif self.balance_mode == 'weighted':
            weights = np.array(self.config.scale_weights) / sum(self.config.scale_weights)
            counts = (weights * n_samples).astype(int)
            remainder = n_samples - counts.sum()
            counts[:remainder] += 1
            
            assignment = []
            for scale, count in zip(self.scales, counts):
                assignment.extend([scale] * count)
            
            random.shuffle(assignment)
            return assignment
        
        else:
            raise ValueError(f"Unknown balance_mode: {self.balance_mode}")
    
    def set_epoch(self, epoch):
        """Epoch마다 스케일 분포 변경"""
        if self.is_train and self.balance_mode == 'shuffle':
            random.seed(epoch)
            random.shuffle(self.scale_assignment)
            random.seed()
    
    def _load_training_data(self, hr_path, lr_path):
        """
        Training 데이터 로딩
        - HR: hr_path/0001000/0000001.png
        - LR: lr_path/0001000/0000001x2.png, 0000001x3.png, 0000001x4.png
        """
        print(f"  → Loading HR from: {hr_path}")
        print(f"  → Loading LR from: {lr_path}")
        
        if not os.path.exists(hr_path):
            print(f"  ❌ HR path does not exist: {hr_path}")
            return
        
        if not os.path.exists(lr_path):
            print(f"  ❌ LR path does not exist: {lr_path}")
            return
        
        # HR 이미지 수집 (재귀적으로 서브폴더 탐색)
        hr_images = sorted(glob.glob(os.path.join(hr_path, '**', '*.png'), recursive=True) + 
                          glob.glob(os.path.join(hr_path, '**', '*.jpg'), recursive=True) + 
                          glob.glob(os.path.join(hr_path, '**', '*.jpeg'), recursive=True))
        
        # train/train 폴더와 val1 폴더 제외 (LR 이미지들)
        hr_images = [img for img in hr_images if '/train/train/' not in img and '/val1/' not in img]
        
        print(f"  → Found {len(hr_images)} HR images")
        
        if len(hr_images) == 0:
            print(f"  ⚠️  No HR images found!")
            return
        
        # 각 HR 이미지에 대해 LR 이미지 찾기
        matched_count = 0
        for hr_img_path in hr_images:
            # HR 경로에서 상대 경로와 파일명 추출
            rel_path = os.path.relpath(hr_img_path, hr_path)
            hr_dir = os.path.dirname(rel_path)  # 예: 0001000
            hr_filename = os.path.basename(hr_img_path)
            hr_basename = os.path.splitext(hr_filename)[0]  # 예: 0000001
            hr_ext = os.path.splitext(hr_filename)[1]  # 예: .png
            
            lr_dict = {}
            
            # 각 스케일에 대해 LR 이미지 찾기
            for scale in self.scales:
                # LR 이미지 경로 구성
                lr_dir = os.path.join(lr_path, hr_dir) if hr_dir else lr_path
                lr_filename = f"{hr_basename}x{scale}{hr_ext}"
                lr_img_path = os.path.join(lr_dir, lr_filename)
                
                if os.path.exists(lr_img_path):
                    lr_dict[scale] = lr_img_path
            
            # 최소 1개 이상의 스케일이 있으면 추가
            if lr_dict:
                self.image_data.append((hr_img_path, lr_dict))
                matched_count += 1
        
        print(f"  → Successfully matched {matched_count} HR-LR pairs")
    
    def _load_validation_data(self, val_base_path):
        """
        Validation 데이터 로딩
        - HR: val_base_path/HR/val/0000001.png
        - X2: val_base_path/X2/val/0000001x2.png
        - X3: val_base_path/X3/val/0000001x3.png
        - X4: val_base_path/X4/val/0000001x4.png
        """
        print(f"  → Loading validation data from: {val_base_path}")
        
        hr_path = os.path.join(val_base_path, 'HR')
        
        if not os.path.exists(hr_path):
            print(f"  ❌ HR validation path does not exist: {hr_path}")
            return
        
        # 재귀적으로 서브폴더에서도 이미지 수집
        hr_images = sorted(glob.glob(os.path.join(hr_path, '**', '*.png'), recursive=True) + 
                          glob.glob(os.path.join(hr_path, '**', '*.jpg'), recursive=True) + 
                          glob.glob(os.path.join(hr_path, '**', '*.jpeg'), recursive=True))
        
        print(f"  → Found {len(hr_images)} HR validation images")
        
        if len(hr_images) == 0:
            print(f"  ⚠️  No HR images found in {hr_path}")
            return
        
        # 각 스케일 폴더 확인
        scale_paths = {}
        for scale in self.scales:
            scale_path = os.path.join(val_base_path, f'X{scale}')
            if os.path.exists(scale_path):
                scale_paths[scale] = scale_path
            else:
                print(f"  ⚠️  Scale folder not found: {scale_path}")
        
        # HR 이미지와 LR 이미지 매칭
        for hr_img_path in hr_images:
            # HR 경로에서 상대 경로 추출
            rel_path = os.path.relpath(hr_img_path, hr_path)
            hr_filename = os.path.basename(hr_img_path)
            hr_basename = os.path.splitext(hr_filename)[0]
            hr_ext = os.path.splitext(hr_filename)[1]
            hr_dir = os.path.dirname(rel_path)  # 예: "val"
            
            lr_dict = {}
            
            for scale, scale_path in scale_paths.items():
                # 서브폴더 구조를 유지하면서 LR 이미지 찾기
                lr_base_dir = os.path.join(scale_path, hr_dir) if hr_dir else scale_path
                
                # 다양한 파일명 패턴 시도
                patterns = [
                    f"{hr_basename}x{scale}{hr_ext}",  # 0000001x2.png
                    f"{hr_basename}{hr_ext}",  # 0000001.png
                    f"{hr_basename}_x{scale}{hr_ext}",
                    f"{hr_basename}X{scale}{hr_ext}",
                ]
                
                lr_found = None
                for pattern in patterns:
                    lr_path = os.path.join(lr_base_dir, pattern)
                    if os.path.exists(lr_path):
                        lr_found = lr_path
                        break
                
                if lr_found:
                    lr_dict[scale] = lr_found
            
            if lr_dict:
                self.image_data.append((hr_img_path, lr_dict))
        
        print(f"  → Matched {len(self.image_data)} validation pairs")
        for scale in self.scales:
            count = sum(1 for _, lr_dict in self.image_data if scale in lr_dict)
            print(f"     x{scale}: {count} pairs")
    
    def __len__(self):
        return len(self.image_data)
    
    def __getitem__(self, idx):
        hr_path, lr_dict = self.image_data[idx]
        
        # Scale selection
        if self.is_train:
            assigned_scale = self.scale_assignment[idx]
            if assigned_scale not in lr_dict:
                assigned_scale = list(lr_dict.keys())[0]
        else:
            assigned_scale = list(lr_dict.keys())[0]
        
        lr_path = lr_dict[assigned_scale]
        
        # Load images
        hr_img = Image.open(hr_path).convert('RGB')
        lq_img = Image.open(lr_path).convert('RGB')
        
        # Crop and augment
        if self.is_train:
            hr_img, lq_img = self._random_crop_pair(hr_img, lq_img, self.patch_size, assigned_scale)
            hr_img, lq_img = self._augment_pair(hr_img, lq_img)
        else:
            w, h = hr_img.size
            hr_img = hr_img.crop((0, 0, w - w % assigned_scale, h - h % assigned_scale))
            lq_w, lq_h = lq_img.size
            lq_img = lq_img.crop((0, 0, lq_w - lq_w % assigned_scale, lq_h - lq_h % assigned_scale))
        
        # Convert to tensor
        hr_tensor = transforms.ToTensor()(hr_img)
        lq_tensor = transforms.ToTensor()(lq_img)
        
        # Normalize LQ size
        if lq_tensor.shape[1] != self.normalize_size or lq_tensor.shape[2] != self.normalize_size:
            lq_tensor = F.interpolate(
                lq_tensor.unsqueeze(0), 
                size=(self.normalize_size, self.normalize_size),
                mode='bicubic',
                align_corners=False
            ).squeeze(0)
        
        # Additional augmentation
        if self.is_train and self.config:
            if self.config.use_color_jitter and random.random() < self.config.color_jitter_prob:
                lq_tensor = self.aug.apply_color_jitter(lq_tensor.unsqueeze(0)).squeeze(0)
            
            if self.config.use_gaussian_noise and random.random() < self.config.noise_prob:
                sigma = random.uniform(0, self.config.noise_sigma)
                lq_tensor = self.aug.add_gaussian_noise(lq_tensor, sigma)
        
        return {
            'lq': lq_tensor, 
            'gt': hr_tensor, 
            'scale': assigned_scale,
            'idx': idx
        }
    
    def _random_crop_pair(self, hr_img, lq_img, crop_size, scale):
        w, h = hr_img.size
        if w < crop_size or h < crop_size:
            hr_img = hr_img.resize((max(w, crop_size), max(h, crop_size)), Image.BICUBIC)
            lq_img = lq_img.resize((max(w, crop_size) // scale, 
                                   max(h, crop_size) // scale), Image.BICUBIC)
            w, h = hr_img.size
        
        x = random.randint(0, w - crop_size)
        y = random.randint(0, h - crop_size)
        
        hr_crop = hr_img.crop((x, y, x + crop_size, y + crop_size))
        lq_crop = lq_img.crop((x // scale, y // scale, 
                               (x + crop_size) // scale, 
                               (y + crop_size) // scale))
        
        return hr_crop, lq_crop
    
    def _augment_pair(self, hr_img, lq_img):
        if random.random() < 0.5:
            hr_img = hr_img.transpose(Image.FLIP_LEFT_RIGHT)
            lq_img = lq_img.transpose(Image.FLIP_LEFT_RIGHT)
        if random.random() < 0.5:
            hr_img = hr_img.transpose(Image.FLIP_TOP_BOTTOM)
            lq_img = lq_img.transpose(Image.FLIP_TOP_BOTTOM)
        if random.random() < 0.5:
            hr_img = hr_img.transpose(Image.ROTATE_90)
            lq_img = lq_img.transpose(Image.ROTATE_90)
        
        return hr_img, lq_img

def collate_fn_with_mixup(batch, config):
    lq = torch.stack([item['lq'] for item in batch])
    gt = torch.stack([item['gt'] for item in batch])
    scales = [item['scale'] for item in batch]
    
    if config.use_mixup and random.random() < config.mixup_prob:
        indices = torch.randperm(lq.size(0))
        lq2 = lq[indices]
        gt2 = gt[indices]
        
        aug = AdvancedAugmentation(config)
        gt, lq = aug.mixup(gt, lq, gt2, lq2, config.mixup_alpha)
    
    elif config.use_cutmix and random.random() < config.cutmix_prob:
        indices = torch.randperm(lq.size(0))
        lq2 = lq[indices]
        gt2 = gt[indices]
        
        aug = AdvancedAugmentation(config)
        gt, lq = aug.cutmix(gt, lq, gt2, lq2, config.cutmix_alpha, config.target_scale)
    
    return {'lq': lq, 'gt': gt, 'scales': scales}

# ================================
# Model Components (HAT Architecture)
# ================================
class MlpWithLoRA(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, 
                 act_layer=nn.GELU, drop=0., use_lora=False, lora_config=None):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
        
        if use_lora and lora_config is not None:
            self.fc1 = LinearWithLoRA(self.fc1, rank=lora_config['rank'],
                                     alpha=lora_config['alpha'], dropout=lora_config['dropout'])
            self.fc2 = LinearWithLoRA(self.fc2, rank=lora_config['rank'],
                                     alpha=lora_config['alpha'], dropout=lora_config['dropout'])

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

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

class WindowAttentionWithLoRA(nn.Module):
    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, 
                 attn_drop=0., proj_drop=0., use_lora=False, lora_config=None):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.softmax = nn.Softmax(dim=-1)
        
        if use_lora and lora_config is not None:
            self.qkv = LinearWithLoRA(self.qkv, rank=lora_config['rank'],
                                     alpha=lora_config['alpha'], dropout=lora_config['dropout'])
            self.proj = LinearWithLoRA(self.proj, rank=lora_config['rank'],
                                      alpha=lora_config['alpha'], dropout=lora_config['dropout'])

    def forward(self, x, mask=None):
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class HABWithLoRAEnhanced(nn.Module):
    def __init__(self, dim, input_resolution, num_heads, window_size=7, shift_size=0,
                 compress_ratio=3, squeeze_factor=30, conv_scale=0.01, mlp_ratio=4.,
                 qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm, 
                 use_lora=False, lora_config=None,
                 use_large_kernel=False, use_multi_scale=False, use_cross_scale=False,
                 use_frequency=False, use_adaptive_mod=False, use_enhanced_res=False):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        
        self.norm1 = norm_layer(dim)
        
        use_attn_lora = use_lora and lora_config is not None and 'qkv' in lora_config.get('targets', [])
        self.attn = WindowAttentionWithLoRA(
            dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop,
            use_lora=use_attn_lora, lora_config=lora_config if use_attn_lora else None)

        self.conv_scale = conv_scale
        self.conv_block = nn.Sequential(
            nn.Conv2d(dim, dim, 3, 1, 1, groups=dim),
            nn.GELU(),
            nn.Conv2d(dim, dim, 3, 1, 1, groups=dim)
        )

        self.use_large_kernel = use_large_kernel
        if use_large_kernel:
            self.large_kernel_attn = LargeKernelAttention(dim)
        
        self.use_multi_scale = use_multi_scale
        if use_multi_scale:
            self.multi_scale_fusion = MultiScaleFeatureFusion(dim)
        
        self.use_cross_scale = use_cross_scale
        if use_cross_scale:
            self.cross_scale_attn = CrossScaleAttention(dim, num_heads=4)
        
        self.use_frequency = use_frequency
        if use_frequency:
            self.freq_enhance = FrequencyAwareEnhancement(dim)
        
        self.use_adaptive_mod = use_adaptive_mod
        if use_adaptive_mod:
            self.adaptive_mod = AdaptiveFeatureModulation(dim)
        
        self.use_enhanced_res = use_enhanced_res
        if use_enhanced_res:
            self.enhanced_res = EnhancedResidualBlock(dim)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        
        use_mlp_lora = use_lora and lora_config is not None and 'mlp' in lora_config.get('targets', [])
        self.mlp = MlpWithLoRA(
            in_features=dim, hidden_features=mlp_hidden_dim, 
            act_layer=act_layer, drop=drop,
            use_lora=use_mlp_lora, lora_config=lora_config if use_mlp_lora else None)

    def forward(self, x, x_size):
        H, W = x_size
        B, L, C = x.shape
        
        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x

        x_windows = window_partition(shifted_x, self.window_size)
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)
        attn_windows = self.attn(x_windows, mask=None)
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)

        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
        
        x = x.view(B, H * W, C)
        
        x_img = shortcut.view(B, H, W, C).permute(0, 3, 1, 2).contiguous()
        conv_x = self.conv_block(x_img)
        
        enhanced_x = conv_x
        
        if self.use_large_kernel:
            enhanced_x = enhanced_x + self.large_kernel_attn(enhanced_x) * 0.01
        
        if self.use_multi_scale:
            enhanced_x = enhanced_x + self.multi_scale_fusion(enhanced_x) * 0.01
        
        if self.use_cross_scale:
            enhanced_x = enhanced_x + self.cross_scale_attn(enhanced_x) * 0.01
        
        if self.use_frequency:
            enhanced_x = self.freq_enhance(enhanced_x)
        
        if self.use_adaptive_mod:
            enhanced_x = self.adaptive_mod(enhanced_x)
        
        if self.use_enhanced_res:
            enhanced_x = self.enhanced_res(enhanced_x)
        
        enhanced_x = enhanced_x.permute(0, 2, 3, 1).contiguous().view(B, H * W, C)

        x = shortcut + self.drop_path(x) + enhanced_x * self.conv_scale
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x

class ResidualGroup(nn.Module):
    def __init__(self, blocks, conv):
        super().__init__()
        self.blocks = blocks
        self.conv = conv

class RHAGWithLoRAEnhanced(nn.Module):
    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 compress_ratio, squeeze_factor, conv_scale, mlp_ratio=4.,
                 qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False,
                 use_lora=False, lora_config=None,
                 use_large_kernel=False, use_multi_scale=False, use_cross_scale=False,
                 use_frequency=False, use_adaptive_mod=False, use_enhanced_res=False):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution

        blocks = nn.ModuleList([
            HABWithLoRAEnhanced(
                dim=dim, input_resolution=input_resolution,
                num_heads=num_heads, window_size=window_size,
                shift_size=0 if (i % 2 == 0) else window_size // 2,
                compress_ratio=compress_ratio, squeeze_factor=squeeze_factor,
                conv_scale=conv_scale, mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop, attn_drop=attn_drop,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer,
                use_lora=use_lora,
                lora_config=lora_config,
                use_large_kernel=use_large_kernel,
                use_multi_scale=use_multi_scale,
                use_cross_scale=use_cross_scale,
                use_frequency=use_frequency,
                use_adaptive_mod=use_adaptive_mod,
                use_enhanced_res=use_enhanced_res)
            for i in range(depth)])
        
        conv = nn.Conv2d(dim, dim, 3, 1, 1)
        self.residual_group = ResidualGroup(blocks, conv)

    def forward(self, x, x_size):
        for blk in self.residual_group.blocks:
            x = blk(x, x_size)
        
        B, L, C = x.shape
        H, W = x_size
        
        x_img = x.transpose(1, 2).view(B, C, H, W)
        x_img = self.residual_group.conv(x_img)
        x = x_img.view(B, C, -1).transpose(1, 2) + x
        
        return x

class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.proj(x).flatten(2).transpose(1, 2)
        if self.norm is not None:
            x = self.norm(x)
        return x

class PatchUnEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]
        self.in_chans = in_chans
        self.embed_dim = embed_dim

    def forward(self, x, x_size):
        B, HW, C = x.shape
        x = x.transpose(1, 2).view(B, self.embed_dim, x_size[0], x_size[1])
        return x

class Upsample(nn.Sequential):
    def __init__(self, scale, num_feat):
        m = []
        if (scale & (scale - 1)) == 0:
            for _ in range(int(math.log(scale, 2))):
                m.append(nn.Conv2d(num_feat, 4 * num_feat, 3, 1, 1))
                m.append(nn.PixelShuffle(2))
        elif scale == 3:
            m.append(nn.Conv2d(num_feat, 9 * num_feat, 3, 1, 1))
            m.append(nn.PixelShuffle(3))
        else:
            raise ValueError(f'scale {scale} is not supported.')
        super(Upsample, self).__init__(*m)

class HATWithLoRAEnhanced(nn.Module):
    def __init__(self, img_size=64, patch_size=1, in_chans=3,
                 embed_dim=96, depths=[6, 6, 6, 6], num_heads=[6, 6, 6, 6],
                 window_size=7, compress_ratio=3, squeeze_factor=30, conv_scale=0.01,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, ape=False, patch_norm=True,
                 use_checkpoint=False, upscale=2, img_range=1., upsampler='', resi_connection='1conv',
                 use_lora=False, lora_r=8, lora_alpha=16, lora_dropout=0.1, lora_targets=['qkv', 'proj', 'mlp'],
                 use_large_kernel=False, use_multi_scale=False, use_cross_scale=False,
                 use_frequency=False, use_adaptive_mod=False, use_enhanced_res=False,
                 **kwargs):
        super(HATWithLoRAEnhanced, self).__init__()
        num_in_ch = in_chans
        num_out_ch = in_chans
        num_feat = 64
        self.img_range = img_range
        self.window_size = window_size
        
        if in_chans == 3:
            rgb_mean = (0.4488, 0.4371, 0.4040)
            self.mean = torch.Tensor(rgb_mean).view(1, 3, 1, 1)
        else:
            self.mean = torch.zeros(1, 1, 1, 1)
        self.upscale = upscale
        self.upsampler = upsampler

        self.conv_first = nn.Conv2d(num_in_ch, embed_dim, 3, 1, 1)

        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.num_features = embed_dim
        self.mlp_ratio = mlp_ratio
        
        self.use_lora = use_lora
        lora_config = {
            'rank': lora_r,
            'alpha': lora_alpha,
            'dropout': lora_dropout,
            'targets': lora_targets
        } if use_lora else None

        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=embed_dim, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)
        num_patches = self.patch_embed.num_patches
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution

        self.patch_unembed = PatchUnEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=embed_dim, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)

        if self.ape:
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
            trunc_normal_(self.absolute_pos_embed, std=.02)

        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = RHAGWithLoRAEnhanced(
                dim=embed_dim,
                input_resolution=(patches_resolution[0], patches_resolution[1]),
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=window_size,
                compress_ratio=compress_ratio,
                squeeze_factor=squeeze_factor,
                conv_scale=conv_scale,
                mlp_ratio=self.mlp_ratio,
                qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                norm_layer=norm_layer,
                downsample=None,
                use_checkpoint=use_checkpoint,
                use_lora=use_lora,
                lora_config=lora_config,
                use_large_kernel=use_large_kernel,
                use_multi_scale=use_multi_scale,
                use_cross_scale=use_cross_scale,
                use_frequency=use_frequency,
                use_adaptive_mod=use_adaptive_mod,
                use_enhanced_res=use_enhanced_res)
            self.layers.append(layer)
        
        self.norm = norm_layer(self.num_features)

        if resi_connection == '1conv':
            self.conv_after_body = nn.Conv2d(embed_dim, embed_dim, 3, 1, 1)
        elif resi_connection == '3conv':
            self.conv_after_body = nn.Sequential(
                nn.Conv2d(embed_dim, embed_dim // 4, 3, 1, 1),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv2d(embed_dim // 4, embed_dim // 4, 1, 1, 0),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv2d(embed_dim // 4, embed_dim, 3, 1, 1))

        if self.upsampler == 'pixelshuffle':
            self.conv_before_upsample = nn.Sequential(
                nn.Conv2d(embed_dim, num_feat, 3, 1, 1),
                nn.LeakyReLU(inplace=True))
            self.upsample = Upsample(upscale, num_feat)
            self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_features(self, x):
        x_size = (x.shape[2], x.shape[3])
        x = self.patch_embed(x)
        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)

        for layer in self.layers:
            x = layer(x, x_size)

        x = self.norm(x)
        x = self.patch_unembed(x, x_size)

        return x

    def forward(self, x):
        H, W = x.shape[2:]
        x = self.check_image_size(x)

        self.mean = self.mean.type_as(x)
        x = (x - self.mean) * self.img_range

        if self.upsampler == 'pixelshuffle':
            x = self.conv_first(x)
            x = self.conv_after_body(self.forward_features(x)) + x
            x = self.conv_before_upsample(x)
            x = self.conv_last(self.upsample(x))

        x = x / self.img_range + self.mean

        return x[:, :, :H*self.upscale, :W*self.upscale]

    def check_image_size(self, x):
        _, _, h, w = x.size()
        mod_pad_h = (self.window_size - h % self.window_size) % self.window_size
        mod_pad_w = (self.window_size - w % self.window_size) % self.window_size
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
        return x

# ================================
# Pretrained Model Loading
# ================================
def load_pretrained_with_lora_support(model, pretrained_path, local_rank=0):
    if not os.path.exists(pretrained_path):
        if local_rank == 0:
            print(f"⚠️  Pretrained model not found: {pretrained_path}")
        return model
    
    if local_rank == 0:
        print(f"\nLoading pretrained model from {pretrained_path}")
    
    checkpoint = torch.load(pretrained_path, map_location='cpu', weights_only=False)
    
    if 'params_ema' in checkpoint:
        state_dict = checkpoint['params_ema']
    elif 'params' in checkpoint:
        state_dict = checkpoint['params']
    elif 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint
    
    new_state_dict = {}
    for key, value in state_dict.items():
        new_key = key
        
        if any(x in key for x in ['overlap_attn']):
            continue
        
        if config.use_lora:
            if '.attn.qkv.weight' in key or '.attn.qkv.bias' in key:
                new_key = key.replace('.attn.qkv.', '.attn.qkv.linear.')
            elif '.attn.proj.weight' in key or '.attn.proj.bias' in key:
                new_key = key.replace('.attn.proj.', '.attn.proj.linear.')
            elif '.mlp.fc1.weight' in key or '.mlp.fc1.bias' in key:
                new_key = key.replace('.mlp.fc1.', '.mlp.fc1.linear.')
            elif '.mlp.fc2.weight' in key or '.mlp.fc2.bias' in key:
                new_key = key.replace('.mlp.fc2.', '.mlp.fc2.linear.')
        
        new_state_dict[new_key] = value
    
    missing_keys, unexpected_keys = model.load_state_dict(new_state_dict, strict=False)
    
    if local_rank == 0:
        print("✓ Pretrained model loaded successfully!")
        print(f"  Missing keys: {len(missing_keys)}")
        print(f"  Unexpected keys: {len(unexpected_keys)}")
        
        critical_loaded = sum(1 for k in new_state_dict.keys() if any(x in k for x in ['attn.qkv', 'attn.proj', 'mlp.fc1', 'mlp.fc2']))
        print(f"  Critical weights loaded: {critical_loaded}")
    
    return model

# ================================
# Metrics
# ================================
def calculate_psnr(img1, img2):
    img1 = torch.clamp(img1, 0, 1)
    img2 = torch.clamp(img2, 0, 1)
    
    mse = torch.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * torch.log10(1.0 / torch.sqrt(mse))

def calculate_ssim(img1, img2, window_size=11):
    img1 = torch.clamp(img1, 0, 1)
    img2 = torch.clamp(img2, 0, 1)
    
    C1 = (0.01) ** 2
    C2 = (0.03) ** 2
    
    sigma = 1.5
    gauss = torch.Tensor([math.exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    gauss = gauss / gauss.sum()
    
    _1D_window = gauss.unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(img1.size(1), 1, window_size, window_size).contiguous()
    window = window.to(img1.device).type_as(img1)
    
    mu1 = F.conv2d(img1, window, padding=window_size//2, groups=img1.size(1))
    mu2 = F.conv2d(img2, window, padding=window_size//2, groups=img2.size(1))
    
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2
    
    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size//2, groups=img1.size(1)) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size//2, groups=img2.size(1)) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size//2, groups=img2.size(1)) - mu1_mu2
    
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    
    return ssim_map.mean()

# ================================
# Training Functions
# ================================
def setup_ddp():
    dist.init_process_group(backend='nccl')
    local_rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local_rank)
    return local_rank

def cleanup_ddp():
    dist.destroy_process_group()

def get_optimizer(model, config):
    trainable_params = []
    
    for name, param in model.named_parameters():
        if param.requires_grad:
            trainable_params.append(param)
    
    if len(trainable_params) == 0:
        raise ValueError("No trainable parameters found!")
    
    return torch.optim.AdamW(
        trainable_params,
        lr=config.lr,
        betas=config.betas,
        weight_decay=config.weight_decay
    )

def get_scheduler(optimizer, config, steps_per_epoch):
    warmup_steps = config.warmup_epochs * steps_per_epoch
    total_steps = config.num_epochs * steps_per_epoch
    
    if total_steps == 0 or total_steps <= warmup_steps:
        print(f"⚠️  Warning: Invalid scheduler parameters!")
        print(f"   total_steps: {total_steps}, warmup_steps: {warmup_steps}")
        print(f"   Using constant LR scheduler instead.")
        return torch.optim.lr_scheduler.ConstantLR(optimizer, factor=1.0)
    
    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps if warmup_steps > 0 else 1.0
        else:
            progress = (step - warmup_steps) / (total_steps - warmup_steps)
            return 0.5 * (1.0 + math.cos(math.pi * progress))
    
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

@torch.no_grad()
def validate(model, val_loader, device, epoch, local_rank, save_dir, num_save_images=10, ema=None):
    if ema is not None:
        parameters = [p for p in model.parameters() if p.requires_grad]
        ema.store(parameters)
        ema.copy_to(parameters)
    
    model.eval()
    
    scale_metrics = {scale: {'psnr': [], 'ssim': []} for scale in config.val_scales}
    
    if local_rank == 0:
        val_img_dir = os.path.join(save_dir, 'validation_images', f'epoch_{epoch:03d}')
        os.makedirs(val_img_dir, exist_ok=True)
    
    saved_count = {scale: 0 for scale in config.val_scales}
    
    for batch_idx, batch in enumerate(val_loader):
        lq = batch['lq'].to(device)
        gt = batch['gt'].to(device)
        scales = batch['scales']
        
        sr = model(lq)
        
        for i in range(lq.size(0)):
            scale = scales[i]
            psnr = calculate_psnr(sr[i:i+1], gt[i:i+1])
            ssim = calculate_ssim(sr[i:i+1], gt[i:i+1])
            
            scale_metrics[scale]['psnr'].append(psnr.item())
            scale_metrics[scale]['ssim'].append(ssim.item())
            
            if local_rank == 0 and saved_count[scale] < num_save_images:
                sr_img = sr[i].cpu().clamp(0, 1).numpy().transpose(1, 2, 0)
                gt_img = gt[i].cpu().clamp(0, 1).numpy().transpose(1, 2, 0)
                lq_img = lq[i].cpu().clamp(0, 1).numpy().transpose(1, 2, 0)
                
                sr_img = (sr_img * 255).astype(np.uint8)
                gt_img = (gt_img * 255).astype(np.uint8)
                lq_img = (lq_img * 255).astype(np.uint8)
                
                prefix = f'x{scale}_{saved_count[scale]:03d}'
                Image.fromarray(sr_img).save(os.path.join(val_img_dir, f'{prefix}_sr.png'))
                Image.fromarray(gt_img).save(os.path.join(val_img_dir, f'{prefix}_gt.png'))
                Image.fromarray(lq_img).save(os.path.join(val_img_dir, f'{prefix}_lq.png'))
                
                h, w = sr_img.shape[:2]
                lq_resized = np.array(Image.fromarray(lq_img).resize((w, h), Image.BICUBIC))
                comparison = np.concatenate([lq_resized, sr_img, gt_img], axis=1)
                
                comp_pil = Image.fromarray(comparison)
                draw = ImageDraw.Draw(comp_pil)
                
                try:
                    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20)
                except:
                    font = ImageFont.load_default()
                
                text = f"x{scale} | PSNR: {psnr.item():.2f} dB | SSIM: {ssim.item():.4f}"
                draw.text((10, 10), text, fill=(255, 255, 0), font=font)
                comp_pil.save(os.path.join(val_img_dir, f'{prefix}_comparison.png'))
                
                saved_count[scale] += 1
        
        if batch_idx >= 20:
            break
    
    avg_metrics = {}
    for scale in config.val_scales:
        if len(scale_metrics[scale]['psnr']) > 0:
            avg_psnr = np.mean(scale_metrics[scale]['psnr'])
            avg_ssim = np.mean(scale_metrics[scale]['ssim'])
            avg_metrics[scale] = {'psnr': avg_psnr, 'ssim': avg_ssim}
    
    if local_rank == 0:
        print(f'\n{"="*60}')
        print(f'Epoch [{epoch}] Validation Results:')
        for scale in config.val_scales:
            if scale in avg_metrics:
                print(f'  x{scale}: PSNR={avg_metrics[scale]["psnr"]:.4f} dB, '
                      f'SSIM={avg_metrics[scale]["ssim"]:.6f}')
        
        all_psnr = [avg_metrics[s]['psnr'] for s in avg_metrics]
        all_ssim = [avg_metrics[s]['ssim'] for s in avg_metrics]
        if all_psnr:
            print(f'  Overall: PSNR={np.mean(all_psnr):.4f} dB, SSIM={np.mean(all_ssim):.6f}')
        print(f'{"="*60}\n')
    
    if ema is not None:
        parameters = [p for p in model.parameters() if p.requires_grad]
        ema.restore(parameters)
    
    overall_psnr = np.mean(all_psnr) if all_psnr else 0
    overall_ssim = np.mean(all_ssim) if all_ssim else 0
    return overall_psnr, overall_ssim, avg_metrics

def train_epoch(model, train_loader, optimizer, scheduler, criterion, scaler, device, 
                epoch, config, local_rank, ema=None, dataset=None):
    model.train()
    total_loss = 0
    total_l1 = 0
    total_perceptual = 0
    total_psnr = 0
    
    scale_counts = {scale: 0 for scale in config.train_scales}
    
    for batch_idx, batch in enumerate(train_loader):
        lq = batch['lq'].to(device)
        gt = batch['gt'].to(device)
        scales = batch['scales']
        
        for scale in scales:
            scale_counts[scale] += 1
        
        optimizer.zero_grad()
        
        if config.use_mixed_precision:
            with autocast('cuda'):
                sr = model(lq)
                loss, loss_dict = criterion(sr, gt)
            
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"[WARNING] NaN/Inf detected at batch {batch_idx}, skipping...")
                continue
            
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.gradient_clip_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            sr = model(lq)
            loss, loss_dict = criterion(sr, gt)
            
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"[WARNING] NaN/Inf detected at batch {batch_idx}, skipping...")
                continue
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.gradient_clip_norm)
            optimizer.step()
        
        if ema is not None:
            parameters = [p for p in model.parameters() if p.requires_grad]
            ema.update(parameters)
        
        scheduler.step()
        
        with torch.no_grad():
            batch_psnr = calculate_psnr(sr.detach(), gt.detach())
        
        if not (torch.isnan(loss) or torch.isinf(loss)):
            total_loss += loss.item()
            total_l1 += loss_dict.get('l1', 0)
            if 'perceptual' in loss_dict:
                total_perceptual += loss_dict['perceptual']
            total_psnr += batch_psnr.item()
        
        if local_rank == 0 and (batch_idx + 1) % config.print_freq == 0:
            avg_loss = total_loss / (batch_idx + 1)
            avg_psnr = total_psnr / (batch_idx + 1)
            lr = optimizer.param_groups[0]['lr']
            
            scale_dist = ", ".join([f"x{s}:{scale_counts[s]}" for s in config.train_scales])
            
            log_str = (f'Epoch [{epoch}/{config.num_epochs}] '
                      f'Batch [{batch_idx + 1}/{len(train_loader)}] '
                      f'Loss: {loss.item():.4f} (Avg: {avg_loss:.4f}) '
                      f'L1: {loss_dict.get("l1", 0):.4f} ')
            
            if config.use_perceptual_loss:
                log_str += f'Percept: {loss_dict.get("perceptual", 0):.4f} '
            
            log_str += f'PSNR: {batch_psnr.item():.2f}dB (Avg: {avg_psnr:.2f}dB) '
            log_str += f'LR: {lr:.6f} | Scales: {scale_dist}'
            print(log_str)
    
    return total_loss / len(train_loader)

def main():
    local_rank = setup_ddp()
    device = torch.device(f'cuda:{local_rank}')
    
    if local_rank == 0:
        os.makedirs(config.save_dir, exist_ok=True)
        print("=" * 80)
        print("✅ HAT with LoRA - Balanced Multi-Scale Training (LSDIR)")
        print("=" * 80)
        print(f"🎯 Training scales: {config.train_scales}")
        print(f"📊 Validation scales: {config.val_scales}")
        print(f"⚖️  Balance mode: {config.scale_balance_mode}")
        print(f"🔧 Unified input size: {config.normalize_input_size}x{config.normalize_input_size}")
        print(f"⬆️  Model output scale: x{config.target_scale}")
        print("=" * 80)
        print(f"\n📁 Dataset Paths:")
        print(f"   HR Training: {config.hr_train_path}")
        print(f"   LR Training: {config.lr_train_path}")
        print(f"   Validation: {config.val_base_path}")
        print("=" * 80)
    
    # Dataset with error handling
    try:
        train_dataset = BalancedMultiScaleLSDirectDataset(
            hr_path=config.hr_train_path,
            lr_path=config.lr_train_path,
            scales=config.train_scales,
            patch_size=config.hr_patch_size,
            normalize_size=config.normalize_input_size,
            is_train=True,
            config=config,
            balance_mode=config.scale_balance_mode
        )
    except RuntimeError as e:
        if local_rank == 0:
            print(f"\n{e}")
            print(f"\n💡 Solution:")
            print(f"   Check if images exist:")
            print(f"   HR: {config.hr_train_path}/0001000/0000001.png")
            print(f"   LR: {config.lr_train_path}/0001000/0000001x2.png")
        cleanup_ddp()
        return
    
    try:
        val_dataset = BalancedMultiScaleLSDirectDataset(
            val_base_path=config.val_base_path,
            scales=config.val_scales,
            patch_size=config.hr_patch_size,
            normalize_size=config.normalize_input_size,
            is_train=False,
            config=None,
            balance_mode='cycle'
        )
        has_validation = True
    except RuntimeError as e:
        if local_rank == 0:
            print(f"\n⚠️  Validation dataset not available: {e}")
            print(f"   Training will continue without validation.")
        has_validation = False
        val_dataset = None
    
    # DataLoaders
    train_sampler = DistributedSampler(train_dataset)
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        sampler=train_sampler,
        num_workers=config.num_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=lambda batch: collate_fn_with_mixup(batch, config)
    )
    
    val_loader = None
    if has_validation and val_dataset is not None:
        val_sampler = DistributedSampler(val_dataset, shuffle=False)
        val_loader = DataLoader(
            val_dataset,
            batch_size=1,
            sampler=val_sampler,
            num_workers=config.num_workers,
            pin_memory=True
        )
    
    # Model
    model = HATWithLoRAEnhanced(
        img_size=config.img_size,
        patch_size=1,
        in_chans=3,
        embed_dim=config.embed_dim,
        depths=config.depths,
        num_heads=config.num_heads,
        window_size=config.window_size,
        compress_ratio=config.compress_ratio,
        squeeze_factor=config.squeeze_factor,
        conv_scale=config.conv_scale,
        mlp_ratio=config.mlp_ratio,
        qkv_bias=config.qkv_bias,
        qk_scale=config.qk_scale,
        drop_rate=config.drop_rate,
        attn_drop_rate=config.attn_drop_rate,
        drop_path_rate=config.drop_path_rate,
        ape=config.ape,
        patch_norm=config.patch_norm,
        use_checkpoint=config.use_checkpoint,
        upscale=config.upscale,
        img_range=config.img_range,
        upsampler=config.upsampler,
        resi_connection=config.resi_connection,
        use_lora=config.use_lora,
        lora_r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        lora_targets=config.lora_targets,
        use_large_kernel=config.use_large_kernel_attn,
        use_multi_scale=config.use_multi_scale,
        use_cross_scale=config.use_cross_scale_attn,
        use_frequency=config.use_frequency_aware,
        use_adaptive_mod=config.use_adaptive_modulation,
        use_enhanced_res=config.use_enhanced_residual
    )
    
    model = load_pretrained_with_lora_support(model, config.pretrained_model, local_rank)
    
    # Freeze parameters
    if config.use_lora:
        if local_rank == 0:
            print("\n" + "=" * 80)
            print("🔒 FREEZING PARAMETERS")
            print("=" * 80)
        
        for param in model.parameters():
            param.requires_grad = False
        
        lora_params = 0
        for name, param in model.named_parameters():
            if 'lora' in name.lower():
                param.requires_grad = True
                lora_params += param.numel()
        
        enhancement_params = 0
        if config.train_enhancement_modules:
            for name, param in model.named_parameters():
                if any(x in name for x in ['large_kernel', 'multi_scale', 'cross_scale', 
                                           'freq_enhance', 'adaptive_mod', 'enhanced_res']):
                    param.requires_grad = True
                    enhancement_params += param.numel()
        
        if local_rank == 0:
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            
            print(f"✓ Total parameters:      {total_params:,}")
            print(f"✓ LoRA parameters:       {lora_params:,}")
            print(f"✓ Enhancement params:    {enhancement_params:,}")
            print(f"✓ Trainable total:       {trainable_params:,}")
            print(f"✓ Trainable ratio:       {100 * trainable_params / total_params:.2f}%")
            print("=" * 80 + "\n")
    
    model = model.to(device)
    model = DDP(model, device_ids=[local_rank], find_unused_parameters=False)
    
    criterion = CombinedLoss(
        use_perceptual=config.use_perceptual_loss,
        perceptual_weight=config.perceptual_weight,
        use_multiscale=config.use_multiscale_loss,
        multiscale_weights=config.multiscale_weights,
        device=device,
        local_rank=local_rank
    ).to(device)
    
    optimizer = get_optimizer(model.module if hasattr(model, 'module') else model, config)
    scheduler = get_scheduler(optimizer, config, len(train_loader))
    
    scaler = GradScaler('cuda') if config.use_mixed_precision else None
    
    ema = None
    if config.use_ema:
        parameters = [p for p in model.parameters() if p.requires_grad]
        ema = ExponentialMovingAverage(parameters, decay=config.ema_decay)
    
    best_psnr = 0
    
    if local_rank == 0:
        metrics_log = os.path.join(config.save_dir, 'metrics_log.txt')
        with open(metrics_log, 'w') as f:
            f.write('Epoch,Loss,Overall_PSNR,Overall_SSIM')
            for scale in config.val_scales:
                f.write(f',x{scale}_PSNR,x{scale}_SSIM')
            f.write('\n')
    
    # Training loop
    for epoch in range(1, config.num_epochs + 1):
        train_sampler.set_epoch(epoch)
        
        if config.scale_balance_mode == 'shuffle':
            train_dataset.set_epoch(epoch)
        
        avg_loss = train_epoch(model, train_loader, optimizer, scheduler, criterion, 
                               scaler, device, epoch, config, local_rank, ema, train_dataset)
        
        if local_rank == 0:
            print(f'\nEpoch [{epoch}/{config.num_epochs}] Training Loss: {avg_loss:.4f}')
        
        if val_loader is not None:
            overall_psnr, overall_ssim, scale_metrics = validate(
                model, val_loader, device, epoch, local_rank, 
                config.save_dir, config.num_val_images, ema
            )
            
            if local_rank == 0:
                with open(metrics_log, 'a') as f:
                    f.write(f'{epoch},{avg_loss:.6f},{overall_psnr:.4f},{overall_ssim:.6f}')
                    for scale in config.val_scales:
                        if scale in scale_metrics:
                            f.write(f',{scale_metrics[scale]["psnr"]:.4f},{scale_metrics[scale]["ssim"]:.6f}')
                        else:
                            f.write(',0.0,0.0')
                    f.write('\n')
                
                if overall_psnr > best_psnr:
                    best_psnr = overall_psnr
                    save_path = os.path.join(config.save_dir, 'best_model.pth')
                    
                    save_dict = {
                        'epoch': epoch,
                        'state_dict': model.module.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'psnr': overall_psnr,
                        'ssim': overall_ssim,
                        'scale_metrics': scale_metrics
                    }
                    
                    if ema is not None:
                        save_dict['ema'] = ema.state_dict()
                        
                        parameters = [p for p in model.module.parameters() if p.requires_grad]
                        ema.store(parameters)
                        ema.copy_to(parameters)
                        save_dict['state_dict_ema'] = model.module.state_dict()
                        ema.restore(parameters)
                    
                    torch.save(save_dict, save_path)
                    print(f'🌟 Best model saved! Overall PSNR: {overall_psnr:.4f} dB')
        
        if local_rank == 0 and epoch % config.save_freq == 0:
            save_dict = {
                'epoch': epoch,
                'state_dict': model.module.state_dict(),
                'optimizer': optimizer.state_dict()
            }
            if ema is not None:
                save_dict['ema'] = ema.state_dict()
            
            torch.save(save_dict, os.path.join(config.save_dir, f'checkpoint_epoch_{epoch}.pth'))
    
    cleanup_ddp()
    
    if local_rank == 0:
        print("\n" + "="*60)
        print("🎉 Training completed!")
        print(f"📊 Best Overall PSNR: {best_psnr:.4f} dB")
        print(f"📁 All results saved to: {config.save_dir}")
        print("="*60)

if __name__ == '__main__':
    main()