# hat/archs/hatx_arch.py
import math
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
import torch.distributed as dist
from typing import Tuple

from basicsr.utils.registry import ARCH_REGISTRY
from basicsr.archs.arch_util import to_2tuple, trunc_normal_
from einops import rearrange

# =========================
# ESC 원본 사용 (esc_arch.py)
# =========================
try:
    from .esc_arch import ConvolutionalAttention, ConvAttnWrapper  # 원본 ESC 모듈
    _HAS_ESC = True
except Exception as _e:
    _HAS_ESC = False
    raise ImportError(
        "[HAT][ESC][FATAL] esc_arch.py 를 불러오지 못했습니다.\n"
        " - 파일 경로: hat/archs/esc_arch.py\n"
        f" - 원인: {repr(_e)}\n"
        "ESC 연동이 필수이니 경로/의존성을 확인하세요."
    )

# ==== DEBUG UTILS ====
HAT_DEBUG = os.environ.get("HAT_DEBUG", "0") == "1"

def _is_master():
    return (not dist.is_available()) or (not dist.is_initialized()) or dist.get_rank() == 0

def _dprint(*args, **kwargs):
    if HAT_DEBUG and _is_master():
        print(*args, **kwargs)

_DEBUG_ONCE = {"hab_shift": False, "hab_masksoft": False}

# ===== DropPath =====
def drop_path(x, drop_prob: float = 0., training: bool = False):
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()
    output = x.div(keep_prob) * random_tensor
    return output

class DropPath(nn.Module):
    def __init__(self, drop_prob=None):
        super().__init__()
        self.drop_prob = drop_prob
    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)

# =========================
# Channel Attention (ECA) + CAB
# =========================
class ECA(nn.Module):
    def __init__(self, channels, k_size=5):
        super().__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=k_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        y = self.gap(x)
        y = y.squeeze(-1).transpose(-1, -2)
        y = self.conv(y)
        y = y.transpose(-1, -2).unsqueeze(-1)
        y = self.sigmoid(y)
        return x * y

class CAB(nn.Module):
    def __init__(self, num_feat, compress_ratio=3, squeeze_factor=30):
        super().__init__()
        self.cab = nn.Sequential(
            nn.Conv2d(num_feat, num_feat // compress_ratio, 3, 1, 1),
            nn.GELU(),
            nn.Conv2d(num_feat // compress_ratio, num_feat, 3, 1, 1),
            ECA(num_feat, k_size=5)
        )
    def forward(self, x):
        return self.cab(x)

# =========================
# Gated Depthwise-Conv FFN (GLU 스타일)
# =========================
class GatedDconvFFN(nn.Module):
    def __init__(self, dim, mlp_ratio: float = 2.0, drop: float = 0.0,
                 dw_kernel_size: int = 3, act_layer: str = "silu", bias: bool = True):
        super().__init__()
        hidden = int(dim * mlp_ratio)
        self.fc1 = nn.Linear(dim, 2 * hidden, bias=bias)
        self.dw  = nn.Conv2d(2 * hidden, 2 * hidden, kernel_size=dw_kernel_size,
                             stride=1, padding=dw_kernel_size // 2, groups=2 * hidden, bias=bias)
        self.act = nn.SiLU() if act_layer.lower() == "silu" else nn.GELU()
        self.fc2 = nn.Linear(hidden, dim, bias=bias)
        self.drop = nn.Dropout(drop) if drop and drop > 0 else nn.Identity()

    def forward(self, x, x_size):
        B, N, C = x.shape
        H, W = x_size
        assert N == H * W, f"GatedDconvFFN: N({N}) != H*W({H*W})"
        x = self.fc1(x)
        x = x.transpose(1, 2).contiguous().view(B, -1, H, W)
        x = self.dw(x)
        x = x.view(B, -1, H * W).transpose(1, 2).contiguous()
        x_proj, x_gate = x.chunk(2, dim=-1)
        x = x_proj * self.act(x_gate)
        x = self.fc2(x)
        x = self.drop(x)
        return x

# =========================
# Window helpers
# =========================
def window_partition(x, window_size):
    b, h, w, c = x.shape
    x = x.view(b, h // window_size, window_size, w // window_size, window_size, c)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, c)
    return windows

def window_reverse(windows, window_size, h, w):
    b = int(windows.shape[0] / (h * w / window_size / window_size))
    x = windows.view(b, h // window_size, w // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(b, h, w, -1)
    return x

# =========================
# ESCConvAttn_BNC: (B,N,C) <-> (B,C,H,W) 래퍼 (HAB용)
# =========================
class ESCConvAttn_BNC(nn.Module):
    def __init__(self, dim: int, pdim: int = 16, kernel_size: int = 13):
        super().__init__()
        if not _HAS_ESC:
            raise RuntimeError("[HAB][ESC] esc_arch 모듈이 필요합니다.")
        self.dim = dim
        self.pdim = pdim
        self.kernel_size = kernel_size
        self.core = ConvAttnWrapper(dim=dim, pdim=pdim, kernel_size=kernel_size)
        self.plk_filter = nn.Parameter(torch.randn(pdim, pdim, kernel_size, kernel_size))
        torch.nn.init.orthogonal_(self.plk_filter)

    @staticmethod
    def _bnc_to_bchw(x, x_size):
        B, N, C = x.shape
        H, W = x_size
        assert N == H * W, f"N({N}) != H*W({H*W})"
        return x.transpose(1, 2).contiguous().view(B, C, H, W)

    @staticmethod
    def _bchw_to_bnc(x):
        B, C, H, W = x.shape
        return x.view(B, C, H * W).transpose(1, 2).contiguous()

    def forward(self, x_bnc, x_size):
        x_bchw = self._bnc_to_bchw(x_bnc, x_size)
        y_bchw = self.core(x_bchw, self.plk_filter)
        y_bnc  = self._bchw_to_bnc(y_bchw)
        return y_bnc

# =========================
# FFT-Gated Branch (저/중/고 3-밴드) ⊕ ESC 듀얼
# =========================
class FFTGatedBranch(nn.Module):
    def __init__(self, channels, bands=3, res_scale=0.15):
        super().__init__()
        assert bands in (2, 3)
        self.bands = bands
        self.proj_per_band = nn.ModuleList([nn.Conv2d(channels, channels, 1) for _ in range(bands)])
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, bands, 1),
            nn.Sigmoid()
        )
        self.res_scale = res_scale

    def _fft_split(self, x):
        B, C, H, W = x.shape
        x32 = x.float()
        X = torch.fft.rfft2(x32, norm="ortho")
        yy = torch.linspace(-1, 1, H, device=x.device).view(H, 1).expand(H, W // 2 + 1)
        xx = torch.linspace(0, 1, W // 2 + 1, device=x.device).view(1, W // 2 + 1).expand(H, W // 2 + 1)
        rr = (yy ** 2 + xx ** 2).sqrt()
        if self.bands == 3:
            t1, t2 = 0.25, 0.6
            masks = [(rr <= t1), ((rr > t1) & (rr <= t2)), (rr > t2)]
        else:
            t = 0.45
            masks = [(rr <= t), (rr > t)]
        outs = []
        for m in masks:
            m = m[None, None, :, :]
            Y = X * m
            y = torch.fft.irfft2(Y, s=(H, W), norm="ortho")
            outs.append(y.type_as(x))
        return outs

    def forward(self, x_img):
        bands = self._fft_split(x_img)
        gated = [self.proj_per_band[i](feat) for i, feat in enumerate(bands)]
        gates = self.gate(x_img).softmax(dim=1)  # (B,bands,1,1)
        out = 0
        for i, t in enumerate(gated):
            out = out + t * gates[:, i:i + 1]
        return x_img + out * self.res_scale

# =========================
# HAB: ESC + FFT-Gated + CAB + GLU-FFN
# =========================
class HAB(nn.Module):
    def __init__(self,
                 dim,
                 input_resolution,
                 num_heads,
                 window_size=7,
                 shift_size=0,
                 compress_ratio=3,
                 squeeze_factor=30,
                 conv_scale=0.01,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm,
                 esc_pdim: int = 16,
                 esc_kernel: int = 13,
                 fft_bands: int = 3,
                 fft_res_scale: float = 0.15):
        super().__init__()
        if not _HAS_ESC:
            raise RuntimeError("[HAB][ESC] ESC가 필수인데 초기화 시점에 존재하지 않습니다.")

        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        self.conv_scale = conv_scale

        if min(self.input_resolution) <= self.window_size:
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size

        self.norm1 = norm_layer(dim)
        self.esc_attn = ESCConvAttn_BNC(dim=dim, pdim=esc_pdim, kernel_size=esc_kernel)
        self.fft_branch = FFTGatedBranch(channels=dim, bands=fft_bands, res_scale=fft_res_scale)
        self.conv_block = CAB(num_feat=dim, compress_ratio=compress_ratio, squeeze_factor=squeeze_factor)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = GatedDconvFFN(dim=dim, mlp_ratio=max(2.0, mlp_ratio/2), drop=drop, dw_kernel_size=3, act_layer="silu")

    def forward(self, x, x_size, params=None):
        h, w = x_size
        b, _, c = x.shape

        if HAT_DEBUG and _is_master() and self.shift_size > 0 and not _DEBUG_ONCE["hab_shift"]:
            _DEBUG_ONCE["hab_shift"] = True
            _dprint(f"[HAB][shift] shift_size={self.shift_size}, window_size={self.window_size}, "
                    f"x_size=({h},{w}), tokens={b*h*w}, dim={c}")

        mask_soft_tokens = None
        if params is not None and ("mask_soft_tokens" in params):
            mask_soft_tokens = params["mask_soft_tokens"]  # (H*W,)

        shortcut = x
        x = self.norm1(x)
        x_img = x.view(b, h, w, c).permute(0, 3, 1, 2).contiguous()

        # FFT-gated + Local conv
        x_fft = self.fft_branch(x_img)
        conv_x = self.conv_block(x_fft)                           # (B,C,H,W)
        conv_x = conv_x.permute(0, 2, 3, 1).contiguous().view(b, h * w, c)

        # ESC (global conv-attn) with optional soft token weights
        attn_x = self.esc_attn(x, (h, w))                         # (B,N,C)
        if mask_soft_tokens is not None:
            attn_x = attn_x * mask_soft_tokens.view(1, -1, 1).to(attn_x.dtype)

        x = shortcut + self.drop_path(attn_x) + conv_x * self.conv_scale
        x = x + self.drop_path(self.mlp(self.norm2(x), (h, w)))
        return x

# =========================
# Cross-Scale Attention + 작은 메모리 뱅크  (OCAB 대체)
# =========================
class CrossScaleAttentionMem(nn.Module):
    """
    Cross-Scale Attention + 작은 메모리 뱅크(슬롯 수: mem_slots).
    - 다운샘플 토큰(T=h2*w2)을 메모리 슬롯수(M)로 요약해 EMA 업데이트.
    - 과거 오류(T!=M) 해결: adaptive avg pooling으로 T->M 축약 후 EMA.
    """
    def __init__(
        self,
        dim: int,
        num_heads: int = 6,
        mem_slots: int = 8,
        ema: float = 0.99,
        down_scale: int = 2,
        qkv_bias: bool = True,
        mlp_ratio: float = 2.0,
        norm_layer=nn.LayerNorm
    ):
        super().__init__()
        assert dim % num_heads == 0, "dim must be divisible by num_heads."
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        # cross-scale 생성
        self.down_scale = down_scale
        self.pool = nn.AvgPool2d(kernel_size=down_scale, stride=down_scale, ceil_mode=False, count_include_pad=False)

        # Q/K/V 투영
        self.norm_q  = norm_layer(dim)
        self.norm_kv = norm_layer(dim)
        self.q_proj  = nn.Linear(dim, dim,  bias=qkv_bias)
        self.k_proj  = nn.Linear(dim, dim,  bias=qkv_bias)
        self.v_proj  = nn.Linear(dim, dim,  bias=qkv_bias)

        # 출력 투영
        self.proj = nn.Linear(dim, dim)

        # 작은 메모리 뱅크(슬롯 기반)
        self.mem_slots = mem_slots
        self.ema = float(ema)
        self.register_buffer("mem_k", torch.zeros(1, mem_slots, dim))
        self.register_buffer("mem_v", torch.zeros(1, mem_slots, dim))
        self._mem_init = False

        # FFN
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Linear(mlp_hidden_dim, dim),
        )

    @torch.no_grad()
    def _init_memory_if_needed(self, x_bnc: torch.Tensor):
        """
        x_bnc: (B, N, C)
        간단 초기화: 입력 평균을 M 슬롯으로 복제 → (1, M, C)
        """
        if not self._mem_init:
            B, N, C = x_bnc.shape
            mean_tok = x_bnc.mean(dim=1, keepdim=True)            # (B,1,C)
            k0 = mean_tok.repeat(1, self.mem_slots, 1)            # (B,M,C)
            v0 = torch.zeros_like(k0)                             # (B,M,C)
            self.mem_k = k0.mean(dim=0, keepdim=True).detach()    # (1,M,C)
            self.mem_v = v0.mean(dim=0, keepdim=True).detach()    # (1,M,C)
            self._mem_init = True

    @staticmethod
    def _pool_tokens_to_slots(x_bnc: torch.Tensor, m: int) -> torch.Tensor:
        """
        x_bnc: (B, T, C) → (B, m, C)
        토큰축(T)을 M으로 줄이기 위해 1D adaptive avg pooling 사용.
        """
        B, T, C = x_bnc.shape
        x = x_bnc.transpose(1, 2).contiguous()     # (B, C, T)
        x = F.adaptive_avg_pool1d(x, m)            # (B, C, m)
        x = x.transpose(1, 2).contiguous()         # (B, m, C)
        return x

    @torch.no_grad()
    def _ema_update(self, new_k_bnc: torch.Tensor, new_v_bnc: torch.Tensor):
        """
        new_k_bnc/new_v_bnc: (B, T, C)  (T=h2*w2)
        → adaptive pooling으로 (B, M, C)로 축약 후 배치 평균으로 EMA 업데이트.
        """
        pooled_k = self._pool_tokens_to_slots(new_k_bnc, self.mem_slots)  # (B,M,C)
        pooled_v = self._pool_tokens_to_slots(new_v_bnc, self.mem_slots)  # (B,M,C)

        k_bar = pooled_k.mean(dim=0, keepdim=True).detach()               # (1,M,C)
        v_bar = pooled_v.mean(dim=0, keepdim=True).detach()               # (1,M,C)

        self.mem_k.mul_(self.ema).add_((1.0 - self.ema) * k_bar)
        self.mem_v.mul_(self.ema).add_((1.0 - self.ema) * v_bar)

    def _bnc_to_bchw(self, x_bnc: torch.Tensor, x_size: Tuple[int, int]) -> torch.Tensor:
        B, N, C = x_bnc.shape
        H, W = x_size
        assert N == H * W, f"N({N}) != H*W({H*W})"
        return x_bnc.transpose(1, 2).contiguous().view(B, C, H, W)

    def _bchw_to_bnc(self, x_bchw: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x_bchw.shape
        return x_bchw.view(B, C, H * W).transpose(1, 2).contiguous()

    def forward(self, x_bnc: torch.Tensor, x_size: Tuple[int, int]) -> torch.Tensor:
        """
        x_bnc: (B, H*W, C)
        x_size: (H, W)
        """
        B, N, C = x_bnc.shape
        H, W = x_size

        self._init_memory_if_needed(x_bnc)

        # === Cross-scale features ===
        x_bchw = self._bnc_to_bchw(x_bnc, x_size)           # (B,C,H,W)
        x_dn_bchw = self.pool(x_bchw)                        # (B,C,H/s,W/s)
        x_dn_bnc  = self._bchw_to_bnc(x_dn_bchw)             # (B, T, C), T = (H/s)*(W/s)

        # === Q from high-res, K/V from downscaled+memory concat ===
        q = self.q_proj(self.norm_q(x_bnc))                  # (B,N,C)
        k_src = self.norm_kv(x_dn_bnc)                       # (B,T,C)
        v_src = k_src

        k_new = self.k_proj(k_src)                           # (B,T,C)
        v_new = self.v_proj(v_src)                           # (B,T,C)

        # 메모리 EMA 업데이트 (T -> M으로 축약)
        self._ema_update(k_new, v_new)                       # updates (1,M,C)

        # 메모리와 concat
        k_cat = torch.cat([k_new, self.mem_k.repeat(B, 1, 1)], dim=1)  # (B, T+M, C)
        v_cat = torch.cat([v_new, self.mem_v.repeat(B, 1, 1)], dim=1)  # (B, T+M, C)

        # === Multi-head attention ===
        def split_heads(x):  # (B,L,C) -> (B,h,L,d)
            return x.view(B, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3).contiguous()

        qh = split_heads(q)         # (B,h,N,d)
        kh = split_heads(k_cat)     # (B,h,T+M,d)
        vh = split_heads(v_cat)     # (B,h,T+M,d)

        attn = (qh * self.scale) @ kh.transpose(-2, -1)      # (B,h,N,T+M)
        attn = attn.softmax(dim=-1)
        out = (attn @ vh).transpose(1, 2).reshape(B, N, C)   # (B,N,C)

        out = self.proj(out)
        out = out + self.mlp(self.norm2(out))
        return out

# =========================
# Blocks 컨테이너 (OCAB → CSA+Mem)
# =========================
class AttenBlocks(nn.Module):
    def __init__(self,
                 dim,
                 input_resolution,
                 depth,
                 num_heads,
                 window_size,
                 compress_ratio,
                 squeeze_factor,
                 conv_scale,
                 overlap_ratio,  # 호환용 (미사용)
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 norm_layer=nn.LayerNorm,
                 downsample=None,
                 use_checkpoint=False,
                 esc_pdim: int = 16,
                 esc_kernel: int = 13,
                 # CSA+Mem 옵션
                 csa_heads: int = 6,
                 csa_down_ratio: int = 2,
                 csa_mem_size: int = 32,
                 fft_bands: int = 3,
                 fft_res_scale: float = 0.15):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        self.blocks = nn.ModuleList([
            HAB(
                dim=dim,
                input_resolution=input_resolution,
                num_heads=num_heads,
                window_size=window_size,
                shift_size=0 if (i % 2 == 0) else window_size // 2,
                compress_ratio=compress_ratio,
                squeeze_factor=squeeze_factor,
                conv_scale=conv_scale,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop,
                attn_drop=attn_drop,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer,
                esc_pdim=esc_pdim,
                esc_kernel=esc_kernel,
                fft_bands=fft_bands,
                fft_res_scale=fft_res_scale
            ) for i in range(depth)
        ])

        # Cross-Scale Attention + Memory (OCAB 대체)
        self.cross_scale = CrossScaleAttentionMem(
            dim=dim,
            num_heads=csa_heads,
            down_scale=csa_down_ratio,   # 이름 정합
            mem_slots=csa_mem_size,      # 이름 정합
            mlp_ratio=mlp_ratio,
            norm_layer=norm_layer
        )

        self.downsample = downsample(input_resolution, dim=dim, norm_layer=norm_layer) if downsample is not None else None

    def forward(self, x, x_size, params):
        for blk in self.blocks:
            x = blk(x, x_size, params=params)
        x = self.cross_scale(x, x_size)
        if self.downsample is not None:
            x = self.downsample(x)
        return x

class PatchMerging(nn.Module):
    def __init__(self, input_resolution, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)
    def forward(self, x):
        h, w = self.input_resolution
        b, seq_len, c = x.shape
        assert seq_len == h * w, 'input feature has wrong size'
        assert h % 2 == 0 and w % 2 == 0
        x = x.view(b, h, w, c)
        x0 = x[:, 0::2, 0::2, :]
        x1 = x[:, 1::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, :]
        x3 = x[:, 1::2, 1::2, :]
        x = torch.cat([x0, x1, x2, x3], -1).view(b, -1, 4 * c)
        x = self.norm(x)
        x = self.reduction(x)
        return x

class RHAG(nn.Module):
    def __init__(self,
                 dim,
                 input_resolution,
                 depth,
                 num_heads,
                 window_size,
                 compress_ratio,
                 squeeze_factor,
                 conv_scale,
                 overlap_ratio,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 norm_layer=nn.LayerNorm,
                 downsample=None,
                 use_checkpoint=False,
                 img_size=224,
                 patch_size=4,
                 resi_connection='1conv',
                 esc_pdim: int = 16,
                 esc_kernel: int = 13,
                 # CSA+Mem 옵션
                 csa_heads: int = 6,
                 csa_down_ratio: int = 2,
                 csa_mem_size: int = 32,
                 fft_bands: int = 3,
                 fft_res_scale: float = 0.15):
        super().__init__()

        self.dim = dim
        self.input_resolution = input_resolution

        self.residual_group = AttenBlocks(
            dim=dim,
            input_resolution=input_resolution,
            depth=depth,
            num_heads=num_heads,
            window_size=window_size,
            compress_ratio=compress_ratio,
            squeeze_factor=squeeze_factor,
            conv_scale=conv_scale,
            overlap_ratio=overlap_ratio,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop=drop,
            attn_drop=attn_drop,
            drop_path=drop_path,
            norm_layer=norm_layer,
            downsample=downsample,
            use_checkpoint=use_checkpoint,
            esc_pdim=esc_pdim,
            esc_kernel=esc_kernel,
            csa_heads=csa_heads,
            csa_down_ratio=csa_down_ratio,
            csa_mem_size=csa_mem_size,
            fft_bands=fft_bands,
            fft_res_scale=fft_res_scale
        )

        if resi_connection == '1conv':
            self.conv = nn.Conv2d(dim, dim, 3, 1, 1)
        elif resi_connection == 'identity':
            self.conv = nn.Identity()
        else:
            raise ValueError(f"Unknown resi_connection: {resi_connection}")

        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=0, embed_dim=dim, norm_layer=None)
        self.patch_unembed = PatchUnEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=0, embed_dim=dim, norm_layer=None)

    def forward(self, x, x_size, params):
        return self.patch_embed(self.conv(self.patch_unembed(self.residual_group(x, x_size, params), x_size))) + x

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
        self.norm = norm_layer(embed_dim) if norm_layer is not None else None
    def forward(self, x):
        x = x.flatten(2).transpose(1, 2)
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
        x = x.transpose(1, 2).contiguous().view(x.shape[0], self.embed_dim, x_size[0], x_size[1])
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
            raise ValueError(f'scale {scale} is not supported. Supported scales: 2^n and 3.')
        super().__init__(*m)

# =========================
# HAT with Iterative Self-Conditioning + CSA+Mem + FFTGated
# =========================
@ARCH_REGISTRY.register()
class HATX(nn.Module):
    r""" Hybrid Attention Transformer (ESC conv-attn + FFT-Gated + CSA+Mem). """
    def __init__(self,
                 img_size=64,
                 patch_size=1,
                 in_chans=3,
                 embed_dim=96,
                 depths=(6, 6, 6, 6),
                 num_heads=(6, 6, 6, 6),
                 window_size=7,
                 compress_ratio=3,
                 squeeze_factor=30,
                 conv_scale=0.01,
                 overlap_ratio=0.5,  # 남겨둠(호환)
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm,
                 ape=False,
                 patch_norm=True,
                 use_checkpoint=False,
                 upscale=2,
                 img_range=1.,
                 upsampler='',
                 resi_connection='1conv',
                 # HAB의 ESC
                 esc_pdim: int = 16,
                 esc_kernel: int = 13,
                 esc_use_dynamic: bool = True,  # (호환용, 내부에서 미사용)
                 # CSA+Mem 옵션
                 csa_heads: int = 6,
                 csa_down_ratio: int = 2,
                 csa_mem_size: int = 16,
                 # FFT 옵션
                 fft_bands: int = 3,
                 fft_res_scale: float = 0.15,
                 # 반복 정련
                 iterative_T: int = 1,
                 **kwargs):
        super().__init__()

        if not _HAS_ESC:
            raise RuntimeError("[HAT][ESC] ESC가 필수인데 초기화 시점에 존재하지 않습니다.")

        self.window_size = window_size
        self.shift_size = window_size // 2
        self.overlap_ratio = overlap_ratio

        num_in_ch = in_chans
        num_out_ch = in_chans
        num_feat = 64
        self.img_range = img_range
        if in_chans == 3:
            rgb_mean = (0.4488, 0.4371, 0.4040)
            self.mean = torch.Tensor(rgb_mean).view(1, 3, 1, 1)
        else:
            self.mean = torch.zeros(1, 1, 1, 1)
        self.upscale = upscale
        self.upsampler = upsampler
        self.iterative_T = max(1, int(iterative_T))

        # relative position index (SA용, OCA는 제거됨)
        relative_position_index_SA = self.calculate_rpi_sa()
        self.register_buffer('relative_position_index_SA', relative_position_index_SA)

        # shallow
        self.conv_first = nn.Conv2d(num_in_ch, embed_dim, 3, 1, 1)

        # deep
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.num_features = embed_dim
        self.mlp_ratio = mlp_ratio

        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=embed_dim,
            embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution

        self.patch_unembed = PatchUnEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=embed_dim,
            embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)

        if self.ape:
            num_patches = self.patch_embed.num_patches
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
            trunc_normal_(self.absolute_pos_embed, std=.02)

        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        # RHAGs
        self.layers = nn.ModuleList()
        start = 0
        for i_layer in range(self.num_layers):
            layer = RHAG(
                dim=embed_dim,
                input_resolution=(patches_resolution[0], patches_resolution[1]),
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=window_size,
                compress_ratio=compress_ratio,
                squeeze_factor=squeeze_factor,
                conv_scale=conv_scale,
                overlap_ratio=overlap_ratio,
                mlp_ratio=self.mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[start:start + depths[i_layer]],
                norm_layer=norm_layer,
                downsample=None,
                use_checkpoint=use_checkpoint,
                img_size=img_size,
                patch_size=patch_size,
                resi_connection=resi_connection,
                esc_pdim=esc_pdim,
                esc_kernel=esc_kernel,
                csa_heads=csa_heads,
                csa_down_ratio=csa_down_ratio,
                csa_mem_size=csa_mem_size,
                fft_bands=fft_bands,
                fft_res_scale=fft_res_scale
            )
            start += depths[i_layer]
            self.layers.append(layer)
        self.norm = norm_layer(self.num_features)

        # reconstruction
        if resi_connection == '1conv':
            self.conv_after_body = nn.Conv2d(embed_dim, embed_dim, 3, 1, 1)
        elif resi_connection == 'identity':
            self.conv_after_body = nn.Identity()
        else:
            raise ValueError(f"Unknown resi_connection: {resi_connection}")

        if self.upsampler == 'pixelshuffle':
            self.conv_before_upsample = nn.Sequential(
                nn.Conv2d(embed_dim, num_feat, 3, 1, 1), nn.LeakyReLU(inplace=True))
            self.upsample = Upsample(upscale, num_feat)
            self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)

        # ---- 반복 정련용 보조조건 투입
        self.cond_proj = nn.Conv2d(2, embed_dim, 1)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def calculate_rpi_sa(self):
        coords_h = torch.arange(self.window_size)
        coords_w = torch.arange(self.window_size)
        coords = torch.stack(torch.meshgrid(coords_h, coords_w, indexing='ij'))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += self.window_size - 1
        relative_coords[:, :, 1] += self.window_size - 1
        relative_coords[:, :, 0] *= 2 * self.window_size - 1
        relative_position_index = relative_coords.sum(-1)
        return relative_position_index

    def _build_attn_mask_and_soft_tokens(self, x_size):
        """ Swin-style attn_mask + soft token weights (H*W,) """
        h, w = x_size
        img_mask = torch.zeros((1, h, w, 1))
        shift_size = self.window_size // 2
        h_slices = (slice(0, -self.window_size), slice(-self.window_size, -shift_size), slice(-shift_size, None))
        w_slices = (slice(0, -self.window_size), slice(-self.window_size, -shift_size), slice(-shift_size, None))
        cnt = 0
        for hs in h_slices:
            for ws in w_slices:
                img_mask[:, hs, ws, :] = cnt
                cnt += 1
        mask_windows = window_partition(img_mask, self.window_size).view(-1, self.window_size * self.window_size)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))

        # soft mask per token: 같은 윈도우=1, 다른 윈도우=0 → 평균값 (num_win,Nw)
        mask_soft = (attn_mask == 0).float().mean(dim=-1)  # (num_win, Nw)

        # (num_win, Nw) → (1,H,W,1)로 역복원 → (H*W,)
        soft_win = mask_soft.view(-1, self.window_size, self.window_size, 1)
        soft_map = window_reverse(soft_win, self.window_size, h, w)  # (1,H,W,1)
        soft_tokens = soft_map.view(-1)  # (H*W,)
        return attn_mask, soft_tokens

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}

    def forward_features_once(self, x):
        """ x: (B,C,H,W) -> (B,C,H,W) 한 번 통과 """
        x_size = (x.shape[2], x.shape[3])

        attn_mask, mask_soft_tokens = self._build_attn_mask_and_soft_tokens(x_size)
        params = {
            'attn_mask': attn_mask.to(x.device),
            'rpi_sa': self.relative_position_index_SA,  # (OCAB 제거, SA만 유지)
            'mask_soft_tokens': mask_soft_tokens.to(x.device)
        }

        tokens = self.patch_embed(x)
        if hasattr(self, "absolute_pos_embed"):
            tokens = tokens + self.absolute_pos_embed
        tokens = self.pos_drop(tokens)

        for layer in self.layers:
            tokens = layer(tokens, x_size, params)

        tokens = self.norm(tokens)
        feat = self.patch_unembed(tokens, x_size)  # (B,C,H,W)
        feat = self.conv_after_body(feat) + x
        return feat

    def _cond_maps(self, x_feat):
        """ 저주파 + 에지 맵 2채널 """
        low = F.avg_pool2d(x_feat, 3, stride=1, padding=1)
        kx = torch.tensor([[1,0,-1],[2,0,-2],[1,0,-1]], device=x_feat.device, dtype=x_feat.dtype).view(1,1,3,3)
        ky = torch.tensor([[1,2,1],[0,0,0],[-1,-2,-1]], device=x_feat.device, dtype=x_feat.dtype).view(1,1,3,3)
        edge_x = F.conv2d(x_feat.mean(1, keepdim=True), kx, padding=1)
        edge_y = F.conv2d(x_feat.mean(1, keepdim=True), ky, padding=1)
        edge = torch.sqrt(edge_x**2 + edge_y**2 + 1e-6)
        cond = torch.cat([low.mean(1, keepdim=True), edge], dim=1)  # (B,2,H,W)
        return cond

    def forward_features_iterative(self, x):
        """ 반복 정련 (iterative_T>1) """
        feat = x
        for t in range(self.iterative_T):
            cond = self._cond_maps(feat)
            feat = feat + self.cond_proj(cond) * 0.1
            feat = self.forward_features_once(feat)
        return feat

    def forward(self, x):
        self.mean = self.mean.type_as(x)
        x_in = (x - self.mean) * self.img_range

        if self.upsampler == 'pixelshuffle':
            feat0 = self.conv_first(x_in)
            feat = self.forward_features_iterative(feat0)
            out = self.conv_before_upsample(feat)
            out = self.upsample(out)
            out = self.conv_last(out)
        else:
            feat0 = self.conv_first(x_in)
            out = self.forward_features_iterative(feat0)

        out = out / self.img_range + self.mean
        return out
