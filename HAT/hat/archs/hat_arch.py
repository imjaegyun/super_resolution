# basicsr/archs/hat_arch.py
import math
import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint

from basicsr.utils.registry import ARCH_REGISTRY
from basicsr.archs.arch_util import to_2tuple, trunc_normal_
from einops import rearrange

# =========================
# ESC 원본 사용 (esc_arch.py)
# =========================
import os, sys
try:
    from .esc_arch import ConvolutionalAttention, ConvAttnWrapper  # 원본 ESC 모듈
    _HAS_ESC = True
except Exception as _e:
    _HAS_ESC = False
    raise ImportError(
        "[HAT][ESC][FATAL] esc_arch.py 를 불러오지 못했습니다.\n"
        " - 파일 경로: basicsr/archs/esc_arch.py\n"
        f" - 원인: {repr(_e)}\n"
        "ESC 연동이 필수이니 경로/의존성을 확인하세요."
    )

# ==== DEBUG UTILS ====
import torch.distributed as dist
HAT_DEBUG = os.environ.get("HAT_DEBUG", "0") == "1"

def _is_master():
    return (not dist.is_available()) or (not dist.is_initialized()) or dist.get_rank() == 0

def _dprint(*args, **kwargs):
    if HAT_DEBUG and _is_master():
        print(*args, **kwargs)

_DEBUG_ONCE = {
    "hab_shift": False,
    "hab_masksoft": False,
    "ocab_once": False,
    "ocab_esc_once": False,
}

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
# Channel Attention (ECA)
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
# Gated Depthwise-Conv FFN (GLU 스타일) — HAB에서만 사용, OCAB는 MLP 유지
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
# HAB: ESC 기반 + 게이티드 FFN
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
                 esc_kernel: int = 13):
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

        self.conv_block = CAB(num_feat=dim, compress_ratio=compress_ratio, squeeze_factor=squeeze_factor)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = GatedDconvFFN(dim=dim, mlp_ratio=mlp_ratio, drop=drop, dw_kernel_size=3, act_layer="silu")

    def forward(self, x, x_size, rpi_sa_unused=None, attn_mask_unused=None):
        h, w = x_size
        b, _, c = x.shape

        shortcut = x
        x = self.norm1(x)
        x_img = x.view(b, h, w, c)

        conv_x = self.conv_block(x_img.permute(0, 3, 1, 2))
        conv_x = conv_x.permute(0, 2, 3, 1).contiguous().view(b, h * w, c)

        attn_x = self.esc_attn(x, (h, w))  # (B,N,C)

        if HAT_DEBUG and _is_master():
            if self.shift_size > 0 and not _DEBUG_ONCE["hab_shift"]:
                _DEBUG_ONCE["hab_shift"] = True
                _dprint(f"[HAB][shift] shift_size={self.shift_size}, window_size={self.window_size}, "
                        f"x_size=({h},{w}), tokens={b*h*w}, dim={c}")

        x = shortcut + self.drop_path(attn_x) + conv_x * self.conv_scale
        x = x + self.drop_path(self.mlp(self.norm2(x), (h, w)))
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

# =========================
# OCAB (ESC-Infused, MLP 유지)
#  - esc_enable=False면 원본과 동일 동작
#  - esc_enable=True면: Q는 x에서, K/V는 ESC로 보강된 y에서 생성
# =========================
class OCAB(nn.Module):
    def __init__(self, dim,
                 input_resolution,
                 window_size,
                 overlap_ratio,
                 num_heads,
                 qkv_bias=True,
                 qk_scale=None,
                 mlp_ratio=2,
                 norm_layer=nn.LayerNorm,
                 # ESC 주입 제어
                 esc_enable: bool = False,
                 esc_pdim: int = 16,
                 esc_kernel: int = 13):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.overlap_win_size = int(window_size * overlap_ratio) + window_size

        self.norm1 = norm_layer(dim)
        # Q/KV를 각각 다른 입력에 적용하기 위해 분리 투영
        self.q_proj  = nn.Linear(dim, dim,  bias=qkv_bias)
        self.kv_proj = nn.Linear(dim, 2*dim, bias=qkv_bias)

        self.unfold = nn.Unfold(kernel_size=(self.overlap_win_size, self.overlap_win_size),
                                stride=window_size, padding=(self.overlap_win_size - window_size) // 2)

        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((window_size + self.overlap_win_size - 1) * (window_size + self.overlap_win_size - 1), num_heads))
        trunc_normal_(self.relative_position_bias_table, std=.02)

        self.softmax = nn.Softmax(dim=-1)
        self.proj = nn.Linear(dim, dim)

        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        # === 요청: OCAB는 MLP 그대로 유지 ===
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Linear(mlp_hidden_dim, dim)
        )

        # === ESC 주입 경로 ===
        self.esc_enable = bool(esc_enable)
        if self.esc_enable:
            if not _HAS_ESC:
                raise RuntimeError("[OCAB][ESC] esc_arch 모듈이 필요합니다.")
            self.esc_core = ConvAttnWrapper(dim=dim, pdim=esc_pdim, kernel_size=esc_kernel)
            self.esc_plk  = nn.Parameter(torch.randn(esc_pdim, esc_pdim, esc_kernel, esc_kernel))
            torch.nn.init.orthogonal_(self.esc_plk)
            self._esc_pdim = esc_pdim
            self._esc_kernel = esc_kernel

    def forward(self, x, x_size, rpi):
        h, w = x_size
        b, _, c = x.shape

        shortcut = x
        x = self.norm1(x)
        x_img = x.view(b, h, w, c)                      # (B,H,W,C)
        x_bchw = x_img.permute(0, 3, 1, 2).contiguous() # (B,C,H,W)

        # === ESC 주입: K/V용 특징 보강 ===
        if self.esc_enable:
            y_bchw = self.esc_core(x_bchw, self.esc_plk)        # (B,C,H,W)  PLK+동적DWConv
            y_img  = y_bchw.permute(0, 2, 3, 1).contiguous()    # (B,H,W,C)
            if HAT_DEBUG and _is_master() and (not _DEBUG_ONCE["ocab_esc_once"]):
                _DEBUG_ONCE["ocab_esc_once"] = True
                _dprint(f"[OCAB-ESC] enable=True, pdim={self._esc_pdim}, kernel={self._esc_kernel}, "
                        f"x/y shape={(tuple(x_img.shape), tuple(y_img.shape))}")
        else:
            y_img = x_img  # ESC 비활성 시 원본과 동일

        # --- Q from x_img ---
        q = self.q_proj(x_img)  # (B,H,W,C)

        # --- KV from y_img (ESC 보강 특징) ---
        kv = self.kv_proj(y_img)  # (B,H,W,2C)
        k, v = kv.split(c, dim=-1)

        # === 윈도우 분할 (Q) ===
        q_windows = window_partition(q, self.window_size) \
            .view(-1, self.window_size * self.window_size, c)  # (B*nW, Nq, C)

        # === Overlap 윈도우 추출 (KV) ===
        # (B,H,W,C) -> (B,C,H,W) -> unfold -> (B, 2C*Ow*Ow, nW)
        kv_cat_bchw = torch.cat([k, v], dim=-1).permute(0, 3, 1, 2).contiguous()
        kv_windows = self.unfold(kv_cat_bchw)  # (B, 2C*Ow*Ow, nW)
        kv_windows = rearrange(
            kv_windows, 'b (nc ch owh oww) nw -> nc (b nw) (owh oww) ch',
            nc=2, ch=c, owh=self.overlap_win_size, oww=self.overlap_win_size).contiguous()
        k_windows, v_windows = kv_windows[0], kv_windows[1]  # ((B*nW), Nk, C)

        # === 멀티헤드로 reshape ===
        b_, nq, _ = q_windows.shape
        _, n, _ = k_windows.shape
        d = self.dim // self.num_heads
        qh = q_windows.reshape(b_, nq, self.num_heads, d).permute(0, 2, 1, 3)
        kh = k_windows.reshape(b_, n, self.num_heads, d).permute(0, 2, 1, 3)
        vh = v_windows.reshape(b_, n, self.num_heads, d).permute(0, 2, 1, 3)

        # === 어텐션 ===
        qh = qh * self.scale
        attn = (qh @ kh.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[rpi.view(-1)].view(
            self.window_size * self.window_size,
            self.overlap_win_size * self.overlap_win_size,
            -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn = self.softmax(attn + relative_position_bias.unsqueeze(0))
        attn_windows = (attn @ vh).transpose(1, 2).reshape(b_, nq, self.dim)

        # === 윈도우 역복원 ===
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, self.dim)
        x = window_reverse(attn_windows, self.window_size, h, w).view(b, h * w, self.dim)

        # === 출력 ===
        x = self.proj(x) + shortcut
        x = x + self.mlp(self.norm2(x))
        return x

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
                 esc_pdim: int = 16,
                 esc_kernel: int = 13,
                 # OCAB ESC 주입 파라미터
                 ocab_esc_enable: bool = False,
                 ocab_esc_pdim: int = 16,
                 ocab_esc_kernel: int = 13):
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
                esc_kernel=esc_kernel
            ) for i in range(depth)
        ])

        self.overlap_attn = OCAB(
            dim=dim,
            input_resolution=input_resolution,
            window_size=window_size,
            overlap_ratio=overlap_ratio,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            mlp_ratio=mlp_ratio,
            norm_layer=norm_layer,
            esc_enable=ocab_esc_enable,
            esc_pdim=ocab_esc_pdim,
            esc_kernel=ocab_esc_kernel
        )

        self.downsample = downsample(input_resolution, dim=dim, norm_layer=norm_layer) if downsample is not None else None

    def forward(self, x, x_size, params):
        for blk in self.blocks:
            x = blk(x, x_size, params.get('rpi_sa', None), params.get('attn_mask', None))

        x = self.overlap_attn(x, x_size, params['rpi_oca'])

        if HAT_DEBUG and _is_master() and (not _DEBUG_ONCE["ocab_once"]):
            _DEBUG_ONCE["ocab_once"] = True
            rpi = params.get('rpi_oca', None)
            _dprint(f"[OCAB] window={self.overlap_attn.window_size}, "
                    f"overlap_win={self.overlap_attn.overlap_win_size}, "
                    f"heads={self.overlap_attn.num_heads}, "
                    f"rpi_oca shape={tuple(rpi.shape) if rpi is not None else None}")

        if self.downsample is not None:
            x = self.downsample(x)
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
                 ocab_esc_enable: bool = False,
                 ocab_esc_pdim: int = 16,
                 ocab_esc_kernel: int = 13):
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
            ocab_esc_enable=ocab_esc_enable,
            ocab_esc_pdim=ocab_esc_pdim,
            ocab_esc_kernel=ocab_esc_kernel
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

@ARCH_REGISTRY.register()
class HAT(nn.Module):
    r""" Hybrid Attention Transformer (ESC conv-attn integrated in HAB + optional OCAB). """
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
                 overlap_ratio=0.5,
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
                 esc_use_dynamic: bool = True,  # 호환용(원본 ESC는 내부에서 자동)
                 # OCAB의 ESC 주입
                 ocab_esc_enable: bool = False,
                 ocab_esc_pdim: int = 16,
                 ocab_esc_kernel: int = 13,
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

        # relative position index
        relative_position_index_SA = self.calculate_rpi_sa()
        relative_position_index_OCA = self.calculate_rpi_oca()
        self.register_buffer('relative_position_index_SA', relative_position_index_SA)
        self.register_buffer('relative_position_index_OCA', relative_position_index_OCA)

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
                ocab_esc_enable=ocab_esc_enable,
                ocab_esc_pdim=ocab_esc_pdim,
                ocab_esc_kernel=ocab_esc_kernel
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

    def calculate_rpi_oca(self):
        window_size_ori = self.window_size
        window_size_ext = self.window_size + int(self.overlap_ratio * self.window_size)

        coords_h = torch.arange(window_size_ori)
        coords_w = torch.arange(window_size_ori)
        coords_ori = torch.stack(torch.meshgrid(coords_h, coords_w, indexing='ij'))
        coords_ori_flatten = torch.flatten(coords_ori, 1)

        coords_h = torch.arange(window_size_ext)
        coords_w = torch.arange(window_size_ext)
        coords_ext = torch.stack(torch.meshgrid(coords_h, coords_w, indexing='ij'))
        coords_ext_flatten = torch.flatten(coords_ext, 1)

        relative_coords = coords_ext_flatten[:, None, :] - coords_ori_flatten[:, :, None]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += window_size_ori - window_size_ext + 1
        relative_coords[:, :, 1] += window_size_ori - window_size_ext + 1
        relative_coords[:, :, 0] *= window_size_ori + window_size_ext - 1
        relative_position_index = relative_coords.sum(-1)
        return relative_position_index

    def calculate_mask(self, x_size):
        h, w = x_size
        img_mask = torch.zeros((1, h, w, 1))
        h_slices = (slice(0, -self.window_size), slice(-self.window_size, -self.shift_size), slice(-self.shift_size, None))
        w_slices = (slice(0, -self.window_size), slice(-self.window_size, -self.shift_size), slice(-self.shift_size, None))
        cnt = 0
        for hs in h_slices:
            for ws in w_slices:
                img_mask[:, hs, ws, :] = cnt
                cnt += 1
        mask_windows = window_partition(img_mask, self.window_size).view(-1, self.window_size * self.window_size)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        return attn_mask

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}

    def forward_features(self, x):
        x_size = (x.shape[2], x.shape[3])

        attn_mask = self.calculate_mask(x_size).to(x.device)
        params = {'attn_mask': attn_mask,
                  'rpi_sa': self.relative_position_index_SA,
                  'rpi_oca': self.relative_position_index_OCA}

        x = self.patch_embed(x)
        if hasattr(self, "absolute_pos_embed"):
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)

        for layer in self.layers:
            x = layer(x, x_size, params)

        x = self.norm(x)
        x = self.patch_unembed(x, x_size)
        return x

    def forward(self, x):
        self.mean = self.mean.type_as(x)
        x = (x - self.mean) * self.img_range

        if self.upsampler == 'pixelshuffle':
            x = self.conv_first(x)
            x = self.conv_after_body(self.forward_features(x)) + x
            x = self.conv_before_upsample(x)
            x = self.conv_last(self.upsample(x))

        x = x / self.img_range + self.mean
        return x
