import torch 
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange 

from torch.nn.init import trunc_normal_

from torch.nn.attention.flex_attention import flex_attention
from typing import Optional

from models import register


def attention(q, k, v, bias):
    score = q @ k.transpose(-2, -1) / q.shape[-1]**0.5
    score = score + bias
    score = F.softmax(score, dim=-1)
    out = score @ v
    return out


def apply_rpe(table: torch.Tensor, window_size: int):
    def bias_mod(score, b, h, q_idx, kv_idx):
        q_h = q_idx // window_size
        q_w = q_idx % window_size
        k_h = kv_idx // window_size
        k_w = kv_idx % window_size
        rel_h = k_h - q_h + window_size - 1
        rel_w = k_w - q_w + window_size - 1
        rel_idx = rel_h * (2 * window_size - 1) + rel_w
        return score + table[h, rel_idx]
    return bias_mod


def feat_to_win(x, window_size, heads):
    return rearrange(
        x, 'b (qkv heads c) (h wh) (w ww) -> qkv (b h w) heads (wh ww)  c', heads=heads, wh=window_size[0], ww=window_size[1], qkv=3
    )
def win_to_feat(x, window_size, h_div, w_div):
    return rearrange(
        x, '(b h w) heads (wh ww) c -> b (heads c) (h wh) (w ww)', h=h_div, w=w_div, wh=window_size[0], ww=window_size[1]
    )


class LayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_first"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape, )

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            if self.training:
                return F.layer_norm(x.permute(0, 2, 3, 1).contiguous(), self.normalized_shape, self.weight, self.bias, self.eps).permute(0, 3, 1, 2).contiguous()
            else:
                return F.layer_norm(x.permute(0, 2, 3, 1), self.normalized_shape, self.weight, self.bias, self.eps).permute(0, 3, 1, 2)


class ConvolutionalAttention(nn.Module):
    def __init__(self, pdim: int, proj_dim_in: Optional[int] = None):
        super().__init__()
        self.pdim = pdim
        self.proj_dim_in = proj_dim_in if proj_dim_in is not None else pdim
        self.sk_size = 3
        self.dwc_proj = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(self.proj_dim_in, pdim // 2, 1, 1, 0),
            nn.GELU(),
            nn.Conv2d(pdim // 2, pdim * self.sk_size * self.sk_size, 1, 1, 0)
        )
        nn.init.zeros_(self.dwc_proj[-1].weight)
        nn.init.zeros_(self.dwc_proj[-1].bias)

    def forward(self, x: torch.Tensor, lk_filter: torch.Tensor) -> torch.Tensor:
        if self.training:
            x1, x2 = torch.split(x, [self.pdim, x.shape[1]-self.pdim], dim=1)
            
            # Dynamic Conv
            bs = x1.shape[0]
            dynamic_kernel = self.dwc_proj(x[:, :self.proj_dim_in]).reshape(-1, 1, self.sk_size, self.sk_size)
            x1_ = rearrange(x1, 'b c h w -> 1 (b c) h w')
            x1_ = F.conv2d(x1_, dynamic_kernel, stride=1, padding=self.sk_size//2, groups=bs * self.pdim)
            x1_ = rearrange(x1_, '1 (b c) h w -> b c h w', b=bs, c=self.pdim)
            
            # Static LK Conv + Dynamic Conv
            x1 = F.conv2d(x1, lk_filter, stride=1, padding=lk_filter.shape[-1] // 2) + x1_
            
            x = torch.cat([x1, x2], dim=1)
        else:
            dynamic_kernel = self.dwc_proj(x[:, :self.proj_dim_in]).reshape(-1, 1, self.sk_size, self.sk_size)
            x[:, :self.pdim] = F.conv2d(x[:, :self.pdim], lk_filter, stride=1, padding=lk_filter.shape[-1] // 2) + \
                rearrange(
                    F.conv2d(rearrange(x[:, :self.pdim], 'b c h w -> 1 (b c) h w'), dynamic_kernel, stride=1, padding=self.sk_size//2, groups=x.shape[0] * self.pdim),
                    '1 (b c) h w -> b c h w', b=x.shape[0]
                )
        return x
    
    def extra_repr(self):
        return f'pdim={self.pdim}, proj_dim_in={self.proj_dim_in}'
    

class ConvAttnWrapper(nn.Module):
    def __init__(self, dim: int, pdim: int, proj_dim_in: Optional[int] = None):
        super().__init__()
        self.plk = ConvolutionalAttention(pdim, proj_dim_in)
        self.aggr = nn.Conv2d(dim, dim, 1, 1, 0)

    def forward(self, x: torch.Tensor, lk_filter: torch.Tensor) -> torch.Tensor:
        x = self.plk(x, lk_filter)
        x = self.aggr(x)
        return x 


class ConvFFN(nn.Module):
    def __init__(self, dim: int, kernel_size: int, exp_ratio: int):
        super().__init__()
        self.proj = nn.Conv2d(dim, int(dim*exp_ratio), 1, 1, 0)
        self.dwc = nn.Conv2d(int(dim*exp_ratio), int(dim*exp_ratio), kernel_size, 1, kernel_size//2, groups=int(dim*exp_ratio))
        self.aggr = nn.Conv2d(int(dim*exp_ratio), dim, 1, 1, 0)

    def forward(self, x):
        x = F.gelu(self.proj(x))
        x = F.gelu(self.dwc(x)) + x
        x = self.aggr(x)
        return x


class WindowAttention(nn.Module):
    def __init__(self, dim: int, window_size: int, num_heads: int, attn_func=None, deployment=False):
        super().__init__()
        self.dim = dim
        window_size = (window_size, window_size) if isinstance(window_size, int) else window_size
        self.window_size = window_size
        self.num_heads = num_heads
        self.to_qkv = nn.Conv2d(dim, dim*3, 1, 1, 0)
        self.to_out = nn.Conv2d(dim, dim, 1, 1, 0)
        
        self.attn_func = attn_func
        self.is_deployment = deployment
        self.relative_position_bias = nn.Parameter(
            torch.randn(num_heads, (2*window_size[0]-1)*(2*window_size[1]-1)).to(torch.float32) * 0.001
        )
        if self.is_deployment:
            self.relative_position_bias = self.relative_position_bias.requires_grad_(False)
            self.get_rpe = apply_rpe(self.relative_position_bias, window_size[0])
        else:
            self.rpe_idxs = self.create_table_idxs(window_size[0], num_heads)

    @staticmethod
    def create_table_idxs(window_size: int, heads: int):
        idxs_window = []
        for head in range(heads):
            for h in range(window_size**2):
                for w in range(window_size**2):
                    q_h = h // window_size
                    q_w = h % window_size
                    k_h = w // window_size
                    k_w = w % window_size
                    rel_h = k_h - q_h + window_size - 1
                    rel_w = k_w - q_w + window_size - 1
                    rel_idx = rel_h * (2 * window_size - 1) + rel_w
                    idxs_window.append((head, rel_idx))
        idxs = torch.tensor(idxs_window, dtype=torch.long, requires_grad=False)
        return idxs
        
    def pad_to_win(self, x, h, w):
        pad_h = (self.window_size[0] - h % self.window_size[0]) % self.window_size[0]
        pad_w = (self.window_size[1] - w % self.window_size[1]) % self.window_size[1]
        x = F.pad(x, (0, pad_w, 0, pad_h), mode='reflect')
        return x
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: input features with shape of (B, C, H, W)
        """
        _, _, h, w = x.shape
        x = self.pad_to_win(x, h, w)
        h_div, w_div = x.shape[2] // self.window_size[0], x.shape[3] // self.window_size[1]
        
        qkv = self.to_qkv(x)
        dtype = qkv.dtype
        qkv = feat_to_win(qkv, self.window_size, self.num_heads)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        if self.is_deployment:
            q = q.contiguous()
            k = k.contiguous()
            v = v.contiguous()
            out = self.attn_func(q, k, v, score_mod=self.get_rpe)
        else:
            bias = self.relative_position_bias[self.rpe_idxs[:, 0], self.rpe_idxs[:, 1]]
            bias = bias.reshape(1, self.num_heads, self.window_size[0]*self.window_size[1], self.window_size[0]*self.window_size[1])
            out = self.attn_func(q, k, v, bias)
        
        out = win_to_feat(out, self.window_size, h_div, w_div)
        out = self.to_out(out.to(dtype)[:, :, :h, :w])
        return out   

    def extra_repr(self):
        return f'dim={self.dim}, window_size={self.window_size}, num_heads={self.num_heads}'


class Block(nn.Module):
    def __init__(
            self, dim: int, pdim: int, conv_blocks: int, 
            window_size: int, num_heads: int, exp_ratio: int, 
            proj_dim_in: Optional[int] = None, attn_func=None, deployment=False
        ):
        super().__init__()
        self.ln_proj = LayerNorm(dim)
        self.proj = ConvFFN(dim, 3, 2)

        self.ln_attn = LayerNorm(dim) 
        self.attn = WindowAttention(dim, window_size, num_heads, attn_func, deployment)
        
        self.pconvs = nn.ModuleList([ConvAttnWrapper(dim, pdim, proj_dim_in) for _ in range(conv_blocks)])
        self.convffns = nn.ModuleList([ConvFFN(dim, 3, exp_ratio) for _ in range(conv_blocks)])
        
        self.ln_out = LayerNorm(dim)
        self.conv_out = nn.Conv2d(dim, dim, 3, 1, 1)

    def forward(self, x: torch.Tensor, plk_filter: torch.Tensor) -> torch.Tensor:
        skip = x
        x = self.ln_proj(x)
        x = self.proj(x)
        x = x + self.attn(self.ln_attn(x))
        for pconv, convffn in zip(self.pconvs, self.convffns):
            x = x + pconv(convffn(x), plk_filter)
        x = self.conv_out(self.ln_out(x))
        return x + skip


def _geo_ensemble(k):
    k_hflip = k.flip([3])
    k_vflip = k.flip([2])
    k_hvflip = k.flip([2, 3])
    k_rot90 = torch.rot90(k, -1, [2, 3])
    k_rot90_hflip = k_rot90.flip([3])
    k_rot90_vflip = k_rot90.flip([2])
    k_rot90_hvflip = k_rot90.flip([2, 3])
    k = (k + k_hflip + k_vflip + k_hvflip + k_rot90 + k_rot90_hflip + k_rot90_vflip + k_rot90_hvflip) / 8
    return k


# @ARCH_REGISTRY.register()
class Net(nn.Module):
    def __init__(
        self, dim: int = 64, pdim: int = 16, kernel_size: int = 13,
        n_blocks: int = 5, conv_blocks: int = 5, window_size: int = 32, num_heads: int = 4,
        upscaling_factor: int = 12345, exp_ratio: int = 1.25, proj_dim_in: Optional[int] = None,
        deployment=False, is_geo=False
    ):
        super().__init__()
        if deployment:
            attn_func = torch.compile(flex_attention, dynamic=True)
        else:
            attn_func = attention
            
        self.plk_func = _geo_ensemble
            
        self.plk_filter = nn.Parameter(torch.randn(pdim, pdim, kernel_size, kernel_size))
        torch.nn.init.orthogonal_(self.plk_filter)
        
        self.proj = nn.Conv2d(3, dim, 3, 1, 1)
        self.blocks = nn.ModuleList([
            Block(
                dim, pdim, conv_blocks, window_size, 
                num_heads, exp_ratio, proj_dim_in, 
                attn_func, deployment
            ) for _ in range(n_blocks)
        ])
        self.last = nn.Conv2d(dim, dim, 3, 1, 1)
        # self.to_img = nn.Conv2d(dim, 3*upscaling_factor**2, 3, 1, 1)
        self.out_dim = dim
        self.window_size = window_size
        self.upscaling_factor = upscaling_factor
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.proj(x)
        skip = feat
        plk_filter = self.plk_func(self.plk_filter)
        for block in self.blocks:
            feat = block(feat, plk_filter)
        feat = self.last(feat) + skip
        # x = self.to_img(feat) + torch.repeat_interleave(x, self.upscaling_factor**2, dim=1)
        # x = F.pixel_shuffle(x, self.upscaling_factor)
        return feat


@register('esc')
def make_swinir(no_upsampling=True, deployment=False):
    return Net(deployment=deployment)
