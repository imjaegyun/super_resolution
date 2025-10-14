# hat/archs/esc_adapter.py
import torch
import torch.nn as nn
import torch.nn.functional as F

# 같은 폴더의 esc_arch.py 에서 ConvAttnWrapper를 import
from .esc_arch import ConvAttnWrapper

class ESCConvAttnWrapper(nn.Module):
    """
    HAT의 HAB에서 사용하는 (B,N,C) 어텐션 대체 모듈.
    - 내부에 ESC의 ConvAttnWrapper(+plk_filter 파라미터)만 씀
    - 입력: x_tokens (B,N,C), x_size=(H,W)
    - 출력: (B,N,C)
    """
    def __init__(self, dim: int, pdim: int = 16, kernel_size: int = 13):
        super().__init__()
        self.dim = dim
        self.pdim = pdim
        self.kernel_size = kernel_size

        # ESC에서 쓰는 큰 커널 필터(learnable)
        self.plk_filter = nn.Parameter(torch.randn(pdim, pdim, kernel_size, kernel_size))
        nn.init.orthogonal_(self.plk_filter)

        # ESC의 conv-attn 블록 래퍼 (입력/출력: (B,C,H,W))
        self.conv_attn = ConvAttnWrapper(dim=dim, pdim=pdim, kernel_size=kernel_size)

    def forward(self, x_tokens: torch.Tensor, x_hw):
        """
        x_tokens: (B, N, C), x_hw=(H,W)
        """
        B, N, C = x_tokens.shape
        H, W = x_hw
        assert N == H * W, "ESCConvAttnWrapper: N must be H*W"

        x = x_tokens.transpose(1, 2).contiguous().view(B, C, H, W)  # (B,C,H,W)
        y = self.conv_attn(x, self.plk_filter)                      # (B,C,H,W)
        y = y.view(B, C, H, W).flatten(2).transpose(1, 2).contiguous()  # (B,N,C)
        return y
