# hat/archs/hybrid_hat_naf_arch.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from basicsr.utils.registry import ARCH_REGISTRY

# 이미 갖고 있는 HATX 재사용
from .hatx_arch import HATX

# ---- NAFNet-style 블록 (경량 구현) ----
class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2

class NAFBlock(nn.Module):
    """
    매우 간단화한 NAFBlock:
    - PWConv -> DWConv -> SimpleGate -> SCA -> PWConv
    - skip 연결 + 드롭경량화
    """
    def __init__(self, c, dw_expand=2, ffn_expand=2, drop_path=0.0):
        super().__init__()
        dwc = c * dw_expand
        ffnc = c * ffn_expand

        self.pw1 = nn.Conv2d(c, dwc, 1, 1, 0)
        self.dw  = nn.Conv2d(dwc, dwc, 3, 1, 1, groups=dwc)
        self.sg  = SimpleGate()

        # SCA (scale-channel attention)
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dwc // 2, dwc // 2, 1, 1, 0),
        )

        self.pw2 = nn.Conv2d(dwc // 2, c, 1, 1, 0)

        # FFN 분기
        self.ffn1 = nn.Conv2d(c, ffnc, 1, 1, 0)
        self.ffn_dw = nn.Conv2d(ffnc, ffnc, 3, 1, 1, groups=ffnc)
        self.ffn_sg = SimpleGate()
        self.ffn2 = nn.Conv2d(ffnc // 2, c, 1, 1, 0)

        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)))
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)))

        self.drop_path = nn.Identity()  # 심플하게

    def forward(self, x):
        shortcut = x

        x = self.pw1(x)
        x = self.dw(x)
        x = self.sg(x)  # (B, dwc//2, H, W)

        x = x * self.sca(x)
        x = self.pw2(x)
        y = shortcut + self.drop_path(self.beta * x)

        # FFN
        x2 = self.ffn1(y)
        x2 = self.ffn_dw(x2)
        x2 = self.ffn_sg(x2)
        x2 = self.ffn2(x2)
        y = y + self.drop_path(self.gamma * x2)
        return y

class NAFStem(nn.Module):
    """ 가벼운 NAFNet 전처리 스템 (해상도 유지) """
    def __init__(self, in_ch=3, width=64, n_blocks=4):
        super().__init__()
        self.head = nn.Conv2d(in_ch, width, 3, 1, 1)
        self.body = nn.Sequential(*[NAFBlock(width) for _ in range(n_blocks)])
        self.tail = nn.Conv2d(width, in_ch, 3, 1, 1)

    def forward(self, x):
        h = self.head(x)
        h = self.body(h)
        h = self.tail(h)
        # 입력에 잔차로 더해 안정화
        return x + h

# ---- 하이브리드: NAFNet(전처리) -> HATX(본처리/업샘플) ----
@ARCH_REGISTRY.register()
class HybridHATNAF(nn.Module):
    """
    직렬 하이브리드:
      x --(NAFStem)--> x_naf --(HATX)--> y
    - NAFStem이 노이즈/저주파 정리
    - HATX가 글로벌 컨텍스트/업샘플 담당
    - HATX는 기존 hatx_arch.py 그대로 사용
    """
    def __init__(
        self,
        # NAF 쪽
        naf_width: int = 64,
        naf_blocks: int = 4,
        # (편의) 최상위에서 직접 받으면 hat_kwargs와 병합
        window_size: int | None = None,
        upscale: int = 2,
        in_chans: int = 3,
        # HATX 전달 파라미터
        hat_kwargs: dict | None = None,
    ):
        super().__init__()
        self.naf = NAFStem(in_ch=in_chans, width=naf_width, n_blocks=naf_blocks)

        # ---- hat_kwargs와 top-level 인자 병합
        hk = {} if hat_kwargs is None else dict(hat_kwargs)
        # window_size 우선순위: top-level 인자 > hat_kwargs > 기본
        if window_size is None:
            window_size = int(hk.get("window_size", 8))
        hk["window_size"] = int(window_size)

        # upscale/in_chans도 빠져있으면 채워줌
        hk.setdefault("upscale", int(upscale))
        hk.setdefault("in_chans", int(in_chans))

        # HATX 생성
        self.hat = HATX(**hk)

        # ---- 바깥에서 바로 참조할 수 있게 편의 속성 노출
        # (HATModel이나 기타 유틸이 net.window_size / net.upscale 등을 직접 읽는 경우 대응)
        self.window_size = int(window_size)
        self.upscale = int(hk["upscale"])
        self.in_chans = int(hk["in_chans"])
        self.img_range = getattr(self.hat, "img_range", 1.0)

    def forward(self, x):
        x_naf = self.naf(x)
        y = self.hat(x_naf)
        return y

    def extra_repr(self) -> str:
        return f"window_size={self.window_size}, upscale={self.upscale}, in_chans={self.in_chans}"
