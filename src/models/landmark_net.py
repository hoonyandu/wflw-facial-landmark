"""Part B 모델: 직접좌표 회귀(direct coordinate). MobileNetV3-small + FC head.

파라미터화 선택 근거(CPU 20ms 예산에서 히트맵 디코드 비용이 부담스러움)와
그 대가(occlusion 하 "평균으로 회귀"하는 실패 모드)는 DESIGN.md §3, E-1 참조.
"""
import torch
import torch.nn as nn
from torchvision.models import mobilenet_v3_small

NUM_POINTS = 98


class LandmarkNet(nn.Module):
    def __init__(self, num_points=NUM_POINTS, pretrained=True):
        super().__init__()
        m = mobilenet_v3_small(weights="DEFAULT" if pretrained else None)
        self.backbone = m.features            # ~1.5M params
        self.pool = nn.AdaptiveAvgPool2d(1)
        c = m.classifier[0].in_features       # 576
        self.head = nn.Sequential(
            nn.Linear(c, 256), nn.Hardswish(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, num_points * 2),   # 196: (x,y) in [0,1]
        )

    def forward(self, x):
        x = self.backbone(x)
        x = self.pool(x).flatten(1)
        x = self.head(x)
        return x.view(x.size(0), NUM_POINTS, 2)  # 정규화 좌표


class WingLoss(nn.Module):
    """Wing loss: 작은 오차에 로그 곡률을 줘 미세 정렬을 개선(Feng et al. 2018).
    landmark 회귀에서 L2/L1보다 눈·입 같은 세밀 구조에 유리."""
    def __init__(self, w=10.0, eps=2.0):
        super().__init__()
        self.w, self.eps = w, eps
        self.C = w - w * torch.log(torch.tensor(1.0 + w / eps))

    def forward(self, pred, target):
        d = (pred - target).abs()
        return torch.where(d < self.w,
                           self.w * torch.log(1 + d / self.eps),
                           d - self.C).mean()


def count_params_mb(model):
    """모델 파라미터 수와 FP32 기준 용량(MB)을 계산

    Args:
        model: nn.Module

    Returns:
        (파라미터 개수, FP32 MB)
    """
    n = sum(p.numel() for p in model.parameters())
    return n, n * 4 / 1e6


if __name__ == "__main__":
    net = LandmarkNet(pretrained=False).eval()
    n, mb = count_params_mb(net)
    print(f"params={n:,}  fp32={mb:.1f} MB  (budget 25MB)")
    # 지연 측정 방법론(DESIGN.md §3): CPU 단일스레드, warmup 20, 측정 200회, median+IQR.
    import time
    torch.set_num_threads(1)
    x = torch.randn(1, 3, 128, 128)
    with torch.no_grad():
        for _ in range(20):
            net(x)
        ts = []
        for _ in range(200):
            t = time.perf_counter(); net(x); ts.append(time.perf_counter() - t)
    import numpy as np
    ts = np.array(ts) * 1e3
    print(f"latency median={np.median(ts):.1f}ms  IQR=[{np.percentile(ts,25):.1f},"
          f"{np.percentile(ts,75):.1f}]  (budget 20ms; state your HW)")
