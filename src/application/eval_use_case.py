"""Part B 추론 use case: 얼굴 crop을 모델에 통과시켜 원본 픽셀 좌표로 복원함.

실행 진입점: `src/notebooks/part_b.ipynb`(직접 호출, `output/results/preds.npz` 생성까지 포함).
"""
import numpy as np
import cv2
import torch

from src.shared.crop_geometry import crop_and_normalize, invert_affine


def run_model(net, paths, boxes, size, dev):
    """얼굴 crop을 모델에 통과시켜 원본 픽셀 좌표로 복원

    Args:
        net: 학습된 모델
        paths: 이미지 경로 리스트
        boxes: (N,4) [x0,y0,x1,y1]
        size: crop 정사각 한 변 길이
        dev: 실행 디바이스

    Returns:
        (N,98,2) 원본 픽셀 좌표계 예측
    """
    net.eval()
    out = []
    with torch.no_grad():
        for p, box in zip(paths, boxes):
            img = cv2.imread(p)
            crop, M = crop_and_normalize(img, box, size)
            x = torch.from_numpy(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)) \
                     .permute(2, 0, 1).float().unsqueeze(0).to(dev) / 255.0
            pred01 = net(x)[0].cpu().numpy()            # (98,2) in [0,1]
            pred_crop = pred01 * size
            out.append(invert_affine(pred_crop, M))     # 원본 픽셀
    return np.stack(out)
