"""crop/좌표 변환 순수 함수. 학습(dataset.py)과 추론(eval_use_case.py)이 공유함

좌표계 규약(전 파이프라인 일관):
  - 모델 타깃/출력: crop 기준 [0,1] 정규화 좌표.
  - NME 계산: [0,1] -> *size -> crop affine 역변환 -> 원본 픽셀 좌표.

학습/추론이 이 crop 로직을 공유하지 않으면 좌표계가 어긋나므로 별도
모듈로 분리해 shared 계층에 배치(파일 I/O 없는 순수 함수).
"""
import numpy as np
import cv2


def crop_and_normalize(img, bbox, size=128, pad=0.15):
    """얼굴 박스를 pad만큼 확장해 정사각 crop 후 size로 리사이즈

    Args:
        img: 원본 이미지 (HWC)
        bbox: (x0,y0,x1,y1)
        size: 출력 정사각 crop 한 변 길이
        pad: 박스 확장 비율

    Returns:
        (crop(HWC uint8), M). M은 원본->crop 2x3 affine 행렬
    """
    x0, y0, x1, y1 = bbox
    w, h = x1 - x0, y1 - y0
    cx, cy = x0 + w / 2, y0 + h / 2
    s = max(w, h) * (1 + pad)
    nx0, ny0 = cx - s / 2, cy - s / 2
    M = np.array([[size / s, 0, -nx0 * size / s],
                  [0, size / s, -ny0 * size / s]], float)
    crop = cv2.warpAffine(img, M, (size, size))
    return crop, M  # landmark_crop = M @ [x,y,1]


def apply_affine(pts, M):
    """crop_and_normalize가 반환한 M으로 원본 좌표를 crop 좌표로 변환

    Args:
        pts: (N,2) 원본 좌표
        M: crop_and_normalize가 반환한 2x3 affine 행렬

    Returns:
        (N,2) crop 좌표
    """
    return (pts @ M[:, :2].T) + M[:, 2]


def invert_affine(pts_crop, M):
    """crop 픽셀 좌표를 원본 픽셀 좌표로 복원(apply_affine의 역변환)

    Args:
        pts_crop: (N,2) crop 좌표
        M: crop_and_normalize가 반환한 2x3 affine 행렬

    Returns:
        (N,2) 원본 픽셀 좌표
    """
    A = M[:, :2]; b = M[:, 2]
    Ainv = np.linalg.inv(A)
    return (pts_crop - b) @ Ainv.T
