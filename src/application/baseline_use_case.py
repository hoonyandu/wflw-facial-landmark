"""Part A 비학습 베이스라인: 평균 형상(mean shape) + 박스 정렬.

학습셋 landmark의 평균(통계량, 학습 아님)을 각 테스트 박스에 맞춤.
"정면·일관된 박스" 가정이 깨지는 pose/occlusion에서 체계적으로 실패하도록
설계됨. 선택 근거와 대안(PDM, off-the-shelf 예측기)은 DESIGN.md §2 참조.
"""
import numpy as np


def compute_mean_shape(train_landmarks, train_bboxes):
    """학습셋 전체 landmark를 각자의 박스 기준으로 정규화한 뒤 평균을 냄

    Args:
        train_landmarks: (M,98,2)
        train_bboxes: (M,4) [x0,y0,x1,y1]

    Returns:
        (98,2) 단위정사각형([0,1]^2) 좌표계의 평균 형상
    """
    norm = []
    for lm, (x0, y0, x1, y1) in zip(train_landmarks, train_bboxes):
        w, h = max(x1 - x0, 1e-6), max(y1 - y0, 1e-6)
        norm.append((lm - [x0, y0]) / [w, h])
    return np.mean(norm, axis=0)  # (98,2) in [0,1]


def fit_to_box(mean_shape, bbox):
    """평균 형상을 테스트 박스로 되돌림(축별 스케일 + 평행이동)

    similarity가 아니라 anisotropic로 두는 편이 박스 aspect에 덜 민감;
    더 엄격히 하려면 solveAffine으로 교체 가능.

    Args:
        mean_shape: (98,2) 단위정사각형([0,1]^2) 좌표계의 평균 형상
        bbox: (x0,y0,x1,y1)

    Returns:
        (98,2) 원본 픽셀 좌표계로 변환된 landmark
    """
    x0, y0, x1, y1 = bbox
    return mean_shape * [x1 - x0, y1 - y0] + [x0, y0]


def predict(mean_shape, bboxes):
    """테스트 박스 전체에 평균 형상을 맞춰 예측 생성

    Args:
        mean_shape: (98,2) 단위정사각형([0,1]^2) 좌표계의 평균 형상
        bboxes: (N,4) [x0,y0,x1,y1]

    Returns:
        (N,98,2) 예측 landmark
    """
    return np.stack([fit_to_box(mean_shape, b) for b in bboxes])
