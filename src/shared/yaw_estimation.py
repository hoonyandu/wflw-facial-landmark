"""Part C: yaw를 정답 landmark에서 PnP로 파생 (예측에서 뽑으면 bin 할당이
측정 대상 오차의 함수가 되므로 GT에서만). WFLW 6점 -> 3D 얼굴 모델 ->
cv2.solvePnP -> Euler yaw. 방법/한계는 DESIGN.md §4.1 참조.
"""
import numpy as np
import cv2

# WFLW 98점 중 PnP에 쓸 6점 (표준 head-pose 6점 모델과 대응)
IDX = dict(nose_tip=54, chin=16, l_eye_outer=60, r_eye_outer=72,
           l_mouth=76, r_mouth=82)

# 일반 3D 얼굴 모델 (mm, 임의 스케일). 코끝 원점.
MODEL_3D = np.array([
    [0.0,    0.0,    0.0],     # nose tip
    [0.0,  -63.6,  -12.5],     # chin
    [-43.3, 32.7,  -26.0],     # left eye outer
    [43.3,  32.7,  -26.0],     # right eye outer
    [-28.9,-28.9,  -24.1],     # left mouth
    [28.9, -28.9,  -24.1],     # right mouth
], dtype=np.float64)

_ORDER = ["nose_tip", "chin", "l_eye_outer", "r_eye_outer", "l_mouth", "r_mouth"]


def yaw_from_landmarks(gt, img_wh):
    """GT 6점을 PnP에 통과시켜 yaw를 구함

    Args:
        gt: (98,2) 픽셀 좌표
        img_wh: (W,H)

    Returns:
        yaw(도, 부호 있음)
    """
    pts2d = np.array([gt[IDX[k]] for k in _ORDER], dtype=np.float64)
    W, H = img_wh
    f = W  # focal 근사; 내부파라미터 미상이라 W로 둠(문서에 한계 명시)
    K = np.array([[f, 0, W / 2], [0, f, H / 2], [0, 0, 1]], dtype=np.float64)
    ok, rvec, _ = cv2.solvePnP(MODEL_3D, pts2d, K, np.zeros(4),
                               flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok:
        return np.nan
    R, _ = cv2.Rodrigues(rvec)
    # yaw = R로부터 Euler(양의 방향은 관례). atan2로 안정적으로 추출.
    yaw = np.degrees(np.arctan2(-R[2, 0], np.hypot(R[0, 0], R[1, 0])))
    return float(yaw)


def bin_by_yaw(yaws, edges=(0, 10, 20, 30, 45, 90)):
    """|yaw|를 구간에 할당

    Args:
        yaws: (N,) yaw(도)
        edges: bin 경계

    Returns:
        (idx, labels): idx는 (N,) bin index 배열, labels는 구간 라벨 리스트
    """
    a = np.abs(np.asarray(yaws))
    idx = np.digitize(a, edges[1:-1])  # 0..len(edges)-2
    labels = [f"{edges[i]}-{edges[i+1]}" for i in range(len(edges) - 1)]
    return idx, labels
