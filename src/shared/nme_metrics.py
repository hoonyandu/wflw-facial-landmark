"""NME(Normalized Mean Error) 계산. Part A/B/C/D가 공유하는 순수 함수 모음

I/O나 상태 없이 좌표 배열만 받는 순수 계산이라 shared 계층에 배치.
Part D의 5% NME 임계값, Part E-2의 정규화 비교가 전부 이 파일에 의존.

정규화 항을 3종 모두 구현해 두는 이유: Part E-2에서 "왜 정규화 선택이
Part C 곡선의 형태를 바꾸는가"를 실측 데이터로 보여줘야 하기 때문.
같은 예측/정답 쌍에 대해 3종을 다 계산해서 yaw 구간별로 겹쳐 그리면
inter-ocular가 yaw 증가에 따라 어떻게 붕괴하는지 그대로 드러남.
"""
import numpy as np

# WFLW 98-point 표준 인덱스 (0-based)
L_EYE_OUTER = 60   # 좌안 바깥 코너
R_EYE_OUTER = 72   # 우안 바깥 코너
L_PUPIL = 96       # 좌 동공 (WFLW 추가 2점)
R_PUPIL = 97       # 우 동공


def _norm_term(gt, kind, bbox=None):
    """정규화 항(스칼라) 계산

    Args:
        gt: (98,2) 정답 landmark
        kind: "inter_ocular" | "inter_pupil" | "bbox_diag"
        bbox: bbox_diag일 때만 사용. 없으면 정답 landmark의 tight box로 대체.

    Returns:
        정규화 스칼라
    """
    if kind == "inter_ocular":
        return np.linalg.norm(gt[L_EYE_OUTER] - gt[R_EYE_OUTER])
    if kind == "inter_pupil":
        return np.linalg.norm(gt[L_PUPIL] - gt[R_PUPIL])
    if kind == "bbox_diag":
        if bbox is None:
            x0, y0 = gt.min(0); x1, y1 = gt.max(0)
        else:
            x0, y0, x1, y1 = bbox
        return np.hypot(x1 - x0, y1 - y0)
    raise ValueError(kind)


def nme_single(pred, gt, kind="inter_ocular", bbox=None):
    """한 이미지의 NME 계산

    Args:
        pred: (98,2) 예측 landmark
        gt: (98,2) 정답 landmark
        kind: "inter_ocular" | "inter_pupil" | "bbox_diag"
        bbox: bbox_diag일 때만 사용. 없으면 정답 landmark의 tight box로 대체

    Returns:
        (NME 스칼라, per-point 정규화 오차 (98,))
    """
    d = _norm_term(gt, kind, bbox)
    per_point = np.linalg.norm(pred - gt, axis=1)
    return per_point.mean() / d, per_point / d


def nme_batch(preds, gts, kind="inter_ocular", bboxes=None):
    """배치 NME 계산

    Args:
        preds: (N,98,2) 예측 landmark
        gts: (N,98,2) 정답 landmark
        kind: "inter_ocular" | "inter_pupil" | "bbox_diag"
        bboxes: bbox_diag일 때만 사용. (N,4) 또는 None

    Returns:
        ((N,) NME, (N,98) per-point 정규화 오차)
    """
    N = len(preds)
    out, pp = np.empty(N), np.empty((N, 98))
    for i in range(N):
        b = None if bboxes is None else bboxes[i]
        out[i], pp[i] = nme_single(preds[i], gts[i], kind, b)
    return out, pp


def failure_rate(nmes, thr=0.10):
    """NME가 임계값을 넘는 프레임 비율

    WFLW 관례상 0.10을 자주 쓰지만, in-cabin 배포 관점이면 Part D의 0.05를
    함께 리포트하는 게 더 정직함. thr는 호출부에서 반드시 문서에 명시할 것.

    Args:
        nmes: (N,) NME
        thr: 실패 판정 임계값

    Returns:
        실패율(0~1)
    """
    return float((np.asarray(nmes) > thr).mean())


def report_by_subset(nmes, subset_masks, thr=0.10):
    """attribute subset(pose/expression/illumination/make-up/occlusion/blur)별 NME + 실패율

    Args:
        nmes: (N,) NME
        subset_masks: {name: bool array (N,)}
        thr: 실패 판정 임계값

    Returns:
        {name: (mean_nme, failure_rate)}, "overall" 포함
    """
    rows = {"overall": (float(np.mean(nmes)), failure_rate(nmes, thr))}
    for name, m in subset_masks.items():
        m = np.asarray(m, bool)
        if m.sum() == 0:
            continue
        rows[name] = (float(np.mean(nmes[m])), failure_rate(nmes[m], thr))
    return rows
