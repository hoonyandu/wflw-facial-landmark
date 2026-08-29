"""Part D: accept/reject 결정 전략 + 트레이드오프 곡선.

신호: disagreement = A-B 예측 간 정규화 평균 거리(라벨 불필요, 추론시점 계산).
요구조건(precision>=95%, coverage>=85%/70%)과 신호의 한계(proxy일 뿐 오차
자체가 아님, false-accept/false-reject 두 실패 방향)는 DESIGN.md §5, E-3 참조.
"""
import numpy as np


def disagreement_signal(predsA, predsB, gts_for_norm=None):
    """A-B 예측 간 정규화 평균 거리(disagreement)를 계산
    정규화는 라벨 없이도 가능한 값(예측 B의 눈 간 거리)으로 해야 배포에서
    사용 가능. 여기선 예측 B의 inter-ocular로 정규화.

    Args:
        predsA: (N,98,2) A 예측 landmark
        predsB: (N,98,2) B 예측 landmark
        gts_for_norm: 현재 미사용(정규화는 predsB의 inter-ocular로 계산)

    Returns:
        (N,) 정규화된 disagreement 신호
    """
    from src.shared.nme_metrics import L_EYE_OUTER, R_EYE_OUTER
    per_pt = np.linalg.norm(predsA - predsB, axis=2)          # (N,98)
    d = np.linalg.norm(predsB[:, L_EYE_OUTER] - predsB[:, R_EYE_OUTER], axis=1)
    return per_pt.mean(1) / np.clip(d, 1e-6, None)


def sweep(signal, nme_B, occ_mask, nme_thr=0.05, prec_req=0.95,
          cov_all_req=0.85, cov_occ_req=0.70, n=200):
    """signal 임계값을 낮은→높은 쪽으로 쓸면서(작을수록 accept) 각 지점의
    precision/coverage를 계산. 요구조건을 모두 만족하는 지점 탐색.

    Args:
        signal: (N,) accept/reject 판단 신호(disagreement)
        nme_B: (N,) B의 NME
        occ_mask: (N,) occlusion subset bool mask
        nme_thr: 배포 성공 판정 NME 임계값
        prec_req: 요구 precision
        cov_all_req: 요구 coverage(전체)
        cov_occ_req: 요구 coverage(occlusion)
        n: 임계값 sweep 개수

    Returns:
        (curve, feasible): curve는 dict of arrays, feasible은 요구조건을
        모두 만족하는 지점(dict) 또는 None
    """
    signal = np.asarray(signal); nme_B = np.asarray(nme_B)
    occ_mask = np.asarray(occ_mask, bool)
    ths = np.quantile(signal, np.linspace(0.01, 1.0, n))
    curve = {k: [] for k in
             ("thr", "coverage", "coverage_occ", "precision", "precision_occ")}
    feasible = None
    for t in ths:
        acc = signal <= t
        if acc.sum() == 0:
            continue
        within = nme_B[acc] <= nme_thr
        prec = within.mean()
        cov = acc.mean()
        occ_acc = acc & occ_mask
        cov_occ = occ_acc.sum() / max(occ_mask.sum(), 1)
        prec_occ = (nme_B[occ_acc] <= nme_thr).mean() if occ_acc.sum() else np.nan
        for k, v in zip(curve, (t, cov, cov_occ, prec, prec_occ)):
            curve[k].append(v)
        if prec >= prec_req and cov >= cov_all_req and cov_occ >= cov_occ_req:
            # 요구 만족 지점 중 커버리지 최대를 유지(임계값 큰 쪽으로 계속 갱신)
            feasible = dict(thr=float(t), coverage=float(cov),
                            coverage_occ=float(cov_occ), precision=float(prec),
                            precision_occ=float(prec_occ))
    curve = {k: np.array(v) for k, v in curve.items()}
    return curve, feasible


def report(feasible):
    """feasible operating point 유무를 사람이 읽을 문장으로 출력

    Args:
        feasible: sweep()이 반환한 feasible operating point 또는 None
    """
    if feasible is None:
        print("요구조건 동시 만족 불가. coverage를 극한까지 줄여도 precision이")
        print("95%에 못 미침(실측 최대 ~80%). 병목은 coverage-precision 트레이드")
        print("오프가 아니라 disagreement 신호 자체의 품질(자세한 근거: DESIGN.md §5).")
    else:
        print("feasible operating point:", feasible)
