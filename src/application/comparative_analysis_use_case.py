"""Part C 비교 분석 use case: output/results/preds.npz를 읽어 그림과 수치 생성함.

실행 진입점: `src/notebooks/part_c.ipynb`(직접 호출).
산출: figs/  (yaw_curve.png, perpoint_heat.png, failure_*.png), 콘솔에 겹침 수치
"""
import numpy as np
import cv2
import matplotlib.pyplot as plt

from src.shared.nme_metrics import nme_batch
from src.shared.yaw_estimation import yaw_from_landmarks, bin_by_yaw


def yaw_curves(gts, img_wh, predsA, predsB, boxes, edges, out):
    """정규화 3종 x (A,B)로 yaw 구간별 평균 NME 곡선 생성. E-2 증거도 겸함.

    Args:
        gts: (N,98,2) 정답 landmark
        img_wh: (N,2) 이미지 (W,H)
        predsA: (N,98,2) A 예측 landmark
        predsB: (N,98,2) B 예측 landmark
        boxes: (N,4) bbox
        edges: yaw bin 경계
        out: 그림 저장 디렉터리

    Returns:
        (N,) GT 기반 yaw(도)
    """
    yaws = np.array([yaw_from_landmarks(g, wh) for g, wh in zip(gts, img_wh)])
    idx, labels = bin_by_yaw(yaws, edges)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
    for ax, norm in zip(axes, ["inter_ocular", "inter_pupil", "bbox_diag"]):
        nA, _ = nme_batch(predsA, gts, norm, boxes)
        nB, _ = nme_batch(predsB, gts, norm, boxes)
        xs = range(len(labels))
        ax.plot(xs, [nA[idx == b].mean() if (idx == b).any() else np.nan
                     for b in xs], "o-", label="A mean-shape")
        ax.plot(xs, [nB[idx == b].mean() if (idx == b).any() else np.nan
                     for b in xs], "s-", label="B net")
        ax.set_xticks(list(xs)); ax.set_xticklabels(labels, rotation=45)
        ax.set_title(f"NME vs |yaw|  ({norm})"); ax.set_xlabel("|yaw| deg")
        ax.legend()
    axes[0].set_ylabel("NME")
    fig.tight_layout(); fig.savefig(f"{out}/yaw_curve.png", dpi=120)
    print(f"[yaw] saved {out}/yaw_curve.png  (정규화별 기울기 차이가 E-2 증거)")
    return yaws


def perpoint_heat(gts, predsB, boxes, occ_mask, out):
    """occlusion subset에서 점별 평균 정규화 오차 -> 평균 얼굴 위에 색칠.
    '어떤 점이 어디서 실패하나'(오차 위치화).

    Args:
        gts: (N,98,2) 정답 landmark
        predsB: (N,98,2) B 예측 landmark
        boxes: (N,4) bbox
        occ_mask: (N,) occlusion subset bool mask
        out: 그림 저장 디렉터리
    """
    _, ppall = nme_batch(predsB, gts, "inter_ocular", boxes)   # (N,98)
    pp = ppall[occ_mask].mean(0)                               # (98,)
    tmpl = np.stack([g - g.mean(0) for g in gts]).mean(0)      # 정규화 평균형상
    fig, ax = plt.subplots(figsize=(5, 5))
    sc = ax.scatter(tmpl[:, 0], -tmpl[:, 1], c=pp, s=60, cmap="hot")
    for i in range(98):
        ax.annotate(str(i), (tmpl[i, 0], -tmpl[i, 1]), fontsize=5)
    plt.colorbar(sc, label="per-point NME (occlusion)")
    ax.set_title("Part B per-point error on occlusion"); ax.axis("equal")
    fig.tight_layout(); fig.savefig(f"{out}/perpoint_heat.png", dpi=120)
    print(f"[perpoint] saved {out}/perpoint_heat.png")


def overlaps(gts, predsA, predsB, boxes, thr=0.10):
    """이미지/점 수준 실패 겹침(Jaccard). 문서 논리 축:
    겹침이 낮음 = 오차가 독립적임 = Part D disagreement 신호가 유효함.

    Args:
        gts: (N,98,2) 정답 landmark
        predsA: (N,98,2) A 예측 landmark
        predsB: (N,98,2) B 예측 landmark
        boxes: (N,4) bbox
        thr: 실패 판정 NME 임계값

    Returns:
        image-level failure Jaccard
    """
    nA, ppA = nme_batch(predsA, gts, "inter_ocular", boxes)
    nB, ppB = nme_batch(predsB, gts, "inter_ocular", boxes)
    failA, failB = nA > thr, nB > thr
    inter = (failA & failB).sum(); union = (failA | failB).sum()
    img_j = inter / max(union, 1)
    # 점 수준: 둘 다 실패한 이미지 내에서 실패 점(정규화오차>thr) Jaccard 평균
    both = np.where(failA & failB)[0]
    pj = []
    for i in both:
        pa, pb = ppA[i] > thr, ppB[i] > thr
        u = (pa | pb).sum()
        if u:
            pj.append((pa & pb).sum() / u)
    print(f"\n[overlap] image-level failure Jaccard = {img_j:.3f}")
    print(f"[overlap] point-level failure Jaccard (both-fail imgs) = "
          f"{np.mean(pj) if pj else float('nan'):.3f}")
    print("  -> 낮을수록 오차가 독립적 -> disagreement 신호 유효(Part D).")
    return img_j


def failure_images(paths, gts, predsA, predsB, boxes, out, k=4):
    """각 방법의 최악 사례에 예측(빨강)·정답(초록) 오버레이.

    Args:
        paths: 이미지 경로 리스트
        gts: (N,98,2) 정답 landmark
        predsA: (N,98,2) A 예측 landmark
        predsB: (N,98,2) B 예측 landmark
        boxes: (N,4) bbox
        out: 그림 저장 디렉터리
        k: 저장할 최악 사례 개수
    """
    nA, _ = nme_batch(predsA, gts, "inter_ocular", boxes)
    nB, _ = nme_batch(predsB, gts, "inter_ocular", boxes)
    for tag, nmes, preds in [("A", nA, predsA), ("B", nB, predsB)]:
        worst = np.argsort(nmes)[-k:]
        for j, i in enumerate(worst):
            img = cv2.imread(str(paths[i]))
            for (x, y) in gts[i]:
                cv2.circle(img, (int(x), int(y)), 2, (0, 255, 0), -1)
            for (x, y) in preds[i]:
                cv2.circle(img, (int(x), int(y)), 2, (0, 0, 255), -1)
            cv2.imwrite(f"{out}/failure_{tag}_{j}_nme{nmes[i]:.3f}.png", img)
    print(f"[failure] saved worst-{k} overlays for A and B -> {out}/")
