# WFLW Facial Landmark under Occlusion

> 가림 상황에서 얼굴 랜드마크를 (A) 비학습 / (B) 학습 두 방식으로 풀고,
> 비교(C)하며, 라벨 없이 스스로 실패를 감지하는 accept/reject(D)를 붙인다.
>
> 설계·추론 근거는 `DESIGN.md` 참조.

## Structure

```
src/
    ├── data/
    │   ├── wflw_annotation_repository.py    # WFLW 어노테이션 파싱(외부 포맷 어댑터)
    │   └── dataset.py                       # torch Dataset
    ├── shared/
    │   ├── crop_geometry.py                 # crop/좌표 변환 순수 함수(학습·평가 공유)
    │   ├── yaw_estimation.py                # yaw(GT PnP) 파생 + 구간화(Part C)
    │   └── nme_metrics.py                   # NME 3종 정규화 + 실패율 + subset 리포트(A/B/C/D 공유)
    ├── application/
    │   ├── baseline_use_case.py             # Part A 비학습 베이스라인
    │   ├── train_use_case.py                # Part B 학습 루프(train/val 분리 + early stopping)
    │   ├── eval_use_case.py                 # Part B 추론(crop -> 모델 -> 원본 좌표 복원)
    │   ├── comparative_analysis_use_case.py # Part C 비교 분석(yaw곡선/겹침/오차위치화/실패이미지)
    │   └── accept_reject_use_case.py        # Part D disagreement 신호 + accept/reject sweep
    ├── models/
    │   └── landmark_net.py                  # Part B 모델(직접좌표) + WingLoss + 지연 벤치
    ├── notebooks/
    │   └── part_a~e.ipynb                   # Part별 실행/분석 노트북(학습/비교/결정 로직의 실제 실행 지점)
    └── settings.py                          # Config 클래스 기반 설정

output/
    ├── weights/landmarknet.pt               # Part B 학습 체크포인트
    └── results/preds.npz                    # Part B가 생성하는 Part C/D/E 공유 예측 캐시
```

## Environment

```
Python: 3.12
OS: macOS (Apple Silicon)
학습 가속: MPS
```

## Prerequisites

```bash
pip install -r requirements.txt
```

WFLW를 다운로드하여 `data/WFLW/`에 배치
```
data/WFLW/
    ├── images/
    └── annotations/list_98pt_rect_attr_{train,test}.txt
```

## Usage

재현 순서 (fixed seed=42)

`src/notebooks/`에서 실행:

1. **Part A:** `part_a.ipynb` (선행 조건 없음)
2. **Part B:** `part_b.ipynb` (가중치 없으면 자동 학습 → `output/weights/landmarknet.pt`, 있으면 스킵; `output/results/preds.npz`도 여기서 생성)
3. **Part C/D/E:** `part_c.ipynb`, `part_d.ipynb`, `part_e.ipynb` (2에서 만든 캐시를 읽음)

## Status

- [x] 지표, 베이스라인, yaw 파생, 모델, 결정 로직
- [x] Part A/B NME 표 (Apple M3 Pro, 60 epoch 완주)
- [x] Part C 그림/겹침 수치/실패 이미지 (`figs/`)
- [x] Part D 곡선/운영점: 요구조건 동시 만족 불가로 판정, 근거는 `DESIGN.md` §5
- [x] `src/notebooks/part_a~e.ipynb` 전부 실행 확인
- [x] `DESIGN.md` §7 정직성 노트

평가·측정 방법론과 결과 해석은 `DESIGN.md` 참조
