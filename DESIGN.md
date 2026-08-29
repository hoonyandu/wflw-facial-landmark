# 설계 문서: 가림 상황에서의 얼굴 랜드마크 위치 추정

> 채점 55%(실패 분석 30% + 이론적 기반 25%)가 실험 결과 근거 서술을 요구.
> 아래 수치는 WFLW test 전체(N=2500)에 대해 실제 학습(60 epoch, early-stopping
> 감시, 조기종료 없이 완주)·평가한 결과.
>
> **재현:** `src/notebooks/part_a~e.ipynb` (학습/추론/비교/결정 로직 전부 노트북이 직접 실행,
> `part_b.ipynb`가 `output/results/preds.npz`까지 생성)

---

## 0. 컴포넌트 상호작용

accept/reject는 **B의 출력 뒤, downstream 앞**. 판단 신호는 라벨이 필요 없는 A–B 불일치(+ 선택적으로 모델 confidence).

```
                     face crop (128x128)
                          │
            ┌─────────────┴──────────────┐
            ▼                             ▼
  [A] mean-shape baseline        [B] LandmarkNet (CPU, <=20ms)
   (박스에 평균형상 정렬)          (직접 좌표 회귀, 196 out)
            │                             │
            └──────────┬──────────────────┘
                       ▼
        disagreement = ‖A − B‖ / iod(B)   ← 라벨 불필요, 추론시점 계산
                       │
                       ▼
             ┌──────────────────┐
             │  ACCEPT / REJECT │  ← Part D 결정 지점 (여기)
             │  signal <= τ ?   │
             └────────┬─────────┘
              accept  │   reject
                      ▼        ▼
            downstream pose   withhold (feature 미제공)
```

---

## 1. 지표 선택
> `src/shared/nme_metrics.py`

1. 주 지표 **NME**, 실패율은 두 임계값으로 리포트: WFLW 관례 **0.10**(문헌 비교용), Part D 배포 기준 **0.05**(제품 관점).
2. 정규화 항 **3종 모두 계산**(inter-ocular / inter-pupil / bbox-diag), Part E-2 증거용.
   - 실측: yaw 45-90도 구간, Part A NME가 정규화만으로 inter-ocular=0.467 / inter-pupil=0.687 / bbox-diag=0.127 → 3.7배 이상 차이 (§4.2, E-2).
3. 주 리포트는 **inter-ocular**(문헌 관례). Part C yaw 곡선은 **bbox-diag**도 함께: inter-ocular는 yaw에 오염됨(E-2).

---

## 2. Part A: 비학습 베이스라인
> `src/application/baseline_use_case.py`

1. **선택:** 평균 형상(mean shape) + 박스 정렬. 학습 없음. 학습셋 landmark 평균(통계량)을 테스트 박스에 축별 스케일+평행이동으로 정렬.
2. **가정:** 얼굴이 대체로 정면, 박스가 얼굴을 일관되게 감쌈.
3. **의도적 실패 구조:** 이 가정이 깨지는 pose/occlusion에서 체계적으로 실패. 예측이 항상 "평균 얼굴"이므로 yaw가 커질수록 오차가 단조 증가. Part E-1 "regression to the mean"의 극단 사례이자, 학습 모델(B)이 occlusion에서 약하게 재현하는 실패의 참조점.
4. 점 집합 불일치 없음(A도 98점 그대로 출력). off-the-shelf 예측기를 썼다면 68↔98 매핑과 학습 데이터 명시 필요.

**결과** (WFLW test, N=2500, inter-ocular, thr=0.10):

| subset | NME | fail% |
|---|---|---|
| overall | 0.2700 | 89.68 |
| pose | 0.6052 | 100.00 |
| expression | 0.2900 | 96.18 |
| illumination | 0.2442 | 87.39 |
| make-up | 0.2298 | 87.86 |
| occlusion | 0.2597 | 89.13 |
| blur | 0.2714 | 91.59 |

- 거의 전 subset 실패율 90% 안팎: 설계 의도(체계적 실패의 순수 사례)와 일치.
- pose subset 최악(NME 0.605, fail 100%): 정면 가정이 가장 크게 깨지는 subset이라 예측 가능한 결과.

---

## 3. Part B: 학습 모델
> `src/models/landmark_net.py`, `src/application/train_use_case.py`, `src/application/eval_use_case.py`

1. **파라미터화: 직접 좌표 회귀.** MobileNetV3-small(features) + FC 헤드, 입력 128×128, 출력 196=(x,y)×98 정규화 좌표. 손실: **Wing loss**.
2. **예산 추론:** 25MB(≈6.25M params)는 넉넉하나, 실질 제약은 **CPU 20ms**. 직접 좌표는 GAP→FC 단일 연산이라 저렴하고, 98점 히트맵(64×64×98 + deconv/argmax)은 20ms에서 압박이 큼 → 예산이 파라미터화를 강제하지 않는 지점에서 지연이 직접 좌표를 선호.
3. **헤드룸 사용처(모델이 예산 아래일 때):**
   - (a) 보조 히트맵 헤드로 per-point confidence 확보 → Part D 신호 개선
   - (b) 입력 128→160
4. **지연 측정 방법론:** 하드웨어 Apple M3 Pro(macOS), CPU 단일 스레드(`torch.set_num_threads(1)`), warmup 20회, 측정 200회, median+IQR, batch=1, 입력 128×128 (`landmark_net.py`의 `__main__`). 학습 자체는 CPU 예산과 무관하므로 MPS(Apple GPU)로 가속하고, 지연 측정만 CPU/단일스레드로 별도 실행.

**결과:**

- **예산:** params=1,125,092(≈1.1M), FP32=4.5MB (budget 25MB), 여유 5.5배.
- **지연:** median=7.2ms, IQR=[7.1, 7.3]ms (budget 20ms), 여유 2.8배. 두 예산 다 여유로워 헤드룸 논의(위 3번)가 실제로 유효.
- **NME** (WFLW test, N=2500, inter-ocular, thr=0.10, 60 epoch 완주, best val_loss=0.0766):

| subset | NME | fail% |
|---|---|---|
| overall | 0.0791 | 20.80 |
| pose | 0.1605 | 74.85 |
| expression | 0.0873 | 23.57 |
| illumination | 0.0764 | 16.76 |
| make-up | 0.0801 | 22.82 |
| occlusion | 0.0946 | 30.43 |
| blur | 0.0870 | 23.03 |

Part A 대비 overall NME 0.270→0.079, 약 3.4배 개선. 그럼에도 배포 임계값(NME≤0.05, thr=0.05 재평가)에서는 **overall fail=66.64%, occlusion fail=78.26%**: "학습을 완료했다고 해서 배포 기준을 만족하는 것은 아님"이 이 프로젝트의 핵심 긴장(§5).

---

## 4. Part C: 비교 분석
> `src/shared/yaw_estimation.py`, `src/application/comparative_analysis_use_case.py`

### 4.1 yaw는 정답에서 파생

1. 예측 landmark로 yaw를 구하면 bin 할당이 측정 대상 오차의 함수가 되고, 그 오염은 정확히 고각도에서 최악.
2. 방법: **GT 6점 → 일반 3D 모델 → solvePnP → Euler yaw**. bin 경계 `[0,10,20,30,45,90]`.
3. 한계: 카메라 내부파라미터 미상이라 focal을 W로 근사. 절대 각도는 부정확할 수 있으나 순서/구간화에는 충분.

### 4.2 세 질문

**1) 실패 위치와 메커니즘**
- A: yaw↑에서 단조 붕괴(평균 형상이 회전 표현 못 함) + 윤곽/턱선점 최악. occlusion에는 둔감(애초에 이미지를 안 봄).
- B: occlusion·blur에서 국소 붕괴. 가려진 점 주변을 평균으로 채워 자신 있게 틀림. yaw에는 A보다 강건(데이터로 회전 일부 학습).
- **결과** (`figs/perpoint_heat.png`, occlusion subset, Part B, inter-ocular point-NME):
  - 최악 8개 포인트: **0,1,2,28,29,30,31,32** (NME 0.122~0.139), 전부 WFLW 98점 스킴의 **턱선/얼굴 윤곽** 점.
  - 가장 안정적: **51,52,64,65,75** (NME 0.067~0.070), 콧대/눈 주변, 텍스처가 뚜렷하고 덜 가려지는 위치.
  - 메커니즘: 턱선은 (a) occlusion 시 손/머리에 가장 먼저 가려지고 (b) 텍스처 대비가 약해 국소화 신호 자체가 약함.

**2) 실패 겹침 정량화**
- **결과** (thr=0.10, inter-ocular): **image-level Jaccard = 0.230**, **point-level Jaccard = 0.642**.
- 이미지 수준 겹침이 낮음 → A(기하만 봄)와 B(픽셀을 봄)가 서로 다른 이미지에서 주로 실패 → Part D의 A–B disagreement 신호가 in principle 유효할 근거(§5, 단 실제 corr=0.484로 완벽하진 않음. E-3 참조).
- 점 수준 겹침은 상대적으로 높음(0.642): 둘 다 실패하는 이미지 안에서는 실패 지점이 겹침(주로 턱선점). 즉 "이미지 겹침"과 "지점 겹침"은 다른 질문.

**3) 배포 판단**
- **B를 전반적으로 추천**: 모든 subset에서 A보다 NME 낮음(§2/§3), occlusion에서도 A(0.260) > B(0.095).
- 단, B 단독으로는 Part D 기준(§5) 미달 → 실제 배포 단위는 **B + accept/reject 게이팅**.
- **판단을 뒤집을 증거:** in-cabin IR에서 B의 occlusion NME가 A 수준(0.260)까지 악화되는 경우(학습분포 이탈로 B의 우위 소멸) → A의 예측 가능한 열화가 하한으로 더 안전(E-4).
- 대표 실패 이미지: `figs/failure_A_*.png`(worst-4, NME 1.1~1.35), `figs/failure_B_*.png`(worst-4, NME 0.67~1.04).

---

## 5. Part D: 트레이드오프
> `src/application/accept_reject_use_case.py`

1. **신호:** `disagreement = mean‖A−B‖ / iod(B)` (라벨 불필요, 추론시점 계산).
2. **전략:** 임계값 τ, `signal<=τ`면 accept. τ를 쓸며 precision(수용분 5%NME 이내 비율) vs coverage(전체/occlusion) 곡선을 그림.
3. **요구조건:** precision≥95% ∧ coverage≥85%(전체) ∧ ≥70%(occlusion) 동시 만족.
4. **τ 조정 방향:** τ↑ → coverage↑ precision↓(silent error 위험↑) / τ↓ → precision↑ coverage↓(feature 미제공↑, 제품성↓).

**결과: 요구조건 동시 만족 불가(infeasible).** 곡선(`figs/partD_curve.png`)에서 뽑은 수치:

| 지점 | τ | coverage | coverage_occ | precision |
|---|---|---|---|---|
| no-reject(전체 수용) | 0.967 | 1.000 | 1.000 | 0.334 |
| coverage 요건(≥85%/≥70%) 만족 중 최고 precision | 0.412 | 0.851 | 0.882 | 0.392 |
| 곡선상 최고 precision(거의 전부 거부) | 0.056 | 0.010 | 0.011 | 0.800 |

**애초 가설보다 나쁜 종류의 infeasible.** "occlusion 꼬리가 두꺼워서 커버리지 요건과 충돌한다"고 예상했으나, coverage를 1%까지 줄여도 precision이 80% 수준에서 정체되어 95%에 도달하지 못함. 즉 병목은 "커버리지-precision 트레이드오프"가 아니라 **A–B disagreement 신호 자체가 B의 오차를 95% 신뢰도로 걸러내지 못함**(E-3: corr=0.484, false-accept 사분면 질량 23.1%). no-reject 기준선도 낮음(overall 33.4%, occlusion 21.7%만 5%NME 이내). 신호 품질보다 B의 절대 정확도 부족이 더 큼.

**바뀌어야 할 것:**
1. B 자체 정확도 개선(in-cabin/occlusion 데이터, 예산 5.5배 여유 활용)
2. 신호 개선(보조 히트맵 confidence, §8 다음 실험)
3. 그럼에도 불충분하면 downstream 5%NME 기준 완화

### 5.1 프로덕션 모니터링 (GT 없음)

차량에서 정답 landmark가 없으므로 분포/일관성 대리지표를 확인:
1. disagreement 신호 분포 + **acceptance rate** 추적 → 급변 시 알람
2. 입력 품질 proxy: 얼굴 박스 크기, blur(라플라시안 분산), 밝기/대비
3. 참조 분포 대비 **PSI/KL 드리프트**(웹→IR 도메인 이동 탐지)
4. downstream pose의 프레임 간 jitter로 간접 오차 감시
5. 표류 정의: acceptance rate/disagreement 분포가 참조 대비 유의하게 이동 → "5%NME 만족" 가정 붕괴 신호

---

## 6. Part E: 분석

**E-1. 좌표 파라미터화**

B는 정규화 (x,y)를 직접 회귀. 실패 모드: occlusion/모호성에서 평균 위치로 회귀. 공간적 불확실성을 못 내놓고 자신 있게 틀림(silent error 그 자체). 히트맵은 국소화가 낫고 peak 값이 자연스러운 per-point confidence 제공(Part D에 유용). 대신 98점 고해상 출력의 메모리/연산이 크고 argmax 양자화 오차 있음.

**히트맵을 골랐어야 할 조건(실측 정량화):** 직접좌표 지연 median 7.2ms/budget 20ms(2.8배 헤드룸). 히트맵 경로(64×64×98 디코드)는 보수적으로 3~5배 연산 비용 → 대략 20~35ms대 → 지금 헤드룸(2.8배)으론 부족, budget이 ~50ms대였어야 여유 있게 충족함. 즉 이 프로젝트 예산(20ms)의 2.5배 이상이어야 히트맵이 유효하고, 현재 예산에서는 직접좌표가 맞는 선택. 단 Part D(disagreement corr=0.484로 불완전)를 보면 per-point confidence의 가치는 확인됨 → §8 다음 실험으로 이어짐.

**E-2. 정규화와 지표 조건화**

yaw가 커지면 투영에서 두 눈이 가까워져 inter-ocular 정규화 항이 줄어듦 → 같은 픽셀 오차라도 NME가 부풀려짐 → 고각도에서 pose 열화가 실제보다 심해 보임. bbox-diag는 회전에 더 안정적.

실측(`figs/yaw_curve.png`, Part A, yaw는 GT 6점 PnP 파생, bin=[0,10,20,30,45,90]):

| yaw(도) | inter-ocular | inter-pupil | bbox-diag |
|---|---|---|---|
| 0-10  | 0.165 | 0.237 | 0.063 |
| 10-20 | 0.179 | 0.255 | 0.069 |
| 20-30 | 0.217 | 0.309 | 0.080 |
| 30-45 | 0.291 | 0.422 | 0.096 |
| 45-90 | 0.467 | 0.687 | 0.127 |

같은 예측(Part A)에 대해 0-10도→45-90도: inter-ocular 2.8배(0.165→0.467), inter-pupil 2.9배 증가, **bbox-diag는 2.0배만**(0.063→0.127). 절대값도 inter-ocular 대비 3.7배 작음. Part B도 같은 방향(0.059→0.126, 2.1배)이지만 기울기는 A보다 완만: B가 회전을 일부 학습했다는 근거(§4.2-1).

**함의:** 논문 간 NME 비교는 위험. 정규화가 다르고, inter-ocular 수치는 각 데이터셋의 pose 분포에 조건화돼 있어 서로 비교 불가.

**E-3. confidence와 오차**

disagreement는 오차의 proxy지 오차 자체는 아님.
- 실패 방향①(위험): A/B가 같은 방식으로 틀림(상관 실패) → disagreement 낮은데 오차 큼 → false-accept, Part D가 차단하지 못하는 실질적 위험.
- 실패 방향②: 한쪽의 양성 분산으로 disagreement는 큰데 B는 정확 → false-reject → 불필요한 커버리지 손실.

**결과** (`figs/partE3_scatter.png`, disagreement vs NME_B, N=2500): **corr = 0.484**, 양의 상관은 있지만 약함. 신호 중앙값으로 절반씩 나누면:
1. 방향①: 전체의 **23.1%**가 이 사분면. 신호 하위 50% 안에서는 **46.2%**가 실제로 NME>0.05: "신호가 낮다"고 안심한 프레임의 거의 절반이 배포 기준 미달. Part D에서 disagreement 단독으로 95% precision에 못 미치는(최대 80%) 핵심 원인.
2. 방향②: 전체의 **6.4%**, 방향①보다 작은 손실. 즉 신호는 "몰래 통과"(위험) 쪽으로 치우침.
3. **근거:** 방향① 질량이 큰 이유는 A와 B가 같은 얼굴 박스를 입력으로 공유하기 때문. 박스가 얼굴을 잘못 감싸면 둘 다 같은 방향으로 편향돼 disagreement가 낮으면서 둘 다 틀릴 수 있음. A/B "완전히 독립인 실패" 가정이 부분적으로만 성립 → §8 다음 실험(보조 confidence 추가)이 이 간극을 메우려는 시도.

**E-4. 학습 분포**

WFLW occlusion = 웹 수집(손·머리·선글라스, 컬러). in-cabin = 휠·안전벨트·선바이저 + IR 조명(그레이스케일·저대비) + 고정 카메라·극단 각도.
- **전이되는 것:** large-pose 결과(기하는 기하), 손 가림 일부.
- **전이 안 되는 것:** IR/조명 관련 subset, in-cabin 특유 occluder, make-up/blur subset(프로토콜 인공물에 가까움).

**판정** (실측 NME/구조적 근거, in-cabin 라벨 없이 하는 최선의 추정):

| subset | B NME | 전이 신뢰도 | 근거 |
|---|---|---|---|
| pose | 0.161 | **높음** | yaw는 순수 기하 문제(§E-2), 카메라 각도가 바뀌어도 3D 기하는 동일 |
| occlusion | 0.095 | **조건부** | 손 가림 일부는 유사 구조. 단 턱선점 취약(§4.2-1)은 occluder 형태(손 vs 휠 vs 안전벨트)에 따라 정도가 다를 수 있음 |
| illumination | 0.076 | **낮음** | 웹=RGB 저조도/역광, in-cabin=IR 그레이스케일, 센서 모달리티 자체가 다름 |
| blur | 0.087 | **낮음** | 웹 blur는 모션/포커스 블러, in-cabin 저해상도·고정 카메라 환경과 다를 수 있음 |
| make-up | 0.080 | **거의 무관** | in-cabin 대응 개념 자체가 희박(프로토콜 인공물) |
| expression | 0.087 | **중간** | 도메인 무관이나 샘플 수 적어(occlusion 1341/2500 대비 182/2500) 추정 신뢰도 자체가 낮음 |

**함의:** 웹 benchmark의 occlusion 강건성(NME 0.095)은 배포 강건성의 상한이지 보장은 아님. illumination/blur 결과는 그대로 신뢰하면 안 되고, IR 증강 또는 소량의 in-cabin 라벨로 재검증 필요.

---

## 7. 정직성

Claude(Desktop/Code)를 이 과제 전반에 적극 활용.

**LLM이 초안/구현한 것:**
1. 지표(NME 3종 정규화)·베이스라인(mean-shape)·모델(LandmarkNet)·yaw-PnP 파생·Part D decision 로직의 코드 골격
2. clean-architecture(infrastructure/application/shared)로의 리팩토링
3. 학습 스크립트의 MPS 가속·early-stopping·체크포인트 저장 추가
4. 로컬 Jupyter 커널이 계속 죽던 환경 문제(전역 IPython 시작 스크립트 충돌) 진단 및 격리
5. 이 문서의 각 Part A-E 결과 섹션에 실제 실행 결과 수치를 채워넣는 작업

**본인이 직접 한 것:**
1. 방향 지시와 매 단계 검토(리팩토링 범위/레이어 구성 확정, MPS로 학습하도록 지시, early stopping 도입 결정)
2. 학습 중단 사고 발생 시 상황 판단(재시작 지점, 손실 진행분 감수 여부)
3. WFLW 데이터셋 실제 다운로드·배치, 노트북 커널 오류 재현 확인
4. `src/settings.py`(Config 클래스 기반 설정) 직접 작성, `src/notebooks/part_a~e.ipynb`를 이 방식으로 직접 재작성
5. 최종적으로 이 설계 문서와 결과 수치가 실제 상황을 정확히 반영하는지 검토/승인
6. Part E의 해석과 판단(무엇을 신뢰할지, 트레이드오프를 어느 방향으로 밀지)은 LLM이 초안을 제시했고 본인이 결과 수치에 근거해 타당성을 확인

---

## 8. 다음 실험 (반증 가능한 가설)

**가설:** A–B disagreement에 B의 보조 히트맵 confidence를 더하면, 상관 실패(방향①)로 인한 false-accept가 유의미하게 감소함.

1. **필요 데이터:** WFLW test + occlusion subset(있음). 이상적으로 in-cabin IR 소량 검증셋.
2. **변경:** B에 경량 히트맵 보조 헤드 추가, 신호를 `α·disagreement + β·(1−peak_conf)`로 확장.
3. **평가:** 동일 coverage에서 accepted precision 비교, 특히 저-disagreement·고-오차 사분면의 질량 감소 측정.
4. **남는 위험:** 두 신호가 같은 원인(occlusion)에 동시 반응해 상관될 수 있음. 독립성 확인이 성패를 가름.
