# 📝 논문 회의록 (최종 수정본)

> **마지막 업데이트**: 2026-05-14 (선행연구 정독 결과 반영, **CTDR 포지셔닝 재조정 필요**)
> **상태**: EDA + 학습 데이터 매칭 + 외부 검증셋 처리 완료, **선행연구 정독 완료** — CTDR 신규성 재평가, 포지셔닝 합의 필요

---

## 1. 논문 개요

| 항목 | 내용 |
|---|---|
| **주제** | 이미지 인식을 통한 식품 품질 및 안전성 학습 모델 유효성 검증 |
| **세부 주제** | 라벨 분리 환경에서 멀티헤드 학습의 효과 검증 (토마토 성숙도/품질) |
| **투고 저널** | Journal of Food Engineering (SCIE Q1, IF 7.06) |
| **연구 형태** | 실증 연구 + 새 모듈 제안 |
| **Contribution 핵심** | **Disjoint-label Partial-MTL** 환경 전용 모듈(CTDR) 제안 |

---

## 2. 연구 범위

- 식품 품질 → 야채 → **토마토** (스마트팜 매칭)
- 공개 데이터셋 활용

---

## 3. 데이터셋

### 메인 학습 데이터: Sher-e-Bangla Tomato Dataset

| 항목 | 내용 |
|---|---|
| **DOI** | 10.17632/s42kpg8h37.1 |
| **링크** | https://data.mendeley.com/datasets/s42kpg8h37/1 |
| **라이선스** | CC BY 4.0 |
| **저자** | Khatun, Razzak, Islam, Uddin (2023) |
| **총 매수** | 12,986장 (.jpg) |

```
Sher-e-Bangla (12,986장)
├── Maturity (5,000)
│   ├── Original (1,000): Mature 500 + Immature 500
│   └── Augment (4,000): Mature 2,000 + Immature 2,000
└── Quality (7,986)
    ├── Original (1,986): Fresh 1,350 + Rotten 636
    └── Augment (6,000): Fresh 3,000 + Rotten 3,000
```

### 외부 검증 데이터: `valid/` 폴더 (Mendeley `x4s2jz55dx` 추정)

| 항목 | 내용 |
|---|---|
| **원본 파일** | 4,800장 |
| **고유 이미지 (hash dedup 후)** | 1,559장 |
| **라벨 view** | Three Classes (Reject/Ripe/Unripe) + Two Classes (Healthy/Reject) — 같은 이미지의 2가지 라벨링 view |
| **사용** | 외부 검증 (학습 데이터와 도메인 동일: close-up 단일 토마토) |

---

## 4. EDA 결과 (★ 핵심 발견)

### 4.1 학습 데이터 핵심 특성

1. **Maturity ↔ Quality 이미지 교집합 = 0**
   - 두 서브셋은 완전히 다른 이미지 집합 → **Partial MTL 환경**
2. **Augmented 파일명 정규식 100% 매칭**
   - `^python_original_(.+\.jpg)_[0-9a-f-]{36}\.jpg$`
3. **해상도**: W ∈ [647, 1280], H ∈ [957, 1024], 모두 RGB
4. **깨진 이미지: 0장** ✅
5. **클래스 분포**:
   - Maturity: 1:1 균형 ✅
   - Quality (Original): Fresh 1,350 : Rotten 636 = 2.12:1 ⚠️

### 4.2 RGB 분석 (★ 연구 motivation의 근거)

```
                R_mean  G_mean  B_mean
maturity/immature  120    124     84    ← 미숙(녹색)
maturity/mature    117    113     72    ← 성숙
                 차이 -3   -11   -12     ← 색상 차이 큼

quality/fresh      182    165    157    ← 신선
quality/rotten     179    163    154    ← 부패
                 차이 -3    -3    -3    ← 색상 차이 매우 작음 ⚠️
```

**결정적 인사이트**:
- **Maturity는 색상 기반** → 단순 CNN/CV 모델로 충분히 분리 가능
- **Quality는 텍스처 기반** → 색만으로 안 됨, 표면 특징 필수
- **두 태스크가 본질적으로 다른 feature를 요구** → naive 공유 백본이 한쪽으로 끌리기 쉬움 = **Negative transfer 가능성** = **CTDR 모듈의 motivation**

### 4.3 데이터 정합성 이슈
- Quality/Rotten Augment에서 추출된 원본명 14개가 Original에 없음 (논문 데이터 카드에 명시)

---

## 5. 학습 데이터 매칭 (★ Option X 적용 완료)

### 결정: 두 태스크 표본 수 + 구조 완전 매칭

```
Maturity:  500 orig + 2000 aug per class  (2,500 × 2 = 5,000)
Quality:   500 orig + 2000 aug per class  (2,500 × 2 = 5,000)
                                          ──────────
                                          Total      10,000
```

| 결정 사항 | 채택 |
|---|---|
| 학습 데이터 분할 (Train/Val/Test) | ❌ 안 함 — **모든 12,986장 학습용** |
| 검증/테스트 | **외부 데이터셋 사용** (`valid/`) |
| 클래스 가중치 | ❌ 불필요 — 매칭으로 자동 균형 |
| 매칭 방법 | Quality에서 다운샘플링 (1350→500 orig, 3000→2000 aug) |
| Random seed | 42 |

**산출물**: `data/processed/train.csv` (10,000 rows), `train_meta.json`

---

## 6. 외부 검증셋 처리 (★ 완료)

### 6.1 발견된 데이터 품질 이슈

| 폴더 | 파일 수 | 고유 이미지 | 메모 |
|---|---:|---:|---|
| Three/Ripe | 800 | 800 | ✅ 깨끗 |
| Three/Unripe | 800 | **93** | 🚨 707장이 중복 (93장 사진을 8.6배씩) |
| Three/Reject | 800 | 669 | ⚠️ 131장 중복 |
| Two/Healthy | 1,600 | 893 | (= Ripe 800 + Unripe 93) |
| Two/Reject | 800 | 669 | (= Three/Reject) |
| **전체** | **4,800** | **1,559** | 3,241장이 중복 |

### 6.2 라벨 매핑 결정

```
Maturity:
  Three/Ripe   → Mature
  Three/Unripe → Immature
  Three/Reject → SKIP (Reject는 품질 라벨)

Quality:
  Two/Healthy → Fresh
  Two/Reject  → Rotten
```

### 6.3 라벨 충돌 제외

같은 이미지가 (Three/Reject, Three/Ripe, Two/Healthy, Two/Reject) **4개 버킷 모두**에 등록된 케이스 **3장** 발견 → 자동 제외 처리

### 6.4 최종 외부 검증셋

| | 클래스 1 | 클래스 2 | 합계 | 비율 |
|---|---:|---:|---:|---:|
| **Maturity** | Mature 797 | Immature **93** | **890** | **8.6:1 ⚠️** |
| **Quality** | Fresh 890 | Rotten 666 | **1,556** | **1.3:1 ✅** |

**산출물**: `data/processed/valid_maturity.csv`, `valid_quality.csv`, `valid_meta.json`

### 6.5 ⚠️ Maturity Immature 93장 한계 대응

| 안 되는 것 | 대응 방법 |
|---|---|
| 증강으로 부풀리기 ❌ | 비독립 샘플 → reject 사유 |
| 다른 클래스를 93으로 다운샘플링 ❌ | 통계력 깎임 |
| 모순된 외부 도메인(크롭/스톡 이미지)로 보강 ❌ | Domain shift confound |

| ✅ 채택한 대응 |
|---|
| 1. **93장 그대로 사용** + 정직한 보고 |
| 2. **Bootstrap CI** (1,000 resample) → Immature 신뢰구간 명시 |
| 3. **Macro-F1 + Balanced Accuracy** 메인 metric |
| 4. **Per-class precision/recall/F1** 보고 |
| 5. **TTA (Test-Time Augmentation)**으로 prediction 보강 |
| 6. **Limitations section**에 명시 |

---

## 7. 연구 가설 (확정)

> **"라벨이 분리된 데이터셋 환경에서, 공유 백본 기반의 멀티헤드 학습이 두 개의 독립된 싱글헤드 모델보다 효과적인가? 그리고 명시적 feature disentanglement가 이를 더 개선하는가?"**

**영문**:
> "In disjoint-label dataset environments, is multi-head learning with a shared backbone more effective than two independent single-task models? Can explicit feature disentanglement further improve this?"

### 학술적 자리매김
- ✅ **Partial Multi-Task Learning**
- ✅ **Multi-Head Learning with Disjoint Labels**
- ✅ **Novel module for orthogonal-task MTL** (Ours)

---

## 8. ⭐ 새 모듈: CTDR (Cross-Task Disentangled Routing)

### 8.1 풀고자 하는 문제 (positioning)

```
일반 MTL:                           우리 (Partial MTL):
같은 이미지에 두 라벨               다른 이미지가 다른 라벨
─────────────────────              ─────────────────────
img1 → (mature, fresh)              img1 → mature   (quality 라벨 없음)
img2 → (immature, rotten)           img2 → fresh    (maturity 라벨 없음)

모든 배치가 두 태스크에 기여         배치마다 한 태스크만 업데이트
                                    → 공유 백본이 번갈아 가며 한쪽으로만 끌림
                                    → Negative transfer 위험 ⬆⬆
```

→ **기존 MTL 방법(Cross-Stitch, MMoE 등)이 가정하지 않는 상황**. 여기가 우리 contribution의 자리.

### 8.2 CTDR 아키텍처

```
이미지
  ↓
[Pretrained Backbone] (ImageNet 가중치 사용)
  ↓
Feature f ∈ R^D
  ↓
┌──────────────────────────────────────────────────┐
│            CTDR Module (NEW, Ours)               │
│                                                  │
│   f_shared = W_s(f)      ← 두 태스크 공통 feature │
│   f_mat    = W_m(f)      ← Maturity 전용         │
│   f_qual   = W_q(f)      ← Quality 전용          │
│                                                  │
│   g_mat  = σ(W_gm(f)) ∈ [0,1]^D   ← 학습 가능 게이트│
│   g_qual = σ(W_gq(f)) ∈ [0,1]^D                  │
│                                                  │
│   z_mat  = g_mat ⊙ f_shared + (1−g_mat) ⊙ f_mat   │
│   z_qual = g_qual⊙ f_shared + (1−g_qual)⊙ f_qual  │
└──────────────────────────────────────────────────┘
  ↓                          ↓
[Head_mat]               [Head_qual]
  ↓                          ↓
Mature/Immature          Fresh/Rotten
```

### 8.3 손실 함수 (3개 합성)

```
L_total = L_cls + λ_orth · L_orth + λ_gate · L_gate

L_cls    = Σ Masked-CE per sample       (라벨 있는 태스크만 BP)
L_orth   = | cos(f_mat, f_qual) |²       (태스크별 feature 직교화)
L_gate   = −H(g_mat) − H(g_qual)          (게이트 결정적으로 갈리도록)
```

### 8.4 기존 방법 대비 차별점 (초기 분석 — 8.6에서 재평가됨)

| 기존 방법 | 우리 CTDR |
|---|---|
| Cross-Stitch (CVPR 2016) | feature 선형결합만 |
| MMoE (KDD 2018) | 다중 expert + gating |
| Sluice Networks (AAAI 2017) | task feature 전체 텐서 결합 |
| **CTDR (Ours)** | **2-branch (shared + task-specific) + 게이트 + 직교 손실** + **disjoint-label 전제** |

→ "**Disjoint-label partial-MTL용 모듈**" 포지셔닝이 under-explored gap (초기 가정)

### 8.5 핵심 코딩 사항 (교수님 요청 반영)

| 항목 | 채택 |
|---|---|
| Pretrained 사용 | ✅ ImageNet 가중치 로드 |
| nn.Module로 직접 구현 | ✅ timm 래퍼 X, 직접 작성 |
| 백본 자체 from-scratch | ❌ (성능 보장 X, JFE 기준 부적합) |
| 새 모듈 코드 자작 | ✅ CTDR은 처음부터 구현 |

---

### 8.6 ⚠️ 선행연구 정독 후 신규성 재평가 (2026-05-14 추가)

**솔직한 결론**: CTDR은 "신규 발명"이 아니라 **기존 3~4개 요소의 재조합**임이 확인됨.

#### 정독한 핵심 prior

1. **Sluice Networks** (Ruder+, AAAI 2019) — **가장 위험한 prior**
   - ✅ Disjoint labels + weighted loss 합산 (우리와 동일)
   - ✅ Shared / private subspace 분리
   - ✅ Orthogonality penalty (Frobenius)
   - ❌ Sigmoid 게이트는 없음 (linear α, β coefficient)
   - ❌ Gate entropy 항 없음
   - → **CTDR 골격의 80%가 이미 여기 있음**

2. **OrthTD** (clinical, arXiv 2605.03570, 2025)
   - ✅ Shared + task-specific projection
   - ✅ Cosine orthogonality (shared-vs-task)
   - ❌ 게이트 없음, joint labels
   - → "Cos orthogonality + shared/specific" 패턴 동일

3. **Task-Specific Normalization (σBN)** (arXiv 2512.20420, 2025)
   - ✅ Sigmoid 게이트 per task (γ_t)
   - ❌ BN scale로 작용, 별도 projection 없음
   - → Sigmoid 게이트 메커니즘 자체는 이미 존재

4. **MMoE / TRL / Routing Networks**
   - 게이트 / 라우터 계열의 다양한 변형 다수 존재

5. **KAIS'25 FDSR** (Liu+, Knowledge and Information Systems 2025)
   - Disentanglement → Selection → Reaggregation 큰 흐름이 CTDR과 매우 유사 (NLP 도메인)

#### 진짜 신규로 주장 가능한 요소 (제한적)

| 요소 | 명확한 Prior 존재? | 비고 |
|---|---|---|
| Shared + task-specific 분리 | ❌ | Sluice, Cross-Stitch, OrthTD 선점 |
| Sigmoid 게이트 | ❌ | MMoE, σBN, Routing Net 선점 |
| Orthogonality 손실 | ❌ | Sluice, OrthTD 선점 |
| **−H(g) 음의 엔트로피 게이트 정규화** | ✅ **prior 미발견** | Routing Networks의 load-balance entropy와 부호/의미 다름 |
| **Task-vs-task cosine²** (위치 차이) | 🟡 약한 차별화 | OrthTD는 shared-vs-task |
| **3-loss 동시 사용 + disjoint + 농식품** | ✅ 응용 차원 신규 | 정확한 조합 + 도메인 미보고 |

### 8.7 ★ 정직한 포지셔닝 (수정)

#### ❌ 안 됨 (이대로 가면 reject)
> "We propose a **novel CTDR module** for multi-task tomato classification..."

#### ✅ 정직한 프레이밍 (수정안)
> "We **adapt and integrate** disentangled routing principles from prior MTL works (Sluice [Ruder+19], OrthTD [2025], σBN [2025]) **for the under-explored partial-MTL setting with disjoint single-task labels** in agricultural quality inspection. We further propose a **decisive gate entropy regularizer** that pushes routing gates toward binary values, and provide systematic ablation across loss terms."

#### Contribution으로 주장할 수 있는 것 (4가지 합산)

1. **응용 신규성**: 농식품 분야 disjoint partial-MTL 첫 체계적 적용
2. **메소드 minor 신규성**: −H(g) gate entropy regularizer (결정적 라우팅)
3. **체계적 비교 분석**: Sluice / OrthTD / σBN / MMoE / Naive multi-head vs Ours
4. **외부 검증 + Ablation 깊이**: 외부 데이터셋 + bootstrap CI + representation analysis

→ 1+2+3+4 합치면 Q1 수준. 단독으로는 부족.

### 8.8 ★ 논문 Related Work에 들어갈 비교 표

| 방법 | Disjoint labels | Shared+Specific | Gate | Orth Loss | Gate Entropy | Domain |
|---|---|---|---|---|---|---|
| Cross-Stitch (2016) | Joint | ✅ | Linear | ❌ | ❌ | Vision |
| Sluice (2019) | **Disjoint** | ✅ | Linear α,β | ✅ Frobenius | ❌ | NLP |
| MMoE (2018) | Joint | Experts | Softmax | ❌ | ❌ | RecSys |
| TRL (2019) | Joint | Channel mask | Binary fixed | ❌ | ❌ | Vision |
| OrthTD (2025) | Joint | ✅ | ❌ | ✅ cos shared-task | ❌ | Clinical |
| σBN (2025) | Joint | Fully shared | Sigmoid γ | ❌ | ❌ | Dense pred |
| KAIS'25 FDSR | Joint | ✅ | Selection | IB loss | ❌ | NLP |
| **Ours (adapted)** | **Disjoint** | **✅** | **Sigmoid D-dim** | **✅ cos task-task** | **✅ −H(g)** | **Agri-food** |

이 표가 motivation section + related work의 핵심.

### 8.9 Q1 통과 가능성 평가

| 시나리오 | 가능성 | 작업량 |
|---|---|---|
| 옵션 1: 현재 plan + 정직한 포지셔닝 | 🟡 Q1 도전 가능, acceptance 불확실 | 보통 |
| 옵션 2: 이론적 분석 추가 (−H(g) 효과 등) | 🟢 Q1 가능성 ↑ | +2~4주 |
| 옵션 3: Venue 재조정 (응용 저널) | 🟢 통과 빠름 | 보통 |

#### Venue 후보 (백업 plan)
- *Computers and Electronics in Agriculture* (Q1, IF 6.5, 응용 친화)
- *Postharvest Biology and Technology* (Q1)
- *Sensors* (Q2)
- *Agronomy* (Q2)

**현재 추천 전략**: **JFE 먼저 시도 (옵션 1) + 응용 저널 백업 plan 준비 (옵션 3)**

---

## 9. 실험 설계 (확정)

### 9.1 비교 모델 (6종)

| # | 모델 | 분류 | 데이터 |
|---|---|---|---|
| ① | CV Baseline (HOG+SVM 등) | 비교군 | 각 태스크 |
| ② | 싱글태스크 - Maturity (Backbone) | 싱글 | Maturity만 |
| ③ | 싱글태스크 - Quality (Backbone) | 싱글 | Quality만 |
| ④ | Joint Single-head (4-class) | 중간 | 통합 |
| ⑤ | Naive Multi-head | 멀티 baseline | 통합 (마스킹) |
| ⑥ | **CTDR (Ours)** ⭐ | 멀티 +α | 통합 (마스킹) |

### 9.2 추가 비교 baseline (Ablation용)

| # | 모델 | 목적 |
|---|---|---|
| ⑦ | Cross-Stitch | 기존 MTL과 비교 |
| ⑧ | MMoE | 기존 MTL과 비교 |

### 9.3 4개 핵심 RQ

```
RQ1: Disjoint-label MTL에서 naive 멀티헤드는 negative transfer를 겪는가?
     → ② 싱글 vs ⑤ Naive 멀티헤드 비교

RQ2: CTDR이 이를 해결하는가?
     → ⑤ Naive 멀티헤드 vs ⑥ CTDR (Ours)

RQ3: CTDR의 각 component가 기여하는가?
     → Ablation (8.6에 상세)

RQ4: 게이트가 의미 있는 분리를 학습하는가?
     → 게이트 시각화 + t-SNE + CKA 분석
```

### 9.4 평가 방식

```
[Maturity 평가] — valid_maturity.csv (890장)
  ①, ②, ④, ⑤, ⑥의 maturity 출력 비교
  + bootstrap CI for Immature (n=93)

[Quality 평가] — valid_quality.csv (1,556장)
  ①, ③, ④, ⑤, ⑥의 quality 출력 비교

❌ Maturity vs Quality 직접 비교 안 함 (다른 평가셋)
```

---

## 10. 통제 변수

| 변수 | 모든 모델 공통 | 통제 |
|---|---|---|
| 데이터셋 출처 | Sher-e-Bangla | ✅ 동일 |
| 백본 아키텍처 | 메인 1개 (TBD: ResNet50 / EfficientNet-B3 / ViT-Base) | ✅ 동일 |
| 입력 크기 | 224×224 | ✅ 동일 |
| Optimizer | Adam | ✅ 동일 |
| Learning rate | 1e-4 | ✅ 동일 |
| Batch size | 32 | ✅ 동일 |
| Seed | 42 (메인) + 5개 추가 (안정성) | ✅ 동일 |
| Pretrained weights | ImageNet | ✅ 동일 |

---

## 11. Ablation Study

### A1. 데이터 양 통제 ⭐⭐⭐ 필수
- 싱글 모델에도 전체 데이터 노출 (자기 태스크 라벨만 사용)
- → "구조 효과만" 분리 검증

### A2. 백본 아키텍처 비교 ⭐⭐
- ResNet50 / EfficientNet-B3 / ViT-Base
- 결과의 일반성

### A3. 손실 가중치 ⭐⭐
- (λ_orth, λ_gate) 그리드 서치
- 0, 0.1, 1.0, 10.0

### A4. 다중 시드 안정성 ⭐⭐ 필수
- 5개 시드 → 평균 ± 표준편차

### A5. **CTDR Component Ablation** ⭐⭐⭐ 필수 (새 모듈 정당화)
- w/o L_orth (직교 손실 제거)
- w/o L_gate (게이트 entropy 제거)
- w/o gating (단순 concat)
- w/o task-specific projection (shared만)

### A6. 기존 MTL 방법과 비교
- vs Cross-Stitch
- vs MMoE

### A7. 캘리브레이션 (선택)
- ECE 비교

---

## 12. 평가 지표

### 12.1 정량
- **Accuracy** (참고용)
- **Balanced Accuracy** (메인, 불균형 대응)
- **Macro-F1** (메인)
- **Per-class Precision / Recall / F1**
- **ROC-AUC**
- **Confusion Matrix**

### 12.2 통계적 신뢰도
- **Bootstrap CI (n=1000)** — 특히 Maturity Immature
- **다중 시드 평균 ± 표준편차**

### 12.3 효율성
- 모델 파라미터 수
- FLOPs
- 추론 시간

### 12.4 해석 가능성 (★ JFE Q1 통과를 위한 필수)
- **Grad-CAM** (모델이 어디를 보는지)
- **t-SNE / UMAP** (feature 공간 시각화)
- **CKA (Centered Kernel Alignment)** (레이어별 representation 유사도)
- **Gate 값 시각화** (CTDR 게이트 분포)

---

## 13. 솔직한 한계 (논문에 명시)

### 13.1 데이터 한계
- 단일 학습 데이터셋(Sher-e-Bangla)
- 외부 검증의 Maturity Immature 클래스 93장 (통계력 제한)

### 13.2 방법론 한계
- 같은 이미지에 두 라벨이 없음 → 진정한 MTL 평가 불가 (이건 contribution으로 전환됨)
- 데이터 양 비대칭 (Ablation A1으로 보완)

### 13.3 보완 전략
- ✅ Option X 매칭으로 표본 수 균형
- ✅ Ablation Study로 효과 분리
- ✅ 다중 시드로 안정성 검증
- ✅ Bootstrap CI로 신뢰구간 명시
- ✅ 외부 검증으로 일반화 검증

---

## 14. 환경 셋업 (완료)

### 14.1 하드웨어
- **GPU**: NVIDIA A40 (48GB)
- **CUDA**: 12.6

### 14.2 소프트웨어
- Python 3.12.13 (conda env: `tomato`)
- PyTorch 2.6.0+cu124
- 주요 라이브러리: torchvision, timm, albumentations, sklearn, wandb

### 14.3 프로젝트 구조
```
C:\Users\smin\IdeaProjects\tomato\
├── .gitignore
├── requirements.txt              (UTF-8)
├── tomato.iml
├── meeting_log.md                ← 이 파일
├── data/
│   ├── raw/                      ← Sher-e-Bangla 12,986장
│   │   ├── maturity/{original,augment}/{immature,mature}/
│   │   └── quality/{original,augment}/{fresh,rotten}/
│   └── processed/
│       ├── inventory.csv         ← 학습 전체 인벤토리
│       ├── inventory_original_with_meta.csv
│       ├── train.csv             ← Option X 매칭 (10,000장)
│       ├── train_meta.json
│       ├── valid_maturity.csv    ← 외부 검증 (890장)
│       ├── valid_quality.csv     ← 외부 검증 (1,556장)
│       └── valid_meta.json
├── valid/                        ← 외부 검증 원본 폴더
├── notebooks/
│   ├── 01_data_exploration.ipynb         ← 완료 ✅
│   ├── 02_training_prep.ipynb            ← 완료 ✅
│   └── 03_external_valid_eda.ipynb       ← 완료 ✅
├── src/
│   ├── data/
│   └── models/
├── results/
│   ├── figures/                  ← 9개 PNG 생성됨
│   └── tables/                   ← 3개 CSV 생성됨
└── docs/
    └── data_card.md              ← 자동 생성됨
```

---

## 15. 진행 상태

### ✅ 완료
- [x] 데이터셋 선정 (Sher-e-Bangla)
- [x] 가상환경 + 라이브러리 설치 (conda env `tomato`)
- [x] 폴더 구조 정리 + data/raw로 표준화
- [x] requirements.txt UTF-8 변환
- [x] EDA (`01_data_exploration.ipynb`)
  - 클래스 분포, 해상도, RGB 분석, 샘플 시각화, 깨진 파일 검사
  - 데이터 카드 자동 생성
- [x] 학습 데이터 매칭 (Option X, `02_training_prep.ipynb`)
  - 10,000장 train.csv 생성
  - 원본:증강 = 20:80 균형 시각화
- [x] 외부 검증셋 처리 (`03_external_valid_eda.ipynb`)
  - 4,800 → 1,559 hash dedup
  - 라벨 충돌 3장 자동 제외
  - 2-class 매핑 + CSV 생성
- [x] 가설 정확하게 수정 + 영문화
- [x] **CTDR 모듈 설계 결정** ⭐ (초기안)
- [x] 비교 모델 6종 + Ablation 항목 확정
- [x] **선행연구 정독** ⭐ — Sluice/OrthTD/σBN/MMoE/TRL/KAIS'25 FDSR 등 10편
- [x] **CTDR 신규성 재평가** — 기존 요소 재조합임 확인, 정직한 포지셔닝 합의
- [x] Related Work 비교 표 작성 (§8.8)
- [x] 모델 아키텍처 설명서 작성 (`docs/model_architectures.md`)

### 🎯 진행 중
- [ ] **Contribution 포지셔닝 최종 결정** (옵션 1/2/3 중)
- [ ] 모듈 이름 재결정 (CTDR 유지 vs 더 솔직한 이름)
- [ ] 백본 결정 (ResNet50 / EfficientNet-B3 / ViT-Base 중 1개)
- [ ] 모듈 코드 구현 (nn.Module 직접 작성)

### 📅 다음 단계 (예정)
- [ ] PyTorch Dataset 클래스 (masked label 지원)
- [ ] CV Baseline 구현 (①)
- [ ] 싱글 모델 학습 (②, ③)
- [ ] Joint Single-head 학습 (④)
- [ ] Naive Multi-head 학습 (⑤)
- [ ] **CTDR 학습 (⑥) ⭐ Ours**
- [ ] Ablation Study (A1~A6)
- [ ] 외부 평가 (Maturity + Quality)
- [ ] Representation 분석 (t-SNE, CKA, Grad-CAM, 게이트 시각화)
- [ ] 결과 분석 및 시각화
- [ ] 논문 초고 작성

---

## 16. 결정 필요한 사항

### 16.1 ⏳ 최우선 — Contribution 포지셔닝 (지도교수 합의 필요)

선행연구 정독 결과 CTDR이 기존 요소 재조합임이 확인됨. 진행 방향 합의 필요:

- **옵션 1: 현재 plan + 정직한 포지셔닝** (현재 추천)
  - "Adapted and integrated" 언어 사용, "novel module" 주장 X
  - −H(g) gate entropy + 농식품 응용 + 체계적 ablation으로 contribution 정당화
  - JFE Q1 도전 가능, 통과 보장 X (응용 저널 백업)

- **옵션 2: Contribution 강화** (작업 +2~4주)
  - −H(g)의 이론적 분석 추가
  - 다른 작물 데이터셋 확장 (사과, 고추 등)
  - 새 ablation/평가 프로토콜 제안
  - JFE Q1 통과 가능성 ↑

- **옵션 3: Venue 재조정**
  - *Computers and Electronics in Agriculture* (Q1, IF 6.5, 응용 친화)
  - *Postharvest Biology and Technology* (Q1)
  - *Sensors* / *Agronomy* (Q2)
  - 통과 빠름, 응용 contribution이 더 자연스러움

### 16.2 ⏳ 즉시 결정 필요 (옵션 결정 후)

- **모듈 이름**:
  - [ ] CTDR (Cross-Task Disentangled Routing) — 옵션 1·2에서 유지
  - [ ] 더 솔직한 이름 (예: ADRR — Adapted Disentangled Routing & Regularization)
  - [ ] 다른 이름: ___________

- **백본 선택**:
  - [ ] ResNet50 (안전, 표준)
  - [ ] EfficientNet-B3 (효율 + 최신)
  - [ ] ViT-Base (transformer)
  - → Ablation A2에서 비교하되, **메인은 하나 선정**

- **구현 시작 순서**:
  - [ ] (a) Baseline → Naive 멀티 → CTDR (motivation 입증 강함)
  - [ ] (b) CTDR 먼저 + baseline 채워가기

### 16.3 ✅ 해결 완료
- ~~데이터 분할 비율~~ → **외부 검증 사용으로 결정 (분할 안 함)**
- ~~클래스 불균형 처리~~ → **Option X 매칭으로 균형 (가중치 불필요)**
- ~~Pretrained 사용 여부~~ → **ImageNet 가중치 사용 + 모듈 코드 자작**
- ~~"새 모듈" novelty 주장~~ → **선행연구 정독 결과 "Adapted + integrated"로 재포지셔닝**

---

## 17. 참고 문헌

### 17.1 데이터셋
1. **Khatun, T., Razzak, A., Islam, M. S., & Uddin, M. S. (2023).**
   *Tomato Maturity Detection and Quality Grading Dataset.*
   Mendeley Data, V1. https://doi.org/10.17632/s42kpg8h37.1

2. **(외부 검증)** Mendeley `x4s2jz55dx` 추정 — Two/Three Classes view 토마토 데이터셋

### 17.2 비교 대상 / 관련 연구 (★ Related Work 핵심 인용 — 반드시 차별화)

3. **Khan, A., Hassan, T., Shafay, M., et al. (2023).**
   *Tomato maturity recognition with convolutional transformers.*
   Scientific Reports, 13(1), 22885.
   → 커스텀 아키텍처 사례 (참고)

#### MTL 메소드 (CTDR의 직접적 prior — 반드시 인용 + 차별화)
4. **Misra, I., et al. (2016).** *Cross-Stitch Networks for Multi-Task Learning.* CVPR.
5. **Rosenbaum, C., et al. (2018).** *Routing Networks: Adaptive Selection of Non-Linear Functions for Multi-Task Learning.* ICLR.
6. **Ma, J., et al. (2018).** *Modeling Task Relationships in Multi-Task Learning with Multi-gate Mixture-of-Experts.* KDD.
7. **Ruder, S., et al. (2019).** *Sluice Networks: Learning what to share between loosely related tasks.* AAAI. ⭐ **가장 가까운 prior**
8. **Strezoski, G., et al. (2019).** *Many Task Learning With Task Routing.* ICCV.

#### 최신 disentanglement / orthogonality MTL (★ 반드시 인용)
9. **OrthTD (2025).** *Disentangling Shared and Task-Specific Representations from Multi-Modal Clinical Data.* arXiv:2605.03570. ⭐ Cos orthogonality 패턴 동일
10. **Task-Specific Normalization (σBN, 2025).** *Simplifying Multi-Task Architectures Through Task-Specific Normalization.* arXiv:2512.20420. ⭐ Sigmoid gate per task
11. **Ortho-LoRA (2026).** *Disentangling Task Conflicts in Multi-Task LoRA via Orthogonal Gradient Projection.* arXiv:2601.09684.
12. **Liu+ (KAIS 2025).** *Feature disentanglement, selection, and reaggregation method for multi-task learning.* Knowledge and Information Systems.
13. **Lippl+ (2024).** *Disentangling Representations through Multi-task Learning.* arXiv:2407.11249.

### 17.3 백본 후보
- He, K., et al. (2016). *Deep Residual Learning.* CVPR. (ResNet)
- Tan, M., & Le, Q. V. (2019). *EfficientNet.* ICML.
- Dosovitskiy, A., et al. (2021). *An Image is Worth 16x16 Words.* ICLR. (ViT)

---

## 18. 회의 기록

### 회의 1 (초기): 주제/범위
- 식품 품질·안전성 → 야채 → 토마토 한정
- 공개 데이터셋 활용 결정

### 회의 2: 데이터셋 검토
- 여러 데이터셋 후보 비교
- Sher-e-Bangla 단독 사용 결정

### 회의 3: 가설 수정
- "MTL이 좋다" → "Partial MTL 환경에서 멀티헤드가 효과적"
- 학술적 정확성 확보

### 회의 4: EDA 결과 검토 (2026-05-14 진행)
- ✅ 클래스 분포 / RGB 분석 / 깨진 파일 검사 완료
- ✅ Maturity는 색상, Quality는 텍스처 의존 — **CTDR motivation 확보**
- 결정: 데이터 분할 없이 전체 학습용 사용 + 외부 검증

### 회의 5: 학습 데이터 & 외부 검증셋 처리 (2026-05-14)
- ✅ Option X 매칭 (500 orig + 2000 aug per class, 총 10,000장)
- ✅ 외부 검증셋 hash dedup + 라벨 충돌 제외 + 2-class 매핑
- 결정: Maturity Immature 93장 한계 → bootstrap CI + macro 지표로 대응

### 회의 6: 새 모듈 설계 (2026-05-14) ⭐
- **JFE Q1 (IF 7.06) 통과를 위해 새 모듈 필요** 합의
- ✅ **CTDR (Cross-Task Disentangled Routing)** 설계 확정 (초기안)
  - Shared/task-specific 분기 + 학습 가능 게이트 + 직교 손실
  - Disjoint-label partial-MTL 전제 (under-explored gap으로 가정)
- ✅ Pretrained 백본 사용 + 모듈은 nn.Module로 직접 구현 (교수님 요청 반영)
- ✅ 4개 RQ + 6개 비교 모델 + 7개 Ablation 항목 확정
- 📄 `docs/model_architectures.md` 작성 (5개 모델 상세 설명서)

### 회의 7: 선행연구 정독 및 신규성 재평가 (2026-05-14) ⭐⭐
- **CTDR이 정말 신규인지 검증 필요** — 정독 진행
- **정독 논문 (10편)**:
  - Sluice Networks (AAAI 2019), Cross-Stitch (CVPR 2016), MMoE (KDD 2018)
  - Task Routing TRL (ICCV 2019), Routing Networks (ICLR 2018)
  - OrthTD (2025), σBN (2025), Ortho-LoRA (2026)
  - KAIS'25 FDSR, Disentangling via MTL (Lippl 2024)
- **결론**: CTDR은 **기존 3~4개 요소의 재조합**
  - Shared+specific 분리: Sluice, Cross-Stitch, OrthTD 선점
  - Sigmoid 게이트: MMoE, σBN 선점
  - Orthogonality 손실: Sluice (Frobenius), OrthTD (cos)
  - **−H(g) gate entropy 정규화만 prior 미발견 (minor novelty)**
- **포지셔닝 재조정 합의**:
  - ❌ "Novel CTDR module" → ✅ "Adapt and integrate ... for partial-MTL"
  - Sluice/OrthTD/σBN/MMoE 등 prior 명시적 인용 + 차별화 필수
  - Contribution: 응용 신규성 + minor methodological novelty + 깊은 ablation + 외부 검증
- **다음 회의에서 결정할 것**:
  - Contribution 강화 옵션 (옵션 1/2/3) 중 선택
  - 모듈 이름 (CTDR 유지 vs 변경)
  - Venue 백업 plan 확정

### 회의 8 (예정): 포지셔닝 최종 결정 + 코드 작성 시작
- Contribution 옵션 1/2/3 중 선택
- 백본 1개 선정 (ResNet50 / EfficientNet-B3 / ViT-Base)
- 모듈 이름 확정
- Dataset 클래스 + 모듈 구현 시작

---

> **Note**: 이 회의록은 살아있는 문서입니다. 진행하면서 계속 업데이트됩니다.
