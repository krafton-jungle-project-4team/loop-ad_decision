# 프로모션 기반 동적 추천과 Uplift-ready 아키텍처

## 구현 경계

### 현재 실제 추천에 사용하는 것

- 프로모션의 목적지, 시즌, 혜택을 정규화한 `promotion_audience_ast.v1`
- AST를 기존 Segment Audience V2 실행 계약으로 결정적으로 컴파일한 조건
- 프로모션 적합도, 행동 기반 예상 예약 가능성, 표본 신뢰도, 후보 차별성
- 후보별 source audience snapshot과 여러 후보 확정 후의 final audience snapshot

카드의 예약 지표는 **행동 기반 예상 예약 전환율**이다. 과거 행동을 바탕으로
추정한 향후 예약 가능성이며 광고로 인한 증가율은 아니다.

### 구현하지만 추천 serving에는 사용하지 않는 것

- treatment와 control을 모두 보존하는 `ad_experiment_units`
- treatment만 광고 발송 대상으로 저장하는 `user_segment_assignments`
- 배정 이전 feature generation과 immutable audience snapshot
- 프로모션별 outcome spec과 outcome window 결과 수집
- Uplift 학습 데이터셋, signed CATE 모델, 합성 및 외부 pipeline 검증
- 모델 lifecycle과 activation guard

### 아직 검증하지 않은 것

- LoopAd 운영 데이터로 추정한 광고 증분 효과
- 실제 숙박 광고의 고객별 CATE
- Uplift 모델을 추천 순위에 사용했을 때의 서비스 성과
- 종료 실험을 탐색하고 학습하는 scheduler와 worker
- persistent model registry, 승인 기록, validated -> active 전환 API

현재 LoopAd 모델 lifecycle은 `collecting_data`, `serving_eligible=false`다.
현재 코드에서는 process 안에서 만든 metadata로 Uplift serving을 활성화할 수 없다.

## 동적 추천 계약

```text
프로모션 자연어
  -> promotion_audience_ast.v1
  -> 등록 template 또는 custom_structured_condition v1/v2
  -> Segment Audience V2 validator/compiler/binder
  -> source audience snapshot
```

AST는 후보 조합, identity, 표시 문구를 위한 내부 표현이다. 실제 membership의
source of truth는 compiled `segment_audience_spec`과 snapshot이다. 새로운 공개
custom structured V3는 만들지 않는다.

- `any_of(jeju, okinawa)`는 등록 template의 정렬·중복 제거된
  `destination_ids=["jeju", "okinawa"]`로 컴파일한다.
- `contains_all(jeju, okinawa)`는 custom structured v1의 두 검색 조건을 동일
  사용자에게 AND로 적용한다.
- source audience에 조건을 더할 때만 custom structured v2를 사용한다.
- 정확히 실행할 수 없는 문구는 `creative_only` 또는
  `unsupported_conditions`로 분리한다.

동적 세그먼트 identity는 프로모션 ID, canonical AST, evaluation window,
compiler version, audience contract version의 SHA-256이다. 목적지 배열과
교환 가능한 조건은 정렬한다. 현재 사용자 목록, 추천 순서, 카드 문구는 identity에
영향을 주지 않는다.

## Snapshot 역할

`audience_snapshot_id`는 조건을 만족한 전체 사용자 집합을 재현한다.
추천 단계에서는 카드 수와 source snapshot member 수가 같다. 여러 후보를 확정한
뒤에는 중복을 제거한 final snapshot member 수가 최종 실험 모집단이 된다.

`vector_generation_id`는 배정 전 특징 `X`를 재현한다. 배정 후 이벤트가 특징에
포함되지 않도록 다음 관계를 검증한다.

```text
generation.window_end <= assigned_at
generation.source_revision_cutoff <= assigned_at
audience_snapshot.source_cutoff <= assigned_at
segment_assignment_execution.source_cutoff_at <= assigned_at
outcome_window_start = assigned_at
outcome_window_start < outcome_window_end
```

## Assignment와 모집단

`segment_assignment_executions.input_manifest_json`이 요청과 experiment design의
source of truth다. `ad_experiment_units`는 해당 설계로 정해진 사용자별 arm,
`user_segment_assignments`는 treatment serving subset이다.

```text
all_treatment:
final snapshot = experiment units = treatment units = serving assignments
control units = 0

randomized_holdout:
final snapshot = experiment units
treatment units = serving assignments
control units = final snapshot - treatment units
control serving assignments = 0
```

같은 promotion run은 여러 execution provenance를 가질 수 있지만 모든
Uplift-ready execution은 하나의 experiment design fingerprint를 공유한다.
다른 비율, outcome window, outcome spec 또는 randomization version으로 재시도하면
`409 experiment_design_conflict`다.

`randomized_holdout`은 명시적 opt-in이다. 각 experiment에서 HMAC-SHA-256 정렬로
결정적 complete randomization을 수행한다.

```text
raw_k = floor(N * requested_treatment_ratio + 0.5)
treatment_count = min(max(raw_k, 1), N - 1)
actual_treatment_ratio = treatment_count / N
```

`N < 2`면 `422 randomized_holdout_audience_too_small`로 전체 요청을 거절한다.
`SEGMENT_HOLDOUT_RANDOMIZATION_SALT`는 randomized 요청에만 필요하며, 없으면 쓰기
전에 `503 randomized_holdout_configuration_unavailable`을 반환한다. 기본
`all_treatment`는 salt 없이 기존과 동일하게 동작한다.

Execution은 `preparing`으로 생성한다. Experiment units와 treatment serving
assignments를 저장한 뒤 `finalize_uplift_assignment_execution()`이 전체 모집단 관계와
manifest quota를 집합 단위로 한 번만 검증하고 `finalized`로 봉인한다. 이 과정은
하나의 PostgreSQL 트랜잭션이며, 학습 데이터는 finalized execution만 읽는다.
봉인된 execution 설계와 experiment unit은 수정하거나 삭제할 수 없다.

## Immutable outcome

Promotion run 생성 시 canonical outcome spec과 hash를 `goal_snapshot_json`에
봉인한다. Uplift v1은 다음 결과만 학습한다.

```text
outcome_metric = booking_conversion_rate
outcome_event_name = booking_complete
assigned_at <= event_time < outcome_window_end
promotion destination이 있으면 destination_id, destination_name,
hotel_city, hotel_country 중 하나의 canonical 목적지가 그 집합에 포함됨
```

제주·오키나와 프로모션에서 서울 예약은 `Y=1`이 아니다. Control의 자연 예약도
관찰해야 하므로 event에 promotion ID가 있는지는 요구하지 않는다. 지원하지 않는
goal metric의 holdout은 저장할 수 있지만 학습에서는 사유와 함께 제외한다.
목적지 필드는 우선순위 `coalesce`가 아니라 각각 정규화한 OR 계약이다. 숫자 ID가
함께 있어도 `destination_name=제주`처럼 지원 필드 하나가 일치하면 제주 예약으로
인정한다. Outcome 조회는 user ID를 배치로 나눠 수행한다.

Data Contract trigger는 run 생성 후 `outcome_spec`, `outcome_spec_hash`,
`outcome_definition_version` 변경을 거절한다. 프로모션 설명이 바뀌어도 이미 시작한
run의 성공 정의는 바뀌지 않는다.

## Uplift 학습 계약

```text
X = vector_generation_id가 가리키는 배정 이전 64차원 행동 특징
T = ad_experiment_units.arm의 ITT treatment 여부
Y = 종료된 window에서 frozen outcome_spec과 일치한 예약 여부
```

실제 발송·노출은 compliance 진단에 사용하며 `T`를 다시 정의하지 않는다. Window
미종료, 특징 누락, 시간 위반, spec hash 불일치, legacy execution, 미지원 metric은
학습에서 제외하고 사유별 수를 남긴다. 누락 특징은 임의 보간하지 않는다.

현재 오프라인 PoC 모델은 T-learner가 아니라 unit별 무작위 배정 확률을 반영한
`transformed-outcome-ridge.v1` Transformed Outcome Regression이다. ATE,
Qini, AUUC, uplift@top-k도 각 unit의 treatment probability를 사용하는 IPW로
평가한다.

호환되는 ACTIVE 모델이 생긴 이후에만 다음 signed 지표를 추천 순위에 사용할 수
있다.

```text
mean_cate = 후보 내 모든 signed CATE 평균
expected_incremental_bookings = 후보 내 모든 signed CATE 합
negative_cate_user_ratio = CATE < 0인 사용자 비율
```

음수 CATE는 제거하거나 0으로 보정하지 않는다. 현재 예측 score를 experiment 단위로
재표집한 값은 정식 CATE confidence interval이 아니라
`predicted_cate_cluster_variability_interval`이며 항상 `reference_only=true`다.
모델 재학습을 포함하는 정식 불확실성 검증은 여러 LoopAd randomized experiment가
축적된 뒤의 후속 범위다.

`uplift-validation.v1`의 실험 20개, arm별 관측 1,000개 등의 값은 statistical
power 분석 결과가 아닌 초기 안전 가드다. 실제 baseline rate, 검출할 최소 uplift,
allocation ratio, 실험 간 분산과 목표 검정력으로 새 정책 버전을 만들어야 한다.
Persistent registry와 수동 승인 provenance가 구현되기 전에는 ACTIVE가 될 수 없다.

## 검증 도구

합성 검증은 알려진 negative, zero, positive treatment effect의 방향과 순서를
signed CATE가 복원하는지 확인한다.

```bash
python -m app.uplift.synthetic_validation \
  --sample-size 6000 \
  --sample-seed 41 \
  --output-path artifacts/uplift/synthetic-report.json
```

Criteo adapter는 로컬 CSV 또는 gzip 파일만 읽는다. 데이터를 저장소에 포함하거나
실행 중 다운로드하지 않는다.

```bash
python -m app.uplift.criteo_adapter \
  --input-path /path/to/criteo-uplift.csv.gz \
  --max-rows 100000 \
  --sample-seed 41 \
  --output-path artifacts/uplift/criteo-report.json
```

Stable row hash와 seed로 treatment/control별 70% train, 30% test를 고정한다. 모델은
train에만 fit하고 ATE, AUUC, Qini, uplift@top-k는 test에서만 계산한다. 리포트에는
split policy, train/test fingerprint와 각 표본 수도 포함된다.
Criteo artifact는 `external_pipeline_validation`, `serving_eligible=false`이며 숙박
프로모션의 성과 근거나 LoopAd ACTIVE 모델로 사용할 수 없다.

## 발표 경계

말할 수 있는 내용:

- 프로모션 조건과 실제 행동을 조합해 프로모션마다 다른 고객군을 동적으로 만든다.
- 현재 순위는 프로모션 적합도와 행동 기반 예상 예약 가능성을 사용한다.
- 발송군·대조군, 배정 전 특징, 이후 프로모션 일치 예약을 재현할 수 있는
  Uplift-ready 구조를 구축했다.
- 실제 모델은 운영 실험 데이터 축적과 별도 검증 및 승인을 통과한 뒤에만
  활성화된다.
- 현재 구현은 treatment/control 수집 계약과 offline Uplift PoC까지이며, 자동 학습
  worker와 운영 activation workflow는 후속 범위다.

말하면 안 되는 내용:

- 현재 표시한 예상 예약 전환율이 광고의 증분 효과라는 주장
- LoopAd 운영 데이터에서 CATE가 이미 검증됐다는 주장
- Criteo 검증이 숙박 광고 성과를 입증한다는 주장
- 운영 데이터가 쌓이면 현재 코드가 자동으로 학습·배포한다는 주장
