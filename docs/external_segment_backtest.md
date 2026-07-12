# 외부 데이터셋 세그먼트 추천 검증

이 도구는 Decision API가 사용하는 `generate_raw_event_segment_definitions`를 외부 데이터의 관찰 신호로 실행하고, 감춰 둔 실제 결과에서 후보별 성과를 측정한다. 데이터셋마다 관측 가능한 사실이 다르므로 결과를 하나의 종합 점수로 합치지 않는다.

## 설치

Parquet 기반 Synerise adapter는 개발 의존성의 `pyarrow`를 사용한다.

```bash
python -m pip install -e '.[dev]'
```

원본 데이터는 다음 기본 경로에 둔다. `artifacts/`는 Git에서 제외된다.

```text
artifacts/external-datasets/
├── airbnb-recruiting-new-user-bookings/
├── booking-com/
└── synerise_dataset/
```

## 실행

### Booking.com

```bash
.venv/bin/python scripts/backtest_external_segments.py booking-com \
  --profile-pool-limit 1000 \
  --max-scenarios 3 \
  --sample-modulo 1
```

`train_set.csv`에서 여행별 마지막 예약 도시를 outcome으로 숨기고 이전 도시만 profile로 만든다. 시나리오 목적지는 이전 도시 이력에서만 선택한다. 기본 validation은 `test_set.csv`와 `ground_truth.csv`를 읽지 않는다.

이 결과는 다음 도시 정합성과 목적지 반복 관심을 검증한다. 검색 후 예약 전환율이나 미예약 사용자의 반응은 검증하지 못한다.

### Airbnb

```bash
.venv/bin/python scripts/backtest_external_segments.py airbnb \
  --profile-pool-limit 1000 \
  --sample-modulo 1
```

`sessions.csv.zip`의 검색·클릭·조회 행동으로 profile을 만들고 `country_destination != NDF`를 관측된 첫 예약 outcome으로 사용한다. 결과와 가까운 `booking_request`, `booking_response`, `partner_callback`은 feature에서 제외하고 목적지 label은 profile에 넣지 않는다.

세션 행에 절대 시각이 없으므로 이 결과는 시간 기반 미래 전환율이 아니라 정적 outcome holdout이다.

### Synerise

```bash
.venv/bin/python scripts/backtest_external_segments.py synerise \
  --profile-pool-limit 1000 \
  --max-scenarios 3 \
  --cutoff 2022-11-10T00:00:00Z \
  --lookback-days 90 \
  --outcome-days 28 \
  --sample-modulo 1
```

cutoff 이전의 검색·장바구니 추가·제거·구매만 profile에 사용하고 cutoff 이후 target category 구매를 outcome으로 사용한다. 시나리오 category도 관찰 구간에서만 선택한다.

`page_visit`는 약 2억 건이지만 URL과 SKU의 관계가 제공되지 않아 상품 상세 조회로 변환하지 않는다. Synerise 결과는 퍼널 후보 생성과 Rank 구조를 검증하는 교차 도메인 자료이며 숙박 성능과 합산하지 않는다.

## 산출물

기본 출력 경로는 `artifacts/external-segment-backtest/<dataset>/validation-<timestamp>`다.

```text
dataset_manifest.json  원본 파일 checksum과 신호별 direct/derived/proxy 계약
results.csv            Rank별 예상값, 실제값, baseline, lift, 후보 중복도
summary.json           데이터 계약, 모델 metadata, 집계 지표
report.md              사람이 읽는 결과와 검증 불가능한 주장
```

`--skip-checksum`은 빠른 smoke test에서만 사용한다. 비교 가능한 검증 기록에는 원본 checksum을 남긴다.

## 지표 해석

- `rank_one_beats_baseline_rate`: Rank 1의 실제 outcome rate가 전체 profile baseline보다 높은 시나리오 비율
- `rank_one_is_best_rate`: 실제 양성 outcome이 있고 후보가 두 개 이상인 시나리오에서 Rank 1이 최고였던 비율
- `mean_rank_one_lift_percentage_points`: Rank 1 실제 outcome rate와 baseline의 평균 차이
- `mean_absolute_prediction_error_percentage_points`: 표시한 예상값과 데이터셋 실제 outcome rate의 평균 절대 차이. outcome 계약이 다른 데이터셋에서는 calibration을 증명하는 값으로 해석하지 않는다.
- `mean_non_first_rank_overlap`: Rank 2 이후 후보와 앞선 후보 간 최대 사용자 Jaccard overlap의 평균

Booking.com이나 Airbnb처럼 후보가 하나뿐이면 Rank 순서의 정확성은 검증되지 않으며 `rank_one_is_best_rate`는 `N/A`다.

## 검증 원칙

1. 외부 데이터의 사용자 ID를 Expedia 사용자와 연결하지 않는다.
2. 존재하지 않는 검색·광고 노출·할인·예약 label을 생성하지 않는다.
3. proxy 변환은 manifest에 원본 필드와 한계를 남긴다.
4. 시나리오 선택과 profile 생성에는 outcome 구간을 사용하지 않는다.
5. 데이터셋별 outcome 의미가 다르므로 성과를 평균내지 않는다.
6. 이 검증은 관측 데이터 기반 적합성 검증이며 광고의 인과적 증분 효과를 증명하지 않는다.
