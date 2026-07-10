# Expedia 세그먼트 추천 백테스트

## 목적

이 도구는 기준일 이전 Expedia 행동만 사용해 현재 AI 세그먼트 후보를 생성하고,
기준일 이후의 실제 `is_booking=1`로 추천 품질을 평가한다.

다음 항목을 확인할 수 있다.

- 추천 후보의 미래 목적지 예약 전환율
- 전체 분석 대상 대비 전환율 향상도
- 예상 전환율과 미래 실제 전환율의 차이
- Rank 1이 Rank 2, Rank 3보다 실제로 성과가 좋았는지 여부

Expedia 데이터에는 광고 노출 대조군이 없으므로 광고 발송의 인과적 효과를 검증하는
도구는 아니다.

## 현재 추천 로직과의 관계

현재 추천의 `raw_event_intent` 경로는 사용자 행동 벡터의 최신 90일 window를 분석 대상
범위로 사용하고, 같은 window의 행동 신호를 다시 집계해 후보를 생성한다. 백테스트는
Expedia 원본에서 동일한 기간과 행동 신호를 직접 집계한 뒤 운영 코드의
`generate_raw_event_segment_definitions`를 그대로 호출한다.

백테스트마다 수천만 건을 `raw_events`와 `user_behavior_vectors`로 복제 저장하지 않으므로
빠르게 여러 기준일을 반복할 수 있다. 후보 타입, 조건 점수, 예상 전환율, Rank 결정은
별도의 백테스트 추천기가 아니라 현재 Decision 구현을 사용한다.

이 도구는 후보 선택과 Rank 품질을 검증한다. `user_behavior_vectors` 배치 적재 자체와
확정 세그먼트의 광고 대상 배정은 기존 통합 테스트 및 운영 흐름의 검증 범위다.

프로모션 intent는 각 시나리오의 목적지와 시즌을 정답 JSON으로 고정한다. LLM의 표현
변동과 후보 랭킹 품질을 섞지 않기 위한 조치다. LLM intent 추출 정확도는 별도의 고정
프로모션 문장 집합으로 평가해야 한다.

## 파일 사용 범위

- `train.csv`: 행동 입력과 미래 `is_booking` 평가 라벨로 사용한다.
- `test.csv`: `is_booking`이 없어 성과 백테스트에는 사용하지 않는다.
- `destinations.csv`: 현재 운영 후보 생성기가 149개 잠재 특성을 사용하지 않으므로 이번
  검증에는 넣지 않는다. 백테스트에만 추가하면 운영과 다른 추천기를 평가하게 된다.
- `sample_submission.csv`: 호텔 클러스터 Kaggle 제출 형식이므로 사용하지 않는다.

## 1. 환경 준비

Decision 가상환경을 활성화한다.

```bash
source .venv/bin/activate
```

Decision 저장소 루트의 `.env`에 로컬 ClickHouse 연결 정보가 있어야 한다.

```dotenv
LOOPAD_CLICKHOUSE_URL=http://localhost:18123
LOOPAD_CLICKHOUSE_DATABASE=...
LOOPAD_CLICKHOUSE_USERNAME=...
LOOPAD_CLICKHOUSE_PASSWORD=...
```

ClickHouse에 `expedia_hotel_events` 테이블과 데이터가 이미 있다면 2단계로 바로 이동한다.

데이터가 없다면 `train.csv`를 스트리밍 적재한다. 3.8GB 파일 전체를 Python 메모리에
올리지 않는다.

```bash
python3 scripts/backtest_expedia_segments.py prepare \
  --train-csv /Users/jinhyuk/Downloads/expedia-hotel-recommendations/train.csv
```

테이블에 데이터가 있으면 기본적으로 중복 적재하지 않는다. 기존 데이터를 지우고 다시
적재해야 할 때만 명시적으로 `--replace`를 사용한다.

`test.csv`에는 `is_booking` 라벨이 없으므로 `prepare` 입력으로 사용할 수 없다.

## 2. 빠른 검증

먼저 전체 사용자의 약 5%를 결정적 해시 샘플링해 한 기준일과 상위 두 목적지만
검증한다.

```bash
python3 scripts/backtest_expedia_segments.py smoke
```

기본 설정은 다음과 같다.

- 기준일: `2014-10-01`
- 행동 관찰: 기준일 이전 90일
- 미래 평가: 기준일 이후 30일
- 분석 사용자 풀: 최근 사용자 최대 1,000명
- 후보 사용자: 현재 운영 로직과 동일하게 후보별 최대 160명
- 사용자 샘플: `cityHash64(user_id) % 20 = 0`

## 3. 월별 백테스트

샘플로 2014년 전체 기준일을 순회한다.

```bash
python3 scripts/backtest_expedia_segments.py run
```

사용자 해시 사전 샘플링을 끄고 전체 데이터에서 운영과 동일하게 최근 1,000명 풀을
선택하려면 다음 명령을 사용한다.

```bash
python3 scripts/backtest_expedia_segments.py run \
  --start-cutoff 2014-01-01 \
  --end-cutoff 2014-12-01 \
  --user-sample-modulo 1
```

분석 풀 크기에 따른 결과 변화도 확인하려면 `--profile-pool-limit`을 늘려 별도로 실행한다.
현재 운영 기본값과 직접 비교하는 결과는 `1000`을 사용해야 한다.

```bash
python3 scripts/backtest_expedia_segments.py run \
  --user-sample-modulo 1 \
  --profile-pool-limit 10000
```

여름 체크인 조건까지 함께 평가하려면 `--season summer`를 추가한다.

```bash
python3 scripts/backtest_expedia_segments.py run \
  --season summer \
  --user-sample-modulo 1
```

## 4. 결과 파일

결과는 기본적으로 다음 경로에 저장된다.

```text
artifacts/expedia-segment-backtest/<mode>-<timestamp>/
├── report.md
├── results.csv
├── summary.json
└── skipped_scenarios.csv
```

먼저 `report.md`를 열면 된다. 세부 후보별 수치는 `results.csv`, 자동 처리 가능한 집계는
`summary.json`에 있다.

주요 컬럼은 다음과 같다.

| 컬럼 | 의미 |
| --- | --- |
| `predicted_conversion_rate` | 추천 시점에 계산한 예상 전환율 |
| `actual_contextual_conversion_rate` | 미래에 해당 목적지를 예약한 추천 사용자 비율 |
| `baseline_contextual_conversion_rate` | 전체 분석 대상의 미래 해당 목적지 예약률 |
| `absolute_lift_percentage_points` | 추천 후보와 전체 기준의 전환율 차이 |
| `calibration_error_percentage_points` | 예상값과 미래 실제값의 절대 오차 |
| `actual_any_conversion_rate` | 목적지와 무관하게 미래에 예약한 사용자 비율 |

`actual_any_conversion_rate`만 높고 `actual_contextual_conversion_rate`의 향상도가 낮으면
프로모션 맞춤 추천이 아니라 원래 예약 가능성이 높은 사용자를 추천했을 가능성이 크다.

## 시간 누수 방지

추천 행동 프로필 쿼리는 항상 다음 조건을 사용한다.

```text
observation_start <= date_time < cutoff
```

실제 성과 쿼리는 별도로 다음 조건을 사용한다.

```text
cutoff <= date_time < cutoff + outcome_days
```

미래 평가 구간의 `is_booking`은 세그먼트 후보 생성과 예상 전환율 계산에 전달되지 않는다.

## 원본에 없는 신호

Expedia에는 `promotion_impression`, `promotion_click`, `campaign_landing`이 없다. 해당 값은
0으로 유지하며 임의 생성하지 않는다. 따라서 `promotion_responsive` 후보는 생성되지 않을
수 있다.

가격도 원본에 없으므로 임의 가격은 사용하지 않는다. 패키지 여부, 검색 시점과 체크인
사이 기간, 숙박 기간처럼 원본 행동에서 결정적으로 유도할 수 있는 혜택 신호만 참고한다.
