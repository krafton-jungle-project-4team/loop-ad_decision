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

## 3. 2013 학습 / 2014 개발 검증

예상 전환율 보정과 추천 로직 개발은 `validation` 명령을 사용한다. `holdout`은 기존
호환성을 위한 alias지만, 2014년 결과를 보고 추천 규칙과 가중치를 수정했기 때문에 더
이상 독립적인 최종 테스트가 아니다.

```bash
python3 scripts/backtest_expedia_segments.py validation
```

이 명령은 다음 순서를 자동으로 수행한다.

1. 2013-05-01부터 2013-12-01까지 기준일 이전 90일 행동으로 후보 feature를 만든다.
   이때 기존 휴리스틱 Top 3가 아니라 생성 가능한 모든 후보 타입을 학습 표본으로 쓴다.
2. 각 기준일 이후 30일의 동일 목적지 `is_booking=1`을 학습 target으로 연결한다.
3. 목적지 일치율, 목적지별 기본 수요, 퍼널 행동, 혜택 반응으로 logistic 보정식을 학습한다.
4. 학습에 사용하지 않은 2014-01-01부터 2014-12-01까지의 예상값과 Rank를 계산한다.
5. 2014년 미래 결과와 비교해 MAE, Brier score, lift, Rank 1 적중률을 기록한다.

전체 후보 풀을 학습하는 이유는 보정 전 휴리스틱이 먼저 선택한 후보만 학습할 때 생기는
선택 편향을 막기 위해서다. 사용자에게 노출되는 개발 검증 결과에는 운영과 동일하게
사용자 중복 제한과 Rank 점수를 적용한 최대 3개 후보만 기록한다.

Expedia 원본에는 무료 취소와 조식 포함 속성이 없다. 따라서 백테스트의 혜택 신호는
실제 `is_package=1`만 `deal_event_count`로 사용하며, `free_cancellation_count`와
`breakfast_included_count`는 0으로 둔다. 관찰할 수 없는 혜택을 체류 기간이나 예약
리드타임으로 임의 생성하지 않는다.

5번 결과를 본 뒤 로직을 수정했다면 2014년은 통계적인 학습 데이터는 아니더라도
human-in-the-loop 개발 검증 데이터다. 이 수치를 최종 일반화 성능으로 발표하지 않는다.

빠른 확인은 기본 5% 결정적 사용자 표본으로 실행한다. 결론이 전체 원천 데이터에서도
유지되는지 확인할 때는 다음처럼 해시 사전 샘플링을 끈다.

```bash
python3 scripts/backtest_expedia_segments.py validation \
  --user-sample-modulo 1 \
  --profile-pool-limit 1000
```

`--user-sample-modulo 1`은 3,770만 원천 행에서 사용자를 해시로 미리 제외하지 않는다는
의미다. `--profile-pool-limit 1000`은 그 전체 원천 데이터에서 현재 운영 추천과 동일하게
최근 사용자 최대 1,000명을 후보 분석 풀로 사용한다. 두 옵션은 서로 다른 단계의 제한이다.

결과 디렉터리에는 학습 모델과 시간 분리 결과가 함께 생성된다.

```text
artifacts/expedia-segment-backtest/validation-<timestamp>/
├── contextual_booking_calibration_v2.json
├── temporal_validation_report.md
├── temporal_validation_summary.json
├── training-2013/
│   ├── results.csv
│   └── summary.json
└── development-validation-2014/
    ├── results.csv
    └── summary.json
```

운영에서 별도 모델 파일을 사용하려면 아래 환경 변수에 검증된 JSON 경로를 지정한다.
지정하지 않으면 저장소에 포함된 기본 보정 모델을 사용한다.

```dotenv
LOOPAD_SEGMENT_PERFORMANCE_MODEL_PATH=/path/to/contextual_booking_calibration_v2.json
```

## 4. 봉인 최종 테스트

### 4.1 코드와 모델 동결

관련 PR을 모두 `dev`에 머지하고 tracked working tree가 clean인 상태에서만 최종 테스트
manifest를 만들 수 있다. 미추적 로컬 파일은 검사에서 제외하지만 tracked 코드를 수정한
상태에서는 실행을 거부한다.

```bash
git switch dev
git pull origin dev
git status --short
```

### 4.2 미래 정답을 열지 않고 manifest 봉인

다음 명령은 2014년 월별 개발 검증 Top 3 목적지의 합집합을 제외한다. 이후 2014년
7월부터 12월까지 각 기준일에서 아직 사용하지 않은 목적지를 3개씩 선택한다. 목적지
선정에는 기준일 이전 90일의 행동과 사용자 수만 사용한다. 미래 `is_booking`은 원천
무결성 체크섬에만 포함하며 목적지 선정이나 성능 계산에는 전달하지 않는다.

```bash
.venv/bin/python scripts/backtest_expedia_segments.py seal-final-test \
  --user-sample-modulo 1 \
  --profile-pool-limit 1000
```

manifest에는 다음 값이 고정된다.

- 정확한 기준일과 목적지 ID 목록
- 개발 검증에서 제외한 목적지 ID 목록
- Decision code commit과 tree hash
- 2013년 학습 모델 SHA-256과 버전
- Expedia 원천 통계와 전체 행 체크섬
- 사용자 sampling, profile pool, 후보 수 설정
- 결과를 보기 전에 등록한 합격 기준

기본 합격 기준은 다음과 같다.

- Rank 1 기준선 승률 `70% 이상`
- Rank 1 실제 최고 후보 비율 `50% 이상`
- 전체 후보 MAE `3.5%p 이하`
- 예측 편향 절댓값 `1.5%p 이하`
- Brier skill score `0 초과`

명령이 출력하는 `confirmation=RUN_FINAL_TEST_...` 값은 최종 실행 전까지 보관한다.

### 4.3 코드 동결 후 단 한 번 실행

manifest 생성 후 코드, 모델, 원천 데이터 중 하나라도 달라지면 실행을 거부한다. 아래
명령은 확인 토큰이 일치할 때 manifest별 실행 ID와 상태 저널을 먼저 생성한 뒤 미래 예약
결과를 조회한다.

실행 시 브랜치 이름은 검사하지 않는다. `dev`가 봉인 이후 앞으로 진행했더라도 manifest의
`code_commit`을 별도 worktree에 checkout하고 tracked working tree가 clean하면 실행할 수
있다. 현재 commit과 tree는 여전히 manifest 값과 정확히 일치해야 한다.

```bash
git worktree add ../loop-ad-expedia-final <manifest-code-commit>
cd ../loop-ad-expedia-final
```

```bash
.venv/bin/python scripts/backtest_expedia_segments.py run-final-test \
  --confirm RUN_FINAL_TEST_<seal-command-output>
```

실행이 시작된 manifest에는 `*.execution-started.json` 저널이 남는다. 재시도 가능 여부는
미래 예약 outcome을 처음 조회한 시점을 기준으로 나뉜다.

이 저널은 해당 filesystem에서 한 manifest의 중복 실행을 막는 로컬 계약이다. 저장소를
다시 clone한 외부 사용자의 계정별 실행 횟수까지 통제하지는 않는다. 계정별 1회 제한은
원본 outcome을 보유한 hosted evaluator와 공용 실행권 저장소에서 별도로 적용해야 한다.

- outcome 조회 전 실패는 저널의 `execution_id`를 지정해 같은 실행으로 재시도할 수 있다.
- outcome 조회 후 계산 실패는 최종 시험 반복을 막기 위해 재실행할 수 없다.
- 계산 완료 후 결과 디렉터리 공개만 실패했다면 같은 실행 ID로 게시만 재개할 수 있다.
- 완료된 실행과 다른 실행 ID를 이용한 재시도는 차단한다.

재개할 때는 최초 실행 명령에 다음 옵션을 추가한다.

```bash
  --resume-execution-id <execution_id>
```

결과를 확인한 뒤 코드를 수정하면 새로운 라벨 데이터 없이는 다시 “최종 테스트”라고
부를 수 없다. 즉 재시도 기능은 결과를 다시 계산하는 기능이 아니라, outcome을 읽기 전
장애 또는 계산이 끝난 뒤 게시 장애를 복구하는 기능이다.

생성되는 최종 산출물은 다음과 같다.

```text
artifacts/expedia-segment-backtest/sealed-final-<manifest-id-prefix>/
├── sealed_final_test_report.md
├── sealed_final_test_summary.json
└── details/
    ├── results.csv
    └── skipped_scenarios.csv
```

이 결과도 완전히 새로운 연도의 외부 테스트는 아니다. 이전에 평가하지 않은 목적지를
봉인한 내부 destination holdout이며, 실제 광고의 인과적 증분 효과를 증명하지 않는다.

## 5. 월별 단순 백테스트

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

## 6. 결과 파일

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
| `prediction_model_version` | 예상값을 계산한 보정 모델 버전 |
| `performance_features` | 미래 결과 없이 추천 시점에 계산한 모델 입력 feature |
| `actual_contextual_conversion_rate` | 미래에 해당 목적지를 예약한 추천 사용자 비율 |
| `baseline_contextual_conversion_rate` | 전체 분석 대상의 미래 해당 목적지 예약률 |
| `absolute_lift_percentage_points` | 추천 후보와 전체 기준의 전환율 차이 |
| `calibration_error_percentage_points` | 예상값과 미래 실제값의 절대 오차 |
| `actual_any_conversion_rate` | 목적지와 무관하게 미래에 예약한 사용자 비율 |

`actual_any_conversion_rate`만 높고 `actual_contextual_conversion_rate`의 향상도가 낮으면
프로모션 맞춤 추천이 아니라 원래 예약 가능성이 높은 사용자를 추천했을 가능성이 크다.

`temporal_validation_summary.json`의 주요 개발 검증 지표는 다음과 같다.

- `all_candidate_mean_absolute_error_percentage_points`: 예상값과 실제값의 평균 절대 오차
- `all_candidate_brier_score`: 사용자별 확률 예측 오차. 0에 가까울수록 좋다.
- `rank_one_beats_baseline_rate`: Rank 1이 전체 분석 대상의 목적지 예약률을 이긴 비율
- `rank_one_is_best_rate`: Rank 1 예약률이 다른 모든 후보보다 엄격하게 높았던 비율. 동률은 승리로 세지 않는다.
- `rank_one_tied_best_rate`: Rank 1이 실제 최고 예약률로 다른 후보와 동률이었던 비율
- `rank_two_beats_baseline_rate`, `rank_three_beats_baseline_rate`: Rank 2·3도 전체 사용자 기준 예약률을 이긴 비율
- `pairwise_rank_accuracy`: 동률이 아닌 후보 쌍에서 앞 Rank의 실제 예약률이 더 높았던 비율
- `pairwise_rank_tie_rate`: 실제 예약률이 같아 순서의 옳고 그름을 판단할 수 없었던 후보 쌍 비율
- `three_rank_scenario_count`: Rank 1·2·3을 모두 생성해 Top 3 전체를 비교할 수 있었던 시나리오 수

봉인 최종 테스트는 관측 outcome과 Rank 2·3 후보가 부족하면 `FAIL` 대신 `INCONCLUSIVE`로 기록한다. 충분한 근거가 있을 때만 예상값 보정, Top 3 유용성, 전체 순위 정확성을 함께 사용해 `PASS` 또는 `FAIL`을 결정한다.

이 기준은 `expedia.sealed-final-test.v2` manifest에 사전 등록된다. 아직 결과를 열지 않은 v1 manifest가 있다면 새 코드와 모델을 동결한 뒤 v2 manifest를 다시 생성해야 한다.

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
