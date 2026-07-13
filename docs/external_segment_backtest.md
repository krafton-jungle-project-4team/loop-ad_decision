# 외부 데이터셋 세그먼트 추천 검증

이 도구는 운영에서 사용하는 세그먼트 후보 생성·랭킹 로직을 외부 데이터의 관찰 신호로 실행하고, 감춰 둔 실제 결과에서 후보별 성과를 측정한다.

외부 데이터는 Expedia 기반 예상 예약률 모델을 학습하거나 보정하는 데 사용하지 않는다. 외부 결과는 추천 후보가 다른 데이터에서도 baseline보다 좋은 사용자를 선별하는지, Rank가 의미 있게 구분되는지 진단하는 용도다.

## 데이터 분할 원칙

| 데이터 | 반복 가능한 개발 진단 | 한 번만 여는 봉인 평가 |
| --- | --- | --- |
| Booking.com | `train_set.csv`에서 여행별 마지막 도시를 숨김 | 공식 `test_set.csv + ground_truth.csv` |
| Airbnb | 사용자 ID 해시 `mod 5`의 remainder `0,1,2,3` | 사용자 ID 해시 `mod 5`의 remainder `4` |
| Synerise | remainder `0,1,2,3`, cutoff `2022-09-29`, `2022-10-13` | remainder `4`, cutoff `2022-11-10` |

Airbnb는 절대 행동 시각이 없으므로 정적 label 기반 사용자 holdout이다. Synerise는 숙박이 아닌 리테일 데이터이므로 후보 생성·랭킹 구조의 교차 도메인 진단일 뿐 숙박 성능의 직접 근거가 아니다.

## 설치와 원본 경로

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

## 1. 개발 진단

개발 중에는 `development`만 반복 실행한다. 이 결과를 보고 일반적인 추천 규칙을 개선할 수 있으므로 최종 성능 수치가 아니라 개발 진단 데이터로 취급한다.

### Booking.com

```bash
.venv/bin/python scripts/backtest_external_segments.py development booking-com \
  --profile-pool-limit 1000 \
  --max-scenarios 3
```

`train_set.csv`의 각 여행에서 마지막 도시를 outcome으로 숨기고 이전 도시만 profile로 만든다. 공식 `test_set.csv`와 `ground_truth.csv`는 읽지 않는다.

### Airbnb

```bash
.venv/bin/python scripts/backtest_external_segments.py development airbnb \
  --profile-pool-limit 1000
```

검색·클릭·상세 탐색 행동으로 profile을 만들고 `country_destination != NDF`를 관측된 첫 예약 label로 사용한다. 결과와 가까운 booking action과 목적지 label은 feature에서 제외한다.

### Synerise

```bash
.venv/bin/python scripts/backtest_external_segments.py development synerise \
  --profile-pool-limit 1000 \
  --max-scenarios 3 \
  --lookback-days 90 \
  --outcome-days 28
```

기본 실행은 두 development cutoff를 각각 평가한다. cutoff 이전 검색·장바구니 추가·제거·구매만 profile에 사용하고, cutoff 이후 target category 구매를 outcome으로 사용한다.

빠른 smoke test에서 전체 사용자를 쓰려면 `--sample-modulo 1`을 지정할 수 있다. 비교 가능한 기록에는 원본 checksum을 남기며 `--skip-checksum`은 로컬 smoke test에서만 사용한다.

## 2. 외부 최종 평가 봉인

추천 로직과 예상 예약률 모델 수정이 끝난 뒤 PR을 `dev`에 병합한다. manifest 봉인은
깨끗한 `dev`에서만 허용한다. 최종 실행은 브랜치 이름과 관계없이 tracked working tree가
깨끗하고 manifest에 기록된 정확한 commit과 tree를 checkout한 환경에서 허용한다.

```bash
git switch dev
git pull --ff-only origin dev

.venv/bin/python scripts/backtest_external_segments.py seal-final-test booking-com
.venv/bin/python scripts/backtest_external_segments.py seal-final-test airbnb
.venv/bin/python scripts/backtest_external_segments.py seal-final-test synerise
```

봉인 명령은 다음을 manifest에 고정한다.

- 원본 파일 크기와 SHA-256
- Expedia에서 학습한 모델 파일과 metadata
- Git commit과 tree
- profile 크기, 후보 수, 최소 표본 등 실행 설정
- development/final 사용자·시간 분할
- 외부 outcome과 모델 정답의 비교 가능성 계약
- 결과를 보기 전에 정한 통과 기준

봉인 과정은 outcome 파일의 checksum만 계산하며 outcome을 이용해 시나리오나 후보를 고르지 않는다. 기존 manifest는 덮어쓰지 않는다.

명령이 성공하면 다음 두 값이 출력된다.

```text
manifest=artifacts/external-segment-backtest/sealed/booking-com-final.manifest.json
confirmation=RUN_EXTERNAL_FINAL_<manifest-id-prefix>
```

## 3. 봉인 결과 한 번만 열기

아직 최종 결과를 보고 싶지 않다면 이 명령은 실행하지 않는다. `run-final-test`는
manifest별 실행 ID와 상태를 `*.execution-started.json` 저널에 기록한다. 같은 manifest로
새 평가를 시작하는 것은 차단하며, 장애 복구는 기존 실행 ID로만 가능하다.

봉인 이후 `dev`에 새 commit이 쌓였다면 manifest의 `code_commit`으로 별도 worktree를
만들어 실행한다.

```bash
git worktree add ../loop-ad-external-final <manifest-code-commit>
cd ../loop-ad-external-final
```

```bash
.venv/bin/python scripts/backtest_external_segments.py run-final-test \
  --manifest artifacts/external-segment-backtest/sealed/booking-com-final.manifest.json \
  --confirm RUN_EXTERNAL_FINAL_<manifest-id-prefix>
```

확인 토큰, 원본 checksum, 모델, Git commit 또는 tree가 봉인 시점과 다르면 실행하지
않는다. 실패 시 재시도 가능 여부는 outcome 개봉 시점을 기준으로 나뉜다.

`*.execution-started.json`은 해당 filesystem에서 manifest 중복 실행을 막는다. 외부
사용자가 저장소를 새로 clone하거나 로컬 저널을 삭제하는 행동까지 신뢰성 있게 막을 수는
없다. GitHub 사용자별 1회 실행은 outcome을 외부에 배포하지 않는 hosted evaluator에서
공용 실행권을 원자적으로 선점하는 별도 경계로 구현해야 한다.

- `reserved`, `retryable_pre_outcome_failure`: outcome을 읽기 전의 파일·환경 오류다. 저널의
  `execution_id`를 `--resume-execution-id`로 전달하면 같은 실행을 재개할 수 있다.
- `outcomes_opened`, `failed_after_outcomes`: 정답 label을 읽은 뒤 계산이 실패한 상태다.
  다시 계산하면 최종 시험을 반복하게 되므로 재실행을 차단한다.
- `result_staged`: 계산과 산출물 기록은 끝났지만 최종 디렉터리 공개가 실패한 상태다.
  같은 실행 ID로 재개하면 label을 다시 읽거나 계산하지 않고 게시만 완료한다.
- `completed`: 최종 평가가 완료된 상태로 다시 실행할 수 없다.

재개 명령은 최초 명령에 실행 ID만 추가한다.

```bash
.venv/bin/python scripts/backtest_external_segments.py run-final-test \
  --manifest artifacts/external-segment-backtest/sealed/booking-com-final.manifest.json \
  --confirm RUN_EXTERNAL_FINAL_<manifest-id-prefix> \
  --resume-execution-id <execution_id>
```

## 산출물

개발 진단은 기본적으로 아래 경로에 생성된다.

```text
artifacts/external-segment-backtest/<dataset>/development-<timestamp>/
├── dataset_manifest.json
├── results.csv
├── summary.json
├── report.md
└── development_diagnostic_summary.json
```

Synerise처럼 cutoff가 여러 개면 cutoff별 하위 디렉터리가 생긴다. 봉인 최종 평가는 manifest ID가 포함된 별도 디렉터리에 기록하며 덮어쓸 수 없다.

## 지표 해석

- `rank_one_beats_baseline_rate`: Rank 1의 실제 outcome rate가 전체 profile baseline보다 높은 시나리오 비율
- `rank_one_is_best_rate`: 비교 가능한 시나리오에서 Rank 1의 실제 성과가 다른 모든 후보보다 **엄격하게 높은** 비율. 동률은 승리로 세지 않는다.
- `rank_one_tied_best_rate`: Rank 1이 다른 후보와 실제 최고 성과로 동률인 비율
- `mean_rank_one_lift_percentage_points`: Rank 1 실제 outcome rate와 baseline의 평균 차이
- `rank_two_beats_baseline_rate`, `rank_three_beats_baseline_rate`: Rank 2·3도 baseline보다 유용했던 시나리오 비율
- `pairwise_rank_accuracy`: 실제 성과가 동률이 아닌 후보 쌍 중 앞 Rank의 성과가 더 높았던 비율. Rank 1>2, Rank 1>3, Rank 2>3을 모두 평가한다.
- `pairwise_rank_tie_rate`: 후보 쌍의 실제 성과가 같아 순서의 옳고 그름을 판단할 수 없었던 비율
- `rank_comparable_scenario_count`: 후보가 둘 이상이고 실제 성과가 다른 후보 쌍이 하나 이상 있어 Rank 순서를 판단할 수 있는 시나리오 수
- `three_rank_scenario_count`: Rank 1·2·3 후보가 모두 생성되어 Top 3 전체를 평가할 수 있는 시나리오 수
- `mean_non_first_rank_overlap`: Rank 2 이후 후보와 앞선 후보 간 최대 사용자 Jaccard overlap 평균
- `mean_absolute_prediction_error_percentage_points`: 외부 outcome의 정답 정의가 Expedia 모델과 달라 기본적으로 `N/A`

Booking.com의 다음 도시, Airbnb의 정적 첫 예약, Synerise의 리테일 구매를 Expedia의 향후 목적지 일치 예약률과 직접 비교하면 안 된다. 따라서 외부 봉인 평가의 통과 기준에는 예상값 MAE를 넣지 않고, baseline lift·Top 3 유용성·Rank 순서 정확성·후보 다양성·중복도를 사용한다.

봉인 평가의 최종 판정은 세 가지다.

- `passed`: 비교 가능한 시나리오와 Rank 2·3 표본이 충분하고 모든 품질 기준을 통과함
- `failed`: 평가 근거는 충분하지만 Top 3 유용성, 순서 정확성, 다양성 또는 중복도 기준을 통과하지 못함
- `inconclusive`: Rank 2·3이 충분히 생성되지 않았거나 비동률 후보 쌍과 관측 outcome이 부족해 순위 품질을 판정할 수 없음

후보가 하나뿐인 데이터셋을 Rank 1 성공으로 간주하지 않는다. 예를 들어 Airbnb처럼 구조상 하나의 정적 label 시나리오만 제공하는 경우 추천 유용성에 대한 참고 진단은 가능하지만 Top 3 순위 평가의 최종 판정은 `inconclusive`가 될 수 있다.

Top 3 판정 기준은 `external.sealed-final-test.v2` manifest에 함께 봉인된다. 이전 v1 manifest는 새 기준을 포함하지 않으므로 결과를 열기 전에 v2 manifest로 다시 생성해야 한다.

## 검증 원칙

1. 외부 데이터의 사용자 ID를 Expedia 사용자와 연결하지 않는다.
2. 존재하지 않는 검색·광고 노출·할인·예약 label을 생성하지 않는다.
3. proxy 변환은 manifest에 원본 필드와 한계를 남긴다.
4. 시나리오 선택과 profile 생성에는 outcome을 사용하지 않는다.
5. 데이터셋별 outcome 의미가 다르므로 결과를 하나의 평균 점수로 합치지 않는다.
6. 외부 결과를 모델 fitting이나 calibration 입력으로 사용하지 않는다.
7. 이 검증은 관측 데이터 기반 선별력 평가이며 광고의 인과적 증분 효과를 증명하지 않는다.
