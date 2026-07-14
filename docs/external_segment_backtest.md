# 외부 데이터셋 세그먼트 추천 검증

이 도구는 운영에서 사용하는 세그먼트 후보 생성·선택 로직을 외부 데이터의 관찰 신호로 실행하고, 감춰 둔 실제 결과에서 후보별 성과를 측정한다.

외부 데이터는 Expedia 기반 예상 예약률 모델을 학습하거나 보정하는 데 사용하지 않는다. 외부 결과는 추천된 후보 묶음이 다른 데이터에서도 baseline보다 좋은 사용자를 선별하는지와 후보가 충분히 다양하게 구성되는지 진단하는 용도다.

외부 개발 진단과 봉인 평가는 운영과 동일한 후보 지원 계약을 사용한다. Expedia 모델
metadata의 후보 타입별 학습 사례 수가 0이면 해당 타입은 예상 예약 전환율 계산과 추천
후보 선택에서 제외된다. 외부 데이터에서 관측된 결과를 이용해 그 타입을 임시 활성화하거나
Expedia 모델을 다시 학습하지 않는다.

## 데이터 분할 원칙

| 데이터 | 반복 가능한 개발 진단 | 한 번만 여는 봉인 평가 |
| --- | --- | --- |
| Booking.com | `train_set.csv`에서 여행별 마지막 도시를 숨김 | 공식 `test_set.csv + ground_truth.csv` |
| Airbnb | 사용자 ID 해시 `mod 5`의 remainder `0,1,2,3` | 사용자 ID 해시 `mod 5`의 remainder `4` |
| Synerise | remainder `0,1,2,3`, cutoff `2022-09-29`, `2022-10-13` | remainder `4`, cutoff `2022-11-10` |

Airbnb는 절대 행동 시각이 없으므로 정적 label 기반 사용자 holdout이다. Synerise는 숙박이 아닌 리테일 데이터이므로 후보 생성·선택 구조의 교차 도메인 진단일 뿐 숙박 성능의 직접 근거가 아니다.

## 설치와 원본 경로

Parquet 기반 Synerise adapter는 개발 의존성의 `pyarrow`를 사용한다.

```bash
python -m pip install -e '.[dev]'
```

원본 데이터는 각 평가자가 공식 배포처에서 직접 내려받아 다음 기본 경로에 둔다.
원본 파일은 이 저장소나 별도 공용 저장소에서 제공하지 않으며, `artifacts/`는 Git에서
제외된다.

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

### 추천 대상 선택 비율 진단

`development` 명령은 일반 추천 품질과 함께 조건 일치 사용자 중 행동 강도 상위
`20/40/60/80/100%`를 추천 대상으로 삼았을 때의 결과를 비교한다. 별도 옵션 없이 위 명령을
그대로 실행하면 현재 운영 비율인 80%를 강조하여 다음 항목을 진단한다.

- 선택 표본의 실제 외부 outcome rate와 전체 조건 일치자 대비 lift
- 전체 조건 일치 positive 중 선택 표본이 보존한 비율
- 조건 일치자 대비 reach와 최소 표본 안정성
- 후보별 baseline 초과율, 후보 수, 후보 간 사용자 중복도

비율 표는 데이터셋별로 따로 해석한다. Booking.com의 다음 도시, Airbnb의 정적 첫 예약,
Synerise의 향후 카테고리 구매를 하나의 절대 전환율로 합치지 않는다. 외부 진단은 80%가
다른 데이터에서도 선별력을 유지하는지 확인하는 자료일 뿐 다음 작업을 수행하지 않는다.

- Expedia 예상 성과 모델 재학습 또는 보정
- 운영 추천 대상 비율 자동 변경
- 봉인된 외부 final partition 접근

비율 격자나 현재 운영 비율을 명시적으로 재현하려면 다음처럼 실행한다.

```bash
.venv/bin/python scripts/backtest_external_segments.py development booking-com \
  --audience-selection-ratios 0.2,0.4,0.6,0.8,1.0 \
  --current-runtime-ratio 0.8 \
  --audience-selection-min-selected-users 30
```

`assessment.status`는 다음 의미다.

- `supported`: 적용 가능한 후보에서 lift와 positive 포착률 기준을 통과하고 후보 baseline 초과율이 크게 악화되지 않음
- `caution`: 비교 근거는 있으나 lift, 포착률 또는 후보 baseline 초과율 중 하나 이상이 기준 미달
- `insufficient_evidence`: 비율 제한이 실제 적용된 후보나 관측 positive가 부족해 판단할 수 없음

외부 데이터에서 `caution`이 나오더라도 이 명령이 정책 파일을 생성하거나 덮어쓰지는 않는다.
원인을 후보 타입·데이터셋별로 분석한 뒤 Expedia development/validation에서도 확인되는
일반화 가능한 수정만 별도 PR로 반영한다.

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

봉인 manifest는 원본 데이터가 아니라 파일명·크기·SHA-256, 모델, 코드, 평가 기준을
담은 작은 공개 계약서다. 팀에서 최종 manifest를 확정하면 다음 명령으로 Git 레지스트리에
복사하고 JSON 파일만 커밋한다. 원본 데이터와 outcome 파일은 커밋하지 않는다.

```bash
.venv/bin/python scripts/backtest_external_segments.py register-final-test \
  --manifest artifacts/external-segment-backtest/sealed/booking-com-final.manifest.json

git add evaluation_manifests/external
git commit -m "test: 외부 최종 평가 manifest 등록"
```

clone 사용자는 등록된 manifest ID, 데이터셋, 봉인 commit을 다음 명령으로 확인할 수 있다.

```bash
.venv/bin/python scripts/backtest_external_segments.py list-final-tests
```

## 3. 봉인 결과 한 번만 열기

아직 최종 결과를 보고 싶지 않다면 이 명령은 실행하지 않는다. `run-final-test`는
manifest별 실행 ID와 상태를 `*.execution-started.json` 저널에 기록한다. 같은 manifest로
새 평가를 시작하는 것은 차단하며, 장애 복구는 기존 실행 ID로만 가능하다.

clone 사용자는 원본 데이터셋을 직접 내려받은 뒤, 봉인 이후 `dev`에 새 commit이 쌓였다면
manifest의 `code_commit`으로 별도 worktree를 만들어 실행한다. manifest는 registry가
있는 원래 clone의 절대 경로를 사용한다.

```bash
git worktree add ../loop-ad-external-final <manifest-code-commit>
cd ../loop-ad-external-final

MANIFEST=/absolute/path/to/original-clone/evaluation_manifests/external/<manifest-id>.json
SOURCE_DIR=/absolute/path/to/downloaded-dataset
```

```bash
.venv/bin/python scripts/backtest_external_segments.py run-final-test \
  --manifest "$MANIFEST" \
  --source-dir "$SOURCE_DIR" \
  --confirm RUN_EXTERNAL_FINAL_<manifest-id-prefix>
```

확인 토큰, 원본 checksum, 모델, Git commit 또는 tree가 봉인 시점과 다르면 실행하지
않는다. 실패 시 재시도 가능 여부는 outcome 개봉 시점을 기준으로 나뉜다.

`*.execution-started.json`은 해당 clone에서 manifest 중복 실행을 막는다. manifest가 Git
레지스트리에 있으면 저널도 그 파일 옆에 생성되며 `.gitignore`로 제외된다. 원본과 실행
환경을 평가자가 소유하므로 새 clone 생성, 저널 삭제, 코드 수정까지 막는 중앙 통제는 하지
않는다. 따라서 이 구조의 “한 번”은 **한 clone의 한 manifest당 한 번**이라는 재현성
규칙이지 GitHub 계정별 보안 제한이 아니다.

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
  --manifest "$MANIFEST" \
  --source-dir "$SOURCE_DIR" \
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
├── audience-selection/
│   ├── ratio_results.csv
│   ├── ratio_summary.json
│   └── report.md
└── development_diagnostic_summary.json
```

Synerise처럼 cutoff가 여러 개면 cutoff별 하위 디렉터리가 생긴다. 봉인 최종 평가는 manifest ID가 포함된 별도 디렉터리에 기록하며 덮어쓸 수 없다.

## 지표 해석

- `portfolio_candidate_beats_baseline_rate`: 추천 후보 전체에서 실제 outcome rate가 전체 profile baseline보다 높은 후보 비율
- `portfolio_scenario_any_candidate_beats_baseline_rate`: 추천 후보 중 하나 이상이 baseline을 넘은 시나리오 비율
- `portfolio_scenario_all_candidates_beat_baseline_rate`: 추천 후보 모두가 baseline을 넘은 시나리오 비율
- `portfolio_mean_candidate_lift_percentage_points`: 추천 후보 전체의 baseline 대비 평균 lift
- `portfolio_mean_worst_candidate_lift_percentage_points`: 시나리오별 최저 성과 후보를 모아 계산한 평균 lift
- `portfolio_multi_candidate_scenario_count`: 후보를 2개 이상 생성해 함께 평가한 시나리오 수
- `portfolio_three_candidate_scenario_count`: 후보 3개를 모두 생성해 함께 평가한 시나리오 수
- `mean_portfolio_candidate_overlap`: 뒤에 선택된 후보와 앞서 선택된 후보 간 최대 사용자 Jaccard overlap 평균
- `mean_absolute_prediction_error_percentage_points`: 외부 outcome의 정답 정의가 Expedia 모델과 달라 기본적으로 `N/A`

Booking.com의 다음 도시, Airbnb의 정적 첫 예약, Synerise의 리테일 구매를 Expedia의 향후 목적지 일치 예약률과 직접 비교하면 안 된다. 따라서 외부 봉인 평가의 통과 기준에는 예상값 MAE를 넣지 않는다. 또한 외부 데이터에서 생성되지 않는 복수 전략 후보를 억지로 평가하지 않는다.

`external.evaluation-contract.v1`은 데이터셋별로 검증 가능한 주장을 고정한다.

- Booking.com: 목적지 관심 후보의 다음 도시 outcome과 baseline 대비 lift
- Airbnb: 행동 활성 후보에 실제 첫 예약 사용자가 더 많이 포함되는지
- Synerise: 리테일 미래 구매 outcome을 이용한 타 도메인 후보 선별력 진단

Booking.com과 Airbnb 결과는 숙박 도메인의 보조 근거다. Synerise의 `verdict_scope`는 `cross_domain_diagnostic_only`이며 제품 최종 합격 여부를 직접 결정하지 않는다. 추천 후보 2~3개의 다양성과 중복도는 충분한 후보 묶음을 생성하는 Expedia 봉인 평가에서 검증한다.

봉인 평가의 최종 판정은 세 가지다.

- `passed`: 해당 데이터셋이 지원하는 주장에 필요한 outcome 근거와 품질 기준을 통과함
- `failed`: 지원하는 주장에 대한 근거는 충분하지만 후보 유용성 기준을 통과하지 못함
- `inconclusive`: 지원하는 주장에 필요한 시나리오나 관측 outcome이 부족함

지원하지 않는 기준은 `criteria_results`에 `applicable=false`, `passed=null`, `operator=not_applicable`로 남는다. 이는 자동 통과가 아니며 최종 verdict 계산에서 제외된다. 예를 들어 Airbnb는 정적 첫 예약 시나리오 하나로 후보 enrichment를 평가할 수 있지만, 복수 후보 다양성을 검증했다는 주장은 할 수 없다.

데이터셋별 판정 계약은 `external.sealed-final-test.v4` manifest에 함께 봉인된다. 이전 manifest는 claim별 적용 범위를 포함하지 않으므로 결과를 열기 전에 v4 manifest로 다시 생성해야 한다.

## 검증 원칙

1. 외부 데이터의 사용자 ID를 Expedia 사용자와 연결하지 않는다.
2. 존재하지 않는 검색·광고 노출·할인·예약 label을 생성하지 않는다.
3. proxy 변환은 manifest에 원본 필드와 한계를 남긴다.
4. 시나리오 선택과 profile 생성에는 outcome을 사용하지 않는다.
5. 데이터셋별 outcome 의미가 다르므로 결과를 하나의 평균 점수로 합치지 않는다.
6. 외부 결과를 모델 fitting이나 calibration 입력으로 사용하지 않는다.
7. 이 검증은 관측 데이터 기반 선별력 평가이며 광고의 인과적 증분 효과를 증명하지 않는다.
