# Audience ANN 검색 정책 벤치마크

이 벤치마크는 runtime 검색 방식을 바꾸지 않고 현재 구현을 그대로 재현하는
`current_runtime`과 `exact_all`, `filter_first_exact`, `ann_first`를 비교해 승리
구간과 ANN 구현값을 산출한다. 최종
산출물인 `candidate-policy.json`과 `implementation-fixtures.json`은 별도 runtime
PR의 입력이다. 측정 전에는 어떤 ANN 수치도 구현값으로 간주하지 않는다.

## 고정 조건과 안전장치

- source cutoff: `2015-01-01T00:00:00Z`
- observation window: 730일 (`2013-01-01T00:00:00Z`부터)
- vector version: `hotel_behavior.v2`
- cohort: 동일 seed로 정렬한 중첩 `50k, 100k, 250k, 500k, 750k, 1M`
- PostgreSQL: 규모마다 최신 Data Contract를 적용한 별도 빈 DB
- ClickHouse: Expedia revision을 가진 로컬 clone
- 품질 기준: precision `1.0`, recall과 95% Wilson lower bound 모두 `0.95`
- 100만 초과, manifest/version 불일치, 비어 있거나 겹치는 정책 bucket,
  미검증 구간은 항상 `exact_all`

`prepare-cohort`와 `run-cell`은 PostgreSQL과 ClickHouse host가 로컬 주소가
아니면 거부한다. cohort 적재는 `--confirm-empty-disposable-postgres`, 셀 실행은
`--confirm-disposable-postgres`를 명시해야 한다. 운영 DB를 대상으로 실행하지
않는다.

## 1. 고정 window production vector 준비

기존 `v1` 벡터를 재사용하지 않는다. Expedia를 backfill한 로컬 `raw_events`에서
production `UserBehaviorVectorBatchService`를 고정 시각으로 실행한다.

```bash
.venv/bin/python scripts/benchmark_audience_search.py prepare-vectors \
  --env-file .env.ann-source \
  --project-id demo_project \
  --window-end 2015-01-01T00:00:00Z \
  --window-days 730 \
  --output artifacts/ann-search/source-vectors.json \
  --confirm-local-clickhouse-write
```

출력의 `manifest_hash`, `source_revision_cutoff`, window, user count를 cohort
입력에 그대로 사용한다. 1M보다 사용자가 적으면 실제 최대 cohort까지만 검증한다.

## 2. 실험·scenario manifest 준비

전체 grid를 고정한 manifest를 먼저 만든다.

```bash
.venv/bin/python scripts/benchmark_audience_search.py template \
  --output artifacts/ann-search/experiment-manifest.json
```

scenario는 `HotelBookingBehaviorSchemaV2.compile_segment_audience()`와 bundled
semantic-selection calibration을 통과한 production candidate만 사용한다. query
vector도 production `SegmentVectorPreparer` 결과를 직렬화한다. Expedia로 만들 수
없는 candidate type/선택도 구간은 합성하지 않고 누락 상태로 둔다.

```json
{
  "experiment_version": "audience_search.benchmark.v1",
  "project_id": "expedia-ann-benchmark-v1",
  "vector_version": "hotel_behavior.v2",
  "manifest_hash": "<production-vector-manifest-hash>",
  "scenarios": [
    {
      "scenario_id": "intent_matched-hard-le-005-a",
      "scenario_set": "tuning",
      "candidate_type": "intent_matched",
      "query_vector": ["64개의 production compiler 출력"],
      "score_threshold": 0.55,
      "hard_predicate_keys": ["hotel_product_interest"],
      "predicate_parameters": {},
      "compiler_provenance": {
        "manifest_hash": "<production-vector-manifest-hash>",
        "calibration_version": "<compiled value>",
        "calibration_hash": "<compiled value>",
        "query_compiler_version": "<compiled value>",
        "query_compiler_hash": "<compiled value>",
        "template_id": "hotel.intent_matched.v1",
        "template_semantic_hash": "<compiled value>"
      }
    }
  ]
}
```

5% exclusion 확인 때는 동일 manifest에 `campaign_id`, `promotion_id`를 함께
넣고 최신 Data Contract 형식의 deterministic exclusion snapshot을 미리 만든다.
runner는 실제 제외 비율과 `--exclusion-ratio 0.05` 차이가 0.5%p를 넘으면
실행을 거부한다.

각 candidate type과 관측되는 hard/expected ratio bucket마다 production 입력이
서로 다른 `tuning`, `confirmation` scenario를 선정한다. 합성기는 같은
candidate type·규모·ratio bucket의 tuning holdout이 없으면 정책 생성을 거부한다.

## 3. 중첩 cohort 준비

Data Contract PostgreSQL schema만 적용된 빈 DB를 규모별로 준비하고 `.env`의
database 이름을 해당 DB로 바꾼다. `source-revision-cutoff`은 모든 규모에서
동일한 Expedia ingestion snapshot 시각을 쓴다.

```bash
.venv/bin/python scripts/benchmark_audience_search.py prepare-cohort \
  --env-file .env.ann-50k \
  --project-id expedia-ann-benchmark-v1 \
  --manifest-hash '<production-vector-manifest-hash>' \
  --source-revision-cutoff 2026-07-18T00:00:00Z \
  --cohort-size 50000 \
  --cohort-seed expedia-ann-v1 \
  --output artifacts/ann-search/cohorts/50000.json \
  --confirm-empty-disposable-postgres
```

같은 명령을 여섯 규모에 반복한다. preparer는
`SHA256(seed || user_id)` 순서의 prefix를 적재하므로 작은 cohort가 큰 cohort에
포함된다. source row는 64차원·finite 여부를 검사하고, Data Contract와 동일한
HNSW DDL로 index를 다시 만든다. 결과 JSON의 `cohort_sha256`, row count, index
크기와 build 시간을 보존한다.

```bash
.venv/bin/python scripts/benchmark_audience_search.py preflight \
  --env-file .env.ann-50k \
  --manifest artifacts/ann-search/scenario-manifest.json
```

preflight는 active generation의 manifest, corpus 수, pgvector/PostgreSQL/
ClickHouse version, HNSW index valid/ready 상태를 확인한다.

## 4. Exact ground truth와 현재 기준선

각 cohort·scenario마다 전체 corpus의 exact cosine 순위와 최종 양성 여부를
저장한다.

```bash
.venv/bin/python scripts/benchmark_audience_search.py ground-truth \
  --env-file .env.ann-50k \
  --manifest artifacts/ann-search/scenario-manifest.json \
  --scenario-id intent_matched-hard-le-005-a \
  --reference-sample-size 50000 \
  --output artifacts/ann-search/ground-truth/50000-intent.csv.gz
```

CSV에는 `cosine_rank`, `user_id`, `behavior_fit_score`, `score_pass`,
`final_positive`가 기록된다. 옆의 `*.summary.json`에는 `H`, 50k 표본의 `P`,
`E=H×P`, 실제 최종 회원 수, corpus hash, 95% Wilson gate를 만족하는 이론적
최소 rank가 기록된다. ground truth 쿼리는 PostgreSQL index/bitmap scan을
끄고 전체 corpus를 exact 정렬한다.

동일 confirmation 셀의 `current_runtime`도 실행한다. 원본 JSONL에는 ANN 실행
횟수, K 이력, audit 행 수, Exact fallback 여부와 단계별 시간이 저장된다.

## 5. K screening

현재 HNSW 값으로 중복 제거된 모든 정책 K와 corpus `1%, 5%, 10%, 25%` K를
warm-up 5회, 측정 10회 실행한다. 일반 측정과 diagnostic `EXPLAIN ANALYZE`는
분리되어 있다.

```bash
.venv/bin/python scripts/benchmark_audience_search.py run-cell \
  --env-file .env.ann-50k \
  --manifest artifacts/ann-search/scenario-manifest.json \
  --scenario-id intent_matched-hard-le-005-a \
  --phase screening \
  --warmups 5 --repetitions 10 \
  --hnsw 100:strict_order:20000 \
  --output artifacts/ann-search/raw.jsonl --append \
  --diagnostics-dir artifacts/ann-search/explain/screening \
  --confirm-disposable-postgres
```

screening survivor는 recall `≥0.90`, 빠른 Exact p95의 `120%` 이내, 지정 HNSW
index 사용, spill/OOM 없음 조건을 모두 통과해야 한다.

```bash
.venv/bin/python scripts/benchmark_audience_search.py phase-report \
  --input artifacts/ann-search/raw.jsonl \
  --phase screening \
  --output artifacts/ann-search/screening-survivors.json
```

## 6. HNSW tuning

survivor의 고유 K만 `--requested-k`에 넣어 24개
`ef_search × iterative_scan × max_scan_tuples` 조합을 warm-up 3회, 측정 10회
실행한다.

```bash
.venv/bin/python scripts/benchmark_audience_search.py run-cell \
  --env-file .env.ann-500k \
  --manifest artifacts/ann-search/scenario-manifest.json \
  --scenario-id intent_matched-hard-le-005-a \
  --phase tuning \
  --warmups 3 --repetitions 10 \
  --requested-k 10000,20000,50000 \
  --hnsw-grid full \
  --output artifacts/ann-search/raw.jsonl --append \
  --diagnostics-dir artifacts/ann-search/explain/tuning \
  --confirm-disposable-postgres

.venv/bin/python scripts/benchmark_audience_search.py phase-report \
  --input artifacts/ann-search/raw.jsonl \
  --phase tuning \
  --output artifacts/ann-search/tuning-winners.json
```

tuning report는 precision/recall/Wilson/index/spill/OOM 품질 gate를 통과한 설정 중
K별 p95 최저 설정을 고른다.

## 7. 정책 경계 확인

tuning과 다른 confirmation scenario에서 정책 경계와 인접 탈락 셀을 warm cache
300회 확인한다. p95 gate는 1,000회 bootstrap의 95% 상한도 통과해야 한다.

```bash
.venv/bin/python scripts/benchmark_audience_search.py run-cell \
  --env-file .env.ann-1m \
  --manifest artifacts/ann-search/scenario-manifest.json \
  --scenario-id intent_matched-hard-le-005-confirm-a \
  --phase confirmation \
  --cache-mode warm \
  --warmups 10 --repetitions 300 \
  --sample-sizes 2000,5000,10000,20000,50000 \
  --requested-k 10000,20000 \
  --hnsw 100:strict_order:20000 \
  --output artifacts/ann-search/raw.jsonl --append \
  --confirm-disposable-postgres
```

DB-cold는 외부에서 PostgreSQL을 완전히 재시작한 직후 한 번만 측정한다. 아래
호출을 재시작과 함께 30회 반복한다. runner는 DB-cold에서 warm-up 또는 한 번
초과 반복을 허용하지 않는다.

```bash
.venv/bin/python scripts/benchmark_audience_search.py run-cell \
  --env-file .env.ann-1m \
  --manifest artifacts/ann-search/scenario-manifest.json \
  --scenario-id intent_matched-hard-le-005-confirm-a \
  --phase confirmation --cache-mode db_cold \
  --warmups 0 --repetitions 1 \
  --sample-sizes 10000 \
  --plan ann_first \
  --requested-k 10000 \
  --hnsw 100:strict_order:20000 \
  --output artifacts/ann-search/raw.jsonl --append \
  --confirm-disposable-postgres
```

Exact 기준도 `--plan exact_all` 또는 `--plan filter_first_exact`로 각각 별도
재시작 뒤 실행한다. Exact 명령에는 `--requested-k`를 주지 않는다. DB-cold
runner는 하나의 sample·plan과, ANN이면 하나의 K·HNSW 설정만 허용하며 vector
table을 세는 preflight와 ground-truth 쿼리를 측정 전에 수행하지 않는다.

동일 확인 셀을 deterministic 5% exclusion snapshot과
`--exclusion-ratio 0.05`로 다시 300회 실행한다.

각 셀의 시간은 hard count와 score-pass estimate부터 최종 회원 materialize까지
포함한다. PostgreSQL backend PID가 같은 OS namespace에서 보이면 10ms 간격으로
peak RSS를 자동 측정한다. PID/RSS를 확인할 수 없는 환경에서는 값이 비어 있고,
정책 합성기는 ANN 메모리 gate를 보수적으로 탈락시킨다.

`--diagnostics-dir`를 지정한 warm 실행은 Exact 두 계획과 각 ANN 셀의
`EXPLAIN (ANALYZE, BUFFERS, WAL, SETTINGS, FORMAT JSON)`을 저장하고 같은 실행
구간의 `system.query_log`를 flush·수집한다. query-log 권한/설정이 없으면
`capture_error`가 JSON에 남는다. DB-cold 명령에는 diagnostics를 지정하지 않아
측정 전 cache를 건드리지 않는다.

## 8. 3-segment 실제 요청 workload

production 기본값인 세 suggestion을 한 요청에서 순차 처리한다. 먼저 macro
입력용 preliminary policy를 합성한 뒤 현재 구현과 후보 정책을 각각 실행한다.

```bash
.venv/bin/python scripts/benchmark_audience_search.py synthesize \
  --input artifacts/ann-search/raw.jsonl \
  --scenario-manifest artifacts/ann-search/scenario-manifest.json \
  --dataset-hash '<sha256>' \
  --environment-json artifacts/ann-search/environment.json \
  --output-dir artifacts/ann-search/preliminary

.venv/bin/python scripts/benchmark_audience_search.py run-macro \
  --env-file .env.ann-1m \
  --manifest artifacts/ann-search/scenario-manifest.json \
  --scenario-id intent-confirm-a \
  --scenario-id funnel-confirm-a \
  --scenario-id benefit-confirm-a \
  --mode current_runtime \
  --concurrency 1,4 \
  --warmup-seconds 120 --duration-seconds 900 \
  --output artifacts/ann-search/macro.jsonl --append \
  --confirm-disposable-postgres

.venv/bin/python scripts/benchmark_audience_search.py run-macro \
  --env-file .env.ann-1m \
  --manifest artifacts/ann-search/scenario-manifest.json \
  --scenario-id intent-confirm-a \
  --scenario-id funnel-confirm-a \
  --scenario-id benefit-confirm-a \
  --mode candidate_policy \
  --policy artifacts/ann-search/preliminary/candidate-policy.json \
  --concurrency 1,4 \
  --warmup-seconds 120 --duration-seconds 900 \
  --output artifacts/ann-search/macro.jsonl --append \
  --confirm-disposable-postgres
```

각 worker는 독립 DB connection을 사용한다. `macro-report`는 동시성 1과 4에서
오류율 0, PostgreSQL `temp_bytes` 증가 없음, candidate p95/p99 비악화를 요구한다.
동시 실행 중 관측한 `temp_bytes`는 DB 전체 누적값이라 다른 worker의 spill도 잡는
보수적인 판정이다.

## 9. 정책·fixture 생성

환경 JSON에는 instance/container CPU·RAM, PostgreSQL, pgvector, ClickHouse,
OS와 cold restart 절차를 기록한다. dataset hash는 여섯 cohort manifest와
ground-truth hash를 정렬 결합해 만든 SHA-256을 사용한다.

```bash
.venv/bin/python scripts/benchmark_audience_search.py synthesize \
  --input artifacts/ann-search/raw.jsonl \
  --scenario-manifest artifacts/ann-search/scenario-manifest.json \
  --dataset-hash '<sha256>' \
  --environment-json artifacts/ann-search/environment.json \
  --macro-input artifacts/ann-search/macro.jsonl \
  --output-dir artifacts/ann-search/result

.venv/bin/python scripts/benchmark_audience_search.py validate-policy \
  --policy artifacts/ann-search/result/candidate-policy.json \
  --fixtures artifacts/ann-search/result/implementation-fixtures.json
```

합성기는 확인 phase만 사용해 다음 파일을 만든다.

- `candidate-policy.json`
- `implementation-fixtures.json`
- `current-vs-candidate.json`
- `unvalidated-fallbacks.json`
- `benchmark-summary.json`, `benchmark-summary.csv`
- `current-runtime-work-amplification.json`
- `macro-benchmark-report.json`
- `report.md`

ANN은 precision, recall/Wilson, p95와 bootstrap 상한, Exact/current-runtime 대비
p99, index, spill/OOM, peak RSS, DB-cold, 5% exclusion을 모두 통과해야 한다.
정책 규칙에는 해당 bucket의 `validated_recall`과 보수적인
`validated_recall_lower_bound`가 함께 저장된다. 같은 bucket의 다섯 candidate type이 모두
통과해야 규칙이 생긴다. 정책의 K는 통과 조합 중 가장 작은
`min_candidates`, `k_safety_factor`, `max_corpus_fraction` 순으로 정한다. cap을
넘는 K는 자르지 않고 Exact로 돌아간다. 첫 검증 cohort보다 작은 규모, 인접
경계 불일치, 누락 bucket도 Exact다.

`implementation-fixtures.json`에는 대표 셀뿐 아니라 사용자 수, hard-match
ratio, expected-member ratio의 각 rule 경계가 들어간다. runtime PR은 이 파일을
직접 읽어 정책 selector와 K 계산을 검증한다.

## 완료 판정

다음 조건이 모두 충족되기 전에는 `candidate-policy.json`을 runtime에 반영하지
않는다.

1. 여섯 cohort manifest와 전체 scenario ground truth가 있다.
2. screening, tuning, warm/cold/exclusion confirmation 원본 JSONL과 EXPLAIN JSON이 있다.
3. 모든 ANN 규칙이 warm 300회와 DB-cold 30회 확인 gate를 통과한다.
4. 정책 fixture replay와 저장소 전체 test가 통과한다.
5. 누락 candidate type/ratio/규모가 `unvalidated-fallbacks.json`에 Exact로 남는다.
6. 동시성 1/4의 3-segment macro gate가 통과한다.
7. runtime 변경은 별도 feature-flag PR이며 flag off의 유일 경로가 Exact-all이다.
