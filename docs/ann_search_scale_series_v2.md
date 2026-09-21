# ANN scale-series benchmark v2

Goal 1은 runtime 정책을 만들지 않는 offline scale experiment다. 기존
`artifacts/ann-search/`의 50K 결과는 pilot evidence로만 보존하고, 새 raw,
ground truth, latency는 모두 `artifacts/ann-search/scale-series-v2/` 아래에
기록한다.

## 고정 계약

- experiment: `audience_search.benchmark.v2`
- vector: `hotel_behavior.v2`
- observation window: `[2013-01-01, 2015-01-01)`
- cohort order: `SHA256(seed + "|" + user_id), user_id`
- target sizes: `50K, 100K, 250K, 500K, 750K, 1M`
- P: cohort 내부 hard-match 중 deterministic 최대 50K sample
- 각 cohort는 별도 빈 Data Contract PostgreSQL DB를 사용한다.
- ClickHouse의 production hard-predicate 식은
  `ann_benchmark_scale_membership`의 `cohort_rank <= N`에 먼저 join해 cohort별
  exact user signal을 `ann_benchmark_user_signals`에 동결한다.
- 측정 중 H, P, Filter-first는 이 frozen signal을 membership relation에 다시
  join해 cohort 범위를 강제한다. signal 값은 production 식과 동일하고,
  cohort마다 raw event를 1,000명 단위로 반복 scan하는 repository 구현 비용은
  latency에서 제외한다.

v2 manifest는 `--scale-series-id`, `--scale-cohort-size`,
`--reference-sample-seed` 없이 live benchmark에 사용할 수 없다. 반대로 v1
manifest에는 이 scope를 줄 수 없다. Goal 1은 exclusion confirmation과 macro
benchmark를 실행하지 않는다.

## Phase 0 gate

다음 gate가 모두 통과하기 전에는 source 적재, vector 생성, membership 적재,
cohort DB 생성을 시작하지 않는다.

```bash
.venv/bin/python -m pytest -q tests/test_ann_search_scale_series.py tests/test_ann_search_experiment.py
.venv/bin/python -m pytest -q
git diff --check
```

검증 범위는 cohort 밖 사용자 차단, 정확한 nested prefix, Exact/Filter-first
집합 동등성, H/E bucket 경계, tuning-only survivor 선택, scale/policy survivor
분리, Goal 2 union, fingerprint 재사용 거부, append-only resume, actual-only
report를 포함한다.

### 전체 source vector build 자원

Expedia 전체 source의 production `hotel_behavior.v2` GROUP BY는 final external
aggregation merge에서 32 GiB를 넘는다. 로컬 scale-series 환경은 Docker
Desktop RAM 36 GiB, swap 4 GiB와
`offline_evaluation/config/clickhouse-scale-series-memory.xml`의 명시적 server
ceiling을 함께 사용한다. 이 설정은 production SQL이나 service를 변경하지
않는 benchmark-only 자원 계약이며 `resource-manifest.json`에 실제 적용값을
기록한다. vector build는 `max_threads=1`,
`max_bytes_before_external_group_by=1 GiB`로 실행한다.

## 실행 순서

1. 고정 window의 전체 Expedia source user/row/disk 예상량을 읽기 전용으로
   확인한다.
2. 별도 project로 raw event를 적재하고 production
   `UserBehaviorVectorBatchService`로 v2 vector를 만든다.
3. source revision cutoff를 동결하고 membership을 한 번만 적재한다.
4. 실제 가능한 cohort prefix와 hash를 확정한다.
5. 실제 destination/month/supported benefit pool을 production compiler와
   calibration으로 compile한다.
6. 각 관측 bucket에서 정의가 다른 tuning/confirmation pair를 동결한다.
7. 각 cohort의 full cosine ground truth를 만든 뒤 tuning에는 전체 K screening,
   confirmation에는 baseline K 하나만 순차 실행한다.
8. scale/policy survivor와 중복 제거 Goal 2 union을 만들고 checkpoint,
   actual-only CSV/Markdown/PNG, unvalidated Exact fallback을 기록한다.

일반 측정과 `EXPLAIN` 진단은 분리한다. checkpoint JSONL은 동일 identity의
동일 row만 건너뛸 수 있고, 내용이 다른 중복이나 기존 immutable artifact
덮어쓰기는 실패한다.

### Scale-ready hard-predicate backend

초기 50K scale precheck에서 현재 repository가 1,000 candidate chunk마다 전체
raw window를 다시 읽는 동작을 확인했다. 이 경로로는 corpus scale보다 raw scan
횟수가 latency를 지배해 Goal 1의 ANN/Exact 계획 비교가 불가능하다. benchmark
v2는 다음 경계를 고정한다.

- hard predicate 의미와 사용자 집합은 production 식의 exact 결과다.
- `exact_all`, `filter_first_exact`, `ann_first`는 동일 frozen signal을 사용한다.
- `current_runtime`은 production selector, K 증가, audit, fallback을 유지하되
  hard predicate backend만 동일 frozen signal로 정규화한다.
- 제외된 raw chunk rescan은 `phase0/scale-ready-backend.json`과
  `unvalidated-fallbacks.json`에 명시하고 ANN latency 근거로 사용하지 않는다.

따라서 Goal 1의 current-runtime 결과는 정책 work amplification 비교값이지,
현재 raw-event chunk implementation의 end-to-end latency 재현값은 아니다.

## 종료 경계

Goal 1은 가능한 모든 cohort의 ground truth, baseline, K screening, 두 survivor
목록, Goal 2 union, report, fingerprint 및 artifact validator까지 완료한다.
candidate policy, runtime selector, full 24-setting HNSW tuning, cold/exclusion
confirmation, macro benchmark는 생성하거나 실행하지 않는다.

## Goal 2 — final offline validation

Goal 2는 `scale-series-v2/goal2/`만 쓰고 Goal 1 디렉터리를 read-only 입력으로
취급한다. 시작 전에 `goal1-input-lock.json`으로 다음을 다시 잠근다.

- Goal 1 artifact integrity와 measurement-code SHA
- manifest, tuning input, source/vector/cohort/scenario hash
- 6개 PostgreSQL DB의 active generation, row count, HNSW DDL/valid/ready
- ClickHouse membership prefix hash와 PostgreSQL vector membership set 동등성
- PostgreSQL/pgvector/ClickHouse 버전과 Docker CPU/RAM/SHM 제한

정책 coverage audit은 `N`, `H/N`, `E/N`만 key로 사용하고 candidate type을
추가 차원으로 사용하지 않는다. 다섯 production candidate type의 서로 다른
tuning/confirmation coverage가 없는 bucket은 장애가 아니라 Exact fallback이라는
제품 음성 결과다. 이 경우 exclusion, score-sample, macro branch를 실행하지
않지만 scale-performance track은 계속한다.

Scale track은 Goal 1의 104개 survivor K와 다음 24개 HNSW 설정을 Cartesian
product로 고정한다.

- `ef_search`: `50, 100, 200, 400`
- `iterative_scan`: `strict_order, relaxed_order`
- `max_scan_tuples`: `20K, 50K, 100K`
- 각 cell: warm-up 3회, measured 10회

19개 고유 cohort/scenario point마다 `current_runtime`, `exact_all`,
`filter_first_exact` matched baseline을 같은 횟수로 다시 측정한다. 따라서 core
실행량은 ANN 2,496 cells/32,448 invocations와 baseline 57 cells/741
invocations, 합계 2,553 cells/33,189 invocations로 고정된다.

```bash
.venv/bin/python scripts/run_ann_scale_goal2.py phase0
.venv/bin/python scripts/run_ann_scale_goal2.py plan
.venv/bin/python scripts/run_ann_scale_goal2.py tune
.venv/bin/python scripts/run_ann_scale_goal2.py aggregate
.venv/bin/python scripts/run_ann_scale_goal2.py diagnostics
.venv/bin/python scripts/run_ann_scale_goal2.py finalize-tuning
```

Live timing은 같은 PostgreSQL host에서 직렬 실행한다. raw는
cohort/scenario block별로 격리하고 complete checkpoint의 manifest/hash/count가
모두 일치할 때만 resume한다. EXPLAIN은 timing 밖에서 prediagnostic winner와
대표 인접 탈락 설정에만 수집한다. 진단 증거가 필요한 winner에서 RSS/index/
spill 값이 없으면 보수적으로 gate를 실패시킨다.

`ann_vs_exact_all_passed`, `ann_policy_candidate`, `policy_rule_eligible`은 별도
필드다. Scale winner가 있으면 Goal 1의 다른 confirmation scenario에서 warm
300회, PostgreSQL 완전 재시작 cold 30회를 통과해야 observed break-even으로
인정한다. 정책 coverage가 없는 경우 cell-level `ann_policy_candidate`를 계산할
수 있어도 `policy_rule_eligible=false`, `ann_min_users=null`이며 candidate policy와
macro를 만들지 않는다.
