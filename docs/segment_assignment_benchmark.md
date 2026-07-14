# 사용자 세그먼트 assignment exact/ANN 검증

이 도구는 production assignment를 변경하지 않고 frozen corpus에서 exact oracle,
고정 FAISS HNSW, adaptive HNSW + exact rescue를 비교한다. `offline_evaluation` 모듈은
Decision repository, DB writer, publication pointer를 import하지 않는다. Shadow 결과도 로컬
JSON report만 만들며 `user_segment_assignments`나 serving 경로에는 쓰지 않는다.

## 설치

Benchmark dependency는 production dependency와 분리되어 있다.

```bash
python -m venv .venv
.venv/bin/pip install -e '.[dev,assignment-benchmark]'
```

현재 extra는 NumPy, `faiss-cpu`, psutil, threadpoolctl을 설치한다. CLI는 기본적으로
`--blas-threads 1 --faiss-threads 1`을 사용한다. NumPy/FAISS를 처음 읽기 전에 OpenMP,
OpenBLAS, MKL, BLIS, NumExpr와 Apple Accelerate용 thread 환경을 설정하고, 각 fresh child에서
threadpoolctl이 인식한 BLAS pool도 같은 값으로 제한한다. 보고서에는 실제
NumPy BLAS build backend, NumPy/FAISS/psutil 버전, CPU, logical CPU 수, 환경 변수,
threadpoolctl이 관측한 BLAS/OMP runtime pool과 FAISS thread 설정을 기록한다.
Apple Accelerate는 threadpoolctl이 runtime pool을 노출하지 않으므로
`VECLIB_MAXIMUM_THREADS`를 NumPy import 전에 설정하지만 report에는
`environment_only_unverified`로 표시한다. OpenMP 기반 OpenBLAS와 FAISS가 같은 runtime을
공유하면 서로 다른 BLAS/FAISS thread 값이 결합될 수 있다. 공식 비교 결과는 두 값을 같은
수(기본값 `1/1`)로 실행하고, 다른 조합은 backend-dependent 탐색 결과로만 취급한다.
Corpus를 만든 commit과 benchmark를 실행한 코드 revision은 별도다. 각 benchmark report의
`benchmark_code_identity`에는 현재 commit/tree/branch, 관련 source dirty 여부, 통합
SHA-256과 파일별 SHA-256이 기록된다. 따라서 이전 commit에서 export한 `--corpus`를
실행해도 현재 matcher/runner 코드가 무엇이었는지 구분할 수 있다.
Shadow report도 같은 corpus manifest, code identity, 실행 환경과 실제 single-process
실행 방식을 함께 기록하므로 report JSON 하나만으로 입력과 실행 코드를 추적할 수 있다.

## Frozen corpus

Corpus는 JSONL 한 파일이다. 첫 줄은 manifest이고 이후 user/segment vector가 이어진다.

- 모든 vector는 finite, non-zero, 정확히 64차원이어야 한다.
- matcher 입력과 hash 대상은 L2-normalized little-endian contiguous float32다.
- 실제 user ID는 domain-separated SHA-256 pseudonym으로만 저장한다.
- segment ID는 production 동점 규칙(`segment_id` 사전순)을 재현하기 위해 원문을
  보존한다. Segment ID는 사용자 식별자가 아니며 artifact 전체는 Git ignore 경로에 둔다.
- raw 입력을 export할 때는 `--id-hash-salt-file`이 필수이며 salt는 artifact에 저장하지 않는다.
- corpus SHA-256은 정렬된 pseudonymous user ID와 lexical segment ID, normalized
  float32 bytes, format version, dimension, vector version, source cutoff, distribution,
  seed, provenance mode, cutoff attestation을 포함한다.
- ZIP/container bytes나 생성 시각은 corpus hash에 포함하지 않는다.
- manifest에는 user/segment 수, D=64, vector version, source cutoff, distribution, seed,
  provenance/cutoff attestation, Git commit, NumPy/FAISS 버전, CPU/thread 설정, matcher
  config가 남는다.
- synthetic corpus는 `synthetic_generator` provenance와 실제 seed를 기록한다. Raw JSONL은
  항상 distribution `provided`, seed `null`이며, exporter가 row timestamp를 검증할 수
  없으므로 호출자가 cutoff 이전 자료임을 명시적으로 attest해야 한다.
- legacy v2 corpus는 읽을 수 있지만 provenance와 cutoff를 추론하지 않고
  `legacy_v2_unattested`로 표시한다.

Deterministic synthetic corpus:

```bash
.venv/bin/python scripts/export_segment_assignment_corpus.py \
  --output artifacts/assignment-benchmark/corpus-random.jsonl \
  --users 10000 \
  --segments 256 \
  --distribution random \
  --seed 20260714
```

Raw JSONL은 `{"kind":"user|segment","id":"...","vector":[64 values]}` 형식이다.
원본 corpus와 salt 파일은 Git에 추가하지 않는다. `artifacts/`는 ignore 대상이다.
Raw 입력 export는 다음처럼 cutoff와 attestation을 모두 요구한다.

```bash
.venv/bin/python scripts/export_segment_assignment_corpus.py \
  --input-jsonl /secure/path/assignment-input.jsonl \
  --id-hash-salt-file /secure/path/id-hash-salt.txt \
  --source-cutoff 2026-07-01T00:00:00Z \
  --attest-source-cutoff \
  --output artifacts/assignment-benchmark/provided.jsonl
```

## Matcher 의미

Exact oracle은 segment ID를 사전순으로 정렬한 뒤 normalized float32 batch matrix
multiplication으로 top-1을 구한다. 동점은 사전순 segment ID가 이기며 raw cosine이
`0.65`보다 작을 때만 fallback이다. 작은 corpus는 FAISS `IndexFlatIP` 전체 검색을 다시
NumPy exact-rerank하여 oracle과 교차 검증한다.

고정 HNSW도 FAISS 반환 순서와 score를 최종 결과로 신뢰하지 않는다. 유효 candidate ID를
중복 제거·검증한 후 NumPy exact score로 rerank한다.

Adaptive HNSW는 다음 조건의 합집합 사용자만 전체 segment에 대해 exact rescue한다.

- candidate가 `min(K, S)`보다 적음
- ANN label/score가 invalid, 중복 또는 범위 밖임
- top score가 threshold band 안에 있음
- top1-top2 margin이 설정값 이하임

`K`, `M`, `efConstruction`, `efSearch`, threshold band, margin, batch size, BLAS/FAISS thread 수는
`MatcherConfig`에 있으며 각 report에 직렬화된다. `K == S`처럼 candidate rerank 자체가
full exact인 경우 threshold/margin 중복 rescue는 하지 않는다.

## 실행

기본 smoke는 U=256, S=256과 `random`, `clustered`, `threshold_near`, `low_margin` 네
분포를 실제 실행한다.

```bash
.venv/bin/python scripts/benchmark_segment_assignments.py \
  --output-dir artifacts/assignment-benchmark/smoke
```

Frozen corpus 한 개와 read-only shadow report를 함께 실행할 수도 있다.

```bash
.venv/bin/python scripts/benchmark_segment_assignments.py \
  --corpus artifacts/assignment-benchmark/corpus-random.jsonl \
  --shadow \
  --output-dir artifacts/assignment-benchmark/corpus-run
```

이 실행의 summary mode는 `provided_corpus`이며 S matrix를 실행한 것이 아니므로 crossover를
계산하지 않는다. Shadow report는 mismatch example limit, 전체 mismatch user 수, example
truncation 여부를 함께 기록하며 production assignment나 publication에는 쓰지 않는다.

Full matrix 명령은 다음과 같다.

```bash
.venv/bin/python scripts/benchmark_segment_assignments.py \
  --full \
  --timing-trials 3 \
  --blas-threads 1 \
  --faiss-threads 1 \
  --output-dir artifacts/assignment-benchmark/full
```

Full matrix는 다음 32개 case를 실행한다.

- U: 10K, 100K
- S: 50, 256, 1K, 5K
- D: 64
- distribution: random, clustered, threshold-near, low-margin

`100K x 5K` 전체 score matrix는 만들지 않고 user batch별 matrix multiplication을 한다.
Full 실행은 CPU와 메모리를 많이 쓴다. 정상 종료 시 실제 실행한 case만 `summary.json`에
기록하며 실행하지 않은 수치나 crossover는 생성하지 않는다. 실행 도중 중단되면 이미 쓴
개별 report는 부분 증거로 남지만 `summary.json`은 `status: incomplete` marker로 남는다.
`status: complete` summary만 공식 결과로 사용한다. `--force` 재실행도 시작 시 기존 summary를
먼저 이 marker로 교체하므로 새 report와 오래된 complete summary가 섞이지 않는다. 현재 runner는 중단
지점 resume을 지원하지 않으므로 같은 output directory에서 다시 시작할 때는 `--force`로
전체 matrix를 재실행하거나 새 output directory를 사용하고, 정상 종료한 summary만 공식
결과로 사용한다.

## 지표

각 matcher report는 다음을 기록한다.

- exact top-1 candidate recall
- rescue 전/후 raw top segment agreement
- 최종 assignment agreement와 fallback agreement
- false fallback / false non-fallback count
- rescue count/rate와 조건별 중첩 count
- median users/sec
- trial별 index build + match + output assembly의 end-to-end p50/p95와 표본 수
- index build, match, output assembly 각각의 trial median
- 전체 trial에서 pooled한 warmup 제외 실제 `matcher.match` batch wall-time p50/p95와 표본 수
- matcher별 격리 child process peak RSS
- observed disagreement count/rate
- one-sided Wilson 95% disagreement upper bound

End-to-end P50/P95는 matcher마다 독립 fresh `spawn` child에서 측정한 trial 표본이다.
각 표본은 index build, warmup 제외 measured match batch 전체, output assembly를 포함하고
spawn 시작, corpus IPC/역직렬화, oracle cross-check, 결과 IPC는 제외한다.
`end_to_end_seconds`와 index/match/assembly scalar는 trial median이다. 별도의 batch P50/P95는
개별 user latency가 아니라 모든 trial에서 pooled한 report batch size의 `matcher.match`
wall-time이며 index build와 output assembly를 제외한다. 기본 timing trial은 3회이고 report에
warmup user 수, trial 수, E2E 표본 수, batch 표본 수를 함께 남긴다. 세 표본은 분포 추정에
작으므로 smoke 결과만으로 성능 승격을 판정하지 않는다. Report는 각 raw trial과
NumPy linear percentile 방법을 함께 기록한다. 요청한 warmup user 수와 실제
`min(requested, corpus user count)`도 구분한다. Matcher 순서는 trial마다 exact/fixed/adaptive를
deterministic rotation하여 고정 순서의 열·주파수 편향을 줄이며 schedule 자체를 report에 남긴다.
각 matcher trial은 새 `spawn` child process에서 순차 실행된다. Peak RSS는 해당 matcher의 모든 trial child 중 최댓값이며 각 child의
corpus 역직렬화·정규화 vector·index·batch output·combined output을 포함하는 absolute
high-water 값이며 parent와 다른 matcher의 과거 peak를 포함하지 않는다. IPC 직렬화는
측정에서 제외한다. OS RSS는 shared library page를 각 child에 중복 계상할 수 있으므로
report의 정의와 함께 해석한다. 각 matcher report에는 child에서 직접 읽은 실제 thread
설정과 process ID도 남는다.

`summary.json`은 동일 U/distribution의 각 S에서 repeated-trial `end_to_end_p95_ms`가 exact보다
빨랐는지를 모두 기록한다. `observed_bracketed_crossover_segment_count`는 바로 아래 S에서
느렸고 해당 S부터 측정된 모든 더 큰 S에서 빨랐을 때만 채운다. 가장 작은 측정 S부터
빨랐다면 `unbracketed_faster_at_minimum`, 결과가 뒤집히면 `non_monotonic_or_unstable`로
표시한다. `null`은 더 큰 S에서도 crossover가 없다는 뜻이 아니다.

## ANN 승격 gate

ANN을 production primary로 바꾸려면 대표 frozen corpus의 각 S/distribution 구간에서
다음을 모두 만족해야 한다.

- adaptive rescue 후 observed disagreement 0
- fallback agreement 100%
- underfill이 전부 exact rescue 또는 명시적 실패
- 충분한 batch 표본에서 exact보다 `match_batch_latency_p95_ms`가 개선되고, 별도 index
  build와 전체 E2E p95도 허용 범위인 구간 확인
- index build 시간과 peak RSS가 worker 한도 이내
- corpus SHA, manifest, Git commit, 라이브러리/CPU/thread 설정으로 재현 가능
- 표본 수와 95% upper bound를 함께 제시

이 smoke 결과만으로 ANN을 승격하지 않는다. 통과하지 못했거나 실행하지 않은 S 구간은
exact를 유지한다. Raw cosine은 benchmark 비교에 사용하고, PostgreSQL `[0,1]` 저장 clamp와
혼합하지 않는다.
