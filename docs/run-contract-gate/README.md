# Run Contract Gate 문서 안내

| 항목 | 내용 |
|---|---|
| 대상 독자 | Decision 개발자, 구현 에이전트, 리뷰어, 포트폴리오 독자 |
| 상태 | PR 0·1·2·milestone 문서 통합 완료 · PR 3A 로컬 PASS · PR 3A CI 제출 전 · PR 3B pending |
| 기준 revision | PR 3A base 6de82a36ddc1cf51c88a431d65e84f873ea8f472 / baseline e1de8b2 / Contract 0ec2cef0290f4659ad21ccc1dd2a20df2801ff50 |
| 마지막 확인 | 2026-09-19 KST |

**선택 고객군으로 run을 생성할 때, 실제 PostgreSQL 계약을 통과하고 재시도·실패 이후에도 올바른 데이터가 남는지 로컬과 PR CI에서 반복 검증하는 도구다. 로컬 Gate와 같은 명령의 PR CI workflow를 구현했다. PR 2의 실제 Actions도 통과했다. PR 3A는 실제 동시 경합과 원본 응답 bundle을 구현했고, PR 3B 소비 검증은 후속 범위다.**

초기 Gate가 **HTTP 200 전송 후 commit 실패·전체 rollback**을 발견했다. 이를 선행 서비스 [PR #394](https://github.com/krafton-jungle-project-4team/loop-ad_decision/pull/394)로 분리해 `integration/run-contract-gate`에 병합했다. 기존 PR 1 작업 파일을 보존하고 통합 commit을 반영한 뒤 전체 로컬 Gate를 구현했다.

[기본 명령](developer-guide.md#2-로컬-기본-실행)은 고정·최신 DDL 각각 21개 DB 시나리오와 54개 제어 검사를 실행하고 JSON/JUnit·소스 증거·정리 결과를 남긴다. 고정 baseline의 실제 API가 만든 행과 provenance도 보존했다. PR 1 #395는 개인 통합에 병합됐다. PR 2 #396의 clean 로컬·CI 성공과 개인 통합 병합은 [E-15](evidence-log.md#e-15-pr-2-ci-성공과-개인-통합-병합)에 기록했다. PR 3A의 실제 경합·응답 bundle·timeout/중단 정리 결과는 [E-17](evidence-log.md#e-17-pr-3a-구현과-실제-경합bundle-검증)에 기록했다. Dashboard 소비는 아직 실행하지 않았다.

## 읽는 순서

| 목적 | 읽을 문서 | 얻을 내용 |
|---|---|---|
| 개발 시작 | [개발 계획](implementation-plan.md) | 문제, 선택 이유, 구현 순서, PR 경계, 중단 기준 |
| 테스트 구현·검토 | [검증 명세](verification-spec.md) | 필수 시나리오, fixture, 결과 판정, 미래 테스트 대응 |
| 도구 사용·실패 조사 | [개발자 사용 안내](developer-guide.md) | 검증한 실행법, 결과 해석, 갱신·정리 절차 |
| 원리 이해 | [학습 안내](learning-guide.md) | 기존 Gate 1~6단원, PR 3A 구현·3B 계획 7~9단원의 예시·질문·해설 |
| 사실 확인·진행 기록 | [근거 기록](evidence-log.md) | 조사 revision, 기존 기여, 이번 변경, 미측정 결과 |
| 제출·면접 준비 | [포트폴리오 초안](portfolio-case-study.md) | 현재 사용할 표현, 완료 후 채울 결과, 설명·데모·질문 |

처음 배우는 사람은 이 안내 → 학습 안내 → 개발 계획 → 검증 명세 순서로 읽는다. 구현자는 개발 계획 → 검증 명세 → 개발자 사용 안내 순서로 읽는다.

## 어디에 무엇을 기록하는가

- 설계와 작업 범위의 기준: 개발 계획.
- 시나리오와 합격 판정의 기준: 검증 명세.
- 명령과 사용 절차의 기준: 개발자 사용 안내.
- 실행 사실·측정값·기여 근거의 기준: 근거 기록.
- 학습 안내와 포트폴리오는 위 문서를 설명하고 인용한다. 새로운 판정 규칙을 만들지 않는다.
- 생성될 JSON·JUnit은 실제 실행 artifact다. 근거 기록은 그 artifact를 가리키는 사람이 편집하는 기록이며 자동 Markdown 보고서 생성기를 뜻하지 않는다.

## 현재 상태

| 항목 | 상태 |
|---|---|
| 실제 service·repository·transaction 경로 조사 | 확인 |
| 원격 Decision dev / Contract main SHA 익명 조회 | 확인 — [조회 범위](evidence-log.md#현재-조사-기록) |
| 새 worktree와 로컬 브랜치 | 준비 |
| Gate 테스트·컨테이너 실행기·CI | PR 3A 로컬 Gate·bundle PASS; 3A CI 제출 전·3B pending |
| 기준 코드가 생성한 기존 row fixture | 생성·새 DB 복원·baseline/candidate 재사용 확인 |
| 로컬·CI 실행 결과와 소요 시간 | PR 3A 로컬 fixed/latest 각 21 passed, controls 54 passed; E-17. 기존 PR 2 CI는 E-15 |
| PR / dev Draft | PR 0 #394·PR 1 #395·PR 2 #396·문서 #397 통합 병합; PR 3A 제출 준비, merge 제외 |

## 구현·제출 경계

PR 0은 응답 전 commit 서비스 수정, PR 1은 로컬 Gate, PR 2는 같은 명령의 CI 적용이다. PR 3은 두 저장소의 PR을 하나의 milestone으로 연결한다.

| 작업 | 저장소 | 끝까지 검증할 경계 | 현재 상태 |
|---|---|---|---|
| PR 3A | Decision | 복수 실제 DB connection의 경합·commit/rollback → 원본 응답·결과 artifact | 구현·로컬 검증 완료 |
| PR 3B | Dashboard | 원본 artifact → 실제 client → 공유 변환 → 실제 launch의 다음 operation 인자 | 계획 확정·미구현 |
| PR 3 milestone | 두 저장소의 증거 연결 | Decision/Dashboard/Contract revision과 실제 소비 bundle hash가 연결된 결과 | 3A local verified / 3B pending |

**개발자·AI의 시작점:** [계획 21절](implementation-plan.md#21-pr-3-milestone-실행-계약) → [새 검증 기준](verification-spec.md#pr-3-검증-경계--3a-구현-3b-계획) → [인수인계 절차](developer-guide.md#9-pr-3a에서-pr-3b로-넘기는-절차). 먼저 학습하려면 [7~9단원](learning-guide.md#7-pr-3a-동시-시작과-실제-경합은-다르다)을 읽는다. 완료 여부는 [E-16 대장](evidence-log.md#e-16-pr-3-milestone-결정과-증거-대장)으로 확인한다.

3B는 run-consumer 통합 검증이다. 브라우저 전체 E2E·실제 assignment/start/발송·배포 smoke까지 실행하는 계획은 아니다. 서비스 결함은 별도 수정 PR로 분리한다. PR 3A만 통과하면 전체 milestone을 완료 처리하지 않는다.

현재 PR 3 전체 merge-safety verdict는 **incomplete evidence**다. 3A의 로컬 성공을 미실행인 3B 소비 검증의 성공으로 확대하지 않는다. 각 저장소의 dev 제출·merge는 검증 상태와 별도로 관리한다.

운영 DB·credential·개인정보를 사용하지 않는다. AGENTS.md, agent/, .codex/는 변경·제출 대상에서 제외한다.
