# Run Contract Gate 문서 안내

| 항목 | 내용 |
|---|---|
| 대상 독자 | Decision 개발자, 구현 에이전트, 리뷰어, 포트폴리오 독자 |
| 상태 | PR 0~3 통합 완료 · Dashboard #246/#247 병합 완료 · 최신 dev 로컬 통합 완료 · 최종 SHA Gate/consumer 재검증 대기 |
| 기준 revision | dev `dd55b38` + integration `61f7e03` → 로컬 merge `004e3e7` / 기존 검증 producer `9ace3b6` / baseline `e1de8b2` / Contract `0ec2cef` |
| 마지막 확인 | 2026-09-22 KST |

**선택 고객군으로 run을 생성할 때, 실제 PostgreSQL 계약을 통과하고 재시도·실패 이후에도 올바른 데이터가 남는지 로컬과 PR CI에서 반복 검증하는 도구다. Decision의 실제 경합·원본 응답 bundle과 Dashboard의 실제 client→공유 변환→launch 소비 검증까지 구현·병합했다. 최신 dev를 통합한 후보의 신규 Gate/consumer 실행과 dev PR/CI는 다음 단계다.**

초기 Gate가 **HTTP 200 전송 후 commit 실패·전체 rollback**을 발견했다. 이를 선행 서비스 [PR #394](https://github.com/krafton-jungle-project-4team/loop-ad_decision/pull/394)로 분리해 `integration/run-contract-gate`에 병합했다. 기존 PR 1 작업 파일을 보존하고 통합 commit을 반영한 뒤 전체 로컬 Gate를 구현했다.

[기본 명령](developer-guide.md#2-로컬-기본-실행)은 고정·최신 DDL 각각 21개 DB 시나리오와 54개 제어 검사를 실행하고 JSON/JUnit·소스 증거·정리 결과를 남긴다. 고정 baseline의 실제 API가 만든 행과 provenance도 보존했다. PR #394~#399는 `integration/run-contract-gate`에 반영됐고, Dashboard #246/#247은 `main`에 병합됐다. 이 결과와 소비 artifact 검증은 [E-17](evidence-log.md#e-17-pr-3a-구현과-실제-경합bundle-검증) 후반에 기록했다. 최신 dev 통합 자체는 완료했지만 그 최종 SHA의 Gate/consumer 재검증은 아직 실행하지 않았다.

## 읽는 순서

| 목적 | 읽을 문서 | 얻을 내용 |
|---|---|---|
| 개발 시작 | [개발 계획](implementation-plan.md) | 문제, 선택 이유, 구현 순서, PR 경계, 중단 기준 |
| 테스트 구현·검토 | [검증 명세](verification-spec.md) | 필수 시나리오, fixture, 결과 판정, 미래 테스트 대응 |
| 도구 사용·실패 조사 | [개발자 사용 안내](developer-guide.md) | 검증한 실행법, 결과 해석, 갱신·정리 절차 |
| 원리 이해 | [학습 안내](learning-guide.md) | 기존 Gate 1~6단원, PR 3A·3B 경계 7~9단원의 예시·질문·해설 |
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
| Gate 테스트·컨테이너 실행기·CI | 구현 및 integration CI 검증 완료; 최신 dev 통합 SHA 재검증 대기 |
| 기준 코드가 생성한 기존 row fixture | 생성·새 DB 복원·baseline/candidate 재사용 확인 |
| 로컬·CI 실행 결과와 소요 시간 | PR 3A 로컬 fixed/latest 각 21 passed, controls 54 passed; E-17. 기존 PR 2 CI는 E-15 |
| PR / dev Draft | Decision #394~#399 integration 병합 완료; Dashboard #246/#247 main 병합 완료; dev Draft·최종 CI는 미제출 |

## 구현·제출 경계

PR 0은 응답 전 commit 서비스 수정, PR 1은 로컬 Gate, PR 2는 같은 명령의 CI 적용이다. PR 3은 두 저장소의 PR을 하나의 milestone으로 연결한다.

| 작업 | 저장소 | 끝까지 검증할 경계 | 현재 상태 |
|---|---|---|---|
| PR 3A | Decision | 복수 실제 DB connection의 경합·commit/rollback → 원본 응답·결과 artifact | 구현·로컬/CI 검증 완료 |
| PR 3B | Dashboard | 원본 artifact → 실제 client → 공유 변환 → 실제 launch의 다음 operation 인자 | #246/#247 병합·CI/artifact 검증 완료 |
| PR 3 milestone | 두 저장소의 증거 연결 | Decision/Dashboard/Contract revision과 실제 소비 bundle hash가 연결된 결과 | 기존 revision 조합 verified; 최신 dev 후보 재검증 대기 |

**개발자·AI의 시작점:** [계획 21절](implementation-plan.md#21-pr-3-milestone-실행-계약) → [새 검증 기준](verification-spec.md#pr-3-검증-경계--3a3b-구현) → [인수인계 절차](developer-guide.md#9-pr-3a에서-pr-3b로-넘기는-절차). 먼저 학습하려면 [7~9단원](learning-guide.md#7-pr-3a-동시-시작과-실제-경합은-다르다)을 읽는다. 완료 여부는 [E-16 대장](evidence-log.md#e-16-pr-3-milestone-결정과-증거-대장)으로 확인한다.

3B는 run-consumer 통합 검증이며 Dashboard main에 반영됐다. 다만 브라우저 전체 E2E·실제 assignment/start/발송·배포 smoke는 여전히 비범위다. 기존 revision 조합의 PR 3 milestone 검증과 최신 dev 통합 후보의 재검증을 구분한다. 이번 단계에서는 dev 통합과 문서 정리만 완료했으며, 최종 SHA Gate/consumer·dev PR/CI는 아직 성공으로 표시하지 않는다.

운영 DB·credential·개인정보를 사용하지 않는다. AGENTS.md, agent/, .codex/는 변경·제출 대상에서 제외한다.
