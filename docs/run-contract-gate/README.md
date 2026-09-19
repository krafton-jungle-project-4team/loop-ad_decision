# Run Contract Gate 문서 안내

| 항목 | 내용 |
|---|---|
| 대상 독자 | Decision 개발자, 구현 에이전트, 리뷰어, 포트폴리오 독자 |
| 상태 | PR 0 개인 통합 병합 완료 · PR 1 로컬 구현·검증 완료 · PR 2 CI 미구현 |
| 기준 revision | 후보 기반 09442f29e8da514df1d1a5f2a52b03646c92e170 / baseline e1de8b2 / Contract 0ec2cef0290f4659ad21ccc1dd2a20df2801ff50 |
| 마지막 확인 | 2026-09-19 KST |

**선택 고객군으로 run을 생성할 때, 실제 PostgreSQL 계약을 통과하고 재시도·실패 이후에도 올바른 데이터가 남는지 로컬과 PR CI에서 반복 검증하는 도구다. 로컬 Gate를 구현했으며 PR CI는 다음 단계다.**

초기 Gate가 **HTTP 200 전송 후 commit 실패·전체 rollback**을 발견했다. 이를 선행 서비스 [PR #394](https://github.com/krafton-jungle-project-4team/loop-ad_decision/pull/394)로 분리해 `integration/run-contract-gate`에 병합했다. 기존 PR 1 작업 파일을 보존하고 통합 commit을 반영한 뒤 전체 로컬 Gate를 구현했다.

[기본 명령](developer-guide.md#2-로컬-기본-실행)은 고정·최신 DDL 각각 17개 DB 시나리오와 30개 제어 검사를 실행하고 JSON/JUnit·소스 증거·정리 결과를 남긴다. 고정 baseline의 실제 API가 만든 행과 provenance도 보존했다. 문서의 E-12는 커밋 전 로컬 검증 근거이며 제출 commit 재검증 결과는 PR 본문에 기록한다. CI·실제 동시성·Dashboard 실행은 미검증이다. [최신 실행 근거](evidence-log.md#e-12-pr-1-로컬-gate-검증)

## 읽는 순서

| 목적 | 읽을 문서 | 얻을 내용 |
|---|---|---|
| 개발 시작 | [개발 계획](implementation-plan.md) | 문제, 선택 이유, 구현 순서, PR 경계, 중단 기준 |
| 테스트 구현·검토 | [검증 명세](verification-spec.md) | 필수 시나리오, fixture, 결과 판정, 미래 테스트 대응 |
| 도구 사용·실패 조사 | [개발자 사용 안내](developer-guide.md) | 검증한 실행법, 결과 해석, 갱신·정리 절차 |
| 원리 이해 | [학습 안내](learning-guide.md) | 여섯 단원의 코드 읽기·질문·해설 |
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
| Gate 테스트·컨테이너 실행기·CI | 로컬 Gate 완료; PR 2 CI 미구현 |
| 기준 코드가 생성한 기존 row fixture | 생성·새 DB 복원·baseline/candidate 재사용 확인 |
| 로컬·CI 실행 결과와 소요 시간 | fixed/latest 각각 17 passed, controls 30 passed; CI 미실행. 시간은 E-12 |
| PR / dev Draft | PR 0 #394 병합 완료; PR 1 제출 단계; PR 2·dev Draft 미생성 |

## 구현·제출 경계

PR 0은 응답 전 commit을 보장하는 선행 서비스 수정, PR 1은 로컬에서 완결된 Gate, PR 2는 같은 명령의 CI 적용이다. PR 0 이후 PR 1·PR 2를 개인 통합 브랜치에 반영하고 최종 commit을 검증한 뒤 dev로 Draft PR 하나를 제출한다. 실제 동시성 검증은 **필수 후속 개발**이다. Dashboard 실제 소비 코드 연결은 후속 단계다.

현재 merge-safety verdict는 **incomplete evidence**다. 로컬 Gate 범위는 통과했지만 CI·Dashboard 배포 여정까지 승인한 판정은 아니다.

운영 DB·credential·개인정보를 사용하지 않는다. AGENTS.md, agent/, .codex/는 변경·제출 대상에서 제외한다.
