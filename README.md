# loop-ad_decision
loop-ad 프로젝트 사용자 행동 분석 및 마케팅 액션 추천하는 AI레포입니다.

## Docker Compose 실행

AI 분석 서버를 실행합니다. 로컬 데이터베이스는 `loop-ad_local-data-source_contract` 레포의 규약대로 별도로 실행합니다.

```bash
cp .env.example .env.local
docker compose --env-file .env.local up decision
```

서비스 상태와 헬스체크는 다음 명령으로 확인할 수 있습니다.

```bash
docker compose ps
curl http://localhost:8000/health
```

서버는 `LOOPAD_*` env contract를 시작 시 검증합니다. 로컬 값은 `.env.local` 또는 개인 shell 환경에서 관리하고 commit하지 않습니다. PostgreSQL 테이블은 `LOOPAD_POSTGRES_AUTO_CREATE_TABLES=true`일 때 서버 시작 시 자동 생성됩니다.
