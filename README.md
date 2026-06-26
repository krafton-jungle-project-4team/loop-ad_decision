# loop-ad_decision
loop-ad 프로젝트 사용자 행동 분석 및 마케팅 액션 추천하는 AI레포입니다.

## Docker Compose 실행

AI 분석 서버와 로컬 ClickHouse, PostgreSQL을 함께 실행합니다.

```bash
cp .env.example .env.local
docker compose --env-file .env.local up decision
```

ClickHouse만 실행하려면 다음 명령을 사용합니다.

```bash
docker compose --env-file .env.local up -d clickhouse
```

서비스 상태와 헬스체크는 다음 명령으로 확인할 수 있습니다.

```bash
docker compose ps
curl http://localhost:8000/health
curl http://localhost:8123/ping
```

서버는 `LOOPAD_*` env contract를 시작 시 검증합니다. 로컬 값은 `.env.local` 또는 개인 shell 환경에서 관리하고 commit하지 않습니다. PostgreSQL 테이블은 `LOOPAD_POSTGRES_AUTO_CREATE_TABLES=true`일 때 서버 시작 시 자동 생성됩니다.

## 로컬 데이터베이스 초기화 스크립트

초기화 SQL은 테이블 생성과 seed 적재를 분리합니다.

- PostgreSQL: `scripts/postgres/postgres-init.sql`, `scripts/postgres/postgres-seed.sql`
- ClickHouse: `scripts/clickhouse/clickhouse-init.sql`, `scripts/clickhouse/clickhouse-seed.sql`

ClickHouse seed는 `ga4_exports/ga4_events_*.csv`를 읽어 새 `events` 스키마의 실험/광고 컬럼을 기본값으로 채웁니다. 기존 ClickHouse 볼륨을 새 스키마로 다시 만들 때는 볼륨을 제거한 뒤 `docker compose up -d clickhouse`를 실행합니다.
