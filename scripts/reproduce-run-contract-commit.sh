#!/usr/bin/env bash
# Bounded baseline reproducer only; not the unfinished fixed/latest Gate entrypoint.
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
CONTRACT_REPO=${1:?Usage: reproduce-run-contract-commit.sh LOCAL_CONTRACT_REPO [NEW_OUTPUT_DIRECTORY]}
BASELINE=e1de8b29b902b54df3a58f21f1daa27c1171fe80
CONTRACT=0ec2cef0290f4659ad21ccc1dd2a20df2801ff50
PG_IMAGE=pgvector/pgvector:0.8.0-pg16@sha256:a132765ec351c65111b5b675928a3a0515a466a40f97277329db8b8209ad8bc9
if [ "$#" -ge 2 ]; then mkdir "$2"; OUTPUT=$(cd "$2" && pwd); else OUTPUT=$(mktemp -d "${TMPDIR:-/tmp}/rcg-repro-output.XXXXXX"); fi
WORK=$(mktemp -d "${TMPDIR:-/tmp}/rcg-repro-work.XXXXXX")
RUN_ID="rcg-$(basename "$WORK" | tr '[:upper:]' '[:lower:]')"
DB="$RUN_ID-db"; RUNNER="$RUN_ID-runner"; SOCKET="$RUN_ID-socket"
LABEL=org.loopad.run-contract-gate
ENDPOINT=${DOCKER_HOST:-$(docker context inspect --format '{{.Endpoints.docker.Host}}')}
case "$ENDPOINT" in unix://*) ;; *) echo 'Only a local Unix Docker socket is supported.' >&2; exit 2;; esac
mkdir -p "$WORK/docker-config" "$WORK/source" "$WORK/repro/run_contract_gate" "$WORK/context/tools/run_contract_gate"
D=(docker --config "$WORK/docker-config" --host "$ENDPOINT")
cleanup() {
    code=$?; trap - EXIT INT TERM
    failed=0
    for name in "$RUNNER" "$DB"; do
        if owner=$("${D[@]}" inspect --format "{{index .Config.Labels \"$LABEL\"}}" "$name" 2>/dev/null); then
            if [ "$owner" = "$RUN_ID" ]; then "${D[@]}" rm -f -v "$name" >> "$OUTPUT/cleanup.log" 2>&1 || failed=1; else failed=1; fi
        fi
    done
    if owner=$("${D[@]}" volume inspect --format "{{index .Labels \"$LABEL\"}}" "$SOCKET" 2>/dev/null); then
        if [ "$owner" = "$RUN_ID" ]; then "${D[@]}" volume rm "$SOCKET" >> "$OUTPUT/cleanup.log" 2>&1 || failed=1; else failed=1; fi
    fi
    if [ "$failed" -eq 0 ]; then echo 'cleanup: owned containers and socket volume removed' >> "$OUTPUT/cleanup.log"; else code=2; fi
    rm -rf "$WORK"
    echo "Reproducer exit: $code; artifacts: $OUTPUT"
    exit "$code"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
# Export immutable tracked application bytes only. No env, Git metadata or candidate app changes.
git -C "$ROOT" archive "$BASELINE" app pyproject.toml | tar -x -C "$WORK/source"
git -C "$CONTRACT_REPO" show "$CONTRACT:postgres/schema.sql" > "$WORK/schema.sql"
cp "$ROOT/tests/run_contract_gate/seed.py" "$ROOT/tests/run_contract_gate/test_run_db.py" "$ROOT/tests/run_contract_gate/__init__.py" "$WORK/repro/run_contract_gate/"
cp "$ROOT/tools/run_contract_gate/requirements.lock" "$WORK/context/tools/run_contract_gate/requirements.lock"
cp "$ROOT/Dockerfile.run-contract-gate" "$WORK/context/Dockerfile.run-contract-gate"
# Context is explicitly assembled; independent of .dockerignore support in the daemon.
LOCK_HASH=$(shasum -a 256 "$WORK/context/tools/run_contract_gate/requirements.lock" | awk '{print $1}')
IMAGE="rcg-repro:$LOCK_HASH"
{
    echo "baseline=$BASELINE"; echo "contract=$CONTRACT"; echo "run_id=$RUN_ID"
    echo "candidate_head=$(git -C "$ROOT" rev-parse HEAD)"; echo 'candidate application is not exercised by this baseline reproducer'
    shasum -a 256 "$WORK/schema.sql" "$WORK/context/tools/run_contract_gate/requirements.lock" "$WORK/repro/run_contract_gate/"*.py
} > "$OUTPUT/provenance.txt"
"${D[@]}" build -f "$WORK/context/Dockerfile.run-contract-gate" -t "$IMAGE" "$WORK/context" > "$OUTPUT/build.log" 2>&1
"${D[@]}" pull "$PG_IMAGE" > "$OUTPUT/postgres-image.log" 2>&1
"${D[@]}" image inspect "$IMAGE" "$PG_IMAGE" --format '{{.Id}} {{.Os}}/{{.Architecture}} {{json .RepoDigests}}' >> "$OUTPUT/provenance.txt"
"${D[@]}" volume create --label "$LABEL=$RUN_ID" "$SOCKET" > "$OUTPUT/resources.txt"
"${D[@]}" create --name "$DB" --label "$LABEL=$RUN_ID" --network none \
    --mount "type=volume,src=$SOCKET,dst=/var/run/postgresql" \
    -e POSTGRES_HOST_AUTH_METHOD=trust -e PGDATA=/tmp/rcg-data \
    "$PG_IMAGE" -c listen_addresses= >> "$OUTPUT/resources.txt"
"${D[@]}" start "$DB" > /dev/null
ready=0
for attempt in $(seq 1 30); do
    if "${D[@]}" exec "$DB" sh -c 'test "$(head -1 /proc/1/comm)" = postgres && pg_isready -U postgres' > /dev/null 2>&1; then ready=1; break; fi
    sleep 1
done
if [ "$ready" -ne 1 ]; then "${D[@]}" logs "$DB" > "$OUTPUT/database.log" 2>&1; exit 2; fi
"${D[@]}" create --name "$RUNNER" --label "$LABEL=$RUN_ID" --network none \
    --mount "type=volume,src=$SOCKET,dst=/var/run/postgresql" \
    --mount "type=bind,src=$WORK/source,dst=/source,readonly" \
    --mount "type=bind,src=$WORK/repro,dst=/repro,readonly" \
    --mount "type=bind,src=$WORK/schema.sql,dst=/schema.sql,readonly" \
    --mount "type=bind,src=$OUTPUT,dst=/output" \
    -e RCG_SCHEMA=/schema.sql -e RCG_OUTPUT=/output \
    "$IMAGE" python -m pytest -q -p no:cacheprovider --confcutdir=/repro \
    /repro/run_contract_gate/test_run_db.py::test_create_commits_exact_scope \
    /repro/run_contract_gate/test_run_db.py::test_deferred_binding_failure_rolls_back --junitxml=/output/junit.xml >> "$OUTPUT/resources.txt"
set +e
"${D[@]}" start -a "$RUNNER" 2>&1 | tee "$OUTPUT/pytest.log"
code=${PIPESTATUS[0]}
set -e
"${D[@]}" logs "$DB" > "$OUTPUT/database.log" 2>&1
exit "$code"
