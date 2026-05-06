#!/usr/bin/env bash
# Qdrant snapshot drill — take + restore.
#
# Per IMPLEMENTATION_PLAN.md → Risks & Gotchas → "Snapshot drill",
# every Vector Store collection should have a documented snapshot
# config and a runnable restore procedure from day one. This script is
# the procedure.
#
# Usage:
#   ./infra/scripts/snapshot.sh take <collection>
#   ./infra/scripts/snapshot.sh list <collection>
#   ./infra/scripts/snapshot.sh restore <collection> <snapshot-name>
#
# Assumes a Qdrant instance reachable at $QDRANT_URL (default
# http://localhost:6333). Run `docker compose -f infra/docker-compose.yml
# up qdrant` first if you need a local one.

set -euo pipefail

QDRANT_URL="${QDRANT_URL:-http://localhost:6333}"

usage() {
  cat <<'EOF'
Qdrant snapshot drill

  ./infra/scripts/snapshot.sh take <collection>
      Create a snapshot of the named collection. Returns the snapshot
      filename which the script also prints to stdout.

  ./infra/scripts/snapshot.sh list <collection>
      List existing snapshots for the named collection.

  ./infra/scripts/snapshot.sh restore <collection> <snapshot-name>
      Restore the collection from a previously taken snapshot. The
      collection is replaced atomically.

Snapshots are stored under the path configured by
QDRANT__STORAGE__SNAPSHOTS_PATH on the Qdrant server (defaults to
/qdrant/storage/snapshots in the bundled compose stack).
EOF
}

cmd="${1:-}"
case "$cmd" in
  take)
    collection="${2:?usage: take <collection>}"
    echo "→ Taking snapshot of '$collection' on $QDRANT_URL …"
    response=$(curl -sf -X POST "$QDRANT_URL/collections/$collection/snapshots")
    echo "$response"
    ;;
  list)
    collection="${2:?usage: list <collection>}"
    echo "→ Listing snapshots for '$collection' on $QDRANT_URL …"
    curl -sf "$QDRANT_URL/collections/$collection/snapshots"
    ;;
  restore)
    collection="${2:?usage: restore <collection> <snapshot-name>}"
    snapshot="${3:?usage: restore <collection> <snapshot-name>}"
    echo "→ Restoring '$collection' from '$snapshot' on $QDRANT_URL …"
    curl -sf -X PUT \
      "$QDRANT_URL/collections/$collection/snapshots/recover" \
      -H 'Content-Type: application/json' \
      -d "{\"location\": \"file:///qdrant/storage/snapshots/$collection/$snapshot\"}"
    ;;
  *)
    usage
    exit 1
    ;;
esac
