#!/bin/sh
# DeedLens container start:
#  1. (as root) make sure the data and model volumes are writable, then drop privileges;
#  2. first run (data/raw has no PDFs): copy the fictional samples in;
#  3. ingest (cached: only new or changed PDFs cost anything; skip with INGEST_ON_START=false);
#  4. start the web server.
set -e

DATA_DIR="${DATA_DIR:-/app/data}"
MODELS="${HF_HOME:-/models}"

if [ "$(id -u)" = "0" ]; then
  mkdir -p "$DATA_DIR" "$MODELS"
  chown -R deedlens:deedlens "$DATA_DIR" "$MODELS" 2>/dev/null || true
  exec setpriv --reuid=1000 --regid=1000 --init-groups "$0" "$@"
fi

mkdir -p "$DATA_DIR/raw"
if ! ls "$DATA_DIR"/raw/*.pdf >/dev/null 2>&1; then
  echo "DeedLens: no PDFs in $DATA_DIR/raw - copying the fictional sample deeds."
  echo "DeedLens: first run downloads ~250 MB of models and ingests the samples (a few minutes)."
  cp /app/samples/*.pdf "$DATA_DIR/raw/"
  INGEST_ON_START=true
fi

if [ "${INGEST_ON_START:-true}" = "true" ]; then
  python -m deedlens.ingest || echo "DeedLens: ingestion reported problems (see above); starting the server anyway."
fi

echo "DeedLens: open http://localhost:${PUBLISHED_PORT:-8000}"
exec uvicorn deedlens.api.main:app --host 0.0.0.0 --port 8000
