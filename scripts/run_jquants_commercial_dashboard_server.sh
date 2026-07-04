#!/usr/bin/env bash
set -euo pipefail

cd /home/rai/projects/jp-stock-analysis-engine
mkdir -p logs

exec /home/rai/projects/jp-stock-analysis-engine/.venv/bin/serve-jquants-commercial-dashboard \
  --host 127.0.0.1 \
  --port 8050
