#!/usr/bin/env bash
set -euo pipefail
docker stop pgsui >/dev/null 2>&1 || true
