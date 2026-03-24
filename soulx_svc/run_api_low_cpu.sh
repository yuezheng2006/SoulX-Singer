#!/usr/bin/env bash
# 启动 SVC API 并限制 CPU 并行（默认约 2 核，可按需改 SVC_CPU_THREADS）
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"
export SVC_CPU_THREADS="${SVC_CPU_THREADS:-2}"
exec "${PYTHON:-python3}" -m soulx_svc.api "$@"
