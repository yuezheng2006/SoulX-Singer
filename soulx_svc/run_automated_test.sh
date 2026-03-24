#!/usr/bin/env bash
# 一键：按 test_config 启动 API → 等 /health → 跑 test_curl.sh → 可选关闭服务
#
# 用法（在仓库根目录）：
#   bash soulx_svc/run_automated_test.sh
#
# 环境变量：
#   SVC_PYTHON          覆盖解释器（默认优先 .venv_svc_full/bin/python）
#   SVC_CPU_THREADS     默认 2
#   SVC_TEST_CURL_MAXTIME  POST 最长秒数，默认 3600（CPU 长歌可设 7200）
#   SVC_AUTOMATION_KEEP_SERVER=1  测完后不杀 uvicorn
#   SVC_TEST_CONFIG     传给 emit_test_env 的配置文件

set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

if [[ -x "${ROOT}/.venv_svc_full/bin/python" ]]; then
  PY="${SVC_PYTHON:-${ROOT}/.venv_svc_full/bin/python}"
else
  PY="${SVC_PYTHON:-python3}"
fi

export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"
export SVC_CPU_THREADS="${SVC_CPU_THREADS:-2}"
export SVC_TEST_CURL_MAXTIME="${SVC_TEST_CURL_MAXTIME:-3600}"

EMIT=( "${PY}" "${ROOT}/soulx_svc/emit_test_env.py" "${ROOT}" )
[[ -n "${SVC_TEST_CONFIG:-}" ]] && EMIT+=( "${SVC_TEST_CONFIG}" )
eval "$("${EMIT[@]}")"

HOST="${SVC_TEST_HOST}"
PORT="${SVC_TEST_PORT}"
BASE="http://${HOST}:${PORT}"
LOG="${ROOT}/outputs/soulx_svc/automated_uvicorn.log"
mkdir -p "$(dirname "$LOG")"

_started_by_us=0

cleanup() {
  if [[ "${SVC_AUTOMATION_KEEP_SERVER:-}" == "1" ]]; then
    echo "# SVC_AUTOMATION_KEEP_SERVER=1，保留 uvicorn (pid ${_uvicorn_pid:-})" >&2
    return
  fi
  if [[ -n "${_uvicorn_pid:-}" ]] && [[ "${_started_by_us}" -eq 1 ]]; then
    kill "${_uvicorn_pid}" 2>/dev/null || true
    sleep 1
    kill -9 "${_uvicorn_pid}" 2>/dev/null || true
    echo "# 已停止本脚本启动的 uvicorn (pid ${_uvicorn_pid})" >&2
  fi
}
trap cleanup EXIT

if curl -sS -m 2 "${BASE}/health" >/dev/null 2>&1; then
  echo "# 已有服务在 ${BASE}，跳过启动" >&2
else
  echo "# 使用 Python: ${PY}" >&2
  echo "# 启动 uvicorn → ${BASE}，日志 ${LOG}" >&2
  : >"${LOG}"
  nohup env PYTHONPATH="${PYTHONPATH}" SVC_CPU_THREADS="${SVC_CPU_THREADS}" \
    "${PY}" -m uvicorn soulx_svc.api:app --host "${HOST}" --port "${PORT}" \
    >>"${LOG}" 2>&1 &
  _uvicorn_pid=$!
  _started_by_us=1
  for _i in $(seq 1 60); do
    if curl -sS -m 2 "${BASE}/health" >/dev/null 2>&1; then
      echo "# /health 就绪" >&2
      break
    fi
    sleep 1
    if [[ "${_i}" -eq 60 ]]; then
      echo "ERROR: 60s 内 ${BASE}/health 仍不可用，见 ${LOG}" >&2
      tail -40 "${LOG}" >&2 || true
      exit 1
    fi
  done
fi

echo "# 运行 test_curl.sh（POST 超时 ${SVC_TEST_CURL_MAXTIME}s）…" >&2
set +e
bash "${ROOT}/soulx_svc/test_curl.sh"
_rc=$?
set -e

if [[ -f "${SVC_TEST_OUT:-}" ]]; then
  _magic=$(head -c 4 "${SVC_TEST_OUT}" | od -An -tx1 | tr -d ' \n')
  if [[ "${_magic}" == "52494646" ]]; then
    echo "# 输出疑似 WAV（RIFF），路径: ${SVC_TEST_OUT}" >&2
  else
    echo "# 输出非 WAV 头，可能为 JSON 错误；路径: ${SVC_TEST_OUT}" >&2
    head -c 200 "${SVC_TEST_OUT}"; echo >&2
  fi
fi

exit "${_rc}"
