#!/usr/bin/env bash
# 本地接口冒烟：测试数据与 API 地址见 soulx_svc/test_config.toml（无则使用 test_config.example.toml）
#
# 首次使用：
#   cp soulx_svc/test_config.example.toml soulx_svc/test_config.toml
#   # 编辑 test_config.toml 中的 [audio]、[api]、[svc_params] 等
#
# 启动 API（仓库根目录）：
#   export PYTHONPATH=$PWD:$PYTHONPATH
#   pip install -r soulx_svc/requirements-api.txt
#   python -m soulx_svc.api --host 0.0.0.0 --port 8088
#
# 另需与 webui_svc 相同的 ML 环境 + 预训练权重，否则 /v1/svc 可能返回 503。

set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

# 可选：指定配置文件路径
#   SVC_TEST_CONFIG=/path/to/my.toml bash soulx_svc/test_curl.sh
EMIT=(python3 "${ROOT}/soulx_svc/emit_test_env.py" "${ROOT}")
if [[ -n "${SVC_TEST_CONFIG:-}" ]]; then
  EMIT+=( "${SVC_TEST_CONFIG}" )
fi
eval "$("${EMIT[@]}")"

echo "# using config: ${SVC_TEST_CONFIG_USED}" >&2
BASE="http://${SVC_TEST_HOST}:${SVC_TEST_PORT}"

curl -sS "${BASE}/health"
echo

rm -f "${SVC_TEST_OUT}"

# 可选：整次 POST 最长秒数（CPU 长音频建议 3600+）；不设则由 curl 默认无上限
POST_OPTS=()
if [[ -n "${SVC_TEST_CURL_MAXTIME:-}" ]] && [[ "${SVC_TEST_CURL_MAXTIME}" =~ ^[0-9]+$ ]] && [[ "${SVC_TEST_CURL_MAXTIME}" -gt 0 ]]; then
  POST_OPTS=( -m "${SVC_TEST_CURL_MAXTIME}" )
fi

curl -sS "${POST_OPTS[@]}" -o "${SVC_TEST_OUT}" -w "POST /v1/svc HTTP %{http_code} bytes %{size_download}\n" \
  -X POST "${BASE}/v1/svc?${SVC_TEST_QUERY}" \
  -F "prompt_audio=@${SVC_TEST_PROMPT}" \
  -F "target_audio=@${SVC_TEST_TARGET}"

if [[ -f "${SVC_TEST_OUT}" ]]; then
  head -c 4 "${SVC_TEST_OUT}" | od -An -tx1 | head -1 || true
fi
