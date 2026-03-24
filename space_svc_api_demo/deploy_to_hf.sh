#!/usr/bin/env bash
# 将「API 演示」Space 推送到**独立**的 Hugging Face 仓库（与 space/ 子模块完全隔离）。
#
# 用法（在本目录执行）:
#   cd space_svc_api_demo
#   bash deploy_to_hf.sh YOUR_USERNAME YOUR_NEW_SPACE_NAME
#
# 注意:
# - 请使用**新的** Space 名称，不要与现有推理 Space（如 SoulX-Singer-with-background）共用同一仓库。
# - 本目录仅含 Gradio + httpx，不含模型与 space/ 内代码。

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if [ "${#}" -lt 2 ]; then
  echo "用法: $0 <YOUR_USERNAME> <NEW_SPACE_NAME>"
  echo "示例: $0 vincenthugging SoulX-SVC-API-Playground"
  echo ""
  echo "说明: 会在 HF 上创建/使用 spaces/<USERNAME>/<NEW_SPACE_NAME>，与主仓库的 space/ 子模块无关。"
  exit 1
fi

USERNAME="$1"
SPACE_NAME="$2"
SPACE_REPO="https://huggingface.co/spaces/${USERNAME}/${SPACE_NAME}"

echo "部署目标（独立 Space）: ${SPACE_REPO}"
echo "与 space/ git submodule 无任何关联。"
echo ""

if ! command -v hf &>/dev/null; then
  echo "未找到 hf 命令，正在安装 huggingface_hub..."
  pip install -U huggingface_hub
fi

if ! hf auth whoami &>/dev/null; then
  echo "请先登录 Hugging Face: hf auth login"
  exit 1
fi

echo "确保 Space 存在（不存在则创建，Gradio SDK）..."
hf repo create "${USERNAME}/${SPACE_NAME}" \
  --repo-type space \
  --space_sdk gradio \
  --exist-ok

if [ ! -d ".git" ]; then
  echo "在本目录初始化独立 Git 仓库（仅跟踪 API 演示文件）..."
  git init
  git add app.py requirements.txt README.md deploy_to_hf.sh
  git commit -m "chore: SoulX SVC API Playground (isolated from space/ submodule)" || true
fi

if git remote get-url origin &>/dev/null; then
  git remote set-url origin "${SPACE_REPO}"
else
  git remote add origin "${SPACE_REPO}"
fi

echo "推送到 Hugging Face..."
git branch -M main 2>/dev/null || true
git push -u origin main

echo ""
echo "完成: ${SPACE_REPO}"
echo "请勿将此 remote 指向原推理 Space 仓库，以保持隔离。"
