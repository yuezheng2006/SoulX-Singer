---
title: SoulX SVC API Playground
emoji: 🔌
colorFrom: indigo
colorTo: gray
sdk: gradio
sdk_version: 4.44.0
python_version: 3.12.8
app_file: app.py
pinned: false
license: apache-2.0
short_description: 演示 HTTP 调用 SoulX-Singer SVC FastAPI（POST /v1/svc）
---

# SoulX SVC API Playground

轻量 **Gradio** Space：不跑模型，只在服务端用 `httpx` 调用你已部署的 **`python -m soulx_svc.api`**，用于对外演示「API 化」集成方式。

## 与原有 Space 的隔离（重要）

| 项目 | 原推理 Space（`space/`） | 本目录 `space_svc_api_demo/` |
|------|---------------------------|------------------------------|
| Git | **子模块** → 独立 HF 仓库（含完整 WebUI + 模型下载逻辑） | **普通目录**，可再 `git init` 推到**另一个** HF Space |
| 依赖 | PyTorch、预处理、权重 | 仅 `gradio`、`httpx` |
| 用途 | 在 HF 上直接跑 SVC | 只演示调用你自托管的 HTTP API |
| 部署脚本 | `space/deploy_to_hf.sh` | **`space_svc_api_demo/deploy_to_hf.sh`**（勿混用） |

推 HF 时请使用**新的 Space 名称**与**新的仓库 URL**，不要覆盖 `SoulX-Singer-with-background` 等现有推理 Space。

## 使用方式

1. 在任意云主机 / 本机（公网可达）启动 API，例如：
   ```bash
   export PYTHONPATH=$PWD:$PYTHONPATH
   python -m soulx_svc.api --host 0.0.0.0 --port 8088
   ```
2. 在本 Space 中填写 **API Base URL**（如 `https://api.example.com:8088`，**不要**末尾 `/`）。
3. 上传 **参考音频**、**目标音频**，点击 **调用 API**。

## Hugging Face 上新建 Space（独立仓库）

全程用官方 **`hf` CLI**（`huggingface_hub` 自带），与网页操作等价。

### 1. 安装与登录

```bash
pip install -U huggingface_hub
hf auth login
hf auth whoami
```

### 2. 一键创建 Space + git 推送（推荐）

在本目录执行（与 `space/` 子模块隔离）：

```bash
cd space_svc_api_demo
bash deploy_to_hf.sh <你的HF用户名> <新Space名>
```

脚本内部等价于：

```bash
hf repo create <用户名>/<新Space名> --repo-type space --space_sdk gradio --exist-ok
# 随后在本地 git push 到 https://huggingface.co/spaces/<用户名>/<新Space名>
```

会在本目录生成**仅用于该演示**的 `.git/`，并推送到 `spaces/<用户>/<新Space名>`；主仓库的 `space/` 子模块不会被修改。

### 3. 纯 CLI、不用脚本时

也可手动创建仓库后再 `git push`：

```bash
hf repo create <用户名>/<新Space名> --repo-type space --space_sdk gradio --exist-ok
cd space_svc_api_demo
git init
git add app.py requirements.txt README.md deploy_to_hf.sh
git commit -m "init API playground"
git remote add origin https://huggingface.co/spaces/<用户名>/<新Space名>
git branch -M main
git push -u origin main
```

**方式 A — 网页**：打开 [New Space](https://huggingface.co/new-space)，SDK 选 **Gradio**，将本目录文件推到该空仓库（与 CLI 二选一即可）。

（可选）在 Space **Settings → Repository secrets** 添加 `SVC_API_BASE_URL`，默认填好你的 API 地址。

## 环境变量（可选）

| 变量 | 说明 |
|------|------|
| `SVC_API_BASE_URL` | 默认 API Base URL |
| `SVC_API_TIMEOUT_SEC` | 请求超时秒数，默认 `3600`（长音频建议保持较大） |

## 与主仓库关系

本目录可单独复制为 HF 仓库；亦位于 [SoulX-Singer](https://github.com/Soul-AILab/SoulX-Singer) 仓库的 `space_svc_api_demo/`，与完整推理 Space（`space/` 子模块）相互独立。

## curl 示例

```bash
BASE="http://127.0.0.1:8088"
curl -sS -X POST "$BASE/v1/svc?n_steps=32&cfg=1.0" \
  -F "prompt_audio=@prompt.wav" \
  -F "target_audio=@target.wav" \
  -o generated.wav
```

API 契约以主项目 `soulx_svc/api.py` 为准。
