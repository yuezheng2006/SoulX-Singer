"""
SoulX-Singer SVC on Hugging Face Space

两种运行方式：
- **嵌入模式**（Docker Space / 本地已 clone 仓库）：同一进程内挂载 `soulx_svc.api` 到 `/api`，Gradio 默认请求 `http://127.0.0.1:$PORT/api`。
- **轻量模式**（仅 Gradio + httpx）：未安装 soulx_svc 时，仅演示调用外部 Base URL（与旧行为一致）。

对外 HTTP 契约：`POST {公网}/api/v1/svc`（与原先 `POST {base}/v1/svc` 等价，多一层 `/api` 前缀）。
"""

from __future__ import annotations

import importlib.util
import os
import tempfile
from pathlib import Path
from urllib.parse import urlencode

import httpx


def _port() -> int:
    return int(os.environ.get("PORT", "7860"))


def _internal_api_base() -> str:
    return f"http://127.0.0.1:{_port()}/api"


def public_api_base() -> str:
    """对外公开的 API 根路径（含 /api），用于展示与 curl；优先环境变量，其次 HF Space 注入变量。"""
    explicit = os.environ.get("SVC_PUBLIC_API_BASE", "").strip().rstrip("/")
    if explicit:
        return explicit
    sid = os.environ.get("SPACE_ID", "").strip()
    if "/" in sid:
        author, name = sid.split("/", 1)
        slug = f"{author}-{name}".lower().replace("_", "-")
        return f"https://{slug}.hf.space/api"
    author = os.environ.get("SPACE_AUTHOR_NAME", "").strip()
    repo = os.environ.get("SPACE_REPO_NAME", "").strip()
    if author and repo:
        slug = f"{author}-{repo}".lower().replace("_", "-")
        return f"https://{slug}.hf.space/api"
    return ""


def use_embedded_api() -> bool:
    if os.environ.get("SVC_API_EMBEDDED", "").lower() in ("0", "false", "no"):
        return False
    try:
        return importlib.util.find_spec("soulx_svc.api") is not None
    except Exception:
        return False


def default_api_base() -> str:
    if use_embedded_api():
        return _internal_api_base()
    return os.environ.get("SVC_API_BASE_URL", "").strip()


DEFAULT_TIMEOUT = float(os.environ.get("SVC_API_TIMEOUT_SEC", "3600"))


def _build_query(
    prompt_vocal_sep: bool,
    target_vocal_sep: bool,
    auto_shift: bool,
    auto_mix_acc: bool,
    pitch_shift: int,
    n_steps: int,
    cfg: float,
    seed: int,
) -> str:
    q = {
        "prompt_vocal_sep": str(prompt_vocal_sep).lower(),
        "target_vocal_sep": str(target_vocal_sep).lower(),
        "auto_shift": str(auto_shift).lower(),
        "auto_mix_acc": str(auto_mix_acc).lower(),
        "pitch_shift": pitch_shift,
        "n_steps": n_steps,
        "cfg": cfg,
        "seed": seed,
    }
    return urlencode(q)


def build_curl(
    base_url: str,
    prompt_path: str | None,
    target_path: str | None,
    prompt_vocal_sep: bool,
    target_vocal_sep: bool,
    auto_shift: bool,
    auto_mix_acc: bool,
    pitch_shift: int,
    n_steps: int,
    cfg: float,
    seed: int,
    *,
    use_public_base: bool = False,
) -> str:
    pub = public_api_base()
    if use_public_base and pub:
        base = pub
    else:
        base = (base_url or "").strip().rstrip("/")
    if not base:
        return (
            "# 未配置 API 地址。\n"
            "# · 嵌入模式：界面已默认本 Space 的 /api；若需在外部执行 curl，"
            "请在 Space 变量中设置 SVC_PUBLIC_API_BASE（或依赖 SPACE_* 自动拼接）。\n"
            "# · 轻量模式：请在上方填写外部 soulx_svc.api 的 Base URL。"
        )
    qs = _build_query(
        prompt_vocal_sep,
        target_vocal_sep,
        auto_shift,
        auto_mix_acc,
        pitch_shift,
        n_steps,
        cfg,
        seed,
    )
    lines = [
        f'BASE="{base}"',
        f'curl -sS -X POST "$BASE/v1/svc?{qs}" \\',
        '  -F "prompt_audio=@/path/to/prompt.wav" \\',
        '  -F "target_audio=@/path/to/target.wav" \\',
        '  -o generated.wav',
    ]
    if use_embedded_api() and not use_public_base and not pub:
        lines.append(
            "# 提示：从你自己电脑调用时，请把 BASE 换成浏览器地址栏里的 Space 域名 + /api"
        )
    return "\n".join(lines)


def call_svc_api(
    base_url: str,
    prompt_audio,
    target_audio,
    api_key: str,
    prompt_vocal_sep: bool,
    target_vocal_sep: bool,
    auto_shift: bool,
    auto_mix_acc: bool,
    pitch_shift: int,
    n_steps: int,
    cfg: float,
    seed: int,
):
    base = (base_url or "").strip().rstrip("/")
    if not base:
        return None, "请填写 API Base URL（嵌入模式启动后应已自动填入本机 /api）。", build_curl(
            base_url,
            None,
            None,
            prompt_vocal_sep,
            target_vocal_sep,
            auto_shift,
            auto_mix_acc,
            pitch_shift,
            n_steps,
            cfg,
            seed,
            use_public_base=False,
        )
    if prompt_audio is None or target_audio is None:
        return None, "请上传参考音频（prompt）与目标音频（target）。", build_curl(
            base,
            None,
            None,
            prompt_vocal_sep,
            target_vocal_sep,
            auto_shift,
            auto_mix_acc,
            pitch_shift,
            n_steps,
            cfg,
            seed,
        )

    p = Path(prompt_audio) if not isinstance(prompt_audio, str) else Path(prompt_audio)
    t = Path(target_audio) if not isinstance(target_audio, str) else Path(target_audio)
    if not p.is_file() or not t.is_file():
        return None, "音频路径无效，请重新上传。", build_curl(
            base,
            str(p) if p.is_file() else None,
            str(t) if t.is_file() else None,
            prompt_vocal_sep,
            target_vocal_sep,
            auto_shift,
            auto_mix_acc,
            pitch_shift,
            n_steps,
            cfg,
            seed,
        )

    qs = _build_query(
        prompt_vocal_sep,
        target_vocal_sep,
        auto_shift,
        auto_mix_acc,
        pitch_shift,
        n_steps,
        cfg,
        seed,
    )
    url = f"{base}/v1/svc?{qs}"
    curl_internal = build_curl(
        base,
        str(p),
        str(t),
        prompt_vocal_sep,
        target_vocal_sep,
        auto_shift,
        auto_mix_acc,
        pitch_shift,
        n_steps,
        cfg,
        seed,
        use_public_base=False,
    )
    curl_public = build_curl(
        base,
        str(p),
        str(t),
        prompt_vocal_sep,
        target_vocal_sep,
        auto_shift,
        auto_mix_acc,
        pitch_shift,
        n_steps,
        cfg,
        seed,
        use_public_base=True,
    )
    curl_preview = (
        f"{curl_public}\n\n# —— 容器内调试（与界面请求一致）——\n{curl_internal}"
        if use_embedded_api() and public_api_base()
        else curl_internal
    )

    headers = {}
    key = (api_key or "").strip()
    if key:
        headers["Authorization"] = f"Bearer {key}"

    try:
        with httpx.Client(timeout=DEFAULT_TIMEOUT, follow_redirects=True) as client:
            with p.open("rb") as fp, t.open("rb") as ft:
                files = {
                    "prompt_audio": (p.name, fp, "application/octet-stream"),
                    "target_audio": (t.name, ft, "application/octet-stream"),
                }
                r = client.post(url, files=files, headers=headers)
    except httpx.RequestError as e:
        return None, f"网络错误: {e}", curl_preview

    if r.status_code != 200:
        body = r.text[:2000] if r.text else ""
        return None, f"HTTP {r.status_code}\n{body}", curl_preview

    ct = (r.headers.get("content-type") or "").lower()
    if "wav" not in ct and r.content[:4] != b"RIFF":
        return None, f"响应不是 WAV（Content-Type: {ct or 'unknown'}）", curl_preview

    fd, out_path = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    Path(out_path).write_bytes(r.content)
    return (
        out_path,
        f"成功，{len(r.content)} 字节，Content-Type: {ct or 'audio/wav'}",
        curl_preview,
    )


def api_urls_markdown() -> str:
    """页首：由本服务给出的、可给第三方的 URL 说明。"""
    pub = public_api_base()
    if use_embedded_api():
        if pub:
            return (
                "### 本 Space 对外提供的 HTTP API（可直接给外部系统使用）\n\n"
                f"| 用途 | URL |\n|------|-----|\n"
                f"| **歌声转换** | `{pub}/v1/svc`（`POST`，multipart：`prompt_audio`、`target_audio`） |\n"
                f"| **健康检查** | `{pub}/health`（`GET`） |\n"
                f"| **OpenAPI 文档** | `{pub}/docs` |\n\n"
                "浏览器或其它服务器均可请求上述地址（已启用 CORS，便于前端跨域调试）。"
                "若域名与上表不符，请在 Space **Settings → Variables** 设置 `SVC_PUBLIC_API_BASE`"
                "（例如 `https://你的用户名-space名.hf.space/api`）。\n"
            )
        return (
            "### 本 Space 已内置 API\n\n"
            "当前未能自动推断公网地址。请在 Space **Variables** 中设置 **`SVC_PUBLIC_API_BASE`**"
            "（值为 `https://<你的 hf.space 域名>/api`，无尾斜杠），保存后刷新本页即可显示完整 URL。\n\n"
            "- 转换：`POST .../api/v1/svc`\n"
            "- 健康：`GET .../api/health`\n"
        )
    if pub:
        return (
            "### 轻量模式 · 文档用 API 根路径\n\n"
            f"已配置 **`SVC_PUBLIC_API_BASE`**：`{pub}`。本 Space **未**内置推理，"
            "仅演示从浏览器/脚本调用该地址；实际推理需在别处部署 `soulx_svc.api`。\n"
        )
    return (
        "### 轻量模式\n\n"
        "本 Space **未**内置 `soulx_svc`。请在下方 **「要调用的 API Base URL」** 填写你已部署的 "
        "`soulx_svc` 地址；或在 Variables 中设置 **`SVC_PUBLIC_API_BASE`** 以便页首展示固定文档 URL。\n"
    )


def build_gradio():
    import gradio as gr

    pub = public_api_base()
    # 嵌入模式：实际请求走容器内回环，不向用户索要 URL
    internal = _internal_api_base() if use_embedded_api() else ""
    manual_default = (
        ""
        if use_embedded_api()
        else (default_api_base() or pub)
    )

    with gr.Blocks(
        title="SoulX SVC API Playground",
        theme=gr.themes.Soft(primary_hue="indigo", secondary_hue="slate"),
    ) as demo:
        gr.Markdown(api_urls_markdown())
        if use_embedded_api():
            gr.Markdown(
                "_下方演示按钮会向 **容器内** `127.0.0.1` 发起请求，与外部用户访问上表公网 URL 等价。_"
            )
            base_in = gr.Textbox(
                label="（内部）请求使用的 Base URL — 已固定，无需修改",
                value=internal,
                interactive=False,
            )
        else:
            base_in = gr.Textbox(
                label="要调用的 API Base URL（不含尾斜杠；可与上方文档地址一致）",
                placeholder="https://your-api.example.com 或 https://xxx.hf.space/api",
                value=manual_default,
            )
        api_key_in = gr.Textbox(
            label="可选：Authorization Bearer（若网关加了鉴权）",
            type="password",
            placeholder="留空则不带 Authorization 头",
        )
        with gr.Row():
            prompt_in = gr.Audio(
                label="参考音色 prompt_audio",
                type="filepath",
                sources=["upload"],
            )
            target_in = gr.Audio(
                label="待转换 target_audio",
                type="filepath",
                sources=["upload"],
            )
        with gr.Accordion("推理参数（query）", open=False):
            with gr.Row():
                pv = gr.Checkbox(label="prompt_vocal_sep", value=False)
                tv = gr.Checkbox(label="target_vocal_sep", value=True)
            with gr.Row():
                ash = gr.Checkbox(label="auto_shift", value=True)
                amx = gr.Checkbox(label="auto_mix_acc", value=True)
            with gr.Row():
                ps = gr.Slider(-36, 36, value=0, step=1, label="pitch_shift")
                ns = gr.Slider(1, 128, value=32, step=1, label="n_steps")
            with gr.Row():
                cf = gr.Slider(0.0, 10.0, value=1.0, step=0.1, label="cfg")
                sd = gr.Number(value=42, label="seed", precision=0)

        btn = gr.Button("调用 API", variant="primary")
        status = gr.Textbox(label="状态", lines=4)
        curl_out = gr.Code(
            label="等效 curl（公网 + 容器内）",
            language="shell",
        )
        audio_out = gr.Audio(label="生成结果", type="filepath")

        inputs = [
            base_in,
            prompt_in,
            target_in,
            api_key_in,
            pv,
            tv,
            ash,
            amx,
            ps,
            ns,
            cf,
            sd,
        ]

        def on_change_curl(b, pr, ta, _k, *rest):
            return build_curl(b, None, None, *rest, use_public_base=True)

        rest_sliders = [pv, tv, ash, amx, ps, ns, cf, sd]
        for comp in [base_in, pv, tv, ash, amx, ps, ns, cf, sd]:
            comp.change(
                on_change_curl,
                [base_in, prompt_in, target_in, api_key_in] + rest_sliders,
                curl_out,
            )

        def _run_call(b, pr, ta, ak, *rest):
            base = _internal_api_base() if use_embedded_api() else (b or "").strip()
            return call_svc_api(base, pr, ta, ak, *rest)

        btn.click(
            _run_call,
            inputs,
            [audio_out, status, curl_out],
        )

    return demo


def create_combined_app():
    """FastAPI：/api → soulx_svc（对外开放 + CORS），/ → Gradio。"""
    from fastapi import FastAPI
    from fastapi.middleware.cors import CORSMiddleware
    from gradio import mount_gradio_app

    from soulx_svc.api import app as svc_app

    demo = build_gradio()
    main = FastAPI(title="SoulX-Singer SVC Space", version="0.2.0")
    _cors = os.environ.get("SVC_CORS_ORIGINS", "*").strip()
    origins = (
        ["*"]
        if _cors == "*"
        else [x.strip() for x in _cors.split(",") if x.strip()]
    )
    _wildcard = origins == ["*"]
    main.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=not _wildcard,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    main.mount("/api", svc_app)
    mount_gradio_app(main, demo, path="/")
    return main


def main() -> None:
    port = _port()
    if use_embedded_api():
        import uvicorn

        app = create_combined_app()
        uvicorn.run(app, host="0.0.0.0", port=port)
        return

    import gradio as gr

    demo = build_gradio()
    demo.queue()
    demo.launch(server_name="0.0.0.0", server_port=port)


if __name__ == "__main__":
    main()
