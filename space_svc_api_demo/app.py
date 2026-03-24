"""
SoulX-Singer SVC API 演示：在 Space 服务端用 HTTP 调用你部署的 FastAPI（soulx_svc.api）。

说明：请求从 Hugging Face 容器发出，需你的 API 公网可达；不受浏览器 CORS 限制。
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from urllib.parse import urlencode

import gradio as gr
import httpx

DEFAULT_BASE = os.environ.get("SVC_API_BASE_URL", "").strip()
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
) -> str:
    base = (base_url or "").strip().rstrip("/")
    if not base:
        return "# 请先填写 API Base URL（例如 https://your-host:8088）"
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
        return None, "请填写 API Base URL。", build_curl(
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
    curl_preview = build_curl(
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


def main():
    with gr.Blocks(
        title="SoulX SVC API 演示",
        theme=gr.themes.Soft(primary_hue="indigo", secondary_hue="slate"),
    ) as demo:
        gr.Markdown(
            "## SoulX-Singer SVC · API 调用演示\n"
            "填写你已部署的 **`soulx_svc.api`** 地址，上传两段音频，即可在服务端发起 "
            "`POST /v1/svc` 并试听返回的 WAV。\n\n"
            "**要求**：API 须公网可达（本 Space 从 HF 机房出站访问你的 URL）。"
        )
        with gr.Row():
            base_in = gr.Textbox(
                label="API Base URL（不含尾斜杠）",
                placeholder="https://your-api.example.com:8088",
                value=DEFAULT_BASE,
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
            label="等效 curl（路径需替换为本机文件）",
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
            return build_curl(b, None, None, *rest)

        rest_sliders = [pv, tv, ash, amx, ps, ns, cf, sd]
        for comp in [base_in, pv, tv, ash, amx, ps, ns, cf, sd]:
            comp.change(
                on_change_curl,
                [base_in, prompt_in, target_in, api_key_in] + rest_sliders,
                curl_out,
            )

        btn.click(call_svc_api, inputs, [audio_out, status, curl_out])

    demo.queue()
    demo.launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT", "7860")),
    )


if __name__ == "__main__":
    main()
