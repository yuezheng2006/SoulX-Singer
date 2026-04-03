"""
SVC HTTP API：multipart 上传 prompt_audio + target_audio，返回 wav。

在仓库根目录执行：

  export PYTHONPATH=$PWD:$PYTHONPATH
  pip install -r soulx_svc/requirements-api.txt
  python -m soulx_svc.api --host 0.0.0.0 --port 8088

CPU 占满（活动监视器里几百 %）时，在启动前限制线程，例如：

  export SVC_CPU_THREADS=2          # 推荐：BLAS + PyTorch 用约 2 核
  # 或
  export SVC_LIMIT_CPU=1            # 最省 CPU，也最慢

也可直接：bash soulx_svc/run_api_low_cpu.sh --host 0.0.0.0 --port 8088
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from soulx_svc.cpu_threads import apply_cpu_thread_limits_from_env

apply_cpu_thread_limits_from_env()

from typing import TYPE_CHECKING, Optional

import httpx
from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import Response

if TYPE_CHECKING:
    from soulx_svc.runner import SVCServiceRunner

_runner: "SVCServiceRunner | None" = None


def get_runner():
    global _runner
    if _runner is None:
        try:
            from soulx_svc.runner import SVCServiceRunner as _SVC, default_repo_root
        except ModuleNotFoundError as e:
            raise HTTPException(
                status_code=503,
                detail=f"SVC 推理依赖未安装: {e}",
            ) from e
        use_fp16 = os.environ.get("SVC_FP16", "").lower() in ("1", "true", "yes")
        _runner = _SVC(repo_root=default_repo_root(), use_fp16=use_fp16)
    return _runner


app = FastAPI(title="SoulX-Singer SVC API", version="0.1.0")

_cors = os.environ.get("SVC_CORS_ORIGINS", "").strip()
if _cors:
    from fastapi.middleware.cors import CORSMiddleware

    _orig = [x.strip() for x in _cors.split(",") if x.strip()]
    _wildcard = len(_orig) == 1 and _orig[0] == "*"
    app.add_middleware(
        CORSMiddleware,
        allow_origins=_orig,
        allow_credentials=not _wildcard,
        allow_methods=["*"],
        allow_headers=["*"],
    )


def _cleanup_session(path: Path) -> None:
    shutil.rmtree(path, ignore_errors=True)


async def _download_or_read_upload(
    url_or_upload: Optional[str],
    upload: Optional[UploadFile],
    dest_path: Path,
    param_name: str,
) -> None:
    """下载 URL 或读取上传文件到 dest_path"""
    if url_or_upload:
        try:
            async with httpx.AsyncClient(follow_redirects=True, timeout=60.0) as client:
                resp = await client.get(url_or_upload)
                resp.raise_for_status()
                dest_path.write_bytes(resp.content)
        except Exception as e:
            raise HTTPException(
                status_code=400,
                detail=f"Failed to download {param_name}: {str(e)}",
            ) from e
    elif upload:
        data = await upload.read()
        if not data:
            raise HTTPException(status_code=400, detail=f"empty file: {upload.filename}")
        dest_path.write_bytes(data)
    else:
        raise HTTPException(status_code=400, detail=f"Either {param_name} or {param_name}_url is required")


@app.post("/v1/svc")
async def convert_svc(
    background_tasks: BackgroundTasks,
    prompt_audio: Optional[UploadFile] = File(None, description="Reference timbre (singing, file upload)"),
    prompt_audio_url: Optional[str] = Form(None, description="Reference timbre (URL)"),
    target_audio: Optional[UploadFile] = File(None, description="Singing to convert (file upload)"),
    target_audio_url: Optional[str] = Form(None, description="Singing to convert (URL)"),
    prompt_vocal_sep: bool = Form(False),
    target_vocal_sep: bool = Form(True),
    auto_shift: bool = Form(True),
    auto_mix_acc: bool = Form(True),
    pitch_shift: int = Form(0),
    n_steps: int = Form(32),
    cfg: float = Form(1.0),
    seed: int = Form(42),
):
    try:
        from soulx_svc.runner import SVCRunParams, new_session_dir
    except ModuleNotFoundError as e:
        raise HTTPException(
            status_code=503,
            detail=f"SVC 推理依赖未安装（请在项目 conda/venv 中安装 requirements.txt 等）: {e}",
        ) from e

    session = new_session_dir()
    try:
        up_dir = session / "uploads"
        up_dir.mkdir(parents=True, exist_ok=True)
        p_path = up_dir / "prompt.wav"
        t_path = up_dir / "target.wav"

        await _download_or_read_upload(prompt_audio_url, prompt_audio, p_path, "prompt_audio")
        await _download_or_read_upload(target_audio_url, target_audio, t_path, "target_audio")

        params = SVCRunParams(
            prompt_vocal_sep=prompt_vocal_sep,
            target_vocal_sep=target_vocal_sep,
            auto_shift=auto_shift,
            auto_mix_acc=auto_mix_acc,
            pitch_shift=pitch_shift,
            n_steps=n_steps,
            cfg=cfg,
            seed=seed,
        )

        ok, msg, out_path = get_runner().run_from_raw_audio_files(
            str(p_path),
            str(t_path),
            session,
            params,
        )
        if not ok or out_path is None:
            raise HTTPException(status_code=500, detail=msg)

        wav_bytes = out_path.read_bytes()
        if os.environ.get("SVC_KEEP_SESSIONS", "").lower() not in ("1", "true", "yes"):
            background_tasks.add_task(_cleanup_session, session)
        return Response(
            content=wav_bytes,
            media_type="audio/wav",
            headers={"Content-Disposition": 'attachment; filename="generated.wav"'},
        )
    except HTTPException:
        if os.environ.get("SVC_KEEP_SESSIONS", "").lower() not in ("1", "true", "yes"):
            _cleanup_session(session)
        raise
    except Exception as e:
        if os.environ.get("SVC_KEEP_SESSIONS", "").lower() not in ("1", "true", "yes"):
            _cleanup_session(session)
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.get("/health")
def health():
    return {"status": "ok", "service": "svc"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8088)
    args = parser.parse_args()
    import uvicorn

    uvicorn.run("soulx_svc.api:app", host=args.host, port=args.port, reload=False)


if __name__ == "__main__":
    main()
