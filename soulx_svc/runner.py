"""
SVC 端到端：原始音频 -> 预处理（F0、可选人声分离）-> SoulX-Singer-SVC。
供 webui_svc、HTTP API 等调用；依赖仓库根目录下的 preprocess / 预训练权重。
"""

from __future__ import annotations

from soulx_svc.cpu_threads import apply_cpu_thread_limits_from_env, apply_torch_cpu_threads

# 须在 import numpy/torch/librosa 之前限制 BLAS/OpenMP 线程
apply_cpu_thread_limits_from_env()

import gc
import random
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

import librosa
import numpy as np
import soundfile as sf
import torch

from preprocess.pipeline import PreprocessPipeline
from soulxsinger.utils.file_utils import load_config
from cli.inference_svc import build_model as build_svc_model, process as svc_process


def default_repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def get_inference_device() -> str:
    return "cuda:0" if torch.cuda.is_available() else "cpu"


def trim_and_save_audio(
    src_audio_path: str,
    dst_wav_path: Path,
    max_sec: int,
    sr: int = 44100,
) -> None:
    audio_data, _ = librosa.load(src_audio_path, sr=sr, mono=True)
    audio_data = audio_data[: max_sec * sr]
    dst_wav_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(dst_wav_path, audio_data, sr)


@dataclass
class SVCRunParams:
    prompt_vocal_sep: bool = False
    target_vocal_sep: bool = True
    auto_shift: bool = True
    auto_mix_acc: bool = True
    pitch_shift: int = 0
    n_steps: int = 32
    cfg: float = 1.0
    seed: int = 42
    prompt_max_sec: int = 30
    target_max_sec: int = 600


class SVCServiceRunner:
    """加载预处理流水线与 SVC 模型；每次请求跑完整链路。"""

    def __init__(
        self,
        repo_root: Optional[Path] = None,
        device: Optional[str] = None,
        use_fp16: bool = False,
        config_rel: str = "soulxsinger/config/soulxsinger.yaml",
    ) -> None:
        self.repo_root = Path(repo_root) if repo_root is not None else default_repo_root()
        self.device = device if device is not None else get_inference_device()
        self.use_fp16 = use_fp16 and ("cuda" in self.device)
        apply_torch_cpu_threads(self.device)

        placeholder = self.repo_root / "outputs" / "soulx_svc" / "_placeholder"
        self.preprocess_pipeline = PreprocessPipeline(
            device=self.device,
            language="Mandarin",
            save_dir=str(placeholder),
            vocal_sep=True,
            max_merge_duration=60000,
            midi_transcribe=False,
        )

        config_path = self.repo_root / config_rel
        self.svc_config = load_config(str(config_path))
        model_path = self.repo_root / "pretrained_models" / "SoulX-Singer" / "model-svc.pt"
        self.svc_model = build_svc_model(
            model_path=str(model_path),
            config=self.svc_config,
            device=self.device,
            use_fp16=self.use_fp16,
        )

    def run_preprocess(
        self,
        audio_path: Path,
        save_path: Path,
        vocal_sep: bool,
    ) -> tuple[bool, str, Optional[Path], Optional[Path]]:
        try:
            self.preprocess_pipeline.save_dir = str(save_path)
            self.preprocess_pipeline.run(
                audio_path=str(audio_path),
                vocal_sep=vocal_sep,
                max_merge_duration=60000,
                language="Mandarin",
            )
            vocal_wav = save_path / "vocal.wav"
            vocal_f0 = save_path / "vocal_f0.npy"
            if not vocal_wav.exists() or not vocal_f0.exists():
                return False, f"preprocess output missing: {vocal_wav} or {vocal_f0}", None, None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return True, "ok", vocal_wav, vocal_f0
        except Exception as e:
            return False, f"preprocess failed: {e}", None, None

    def run_svc_inference(
        self,
        prompt_wav_path: Path,
        target_wav_path: Path,
        prompt_f0_path: Path,
        target_f0_path: Path,
        session_base: Path,
        params: SVCRunParams,
    ) -> tuple[bool, str, Optional[Path]]:
        try:
            torch.manual_seed(params.seed)
            np.random.seed(params.seed)
            random.seed(params.seed)

            save_dir = session_base / "generated"
            save_dir.mkdir(parents=True, exist_ok=True)

            class Args:
                pass

            args = Args()
            args.device = self.device
            args.prompt_wav_path = str(prompt_wav_path)
            args.target_wav_path = str(target_wav_path)
            args.prompt_f0_path = str(prompt_f0_path)
            args.target_f0_path = str(target_f0_path)
            args.save_dir = str(save_dir)
            args.auto_shift = params.auto_shift
            args.pitch_shift = int(params.pitch_shift)
            args.n_steps = int(params.n_steps)
            args.cfg = float(params.cfg)
            args.use_fp16 = self.use_fp16

            svc_process(args, self.svc_config, self.svc_model)

            generated = save_dir / "generated.wav"
            if not generated.exists():
                return False, f"inference finished but output not found: {generated}", None

            if params.auto_mix_acc:
                acc_path = session_base / "transcriptions" / "target" / "acc.wav"
                if acc_path.exists():
                    vocal_shift = args.pitch_shift
                    mul = -1 if vocal_shift < 0 else 1
                    acc_shift = abs(vocal_shift) % 12
                    acc_shift = mul * acc_shift
                    if acc_shift > 6:
                        acc_shift -= 12
                    if acc_shift < -6:
                        acc_shift += 12

                    mix_sr = self.svc_config.audio.sample_rate
                    vocal, _ = librosa.load(str(generated), sr=mix_sr, mono=True)
                    acc, _ = librosa.load(str(acc_path), sr=mix_sr, mono=True)
                    if acc_shift != 0:
                        acc = librosa.effects.pitch_shift(acc, sr=mix_sr, n_steps=acc_shift)

                    mix_len = min(len(vocal), len(acc))
                    if mix_len > 0:
                        mixed = vocal[:mix_len] + acc[:mix_len]
                        peak = float(np.max(np.abs(mixed))) if mixed.size > 0 else 1.0
                        if peak > 1.0:
                            mixed = mixed / peak
                        mixed_path = save_dir / "generated_mixed.wav"
                        sf.write(str(mixed_path), mixed, mix_sr)
                        generated = mixed_path
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return True, "svc inference done", generated
        except Exception as e:
            return False, f"svc inference failed: {e}", None

    def run_from_raw_audio_files(
        self,
        prompt_audio_path: str,
        target_audio_path: str,
        session_base: Path,
        params: Optional[SVCRunParams] = None,
    ) -> tuple[bool, str, Optional[Path]]:
        """
        完整流程：任意可读音频路径 -> 单一输出 wav。
        中间结果写在 session_base（transcriptions/、generated/）。
        """
        params = params or SVCRunParams()
        session_base.mkdir(parents=True, exist_ok=True)
        audio_dir = session_base / "audio"
        prompt_raw = audio_dir / "prompt.wav"
        target_raw = audio_dir / "target.wav"

        try:
            trim_and_save_audio(
                prompt_audio_path, prompt_raw, params.prompt_max_sec, sr=44100
            )
            trim_and_save_audio(
                target_audio_path, target_raw, params.target_max_sec, sr=44100
            )
        except Exception as e:
            return False, f"load/trim audio failed: {e}", None

        prompt_ok, prompt_msg, prompt_wav, prompt_f0 = self.run_preprocess(
            audio_path=prompt_raw,
            save_path=session_base / "transcriptions" / "prompt",
            vocal_sep=params.prompt_vocal_sep,
        )
        if not prompt_ok or prompt_wav is None or prompt_f0 is None:
            return False, prompt_msg, None

        target_ok, target_msg, target_wav, target_f0 = self.run_preprocess(
            audio_path=target_raw,
            save_path=session_base / "transcriptions" / "target",
            vocal_sep=params.target_vocal_sep,
        )
        if not target_ok or target_wav is None or target_f0 is None:
            return False, target_msg, None

        return self.run_svc_inference(
            prompt_wav_path=prompt_wav,
            target_wav_path=target_wav,
            prompt_f0_path=prompt_f0,
            target_f0_path=target_f0,
            session_base=session_base,
            params=params,
        )


def new_session_dir(base: Optional[Path] = None) -> Path:
    root = base if base is not None else default_repo_root()
    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    path = root / "outputs" / "soulx_svc" / "sessions" / ts
    path.mkdir(parents=True, exist_ok=True)
    return path
