"""
控制 CPU 推理时的并行线程，避免本机「多核打满」（例如活动监视器里 400%）。

务必在「首次 import numpy/torch」之前调用 apply_cpu_thread_limits_from_env()。
PyTorch 加载后再调用 apply_torch_cpu_threads(device)。
"""

from __future__ import annotations

import os

# BLAS / OpenMP（在 import numpy/torch 前写入 os.environ 才可靠）
_BLAS_LIKE_ENV = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "NUMBA_NUM_THREADS",
)


def _effective_cpu_thread_count() -> int | None:
    if os.environ.get("SVC_LIMIT_CPU", "").lower() in ("1", "true", "yes"):
        return 1
    raw = os.environ.get("SVC_CPU_THREADS", "").strip()
    if raw:
        return max(1, int(raw))
    return None


def apply_cpu_thread_limits_from_env() -> None:
    """在进程内、import numpy/torch 之前调用。"""
    n = _effective_cpu_thread_count()
    if n is None:
        return
    for k in _BLAS_LIKE_ENV:
        os.environ[k] = str(n)


def apply_torch_cpu_threads(device: str) -> None:
    """在已 import torch 后调用；GPU 上不修改。"""
    if "cuda" in device:
        return
    n = _effective_cpu_thread_count()
    if n is None:
        return
    import torch

    torch.set_num_threads(n)
    # 降低算子间并行，通常能再省一点 CPU 抖动
    torch.set_num_interop_threads(min(2, max(1, n)))
