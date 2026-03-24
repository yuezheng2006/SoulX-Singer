"""SoulX-Singer SVC 独立封装：端到端推理 + 可选 HTTP API。"""

from typing import TYPE_CHECKING, Any

__all__ = [
    "SVCRunParams",
    "SVCServiceRunner",
    "default_repo_root",
    "get_inference_device",
    "new_session_dir",
    "trim_and_save_audio",
]

if TYPE_CHECKING:
    from soulx_svc.runner import (
        SVCRunParams,
        SVCServiceRunner,
        default_repo_root,
        get_inference_device,
        new_session_dir,
        trim_and_save_audio,
    )


def __getattr__(name: str) -> Any:
    if name in __all__:
        from soulx_svc import runner

        return getattr(runner, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
