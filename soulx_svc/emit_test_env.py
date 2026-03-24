#!/usr/bin/env python3
"""
从 TOML 读出测试配置，向 stdout 打印可被 bash eval 的 export 语句。
用法: eval "$(python3 soulx_svc/emit_test_env.py /path/to/repo)"
"""
from __future__ import annotations

import shlex
import sys
import tomllib
from pathlib import Path
from urllib.parse import urlencode


def main() -> int:
    if len(sys.argv) < 2:
        print("usage: emit_test_env.py <repo_root> [path/to/test_config.toml]", file=sys.stderr)
        return 2

    root = Path(sys.argv[1]).resolve()
    if len(sys.argv) >= 3:
        cfg_path = Path(sys.argv[2]).expanduser().resolve()
    else:
        local = root / "soulx_svc" / "test_config.toml"
        example = root / "soulx_svc" / "test_config.example.toml"
        cfg_path = local if local.is_file() else example

    if not cfg_path.is_file():
        print(f"config not found: {cfg_path}", file=sys.stderr)
        return 1

    with cfg_path.open("rb") as f:
        cfg = tomllib.load(f)

    api = cfg.get("api") or {}
    audio = cfg.get("audio") or {}
    out = cfg.get("output") or {}
    params = cfg.get("svc_params") or {}

    host = str(api.get("host", "127.0.0.1"))
    port = int(api.get("port", 8088))

    def resolve_audio(p: str) -> Path:
        path = Path(p).expanduser()
        if not path.is_absolute():
            path = (root / path).resolve()
        return path

    prompt = resolve_audio(str(audio.get("prompt", "")))
    target = resolve_audio(str(audio.get("target", "")))
    out_rel = str(out.get("file", "outputs/soulx_svc_test_out.wav"))
    out_path = Path(out_rel).expanduser()
    if not out_path.is_absolute():
        out_path = (root / out_path).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    def b(v: object, default: bool) -> bool:
        return default if v is None else bool(v)

    pairs = [
        ("prompt_vocal_sep", "true" if b(params.get("prompt_vocal_sep"), False) else "false"),
        ("target_vocal_sep", "true" if b(params.get("target_vocal_sep"), True) else "false"),
        ("auto_shift", "true" if b(params.get("auto_shift"), True) else "false"),
        ("auto_mix_acc", "true" if b(params.get("auto_mix_acc"), True) else "false"),
        ("pitch_shift", str(int(params.get("pitch_shift", 0)))),
        ("n_steps", str(int(params.get("n_steps", 32)))),
        ("cfg", str(float(params.get("cfg", 1.0)))),
        ("seed", str(int(params.get("seed", 42)))),
    ]
    query = urlencode(pairs)

    print(f"export SVC_TEST_HOST={shlex.quote(host)}")
    print(f"export SVC_TEST_PORT={shlex.quote(str(port))}")
    print(f"export SVC_TEST_PROMPT={shlex.quote(str(prompt))}")
    print(f"export SVC_TEST_TARGET={shlex.quote(str(target))}")
    print(f"export SVC_TEST_OUT={shlex.quote(str(out_path))}")
    print(f"export SVC_TEST_QUERY={shlex.quote(query)}")
    print(f"export SVC_TEST_CONFIG_USED={shlex.quote(str(cfg_path))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
