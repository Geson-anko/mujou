#!/usr/bin/env python3
"""PostToolUse hook (matcher: Edit|Write|MultiEdit): warn on global backprop in src/.

mujou の学習則は PC-ALM の層局所更新であり、ネットワーク全体を貫く誤差逆伝播
（``loss.backward()`` 等）を ``src/mujou/`` に持ち込まない。層内のローカルエネルギーの
勾配を autograd で求めるのは許容されるので、ブロックせず警告だけを返す。

意図的な層局所 autograd は、その行に ``# mujou: local-autograd`` を付けると警告対象から
外れる。fail-open（常に exit 0）。依存は標準ライブラリのみ。
"""

import json
import os
import re
import sys
from pathlib import Path

_PATTERNS = [
    (re.compile(r"\.backward\s*\("), "`.backward()`"),
    (re.compile(r"\bautograd\.grad\s*\("), "`torch.autograd.grad()`"),
    (re.compile(r"\bautograd\.backward\s*\("), "`torch.autograd.backward()`"),
    (re.compile(r"\bretain_graph\s*=\s*True"), "`retain_graph=True`"),
    (re.compile(r"\bjax\.(grad|value_and_grad|vjp)\s*\("), "JAX global gradient"),
]
_ALLOW_MARKER = "# mujou: local-autograd"


def _is_library_source(path: Path, project_dir: Path) -> bool:
    try:
        relative = path.resolve().relative_to(project_dir.resolve())
    except ValueError:
        return False
    return relative.parts[:2] == ("src", "mujou") and path.suffix == ".py"


def _find_hits(text: str) -> list[str]:
    hits: list[str] = []
    for number, line in enumerate(text.splitlines(), start=1):
        if _ALLOW_MARKER in line:
            continue
        for pattern, label in _PATTERNS:
            if pattern.search(line):
                hits.append(f"  L{number}: {label}: {line.strip()}")
    return hits


def main() -> int:
    try:
        data = json.load(sys.stdin)
        path = Path(data["tool_input"]["file_path"])
        project_dir = Path(
            os.environ.get("CLAUDE_PROJECT_DIR") or data.get("cwd") or "."
        )
        if not _is_library_source(path, project_dir):
            return 0
        hits = _find_hits(path.read_text(encoding="utf-8"))
    except Exception:
        return 0
    if not hits:
        return 0

    context = "\n".join(
        [
            f"[LOCALITY GUARD] {path} にグローバル勾配計算に見える箇所がある。",
            *hits,
            "- mujou は BP を使わず PC-ALM の層局所更新で学習する（skill `pc-alm`, `locality-and-async`）",
            "- 層をまたいで計算グラフが繋がっていないか確認せよ（入力側の活動は detach されているか）",
            f"- 層内ローカルエネルギーの勾配として意図的に使うなら、その行に `{_ALLOW_MARKER}` を付けて理由をコメントせよ",
        ]
    )
    print(
        json.dumps(
            {
                "hookSpecificOutput": {
                    "hookEventName": "PostToolUse",
                    "additionalContext": context,
                }
            }
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
