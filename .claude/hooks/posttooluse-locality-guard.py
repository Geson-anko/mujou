#!/usr/bin/env python3
"""PostToolUse hook (matcher: Edit|Write|MultiEdit): warn on global backprop in src/.

対象は ``src/mujou/`` 配下の ``.py``。``.claude/worktrees/<name>/src/mujou/`` も含む。

mujou はノード（内側は同期、ノード間は非同期のメッセージ）単位で計算し、
ノードをまたぐ誤差逆伝播（``loss.backward()`` 等）を ``src/mujou/`` に持ち込まない。
ノード内に閉じた autograd（ローカルエネルギーの勾配やノード内の BP）は許容される
ので、ツール呼び出しは止めず警告だけを返す。

意図的なノード内 autograd は、その行に ``# mujou: local-autograd`` を付けると警告対象から
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
        parts = path.resolve().relative_to(project_dir.resolve()).parts
    except ValueError:
        return False
    # A worktree (.claude/worktrees/<name>/) is a checkout of its own.
    if parts[:2] == (".claude", "worktrees"):
        parts = parts[3:]
    return parts[:2] == ("src", "mujou") and path.suffix == ".py"


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
            "- mujou はノード間を非同期のメッセージでつなぎ、ノードをまたぐ勾配計算を持ち込まない（skill `locality-and-async`）",
            "- ノードをまたいで計算グラフが繋がっていないか確認せよ（入力側の活動は detach されているか）",
            f"- ノード内に閉じた勾配（ローカルエネルギーやノード内 BP）として意図的に使うなら、その行に `{_ALLOW_MARKER}` を付けて理由をコメントせよ",
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
