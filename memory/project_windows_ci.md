---
name: project-windows-ci
description: Test のワークフローは Ubuntu・macOS・Windows で回す（2026-10-23）。.claude/ のツールは Windows を対象にせず、シェルの hook のテストは Windows で skip
metadata:
  type: project
---

2026-10-23、ユーザーの依頼「Windows 対応を CI で試行」で、`.github/workflows/test.yml` の rust と pytest のマトリクスに `windows-latest` を足した（PR #105）。

- Windows で落ちたのはテストとサンプルの側だけで、`src/mujou/` と Rust のコアは直さずに通った
- ファイルの読み書きは `encoding` を明示する（Windows の既定は cp1252。`viz.html` の `⇄` などは書けない）
- テストで SIGINT を起こすときは `signal.raise_signal`。Windows の `os.kill(pid, SIGINT)` はプロセスを終了させる
- `.claude/` のツール（hook・statusline）は `python3` に依存し、Windows を対象にしない。`TestCompactionRecovery` は Windows で skip
- Windows の wheel の公開（publish.yml）は別の判断。リリース戦略は [[project-release-strategy]]

**Why:** 利用者が Windows で使えることを CI で担保するため。

**How to apply:** テストやサンプルで OS に依存する書き方（encoding の省略、`os.kill`、`python3` 前提）を避ける。
