# CLAUDE.md

このファイルは Claude Code がこのリポジトリを扱う際のガイダンス。常時ロードされるので、手続きの詳細は skill に置き、ここには判断の拠り所だけを書く。

## 応答言語

ユーザーへの応答は日本語で行う。

- 散文・説明・要約・確認・質問・計画は日本語で書く
- コード・コマンド・ファイルパス・識別子・ログ/エラーの引用は原語のまま残す
- コミットメッセージ・PR タイトルは「Git 運用」の規約に従い、日本語化しない
- 仕様書・計画書・skill・memory は日本語で書く

## プロジェクト概要

`mujou` は **Asynchronous Distributed Deep Learning Library for Continuous Learning**。PyTorch のような深層学習フレームワークだが、重みの更新に誤差逆伝播（BP）を使わず、**PC-ALM**（Augmented Lagrangian Predictive Coding、<https://pub.sakana.ai/pc-alm/>）を使う。

- 名前は「諸行無常」の「無常」から取った。学習後に重みを固定して推論する従来の機械学習と違い、ストリームから学び続け、常に変化し続ける学習機械を作る
- 狙う用途はリアルタイム処理・ロボティクスなど、生涯学習し続ける機械
- PC-ALM の更新は層局所（隣接層とのやり取りだけで決まる）なので、層・ニューロン単位で非同期に推論と学習を実行できるシステムを作る

## 開発原則

慎重さを速度に優先する。trivial なタスクでは判断で簡略化してよい。

### 1. 実装前に考える

- 仮定を明示し、不確かな点は確認する。複数の解釈やトレードオフがあれば表に出す
- より単純な解決策があれば提示する。不明瞭なまま実装を進めない

### 2. シンプルさを優先

- 要求されていない機能・柔軟性・抽象化を追加しない
- 起こり得ないシナリオ向けの処理を増やさない。問題を解く最小限のコードにする

### 3. 必要な範囲だけ変更

- 周辺コードをついでに整形・リファクタしない。既存スタイルと公開インターフェースを尊重する
- 自分の変更で生じた未使用コードだけを片付ける。diff の各行をユーザー要求へ直接トレースできる状態にする

### 4. ゴール駆動

検証可能な成功条件を先に置き、満たすまで実装と検証を繰り返す。複数ステップの作業では短い計画と各ステップの検証方法を示す。

## 不変の原則

### BP を持ち込まない・局所性を壊さない

mujou の存在理由。`src/mujou/` にネットワーク全体を貫く `.backward()` を書かない。層（計算単位）は自分の状態と隣接単位からのメッセージだけで更新する。グローバルなバリア・学習/推論モードの固定・エポック前提の API を持ち込まない。

- 学習則の式・平衡点の性質・落とし穴: skill `pc-alm`
- 実行モデルの不変条件・許容される例外（`# mujou: local-autograd`）: skill `locality-and-async`
- 違反しそうな箇所は PostToolUse hook が警告する。破る必要があるときはユーザーに確認し、理由を `memory/` に残す

### テスト

- 「損失が下がった」ではなく理論から導いたオラクル（有限差分、平衡点での BP 一致、α=0 で PC に退化）で検証する
- 乱数とスケジュールを固定して決定的に保つ。`class TestXxx` に集約し、`tests/mujou/` は `src/mujou/` をミラーする
- 詳細: skill `testing-strategy`

### カプセル化

- 内部実装と `__init__` で設定する属性は原則 `_` prefix。外部から必要なものだけ public にする
- 詳細: skill `refactor-conventions`

## ツーリング

- パッケージ・環境管理: `uv`（`uv.lock` をコミットする）。**`pip install` は使わない**（hook がブロックする）。依存追加は `uv add` / `uv add --dev`
- Python: 3.13 以上（`.python-version` は 3.13）
- タスクランナー: `just`（1.58 以上。`[windows] set shell` 構文を使うため）
- 型チェッカー: `pyright` を `src/` に対し strict
- リンター/フォーマッター: `ruff`（line-length 88）。pre-commit: ruff、pyupgrade、mdformat、codespell、uv-lock ほか
- テンソル基盤は未決定（[memory/project_open_decisions.md](memory/project_open_decisions.md)）。決まるまで依存に入れない

## コマンド

| レシピ           | 内容                                                                 |
| ---------------- | -------------------------------------------------------------------- |
| `just setup`     | 開発環境のセットアップ（`uv venv` + `uv sync` + pre-commit install） |
| `just format`    | pre-commit フックを全ファイルに実行                                  |
| `just test`      | pytest（slow を含む全テスト、coverage 付き）                         |
| `just test-fast` | slow を除いた pytest。反復中に使う                                   |
| `just type`      | pyright strict                                                       |
| `just run`       | format → test → type。**コミット前の検証はこれ**                     |

単一テスト: `uv run pytest tests/mujou/test_x.py -v` / キーワード: `uv run pytest -k "<expr>"`

## Git 運用

- `main` から `<種別>/<日付>/<内容>` で分岐する（例: `feature/2026-10-11/pc-alm-layer`）。種別は `feature`, `fix`, `refactor`, `docs`, `chore`
- `main` に直接 commit しない。main への merge はユーザー判断
- commit は `<種別>(<スコープ>): <内容>`、1 commit 1 関心事。`just run` が通る前に commit しない
- force push、`git reset --hard`、`--no-verify`、未確認の破壊的操作を行わない

## エージェント運用

- **既定はシングルエージェント**。ユーザーの指定がない限り skill `solo-dev-cycle` で 計画 → テスト → 実装 → リファクタ → ドキュメント を自分で回す
- 「エージェントチームで」「並列で」と明示されたときだけ skill `agent-team-startup` に従い、自分が `orchestrator` の役割を担う
- agent 構成（`.claude/agents/`）: `implementation-planner` →（任意 `spec-test-author`）→ `plan-implementer` → `code-reviewer`（＋学習則に触れるなら `algorithm-reviewer`）⇄ `code-simplifier`。文書改善用に `doc-teacher` / `doc-student`（haiku）
- `doc-student` を除く全 agent が `model: inherit`。深さは `effort` で差別化する（reviewer 2 体は high、他は medium。セッション既定は `settings.json` の `effortLevel: medium`）
- **ユーザーに質問できるのはメインエージェントだけ**。サブエージェントの質問は報告に含めて返し、メインが中継する
- 中間メモは `memory/agents/<agent-name>/<task-slug>.md`

## Hooks（`.claude/settings.json`）

| hook                               | 役割                                                              |
| ---------------------------------- | ----------------------------------------------------------------- |
| `pretooluse-guard-bash.py`         | `pip install` 系と `git commit --no-verify` をブロック            |
| `posttooluse-locality-guard.py`    | `src/mujou/` へのグローバル勾配計算の混入を警告（ブロックしない） |
| `compaction-recovery.sh` ほか 2 本 | compact 前後の状態退避と復旧（下記）                              |

hook は `python3` と `bash` だけに依存し、fail-open。テストは `tests/test_claude_hooks.py`。

## コンテキスト管理 (compact)

- statusLine（`.claude/scripts/statusline.sh`）が context 使用率を表示し、60% を超えると次のプロンプトで `/compact-prep` を促す
- `/compact-prep` は `/compact` 直前に、要約に残りにくい判断構造（採用/却下した案・現在フェーズ・委譲先）を `${TMPDIR:-/tmp}/claude-compact-state/<session_id>.md` へ退避する
- 圧縮後は hook が state file・TaskList・CLAUDE.md の決定事項を読み戻すよう指示する。圧縮サマリーの next step は仮説として扱う

## メモリ参照

プロジェクト固有の規約・知見・ユーザーの好み・未決定事項は repo ルートの [memory/](memory/) に保存する（git 管理対象）。harness が自動ロードする `~/.claude/projects/.../memory/` は使わない。

セッション開始時、または規約が関係しそうなタスクに着手する前に [memory/MEMORY.md](memory/MEMORY.md) を確認する。新しい規約・フィードバック・決定が判明したら 1 メモリ 1 ファイルで足し、`MEMORY.md` から 1 行リンクを張る。

## 索引（skill）

- 理論: `/pc-alm` / 実行モデル: `/locality-and-async`
- 開発フロー: `/solo-dev-cycle`（既定）/ `/agent-team-startup`（明示時）/ `/maximize-parallels`
- 規約: `/testing-strategy` / `/refactor-conventions`
- Git: `/github-pr` / `/merge-main` / `/do-on-worktree`
- 文書改善: `/doc-teacher-student` / `.claude/` 編集: `/edit-dot-claude`
- context 管理: `/compact-prep` / 言語切替: `/japanese`
