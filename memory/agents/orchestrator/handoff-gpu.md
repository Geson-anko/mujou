# 引き継ぎ: 作り直しの後の GPU テスト

2026-10-21、コンテナを作り直す前に書いた引き継ぎ。新しいセッションは、このファイルと `memory/MEMORY.md` を読んでから始める。

## 今の状態

- ブランチ `refactor/2026-10-20/restructure`（PR は下の「PR」）。mujou の作り直しは全ステップと最終レビューまで終わり、commit 済み。`just run` は GPU の無い環境で通っている（pytest 710 passed / 22 skipped、pyright 0 errors、docs の strict ビルド成功）
- 決定はすべてユーザーが承認済み（`memory/project_restructure_2026_10_20.md`）。設計の正本は `specs/architecture.md` と `specs/scenarios.md`。作り直しの設計書と、作業中の裁定は同じフォルダの `restructure-design.md` と `restructure-carryover.md`（経緯の記録。正本は specs）
- 前のコンテナには GPU が無かった。torch は CUDA 版（`2.14.1+cu130`）だが、`.devcontainer/devcontainer.json` に GPU を渡す設定（`"runArgs": ["--gpus", "all"]` など）が無いので、`torch.cuda.is_available()` が False だった。GPU のテストの前に、コンテナから GPU が見えることを確かめる:

  ```bash
  nvidia-smi
  uv run python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.device_count())"
  ```

## GPU でやること

### 1. CUDA のテスト（必須）

CUDA のテストは `@pytest.mark.cuda(n)` が付いていて、GPU が無いと skip される（`tests/mujou/conftest.py`）。GPU のある環境で slow を含めて流す:

```bash
uv run pytest -m cuda -v          # CUDA のテストだけ（slow を含む）
just test                         # 全体（skip が CUDA の分だけ減るはず）
```

特に見るもの（どれも GPU の無い環境では一度も実行していない）:

- `tests/mujou/_core/test_driver.py` の `TestCuda`: Driver が hook ごとに device と stream を設定すること、`on_update` と `load_state_dict` の後の stream の同期、`super()` を呼ばない override でも同期が飛ばないこと、capture での stream、CPU のノードが CUDA を初期化しないこと（サブプロセス）
- `tests/mujou/_core/test_run.py` の CUDA のテスト（GPU のノードの状態の取得と復元）、`test_node.py`・`test_port.py`・`test_spec.py`・`test_state.py` の CUDA のテスト
- 2 GPU が要るテスト（`cuda(2)`）は、GPU が 1 枚なら skip のままでよい

落ちたら、実装（`src/mujou/_core/driver.py`、`state.py`、`codec.py`）を直す。テストの期待値は理論や仕様から書いてあるので、安易に変えない。

### 2. 計測（R10・R12・R13）

結果は `specs/research/benchmarks.md` の該当の節（「未計測（GPU なし）」と書いてある箇所）に書き足す。

| 項目 | コマンド | 何を見るか |
| --- | --- | --- |
| R10: GPU のノードの capture | `uv run python benchmarks/capture.py --device cuda` | 素の dict の木（停止中に device の上で複製し、再開の後に CPU へ移す）の ms/capture。CPU の値と並べる |
| R13: CUDA のノードの hook ごとの設定のコスト | `benchmarks/mn_scheduler.py` か小さなスクリプトで、CUDA のノードを多数ワーカーに載せた更新回数/s | Driver が hook ごとに `set_device` / `set_stream` するコスト。目立つなら `specs/open-decisions.md` の案（スレッドが切り替わったかを Driver に渡す）を検討する。core に enter のキャッシュは戻さない |
| R12: CUDA のノードで y を運ぶコスト | `benchmarks/pc_dense.py` に `--device` を足して測る（今は CPU だけ） | carry を廃止して、各ホップで y・mask・time を device と往復するようになったコスト。記録だけでよい（GPU の効率は設計目標ではない） |

MNIST を GPU で回す `--device` の引数は `examples/mnist/main.py` に無い。要るなら足してよい（examples は手引きなので、足すなら docs も合わせる）。

### 3. GIL ありの通常ビルド（GPU とは別。できれば同じ機会に）

GIL ありの環境では、これまで 3.14t に `PYTHON_GIL=1` を付けて代用しただけ。通常ビルドで確かめる:

```bash
UV_PROJECT_ENVIRONMENT=.venv-gil uv run --python 3.14 pytest -v      # slow を含む
UV_PROJECT_ENVIRONMENT=.venv-gil uv run --python 3.14 python benchmarks/ports.py
UV_PROJECT_ENVIRONMENT=.venv-gil uv run --python 3.14 python benchmarks/mn_scheduler.py
```

見るもの: インタプリタの終了のテスト（`test_run.py` の `TestInterpreterExit`。監視スレッドを消して atexit だけにした R5）、ワーカー数の既定を半分にする経路。CI の 3.14 のジョブでも流れる。

## 終わったら

- `specs/research/benchmarks.md` に数値を書き、`specs/open-decisions.md` の「検証が残っているもの」から済んだ項目を消す
- `memory/project_restructure_2026_10_20.md` の最後の段落（GPU と GIL ありの確認が残っている）を直す
- 直したものを commit する（CLAUDE.md の Git 運用。`just run` を通してから）。このファイルは、すべて済んだら消してよい

## PR

（PR を出した後に URL を書く）
