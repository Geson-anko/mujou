# devcontainer（PR 1）

2026-10-15、solo-dev-cycle。

## 計画

- `.devcontainer/Dockerfile`: devcontainers/base:ubuntu-24.04 + uv 0.13.0 + just 1.58.0 + rustup（minimal、rustfmt・clippy）。Python 3.14t は uv が `.python-version` から入れる
- `.devcontainer/devcontainer.json`: `.venv` と `target/` を名前付き volume にする、`uv sync`、VS Code 拡張
- 段階 2 は省略（テスト対象のコードが無い）。検証はコンテナ内の `just run`

## 計画外の判断

- `just setup` を使わない: `pre-commit install` はフックに Python の絶対パスを書くので、コンテナで実行するとホストのフックを上書きする。フックが無いときだけ入れ、`.venv/bin` を PATH に入れてホストのフックからも pre-commit が見えるようにした
- Docker Desktop の bind mount（fakeowner）は 644 のファイルでも `access(X_OK)` が真を返し、`check-executables-have-shebangs` が全ファイルで落ちた。コンテナ内だけ `GIT_CONFIG_*` で `core.fileMode=false` にした（このフックは fileMode=false なら git index のモードを見る）。共有の `.git/config` は変えない
- `check-json` がコメント付き JSON を弾くので、devcontainer.json の説明は README に置いた
- Rust ツールチェーンは PR 2 より先にここで入れた（コアが Rust であることは決定済み）

## 検証

- `devcontainer up` → `devcontainer exec just run` が green（pytest 21 passed、pyright 0 errors）
- ホストの `.venv` と `.git/hooks/pre-commit` は変化なし

## レビュー（code-reviewer、approve）と対応

- should-fix `target/` が ignore されていない → `.gitignore` に追加。PR 2 と `.gitignore` が衝突しないよう、PR 2 をこのブランチの上に積む
- should-fix worktree をコンテナで開くと git が動かない → README に「メインのチェックアウトを開く」と書いた
- nit `core.fileMode=false` の副作用、新しい clone をコンテナで先に開いたときのフック、CLI の入手先 → README に追記
- nit rustup が `~/.profile` などに PATH を追記する → `--no-modify-path`
- nit ツールの版の固定 → 見送り。Rust の版を固定するかどうかは別途決める

## ユーザーのフィードバック（2026-10-15）と対応

- 「worktree も devcontainer で実行する」→ ワークスペースをホストと同じ絶対パスにマウントした（worktree の `.git` ファイルがホストの絶対パスを持つため）。`.venv` の volume はやめ、`UV_PROJECT_ENVIRONMENT=.venv-devcontainer`（相対パスはプロジェクトのルート基準で解決されることを確認）で checkout ごとに Linux の venv を分けた。cargo の target は volume 上の `CARGO_TARGET_DIR` を全 checkout で共有（cargo がロックする）。`just container <cmd>` で今いる checkout のコマンドをコンテナで実行する
- 「root で書き込まれないか、uid/gid を合わせるユーザーと NOPASSWD が要るか」→ remoteUser は `vscode`。macOS（Docker Desktop）ではコンテナで作ったファイルもホストの uid 501 になることを確認。Linux では Dev Container CLI の `updateRemoteUserUID`（既定 true、明示した）が vscode の UID/GID をホストに合わせる。base イメージの vscode は NOPASSWD の sudo を持つが、volume の chown をやめたので postCreate で sudo は使わない。自前のユーザー作成は不要
- `.vscode/settings.json` の `python.defaultInterpreterPath`（`.venv`）はワークスペース設定でコンテナ側の設定より優先されるので削除し、コンテナ側は customizations で `.venv-devcontainer` を指定。ホストは Python 拡張が `.venv` を自動で選ぶ
