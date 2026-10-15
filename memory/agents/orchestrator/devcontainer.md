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
