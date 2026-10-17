# tooling-py314 の判断ログ

## pre-commit のフック環境を 3.14 にする

- 原因: `default_language_version: python: python3` が、フック環境を作った時点の `python3`（Python 3.12.8）に解決されていた。`debug-statements` などは 3.13 以降の構文（型パラメータの既定値）を parse できない
- 変更: `python: python3.14`。pre-commit は言語バージョンごとに環境ディレクトリを分ける（`py_env-python3` と `py_env-python3.14`）。そのため、`~/.cache/pre-commit` を消さずに済んだ（`pre-commit clean` は使っていない）。他の worktree が使っている `py_env-python3` はそのまま残っている
- ローカルでは `uv run pre-commit` の PATH の先頭に `.venv/bin` が来るので、`python3.14` は venv の 3.14t（free-threaded）に解決される。この環境で全フックが通ることを確かめた
- CI の pre-commit ジョブは `setup-python` の `3.14t` を `3.14` に変えた。3.14t のセットアップで `python3.14` という名前の実行ファイルが PATH に入る保証を確認できなかったためである。フックに free-threaded は要らない。テストと型チェックのジョブは触っていない
- `docs/python-environment.md` に上記を 1 段落で足した。他の文書に 3.12 前提の記述はなかった

## numpy

- `uv add numpy`（numpy 2.5.4、本体の依存）。torch の import 時に出ていた "Failed to initialize NumPy" の警告が消えた
- numpy を import しても `sys._is_gil_enabled()` は False のまま（free-threaded 対応済み）

## `_port.py` の PEP 695 化

- `Port` / `PrevPort` / `NextPort` を `class Port[S = Never, R = Never, M: (Single, Multi) = Single]` の形に戻した。`NextPort` の既定は `M = Multi`。そのため、旧 `MN` という別名の型変数は不要になった
- オーバーロードの method スコープの型パラメータ `S2` / `R2` は、クラスの `S` / `R` を隠さないための名前なので残した
- pyright strict（`src/`）と `tests/mujou/test_port.py` の `assert_type` 12 か所が通ることを確かめた。`assert_type` が実際に検査されていることも確かめた（1 か所を `int` に書き換えたコピーで `reportAssertTypeFailure` が出た）
- `grep -rn "3.13\|3.12\|pre-commit" src tests` で引っかかった回避は `_port.py` の 1 か所だけだった（`tests/test_claude_hooks.py` の "pre-commit" は無関係）。`TypeVar` / `Generic[` の残りもない
