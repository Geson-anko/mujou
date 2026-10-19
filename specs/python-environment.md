# Python 環境

## 対応バージョン

開発は free-threaded 版の Python 3.14t で行う。`.python-version` は `3.14t`、`requires-python` は `>=3.14` である。GIL ありの通常ビルドもサポートし、CI のテストは 3.14t と 3.14 の両方で回す。

## free-threaded 版を使う理由

Python のノード（PyTorch を使うものなど）を、ワーカーと専用スレッドで並列に動かしたいからである（[ネイティブコア](native-core.md#mn-スケジューリング)）。

GIL ありの環境では、ワーカーと専用スレッドが hook を呼ぶ間だけ GIL を取り合う。取り合うのはノードの数でなく、その本数である。torch の演算と Rust のコアは GIL を解放して並列に動くので、利用者の環境を free-threaded 版に限定する理由はない。

3.13t を選ばなかったのは、3.13 の free-threaded ビルドが実験的な扱いで、1 スレッドでの性能低下も大きいためである。free-threaded ビルドは 3.14 から公式サポートになった（PEP 779）。

## 実装上の注意

依存（特に C 拡張）を追加するときは、free-threaded に対応しているかを確認する。未対応の C 拡張を import すると、GIL が自動で有効に戻ってしまう。

Rust 拡張では、PyO3 で free-threaded 対応を明示する（`#[pymodule(gil_used = false)]`）。また、テストでも実装でも、GIL による暗黙の排他に頼らない。

pre-commit のフックも Python 3.14 の環境で動かす（`.pre-commit-config.yaml` の `default_language_version`）。フックはソースを parse するので、3.13 以降の構文（型パラメータの既定値など）を読めるバージョンでなければならない。CI の pre-commit ジョブは、`python3.14` という名前の実行ファイルを確実に用意するために GIL ありの 3.14 を使う。

## 依存

実行時の依存は `torch` と `numpy` である（`pyproject.toml` の `dependencies`）。
