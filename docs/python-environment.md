# Python 環境

## 対応バージョン

- **開発環境は free-threaded 版の Python 3.14t** とする。`.python-version` は `3.14t`、`requires-python` は `>=3.14`
- **GIL ありの通常ビルドもサポートする。** CI のテストは 3.14t と 3.14 の両方で回す

## free-threaded 版を使う理由

Python のブロック（PyTorch を使うものなど）を専用スレッドで並列に動かしたいためである（[ネイティブコア](native-core.md)）。

GIL ありの環境では Python のブロック同士が GIL を取り合うが、Rust のコアは GIL を解放して並列に動く。したがって、利用者の環境を free-threaded 版に限定する理由はない。

3.13t を選ばなかったのは、3.13 の free-threaded ビルドが実験的扱いで、1 スレッド性能の低下も大きいためである。3.14 から公式サポートになった（PEP 779）。

## 実装上の注意

- 依存（特に C 拡張）を追加するときは、free-threaded 対応を確認する。未対応の C 拡張を import すると、GIL が自動で有効に戻る
- Rust 拡張は PyO3 で free-threaded 対応を明示する（`#[pymodule(gil_used = false)]`）
- テストや実装で、GIL の存在（暗黙の排他）に頼らない
