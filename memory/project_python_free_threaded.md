---
name: project-python-free-threaded
description: 開発は free-threaded 版 Python 3.14t で行う。GIL ありの通常ビルドもサポートし、CI で両方を検証する
metadata:
  type: project
---

2026-10-11 決定。

- **開発環境は free-threaded 版の Python 3.14t。** `.python-version` は `3.14t`、`requires-python` は `>=3.14`
- **GIL ありの通常ビルドもサポートする。** CI のテストは 3.14t と 3.14 の両方で回す
- 3.13t を選ばなかった理由: 3.13 の free-threaded ビルドは実験的扱いで、1 スレッド性能の低下も大きい。3.14 から公式サポート（PEP 779）

**Why:** Python のノード（PyTorch など）を専用スレッドで並列に動かしたい（[[project-native-core-rust]]）。GIL ありの環境では Python のノード同士が GIL を取り合うが、Rust のコアは GIL を解放して並列に動くので、利用者の環境を free-threaded 版に限定する理由はない。

**How to apply:**

- 依存（特に C 拡張）を足すときは free-threaded 対応を確認する。未対応の C 拡張を import すると GIL が自動で有効に戻る
- Rust 拡張は PyO3 で free-threaded 対応を明示する（`#[pymodule(gil_used = false)]`）
- テストや実装で GIL の存在（暗黙の排他）に頼らない

関連: [[project-native-core-rust]]
