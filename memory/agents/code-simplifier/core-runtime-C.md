# core-runtime WP-C（Python の構造）の簡素化

対象: `src/mujou/{_spec,_port,_network,_node}.py`（`chore/2026-10-15/core-scaffold..HEAD`）。公開 IF（計画書と判断ログの IF 変更通知にある名前・引数・戻り値・例外）は変えていない。振る舞いと例外メッセージの文面も同じ。

## 変更

- `_network.Network.build` の `visit` を短くした（大きな書き換えではない）
    - 宣言の検査（クラス属性の Port、PrevPort / NextPort が 2 つ以上）を `_check_declarations(component, where)` に切り出した。`visit` は木を辿ることと、ノード・Port の登録だけになった
    - 2 回ずつ書いていた式を `_child(path, name)`（子のパス）と `_where(path)`（エラーメッセージでの位置。ルートは `the root`）にまとめた
    - `Graph.port_path` と build のエラーメッセージで同じ「ノードのパス + 属性名」を組み立てていたので、`_path` を共有した（引数を `Sequence` にした）
- `_network._sides` の重複除去を `all(port is not p ...)` から `port not in found` にした（Port は `__eq__` を持たず identity で比べる）
- `_spec.DictSpec.__eq__` を、items の tuple の比較からマッピングの比較にした。同じ TypedDict ならキーの順は同じなので結果は変わらない

## 変えなかったもの

- `_port.py`: 3 つの `__init__` が `_init` を呼ぶだけなのは、self 型 overload のため（判断ログ）。`_init`・`freeze`・`is_frozen` は D が使うので残した
- `_spec.py`: `DictSpec.__init__` と `split` のキーの過不足の検査は似ているが 2 か所だけで、メッセージも違うので抽出しなかった
- テスト: 重複が少なく、公開 IF を通している。変更なし
- docstring: 振る舞いを変えていないので、公開 API の docstring はそのまま現状と合っている。レビュー N7（`recv` が接続済みでも未送信なら初期値を返す）は D で本体を入れるときに直す

## 検証

`just run`: pre-commit 変更なし、pytest 134 passed（coverage 100%）、pyright 0 errors
