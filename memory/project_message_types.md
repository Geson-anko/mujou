---
name: project-message-types
description: メッセージの型とスロットの決定。利用者が扱う値は torch.Tensor だけ、コアとは DLPack で必ずコピー、メッセージ型は torch.Tensor か TypedDict、TensorSpec(shape, dtype=torch.float32)、初期値必須、スロットの書き手は 1 つ、時刻（ns、None はまだ書かれていない）自動付与、バッファプールのみ
metadata:
  type: project
---

2026-10-13 決定。詳細は `specs/messages.md`。

- **スロットの書き手は 1 つ。** 送るポートが outbox と相手ごとのスロットに書き、ホストが未接続の受け取るポートのスロットに書く。読み手は N 個可
- **利用者が扱う値は `torch.Tensor` だけ（2026-10-15）。** コアの値の型は `mujou._native` の内部に閉じる
- **dtype は torch の dtype。** 一覧は `TensorSpec.dtypes`（正本は `spec.DLPACK`）。バッチは shape の一部で特別扱いしない
- **コアの値はホストのメモリ上の不変なバッファ。** torch とは DLPack で受け渡し、読み書きとも必ずコピーする（ゼロコピーにすると、利用者が受け取った値を書き換えたときに他の読み手のバッファまで変わる）。変換は `codec.py` の 1 か所、shape と dtype の検査は Rust の `copy_from` の 1 か所
- **メッセージの型は `torch.Tensor` と `TypedDict` の 2 種類。** `TensorSpec` と `DictSpec(TD, **specs)`。値が無いこと（`Tensor | None`）は表さない
- **`TensorSpec(shape, dtype=torch.float32)`。** 既定を固定したのは、module のレベルの spec が import の順と既定の dtype の設定で変わらないようにするため。`zeros()` / `empty()` / `rand(*, generator=)`。spec の一致は dtype を含み、暗黙の dtype 変換はしない
- **スロットは常に宣言どおりの型の値を持つ。** 既定はゼロ。送るポートは `initial=` で初期値を宣言できる（2026-10-17、ユーザーの決定。`on_start` での send は復元した値を上書きするので、初期値の手段にしない）
- **スキーマは段 2 で spec から作り、以後は変えない。** ネットワークは宣言的に組み、`nn.Module` のような API で書く（Module とノードは対応しない比喩）
- **スロットは書き込み時のその実行の時計の時刻を ns で添える。** `None` は書き手がまだ書いていない
- **メモリ管理はバッファプールだけ。** seqlock + copy は、メモリが増えても安全性と速度を優先して採らない
- **複数マシンにまたがる実行は想定しない**

**Why（バッファプール）:** 実行中にヒープへの確保と解放を毎回起こしたくない、というのがユーザーの動機。読み手は全員読み取り専用なので N 個の読み手が同じ最新バッファを共有でき、書き手は常に別のバッファに書くので競合しない。

**How to apply:** 空きが無ければ書き手が追加で確保し、読み手を待たない。プールは空で始め、縮小しない。コアの値に演算を足さない。演算はノードの内側で torch を使う。

関連: [[project-async-node-execution]], [[project-native-core-rust]], [[project-core-message-framework]]
