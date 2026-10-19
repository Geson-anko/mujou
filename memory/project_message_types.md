---
name: project-message-types
description: ノード間メッセージの型とスロットの仕様の決定。利用者が扱う値は torch.Tensor だけ（コアの値の型・mujou.dtypes・mujou.default.dtype は公開しない）、コアとは DLPack で必ずコピー、メッセージ型は torch.Tensor / TypedDict、TensorSpec(shape, torch.dtype)、zeros/rand、初期値必須、片方向スロット 2 本で双方向、時刻（秒 f64 + ns u64）自動付与、バッファプールのみ、ネットワークは define-and-run のネットワークオブジェクト
metadata:
  type: project
---

2026-10-13 決定。詳細は `specs/messages.md`。

- **双方向通信は片方向スロット 2 本。** 各スロットの書き手は所有ノード 1 つ、読み手は N 個可
- **利用者が扱う値は `torch.Tensor` だけ（2026-10-15 変更）。** Port の `recv()` / `send()`、ホストの `write` / `read`、`TensorSpec.zeros()` / `rand()` はすべて `torch.Tensor`。コアの値の型（旧 `mujou.Tensor`）は `mujou._core._native` の内部に閉じ、`mujou.dtypes` と `mujou.default.dtype` は廃止。NumPy との変換と `ml_dtypes` も不要になった
- **dtype は torch の dtype。** float16・bfloat16・float32・float64・int8・int16・int32・int64・uint8・bool。バッチは shape の一部で特別扱いしない
- **コアの中の値は、ホストのメモリ上の不変なバッファ（連続 row-major）。** torch とは DLPack で受け渡し、読み書きとも必ずコピーする（ゼロコピーにすると、利用者が受け取った値をその場で書き換えたときに他の読み手のバッファまで変わるため）。コアは独自の dtype を持たず DLPack の dtype 表現を使う。NumPy 経由では bf16 を渡せないので DLPack を選んだ。書くときは `detach()` し（`requires_grad` の値は `__dlpack__` で書き出せない）、GPU 上の値はホストへコピーしてから渡す。ノードの `send()` とホストの `write()` で同じ。ホストの `read` / `subscribe` と `TensorSpec.zeros()` / `rand()` は CPU 上の値を返す。コアが受け取るのは `kDLCPU` と `kDLCUDAHost`（pinned）の値だけ。`torch.dtype` → DLPack dtype の対応表は torch 依存の側が持つ
- **メッセージの型は `torch.Tensor` と `TypedDict` の 2 種類。** TypedDict はキー名と値の型（`torch.Tensor`）を明示し、入れ子は許さない
- **`torch.Tensor | None`（値が無いこと）は入れない（2026-10-14 変更）。** 一度は入れると決めたが取り消した。必要になったときに拡張する
- **`TensorSpec(shape, dtype)` で値の形を表す。** dtype は torch の dtype。省略すると、TensorSpec を作った時点の `torch.get_default_dtype()` を読み取って確定し、あとで変えても既存の spec・スキーマは変わらない（スキーマを後から変えない決定と揃えるため）。`TensorSpec.zeros()`（全 dtype）と `TensorSpec.rand()`（浮動小数点のみ。一様 [0, 1)、seed か `torch.Generator` を渡せば決定的）で初期値の `torch.Tensor` を作れる。正規分布が要れば `randn` を足す。spec の一致は dtype を含み、暗黙の dtype 変換はしない（dtype が違う Port は接続できない）
- **Port には spec を渡す。** `torch.Tensor` のメッセージには `TensorSpec`、`TypedDict` のメッセージには TypedDict と各キーの TensorSpec の対応表（例: `DictSpec(Feedback, error=TensorSpec(...))`）。対応表は TypedDict と照合し、キーの過不足はエラー
- **スロットは常に宣言どおりの型の値を持つ。** スキーマ作成時に初期値を与える。既定はゼロ（`TensorSpec.zeros()`）。2026-10-17、ユーザーの決定で、送る側の Port が `initial=` で初期値を宣言できるようにした（`NextPort(send=spec, initial=x0)`。各 Channel と外向きのスロットがその値で始まり、復元では木の値が勝つ）。`on_start` での `send` は復元した値を上書きするので、初期値の手段にしない。`initial` が `send` の型推論を広げないよう、`MessageSpec` の型引数は不変にした
- **スキーマ（型 + shape + 初期値）は Port の spec と `initial`（ホストが書くスロットは `mujou.run(inputs=)`）で決まり、以後は変えない。スロットは `build` ではなく `mujou.run` の時点で作る（`specs/network.md`）。** 非同期処理の単位はノードオブジェクト。mujou はノードの組み合わせと双方向の接続を定義するネットワークオブジェクトを提供し、define-and-run で、`nn.Module` のように宣言的に組み合わせられる API で記述して最終的にスキーマオブジェクトを作る
- **「Module API」は比喩。** `nn.Module` に相当する単位がノードになるわけではない（Module とノードは対応しない）
- **スロットは書き込み時のその実行のタイマーの時刻を自動で添える**（2026-10-16 にグローバル時刻から改めた、[[project-timer-per-run]]）。時刻はナノ秒（u64）で持ち、秒（f64）はそこから計算する
- **メモリ管理はバッファプールだけ。** seqlock + copy は、メモリが増えても安全性と速度を優先して不採用
- **複数マシンにまたがる実行は現状想定しない**

**Why（バッファプール）:** 実行中にヒープへの確保と解放を毎回起こしたくない、というのがユーザーの動機。読み手は全員読み取り専用なので、N 個の読み手が同じ最新バッファを共有できる。書き手は常に別のバッファに書くので、書き込み先は競合しない。だから読み手の数に比例したバッファは要らず、参照カウントで解放されたバッファをプールに戻して再利用すればよい。

**How to apply:**

- 空きが無ければ書き手が追加で確保し、読み手を待たない。定常状態でヒープへの確保と解放が起きないことを目標にする。プールは空で始め、必要なだけ確保し、縮小しない（実装）
- Python の読み手（`mujou.Node`・ホスト）は読むときにコピーするので、バッファを参照するのはコピーの間だけ（Rust のネイティブなノードは `on_update` の間だけ直接参照する）。読み手の数から必要なバッファ数を見積もる設計にしない
- コアの値に演算を足さない。演算はノードの内側で torch を使って行う
- 未決定（勝手に決めずにユーザーに確認する）: ネットワークオブジェクトの API の具体的な形（shape の決め方など）

関連: [[project-async-node-execution]], [[project-native-core-rust]], [[project-core-message-framework]], [[project-open-decisions]]
