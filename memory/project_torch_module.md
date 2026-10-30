---
name: project-torch-module
description: PyTorch を必須の依存として全面採用（mujou.torch に分けない）。torch に依存させないのは _native と mujou.time だけ（ほかは要るなら使ってよい）。利用者が継承する mujou.Node は純 Python で、Rust は Driver を hook 1 回につき 1 回呼ぶ（R3）。Driver が CUDA の device・stream と同期を受け持つ。乱数生成器は super().__init__ で作り状態に入る。CPU と CUDA だけ
metadata:
  type: project
---

2026-10-15 決定（PyTorch の全面採用）。モジュールの構成は `specs/architecture.md`、ノードの仕様は `specs/nodes.md`。

- **PyTorch は必須の依存。** 利用者の名前は `mujou.*` から import する（モジュールの置き方は [[project-public-modules]]）。torch に依存させないと決めているのは Rust の拡張モジュール `mujou._native` と `mujou/time.py` だけで、ほかのモジュールは要るなら torch を使ってよい。バックエンドで分けた `mujou.torch` / `mujou.jax` は持たない（必要なら正式版で検討する）
- **`mujou.Node` は純 Python。** Rust は Python の `Driver`（`driver.py`、ノードに 1 つ）のメソッドを hook 1 回につき 1 回だけ呼ぶ。Driver が、CUDA の device と stream の設定、ノードごとの `contextvars.Context`、`on_update` と `load_state_dict` の後の stream の同期、`state_dict` の複製を、hook の外で行う。R3 として 2026-10-21 にユーザーが承認した（以前は `mujou.Node` がコアの `Node` を継承し、private hook で設定していた。[[project-restructure-2026-10-20]]）
- **乱数生成器は `super().__init__` で作る（2026-10-18、ユーザーの決定）。** `__init__` の中から使え、module の初期化にも `generator=self.generator` を渡す。既定の `state_dict` が生成器の状態を含める
- **状態は `__init__` で作り、その場で更新する。** 毎ステップ大きなテンソルを確保すると、多数のスレッドでメモリ確保が律速になる（7950X、n=1024、16 スレッドで 1.1k → 36.9k ステップ/秒）
- **CUDA のテンソルやモジュールは `on_start` 以降で作るよう docs で勧める（2026-10-19、ユーザーの決定）。** キャッシュアロケータがメモリを確保した時点の stream に結び付けるため。GPU の実機では未確認
- torch の計算をするスレッド（ワーカー + 専用スレッドのノード）× 内部スレッド数は物理コア数を超えないようにする。内部スレッド数は mujou が設定せず、ホストが決める
- 対象デバイスは CPU と CUDA だけ（[[project-gpu-policy]]）

**Why:** コアは GPU の意味論（device・stream・event）を持たないので、ノードの内側で GPU を使うときと CUDA のメッセージを送受信するときにノード間の独立性を保つ設定を、作者が毎回書かずに済むようにする。Driver を Python に置き hook の外で設定するので、`super()` を呼び忘れても CUDA の同期は飛ばず、1 回の update で FFI をまたぐのも 1 回になる。

**JAX は不採用（2026-10-15）。** 3.14t で多数スレッドからの呼び出しが伸びず、論理コア数のスレッドで崩壊した。GPU でもノードごとに stream を分けられない（`specs/research/benchmarks.md` の「ノードの内側のライブラリ」）。

関連: [[project-gpu-policy]], [[project-node-lifecycle]], [[project-grad-on-by-default]]
