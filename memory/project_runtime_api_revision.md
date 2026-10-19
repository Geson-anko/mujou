---
name: project-runtime-api-revision
description: 2026-10-16 のユーザーの決定。_core パッケージへの集約と Rust の _core._native、stub の自動生成、run がハンドルを返す、Group、実行ごとのタイマーと Timestamp、終了後の state_dict、exception_state_dict の削除、自動の状態収集、hook の名前ほか
metadata:
  type: project
---

2026-10-16、コア機能の PR stack（#7〜#13）へのユーザーの回答。

- **モジュール構成:** Node・Port・Network などの Python 実装は `mujou._core` パッケージに集める。Rust の拡張モジュールは `mujou._core._native`。torch に依存しないのは `_native` だけになる（それ以外の `_core` は torch を使う。のちに時刻まわりの `_core/timer.py` と `mujou.time` も torch を import しない形になった）。stub は手書きせず自動生成する
- **run の API:** `build()` は Graph を返し、何度呼んでも同じ Graph を返す。`Network.run` は無くす（build 後の Graph を mujou が実行する）。`h = mujou.run(*networks, timer=None, inputs={port: x}, state=tree)`（2026-10-17 に `mujou.run(component, *components, ...)` に改めた、[[project-run-node-and-write-time]]）がハンドルを返し、ハンドルが write / read / subscribe / state_dict / shutdown / join を持つ（with 文で使える）。初期値と復元は run の引数で渡す
- **複数のネットワーク:** 同時に走らせる想定はないが、`mujou.run(*networks)` と `Group(*networks)` を用意する（2026-10-17 に `mujou.run(component, *components)` / `Group(*components)` に改め、ノードも受け付ける。[[project-run-node-and-write-time]]）。パスは位置なら `"0.l1"`、キーワード（`Group(enc=..., dec=...)`）なら `"enc.l1"`。独立に回すときは `mujou.run(timer=...)` で実行ごとにタイマーを渡す。`timer=None` は既定のグローバルタイマー。ノードはタイマーを属性として持つ
- **時刻:** `Timestamp` 型。`float()` / `int()` は秒、ナノ秒は property
- **状態:** `state_dict()` は実行中と正常終了の後に呼べる（作る → 走らせる → 保存する）。例外で終了した後は `NetworkError`。例外時の状態は保存しない（エラー解析にしか使わないため）。`exception_state_dict()` と `capture_state_on_exception` は削除。mujou 管理下のオブジェクトの救済は、自動の状態収集が入ってから別に設計する
- **自動の状態収集:** nn.Module のように `__setattr__` をフックし、名前の `_` の有無にかかわらず torch.Tensor と nn.Module の属性を集める（Port は除く）。既定の load_state_dict はその場で copy_ する。override すれば自分で書ける
- **hook:** `on_thread_start` は利用者が override してよい公開の hook にする（`_` を外す。2026-10-19 にユーザーの決定で廃止した、[[project-mn-scheduler]]）。`_after_update` / `_capture_state` は内部の hook として `_` のまま。2026-10-17 に改めた: 外せない設定（grad の無効化・CUDA のデバイスとストリーム・乱数生成器・所有スレッドの記録・ホストのメソッドを呼ばせないガード）は内部の hook `_on_thread_start` で済ませ、Rust から先に呼ぶ。公開の `on_thread_start` は既定で何もしない hook で、`super()` を呼ぶ規約は無い。2026-10-18 のユーザーの決定で、乱数生成器は `_on_thread_start` ではなく `super().__init__` で作るように変えた（`__init__` の中から使える）
- **自動の状態収集の細部（2026-10-17）:** `state_dict` の型は `Any`、nn.Module の不一致は torch と同じ `RuntimeError`、optimizer と乱数生成器は自動で集めない、`__setattr__` でフックする（実装者の推奨案を承認）
- **終了:** `shutdown` / `join` は `timeout=math.inf`。停止のタイムアウトの既定は実時間 30 秒。threading の private API は使わない
- **subscribe:** 呼んだ時点の値を最初に 1 つ返す
- **テスト用の口:** 利用者が使うべきでないもの（`_manual` など）は公開の API に見せない
- **TensorSpec:** `empty()` を足し、`zeros` / `rand` / `empty` に `device` を足す。それ以外の初期化は `fill_` などのその場の操作に任せる
- **ツール:** pre-commit は Python 3.14。numpy を依存に足す
- **PC-ALM:** まだ着手しない
- on_start の例外を run 自身が送出する形は承認

**Why:** ビルド後の構造は宣言とは別のオブジェクトで、mujou がそれを実行する（ユーザー）。実行ごとのタイマーはデバッグしやすくする。利用者が使うべきでないものは見えてはいけない。

関連: [[project-run-node-and-write-time]], [[project-host-io]], [[project-network-ports]], [[project-node-lifecycle]], [[project-pause-and-state]], [[project-torch-module]]
