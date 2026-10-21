---
name: project-runtime-api-revision
description: 実行の API の形。実装は mujou._core、Rust は mujou._core._native。build と Graph は持たず、mujou.run がハンドル Run を返す。state_dict は実行中と正常終了の後、例外の後は NetworkError。shutdown / join の timeout は既定 math.inf
metadata:
  type: project
---

2026-10-16 のユーザーの回答と、その後の作り直しで決まった形。

- **モジュール構成:** Python の実装は private な `mujou._core`、Rust の拡張モジュールは `mujou._core._native`。利用者の名前は `mujou`、`mujou.time`、`mujou.testing`、`mujou.pc` が再エクスポートする（`__all__` と `tests/mujou/test_init.py` が正本）。stub は自動生成する（[[project-core-stub]]）
- **build と Graph は持たない。** `mujou.run` が検査を兼ね、走った事実は `run.py` の印だけが持つ。build は「build した後は Group に入れられない」という凍結の罠を生んでいた
- **run がハンドルを返す。** `mujou.run(root, ...)` が `Run` を返し、`Run` が write / read / read_with_time / subscribe / wait_for / state_dict / shutdown / join と `timer` を持つ（with 文で使える）。初期値と復元は run の引数で渡す
- **状態:** `state_dict()` は実行中と正常に終わった後に呼べる（作る → 走らせる → 保存する）。例外で終わった後は `NetworkError`（例外時の状態は保存しない）
- **終了:** `shutdown` / `join` / `state_dict` の `timeout` は実時間の秒で、既定は `math.inf`
- **subscribe:** 呼んだ時点の値を最初に 1 つ返す
- **TensorSpec:** `zeros` / `empty` / `rand` に `device` がある。それ以外の初期化は `fill_` などのその場の操作に任せる
- on_start の例外を run 自身が送出する形は承認

**Why:** 走らせる構造は宣言とは別に実行の側が持つ（ユーザー）。利用者が使うべきでないものは見えてはいけない。

関連: [[project-run-node-and-write-time]], [[project-host-io]], [[project-pause-and-state]]
