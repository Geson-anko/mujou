---
name: project-pause-and-state
description: 停止と状態の取得・復元。フラグを立ててすぐその実行のタイマーを凍結 → 全ノードが区切りで停止 → device の上で状態を複製 → 再開 → CPU へ移してホストに返す。状態の木は素の dict（timer と nodes。ノードごとに state・last_update_ns・ports）。保存はユーザー。グローバルなバリア禁止の例外
metadata:
  type: project
---

2026-10-14 決定（停止の流れ）。木の形の正本は `Run.state_dict` の docstring、流れは `specs/execution-model.md` の「停止と状態の取得・復元」。停止の流れは pamiq-core の `ThreadController` を参考にした。

- **停止:** 停止要求のフラグを立て、続けてすぐにその実行のタイマーを時刻 T で凍結する → 各ノードは実行中の `on_update` を終え、区切りで止まって `on_pause()` → 全ノードの停止を確かめる。pamiq-core と順序が逆なのは、待ってから凍結すると早く止まったノードの待ち時間が再開後の Δt に混ざるから
- **再開:** タイマーを再開し、続けてすぐにフラグを下ろす（最初の Δt が 0 にならない）
- **取得:** 停止中に各ノードの Driver がノードの stream の上で `state_dict()` を複製し（テンソルは detach してから）、再開した後にホストのスレッドで CPU へ移して返す。スロットが device の上に持つ値（[[project-gpu-policy]]）も、停止中は参照だけを木に入れ（`codec.peek`）、再開した後に CPU へ移す。停止中に CPU へコピーすると停止が延びる
- **正常に終わった後も取れる（2026-10-16、ユーザーの決定）。** 木の時刻は停止を求めた時刻（RunClock の end）。終わった後の `write` は断る。例外で終わった後は `NetworkError`
- **木は素の dict:** `{"timer": {"time_ns", "time_scale"}, "nodes": {パス: {"state", "last_update_ns", "ports": {属性: {"out", "to": {相手のパス: E}, "in"}}}}}`、`E = {"value", "time_ns": int | None}`。テンソルは CPU のコピーなので、`torch.save` と `torch.load(weights_only=True)` がそのまま通る
- **スロットも木に入れる:** 系の状態はノードの状態とスロットの中身。スロットを初期値に戻すと再開直後に平衡から外れる
- **復元:** `mujou.run(..., state=tree)` で新しいインスタンスに戻す。保存した時刻から始まる専用のタイマーで走り（[[project-timer-per-run]]）、`on_start()` → `load_state_dict()` の順。木の形が違えば段 2 で断る（K21）
- **保存はユーザーが行う（2026-10-14、ユーザーの指摘）。** フレームワークもノードもファイルを扱わない

**Why（バリアの例外）:** 停止はシステム全体のバリアだが、停止中はその実行の時間が進まないので、系のダイナミクスからは存在しないのと同じ。ユーザーが提案し、2026-10-14 に例外として認めた。

**How to apply:** 停止を使ってよいのは状態の取得・復元や観測などの制御のための操作だけ。定期的に止めて重みを平均するなど、停止を学習アルゴリズムの一部に使う実装は不変条件違反として扱う。

関連: [[project-node-lifecycle]], [[project-async-node-execution]]
