---
name: project-shutdown-on-exception
description: ノードのどの hook が例外を投げてもシステム全体を終了する（pamiq-core 参考）。end は失敗を検出した時刻。全ノードの on_exception → on_stop を呼び、以後のホストの呼び出しが NetworkError を出す。例外のときの状態は取得しない。例外を投げたノードだけを止める・再起動する・スキップすることはしない
metadata:
  type: project
---

2026-10-14 決定（ユーザーの指示）。詳細は `specs/execution-model.md` の「終了と例外」。

- **どの hook の例外でも全体を終了する。** `on_update` に限らない
- **流れ:** ノードを走らせているスレッドが例外を捕まえ、ノードの名前とトレースバックをログに出して記録 → その場で実行の時計（RunClock）に end を付け、終了要求のフラグを立てる → 全ノードが区切りで止まり、それぞれ `on_exception()`（例外を投げていないノードも含む。例外を投げたノードが先）→ 全ノードの `on_stop()` → ホストに最初の例外を伝える（2 つ目以降はログのみ）。`on_exception` と `on_stop` の例外はログに出して続ける
- **end は失敗を検出した時刻。** 終わった後の `on_exception` / `on_stop` では `self.timer` は止まっているので、減速のような処理は実時間で測る
- **例外のときの状態は取得しない（2026-10-16、ユーザーの決定）。** 例外で終わった後の `state_dict()` は `NetworkError`。保存したい状態は、実行中か正常に終わった後に取る（[[project-pause-and-state]]）
- **ホストへの伝え方:** 起動中の例外は `mujou.run` が送出する。起動した後の例外は、その後のホストの呼び出し（write、read、read_with_time、subscribe とそのイテレータ、wait_for、state_dict、shutdown、join、step）が `NetworkError` で送出し、最初の例外を `__cause__` につなぐ
- **pamiq-core との違い:** pamiq は制御スレッドが tick ごとに例外フラグをポーリングする。mujou はループをフレームワークが持つので、例外を捕まえた場所で直接終了フラグを立てる

**Why:** 例外を投げたノードだけを止めると、そのスロットに最後の値が残り、隣のノードが古い値を読み続けて、気づかれないまま平衡から外れていく。fail fast にする。

**How to apply:** ランタイムに、ノード単位の再起動やステップのスキップのような回復処理を足さない。フレームワークは保存しない。

関連: [[project-node-lifecycle]], [[project-pause-and-state]]
