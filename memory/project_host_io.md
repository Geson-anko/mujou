---
name: project-host-io
description: ホストの入出力。run はバックグラウンドで走り、ホストは Run の write / read / read_with_time / subscribe / wait_for / state_dict で話す。時刻は Timestamp | None（None はまだ書かれていない）。has_received は無い。inputs の時刻は起動の時刻。複数 Port の write は版番号で一斉に切り替わる
metadata:
  type: project
---

2026-10-14 決定（方針はユーザー、細部はユーザーの指示で Claude が考案）。詳細は `specs/network.md` の「ホストの入出力」。

- **`run()` は常にバックグラウンドで実行する（ユーザーの決定）。** 全ノードの `on_start`（と復元）が終わってから戻る（起動中の例外を `run()` が送出するため）。ホストは `Run` のメソッドだけでやり取りする
- **書けるのは未接続で受け取る `Port` だけ**（K17）。ホストはそのスロットの唯一の書き手。`MultiPort` には書かない
- **初期値は 3 つの道:** 送るポートの `initial`（時刻 `None`）、`mujou.run(..., inputs=)`（起動の時刻、復元なら保存した時刻）、走り出した後の `run.write`（書いた時刻）
- **時刻は `Timestamp | None`。** `None` は書き手がまだ書いていない（initial か zeros）。`recv_with_time` / `read_with_time` / `subscribe` / `wait_for` が同じ型で返す。`has_received` は無い。「送られたか」と時刻を 1 回の読み出しから得るので、両者が別のメッセージの情報になる競合が起きない
- **複数のポートへの同時書き込み（ユーザーの決定）。** 1 回の `write` の値はどのノードからも一斉に切り替わり、同じ時刻が付く。方式は版番号（外から書けるスロットは現在の値と版番号付きの次の値を持つ）。書き手も読み手も待たない
- **読み出しは `read`（最新値）、`read_with_time`、`subscribe`（呼んだ時点の値を最初に返し、新しい値ごとに最新値を返すイテレータ）、`wait_for`（条件を満たす最初の値）。** 消費者だけが待ち、途中の値は飛ばす。正常終了でイテレータが終わる。`send_to` は outbox に書かないので、`send_to` だけを使うポートを読むと `initial` か zeros が返る（推奨案で、ユーザーの確認待ち。`specs/open-decisions.md`）
- **write の値は、ポートを持つノードの CUDA device の上のテンソルだけ device に置く（2026-10-25、ユーザーの決定）。** `inputs=` も同じ。持ち主が CPU のノードのとき、持ち主と別の device のテンソル、CPU のテンソルはホストのメモリ。最初は「渡したテンソルの device」と決めたが、持ち主以外の読み手がホストの GPU の仕事を待つと分かり、同じ日に狭めた。`device=` 引数は足さない。同期は S2 と同じで、write が呼んだスレッドのその device の current stream を待つ（[[project-gpu-policy]]）
- **入力はその実行のタイマー上の信号（ユーザー了承）。** スロットは最新値なので sample-and-hold。データセットでは提示時間をユーザーが決め、ホストは `run.timer.sleep` で待つ
- **持たないもの:** 書き込みごとのコールバック、全値の記録（future work）、外部プロセスへの転送、サンプル ID の伝搬

**Why:** 外の消費者や生産者の速さがノードの歩調に影響しないこと（非同期性）を、入出力の境界でも守るため。

関連: [[project-network-ports]], [[project-shutdown-on-exception]], [[project-message-types]], [[project-run-node-and-write-time]]
