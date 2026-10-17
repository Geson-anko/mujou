---
name: project-host-io
description: ホスト入出力の設計。run はバックグラウンドで実行し、ホストは公開インターフェイス（write / read / subscribe / state_dict / shutdown / join）でやり取りする。複数 Port の write は版番号で一斉に切り替わる。入力はその実行のタイマー上の信号。run の戻り値・初期値・復元の渡し方は 2026-10-16 の決定で Run ハンドルと inputs= / state= に上書きされた
metadata:
  type: project
---

**2026-10-16 の [[project-runtime-api-revision]] と 2026-10-17 の [[project-run-node-and-write-time]] で、run の戻り値・初期値・復元の渡し方が上書きされた。** 今の形は `h = mujou.run(component, *components, timer=..., inputs={port: x}, state=tree)` がハンドル `Run` を返し、`Run` が write / read / subscribe / state_dict / shutdown / join を持つ。2026-10-17 に推奨案で `subscribe(port, *, timeout=)` と `wait_for(port, condition, *, timeout=)`（条件を満たす最初の値まで待つ。例の読み直しループを 1 行に）を足した（ユーザー確認待ち）。以下の打ち消し線と注記はその前の記録。

2026-10-14 決定（暫定）。ユーザーが方針（バックグラウンド実行、run 前の初期値書き込み、複数 Port の同時書き込み、read と subscribe）を決め、細部はユーザーの指示で Claude が暫定案として考案した。詳細は `specs/network.md` のネットワークの入力と出力の節。

- **2026-10-16 に実装。** ~~`mujou.run(network)` は未 build なら build して run し、同じ network を返す。~~（同日の決定で上書き: `Run` ハンドルを返す。[[project-runtime-api-revision]]）`run()` は全ノードの `on_start`（と復元）が終わり、タイマーが動き出してから戻る（起動中の例外を `run()` が送出するため。当初の「すぐ戻る」から変更）。~~`state_dict()` は実行中だけ（run 前・正常終了後は `RuntimeError`、例外終了後は `NetworkError`）。`subscribe` は呼んだ後に書かれた値だけを返す。~~（2026-10-16 の決定で上書き: `state_dict()` は正常終了後も呼べる、`subscribe` は呼んだ時点の値を最初に返す。[[project-runtime-api-revision]]、[[project-pause-and-state]]）`send_to` だけの Port の `read` は初期値を返す。詳細は `specs/network.md` の実装の節
- **`run()` は常にバックグラウンドで実行する（ユーザーの決定）。** ホストは公開インターフェイスだけでやり取りする。~~インターフェイスは `build()` を呼んだルートのインスタンスが持つ~~（2026-10-16 の決定で上書き: `mujou.run` が返す `Run` が持つ。[[project-runtime-api-revision]]）
- **操作:** `mujou.run(...)` と、`Run` の `write(...)`（`Timestamp` を返す、[[project-run-node-and-write-time]]）、`read(port)`、`subscribe(port)`、`state_dict()`、`shutdown()`、`join()`。~~`build()`、`load_state_dict(state)`（run 前）、`run()`~~（2026-10-16 の決定で上書き: 復元は `mujou.run(..., state=tree)`）
- **paused 起動は持たない（ユーザーの決定）。** ~~run 前に `write` した値がそのスロットの初期値になる。`load_state_dict` の後に書いた値が優先~~（2026-10-16 の決定で上書き: 初期値は `mujou.run(..., inputs={port: x})` で渡す。[[project-runtime-api-revision]]）
- **書く値・読む値は `torch.Tensor`（2026-10-15）。** `read` / `subscribe` は CPU 上の値を返す。`write` は値を `detach()` し、GPU 上の値はホストへコピーしてから受け取る（[[project-message-types]]）
- **入力はその実行のタイマー上の信号（ユーザー了承。2026-10-16 にグローバル時間から実行ごとのタイマーへ、[[project-timer-per-run]]）。** スロットは最新値なので sample-and-hold。データセットでは提示時間をユーザーが決め、ホストはその実行のタイマーで待つ（`Timer.sleep`。既定のグローバルタイマーなら `mujou.time.sleep`）。速くしたいときはタイマーを加速
- **複数 Port への同時書き込みは必要（ユーザーの決定）。** 1 回の `write` の値はどのノードからも一斉に切り替わり、同じ時刻が付く。暫定の方式は版番号: ネットワークが atomic な版番号 V を持ち、外から書けるスロットは「現在の値 + 版番号付きの次の値」を持つ。write は次の値（版 V+1）を全部置いてから V を進める。読み手は次の値の版が V 以下なら次の値を読む。書き手も読み手も待たない。停止してから書く方式は、書くたびに最も遅い `on_update` を待つので採らない
- **読み出しは `read`（最新値と時刻）と `subscribe`（新しい値ごとに最新値を返すイテレータ）（ユーザーの決定）。** 消費者だけが待ち、書き手は通知するだけ。途中の値は飛ばす。正常終了でイテレータが終わる。`subscribe` は呼んだ時点の最新値を最初に 1 つ待たずに返し、その後は新しい値ごとに返す（2026-10-16、ユーザーの決定）
- **持たないもの:** 書き込みごとのコールバック（消費者の遅さが書き手に伝わる）、全値の記録（future work）、外部プロセスへの転送（ユーザーがホストのスレッドか出力ノードで書く）、サンプル ID の伝搬（全ノードに負担）。入力と出力の対応は提示時間の終わりに読んで取る

**Why:** 外の消費者や生産者の速さがノードの歩調に影響しないこと（非同期性）を、入出力の境界でも守るため。

**How to apply:** 実装中に問題が出たら見直してよい暫定案として扱い、変える場合はユーザーに確認する。run のシグネチャは 2026-10-17 にユーザーが決めた（[[project-run-node-and-write-time]]）。

関連: [[project-network-ports]], [[project-shutdown-on-exception]], [[project-message-types]], [[project-runtime-api-revision]], [[project-run-node-and-write-time]]
