# 未決定事項

2026-10-14 時点で、次の設計判断はまだ決まっていない。これらに関わる実装は、決定を待ってから進める。

## ネットワークスキーマの API

ノードの組み合わせと双方向の接続を定義するネットワークオブジェクトを提供し、define-and-run で記述してスキーマオブジェクトを作ることは決定済み。Port・Channel による双方向接続の仕組み、Node と Network の Composite、`a.connect(b)`、PrevPort・NextPort も決定済み。

クラス名・メソッド名と、実行と外からの読み書き（`mujou.run`、`write`・`read`・`subscribe`、版番号による一斉切り替え、例外の伝え方）は、推奨案で実装した（2026-10-16、ユーザー確認待ち。[実装](network.md#実装)）。そのうち次の 3 点は仕様書（specs/）の当初の案から変えた。`run()` は全ノードの起動が終わってから戻る（当初は「すぐに戻る」）。`state_dict()` は実行中と正常に終了した後に呼べる（当初は「いつでも」。2026-10-16 にユーザーの決定で終了後も呼べるようにした）。相手 1 つの Port への 2 本目の接続は `connect` の時点でエラーにする（当初は「構築時」）。また、`send_to` だけを使う Port を外から読むと初期値が返る。

shape の決め方は未決定。すべての値を記録する読み出しは future work。関連: [ネットワーク](network.md)

## 実行の配置

複数のノードを同じワーカーで決まった順に回すこと（[ノード](nodes.md#非同期の境界と実行の配置)）を、利用者がどう指定するか。ノードごとの実行クラス（タスク・専用スレッド）と目標周期の指定方法も未決定。関連: [ネイティブコア](native-core.md#mn-スケジューリング)

## ライフサイクルの細部

hook の構成（`on_start`・`on_update`・`on_pause`・`on_resume`・`on_exception`・`on_stop`・`state_dict`・`load_state_dict`）と、停止・状態の取得・復元の流れは決定済み。hook が例外を投げたらシステム全体を終了し、全ノードの引数なしの `on_exception()` を呼ぶことも決定済み。例外のときの状態は取得しない（2026-10-16、ユーザーの決定。当初は設定で取得できた）。状態の保存はユーザーが行う。

停止のタイムアウトの既定値は実時間 30 秒、`shutdown()` / `join()` の `timeout` は既定で `math.inf` とした（2026-10-16、ユーザーの決定）。スロットを識別する名前（`"body.l1.next/out"` など）と、例外で終了した後にタイマーを再開することは、推奨案で実装した（2026-10-16、ユーザー確認待ち。[実装](execution-model.md#実装-1)）。関連: [ノード](nodes.md#ライフサイクル-hook)、[実行モデル](execution-model.md#停止と状態の取得復元)

## 時定数

連続時間形の $\tau_h, \tau_\lambda, \tau_\theta$ の比と、$\Delta t$ の上限の扱い。関連: [PC-ALM の連続時間系](pc-alm-continuous-time.md#時定数の比と未解決の問い)

## 双対変数の扱い

ストリームのステップ間で $\lambda$ をリセットするか持ち越すか、dual leak を入れるか。関連: [PC-ALM の連続時間系](pc-alm-continuous-time.md#時定数の比と未解決の問い)

## mujou.Node の細部

GPU をサポートしないこと、private な拡張モジュール `mujou._core._native` だけを torch 非依存にして残りは PyTorch を全面的に採用すること、利用者が継承する `mujou.Node` がコアの `Node` を継承し、勾配の記録の停止・ノードごとのストリーム・乱数生成器を設定することは決定済み。

次は推奨案で実装した（2026-10-16、ユーザー確認待ち）。設定は `on_start` の前に同じスレッドで呼ぶ `_on_thread_start()` で行い、`on_start` の override で漏れないようにした。トップレベルの公開名は `Node, Network, Port, PrevPort, NextPort, Channel, TensorSpec, DictSpec, NetworkError, PauseTimeoutError, run`（と `mujou.time`）。torch の内部スレッド数は mujou が設定せず、ホストが実行前に決める（目安は [PyTorch の利用](torch.md#プロセス全体に効くもの)）。PC-ALM のクラスは `mujou.pc` からだけ公開すると決定済み（[更新手法](update-rules.md#実装と公開元)）。関連: [PyTorch の利用](torch.md)

## グローバルタイマーとスケジューラの境界

ノードの実行と制御は実装した（[実行モデル](execution-model.md#実装-1)）。pamiq-core の `thread_control.py` に倣い、制御は 1 本のスレッドに直列化した。$\Delta t = 0$ で待つノードは、目標の時刻か制御の要求で起きる。

残りは次の 2 点である。時刻の読み取りが Mutex を取ることによる競合（性能が問題になったら seqlock などを検討する）。実経過を f64 で扱うことによる、長時間の実行での分解能の低下。

同じタイマーで動くネットワーク同士が停止で時間を止め合う点は、2026-10-16 にユーザーが決めた。時間軸は実行ごとに 1 本で、既定はグローバルタイマーである。止め合いたくないネットワークは別のタイマーで動かす（[複数のネットワーク](execution-model.md#複数のネットワーク)）。コアと `mujou._core._native` は実装済みで、`mujou.run(..., timer=...)` はホスト側の作業で入れる。

## ライセンス

LICENSE ファイルが未作成。
