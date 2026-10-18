# 未決定事項

2026-10-17 時点で、次の設計判断はまだ決まっていない。これらに関わる実装は、決定を待ってから進める。

## ネットワークスキーマの API

ノードの組み合わせと双方向の接続を定義するネットワークオブジェクトを提供し、define-and-run で記述してスキーマオブジェクトを作ることは決定済み。Port・Channel による双方向接続の仕組み、Node と Network の Composite、PrevPort・NextPort も決定済み。接続は Port 同士だけ（`a.next.connect(b.prev)`）で、Port と子は代入時に登録する（2026-10-17、[属性の登録](network.md#属性の登録)）。

クラス名・メソッド名と、実行と外からの読み書き（`mujou.run`、`write`・`read`・`subscribe`、版番号による一斉切り替え、例外の伝え方）は、推奨案で実装した（2026-10-16、ユーザー確認待ち。[実装](network.md#実装)）。そのうち次の 3 点は仕様書（specs/）の当初の案から変えた。`run()` は全ノードの起動が終わってから戻る（当初は「すぐに戻る」）。`state_dict()` は実行中と正常に終了した後に呼べる（当初は「いつでも」。2026-10-16 にユーザーの決定で終了後も呼べるようにした）。相手 1 つの Port への 2 本目の接続は `connect` の時点でエラーにする（当初は「構築時」）。また、`send_to` だけを使う Port を外から読むと初期値が返る。

shape の決め方は未決定。すべての値を記録する読み出しは future work。関連: [ネットワーク](network.md)

## 実行の配置

複数のノードを同じワーカーで決まった順に回すこと（[ノード](nodes.md#非同期の境界と実行の配置)）を、利用者がどう指定するか。既定のワーカー数（`min(ワーカーに載るノード数, 論理 CPU 数)`）が妥当かは、ベンチマークで確かめる。関連: [ネイティブコア](native-core.md#mn-スケジューリング)

目標周期は、ノードの `period`（`Node.__init__` のキーワード、その実行のタイマーの秒、既定は 0.01、`0.0` はできるかぎり速く。既定は 2026-10-17 のユーザーの決定）として推奨案で実装した（2026-10-17、ユーザー確認待ち。[実行の周期](nodes.md#実行の周期)）。待つのはランタイムで、目標は前回の目標 + `period`、`period` 以上遅れたときだけ数え直す（2026-10-17、ユーザーの決定。pamiq-core の「起きた時刻から数え直す」から変えた）。クラス属性で既定の周期を置く書き方は認めず、`period` を予約名にした（同日、ユーザーの決定）。ワーカーは同じ規則で求めた目標を期限とし、期限の早い順に回す（[ワーカープールの実装](native-core.md#ワーカープールの実装)）

## ライフサイクルの細部

hook の構成（`on_thread_start`（専用スレッドのノードだけ）・`on_start`・`on_update`・`on_pause`・`on_resume`・`on_exception`・`on_stop`・`state_dict`・`load_state_dict`）と、停止・状態の取得・復元の流れは決定済み。hook が例外を投げたらシステム全体を終了し、全ノードの引数なしの `on_exception()` を呼ぶことも決定済み。例外のときの状態は取得しない（2026-10-16、ユーザーの決定。当初は設定で取得できた）。状態の保存はユーザーが行う。

停止のタイムアウトの既定値は実時間 30 秒、`shutdown()` / `join()` の `timeout` は既定で `math.inf` とした（2026-10-16、ユーザーの決定）。スロットを識別する名前（`"body.l1.next/out"` など）と、例外で終了した後にタイマーを再開することは、推奨案で実装した（2026-10-16、ユーザー確認待ち。[実装](execution-model.md#実装-1)）。関連: [ノード](nodes.md#ライフサイクル-hook)、[実行モデル](execution-model.md#停止と状態の取得復元)

## 時定数

連続時間形の $\tau_h, \tau_\lambda, \tau_\theta$ の比と、$\Delta t$ の上限の扱い。周期を入れても $\Delta t$ は実際の経過時間のままで、上限（`max_dt`）はまだ持たない（2026-10-17）。関連: [PC-ALM の連続時間系](pc-alm-continuous-time.md#時定数の比と未解決の問い)

## 双対変数の扱い

ストリームのステップ間で $\lambda$ をリセットするか持ち越すか、dual leak を入れるか。関連: [PC-ALM の連続時間系](pc-alm-continuous-time.md#時定数の比と未解決の問い)

## mujou.Node の細部

GPU をサポートしないこと、private な拡張モジュール `mujou._core._native` だけを torch 非依存にして残りは PyTorch を全面的に採用すること、利用者が継承する `mujou.Node` がコアの `Node` を継承し、勾配の記録の停止・ノードごとのストリーム・乱数生成器を設定することは決定済み。

次は推奨案で実装した（2026-10-16、ユーザー確認待ち）。設定は内部の hook `_on_enter()` で行い、`on_start` の override で漏れないようにした。`_on_enter()` はスレッドがそのノードに切り替わるたびに次の hook の直前で呼ぶ（2026-10-18 に、起動時に 1 回だけ呼ぶ `_on_thread_start()` から置き換えた）。トップレベルの公開名は `Node, Network, Group, Graph, Run, Port, PrevPort, NextPort, Channel, TensorSpec, DictSpec, NetworkError, PauseTimeoutError, run`（と `mujou.time` と `mujou.typing`。テスト用の口は `mujou.testing`）。torch の内部スレッド数は mujou が設定せず、ホストが実行前に決める（目安は [PyTorch の利用](torch.md#プロセス全体に効くもの)）。PC-ALM のクラスは `mujou.pc` からだけ公開すると決定済み（[更新手法](update-rules.md#実装と公開元)）。関連: [PyTorch の利用](torch.md)

## グローバルタイマーとスケジューラの境界

ノードの実行と制御は実装した（[実行モデル](execution-model.md#実装-1)）。pamiq-core の `thread_control.py` に倣い、制御は 1 本のスレッドに直列化した。$\Delta t = 0$ で待つノードは、目標の時刻か制御の要求で起きる。

残りは次の 3 点である。時刻の読み取りが Mutex を取ることによる競合（性能が問題になったら seqlock などを検討する）。スロットの読み書きがスロットごとの mutex を取ることによる競合（[スロットの実装](native-core.md#スロットの実装)。lock を持つスレッドが横取りされるとその間は待つ。問題になったら RwLock や parking_lot などを検討する）。実経過を f64 で扱うことによる、長時間の実行での分解能の低下。

同じタイマーで動くネットワーク同士が停止で時間を止め合う点は、2026-10-16 にユーザーが決めた。時間軸は実行ごとに 1 本で、既定はグローバルタイマーである。止め合いたくないネットワークは別のタイマーで動かす（[複数のネットワーク](execution-model.md#複数のネットワーク)）。コアと `mujou._core._native` は実装済みで、`mujou.run(..., timer=...)` で実行ごとに渡す（実装済み）。

## 2026-10-17 の推奨案（ユーザー確認待ち）

PC-ALM を実装する前に、ライブラリを使いやすく整えた（PR #30 から始まるスタック）。次は推奨案で実装し、ユーザーの確認を待っている。

- ホストの待ち方: `Run.wait_for(port, condition, timeout=)` と `Run.subscribe(port, timeout=)`（[ネットワーク](network.md)）
- テスト: `mujou.testing.run` の既定のタイマーを新しい `VirtualTimer` にし、`ManualRun.timer` で返す。`mujou.time.now()`
- Port: エラーと repr に `Owner.attr` の名前を出す。`Channel` は `connect` だけが作る。型付きの overload は `None` の spec を受け付けない

次はまだ決めていない。

- 手動の実行（`ManualRun`）で `period` を守る選択肢（2026-10-17 にユーザーから用途を問われた。今は守らず、テストでは `step` と `advance` で更新の順と `dt` を決める）

LICENSE は MIT に決めた（2026-10-17、ユーザーの決定）。

`Run` の API は次のように整理した（2026-10-17、ユーザーの決定。[ネットワーク](network.md#実装)）。停止のタイムアウトは `mujou.run(..., pause_timeout=)` から `Run.state_dict(timeout=30.0)` に移した。`subscribe` と `wait_for` の `timeout` は位置でも渡せる。`Run.timer` はすべての実行が持ち、既定はグローバルタイマーである。`Run.read(port)` は値だけを返し、時刻との組は `Run.read_with_time(port)` が返す。

ノードのデバイスは CPU と CUDA だけにし、MPS など CUDA 以外のアクセラレータのサポートを取りやめた（2026-10-17、ユーザーの決定）。PyTorch の MPS は 2 スレッドから同時に使うとプロセスが落ち、複数のスレッドが並行してノードを計算する実行モデルと合わないためである。これに伴い、MPS のノードを 1 つに限る規則と、MPS のノードの hook をロックで直列にする案は不要になった（[PyTorch の利用](torch.md#cuda-以外のアクセラレータ)）。

GPU（CUDA）のテストは、自前の GPU runner（ラベル `gpu`）で手動で起動するワークフロー `.github/workflows/gpu.yml` で走らせる（2026-10-17、ユーザーの決定）。runner はまだ無く、登録するスクリプトを `.github/runner/` に置いた。fork からの PR のコードを自前のマシンで走らせないよう、PR では起動しない。
