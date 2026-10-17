# 未決定事項

2026-10-17 時点で、次の設計判断はまだ決まっていない。これらに関わる実装は、決定を待ってから進める。

## ネットワークスキーマの API

ノードの組み合わせと双方向の接続を定義するネットワークオブジェクトを提供し、define-and-run で記述してスキーマオブジェクトを作ることは決定済み。Port・Channel による双方向接続の仕組み、Node と Network の Composite、`a.connect(b)`、PrevPort・NextPort も決定済み。

クラス名・メソッド名と、実行と外からの読み書き（`mujou.run`、`write`・`read`・`subscribe`、版番号による一斉切り替え、例外の伝え方）は、推奨案で実装した（2026-10-16、ユーザー確認待ち。[実装](network.md#実装)）。そのうち次の 3 点は仕様書（specs/）の当初の案から変えた。`run()` は全ノードの起動が終わってから戻る（当初は「すぐに戻る」）。`state_dict()` は実行中と正常に終了した後に呼べる（当初は「いつでも」。2026-10-16 にユーザーの決定で終了後も呼べるようにした）。相手 1 つの Port への 2 本目の接続は `connect` の時点でエラーにする（当初は「構築時」）。また、`send_to` だけを使う Port を外から読むと初期値が返る。

shape の決め方は未決定。すべての値を記録する読み出しは future work。関連: [ネットワーク](network.md)

## 実行の配置

複数のノードを同じワーカーで決まった順に回すこと（[ノード](nodes.md#非同期の境界と実行の配置)）を、利用者がどう指定するか。ノードごとの実行クラス（タスク・専用スレッド）の指定方法も未決定。関連: [ネイティブコア](native-core.md#mn-スケジューリング)

目標周期は、ノードの `period`（`Node.__init__` のキーワード、その実行のタイマーの秒、`None` はできるかぎり速く）として推奨案で実装した（2026-10-17、ユーザー確認待ち。[実行の周期](nodes.md#実行の周期)）。待つのはランタイムで、目標は前回の目標 + `period`、`period` 以上遅れたときだけ数え直す（2026-10-17、ユーザーの決定。pamiq-core の「起きた時刻から数え直す」から変えた）。クラス属性で既定の周期を置く書き方は認めず、`period` を予約名にした（同日、ユーザーの決定）。残り: M:N のワーカーで期限の早い順に回すときの周期の扱い

## ライフサイクルの細部

hook の構成（`on_thread_start`・`on_start`・`on_update`・`on_pause`・`on_resume`・`on_exception`・`on_stop`・`state_dict`・`load_state_dict`）と、停止・状態の取得・復元の流れは決定済み。hook が例外を投げたらシステム全体を終了し、全ノードの引数なしの `on_exception()` を呼ぶことも決定済み。例外のときの状態は取得しない（2026-10-16、ユーザーの決定。当初は設定で取得できた）。状態の保存はユーザーが行う。

停止のタイムアウトの既定値は実時間 30 秒、`shutdown()` / `join()` の `timeout` は既定で `math.inf` とした（2026-10-16、ユーザーの決定）。スロットを識別する名前（`"body.l1.next/out"` など）と、例外で終了した後にタイマーを再開することは、推奨案で実装した（2026-10-16、ユーザー確認待ち。[実装](execution-model.md#実装-1)）。関連: [ノード](nodes.md#ライフサイクル-hook)、[実行モデル](execution-model.md#停止と状態の取得復元)

## 時定数

連続時間形の $\tau_h, \tau_\lambda, \tau_\theta$ の比と、$\Delta t$ の上限の扱い。周期を入れても $\Delta t$ は実際の経過時間のままで、上限（`max_dt`）はまだ持たない（2026-10-17）。関連: [PC-ALM の連続時間系](pc-alm-continuous-time.md#時定数の比と未解決の問い)

## 双対変数の扱い

ストリームのステップ間で $\lambda$ をリセットするか持ち越すか、dual leak を入れるか。関連: [PC-ALM の連続時間系](pc-alm-continuous-time.md#時定数の比と未解決の問い)

## mujou.Node の細部

GPU をサポートしないこと、private な拡張モジュール `mujou._core._native` だけを torch 非依存にして残りは PyTorch を全面的に採用すること、利用者が継承する `mujou.Node` がコアの `Node` を継承し、勾配の記録の停止・ノードごとのストリーム・乱数生成器を設定することは決定済み。

次は推奨案で実装した（2026-10-16、ユーザー確認待ち）。設定は `on_start` の前に同じスレッドで呼ぶ内部の hook `_on_thread_start()` で行い（2026-10-17 に、その後に呼ぶ公開の `on_thread_start()` と分けた）、`on_start` の override で漏れないようにした。トップレベルの公開名は `Node, Network, Compose, Graph, Run, Port, PrevPort, NextPort, Channel, TensorSpec, DictSpec, NetworkError, PauseTimeoutError, run`（と `mujou.time`。テスト用の口は `mujou.testing`）。torch の内部スレッド数は mujou が設定せず、ホストが実行前に決める（目安は [PyTorch の利用](torch.md#プロセス全体に効くもの)）。PC-ALM のクラスは `mujou.pc` からだけ公開すると決定済み（[更新手法](update-rules.md#実装と公開元)）。関連: [PyTorch の利用](torch.md)

## グローバルタイマーとスケジューラの境界

ノードの実行と制御は実装した（[実行モデル](execution-model.md#実装-1)）。pamiq-core の `thread_control.py` に倣い、制御は 1 本のスレッドに直列化した。$\Delta t = 0$ で待つノードは、目標の時刻か制御の要求で起きる。

残りは次の 3 点である。時刻の読み取りが Mutex を取ることによる競合（性能が問題になったら seqlock などを検討する）。スロットの読み書きがスロットごとの mutex を取ることによる競合（[スロットの実装](native-core.md#スロットの実装)。lock を持つスレッドが横取りされるとその間は待つ。問題になったら RwLock や parking_lot などを検討する）。実経過を f64 で扱うことによる、長時間の実行での分解能の低下。

同じタイマーで動くネットワーク同士が停止で時間を止め合う点は、2026-10-16 にユーザーが決めた。時間軸は実行ごとに 1 本で、既定はグローバルタイマーである。止め合いたくないネットワークは別のタイマーで動かす（[複数のネットワーク](execution-model.md#複数のネットワーク)）。コアと `mujou._core._native` は実装済みで、`mujou.run(..., timer=...)` で実行ごとに渡す（実装済み）。

## 2026-10-17 の推奨案（ユーザー確認待ち）

PC-ALM を実装する前に、ライブラリを使いやすく整えた（PR #30〜#44）。次は推奨案で実装し、ユーザーの確認を待っている。

- ホストの待ち方: `Run.wait_for(port, condition, *, timeout=)` と `Run.subscribe(port, *, timeout=)`（[ネットワーク](network.md)）
- テスト: `mujou.testing.run` の既定のタイマーを新しい `VirtualTimer` にし、`ManualRun.timer` で返す。`mujou.time.now()`
- Port: エラーと repr に `Owner.attr` の名前を出す。`Channel` は `connect` だけが作る。型付きの overload は `None` の spec を受け付けない
- デバイス: CUDA 以外のアクセラレータ（MPS など）はストリーム無しで使い、デバイス全体を同期する。MPS は 2 スレッドから同時に使うとプロセスが落ちるので、走っている実行の全体で MPS のノードは 1 つまでとし、開始前に `ValueError` で拒む（[PyTorch の利用](torch.md)）

次はまだ決めていない。

- MPS のノードを複数許すために、MPS のノードの hook を 1 本のロックで直列にするか（ノード同士が計算を待ち合うことになる）
- `Run.state_dict()` の木のテンソルを常に CPU へ移すか（今はノードのデバイスのまま。MPS のノードが走っている間にホストで触ると落ちうる）
- `Timer.pause` / `resume` / `is_paused`（手動の一時停止）は src で使っておらず、テストの基盤だけが使う。残すか消すか
- `pause_timeout` を `mujou.run` の引数から `Run.state_dict(timeout=)` へ移すか
- `Compose` の名前（torchvision の `Compose` は直列につなぐ意味で使われる）
- 相手がまだ送っていないことを判定する API（今は `on_start` で `recv_with_time()` の時刻を読む書き方を案内している）
- 手動の実行（`ManualRun`）で `period` を守る選択肢
- `Run.timer` を `ManualRun` だけでなく `Run` 全体に持たせるか
- GPU（CUDA）のテストを CI で走らせるか（今は GPU のある手元でだけ走る）

LICENSE は MIT に決めた（2026-10-17、ユーザーの決定）。
