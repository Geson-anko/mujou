# 未決定事項

2026-10-14 時点で、次の設計判断はまだ決まっていない。これらに関わる実装は、決定を待ってから進める。

## ネットワークスキーマの API

ノードの組み合わせと双方向の接続を定義するネットワークオブジェクトを提供し、define-and-run で記述してスキーマオブジェクトを作ることは決定済み。Port・Channel による双方向接続の仕組みも決定済み。Node と Network の Composite、`a.connect(b)`、PrevPort・NextPort も決定済み。実行と外からの読み書き（バックグラウンドでの `run`、`write`・`read`・`subscribe`、版番号による一斉切り替え、例外の伝え方）は暫定案を置いた（[ネットワークの入力と出力](network.md#ネットワークの入力と出力)）。API の具体的な形（クラス名・メソッド名、shape の決め方など）は未決定。すべての値を記録する読み出しは future work。関連: [ネットワーク](network.md)

## 実行の配置

複数のノードを同じワーカーで決まった順に回すこと（[ノード](nodes.md#非同期の境界と実行の配置)）を、利用者がどう指定するか。ノードごとの実行クラス（タスク・専用スレッド）と目標周期の指定方法も未決定。関連: [ネイティブコア](native-core.md#mn-スケジューリング)

## ライフサイクルの細部

hook の構成（`on_start`・`on_update`・`on_pause`・`on_resume`・`on_exception`・`on_stop`・`state_dict`・`load_state_dict`）と、停止・状態の取得・復元の流れは決定済み。hook が例外を投げたらシステム全体を終了し、全ノードの引数なしの `on_exception()` を呼び、設定しておけばそのときの状態の木を取得することも決定済み。状態の保存はユーザーが行う。停止のタイムアウトの既定値、スロットを識別する名前の付け方、例外のときの木を受け取る API の名前は未決定。関連: [ノード](nodes.md#ライフサイクル-hook)、[実行モデル](execution-model.md#停止と状態の取得復元)

## 時定数

連続時間形の $\tau_h, \tau_\lambda, \tau_\theta$ の比と、$\Delta t$ の上限の扱い。関連: [PC-ALM の連続時間系](pc-alm-continuous-time.md#時定数の比と未解決の問い)

## 双対変数の扱い

ストリームのステップ間で $\lambda$ をリセットするか持ち越すか、dual leak を入れるか。関連: [PC-ALM の連続時間系](pc-alm-continuous-time.md#時定数の比と未解決の問い)

## mujou.Node の細部

GPU をサポートしないこと、private な `mujou._core` だけを torch 非依存にして残りは PyTorch を全面的に採用すること、利用者が継承する `mujou.Node` がコアの `Node` を継承し、勾配の記録の停止・ノードごとのストリーム・乱数生成器を `on_start` で設定することは決定済み。`mujou.*` のトップレベルに公開する名前の一覧、サブクラスが `on_start` を override したときに設定が漏れない仕組み、torch の内部スレッド数を誰がどう決めるかは未決定。関連: [PyTorch の利用](torch.md)

## ライセンス

LICENSE ファイルが未作成。
