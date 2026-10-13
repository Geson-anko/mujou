# 未決定事項

2026-10-13 時点で、次の設計判断はまだ決まっていない。これらに関わる実装は、決定を待ってから進める。

## ネットワークスキーマの API

ブロックの組み合わせと双方向の接続を定義するネットワークオブジェクトを提供し、define-and-run で記述してスキーマオブジェクトを作ることは決定済み。Port・Channel による双方向接続の仕組みも決定済み。Block と Network の Composite、`a.connect(b)`、PrevPort・NextPort も決定済み。API の具体的な形（クラス名・メソッド名、shape の決め方、`build` をどこから呼ぶか、実行と外からの読み書きの API など）は未決定。関連: [ネットワーク](network.md)

## 時定数

連続時間形の $\tau_h, \tau_\lambda, \tau_\theta$ の比と、$\Delta t$ の上限の扱い。関連: [PC-ALM の連続時間系](pc-alm-continuous-time.md#時定数の比と未解決の問い)

## 双対変数の扱い

ストリームのステップ間で $\lambda$ をリセットするか持ち越すか、dual leak を入れるか。関連: [PC-ALM の連続時間系](pc-alm-continuous-time.md#時定数の比と未解決の問い)

## ライセンス

LICENSE ファイルが未作成。
