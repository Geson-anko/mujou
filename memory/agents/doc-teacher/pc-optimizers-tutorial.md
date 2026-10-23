# docs/en/tutorial/5-learn-with-pc.md 理解度テスト

## 対象
- ファイル: `docs/en/tutorial/5-learn-with-pc.md`（`docs/ja/` 版は同じ内容に同期する）
- 読者と用途: チュートリアルを順に進めている mujou の初学者（PyTorch は知っている）。`pc.Chain` に `pc.TimeConstants(h=, lam=)` と `optimizer=pc.SGD(tau=...)` を渡して学習を走らせ、各値が何を決めるかがわかる

## 事実確認（R1）
- `TimeConstants(*, h, lam)`、`period = h / 4`（`src/mujou/pc/time_constants.py`）
- `pc.SGD(tau=)`: 1 歩で `p -= (elapsed / tau) * g`、g はバッチの平均の勾配（ノードが 1/B を掛ける、`layer.py` 311 行）。`elapsed` は前に学んでからの時間
- `optimizer` は `Layer`/`Output`/`Chain` の必須引数、既定なし。各ノードが `optimizer.bind(params)` するので 1 つの設定を共有してよい（`specs/pc.md`「重みの optimizer」）
- ノードの状態に `optimizer` 属性（束縛）が入る
- `examples/pc_regression.py` を実行し、出力が 0.0458→0.0476、0.0060→0.0059 に変わったのを確認して出力例を更新
- topics/pc.md は別の teacher が改稿中で optimizer 節のアンカーが未定。リンクはページ単位（`../topics/pc.md`）にした

## 評価用問題
### Q1 1 バッチを 0.05 秒見せ、各ノードがバッチごとに 1 回学ぶ。SGD で 1 バッチあたりの学習率を 0.1 にしたい。`optimizer=` に何を渡すか。理由も。
- 要点: `pc.SGD(tau=0.5)`
- 要点: 1 バッチあたりの学習率 = 見せる時間（前に学んでからの時間）/ `tau` なので 0.05 / 0.1 = 0.5

### Q2 `pc.Chain` を作るとき、`optimizer=` を省いて既定の optimizer に任せてよいか。
- 要点: よくない。`optimizer` には既定が無く、渡す必要がある

### Q3 3 つのモジュールから `pc.Chain` を作る。ノードごとに別の `pc.SGD` を作る必要があるか。
- 要点: 不要。1 つの `pc.SGD` を `Chain` に渡せばよい
- 要点: 各ノードが optimizer を自分の重みに結び付ける（bind する）から

### Q4 重みの時定数（学習の速さ）は `pc.TimeConstants` のどの引数で決めるか。
- 要点: `TimeConstants` では決めない。`TimeConstants` は `h`（活動）と `lam`（双対）だけ
- 要点: 重みは `optimizer` の `tau`（`pc.SGD(tau=...)`）で決まる

### Q5 例の設定（SGD、バッチごとに 1 回学ぶ）から、`learn_interval` を変えて 1 バッチに 2 回学ぶようにした。`optimizer` は同じ。1 バッチの間に重みが学ぶ量（学習率の合計）はおおよそどうなるか。理由も。
- 要点: ほぼ変わらない（1 回あたりが約半分になり、2 回で同じ）
- 要点: 1 回の学習は、前に学んでからの時間 / `tau` に比例するから

### Q6 学習中の同じ鎖で、重みを動かさずに評価用のバッチの予測を得たい。どうするか。
- 要点: そのバッチの `mask` をすべて False にして `pc.show` で見せる
- 要点: 推論モードへの切り替えは無い（活動は動くが重みは動かない）

### Q7 `tau = pc.TimeConstants(h=8e-3, lam=16e-3)` の鎖で、各ノードは何秒ごとに更新するか。
- 要点: 2 ms（`tau.period` = `tau.h / 4`）

## 保留問題
### H1 SGD ではなく Adam 系で重みを学ばせたい。何を渡すか。値の決め方はどこを見ればよいか。
- 要点: `optimizer=pc.AdamW(...)`
- 要点: topics の PC-ALM ページ（`../topics/pc.md`）

### H2 `SHOWN = 0.032`、バッチごとに 1 回学ぶ鎖に `optimizer=pc.SGD(tau=0.16)` を渡した。1 バッチあたりの学習率はいくつか。
- 要点: 0.2（0.032 / 0.16）

### H3 実行の終わりに、`layer1` の optimizer の状態を見たい。どこにあるか。
- 要点: `run.state_dict()["nodes"]["layer1"]["state"]` の中（ノードの状態に h、lam、モジュール、クロックと並んで入る）

## ラウンド記録
### R0（執筆、49 → 50 行）
- `tau.theta` 前提の記述を `optimizer` に置き換え。学習率の説明を `learn_interval` の項から `optimizer` の項へ移した

### R1（50 → 50 行）正解 7/7
| 問 | 判定 | 原因 | 直したこと |
| -- | ---- | ---- | ---------- |
| Q1〜Q7 | 正解 | - | - |

生徒の「読みにくかった箇所」への対応:
- `tau` が `TimeConstants` と `pc.SGD(tau=)` の 2 通りに読める（曖昧）: `tau` の項に「重みの時定数は optimizer にある」、`optimizer` の項に「`tau` は重みの時定数」と書いた
- `SHOWN` が本文で定義されていない: 式を「見せる時間 `SHOWN` / `optimizer.tau`」にした（snippet では定義済みだが、本文だけで読めるように）
- `mask` の渡し方、`learn_interval` で 2 回学ぶときの間隔: snippet と topics/pc.md が扱うので直さない（生徒は snippet を展開して読めない）

### R2 前（50 → 50 行）
- topics/pc.md の見出しが「Choosing the optimizer」/「optimizer を選ぶ」に決まったので、`pc.AdamW` の文にアンカー付きリンクを足した（ja の slug は pymdownx.slugs で `optimizer-を選ぶ`、既存の ja リンクに合わせて percent-encode）。当初からの依頼（optimizer の選び方は topics へリンク）に基づく変更で、保留問題への対策ではない
- 保留問題 H1〜H3 を出題

### R2（50 → 50 行）保留問題 3/3 正解
| 問 | 判定 | 原因 | 直したこと |
| -- | ---- | ---- | ---------- |
| H1 | 正解 | - | - |
| H2 | 正解（ただし「divided by `optimizer.tau`, 0.3」の 0.3 が学習率か tau か読み分けにくいと指摘） | 曖昧 | 「`SHOWN / optimizer.tau` = 0.3, where `SHOWN` is the time each batch is shown」に変えた。H2 を見て直したので、H2 は評価用 Q8 に移した |
| H3 | 正解（キー名 `"optimizer"` は要点に含めていない） | - | - |

- 「`pc.AdamW` の引数」「counted from each write」の指摘は、topics/pc.md へのリンクとチュートリアル前ページの範囲なので直さない
- 保留問題は H1、H3 の 2 問が残る（どちらも既に生徒に見せたので、次に汎化確認するなら新しい保留問題が要る）

### R3（50 行、変更なし）Q8 正解
- (a) 0.2 正解。(b) 例の 0.3 は本文どおりに答えたが、`SHOWN` と `tau` の値が snippet の中にしかないため確信は低かった。orchestrator の判断で正解とした。生徒には snippet が展開されないだけで、利用者には見える
- 最終状態: 評価用 Q1〜Q8 は全問正解、保留問題 H1 と H3 も正解で収束。en と ja は同期済み。保留問題は使い切った（再利用するなら新しく作る）

### Q8（旧 H2）`SHOWN = 0.032` で、ノードがバッチごとに 1 回学ぶ鎖に `optimizer=pc.SGD(tau=0.16)` を渡した。1 バッチあたりの学習率はいくつか。
- 要点: 0.2（0.032 / 0.16）
