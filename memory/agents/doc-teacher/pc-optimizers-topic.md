# docs/en/topics/pc.md（optimizer）理解度テスト

## 対象
- ファイル: `docs/en/topics/pc.md`（テスト対象）、`docs/ja/topics/pc.md`（en に同期。日本語の推敲は別途）
- 読者と用途: PyTorch を知っていて `mujou.pc` でネットワークを組んで学習させる利用者。読んで、重みの optimizer（`pc.SGD` / `pc.AdamW`）を選んで設定でき、自作のノードで optimizer を bind して使えるようになる

## 評価用問題
### Q1 各バッチを 40 ms 見せ、ノードは 1 バッチに 1 回学ぶ。`pc.SGD` で 1 バッチあたりの学習率を 0.2 にしたい。`optimizer=` に何を渡すか
- 要点: `pc.SGD(tau=0.2)`（0.2 s = 40 ms / 0.2）。見せる時間 / lr

### Q2 `pc.SGD(tau=...)` は変えずに、`learn_interval` を見せる時間の 0.75 倍から 0.4 倍に変えた。ノードは 1 バッチに 2 回学ぶようになる。1 バッチの間に重みが動く量は変わるか。理由も
- 要点: 変わらない
- 要点: 各歩みの `elapsed` は前に学んでからの時間なので、1 バッチの間の `elapsed` の合計は何回学んでも見せる時間になる
- 旧 Q2（「3/4 から 0.6 倍」）は「3/4 の 0.6 倍」とも読めたので R1 で差し替えた（問題不良）

### Q3 PyTorch で `torch.optim.AdamW(lr=1e-3, betas=(0.9, 0.999))` を使っていた。mujou では各バッチを 64 ms 見せ、1 バッチに 1 回学ぶ。`pc.AdamW` の `tau` と `tau_m` をいくつにするか（式と値）
- 要点: Δt = 64 ms（2 回の学習の間の時間 = 見せる時間）
- 要点: `tau = Δt / lr` = 64 s
- 要点: `tau_m = -Δt / ln(beta1)` ≈ 0.61 s（0.6 前後なら可）

### Q4 あるノードを学習させたくない。`pc.SGD(tau=math.inf)` を渡してよいか。だめならどうするか
- 要点: だめ。`tau` は有限で > 0 でないと `ValueError`
- 要点: `learn_interval=math.inf` を渡す

### Q5 自作のノードを 2 つ書く。1 つの `pc.AdamW(...)` を `bind` した結果を 2 つのノードの `self.optimizer` に共有してよいか。設定の `pc.AdamW(...)` 自体はどうか
- 要点: bind した結果の共有はだめ。各ノードで bind する
- 要点: 設定（`pc.AdamW(...)`）は状態を持たないので複数のノードで共有してよい（`pc.Chain` も全ノードに同じものを渡す）

### Q6 自作のノードの `on_update` で、学ぶ更新のとき optimizer に何をどう渡すか。学ばない条件も
- 要点: `self.optimizer.step_([g / B for g in g_theta], t.elapsed)`
- 要点: `B`（バッチのサンプルの数）で割るのはノード（optimizer はバッチを知らない）
- 要点: `t.learn` が True で、目標を持つサンプル（mask True）があるときだけ

### Q7 埋め込み層のパラメータを `requires_grad=False` で凍結し、`pc.AdamW(..., weight_decay=1e-2)` の `pc.Chain` に入れた。凍結したパラメータは weight decay で縮むか
- 要点: 縮まない。ノードは `requires_grad=True` のパラメータだけを bind し、凍結したパラメータには weight decay も掛からない

### Q8 `pc.AdamW(tau=10.0, tau_m=0.3, tau_v=30.0, weight_decay=0.01)` で、weight decay だけを見た重みの減衰の時定数は何秒か
- 要点: `tau / weight_decay` = 1000 s

## 保留問題
### H1 入力がフレームごとに変わるストリームで `learn_interval=None`、`pc.SGD(tau=2.0)` にした。学習率はどう表され、値はいくつか
- 要点: 1 秒あたりの学習率が `1 / tau` = 0.5（毎回の更新で学ぶ）

### H2 実行中に "layer1" という `pc.Layer` の AdamW の moment を見たい。どこから読むか。ノードの属性を直接読んでよいか
- 要点: `run.state_dict()["nodes"]["layer1"]["state"]["optimizer"]`（`m`、`v`、`c_m`、`c_v`）
- 要点: 属性を走っている間に読むと更新と競合するので、state_dict から読む

### H3 自作のノードの `__init__` で optimizer を bind する。何を、いつ bind し、`vjp` には何を渡すか
- 要点: 学ぶパラメータ（`requires_grad`）を bind する
- 要点: module を device に載せた後
- 要点: `vjp` には bind したのと同じ `params` を同じ順で渡す

## ラウンド記録
### R0（140 → 168 行）
- 初稿。`tau.theta` / `functional.learn_` を新 API に置換、「Choosing the optimizer」節を追加、自作ノードの手順 1・3・7 を更新
- 行数 +20%: 計画書が要求する新しい節（SGD/AdamW の表、時間で進める理由、torch からの換算、weight_decay、不正値、状態）の追加（欠落）による

### R1（168 → 169 行）
| 問 | 判定 | 原因 | 直したこと |
| -- | ---- | ---- | ---------- |
| Q1 | 正解 | - | - |
| Q2 | 不正解 | 問題不良（倍率が二通りに読めた） | 問題を差し替えた。生徒の指摘から、文書の「Time, not steps」が 1 バッチ 1 回の前提でしか根拠を示していなかった（曖昧）ので、何回学んでも 1 バッチの `elapsed` の合計が見せる時間になる、と書き直した |
| Q3 | 正解 | - | 生徒の指摘（Δt の定義が埋もれる）で、torch からの換算を独立した箇条にし、Δt の定義を先頭に置いた（埋没） |
| Q4〜Q8 | 正解 | - | 手順 7 の「target」が未定義との指摘で、「`mask` が True のサンプル」に言い換えた（曖昧） |
- 手順 4 の式が詰まっているとの指摘は、optimizer と関係しない既存の記述なので直さない

### R2（169 → 169 行）
| 問 | 判定 | 原因 | 直したこと |
| -- | ---- | ---- | ---------- |
| Q1〜Q8 | 正解 | - | - |
- 生徒の指摘から、optimizer に関わる 2 件を直した: 学ばないノードの `learn_interval=math.inf` は `pc.Layer` / `pc.Output` に渡すこと（`pc.Chain` は全ノードに 1 つ）、AdamW の例のコメントを torch の値を換算して丸めたものと明記
- 範囲外として残した指摘: credit の往復が必須か推奨かの表現、1 バッチの最初の学習の elapsed
- 次: 保留問題 H1〜H3 で汎化確認

### 汎化確認（169 → 169 行）
| 問 | 判定 | 原因 | 直したこと |
| -- | ---- | ---- | ---------- |
| H1〜H3 | 正解 | - | - |
- 生徒の指摘から 1 件だけ直した: 状態の箇条に `state["optimizer"]` の中身（AdamW は `"m"`・`"v"`・`"c_m"`・`"c_v"` の dict、SGD は空）を明記（曖昧）
- 残した指摘: per batch と per second の切り替え（ストリームの箇条で明示済み）、手順 1 の「state に入る」、vjp の他の引数と relax_ の tau（API リファレンスと既存の記述の範囲）
