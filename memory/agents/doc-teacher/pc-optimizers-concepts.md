# docs/en/topics/pc-concepts.md（optimizer 改稿）理解度テスト

## 対象
- ファイル: docs/en/topics/pc-concepts.md（ja 版 docs/ja/topics/pc-concepts.md は同じ内容に同期）
- 読者と用途: `mujou.pc` の仕組み（h、λ、θ が同時に時間で動くこと）を理解したい利用者。重みの optimizer が時間で進むこと（ステップ数ではなく経過時間、AdamW の時定数と質量 c による bias 補正、torch.optim を使わない理由、1/B をノードが掛けること）を説明できるようになる
- 前のテスト（pc.md 以前の版）は `pc-concepts.md`。今回は 6 節「Optimizers」の新設と 1・3・4・5 節の `tau.theta` / `learn_` の置き換え
- 事実の出典: `src/mujou/pc/optim.py`（eps > 0 は simplifier の変更後のコードに合わせた。specs/pc.md の K31 は eps >= 0 のまま）、`layer.py`、`output.py`、`specs/pc.md`「重みの optimizer」

## 評価用問題
### Q1 pc.SGD(tau=1.2)、learn_interval=24e-3 のノードに同じ入力を長く見せ続けている（mask は True、dt は打ち切られない）。learn_interval を 48e-3 に変えると、同じ入力を見せている間に重みが 1 秒あたりに動く速さは変わるか。1 回の学習の学習率はいくつになるか。
- 要点: 1 秒あたりの速さは変わらない（1 歩が elapsed 秒ぶん進むので歩幅が倍になる）／elapsed は前に学んでからの時間で約 0.048 秒、学習率は約 0.048 / 1.2 = 0.04
- R0 の差し戻しで前提（同じ入力が続く間）を問題文に入れた

### Q2 ある Layer に決して学ばせたくない。optimizer=pc.SGD(tau=math.inf) を渡してよいか。どうするか。
- 要点: 渡せない（tau は有限で > 0 が必須で断られる）／ノードに learn_interval=math.inf を渡す

### Q3 pc.AdamW は torch.optim.AdamW の「ステップ数による bias 補正」の代わりに何を使うか。t 秒ぶんの勾配の後のその値と、それで m をどう補正するか。
- 要点: 質量 c（moment が集めた重み）／c = 1 − e^{−t/τ}（τ はその moment の時定数）／m̂ = m / c（v も c_v で同様）

### Q4 同じ一定の勾配で、pc.AdamW のノード A は 0.05 秒の歩みを 2 回、ノード B は 0.1 秒の歩みを 1 回した。m・v・c は A と B で同じか。固定の β でステップを数える optimizer ならどうか。
- 要点: pc.AdamW では同じ（歩みの秒数の合計で決まる。流れを厳密に解く）／ステップを数える方は違う（2 回と 1 回で補正が変わる）

### Q5 自分のノードで学ぶとき、バッチのサンプル数 B で割るのはノードか optimizer か。step_ には何を渡すか。
- 要点: ノードが g_theta を B で割ってから渡す（optimizer はバッチを知らない）／step_([g / B for g in g_theta], tick.elapsed)

### Q6 torch の Adam を lr=1e-3、betas=(0.9, 0.999) で使っていた。mujou では各入力を 32 ms 見せ、ノードは 1 入力につき 1 回学ぶ。pc.AdamW の tau、tau_m、tau_v をおよそいくつにするか。
- 要点: Δt（2 回の学習の間の時間）= 32 ms／tau = Δt / lr = 32 秒／tau_m ≈ 0.30 秒（−Δt / ln β1）／tau_v ≈ 32 秒
- R0 の差し戻し: 換算の Δt は learn_interval ではなく 2 回の学習の間の時間。問題文で Δt が決まるようにした

### Q7 mujou.pc が torch.optim.AdamW を使わない理由を 2 つ挙げよ。
- 要点（次のうち 2 つ、正しく）: ステップを数え時間を数えない（β・lr が 1 ステップあたりで、1 秒あたりの速さが learn_interval や elapsed で変わる）／ステップ数が float32 で 2^24 回で増えなくなる／作るときに torch._dynamo を import し、3.14t でプロセス全体の GIL が戻る

### Q8 手で組んだネットワークで、1 つの pc.AdamW(...) を 3 つのノードに渡した。ノード同士で moment を共有するか。各ノードの moment はどこにあり、状態のどのキーに入るか。
- 要点: 共有しない（設定は状態を持たず、各ノードが自分の重みに bind する）／各ノードの束縛した optimizer（self.optimizer）の中／キー "optimizer"

### Q9 pc.AdamW(tau=10.0, ..., weight_decay=0.01) で、weight decay が θ を減衰させる時定数は何秒か。
- 要点: τθ / w = 10 / 0.01 = 1000 秒（weight_decay は無次元）

### Q10 Layer の 1 回の on_update で、optimizer.step_ は Activity を送る前か後か。mask がすべて False の入力では step_ を呼ぶか。
- 要点: Activity を送った後（最後の段 6）／呼ばない（tick.learn が True で mask に True があるときだけ）

## 保留問題
### H1 自分のノードで module.parameters() に requires_grad=False の重みが混じっている。bind には何を渡すか。
- 要点: 学ぶ重み（勾配を取る重み）だけを渡す

### H2 一定の勾配 g を受け続ける pc.AdamW の最初の 1 歩で、補正した m̂ はいくつか。理由も。
- 要点: m̂ = g／m = c·g なので m / c = g

### H3 elapsed == 0 で step_ が呼ばれたら重みや moment はどうなるか。
- 要点: 何も動かない

## ラウンド記録
### R0 執筆（211 → 281 行）
- 6 節「Optimizers」を新設（要点、bind の図、コード 3 行、SGD/AdamW の表、質量 c と 2 回 vs 1 回の表、torch との換算、torch.optim を使わない理由の question、自作の optimizer）。1・3・4・5 節の tau.theta と learn_ を置き換え
- 差し戻し（coordinator）: 6 節の `tau = learn_interval / lr` は誤り（optim.py の docstring 由来）。「2 回の学習の間の時間 Δt / lr」に直し、β の換算も同じ Δt と明記。Q1・Q6 の前提を問題文に入れた
- 増えた理由: 新しい節（約 70 行）を足した。既存節は行数を増やしていない
- ja 版を同じ内容で同期（日本語の推敲は後で別に行う）
- pre-commit（mdformat）と just docs-build（strict）通過

### R1（281 → 282 行）
| 問 | 判定 | 原因 | 直したこと |
| -- | ---- | ---- | ---------- |
| Q1 | 不正解（学習率を 0.02 と答え、0.04 は括弧の補足。要点の 0.04 が主答でない） | 曖昧 | 4 節 learn_interval の「Positive seconds」行の "each of learn_interval, twice it, … past time" を `time + learn_interval`、`time + 2 * learn_interval`、… 以降の最初の更新に言い換えた。箇条に「同じ入力が続く間は learn_interval 秒ごとに学び、elapsed はおよそ learn_interval」を 1 行足した（Tick.elapsed の値は自作のノードを書く読者にも要る） |
| Q2〜Q10 | 正解 | — | — |
| （指摘: c の表が c_m だけ） | — | 直さない | 表の直前の文が「moments and masses」を述べ、m = c_m g もある。Q4 は正解したので、行数を増やさないことを優先した |
- ja 版も同じ 2 か所を同期。mdformat、just docs-build 通過

### R2（282 → 282 行）
| 問 | 判定 | 原因 | 直したこと |
| -- | ---- | ---- | ---------- |
| Q1〜Q10 | 正解（Q1 は変更後の 0.04 を答え、根拠に 4 節の elapsed ≈ learn_interval を挙げた） | — | — |
| （指摘: c の表の固定 β が 0.05 s の歩みに対応すると書いていない） | — | 曖昧 | 表の見出しを「β1 = e^{-0.05/0.1}, fixed for steps of 0.05 s」に（行数は増えない） |
| （ほかの指摘: 3 節の段 2 の参照、SGD の数値例、表が c_m だけ） | — | 直さない | 誤答に結びついておらず、足すと長くなる |
- 次は保留問題 H1〜H3 で汎化確認

### 汎化確認（282 → 282 行）
| 問 | 判定 | 原因 | 直したこと |
| -- | ---- | ---- | ---------- |
| H1〜H3 | 正解 | — | — |
| （指摘: elapsed == 0 の「moves nothing」が moment・mass を含むか） | — | 曖昧 | 「neither the weights nor the state」を同じ行に足した（ja も） |
- 最終状態: 評価用 10/10（R2）、保留 3/3。ループ終了。en・ja とも 282 行、mdformat と just docs-build 通過。ja の日本語の推敲（yomiyasu）は未実施
- 残した指摘: 「weights that require gradients」と「weights that learn」の言い分け、コード例の params の出どころ、1 歩目の計算が結論だけ（誤答なし）
