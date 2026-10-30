# docs/en/topics/pc-concepts.md 理解度テスト

## 対象
- ファイル: docs/en/topics/pc-concepts.md（同じ内容を docs/ja/topics/pc-concepts.md にも反映）
- 読者と用途: mujou.pc の部品（Layer, Input, Output, LearnClock, TimeConstants, optimizer, relaxer）のつながりと、Layer の 1 回の on_update を理解し、自分のノードや relaxer を書く利用者・開発者。

## 評価用問題
### Q1 `pc.Layer` の 1 回の `on_update` で、前のノードへ送る `g_in` の元になる `delta` は、その更新で `r` と `lam` を動かす前の値と後の値のどちらで作るか。
- 要点: 動かす前（段 2 の vjp が段 3 の relax_ より先）

### Q2 速度を覚える自作の relaxer を書いた。ノードはどのメソッドを、どの場面で呼んでその記憶を捨てさせるか。そのとき `r` と `lam` はどうなるか。
- 要点: `reset_()`／新しいサンプル（`tick.new` が True）の更新で、`relax_` の前／`r` と `lam` は保たれる（リセットしない）

### Q3 自作の relaxer の中で `lam` も一緒に動かしてよいか。また relaxer の不動点はどこでなければならないか。
- 要点: relaxer は `lam` を動かさない（`lam` は動いた `r` で後から進む）／不動点は力 F = 0 かつ relaxer の状態が静止／そうでないと `lam` が BP の credit にならない

### Q4 `pc.TimeConstants(h=4e-3, lam=8e-3, r=2e-3)` で `period=` を渡さずに `pc.Chain` を作った。何が起き、どう直すか。
- 要点: 既定の period は tau.h/4 = 1 ms、tau.max_dt = min(4e-3/(2*3), 24e-3) ≈ 0.67 ms で period が max_dt より長い → ValueError／`period=` に tau.max_dt 以下を渡す

### Q5 `pc.Layer` が活動 `h` ではなくずれ `r` を状態に持つのはなぜか。
- 要点: 予測 mu は新しいサンプルや学習（自分・下のノード）で一度に跳ぶ／h を持つと r = h - mu が急に大きくなり、偽の誤差が g_in と重みの勾配に入る／r を持てば h = mu + r が予測と一緒に動く

### Q6 同僚が「`tau.r` は平衡点を変えないから、ResBlock 8 段の深い鎖でも `tau.r=math.inf` で十分」と言う。正しいか。何を勧めるか。
- 要点: 平衡点を変えない点は正しい（lam が r を溜めるので r=0）／だが減衰が無いと深い鎖は発散する（r を動かすと上の h がヤコビアンの積で動き、勾配流でない）／ふつうは tau.r = tau.h

### Q7 順伝播が強く増幅する鎖（ヤコビアンの積が 1 よりずっと大きい）が落ち着かない。`tau.r` を短くしていけば解決すると文書から言えるか。
- 要点: 言えない。どの一定の tau.r でも落ち着かない（減衰の限界）

### Q8 新しいサンプルが届いた更新で、`pc.Layer` は `r` と `lam` を 0 に戻すか。その更新で次のノードへ送る `h` は何か。
- 要点: 戻さない（保つ）／新しい入力から計算した mu にその更新で動かした r を足した h = mu + r

### Q9 `learn_interval=24e-3` の `Layer` で、`tick.new` が True の更新は重みを学ぶか。理由は何か。
- 要点: 学ばない／その更新の delta はまだ前の入力のもの

### Q10 `Layer` や `Chain` の `relaxer=` に渡せるものは何か。既定はどれで、その状態（キー `"relaxer"`）には何が入るか。
- 要点: pc.relax.Euler() か自作の relaxer だけ／既定は pc.relax.Euler()／状態は空

## 保留問題
### H1 あるバッチで `mask` がすべて False だった。その更新で `Layer` の `r` と `lam` は動くか。重みは学ぶか。
- 要点: r と lam は緩和する（動く）／重みは学ばない

### H2 `tau.lam=math.inf` かつ `tau.r=math.inf` にすると、平衡点は BP と一致するか。
- 要点: 一致しない／lam は 0 のまま、r は 0 に戻らない／ただの predictive coding

### H3 relaxer の `step_` に渡される力 F の式と、`pc.relax.Euler` の 1 回の動きを書け。
- 要点: F = (1 + tau_h/tau_r) r + lam + g／r ← r − (dt/tau_h) F

## ラウンド記録
### R0（執筆、287 → 291 行）
- 「ノードが r を持つ理由」を 4 節の下から冒頭の「ノードが持つ量」の直後へ移動（埋没: LearnClock と無関係な位置だった）
- 減衰が要る理由に「g は h を固定した勾配で、緩和が勾配流でない」を追記、増幅する鎖の限界を追記（specs/pc.md「減衰が要る理由と限界」）
- 段 3 に tick.new での reset_ を明記（コードは relax_ の前に呼ぶ）
- TimeConstants 表の tau.r 行から inf 列に紛れた「ふつうは tau.h／平衡点は変わらない」を表の下へ。period > max_dt で ValueError になる条件（tau.r < tau.h）を追記（time_constants.py, clock.py）
- 7 節: relaxer は Euler と自作だけと明記

### R1（291 → 292 行）
| 問 | 判定 | 原因 | 直したこと |
| -- | ---- | ---- | ---------- |
| Q1 | 正解 | - | - |
| Q2 | 正解 | - | - |
| Q3 | 正解 | - | - |
| Q4 | 正解 | - | 生徒は計算で導いた（確信: 中）。数値の例は Q4 専用の注記になるので足さない |
| Q5 | 正解 | - | - |
| Q6 | 正解 | - | - |
| Q7 | 正解 | - | - |
| Q8 | 正解 | - | 「保たれた r」と答えた。この更新で緩和した r だという点はぼかしているが、誤りではないので正解にした |
| Q9 | 正解 | - | - |
| Q10 | 正解 | - | - |

生徒の申告に応じた直し:
- 段 3 のセルから reset_ を外し、表の下の箇条書きに「tick.new なら段 3 の直前に reset_」と移した（詰め込み）
- tau.r 行の「dt / tau.r of r more」を「A further dt / tau.r of r toward 0」にした（曖昧）
- max_dt の括弧書き「tau.h / 4 with tau.r = tau.h」を削った。表の下の段落と重複していた
- 見送ったもの: ValueError の具体例（Q4 専用になる）、質量 c の表の見出し（今回の変更の範囲外）、F と不動点を並べる件（7 節で隣り合う箇条書きにすでになっている）

### 保留問題（292 → 292 行）
| 問 | 判定 | 原因 | 直したこと |
| -- | ---- | ---- | ---------- |
| H1 | 正解 | - | - |
| H2 | 不正解（要点「ただの predictive coding」が欠けた） | 曖昧: tau.lam 行の括弧書き「plain PC with tau.r = inf」が、両方 inf の組み合わせだと読めなかった | 括弧書きを表から外した。表の下の段落に「両方 math.inf なら、平衡点はただの predictive coding のもの」と 1 文で書いた |
| H3 | 正解 | - | 生徒の申告に応じて、Euler の状態の箇条書きに「状態を持たず、reset_ は何もしない」と書き足した（relax.py の _BoundEuler で確認） |
- H2 の修正後に保留問題をもう一度出しても、汎化の確認にはならない（答えを見た後の改稿のため）。汎化を確かめるなら、新しい保留問題が要る
