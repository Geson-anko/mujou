# docs/en/user-guide/writing-a-node.md・running.md 理解度テスト

## 対象
- ファイル: `docs/en/user-guide/writing-a-node.md`、`docs/en/user-guide/running.md`（取り込むサンプル `examples/leaky_chain/leaky.py`、`examples/quickstart.py`、`examples/subscribe.py` も渡す）
- 読者と用途: mujou を初めて使う利用者。自分のノードのクラスを書き、ホストのプログラムからネットワークを走らせ、入力を書いて出力を読む

## 評価用問題
### Q1 `on_update` で状態を `self.x = self.x + dt * f` と書いた。このままでよいか。どう書くか、理由も
- 要点: その場で更新する（`+=`、`copy_` など）
- 要点: 代入し直すと、`on_start` で作ったテンソルへのほかの参照（optimizer など）が古いテンソルを指したままになる（復元もそのテンソルに書き込む）

### Q2 センサーに合わせて平均 10 ms ごとに更新したい。どう指定するか。`on_update` は `dt == 0.01` と仮定してよいか
- 要点: `super().__init__(period=0.01)` で渡す（クラス属性 `period` は `TypeError`、hook の中で待たない）
- 要点: 仮定しない。`dt` は実際の経過時間で、`period` ちょうどではない。`dt` に比例して更新する

### Q3 厳密解のない系で、`dt` が長いと陽的 Euler 法が発散する。`dt` に上限を設けて切り捨ててよいか。どうするか
- 要点: 切り捨てない（経過時間が失われ、連続時間の解からずれる）
- 要点: 1 回の `on_update` の中で `dt / n` の刻みに分け、`for` ループで `n` 回更新する

### Q4 最初の `on_update` からノードにゼロでない入力を見せたい。ホストはどうするか
- 要点: `mujou.run(net, inputs={port: value})` で渡す
- 要点: `mujou.run` の後の `write` では、それまでの更新でノードはゼロを読む

### Q5 入力 x を書き、x への応答を出力から読みたい（グローバルタイマー）。いつ・どう待って読むか
- 要点: `mujou.time.sleep(seconds)` で見せる時間だけ待ち、その終わりに `read` する（標準の `time.sleep` は使わない）
- 要点: `write` の直後に読むと前の入力への出力が返る

### Q6 入力と教師信号を、ノードが必ず組で新しい値を読むように切り替えたい。どうするか
- 要点: 1 回の `write` に両方の Port を入れる（同時に切り替わり、片方だけ新しい値を読むことはない）

## 保留問題
### H1 出力が新しくなるたびに処理し、実時間で 2 秒たったらやめたい。何を使い、2 秒たつと何が起き、実行はどうなるか
- 要点: `run.subscribe(port, timeout=2.0)`
- 要点: `TimeoutError` が出る。実行は走り続ける（ループ本体の時間も数える）

### H2 復元した実行で、`load_state_dict` は `on_start` の前と後のどちらで呼ばれるか。復元しない実行では呼ばれるか
- 要点: `on_start` の後
- 要点: 復元するときだけ呼ばれる

### H3 一度走らせて終わったネットワークのインスタンスを、もう一度 `mujou.run` に渡してよいか
- 要点: 不可。`RuntimeError`
- 要点: ノードも含めて新しいインスタンスを作る

## ラウンド記録
### R1（writing-a-node 144 行、running 111 行）

改稿前の行数（main からの単純化の前）: writing-a-node 136 行、running 134 行。

| 問 | 判定 | 原因 | 直したこと |
| -- | ---- | ---- | ---------- |
| Q1 | 正解 | - | - |
| Q2 | 正解 | - | - |
| Q3 | 正解 | - | - |
| Q4 | 正解 | - | - |
| Q5 | 正解 | - | 読みにくさの指摘（待つ秒数の決め方）に対し、期待する出力が分かるなら `wait_for` で待つ一文を足した |
| Q6 | 正解 | - | - |

正解 6/6。読みにくかった箇所への対応: super() の段落の「keeps the node right」を言い換え、「The cut-off time is lost」を「The time you cut off is lost」にした。番号付きの節と番号なしの節が混ざる点は、番号が例のノードの 3 手順に対応しているので残した。

### 汎化確認（保留問題）

H1・H2・H3 とも正解（3/3）。指摘: subscribe の `timeout` が手順 4 にしか無い、`on_pause`/`on_resume` の「状態の取得」が state.md を読まないと分からない。どちらもリンク先で説明しているので直さない。
