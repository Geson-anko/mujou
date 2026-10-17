# user-guide（後半 6 ページ・日本語）理解度テスト

## 対象
- ファイル: `docs/ja/user-guide/` の `state.md`・`time.md`・`torch.md`・`testing.md`・`errors.md`・`update-rules.md`
- 生徒が読むファイル: 上の 6 ページと、取り込まれるサンプル `examples/save_and_restore.py`・`examples/timers.py`・`examples/torch_node.py`・`examples/testing_a_node.py`・`examples/errors.py`
- 読者と用途: Concepts の章を読み、ノードを書ける利用者。状態の保存・復元、時間の制御、ノード内の PyTorch、決定的なテスト、例外と終了の扱いを、これらのページを見ながら正しく書けるようになる

## 執筆の判断（R0）
- Concepts（`concepts/state.md`・`concepts/time.md`）が木の中身・自動収集の表・停止の仕組み・タイマーの考え方を持つので、ここでは手順・注意・コード例に絞り、考え方はリンクで済ませた
- 見出しへのアンカー付きリンクは使わない。`toc` に slugify の設定が無く、日本語の見出しはアンカーが空になり、`mkdocs build --strict` で INFO が出るため
- 書いた内容を実物で確かめた: 復元でグローバル時刻が巻き戻る、同じタイマーで別の実行中は `ValueError`、木は既定の `torch.load`（weights_only）で読める、生成器の状態の override 例が動く、VirtualTimer で途中保存→復元が止めずに続けた結果とビット一致、例外のログは logger "mujou" に ERROR（`crates/mujou-python/src/runtime.rs` の `log_error`）
- orchestrator の元メモ（errors.md）の「停止のタイムアウト（既定は実時間で 30 秒）」は `state_dict` の `pause_timeout` のこと。`shutdown`/`join` の既定は `math.inf`。実物に合わせて表にした
- `run.write` が `Timestamp` を返すことは前提にしてよいと指示があったので time.md に 1 か所だけ書いた

## 評価用問題
### Q1 ノードに `self.count = 0`（int）を持たせ、`on_update` で 1 ずつ増やしている。保存した木から復元すると、count が 0 から数え直された。なぜか。count を保存・復元させるにはどうするか
- 要点: int（数値の属性）は自動では状態に集まらない（集まるのは Tensor と nn.Module だけ）
- 要点: `state_dict` と `load_state_dict` の両方を override し（super() を呼び）、count を足して戻す（Tensor の属性にしてその場で更新する、も可）

### Q2 状態を取り終えた `net`（with を抜けた後）を、そのまま `mujou.run(net, state=tree)` に渡して続きを走らせてよいか
- 要点: いけない。1 つのネットワークは 1 回しか走らせられず `RuntimeError` になる
- 要点: 同じ構成の新しいインスタンスを作って渡す

### Q3 既定のグローバルタイマーで走らせている。ホストは復元の直前に `t0 = mujou.time.time()` を記録し、`mujou.run(new_net, state=tree)` の後に `mujou.time.time() - t0` で経過時間を測った。この測り方の問題は何か
- 要点: 復元するとタイマーの時刻が木の時刻に戻り、グローバル時刻（`mujou.time.time()`）も巻き戻る
- 要点: 復元の前後の時刻を比べても経過時間にならない（負にもなりうる）

### Q4 グローバルタイマーで実行 A が走っている間に、保存した木から実行 B を復元して走らせたい。そのまま `mujou.run(netB, state=tree)` を呼ぶと何が起きるか。どうすれば B を走らせられるか
- 要点: `ValueError` になり、何も起動しない
- 要点: 先に A を終了させる、または B に自分の `mujou.time.Timer()` を渡す

### Q5 長時間の実行が、ノードの例外で止まった。止まる直前の状態を `run.state_dict()` で取れるか。状態を失わないためには、普段どうしておくか
- 要点: 取れない（`NetworkError` が出る。例外で終了したときの状態は保存されない）
- 要点: 実行中に定期的に `run.state_dict()` を取って保存しておく

### Q6 同僚から受け取った状態のファイル（Tensor・数値・dict だけの木）を読み込みたい。`pickle.load` と既定の `torch.load` のどちらを使うか。理由も
- 要点: 既定の `torch.load`（`weights_only=True`）を使う。その木なら読める
- 要点: `pickle.load`（と `weights_only=False`）は読み込みでファイルの中のコードを実行できるので、信頼できないファイルには使わない

### Q7 ノードの `on_update` で、送る前に 0.01 秒待ちたいので `mujou.time.sleep(0.01)` を書いた。何が起きるか。一定間隔で更新したいなら代わりにどうするか
- 要点: `RuntimeError` になる（hook の中から時間を動かす・待つことはできない）
- 要点: `super().__init__(period=0.01)` のように周期を渡す

### Q8 `timer = mujou.time.Timer()` を渡して走らせた実行で、入力を 0.5 秒（その実行の時間）見せてから出力を読みたい。待つのに `mujou.time.sleep(0.5)` と `timer.sleep(0.5)` のどちらを使うか。また `run.read` が返した時刻は何と比べるか
- 要点: `timer.sleep(0.5)`。`mujou.time` の関数はグローバルタイマーで、この実行のタイマーとは別だから
- 要点: `run.read` の時刻は `timer.now()`（同じタイマーの時刻）と比べる。`mujou.time.time()` とは比べない

### Q9 PC-ALM で学習するネットワークを作りたい。`mujou.pc` から PC-ALM のノードを import して使えるか。使えないなら今どうするか
- 要点: 使えない。`mujou.pc` は予定で、まだ実装されていない
- 要点: `mujou.Node` を継承し、`on_update(dt)` に更新式を自分で書く

### Q10 ノードの中で、損失の勾配を autograd で計算してパラメータを更新したい。どこを何で囲み、何の印を付け、パラメータの更新はどこで行うか
- 要点: 順伝播と勾配の計算（`torch.autograd.grad` / `.backward()`）の両方を `with torch.enable_grad():` で囲む
- 要点: その行に `# mujou: local-autograd` を付ける
- 要点: パラメータの更新はブロックの外で、その場で行う

### Q11 テストを再現できるように、ノードの `on_start` で `torch.manual_seed(0)` を呼んでよいか。代わりにどうするか
- 要点: いけない。既定の生成器はプロセス全体で共有され、ほかのノードの乱数まで変わる
- 要点: `super().__init__(seed=0)` で seed を渡し、乱数は `self.generator` から引く

### Q12 CPU で計算するノードを 8 つ走らせる。マシンの物理コアは 8。torch の内部のスレッド数を、誰が・いつ・何の値に設定するか
- 要点: ホストが `mujou.run` の前に `torch.set_num_threads(n)` で設定する（ノードの中では変えない）
- 要点: ノード数 × n が物理コア数を超えないように、n = 1

### Q13 `VirtualTimer` を使うテストのループで、`timer.advance(h)` と `run.step(...)` はどちらを先に呼ぶか。`advance` を呼び忘れると何が起きるか
- 要点: `advance` が先、`step` が後（各ノードの dt がちょうど h になる）
- 要点: 時刻が進んでいないノードは更新されない（何も起きない）

### Q14 起動の後に、あるノードの `on_update` が例外を投げた。ホストは何も呼ばずに `mujou.time.sleep(60)` で待っている。ホストはいつ例外に気づくか。止まるのを待つにはどう書くべきか。元の例外はどこから取り出すか
- 要点: 次に `Run` のメソッド（write/read/state_dict/shutdown/join など）を呼ぶまで気づかない
- 要点: `run.join(timeout)` で待つ（`NetworkError` が送出される）
- 要点: 元の例外は `NetworkError` の `__cause__`

### Q15 ロボットのノードで、異常で止まったときだけモータを安全な姿勢に戻したい。`on_exception` と `on_stop` のどちらに書くか。どの例外で止まったかで処理を変えられるか
- 要点: `on_exception`（例外による終了のときだけ、`on_stop` の前に、全ノードで呼ばれる）。`on_stop` は正常な終了でも呼ばれる
- 要点: 変えられない。`on_exception` は引数を取らず、ノードは例外を知れない（原因への対処はホストが `__cause__` で行う）

## 保留問題
### H1 with 文の本体で、ホストのコードが `KeyError` を投げた。同じころノードも例外で止まっていた。with を抜けたときに呼び出し元へ伝わるのはどちらの例外か。ノードの例外はどこで確かめるか
- 要点: 本体の `KeyError`（`NetworkError` は捨てられる）
- 要点: ノードの例外はログ（logger "mujou"）で確かめる

### H2 `mujou.testing.run(net, inputs=...)` で、`timer` を渡さずにテストを書いた。`advance` は呼べないので `run.step()` だけを繰り返したら、結果が毎回わずかに違う。なぜか。どう直すか
- 要点: `timer` の既定はグローバルタイマーで、時計がひとりでに進み、dt が step の間の実時間になる
- 要点: `VirtualTimer()` を渡し、`advance` で進める

### H3 実行中に `run.state_dict()` を呼ぶと `PauseTimeoutError` が出た。このとき実行はどうなっているか。待ち時間を延ばすにはどうするか
- 要点: 実行は走り続けている（状態は取れていない）
- 要点: `mujou.run(..., pause_timeout=秒)` で延ばす（既定は実時間 30 秒）

## ラウンド記録
### R0（400 行: state 79 / time 71 / torch 83 / testing 75 / errors 81 / update-rules 11）
執筆と出題のみ。

### R1（400 行 → 400 行。文を既存の行に足したため行数は変わらない）
全 15 問正解（15/15）。どの解答も、根拠に挙げた箇所が答えを支えていた。

| 問 | 判定 | 原因 | 直したこと |
| -- | ---- | ---- | ---------- |
| Q1〜Q15 | 正解 | - | - |

生徒が読みにくかった箇所への対応（どれも正解した問題についての指摘。妥当なものだけ直した）:
- state.md: 数値を状態に入れる方法が、乱数生成器の例からの類推しかなかった → 0 次元の Tensor で持てば自動で集まる、と 1 文足した
- state.md: 復元の前後で比べるなとだけあり、代わりの測り方が無かった → 復元の後に読んだ時刻同士の差で測る、と足した
- state.md: どの方法で保存したらどの方法で読むかが無かった → 保存と同じ方法で読む、と足した
- time.md: period の意味がリンク先にしか無かった → 周期は 2 回の更新の間の平均の時間（実行のタイマーの秒）、と足した
- torch.md: on_thread_start の説明が無かった → ノードのスレッドで on_start の直前に 1 回呼ばれる hook、と足した
- testing.md: コード片の `u` と `Chain` の出どころが書かれていなかった → 上のサンプルのもの、と足した
- update-rules.md の「既定の更新手法の説明が無い」は直していない。PC-ALM はまだ実装されておらず、名前と出典の URL と選んだ理由で、このページの用途には足りるため

次は保留問題 H1〜H3 を出す。

### 汎化確認（保留問題 H1〜H3）
3/3 正解。保留問題を評価用に移す必要はない。

保留問題の解答で生徒が挙げた読みにくい箇所を、最小限の改稿で直した（400 → 411 行）:
- errors.md: with 文の節の「ログで確かめる」に、「ログ」の節への案内を足した
- errors.md: タイムアウトの表を `shutdown`/`join` の 2 行だけにした。`state_dict` の `pause_timeout` は `mujou.run` の引数なので、表から外して段落に分けた。時間切れのとき、止まっていたノードも再開することを足した（`crates/mujou-core/src/runtime.rs` の `resume_after_pause` で確かめた）。ページ冒頭の説明にも、状態を取るときのタイムアウトを加えた
- testing.md: 保存と復元の例に import と `u` の定義を足した。`Chain` だけを上のサンプルから使う。この断片をサンプルの `Chain` と組み合わせて動かし、動くことを確かめた
- testing.md: 「VirtualTimer を渡し忘れない」の節に、直し方と「手順」の 2・3 への案内を足した

ループを終える。

### 英語版 R1 の反映（411 → 415 行）
英語版（`guide-b-en.md` の R1）で見つかった 6 か所を、日本語版にも最小の改稿で反映した。torch.md は、dt の件で examples を直すときに一緒に直すので、今回は触っていない。
- state.md: pickle の項に「木を他人とやり取りするなら `torch.save` で保存し、相手が既定の `torch.load` で読めるようにする」を足した（英語版の Q6 は「曖昧」に分類されている）
- state.md: 時刻が戻るタイミングを「`mujou.run(..., state=tree)` が戻った時点で巻き戻っている」とした
- time.md: 停止を括弧で定義した（状態を取る間、実行は止まる）
- testing.md: 同じ `timer` を使い回してよい理由（前半の実行が終わっている）を足した。例外のテストの断片に import を足した
- errors.md: 出力先を変えている箇所として `examples/errors.py` を示した

### 英語版 R2 の反映（行数は変わらず）
- time.md: 2 か所とも「Python 標準の `time.sleep`」とした（`mujou.time.sleep` と取り違えないように）。`mujou.time` の関数名を挙げ、「グローバルタイマーで働き、この実行のタイマーには効かない」とした
- state.md: 0 次元の Tensor の作り方の例を括弧で足した。時刻が戻る文を「復元すると戻る」と「`mujou.run` が戻った時点で、すでに戻っている」の 2 文に分けた

### 英語版 R3 の反映（state 79 → 84 行）
- state.md: 保存の方法が分からないファイルの読み方を足した（まず既定の `torch.load` で読み、失敗したら、送り手を信頼できるときだけ pickle で読む）。`Noisy` の例に import を足した
- time.md: 1 つ目の箇条の主文を規則にし、例は括弧に入れた

### 英語版 R4 の反映（state 84 → 86 行）
- state.md「保存の形式」を英語版にそろえた。冒頭を 2 段落に分けた。1 つ目は保存の方法の使い分け（自分用ならどちらでもよい、共有するなら `torch.save`）、2 つ目は読み方（保存と同じ方法で読む。分からなければまず既定の `torch.load` で読み、失敗したら、送り手を信頼できるときだけ pickle で読む）。pickle の箇条は警告だけに戻した

### 英語版の保留問題の反映（errors 84 行のまま）
- errors.md「タイムアウト」: `pause_timeout` は実行を起動するときに決まり、走っている間は変えられない、と 1 文足した
