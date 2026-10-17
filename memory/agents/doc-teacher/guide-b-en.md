# user-guide（後半 6 ページ・英語）理解度テスト

## 対象
- ファイル: `docs/en/user-guide/` の `state.md`・`time.md`・`torch.md`・`testing.md`・`errors.md`・`update-rules.md`
- 生徒が読むファイル: 上の 6 ページと、取り込まれるサンプル `examples/save_and_restore.py`・`examples/timers.py`・`examples/torch_node.py`・`examples/testing_a_node.py`・`examples/errors.py`
- 読者と用途: Concepts の章を読み、ノードを書ける利用者。状態の保存・復元、時間の制御、ノード内の PyTorch、決定的なテスト、例外と終了の扱いを、これらのページを見ながら正しく書けるようになる（日本語版と同じ）

## 執筆の判断（R0）
- 完成した日本語版 `docs/ja/user-guide/` の同名ページ（R1 と汎化確認の後、411 行）を翻訳した。節・表・リスト・admonition・`--8<--` の行範囲・コード片をそろえた。行数も 411 行で一致する
- 用語は `docs/en/concepts/` に合わせた: host / run / the run's timer / global timer / global time / time scale / state tree / port / pause / update rule。散文は "Mujou"、コードは `mujou`
- 見出しへのアンカーリンクは使わない（日本語版と同じく、ページへのリンクだけ）
- 「テストの保存と復元」のコード片を `examples/testing_a_node.py` の `Chain` と組み合わせて実行し、通ることを確かめた
- `uv run pre-commit run --files <6 ファイル>`（mdformat が表を整形）と `uv run mkdocs build --strict` が通る
- 日本語版との食い違い（報告のみ）: `torch.md` の「例では学習率に `min(dt, 0.01)` を掛けて…発散しないようにしている」と `examples/torch_node.py` の `min(dt, 0.01)`、`examples/testing_a_node.py` の `min(dt / self.tau, 1.0)` は、Concepts（`concepts/time.md`）の「`dt` を打ち切らない。刻みを分割する」と食い違う。英語版も日本語版に合わせて訳した

## 評価用問題（`guide-b-ja.md` の Q1〜Q15 の英訳。要点も同じ）
### Q1 A node keeps `self.count = 0` (an int) and adds 1 to it in each `on_update`. After restoring from a saved tree, count started again from 0. Why? How do you make count saved and restored?
- 要点: int（数値の属性）は自動では状態に集まらない（集まるのは Tensor と nn.Module だけ）
- 要点: `state_dict` と `load_state_dict` の両方を override し（super() を呼び）、count を足して戻す（0 次元の Tensor にしてその場で更新する、も可）

### Q2 After taking the state of `net` (after the `with` statement has exited), may you pass the same `net` to `mujou.run(net, state=tree)` to run the rest?
- 要点: いけない。1 つのネットワークは 1 回しか走らせられず `RuntimeError` になる
- 要点: 同じ構成の新しいインスタンスを作って渡す

### Q3 A network runs on the default global timer. Just before restoring, the host records `t0 = mujou.time.time()`, and after `mujou.run(new_net, state=tree)` it measures the elapsed time as `mujou.time.time() - t0`. What is wrong with this measurement?
- 要点: 復元するとタイマーの時刻が木の時刻に戻り、グローバル時刻（`mujou.time.time()`）も巻き戻る
- 要点: 復元の前後の時刻を比べても経過時間にならない（負にもなりうる）。復元の後に読んだ時刻同士の差で測る

### Q4 Run A is running on the global timer. While it runs, you want to restore run B from a saved tree. What happens if you just call `mujou.run(netB, state=tree)`? How can you run B?
- 要点: `ValueError` になり、何も起動しない
- 要点: 先に A を終了させる、または B に自分の `mujou.time.Timer()` を渡す

### Q5 A long run stopped because a node raised. Can you get the state just before it stopped with `run.state_dict()`? What should you do routinely so as not to lose the state?
- 要点: 取れない（`NetworkError` が出る。例外で終了したときの状態は保存されない）
- 要点: 実行中に定期的に `run.state_dict()` を取って保存しておく

### Q6 A colleague sent you a state file (a tree of only tensors, numbers and dicts). Do you read it with `pickle.load` or with the default `torch.load`? Why?
- 要点: 既定の `torch.load`（`weights_only=True`）を使う。その木なら読める
- 要点: `pickle.load`（と `weights_only=False`）は読み込みでファイルの中のコードを実行できるので、信頼できないファイルには使わない

### Q7 To wait 0.01 seconds before sending, a node calls `mujou.time.sleep(0.01)` in `on_update`. What happens? What should you do instead to update at regular intervals?
- 要点: `RuntimeError` になる（hook の中から時間を待つ・動かすことはできない）
- 要点: `super().__init__(period=0.01)` のように周期を渡す

### Q8 A run was started with `timer = mujou.time.Timer()`. You want to show an input for 0.5 seconds (of that run's time) and then read the output. Do you wait with `mujou.time.sleep(0.5)` or `timer.sleep(0.5)`? And what do you compare the time returned by `run.read` with?
- 要点: `timer.sleep(0.5)`。`mujou.time` の関数はグローバルタイマーで、この実行のタイマーとは別だから
- 要点: `run.read` の時刻は `timer.now()`（同じタイマーの時刻）と比べる。`mujou.time.time()` とは比べない

### Q9 You want to build a network that learns with PC-ALM. Can you import the PC-ALM nodes from `mujou.pc` and use them? If not, what do you do now?
- 要点: 使えない。`mujou.pc` は予定で、まだ実装されていない
- 要点: `mujou.Node` を継承し、`on_update(dt)` に更新式を自分で書く

### Q10 In a node, you want to compute the gradient of a loss with autograd and update the parameters. What do you wrap in what, what mark do you add, and where do you update the parameters?
- 要点: 順伝播と勾配の計算（`torch.autograd.grad` / `.backward()`）の両方を `with torch.enable_grad():` で囲む
- 要点: その行に `# mujou: local-autograd` を付ける
- 要点: パラメータの更新はブロックの外で、その場で行う

### Q11 To make a test reproducible, may a node call `torch.manual_seed(0)` in `on_start`? What should it do instead?
- 要点: いけない。既定の生成器はプロセス全体で共有され、ほかのノードの乱数まで変わる
- 要点: `super().__init__(seed=0)` で seed を渡し、乱数は `self.generator` から引く

### Q12 You run 8 nodes that compute on the CPU, on a machine with 8 physical cores. Who sets torch's internal thread count, when, and to what value?
- 要点: ホストが `mujou.run` の前に `torch.set_num_threads(n)` で設定する（ノードの中では変えない）
- 要点: ノード数 × n が物理コア数を超えないように、n = 1

### Q13 In a test loop with a `VirtualTimer`, which do you call first, `timer.advance(h)` or `run.step(...)`? What happens if you forget to call `advance`?
- 要点: `advance` が先、`step` が後（各ノードの dt がちょうど h になる）
- 要点: 時刻が進んでいないノードは更新されない（何も起きない）

### Q14 After the start, a node's `on_update` raised. The host is waiting with `mujou.time.sleep(60)` and calls nothing else. When does the host notice the exception? How should it wait for the run to stop? Where does it get the original exception?
- 要点: 次に `Run` のメソッド（write/read/state_dict/shutdown/join など）を呼ぶまで気づかない
- 要点: `run.join(timeout)` で待つ（`NetworkError` が送出される）
- 要点: 元の例外は `NetworkError` の `__cause__`

### Q15 In a robot node, you want to move the motors to a safe position only when the run stops abnormally. Do you write it in `on_exception` or `on_stop`? Can you vary what you do depending on which exception stopped the run?
- 要点: `on_exception`（例外による終了のときだけ、`on_stop` の前に、全ノードで呼ばれる）。`on_stop` は正常な終了でも呼ばれる
- 要点: 変えられない。`on_exception` は引数を取らず、ノードは例外を知れない（原因への対処はホストが `__cause__` で行う）

## 保留問題（`guide-b-ja.md` の H1〜H3 の英訳）
### H1 In the body of a `with` statement, the host's code raised `KeyError`. Around the same time, a node also stopped with an exception. Which exception reaches the caller when the `with` exits? Where do you check the node's exception?
- 要点: 本体の `KeyError`（`NetworkError` は捨てられる）
- 要点: ノードの例外はログ（logger "mujou"）で確かめる

### H2 You wrote a test with `mujou.testing.run(net, inputs=...)` without passing `timer`. Since you cannot call `advance`, you just repeat `run.step()`, and the results differ slightly every time. Why? How do you fix it?
- 要点: `timer` の既定はグローバルタイマーで、時計がひとりでに進み、dt が step の間の実時間になる
- 要点: `VirtualTimer()` を渡し、`advance` で進めてから `step` する

### H3 Calling `run.state_dict()` on a running network raised `PauseTimeoutError`. What state is the run in now? How do you make the wait longer?
- 要点: 実行は走り続けている（状態は取れていない。止まっていたノードも再開する）
- 要点: `mujou.run(..., pause_timeout=seconds)` で延ばす（既定は実時間 30 秒）

## ラウンド記録
### R0（411 行: state 79 / time 71 / torch 83 / testing 83 / errors 84 / update-rules 11）
執筆と出題のみ。

### R1（411 行 → 415 行）
14/15 正解。

| 問 | 判定 | 原因 | 直したこと |
| -- | ---- | ---- | ---------- |
| Q1〜Q5, Q7〜Q15 | 正解 | - | - |
| Q6 | 不正解（「文書から判断できない」と留保した） | 曖昧 | state.md「Formats」で「保存したのと同じ方法で読む」と「信頼できないファイルを pickle で読まない」の 2 つの規則が、他人から受け取ったファイルに対してどちらを優先するか読めなかった。pickle の項に「他人と木をやり取りするなら `torch.save` で保存し、相手は既定の `torch.load` で読む」と 1 文足した。日本語版にも同じ曖昧さがある（報告） |

生徒が読みにくかった箇所への対応（妥当なものだけ）:
- state.md: 「as soon as the run is restored」がどの呼び出しの時点か曖昧 → 「`mujou.run(..., state=tree)` が戻った時点で巻き戻っている」に変えた
- time.md: pause が未定義 → 「状態を取る間、実行は止まる」と括弧で足した
- testing.md: 保存と復元の例で同じ `timer` を使い回してよい理由が無い → 前半の実行は終わっているので許される、と 1 文足した（state.md の「同じタイマーでほかの実行が走っている間は復元できない」と対になる）
- testing.md: 例外のテストの断片に `import pytest` と `from mujou import NetworkError` を足した
- errors.md: 「上のサンプルは標準出力に出している」の指す先が曖昧 → `examples/errors.py` の次の行で、と明示した
- 直さなかった: torch.md の `on_thread_start` で変えてよい設定の一覧、`n = 1` の "small matrices"（torch.md はオーケストレーターの指示で dt の件以外も今回は最小限にとどめた。Q12 は正解している）、木の配置（Concepts の担当）、period の詳細（writing-a-node.md の担当）

### R2（415 行 → 415 行。文を既存の行に足したため行数は変わらない）
14/15 正解。

| 問 | 判定 | 原因 | 直したこと |
| -- | ---- | ---- | ---------- |
| Q1〜Q7, Q9〜Q15 | 正解 | - | - |
| Q6 | 正解（R1 の改稿で結論を言い切った。保存の方法が問題文に無いという留保は残ったが、要点は 2 つとも満たす） | - | - |
| Q8 | 不正解（`timer.sleep` は正しいが、理由を「`mujou.time.sleep` は実時間で待つ」とした。正しくはグローバルタイマーで待つ） | 曖昧 | time.md「What to use where」の箇条で、2 度目の `time.sleep` から「standard」が落ちていて `mujou.time.sleep` と混同できた。「the standard `time.sleep` of Python」と両方に付けた。「A timer for one run」の段落に、`mujou.time` の関数（`mujou.time.sleep`・`mujou.time.time()`）はグローバルタイマーで働き、この実行のタイマーではない、と名前を挙げた。日本語版にも同じ書き方がある（報告） |

生徒が読みにくかった箇所への対応:
- state.md: 時刻が巻き戻る文が読みにくい（R1 の改稿で入れた「by the time ... returns」）→ 「復元するとグローバル時刻が木の時刻に戻る。`mujou.run(..., state=tree)` が戻った時点で、すでに戻っている」の 2 文に分けた
- state.md: 0 次元 Tensor の作り方の例が無い → `self.count = torch.zeros((), device=self.device)` を括弧で足した
- 直さなかった: time.md の表に `run.read` との比較を足す（段落で足りる。Q8 の誤りは理由の側）、period と seed の併用（writing-a-node.md の担当）、CUDA ノードがあるときのスレッド数の数え方（torch.md は今回触らない）、例外のテストの断片の `run`（断片であることは前後で分かる）

次は R3 として、改稿した箇所に関わる Q1・Q3・Q6・Q8 を出す。全問正解なら保留問題へ。

### R3（415 行 → 420 行。Q1・Q3・Q6・Q8 だけを出題）
3/4 正解（生徒への番号は 1〜4 = Q1・Q3・Q6・Q8）。

| 問 | 判定 | 原因 | 直したこと |
| -- | ---- | ---- | ---------- |
| Q1 | 正解 | - | - |
| Q3 | 正解 | - | - |
| Q6 | 不正解（R2 に続いて「保存の方法による」と留保した。2 ラウンド続いたので曖昧として扱う） | 欠落 | 保存の方法が分からないファイルの扱いが文書に無かった。state.md「Formats」の pickle の項に「保存の方法が分からなければ、まず既定の `torch.load` で読む。失敗したら、送り手を信頼できるときだけ `pickle` で読む」を足した。`pickle.dumps` で保存した木（Tensor 入り）を既定の `torch.load` で読むと `UnpicklingError` になることを実物で確かめた。日本語版にも同じ欠落がある（報告） |
| Q8 | 正解（R2 の改稿で理由も正しくなった） | - | - |

生徒が読みにくかった箇所への対応:
- state.md: `Noisy` の例に import が無い → `from typing import Any, override` と `from mujou import Node` を足した（実行して import が通ることを確かめた）
- time.md: 「for example」が規則の文に付いていて、例なのか規則なのか分からない → 「ホストは実行のタイマーの `sleep` で待つ（たとえば入力を一定時間見せるとき）」と規則を主文にした
- 直さなかった: 0 次元 Tensor にすると読む側が `.item()` を要する（自明）、時刻の巻き戻りの記述が 2 ページにある（state.md が詳細、time.md が案内で、役割が分かれている）

行数が 2 割以上は増えていない（415 → 420）。次は R4 として Q6 だけを出す。正解なら保留問題へ。

### R4（420 行 → 422 行。Q6 だけを出題）
1/1 正解。Q6 は R3 の改稿で留保なく答えられた。

生徒が読みにくかった箇所への対応:
- state.md「Formats」: pickle の項 1 つに、信頼の警告・共有の勧め・保存の方法が分からないときの順序が詰め込まれ、見つけにくかった。冒頭の段落を 2 つに分けて書き直した。1 つ目の段落は、自分のファイルならどちらでもよい（例は pickle）、共有するなら `torch.save` という使い分け。2 つ目の段落は、保存と同じ方法で読むことと、保存の方法が分からないときの順序。pickle の項は信頼の警告だけに戻した
- 直さなかった: testing.md「例外のテスト」の断片に `run` の定義が無い（R2 でも挙がった。断片の前の文が `step` から届くと言っており、`run` が `ManualRun` であることは「Tools」の表から分かる。足すと例が長くなる）

次は保留問題 H1〜H3 を出す。

### 汎化確認（保留問題 H1〜H3。422 行 → 422 行）
3/3 正解。H3 は「走行中の実行の待ち時間を変えられるかは文書から分からない」と留保したが、要点（実行は走り続ける、`mujou.run(..., pause_timeout=秒)` で延ばす、既定は実時間 30 秒）は満たす。

改稿（最小）:
- errors.md「Timeouts」: `pause_timeout` は実行の起動時に決まり、走っている間は変えられない、と明示した。実装で確かめた: `Run.state_dict()` は引数を取らず（`src/mujou/_core/run.py`）、`pause_timeout` は `mujou.run` / `mujou.testing.run` の引数だけでネイティブの runtime に渡る。日本語版にも同じ欠落がある（報告）
- 直さなかった: with の本体も例外を投げたときにノードの例外をコードで取り出す方法（実装に無い。ログで確かめる、と書いてある）、testing.md の `run` の定義（R2・R4 と同じ判断）、advance を忘れたときの規則を「Steps」にも書く（「Rules of step」にあり、H2 は正解）

ループを終える。
