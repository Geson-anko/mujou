# docs/en/tutorial/3-test.md 理解度テスト

## 対象
- ファイル: `docs/en/tutorial/3-test.md`（snippet の取り込み元 `examples/test_leaky.py`）
- 読者と用途: mujou のチュートリアルを順に読んでいる利用者（PyTorch に慣れた Python 開発者）が、自分のノードを `mujou.run_manually` と `ManualRun.step` で決定的にテストできるようになる
- 事実の出典: `src/mujou/execution.py`（`ManualRun.step` / `run_manually` の docstring）、`specs/execution-model.md` の「手動の実行（シミュレーション）」、`memory/project_manual_run_simulation.md`

## 評価用問題

### Q1 1 つのノードのテストで、ノードの `on_update` が毎回ちょうど `dt = 0.002` を受け取るようにしたい。実行を始める関数と、各ステップで呼ぶものを答えよ。
- 要点:
  - `mujou.run_manually(node, ...)` で実行を始める（`with` で使う）
  - 各ステップで `run.step(0.002)` を呼ぶ（タイマーを 0.002 秒進めてから更新するので dt がちょうどその値）

### Q2 chain `a -> b`（`a` の出力を `b` が受ける。`net.a` と `net.b`）で、各ステップの中で `b` が「同じステップで `a` が送った値」を読むようにしたい。`run.step(h)` と `run.step(h, net.a, net.b)` のどちらを使うか。理由も。
- 要点:
  - `run.step(h, net.a, net.b)`
  - 渡したノードは渡した順に 1 つずつ更新され、後のノードは前のノードがこのステップで送った値を読む
  - `step(h)` では全ノードがステップ開始時の値を読むので、`b` は前のステップの `a` の値を読む

### Q3 ノード `a` と `b` が互いにメッセージを送り合う（`b` から `a` へも送る）ネットワークがある。`run.step(h)` の結果を、`run.step(h, ...)` にノードを何らかの順で渡して再現できるか。
- 要点:
  - できない（どの順でも一致しない）
  - 一致するのはメッセージが出力の方向にだけ流れる chain で、出力側から入力側への順に渡したときだけ

### Q4 `period=0.05` のノードを手動の実行で `run.step(0.001)` を 10 回呼んで進める。ノードは何回更新され、各回の `dt` はいくつか。
- 要点:
  - 10 回（手動の実行では周期は使われない）
  - 各回 `dt = 0.001`

### Q5 ノードが `ValueError` を出すことを確かめるテストを書く。`pytest.raises` をどこに置き、何を確かめるか。
- 要点:
  - `pytest.raises(NetworkError)` を `mujou.run_manually(...)` と同じ `with` 文に置き、ブロック全体を囲む
  - 理由: その後の呼び出し（`with` の終わりの `shutdown` を含む）も `NetworkError` を出し直すため
  - 例外の `__cause__` が `ValueError` であることを確かめる

### Q6 2 つのノードが同じ Python のリスト（属性として共有）に追記する。これを `run.step(dt)` で進めるテストは決定的になるか。どうすべきか。
- 要点:
  - 決定的にならない（`step(dt)` は全ノードを同時に更新するので、どちらが先かで結果が変わる）
  - ノード同士で変更する Python のオブジェクトを共有しない

## 保留問題

### H1 途中で取った状態 `state` から新しいインスタンスで手動の実行を再開した。その実行のタイマーはどれで、時刻はいくつから始まるか。
- 要点:
  - その実行専用の `VirtualTimer`
  - 状態を取った時刻から始まる

### H2 2 つの手動の実行を同じ時間軸に乗せ、一方の `step` で他方の時間も進めたい。どうするか。
- 要点:
  - 同じ `mujou.time.VirtualTimer()` を両方の `run_manually` に `timer` として渡す

### H3 `run.step(h, net.a, net.a)` と同じノードを 2 回渡した。`a` は何回更新されるか。理由も。
- 要点:
  - 1 回
  - 2 回目は時刻が進んでいない（dt が 0）ので飛ばされる

## ラウンド記録

### R0（改稿前 72 行 → 73 行）
- `mujou.testing.run` → `mujou.run_manually`。「`import mujou.testing` が要る」の文を削除
- ツール表: `workers` を持つことに、`step(dt)` と `step(dt, *nodes)` を別の行に
- 「The order of the updates」: `step(dt)` は同時・ステップ開始時の値（Jacobi）、`step(dt, *nodes)` は逐次。逆順の一致は一方向の chain に限ると明記。「A node that raises ...」の行は Exceptions 節と重複のため削除し、「at once」を Exceptions 節へ
- 決定性の条件に「変更する Python のオブジェクトを共有しない」を追加（並列の step の帰結。spec の「docs で注意する」）
- 末尾から `../topics/simulation.md` へリンク（別担当が執筆中。存在しないと docs-build が失敗するので合流時に確認）

### R1（73 → 75 行）
| 問 | 判定 | 原因 | 直したこと |
| -- | ---- | ---- | ---------- |
| Q1 | 正解 | - | - |
| Q2 | 正解 | （問題不良・軽微） | 問題文の `u` がノードかポートか曖昧と指摘された。問題文を `a -> b` に直した |
| Q3 | 正解 | - | 「the second test」をテスト名で指すようにした（生徒の指摘、埋没） |
| Q4 | 正解 | - | - |
| Q5 | 正解 | - | 例に `from mujou import NetworkError` を足した（生徒の指摘、利用者がテストを書くのに要る） |
| Q6 | 正解 | - | 共有が避けられないときは `step(dt, *nodes)` で固定の順に、を 1 文足した（逐次なので決定的。読者の次の疑問に答える） |

- 生徒の「ノードに例外を出させる仕組みが書かれていない」は見送り。例外を出すのは利用者のノードの都合で、この節の用途（例外の確かめ方）に要らない
- 「Passing every node ...」の段落が詰まっているという指摘は見送り。2 回渡しは別の項目にあり、段落は 3 文で正解率にも影響していない

### R2 汎化確認（保留問題、75 行、改稿なし）
| 問 | 判定 | 原因 | 直したこと |
| -- | ---- | ---- | ---------- |
| H1 | 正解 | - | - |
| H2 | 正解 | - | - |
| H3 | 不正解 | 誤読 | なし。答えが「2 回」で、説明は「更新は 1 回だけ」と自己矛盾。文書は「updates once」と明確 |

- 保留問題を見て文書を直すのは禁止なので、改稿していない
- 生徒の指摘の候補（未適用）:
  - 「the time of the state」を「the time the state was taken」に言い換える
  - `state` と `timer` を一緒に渡せない（`ValueError`）ことを書く

### 日本語版（docs/ja/tutorial/3-test.md、72 → 75 行、テストなし）
- 英語版の最終稿に合わせて同じ箇所を直した。文体は `docs/ja/index.md` に合わせ、既存の訳語（ノード・実行・ステップ・ワーカー）を使った
