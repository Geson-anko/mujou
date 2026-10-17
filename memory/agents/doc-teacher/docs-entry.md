# docs/en/concepts.md と docs/en/getting-started/quickstart.md 理解度テスト

## 対象
- ファイル: docs/en/concepts.md（docs/concepts/{index,nodes,ports,networks,time,state}.md を 1 ページに統合）、docs/en/getting-started/quickstart.md
- 読者と用途: PyTorch が使える初めての利用者。quickstart を走らせた後、ノードを書く前に mujou の考え方（非同期・dt 比例・状態）をつかむ

## 評価用問題
### Q1 2 回続けた `recv()` が同じテンソルを返した。故障か。新しい値か知るには
- 要点: 故障ではない（値は溜まらず、同じ値を何度も読むのは正常）／`recv_with_time()` で時刻を読み前回と比べる
### Q2 層をノード A・B に分け、1 つの loss で `backward()` したい
- 要点: できない（送る値は計算グラフから切り離される）／2 層を 1 つのノードのクラスに入れ、ノード内で BP する
### Q3 負荷で `dt` が大きく Euler が不安定。`dt = min(dt, 0.01)` は
- 要点: しない（実際に経った時間を捨てる）／1 回の `on_update` の中で刻みを分割する
### Q4 時間 2 倍速でホストが mujou の時間で 2 秒入力を見せる
- 要点: `mujou.time.sleep`／`time.sleep` は実時間で待つ
### Q5 正常終了後の `run.state_dict()`。(a) 値と時刻 (b) `on_stop` で `self.w = None` がだめな理由
- 要点: (a) `on_stop` の後にノードが残した値、時刻は shutdown が終了を求めた時点（後の更新・送信があればその時刻）(b) 木は `on_stop` の後に取るので変更が木に入り、`w` が木から消える
### Q6 quickstart で 1 回読まず `wait_for` を使う理由
- 要点: `recv()` は待たず、`x` が通り抜けるまで古い入力（最初はゼロ）からの値が出る／`wait_for` は条件を満たすまで待つ

## 保留問題
### H1 既定設定の 2 つの実行。片方の状態を取るともう片方の時間は。避けるには
- 要点: 同じグローバルタイマーなので止まる／片方に専用のタイマーを持たせる
### H2 ノード A・B を Network で囲めば同期して更新するか
- 要点: しない（ネットワークは組み立てだけ）／同期させたい層は 1 つのノードのクラスにまとめる

## ラウンド記録
### R1（concepts 101 行、quickstart 43 行）
| 問 | 判定 | 原因 | 直したこと |
| -- | ---- | ---- | ---------- |
| Q1–Q4, Q6 | 正解 | - | - |
| Q5 | 不正解（(b) の理由を推測） | 欠落 | `on_stop` の変更が終了後の木に入ること、`self.w = None` で `w` が消える例を足した（ja も同じ） |
| 読みにくさ | - | 曖昧 | 表の時刻の括弧書きを文に分けた。quickstart の「So the host does not read once」を「直後に読んだ出力は古いことがある」に直した |

### R2（concepts 101 行、quickstart 43 行）
| 問 | 判定 | 原因 | 直したこと |
| -- | ---- | ---- | ---------- |
| Q1–Q6 | 6/6 正解 | - | 表の「The time」を「The timer's time」にした（何の時刻か曖昧という指摘） |
| H1, H2 | 2/2 正解 | - | 専用タイマーの渡し方・速さの変え方が無いという指摘は、ガイド（user-guide/time.md）の担当なので本文に足さない |
