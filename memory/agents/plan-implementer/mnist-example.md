# MNIST の example の実装ログ（plan-implementer）

計画: `memory/agents/implementation-planner/pc-module.md`「MNIST の example」、`specs/pc-module.md`「MNIST の example」。実装: `examples/mnist/{main,data,bp}.py`（2026-10-20）。`src/`・`tests/` は変えていない。

## 計画からの逸脱・計画に無かった判断

- **dt の監視はノードの状態に持つヒストグラム**。計画は「最初の 100 バッチで `subscribe(out.next)` の時刻の差の中央値」だったが、`subscribe` は間の送信を飛ばすので差が過大になる。`Watch(mujou.Node)` を mixin にして `WatchedLayer(Watch, pc.Layer)`・`WatchedOutput(Watch, pc.Output)` を作り、`on_update` で `dt / period` を 1/4 周期のビンに数える（`"dt_histogram"`、状態の木で読む）。100 バッチ目に `run.state_dict()` で 1 回見て（観測のための停止）、終わりに分布（50/90/99%）を出す。警告は中央値 > 1.5 周期か 99% 点 > 2.8 周期（実効 η > 0.7）
- **BP の更新は `torch.optim.SGD` を使わず `p -= lr * g` と書いた**。3.14t では `torch.optim` の `step()` が triton を読み込み、プロセスの残りで GIL が有効になる（`sys._is_gil_enabled()` が True に変わる。`foreach=False`・`fused=False` でも同じ）。BP を PC の後に回すので PC には影響しないが、順番に依存しないようにした。式は同じ SGD
- **BP の損失は `where(mask, CE, 0).sum() / B`**。端数のバッチ（0 埋め、mask False）でも PC の `learn_` の `g_theta / B` と同じ量。満杯のバッチでは平均の CE
- **テスト精度は host が持つ正解と、S 周期後に `run.read(out.next)` した予測で測る**。テストのバッチは `mask` を False にして書き、`y` も一緒に運ばれるが学習には使われない
- **発散したら止める**: 最初の予測が非有限なら `SystemExit`
- 既定の `--time-scale` は 2（下の測定）。`--workers` は 4（ノード数）。`torch.set_num_threads(1)`

## 測定（32 コア、3.14t、GIL off、float32、B=64、S=32、τh=4 ms、周期 1 ms）

time scale ごとの 1 epoch（seed 0、無負荷）。l1（784 入力）が一番遅いノード:

| scale | l1 の dt（50/90/99%、周期単位） | PC 1 epoch の実時間 | test |
| ----- | ------------------------------- | ------------------- | ---- |
| 1 | ≤1.25 / 1.25 / 1.25 | 30.4 s | 96.49% |
| 2 | ≤1.25 / 1.25 / 1.75 | 15.3 s | 96.20% |
| 3 | ≤1.75 / 1.75 / 2.5（警告） | 10.3 s | 96.00% |
| 4 | ≤2.25 / 2.25 / 3.25（警告） | 7.8 s | 96.13% |

既定（3 epoch、scale 2、seed 0）の最後の実行:

| epoch | prequential PC / BP | test PC / BP | 学習の実時間 PC / BP |
| ----- | ------------------- | ------------ | -------------------- |
| 1 | 92.04% / 91.78% | 96.20% / 96.22% | 15.3 s / 0.8 s |
| 2 | 96.93% / 96.83% | 97.14% / 96.95% | 15.2 s / 0.8 s |
| 3 | 97.83% / 97.92% | 97.18% / 97.54% | 15.2 s / 0.8 s |

- 1 バッチ: PC 16.2 ms（各ノード forward と VJP を 32 回）、BP 0.82 ms（1 スレッド）。約 20 倍
- 3 epoch 目の PC の test は実行ごとに揺れる: seed 0 で 97.35 / 97.27 / 97.18%（scale 2）、97.12%（scale 1）。seed 1: 97.59%（BP 97.45%）、seed 2: 97.45%（BP 97.54%）。BP との差は −0.4〜+0.1pt。同期シミュレーション（BP 97.65%、PC ≈97.6%）より差がやや大きいが、scale 1 でも縮まないので dt の揺らぎは原因ではない

## 負荷の下での発散（src の設計に関わる観察）

他のプロセスがコアを占めると発散する（予測が NaN、精度 9.80% = クラス 0 の割合）。再現: 26〜40 本の busy loop と並べて scale 2 で 1 epoch。無負荷では 10 回以上の実行で一度も発散していない。

- 発散の直前、全ノードがそろって約 230 ms（タイマー時間）止まり、その直後の学習で ΣΔt が 8〜9 バッチ分になった（学習率 0.8〜0.9 を 1 バッチの勾配で 1 歩）。`Output` の重みの最大絶対値が 0.21 → 6.1 に跳び、続いて u・λ が指数的に増えた
- ただし `Learner._elapsed` を 1 バッチ（32 ms）で頭打ちにしても（scratchpad で `_tick` を差し替え）、同じ負荷で発散した。そのとき l1 の dt の 90% 点が 3〜inf 周期（8 周期超が数千回）で、緩和のノード間の陽的な結合（`specs/pc-module.md`「dt が伸びたとき」の想定どおり）が主因と見る
- scale 1 でも 26 本の負荷では 2 回に 1 回発散した
- example 側の対策は、docstring で無負荷の機械で回すよう案内し、警告と発散の検出を入れるところまで。src 側の候補（未実装、提案）: ΣΔt の上限（`learn_` の 1 歩の学習率の上限）、dt が長いときに u・λ の更新を打ち切る/ノード間の結合を弱める仕組み
