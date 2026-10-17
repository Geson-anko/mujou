# docs/ja, docs/en のサンプル・API 追従 理解度テスト

## 対象
- ファイル: docs/{ja,en}/user-guide/{writing-a-node,torch,testing,building-a-network,running,compose,time}.md, docs/{ja,en}/concepts/networks.md, docs/{ja,en}/examples/index.md
- 読者と用途: ノードとネットワークを書く利用者。長い `dt` への対処、分岐の送り先の持ち方、ノード 1 つを走らせたときのパスを、サンプルの変更（厳密解の Leaky、刻み分割の torch_node、Channel を持つ Split、`mujou.run(node)`）どおりに理解して書けるようにする

## 評価用問題（ja/en 共通の論点。en は同じ問いを英語で出す）
### Q1 （writing-a-node.md）厳密解の無い系のノードで、混んで `dt` が長くなりうる。on_update をどう書くか。`dt` を上限で切り詰めてよいか
- 要点: 1 回の on_update の中で刻みを n 個に分割し（dt/n が安定な長さ以下）n 回更新する / dt を切り詰めない / 理由: 切った分の経過時間が失われ連続時間の解からずれる
### Q2 （writing-a-node.md）例の Leaky は陽的 Euler 法か。x を u へどれだけ動かすか。dt が非常に長いとき u を行き過ぎるか、理由も
- 要点: Euler ではない（厳密解） / 係数 1 - exp(-dt/tau)（u までの距離に掛ける） / 行き過ぎない: 係数はどんな dt でも 1 未満 / 入力が一定の間は厳密
### Q3 （torch.md + examples/torch_node.py）dt = 0.035 秒の on_update で、パラメータを何回更新し、1 回の刻み h は何秒か。勾配は何回計算するか
- 要点: n = ceil(0.035/0.01) = 4 / h = 0.00875 / 勾配も刻みごとに 4 回計算し直す
### Q4 （building-a-network.md + examples/branching.py）Split が x を double へ、-x を square へ send_to で送る。送り先の Channel をどう用意するか。self.next.channels を使うと何が困るか
- 要点: ネットワークで connect が返す Channel を split の属性（plus/minus）に代入する / channels はつないだ順なので connect の順を入れ替えると送り先が気づかないうちに入れ替わる
### Q5 （concepts/networks.md）ネットワークで包まずに mujou.run(node) で走らせた。ノードのパスと、Port prev のパスは何か
- 要点: ノードは "0" / Port は "0.prev"（mujou.run(Compose(node)) と同じ）

## 保留問題
### H1 （testing.md）testing_a_node の参照計算は、各ステップで a と b をどの係数・どの順で更新するか
- 要点: 係数 1 - exp(-h/tau) / a を先に更新し、その新しい a で b を更新
### H2 （examples/index.md）leaky_chain は出力を厳密解と比べるとき、経過時間をどう測るか
- 要点: run.write が返す書き込みの時刻から、read が返す出力の時刻までの差

## ラウンド記録
### R0（追従の改稿のみ）
- 行範囲 8 箇所を更新（ja/en）、ビルド後 HTML で取り込み内容を確認
- building-a-network の RoutedSplit/Router の例を削除（本例と重複）。約 30 行減

### R1
| 問 | 判定 (ja/en) | 原因 | 直したこと |
| -- | ------------ | ---- | ---------- |
| Q1 | 正解/正解 | - | - |
| Q2 | 正解/正解 | - | - |
| Q3 | 正解/正解 | - | - |
| Q4 | 正解/正解 | - | - |
| Q5 | 正解/正解 | 埋没（両生徒の指摘） | concepts/networks.md: ノード 1 つのパスの規則を「構築と実行」から「パス」節へ移し、ネットワーク 1 つ（包まず根になる）との違いを理由付きで添えた |
- 見送り: 安定な刻みの一般則、刻みの途中で u が変わる場合、Channel 属性のクラス注釈の説明（事実の追従の範囲外）
