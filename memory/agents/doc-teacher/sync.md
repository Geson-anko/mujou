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

### R2（API 変更 74e6472 への追従）
- 事実: run(component, *components)。1 つだけならノードも根（パス ""、Port "prev"、状態の木のキー ""）。Compose(node) は "0"
- 直した: running.md 引数表と単独ノードの文、compose.md 表の後の文、concepts/networks.md パス節（R1 で足した「ノードは包む」の理由文は削除）。ja/en
- 新問題: N1 run(node) の Port パス（running.md）/ N2 run(node) と Compose(node) のパスの違い（compose.md）/ N3 run(node) の状態の木のキー（concepts/networks.md）

### R3（R2 の問題の採点）
| 問 | 判定 (ja/en) | 原因 | 直したこと |
| -- | ------------ | ---- | ---------- |
| N1 | 正解/正解 | - | - |
| N2 | 正解/正解 | - | - |
| N3 | 不正解/不正解 | 問題不良 | 問題文が、どの文書にも説明の無い状態の木の `nodes` 階層を前提にしていた（`nodes` は testing.md の例のコードにしか出てこない）。問題文から `nodes` を外した。concepts/networks.md の「パス」節は段落を 2 つに分け（Compose の規則と、1 つだけ渡したときの規則）、状態の木の文に state.md へのリンクを付けた（埋没の指摘への対応） |
- running.md の引数表は列が実際にずれていた（docs/ 用の mdformat が整列しない）。手で整列した

### R4（改題 Q3）
- ja/en とも `""` は正解、`Compose(node)` の `"0"` は推測止まり → 欠落（Compose がノードを受け付けることを networks.md が書いていない）
- 直した: 「パス」節に Compose はノードも受け、`Compose(node)` は `"0"` と 1 文。「構築と実行」の「ノード 1 つも渡せる」を「ネットワークの代わりにノードを渡してよい。1 つだけなら包まずに走らせる」に（曖昧）

### R5（状態の木の構造をフックとして追記）
- 置き場所: concepts/state.md「状態の木」の表にキー列（nodes / slots / last_update_ns / timer）を足し、取り出し例 1 文（`tree["nodes"]["l1"]`、単独ノードは `tree["nodes"][""]`）。user-guide/state.md は既に concepts へリンクしているので触らない
- 確認元: src/mujou/_core/session.py `_tree` / `_restore`、run.py `state_dict` docstring
- 但し書き（形は変わりうる・書き換えて復元は想定外）は specs/実装に根拠が無いので書かなかった
- 問題 S1: `mujou.run(net)` で走らせた `l1` の状態を木から取り出す書き方（concepts/state.md）

### R6（examples-timing: サンプルが「出力が届くまで読み直す」形になったことへの追従）
- 行範囲: running.md quickstart.py:54:68、compose.md compose.py:38:70、time.md timers.py:36:62 / 64:69
- quickstart.md: 出力例を t = 0.017 s に（実測 0.017〜0.024）。「t は実行ごとに変わる」「t は y を読んだ時刻ではなく送った時刻」。手順 3/4 を「read → 期待値まで 0.01 s ごとに読み直し、グローバル時間で 5 s で諦めて assert が失敗」に。待たずに読む話は読み直す理由の段落に統合
- running.md 手順 3、examples/index.md の quickstart、installation.md の出力例も追従

| 問 | 判定 (ja/en) | 原因 | 直したこと |
| -- | ------------ | ---- | ---------- |
| Q1 手順 4 の動きと理由、諦めたとき | 正解/正解 | - | 両者が「5 秒がどの時計か書かれていない」と指摘したので、手順 4 に「グローバル時間で」を足した |
| Q2 t は何の時刻か、0.017 と違うのは異常か | 正解/正解 | - | - |
- 見送り: グローバル時間の 0 点の定義、送る前の read の戻り値、失敗時に print が先に出る順序（事実の追従の範囲外）
