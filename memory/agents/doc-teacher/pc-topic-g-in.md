# docs/en/topics/pc.md 理解度テスト（g_in を緩和の後に送る変更）

## 対象
- ファイル: docs/en/topics/pc.md
- 読者と用途: mujou.pc（PC-ALM）でネットワークを組み学習させる利用者（PyTorch に慣れた Python 開発者）。周期・見せる時間・learn_interval を決め、自分の pc ノード（Nodes of your own）を正しい順序で書く
- 事実の出典: memory/project_pc_g_in_after_relax.md、計画書 memory/agents/implementation-planner/pc-g-in-after-relax.md、src/mujou/pc/layer.py の docstring と on_update、examples/pc_merge.py の Merge.on_update、specs/pc.md の「learn_interval の目安」、specs/research/pc-experiments.md（MNIST 97.69% / BP 97.53%、prequential の差は最大 0.11 ポイント）

## 評価用問題
### Q1 Input → Layer × 3 → Output の鎖を周期 2 ms で組む。learn_interval の下限として勧められる値は何 ms か。計算も示せ
- 要点: hops = 4（Input から Output までの接続の数）
- 要点: 2 × hops × period = 16 ms より長くする

### Q2 hops = 3 の鎖で、見せる時間 S（周期の数）はどう決めるか
- 要点: 2 × hops = 6 周期（credit の最長の往復）より数周期多くする
- 要点: （例として MNIST は 16 周期。無くても可）

### Q3 自分の pc ノードの on_update で、relax_ と vjp はどちらを先に呼ぶか。理由も
- 要点: relax_ が先
- 要点: g_in がその更新で受け取った credit を運び、credit が 1 更新に 1 ホップ戻る（逆だと各ホップで 1 更新余分に待つ）

### Q4 自分の pc ノードで vjp に渡す delta は何から作るか。mask が False のサンプルはどうするか
- 要点: relax_ で動かした後の r と lam の和 r + lam
- 要点: mask が False のサンプルは 0

### Q5 自分の pc ノードで t.new が True のとき、relax_ より前に何をするか
- 要点: reset が立ったサンプルの r と lam を 0 にする
- 要点: relaxer を持つなら relaxer.reset_(act["reset"]) を呼ぶ

### Q6 mujou.run（全ノード同じ周期）で credit の往復が約 hops × period だった。learn_interval をそれより少し長い値にしてよいか
- 要点: 勧めない。2 × hops × period より長くする
- 要点: 理由: ManualRun の run.step(dt) では往復がちょうど 2 × hops ステップかかり、どちらの実行でも credit が戻るようにするため（周期が違えば約半分も保証されない、でも可）

### Q7 CIFAR-10 の例（ノード 5 個の鎖、12 周期見せ、0.9 で学ぶ）で、学ぶ時刻は credit の往復の後か。数で答えよ
- 要点: hops = 4、往復は最大 8 周期
- 要点: 学ぶのは 12 × 0.9 = 10.8 周期で、8 周期より後

### Q8 MNIST の例の 3 epoch 後のテスト精度（PC-ALM と BP）
- 要点: PC-ALM 97.69%、BP 97.53%（seed 0）

## 保留問題
### H1 Input → Layer × 2 → Output、周期 5 ms、見せる時間 40 ms。learn_interval = 30 ms は勧められる範囲か
- 要点: hops = 3、2 × 3 × 5 = 30 ms より「長く」なければならないので不可（等しい）
- 要点: 半分（20 ms）より長く 40 ms より短い条件は満たす。例えば 35 ms にする

### H2 自分のノードで vjp を relax_ の前に呼ぶと、credit の戻り方はどう変わるか
- 要点: 各ホップで 1 更新余分に待つ（遅くなる）

### H3 周期の違うノードを混ぜた mujou.run で、往復を hops × period と見積もってよいか
- 要点: よくない。約半分になるのは全ノードが同じ周期のとき。言えるのは 1 ホップに最大 1 周期、往復は最大 2 × hops × period

## ラウンド記録
### R0 改稿（245 行 → 245 行）
- 57 行（周期）・59-60 行（見せる時間・learn_interval）・68 行（往復: 最大 2×hops×period、step(dt) でちょうど、mujou.run で約半分）・71 行・147 行（CIFAR）を「1 ホップに最大 1 周期」に
- MNIST の数字を 97.69% / 97.53%、prequential の差 0.11 ポイントに
- Nodes of your own の手順 3 と 4 を入れ替え、relax_ を先にする理由を 1 文足した
- 用語を「per hop」に統一（往復の節で hops を定義）

### R1（245 行 → 245 行）
| 問 | 判定 | 原因 | 直したこと |
| -- | ---- | ---- | ---------- |
| Q1 | 正解 | - | 生徒の指摘（hops の数え方が無い）が妥当なので、「`pc.Chain` のモジュールの数」と足した（往復の項） |
| Q2 | 正解 | - | 「a few」の具体数は表の MNIST 列（16）で足りるとみて直さない |
| Q3 | 正解 | - | - |
| Q4 | 正解 | - | - |
| Q5 | 正解 | - | - |
| Q6 | 正解 | - | 推奨の強さの指摘が妥当なので「Nothing checks it」を足した |
| Q7 | 正解 | - | 「five nodes in all」を「four modules, and five nodes with the Input」に（曖昧） |
| Q8 | 正解 | - | - |
- 手順 1 の relaxer の文の曖昧さの指摘は、今回の変更範囲外で、誤答も出ていないので直さない

### 保留問題の確認（245 行 → 245 行）
| 問 | 判定 | 原因 | 直したこと |
| -- | ---- | ---- | ---------- |
| H1 | 正解 | - | hops の 2 通りの書き方が紛らわしいという指摘を受け、「in a `pc.Chain`, as many as its modules」と、定義は 1 つで Chain での数え方を添える形にした |
| H2 | 正解 | - | - |
| H3 | 不正解（文書から判断できない） | 曖昧 | 「約半分」は同じ周期のときだけと書いてあったが、それ以外で何が言えるかが無かった。「周期が違えば言えるのは上限だけ: 1 ホップは行き先のノードの約 1 周期まで待つ」を足した（出典: 計画書の credit の速さの節「メッセージは受け手が次に更新するまで待つ」「ジッタのある自由実行で言えるのは 1 ホップあたり最大 1 周期という上限だけ」） |
- 規則に従い H3 を評価用に移す（Q9）。保留問題は残り H1・H2。汎化の確認には新しい保留問題が要る
- 表の learn_interval 行が必須条件に見えるという指摘は、本文に「recommended」「Nothing checks it」があり誤答も出ていないので直さない

## 評価用問題（追加）
### Q9 周期の違うノードを混ぜた mujou.run で、credit の往復を hops × period と見積もってよいか（元 H3）
- 要点: よくない。約半分になるのは全ノードが同じ周期のとき
- 要点: 言えるのは上限だけ（1 ホップは行き先のノードの約 1 周期まで待つ）

### R2（245 行 → 245 行）
| 問 | 判定 | 原因 | 直したこと |
| -- | ---- | ---- | ---------- |
| Q9 | 正解 | - | - |
- 全問正解でループを終えた
- 未対応の指摘: 周期が混在するとき、目安 `2 * hops * period` の period がどのノードの周期か（最長か、経路の和か）が書かれていない。出典（specs/pc.md、計画書）にも決めた記述が無いので、文書には足さずに報告に回した
