# docs/en/topics/pc-concepts.md 理解度テスト（g_in を緩和の後に送る）

## 対象
- ファイル: docs/en/topics/pc-concepts.md（あわせて docs/en/tutorial/5-learn-with-pc.md 28 行の事実だけを直した。テストの対象外）
- 読者と用途: mujou.pc（PC-ALM）を使う・自分の pc ノードを書く利用者（PyTorch に慣れた Python 開発者）が、1 回の更新の中で何をどの順に計算して送るか、credit がどの速さで前へ戻るかを正しく理解する
- 事実の出典: memory/project_pc_g_in_after_relax.md、src/mujou/pc/layer.py（on_update と docstring）、specs/pc.md「g_in を緩和の後に送る」、examples/pc_merge.py の on_update

## 評価用問題
### Q1 自分の pc ノードの on_update で、next から受け取った g_in の和 g で relax_ を呼ぶことと、vjp で g_in を作って prev へ送ることは、どちらを先にするか。理由も。
- 要点: relax_ が先／受け取った credit を同じ更新のうちに前のノードへ渡すため（vjp の delta は緩和した後の r + lam）

### Q2 pc.Layer が vjp に渡す delta は、どの時点の何から作るか。mask が False のサンプルの delta はどうなるか。
- 要点: 段 2（relax_）で動かした後の r + lam／mask が False の行は 0

### Q3 Input → Layer → Layer → Output の鎖（接続は 3）を既定の TimeConstants で作り、ManualRun.step(dt) で全ノードを 1 ステップに 1 回ずつ更新する。host が書いた入力が Output へ届き、credit が先頭の Layer まで戻ってくる往復は何ステップか。
- 要点: 2 × hops = 6 ステップ（前へ 1 ステップに 1 ノード、後ろへ 1 ステップに 1 ノード）

### Q4 Q3 の鎖を、周期の等しい自由実行で動かすと、credit の往復は約何周期か。また後ろ向きの 1 ホップ（接続 1 つ）あたりは最大何周期か。
- 要点: 約 hops = 3 周期／1 ホップあたり最大 1 周期

### Q5 learn_interval=0.024 の Layer に、書き込みから 0.03 秒経った新しい入力が初めて届いた。この更新で学ぶか。理由も。
- 要点: 学ばない／緩和に使う g がまだ前の入力のもの（次のノードがまだ新しい入力を見ていない）

### Q6 次のノードから受け取った g を、Layer はそのまま前のノードへ渡すか。渡さないなら何を送るか。
- 要点: 渡さない。g は自分の r と lam を動かす力に使う／自分の（緩和した）delta から作った g_in = -J_h^T delta を送る

### Q7 pc.Output の 1 回の更新で relax_ を呼ぶか。g_in は何から作るか。
- 要点: 呼ばない（r も lam も持たない、段 2 が無い）／functional.loss_vjp で損失の勾配を g_in として送る

### Q8 reset が立ったサンプルが初めて届いた更新で、r と lam を 0 にするのは relax_ の前か後か。
- 要点: 前（段 1）。その更新の緩和と vjp は 0 から始まる

### Q9（H1 から移した） 自作ノードで、vjp を `lambda mu: r + lam` で取って g_in を送ってから relax_ を呼ぶと、pc.Layer と比べて credit の戻り方はどう変わるか。
- 要点: 受け取った credit が前へ渡るのが 1 回の更新ぶん遅れる（g_in が緩和前の delta から作られる）

## 保留問題
### H2 学ぶ更新で、g_theta は g_in と同じ delta から取るか。VJP は何回取るか。
- 要点: 同じ（緩和した後の delta）／1 回

### H3 既定の TimeConstants で 1 回の更新は delta をどこまで動かし、その delta はいつ前のノードへ送られるか。
- 要点: -g ちょうど（行き過ぎない）／同じ更新のうちに

## ラウンド記録
### R0 改稿（297 → 297 行）
- 38 行・3 節（要点・段の表の 2 と 3 の入れ替え・注記・Output の段落・BP との違いの question）・4 節の new で学ばない理由・5 節の速さ（ManualRun.step は往復 2 × hops ステップ、自由実行は約 hops 周期）を直した
- just docs-build 通過

### R1（297 → 304 行）
| 問 | 判定 | 原因 | 直したこと |
| -- | ---- | ---- | ---------- |
| Q1〜Q8 | 正解 | — | — |
| （Q3 の確信「中」・指摘: TimeConstants は全部必須なのに「defaults」） | — | 曖昧 | 38 行の「the defaults of」を 5 節と同じ「the usual values of」にした |
| （Q2 の確信「中」・指摘: delta が 0 と g_in の行が 0 が揃わない） | — | 曖昧 | 2 節の mask の項を「delta の行を 0、だから g_in の行も 0」にした |
| （指摘: 手動の実行と自由実行の速さが読み比べにくい） | — | 埋没 | 5 節の速さの 2 文を、実行の仕方ごとの 1 ノードあたりと往復の表にした |
- 7 行増えた（2.4%）。表の枠のぶんで、2 割に届かない

### 汎化確認（保留問題、304 → 304 行）
| 問 | 判定 | 原因 | 直したこと |
| -- | ---- | ---- | ---------- |
| H1 | 不正解（「文書からは断定できない」、内容は合っていた） | 欠落 | 3 節の注記に「自作ノードでもこの順を守る。g_in を relax_ の前に送ると、受け取った credit を 1 回の更新ぶん遅れて送る」を足した。読者は自分のノードを書くので必要な情報。H1 は Q9 として評価用に移した |
| H2 | 正解 | — | — |
| H3 | 正解 | — | （指摘: exactly の前提）5 節を「An update of one period then moves ...」に直した（曖昧） |
- 指摘「tau.period と max_dt の段落が密」は、今回の変更と関係しないので直していない

### S1 の反映（304 → 304 行）
- algorithm-reviewer の S1 を反映: 「1 周期に最大 1 ノード」は向きが逆（出力側からの逐次では 1 ステップで全層に届く）。38 行・5 節の表の列・tutorial 28 行を「1 ホップあたり最大 1 周期」にした。Q4 の要点「1 ノードあたり最大 1 周期」も「1 ホップあたり最大 1 周期」と読む

### R3（Q1〜Q9、304 → 304 行）
| 問 | 判定 | 原因 | 直したこと |
| -- | ---- | ---- | ---------- |
| Q1〜Q9 | 正解 | — | — |
| （Q3 の確信「中」・指摘: 往復の終点） | — | 曖昧 | 5 節の表の列名を「the input forward to the Output, the credit back to the first Layer」にした（step(h) で層 i に 2·hops − i ステップ目、先頭の Layer に 2·hops で届く。memory/project_pc_g_in_after_relax.md） |
| （指摘: as step 2 left them） | — | 曖昧 | 「after step 2」にした |
- 直していない指摘: tau.r の行の「further」と learn_interval の例（今回の変更と関係しない節）、前向きの 1 ホップあたりのステップ数（往復 2 × hops と後ろ 1 ステップから決まるので足さない）
- 全問正解、汎化確認も済んだのでループを終えた。just docs-build は走らせていない（オーケストレーターの指示。mdformat だけ通した）

### ja の追従（docs/ja、テストなし）
- docs/ja/topics/pc-concepts.md を英語版の最終稿に合わせた（38 行・2 節の mask・3 節の要点と段の表と注記と Output と question・4 節の new の理由・5 節の速さの表）。docs/ja/tutorial/5-learn-with-pc.md 28 行を「1 ホップを最大 1 周期で戻る」に。mdformat のみ通した
