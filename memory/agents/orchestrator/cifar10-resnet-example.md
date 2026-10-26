# cifar10-resnet-example（solo-dev-cycle の記録）

2026-10-23〜24。依頼: 「example に CIFAR-10 + ResNet を追加。GPU で実行」「ドキュメントにも追記 /yomiyasu」「AdamW でやるべきでは」。worktree `feature/2026-10-23/cifar10-resnet-example`。

## 計画

- `examples/cifar10/`（main / data / resnet / bp / metrics）を `examples/mnist/` に倣って作る。src/ は変えない
- 段階 2（テスト）は省略: example はテストで実行しない（`examples/README.md`）。代わりに scratch で平衡点のオラクル（学習なしで緩和した lam と BP の -dl/dh）を確かめた

## 測ったことと判断

- ブロックごとに 11 ノード: 1 更新 ~15 ms。原因は境界の device<->host のコピー（B=128 で 8 MB/メッセージ、`cudaMemcpyAsync` が CPU 時間の 6 割）。ブロック単体の forward は 0.14 ms
- ステージごとに 4 ノード: 初期化が 1x1 conv + GN のショートカットだと stage の σ(J)≈8 で緩和が発散（lam が勾配の 600 倍、cos≈0）。ブロックを恒等写像から始め（最後の GN の γ=0）、option A のショートカットにすると初期 σ=1.00、オラクル通過（cos 1.000）
- しかし BP で 3 epoch 学習すると stage の σ が 5〜9 に育ち、4 ノードでは再び発散。ブロック単体でも 3〜6。陽的 Euler（Δt=τ_h/4）の安定条件 1+σ² < 8 → σ < 2.65 を超える
- head を含む合成だけ小さい: J(stage3∘head)≈1.3、J(stage2∘stage3∘head)≈1.8（学習後）。→ 2 ノード（stem+stage1+stage2 / stage3+head）に決定
- 正規化: GroupNorm(8 群) は BP でも遅い（3 epoch 46%）。BatchNorm は速いがバッチ統計で標本が結びつくので不採用。GN の 2 チャネル/群を採用
- optimizer: momentum なし SGD は GN の ResNet で不安定（lr 0.1〜2 で 10〜45% を行き来）。ユーザーの提案どおり AdamW に。BP にも同じ `pc.AdamW` を bind し `elapsed=shown` で step_。lr は B=64 で 8e-3 が最良（3 epoch 60%）
- 手動実行（遅れなし）でも lam=2τ_h, S=32 では PC 28.6% / BP 37.0%（400 バッチ）。S=64 で 37.7%、lam=τ_h, S=32 で 37.75%。→ `tau.lam = tau.h`
- 速度: 2 ノード・B=64 で time_scale 0.4（dt 中央値 ≤1.5 周期）。3 epoch 約 4 分。結果 prequential PC 56.96% / BP 56.42%、test 55.69% / 57.86%（GPU は非決定的で ±1 pt）

## 却下した案

- ブロック/ステージごとのノード（σ の安定条件、コピーのコスト）
- 周期を τ_h/16 などに短くする（更新回数が増え、コピー律速でさらに遅い。未測定）
- BatchNorm（サンプル間の結合）、SGD（不安定）

## 自己レビュー

- `--blocks` / `--width` を外した（要求外）。resnet() は 2 ノードのモジュールを返す形にし、切れ目の理由を resnet.py の docstring に置いた
- docs の数値は最終版のコードの実行に合わせた。ja は yomiyasu で推敲（ラベルを名前にそろえ、「〜ため。」の切り出しと括弧の手順を直した）

## レビュー対応（code-reviewer: approve、algorithm-reviewer: 承認。must-fix なし）

- 「更新は約 3 ms、大半がコピー」は 4 ノード・B=128 の測定の流用だった。2 ノード・B=64 で測り直すと、コピー（cudaMemcpyAsync）は更新の CPU 時間の 1/4〜1/3
- 倍率 0.4 では layer0 の dt が中央値 1.5 周期・99% 点 3.5 周期で、max_dt を超えていた。0.3 で 99% 点 2 周期、0.25 で 1.5 周期。既定を 0.3 に下げた（3 epoch 約 5 分）
- 安定条件: 「約 2.6」は遅れのないときの境界。dt=max_dt や 1 周期古い値で 1.2〜1.8 に下がる（algorithm-reviewer が線形の pc.Chain で測定）。次が Output なら CE の曲率（≤1/2）が σ² に掛かる。docs と resnet.py に書いた
- 10 epoch: J(stage3∘head) は 2.2（9 epoch で 2.65）、ステージは 12 まで。倍率 0.4 の 10 epoch の実行は発散せず、prequential PC 74.32% / BP 73.16%、test 71.97% / 74.34%（epoch ごとに数 pt 揺れる）
- 最終の既定値の実行: prequential PC 55.00% / BP 55.76%、test 55.98% / 58.23%
- nit: data.py の float32 変換の重複を除いた、bp.py の不要な model.to を消した。shown の式の重複は docs の snippet のために残した

## main の device 上の受け渡しへの追従（2026-10-26）

ユーザー: 「ノード間で GPU 上でやり取りできるようになりました。ノード分割の構造など修正」（/merge-main）。

- main（PR #107 ほか）を merge。CUDA のノード間のメッセージは device に置いたまま。HtoD / DtoH は 0 µs に（DtoH 3 µs は予測をホストが読む分）
- ノードの切り方は 2 ノードのまま。理由は緩和の安定条件（次のノードの σ）で、転送の仕組みとは無関係。ステージ単位では遅延のない手動実行でも発散する（前回の測定）。分割を細かくするには src の pc の変更（陰的・前処理つきの緩和、ノードごとの周期と max_dt）が要る → ユーザーに確認
- 変えたこと: layer0 の 1 更新は B=64 で 3.60 ms、B=128 で 3.69 ms、B=256 で 5.28 ms（手動実行で単体）。CPU 側が律速なので B=128 に。lr は 1.6e-2（B=128 で 1e-2 より良い）。倍率 0.3 で layer0 の dt の 99% 点 2 周期（0.4 で 2.75、0.6 で 3）
- 1 epoch 84.6 s → 42.7 s。3 epoch 約 3 分、10 epoch 約 9 分
- 結果（seed 0）: 3 epoch 目 prequential PC 55.11% / BP 54.45%、test 57.82% / 59.02%。10 epoch 目 74.00% / 73.27%、73.12% / 72.61%
- J(stage3∘head)（lr 1.6e-2、B=128 の BP）: 3 epoch で 1.23、10 epoch で 2.02。stage3 単体は 11 まで
- docs: threads-and-devices.md の「メッセージはホストのメモリ」の段落を削除（main の説明と矛盾）。pc.md の GPU の項を書き直し

## /goal: ResBlock 単位のノード + 時間倍率 1.0（2026-10-26）

ユーザーの /goal: 「Batch サイズを 1 にしてもよく、時間スケール 1.0 になるように調整、ノードは Resblock レベルで分割し、それでも学習できるよう新しい feature の実装（緩和の手法設定など）も考慮して試行。BP と比較する際は BP 側の学習条件を合わせること」。

測定（scratch: `blocks.py` 手動・遅延なし、`async_blocks.py` 非同期、`equil_blocks.py` 平衡のオラクル。B=128、lr 1.6e-2、S=64、lam=τ_h）:

- ノードは stem+block1、block2..9、head（Output）の 10 個。1 更新は約 1.1〜1.4 ms で B=1 でも変わらない（CPU 律速）。倍率 1.0 で追いつく周期は約 5 ms（τ_h=20 ms、11 ワーカー）。τ_h=12 ms（周期 3 ms）は dt の中央値 1.75 周期で追いつかない
- 手動（遅延なし）固定刻み: 3 epoch で PC test 53.45% / BP 53.65%。非同期・倍率 1.0・固定刻み: 3 epoch で 47.7%、10 epoch で prequential 51.4% / test 50.7%（BP は 72.5% / 76.0%）。dt は 99% 点 1.25 周期で追いついているので、遅れではなく緩和の精度の問題
- 平衡のオラクル（凍結した重みで λ と BP の credit の cos / 大きさの比）: BP で 391 バッチ学習した重みでは固定刻みの cos が 0.0〜0.2（比 25〜120 倍）で λ が発散。PC（非同期 3 epoch）で学習した重みでも、固定刻みは前半のノード（stem〜block5）で cos 0.5〜0.68、比 1.8〜2.0
- 次のノードの J^T J の最大固有値（サンプルごと、391 バッチ BP 後）: block2 77、block3 31、block4 6、…、block8 12.6、block9 10.1、stem+block1 714。固定刻みで安定なのは 1+c < 8
- 試作 AdaptiveStep（次のノードの曲率 c をべき乗法で見積もって前のノードへ送り、h と λ の刻みを dt·min(1, margin·τ_h/(dt(1+c))) にサンプルごとに縮める）: BP の重みでも PC の重みでも cos 0.86〜0.97（64〜128 ステップ）、256 で比 1.0。margin 1.5 は振動（1.0 を採用）。every=4 は every=1 と同じ
- ただし学習では遅くなる。手動 S=64 の 293 バッチで 33.6%（固定刻み 39.9%）。非同期（τ_h=20 ms、倍率 1.0、every=4）は 600 バッチ付近から崩れて 3 epoch 27%。見積もりが極端に大きい行（c が 1e4）がある（GroupNorm の分散が小さい群の局所的な利得）。更新が重くなって dt も遅れた（99% 点 3.5 周期）

結論（ユーザー: 「BP の性能に勝たなくてもよい。戦っている場所が違う」→ BP との差は報告にとどめる）:

- 新しい feature（AdaptiveStep: 次のノードの曲率で刻みをサンプルごとに縮める）は実装して試し、却下して src と tests を戻した。凍結した重みでは credit が正しくなったが、学習では縮めるほど悪化（600 バッチ test: 固定 46.5%、下限 0.25 で 42.2%、0.1 で 27.9%、下限なし 19.6%、BP 47.8%）。縮めた行が前のバッチの λ を長く引きずる。非同期でも崩れた。詳細は `project_pc_jacobian_stability.md`
- 固定刻みのまま、ブロックごとの 11 ノードを S=128、τ_h=20 ms、倍率 1.0、ワーカー 11 本で動かすと学べる。S=64 は 10 ホップの遅れで実効の緩和時間が足りず BP より約 10 ポイント低い（10 epoch で頭打ち 51%）。ノードの 1 更新は B=1 でも 128 でも約 1.2 ms なので、B=128 のまま（B=1 は 1 epoch が 128 倍遅い）
- 例を更新: `resnet()` が 10 モジュール（stem+block1、block2〜9、head）、`main.py` の既定を steps 128、tau-h 20e-3、time-scale 1.0、workers 11 に。結果（seed 0、main.py の既定）: 3 epoch 目 prequential PC 50.72% / BP 54.45%、test 54.18% / 59.02%。10 epoch 目 68.73% / 73.27%、68.95% / 72.61%。1 epoch は PC 251 s（実時間）、BP 2.8 s。3 epoch 約 14 分、10 epoch 約 45 分
- docs（en / ja の pc.md、examples.md）を更新。ja は yomiyasu（run 20261026T163945Z-51101be3、verify 済み）。落とし穴の項を「発散する」から「正確にならない」に直し、効いたもの・効かなかったものを書いた
