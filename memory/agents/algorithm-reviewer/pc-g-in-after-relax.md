# algorithm-reviewer: pc-g-in-after-relax

verdict: approve（レビューのファイルは書けなかったため、orchestrator から受けた指摘を code-simplifier が写した）

- S1（確信度 高）: 「at most one node (back) in each period」は向きが逆。正しくは「1 ホップに最大 1 周期」（from_next では 1 ステップで全層に届くので「1 周期に最大 1 ノード」は偽）。逆の箇所: src/mujou/pc/chain.py:92、src/mujou/pc/time_constants.py:36、examples/cifar10/main.py:23 と :91、examples/cifar10/resnet.py:10。正しい言い回しは layer.py:55・88 の "at most one period per hop" に揃える
- N2: specs/pc.md のテスト番号の重複（14 が 2 つ）
- N3: specs/pc.md の自由実行の φ の定義（受け手の位相 − 送り手の位相）を、後ろ向きの辺で定義することを明記
- N4: specs/execution-model.md の μ の遅れの原因は「読んだ値を中継する」ことで、その言い方に統一（skill 側は orchestrator が直す）
- N5: リセット直後の更新で前のサンプルの credit を下へ送る（δ' = −g_old）が、往復の間に消えるので学習には効かない（code-reviewer の must-fix と同じ）
- N7: MNIST の BP が 97.54 → 97.53% になったのは BP の経路を変えていないので走らせ直しの揺れ、と specs/research/pc-experiments.md に一言
