# code-simplifier: pc-g-in-after-relax

code-reviewer（request-changes）と algorithm-reviewer（approve）の指摘を反映した。公開 IF は変えていない。

## 変更
- specs/pc.md
  - サンプルごとのリセット: 「δ を取る前に消すので credit を下へ送らない」を事実に直した（消した後、まだ前の入力にいる次のノードの g で同じ更新のうちに緩和し、臨界減衰では δ' = −g_old で前のサンプルの credit をもう 1 度送る。往復の間に消えるので学習には効かない）
  - 「g_in を緩和の後に送る」: 「μ は δ に依らない」の箇条を、学ぶと θ がずれて以後の μ が変わる、に直した。「リセットの直後」の箇条を足した。φ を後ろ向きの辺で定義すると明記
  - 自作のノードの手順を relax_ → vjp（g_in を送る）の順に
  - テスト番号の重複（14 が 2 つ）を 14/15/16 に振り直した
- specs/execution-model.md: `step(h)` が陽的 Euler でない理由を「μ が読んだ値（h_in）を中継する」に統一
- specs/research/pc-experiments.md: MNIST の BP 97.54 → 97.53% は走らせ直しの揺れ、と 1 行
- src/mujou/pc/layer.py: クラス docstring のリセットの記述を事実に合わせ、learn_interval の目安の段落を 88 桁で折り返し、自由実行の hops*period は上限ではないと読めるように
- src/mujou/pc/chain.py、time_constants.py、examples/cifar10/main.py（2 箇所）、examples/cifar10/resnet.py: 「1 周期に最大 1 ノード」→「1 ホップに最大 1 周期」。resnet.py は ManualRun.step(dt) に限定し、88 桁で折り返した
- tests/mujou/pc/_reference.py: self.jacobi → self._jacobi。g_sent を `list(self.g_in) if self._jacobi else self.g_in` の 1 行にし、jacobi=False でコピーを作らない

## 検証
- ruff check / format --check: 通過
- uv run pytest tests/mujou/pc -q: 250 passed, 6 skipped
- pyright: 0 errors
- just test-fast: pytest 936 passed, 63 skipped
