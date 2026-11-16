# code-reviewer: pc-g-in-after-relax

verdict: request-changes（レビューのファイルは書けなかったため、orchestrator から受けた指摘を code-simplifier が写した）

- must-fix: specs/pc.md:125（サンプルごとのリセット）「δ を取る前に消すので、その更新で前のサンプルの credit を下へ送らない」は新しい順序と逆。新しい順序では、リセットで r と λ を 0 にした後、まだ前の入力にいる次のノードの g で同じ更新のうちに緩和し、臨界減衰では δ' = −g_old で前のサンプルの credit を下へ送る（以前は 1 更新遅れて同じ値を送っていた）。往復の間に消えるので学習には効かない（algorithm-reviewer N5）。事実どおりに直し、specs/pc.md の新しい節「g_in を緩和の後に送る」にもこの点を書く
- should-fix: specs/pc.md:375 の自作ノードの手順が古い順序（vjp → relax_）。relax_ → vjp に
- should-fix: specs/pc.md:409-410 のテスト番号の重複（14 が 2 つ）を振り直す
- should-fix: src/mujou/pc/layer.py:92（104 桁）と examples/cifar10/resnet.py:12（116 桁）を 88 桁で折り返す
- should-fix: specs/pc.md:117「前向きのメッセージは以前とビット単位で同じ」→「1 回の更新の中で μ は δ に依らない（学ばない間は前向きのメッセージが以前と同じ。学ぶと θ がずれるので以後の μ は変わる）」
- nit: examples/cifar10/resnet.py:10-11「exactly one under a stepped run」→ `ManualRun.step(dt)` に限る。layer.py:89-92 が自由実行なら hops*period で足りると読めないように（目安は 2*hops*period）。layer.py:57-61 の「no credit of an unrelated sample is left」を must-fix と同じ事実に合わせる。tests/mujou/pc/_reference.py:240 は jacobi=False のとき g_sent を作らない、self.jacobi を _jacobi に
