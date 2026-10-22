# pc-optimizers（plan-implementer の判断ログ）

計画書 `memory/agents/implementation-planner/pc-optimizers.md` のステップ 1〜5 の src・examples・specs・memory を実装した。tests は spec-test-author、docs（`docs/en/api/pc.md` の members を除く）は後の段。

## 計画の確認事項の確定（orchestrator から）

- Protocol 名は `BoundOptimizer`、`AdamW` の既定は `eps=1e-8`、`weight_decay=1e-2`

## 計画外の小さな判断

- `_BoundAdamW` は設定（`AdamW`）をそのまま持ち、フィールドを写さない。設定は frozen なので共有してよい
- `BoundOptimizer.state_dict` はモーメントのテンソルを参照のまま返す（`Node.state_dict` と同じく、木のコピーは基盤が取る）。`load_state_dict` は `copy_` で写し、`c_m` / `c_v` は `float()` で代入する
- `SGD` / `AdamW` の検査は module private の `_check_positive` / `_check_non_negative` にまとめた（メッセージは計画の表どおり）
- `Layer` は `relax_` に使うので `self._tau` を残した。`Output` は `self._tau` を使わなくなったので消した
- `Layer` の `self._batch` は `__init__` の局所変数 `batch` にし、`self._samples = math.prod(batch)` を持つ（計画どおり）。`Output` も同じく `self._samples`
- `examples/feedback_alignment.py`: 計画どおり定数名は `SGD`。モジュールの docstring の箇条（「Telling what is new」）に「`pc.SGD` を bind した束縛が 1 歩を進める」を 1 句足した。`self.w1 -= ...`（属性の付け替え）が束縛の in-place の `step_` に変わる
- `examples/pc_merge.py`: `Merge` の `vjp` に渡す `params` はそのまま残し、bind は `__init__` で `self.module.parameters()` から
- specs/pc.md の新しい節「重みの optimizer」は「時定数」の節の直後（「dt の打ち切り」の前）に置いた
- **計画の漏れ**: `benchmarks/pc_dense.py` が `TimeConstants(theta=)` を使っていて `just type` が落ちた。計画の「変更するファイル」に無かったが、`TAU` から theta を消し `OPTIMIZER = pc.SGD(tau=1.28)` を `common` に足す最小の修正をした
