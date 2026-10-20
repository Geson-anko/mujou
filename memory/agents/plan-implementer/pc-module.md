# mujou.pc の実装ログ（plan-implementer）

計画: `memory/agents/implementation-planner/pc-module.md`（orchestrator の決定 3 件を「確認事項」節に反映済み）。

## ステップ 1（2026-10-20）: 時定数・付随データ・損失・functional

実装: `src/mujou/_core/pc/{__init__,time_constants,carry,losses,functional}.py`、テスト: `tests/mujou/_core/pc/test_{time_constants,carry,losses,functional}.py`。公開（`src/mujou/pc/`）とノードは carry の PR の後。

### 計画からの逸脱・計画に無かった判断

- **`loss_vjp` は mask が False の行の `y` を 0 に置き換えてから `loss` に渡す**。計画は「`torch.where` で損失を選べば NaN が混ざらない」としていたが、それは前向きだけで、後ろ向きでは `where` の勾配 0 が `loss` の微分（`y` が NaN なら NaN）に掛かり `0 * NaN = NaN` が `g_in`・`g_theta` に入る。`y` を先に置き換えれば前向きも後ろ向きも有限になる。損失の選択の `where` も残す（置き換えた値の損失を足さないため）。`TestLossVjp.test_a_masked_out_target_does_not_matter` が検証する
- **`learn_` は長さを先に比べる**。`zip(strict=True)` だけだと、短い方が尽きるまでのパラメータを動かしてから `ValueError` になる。例外のときは何も動かさない
- **`carry_spec` の `target` がスカラー（`()`）** は先頭次元がバッチと違うものとして `ValueError`（`target.shape[:1] != (batch,)`）
- **`TimeConstants` は `int` も受け付ける**（`bool` は `TypeError`）。値は変換せずそのまま持つ
- **`vjp(wrt_input=False, params=())`** でも `delta` の shape を検査する（どちらの経路でも同じ契約にする）
- テストに `TestMessages`（`Sample`・`Prediction` が `DictSpec` を作れる）を足した。TypedDict の注釈が `torch.Tensor` でないと `DictSpec` が弾くので、Port に使えることの確認になる
