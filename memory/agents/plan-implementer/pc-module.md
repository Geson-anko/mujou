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

## ステップ 2・3（2026-10-20）: Learner・Input・Layer・Output と公開

実装: `src/mujou/_core/pc/{learner,input,layer,output}.py`、`carry.py` に `batch_of`・`check_batch`、公開 `src/mujou/pc/{__init__,functional}.py`。テスト: `tests/mujou/_core/pc/test_{layer,output,input}.py`、`tests/mujou/test_init.py` の `TestPcPublicNames`。carry の c70801b をマージ済み（05c4b74）。

### carry の API（計画の仮定との差）

- 計画どおり（`send_carry=` / `recv_carry=`、`send(x, carry=)`、`carried()` / `carried(key)`）。差は無かった
- `Run.read` は `send_carry` を持つ Port を読めない（型でも実行時でも）。テストで `Layer.next`・`Input.next` の送った値と付随データは `Run.state_dict()["slots"]["<node>.next/out"]`（`(value, time_ns, sent, carried)`）で読む

### 計画からの逸脱・計画に無かった判断

- **`Learner` が `module`・学習するパラメータ・バッチ・入力の spec・probe も持つ**。計画の `Learner` は位相と ΣΔt だけだったが、`Layer` と `Output` の構築（型と device の検査、probe、`input_shape` の検査）が同じなので基底に寄せた。`Learner.__init__(module, *, input_shape, carry, tau, learn_interval, period, dtype, device)`。非公開なので公開 IF は計画どおり
- **`_seen_time` の初期値は `-1`**（orchestrator の K2 の指示）。最初の更新で必ず位相を始める。最初のサンプルが届く前の付随データはゼロ（`time` 0、`mask` すべて False）で、VirtualTimer の時刻 0 に書いた最初のサンプルも `time` 0 なので、時刻だけでは新しい入力と区別できない。その場合、位相はノードの最初の更新から数え、学習するかは `mask` で決まる（ゼロの付随データでは学習せず ΣΔt を戻す）。テスト `TestLearnInterval.test_a_first_input_written_at_time_zero_is_learned`。`specs/pc-module.md`「θ の更新間隔」と `Layer`・`Output` の docstring に記載
- **`Learner` の検査**: `learn_interval` が None でなく有限の > 0 でないと `ValueError`（`inf` も弾く）。`max_step` の検査は `Layer.__init__` でも行う（`relax_` に任せると最初の更新まで気付かない）
- **`Output` の `loss` の検査**は `no_grad` の下で probe の出力と `carry` の `y` のゼロで行う
- **`Input` は前のノードから返る `g` を読まない**（計画どおり）。`next` の `recv=X` は接続のためだけ
- テストの参照実装は `test_layer.py` の `Reference`（素の PyTorch、`functional` を使わない）。Jacobi のテストは途中で入力を変え、付随データの中継も 1e-12 で照合する
- 不動点のテストは `RELAX_STEPS = 3000`（tau.h = 4 ms、dt = 1 ms で u が 1e-10、λ が 1e-8 に届く）。slow の合計は約 23 秒
- `test_learner.py` は作らない（`Learner` は `Layer`・`Output` のテストで検証する）
- specs/pc-module.md の修正: `Input` の `period` が必須、`pc.Sample`・`pc.Prediction`・`loss_vjp` を MVP の表に追加、レート $1/\tau$ を別に持たないこと、`Layer` も `mask` を読むこと、最初のサンプルが届く前の扱い
