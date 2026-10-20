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

## レビューと実験への対応（2026-10-20）

依頼: algorithm-reviewer の must 1（u の形は深い ResBlock 鎖で発散）・should 1〜3・検討 1・3・4・5、code-reviewer の should-fix 1・2 と nit 1〜8、orchestrator の決定（状態を ra2 に、`max_dt`）。

### 実装した形

- `Layer` の状態は `h`・`lam`・`_mu`（前回の更新の μ）・`_moved`。付随データの `time` が変わった更新と、θ を動かした直後の更新で 1 回だけ `h <- mu + (h - mu_last)`
- `relax_(h, lam, mu, g, dt, tau)` は 1 歩の陽的 Euler（刻みの分割は削除）。τλ = ∞ なら λ に触れない
- `Learner` に `max_dt`（既定 min(τh/2, 3τλ)）。h・λ の緩和と ΣΔt に打ち切った dt を使う。`Output` も同じ
- 学習しないまま入力が切り替わったら ΣΔt を捨てる（状態 `_learned_input`）
- `Layer` の VJP の cotangent は mask が False の行を 0

### 計画からの逸脱・計画に無かった判断

- **`functional.vjp` の `delta` は Tensor か「μ から δ を作る関数」**。付け替えない更新の δ = h − μ + λ は μ が要り、μ は vjp の中で計算するため。付け替える更新では δ = h − μ_last + λ で μ が要らないが、Layer は 1 つの関数で両方を書いた。forward を 2 回計算する案（μ を先に no_grad で求める）は計算量が増えるので採らなかった
- **λ は更新後の h の r で進める（Gauss-Seidel の順）**。orchestrator の比較実験（chain.py の ra2）は更新前の r で λ を進めているが、spec の「公式実装と同じ順」と algorithm-reviewer の検討 1（max_dt の既定の安定条件）はこの順が前提なので、元の順を保った。テストの参照実装もこの順
- **学習の時刻をホストが書いた時刻（付随データの `time`）から数える（要確認）**。最初は orchestrator の指示どおり「ノードが新しい入力を見た更新から learn_interval」で実装したが、MNIST の example が無負荷でも数十〜数百バッチで発散した。`ManualRun` で 3 ノードを揺らいだ順（ジッター ±10% や更新順のシャッフル）に回すと 27〜65 バッチで決定的に再現し、固定の Jacobi 順では 1 epoch 安定。原因: 下のノードが θ を更新すると次の更新で h を付け替え、上のノードの h_in が跳ぶ。上のノードは自分の θ でないので付け替えず、跳んだ分が r に入る。Jacobi では上のノードがちょうど 1 更新先に学習するので無害だが、順序が揺らぐと跳んだ後の δ で学習する。l1 を凍結しても、l2 を凍結しても安定（両方が学習するときだけ発散）。書いた時刻から数えると経路上のノードがほぼ同時に学習し、同じシミュレーションの 4 条件すべてで 150 バッチ安定、6 条件で 1 epoch 安定。状態 `_phase` を `_learn_at`（次に学習する時刻、ns）に置き換えた。学習の時刻は実行の時間（打ち切った時間を含む）で決まる。spec の「入力が変わってから」を「入力を書いてから」と読み替えた形で、ユーザーの承認（2026-10-19）の範囲かは確認が要る
- **ΣΔt を捨てる条件は「その入力で学習（またはラベル無しで飛ばす）していないまま切り替わったとき」**。algorithm-reviewer の案（位相が learn_interval に届いていなければ）だと、学習した後に残った時間（MNIST では learn_interval の後の 8 周期）も毎バッチ捨て、学習率が変わる。切り替えの更新の dt は新しい入力の分として数える
- **最初のノードの更新（ゼロの付随データ）は新しい入力**。最初の本物の入力が来るとその時間は捨てる（`TestLearnInterval` の最初の重みが 4H → 3H）
- **時刻 0 に書いた最初のサンプルでは付け替えない**（時刻ではゼロと区別できない）。h は τh で追いつく。`TestWithoutTeacher` は、下から回す（各ノードが同じステップで送られた値を読む）テストと、r = 0 から入力を切り替えて上から回すテストに分けた。上から回し始めると、l0 の最初の付け替え（ゼロ → μ0(0)）は時刻が変わらないまま l1 に届き、l1 に r ≠ 0 が残るので、forward と厳密には一致しない
- **ResBlock 8 ノードのテストの目標は forward の出力に雑音を足したもの**。forward と無関係な乱数の目標（credit ≈ 2.7）では、h の形でもサンプルの一部が収束せず有界な振動が続いた（dt を τh/20 まで小さくしても止まない。連続時間の性質）。credit ≈ 0.06〜0.7 の 9 条件（seed 3 × 雑音 3）では h の形は収束、u の形（毎回付け替え）は 1e37〜1e45 に発散。この振動は spec の未決定事項に記録
- **`Learning`（テスト）に `state=`・`steps=`**（code-reviewer nit 5）。復元は `start_manual(..., state=)` で行う
- **docs/en,ja/concepts.md の「dt を打ち切らない」の警告を「長い dt に気を付ける」に変えた**。コアは打ち切らないが、ノード間の結合が不安定なノードは打ち切ってよい、と書いた（spec の nodes.md・execution-model.md・pc-alm-continuous-time.md・open-decisions.md も同じ趣旨で整合）
- **example の既定の `--time-scale` を 2 → 1**。2 では h の形で数百バッチ後に発散した（l1 の dt は 1.25 周期程度で max_dt に届かないので、dt ではなく、最も遅いノードが周期の大半を計算に使い読む値が古くなることが原因と見る。`ManualRun` のシミュレーションでは再現しない）。1 では 3 epoch × 2 seed で安定
- **example の警告の閾値（99% 点）を 2.8 → 2 周期（max_dt）**

### 負荷の下（未解決）

- 再現: `python busy.py N 100`（N 本の busy loop）と並べて 1 epoch。32 コア
- 16 本・スケール 1: 完走（test 96.41%）。26 本・スケール 1: 発散。26 本・スケール 0.5: 完走（96.30%）。32 本・スケール 1 / 0.5: 発散
- 機構: あるノード（l2）がおよそ 115 ms 走れない間、隣のノード（l1）はその最後の g で learn_interval ごとに学習を続け、θ が同じ向きに走る（l1 の μ の最大が 9 → 16 → 21 → 28）。dt の伸びではないので max_dt では防げない
- 試した案: 上のノードの g がその前の学習より後に届いたときだけ学習する（`recv_with_time` の時刻で判定、scratchpad でサブクラス）。32 本では発散を防げなかった。src には入れていない

### その他の確認

- `Port.send` は値を同期的にコピーする（`h` をそのまま送っても後の in-place 更新は届かない）。発散の調査で疑ったが、`h.clone()` を送っても同じように発散した
