# algorithm-reviewer: mujou.pc（e0b8774・8e5d1f4・003ddf1）

> 2026-10-20。対象は `src/mujou/_core/pc/`・`src/mujou/pc/`・`tests/mujou/_core/pc/`・`specs/pc-module.md`。carry のマージ分は除く。

## verdict

request-changes（確信度: 高。must-1 の数値事実は実ノードと素の PyTorch の両方で再現した。直し方はユーザーの設計判断が要る）

更新式の符号・係数・順序は spec と skill `pc-alm` に 1 項ずつ一致する。ノードをまたぐ計算グラフ・グローバル同期・更新手法の基盤への焼き付けは無い。オラクルも理論から導いてある。問題は、spec が「未決定」とした u の形の力学の安定性で、深い鎖（ResBlock 6 ノード以上）では連続時間の時点で発散する。

## must

1. **u の形の力学は、深い鎖で連続時間の時点で不安定になる。** 記事の h の形は安定なままである
   - 実ノード（`pc.Layer` × 8、`ResBlock(8)` の既定の初期化、Linear + mse、dt = τh/4、τλ = 2τh、上から同期）: λ が 4000 ステップで 1e40 まで発散する。12 ノードでも同じ。tanh MLP × 8 と恒等 × 8 は収束する
   - 素の PyTorch で同じ鎖を u の形と h の形で回した: ResBlock 4〜5 ノードではどちらも同じ不動点に収束する。6 ノード以上は u の形だけが発散し（η_h = 0.25 で 1e12〜1e32）、h の形は収束する。η_h = 0.05 や 0.01 でも u の形は発散するので、離散化や Jacobi の遅れが原因ではない
   - スカラー線形鎖の連続時間の固有値（w = 1、w_L = 1）: u の形は D = 8 で max Re = +0.089、h の形は −0.068。w = 1.5 なら u の形は D = 4 で不安定になる
   - 原因: u の形では h_j = μ_j(h_{j−1}) + u_j なので、u_i が上のすべての h を即座に動かす。その結果、損失のヘッセ行列が順伝播のヤコビアンの積を通して u_i に返ってくる（ループゲインは ‖Π J‖²）。h の形の結合は隣接層の J^T J だけ。ヤコビアンが 1 に近い ResBlock で効く
   - 現状の範囲: MVP の 784→256→256→10（Layer 2 つ）は安定。ただし docstring と spec は任意の module・ResBlock・深さを謳い、skill は深さ 1000 の residual を前提にしている
   - 代案:
     (a) 状態は h の形で持ち、付随データの `time` が変わったときだけ h ← μ_new + (h − μ_old) と付け替える（u の形の「入力の切り替えに即座に追従する」利点はこの付け替えで得られる。入力が一定の間は記事の勾配流そのものになる）
     (b) u の形を残すなら、深さとヤコビアンのノルムに対する安定条件を spec に書き、ResBlock を何ノードも並べる使い方を docs から外し、6 ノード以上の ResBlock 鎖が不動点に収束するテスト（今は落ちる）を xfail で置く
   - どちらにするかはユーザーの判断。example と docs（実装の順番 5）の前に決める

## should

1. **入力が `learn_interval` より速く変わると `_elapsed` が際限なく溜まり、次の学習 1 回にまとめて掛かる。** `Input → Output`、`learn_interval = 3H`、2 ステップごとに新しい入力を書いた。この場合、29 ステップの間 1 回も学習せず、入力が止まった最初の学習が 33H の重み（通常の 11 倍）になった（`_tick` は入力の切り替えで `_phase` だけを戻す）。ストリームでは学習率が無制限に跳ね上がり、θ が発散しうる
   - 代案: 学習しないまま入力が切り替わったら、ラベルの無いときと同じく ΣΔt を捨てる（`_tick` の `time != _seen_time` の枝で、位相が `learn_interval` に届いていなければ `_elapsed = 0`）。または `_elapsed` を `learn_interval`（None なら dt）で頭打ちにする。spec の「ΣΔt は入力の切り替えでは戻さない」もあわせて直す
2. **`Layer` は mask が False の行の δ も g_θ に入れる。** `Output` は mask の無いサンプルを損失から外し、`Layer` もすべて False なら学習を飛ばす。ところが一部だけ False の行では、前の（ラベルのあった）入力から持ち越した u + λ が `vjp` の cotangent にそのまま入る。τh = 4 ms・τλ = 8 ms の局所モードの減衰率は 1/8 ms⁻¹ なので、`learn_interval` = 24 ms の時点でも約 5% 残り、下の層ほど多く残る。半教師ありのバッチでは、目標 0 の credit に雑音が乗る
   - 代案: `Layer.on_update` で cotangent を `torch.where(mask[:, None…], delta, 0)` にする。g_in も同じ行が 0 になるが、その行の不動点はもともと λ = 0 なので不動点は変わらない（緩和が速くなるだけ）。「すべて False なら飛ばす」を行ごとに一般化したものになり、`Output` の扱いとも揃う
3. **`Layer` の docstring の「Read `module`, `u` and `lam` to watch the node」。** `mujou.run` で自由に回している間に、ホストのスレッドからこれらを読むと、ワーカーの in-place 更新（`u.sub_`・`p.sub_`）と競合し、書きかけの値を読みうる。`Run.state_dict()`（停止して読む）か、止めた後に読むよう書き換える

## 検討

1. **`max_step` の既定は τh だけで決まる。** 局所の 2 変数系（u を進めてから、進めた u で λ）の陽的 Euler が安定な条件は、a = s/τh、b = s/τλ として 0 < a < 2 かつ ab < 4 − 2a。既定の a = 0.5 なら τλ > τh/12 が要る。`TimeConstants` で τλ < τh/12 を弾くか、`max_step` の既定を min(τh/2, 3τλ) にする（a ≤ 0.5 のとき b ≤ 3 なら安定）
2. **dt が伸びたときの余裕はモデルに依存する。** スカラー線形の 2 層鎖を上から Jacobi で回すと、刻みを分割しても η_h = dt/τh が w = 1 なら 1.0 で、w = 1.5 なら 0.5 で発散する。spec の「周期の 2.8 倍まで」は MNIST での実測で、一般の保証ではないと注記する。θ も `elapsed/τθ` に上限が無い（should-1 と同じ根）
3. **ノードの分け方を変えても結果が同じことのテスト**（skill `locality-and-async`）は無い。不動点で BP と一致するテストから間接に従うが、「2 つの Linear+tanh を 1 つの Layer にした鎖」と「2 つの Layer にした鎖」で 1 回の θ 更新が一致するテストは安く書ける
4. `TestLearnInterval` の `layer_update` は、実装の `vjp`・`learn_` をオラクルに使っている。両者は別のテストで有限差分と明示式に照合済みなので実害は無いが、時刻の検証に絞ったテストだと docstring に書いておくとよい
5. `relax_` で τλ = ∞ のとき `lam.add_(u, alpha=0.0)` は、u が inf なら NaN を作る（λ を「そのまま残す」という docstring に反する）。`if math.isfinite(tau.lam)` で飛ばせば厳密になる
6. 同じ ns に 2 回書くと（`VirtualTimer` を進めずに書く）、新しい入力として検出されない。`time` の代わりに書き込みの通し番号を運べば避けられる。実時間では起きにくい

## 検証結果

- `uv run pytest tests/mujou/_core/pc tests/mujou/test_init.py -q -m "slow or not slow"`: 100 passed（22 s）
- `uv run pyright src/mujou/_core/pc src/mujou/pc`: 0 errors
- `just run` は実行していない（format が作業ツリー全体を書き換えうるので、同じ worktree の examples/mnist/ の作業を壊さないため）
- 数値実験のスクリプトは worktree の外（scratchpad）に置いた: スカラー線形鎖の固有値、実ノードの深い鎖、u の形と h の形の素の PyTorch の比較、`_elapsed` の蓄積

## 照合メモ

- `vjp`: E = ½‖h − f(h_in) + λ‖²（h は固定）。∂E/∂h_in = −J_h^T δ、∂E/∂θ = −J_θ^T δ。`grad_outputs=-delta` で一致する。f = σ(Wh) なら −J_W^T δ = −ε h^T なので、θ −= rate·g_θ は W += rate·ε h^T（skill 3 節）
- `loss_vjp`: g_in = ∂ℓ/∂h_in = −W_L^T e_L（mse、線形の head）。skill の ∇_{h_{L−1}} の第 2 項と同符号
- 不動点: u = 0、λ = −g。最上層では λ = −∂ℓ/∂h = W_L^T e_L（skill 4 節）。テストは autograd の credit と記事の漸化式の両方で照合している
- 1/B: `learn_` の rate = elapsed/τθ/B だけに掛かる。g_in と λ には掛からない。1/B を g_in に掛けると λ = credit/B になり、不動点のテストで検出される
- 順序: VJP は更新前の δ、u → λ（更新後の u）、h = μ + u_new を送り、θ は更新前の δ の g_θ を ΣΔt（今回の dt を含む）で動かす。spec の手順 1〜7 と一致する
- α = 0: τλ = ∞ → s/τλ = 0。テストでは λ ≡ 0 で、PC のエネルギーの h 勾配が 0（u = −g）。オラクルとして正しい
- Jacobi: 上から回すと、各ノードは前の時刻の h と、前の時刻の状態から作った g を読む（Output は g_in を学習の前に計算して送る）。`Reference` は functional を使わない独立の実装だが、u の形・順序の設計判断は同じものを書き写している。数理のオラクルは不動点・SGD・PC のテストが担っている
- 局所性: `# mujou: local-autograd` の 2 行は、どちらも detach した葉 `h_in` と自分の `params` だけを微分する。`.backward()` は無い。`torch.enable_grad`・`no_grad` は `with` に閉じている。送る値（μ.detach() + u、g_in）は grad_fn を持たない
- 焼き付け: λ・u・位相は `mujou._core.pc` に閉じている。carry・`register_state` は汎用で、PC 固有の量を持たない。グローバルなバリアやモードも無い（位相は付随データの `time` の変化で、ノードごとに決まる）
