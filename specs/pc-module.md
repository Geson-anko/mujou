# mujou.pc の実装計画

> 状態: 計画（2026-10-19）。`mujou.pc` はまだ無い。

PC-ALM を `mujou.Node` の上に実装するモジュール `mujou.pc` の設計と実装の順番。最初の目標は `nn.Linear` + 任意の活性化関数の 3 層ネットワークでバッチサイズ $B$ の MNIST を解き、同じネットワークの BP と性能を比べること。

式と記号は skill `pc-alm`、連続時間形は [PC-ALM の連続時間系](pc-alm-continuous-time.md)、公開元の方針は [更新手法](update-rules.md#実装と公開元) に従う。

## 一般化した定式化

記事の層 $\sigma(W_i h_{i-1})$ を、ノードが持つ任意の微分可能な関数 $f_i(\cdot;\theta_i)$ に置き換える。ノードの内側は BP、ノードの間は PC-ALM になる。

$$\mu_i = f_i(h_{\mathrm{in}};\theta_i),\qquad r_i = h_i - \mu_i,\qquad \delta_i = r_i + \lambda_i$$

ノードの内側で autograd の VJP を 1 回取り、下へ送る値と重みの勾配を同時に得る。

$$g_{\mathrm{in}} = -J_h^\top \delta_i,\qquad g_{\theta} = -J_\theta^\top \delta_i$$

```python
g_in, *g_theta = torch.autograd.grad(mu, [h_in, *params], grad_outputs=-delta)
```

$f = \sigma(Wh)$ なら $J_h^\top\delta = W^\top\varepsilon$、$J_W^\top\delta = \varepsilon h^\top$ となり記事の式に戻る。上のノードから届く値の和を $g_i$ とすると、連続時間形は次のとおり。

$$\tau_h \dot h_i = -(\delta_i + g_i),\qquad \tau_\lambda \dot\lambda_i = r_i,\qquad \tau_\theta \dot\theta_i = -g_\theta / B$$

- 受け取る $g_i$ は全体の損失の勾配ではなく、上のノードのエネルギー項の偏微分。平衡点（$r = 0$）で $\lambda_i = -\partial\ell/\partial h_i$ となり BP の credit に一致する
- `h_in` はスロットから読んだコピーなので計算グラフは葉で切れる。ノードをまたぐ `.backward()` は無い
- tanh の層 1 つと ResBlock 2 つ（各 1 ノード）を Linear + MSE の出力ノードにつなぎ同期的に回すと、不動点で重みの勾配が BP と相対誤差 $4\times10^{-15}$ で一致（2026-10-19）
- $h$ が準平衡なら、$\lambda$ の値によらず $\delta_i = J_{i+1}^\top\delta_{i+1}$（BP の漸化式）が成り立つ。$\lambda$ の役割は $r = \delta - \lambda$ を 0 に寄せ、ヤコビアンを forward の点で評価させること。$\alpha = 0$（PC）では $r = \delta \ne 0$ のまま止まり、ずれた点で評価した勾配は縮む（MNIST でノルム比 0.58、cos 0.97）

任意の $f$ への一般化と、損失を持つ出力ノードの分離は記事の範囲外にある mujou の拡張（skill `pc-alm` 7 節に追記する）。

### 符号・バッチ・損失の規約

- 下へ送る値は上のノードのエネルギーの $h$ に対する勾配。複数の相手からの値は足すだけ
- VJP はバッチの和で取り、$1/B$ は $\theta$ の勾配にだけ掛ける（2026-10-19、ユーザーの承認）。局所のエネルギーを `.mean()` で作ると $h$ の勾配まで $1/B$ 倍になるので使わない。この規約で $\lambda$ はサンプルごとの credit $-\partial\ell_b/\partial h_b$、$\theta$ の更新は平均損失の勾配降下になる
- 出力ノードの損失は**サンプルごとの値** $\ell_b$（shape `(B,)`）を返す関数とし、和はノードが取る。`nn.MSELoss` の既定（平均、½ なし）をそのまま使うと $g_{\mathrm{in}}$ が $2/B$ 倍になり、$\theta$ には $1/B$ が二重に掛かる。`pc.mse` は $\tfrac12\|y-\hat y\|^2$ を特徴の次元で和にしたもの、`pc.cross_entropy` は `reduction="none"` の交差エントロピー
- `f` はサンプルごとに独立であること。学習モードの BatchNorm のようにバッチの中で混ざる演算は使えない（LayerNorm や GroupNorm は使える）

## ノードの構成

### 状態

| ノード   | 状態                                   | 時定数                               |
| -------- | -------------------------------------- | ------------------------------------ |
| `Layer`  | $\theta_i$（`f`）、$h_i$、$\lambda_i$ | $\tau_\theta$、$\tau_h$、$\tau_\lambda$ |
| `Output` | $\theta_L$（`f_L`）                    | $\tau_\theta$                        |

ほかに版番号・学習の位相・前回の学習からの経過時間を持つ。どれも 0 次元の `torch.Tensor` にし、`Node` の状態の自動収集に乗せる（Python の数値で持つと復元後に決定性が崩れる）。$h, \lambda$ も属性に `Tensor` を代入するだけで集まるので、専用のクラスは作らない。

$\mu, r, \delta$ は状態にせず、更新のたびに今の `h_in` から計算し直す。落とし穴 2（$\sigma'$ をキャッシュで評価しない）は自動で守られる。

### メッセージと Port

メッセージは値と**版番号**（`torch.int64` の 0 次元）の組にする。版番号はサンプルの同一性を表し、ホストが決める（[バッチの切り替えと状態](#バッチの切り替えと状態)）。

| 型（仮称）  | 中身                      | 向き                 |
| ----------- | ------------------------- | -------------------- |
| `pc.Up`     | `h`、`version`            | 下 → 上（活動）      |
| `pc.Down`   | `g`、`version`            | 上 → 下（勾配）      |
| `pc.Target` | `y`、`version`            | ホスト → 出力（教師）|

| ノード   | Port     | 受け取る                 | 送る                    |
| -------- | -------- | ------------------------ | ----------------------- |
| `Layer`  | `prev`   | `Up`（$h_{\mathrm{in}}$）| `Down`（$g_{\mathrm{in}}$） |
| `Layer`  | `next`   | `Down`（相手ごと、和を取る） | `Up`（$h_i$）       |
| `Output` | `prev`   | `Up`                     | `Down`                  |
| `Output` | `target` | `Target`                 | —                       |
| `Output` | `next`   | —                        | 予測 $\mu_L$ と `version` |

入力専用のノードは置かず、ホストが最初の `Layer` の未接続の `prev` に `Up` を書く。どの `Layer` の `prev` も同じ型になる。最初の `Layer` は $g_{\mathrm{in}}$ の計算を省いてよい（読み手がいないので）。

### ResBlock などの入れ方

ノードの境界は、制約 $h_i = f_i(\cdot)$ を置いて自由変数 $h, \lambda$ を持つ場所。

- ResBlock を 1 つの `Layer` にする（既定）: $f(h) = h + g_\theta(h)$ を渡す。スキップ接続と中間の活性はノードの内側で BP される
- ノードをまたいで分ける: 合流するノードが入力ごとに Port を持ち、VJP を入力ごとに返す。分岐する側は `next` に届いた値を足す。一般形は $\nabla_{h_j}\mathcal L = \delta_j + \sum_{i \in \mathrm{children}(j)} \partial E_i/\partial h_j$

細かく分けるほど並列度と局所性が上がり、$h, \lambda$ のメモリと緩和の時間（経路上のノード数に比例）が増える。モデル全体を 1 ノードにして出力ノードをつなぐとオンラインの BP になる。

## on_update の手順

`Layer` は 1 回の `on_update(dt)` で次を行う。

1. `prev` から $h_{\mathrm{in}}$ と版、`next` から $g$ を読む。自分と版の違う $g$ は 0 とみなす
2. $\mu = f(h_{\mathrm{in}})$ を計算する
3. 版が変わった（または未初期化）なら $h \leftarrow \mu$、$\lambda \leftarrow 0$、版と学習の位相を更新し、$h$ を送って終わる（ダイナミクスは進めない）
4. **更新前の** $\delta$ で VJP を取る。学習するステップなら $g_\theta$ も取り、しないステップは $h_{\mathrm{in}}$ についてだけ取る
5. $g_{\mathrm{in}}$ を送る
6. $h, \lambda$ を `dt` だけ進める（陽的 Euler。$\lambda$ も更新前の $r$ を使う）
7. $h$ を送る
8. 学習するステップなら $\theta$ を更新する

状態の更新（6、8）は `torch.no_grad()` の下で行う。autograd の行には `# mujou: local-autograd` を付ける。

`Output` は 1、2 の後、`target` の版が `h_in` の版と一致するときだけ $\ell = \sum_b \ell_b$ を微分して $g_{\mathrm{in}}$ と（学習するステップなら）$g_\theta$ を得る。一致しなければ教師なしとして $g_{\mathrm{in}} = 0$ を送り、学習しない。予測 $\mu_L$ は毎回送る。

### 同期スケジュールとの対応

`ManualRun.step` で**上から**（`Output` → … → 最初の `Layer`）固定の `dt` で回すと、各ノードは下の $h$ を前の時刻の値で読み、上の $g$ も前の時刻の状態から作られた値を読む。$g_{\mathrm{in}}$ は更新前の $\delta$ から作るので、全体は記事の Jacobi 更新（全層が時刻 $t$ の値を使う）と一致する。下から回すと Gauss-Seidel になる。

記事は $\lambda$ の更新に更新前と更新後のどちらの $r$ を使うかを明記していない。mujou は更新前の $r$ に固定し、オラクルの参照実装も同じにする（差は $O(\eta_h\alpha)$）。

## θ の更新間隔

$h, \lambda$ は毎回の `on_update` で更新し、$\theta$ は間隔を空けて更新する（ユーザーの提案）。学習するステップでは、前回の $\theta$ の更新からの経過時間 $\Sigma\Delta t$ を重みにして

$$\theta \leftarrow \theta - \frac{\Sigma\Delta t}{\tau_\theta}\cdot\frac{g_\theta}{B}$$

とする。学習の速さ（時間あたり）は $\tau_\theta$ で決まり、間隔は $g_\theta$ を**どの時点でサンプルするか**と計算量を決める。

間隔はステップ数でなく時間 `learn_interval`（秒）で持ち、**版が変わったときと学習したときに位相を 0 に戻す**。版が変わってから `learn_interval` だけ経った時点で学習する。$\Sigma\Delta t$ は版の切り替えでは戻さない。`ManualRun` を固定の `dt` で回すと、これは「$K$ ステップに 1 回、バッチの切り替えから数えて」と同じになる。`learn_interval=None` は毎回学習する（連続時間形そのまま）。

サンプルの時点が効くのは、版が変わった直後は $\delta = 0$ で $g_\theta = 0$ になり、credit が出力から 1 ステップに 1 層ずつ下りてくるからである。時刻によらず一定の間隔で取ると、同じ $\Sigma\Delta t$ でも実効の学習率が 0 から収束値まで揺れる。

MNIST の実験（下表、relu・CE・バッチを $S = 32$ ステップ見せる）:

| 学習のタイミング                        | BP との差 | $g_{W_1}$ の cos / ノルム比 | 1 バッチの時間 |
| --------------------------------------- | --------- | --------------------------- | -------------- |
| 毎ステップ（$K = 1$）                   | +0.13pt   | 0.85 / 0.68                 | 26.6 ms        |
| $K = S/4$                               | 発散      | —                           | 16.3 ms        |
| バッチの最後（切り替えから $S$）        | −0.13pt   | 0.999 / 0.97                | 15.1 ms        |
| バッチの途中（切り替えから $S/2$）      | +0.56pt   | 0.995 / 1.18                | 15.1 ms        |

- バッチの途中で学習した結果は、$S/2$ だけ見せてバッチの最後で学習した結果と完全に一致した。効くのは切り替えからの時間だけ
- $K = 1$ と $K = S/4$ はバッチの途中で $\theta$ が動き、relu で発散することがある。原因は推定（$W_1$ 方向の曲率が入力の共分散の最大固有値に比例して大きく、$K\eta_\theta$ に上限ができる）で未確定
- 既定の推奨は「バッチの切り替えから一定時間後に 1 回」。毎ステップの学習は選べるが既定にしない

## 時定数

`pc.TimeConstants(h, lam, theta)` に秒で持つ。記事の離散パラメータから作る `TimeConstants.from_rates(period, eta_h, alpha, eta_theta)`（$\tau = P/\eta$）を置く。$\alpha = 0$ は $\tau_\lambda = \infty$ として受け付ける（内部ではレート $1/\tau$ で持つ）。

安定な範囲（MNIST、$S = 32$、同期、2 万サンプル）:

- relu: $\eta_h \le 0.5$ なら $\alpha \in [0, 1]$ のすべてで安定。$\eta_h = 0.7$ は $\alpha \le 0.1$ で発散、$\eta_h = 1$ はほぼ発散
- tanh: $\eta_h \le 0.7$ で安定。CE で $\alpha \le 0.1$ だと劣化
- 拡張ラグランジアンの $h$ についてのヘッセ行列の最大固有値は約 3 で、$\eta_h <$ 約 0.7 と整合する
- 1 つのモードを線形化すると $u'' + \eta_h c\,u' + \eta_h\alpha\,u = 0$（$c \approx 1 + \|J\|^2$）。臨界減衰は $\alpha \approx c^2\eta_h/4$ で、実測の最良帯 $\eta_h \in [0.3, 0.7]$、$\alpha \in [0.1, 0.3]$ と一致する。$\alpha$ が大きすぎると不足減衰で行き過ぎ（ノルム比 1.6〜1.8）、$\alpha = 1$ では勾配の向きも崩れる（cos 0.5〜0.9）

既定の推奨は $\eta_h = 0.3$、$\alpha = 0.1$（$\tau_h : \tau_\lambda \approx 1 : 3$）。$\eta_h = 0.5$、$\alpha = 0.2$ も同等の精度だが、実時間の実行では `dt` が周期より伸びるので余裕を取る。

### dt が伸びたとき

`dt` が周期より長いと実効の $\eta_h = \Delta t/\tau_h$ が上がり、0.7 を超えると発散する。手順 6 では $\mu$ と $g$ を固定したまま、$h, \lambda$ を `max_step`（既定 $\tau_h/2$）以下の刻みに分けて Euler で進める。$\mu, g$ が固定なので刻みの分割は要素ごとの演算だけで済み、VJP を取り直さない。

ただし $g$ は $h_i$ にも依存する（上のノードの $J^\top J$ を通じた結合）。ノードの内側で刻みを分けても、この結合は陽的なまま残り、無条件には安定しない。ノードの周期は $\tau_h$ の 1/3 以下を目安にする。$(h, \lambda)$ の局所系を要素ごとに厳密に積分する案（2×2 の行列指数）は、同じ理由で無条件の安定は得られないので後回しにする。

## バッチの切り替えと状態

i.i.d. のバッチを順に見せるとき、切り替えで $h, \lambda$ をどう扱うかを MNIST で比べた（BP との差 [pt]、$\eta_h = 0.5$、$\alpha = 0.2$、バッチの最後で学習）。

| $S$ | (a) リセット（relu-ce / relu-mse / tanh-ce / tanh-mse） | (b) $\lambda$ だけ 0、(c) 持ち越し |
| --- | ------------------------------------------------------- | ---------------------------------- |
| 1   | −20.2 / −15.8 / −9.4 / −6.7                             | どれも約 11%（tanh-mse は NaN）    |
| 4   | −0.7 / −0.7 / −0.7 / −1.8                               | 同上                               |
| 8   | +0.4 / +0.2 / +0.2 / +0.1                               | relu は NaN、tanh-ce −3.1          |
| 32  | −0.1 / 0.0 / −0.1 / 0.0                                 | relu-ce は NaN、他は −0.6〜−1.1    |
| 128 | ±0.05                                                   | ようやく BP と同等                 |

- リセット（$h \leftarrow$ forward の値、$\lambda \leftarrow 0$）だけがまともに動く。$S = 1, 2$ では $W_1$ の勾配が厳密に 0（credit が 1 ステップに 1 層ずつ下りるため）で、$S$ は層数より大きくなければならない
- 持ち越すと、前のサンプルの $h$ と新しい $\mu$ の差（活動と同じ大きさ）が credit よりずっと大きく、$\theta$ に 10 倍規模の誤った勾配が入る
- 時間的に相関したストリーム（ロボットの各環境のサンプルが続く場合など）で持ち越しが効くかは未検証

そこで、リセットを**版番号**で表す。

- ホストは入力の `Up` と教師の `Target` に版番号を付けて書く。新しいサンプルなら版を変え、同じサンプルが続く（相関したストリームで状態を持ち越したい）なら版を変えない。方針をノードの引数にせず、データを書く側が決める
- `Layer` は `prev` の版が自分の版と違えば、手順 3 で $h \leftarrow \mu$、$\lambda \leftarrow 0$ に戻し、新しい版を付けて $h$ を送る。下から上へ波として伝わり、層数回の更新で forward での初期化が再現される。初期化（スロットの初期値は版 0、ノードの初期の版は −1）も同じ経路で行う
- $g$ にも版を付け、自分と版の違う $g$ は 0 とみなす。版を $h$ だけに付けると、上のノードが新しい版を見るまで前のサンプルの $g$ が新しいサンプルに注入される
- `Output` は `target` と `h_in` の版が一致するときだけ教師ありとして扱う。教師なしのステップ（ラベルの無いサンプル、テスト）は、ホストが版を変えた `Up` だけを書けばよい

各ノードは隣から届いた版を見て自分の状態だけを戻すので、グローバルなバリアではなく局所性・非同期性を保つ。学習モードと推論モードの区別も持ち込まない（教師があれば学習し、無ければ推論する）。

## 機能

### MVP（MNIST に必要なもの）

| 公開名（仮称）                    | 内容                                                                                       |
| --------------------------------- | ------------------------------------------------------------------------------------------ |
| `pc.Layer`                        | 任意の `nn.Module` を包むノード。$h, \lambda$ を持つ。出力の shape は構築時に `f` を 1 回通して決める |
| `pc.Output`                       | $f_L$ とサンプルごとの損失を持つノード                                                     |
| `pc.TimeConstants`                | $\tau_h, \tau_\lambda, \tau_\theta$（秒）と `from_rates`                                   |
| `pc.Up` / `pc.Down` / `pc.Target` | メッセージの `TypedDict` と、shape から `DictSpec` を作る関数                               |
| `pc.mse` / `pc.cross_entropy`     | サンプルごとの損失                                                                         |
| `pc.functional`                   | 状態を持たない計算: `vjp`、`relax_`（刻みの分割つき Euler）、`reset_`（$h \leftarrow \mu, \lambda \leftarrow 0$）、`learn_` |

`pc.functional` は、多入力のノードなどを利用者が自分で書くときの部品で、`Layer` と `Output` もこれで書く。重みの初期化は利用者が `module` を作るときに行う。

```python
import mujou.pc as pc

B = 64
tau = pc.TimeConstants.from_rates(period=1e-3, eta_h=0.3, alpha=0.1, eta_theta=...)
l1 = pc.Layer(nn.Sequential(nn.Linear(784, 256), nn.ReLU()), input_shape=(B, 784),
              tau=tau, learn_interval=24e-3, period=1e-3)
l2 = pc.Layer(nn.Sequential(nn.Linear(256, 256), nn.ReLU()), input_shape=(B, 256), ...)
out = pc.Output(nn.Linear(256, 10), loss=pc.cross_entropy, input_shape=(B, 256),
                target_spec=TensorSpec((B,), torch.int64), ...)
l1.next.connect(l2.prev)
l2.next.connect(out.prev)
```

### 後で足すもの

- `pc.Chain(*layers, output)`: 直列につないで `prev`・`target`・`next` を公開するネットワーク
- 多入力の `Layer`（ノードをまたぐスキップ接続、合流）
- 監視用の Port（ノードごとの $\tfrac12\|\delta\|^2$、$\|r\|$）
- 重みの更新に `torch.optim`（Adam など）を差し込む口。学習率は $\Sigma\Delta t/\tau_\theta$ から作る
- dual leak、$(h, \lambda)$ の局所系の厳密積分
- `torch.compile`（3.14t の多数のスレッドで使えるかを確かめてから）

## MNIST の example

1 本のスクリプトで、mujou で PC-ALM の 3 層を学習し、同じ構成の BP と比べる。

- **データ**: <https://ossci-datasets.s3.amazonaws.com/mnist/> から idx 形式をダウンロードしてキャッシュし、numpy で読む（torchvision に依存しない）。画素を標準化
- **PC-ALM**: 上の 3 ノードを `mujou.run` で実時間に非同期で回す。周期 $P = 1$ ms、各バッチを $S = 32$ 周期（32 ms）見せ、`learn_interval` は 24 ms。ホストはバッチごとに版を変えて `Up` と `Target` を書く
- **BP**: 同じ初期値・同じバッチ順で素の PyTorch の SGD。1 バッチあたりの学習率を $\Sigma\Delta t/\tau_\theta$ と同じ値（0.1）にそろえる
- **測るもの**:
  - prequential 精度: 版を変えた直後、学習前の予測の精度。継続学習の自然な指標
  - テスト精度: テストのバッチを教師なし（`Up` だけ）で見せ、`Output` の予測で測る。推論モードを持たないことの実演になる
  - 1 epoch あたりの実時間
- **既定値**: relu、CE、$\eta_h = 0.3$、$\alpha = 0.1$、$\tau_\theta = 10 \times 32$ ms、3 epoch

同期のシミュレーション（2026-10-19、60k × 3 epoch、3 seed）では BP との差が seed のばらつき（約 0.1pt）に収まった:

| 条件                                     | relu-ce        | relu-mse | tanh-ce |
| ---------------------------------------- | -------------- | -------- | ------- |
| BP                                       | 97.65 ± 0.04   | 97.20    | 97.44   |
| $S = 32$, $\eta_h = 0.3$, $\alpha = 0.1$ | 97.77          | 97.28    | 97.48   |
| $S = 32$, $\eta_h = 0.5$, $\alpha = 0.2$ | 97.67          | 97.14    | 97.39   |

計算時間は BP の約 17 倍（同期で 15 ms/バッチ）。小さな行列では autograd のオーバーヘッドが大きい。実時間で回すと 1 epoch（938 バッチ）が約 30 秒になる。

## テスト

`tests/mujou/_core/pc/` に実装をミラーして置き、float64 と `mujou.testing.run` で決定的に回す。損失が下がったかでは判定しない。

1. `functional.vjp` の $g_{\mathrm{in}}, g_\theta$ が、$E_i = \tfrac12\|r_i+\lambda_i\|^2$ と $\sum_b \ell_b$ の有限差分と一致する
2. 上から回す同期 step が、純粋な PyTorch で書いた参照実装の Jacobi 更新と ~1e-12 で一致する。$f = \sigma(Wh)$ では skill `pc-alm` 3 節の式とも一致する
3. $\alpha = 0$（$\tau_\lambda = \infty$）で $\lambda \equiv 0$、PC と一致し、不動点で $h$ が forward の値からずれる
4. 不動点で $\lambda_i$ がネットワーク全体を autograd した $-\partial\ell/\partial h_i$ と一致し、$\theta$ の 1 回の更新が SGD と一致する（線形と tanh、Linear と ResBlock）
5. スカラーの線形の例で $\lambda = w_2(y - \hat y)$
6. 上から・下から・固定 seed のランダム順で回しても同じ不動点に収束する
7. 版が変わると、層数回の step で $h$ が forward の値に厳密に戻り、版の違う $g$ を使わない
8. 教師が無いとき、平衡で $h = \mu$、$\lambda = 0$、$g_\theta = 0$
9. `learn_interval` を固定の `dt` で回すと、$K$ ステップごとに $\Sigma\Delta t$ の重みで更新される。刻みの分割が細かい Euler と一致する
10. `Run.state_dict` で取得した状態から復元すると、同じスケジュールで同じ値になる（版・位相・$\Sigma\Delta t$ を含む）

## 実装の順番

1. `pc.functional`・損失・`TimeConstants`・メッセージの型と、テスト 1・2（参照実装との一致）
2. `pc.Layer`・`pc.Output` と、テスト 3〜10
3. MNIST の example、docs（英語と日本語）の使い方のページ、skill `pc-alm` 7 節への一般化と実験結果の追記
4. code-reviewer と algorithm-reviewer のレビュー

## 未決定事項

- 公開名（`Layer`・`Output`・`Up`・`Down`・`Target`・`TimeConstants`）
- MNIST の example の置き場所。`examples/` の規約（ネットワーク不要、60 秒以内、テストが実行する）に合わない
- 時間的に相関したストリームで状態を持ち越したときの振る舞い
- `learn_interval` が版の切り替えに揃わない（相関したストリームで版を変えない）ときの学習のサンプリング。不偏にするなら、各ステップを確率 $p$ で選び重み $\Delta t/p$ を掛ける案がある
- 非同期の実行（`dt` の揺らぎ、読む値の古さ）で同期と同じ精度が出るか。example で測る
