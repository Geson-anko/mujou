# mujou.pc の実装計画

> 状態: 計画（2026-10-19）。`mujou.pc` はまだ無い。

PC-ALM を `mujou.Node` の上に実装するモジュール `mujou.pc` の設計と実装の順番。最初の目標は `nn.Linear` + 任意の活性化関数の 3 層ネットワークでバッチサイズ $B$ の MNIST を解き、同じネットワークの BP と性能を比べること。

式と記号は skill `pc-alm`、連続時間形は [PC-ALM の連続時間系](pc-alm-continuous-time.md)、公開元の方針は [更新手法](update-rules.md#実装と公開元) に従う。数値は 2026-10-19 の同期シミュレーション（素の PyTorch、784→256→256→10、$B = 64$、float32）で測った。

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
- tanh の層 1 つと ResBlock 2 つ（各 1 ノード）を Linear + MSE の出力ノードにつなぎ同期的に回すと、不動点で重みの勾配が BP と相対誤差 $4\times10^{-15}$ で一致
- $h$ が準平衡なら、$\lambda$ の値によらず $\delta_i = J_{i+1}^\top\delta_{i+1}$（BP の漸化式）が成り立つ。$\lambda$ の役割は $r = \delta - \lambda$ を 0 に寄せ、ヤコビアンを forward の点で評価させること。$\alpha = 0$（PC）では $r = \delta \ne 0$ のまま止まり、ずれた点で評価した勾配は縮む（ノルム比 0.58、cos 0.97）

任意の $f$ への一般化、損失を持つ出力ノードの分離、次節の状態の持ち方は記事の範囲外にある mujou の拡張（skill `pc-alm` 7 節に追記する）。

### 状態はずれ $u$ で持つ

自由変数を $h_i$ そのものでなく、予測からのずれ $u_i = h_i - \mu_i$（$= r_i$）で持ち、送るときに $h_i = \mu_i + u_i$ を組み立てる。

$$\tau_h \dot u_i = -(u_i + \lambda_i + g_i),\qquad \tau_\lambda \dot\lambda_i = u_i$$

- 不動点（$u = 0$、$\lambda = -g$）は $h$ で持つ場合と同じ
- 違いは $h$ が入力の変化に即座に追従すること（$\dot h = \dot\mu + \dot u$）。教師が無ければ $h$ は forward pass そのもので、入力が届いてから出力までの遅れは経路上のノードの更新 1 回ずつで済む。$h$ で持つと、入力が変わるたびに各層が $\tau_h$ をかけて追いつく
- 入力が不連続に変わったときにリセットが要らなくなる（[入力の切り替えと状態](#入力の切り替えと状態)）
- 入力が一定でも下の層の $h$ は緩和中に動くので、過渡は記事の力学と一致しない。比べられるのは不動点だけで、記事の離散更新との一致はオラクルにできない

### 符号・バッチ・損失の規約

- 下へ送る値は上のノードのエネルギーの $h$ に対する勾配。複数の相手からの値は足すだけ
- VJP はバッチの和で取り、$1/B$ は $\theta$ の勾配にだけ掛ける（2026-10-19、ユーザーの承認）。局所のエネルギーを `.mean()` で作ると $h$ の勾配まで $1/B$ 倍になるので使わない。この規約で $\lambda$ はサンプルごとの credit $-\partial\ell_b/\partial h_b$、$\theta$ の更新は平均損失の勾配降下になる
- 出力ノードの損失は**サンプルごとの値** $\ell_b$（shape `(B,)`）を返す関数とし、和はノードが取る。`nn.MSELoss` の既定（平均、½ なし）をそのまま使うと $g_{\mathrm{in}}$ が $2/B$ 倍になり、$\theta$ には $1/B$ が二重に掛かる。`pc.mse` は $\tfrac12\|y-\hat y\|^2$ を特徴の次元で和にしたもの、`pc.cross_entropy` は `reduction="none"` の交差エントロピー
- `f` はサンプルごとに独立であること。学習モードの BatchNorm のようにバッチの中で混ざる演算は使えない（LayerNorm や GroupNorm は使える）

## ノードの構成

### 状態

| ノード   | 状態                                   | 時定数                                  |
| -------- | -------------------------------------- | --------------------------------------- |
| `Layer`  | $\theta_i$（`f`）、$u_i$、$\lambda_i$ | $\tau_\theta$、$\tau_h$、$\tau_\lambda$ |
| `Output` | $\theta_L$（`f_L`）                    | $\tau_\theta$                           |
| `Input`  | なし                                   | —                                       |

ほかに、学習の位相・前回の学習からの経過時間・最後に見た入力の時刻を Python の数値で持ち、`Node.register_state`（[コアの変更](#コアの変更-noderegister_state)）で状態の木に入れる。$u, \lambda$ は属性に `Tensor` を代入するだけで集まるので、専用のクラスは作らない。

$\mu, \delta$ は状態にせず、更新のたびに今の `h_in` から計算し直す。落とし穴 2（$\sigma'$ をキャッシュで評価しない）は自動で守られる。

### メッセージと Port

ターゲットは入力に**付随させて**、活動と同じ経路で運ぶ（2026-10-20、ユーザーの決定）。付随データはコアの carry（[コアの変更: carry](#コアの変更-carry)）で、活動と同じメッセージに載せ、途中のノードはゼロコピーで中継する。

付随データ `pc.Carried`（`TypedDict`、spec は `pc.carry_spec(target, batch)` で作る）:

| フィールド | 中身                                                         |
| ---------- | ------------------------------------------------------------ |
| `y`        | ターゲット（spec は利用者が決める。クラス番号なら `(B,)` の int64） |
| `mask`     | `(B,)` の bool。ラベルのあるサンプル                          |
| `time`     | `()` の int64。ホストが入力を書いた時刻（ns）                 |

| ノード   | Port   | 受け取る                              | 送る                                              |
| -------- | ------ | ------------------------------------- | ------------------------------------------------- |
| `Input`  | `prev` | $\{x, y, \mathrm{mask}\}$（ホストが書く） | —                                             |
| `Input`  | `next` | $g$（捨てる）                         | $x$ + 付随データ（書いた時刻を `time` に入れる）   |
| `Layer`  | `prev` | $h_{\mathrm{in}}$ + 付随データ        | $g_{\mathrm{in}}$                                 |
| `Layer`  | `next` | $g$（相手ごと、和を取る）             | $h_i$ + 付随データ（`prev` から中継）             |
| `Output` | `prev` | $h_{\mathrm{in}}$ + 付随データ        | $g_{\mathrm{in}}$                                 |
| `Output` | `next` | —                                     | $\{\hat y, y, \mathrm{mask}\}$                    |

- `Output` は `prev` から届いた $h_{\mathrm{in}}$ と、同じメッセージの $y$・`mask` を組む。$y$ は $x$ と同じ経路を同じ遅れで進むので、組み違いが構造的に起きない。時刻の照合も、ホストが $x$ と $y$ を同時に書く必要も無い
- ホストが伝播より速く次のサンプルを書いても組み違えない（時刻で揃える方式は、`target` のスロットが常に新しいサンプルで上書きされ、一致する瞬間が来なくなる）
- 教師なしのサンプルは `mask` を False にする。損失は $\sum_b \mathrm{mask}_b\,\ell_b$ で、1 つのバッチにラベルのあるサンプルと無いサンプルが混ざってよい
- `Layer` は付随データを使わず中継するだけ。読むのは学習の位相のための `time` だけ
- `Output` の `next` は予測と $y$・`mask` を普通のメッセージで送り、ホストが prequential 精度を測れるようにする（ホストの読み書きは carry に対応させない）

`Input` は、ホストが書いた $\{x, y, \mathrm{mask}\}$ を活動と付随データに分けて送るだけのノード。これでどの `Layer` の `prev` も同じ型になる。`Input` の次の `Layer` は $g_{\mathrm{in}}$ の計算を省いてよい（読み手が捨てるので）。

### ResBlock などの入れ方

ノードの境界は、制約 $h_i = f_i(\cdot)$ を置いて自由変数 $u, \lambda$ を持つ場所。

- ResBlock を 1 つの `Layer` にする（既定）: $f(h) = h + g_\theta(h)$ を渡す。スキップ接続と中間の活性はノードの内側で BP される
- ノードをまたいで分ける: 合流するノードが入力ごとに Port を持ち、VJP を入力ごとに返す。分岐する側は `next` に届いた値を足す。一般形は $\nabla_{h_j}\mathcal L = \delta_j + \sum_{i \in \mathrm{children}(j)} \partial E_i/\partial h_j$。付随データは入力のうちの 1 つから中継する

細かく分けるほど並列度と局所性が上がり、$u, \lambda$ のメモリと緩和の時間（経路上のノード数に比例）が増える。モデル全体を 1 ノードにして出力ノードをつなぐとオンラインの BP になる。

## on_update の手順

`Layer` は 1 回の `on_update(dt)` で次を行う。

1. `prev` から $h_{\mathrm{in}}$ と付随データの `time`、`next` から $g$ を読む
2. `time` が最後に見た値から変わったら、それを覚え、学習の位相を 0 に戻す（$u, \lambda$ はそのまま）
3. $\mu = f(h_{\mathrm{in}})$ を計算し、**更新前の** $\delta = u + \lambda$ で VJP を取る。学習するステップなら $g_\theta$ も取り、しないステップは $h_{\mathrm{in}}$ についてだけ取る
4. $g_{\mathrm{in}}$ を送る
5. $u$ を進め、続けて**更新後の** $u$ で $\lambda$ を進める（陽的 Euler、公式実装と同じ順）
6. $h = \mu + u$ を、`prev` で読んだ付随データとともに送る（中継）
7. 学習するステップなら $\theta$ を更新する

状態の更新（5、7）は `torch.no_grad()` の下で行う。autograd の行には `# mujou: local-autograd` を付ける。

`Output` は 1〜3 で $\ell = \sum_b \mathrm{mask}_b\,\ell_b(\hat y_b, y_b)$ を微分して $g_{\mathrm{in}}$ と（学習するステップなら）$g_\theta$ を得る。`mask` がすべて False なら $g_{\mathrm{in}} = 0$ で、学習しない。

ホストは $\{x, y, \mathrm{mask}\}$ を `Input` の `prev` に書く。教師なしのサンプル（ラベルが無いとき、テスト）は `mask` を False にする。学習モードと推論モードの区別は持ち込まない。

## 入力の切り替えと状態

### リセットとは何か、なぜ要ったか

$h, \lambda$ はサンプルごとの状態で、$h_i$ は今のサンプルに対する層 $i$ の活動、$\lambda_i$ はそのサンプルに対する credit を表す。i.i.d. のバッチを順に見せると、入力が切り替わった瞬間に前のサンプルの $h$ が残り、新しい予測 $\mu$ との差（活動そのものと同じ大きさ）が予測誤差として扱われる。それが credit よりずっと大きいので、$\theta$ に誤った勾配が入る。

記事の公式実装は、ミニバッチごとに $h$ を forward pass の値に、$\lambda$ を 0 に初期化してこれを避けている。これがリセットである。$h$ で状態を持つ場合の比較（BP との差 [pt]、$\eta_h = 0.5$、$\alpha = 0.2$、バッチの最後で学習）:

| $S$ | (a) リセット（relu-ce / relu-mse / tanh-ce / tanh-mse） | (b) $\lambda$ だけ 0、(c) 持ち越し |
| --- | ------------------------------------------------------- | ---------------------------------- |
| 1   | −20.2 / −15.8 / −9.4 / −6.7                             | どれも約 11%（tanh-mse は NaN）    |
| 4   | −0.7 / −0.7 / −0.7 / −1.8                               | 同上                               |
| 8   | +0.4 / +0.2 / +0.2 / +0.1                               | relu は NaN、tanh-ce −3.1          |
| 32  | −0.1 / 0.0 / −0.1 / 0.0                                 | relu-ce は NaN、他は −0.6〜−1.1    |
| 128 | ±0.05                                                   | ようやく BP と同等                 |

$S$ は 1 バッチを見せるステップ数。$S \le 2$ では $W_1$ の勾配が厳密に 0 になる（credit が 1 ステップに 1 層ずつ下りるため）。

### ずれ $u$ で持てばリセットは要らない

ずれ $u$ で持つと、入力が変わっても $h = \mu + u$ が即座に新しい予測に乗り、持ち越すのは credit 程度の大きさの $u, \lambda$ だけになる。リセットせずに持ち越した結果（ce、$\eta_h = 0.5$、$\alpha = 0.2$、バッチの最後で学習）:

| 条件                       | relu                | tanh               |
| -------------------------- | ------------------- | ------------------ |
| BP                         | 97.65               | 97.44              |
| (a) $h$ で持ちリセット、$S = 32$ | 97.67         | 97.39              |
| $u$ で持ち持ち越す、$S = 32$ | **97.60**（−0.05） | **97.44**（0.00）  |
| $u$ で持ち持ち越す、$S = 16$ | 34.6〜90.4（崩壊） | 96.9（−0.6）       |

（60k × 3 epoch × 3 seed。$S = 32$ では $g_{W_1}$ が BP と cos 1.000、ノルム比 1.00）

$S = 16$ で崩れた原因は $u, \lambda$ の持ち越しではない。$u$ だけ 0 に戻しても、$\lambda$ だけ 0 に戻しても結果はほぼ変わらなかった。原因は**入力とラベルの時刻のずれ**である。新しい $x$ は経路上のノードの更新を 1 回ずつ経て出力に届くが、ラベルは出力にすぐ届く。その間、出力は古い入力から作った $h$ を新しいラベルと組んで微分し、別のサンプルのラベルに対する勾配を流す。切り替え直後の $\delta_2$ は新しい credit の 3〜5 倍まで膨らんだ。

ラベルを伝播の遅れの分だけ遅らせると、この行き過ぎは消えた（$S = 16$ の relu が崩壊から −1.9pt、毎ステップ学習の relu が −9.9 から −0.87pt、tanh が −1.7 から −0.19pt）。

そこで次のように決める。

- 状態は $u, \lambda$ で持ち、入力が変わっても持ち越す。リセットの機構は持たない
- ターゲットは入力に付随させ、活動と同じ経路で運ぶ（[メッセージと Port](#メッセージと-port)）。伝播の遅れの分だけラベルを遅らせることが、遅れの段数を数えずに実現できる
- 古い credit が新しい入力のヤコビアンを通るずれは残る。$S$ の下限はリセットより厳しい（リセットは $S \ge 8$、持ち越しは $S \ge 32$）

ターゲットを付随させたうえでの $S$ の下限と精度は、実装後に測り直す。

## θ の更新間隔

$u, \lambda$ は毎回の `on_update` で更新し、$\theta$ は間隔を空けて更新する。学習するステップでは前回の $\theta$ の更新からの経過時間 $\Sigma\Delta t$ を重みにして

$$\theta \leftarrow \theta - \frac{\Sigma\Delta t}{\tau_\theta}\cdot\frac{g_\theta}{B}$$

とする。時間あたりの学習の速さは $\tau_\theta$ で決まり、間隔は $g_\theta$ を**どの時点でサンプルするか**と計算量を決める。

間隔は時間 `learn_interval`（秒）で持つ（2026-10-19、ユーザーの承認）。位相は付随データの `time` が変わったときと学習したときに 0 に戻し、入力が変わってから `learn_interval` だけ経った時点で学習する。$\Sigma\Delta t$ は入力の切り替えでは戻さない。`ManualRun` を固定の `dt` で回すと「入力の切り替えから数えて $K$ ステップごと」と同じになる。`learn_interval=None` は毎回学習する（連続時間形そのまま）。

サンプルの時点が効くのは、入力が変わった直後の $\delta$ が新しいサンプルの credit になっていないからである。MNIST での比較（$h$ で持ちリセット、relu-ce、$S = 32$）:

| 学習のタイミング                   | BP との差 | $g_{W_1}$ の cos / ノルム比 | 1 バッチの計算時間 |
| ---------------------------------- | --------- | --------------------------- | ------------------ |
| 毎ステップ（$K = 1$）              | +0.13pt   | 0.85 / 0.68                 | 26.6 ms            |
| $K = S/4$                          | 発散      | —                           | 16.3 ms            |
| バッチの最後（切り替えから $S$）   | −0.13pt   | 0.999 / 0.97                | 15.1 ms            |
| バッチの途中（切り替えから $S/2$） | +0.56pt   | 0.995 / 1.18                | 15.1 ms            |

- 効くのは切り替えからの時間だけで、途中で学習した結果は $S/2$ だけ見せて最後に学習した結果と完全に一致した
- 切り替えと無関係な位相で一定間隔ごとに学習すると、$u$ で持つ場合も relu で −2〜−19pt と崩れた
- 既定は「入力が変わってから `learn_interval` 後に 1 回」。入力がフレームごとに変わり続けるストリームでは `learn_interval=None` を使う

## 時定数

`pc.TimeConstants(h, lam, theta)` に秒で持つ。$\alpha = 0$ は $\tau_\lambda = \infty$ として受け付ける（内部ではレート $1/\tau$ で持つ）。

ノードの周期の既定は $\tau_h / 4$（$\eta_h = 0.25$）にする。利用者は時定数だけを決めればよく、周期を明示すればそちらを使う。

安定な範囲（$h$ で持ちリセット、$S = 32$）:

- relu: $\eta_h \le 0.5$ なら $\alpha \in [0, 1]$ のすべてで安定。$\eta_h = 0.7$ は $\alpha \le 0.1$ で発散、$\eta_h = 1$ はほぼ発散
- tanh: $\eta_h \le 0.7$ で安定。CE で $\alpha \le 0.1$ だと劣化
- 1 つのモードを線形化すると $u'' + \eta_h c\,u' + \eta_h\alpha\,u = 0$（$c \approx 1 + \|J\|^2$）。臨界減衰は $\alpha \approx c^2\eta_h/4$ で、実測の最良帯 $\eta_h \in [0.3, 0.7]$、$\alpha \in [0.1, 0.3]$ と一致する。$\alpha$ が大きすぎると行き過ぎ（ノルム比 1.6〜1.8）、$\alpha = 1$ では勾配の向きも崩れる（cos 0.5〜0.9）

記事の公式実装（SakanaAI/pc-alm）は $\eta_h = 1/\lambda_{\max}$（約 0.2〜0.25）、$\alpha = 1$、$T = 64$ か $2L$、Adam を使う。$\eta_h$ は周期 $\tau_h/4$ の既定とほぼ同じ。$\alpha = 1$ は上の結果では大きすぎるが、公式の実験は幅 32 の深い residual MLP で、曲率 $c$ が違う。example の既定は $\tau_\lambda = 2\tau_h$（$\alpha = 0.125$）とし、実装後に確かめる。

### dt が伸びたとき

実時間の実行では `dt` が周期より長くなることがあり、実効の $\eta_h = \Delta t/\tau_h$ が上がって 0.7 を超えると発散する。周期を $\tau_h/4$ にしておけば、`dt` が周期の 2.8 倍まで伸びても上限に届かない。

手順 5 では、$\mu$ と $g$ を固定したまま $u, \lambda$ を `max_step`（既定 $\tau_h/2$）以下の刻みに分けて進める。刻みの分割は要素ごとの演算だけで、VJP を取り直さない。ただし $g$ は $h_i$ にも依存する（上のノードの $J^\top J$ を通じた結合）ので、ノード間の結合は陽的なまま残り、刻みの分割では安定化できない。

## コアの変更: Node.register_state

`Node` の状態の自動収集は今 `torch.Tensor` と `nn.Module` だけを集める。入力の時刻・学習の位相・経過時間のような数値を状態に入れるには、0 次元の Tensor にするか `state_dict` を override するしかない。PC のノードに限らず使えるよう、コアに `Node.register_state(name)` を足す。

- `self.register_state("name")` で、属性 `name` の値（任意の値。`int`・`float`・`str`・`None` とそのコンテナを想定）を既定の `state_dict` に入れる
- 既定の `load_state_dict` は、その名前の属性に木の値を代入する（その場の更新ではなく差し替え）
- Tensor と `nn.Module` の自動収集はそのまま
- `torch.load(weights_only=True)` で読めるのは基本的な型だけなので、docstring でそう案内する

仕様は [ノード](nodes.md#状態の自動収集) を更新する。

## コアの変更: carry

付随データを活動と同じメッセージに載せ、途中のノードがゼロコピーで中継できるようにする（2026-10-20、ユーザーの決定）。PC に限らず、サンプル ID・マスク・時刻など、経路に沿って運びたいデータに使える。仕様は [ネットワーク](network.md#付随データ) と [メッセージ](messages.md#付随データ) にある（実装済み、2026-10-20）。

付随データを別の Port（別のスロット）に分けない。読み手が 2 つのスロットを別々の瞬間に読むと、活動と付随データが別のサンプルの組になりうるからである。1 つのメッセージに入れておけば、読み手は常に同じサンプルの組を読む。

確定した API:

- Port の宣言で、受け取る向きを `recv_carry=spec`（相手 1 つの Port だけ）、送る向きを `send_carry=` で持つ。`send_carry` に spec を渡すと付随データを作る側で、`send(x, carry=...)` で値を渡す（コピー）。同じノードの Port を渡すと中継する側で、その Port で**最後に `recv` したメッセージ**の付随データをそのまま添える。活動の計算に使った入力と付随データが同じスナップショットになる
- `connect` は付随データの spec も向きごとに照合する。中継元が同じノードの Port であることは構築で確かめる
- `recv` は付随データを torch へ移さない。`carried()` / `carried(key)` で読み、そのときだけコピーする
- コアの `Message` はバッファを `Vec<Arc<Buffer>>` で持ち、中継はバッファの参照の複製だけにする。最後の参照が消えたら元の書き手のプールへ戻る
- 付随データはスロットの値として状態の木に入り、各 Port が最後に受け取った付随データも木の `"carries"` に入って復元される
- ホストの `Run.write` / `Run.read` は付随データを持つ Port を断る（`ValueError`。必要になったら足す）

PC のノードでの使い方:

```python
carry = pc.carry_spec(TensorSpec((B,), torch.int64), batch=B)
# pc.Input: ホストが書いた {x, y, mask} を分けて送る（付随データを作る側）
self.next = NextPort(send=TensorSpec((B, 784)), recv=..., send_carry=carry)
# pc.Layer / pc.Output: 受け取り、Layer は次へ中継する
self.prev = PrevPort(send=..., recv=TensorSpec((B, 784)), recv_carry=carry)
self.next = NextPort(send=TensorSpec((B, 256)), recv=..., send_carry=self.prev)
```

## 機能

### MVP（MNIST に必要なもの）

| 公開名                        | 内容                                                                                         |
| ----------------------------- | -------------------------------------------------------------------------------------------- |
| `pc.Input`                    | ホストが書いた $\{x, y, \mathrm{mask}\}$ を、活動と付随データに分けて送るノード               |
| `pc.Layer`                    | 任意の `nn.Module` を包むノード。$u, \lambda$ を持ち、付随データを中継する。出力の shape は構築時に `f` を 1 回通して決める |
| `pc.Output`                   | $f_L$ とサンプルごとの損失を持つノード                                                       |
| `pc.TimeConstants`            | $\tau_h, \tau_\lambda, \tau_\theta$（秒）                                                    |
| `pc.Carried` / `pc.carry_spec` | 付随データの `TypedDict` と、ターゲットの spec とバッチサイズから `DictSpec` を作る関数      |
| `pc.mse` / `pc.cross_entropy` | サンプルごとの損失                                                                           |
| `pc.functional`               | 状態を持たない計算: `vjp`、`relax_`（刻みの分割つき Euler）、`learn_`                         |

`pc.functional` は、多入力のノードなどを利用者が自分で書くときの部品で、`Layer` と `Output` もこれで書く。重みの初期化は利用者が `module` を作るときに行う。

```python
import mujou.pc as pc

B = 64
tau = pc.TimeConstants(h=4e-3, lam=8e-3, theta=...)  # 周期の既定は tau.h / 4 = 1 ms
carry = pc.carry_spec(TensorSpec((B,), torch.int64), batch=B)
inp = pc.Input((B, 784), carry=carry)
l1 = pc.Layer(nn.Sequential(nn.Linear(784, 256), nn.ReLU()), input_shape=(B, 784),
              carry=carry, tau=tau, learn_interval=24e-3)
l2 = pc.Layer(nn.Sequential(nn.Linear(256, 256), nn.ReLU()), input_shape=(B, 256), ...)
out = pc.Output(nn.Linear(256, 10), loss=pc.cross_entropy, input_shape=(B, 256),
                carry=carry, ...)
inp.next.connect(l1.prev)
l1.next.connect(l2.prev)
l2.next.connect(out.prev)
```

### 後で足すもの

- `pc.Chain(input, *layers, output)`: 直列につないで `prev`・`next` を公開するネットワーク
- 多入力の `Layer`（ノードをまたぐスキップ接続、合流）
- 監視用の Port（ノードごとの $\tfrac12\|\delta\|^2$、$\|u\|$）
- 重みの更新に `torch.optim`（Adam など）を差し込む口。学習率は $\Sigma\Delta t/\tau_\theta$ から作る
- dual leak
- `torch.compile`（3.14t の多数のスレッドで使えるかを確かめてから）

## MNIST の example

`examples/mnist/` に置く。example は利用者への手引きで、動作確認のためのものではないので、テストでは実行しない（[実装の順番](#実装の順番) の 0）。

- **データ**: <https://ossci-datasets.s3.amazonaws.com/mnist/> から idx 形式をダウンロードしてキャッシュし、numpy で読む（torchvision に依存しない）。画素を標準化
- **PC-ALM**: 上の 4 ノードを `mujou.run` で非同期に回す。ホストは各バッチの $\{x, y, \mathrm{mask}\}$ を `Input` に書き、$S = 32$ 周期見せる
- **BP**: 同じ初期値・同じバッチ順で素の PyTorch の SGD。1 バッチあたりの学習率を $\Sigma\Delta t/\tau_\theta$ と同じ値（0.1）にそろえる
- **速さの比べ方**: 実時間の周期のまま回すと待ち時間が入り、BP と比べられない。timer の時間のスケールを上げ、ノードが計算の限界の速さで回るようにする。上げすぎると `dt` が周期より伸びて不安定になるので、ノードが追いつける上限を選ぶ（`dt` を監視する）
- **測るもの**:
  - prequential 精度: `Output` の `next` から届く、学習前の予測とそのサンプルの $y$ から測る。継続学習の自然な指標
  - テスト精度: テストのバッチを `mask` を False にして見せ、`Output` の予測で測る。推論モードを持たないことの実演
  - 同じコア数での、ある精度に達するまでの実時間と、1 サンプルあたりの forward と VJP の回数
- **既定値**: relu、CE、$\tau_\lambda = 2\tau_h$、周期 $\tau_h/4$、$\tau_\theta$ は 1 バッチの学習率が 0.1 になる値、3 epoch

計算量の目安: 同期のシミュレーションを 1 スレッドで回すと、1 バッチが BP の 0.9 ms に対して 15 ms（約 17 倍）だった。実時間の待ちは含まない純粋な計算の差で、1 バッチを $S = 32$ ステップ緩和し、各ステップで全ノードが forward と VJP を 1 回ずつ取るため。mujou ではノードが別々のワーカーで同時に計算するので、3 ノードなら最大でおよそ 3 倍縮む。

## テスト

`tests/mujou/_core/pc/` に実装をミラーして置き、float64 と `mujou.testing.run` で決定的に回す。損失が下がったかでは判定しない。

1. `functional.vjp` の $g_{\mathrm{in}}, g_\theta$ が、$E_i = \tfrac12\|u_i+\lambda_i\|^2$ と $\sum_b \ell_b$ の有限差分と一致する
2. 上から（`Output` → … → `Input`）回す同期 step が、純粋な PyTorch で書いた $u$ の形の参照実装の Jacobi 更新と ~1e-12 で一致する。上から回すと、各ノードは下の $h$ を前の時刻の値で読み、上の $g$ も前の時刻の状態から作られた値を読む
3. 入力を固定して回した不動点で、$\lambda_i$ がネットワーク全体を autograd した $-\partial\ell/\partial h_i$ と一致し、$\theta$ の 1 回の更新が SGD と一致する（線形と tanh、Linear と ResBlock）。$f = \sigma(Wh)$ では記事の式（skill `pc-alm` 3 節）の不動点とも一致する
4. $\alpha = 0$（$\tau_\lambda = \infty$）で $\lambda \equiv 0$、不動点で $h$ が forward の値からずれる（PC）
5. スカラーの線形の例で $\lambda = w_2(y - \hat y)$
6. 上から・下から・固定 seed のランダム順で回しても同じ不動点に収束する
7. 教師が無いとき、$h$ は経路の分の更新で forward の値になり、平衡で $u = 0$、$\lambda = 0$、$g_\theta = 0$
8. `Output` は常に、受け取った活動と同じメッセージの $y$ を使う。`mask` が False のサンプルは損失に入らない
9. `learn_interval` を固定の `dt` で回すと、入力の切り替えから $K$ ステップごとに $\Sigma\Delta t$ の重みで更新される。刻みの分割が細かい Euler と一致する
10. `Run.state_dict` で取得した状態から復元すると、同じスケジュールで同じ値になる（付随データ・位相・$\Sigma\Delta t$ を含む）

## 実装の順番

0. example をテストから外す: `tests/test_examples.py` を削除し、`examples/README.md`（60 秒・ネットワーク不要の規約）、CLAUDE.md、`memory/project_docs_site.md` を直す（chore、別 PR）
1. `Node.register_state` と、[ノード](nodes.md#状態の自動収集) の仕様とテスト（別 PR）
2. コアの carry（Rust の `Message`、bindings、`Port`、スタブ）と、[ネットワーク](network.md)・[メッセージ](messages.md) の仕様（別 PR）
3. `pc.functional`・損失・`TimeConstants`・`Carried` と、テスト 1・2
4. `pc.Input`・`pc.Layer`・`pc.Output` と、テスト 3〜10
5. MNIST の example、docs（英語と日本語）の使い方のページ、skill `pc-alm` 7 節への一般化と実験結果の追記
6. 各 PR で code-reviewer（学習則とコアの carry には algorithm-reviewer も）のレビュー

## 記事の公式実装との違い

| 項目                 | 公式実装（SakanaAI/pc-alm）                     | mujou.pc                                         |
| -------------------- | ----------------------------------------------- | ------------------------------------------------ |
| 層                   | $\sigma(W h)$ の residual MLP                    | 任意の微分可能な `nn.Module`                     |
| 自由変数             | $h$                                             | ずれ $u = h - \mu$                               |
| ミニバッチの始め     | $h$ を forward の値、$\lambda$ を 0 に初期化     | 何もしない（持ち越す）                           |
| 推論と学習           | $T$ ステップ推論してから重みを 1 回更新          | 同時に進む。入力が変わってから `learn_interval` 後に重みを更新 |
| $\lambda$ の更新     | 更新後の $h$ の $r$                              | 同じ                                             |
| 重みの勾配の $\lambda$ | 最後の dual 更新の前                           | VJP を取ったときの更新前の値                     |
| ステップの大きさ     | 離散の $\eta_h, \alpha$                          | 時定数と実経過の `dt`                            |
| 重みの最適化         | Adam                                            | 勾配流（SGD）。Adam は後で                       |
| 出力の損失           | バッチ平均の $\tfrac12\|y-\hat y\|^2$            | サンプルごとの任意の損失の和                     |

## 未決定事項

- ターゲットを付随させたうえでの $S$ の下限と精度（実装後に測る）
- 入力がフレームごとに変わり続けるストリーム（時間的に相関したデータ）での振る舞い
- $u$ で持つ力学の安定性。不動点は同じだが、過渡はラグランジアンの勾配流ではない
- 非同期の実行（`dt` の揺らぎ、読む値の古さ）で同期と同じ精度が出るか。example で測る
