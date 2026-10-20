# mujou.pc の実装計画

> 状態: `pc.Input`・`pc.Layer`・`pc.Output`・`pc.functional`、MNIST の example、docs の使い方のページ（`docs/en|ja/user-guide/pc.md`）と API リファレンス（`docs/en/api/pc.md`）は実装済み（2026-10-20）。skill `pc-alm` への追記は未着手（[実装の順番](#実装の順番) の 5）。

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

### 状態は $h$ で持ち、予測が離散的に跳んだら付け替える

自由変数は記事と同じく活動 $h_i$ と双対 $\lambda_i$ で持ち、記事の $h$ の力学に従う（2026-10-20、orchestrator の決定）。

$$\tau_h \dot h_i = -(r_i + \lambda_i + g_i),\qquad \tau_\lambda \dot\lambda_i = r_i,\qquad r_i = h_i - \mu_i$$

下のノードの緩和で $h_{\mathrm{in}}$ が連続的に動いて $\mu_i$ が変わっても、$h$ は動かさない（緩和で追う）。$\mu_i$ が**離散的に跳ぶ**ときだけ、$h$ を 1 回動かして $r$ と $\lambda$ を持ち越す（2026-10-20、orchestrator の決定。algorithm-reviewer の 2 回目の監査の must-1 と検討 1）。跳びは次の 2 種類。

1. **$\mu$ の入力側の跳び**: 付随データの `time` が変わった更新（新しいサンプル）と、受け取った活動の `epoch` が変わった更新（前のノードの $h$ が跳んだ）。$\mu^{\mathrm{last}}$ を前回の更新での予測として
   $$h_i \leftarrow \mu_i^{\mathrm{new}} + (h_i - \mu_i^{\mathrm{last}})$$
   と付け替える。両方が同じ更新で重なっても 1 回だけ。付け替える更新では $\delta = (h - \mu^{\mathrm{last}}) + \lambda$ が $\mu$ を計算する前に決まり、VJP は付け替えた後の $\delta$ で取る
2. **自分の $\theta$ の跳び**: $\theta$ を更新した更新で、同じ $h_{\mathrm{in}}$ に対して $\mu' = f(h_{\mathrm{in}};\theta_{\mathrm{new}})$ を `no_grad` で 1 回計算し、$\Delta = \mu' - \mu$ を持つ。**次の更新の最初に** $h \leftarrow h + \Delta$ とし、$\mu^{\mathrm{last}}$ は $\mu'$ にしておく。動かすのは $\theta$ による跳びだけで、その間の $h_{\mathrm{in}}$ の変化は $h$ に乗らない

どちらも $r = h - \mu$ を保つので、持ち越すのは予測からのずれ $r$ と $\lambda$ だけになる。

- 不動点（$r = 0$、$\lambda = -g$）は記事と同じ
- 教師が無ければ $r$ は 0 に減衰し、$h$ は forward pass になる。$r = 0$ のまま入力が切り替わると付け替えで新しい forward pass に乗るので、経路上のノードの更新 1 回ずつで出力に届く
- 時刻 0 に書いた最初のサンプルは、その前のゼロの付随データと時刻では区別できないが、`Input` の `epoch`（書き込みを数える）が変わるので付け替える

#### 跳びの連鎖（epoch）

ノードは $h$ と一緒に「$h$ が離散的に跳んだ回数」`epoch` を送る（メッセージ `pc.Activity`、[メッセージと Port](#メッセージと-port)）。ノードは次の更新で `epoch` を 1 進める。

- 新しいサンプルか、受け取った `epoch` の変化で付け替えた（1.）
- 自分の $\theta$ の跳び $\Delta$ を足した（2.）。ただし `learn_interval` を指定したときだけ

同じ更新で重なっても 1 進めるだけ。`Input` は書き込みのたびに進める。受け取ったノードはそれを見て自分も 1 回付け替え、自分の `epoch` を進める。下のノードの $\theta$ による $h$ の跳びが、経路に沿って $r$ を保ったまま上へ伝わるので、上のノードがいつ学習しても、跳びが $r$ に入った $\delta$ で学習することは無い（下の [学習の時刻の仮定](#学習の時刻の仮定)）。

- `learn_interval=None`（毎回学習する）では、$\theta$ は毎回の更新で $\Delta t$ に比例して連続的に動く。自分の $h$ には $\Delta$ を足すが、`epoch` は進めず、上のノードに連鎖させない。毎回連鎖させると上のすべてのノードが毎回付け替えることになり、$u$ の形（下の比較）に戻る
- 同じ理由で、`learn_interval` が経路の往復時間より短いと $u$ の形に戻る（[θ の更新間隔](#θ-の更新間隔) の安定性の条件）。間隔が短いと各ノードがほぼ毎回の更新で学習し、そのたびに `epoch` が進む。上のノードは毎回付け替え、付け替えは $\mu(h_{\mathrm{in}}) - \mu^{\mathrm{last}}$ なので、跳びだけでなく 1 周期分の $h_{\mathrm{in}}$ のドリフトも $h$ に乗る
- $\Delta$ を足すのを同じ更新にして、その更新で送る形は、非同期の MNIST のシミュレーションで 3 seed すべてが 40 バッチ前後で発散した（algorithm-reviewer）。上のノードが自分の学習の前に跳びを読む確率が上がるためで、1 更新遅らせる
- `epoch` は活動のメッセージに入れる。付随データ（carry）は中継専用で、途中のノードが書き換えられないからである。多入力（合流）のノードは入力ごとに `epoch` を持つ（後で足すもの）
- `Output` は活動を持たないので `epoch` を読まない

#### 比較実験と $u$ の形を採らなかった理由

最初の実装は、予測からのずれ $u_i = h_i - \mu_i$ を状態にし、送るときに $h_i = \mu_i + u_i$ を組み立てていた（$u$ の形）。毎回の更新で付け替えるのと同じで、入力の変化に即座に追従するが、深い鎖で連続時間の時点で不安定になる（algorithm-reviewer、2026-10-20）。

- $u$ の形では $u_i$ が上のすべての $h$ を即座に動かすので、損失のヘッセ行列が順伝播のヤコビアンの積を通して $u_i$ に返ってくる（ループゲインは $\|\prod J\|^2$）。$h$ の形の結合は隣接層の $J^\top J$ だけ。スカラーの線形鎖（$w = 1$）の連続時間の固有値の最大実部は、深さ 8 で $u$ の形が $+0.089$、$h$ の形が $-0.068$
- ヤコビアンが 1 に近い ResBlock で効く。ResBlock 6〜12 ノードの鎖（既定の初期化、Linear + MSE、$\eta_h = 0.25$、$\tau_\lambda = 2\tau_h$）で $u$ の形は必ず発散し（$\lambda$ が 4000 ステップで $10^{40}$）、$\eta_h$ を 0.05、0.01 に下げても発散する。$h$ の形は記事どおり収束する

素の PyTorch の同期 Jacobi での MNIST の比較（784→256→256→10、ce、$S = 32$、$\eta_h = 0.5$、$\alpha = 0.25$、60k × 3 epoch × 3 seed、BP との差 [pt]）:

| 状態の持ち方                                  | relu  | tanh  |
| --------------------------------------------- | ----- | ----- |
| $h$ で持ち、新サンプルと自分の $\theta$ の更新の次の更新で付け替える | −0.06 | −0.02 |
| ずれ $u$ で持つ                               | −1.23 | −0.57 |

付け替えを新サンプルのときだけにすると、$\theta$ の更新で $\mu$ が跳んだ分が $r$ に入り、relu で発散した（20k × 1 epoch、$S \in \{16, 32\}$、$\eta_h \in \{0.25, 0.5\}$ の 8 通りのうち 4 通り。tanh は 8 通りとも収束）。

#### θ の跳びだけを付け替える理由と連鎖の比較

表の 1 行目（前の規則）は、$\theta$ を更新した次の更新で $h \leftarrow \mu^{\mathrm{new}} + (h - \mu^{\mathrm{last}})$ と付け替えていた。この付け替えは $\theta$ による跳びだけでなく、その 1 更新の間の $h_{\mathrm{in}}$ の変化も $h$ に乗せる。`learn_interval=None` では毎回の更新で付け替えるので、$u$ の形そのものに戻る（algorithm-reviewer の 2 回目の監査）。

- ResBlock 8 ノードの鎖（目標は forward + 雑音、credit 0.62）を上から同期に回すと、前の規則では $\tau_\theta = 10^9$（$\theta$ はほぼ動かない）でも $\lambda$ が 500 ステップで $10^5$ に増えた。$\theta$ の跳び $\Delta = f(h_{\mathrm{in}};\theta_{\mathrm{new}}) - f(h_{\mathrm{in}};\theta)$ だけを足す今の規則では、$\lambda$ は credit から離れず、6000 ステップで credit と $7\times10^{-10}$ で一致する（テスト 14）
- 非同期の MNIST（784-256-256-10、実ノード、各ノードの周期 1 ms ± 10%、毎ティックで順序をシャッフル、600 バッチ × 3 seed。algorithm-reviewer の `asim.py`）での比較:

| 方式 | 停止なし | 1 ノードが 0.5 s ごとに 115 ms 止まる | 0.25 s ごと（l2 / l1） | prequential（停止なし） |
| --- | --- | --- | --- | --- |
| 前の規則、書いた時刻から学習 | 3/3 | 2/3 | 3/3 / 3/3 | 0.989〜0.991 |
| 前の規則、入力を見た時刻から学習 | 1/3 | 0/3 | — | — |
| 連鎖、入力を見た時刻から学習 | 3/3 | 3/3 | 3/3 / 3/3 | 0.9956〜0.9958 |
| 連鎖、書いた時刻から学習 | 3/3 | 3/3 | 3/3 / 3/3 | 0.993〜0.994 |
| $\theta$ の跳びだけ + 連鎖、書いた時刻から学習（採用） | 3/3 | 3/3 | 3/3 / 3/3 | 0.993〜0.994 |

数字は発散しなかった seed の数。連鎖の方式は 24 回中 0 回の発散だった。この停止は 1 つのノードが周期的に止まるだけで、全ノードが不規則に遅れる実機の過負荷より穏やか（実機の結果は [MNIST の example](#mnist-の-example)）。

### 符号・バッチ・損失の規約

- 下へ送る値は上のノードのエネルギーの $h$ に対する勾配。複数の相手からの値は足すだけ
- VJP はバッチの和で取り、$1/B$ は $\theta$ の勾配にだけ掛ける（2026-10-19、ユーザーの承認）。局所のエネルギーを `.mean()` で作ると $h$ の勾配まで $1/B$ 倍になるので使わない。この規約で $\lambda$ はサンプルごとの credit $-\partial\ell_b/\partial h_b$、$\theta$ の更新は平均損失の勾配降下になる
- 出力ノードの損失は**サンプルごとの値** $\ell_b$（shape `(B,)`）を返す関数とし、和はノードが取る。`nn.MSELoss` の既定（平均、½ なし）をそのまま使うと $g_{\mathrm{in}}$ が $2/B$ 倍になり、$\theta$ には $1/B$ が二重に掛かる。`pc.mse` は $\tfrac12\|y-\hat y\|^2$ を特徴の次元で和にしたもの、`pc.cross_entropy` は `reduction="none"` の交差エントロピー
- `f` はサンプルごとに独立であること。学習モードの BatchNorm のようにバッチの中で混ざる演算は使えない（LayerNorm や GroupNorm は使える）
- `f` は決定的で、自分の状態を持たないこと（Dropout や学習モードの BatchNorm は使えない）。学習した更新で $\theta$ の跳びを求めるため、同じ $h_{\mathrm{in}}$ で `f` をもう 1 回計算する。Dropout では跳びに雑音が入り、BatchNorm は running stats を余分に更新する（2026-10-20、algorithm-reviewer の 3 回目の監査）

## ノードの構成

### 状態

| ノード   | 状態                                   | 時定数                                  |
| -------- | -------------------------------------- | --------------------------------------- |
| `Layer`  | $\theta_i$（`f`）、$h_i$、$\lambda_i$、$\mu_i^{\mathrm{last}}$、$\Delta_i$（次の更新で足す $\theta$ の跳び） | $\tau_\theta$、$\tau_h$、$\tau_\lambda$ |
| `Output` | $\theta_L$（`f_L`）                    | $\tau_\theta$                           |
| `Input`  | 送った `epoch`、最後に見た書き込みの時刻 | —                                       |

ほかに、最後に見た入力の時刻・次に学習する時刻・前回の学習からの経過時間・その入力で学習したか・足すべき $\Delta$ があるか・送る `epoch`・最後に受け取った `epoch` を Python の数値で持ち、`Node.register_state`（[コアの変更](#コアの変更-noderegister_state)）で状態の木に入れる。$h, \lambda, \mu^{\mathrm{last}}, \Delta$ は属性に `Tensor` を代入するだけで集まるので、専用のクラスは作らない。

$\delta$ は状態にせず、更新のたびに今の `h_in` から $\mu$ を計算し直して作る。落とし穴 2（$\sigma'$ をキャッシュで評価しない）は自動で守られる。$\mu^{\mathrm{last}}$ は付け替えにだけ使う。

### メッセージと Port

活動は `pc.Activity`（`TypedDict`）で送る（2026-10-20、orchestrator の決定）。

| フィールド | 中身 |
| ---------- | ---- |
| `h`        | 活動、`(B, *out)` |
| `epoch`    | `()` の int64。送り手の $h$ が離散的に跳んだ回数（[跳びの連鎖](#跳びの連鎖epoch)） |

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
| `Input`  | `next` | $g$（捨てる）                         | $\{h = x, \mathrm{epoch}\}$ + 付随データ（書いた時刻を `time` に入れる） |
| `Layer`  | `prev` | $\{h_{\mathrm{in}}, \mathrm{epoch}\}$ + 付随データ | $g_{\mathrm{in}}$                          |
| `Layer`  | `next` | $g$（相手ごと、和を取る）             | $\{h_i, \mathrm{epoch}\}$ + 付随データ（`prev` から中継） |
| `Output` | `prev` | $\{h_{\mathrm{in}}, \mathrm{epoch}\}$ + 付随データ | $g_{\mathrm{in}}$                          |
| `Output` | `next` | —                                     | $\{\hat y, y, \mathrm{mask}\}$                    |

- `Output` は `prev` から届いた $h_{\mathrm{in}}$ と、同じメッセージの $y$・`mask` を組む。$y$ は $x$ と同じ経路を同じ遅れで進むので、組み違いが構造的に起きない。時刻の照合も、ホストが $x$ と $y$ を同時に書く必要も無い
- ホストが伝播より速く次のサンプルを書いても組み違えない（時刻で揃える方式は、`target` のスロットが常に新しいサンプルで上書きされ、一致する瞬間が来なくなる）
- 教師なしのサンプルは `mask` を False にする。損失は $\sum_b \mathrm{mask}_b\,\ell_b$ で、1 つのバッチにラベルのあるサンプルと無いサンプルが混ざってよい
- `Layer` は付随データを中継する。読むのは学習の時刻と付け替えのための `time` と、`mask` だけ。付け替えの連鎖には活動の `epoch` を読む。学習するステップで `mask` がすべて False なら、前の入力から持ち越した $\delta$ で学習しないよう、$\theta$ を動かさず $\Sigma\Delta t$ を 0 に戻す（`Output` も同じ。2026-10-20、orchestrator の決定）
- `mask` が False の行は、VJP の cotangent $\delta$ を 0 にする（2026-10-20、algorithm-reviewer の指摘）。その行の $\delta$ は前の入力から持ち越した値かもしれず、そのまま流すと下のノードの credit と $g_\theta$ に雑音が乗る。その行の不動点はもともと $\lambda = 0$ なので不動点は変わらない。「すべて False なら学習しない」を行ごとに一般化したもので、`Output` の扱いと揃う
- `Output` の `next` は予測と $y$・`mask` を普通のメッセージで送り、ホストが prequential 精度を測れるようにする（ホストの読み書きは carry に対応させない）

`Input` は、ホストが書いた $\{x, y, \mathrm{mask}\}$ を活動と付随データに分けて送るだけのノード。これでどの `Layer` の `prev` も同じ型になる。`Input` の次の `Layer` は $g_{\mathrm{in}}$ の計算を省いてよい（読み手が捨てるので）。

### ResBlock などの入れ方

ノードの境界は、制約 $h_i = f_i(\cdot)$ を置いて自由変数 $h, \lambda$ を持つ場所。

- ResBlock を 1 つの `Layer` にする（既定）: $f(h) = h + g_\theta(h)$ を渡す。スキップ接続と中間の活性はノードの内側で BP される
- ノードをまたいで分ける: 合流するノードが入力ごとに Port を持ち、VJP を入力ごとに返す。分岐する側は `next` に届いた値を足す。一般形は $\nabla_{h_j}\mathcal L = \delta_j + \sum_{i \in \mathrm{children}(j)} \partial E_i/\partial h_j$。付随データは入力のうちの 1 つから中継する

細かく分けるほど並列度と局所性が上がり、$h, \lambda$ のメモリと緩和の時間（経路上のノード数に比例）が増える。モデル全体を 1 ノードにして出力ノードをつなぐとオンラインの BP になる。

## on_update の手順

`Layer` は 1 回の `on_update(dt)` で次を行う。

1. `dt` を `max_dt` で打ち切る（[dt が伸びたとき](#dt-が伸びたとき)）。以下の `dt` は打ち切った値
2. `prev` から $h_{\mathrm{in}}$ と `epoch`、付随データの `time`・`mask`、`next` から $g$ を読む
3. `time` が最後に見た値から変わったら、新しい入力として覚え、次に学習する時刻を `time + learn_interval` にする。この更新では（`learn_interval` を指定したとき）学習しない
4. 前の更新で $\theta$ を動かしていたら、$h \leftarrow h + \Delta$ とする
5. $\mu = f(h_{\mathrm{in}})$ を計算し、**更新前の** $\delta = r + \lambda$（`mask` が False の行は 0）で VJP を取る。新しい入力の更新と、受け取った `epoch` が変わった更新では、$r = h - \mu^{\mathrm{last}}$ とし、VJP の後で $h \leftarrow \mu + (h - \mu^{\mathrm{last}})$ と付け替える。学習するステップなら $g_\theta$ も取り、しないステップは $h_{\mathrm{in}}$ についてだけ取る
6. $g_{\mathrm{in}}$ を送る
7. $h$ を進め、続けて**更新後の** $h$ の $r$ で $\lambda$ を進める（1 歩の陽的 Euler、公式実装と同じ順）。$\mu^{\mathrm{last}} \leftarrow \mu$
8. 5 で付け替えたか、4 で（`learn_interval` を指定していて）$\Delta$ を足したら `epoch` を 1 進め、$\{h, \mathrm{epoch}\}$ を、`prev` で読んだ付随データとともに送る（中継）
9. 学習するステップなら $\theta$ を更新し、$\mu' = f(h_{\mathrm{in}};\theta_{\mathrm{new}})$ を `no_grad` で計算して $\Delta = \mu' - \mu$、$\mu^{\mathrm{last}} \leftarrow \mu'$ とする

状態の更新（4・5 の付け替え、7、9）は `torch.no_grad()` の下で行う。autograd の行には `# mujou: local-autograd` を付ける。

`Output` は 1〜4 で $\ell = \sum_b \mathrm{mask}_b\,\ell_b(\hat y_b, y_b)$ を微分して $g_{\mathrm{in}}$ と（学習するステップなら）$g_\theta$ を得る。`mask` がすべて False なら $g_{\mathrm{in}} = 0$ で、学習しない。

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

### 付け替えればリセットは要らない

予測が跳ぶ更新で $h$ を付け替えると、入力が変わっても $h$ は新しい予測に乗り、持ち越すのは credit 程度の大きさの $r, \lambda$ だけになる。リセットせずに持ち越した結果は [比較実験](#比較実験と-u-の形を採らなかった理由) の表のとおりで、$S = 32$ で BP と同等。

以前の $u$ の形での測定（ce、$\eta_h = 0.5$、$\alpha = 0.2$、バッチの最後で学習）では、$S = 16$ で崩れた（relu は 34.6〜90.4%、tanh は −0.6pt）。原因は $u, \lambda$ の持ち越しではなかった。$u$ だけ 0 に戻しても、$\lambda$ だけ 0 に戻しても結果はほぼ変わらなかった。原因は**入力とラベルの時刻のずれ**である。新しい $x$ は経路上のノードの更新を 1 回ずつ経て出力に届くが、ラベルは出力にすぐ届く。その間、出力は古い入力から作った $h$ を新しいラベルと組んで微分し、別のサンプルのラベルに対する勾配を流す。切り替え直後の $\delta_2$ は新しい credit の 3〜5 倍まで膨らんだ。

ラベルを伝播の遅れの分だけ遅らせると、この行き過ぎは消えた（$S = 16$ の relu が崩壊から −1.9pt、毎ステップ学習の relu が −9.9 から −0.87pt、tanh が −1.7 から −0.19pt）。

そこで次のように決める。

- 状態は $h, \lambda$ で持ち、入力が変わっても持ち越す（付け替えはするが、リセットの機構は持たない）
- ターゲットは入力に付随させ、活動と同じ経路で運ぶ（[メッセージと Port](#メッセージと-port)）。伝播の遅れの分だけラベルを遅らせることが、遅れの段数を数えずに実現できる
- 古い credit が新しい入力のヤコビアンを通るずれは残る。$S$ の下限はリセットより厳しい（リセットは $S \ge 8$、持ち越しは $S \ge 32$）

ターゲットを付随させたうえでの $S$ の下限と精度は、実装後に測り直す。

## θ の更新間隔

$h, \lambda$ は毎回の `on_update` で更新し、$\theta$ は間隔を空けて更新する。学習するステップでは前回の $\theta$ の更新からの経過時間 $\Sigma\Delta t$ を重みにして

$$\theta \leftarrow \theta - \frac{\Sigma\Delta t}{\tau_\theta}\cdot\frac{g_\theta}{B}$$

とする。時間あたりの学習の速さは $\tau_\theta$ で決まり、間隔は $g_\theta$ を**どの時点でサンプルするか**と計算量を決める。

間隔は時間 `learn_interval`（秒）で持つ（2026-10-19、ユーザーの承認）。ノードは、ホストが入力を書いた時刻（付随データの `time`）から `learn_interval` ごとの時刻 `time + k * learn_interval`（$k = 1, 2, \dots$）を過ぎた最初の更新で学習する。ただし、その入力を初めて見た更新では学習しない（2026-10-20、orchestrator の決定。下の 2 つ目）。`ManualRun` を固定の `dt` で回すと「入力を書いてから $K$ ステップごと」と同じになる。`learn_interval=None` は毎回学習する（連続時間形そのまま。新しい入力を見た更新でも学習する）。

- 学習の時刻を書いた時刻から数える理由と、その仮定は [学習の時刻の仮定](#学習の時刻の仮定)
- 新しい入力を初めて見た更新の $\delta$ は、前の入力の $r + \lambda$ に新しい入力のヤコビアンを掛けたもので、新しい入力の credit ではない。経路の深いノード（入力が届くのが `time + learn_interval` 以降）や、止まっていて戻ったノードはその更新で学習しうるので、学習を次の更新に回す（algorithm-reviewer・code-reviewer の 2 回目の指摘）。`learn_interval` の下限は下の安定性の条件
- $\Sigma\Delta t$ は、学習した後の入力の切り替えでは戻さない。学習しないまま入力が切り替わったら（入力が `learn_interval` より速く変わる）、ラベルの無いときと同じく $\Sigma\Delta t$ を捨てる。溜めたままにすると、入力が止まった最初の学習にまとめて掛かり、学習率が際限なく跳ね上がる（2026-10-20、algorithm-reviewer の指摘）
- $\Sigma\Delta t$ は `max_dt` で打ち切った `dt` の和。学習の時刻は実行の時間で決まる（打ち切った時間も含む）
- **安定性の条件**: `learn_interval` は credit が経路を往復する時間 $2 \times$ 深さ $\times$ 周期以上にする（docstring にも書く）。これは精度の目安ではなく、これより短いと $\theta$ がほぼ止まっていても深い鎖が発散しうる。間隔が短いと毎回の更新で学習して跳びの `epoch` が進み、次のノードが毎回付け替えるので、[跳びの連鎖](#跳びの連鎖epoch) が $u$ の形に戻る。ResBlock 8 ノードの鎖（テスト 14 と同じ構成、$\tau_\theta = 10^9$、credit 0.62、往復は 18 周期）で、`None` と 16 周期は $\lambda$ が credit から離れず、4 周期は有界だが credit の約 4 倍、2 周期と 1 周期は発散した（2026-10-20、algorithm-reviewer の 3 回目の監査）。MNIST の example（3 ノード、間隔は $S$ 周期）は条件を満たす

最初のサンプルが届くまで、付随データはゼロ（`time` は 0、`mask` はすべて False）である。時刻 0 に書いた最初のサンプルは、時刻ではゼロの付随データと区別できないので、学習の時刻の上では新しい入力にならない（学習の時刻は同じ `time` 0 から数えるので変わらない。付け替えは `Input` の `epoch` で起きる）。学習するかは `mask` で決まり、ゼロの付随データでは学習しない（2026-10-20、carry のレビューを受けて決定）。

### 学習の時刻の仮定

学習の時刻を「ノードが入力を見た更新から」ではなく「ホストが書いた時刻から」数えるのは、経路上のノードの学習をほぼそろえるためである（2026-10-20、plan-implementer。[未決定事項](#未決定事項)）。

- 入力を見た更新から数えると、経路上で深いノードほど 1 更新ずつ遅れて学習する。下のノードが $\theta$ を更新すると、その次の更新で $h$ が跳び、上のノードの $h_{\mathrm{in}}$ が跳ぶ。上のノードがそれを読んだ後に学習すると、跳んだ分が $r$ に入った $\delta$ で学習してしまう。同期の Jacobi では上のノードがちょうど 1 更新先に学習するので起きないが、非同期の実行では更新の順序の揺らぎで起きる。MNIST の 3 ノードを `ManualRun` で揺らいだ順に回すと、数十バッチで発散した（固定の Jacobi の順では 1 epoch 安定）
- 書いた時刻から数える方式は、次を仮定している（実行モデルの不変条件 2「値の古さや更新周期に依存する仮定は明示する」）
  - 1 つの実行の全ノードが同じタイマーを読む（skill `locality-and-async` の「1 つの実行の時間軸は 1 本」）。読むのはメッセージの付随データの `time` と実行のタイマーだけなので、非局所な読み取りでもバリアでもない
  - 経路上のノードの学習が、1 更新程度の差でそろう。上のノードが下のノードより 1 周期以上遅れて学習すると（負荷で止まったノードなど）、上の理由で揺れた $\delta$ で学習しうる
- [跳びの連鎖](#跳びの連鎖epoch) はこの仮定を緩める。下のノードの $\theta$ による跳びは `epoch` で上のノードの付け替えを起こし、上のノードの $r$ に入らない。そのため、学習のずれで揺れるのは $\delta$ ではなく、学習に使う $\delta$ がどこまで緩和したか（credit がどれだけ届いたか）だけになる。非同期のシミュレーションでは、連鎖があれば「入力を見た時刻から」でも発散しなかった（[θ の跳びだけを付け替える理由と連鎖の比較](#θ-の跳びだけを付け替える理由と連鎖の比較)）
- 「この順序に依らない」とは言えない。1 つの経路のノードが同じ更新ステップで学習することは、`ManualRun` で全ノードを毎ステップ回すとき（どの順でも）に限って保証される（テスト 16）

サンプルの時点が効くのは、入力が変わった直後の $\delta$ が新しいサンプルの credit になっていないからである。MNIST での比較（$h$ で持ちリセット、relu-ce、$S = 32$）:

| 学習のタイミング                   | BP との差 | $g_{W_1}$ の cos / ノルム比 | 1 バッチの計算時間 |
| ---------------------------------- | --------- | --------------------------- | ------------------ |
| 毎ステップ（$K = 1$）              | +0.13pt   | 0.85 / 0.68                 | 26.6 ms            |
| $K = S/4$                          | 発散      | —                           | 16.3 ms            |
| バッチの最後（切り替えから $S$）   | −0.13pt   | 0.999 / 0.97                | 15.1 ms            |
| バッチの途中（切り替えから $S/2$） | +0.56pt   | 0.995 / 1.18                | 15.1 ms            |

- 効くのは切り替えからの時間だけで、途中で学習した結果は $S/2$ だけ見せて最後に学習した結果と完全に一致した
- 切り替えと無関係な位相で一定間隔ごとに学習すると、$u$ で持つ場合も relu で −2〜−19pt と崩れた
- 既定は「入力を書いてから `learn_interval` 後に 1 回」。入力がフレームごとに変わり続けるストリームでは `learn_interval=None` を使う
- `period` が `max_dt` より長いと、毎回の更新が `max_dt` に打ち切られ、緩和も学習も時間の一部を黙って捨てる。構築時に `ValueError` にする（2026-10-20、code-reviewer の指摘）

## 時定数

`pc.TimeConstants(h, lam, theta)` に秒で持つ。$\alpha = 0$ は $\tau_\lambda = \infty$ として受け付ける（`dt / tau.lam` が 0 になるので、レート $1/\tau$ を別に持たない。`relax_` は $\tau_\lambda = \infty$ なら $\lambda$ に触れないので、$\lambda$ はちょうど 0 のまま）。

ノードの周期の既定は $\tau_h / 4$（$\eta_h = 0.25$）にする。利用者は時定数だけを決めればよく、周期を明示すればそちらを使う。

安定な範囲（$h$ で持ちリセット、$S = 32$）:

- relu: $\eta_h \le 0.5$ なら $\alpha \in [0, 1]$ のすべてで安定。$\eta_h = 0.7$ は $\alpha \le 0.1$ で発散、$\eta_h = 1$ はほぼ発散
- tanh: $\eta_h \le 0.7$ で安定。CE で $\alpha \le 0.1$ だと劣化
- 1 つのモードを線形化すると $u'' + \eta_h c\,u' + \eta_h\alpha\,u = 0$（$c \approx 1 + \|J\|^2$）。臨界減衰は $\alpha \approx c^2\eta_h/4$ で、実測の最良帯 $\eta_h \in [0.3, 0.7]$、$\alpha \in [0.1, 0.3]$ と一致する。$\alpha$ が大きすぎると行き過ぎ（ノルム比 1.6〜1.8）、$\alpha = 1$ では勾配の向きも崩れる（cos 0.5〜0.9）

記事の公式実装（SakanaAI/pc-alm）は $\eta_h = 1/\lambda_{\max}$（約 0.2〜0.25）、$\alpha = 1$、$T = 64$ か $2L$、Adam を使う。$\eta_h$ は周期 $\tau_h/4$ の既定とほぼ同じ。$\alpha = 1$ は上の結果では大きすぎるが、公式の実験は幅 32 の深い residual MLP で、曲率 $c$ が違う。example の既定は $\tau_\lambda = 2\tau_h$（$\alpha = 0.125$）とし、実装後に確かめる。

### dt が伸びたとき

実時間の実行では `dt` が周期より長くなることがある。PC のノードは 1 回の更新で使う `dt` を `max_dt` で打ち切り、$h, \lambda$ の緩和にも $\theta$ の $\Sigma\Delta t$ への加算にも打ち切った `dt` を使う（2026-10-20、orchestrator の決定）。既定は $\min(\tau_h/2,\ 3\tau_\lambda)$ で、周期 $\tau_h/4$ の 2 倍。

- 負荷で全ノードがそろって約 230 ms 止まった直後に、学習率 0.9 の 1 歩を踏み（$\Sigma\Delta t$ が 8〜9 バッチ分）、続いて `dt` の伸びでノード間の陽的な結合が不安定になって発散した（MNIST の example、[memory/agents/plan-implementer/mnist-example.md](../memory/agents/plan-implementer/mnist-example.md)）
- 以前は、$\mu$ と $g$ を固定したまま $u, \lambda$ を `max_step` 以下の刻みに分けて進めていた。刻みの分割は自分の状態の陽的 Euler を安定にするだけで、$g$ は $h_i$ にも依存する（上のノードの $J^\top J$ を通じた結合）ので、ノード間の結合は陽的なまま残る。分割では安定化できないので、分割をやめて `max_dt` に置き換えた。1 回の更新は 1 歩の陽的 Euler
- 既定の値の根拠: 局所の 2 変数系（$h$ を進めてから、進めた $h$ の $r$ で $\lambda$）の陽的 Euler は、$a = \Delta t/\tau_h$、$b = \Delta t/\tau_\lambda$ として $0 < a < 2$ かつ $ab < 4 - 2a$ で安定。$a \le 1/2$ なら $b \le 3$ で足りる（algorithm-reviewer）
- 打ち切ると経過時間が失われ、連続時間の定式化からずれる。コアは `dt` を打ち切らない（[ノードの実行の周期](nodes.md#実行の周期)）。打ち切るのは PC のノードの選択で、失われた時間の分だけ学習が遅れる
- ノード間の結合の安定の余裕はモデルに依存する。スカラーの線形の 2 層鎖を上から Jacobi で回すと、$\eta_h = \Delta t/\tau_h$ が $w = 1$ なら 1.0、$w = 1.5$ なら 0.5 で発散する（algorithm-reviewer）。`max_dt` の既定（$\eta_h \le 0.5$）は一般の保証ではない
- MNIST の example では、timer の時間のスケールを 2 にすると、最も遅いノード（784 入力）が周期の大半を計算に使い、読むメッセージが古くなって、数百バッチで発散した（`dt` は 1.25 周期程度で `max_dt` に届かない。前の付け替えの規則のとき）。跳びの連鎖を入れてからは、スケール 2 でも 1 epoch を完走した。スケール 1 では 3 epoch を通して安定。`max_dt` は `dt` の伸びを抑えるが、読む値の古さは抑えない

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
self.next = NextPort(send=activity((B, 784)), recv=..., send_carry=carry)
# pc.Layer / pc.Output: 受け取り、Layer は次へ中継する
# activity(shape) は DictSpec(pc.Activity, h=TensorSpec(shape), epoch=TensorSpec((), torch.int64))
self.prev = PrevPort(send=..., recv=activity((B, 784)), recv_carry=carry)
self.next = NextPort(send=activity((B, 256)), recv=..., send_carry=self.prev)
```

## 機能

### MVP（MNIST に必要なもの）

| 公開名                        | 内容                                                                                         |
| ----------------------------- | -------------------------------------------------------------------------------------------- |
| `pc.Input`                    | ホストが書いた $\{x, y, \mathrm{mask}\}$ を、活動と付随データに分けて送るノード               |
| `pc.Layer`                    | 任意の `nn.Module` を包むノード。$h, \lambda$ を持ち、付随データを中継する。出力の shape は構築時に `f` を 1 回通して決める |
| `pc.Output`                   | $f_L$ とサンプルごとの損失を持つノード                                                       |
| `pc.TimeConstants`            | $\tau_h, \tau_\lambda, \tau_\theta$（秒）                                                    |
| `pc.Activity`                 | ノードが次のノードへ送る活動の `TypedDict`（$h$ と `epoch`）                                 |
| `pc.Carried` / `pc.carry_spec` | 付随データの `TypedDict` と、ターゲットの spec とバッチサイズから `DictSpec` を作る関数      |
| `pc.Sample` / `pc.Prediction` | ホストが `Input.prev` に書くもの（$x, y$, `mask`）と、`Output.next` が送るもの（$\hat y, y$, `mask`, `time`） |
| `pc.mse` / `pc.cross_entropy` | サンプルごとの損失                                                                           |
| `pc.functional`               | 状態を持たない計算: `vjp`（$\delta$ は値か、$\mu$ から作る関数）、`loss_vjp`（出力ノードの損失の勾配）、`relax_`（1 歩の陽的 Euler）、`learn_` |

`pc.functional` は、多入力のノードなどを利用者が自分で書くときの部品で、`Layer` と `Output` もこれで書く。重みの初期化は利用者が `module` を作るときに行う。

```python
import mujou.pc as pc

B = 64
tau = pc.TimeConstants(h=4e-3, lam=8e-3, theta=...)  # 周期の既定は tau.h / 4 = 1 ms
carry = pc.carry_spec(TensorSpec((B,), torch.int64), batch=B)
inp = pc.Input((B, 784), carry=carry, period=tau.h / 4)  # period は必須
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

- **結果**（2026-10-20、$\theta$ の跳びだけの付け替え + 跳びの連鎖、32 コア、3.14t、時間のスケール 1、4 ワーカー）: 3 epoch 後の test は PC 97.82% / BP 97.54%（seed 0）、PC 97.59% / BP 97.45%（seed 1）。prequential は 3 epoch 目で PC 98.07〜98.08% / BP 97.92〜97.99%。1 バッチの実時間は PC 32 ms、BP 0.8 ms。`dt` は 99% 点まで 1.25 周期。スケール 2 も 1 epoch を完走した（96.47% / 96.23%。l1 の `dt` の 99% 点が 2.25 周期で警告が出る。前の規則では数百バッチで発散した）
- **負荷の下**（`busy.py` の busy loop と並べて 1 epoch、seed 0）: スケール 1 で 16 本 96.62%、26 本 96.52% と 96.23%（2 回）、32 本は発散（prequential 19.5%、test 9.8%）。32 本でスケール 0.5 は完走したが 93.52% に落ちた。前の規則では 26 本・スケール 1 と 32 本・スケール 0.5 が発散していた

計算量の目安: 同期のシミュレーションを 1 スレッドで回すと、1 バッチが BP の 0.9 ms に対して 15 ms（約 17 倍）だった。実時間の待ちは含まない純粋な計算の差で、1 バッチを $S = 32$ ステップ緩和し、各ステップで全ノードが forward と VJP を 1 回ずつ取るため。mujou ではノードが別々のワーカーで同時に計算するので、3 ノードなら最大でおよそ 3 倍縮む。

## テスト

`tests/mujou/_core/pc/` に実装をミラーして置き、float64 と `mujou.testing.run` で決定的に回す。損失が下がったかでは判定しない。

1. `functional.vjp` の $g_{\mathrm{in}}, g_\theta$ が、$E_i = \tfrac12\|h_i - \mu_i + \lambda_i\|^2$ と $\sum_b \ell_b$ の有限差分と一致する。$\delta$ を $\mu$ から作る関数で渡しても同じ
2. 上から（`Output` → … → `Input`）回す同期 step が、純粋な PyTorch で書いた $h$ の形（付け替え・$\theta$ の跳び・`epoch` の連鎖・学習の時刻を含む）の参照実装の Jacobi 更新と ~1e-12 で一致する。`learn_interval=None`（毎回学習）と、間隔を空けて学習する場合の両方。上から回すと、各ノードは下の $h$ を前の時刻の値で読み、上の $g$ も前の時刻の状態から作られた値を読む
3. 入力を固定して回した不動点で、$h_i$ が forward の値、$\lambda_i$ がネットワーク全体を autograd した $-\partial\ell/\partial h_i$ と一致し、$\theta$ の 1 回の更新が SGD と一致する（線形と tanh、Linear と ResBlock）。$f = \sigma(Wh)$ では記事の式（skill `pc-alm` 3 節）の不動点とも一致する
4. $\alpha = 0$（$\tau_\lambda = \infty$）で $\lambda \equiv 0$、不動点で $r = -g$、$h$ が forward の値からずれる（PC）
5. スカラーの線形の例で $\lambda = w_2(y - \hat y)$
6. 上から・下から・固定 seed のランダム順で回しても同じ不動点に収束する。間隔を空けて 2 回学習しても、どの順でも各回の更新が SGD と一致する（1 回目の跳びが連鎖し、次の不動点に緩和する）
7. 教師が無いとき、$h$ は forward の値で、$\lambda = 0$、$\theta$ は動かない。$r = 0$ から入力を切り替えると、新しい forward pass は経路の分の更新で届く
8. `Output` は常に、受け取った活動と同じメッセージの $y$ を使う。`mask` が False のサンプルは損失に入らず、`Layer` もその行の credit を返さない
9. `learn_interval` を固定の `dt` で回すと、入力を書いてから $K$ ステップごとに $\Sigma\Delta t$ の重みで更新される。学習しないまま入力が切り替わったら $\Sigma\Delta t$ を捨てる
10. `Run.state_dict` で取得した状態から復元すると、同じスケジュールで同じ値になる（付随データ・学習の時刻・$\Sigma\Delta t$・$\mu^{\mathrm{last}}$・$\Delta$・`epoch` を含む）
11. ResBlock 8 ノードの鎖が不動点に収束する（$u$ の形では発散した条件。目標は forward の出力に雑音を足したもので、credit は 1 程度）
12. Linear + tanh を 2 つまとめた 1 ノードの鎖と、2 ノードに分けた鎖で、不動点での $\theta$ の更新が一致する（ノードの分け方に依らない）
13. 新しい入力の更新と、受け取った `epoch` が変わった更新でだけ付け替え、そのとき $r = h - \mu$ が付け替えの前後で保たれる。$\theta$ を動かした次の更新では、$\Delta = f(h_{\mathrm{in}};\theta_{\mathrm{new}}) - f(h_{\mathrm{in}};\theta)$ だけを足す。`epoch` は付け替えと（間隔を空けた学習の）$\Delta$ で 1 ずつ進み、重なっても 1 回。`learn_interval=None` では $\Delta$ で進めない。`Input` の `epoch` は書き込みを数える。`dt` が `max_dt` を超えると、緩和と $\Sigma\Delta t$ に `max_dt` を使う（`Output` も）
14. ResBlock 8 ノードの鎖を $\tau_\theta = 10^9$ で、`learn_interval=None` と 16 周期（往復の 18 周期に近い）で回すと、$\lambda$ は credit の 2 倍を超えず、credit に収束する（前の規則では `None` で $10^5$ まで増えた。今の規則でも 2 周期では credit の 2 倍を超える）
15. 新しい入力を初めて見た更新では（`learn_interval` を指定したとき）学習しない。経路より短い `learn_interval` では、入力が届くのが遅いノードは 1 更新遅れて学習する
16. `learn_interval` が経路の往復より長ければ、経路上の全ノードが同じステップで学習する（上から・下から・ランダム順）

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
| 自由変数             | $h$                                             | $h$。予測が離散的に跳ぶ更新で 1 回だけ付け替える（新しいサンプル、前のノードの $h$ の跳び、自分の $\theta$ の跳び） |
| 送る活動             | $h$                                             | $h$ と跳びの回数 `epoch`（`pc.Activity`）。上のノードは `epoch` が変わったら付け替える |
| ミニバッチの始め     | $h$ を forward の値、$\lambda$ を 0 に初期化     | $h$ を予測に付け替え、$r, \lambda$ を持ち越す    |
| 推論と学習           | $T$ ステップ推論してから重みを 1 回更新          | 同時に進む。入力を書いてから `learn_interval` 後に重みを更新（全層一斉の更新を、共通のタイマーで近似する） |
| 重みの更新の後       | 次のミニバッチで $h$ を forward に初期化         | 次の更新で $h$ に $\theta$ の跳びを足し、経路に沿って連鎖させる |
| $\lambda$ の更新     | 更新後の $h$ の $r$                              | 同じ                                             |
| 重みの勾配の $\lambda$ | 最後の dual 更新の前                           | VJP を取ったときの更新前の値                     |
| ステップの大きさ     | 離散の $\eta_h, \alpha$                          | 時定数と実経過の `dt`（`max_dt` で打ち切る）     |
| 重みの最適化         | Adam                                            | 勾配流（SGD）。Adam は後で                       |
| 出力の損失           | バッチ平均の $\tfrac12\|y-\hat y\|^2$            | サンプルごとの任意の損失の和                     |

## 未決定事項

- ターゲットを付随させたうえでの $S$ の下限と精度（実装後に測る）
- 入力がフレームごとに変わり続けるストリーム（時間的に相関したデータ）での振る舞い
- $h$ の形でも、ResBlock 8 ノードの鎖で、目標を forward と無関係な乱数にしたサンプル（credit が 2.7 程度）では、サンプルの一部が不動点に収束せず有界な振動を続けた。`dt` を小さくしても止まないので、連続時間の性質（拡張ラグランジアンの罰則の重みが 1 に固定で、$\lambda^\top \nabla^2 f$ の項に対して局所の凸性が足りない可能性）。credit が 1 程度なら収束する（テスト 11）
- 学習の時刻をホストが書いた時刻から数える決定（[学習の時刻の仮定](#学習の時刻の仮定)）は plan-implementer の判断で、ユーザーの確認が要る。跳びの連鎖を入れたので、入力を見た時刻から数える方式に戻す選択肢もある（非同期のシミュレーションではこちらの prequential 精度が 0.3pt 高かった）
- `learn_interval` の安定性の条件（[θ の更新間隔](#θ-の更新間隔)）を外す根本の代案（algorithm-reviewer の 3 回目の監査、未検証）: `Activity` に跳び $\Delta$（または跳ぶ前の $h$）を載せ、上のノードは付け替えで $\mu(h_{\mathrm{in}}) - \mu(h_{\mathrm{in}} - \Delta)$ だけ動く。$h_{\mathrm{in}}$ のドリフトが付け替えに入らないので、学習の間隔に依らなくなる。forward が 1 回増える
- 非同期の実行で読む値の古さ: 最も遅いノードが周期の大半を計算に使うと、MNIST の example が発散した（[dt が伸びたとき](#dt-が伸びたとき)。跳びの連鎖を入れてからはスケール 2 でも 1 epoch 完走した）。負荷の下では、あるノードが 100 ms ほど走れない間に、隣のノードがそのノードの最後のメッセージで `learn_interval` ごとに学習を続け、$\theta$ が同じ向きに走る。跳びの連鎖で 26 本の busy loop と並べた 32 コアでも完走するようになったが、32 本ではスケール 1 で発散し、0.5 でも精度が落ちる（[MNIST の example](#mnist-の-example)）。`max_dt` はこれを防がない。残る代案は algorithm-reviewer の検討 2（学習の時刻を打ち切った `dt` の和で数える、`g` の時刻と自分の `time` の一致を学習の条件にする）。過負荷への対応は範囲外で未着手。また example は最初の予測が有限でないときだけ止まるので、busy 32・スケール 1 のように予測が有限のまま発散する（精度が偶然の水準に落ちる）場合は検出しない（algorithm-reviewer の 3 回目の監査）
