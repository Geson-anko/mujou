# mujou.pc

PC-ALM を `mujou.Node` の上に実装するモジュール `mujou.pc` の設計。式と記号は skill `pc-alm`、連続時間形は [PC-ALM の連続時間系](pc-alm-continuous-time.md)、公開元の方針は [更新手法](update-rules.md) にある。ここには mujou が選んだことと、その理由を書く。選択を決めた実験の数値は [PC の比較実験](research/pc-experiments.md) にある。

## 一般化した定式化

記事の層 $\sigma(W_i h_{i-1})$ を、ノードが持つ任意の微分可能な関数 $f_i(\cdot;\theta_i)$（`nn.Module`）に置き換える。ノードの内側は autograd（BP）、ノードの間は PC-ALM になる。

$$\mu_i = f_i(h_{\mathrm{in}};\theta_i),\qquad r_i = h_i - \mu_i,\qquad \delta_i = r_i + \lambda_i$$

ノードの内側で VJP を 1 回取り、前のノードへ送る値と重みの勾配を同時に得る。

$$g_{\mathrm{in}} = -J_h^\top \delta_i,\qquad g_{\theta} = -J_\theta^\top \delta_i$$

$f = \sigma(Wh)$ なら $J_h^\top\delta = W^\top\varepsilon$、$J_W^\top\delta = \varepsilon h^\top$ となり記事の式に戻る。次のノードから届く値の和を $g_i$ とすると、連続時間形は次のとおり。

$$\tau_h \dot h_i = -(\delta_i + g_i),\qquad \tau_\lambda \dot\lambda_i = r_i$$

$\theta_i$ は、平均の勾配 $g_\theta / B$ を受け取るノードの optimizer が動かす（「重みの optimizer」）。既定の形は SGD の勾配流 $\tau_\theta \dot\theta_i = -g_\theta / B$。

- 受け取る $g_i$ は全体の損失の勾配ではなく、次のノードのエネルギー項の偏微分。平衡点（$r = 0$）で $\lambda_i = -\partial\ell/\partial h_i$ となり BP の credit に一致する
- `h_in` はスロットから読んだコピーなので、計算グラフは葉で切れる。ノードをまたぐ `.backward()` は無い
- $h$ が準平衡なら、$\lambda$ の値によらず $\delta_i = J_{i+1}^\top\delta_{i+1}$（BP の漸化式）が成り立つ。$\lambda$ の役割は $r = \delta - \lambda$ を 0 に寄せ、ヤコビアンを forward の点で評価させること。$\alpha = 0$（PC）では $r = \delta \ne 0$ のまま止まり、ずれた点で評価した勾配は縮む

任意の $f$ への一般化、損失を持つ出力ノードの分離、次節の状態の持ち方は、記事の範囲外にある mujou の拡張（skill `pc-alm` 7 節）。

## 状態は h で持ち、跳びだけ付け替える

自由変数は記事と同じく活動 $h_i$ と双対 $\lambda_i$ で持ち、記事の $h$ の力学に従う。前のノードの緩和で $h_{\mathrm{in}}$ が連続的に動いて $\mu_i$ が変わっても、$h$ は緩和で追うだけ。$\mu_i$ が**離散的に跳ぶ**ときだけ、$h$ を 1 回動かして $r$ と $\lambda$ を持ち越す。跳びは 2 種類ある。

1. **入力側の跳び**: 受け取った活動の `time` が変わった更新（新しいサンプル）と、`jumps` が変わった更新（前のノードの $h$ が跳んだ）。前回の予測 $\mu^{\mathrm{last}}$ から $h \leftarrow \mu^{\mathrm{new}} + (h - \mu^{\mathrm{last}})$ と付け替える。両方が同じ更新で重なっても 1 回だけ。その更新の $\delta$ は $(h - \mu^{\mathrm{last}}) + \lambda$
2. **自分の $\theta$ の跳び**: $\theta$ を更新した更新で、同じ $h_{\mathrm{in}}$ に対して $\mu' = f(h_{\mathrm{in}};\theta_{\mathrm{new}})$ を勾配なしで 1 回計算し、$\Delta = \mu' - \mu$ を持つ。**次の更新の最初に** $h \leftarrow h + \Delta$ とし、$\mu^{\mathrm{last}} = \mu'$ にしておく。動かすのは $\theta$ による跳びだけで、その間の $h_{\mathrm{in}}$ のドリフトは $h$ に乗らない

どちらも $r = h - \mu$ を保つので、持ち越すのは予測からのずれ $r$ と $\lambda$ だけになる。

- 不動点（$r = 0$、$\lambda = -g$）は記事と同じ
- 教師が無ければ $r$ は 0 に減衰し、$h$ は forward pass になる。$r = 0$ のまま入力が切り替わると、付け替えで新しい forward pass に乗る
- 入力が変わってもリセット（$h$ を forward、$\lambda$ を 0 に戻す）は要らない。付け替えれば $h$ は新しい予測に乗り、持ち越すのは credit 程度の大きさの $r$、$\lambda$ だけになる
- $\Delta$ を足すのを 1 更新遅らせるのは、次のノードが自分の学習の前に跳びを読む確率を下げるためである（同じ更新で足すと、非同期の MNIST で発散した）

ずれ $u = h - \mu$ を状態にする形は採らない。$u_i$ を動かすと次以降のすべての $h$ が即座に動くので、損失のヘッセ行列が順伝播のヤコビアンの積を通じて $u_i$ に返り、深い鎖で連続時間の系として発散する（ResBlock 6 ノード以上の鎖）。

### 跳びの連鎖（jumps）

ノードは $h$ と一緒に「$h$ が離散的に跳んだ回数」`jumps` を送る。ノードは次の更新で `jumps` を 1 進める。

- 新しいサンプルか、受け取った `jumps` の変化で付け替えた
- 自分の $\theta$ の跳び $\Delta$ を足した。ただし学習を間隔を空けて行う（`learn_interval` が `None` でない）ときだけ

同じ更新で重なっても 1 進めるだけ。`Input` はホストの書き込みのたびに進める。受け取ったノードはそれを見て自分も 1 回付け替え、自分の `jumps` を進める。前のノードの $\theta$ による $h$ の跳びが、経路に沿って $r$ を保ったまま次へ伝わるので、次のノードがいつ学習しても、跳びが $r$ に入った $\delta$ で学習することは無い。

- `learn_interval=None`（毎回学習する）では、$\theta$ は毎回の更新で $\Delta t$ に比例して連続的に動く。自分の $h$ には $\Delta$ を足すが `jumps` は進めず、次のノードに連鎖させない。毎回連鎖させると次のすべてのノードが毎回付け替えることになり、$u$ の形に戻る
- `jumped`（足すべき $\Delta$ があるか）のフラグは $\Delta$ が非零かどうかでは代用しない。勾配が 0 の学習でも次のノードに付け替えさせ、参照実装の Jacobi 更新と一致させるため
- `Output` は活動を持たないので `jumps` を読まない

## 符号・バッチ・損失の規約

- 前のノードへ送る値は、自分のエネルギーの $h_{\mathrm{in}}$ に対する勾配。複数の相手からの値は足すだけ
- バッチは shape `batch`（型は `mujou.types.Size`、`type Size = int | tuple[int, ...]`。既定は `()`）で、メッセージのテンソルはどれもこれで始まる（スカラーの `jumps` と `time` はバッチを持たない）。`()` はバッチの無い 1 サンプル、int の `B` は `(B,)`、`(E, A)` のような多次元も取る。$B$ はサンプルの数 `math.prod(batch)`（`()` なら 1）。既定を `()` にしたのは、mujou が狙うストリームの学習ではサンプルを 1 つずつ受け取るのが普通で、ダミーの次元 `(1, d)` を書かせないため（2026-10-21、ユーザーの決定）。多次元のバッチは平坦化した `(E*A,)` と同じ更新になる
- バッチは `activity_spec(h, y, batch)`、`Input(batch=)`、`Chain(batch=)` で宣言する。`Layer` と `Output` は受け取る `act_spec` の `mask` の shape から読み、引数を持たない。どの次元がバッチかは shape からは決まらないので、推論しない
- VJP はバッチの和で取り、$1/B$ は $\theta$ の勾配にだけ掛ける。局所のエネルギーを `.mean()` で作ると $h$ の勾配まで $1/B$ 倍になるので使わない。この規約で $\lambda$ はサンプルごとの credit $-\partial\ell_b/\partial h_b$、$\theta$ の更新は平均の損失の勾配降下になる
- 出力ノードの損失は**サンプルごとの値** $\ell_b$（shape `batch`）を返す関数とし、和はノードが取る。平均を取る損失（`nn.MSELoss()` の既定など）を渡すと $g_{\mathrm{in}}$ が $1/B$ 倍、$\theta$ には $1/B$ が二重に掛かるので断る（K27）。バッチが `()` のときは損失も `()` なので見分けられないが、平均を取るバッチが無いので $1/B$ の問題は起きない
- 損失は `pc.loss.MSELoss(batch)` と `pc.loss.CrossEntropyLoss(batch)`。中身は `pc.functional.mse(y_hat, y, batch)` と `pc.functional.cross_entropy(y_hat, y, batch)` で、クラスは `batch` を持つだけ。torch の `F.mse_loss` と `nn.MSELoss` の関係に倣う。どちらもバッチでない次元（1 サンプルの中）で平均を取り、$\tfrac12$ は付けない。$1/B$ と合わせると、$\theta$ の更新は `nn.MSELoss()`・`nn.CrossEntropyLoss()` の既定（全要素の平均）の勾配降下と一致する。交差エントロピーのクラスの次元はバッチの直後（`len(batch)` 番目）。torch の「クラスは dim 1、バッチが無ければ dim 0」を一般化したもの。1 サンプルの目標が構造を持つ（画素ごとのクラス、`(*batch, C, H, W)`）ときに畳む次元もバッチから決まるので、交差エントロピーもバッチを受け取る
- 以前の `pc.mse` は $\tfrac12\|y-\hat y\|^2$ を特徴で和にしていた。出力の次元が $d$ なら、今の勾配はその $2/d$ 倍（$d = 2$ で同じ、$d = 1$ で 2 倍）。出力側の結合が強くなる分、陽的 Euler の安定の余裕は減る
- `f` はサンプルごとに独立であること。学習モードの BatchNorm のようにバッチの中で混ざる演算は使えない。これはバッチのすべての次元にかかる。系列の時間のように module が混ぜる次元はバッチではなく、`h` の次元に置く
- `f` は決定的で、自分の状態を持たないこと（Dropout や学習モードの BatchNorm は使えない）。学習した更新で $\theta$ の跳びを求めるため、同じ $h_{\mathrm{in}}$ で `f` をもう 1 回計算する

## メッセージ

| 名前 | フィールド | 流れ |
| --- | --- | --- |
| `Sample` | `x`、`y`、`mask` | ホスト → `Input.prev` |
| `Activity` | `h`、`jumps`、`y`、`mask`、`time` | `Input` / `Layer` の `next` → 次のノードの `prev` |
| `Prediction` | `y_hat`、`y`、`mask`、`time` | `Output.next` → ホスト |

- `time` は、その入力をホストが書いた時刻（`write` が返した `Timestamp`、`TimestampSpec()`）。書き込みの前は `Timestamp(ns=0)`。`jumps` は `int`（`IntSpec()`）
- 目標 `y` と `mask` は入力に付随させ、活動と同じメッセージで運ぶ。各ノードは受け取った `y`、`mask`、`time` をそのまま次のメッセージに写す。読み手は 1 回の読み出しで同じサンプルの組を得るので、`Output` は常に、受け取った活動と同じメッセージの `y` と予測を組む。$y$ は $x$ と同じ経路を同じ遅れで進むので、ホストが伝播より速く次のサンプルを書いても組み違えない。別のポートで運ぶと、読み手は 2 つのスロットを別々の瞬間に読むので、別のサンプルの組になりうる
- 教師なしのサンプルは `mask` を False にする。損失は $\sum_b \mathrm{mask}_b\,\ell_b$ で、1 つのバッチにラベルのあるサンプルと無いサンプルが混ざってよい。学習モードと推論モードの区別は持ち込まない
- 1 ホップごとに `y`、`mask`、`time` をコピーするコストは、分類（`y` が `batch`）では h より小さく、回帰（`y` が `(*batch, d)`）では h と同じ大きさになる（[性能の基準値](research/benchmarks.md)）
- `act_spec` は、ノードが送る `Activity` の spec。`Input` と `Layer` が公開し、次のノードが `act_spec=` で受け取る。名前を `act` にしないのは、研究者が `act` をまず活性化関数と読むから。バッチは `act_spec` の `mask` の shape から読む。自作のノードは `activity_spec(h, y, batch)` か前のノードの `act_spec` で繋ぐ。h と y がバッチで始まることの照合は `activity_spec` の 1 か所だけで行う（K28）。`Layer` は自分の出力の `act_spec` を `activity_spec` で作るので、module の出力がバッチで始まることの検査も同じ 1 か所を通る

| ノード | ポート | 受け取る | 送る |
| --- | --- | --- | --- |
| `Input` | `prev`（`Port`） | `Sample`（ホストが書く） | — |
| `Input` | `next`（`MultiPort`） | $g$（読まない） | `Activity`（`h = x`） |
| `Layer` | `prev`（`Port`） | `Activity` | $g_{\mathrm{in}}$ |
| `Layer` | `next`（`MultiPort`） | $g$（相手ごと、和を取る） | `Activity` |
| `Output` | `prev`（`Port`） | `Activity` | $g_{\mathrm{in}}$ |
| `Output` | `next`（`Port`） | — | `Prediction` |

`Input` は、ホストが書く前（時刻 `None`）には `time` 0、`jumps` 0、`mask` がすべて False のゼロの `Activity` を送る。`Input` の `period` は必須で、既定を持たない（`Input` には時定数が無い）。

## 状態

状態の属性は `_` の無い意味のある名前にし、そのまま状態の木のキーになる。

| ノード | 状態 |
| --- | --- |
| `Layer` | `module`、`optimizer`（optimizer の束縛。`pc.optim.AdamW` なら `m`、`v`、`c_m`、`c_v`）、`relaxer`（relaxer の束縛。`pc.relax.HeavyBall` なら `velocity`）、`h`、`lam`、`mu`（前回の予測）、`jump`（次の更新で足す $\theta$ の跳び）、`jumped`、`jumps`、`seen_jumps`、`clock`（`LearnClock`） |
| `Output` | `module`、`optimizer`、`clock` |
| `Input` | `jumps`、`seen_write`（最後に見た書き込みの時刻、書き込みの前は -1） |

$\delta$ は状態にせず、更新のたびに今の `h_in` から $\mu$ を計算し直して作る。$\sigma'$ をキャッシュで評価しない（skill `pc-alm` の落とし穴 2）ことが、これで自動で守られる。

## on_update の段

`Layer` の 1 回の `on_update(dt)` は、段ごとの private なメソッドに分けてある。

1. 活動を読み、`clock.tick` を呼ぶ（`dt` を `max_dt` で打ち切る）
2. 前の更新で $\theta$ を動かしていれば、$h \leftarrow h + \Delta$（`_take_jump`。加算は無条件に行い、$\Delta$ は足した後に 0 にする）
3. $\mu$ と $g_{\mathrm{in}}$、学習するなら $g_\theta$ を、**更新前の** $\delta$ で取る。`mask` が False の行の $\delta$ は 0。$g_{\mathrm{in}}$ を送る（`_predict`）
4. 新しい入力か `jumps` の変化なら $h$ を $\mu$ に付け替え、次のノードから届いた $g$ で $h$ を進め、続けて**更新後の** $h$ の $r$ で $\lambda$ を進める（`_relax`。1 歩の陽的 Euler、公式実装と同じ順）
5. `jumps` を進めるなら進め、`Activity` を送る（`_send`）
6. 学習するなら $\theta$ を動かし、$\Delta$ を求める（`_learn`）

- `mask` が False の行の $\delta$ を 0 にするのは、その行の $\delta$ が前の入力から持ち越した値かもしれず、そのまま流すと前のノードの credit と $g_\theta$ に雑音が乗るからである。その行の不動点はもともと $\lambda = 0$ なので、不動点は変わらない
- ResBlock を 1 つの `Layer` にするなら $f(h) = h + g_\theta(h)$ を渡す。スキップ接続と中間の活性はノードの内側で BP される。ノードをまたいで分けるなら、合流するノードが入力ごとにポートを持ち、VJP を入力ごとに返す。細かく分けるほど並列度と局所性が上がり、$h$、$\lambda$ のメモリと緩和の時間が増える。モデル全体を 1 ノードにして出力ノードをつなぐとオンラインの BP になる

`Output` は $\hat y = f(h_{\mathrm{in}})$ と $\ell = \sum_b \mathrm{mask}_b\,\ell_b(\hat y_b, y_b)$ を微分して $g_{\mathrm{in}}$ と、学習するなら $g_\theta$ を得る（`functional.loss_vjp`）。教師の有無の分岐は `loss_vjp` の mask の処理に一本化した。mask が False の行の $g_{\mathrm{in}}$ は $J^\top 0$ で、$\hat y$ が有限なら厳密に 0、$\hat y$ が inf か NaN のときだけ NaN になる（そのときは出力がすでに壊れている）。学習の条件は `Layer` と同じ `tick.learn and mask.any()`。

## 学習の時刻（LearnClock）

$h$、$\lambda$ は毎回の更新で動かし、$\theta$ は間隔を空けて動かす。学習するときは、前回の学習からの経過時間 $\Sigma\Delta t$（`Tick.elapsed`）を重みにする。

$$\theta \leftarrow \theta - \frac{\Sigma\Delta t}{\tau_\theta}\cdot\frac{g_\theta}{B}$$

（`pc.optim.SGD` の場合。一般には optimizer の `step_` が $g_\theta / B$ と $\Sigma\Delta t$ で動かす）。時間あたりの学習の速さは optimizer の $\tau_\theta$ で決まり、間隔は $g_\theta$ を**どの時点でサンプルするか**と計算量を決める。

学習の時刻は `LearnClock(*, period, learn_interval, max_dt)` が決める。`Layer` と `Output` の 2 人の使い手がいて、PC 以外の則でも使える（`examples/feedback_alignment.py`）ので、時定数を受け取らない部品として公開した。PC のノードの組み立ては `LearnClock.of(tau, *, period, learn_interval, max_dt=None)` の 1 か所にまとめる（`max_dt` の既定 `tau.max_dt`）。周期の既定 `tau.period` と `max_dt` の既定は、根拠とともに `TimeConstants` の property に置く。`tick(dt, time, now)`（どちらも `Timestamp`） は 1 回の更新を数える純粋な遷移で、`Tick(dt, new, learn, elapsed)` を返す。

- ホストが入力を書いた時刻（`time`）から `learn_interval` ごとの時刻 `time + k * learn_interval`（$k = 1, 2, \dots$）を過ぎた最初の更新で学習する。ただし、その入力を初めて見た更新では学習しない。その更新の $\delta$ は、前の入力の $r + \lambda$ に新しい入力のヤコビアンを掛けたもので、新しい入力の credit ではない
- `learn_interval=None` は毎回学習する（連続時間形そのもの。新しい入力を見た更新でも学習する）。`math.inf` は学習しない（学ばないノードの表し方はこの 1 通り）。格子の点が無限の先にあるので、学習の時刻に達しない
- `learn` を返した時点で学習したものとして数え直す（$\Sigma\Delta t$ と次の格子）。ノードが mask の無い入力で学習を飛ばしても同じで、その時間は捨てる。前の入力から持ち越した $\delta$ で学習しないため
- 学習しないまま入力が切り替わったら（入力が `learn_interval` より速く変わる）$\Sigma\Delta t$ を捨てる。溜めたままにすると、入力が止まった最初の学習にまとめて掛かり、学習率が際限なく跳ね上がる
- $\Sigma\Delta t$ は `max_dt` で打ち切った `dt` の和。学習の時刻は実行の時間で決まる（打ち切った時間も含む）
- 学習した時刻は `tick` に渡した `now_ns` で記録する。手動モードでは学習の後に読んだ時刻と同じで、実時間では更新の途中で格子の点を越えたときだけ 1 区間ずれうる
- `ManualRun` を固定の `dt` で回すと、「入力を書いてから $K$ ステップごと」と同じになる

学習の時刻を「ノードが入力を見た更新から」ではなく「ホストが書いた時刻から」数えるのは、経路上のノードの学習をそろえるためである。入力を見た更新から数えると、深いノードほど 1 更新ずつ遅れて学習し、前のノードの $\theta$ の更新による $h$ の跳びが $r$ に入った $\delta$ で学習してしまう（非同期の更新順で発散した）。この方式は次を仮定する。

- 1 つの実行の全ノードが同じタイマーを読む（「1 つの実行の時間軸は 1 本」）。読むのはメッセージの `time` と実行のタイマーだけなので、非局所な読み取りでもバリアでもない
- 経路上のノードの学習が、1 更新程度の差でそろう。[跳びの連鎖](#跳びの連鎖jumps) がこの仮定を緩め、学習のずれで揺れるのは、学習に使う $\delta$ がどこまで緩和したかだけになる
- 1 つの経路のノードが同じ更新ステップで学習することは、`ManualRun` で全ノードを毎ステップ回すとき（どの順でも）に限って保証される

### learn_interval の目安

`learn_interval` は、credit が経路を往復する時間（2 × ホップ数 × period。ホップ数は Input から Output までの接続の数）より長くすることを勧める。検査はしない。既定値は置かない（`Layer`・`Output`・`Chain` の必須のキーワード）。毎回学ぶ `None` はストリーム向けで、よい値をまだ測っていないので、省いたときに黙ってそのモードで走らせない。

間隔が短いと各ノードがほぼ毎回の更新で学習して `jumps` が進み、次のノードが毎回付け替えるので、跳びの連鎖が $u$ の形に戻る。ResBlock 8 ノードの鎖（往復 18 周期）での実測では、`None` と 16 周期は credit に収束し、4 周期は有界だが credit の約 4 倍、発散したのは 2 周期と 1 周期だけだった。2 × ホップ数 × 周期は安定の境界ではないので、それより短い値を断ると安定な設定まで拒むことになる。

## 不変条件

- mask がすべて False の `Activity` では、どのノードも optimizer の `step_` を呼ばない（$\theta$ が動かない。$h$ と $\lambda$ は動く）。テストは全ノードの $\theta$ のビット一致で確かめる
- 書き込みの前の `Activity` の `time` は 0、`LearnClock` の初期値（最後に見た入力の時刻）は -1。最初の `Activity`（ゼロ、mask がすべて False）は新しい入力として見られ、学習の格子を 0 から始める。時刻 0 に書いたサンプルも同じ格子に乗り、`Input` の `jumps` が変わるので付け替えも起きる。学習するかどうかは mask で決まるので、ゼロの `Activity` では学習しない

## 時定数

`TimeConstants(*, h, lam)` に秒で持つ（K25: h は有限で > 0、lam は > 0。lam の inf は可）。$\tau_\theta$ は optimizer が持つ（「重みの optimizer」）。$h$ の動かし方は relaxer が持つ（「h の relaxer」）。

- $\alpha = 0$（PC）は `lam=inf` で表す。`dt / tau.lam` が 0 になるので、レートを別に持たない。`relax_` は lam が inf なら $\lambda$ に触れないので、$\lambda$ はちょうど 0 のまま
- `pc.optim.SGD` での 1 回の学習は、平均の勾配に対する学習率 $\Sigma\Delta t/\tau_\theta$ の SGD に当たる
- ノードの周期の既定は $\tau_h / 4$（$\eta_h = 0.25$）。利用者は時定数だけを決めればよい。記事の公式実装の $\eta_h = 1/\lambda_{\max}$（約 0.2〜0.25）とほぼ同じ
- 安定な範囲の実測（$\eta_h \le 0.5$ で安定、臨界減衰は $\alpha \approx c^2\eta_h/4$、$c \approx 1 + \|J\|^2$）は [PC の比較実験](research/pc-experiments.md#時定数の安定な範囲)。example の既定は $\tau_\lambda = 2\tau_h$（$\alpha = 0.125$）

## 重みの optimizer

$\theta$ の時間発展は差し替えられる部品にする（`src/mujou/pc/optim.py`）。

- **設定 + bind**: `pc.optim.SGD(tau=)` と `pc.optim.AdamW(tau=, tau_m=, tau_v=, eps=1e-8, weight_decay=1e-2)` は値だけを持つ frozen dataclass で、ABC の `pc.optim.Optimizer` を継承する。状態を持たないので `Chain` の全ノードで共有してよい。ノードは `optimizer.bind(params)` で自分の param の束縛（ABC の `pc.optim.BoundOptimizer`）を作り、`self.optimizer` に置く。束縛は `step_(grads, elapsed)` と `state_dict` / `load_state_dict` を実装し、`mujou.Stateful` なので `LearnClock` と同じく属性に置くだけでノードの状態に入る（キー `"optimizer"`）。束縛はノードごとに別で、同じ束縛を 2 ノードに置くと `tree.walk` が断る（K24）
- ABC にしたのは、メソッドを書き忘れた自作の optimizer を、最初に学ぶ更新（実行中の hook の例外で全体が止まる）ではなく、作った時点の `TypeError` で知らせるため（2026-10-23、ユーザーの決定。最初は Protocol だった）。mujou はダックタイピングを使わない（[ノード](nodes.md#状態)）
- `optimizer` は `Layer`・`Output`・`Chain` の必須の引数。時間で表した既定の学習率は無い（`learn_interval` に既定を置かないのと同じ理由）
- ノードは `requires_grad` の param だけを bind する。frozen な param には weight decay も掛からない
- **$1/B$ の位置**: ノードが $g_\theta$ を $B$ で割ってから `step_` に渡す。optimizer はバッチを知らない
- **`elapsed` の意味**: その歩みが重みとする時間（`Tick.elapsed`。前に学んでからの、`max_dt` で打ち切った dt の和で、時計が捨てた時間を含まない）。前回の `step_` からの実時間とは限らない。捨てた時間は AdamW のモーメントと $c$ も動かさず（流れが止まっている扱い）、weight decay もラベルのある時間にだけ掛かる

### 連続時間形

$g = g_\theta / B$ として、

- SGD: $\tau_\theta \dot\theta = -g$
- AdamW: $\tau_m \dot m = g - m$、$\tau_v \dot v = g^2 - v$、$\tau_\theta \dot\theta = -\left(\hat m / (\sqrt{\hat v} + \varepsilon) + w\,\theta\right)$
- 質量（bias 補正）: $\tau_m \dot c_m = 1 - c_m$、$\tau_v \dot c_v = 1 - c_v$（$c(0) = 0$）。$\hat m = m / c_m$、$\hat v = v / c_v$。ステップのカウンタの代わりに、moment が集めた重みを時間で持つ

### 離散化（`elapsed` = $\Delta t$ の 1 歩）

SGD は `p.sub_(g, alpha=elapsed / tau)`。状態は空。AdamW は torch の `_single_tensor_adam`（decoupled weight decay、非 capturable）と同じ順にする。

1. `elapsed == 0` なら何もしない（0 秒の流れは恒等写像。初回の前は $c = 0$ で $\hat m = 0/0$ になる。`ManualRun.step(0.0)` や `learn_interval=None` で起こりうる）
2. $a = -\mathrm{expm1}(-\Delta t/\tau)$（$1 - e^{-\Delta t/\tau}$ を小さな $\Delta t$ でも精度よく）を $m$ と $v$ それぞれで取り、$c \leftarrow c + a(1 - c)$
3. $\eta = \Delta t/\tau_\theta$。param ごとに $\theta \leftarrow \theta(1 - \eta w)$（decay を先に）、$m \leftarrow m + a_m(g - m)$、$v \leftarrow v + a_v(g^2 - v)$、$\theta \leftarrow \theta - \frac{\eta}{c_m}\cdot\frac{m}{\sqrt{v}/\sqrt{c_v} + \varepsilon}$（ε は $\sqrt{\hat v}$ に足す）

$m$、$v$、$c$ は $g$ を $\Delta t$ の間一定とした流れを厳密に解くので、同じ $g$ なら $\Delta t$ を 2 回と $2\Delta t$ を 1 回で同じ値になる。$\theta$ は torch と同じ順で、decay は歩みの前の $\theta$ に陽的に掛け、Adam の項はその歩みで更新した後の $m$・$v$・$c$ で取る（$\Delta t \to 0$ で 1 次の精度。decay を $e^{-\eta w}$ にする厳密な案は torch と一致しなくなるので採らない。$\eta w > 1$ で符号が反転するが、`elapsed` は `learn_interval` 程度で $\eta$ は小さい）。

$\Delta t$ が一定なら $\beta = e^{-\Delta t/\tau}$、$c = 1 - \beta^t$ となり、`torch.optim.AdamW(lr=Δt/τθ, betas=(e^{−Δt/τm}, e^{−Δt/τv}), eps=ε, weight_decay=w)` と丸めの範囲で一致する（テストのオラクル）。torch の β からの換算は $\tau_m = -\Delta t/\ln\beta_1$（32 ms に 1 回学ぶなら β1 = 0.9 で約 0.30 s、β2 = 0.999 で約 32 s）。

### 決めたこと

- **w の単位**: 無次元で $1/\tau_\theta$ と組む（decay の時定数は $\tau_\theta / w$）。torch の `weight_decay` と同じ値が同じ意味になる。独立の時定数にする案は、torch との対応が崩れ、学習率を変えると decay の比も変わる AdamW の decoupled の定義と合わないので採らない
- **c は Python の float**: `torch.load(weights_only=True)` で読め、復元は代入で同じ値になる。torch の非 capturable の bias 補正も倍精度の Python の float。0-d テンソルにすると dtype を選ぶ必要があり、float32 では 1 に近い c の精度が落ちる。倍精度では約 $37\tau$ で $c$ がちょうど 1.0 になり、そのとき $\hat m = m$ で害はない
- **状態の dtype/device**: $m$、$v$ は `torch.zeros_like(p)`。bind は module が device に載った後（ノードの `__init__`）に呼ぶ
- **`torch.optim` を使わない**: (1) 3.14t ではコンストラクタが `torch._dynamo` → `triton._C` を import し、プロセス全体の GIL が戻る。(2) β が 1 ステップあたりなので、実効の時定数が学習の間隔に依存する。(3) `state["step"]` が float32 で $2^{24}$ 回で増えなくなる。自前の optim は `torch.optim` を実行時に import しない
- 構築時の検査（K31）: `tau`・`tau_m`・`tau_v`・`eps` は有限で > 0、`weight_decay` は有限で >= 0。`eps=0` を断るのは、勾配がずっと 0 の要素（batch 全体で死んだ ReLU の行など、PC でよく出る）が $m = v = 0$ のまま $0/0$ で NaN になり、跳び Δ を通して鎖全体に広がるため（torch は 0 を受け付けるが合わせない）。`tau=inf` を断るのは、学ばないことを `learn_interval=math.inf` の 1 通りで表すため（`tau_m=inf` は $\hat m = 0/0$ になる）。フィールドはキーワード専用
- AdamW の推奨の時定数は未測定（`open-decisions.md`）
- AdamW は勾配の大きさで歩みを縮めない（成分ごとに約 $\eta$ 動く）。そのため、緩和しきらない $\delta$ の残差が、BP の勾配がほぼ 0 の成分でも sign 程度の歩みになりうる。SGD なら残差 $e$ による $\theta$ の誤差は $\eta e$ で小さい。`learn_interval` を往復より長くする推奨は SGD より強く効く可能性がある（未測定）

## h の relaxer

$h$ の緩和の仕方は差し替えられる部品にする（`src/mujou/pc/relax.py`、2026-10-26）。きっかけは、$h$ に慣性（heavy ball）を足せるようにした（ユーザーの「それでも学習できるよう新しい feature の実装（緩和の手法設定など）も考慮して試行」）ところ、ユーザーが「$\theta$ の optimizer のように、$h$ にも更新の仕方を自由に変えられる仕組みが必要」と求めたこと。最初は `TimeConstants.momentum` に置いたが、それは慣性という 1 つの手法を時定数の 1 項目にしただけなので、重みの optimizer と同じ形（設定 + bind）に載せ替えた。

- **設定 + bind**: `pc.relax.Euler()` と `pc.relax.HeavyBall(tau=)` は値だけを持つ frozen dataclass で、ABC の `pc.relax.Relaxer` を継承する。状態を持たないので `Chain` の全 `Layer` で共有してよい。`Layer` は `relaxer.bind(h)` で自分の $h$ の束縛（ABC の `pc.relax.BoundRelaxer`）を作り、`self.relaxer` に置く。束縛は `step_(h, force, dt, tau_h)`、`reset_()`、`state_dict` / `load_state_dict` を実装し、`mujou.Stateful` なので属性に置くだけでノードの状態に入る（キー `"relaxer"`）。束縛はノードごとに別（`h` と同じ shape・dtype・device の状態を持つ）
- ABC なのは optimizer と同じ理由（書き忘れを作った時点の `TypeError` で知らせる）。`relaxer` は `Layer` と `Chain` の省略できる引数で、既定は `pc.relax.Euler()`（これまでの挙動）。`Output` は $h$ を持たないので取らない
- **差し替えの範囲は $h$ だけ**: 力 $F = r + \lambda + g$ を作るのは `functional.relax_` で、束縛が $h$ をその力に沿って動かす。$\lambda$ は動かした後の $h$ の $r$ で従来どおり進める。`relax_(h, lam, mu, g, dt, tau, relaxer=None)` の `None` は素の 1 歩
- **`reset_()`**: 新しいサンプルか $h_{\mathrm{in}}$ の跳び（付け替え）のとき、ノードが呼ぶ。付け替えは $r$ と $\lambda$ を保ち、$g$ は下流の最後の値のままなので、その瞬間に力は連続する。それでも、速度が積んだのは前の入力の力の場で、新しい入力のものではないので捨てる。$\theta$ の跳びは $h$ と $\mu$ を一緒に動かし、$r$ も力も連続で、前の入力の力の場が続くので、呼ばない。記憶の無い relaxer は何もしない
- **不動点の条件**: 自作の relaxer は、力が 0 で relaxer 自身の状態が静止しているところ（heavy ball なら $F = 0$ かつ $v = 0$）で $h$ を動かさないものでなければならない。そうでないと不動点が動き、$\lambda$ が BP の credit にならない（$h$ に decay や leak を入れると、$\lambda$ の不動点は credit からずれる）
- **持てる状態の条件**: ノードは `step_` の外でも $h$ を動かす（付け替えの $h \leftarrow \mu^{\mathrm{new}} + (h - \mu^{\mathrm{last}})$ と、$\theta$ の跳びの足し込み）。だから relaxer の状態は $h$ の位置に依らないもの（速度のような力の型）に限る。位置の型の状態（先読みの位置、前回の $h$）は、$\theta$ の跳びで古くなるが、$\theta$ の跳びは relaxer に知らされない。要るようになったら `shift_` のような通知を足す（今は実装しない）
- 局所性: 状態はノードの内側で、メッセージは増えない

### `pc.relax.HeavyBall(tau)`

速度 $v$ を持ち、1 回の更新（$dt > 0$）で

$$
v \leftarrow e^{-\Delta t/\tau}\,v + F,\qquad
h \leftarrow h - \frac{\Delta t}{\tau_h}\,v.
$$

$dt = 0$ の更新は何もしない（力も蓄えない）。状態は `{"velocity": v}`。

- 不動点は変わらない。$F = 0$ では速度も 0 になる。持続する力は $1/(1 - e^{-\Delta t/\tau})$ 倍まで速く $h$ を動かすので、深い鎖で $\lambda$ が credit に届くまでの周期が減る
- 実測（CIFAR-10、ResNet-20 を残差ブロックごとの 11 ノードにして実時間、周期 $\tau_h/4$、$\tau_\lambda = \tau_h$、$\tau_h$ 20 ms）: $\tau = \tau_h/4$（1 更新あたり $e^{-1} \approx 0.37$）で 1 バッチ 64 周期が、`Euler` の 128 周期と 3 epoch 目の精度で並ぶ（prequential 49.5〜51.7% 対 50.7%）。実時間は半分。ただし 10 epoch 目は `HeavyBall` の 64 周期が 62.8%、`Euler` の 128 周期が 68.7%（BP は 73.3%）。128 周期に `HeavyBall` を足しても 3 epoch 目は変わらない（50.2%）。$\tau = \tau_h/3$（0.47）は 3 回に 1 回、学習が崩れた。$e^{-\Delta t/\tau} = 0.8$ は発散。32 周期では `HeavyBall` があっても学習できない
- algorithm-reviewer の監査（承認）: 線形のスカラー鎖（恒等写像、10 ノード、$\alpha = \eta = 0.25$）で更新写像のスペクトル半径を調べると、安定な最大の $e^{-\Delta t/\tau}$ は 0.78（ノード数 2〜20 でほぼ同じ）で、実測の境界と合う。$\alpha$ を大きくすると下がる（0.0625 で 0.94、0.5 で 0.60、1.0 で 0.30）。非同期のジッタ（周期の 1〜2 倍）では約 0.75。慣性は高曲率側の境界（$\sigma$）を広げる（0.47 で $\sigma_{crit}$ 1.94 → 2.58）が、ジッタの下では救えない。速くなるのは過渡（credit の先頭）だけで、恒等写像に近い鎖の遅いモードの漸近の速さは良くならない（0.968 → 0.976）。力の重みは 1 更新あたりで $dt$ に比例しないので、利得は周期に依存する（$dt$ に整合させると、ジッタのある非同期で $\tau = 0.5\tau_h$ で発散した）

### 採らなかった案

- 次のノードの曲率から刻みをサンプルごとに縮める `AdaptiveStep`（凍結した重みでは $\lambda$ が credit に合うが、縮めた行が前のバッチの $\lambda$ を長く引きずって学習が悪化した）
- 新しいサンプルで $\lambda$ と $r$ を捨てる設定（学習が進まなかった）
- $h$ と $\lambda$ を 1 組で差し替える部品: 不動点（$r = 0$、$\lambda = -g$）は、勾配の零点だけを停留点に持つ則なら保てるが、陽的な結合の安定性、跳びでの付け替え（$r$ を保つこと）との整合、サンプルごとの状態（バッチの各行に別の状態）が課題で、今は実装しない。$\lambda$ の差し替えも同じ

## dt の打ち切り

実時間の実行では `dt` が周期より長くなることがある。PC のノードは 1 回の更新で使う `dt` を `max_dt` で打ち切り、$h$、$\lambda$ の緩和にも $\Sigma\Delta t$ への加算にも打ち切った `dt` を使う。既定は $\min(\tau_h/2,\ 3\tau_\lambda)$ で、周期 $\tau_h/4$ の 2 倍。`period > max_dt` は `LearnClock` が断る（K26。毎回の更新が黙って時間を捨てる）。

- ノード間は陽的に結合している（$g$ は次のノードが最後に更新したときの $h$ から作られる）。負荷で全ノードがそろって止まった直後は全ノードの $\Delta t/\tau_h$ が一度に伸び、その結合でネットワークが発散する。ノードの内側で刻みを分割しても、自分の状態の陽的 Euler が安定になるだけで、ノード間の結合は陽的なまま残るので防げない
- 既定の値の根拠: 局所の 2 変数系（$h$ を進めてから、進めた $h$ の $r$ で $\lambda$）の陽的 Euler は、$a = \Delta t/\tau_h$、$b = \Delta t/\tau_\lambda$ として $0 < a < 2$ かつ $ab < 4 - 2a$ で安定。$a \le 1/2$ なら $b \le 3$ で足りる
- ノード間の結合の安定の余裕はモデルに依存し、`max_dt` の既定は一般の保証ではない
- 打ち切ると経過時間が失われ、連続時間の定式化からずれる。コアは打ち切らず（[ノード](nodes.md#実行の周期)）、打ち切るのは PC のノードの選択で、失われた時間の分だけ学習が遅れる。`max_dt` は `dt` の伸びを抑えるが、読む値の古さは抑えない

## Chain と show

`Chain(*modules, x=, y=, loss=, tau=, optimizer=, relaxer=, learn_interval=, max_dt=, device=)` は `Input → Layer → … → Output` の `Network`。`modules[:-1]` が `Layer`、`modules[-1]` が `Output` になる。

- 子のパスは `"input"`、`"layer0"`、…、`"output"`（`modules[i]` が `layer{i}`）。`*modules` は `nn.Sequential` と同じ形なので、番号も 0 から始める
- period はすべて $\tau_h / 4$。period を変えたい利用者は個別のノードで組む
- `prev`（`Sample` を受ける）と `next`（`Prediction` を送る）を公開する
- 名前を `Sequential` にしなかったのは、同じ式の中で `nn.Sequential` と並び、読み違えを生むから。`x` / `y` は `Sample` のキーと一致し、`input` は組み込み関数を隠す

`show(run, chain, sample, *, seconds) -> Prediction` は、`sample` を書き、その書き込みに対する最初の予測（学習する前の、prequential な予測）を返す。返すのは、書き込みから `seconds` の間サンプルを提示し続けた後。`chain` は ABC の `pc.Shown`（`Sample` を受け取る `prev` と `Prediction` を送る `next` を型注釈で宣言する）を取る。`Chain` が継承し、周期や形を変えて手で組んだネットワークも `mujou.Network` とあわせて継承すれば、同じ 1 つのやり方で見せられる。ポートはインスタンスの属性（`__dict__` が唯一のレジストリ）なので、抽象 property にはせず、抽象メソッドの無い名目上の型にした。

- 評価は mask をすべて False にした `Sample` で行う。教師の有無は `Sample` の mask が表すので、`y=None` のような特別扱いは持たない。評価では、伝播に足りる短い `seconds` でよい
- outbox は最新値のスロットなので、ホストが学習の始まりより遅れると、学習を始めた後の予測を読みうる
- 誰も時計を進めないので、`ManualRun` の上では返らない。テストは実時間の run で書く

## 自作のノード

`functional` は状態を持たない数式で、`Layer` と `Output` もこれで書く。

- `vjp(f, h_in, delta, params) -> (mu, g_in, g_theta)`: δ は $\mu$ から作る callable（`Layer` は付け替えの更新で $\mu$ に依存しない δ を、ほかの更新で $r + \lambda$ を作る）。δ と μ の shape が違えば断る（K29）
- `loss_vjp(f, h_in, y, mask, loss, params) -> (y_hat, g_in, g_theta)`: 損失の shape が `mask` の shape（`batch`）でなければ断る（K27）。`Output` の構築時の probe も同じ関数を通る
- `relax_(h, lam, mu, g, dt, tau)`: 1 歩の陽的 Euler
- `mse(y_hat, y, batch=())`、`cross_entropy(y_hat, y, batch=())`: サンプルごとの損失
- `vjp` と `loss_vjp` は骨格 `_grad` を共有し、`torch.autograd.grad` はそこ 1 か所だけにある

手順は、`self.clock = pc.clock.LearnClock.of(tau, ...)` と `self.optimizer = optimizer.bind(学ぶ params)`（`h` の動かし方を選ぶなら `self.relaxer = relaxer.bind(h)` も）を属性に置き（それだけで状態に入る。`max_dt` の既定は `Layer`・`Output` と同じ 1 か所から来る）、毎回の更新で `tick = self.clock.tick(dt, act["time"], self.timer.now())` を呼び、新しい入力か `jumps` の変化を見たら `h` を予測と一緒に動かして自分の `jumps` を進め、`relax_(..., tick.dt, tau, self.relaxer)` で活動を動かし（付け替えたときは先に `self.relaxer.reset_()`）、`h` を自分の `jumps` と組にして送り、`tick.learn` なら `self.optimizer.step_([g / B for g in g_theta], tick.elapsed)` で学ぶ。ほかの pc のノードと繋ぐポートの spec には、前のノードの `act_spec` か `activity_spec(h, y, batch)` を使う。

## 記事の公式実装との違い

| 項目 | 公式実装（SakanaAI/pc-alm） | mujou.pc |
| --- | --- | --- |
| 層 | $\sigma(W h)$ の residual MLP | 任意の微分可能な `nn.Module` |
| 自由変数 | $h$ | $h$。予測が離散的に跳ぶ更新で 1 回だけ付け替える（新しいサンプル、前のノードの $h$ の跳び、自分の $\theta$ の跳び） |
| 送る活動 | $h$ | $h$ と跳びの回数 `jumps`、目標 `y`・`mask`・`time` |
| ミニバッチの始め | $h$ を forward の値、$\lambda$ を 0 に初期化 | $h$ を予測に付け替え、$r$、$\lambda$ を持ち越す |
| 推論と学習 | $T$ ステップ推論してから重みを 1 回更新 | 同時に進む。入力を書いてから `learn_interval` 後に重みを更新 |
| 重みの更新の後 | 次のミニバッチで $h$ を forward に初期化 | 次の更新で $h$ に $\theta$ の跳びを足し、経路に沿って連鎖させる |
| $\lambda$ の更新 | 更新後の $h$ の $r$ | 同じ |
| 重みの勾配の $\lambda$ | 最後の dual 更新の前 | VJP を取ったときの更新前の値 |
| ステップの大きさ | 離散の $\eta_h, \alpha$ | 時定数と実経過の `dt`（`max_dt` で打ち切る） |
| 重みの最適化 | Adam | 時間で進める SGD か AdamW（`pc.optim.SGD` / `pc.optim.AdamW`） |
| 出力の損失 | バッチ平均の $\tfrac12\|y-\hat y\|^2$ | サンプルごとの任意の損失の和（`MSELoss` は特徴の平均で $\tfrac12$ なし） |

## テスト

`tests/mujou/pc/` に置き、float64 と `mujou.testing.run` で決定的に回す。損失が下がったかでは判定しない。

1. `functional.vjp` の $g_{\mathrm{in}}$、$g_\theta$ が、$E_i = \tfrac12\|h_i - \mu_i + \lambda_i\|^2$ と $\sum_b \ell_b$ の有限差分と一致する
2. 出力側から回す同期の step が、純粋な PyTorch で書いた参照実装（`_reference.py`。付け替え・$\theta$ の跳び・`jumps` の連鎖・学習の時刻を含む）の Jacobi 更新と 1e-12 で一致する。`learn_interval=None` と、間隔を空けて学習する場合の両方
3. 入力を固定して回した不動点で、$h_i$ が forward の値、$\lambda_i$ がネットワーク全体を autograd した $-\partial\ell/\partial h_i$ と一致し、$\theta$ の 1 回の更新が SGD と一致する（線形と tanh、Linear と ResBlock）
4. $\alpha = 0$（`lam=inf`）で $\lambda \equiv 0$、不動点で $r = -g$、$h$ が forward の値からずれる（PC）
5. 出力側から・入力側から・固定 seed のランダム順で回しても同じ不動点に収束する
6. 教師が無いとき、$h$ は forward の値で、$\lambda = 0$、$\theta$ は動かない。mask がすべて False の間は全ノードの $\theta$ がビット一致で動かない
7. `Output` は常に、受け取った活動と同じメッセージの $y$ を使う。`mask` が False のサンプルは損失に入らず、`Layer` もその行の credit を返さない
8. `LearnClock` を、仕様の式から書いた期待値で検査する（`test_clock.py`）
9. 状態から復元すると、同じスケジュールで同じ値になる
10. ResBlock 8 ノードの鎖を `learn_interval=None` と 16 周期で回すと、$\lambda$ は credit の 2 倍を超えず、credit に収束する
11. 2 つの制約を 1 ノードにまとめた鎖と、2 ノードに分けた鎖で、不動点での $\theta$ の更新が一致する（ノードの分け方に依らない）
12. 時刻 0 に書いた最初の入力を学ぶ
13. `Chain` が手組みと `torch.equal` で一致する。`show` が書き込みに対する最初の予測を返す（実時間の run）
14. optimizer（`test_optim.py`）: SGD が `p - (elapsed/τ) g` と一致する。AdamW が Δt 一定で手で書いた torch の AdamW の式と 1e-12 で一致し、初回の 1 歩が閉じた形 $\theta(1-\eta w) - \eta g/(|g|+\varepsilon)$ になる。Δt を 2 回と 2Δt を 1 回で m・v・c が一致し、c が $1 - e^{-\sum\Delta t/\tau}$ になる。状態の復元で同じ値になる。`torch.optim.AdamW` との一致は subprocess の 1 本（slow）で確かめ、自前の optimizer が 3.14t の GIL を戻さないことも subprocess で確かめる
15. relaxer（`test_relax.py`）: `Euler` が `h - (dt/τ_h) F` と一致する。`HeavyBall` が $v \leftarrow e^{-\Delta t/\tau}v + F$、$h \leftarrow h - (\Delta t/\tau_h)v$ の式と 1e-14 で一致し、最初の 1 歩は `Euler` と同じ。$dt = 0$ は何も動かさず速度も蓄えない。`reset_` で速度が 0 になる。$\tau$ が小さい極限は `Euler`。一定の力で速度が $1/(1-e^{-\Delta t/\tau})$ に収束する。状態の復元で同じ値になり、状態は束縛とメモリを共有し、読み込みはコピーする。`Layer` では、`HeavyBall` でも $\lambda$ が BP の credit に一致し、credit の先頭が `Euler` より速く届く。新しいサンプルで `reset_` が呼ばれ、$\theta$ の跳びでは呼ばれない。`Chain` は各 `Layer` に別々の束縛を持たせる
