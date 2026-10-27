---
name: pc-alm
description: mujou の既定の更新手法 PC-ALM（Augmented Lagrangian Predictive Coding）の理論リファレンス。制約付き最適化としての定式化、拡張ラグランジアン、活動の primal descent・双対の dual ascent・重み更新の具体式（勾配展開済み）、平衡点で BP と一致する条件、ハイパーパラメータ、安定性、実装時の落とし穴チェックリスト。学習則・推論ダイナミクス・レイヤーの更新式を設計・実装・レビュー・テストする前に読む。
---

# PC-ALM 理論リファレンス

出典: Sakana AI, "PC-ALM: Augmented Lagrangian Predictive Coding" (<https://pub.sakana.ai/pc-alm/>)。
PC-ALM は mujou の**既定の更新手法として暫定的に採用したもの**で、更新手法は差し替えられる（skill `locality-and-async`）。この skill は PC-ALM を使うノードについての文書である。

この skill の式は記事の定義に沿って勾配まで展開したもの。**記事に無い mujou 独自の拡張は「mujou の研究課題」節に分けて書き、理論と混ぜない。**

## 1. 定式化

深さ $L$、入力 $h_0 = x$、教師 $y$、重み $\theta = \{W_i\}$、活性化 $\sigma$。活動 $h_i$ も最適化変数として扱う。

$$\min_{\theta,h}\ \tfrac12\|y - W_L h_{L-1}\|^2 \quad \text{s.t.} \quad h_i = \sigma(W_i h_{i-1}),\ i = 1,\dots,L-1$$

記号（以降すべてこれで書く。コードの変数名もこれに揃える）:

| 記号 | 定義 | 意味 |
| --- | --- | --- |
| $a_i$ | $W_i h_{i-1}$ | 層 $i$ の pre-activation |
| $r_i$ | $h_i - \sigma(a_i)$ | 層 $i$ の予測誤差（制約違反）。**向きは「活動 − 予測」** |
| $e_L$ | $y - W_L h_{L-1}$ | 出力誤差。**向きは「教師 − 予測」** |
| $\lambda_i$ | $\in \mathbb{R}^{n_i}$ | 層 $i$ の双対変数（ラグランジュ乗数）。$h_i$ と同じ次元 |
| $\varepsilon_i$ | $\sigma'(a_i) \odot (r_i + \lambda_i)$ | 層 $i$ が下の層へ返す誤差信号（実装上の中間量） |

## 2. 目的関数

PC の自由エネルギー:

$$F_{PC} = \tfrac12\|e_L\|^2 + \tfrac12\sum_{i=1}^{L-1}\|r_i\|^2$$

PC-ALM の拡張ラグランジアン（PC との違いは線形項 $\lambda_i^\top r_i$ だけ）:

$$\mathcal{L}(h,\theta,\lambda) = \tfrac12\|e_L\|^2 + \sum_{i=1}^{L-1}\lambda_i^\top r_i + \tfrac12\sum_{i=1}^{L-1}\|r_i\|^2$$

平方完成すると、$h,\theta$ についての勾配が同じ等価形:

$$\mathcal{L} \simeq \tfrac12\|e_L\|^2 + \tfrac12\sum_{i=1}^{L-1}\|r_i + \lambda_i\|^2$$

つまり各 primal step は「双対でずらした予測誤差 $r_i + \lambda_i$」に対する PC step である。ペナルティ係数は記事では ½ 固定で、別の調整パラメータは無い。

## 3. 更新則（勾配展開済み）

各ミニバッチ（mujou ではストリームの各ステップ）は forward pass で活動を初期化してから始める。$h_0 = x$ と出力側の教師 $y$ は clamp する。

### 推論フェーズ（$T$ ステップ）

**活動の primal descent**（$h_i \leftarrow h_i - \eta_h \nabla_{h_i}\mathcal{L}$）:

$$\nabla_{h_i}\mathcal{L} = (r_i + \lambda_i) - W_{i+1}^\top \varepsilon_{i+1} \qquad (1 \le i \le L-2)$$

$$\nabla_{h_{L-1}}\mathcal{L} = (r_{L-1} + \lambda_{L-1}) - W_L^\top e_L$$

**双対の dual ascent**:

$$\lambda_i \leftarrow \lambda_i + \alpha\, r_i$$

### 学習フェーズ（1 ステップ）

$W_i \leftarrow W_i - \eta_\theta \nabla_{W_i}\mathcal{L}$ を展開すると:

$$W_i \leftarrow W_i + \eta_\theta\, \varepsilon_i\, h_{i-1}^\top \quad (1 \le i \le L-1), \qquad W_L \leftarrow W_L + \eta_\theta\, e_L\, h_{L-1}^\top$$

「後シナプス側の誤差 × 前シナプス側の活動」の Hebbian 型。バッチでは外積をバッチ平均する。

### 局所性

$\nabla_{h_i}$ は $h_{i-1}, h_i, h_{i+1}$（と $W_i, W_{i+1}, \lambda_i, \lambda_{i+1}$）だけに依存する。上の層 $i+1$ が $W_{i+1}^\top \varepsilon_{i+1}$ を下へ送り、下の層 $i$ が $h_i$ を上へ送れば、通信は隣接層間で閉じる。成分ごとに見ると、ニューロン $j$ の更新も、結合でつながったニューロンとの値（$h_{i-1,k}$ と $W_{i+1}[k,j]\,\varepsilon_{i+1,k}$）だけで閉じる。そのため、PC-ALM はニューロンノードまで分割できる。実行モデル側の規約は skill `locality-and-async`。

## 4. 平衡点と BP の関係

推論ダイナミクスの不動点では:

- dual の停留条件 $r_i = 0$ → 活動は forward pass の値に戻る（$h_i = \sigma(a_i)$）
- primal の停留条件 → $\lambda_{L-1} = W_L^\top e_L$、$\lambda_i = W_{i+1}^\top(\sigma'(a_{i+1}) \odot \lambda_{i+1})$

これは BP の credit $-\partial \ell / \partial h_i$ の漸化式そのもの。よって**不動点での重み更新は BP の負勾配と一致する**。

- スカラー例: $\hat y = w_2 h,\ h = w_1 x$ なら不動点で $\lambda = w_2 (y - \hat y)$
- 不動点が存在しそこへ収束することの解析的保証は**線形ネットのみ**（記事の主張）。非線形は経験的
- PC（$\alpha = 0$）は不動点で活動が forward 値からずれる（prospective configuration）。PC-ALM は収束するとこれを失う。$T$・$\alpha$・dual leak を中間的にするとトレードオフを調整できる可能性がある（記事の示唆）

この関係は**テストのオラクル**として使う（skill `testing-strategy`）。

## 5. ハイパーパラメータと安定性

| 記号 | 意味 | 記事での扱い |
| --- | --- | --- |
| $T$ | 推論ステップ数 | 深さに比例させる（1000 層実験で $T = 2L$） |
| $\eta_h$ | 活動のステップサイズ | |
| $\eta_\theta$ | 重みのステップサイズ | |
| $\alpha$ | 双対のステップサイズ | $\alpha = 0$ で PC に退化。大きすぎると不安定 |
| dual leak | 双対の減衰 | 将来課題として言及のみ |

- $\alpha > 0$ で線形化系の固有値が複素になり、減衰振動する。$\alpha$ を上げすぎると発散する（深さ 8・幅 1 の線形例）
- credit の伝播は PC の拡散的（熱方程式的）な広がりではなく、出力から入力へ波面が進む「ballistic」な伝播。だから $T \propto L$ で足りる
- 制御理論の見方: $r_i$ が比例項、$\lambda_i$ が積分項の PI 制御器が各層にある
- 実験は幅の狭い residual MLP（幅 32、ReLU、MNIST で深さ 1000 まで BP と約 2pt 差）

## 6. 実装チェックリスト（落とし穴）

1. **誤差の向き**: $r_i = h_i - \sigma(a_i)$、$e_L = y - \hat y$。片方だけ逆にすると双対の符号が反転し、学習が発散するか無意味になる
2. **$\sigma'$ の評価点**: $a_i = W_i h_{i-1}$（現在の活動から計算した pre-activation）。forward 時のキャッシュを使い回さない
3. **clamp**: $h_0$ と教師は推論中に更新しない。推論専用（教師なし）ステップでは出力層の扱いを明示する
4. **双対の初期化**: 記事のスカラー例では $\lambda = 0$ 始まり。ミニバッチ間・ストリームのステップ間でリセットするか持ち越すかは明示的に決めて実装する（下の研究課題）
5. **重み更新の順序**: 学習フェーズは推論を終えた状態の $h, \lambda$ を使う。推論中に重みを動かす設計にするなら、それは記事の手順からの逸脱として明示する
6. **計算グラフをノード間で繋げない**: 層 $i$ の更新で $h_{i-1}$ の勾配が流れないよう detach する。ノードをまたぐ `.backward()` は使わない（hook が警告する）
7. **バッチ平均と学習率**: 外積をバッチ和にするか平均にするかで $\eta_\theta$ の意味が変わる。どちらかに統一して docstring に書く
8. **$\alpha = 0$ の退化テスト**: PC-ALM 実装は $\alpha = 0$ で PC と一致するはず。回帰テストとして持つ

## 7. mujou の研究課題（記事の範囲外）

以下は mujou が答えを探す問いで、確立した理論ではない。実装するときは仮説として扱い、決定を `memory/` に記録する。

### 7.1 連続時間形（mujou の定式化）

mujou のノードは、それぞれ独自の周期で非同期に動く（`memory/project_async_node_execution.md`）。そのため 3 節の離散更新を、その実行のタイマーの時間 $t$ に対する微分方程式として扱い直す。時定数 $\tau_h, \tau_\lambda, \tau_\theta$ を導入する。

$$\tau_h \frac{dh_i}{dt} = -\nabla_{h_i}\mathcal{L}, \qquad \tau_\lambda \frac{d\lambda_i}{dt} = r_i, \qquad \tau_\theta \frac{dW_i}{dt} = \varepsilon_i\, h_{i-1}^\top \ \ (\text{出力層は } e_L h_{L-1}^\top)$$

- 活動は拡張ラグランジアンの勾配流、双対はその上昇流、重みは勾配流。入力 $x(t)$ と教師 $y(t)$ は時間とともに変わるストリームとして与える
- 各ノードは、前回の更新からその実行のタイマーで測った経過時間 $\Delta t$ で離散化する。陽的 Euler なら $h_i \leftarrow h_i - \frac{\Delta t}{\tau_h}\nabla_{h_i}\mathcal{L}$
- 3 節の離散パラメータとの対応: $\eta_h = \Delta t/\tau_h$、$\alpha = \Delta t/\tau_\lambda$、$\eta_\theta = \Delta t/\tau_\theta$。全ノードを固定の $\Delta t$ で同期的に回すと、3 節の更新に一致する（テストのオラクル）
- 記事の「推論 $T$ ステップ → 学習 1 ステップ」は、$\tau_\theta \gg T\,\tau_h$ という時間スケールの分離として近似される。フェーズの切替は持たず、推論と学習は同時に進む
- 安定性: 陽的 Euler は $\Delta t/\tau$ が大きいと発散する。5 節の $\alpha$ の上限は $\Delta t/\tau_\lambda$ の上限として読み替える。ノードが遅れて $\Delta t$ が大きくなった場合の扱いは `specs/nodes.md` の「実行の周期」と `specs/pc.md` の「dt の打ち切り」
- 時定数どうしの比、$\Delta t$ の揺らぎや読み取る値の古さに対する収束性は未検証。下の「非同期更新」の問いに含まれる

### 7.2 mujou.pc の定式化（設計は `specs/pc.md`）

仮説として実装し、同期シミュレーションと MNIST で確かめたもの。確立した理論ではない。

- **任意の module への一般化**: 層 $\sigma(W_i h_{i-1})$ を任意の微分可能な $f_i(\cdot;\theta_i)$ に置き換える。ノードの内側で VJP を 1 回取り、cotangent $-\delta_i$ から $g_{\mathrm{in}} = -J_h^\top\delta_i$（前のノードへ送る値）と $g_\theta = -J_\theta^\top\delta_i$ を得る。$f = \sigma(Wh)$ で 3 節の式に戻り、不動点で BP と一致する
- **バッチの規約**: VJP はバッチの和、$1/B$ は $\theta$ の勾配にだけ掛ける。出力の損失はサンプルごとの値を mask 付きで足す
- **状態は h で持ち、離散的な跳びだけ付け替える**: 新しいサンプル、前のノードの $h$ の跳び、自分の $\theta$ の更新による $\mu$ の跳び（1 更新遅らせて足す）のときだけ $h \leftarrow \mu + r$ とし、跳びの回数 `jumps` で次のノードへ連鎖させる。ずれ $u = h - \mu$ を状態にする形は、$u_i$ を動かすと次以降の $h$ が全部動いて損失のヘッセ行列が順伝播のヤコビアンの積を通じて戻るため、深い鎖で連続時間の系として発散した。学習が毎更新に近いと付け替えが毎回起きて同じ形に戻るので、`learn_interval` は 2 × ホップ数 × 周期より長くすることを勧める（安定の境界ではない）
- **目標は入力に付随させる**: 入力とラベルの到着時刻のずれ（伝播の段数）が持ち越しの崩れの主因だったので、目標 `y`・`mask`・書き込みの時刻 `time` を活動と同じメッセージ（`pc.messages.Activity` のフィールド）で運び、各ノードが次へ写す
- **λ は持ち越す**: i.i.d. のバッチでも、上の付け替えがあればリセットは要らない（$h$ をそのまま持ち越すと $h_{\mathrm{old}} - \mu_{\mathrm{new}}$ が活動と同じ大きさになり崩れる）
- **学習の時刻**: ホストが書いた時刻から `learn_interval` ごと（`pc.clock.LearnClock`）。ノードが入力を見た時刻から数えると、非同期の更新順で前のノードの $\theta$ の跳びが $\delta$ に入ったまま学習して発散した
- **dt の打ち切り**: 1 回の更新の $\Delta t$ を `max_dt`（既定 $\min(\tau_h/2, 3\tau_\lambda)$）で打ち切る。刻みの分割はノード間の陽的な結合を安定化しない
- **実測**: 安定な時定数の範囲、MNIST での BP との比較、過負荷での振る舞いは `specs/research/pc-experiments.md`

- **フェーズの撤廃**: 推論 $T$ ステップ → 学習 1 ステップの同期的な区切りを、ストリーム上で連続的に回すダイナミクスに置き換えられるか。7.1 の連続時間形がその定式化で、時定数の選び方が問いとして残る
- **非同期更新**: 各層が隣接層の古い（stale）メッセージで更新しても収束・安定するか。遅延と $\alpha$ の関係
- **双対の持ち越し**: 非定常なストリームで $\lambda$ を warm start するか、dual leak で忘却させるか
- **教師信号の欠落**: ラベルの無いステップ（自己教師・予測タスク）で出力側をどう扱うか
- **時間方向の credit assignment**: 記事が将来課題として挙げる領域
