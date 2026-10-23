# pc-optimizers: mujou.pc の重みの optimizer（SGD と AdamW）を時間で進め、差し替えられるようにする

## 概要

τθ を `TimeConstants` から optimizer に移し、`pc.SGD` / `pc.AdamW`（frozen dataclass の設定）をノードが `bind(params)` して状態を持つ束縛を作る形にする。`Layer`・`Output`・`Chain` は `optimizer=` を受け取り、`functional.learn_` は消す。背景と決定事項は `memory/agents/orchestrator/pc-optimizers.md`（覆さない）。

## 公開インターフェース案

新しいモジュール `src/mujou/pc/optim.py`。import するのは `math`・`dataclasses`・`collections.abc`・`typing`・`torch` だけ（mujou も `torch.optim` も import しない）。

```
class Optimizer(Protocol):
    def bind(self, params: Iterable[torch.Tensor]) -> BoundOptimizer: ...

class BoundOptimizer(Protocol):
    def step_(self, grads: Sequence[torch.Tensor], elapsed: float) -> None: ...
    def state_dict(self) -> dict[str, Any]: ...
    def load_state_dict(self, state: Mapping[str, Any]) -> None: ...

@dataclass(frozen=True, kw_only=True)
class SGD:
    tau: float
    def bind(self, params: Iterable[torch.Tensor]) -> BoundOptimizer: ...

@dataclass(frozen=True, kw_only=True)
class AdamW:
    tau: float
    tau_m: float
    tau_v: float
    eps: float = 1e-8
    weight_decay: float = 1e-2
    def bind(self, params: Iterable[torch.Tensor]) -> BoundOptimizer: ...
```

具体的な束縛は module private の `_BoundSGD` / `_BoundAdamW`（optim.py の中だけで使うので `_`）。`bind` の戻り値の型は `BoundOptimizer`。

### Protocol を選ぶ理由

- 共有する実装が無い（SGD と AdamW で式も状態も別）。ABC にしても継承させるだけになる
- 利用者の optimizer は継承なしに書ける。`Node.is_state` も `state_dict` / `load_state_dict` の有無で見るダックタイピングで、構造的な型と揃う。`pc.show` の `_Shown` も Protocol
- `runtime_checkable` にしない。isinstance での検査は持たない（型の誤りは pyright が見る、G3）

### docstring の要点

- `Optimizer`: 「重みの時間発展の則の設定。値だけを持ち、状態を持たない（ノード間で共有してよい）。`bind(params)` が 1 つのノードの params の束縛を作る」。自作の例を短く 1 つ
- `BoundOptimizer`:
  - `step_(grads, elapsed)`: params を in place に `elapsed` 秒ぶん動かす。`grads` は params と同じ順・同じ shape/dtype/device の**平均済み**勾配（バッチの $1/B$ はノードが掛ける。optimizer はバッチを知らない）。`elapsed` は前の `step_` からの時間（秒、`Tick.elapsed`）。勾配を記録せずに動かす
  - `state_dict` / `load_state_dict`: ノードの属性に置けば、それだけでノードの状態に入る（`LearnClock` と同じ書き方）。`load_state_dict` は既存のテンソルへ `copy_` し、渡された木のテンソルを保持しない
  - 束縛は params を参照で持つ。1 つのノードが所有し、そのノードの hook の中だけで呼ぶ
- `SGD`: $\tau_\theta \dot\theta = -g$。1 歩は `p.sub_(g, alpha=elapsed / tau)`（学習率 `elapsed / tau` の SGD）。状態は空の dict
- `AdamW`: 下の「理論上の対応」の連続時間形と離散化、Δt 一定での `torch.optim.AdamW` との対応、`weight_decay` の単位、状態のキー（`"m"`、`"v"`、`"c_m"`、`"c_v"`）、`elapsed == 0` は何も動かさないこと。torch の per-step の β からの換算 $\tau_m = -\Delta t / \ln\beta_1$（例: 32 ms に 1 回学ぶなら β1 = 0.9 で約 0.30 s、β2 = 0.999 で約 32 s）

### 例外（構築時、`__post_init__`。`TimeConstants` と同じ流儀）

| クラス | 条件 | 例外とメッセージの頭 |
| --- | --- | --- |
| `SGD` | `tau` が有限で > 0 でない（NaN、inf を含む） | `ValueError("SGD.tau must be finite and > 0, got ...")` |
| `AdamW` | `tau`・`tau_m`・`tau_v` のどれかが有限で > 0 でない | `ValueError("AdamW.<name> must be finite and > 0, got ...")` |
| `AdamW` | `eps` が有限で >= 0 でない | `ValueError("AdamW.eps must be finite and >= 0, got ...")` |
| `AdamW` | `weight_decay` が有限で >= 0 でない | `ValueError("AdamW.weight_decay must be finite and >= 0, got ...")` |

- `tau=inf` を断るのは、学ばないのを `learn_interval=math.inf` の 1 通りにした決定（orchestrator メモの 1）に合わせるため。`tau_m=inf` は $c_m$ が 0 のままになり $\hat m = 0/0$ になる
- フィールドはすべてキーワード専用（全部 float で、位置の入れ違いを型で捕まえられない）
- `step_` の `grads` の数の違いは `zip(strict=True)` の `ValueError`（G2、今の `learn_` と同じ）。shape の違いは torch の in-place 演算が出す例外（G2）

### 変更する公開 IF

- `TimeConstants(*, h: float, lam: float)`。フィールド `theta` を消す。`period` / `max_dt` の property はそのまま
- `LearnClock.of(tau, *, period, learn_interval, max_dt=None)`: シグネチャは変えない。「`tau.theta` が inf なら学ばない」規則を消し、`max_dt` の既定 `tau.max_dt` を与えるだけになる
- `Layer(module, *, act_spec, tau, optimizer: Optimizer, learn_interval, period=None, max_dt=None, device="cpu")`
- `Output(module, *, act_spec, loss, tau, optimizer: Optimizer, learn_interval, period=None, max_dt=None, device="cpu")`
- `Chain(*modules, x, y, loss, tau, optimizer: Optimizer, learn_interval, batch=(), max_dt=None, device="cpu")`。全ノードに同じ設定を渡し、各ノードがそれぞれ bind する
- `Layer.optimizer` / `Output.optimizer`: 束縛（`BoundOptimizer`）を置く public の属性。状態の木のキーは `"optimizer"`
- 削除: `pc.functional.learn_`
- `mujou.pc.__all__` に `Optimizer`、`BoundOptimizer`、`SGD`、`AdamW` を足す

### optimizer を必須にする理由

時間で表した既定の学習率は無い（`tau` は見せる時間と 1 バッチあたりの学習率から決まる）。`learn_interval` に既定を置かなかったのと同じ理由で、省いたときに黙って何かの値で走らせない。`learn_interval=math.inf` のノードにも optimizer は要るが、呼ばれないだけで害はない。

## 理論上の対応

### 連続時間形

ノードが渡す平均の勾配を $g = g_\theta / B$ とする。

- SGD: $\tau_\theta \dot\theta = -g$
- AdamW: $\tau_m \dot m = g - m$、$\tau_v \dot v = g^2 - v$、$\tau_\theta \dot\theta = -\left(\hat m / (\sqrt{\hat v} + \varepsilon) + w\,\theta\right)$
- 質量: $\tau_m \dot c_m = 1 - c_m$、$\tau_v \dot c_v = 1 - c_v$（$c(0) = 0$）。$\hat m = m / c_m$、$\hat v = v / c_v$

### 離散化（`elapsed` = Δt の 1 歩、AdamW）

torch の `_single_tensor_adam`（decoupled weight decay、非 capturable）と同じ順にする。

1. `elapsed == 0` なら何もせず戻る（0 秒の流れは恒等写像。初回の前は $c = 0$ で $\hat m = 0/0$ になる。`ManualRun.step(0.0)` や `learn_interval=None` で起こりうる）
2. $a_m = -\mathrm{expm1}(-\Delta t/\tau_m)$、$a_v = -\mathrm{expm1}(-\Delta t/\tau_v)$（$1 - e^{-x}$ を小さな $x$ でも精度よく。Python の float）
3. $c_m \leftarrow c_m + a_m(1 - c_m)$、$c_v \leftarrow c_v + a_v(1 - c_v)$（Python の float）
4. $\eta = \Delta t / \tau_\theta$。param ごとに
   - $\theta \leftarrow \theta\,(1 - \eta w)$（decay を先に。torch の `param.mul_(1 - lr * weight_decay)`）
   - $m \leftarrow m + a_m(g - m)$（`m.lerp_(g, a_m)`。torch の `exp_avg.lerp_(grad, 1 - beta1)`）
   - $v \leftarrow v + a_v(g^2 - v)$（`v.lerp_(g * g, a_v)`。torch は `mul_(beta2).addcmul_(grad, grad, value=1 - beta2)` で、丸めだけが違う）
   - $\theta \leftarrow \theta - \frac{\eta}{c_m}\cdot\frac{m}{\sqrt{v}/\sqrt{c_v} + \varepsilon}$（torch の `denom = (exp_avg_sq.sqrt() / bias_correction2_sqrt).add_(eps)`、`param.addcdiv_(exp_avg, denom, value=-lr / bias_correction1)`）。ε は $\sqrt{\hat v}$ に足す

Δt が一定なら $1 - a = e^{-\Delta t/\tau}$ を β と置いて $c = 1 - \beta^t$ になり、`torch.optim.AdamW(lr=Δt/τθ, betas=(e^{−Δt/τm}, e^{−Δt/τv}), eps=ε, weight_decay=w)` と丸めの範囲で一致する。

### 決めたこと

- **w の単位**: 無次元で $1/\tau_\theta$ と組む（decay の時定数は $\tau_\theta / w$）。torch の `weight_decay` と同じ値がそのまま同じ意味になり、上の一致が成り立つ。独立の時定数 $\tau_w$ にする案は、torch との対応が崩れ、学習率を変えると decay の比が変わる直感（AdamW の decoupled の定義）とも合わないので採らない
- **θ の離散化は陽的 Euler**（SGD と同じ）。decay を $e^{-\eta w}$ にする厳密な案は torch と一致しなくなるので採らない。$\eta w > 1$ で符号が反転するが、`elapsed` は `learn_interval` 程度（MNIST で 24 ms、τθ が 0.3 s なら $\eta \approx 0.08$）で問題にならない
- **c は Python の float**。`torch.load(weights_only=True)` で読め、復元は代入なので同じ値になる。torch の非 capturable の bias 補正も Python の float（倍精度）で計算する。0-d テンソルにすると dtype を選ぶ必要があり、float32 では 1 に近い c の精度が落ちる。GPU の param でも Python のスカラーとの演算は同期を起こさない
- **c は [0, 1]**: 倍精度では約 $37\tau$ 経つと $1 - c$ が 0 に丸まり、c はちょうど 1.0 になる（[0, 1) ではない）。そのとき $\hat m = m$ で害はない。ステップのカウンタは持たないので、float32 の `state["step"]` が $2^{24}$ で止まる問題は起きない
- **状態の dtype/device**: m、v は `torch.zeros_like(p)`（param と同じ dtype・device・memory format）。bind は module が device に載った後（ノードの `__init__` の中）に呼ぶ
- **frozen params**: ノードは `requires_grad` の param だけを bind する（今の `self._params`）。weight decay も frozen な param に掛からない
- **SGD のビット一致**: 今は `p.sub_(g_sum, alpha=(elapsed/τ)/B)`、新しくは `p.sub_(g_sum / B, alpha=elapsed/τ)`。B が 2 の冪なら `g/B` も `(elapsed/τ)/B` も厳密で、積が同じ実数になるのでビット一致する（MNIST の既定 B = 64、pc_merge の 32 も同じ）。B が 2 の冪でない（テストの B = 3）と更新の項が 1 ulp 程度ずれうる。既存のテストは参照実装と 1e-14〜1e-10 の許容で比べているので影響しない。`rtol=1e-15` の `TestLearn` は `learn_` と一緒に消え、`test_optim.py` の新しい式のテストに置き換わる

### 局所性・非同期性

- 束縛は自分のノードの param と、ノードが渡す勾配と `elapsed` だけを読む。ノード間の読み取りもバリアも無い
- 設定（frozen dataclass）は状態を持たず、`state_dict` も無いので `is_state` に入らない。`Chain` で全ノードが共有してよい。束縛はノードごとに別（同じ束縛を 2 ノードに置くと `tree.walk` の K24 が断る）
- 更新手法の焼き付け: optimizer は `mujou.pc` の部品で基盤は知らない。`LearnClock` と同じく PC 以外の則でも使える（`examples/feedback_alignment.py`）
- `torch.optim` を実行時に import しない。3.14t でコンストラクタが `torch._dynamo` → `triton._C` を import し、プロセス全体の GIL が戻るため

## 実装ステップ

1. **optim.py と単体テスト**
   - `src/mujou/pc/optim.py` を上の IF で書く。`step_` は `torch.no_grad()` の中で動かす
   - `tests/mujou/pc/test_optim.py` を書く（テスト観点の TestSGD〜TestFreeThreaded）
   - `tests/test_layers.py` の `LAYERS` で `mujou.pc.optim` を `{"mujou.pc.messages", "mujou.pc.time_constants"}` の段に足す
   - `src/mujou/pc/__init__.py` で 4 つの名前を import し `__all__` に足す。`tests/mujou/test_init.py` の pc の一覧に足す
   - 検証: `uv run pytest tests/mujou/pc/test_optim.py tests/test_layers.py tests/mujou/test_init.py -v`、`uv run pyright src/mujou/pc/optim.py`
2. **TimeConstants と LearnClock.of**
   - `time_constants.py`: `theta` と、その検査・docstring（式・`theta=math.inf`・例）を消す。θ は optimizer が持つ旨を 1 行
   - `clock.py:112-131`: `of` から theta の規則を消し、docstring を直す
   - `tests/mujou/pc/test_time_constants.py`: theta を消し、`lam` だけの検査と「lam は inf 可」にする
   - `tests/mujou/pc/test_clock.py:102-116`: `TimeConstants(h, lam)` にする。`test_never_learns_with_an_infinite_theta` は消す（「inf は学ばない」は `TestWithoutInterval.test_inf_never_learns` が持つ）
   - 検証: `uv run pytest tests/mujou/pc/test_time_constants.py tests/mujou/pc/test_clock.py`
3. **Layer・Output・Chain と functional**
   - `layer.py`: `optimizer` 引数を足し、`self._params` の直後に `self.optimizer = optimizer.bind(self._params)`（状態のキーの順は module の後）。`self._batch` は `__init__` の中だけで使うので、サンプルの数 `self._samples = math.prod(batch)` に置き換える。`_learn`（289-303）は `self.optimizer.step_([g / self._samples for g in g_theta], elapsed)`。docstring の式（25-29 行）を「θ は optimizer が平均の勾配 $g_\theta/B$ で動かす」に、86-87 行の theta の規則を消し、状態の一覧に `"optimizer"`、Args に `optimizer`、Example を直す
   - `output.py`: 同じく（108-117、158-159）。docstring の式（25-28）、Args の `tau`（79-80）、状態の一覧、Example
   - `chain.py`: `optimizer` を受け取り各ノードへ渡す。docstring の Args と Example
   - `functional.py`: `learn_` を消し、`__all__`、モジュールの docstring（13、17-19 行: $1/B$ はノードが optimizer に渡す前に掛ける）、`mse` の docstring（242 行）を直す。使われなくなる import（`batch_shape` は `_starting` が使うので残る。`Size` も残る）を確かめる
   - `__init__.py` の docstring の例と部品の説明（`SGD`・`AdamW` と optimizer）
   - テストを直す（テスト観点の「既存テストの書き換え」）
   - 検証: `uv run pytest tests/mujou/pc -v`（slow を含む）。`grep -rn "theta=\|learn_(\|\.theta\b" src tests examples` が 0 件
4. **examples**
   - `pc_regression.py:25-38`、`visualize.py:82`、`mnist/main.py:83-93`: `TimeConstants(h, lam)` と `optimizer=pc.SGD(tau=...)`。コメントの「SHOWN / theta」を「SHOWN / optimizer の tau」に
   - `mnist/bp.py:30` の docstring の `sum_dt / tau.theta` を `elapsed / tau`（`pc.SGD` の）に
   - `pc_merge.py`: `TAU` から theta を消し、`OPTIMIZER = pc.SGD(tau=SHOWN / 0.3)` を足す。`Merge.__init__` で `self.optimizer = OPTIMIZER.bind(self.module.parameters())`、79 行を `self.optimizer.step_([g / B for g in g_theta], t.elapsed)`。tower と Output に `optimizer=OPTIMIZER`。snippet の印はそのまま
   - `feedback_alignment.py`: `TAU` を `SGD = pc.SGD(tau=SHOWN / 0.2)` にし、`Hidden` は `self.optimizer = SGD.bind([self.w1])` と `self.optimizer.step_([(delta.T @ x) / BATCH], tick.elapsed)`、`Output` も w2 で同じ。optimizer が `LearnClock` と同じく PC 以外の則の部品であることを見せる（採用理由）
   - 検証: `just type`（examples は standard で検査される）。example はテストで実行しない
5. **specs と memory**（下の「変更するファイル」）。検証: `grep -rn "theta\b" specs/pc.md specs/scenarios.md` で残りが τθ の説明だけか目で確かめる
6. **docs（en/ja）と API リファレンス**。doc-teacher-student と yomiyasu は orchestrator の段で行う。検証: `just docs-build`
7. **全体**: `just run`

## 変更するファイル

### src

| ファイル | 変更 |
| --- | --- |
| `src/mujou/pc/optim.py` | 新規。`Optimizer`、`BoundOptimizer`、`SGD`、`AdamW`、`_BoundSGD`、`_BoundAdamW` |
| `src/mujou/pc/time_constants.py` | `theta` を消す |
| `src/mujou/pc/clock.py` | `LearnClock.of` の theta の規則を消す（112-131） |
| `src/mujou/pc/layer.py` | `optimizer` 引数、`self.optimizer`、`_learn`、docstring |
| `src/mujou/pc/output.py` | 同上 |
| `src/mujou/pc/chain.py` | `optimizer` 引数を各ノードへ |
| `src/mujou/pc/functional.py` | `learn_` を消す、docstring |
| `src/mujou/pc/__init__.py` | import、`__all__`、docstring |

### tests

| ファイル | 変更 |
| --- | --- |
| `tests/mujou/pc/test_optim.py` | 新規（下のテスト観点） |
| `tests/mujou/pc/_reference.py` | `TAU = pc.TimeConstants(h=4e-3, lam=8e-3)`、`TAU_THETA = 0.05`、`SGD = pc.SGD(tau=TAU_THETA)`。`NO_LEARNING` を消す。`_learn`（261 行）は `TAU_THETA` の手書きの式のまま（optimizer を使わない参照実装に保つ） |
| `tests/mujou/pc/test_layer.py` | `HandChain` に `optimizer: pc.Optimizer = SGD` を足す。`tau=NO_LEARNING` は `learn_interval=math.inf` に（`LearnClock.of` が今そう変換しているので振る舞いは同じ）。252・498-499 の `TAU.theta` / `theta=` を optimizer の tau に。316 の `theta=1e9` は `optimizer=pc.SGD(tau=1e9)`。385 は `TimeConstants(h, lam=inf)` と `learn_interval=math.inf`。924-931 は theta を消すだけ。`layer_update`（640-666）は `pc.SGD(tau=TAU_THETA).bind(params).step_([g / B ...], elapsed)` に。`test_without_learning_the_interval_changes_nothing`（734-761）は前提（theta=inf が interval を上書きする）が無くなるので、「`learn_interval=math.inf` ではラベルのある入力を 30 ステップ見せても全ノードの θ がビット一致で動かない」テストに書き換える。`TestState` のキーに `"optimizer"` を足す。新しいテストは下 |
| `tests/mujou/pc/test_output.py` | `TAU` から theta、`optimizer=` を渡す、`TAU.theta` を `TAU_THETA` 相当に。新しいテストは下 |
| `tests/mujou/pc/test_chain.py`、`test_show.py`、`test_input.py` | `optimizer=` を渡し、`TimeConstants` から theta を消す |
| `tests/mujou/pc/test_functional.py` | `TestLearn` と `learn_` の import を消す。`TimeConstants(h, lam)` に。344 行のコメントを直す |
| `tests/mujou/pc/test_time_constants.py`、`test_clock.py` | ステップ 2 |
| `tests/mujou/test_init.py` | pc の `__all__` に 4 つ、functional から `learn_` を消す |
| `tests/test_layers.py` | `mujou.pc.optim` を `LAYERS` に |

### examples

`pc_regression.py`、`pc_merge.py`、`feedback_alignment.py`、`visualize.py`、`mnist/main.py`、`mnist/bp.py`（ステップ 4）。AdamW の example は足さない。MNIST での AdamW の値を測っていないので、数字を出す example を作れない。

### specs

| ファイル | 変更 |
| --- | --- |
| `specs/pc.md` | 一般化した定式化（17 行）の θ の式を optimizer に。「学習の時刻」の式（123 行）と `LearnClock.of` の説明（127、130 行）。「時定数」（156-159 行）を `TimeConstants(*, h, lam)` と K25 に。新しい節「重みの optimizer」: 設定 + bind の形、連続時間形、離散化と質量 c、Δt 一定での torch との一致、w の単位、c が float の理由、dtype/device、`elapsed == 0`、$1/B$ の位置、torch.optim を使わない理由（GIL、β が 1 ステップあたりで学習間隔に依存、step カウンタの 2^24）、段階 2 の方針。「自作のノード」（192-198 行）の `learn_` を束縛の `step_` に。「状態」の表に `optimizer`。「公式実装との違い」の「重みの最適化」の行を「時間で進める SGD か AdamW（`pc.SGD` / `pc.AdamW`）」に。「テスト」に optimizer の項目 |
| `specs/scenarios.md` | K25 を「h が有限で > 0 でない、lam が > 0 でない（inf は可）」に。K31 を足す（S7、`SGD` / `AdamW` の値、`__post_init__`、`ValueError`、「NaN になるか、学習しない」、`pc/test_optim.py` の `TestChecks`）。127 行の `learn_` を `BoundOptimizer.step_` に |
| `specs/open-decisions.md` | 「`theta=inf` を `learn_interval=inf` で表すこと」を消す（ユーザーが決めた）。「`torch.optim` を差し込む口」を消す（自前の時間の optimizer に決めた）。mujou.pc に「h と λ の則の差し替え（段階 2）」と「AdamW の推奨の時定数（未測定）」を足す |
| `specs/update-rules.md` | 「mujou.pc は基盤の外の利用者」に、θ の時間発展も `pc.Optimizer` で差し替えられることを 1 行 |
| `specs/pc-alm-continuous-time.md` | 61 行の表の $\tau_\theta$ に「`pc.SGD(tau=)` / `pc.AdamW(tau=)` が持つ」と注記するだけ |

段階 2 の方針の文案（specs/pc.md の新しい節の末尾）: 「段階 2: 緩和の則として h と λ を 1 組で差し替える。不動点（$r = 0$、$\lambda = -g$）は保てるが、陽的な結合の安定性、跳びでの付け替え（$r$ を保つこと）との整合、サンプルごとの状態（バッチの各行に別の状態）が課題で、研究から始める。今は実装しない」

### docs（en と ja の両方）

新しい topic は作らず、既存のページへの追記で足りる。optimizer は `pc` の部品の 1 つで、時定数の選び方と同じ文脈で読まれるため。

| ページ | 変更 |
| --- | --- |
| `docs/{en,ja}/tutorial/5-learn-with-pc.md` | `tau` の箇条（h と lam だけ）、`optimizer` の箇条を足す（`pc.SGD(tau=SHOWN / 0.3)`、1 バッチあたりの学習率は見せる時間 / `tau`、AdamW もある旨と topics へのリンク）、`learn_interval` の箇条の `tau.theta` |
| `docs/{en,ja}/topics/pc.md` | 「時定数を決める」の表の `tau.theta` の行を optimizer の `tau` に、学習率とストリームの箇条を直す。新しい節「optimizer を選ぶ」: SGD と AdamW の式、時間で進める理由、torch の β・lr からの換算、`weight_decay` の単位、AdamW の値は未測定。「自作のノード」の手順 1 に bind、手順 7 を `self.optimizer.step_([g / B for g in g_theta], t.elapsed)` に |
| `docs/{en,ja}/topics/pc-concepts.md` | 1 節の部品の表に `pc.SGD`・`pc.AdamW`（設定）を足し、`functional` の行から `learn_` を消す。3 節の段 6 を `optimizer.step_` に、箇条の `learn_` を直す。4 節の `LearnClock` の表の `math.inf` の行から theta の規則を消し、`learn` / `elapsed` の「使い道」を `step_` に。5 節の TimeConstants の表から `tau.theta` を消す。6 節「optimizer」を足す（要点、式、質量 c の説明、Δt を 2 回と 2Δt を 1 回で m・v・c が同じになる表か図。`feedback_docs_teaching_quality` の基準） |
| `docs/{en,ja}/topics/update-rules.md` | `pc.LearnClock` の節に、重みの 1 歩は `pc.SGD` の束縛で書けることを 1〜2 文（feedback_alignment の snippet が変わるので） |
| `docs/en/api/pc.md` | `mujou.pc` の members の `TimeConstants` の後に `Optimizer, BoundOptimizer, SGD, AdamW`。functional の members から `learn_` を消す |

### memory

- `memory/project_pc_optimizers.md` を新規（決定と理由: τθ の移動、設定 + bind、SGD と AdamW、式、c は float、必須の引数、torch.optim を使わない、段階 2）。`memory/MEMORY.md` に 1 行
- `memory/project_pc_module.md` の公開名の行と LearnClock の行（`math.inf` は学習しない、theta の規則が消えたこと）を直す

## テスト観点

### `tests/mujou/pc/test_optim.py`（float64、seed 固定、期待値は手で書いた式）

- **TestSGD**
  - 正常系: `step_` が `p0 - (elapsed / tau) * g` と一致する（rtol 1e-15）。複数の param で順を保つ
  - 数値的オラクル: B = 4 の和の勾配 g_sum に対し、`step_([g_sum / 4], elapsed)` が旧 `learn_` の式 `p0.sub(g_sum, alpha=elapsed / tau / 4)` と `torch.equal`（2 の冪のバッチでビット一致することの回帰）
  - エッジ: `elapsed == 0` で params が `torch.equal` のまま。`state_dict() == {}`
- **TestAdamW**
  - 数値的オラクル（Δt 一定の離散 AdamW）: 10 歩を、テストに手で書いた torch の AdamW の式（β = e^{−Δt/τ}、ステップ数 t、bias 補正 1 − β^t、decay を先に、ε は $\sqrt{\hat v}$ に足す）と比べる（rtol 1e-12）。ε を $\sqrt{\hat v}$ と同じ桁にし（勾配の大きさ 1e-3、ε = 1e-3）、ε の位置の誤りが見える設定にする
  - 数値的オラクル（初回の閉じた形）: 新しい束縛の 1 歩は $\theta(1 - \eta w) - \eta\, g / (|g| + \varepsilon)$（τm・τv に依らない）
  - 時間の分割: 1 歩の warm-up の後、g 一定で Δt を 2 回と 2Δt を 1 回にした 2 つの束縛の m・v・c_m・c_v が一致する（rtol 1e-12。θ は陽的 Euler なので比べない）
  - 質量: 異なる Δt を混ぜた列で、c_m が $1 - \exp(-\sum\Delta t/\tau_m)$ と一致する（rtol 1e-12）。各歩で $0 < c \le 1$。`elapsed = 1e6 * tau_m` の 1 歩で c がちょうど 1.0、θ が有限。状態のキーが `{"m", "v", "c_m", "c_v"}` だけ（カウンタが無い）
  - weight decay: 勾配 0 で k 歩回すと θ が $\theta_0 (1 - \eta w)^k$（m = v = 0 なので Adam の項は 0）。w が $1/\tau_\theta$ と組むことの確認
  - dtype/device: m、v が param の dtype（float32 と float64 を parametrize）。`@pytest.mark.cuda` で CUDA の param なら状態も CUDA、`step_` が CPU の値と一致
  - エッジ: `elapsed == 0` の初回で NaN にならず params がそのまま、c は 0 のまま。param 0 個の bind で `step_([], dt)` が通る
  - 異常系: grads の数が違うと `ValueError`
- **TestState**（SGD と AdamW を parametrize）
  - 3 歩回した束縛 A の `state_dict` を、同じ初期 params の複製に bind した B に `load_state_dict` し、両方を同じ勾配で 5 歩回すと params と状態が `torch.equal`
  - `load_state_dict` の後で渡した木のテンソルを書き換えても B の状態は変わらない（コピーする）
  - `state_dict` を `torch.save` → `torch.load(weights_only=True)` で読み戻せ、c が `float`
- **TestChecks**（K31）: SGD の tau、AdamW の tau・tau_m・tau_v に 0、-1、nan、inf → `ValueError`（メッセージに `"SGD.tau"` / `"AdamW.tau_m"` などを substring で）。eps と weight_decay に -1、nan、inf → `ValueError`。eps = 0 と weight_decay = 0 は通る。位置引数は `TypeError`
- **TestTorchAdamW**（`@pytest.mark.slow`）: subprocess で `sys.executable -c` のスクリプトを 1 本。float64 の params、seed 固定の勾配で 20 歩、`torch.optim.AdamW(lr=Δt/τ, betas=(exp(−Δt/τm), exp(−Δt/τv)), eps=1e-3, weight_decay=0.1)` と `pc.AdamW(...).bind(...)` を並べ、最大の相対誤差が 1e-10 未満なら `ok` を出力。親は returncode 0 と出力を見る（torch.optim はこのテストのプロセスに入らない）
- **TestFreeThreaded**（`sysconfig.get_config_var("Py_GIL_DISABLED")` でなければ skip）: subprocess で `pc.AdamW` と `pc.SGD` を bind して `step_` した後の `sys._is_gil_enabled()` が False

### 既存のテストファイルへの追加

- `test_layer.py`
  - frozen params: 1 つの param を `requires_grad_(False)` にした module の `Layer` を `pc.AdamW(weight_decay>0)` で学習させると、その param は `torch.equal` で動かず、`state["optimizer"]["m"]` の長さが学ぶ param の数
  - AdamW で学ぶ: `Learning` 相当の最初の学習で、θ が初回の閉じた形（`layer_update` と同じ g_theta / B から）と一致する（1e-12）
  - 状態の復元: `TestState.test_a_restored_run_goes_on_as_the_same_run` を `[pc.SGD(...), pc.AdamW(...)]` で parametrize（`ManualRun` の復元で c も含めて同じ値になる）
- `test_output.py`: `pc.AdamW` を渡した `Output` の最初の学習が初回の閉じた形と一致（g_theta は既存の `gradients` helper から、/B して）
- `test_chain.py`: `Chain(..., optimizer=pc.AdamW(...))` で回すと、`input` 以外の各ノードの状態に `"optimizer"` があり、`m` の長さがそのノードの module の param の数（ノードごとに別に bind されること。共有なら K24 で run が始まらない）

## 想定リスク・トレードオフ

- AdamW の時定数の推奨値は未測定。docs には換算式だけを書き、値を勧めない
- B が 2 の冪でないと、旧 `learn_` と更新の項が 1 ulp 程度ずれる。既存テストの許容（1e-14 以上）の中
- `v.lerp_(g * g, a_v)` は param ごとに g² を 1 つ確保する。学習は間隔を空けて走るので無視できる。torch と同じ `mul_` + `addcmul_` にすると確保は消えるが、式との対応が読みにくくなる
- SGD → AdamW に変えて古い状態を復元すると `KeyError("m")` が `NetworkError` の原因として届く（G2）。AdamW → SGD は moment を黙って捨てるが、利用者が optimizer を変えたのでそれで正しい。新しいガードは足さない
- `learn_interval=math.inf` のノードにも optimizer を渡す手間が残る（必須にした代償）
- `Optimizer.bind` を `Iterable` にしたので `module.parameters()` を直接渡せるが、frozen な param も入る。`pc.Layer` / `pc.Output` は自分で `requires_grad` の param を選ぶ。自作のノードの docs に「学ぶ param だけを渡す」と書く
- 採らなかった案: ABC（共有する実装が無い）、w を独立の時定数にする（torch と対応しない）、c を 0-d テンソルにする（dtype の選択と精度）、decay の厳密な指数（torch と一致しない）、optimizer の既定に SGD を置く（既定の学習率が無い）

## 確認事項

1. **公開名 `Optimizer` / `BoundOptimizer`**: 設定の Protocol と束縛の Protocol の名前。代案は `Optimizer` / `Binding`、`Optimizer` / `OptimizerState`。暫定案は `BoundOptimizer`（`bind` が返す物と読め、torch の `Optimizer`（状態を持つ物）との混同も「Bound」で避けられる）
2. **AdamW の既定**: `tau`・`tau_m`・`tau_v` は必須（per-step の β の既定は学習の間隔が無いと秒にできない）、`eps=1e-8`、`weight_decay=1e-2`（torch の AdamW の既定。w が無次元で $1/\tau_\theta$ と組むので同じ意味になる）。代案は `weight_decay=0.0`。暫定案は torch と同じ `1e-2`

## 参照

- 決定事項: `memory/agents/orchestrator/pc-optimizers.md`
- 今の重みの更新: `src/mujou/pc/functional.py:206-233`（`learn_`）、`src/mujou/pc/layer.py:179-218, 289-303`、`src/mujou/pc/output.py:96-159`、`src/mujou/pc/chain.py:68-106`、`src/mujou/pc/clock.py:112-131`
- 状態の取り込み: `src/mujou/node.py:256-431`（`register_state`、`load_state_dict`、`is_state`、`map_tensors`）
- ガードの台帳: `specs/scenarios.md`（K22、K24、K25、K26）
- 参照実装と helper: `tests/mujou/pc/_reference.py:250-262`、`tests/mujou/pc/test_layer.py:36-89, 597-666, 734-761, 1114-1145`
- subprocess のテストの書き方: `tests/mujou/test_init.py:32-44`、`tests/mujou/test_driver.py:460-480`
- skill: `pc-alm`、`locality-and-async`（不変条件 1・4）、`testing-strategy`、`refactor-conventions`
