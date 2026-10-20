# mujou.pc（PC-ALM のノード群）

> 2026-10-20 implementation-planner。設計は `specs/pc-module.md`（ブランチ `docs/2026-10-19/pc-module-plan`、ユーザーと合意済み）。前提の PR: `Node.register_state`（`feature/2026-10-20/node-register-state`、実装済み）とコアの carry（`memory/agents/implementation-planner/carry.md`、未実装）。この計画は `specs/pc-module.md` の実装の順番 3〜5 にあたる。

## 概要

PC-ALM を `mujou.Node` の上に PyTorch で実装し、`mujou.pc` から公開する。ノードの内側は VJP（BP）、ノードの間は u の形の PC-ALM。ターゲットは carry で入力に付随させ、MNIST で BP と比べる example と docs を付ける。

## 公開インターフェース案

### ファイル構成

実装は `src/mujou/_core/pc/`、公開は `src/mujou/pc/`。`mujou/__init__.py` は変えない（トップレベルに出さない。`import mujou.pc as pc` で使う）。

| ファイル | 中身 |
| --- | --- |
| `src/mujou/_core/pc/__init__.py` | docstring だけ |
| `src/mujou/_core/pc/time_constants.py` | `TimeConstants` |
| `src/mujou/_core/pc/carry.py` | `Carried`・`Sample`・`Prediction`・`carry_spec` |
| `src/mujou/_core/pc/losses.py` | `mse`・`cross_entropy` |
| `src/mujou/_core/pc/functional.py` | `vjp`・`loss_vjp`・`relax_`・`learn_` |
| `src/mujou/_core/pc/learner.py` | `Learner(Node)`（非公開の基底。学習の位相・ΣΔt・最後に見た時刻） |
| `src/mujou/_core/pc/input.py` / `layer.py` / `output.py` | `Input` / `Layer` / `Output` |
| `src/mujou/pc/__init__.py` | 下の公開名を再公開し `__all__` を持つ。`from mujou.pc import functional as functional` |
| `src/mujou/pc/functional.py` | `vjp`・`loss_vjp`・`relax_`・`learn_` を再公開（`import mujou.pc.functional` を通すためパッケージにする） |

`mujou.pc.__all__ = ["Carried", "Input", "Layer", "Output", "Prediction", "Sample", "TimeConstants", "carry_spec", "cross_entropy", "functional", "mse"]`

### 時定数

- `@dataclass(frozen=True, kw_only=True) class TimeConstants: h: float; lam: float; theta: float`（秒）
  - `__post_init__` で検査: `h` は有限で > 0、`lam`・`theta` は > 0（`math.inf` 可。NaN 不可）。違えば `ValueError`、`bool` や数でない値は `TypeError`
  - `lam=math.inf` が α=0（PC）、`theta=math.inf` が学習しない。レート $1/\tau$ は `dt / tau.lam` が `inf` で 0 になるので専用の属性を持たない
  - kw_only の理由: 3 つとも float で、位置で渡すと取り違えが型で検出できない

### 付随データと型（`carry.py`）

- `class Carried(TypedDict): y: torch.Tensor; mask: torch.Tensor; time: torch.Tensor`（`y`: 利用者の spec、`mask`: `(B,)` bool、`time`: `()` int64、ホストが入力を書いた時刻の ns）
- `class Sample(TypedDict): x: torch.Tensor; y: torch.Tensor; mask: torch.Tensor`（ホストが `Input.prev` に書くもの）
- `class Prediction(TypedDict): y_hat: torch.Tensor; y: torch.Tensor; mask: torch.Tensor; time: torch.Tensor`（`Output.next` が送るもの。`time` を入れるのは、ホストが新しいサンプルの最初の予測を `wait_for` で `time == write の ns` として拾うため）
- `def carry_spec(target: TensorSpec, batch: int) -> DictSpec[Carried]`
  - `DictSpec(Carried, y=target, mask=TensorSpec((batch,), torch.bool), time=TensorSpec((), torch.int64))`
  - `TypeError`: `target` が `TensorSpec` でない。`ValueError`: `target.shape[0] != batch`（`batch < 1` は `TensorSpec` が投げる）
  - 同じ引数の 2 回の呼び出しは等しい spec（`DictSpec` の等価）なので、ノードごとに作っても `connect` が通る

### 損失（`losses.py`、サンプルごと、shape `(B,)`）

- `def mse(y_hat: torch.Tensor, y: torch.Tensor) -> torch.Tensor`: $\tfrac12\|y_b-\hat y_b\|^2$ を先頭以外の次元で和。`y_hat.shape != y.shape` は `ValueError`（broadcast で `(B,B)` になって黙って通るのを防ぐ）
- `def cross_entropy(y_hat: torch.Tensor, y: torch.Tensor) -> torch.Tensor`: `F.cross_entropy(y_hat, y, reduction="none")`。`y_hat: (B, C)`、`y: (B,)` int64 か `(B, C)` の確率

### `pc.functional`（状態を持たない計算。`Layer`・`Output` もこれで書く）

- `def vjp(module: Callable[[torch.Tensor], torch.Tensor], h_in: torch.Tensor, delta: torch.Tensor, params: Sequence[torch.Tensor] = (), *, wrt_input: bool = True) -> tuple[torch.Tensor, torch.Tensor | None, list[torch.Tensor]]`
  - 戻り値 `(mu, g_in, g_theta)`。`mu = module(h_in)`（detach 済み、`(B, *out)`）、$E=\tfrac12\|h-\mu+\lambda\|^2$（$h$ 固定）の勾配 $g_{\mathrm{in}}=-J_h^\top\delta$（`(B, *in)`、`wrt_input=False` なら `None`）、$g_\theta=-J_\theta^\top\delta$（`params` の順、バッチの和）
  - `delta = u + lam`（`(B, *out)`）。`delta.shape != mu.shape` は `ValueError`
  - 中身: `torch.enable_grad()` の下で `h_in.detach().requires_grad_(wrt_input)`、`torch.autograd.grad(mu, [...], grad_outputs=-delta, materialize_grads=True)`（使われないパラメータも 0 で返す）。その行に `# mujou: local-autograd`。`wrt_input=False` かつ `params` が空なら `no_grad` で `mu` だけ
- `def loss_vjp(module: Callable[[torch.Tensor], torch.Tensor], h_in: torch.Tensor, y: torch.Tensor, mask: torch.Tensor, loss: Callable[[torch.Tensor, torch.Tensor], torch.Tensor], params: Sequence[torch.Tensor] = (), *, wrt_input: bool = True) -> tuple[torch.Tensor, torch.Tensor | None, list[torch.Tensor]]`
  - 戻り値 `(y_hat, g_in, g_theta)`。$\ell=\sum_b \mathrm{where}(\mathrm{mask}_b,\ell_b,0)$ の $\partial\ell/\partial h_{\mathrm{in}}$ と $\partial\ell/\partial\theta$。`loss` の戻り値が `(B,)` でなければ `ValueError`
  - mask は掛け算でなく `torch.where`: ラベルの無いサンプルの `y` が何であっても前向きに NaN を混ぜない（意味は「mask を掛けて和」と同じ）
- `def relax_(u: torch.Tensor, lam: torch.Tensor, g: torch.Tensor, dt: float, tau: TimeConstants, *, max_step: float | None = None) -> None`
  - `max_step=None` は `tau.h / 2`。`n = max(1, ceil(dt / max_step))`、`s = dt / n` で n 回: `u -= (s / tau.h) * (u + lam + g)`、続けて**更新後の** `u` で `lam += (s / tau.lam) * u`。`no_grad` の下で in-place（状態の Tensor は `load_state_dict` がその場に復元するので差し替えない）
  - `max_step <= 0` か非有限は `ValueError`
- `def learn_(params: Sequence[torch.Tensor], g_theta: Sequence[torch.Tensor], elapsed: float, tau: TimeConstants, batch: int) -> None`
  - `p -= (elapsed / tau.theta) * g / batch`（`no_grad`、in-place、`zip(strict=True)` で長さ違いは `ValueError`）

### ノード

共通: `period=None` は `tau.h / 4`。`dtype=None` は `torch.get_default_dtype()`（`TensorSpec` と同じ）。`device` は `Node` に渡す。Port の型引数はコアの carry の計画どおり `Port[S, R, M, CS, CR]`。

**`Learner(Node)`**（非公開、`learner.py`）

- `__init__(self, *, tau: TimeConstants, learn_interval: float | None, period: float | None, device: torch.device | str) -> None`
- 状態（`register_state`、`__init__` で登録）: `_seen_time: int | None = None`（最後に見た付随データの `time`）、`_phase: float = 0.0`、`_elapsed: float = 0.0`（前回の θ の更新からの ΣΔt）
- `_tick(self, dt: float, time: int) -> bool`: `_elapsed += dt`。`time != _seen_time` なら `_seen_time = time`・`_phase = 0.0`、そうでなければ `_phase += dt`。学習するかを返す: `math.isfinite(tau.theta) and (learn_interval is None or _phase >= learn_interval - 5e-10)`
  - 許容 0.5 ns の理由: タイマーの分解能は 1 ns で `dt` は整数 ns。float の和の誤差でちょうど K ステップ目を取りこぼさないため
- `_learned(self) -> None`: `_elapsed = 0.0`・`_phase = 0.0`
- `ValueError`: `learn_interval` が `None` でなく、有限の > 0 でない。`TypeError`: `tau` が `TimeConstants` でない

**`pc.Input(Node)`**

- `__init__(self, shape: Sequence[int], *, carry: DictSpec[Carried], period: float, dtype: torch.dtype | None = None, device: torch.device | str = "cpu") -> None`（`period` は既定なしの必須キーワード。確認事項 2 の決定）
- `X = TensorSpec(shape, dtype)`
- `self.prev = PrevPort(recv=DictSpec(Sample, x=X, y=carry.specs["y"], mask=carry.specs["mask"]))`（ホストが書く片方向）
- `self.next = NextPort(send=X, recv=X, send_carry=carry)`（受け取る $g$ は読まない。`recv=X` は `Layer.prev` と接続するため）
- `TypeError`: `carry` が `Carried` の `DictSpec` でない。`ValueError`: `shape[0]` が `carry` のバッチと違う
- 状態なし

**`pc.Layer(Learner)`**

- `__init__(self, module: nn.Module, *, input_shape: Sequence[int], carry: DictSpec[Carried], tau: TimeConstants, learn_interval: float | None = None, period: float | None = None, max_step: float | None = None, dtype: torch.dtype | None = None, device: torch.device | str = "cpu") -> None`
- 構築時に `module(zeros(input_shape, dtype, device))` を `no_grad` で 1 回通し、出力の shape と dtype で `H` を決める
- `self.prev = PrevPort(send=X, recv=X, recv_carry=carry)`、`self.next = NextPort(send=H, recv=H, send_carry=self.prev)`
- 状態（自動収集）: `self.module`（$\theta$）、`self.u`、`self.lam`（`H` の zeros、`__init__` で確保）。公開名にする理由: 状態の木のキーがそのまま `"module"`・`"u"`・`"lam"` になり、利用者が監視・取り出しに読む。ノードの外から代入しないと docstring に書く
- 学習するパラメータは `[p for p in module.parameters() if p.requires_grad]`（凍結したものは除く）
- `TypeError`: `module` の出力が Tensor でない、`carry` が違う型。`ValueError`: `input_shape[0]` か出力の先頭次元が `carry` のバッチと違う、`module` のパラメータ・バッファが `device` に無い、`max_step` が非正か非有限。`module` が `input_shape`・`dtype` を受け付けないときは PyTorch の例外がそのまま出る

**`pc.Output(Learner)`**

- `__init__(self, module: nn.Module, *, loss: Callable[[torch.Tensor, torch.Tensor], torch.Tensor], input_shape: Sequence[int], carry: DictSpec[Carried], tau: TimeConstants, learn_interval: float | None = None, period: float | None = None, dtype: torch.dtype | None = None, device: torch.device | str = "cpu") -> None`
- 構築時に `module` を 1 回通して `Ŷ` の spec を決め、`loss(ŷ, carry y の zeros)` が `(B,)` を返すか確かめる（`nn.MSELoss()` のようにスカラーを返す損失は `ValueError` で、メッセージで `pc.mse` を案内）
- `self.prev = PrevPort(send=X, recv=X, recv_carry=carry)`、`self.next = NextPort(send=DictSpec(Prediction, y_hat=Ŷ, y=carry y, mask=carry mask, time=carry time))`（ホストが読む片方向。carry は使わない）
- 状態: `self.module` と `Learner` の 3 つ。`tau.h` は周期の既定にだけ使う
- 例外は `Layer` と同じ（`max_step` を除く）

## 理論上の対応

式は `specs/pc-module.md`「一般化した定式化」「状態はずれ u で持つ」。

- `Layer`: $\mu=f(h_{\mathrm{in}};\theta)$、$\delta=u+\lambda$、$g_{\mathrm{in}}=-J_h^\top\delta$、$g_\theta=-J_\theta^\top\delta$、$\tau_h\dot u=-(u+\lambda+g)$、$\tau_\lambda\dot\lambda=u$、$\theta\leftarrow\theta-\frac{\Sigma\Delta t}{\tau_\theta}\frac{g_\theta}{B}$。$g$ は `next` の全チャネルの和
- `Output`: $\ell=\sum_b\mathrm{mask}_b\ell_b(f_L(h_{\mathrm{in}})_b,y_b)$、$g_{\mathrm{in}}=\partial\ell/\partial h_{\mathrm{in}}$、$\theta$ も同じ規則
- バッチの規約: VJP はバッチの和、$1/B$ は $\theta$ にだけ（`learn_` の `batch`）。$B$ は `input_shape[0]` で、ラベルのあるサンプルの数ではない（ラベルの無いサンプルは 0 を足す）
- α=0 は `TimeConstants(lam=math.inf)` で `relax_` の `s / inf = 0`、λ は 0 のまま
- 局所性: 各ノードが読むのは自分の Port（`prev.recv`・`prev.carried`・`next.recv`）と自分の状態だけ。`h_in` はスロットからのコピーで葉。autograd はノード内の `module` に閉じ、`functional.vjp` / `loss_vjp` の 2 行だけに `# mujou: local-autograd`。付随データはコアの中継で運び、PC 固有の意味（`y`・`mask`・`time`）は `mujou.pc` 側にだけある
- 非同期性: バリア無し。`time` の変化で位相を戻すだけで、他のノードを待たない。同期の step は特殊なスケジュール

### on_update の手順

`Input.on_update(dt)`:

1. `msg, t = self.prev.recv_with_time()`
2. `self.next.send(msg["x"], carry={"y": msg["y"], "mask": msg["mask"], "time": torch.tensor(t.ns, dtype=torch.int64)})`（毎回送る。同じ書き込みなら `time` は同じなので、次のノードの位相は戻らない）

`Layer.on_update(dt)`（`specs/pc-module.md` の手順 1〜7）:

1. `h_in = self.prev.recv()`、`c = self.prev.carried()`（`time`・`mask`）、`g = Σ self.next.recv().values()`（チャネルが無ければ zeros）
2. `learn = self._tick(dt, int(c["time"]))`。`learn` かつ `c["mask"]` がすべて False なら、学習せず `self._learned()` で ΣΔt を 0 に戻し、`learn = False` とする（持ち越した δ による学習を避ける。確認事項 3 の決定）
3. `mu, g_in, g_theta = vjp(self.module, h_in, self.u + self.lam, params if learn else ())`（**更新前の** δ）
4. `self.prev.send(g_in)`
5. `relax_(self.u, self.lam, g, dt, tau, max_step=...)`
6. `self.next.send(mu + self.u)`（付随データは `send_carry=self.prev` で 1 の `recv` のものが自動で添わる）
7. `learn` なら `learn_(params, g_theta, self._elapsed, tau, B)`、`self._learned()`（mask がすべて False の学習のステップは手順 2 で処理済み）

`Output.on_update(dt)`:

1. `h_in = self.prev.recv()`、`c = self.prev.carried()`（`y`・`mask`・`time` を同じメッセージから）
2. `learn = self._tick(dt, int(c["time"]))`
3. `c["mask"].any()` なら `y_hat, g_in, g_theta = loss_vjp(self.module, h_in, c["y"], c["mask"], self._loss, params if learn else ())`。すべて False なら `no_grad` で `y_hat = module(h_in)`、`g_in = zeros`
4. `self.prev.send(g_in)`
5. `self.next.send({"y_hat": y_hat, **c})`
6. `learn` なら、ラベルがあれば `learn_(...)`、どちらでも `self._learned()`（ラベルの無い区間は $g_\theta=0$ の区間として ΣΔt を消費する。確認事項 3）

## 実装ステップ

前提: carry の PR と register_state の PR が main に入っていること（ステップ 2 からは必須）。ブランチ `feature/<日付>/pc-module`。commit はステップごと。

1. **時定数・付随データ・損失・functional**（`uv run pytest tests/mujou/_core/pc/test_time_constants.py test_carry.py test_losses.py test_functional.py`、`just type`）
   - `time_constants.py`・`carry.py`・`losses.py`・`functional.py`。docstring に式と shape（`h_in: (B, *in)` など）、符号（$g$ はエネルギーの勾配で、送る側が負号を付ける）、バッチの規約を書く
   - `carry.py` はコアの carry に依存しない（`DictSpec` だけ）ので、carry の PR を待たずに書ける
2. **`Learner`・`Input`・`Layer`・`Output`**（`uv run pytest tests/mujou/_core/pc/`、`just type`）
   - 上の手順どおり。`state_dict` / `load_state_dict` は override しない（自動収集と `register_state` で足りる）
   - docstring に例（`super()` を呼ぶ形、`feedback_super_calls.md`）、状態の木のキー、`module` はサンプルごとに独立であること（学習モードの BatchNorm・Dropout は不可）
3. **公開**（`src/mujou/pc/__init__.py`・`functional.py`）と公開名のテスト（`tests/mujou/test_init.py`）
4. **MNIST の example**（下記。手で 1 回走らせて数値を docs と `specs/pc-module.md` に記録）
5. **docs と仕様**（`just docs-build`、最後に `just run`）
   - `docs/en/user-guide/pc.md`・`docs/ja/user-guide/pc.md`「Learning with PC-ALM / PC-ALM で学習する」: 3 つのノードと接続、`carry_spec` と `Run.write` に書く `Sample`、時定数と周期の既定、`learn_interval`、教師なしのサンプル（`mask`）とテスト精度、`Output.next` で prequential 精度を測る方法、`pc.functional` で自分のノードを書く方法、バッチと損失の規約、BP との違い。コードは `examples/mnist/` から snippets で取り込む
   - `mkdocs.yml`: nav の User Guide に `Learning with PC-ALM: user-guide/pc.md`（Testing の前）、API Reference に `PC-ALM: api/pc.md`（Testing の前）。`nav_translations` に `Learning with PC-ALM: PC-ALM で学習する`
   - `docs/en/api/pc.md`: `::: mujou.pc`（`members` を `__all__` の順に、`inherited_members: false`。Node の hook を繰り返さない）と `::: mujou.pc.functional`。`docs/en/api/index.md` の表に 1 行と「`import mujou.pc` が要る」旨
   - `docs/{en,ja}/user-guide/index.md` のカード、`docs/{en,ja}/examples/index.md` の表と説明に MNIST
   - `specs/pc-module.md`: 状態を「実装済み」に、確定した API（`Sample`・`Prediction`・`loss_vjp`・`Input` の周期・ラベルの無い学習の扱い）、MNIST の結果、`S` の下限の測り直し。skill `pc-alm` 7 節に一般化（任意の $f$、出力ノードの分離、u の形）と実験結果を追記（`/edit-dot-claude`）
   - `memory/project_pc_module.md`: テストの場所を `tests/mujou/_core/pc/` に直し、確定事項を 1〜2 行

### MNIST の example（`examples/mnist/`、テストでは実行しない）

- `data.py`: <https://ossci-datasets.s3.amazonaws.com/mnist/> の `train-images-idx3-ubyte.gz` ほか 4 本を `urllib.request` で `--data`（既定 `~/.cache/mujou/mnist`）にキャッシュし、`gzip` + `numpy.frombuffer` で読む（magic 2051 / 2049 を確かめる）。訓練データの平均と標準偏差で画素を標準化。`batches(images, labels, batch, generator) -> Iterator[Sample]`: 端数のバッチは 0 で埋め `mask=False`
- `bp.py`: 同じ初期値（`copy.deepcopy` した 3 つの module を `nn.Sequential` に）、同じバッチ順で素の PyTorch の SGD。1 バッチの学習率 0.1、損失は平均の CE（= PC の $\Sigma\Delta t/\tau_\theta \cdot g_\theta/B$ と同じ量）
- `main.py`:
  - 既定値: `B=64`、784→256→256→10、relu、`pc.cross_entropy`、`tau = pc.TimeConstants(h=4e-3, lam=8e-3, theta=S * tau_h/4 / 0.1)`（周期 1 ms、$\tau_\lambda=2\tau_h$、1 バッチの学習率 $\Sigma\Delta t/\tau_\theta = S\cdot\text{周期}/\tau_\theta = 0.1$）、`S=32`、`learn_interval = 24 * 周期`（S の 3/4）、3 epoch、float32、`torch.set_num_threads(1)`
  - `timer = mujou.time.Timer()` に `--time-scale` を設定して `mujou.run(net, timer=timer)`。訓練: バッチごとに `t = run.write({inp.prev: sample})`、`run.wait_for(out.next, lambda p: int(p["time"]) == t.ns)` で最初の予測を取り prequential 精度に足し、`timer.sleep` で書き込みから `S * 周期` まで待つ
  - テスト精度: 各 epoch の後、テストのバッチを `mask=False` で同じく `S` 周期ずつ見せ、次を書く直前の `run.read(out.next)`（`time` が一致するもの）で測る。推論モードを持たないことの実演
  - 速さ: 計算の限界で回すため `--time-scale` で時間を加速する。最初の 100 バッチで `subscribe(out.next)` の時刻の差の中央値を測り、周期の何倍で `Output` が更新されたかを表示する。1.5 倍を超えたら警告（`dt` が伸びて不安定になる）。既定値は手元で測って決め、docstring に手順を書く
  - 表示: epoch ごとの prequential 精度・テスト精度、BP のテスト精度、ある精度（例 97%）に達するまでの実時間、1 サンプルあたりの forward と VJP の回数（PC は各ノード `S` 回、BP は 1 回ずつ）、使ったワーカー数
  - docstring: ネットワークからのダウンロード（約 11 MB）、数分かかること、CPU で動くこと

## テスト観点

すべて `tests/mujou/_core/pc/`、float64（`dtype=torch.float64` と `module.double()`）、`mujou.testing.run` と `step` fixture（1 ms）で決定的に回す。読むのは公開の口（`Run.read`・`Run.state_dict()["nodes"][path]["u"]` など）だけ。基本の時定数は `tau.h = 4 ms`（η=0.25）。

| ファイル | クラス | 中身 |
| --- | --- | --- |
| `test_functional.py` | `TestVjp` | テスト 1（Layer 側）: $E=\tfrac12\|h-f(h_{\mathrm{in}})+\lambda\|^2$ を素で書き、`h_in`・各パラメータの中心差分（eps 1e-6、rtol 1e-7）と `g_in`・`g_theta` が一致。Linear+tanh と ResBlock。`wrt_input=False` で `None`、使われないパラメータは 0、`delta` の shape 違いで `ValueError` |
| | `TestLossVjp` | テスト 1（Output 側）: $\sum_b\mathrm{mask}_b\ell_b$ の中心差分と一致（mse と CE）。mask が False の行の `g_in` は 0、`y` に NaN を入れたサンプルを mask しても結果が有限。`loss` が `(B,)` を返さないと `ValueError` |
| | `TestRelax` | `max_step` で n 分割した結果が、`mu`・`g` 固定の刻み `dt/n` の陽的 Euler n 回（テスト内で素に書く）と一致（テスト 9 の後半）。`dt <= max_step` で 1 回、`lam=inf` で λ が厳密に不変、`dt=0` で不変、λ が**更新後の** u で進む（1 ステップの閉形式） |
| | `TestLearn` | $\theta$ の変化が $-(\Sigma\Delta t/\tau_\theta)g/B$、`theta=inf` で不変、長さ違いで `ValueError` |
| `test_losses.py` | `TestMse` / `TestCrossEntropy` | `(B,)` を返し、mse は $\tfrac12$ の和、shape 違いで `ValueError`。CE の平均が `nn.CrossEntropyLoss()` と一致（1/B を損失に入れていないことの確認） |
| `test_time_constants.py` | `TestTimeConstants` | `h` の 0・負・inf・NaN、`lam`・`theta` の 0・負・NaN で `ValueError`、`inf` は受け付ける、位置引数は `TypeError` |
| `test_carry.py` | `TestCarrySpec` | フィールドの spec、2 回の呼び出しが等しい、`target.shape[0] != batch` で `ValueError` |
| `test_input.py` | `TestInput` | 書いた `x` が `next` から出て、付随データの `y`・`mask` が同じ書き込みのもの、`time` が `Run.write` の返す ns。書き直すと `time` が変わる。`Layer.prev` と接続でき、バッチ違いの `carry` で `ValueError` |
| `test_layer.py` | `TestConstruction` | 出力の shape を probe で決める、`period` の既定が `tau.h/4`、明示すればそれ、各 `ValueError` / `TypeError` |
| | `TestJacobi` | テスト 2（下の参照実装と 1e-12） |
| | `TestFixedPoint` | テスト 3（`slow`） |
| | `TestPcLimit` | テスト 4（`slow`） |
| | `TestScalar` | テスト 5 |
| | `TestSchedules` | テスト 6（`slow`） |
| | `TestWithoutTeacher` | テスト 7 |
| | `TestLearnInterval` | テスト 9（前半）、`max_step` を超える `dt` で `relax_` と同じ分割になる |
| | `TestState` | テスト 10 |
| | `TestUnlabeled` | mask がすべて False の学習のステップで $\theta$ が不変、ΣΔt が戻る（確認事項 3） |
| `test_output.py` | `TestConstruction` | `nn.MSELoss()` で `ValueError`、`Prediction` の spec |
| | `TestPairing` | テスト 8 |
| | `TestUnlabeled` | mask がすべて False の学習のステップで $\theta$ が不変、ΣΔt が戻り、次のラベルのある学習が直前の区間の ΣΔt だけで更新する |
| `tests/mujou/test_init.py` | `TestPcPublicNames` | `sorted(mujou.pc.__all__)`、`"pc"` がトップレベルの `__all__` に無い。`tests/mujou/pc/test_init.py` にしないのは、テストに `__init__.py` が無く basename が衝突するため |

### 各テストの中身

- **テスト 2（Jacobi）**: Input → Layer(Linear+tanh) → Layer(ResBlock) → Output(Linear, mse)、B=3、`learn_interval=None`（毎ステップ学習）、`tau.theta` は有限。各ステップ `step(run, out, l2, l1, inp)` を 50 回、毎回 `u`・`lam`・`module` の重み（`state_dict`）と各ノードの送った値（`Run.read`）を参照実装と 1e-12 で比べる
  - 上から回すと一致する理由: `Output` は `l2` の前のステップの $h$ を読み、更新前の $\theta_L$ で $g$ を作る。`l2` が読む $g$ は今のステップに作られたが、前のステップの状態だけの関数。`l2` が読む `h_in` は `l1` がまだ回っていないので前のステップの値。`Input` は最後に回るので `l1` は前のステップの `x` を読む。どのノードも前の時刻の状態だけから今の時刻を作るので、全ノードを同時に進める Jacobi と同じになる
  - 参照実装は `test_layer.py` のモジュール関数 `reference(...)`（素の PyTorch、状態はリスト）: 状態 $u_i,\lambda_i,\theta_i$ と、送った値 $h_i$（初期値 0）と、$h_i$ に添う付随データ（初期値 0、mask False）。1 ステップで、前の時刻の値から全ノードの $\mu_i=f_i(h_{i-1})$・$g_{\mathrm{in}}$・$g_\theta$ を計算し（$h_0$ は `Input` が前のステップに送った x、最初は 0）、$u,\lambda$ を Euler で進め、$h_i=\mu_i+u_i$、付随データを 1 段ずらし、$\theta$ を更新する。$u$ で持つので、$h$ は前の層の 1 ステップ前の値を通るパイプラインになる点も同じ
  - `functional` を使わず autograd を直接書く（実装を写さない）
- **テスト 3**: 入力を固定し `tau.theta=inf` で 3000 ステップ程度（収束が 1e-10 に届く数を spec-test-author が決める）回した不動点で、$u\approx0$、$\lambda_i$ がネットワーク全体を autograd した $-\partial\ell/\partial h_i$ と 1e-8 で一致。そのあと `tau.theta` 有限の別インスタンスで `learn_interval` をちょうど不動点に届くステップに置き、$\theta$ の 1 回の更新がテスト内の BP（平均損失、学習率 $\Sigma\Delta t/\tau_\theta$）と一致。線形・tanh × Linear・ResBlock を parametrize。$f=\sigma(Wh)$ では skill `pc-alm` 4 節の漸化式 $\lambda_{L-1}=W_L^\top e_L$、$\lambda_i=W_{i+1}^\top(\sigma'(a_{i+1})\odot\lambda_{i+1})$ とも比べる
- **テスト 4**: `lam=inf` で全ステップ λ が厳密に 0、不動点で $u=-g\ne0$、その $h$ で PC のエネルギー $\tfrac12\sum\|r_i\|^2+\ell$ の $\nabla_h$ が 0（テスト内の autograd）、$h$ が forward の値と違う
- **テスト 5**: B=1、`Linear(1,1,bias=False)` 2 つ、mse。不動点で $\lambda=w_2(y-\hat y)$
- **テスト 6**: 上から・下から・`random.Random(0)` でステップごとに並べ替えた順で、同じ不動点（λ を 1e-8）
- **テスト 7**: mask を False、`u`・`lam` は 0 から。経路のノード数ぶんのステップで各 `next` の値が forward の値と厳密に一致（`torch.equal`）、その後も $u=\lambda=0$、学習しても $\theta$ が不変
- **テスト 8**: Input → Output（Linear）で、ホストが毎ステップ別のサンプル（$x_k$, $y_k$, 一部 mask False）を書く（伝播より速い）。`Output.next` の `y`・`mask`・`time` が、その `y_hat` を作った `x` と同じ書き込みのもの。`Output.prev` の $g$ が、その組で計算した $\partial\ell/\partial h_{\mathrm{in}}$ で mask が False の行は 0。$\theta$ の更新もその組の勾配
- **テスト 9（前半）**: Input → Layer → Output、`learn_interval = K * H`、ホストが M ステップごとに入力を変える。$\theta$ が変わるのは各ノードが新しい `time` を見てから K ステップ目ごと（ほかのステップは `torch.equal` で不変）。変化量が、直前の `state_dict` と `Run.read` の `h_in` から `functional.vjp` で作った $g_\theta$ に $-(\Sigma\Delta t/\tau_\theta)/B$ を掛けたもの（$\Sigma\Delta t$ = 前回の学習からのステップ数 × H。入力の切り替えでは戻らない）
- **テスト 10**: テスト 9 の構成で、位相の途中・付随データの中継が「recv 後に前のノードが更新済み」となる時点で `state_dict` を取り、新しいインスタンスに `mujou.testing.run(..., state=tree)` で復元して続けた結果と、止めずに回した結果が `torch.equal`（`u`・`lam`・重み・送った値・付随データ）。木に `_seen_time`・`_phase`・`_elapsed` が入っていることも見る

## 想定リスク・トレードオフ

- **依存**: ステップ 2 以降はコアの carry の PR が必須。carry の API は計画どおり（`send_carry=` / `recv_carry=`、`send(msg, carry=)`、`carried()` / `carried(key)` は所有ノードのデバイスへのコピー、`recv` 前はゼロ、中継のスナップショットは木の `"carries"` に入る）と仮定した。変われば `Input`・`Layer`・`Output` の手順 1 と 6 だけが変わる
- **`Input` の次の `Layer` の $g_{\mathrm{in}}$**: 読み手が捨てるが MVP では毎回計算する（784→256 で forward と同程度の計算）。省く口（`Layer(..., send_grad=False)` と片方向の Port）は後で
- **ラベルの無いバッチでは `Layer` も学習しない**（確認事項 3 の決定）: 付随データの `mask` がすべて False の学習のステップでは θ を動かさず ΣΔt を 0 に戻す。前のバッチから持ち越した $u,\lambda$ による小さな $g_\theta$ で重みが動くのを避ける。一部だけ False のバッチは持ち越した δ を含めて学習する
- **`mask.any()`** は CUDA で同期を 1 回生む。CPU が既定なので MVP では許容
- **`Output` の $1/B$** はラベルのあるサンプルの数でなく $B$。ラベルが少ないバッチでは更新が小さくなる（平均の分母を変える案は採らない。`specs/pc-module.md` の規約どおり）
- **位相を float の和で持つ**: 0.5 ns の許容で K ステップ目を取りこぼさない。タイマーの時刻（ns）で持つ案は、`relax_` が使う `dt` と別の時間の取り方になるので採らない
- **テストの時間**: 不動点のテストは数千ステップを手動で回すので `slow`。テスト 2 は参照実装との一致で同期の正しさを、テスト 3〜6 は不動点の性質を受け持ち、二重にしない
- **u の形の過渡は記事と一致しない**: オラクルは参照実装（同じ u の形）と不動点だけ。記事の離散更新との一致は使わない（`specs/pc-module.md`）
- **公開の状態名** `module`・`u`・`lam` は外から代入できてしまう。property にすると状態に集まらないので、docstring で禁じるにとどめる
- **example の速さの比較**はワーカー数とタイマーのスケールに左右される。表示に条件を必ず出す

## 確認事項（2026-10-20 orchestrator の決定）

1. **公開名の追加**: `pc.Sample`・`pc.Prediction`・`pc.functional.loss_vjp` を足す（承認）
2. **`pc.Input` の周期**: `period` は既定なしの必須キーワード引数にする
3. **ラベルの無い区間**: `Output` は学習のステップで mask がすべて False なら θ を動かさず ΣΔt を 0 に戻す。`Layer` も付随データの `mask` を読み、すべて False なら同じく学習せず ΣΔt を 0 に戻す（持ち越した δ による学習を避ける）。テスト観点: `test_layer.py` の `TestUnlabeled` を足す（mask がすべて False の学習のステップで θ が不変、ΣΔt が戻る）

## 参照

- `specs/pc-module.md`（`git show docs/2026-10-19/pc-module-plan:specs/pc-module.md`）、`specs/pc-alm-continuous-time.md`、`specs/update-rules.md`
- `memory/agents/implementation-planner/carry.md`（Port の型引数・`send_carry` / `recv_carry`・`carried`）
- `src/mujou/_core/node.py`（`register_state` 278、`load_state_dict` は Tensor をその場に、ほかは deepcopy で代入 338）、`port.py`（`Port.__init__` 99、`send` 307）、`spec.py`（`DictSpec` 180 付近）、`run.py`（`write` 102、`wait_for` 193、`ManualRun.step` 378）、`timer.py`（`Timer` 17）、`src/mujou/testing.py`
- `tests/mujou/conftest.py`（`virtual`・`step`・`H`）、`tests/mujou/test_init.py`
- `examples/torch_node.py`（ノード内 BP と `dt` の分割の書き方）、`examples/README.md`、`docs/en/examples/index.md`、`mkdocs.yml`（nav、`nav_translations`、mkdocstrings の設定）
- skill `pc-alm`（3・4・6 節）、`locality-and-async`、`testing-strategy`、`refactor-conventions`、memory `feedback_direction_names.md`・`feedback_super_calls.md`・`feedback_examples_are_guides.md`
