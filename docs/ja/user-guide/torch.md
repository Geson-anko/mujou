# ノードの中の PyTorch

ノードの内側の計算には PyTorch を使う。このページでは、デバイスと乱数、ノードの中での誤差逆伝播（BP）、CUDA とほかのアクセラレータの扱い、プロセス全体に効く設定を書く。

## 例

次のノードは、入力から目標への線形写像をオンラインで学び、予測を送る。BP は 1 つのノードの中に閉じている。

```python
--8<-- "examples/torch_node.py:19:57"
```

## デバイスと seed

デバイスと乱数の seed は、`__init__` で `super().__init__(device=..., seed=...)` に渡す。

- `device` は CPU（既定の `"cpu"`）か、PyTorch の現在のアクセラレータ（CUDA、MPS、XPU など）のデバイスである。使えないデバイスと、それ以外のデバイス（`"meta"` など）は `ValueError` になる
- 受け取ったメッセージ（`recv()` の値）は、ノードのデバイスに届く
- テンソルは `device=self.device` を付けて確保する

## 乱数生成器

乱数は、ノード自身の生成器 `self.generator` から引く。生成器は `seed` から作られ、ノードのデバイスにある。

```python
nn.init.normal_(self.model.weight, std=0.1, generator=self.generator)
noise = torch.randn(4, generator=self.generator, device=self.device)
```

- `self.generator` はノードのスレッドで作られるので、`on_start` から使える。`__init__` の中で使うと `RuntimeError` になる
- ノードの中で `torch.manual_seed` を呼ばない。既定の生成器はプロセス全体で共有されるので、ほかのノードの乱数まで変わる
- 生成器は起動のたびに `seed` から作り直される。状態から復元した後も同じ乱数列を続けたいときは、生成器の状態を状態の木に入れる（[状態の保存と復元](state.md)）

## ノードの中の BP

mujou は、ノードのスレッドで勾配の記録を切っている（`torch.set_grad_enabled(False)`）。既定の更新手法 PC-ALM（実装予定。[更新手法](update-rules.md)）は更新式を直接計算するので、autograd のグラフは要らないからである。

ノードの中で autograd を使うときは、次のように書く。

1. 順伝播と、勾配の計算（`torch.autograd.grad` か `.backward()`）の両方を `with torch.enable_grad():` で囲む
2. その `with` の行に `# mujou: local-autograd` を付ける。BP がこのノードの中に閉じていることを示す印である
3. パラメータの更新はブロックの外で、その場で行う（例の `p -= h * self.rate * g`）。ブロックの外では勾配を記録しないので、更新が autograd のグラフに入らない

BP はノードの中に閉じる。ほかのノードへ勾配を流すことはできない。`recv()` が返す値はコピーで、`send()` は値を `detach()` してから送る。

- `torch.inference_mode()` は使わない。その中で作ったテンソルは backward に使えない
- `.backward()` で `.grad` に溜めた勾配と、optimizer の状態は、状態の木に自動では入らない。復元しても続きを同じにしたいときは、`state_dict` を override して足す（[状態の保存と復元](state.md)）
- 更新量は `dt` に比例させる。例では、`dt` を長さ `MAX_STEP` 以下の `n` 個の刻み `h = dt / n` に分け、刻みごとに勾配を計算し直して `n` 回更新する。`dt` が長くても陽的 Euler 法が発散せず、経過時間も失われない

## CUDA

`device` に CUDA を渡したノードには、mujou が次を設定する。

- ノードのスレッドの現在のデバイスを、ノードのデバイスにする
- ノード専用の CUDA ストリームを作り、そのスレッドの現在のストリームにする。ノードの演算はすべてこのストリームに積まれる
- 各 `on_update` の直後、`load_state_dict` の後、`state_dict` の前後に、ノードのストリームを同期する

そのため、ノードの作者は同期を書かなくてよい。自分で同期するときは、自分のストリーム（`torch.cuda.current_stream().synchronize()`）だけを同期する。デバイス全体を待つ `torch.cuda.synchronize()` は、ほかのノードの仕事まで待つので使わない。

- `device="cuda"` のように番号を付けないと、ノードを作ったときのホストの現在のデバイスに決まる
- PyTorch はストリームをデバイスごとに 32 本のプールから割り当てる。1 つのデバイスに 32 を超えるノードを置くと、ストリームを共有するノードが出る。結果は変わらないが、同期でほかのノードの仕事も待つ
- `on_update` で起きた CUDA のエラー（device-side assert など）は、直後の同期で表に出る。そのため `NetworkError` のメッセージには、mujou の内部の hook の名前 `_after_update` が出る。原因は `on_update` の中にある

## ほかのアクセラレータ（MPS など）

`device="mps"` のようにほかのアクセラレータを渡したノードにも、CUDA と同じ設定をする。違いはストリームだけで、ノード専用のストリームは作らず、デバイスの現在のストリームで計算する。CUDA でノードのストリームを同期する箇所では、デバイス全体を同期する。

- `device="mps"` のように番号を付けないと、ホストの現在のデバイス（`mps:0`）に決まる
- PyTorch の MPS は、2 つのスレッドから同時に使うとプロセスが落ちる。そのため、動いているすべての実行を合わせて、MPS で計算できるノードは 1 つだけである。2 つ目になる起動は、`mujou.run`（と `mujou.testing.run`）がどのノードも起動する前に `ValueError` にする。ほかのノードは CPU に置くか、1 つのノードにまとめて計算する。MPS のノードが動いている間は、ホストでも MPS の計算をしない
- MPS には float64 が無い。MPS のノードは float64 の spec のメッセージを受け取れない

## プロセス全体に効く設定

次の設定はプロセス全体で共有される。ノードの中では変えない。必要なら、ホストが `mujou.run` の前に 1 回だけ設定する。

- `torch.manual_seed` と既定の乱数生成器
- `torch.set_num_threads`、`torch.use_deterministic_algorithms`、`torch.backends.*` のフラグ
- `torch.cuda.synchronize()`、`torch.cuda.empty_cache()` などデバイス全体に効く操作

ノードの中で変えてよいのは、スレッドごとの設定だけである。スレッドごとの設定は `on_thread_start` に書く。`on_thread_start` は、ノードのスレッドで `on_start` の直前に 1 回呼ばれる hook である。

### 内部のスレッド数

各ノードは自分のスレッドで計算する。CPU の torch の演算は、そのスレッドごとにさらに内部のスレッドを使う。内部のスレッドが多いと、ノード同士で CPU を取り合って全体が遅くなる。mujou は内部のスレッド数を設定しないので、ホストが `mujou.run` の前に決める。

```python
--8<-- "examples/torch_node.py:60:62"
```

目安は、CPU で計算するノードの数と `torch.set_num_threads(n)` の `n` の積が、物理コア数を超えないことである。小さな行列を扱うノードが多いときは `n = 1` でよい。
