# 🔥 ノードの中の PyTorch

ノード内の計算には PyTorch を使う。このページでは更新手法、デバイスと乱数、ノード内での誤差逆伝播（BP）、CUDA の扱い、プロセス全体に効く設定を説明する。

## 更新手法

`on_update(dt)` に書く更新が、そのノードの更新手法（学習則）である。mujou はノード・Port・実行の基盤に更新手法を組み込まないので、ノードごとに選んで差し替えられる。

- 既定の更新手法は PC-ALM（Augmented Lagrangian Predictive Coding、<https://pub.sakana.ai/pc-alm/>）。局所的な更新で BP に近い性能を出せると報告されている。ただし暫定的な選択である
- BP も 1 つのノードの内側に閉じていれば、更新手法の 1 つとして使える（下の「ノードの中の BP」）

!!! note "PC-ALM のノードは未実装"

    PC-ALM のノードはモジュール `mujou.pc` から提供する予定。それまでは `mujou.Node` を継承し、`on_update(dt)` に更新式を自分で書く（[ノードを書く](writing-a-node.md)）。

## 例

次のノードは入力から目標への線形写像をオンラインで学び、予測を送る。BP はこのノードの中に閉じている。

```python
--8<-- "examples/torch_node.py:node"
```

[サンプル](../examples/index.md)の `online_learning.py` では、同様のノードが実行中に目標の変わるストリームから学び、その変化に追従する。

## デバイスと seed

デバイスと乱数の seed は `__init__` で `super().__init__(device=..., seed=...)` に渡す。

- `device` は CPU（既定の `"cpu"`）か CUDA のデバイスで、mujou はこの 2 つだけをサポートする。それ以外のデバイス（`"mps"` や `"meta"` など）は `ValueError` になる。CUDA が使えない環境で CUDA を渡しても `ValueError` になる
- 受け取ったメッセージ（`recv()` の値）はノードのデバイスに届く
- テンソルは `device=self.device` を付けて確保する

## 乱数生成器

乱数はノード自身の生成器 `self.generator` から引く。生成器は `seed` から作られ、ノードのデバイス上にある。

```python
nn.init.normal_(self.model.weight, std=0.1, generator=self.generator)
noise = torch.randn(4, generator=self.generator, device=self.device)
```

- `self.generator` は `super().__init__()` が作るので、`__init__` の中から使える
- ノードの中で `torch.manual_seed` を呼ばない。既定の生成器はプロセス全体で共有されるので、ほかのノードの乱数まで変わる
- 復元では新しいインスタンスを使うので、生成器も `seed` から作り直す。状態から復元した後も同じ乱数列を続けたいときは、生成器の状態を状態の木に入れる（[状態の保存と復元](state.md)）

## ノードの中の BP

既定の更新手法 PC-ALM は更新式を直接計算し、autograd のグラフを必要としない。そのため mujou はどのノードでも `state_dict` 以外の hook の前に毎回勾配の記録を切っている（`torch.set_grad_enabled(False)`）。

ノードの中で autograd を使うときは次のように書く。

1. 順伝播と勾配の計算（`torch.autograd.grad` か `.backward()`）の両方を `with torch.enable_grad():` で囲む
2. その `with` の行に、BP がこのノードの中に閉じていることを示す印 `# mujou: local-autograd` を付ける
3. パラメータはブロックの外でその場で更新する（例の `p -= h * self.rate * g`）。ブロックの外では勾配を記録しないので、更新が autograd のグラフに入らない

BP はノードの中に閉じ、ほかのノードへ勾配を流すことはできない。`recv()` が返す値はコピーで、`send()` は値を `detach()` してから送る。

- `torch.inference_mode()` は使わない。その中で作ったテンソルは backward に使えない
- `.backward()` で `.grad` に溜めた勾配と optimizer の状態は、状態の木に自動では入らない。復元後も同じように続けたいときは `state_dict` を override して足す（[状態の保存と復元](state.md)）
- 更新量は `dt` に比例させる。例では `dt` を長さ `MAX_STEP` 以下の `n` 個の刻み `h = dt / n` に分け、刻みごとに勾配を計算し直して `n` 回更新する。こうすると `dt` が長くても陽的 Euler 法が発散せず、経過時間も失われない

## CUDA

`device` に CUDA を渡したノードには mujou が次を設定する。

- ノードの hook の中の現在のデバイスをノードのデバイスにする
- ノード専用の CUDA ストリームを作り、ノードの hook の中の現在のストリームにする。ノードの演算はすべてこのストリームに積まれる
- 各 `on_update` の後と状態の復元の後に、ノードのストリームを同期する。その間はブロッキングする hook と同じくワーカーが待つ（[ネットワークを走らせる](running.md#%E5%AE%9F%E8%A1%8C%E3%81%AE%E3%82%B9%E3%83%AC%E3%83%83%E3%83%89)）
- 状態を取得するときは `state_dict` の前後でノードのストリームを同期する。ワーカーでなく mujou の別のスレッド（実行の終了後はホストのスレッド）で行うので、ワーカーは待たない

デバイスとストリームはワーカーが CUDA のノードに切り替わるときに設定する。CPU のノードは設定しないので、直前に CUDA のノードを動かしたワーカーではそのノードのストリームが現在のストリームのまま残る。CPU のノードは CUDA を使わない。

そのためノードの作者は同期を書かなくてよい。自分で同期するときは自分のストリーム（`torch.cuda.current_stream().synchronize()`）だけを同期する。

!!! warning "デバイス全体を待たない"

    デバイス全体を待つ `torch.cuda.synchronize()` はほかのノードの仕事まで待つので使わない。

`device="cuda"` のように番号を付けないと、ノード作成時のホストの現在のデバイスになる。

!!! tip "CUDA のテンソルとモジュールは `on_start` で作る"

    `mujou.run` がノードを起動する前にホストのストリームを同期するので、`__init__` で CUDA に確保しても正しく動く。それでも CUDA のテンソルやモジュールは `on_start` 以降で作るほうがよい。PyTorch のキャッシュアロケータはメモリを確保した時点のストリームに結び付けるので、`__init__` で作ったメモリはホストのストリームに結び付いたまま残る。ノードがそのテンソルを捨てると、ノードのストリームの計算が終わる前にホストの確保に再利用されうる。どちらで作ってもカーネルはノードのストリームで走り、ほかのノードと並列に動く。

`run.state_dict()` が返す木では、CUDA のノードの状態は CUDA のテンソルのまま。保存前に `cpu()` で木のテンソルをすべて CPU に移せば、GPU の無いマシンでも読める。復元時は既定の `load_state_dict` がノードのデバイスへコピーし直す（[状態の保存と復元](state.md#%E4%BF%9D%E5%AD%98%E3%81%AE%E5%BD%A2%E5%BC%8F)）。

```python
torch.save(run.state_dict().cpu(), "state.pt")
```

## プロセス全体に効く設定

次の設定はプロセス全体で共有されるので、ノードの中では変えない。必要ならホストが `mujou.run` の前に 1 回だけ設定する。

- `torch.manual_seed` と既定の乱数生成器
- `torch.set_num_threads`、`torch.use_deterministic_algorithms`、`torch.backends.*` のフラグ
- `torch.cuda.synchronize()`、`torch.cuda.empty_cache()` などデバイス全体に効く操作

ノードの中で変えてよいのはスレッドごとの設定だけで、それも hook の中に限る。ノードの次の hook は別のスレッドで走りうるうえ、その間スレッドはほかのノードを動かす（[専用のスレッドで動かす](writing-a-node.md#%E5%B0%82%E7%94%A8%E3%81%AE%E3%82%B9%E3%83%AC%E3%83%83%E3%83%89%E3%81%A7%E5%8B%95%E3%81%8B%E3%81%99)）。設定は hook の中の `with` に閉じ（`with torch.autocast(...):`）、hook を抜ける前に戻す。戻さないと同じワーカーで次に走るノードに漏れる。mujou 自身が戻すのは、`state_dict` 以外の hook の前に毎回切る勾配の記録と、ワーカーがノードに切り替わるたびに設定する CUDA の現在のデバイス・ストリームだけである。`dedicated_thread=True` のノードは全 hook を 1 本のスレッドで実行するので、スレッドごとの設定を `on_start` に 1 回だけ書ける。

### 内部のスレッド数

ノードは実行のワーカースレッド（と `dedicated_thread=True` のノードのスレッド）で計算し、CPU 上の torch の演算はそのスレッドごとにさらに内部のスレッドを使う。内部のスレッドが多いとノード同士で CPU を取り合い、全体が遅くなる。

mujou は内部のスレッド数を設定しないので、ホストが `mujou.run` の前に決める。

```python
--8<-- "examples/torch_node.py:threads"
```

!!! tip "スレッド数の目安"

    ワーカーの数に `dedicated_thread=True` のノードの数を足し、`torch.set_num_threads(n)` の `n` を掛けた値が物理コア数を超えないようにする。小さな行列を扱うノードが多いときは `n = 1` でよい。
