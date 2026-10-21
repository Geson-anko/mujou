# 🧵 スレッドとデバイス

実行が計算に使うスレッド、ノードの設定（`device`・`seed`・`dedicated_thread`）、CUDA、プロセス全体で共有する torch の設定を説明する。

## `Node` の引数

`super().__init__` は次のキーワード引数を取る。いずれも省略可能。

| 引数               | 意味                                             | 既定               |
| ------------------ | ------------------------------------------------ | ------------------ |
| `period`           | 更新と更新の間の平均の時間。実行のタイマーの秒   | `0.01`             |
| `seed`             | ノード専用の乱数生成器 `self.generator` の seed  | `None`（毎回違う） |
| `device`           | ノードが計算する場所。`"cpu"` か CUDA のデバイス | `"cpu"`            |
| `dedicated_thread` | 実行のワーカーでなく専用のスレッドで動かす       | `False`            |

ノードはこれらを読み取り専用の property `period`・`generator`・`device`・`dedicated_thread` で読める。

## ワーカー

実行はノードごとにスレッドを立てない。ワーカースレッドのプールを作り、ワーカーがノードを分け合って動かす。ワーカーは 1 つのノードの hook を 1 回実行したら、期限の早い更新から次の仕事を取る。ノードが多くてもスレッドは少ないままで、OS がスレッドの切り替えに時間を取られない。

- ワーカーの数は `mujou.run(net, workers=n)` で決める。既定はプロセスが使える CPU の数（論理コア数。CPU の affinity や割り当ての上限があればそれより少ない）、GIL が有効ならその半分で、ワーカーに載るノードの数を超えない
- 1 つのノードの hook が同時に走ることはなく、前の hook が書いた値は次の hook から見える。どのワーカーがノードを動かすかは hook ごとに変わりうる
- ブロッキングする hook はその間ワーカーをふさぎ、ほかのノードの `dt` が伸びる（[ノードが守る規則](../concepts.md#%E3%83%8E%E3%83%BC%E3%83%89%E3%81%8C%E5%AE%88%E3%82%8B%E8%A6%8F%E5%89%87)）

GIL が有効なとき（GIL ありのビルド、または free-threaded のビルドで `PYTHON_GIL=1` のとき）は、Python のコードを同時に実行できるスレッドは 1 本だけなので、ノードが並列に動くのは torch の演算の中に限られる。CPU の数の半分を超えてワーカーを増やしても GIL を取り合うだけなので、既定を半分にしている。Python の処理が大半を占めるノードなら、さらに少なくても速さはほとんど変わらない。

## 専用のスレッド

ノードが専用のスレッドを要るときは `dedicated_thread=True` を渡す。

- hook がデバイス・ソケット・lock を待ってブロッキングする。ワーカーの上だと、その間ワーカーをふさぐ
- 1 回だけ設定するスレッドごとの状態に頼る。`on_start` はこのノードのスレッドで呼ばれるので、そこで設定する

```python
class Camera(Node):
    def __init__(self) -> None:
        super().__init__(period=1 / 30, dedicated_thread=True)
```

## スレッドごとの設定

ワーカーの上では、ノードの hook ごとに別のスレッドで走りうるうえ、その間スレッドはほかのノードを動かす。hook がスレッドに設定したものを次の hook が見るかどうかはスケジュール次第で、同じワーカーで次に走る別のノードにも見える。

- `threading.local`・`threading.current_thread()`・`threading.get_ident()` は hook ごとに変わりうる
- `torch.autocast` や勾配の記録などスレッドごとの torch の設定は、hook の中の `with` に閉じ（`with torch.no_grad():`）、hook を抜ける前に戻す。戻さないと同じワーカーで次に走るノードに漏れる
- `contextvars` は持ち越される。各ノードは自分専用の `contextvars.Context` の中で hook を実行する。この文脈は空から始まり、ホストの文脈を引き継がない

mujou 自身が設定するのは、CUDA のノードの hook の前ごとの現在のデバイスとストリームだけ。

## seed と乱数

乱数はノード自身の生成器 `self.generator` から引く。生成器は `seed` から作られ、ノードのデバイス上にある。`super().__init__()` の直後から、`__init__` の中でも使える。

```python
nn.init.normal_(self.model.weight, std=0.1, generator=self.generator)
noise = torch.randn(4, generator=self.generator, device=self.device)
```

`seed` で再現できるのは `self.generator` だけ。`nn.Linear` などの既定の初期化は全ノードが共有する生成器を使うので、`self.generator` で初期化し直す。ノードの中で `torch.manual_seed` を呼ばない。ほかのノードの乱数まで変わる。`self.generator` の状態はノードの状態に入るので、再開した実行は同じ乱数列の続きから引く。

## デバイス

`device` は CPU か CUDA のデバイス。それ以外（`"mps"` や `"meta"` など）は `ValueError` になり、CUDA が使えない環境での CUDA も `ValueError` になる。受け取ったメッセージはノードのデバイスに届くので、テンソルは `device=self.device` で確保する。番号を付けない `"cuda"` は、ノードを作った時点のホストの現在のデバイスになる。

CUDA のノードには、mujou が次のことをする。

- ノードの各 hook の中で、ノードのデバイスとノード専用の CUDA ストリームを現在のものにする。ノードの演算はすべてこのストリームに積まれる
- 各 `on_update` の後と状態の復元の後に、そのストリームに積んだ仕事を待つ（その間は、ブロッキングする hook と同じくワーカーが待つ）
- `state_dict` の前後にも待つ。ワーカーでなく mujou の別のスレッド（実行の終了後はホストのスレッド）で行うので、ワーカーは待たない

そのためノードは同期を書かなくてよい。自分で同期するときは自分のストリームだけを待つ（`torch.cuda.current_stream().record_event().synchronize()`）。デバイス全体を待つ `torch.cuda.synchronize()` はほかのノードの仕事まで待つので使わない。ストリームはデバイスごとに 32 本のプールから割り当てるので、1 つのデバイスに 32 を超える CUDA のノードを置くと、いくつかのノードが 1 本を共有する。

!!! tip "CUDA のテンソルとモジュールは `on_start` で作る"

    `mujou.run` がノードを起動する前にホストのストリームを同期するので、`__init__` で CUDA に作っても正しく動く。それでも `on_start` 以降で作るほうがよい。PyTorch のキャッシュアロケータはメモリを確保した時点のストリームに結び付けるので、`__init__` で確保したメモリはホストのストリームに結び付いたまま残る。ノードがそのテンソルを捨てると、ノードの計算が終わる前にホストの確保に再利用されうる。復元は `on_start` の後に状態を読み込むので、そこで作った状態も復元される。

CPU のノードは CUDA に触れない。直前に CUDA のノードを動かしたワーカーではそのノードのストリームが現在のまま残るが、CPU のノードはそれを使わない。

状態の木は CUDA のノードの状態も CPU のコピーで持ち、既定の `load_state_dict` がノードのデバイスへコピーし直す。ただし乱数生成器の状態は CPU と CUDA で違うので、CUDA のノードから保存した木を復元できるのは CUDA のノードだけ（別の GPU のマシンでもよい）。

## プロセス全体の設定

次の設定はプロセス全体で共有されるので、ノードの中では変えない。ホストが `mujou.run` の前に 1 回だけ設定する。

- `torch.manual_seed` と既定の乱数生成器
- `torch.set_num_threads`、`torch.use_deterministic_algorithms`、`torch.backends.*` のフラグ
- `torch.cuda.synchronize()`、`torch.cuda.empty_cache()` などデバイス全体に効く操作

CPU 上の torch の演算は、計算するスレッドごとにさらに内部のスレッドを使う。多すぎるとノード同士でコアを取り合い、全体が遅くなるので、ホストが少なくしておく。

```python
--8<-- "examples/online_learning.py:threads"
```

!!! tip "スレッド数の目安"

    ワーカーの数に `dedicated_thread=True` のノードの数を足し、`torch.set_num_threads(n)` の `n` を掛けた値が物理コア数を超えないようにする。小さな行列を扱うノードが多いときは `n = 1` でよい。
