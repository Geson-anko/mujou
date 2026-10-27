# 💡 サンプル

mujou の使い方を示すサンプルプログラムの一覧。どれもリポジトリの `examples/` にある。サイトのページはこれらをそのまま取り込んでいるので、載っているコードはそのまま実行可能。テストではなく手引きなので、テストでは実行しない。

## やりたいことから探す

| やりたいこと                                         | プログラム              | 詳しい説明                                                                                                                       |
| ---------------------------------------------------- | ----------------------- | -------------------------------------------------------------------------------------------------------------------------------- |
| とにかく 1 つ動かす                                  | `quickstart.py`         | [1. ネットワークを走らせる](tutorial/1-run.md)                                                                                   |
| ネットワークを分岐させ、合流させる                   | `branching.py`          | [1. ネットワークを走らせる](tutorial/1-run.md#%E5%88%86%E5%B2%90%E3%81%A8%E5%90%88%E6%B5%81)、[Port と Channel](topics/ports.md) |
| 連続時間で変わる状態を持つノードを書く               | `leaky.py`              | [2. ノードを書く](tutorial/2-node.md)                                                                                            |
| ノードを決定的にテストする                           | `test_leaky.py`         | [3. ノードをテストする](tutorial/3-test.md)                                                                                      |
| ノードの中の BP で、ストリームから学び続ける         | `online_learning.py`    | [4. ノードの中で学習する](tutorial/4-learn-in-a-node.md)                                                                         |
| 自作の則でノードをまたいで学ぶ                       | `feedback_alignment.py` | [更新則](topics/update-rules.md)                                                                                                 |
| PC-ALM で回帰を数十行で学ぶ                          | `pc_regression.py`      | [5. PC-ALM で学習する](tutorial/5-learn-with-pc.md)                                                                              |
| PC-ALM のノードを自分で書く（入力が 2 つ）           | `pc_merge.py`           | [PC-ALM](topics/pc.md#%E8%87%AA%E5%88%86%E3%81%AE%E3%83%8E%E3%83%BC%E3%83%89%E3%82%92%E6%9B%B8%E3%81%8F)                         |
| PC-ALM で MNIST を学び、BP と比べる                  | `mnist/`                | [PC-ALM](topics/pc.md#mnist)                                                                                                     |
| GPU 上の ResNet で CIFAR-10 を学び、BP と比べる      | `cifar10/`              | [PC-ALM](topics/pc.md#cifar-10)                                                                                                  |
| 状態を保存し、その続きから再開する                   | `save_and_restore.py`   | [6. 保存と再開](tutorial/6-save-and-resume.md)                                                                                   |
| 複数のネットワークを一緒に走らせる                   | `group.py`              | [Group](topics/groups.md)                                                                                                        |
| 新しい出力を追い続ける（途中の値は飛ばしうる）       | `subscribe.py`          | [ホストの入出力](topics/host-io.md#%E8%AA%AD%E3%82%80)                                                                           |
| 実行専用のタイマー、時間の速さ、複数の実行の一時停止 | `timers.py`             | [時間](topics/time.md)                                                                                                           |
| ノードが例外を投げたときの動きを知る                 | `errors.py`             | [例外と終了](topics/errors.md)                                                                                                   |
| ネットワークのグラフを描いてブラウザで開く           | `visualize.py`          | [ネットワークの可視化](topics/visualize.md)                                                                                      |

## 各プログラムが示すこと

- `quickstart.py`: 2 ノードのネットワークを `with mujou.run(...)` で走らせる。ホストは `write` で入力を書き、期待した出力が届くまで `wait_for` で待つ
- `branching.py`: 送るだけの `Port` を 2 つの枝につなぎ、`MultiPort` で合流させる。後半は `MultiPort` が `connect` の返した Channel で、相手ごとに違う値を `send_to` で送る
- `leaky.py`: リーク積分器。連続時間で定義し厳密解で進めるので、どんな `dt` でも安定する。実行すると 2 つ並べた鎖の入力を切り替え、各出力を送られた時刻の厳密解と比べる
- `test_leaky.py`: `leaky.py` のノードを `mujou.testing` の仮想タイマーと手動の更新でテストする。厳密解との比較、離散アルゴリズムとのビット単位の一致、再開した実行と止めずに続けた実行の一致。`uv run pytest examples/test_leaky.py` で実行する
- `online_learning.py`: 教師がホストの書いて変える `a` で y = a x + b のバッチを送る。学習するノードは新しいバッチごとに、前のバッチからの時間だけ、素の PyTorch の autograd で学ぶ。ホストは損失が下がり、跳ね上がり、また下がるのを追う
- `feedback_alignment.py`: 2 つのノードが 2 層のネットワークを feedback alignment で学ぶ。両方向の Port、活動と一緒に中継する目標、`pc.clock.LearnClock` によるサンプルごとに 1 回の学習
- `pc_regression.py`: `pc.Chain` と `pc.show` で、流れてくるバッチから回帰を学ぶ。損失は各バッチから学ぶ前に測る
- `pc_merge.py`: 入力を 2 つ持つ PC-ALM のノード `Merge` を `pc.functional` と `pc.clock.LearnClock` で書き、2 本の塔を合流させて回帰を学ぶ
- `mnist/`: 784-256-256-10 のネットワークを `pc.Chain` で MNIST に学ばせ、同じネットワークを同じ初期重みとバッチで BP でも学ばせる。入口は `main.py`。`data.py` がデータを読み、`metrics.py` が測り、`bp.py` が BP の基準を学習させ、`watch.py` がノードの `dt` を数える（`--watch`）
- `cifar10/`: CUDA のデバイスに載せた ResNet-20 を残差ブロックごとのノードに分け、`pc.Chain` と `pc.optim.AdamW` で CIFAR-10 に学ばせる。同じネットワークを同じ初期重み・バッチ・optimizer で、GPU 上の BP でも学ばせる。入口は `main.py`。`data.py` がデータを読んで拡張し、`resnet.py` がネットワークを組んでブロックが緩和をどれだけ硬くするかを述べ、`metrics.py` が測り、`bp.py` が BP の基準を学習させる
- `save_and_restore.py`: `run.state_dict()` で状態を取得し、`torch.save` で保存し、新しいインスタンスに `mujou.run(..., state=...)` で再開する。再開した実行は専用のタイマーで走る
- `group.py`: `Group(enc=..., dec=...)` と、ノードのパス（`"enc.scale"`）の付き方
- `subscribe.py`: `run.subscribe` がまず最新の出力を、その後は新しい出力を 1 つずつ返す。`break` で止める
- `timers.py`: `Timestamp`、時間の速さを変えた実行専用の `Timer`、グローバルタイマーの速さ、グローバルタイマーを止めて 2 つの実行の状態を同じ時刻で取得すること。一方から再開した実行は専用のタイマーで続く
- `errors.py`: ホストが受け取る `NetworkError` とその `__cause__`、`on_exception` → `on_stop` の順、`on_start` の例外は `mujou.run` が送出すること
- `visualize.py`: 2 つのネットワークと 1 つのノードからなるネットワークと `pc.Chain` のページを、走らせずに `mujou.viz.to_html` で書き出す

## 走らせ方

ソースからビルドした環境（[インストール](install.md)）で、リポジトリのルートから実行する。

```sh
uv run python examples/quickstart.py
uv run python examples/mnist/main.py
uv run python examples/cifar10/main.py
```

GPU を使うのは `cifar10/` だけで、CUDA が要る。`mnist/` と `cifar10/` 以外は 15 秒以内に終わる。`mnist/` は初回に MNIST（約 11 MB）をダウンロードし、既定の 3 epoch で約 2 分かかる。`cifar10/` は初回に CIFAR-10（約 160 MB）をダウンロードし、既定の 3 epoch で RTX 4090 なら実時間で約 8 分かかる。
