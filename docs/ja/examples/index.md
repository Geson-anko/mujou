# 💡 サンプル

mujou の使い方を示すサンプルプログラムの一覧。どれもリポジトリの `examples/` にある。

!!! tip "そのまま動くコード"

    ドキュメントにはプログラムのファイルをそのまま取り込んで載せている。載っているコードはそのまま実行可能。

## 📋 一覧

やりたいことに近い行を探す。「詳しい説明」のページでは、そのプログラムを取り込んで解説している。

| やりたいこと                                       | プログラム            | 詳しい説明                                                                                                                                        |
| -------------------------------------------------- | --------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------- |
| とにかく 1 つ動かす                                | `quickstart.py`       | [クイックスタート](../getting-started/quickstart.md)、[ネットワークを走らせる](../user-guide/running.md)                                          |
| 時間とともに変わる状態を持つノードを書く           | `leaky_chain/`        | [ノードを書く](../user-guide/writing-a-node.md)、[グローバル時間](../concepts.md#%E3%82%B0%E3%83%AD%E3%83%BC%E3%83%90%E3%83%AB%E6%99%82%E9%96%93) |
| ネットワークを枝分かれさせ、合流させる             | `branching.py`        | [ネットワークを組む](../user-guide/building-a-network.md)、[Port と Channel](../concepts.md#port-%E3%81%A8-channel)                               |
| 複数のネットワークを一緒に走らせる                 | `group.py`            | [ネットワークを束ねる](../user-guide/group.md)、[ネットワーク](../concepts.md#%E3%83%8D%E3%83%83%E3%83%88%E3%83%AF%E3%83%BC%E3%82%AF)             |
| 学習した状態を保存し、続きから走らせる             | `save_and_restore.py` | [状態の保存と復元](../user-guide/state.md)、[状態](../concepts.md#%E7%8A%B6%E6%85%8B)                                                             |
| 複数の実行の状態を同じ時刻で取る                   | `pause.py`            | [状態の保存と復元](../user-guide/state.md)、[時間の制御](../user-guide/time.md)                                                                   |
| 最新の出力を受け取り続ける（途中の値は飛ばしうる） | `subscribe.py`        | [ネットワークを走らせる](../user-guide/running.md)                                                                                                |
| ノードの中で PyTorch のモデルを BP で学習させる    | `torch_node.py`       | [ノードの中の PyTorch](../user-guide/torch.md)                                                                                                    |
| 目標が変わっていくストリームから学び続ける         | `online_learning.py`  | [ノードの中の PyTorch](../user-guide/torch.md)、[ノードを書く](../user-guide/writing-a-node.md)                                                   |
| ノードをテストする                                 | `testing_a_node.py`   | [テスト](../user-guide/testing.md)                                                                                                                |
| 時間の進み方を変える                               | `timers.py`           | [時間の制御](../user-guide/time.md)、[グローバル時間](../concepts.md#%E3%82%B0%E3%83%AD%E3%83%BC%E3%83%90%E3%83%AB%E6%99%82%E9%96%93)             |
| ノードが例外を投げたときの動きを知る               | `errors.py`           | [例外と終了](../user-guide/errors.md)                                                                                                             |
| PC-ALM でモデルを学習させ、BP と比べる             | `mnist/`              | [PC-ALM で学習する](../user-guide/pc.md)                                                                                                          |

### 各プログラムが示すこと

- `quickstart.py`: 2 ノードのネットワークを `with mujou.run(...)` で走らせて `write` で入力を書き、期待した出力が届くまで `run.wait_for(..., timeout=...)` で待つ
- `leaky_chain/`: 連続時間で定義したノード（リーク積分器）。厳密解の係数で更新するので、どんな `dt` でも安定する。`run.write` が返す書き込みの時刻から出力の時刻までの経過時間を使い、出力を厳密解と比べる
- `branching.py`: 1 つの `MultiPort` から複数のノードへ送り、もう 1 つの `MultiPort` で合流して受け取る。`send_to` を使うと、`connect` が返した Channel の相手にだけ送れる
- `group.py`: `Group(enc=..., dec=...)`。ノードのパス（`"enc.scale"`）の付き方
- `save_and_restore.py`: `run.state_dict()` で状態を取得し、`torch.save` で保存し、`mujou.run(..., state=...)` で復元して続ける
- `pause.py`: `mujou.time.global_timer.pause()` でグローバルタイマーを止め、そのタイマーで走る 2 つの実行の状態を同じ時刻で取得してから `mujou.time.global_timer.resume()` で再開する。一方の状態から復元した実行は、その時刻から専用のタイマーで続く
- `subscribe.py`: `run.subscribe` でまずその時点の最新の出力を受け取り、その後はより新しい出力が送られるたびに最新の 1 つを受け取る。受け取る前に次の値が送られると途中の値は飛ばされる。20 個受け取ったら `break` でループを抜ける
- `online_learning.py`: 教師のノードが、ホストの書く `a` を使って y = a x + b のバッチを 10 ms ごとに送る。ホストは実行の途中で `a` を変える。学習するノードは新しいバッチごとに 1 回学び、学習と推論の段階を分けない。ホストは学習するノードの損失を `run.wait_for` で追う。損失は下がり、目標が変わると跳ね上がり、学習するノードが追いつくとまた下がる
- `torch_node.py`: ノードの中の `nn.Module` を、ノードの内側に閉じた BP で学習させる。シードの指定と、`dt` を刻みに分けて進める更新
- `testing_a_node.py`: `mujou.testing` の仮想タイマーと手動の更新で、ノードを決まった `dt` と順序で動かし、結果を参照の計算とビット単位で比べる。途中で取得した状態から復元した実行が、止めずに続けた実行と一致することも確かめる
- `timers.py`: 時刻 `Timestamp`、実行ごとのタイマー `Timer`、時間の速さ（実行のタイマーとグローバルタイマー）
- `errors.py`: ホストが受け取る `NetworkError` とその `__cause__`、`on_exception` → `on_stop` の順、`on_start` の例外は `mujou.run` が送出する
- `mnist/`: 784-256-256-10 の ReLU ネットワークを `mujou.pc` の 4 つのノードに分け、MNIST をストリームで流して PC-ALM で学習させる。同じ初期重みとバッチで BP でも学習させ、prequential 精度とテスト精度、ノードの `dt` を比べる

`leaky_chain/` と `mnist/` はディレクトリになっている。入口はどちらも `main.py`。`leaky_chain/` はノードの定義を `leaky.py` に、`mnist/` はデータの読み込みを `data.py` に、BP での学習を `bp.py` に分けてある。

## ▶️ 走らせ方

ソースからビルドした環境（[インストール](../getting-started/installation.md)）で、リポジトリのルートから実行する。

```sh
uv run python examples/quickstart.py
uv run python examples/leaky_chain/main.py
```

どのプログラムも GPU を使わない。`mnist/` 以外はネットワーク接続も使わず、数秒以内に終わる。`mnist/` は初回に MNIST（約 11 MB）をダウンロードし、既定の 3 epoch で約 2 分かかる。
