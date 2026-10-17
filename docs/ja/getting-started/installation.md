# インストール

インストールの方法は 2 通りある。いまはソースからビルドする方法だけが使える。PyPI にはまだ公開していない。

## 必要なもの

| もの    | 条件                                                                                |
| ------- | ----------------------------------------------------------------------------------- |
| Python  | 3.14 以上。free-threaded 版の 3.14t を勧める。GIL ありの通常版 3.14 でも動く        |
| PyTorch | mujou の依存として自動で入る。自分で入れる必要はない                                |
| Rust    | ソースからビルドするときだけ必要。1.85 以上を [rustup](https://rustup.rs/) で入れる |

free-threaded 版を勧めるのは、各ノードが自分のスレッドで動くからである。GIL ありの版では、Python で書いたノード同士が GIL を取り合うため、並列に動きにくい。

## ソースからビルドする

Rust のツールチェーンと [uv](https://docs.astral.sh/uv/) を先に入れておく。

1. リポジトリを取得する

    ```sh
    git clone https://github.com/Geson-anko/mujou.git
    cd mujou
    ```

2. 環境を作る。`uv sync` は Python を用意して `.venv/` を作り、依存を入れ、Rust で書かれたコアをビルドする

    | 使う Python            | コマンド                |
    | ---------------------- | ----------------------- |
    | free-threaded 版 3.14t | `uv sync`               |
    | GIL ありの通常版 3.14  | `uv sync --python 3.14` |

Linux では、`uv sync` は PyTorch を CPU 版の配布元から入れる。mujou のコアは GPU を使わないので、数 GB ある CUDA 版を避けている。ノードの中で GPU を使いたいなら、[PyTorch のインストール案内](https://pytorch.org/get-started/locally/)に従って CUDA 版の PyTorch に入れ替える。

## PyPI から入れる（公開後）

!!! warning "まだ PyPI に公開していない"

    この節のコマンドは、PyPI に公開した後に使えるようになる。いまはソースからビルドする。

[uv](https://docs.astral.sh/uv/) のプロジェクトに入れるなら `uv add mujou` を使う。pip を使うなら、Python 3.14 以上の仮想環境で `pip install mujou` を実行する。pip で入れたときは、仮想環境を有効にしてから `python -c "import mujou; print(mujou.__version__)"` を実行する。`0.0.1` のようなバージョンが表示されれば入っている。

## 動作を確かめる

バージョンを表示する。

```sh
uv run python -c "import mujou; print(mujou.__version__)"
```

`0.0.1` のようなバージョンが表示されれば、インストールできている。

ソースからビルドした場合は、クイックスタートのプログラムも走らせられる。

```sh
uv run python examples/quickstart.py
```

`y = tensor([3., 5., 7.]) at t = 0.106 s` のような 1 行が表示されれば動いている。`t` の値は実行ごとに少し変わる。この出力の読み方は[クイックスタート](quickstart.md)で説明する。
