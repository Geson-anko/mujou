# 📦 インストール

!!! note "PyPI では未公開"

    mujou はまだ PyPI に公開していないので、現状はソースからビルドする。

## 必要なもの

| もの    | 条件                                                                          |
| ------- | ----------------------------------------------------------------------------- |
| Python  | 3.14 以上。free-threaded 版の 3.14t を推奨。GIL ありでも良い。                |
| PyTorch | mujou の依存として自動でインストールされる                                    |
| Rust    | 1.85 以上を [rustup](https://rustup.rs/) でインストール。ソースからのビルド用 |
| uv      | [uv](https://docs.astral.sh/uv/)。ソースからのビルド用                        |

ノードは複数のワーカースレッドで同時に動くので、free-threaded 版を勧める。GIL ありの版では Python で書いたノード同士が GIL を取り合い、並列に動きにくい。

## ソースからビルドする

```sh
git clone https://github.com/Geson-anko/mujou.git
cd mujou
uv sync
```

`uv sync` は free-threaded 版の Python 3.14t を用意して `.venv/` を作り、依存をインストールし、Rust で書かれたコアをビルドする。GIL ありの通常版を使うなら、代わりに `uv sync --python 3.14` を実行する。

PyTorch はほかの依存と同じく PyPI からインストールされる。CPU 版や別の CUDA 版を使いたい場合は [PyTorch のインストール案内](https://pytorch.org/get-started/locally/)を参照。

!!! note "PyPI に公開した後"

    公開後は、uv のプロジェクトなら `uv add mujou`、pip なら Python 3.14 以上の仮想環境で `pip install mujou` でインストールできるようになる。

## 動作を確かめる

```sh
uv run python -c "import mujou; print(mujou.__version__)"  # 0.0.1 のようなバージョンを表示する
uv run python examples/quickstart.py  # y = tensor([3., 5., 7.]) at t = 0.017 s のような 1 行を表示する
```

2 つ目のプログラムとその出力については[クイックスタート](quickstart.md)で説明する。
