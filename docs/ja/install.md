# 📦 インストール

## 必要なもの

| もの    | 条件                                                                          |
| ------- | ----------------------------------------------------------------------------- |
| Python  | 3.14 以上。free-threaded 版の 3.14t を推奨。GIL ありでも良い。                |
| PyTorch | mujou の依存として自動でインストールされる                                    |
| Rust    | 1.85 以上を [rustup](https://rustup.rs/) でインストール。ソースからのビルド用 |
| uv      | [uv](https://docs.astral.sh/uv/)。ソースからのビルド用                        |

ノードは複数のワーカースレッドで同時に動くので、free-threaded 版を勧める。GIL ありの版では Python で書いたノード同士が GIL を取り合い、並列に動きにくい。

!!! warning "API はまだ変わる"

    mujou は 0.x の段階にある。マイナーリリース（0.x.0）では API が壊れることがあり、パッチリリース（0.x.y）はバグの修正だけ。`mujou~=0.1.0` のようにマイナーバージョンを固定し、上げる前に[変更履歴](changelog.md)を読む。

## PyPI からインストールする

uv のプロジェクトでは:

```sh
uv add mujou
```

pip なら、Python 3.14 以上の仮想環境で:

```sh
pip install mujou
```

wheel は Linux（x86_64、aarch64）、macOS（arm64）、Windows（x86_64）向けに、それぞれ 3.14 と 3.14t の分を公開している。ほかのプラットフォームでは pip が sdist からビルドするので、Rust が要る。

PyTorch はほかの依存と同じく PyPI からインストールされる。CPU 版や別の CUDA 版を使いたい場合は [PyTorch のインストール案内](https://pytorch.org/get-started/locally/)を参照。

## ソースからビルドする

サンプルプログラムを動かすときや mujou 自体を開発するときは、リポジトリを clone してビルドする:

```sh
git clone https://github.com/Geson-anko/mujou.git
cd mujou
uv sync
```

`uv sync` は free-threaded 版の Python 3.14t を用意して `.venv/` を作り、依存（サンプルプログラムが使う numpy なども）をインストールし、Rust で書かれたコアをビルドする。GIL ありの通常版を使うなら、代わりに `uv sync --python 3.14` を実行する。

## 動作を確かめる

clone したリポジトリで:

```sh
uv run python -c "import mujou; print(mujou.__version__)"  # 0.1.0 のようなバージョンを表示する
uv run python examples/quickstart.py  # y = tensor([3., 5., 7.]) at t = 0.017 s のような 1 行を表示する
```

`pip install mujou` でインストールした場合は、1 行目を `uv run python` ではなく `python` で実行し、[`examples/quickstart.py`](https://github.com/Geson-anko/mujou/blob/main/examples/quickstart.py) は手元にコピーして実行する。

2 つ目のプログラムとその出力は[チュートリアルの最初のページ](tutorial/1-run.md)で説明する。
