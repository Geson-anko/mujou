# docs/en のホーム・Getting Started・Examples 理解度テスト

## 対象

- ファイル（`docs/en/` 配下）: `index.md`、`getting-started/index.md`・`installation.md`・`quickstart.md`、`examples/index.md`。生徒は加えて `examples/quickstart.py` を読む
- 読者と用途: Mujou を初めて触る利用者（Python と PyTorch は使える）。これらのページだけで Mujou が何かを理解し、インストールし、最初のネットワークを走らせて出力を読み、サンプルの一覧から目的のサンプルを選ぶ
- 方針: 日本語版（`docs/ja/` の同名ページ、`start-ja.md` のループ済み）の翻訳。節・表・リスト・mermaid・admonition・`--8<--` の構造と事実をそろえる。散文は "Mujou"、コードは `mujou`、port は小文字（`docs/en/concepts/` に合わせる）

## 確かめた事実（R1）

- `uv run python examples/quickstart.py` → `y = tensor([3., 5., 7.]) at t = 0.103 s`
- `rust-version = "1.85"`、Linux の CPU 版 torch は `[tool.uv.sources]`、`__version__` は `_native` から
- `period` は `src/mujou/_core/node.py`、`recv_with_time` は `_core/port.py`
- examples の docstring と一覧の説明が合う（errors / timers / compose / subscribe / testing_a_node）
- subscribe は「途中で送られたものは飛ばす」（最新→より新しいもの）。日本語版の表「出力が送られるたびに受け取る」は少し強い。英語版は "Receive outputs as they are sent" にした
- 見出しへのアンカーリンクは使っていない。`uv run mkdocs build --strict` 通過、`uv run pre-commit run --files <5 ファイル>` 通過（mdformat 適用済み）

## 評価用問題

### Q1 You want to install Mujou on your machine right now. Do you install it from PyPI or build it from source? Give the reason.

- 要点: build from source
- 要点: Mujou is not on PyPI yet

### Q2 Before building from source, which two tools do you need to install besides Python?

- 要点: the Rust toolchain (via rustup, 1.85 or later)
- 要点: uv

### Q3 You only have the regular Python 3.14 build with the GIL. Can you use Mujou? If you build from source, what command creates the environment?

- 要点: yes (3.14t is recommended, but the regular build works)
- 要点: `uv sync --python 3.14`

### Q4 You built Mujou from source on Linux and want to use a GPU inside a node. What do you need to do, and why?

- 要点: replace PyTorch with a CUDA build
- 要点: on Linux, `uv sync` installs the CPU-only PyTorch

### Q5 In the quickstart program, you delete the `mujou.time.sleep(0.1)` line and call `read` right after `write`. Is `y` always `[3., 5., 7.]`? Why?

- 要点: not always
- 要点: the value takes time to pass through scale and shift; until it arrives, shift sends a value computed from the previous input (zeros at first)

### Q6 In the host program, you want the network to see an input for 0.5 seconds. May you wait with the standard library's `time.sleep(0.5)`? What should you use, and why?

- 要点: no; use `mujou.time.sleep(0.5)`
- 要点: it waits on Mujou's global time, so it stays correct when the speed of time changes or the run is paused

### Q7 In the quickstart output `y = tensor([3., 5., 7.]) at t = 0.106 s`, what does `t` mean? From what moment is it measured?

- 要点: the global time (seconds) at which shift (the output port) sent `y`
- 要点: 0 is when the process first uses the global time (roughly `import mujou`)

### Q8 In your own node, you want to write a `while` loop inside `on_update` that waits until a new value arrives from the previous node. May you write it that way? Why?

- 要点: no (do not write a loop)
- 要点: Mujou runs the loop; `on_update` holds one step; `recv` returns the latest value without waiting

### Q9 You ran nodes written like those in the quickstart, and one CPU core was fully used. What is the cause, and how do you make the nodes update at a regular interval?

- 要点: the nodes have no update period (they repeat `on_update` without waiting)
- 要点: call `super().__init__(period=seconds)` in `__init__`

### Q10 While the network is running, one node raises an exception. What happens when the host then calls `run.read(...)`? Where do you find the exception the node raised?

- 要点: `mujou.NetworkError` is raised (the whole network has stopped)
- 要点: its `__cause__` is the original exception

### Q11 You want to write a program that saves the learned state and later continues from it. Which example do you look at? In that example, which method takes the state, and which argument restores it?

- 要点: `save_and_restore.py`
- 要点: `run.state_dict()` to take it, `mujou.run(..., state=...)` to restore

### Q12 You want to test your own node with a fixed `dt` so that it gives the same result every time. Which example do you look at? What from which module do you use?

- 要点: `testing_a_node.py`
- 要点: the virtual timer (and manual updates) of `mujou.testing`

### Q13 In a Mujou network, how do you switch to inference mode after training is done?

- 要点: there is no mode to switch
- 要点: the network keeps taking input and producing output while it runs, and keeps learning at the same time

## 保留問題

### H1 When node A receives values from node B, does A wait for B to finish computing? If B has never sent anything, what does A receive?

- 要点: no; A reads the last value B sent
- 要点: a tensor of zeros if nothing has arrived

### H2 From the root of the repository, what command runs the `leaky_chain` example?

- 要点: `uv run python examples/leaky_chain/main.py` (a directory whose entry point is `main.py`)

### H3 You installed Mujou with pip. What do you run to check that it is installed, and what output tells you it is?

- 要点: activate the virtual environment and run `python -c "import mujou; print(mujou.__version__)"`
- 要点: it prints a version such as `0.0.1`

（H3 は日本語版の汎化確認で「問題不良」とした点を直し、問題文で表示の読み方も問う）

## ラウンド記録

### R0（初稿 271 行: index 49、getting-started/index 12、installation 59、quickstart 105、examples/index 46）

日本語版（271 行）と同じ構造・行数。

### R1（271 → 271 行）

| 問      | 判定 | 原因 | 直したこと                                                                                               |
| ------- | ---- | ---- | -------------------------------------------------------------------------------------------------------- |
| Q1–Q13  | 正解 | -    | （指摘を受け）quickstart「Update period」の `dt` を「since the previous call」まで書き、表の定義と同じ言い方にそろえた |

正解 13/13。読みにくい箇所の指摘のうち、次は直さなかった。

- GPU の入れ替えコマンド: 日本語版も外部の案内に任せている。uv 環境での正しい手順を確かめていないので、書き足さない
- `t` の 0 点が「おおよそ」: 実装でも厳密な時点は決まっておらず、日本語版のループで目安と明記済み
- 「falls behind」・global time・virtual timer: 概念とユーザーガイドのページが説明する。生徒はそれらを読めない制約のため

次は保留問題 H1〜H3 を出す。

### 汎化確認（保留問題、271 → 271 行）

| 問  | 判定 | 備考 |
| --- | ---- | ---- |
| H1  | 正解 | -    |
| H2  | 正解 | -    |
| H3  | 正解 | 「公開前なのに pip で入れた」という前提の食い違いを生徒が指摘。問題文の前提は「公開後」を想定している。再利用時は「After Mujou is published on PyPI, you install it with pip.」と前置きする |

保留問題の後の最小改稿（orchestrator の指示）:

1. installation.md: pip で入れた場合の確かめ方で、「`uv run` を除いて実行する」と読者に推測させていた。`python -c "import mujou; print(mujou.__version__)"` をそのまま書き、表示の読み方も 1 文で添えた（埋没・曖昧。日本語版も同じ書き方）
2. quickstart.md: グローバル時間の 0 は「プロセスが初めて使った時点」が定義で、`import mujou` は目安だと分かる言い方にした（曖昧。日本語版は「おおよそ `import mujou` のころ」で同じ問題が残りうる）

直さなかった指摘: previous/next side の向き（直後の「Connect the nodes」で分かる）、`dt` に比例する更新・mermaid の "latest value"・speed of time・run's timer（概念のページの担当）、`dt` の打ち切り（examples 側を直す予定と orchestrator から連絡あり）。
