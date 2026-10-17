# run-handle 再レビュー（レビュー対応 5f2282d..4f7eb58）

対象: `feature/2026-10-16/run-handle`、`git diff 4c73e0d..HEAD`（HEAD = 4f7eb58）。前回の指摘は `memory/agents/code-reviewer/run-handle.md` と `memory/agents/algorithm-reviewer/run-handle.md`。

検証は HEAD を detached worktree（scratchpad の `rh-head`）に取り出して行った。レビューの途中で、この worktree に別のエージェントの未コミットの作業（`Clock` → `ReadOnlyTimer`、native の `TimerReader`）が入ってきたためである。そのため、ここでの指摘はすべて 4f7eb58 についてのものである。

## verdict: request-changes

前回の must-fix（M1・M2・algorithm の must-fix）と should-fix は、どれも直っていることを確かめた。

- M1: 断られた `mujou.run(a, b)` を呼び直せる
- M2: 30 ノード・200 回の並行の run で、両方が走ることは 0 回だった。free-threaded と gil=1 の両方、単独と `(a, b)` の両方で確かめた
- algorithm の must-fix: hook の中からタイマーを動かす操作は `RuntimeError` になる

ただし、M2 と algorithm should-fix 2 を直すために入れた `_BUILT`（`weakref.WeakSet`）が、利用者の `__eq__` / `__hash__` に頼っている。このため、正しいプログラムが build できなくなる場合がある（must-fix 1 件）。

## must-fix

### M1. `_BUILT` が WeakSet なので、hash できないコンポーネントは build できず、等しいと判定される別のコンポーネントは誤って「構築済み」と判定される

- 対象: `src/mujou/_core/network.py:243`（`if component in _BUILT`）、`:302`（`_BUILT.update(members)`）、`:318`（`_BUILT.difference_update`）、`:324`（`_BUILT` の定義）
- 問題: WeakSet は要素を `hash()` と `==` で探す。そのため、利用者のクラスが `__eq__` / `__hash__` を定義していると、判定がそれに左右される
  - `@dataclasses.dataclass` を付けた Network（`eq=True` の既定で `__hash__ = None` になる）は、`build` が `TypeError: cannot use 'weakref.ReferenceType' as a set element (unhashable type: 'DNet')` で落ちる
  - `__eq__` / `__hash__` で値を比べる Node は、等しい別のインスタンスを含む木の `build` が `RuntimeError: EqNode at 'node' belongs to a network that is already built` になる
  - この差分の前は、木を歩くときの識別がすべて `id()`（`seen`）と Port の同一性だけだったので、どちらも build できた。つまり後退である
- 根拠: `specs/network.md` の「構築済みの印」は同じオブジェクトに付く印として書かれていて、値の等しさは条件に入っていない。`Network` の docstring も、属性として持ったコンポーネントを歩くとしか書いていない
- 再現（scratchpad の `r2/p_hash.py`）:
  ```python
  @dataclasses.dataclass
  class DNet(Network):
      node: Node
  DNet(Lone()).build()  # TypeError: ... unhashable type: 'DNet'

  class EqNode(Node):  # __eq__/__hash__ が self.k で比べる
      ...
  Solo(EqNode(1)).build()
  Solo(EqNode(1)).build()  # RuntimeError: ... already built
  ```
- 補足: 利用者の `__hash__` / `__eq__` は `_BUILD_LOCK` を持ったまま呼ばれる。印を id で引けるもの（ノードやネットワークの private 属性、または `id()` をキーにして `weakref.finalize` で消すもの）にすれば、この点も無くなる
- 確信度: 高（再現済み）／深刻度: 中（dataclass の Network はありうる書き方である。等しさで比べる Node はまれ）

## should-fix

### S1. 終了の後の `state_dict` の最中に、ほかのホストのスレッドが `read` / `check` / `shutdown` / `join` を呼ぶと、ノードの `self.timer` が `RuntimeError` になる

- 対象: `src/mujou/_core/session.py:315-325`（`_settle` が、終わっていれば毎回、全ノードに `set_timer(node, None)` する）、`:459-465`（`_ended_tree` が止まった `Clock` を付けてから `capture_state` する）
- 問題: `_settle` は `read()` / `subscribe` → `check()`、`shutdown`、`join` から呼ばれる。終わった後は、呼ばれるたびに無条件でノードの timer を外す。そのため、`_ended_tree` が付けた止まった `Clock` を、別のスレッドの `_settle` が `state_dict` の途中で外してしまう。同じ run の `state_dict()` を 2 つのスレッドから呼んだ場合も、片方の `finally` がもう片方の取得の途中で外す。前回の S5 で直した「終了後の `state_dict` でも `timer` が読める」が、並行に呼ばれたときに崩れる
- 根拠: `Node.timer` の docstring と `specs/nodes.md:77` は、正常に終了した後の `state_dict` の中で読めると約束している
- 再現（`r2/p_settle.py`）: `state_dict` で 50 ms 待ってから `self.timer.now()` を返すノードを、`shutdown` の後に `run.state_dict()` する。同時に、別のスレッドで `run.read(port)` をループで呼ぶ。10 回中 10 回、`RuntimeError: the node is not in a run; ...` になった
- 確信度: 高（再現済み）／深刻度: 低〜中（観測用のスレッドで `read` しながら終了後に保存する書き方で起きる）

### S2. 公開名 `Clock` は、既存の「clock」（時計の源）と意味がぶつかる。`Node.timer` が `Clock` を返すのも名前がずれる

- 対象: `src/mujou/_core/timer.py:113`（`class Clock`）、`src/mujou/time.py:17`、`src/mujou/_core/node.py:159`
- 問題: このコードベースでは「clock」を、タイマーの時刻の源（実時計 / 仮想時計）の意味ですでに使っている。`use_virtual_clock`、`VirtualTimer` の「on the virtual clock」、`mujou.testing.run` の docstring の「the clock runs by itself」などである。`mujou.time.Clock` が「読み取り専用のタイマー」を指すと、`Timer` と `Clock` のどちらが時計の源で、どちらが view なのかが名前から分からない。さらに、属性名は `timer` で、型名は `Clock` である。docstring を読めば分かるが、名前だけでは読み取り専用であることが伝わらない
- 根拠: 依頼の観点（`Clock` という公開名が利用者に分かるか）への答え。worktree で進んでいる未コミットの改名（`ReadOnlyTimer`）は、この問題を解消する方向である。その改名をコミットするなら、この指摘は解消する
- 確信度: 中（命名の判断）／深刻度: 低

## nit

- `node.timer._timer` で native の `Timer`（`pause` / `resume` / `load_state_dict` / `advance` / `set_time_scale`）に届き、これらは `host_only` を通らない（`timer.py:140-145`）。private を経由しなければ届かないので nit にとどめる。`test_is_a_view_that_only_reads` は public の名前だけを見ている。worktree で進んでいる `TimerReader` は、この穴を塞ぐ方向である。確信度: 高
- `Clock` の docstring（`timer.py:114-120`）の「After the run has ended normally ... it holds the time at which the run stopped」は、実際の動作と食い違う。終了後の `state_dict` の間だけ、ノードの `timer` が別の止まった `Clock` に差し替わる。ノードが実行の途中で取って保持した `Clock` は、終了後も生きたタイマーを読み続ける（`r2/p_misc.py` で確かめた）。確信度: 高
- 単独で断られた `mujou.run(a)` の後に `mujou.run(a, b)` を呼ぶと、`RuntimeError: Solo at '0' belongs to a network that is already built` になる。単独の build は取り消さないためで、意図どおりである（単独の `mujou.run(a)` ならやり直せる）。ただし docstring の「Nothing started: the networks may run once the cause is gone」と `specs/network.md` の「原因を取り除けば同じネットワークで走らせ直せる」は、別の形で走らせ直せるようにも読める。「同じ形で」と一言あるとよい。確信度: 中
- `Timestamp.__reduce__` は `copyreg.__newobj_ex__` を返すので、protocol 2/3 では `functools.partial(Timestamp.__new__, ...)` として pickle される。`torch.save` の既定は protocol 2 である。そのため、ノードの状態に `Timestamp` を入れた木は、`torch.load` の既定（`weights_only=True`）で読めない。`add_safe_globals([Timestamp])` を足しても読めない。spec の例は `pickle.dump` なので、仕様違反ではない。ただ、`specs/network.md:261` が「状態に入れてよい」と書いているので、torch の保存の流れで引っかかる。確信度: 中
- `specs/network.md:251` は KeyboardInterrupt について「起動がバックグラウンドで続くので走ったものとして数え」と書いている。しかし、`runtime.start` より前（`_restore` や `inputs` の書き込みの途中）で割り込まれた場合は、何も起動していないのに走ったものとして数える。保守的な扱いなので害は無い。直すなら文書だけでよい。確信度: 中
- `TestOnlyTheHostControlsTheTime` は、退行したときに失敗せず止まって戻らない形になっている。`test_a_node_cannot_scale_the_time` の `while not node.errors:` には上限が無い。`test_a_node_cannot_wait_on_the_time` は、退行すると `step()` が戻らない。また、退行して `mujou.time.set_time_scale(1000.0)` が通ると、グローバルのスケールが後のテストまで残る（最後の assert で失敗はするが、戻さない）。確信度: 中
- `Clock.__new__(Clock)` で初期化されていない `Clock` が作れ、その `now()` は `AttributeError` になる。`__init__` だけで作成を塞いでいるためである。確信度: 高（害は小さい）

## 前回の指摘の確認

- M1（断られた `run(a, b)` をやり直す）: 直った。`unbuild` は Port を解凍し、印を外し、`_graph` を戻す。Port には古い `Binding` が残るが、次の run が結び直すので害は無い。`test_a_refused_run_of_several_networks_can_be_retried` で確かめている
- M2（並行の run）: 直った。build は `_BUILD_LOCK` の下で 1 回だけ行う。lock の順序は「build の lock → 解放 → Graph の lock → 解放 → (unbuild の) build の lock」で、入れ子にならないのでデッドロックは無い。hook からの `mujou.run` は build の前に `in_hook` で止まる。ノードの `on_start` が別のネットワークを build しても、ホストは build の lock を持っていないので詰まらない
- algorithm の must-fix: 直った。`Timer.sleep` / `set_time_scale`、`VirtualTimer.advance`、`mujou.time.sleep` / `set_time_scale` は、hook の中（ノードのスレッド、`_capture_state` の中）で `RuntimeError` になる。ノードには `Clock` を渡している。上の nit の private 経路は残る
- `mujou.time.sleep` を関数にした影響: ホストのスレッドからは前と同じように呼べる。Ctrl-C は 0.32 s で `KeyboardInterrupt` になった（native の sleep のシグナル確認がそのまま効く）。`host_only` は呼ぶたびに import を引くが、`sys.modules` を引くだけなので軽い
- `Timestamp.__reduce__`: pickle の protocol 0〜5、`copy.copy`、`copy.deepcopy` で、`Timestamp(ns=2**64-1)` が元に戻ることを確かめた
- S1〜S7・nit 群・algorithm の should-fix 1 / 2: どれも直ったか、docstring や spec に書かれた（`TypeError` の検査、`ManualRun.step` の `RuntimeError`、Raises、予約名、`Compose` の名前の検査、`Graph` の docstring、`Run.read` の時間軸、`mujou.testing.run` の docstring、`Compose` のオラクルのパラメタライズ、`_BUILT`）。S8 は対象外（node-state 側で対応）

## 検証結果

- just run 相当（4f7eb58 の detached worktree で、各ステップを個別に実行）: pass
  - pre-commit -a: pass（変更なし）
  - cargo test --workspace: 28 + 14 + 37 passed
  - pytest --cov: 422 passed, 14 skipped
  - pyright: 0 errors
  - mkdocs build --strict: pass
- 繰り返し: `tests/mujou` を free-threaded（`sys._is_gil_enabled()` が False）で 5 回、`-X gil=1` で 5 回回し、どれも 401 passed, 13 skipped
- probe（scratchpad の `r2/p_*.py`）:
  - `p_m2.py`（M2 の再現条件）: 単独・`(a, b)` とも、両方が走ったのは 200 回中 0 回。free-threaded と gil=1 の両方で確かめた
  - `p_hash.py`: must-fix M1 を再現
  - `p_settle.py`: should-fix S1 を再現
  - `p_pickle.py`: pickle の全 protocol・copy・`torch.load`
  - `p_misc.py`: `Clock` と、単独で断られた後の composed
  - `p_sigint.py`: Ctrl-C
- 成果物の汚染（`</content>` など）: 無し
- 注意: レビューの途中、この worktree で `just run` を 1 回実行した。そのとき pre-commit の ruff format が、別のエージェントの未コミットのファイルを 1 つ整形した（どのファイルかは特定できない。整形だけ）。main には知らせた
