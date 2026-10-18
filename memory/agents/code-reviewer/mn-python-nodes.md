# M:N PR 2（Python のノードをワーカーに載せ、既定を切り替える）レビュー

対象: `git diff d1ca90f..HEAD`（feature/2026-10-18/mn-python-nodes、82183ce まで）。計画書 `memory/agents/implementation-planner/mn-scheduler.md` の PR 2 と「ユーザーの決定（2026-10-18）」に照らした。

## verdict: request-changes

must-fix は 1 件だけ。エラー経路に限られ、直し方も行の並べ替えで済む。裁定で should-fix に下げるなら approve でもよい。

## must-fix

### M1. `_on_enter` が途中で失敗すると、前のノードの束縛と勾配モードが残ったまま `on_exception` / `on_stop` が走る

- 対象: `src/mujou/_core/node.py:387-400`（`enter_node(self)` が最後の行にある）、`crates/mujou-core/src/runtime/mod.rs:280-291`（`Entry::BestEffort` は `enter` が失敗しても hook を呼ぶ）
- 問題: `torch.cuda.set_device` か `torch.cuda.Stream(...)`、または `_make_generator` が例外を投げると、`enter_node(self)` に届かない。コアは `on_exception` と `on_stop` を `BestEffort` で呼ぶので、そのワーカーで直前に走ったノード A の束縛が残ったまま、失敗したノード B の hook が走る。その結果、
  - B の `on_stop` / `on_exception` から A の Port を `recv` / `send` できてしまう（`current_node() is A`）。B は自分の Port を使えない
  - A が `with` なしで有効にした勾配の記録が、B の hook でも有効のまま残る
  - そのスレッドが一度もノードに入っていなければ `inside` が False のままなので、B の hook からホスト専用の操作（`mujou.run` など）が断られない（専用スレッドでも同じ。`_on_thread_start` の頃からある形だが、ワーカーで「別のノードの束縛が残る」のは今回からの問題）
- 根拠: `specs/network.md`「ほかのノードの hook から呼ぶと `RuntimeError`」、`specs/native-core.md:59`（`on_exception` / `on_stop` は `enter` が失敗しても呼ぶ）。再現スクリプト `/private/tmp/claude-501/-Users-geson-Projects-mujou/cbc74a71-cf1a-4b70-95f1-2db7e241a0ac/scratchpad/repro/stale_owner.py`。CUDA の失敗の代わりに `_on_enter` の先頭で例外を投げ、workers=1 で動かすと、次の出力になった。
  ```
  run ended: NetworkError the network stopped because node 'b' raised in on_update
  B.on_stop read A's port -> tensor([0.])
  grad enabled in B.on_stop -> True
  ```
- 直し方の方向: `enter_node(self)` と `torch.set_grad_enabled(False)` を `_on_enter` の先頭に置き、CUDA と generator の設定はその後に回す。テストは無いので、`_on_enter` が失敗した後の `on_stop` について、自分の Port だけを使えて勾配が切れていることを確かめるテストを足す
- 確信度: 高（再現済み。ただし通る経路はエラー時だけ）。深刻度: 中

## should-fix

### S1. docs は勾配・device・stream を「hook のたびに」戻すと書いているが、実装は切り替わったときだけ戻す

- 対象: `docs/en/user-guide/torch.md:49`（"before the hooks of every node"）、`:93`（"back to the node's before each of its hooks"）、`docs/en/user-guide/writing-a-node.md:230`（"set back by Mujou before each hook"）、`docs/ja/user-guide/torch.md:49`・`:93`（「hook の前に毎回」）、`docs/ja/user-guide/writing-a-node.md:230`（同じ）、`examples/torch_node.py:3`（"before every hook"）、`.claude/skills/locality-and-async/SKILL.md:47`・`memory/project_mn_scheduler.md:14`（「hook の前に戻す」）
- 問題: `_on_enter` が走るのは、スレッドが別のノードに切り替わったときだけ（`mod.rs:280` の `LAST_NODE`）。あるノードが `on_update` の中で `with` を使わずに `torch.set_grad_enabled(True)` を呼んだとする。同じワーカーで続けて走れば次の `on_update` でも勾配は有効のまま、間に別のノードが挟まれば無効に戻る。つまり振る舞いがスケジュール次第になる。docs を信じた利用者は、毎回戻ると思い込む
- 根拠: `Node` の docstring（`node.py` の "Before every hook that a thread runs after another node's (or as its first)"）と `specs/torch.md`「切り替えごとの設定」は正しく書いている。利用者向けの docs と examples だけがこれと食い違う
- 確信度: 高。深刻度: 低〜中（利用者の誤解を招く）

### S2. 「ノードのスレッド」「each on its own thread」が docstring に残っている

- 対象: `src/mujou/_core/run.py:26`（`Run` の docstring "The nodes run in the background, each on its own thread"。API リファレンスに出る）、`src/mujou/_core/port.py:265-266`・`:319`（`recv` / `send` の Raises の "the caller is not the owning node's thread"）、`crates/mujou-core/src/slot.rs:503`（"accepts sends only from its node's thread"）
- 問題: 既定はワーカーになり、Port の所有はスレッドでなくノードで判定するようになった。上の記述はどちらとも合わない
- 根拠: `specs/network.md` の改訂（「判定はスレッド ID でなく、呼んだスレッドが今走らせているノードで行う」）
- 確信度: 高。深刻度: 低

### S3. 「スレッドに設定したものは次の hook に持ち越されない」は言い過ぎで、肝心の「ほかのノードに漏れる」が抜けている

- 対象: `src/mujou/_core/node.py:84-87`、`docs/en/user-guide/writing-a-node.md:227`、`docs/ja/user-guide/writing-a-node.md:227`
- 問題: 実際には、同じワーカーで続けて走れば持ち越される。危ないのはむしろ、同じワーカーで次に走る別のノードに漏れることだ（`torch.set_default_device` のように `with` を使わない torch の設定、`asyncio.set_event_loop` など）。`docs/*/user-guide/torch.md:93` には「その間スレッドはほかのノードを動かす」とあるが、`writing-a-node` と docstring は「持ち越されない」と言い切っている
- 根拠: 実装はスレッドごとの状態のうち、勾配と CUDA だけを戻す（`node.py:391-396`）
- 確信度: 中。深刻度: 低

## nit

- N1. `run.py:442`、`docs/*/user-guide/running.md:38` は既定を「論理コア数」と書いている。実際の値は `std::thread::available_parallelism()` で、affinity や cgroup の割り当てで論理コア数より小さくなりうる（`specs/native-core.md:49` は正しい）。確信度: 中
- N2. `tests/mujou/_core/test_run.py:714-716`・`:766-776` は、コアの既定（`available_parallelism`）を `os.cpu_count()` で近似している。`PYTHON_CPU_COUNT` や `-X cpu_count` で `os.cpu_count()` を小さくすると `n <= 既定のワーカー数` になり、`len(idents) <= os.cpu_count()` が落ちうる。普通の CI では起きない。確信度: 中
- N3. `tests/mujou/_core/test_node.py` の `StreamRecorder` の docstring は "switches both away" だが、コードが替えるのはストリームだけ（自分のデバイスの既定のストリームに戻す）。デバイスは替えない。確信度: 高
- N4. CPU のノードの `_on_enter` は CUDA の current device と stream に触れない。そのため CUDA のノードの直後に同じワーカーで走る CPU のノードが CUDA を使うと、前のノードのストリームに積まれる。CPU のノードが CUDA を使うのは契約の外なので、docs に一言あれば足りる。確信度: 低
- N5. `mujou.run(workers=2**70)` は Python 側の検査を通り、`Runtime.start` の引数の変換で `OverflowError` になる（`ValueError` ではない）。`Session.start` の `except Exception` で取り消されるので、再実行はできる。確信度: 高、深刻度: ごく低
- N6. ja の文体: `docs/ja/user-guide/running.md:42` は本文の文を「torch の演算の中だけ。」と体言止めで終えている（memory/feedback_japanese_style.md は体言止めを短い項目に限っている）。`docs/ja/user-guide/torch.md` の tip「ワーカーの数（と `dedicated_thread=True` のノードの数）の和と…の積」は括弧の中身に「の和」が掛かっていて読みにくい。`docs/ja/user-guide/writing-a-node.md:227`「ノードは既定で、ほかの…」の読点は要らない。確信度: 中
- N7. 利用者向けの docs（`writing-a-node.md:231` の en/ja）は、ノードの `contextvars.Context` が空から始まる（ホストの文脈を引き継がない）ことと、`state_dict` がその文脈の外で呼ばれることを書いていない。docstring（`_native.pyi`）には "(but `_capture_state`)" とある。確信度: 中

## 観点ごとの所見（依頼された点）

- `_on_enter` で設定し直すもの: 勾配・CUDA の device / stream・`current_node` で、計画の表どおり。漏れは M1（失敗時の順序）、S1（毎回ではない）、N4（CPU のノードは CUDA を戻さない）の 3 つ。`with` を使わない torch のスレッドごとの設定（`set_default_device`、autocast の直接設定、forward AD の level など）と `threading.local` は利用者の責任とされ、docs の「`with` に閉じる」がそれを担う。ユーザーの決定 Q3 の範囲内
- Port の所有の判定: 制御スレッドの `state_dict`（`Entry::Skip`。制御スレッドは `enter_node` を呼ばないので None）、ホスト、hook の中で利用者が立てたスレッド（`threading.local` が新しいので None）、終了後（ワーカーと専用スレッドは抜け、スレッドの状態ごと `threading.local` が消える）のどれでも判定は甘くならない。以前のスレッド ID と違って ID の再利用も無い。甘くなるのは M1 の経路だけ
- 計画外の判断 8（束縛を外さない）: 上の理由で妥当。ただし M1 を直すことが前提
- 判断 1（空の `Context()`）: 以前の Rust のスレッドも空の文脈だったので、振る舞いは変わらない。妥当。判断 2（`state_dict` を文脈の外で呼ぶ）: 以前も制御スレッドだったので回帰ではない。docs への記載は N7
- `on_thread_start` の検査: `inspect.getattr_static(type(node), ...)` でクラスを見ていて、継承した override も捕まえる。`Session.start` の `try` の中、`runtime.start` より前に走り、`except Exception` で `lifetime` とタイマーを戻すので、ネットワークを使い切らない（テストで確認）。予約名 `dedicated_thread` の検査も `checked_period` と同じ形になっている
- `workers`: `start_run` の `in_hook` の検査の直後、構築の前に検査している。`bool` と `float` は `TypeError`、1 未満は `ValueError`。`mujou.testing.run` は `workers=None` を渡し、シグネチャにも足していない（`test_testing.py` で固定）。計画どおり
- テスト: 公開インターフェース（`mujou.run`、`mujou.testing.run`）と bindings の `_native.Runtime`（既存の方針）で振る舞いを確かめていて、どれも実装の違いを見分ける（例: `TestOwnerOnAWorker` はスレッド ID の判定なら落ちる）。仮想タイマーと手動ステップで決定的になっている。新しいテストを 8 回続けて回し、すべて通った。spec-test-author の記録と実装の食い違いは見つからなかった。足りないのは M1 の経路のテストだけ
- docs・specs・memory・skill: S1・S2・S3 以外は実装と合っている。`docs/ja/index.md` は「自分のスレッドで」→「自分の周期で」の 1 語だけで、ほかの文には触れていない
- GIL ありのビルド: ワーカーは `on_thread` で生きている間ずっと attach し、待つ間は detach する。hook は `Python::attach` で GIL を取り直すだけ。lock を持ったまま attach する経路は増えていない（`hooks[i]` は、GIL を持たずに待つワーカーと、停止中の制御スレッドしか取らない）。Homebrew の 3.14.8（GIL あり、`sys._is_gil_enabled() == True`）を別の venv に入れて `tests/mujou` と `tests/test_examples.py` を回し、551 passed / 21 skipped だった

## 検証結果

- just run: pass（cargo test・pytest 586 passed / 20 skipped・pyright 0 errors・mkdocs build --strict）。作業ツリーは clean のまま
- GIL ありの 3.14.8: `tests/mujou` + `tests/test_examples.py` で 551 passed / 21 skipped
- 新しいテスト（Workers / FrameworkHooks / Dedicated / OnThreadStart / OwnerOnAWorker / ThreadSetup / TestStart）を 8 回繰り返し: すべて 55 passed
- 成果物の汚染（`</content>` など）: 見つからなかった
