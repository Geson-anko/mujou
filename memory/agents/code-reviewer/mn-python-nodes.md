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

---

# 再レビュー（2026-10-19、`git diff 82183ce..HEAD`、2f3bb91 まで）

## verdict: approve

前回の M1・S1・S2・S3 と nit（N2 を除く）は解消した。N2 を見送った理由（実際のワーカー数を読む公開の口が無い）も妥当。新しく見つけたのは、どれも動作に影響しない should-fix 1 件と nit だけ。

## 前回の指摘の状況

- M1 → 解消（確信度: 高）
  - `_on_enter` の先頭で `enter_node(self)` を呼ぶようになった（`src/mujou/_core/node.py` の `_on_enter`）。勾配は `_before_hook` で切る。`enter` が失敗したときも、BestEffort の `on_exception` / `on_stop` は bindings の `call_as` を通るので、`_before_hook` が先に走る
  - テスト `test_a_node_whose_setup_failed_cleans_up_as_itself` は意図した経路を通る
    - workers=1 で、タスクは添字の順に並ぶ。leaky の `on_start`（勾配を有効にして残す）→ broken の `enter` が失敗（`Required` なので `on_start` は呼ばない）→ leaky の `on_exception` → broken の `enter`（また失敗）→ BestEffort で `on_exception` の順になる
    - 直す前の順序なら、broken の `on_exception` の時点で `current_node()` が leaky なので、leaky の Port を読めてしまい、勾配も有効のままになる。テストはその両方を確かめている
- S1 → 解消（確信度: 高）。勾配を切るのを「切り替えのとき」から「毎回」に変えた。実装・docstring・docs・specs・skill・memory が揃っている
- S2・S3 → 解消。autocast などの漏れは、ユーザーの決定どおり文書だけで扱っている。記述は実装の事実（フレームワークが戻すのは勾配と CUDA の device・stream だけ）に合っている
- nit: N1・N3・N4・N5・N6・N7 → 解消。N2 → 理由つきで見送り（妥当）

## `_before_hook` の呼ばれ方の確認

- 呼ばれる条件（`crates/mujou-python/src/runtime.rs:293-303`）: `call_as` を通る hook のうち、`_on_enter` と `_after_update` を除くすべての前に呼ばれる
  - 対象は `on_thread_start`、`on_start`、`load_state_dict`（実体は `_restore_state`）、`on_update`、`on_pause`、`on_resume`、`on_exception`、`on_stop`
  - `state_dict` は `NodeHooks::state_dict` の別の経路なので呼ばれない
  - 順序は `_on_enter`（切り替えのときだけ）→ `_before_hook` → hook
  - `enter` が失敗したとき: `Required` の hook は呼ばれないので `_before_hook` も呼ばれない。BestEffort（`on_exception` / `on_stop`）では呼ばれる
- エラーの報告名: `call_as` の `reported` に、その hook の public な名前が入る。specs/nodes.md の「`_before_hook` は次に呼ぶはずだった公開の hook」と一致する
- `Context.run` の外で呼ぶこと: 勾配モードは torch の C++ のスレッドローカルで、contextvars ではない。だから文脈の外でも中でも効果は同じで、妥当。ただし文書との食い違いが残る（下の N8）

## must-fix

なし

## should-fix

### S4. `_before_hook` の契約を bindings の層で確かめるテストが無い

- 対象: `tests/mujou/_core/test_native_runtime.py`（`FrameworkHooks` は `_before_hook` を記録しない）
- 問題: `_before_hook` を確かめているのは、`mujou.Node` の勾配を見る 2 本だけ（`test_node.py`）。この 2 本が通るのは `on_start`・`on_update`・`on_exception`・`on_stop` だけだ。次の性質はどのテストも固定していない
  - `on_thread_start`・`load_state_dict`・`on_pause`・`on_resume` の前にも呼ぶこと
  - `state_dict` と `_after_update` の前には呼ばないこと
  - 失敗をその hook の名前で報告すること
  - `_on_enter` → `_before_hook` の順
- 根拠: skill testing-strategy（契約を公開の層で固定する。bindings の内部の hook は、既存の `TestFrameworkHooks` と同じく `_native.Runtime` で確かめる方針）。`_on_enter` には同種のテスト（`test_a_worker_enters_a_node_only_when_it_switches_to_it`、エラーの名前の 2 本）がある
- 確信度: 高（テストが無いこと）。深刻度: 低

## nit

- N8. `_before_hook` は `Context.run` の外で呼ばれる。一方、`specs/native-core.md:67` は「hook はすべてノードごとの `contextvars.Context()` の中で `Context.run` で呼ぶ。`state_dict`（`_capture_state`）は文脈の外で呼ぶ」と書いている。specs/torch.md の「hook はすべてそのノードの `Context` の中で呼ぶ」と、`Node` の doc comment（`runtime.rs` の「each runs in the node's own `contextvars.Context` (but `_capture_state`)」）も同じ。例外として `_before_hook` を挙げるか、文脈の中で呼ぶかのどちらかに揃える。将来 contextvars に依る設定を `_before_hook` に足すと効かない、という注意にもなる。確信度: 高
- N9. `_before_hook` が例外を投げると、`call_as` の `?` で抜けるので `on_exception` / `on_stop` の本体も呼ばれない。`specs/native-core.md:59` の「`on_exception` と `on_stop` は `enter` が失敗しても呼び、資源を解放させる」と同じ扱いにはなっていない。`torch.set_grad_enabled(False)` が失敗することはまず無いので、実害はほぼ無い。確信度: 高（振る舞い）／影響: ごく低
- N10. 利用者向け docs の「before every hook of every node」（`docs/en/user-guide/torch.md:49`）と「どのノードでも hook の前に毎回」（`docs/ja/user-guide/torch.md:49`）は、`state_dict` を除くことを書いていない。docs の hook の一覧には `state_dict` も入っている（docstring と specs は「`state_dict` 以外」と書いている）。確信度: 中
- N11. `docs/en/user-guide/torch.md` は CUDA の同期について「The worker waits meanwhile」と書き、`docs/ja/user-guide/torch.md` も「その間は…ワーカーが待つ」と書いている。しかし、状態を取得するときの同期は制御スレッド（終了後はホストのスレッド）で行うので、ワーカーが待つのは `on_update` の後と復元の後だけ。確信度: 中
- N12. ja の文体: 前回 N6 で直した体言止めが、新しい段落で 2 か所に出ている。`docs/ja/user-guide/torch.md` の「デバイスとストリームを設定するのは、ワーカーが CUDA のノードに切り替わるとき。」と「mujou 自身が戻すのは、…CUDA の現在のデバイス・ストリームだけ。」で、どちらも本文の文。箇条書きの末尾の「…だけ（[ノードの中の PyTorch](torch.md)）」は短い項目なので構わない。確信度: 中
- N13. `test_a_node_whose_setup_failed_cleans_up_as_itself` は、private の `_make_generator` を override し、`_generator` を直接 `None` にして失敗を起こしている。本来の失敗の経路（CUDA の設定）は GPU が無いと起こせず、公開の口からセットアップを失敗させる手段も無いので、やむを得ない。そう選んだ理由をテストのコメントに一言残すとよい。確信度: 中

## 検証結果

- just run: pass（cargo test・pytest 589 passed / 20 skipped・pyright 0 errors・mkdocs build --strict）
- 作業ツリー: just run の後、`memory/agents/algorithm-reviewer/mn-python-nodes.md` に未コミットの追記がある。algorithm-reviewer が同時に書いた再監査で、just run の整形によるものではない
- 成果物の汚染（`</content>` など）: なし

---

# 再レビュー 2（2026-10-19、`git diff b39257e..HEAD`、c89a408 まで。`on_thread_start` の廃止と CUDA の推奨）

## verdict: approve

must-fix と should-fix は無い。決定 (1)(2) は範囲どおりに入っている。指摘は nit だけ。

## 見てほしい点への所見

- `on_thread_start` の消し残し
  - コード（`crates/`、`src/`）、テスト、docs（en/ja）、specs、skill には、現役の記述として残っていない
  - 残っているのは、経緯として書いた次の箇所だけで、どれも打ち消し線か「廃止した」の注記が付いている。正しい残し方
    - `memory/project_*.md`、`memory/MEMORY.md`
    - `specs/native-core.md:65`（廃止の記録）
    - `specs/open-decisions.md:37`（`_on_thread_start` の経緯）
  - 決定どおり、検出も案内もしない。`checked_dedicated` に残したのは予約名の検査だけ
- 消したテストが押さえていた性質の行き先
  - 最初の公開 hook の時点で勾配が切れていること: `TestThreadSetup`（`Probe.on_start`）と `_before_hook` のテスト
  - generator が続けて使えること: 既存の `InitDraws` などのテスト
  - 専用スレッドが全 hook を 1 本のスレッドで走らせること: `TestDedicatedThread`
  - 専用スレッドで `on_start` に設定した `threading.local` が後の hook でも保たれること: `test_the_hooks_of_a_dedicated_node_share_one_python_thread_state`。決定の前提「スレッドごとの設定は `on_start` に書く」はこれで保証される
  - 「専用スレッドの中でもホスト用のメソッドを断る」: 実装者の判断（既存のテストで足りる）は妥当。根拠は 2 本のテストの組み合わせで、次の 2 点が確かめられているから
    - 専用スレッドでも `on_start` の前に `_on_enter` が 1 回だけ呼ばれる（bindings の `test_the_framework_enters_a_dedicated_node_once_before_on_start`）
    - `mujou.Node._on_enter` が `enter_node` を呼ぶ（ワーカーの上の in_hook のテストと M1 のテスト）
  - ただし推論が 2 段になるので、補強の案を N14 に挙げた
- `PyNode` の `dedicated` を消し、コアの `RuntimeConfig::dedicated` だけにした変更: 妥当
  - bindings の `dedicated` は `on_thread_start` を呼び分けるためだけにあった。今はどのノードでも同じ hook の列になる
  - 長さの検査は今までどおりコアが行い、`test_invalid_arguments_start_nothing[dedicated]` が残っている
  - `flags` の複製と `enumerate` も無くなり、簡単になった
- CUDA の推奨の記述: PyTorch の仕様に照らして正確
  - キャッシュアロケータは、確保した時点の現在のストリームにブロックを結び付ける。解放すると、`record_stream` が無ければ、そのストリームの次の確保にすぐ再利用されうる
  - ノードのストリームの仕事が残っているうちにホストのストリームで再利用される危険は、確かに `__init__` で確保したときに生じる
  - `on_start` 以降で作ればノードのストリームに結び付き、ストリームの順序で守られる。この利点も正しく書けている
  - 「`__init__` でも `mujou.run` の同期で正しく動く」は、specs/torch.md の同期の表の最後の行と一致する
  - specs には「実機で確かめていない」の注記がある
  - examples の 3 つ（`torch_node.py`、`online_learning.py`、`save_and_restore.py`）はもともと `on_start` で `nn.Linear(..., device=self.device)` を作っているので、推奨と食い違わない
  - `torch.md#cuda` のアンカーは en/ja とも見出し `## CUDA` で、ビルドも通る

## must-fix

なし

## should-fix

なし

## nit

- N14. 専用スレッドでのホスト用メソッドの拒否を直接確かめるテストが無い（上の 2 段の推論で足りてはいる）。既存の `TestHostMethodsFromANode.test_raise`（`test_run.py`）を `dedicated_thread` の True/False でパラメータ化すれば、1 行で直接押さえられる。確信度: 中
- N15. `src/mujou/_core/node.py:368`（`timer` の docstring）の 1 行が 120 桁ある。「(also in ``on_thread_start`` and ``on_start``)」を消したときに折り返し直していない。ruff format は docstring を折り返さないので、just run は通る。確信度: 高
- N16. ja の文体: `docs/ja/user-guide/torch.md` の tip「…正しく動く。`mujou.run` がノードを起動する前にホストのストリームを同期するからである。」は、memory/feedback_japanese_style.md が避けるとしている「理由を『〜からである』で付け足す」形になっている。「`mujou.run` がノードを起動する前にホストのストリームを同期するので、`__init__` で CUDA に確保しても正しく動く。」のように、理由を前に置けば直る。確信度: 中

## 検証結果

- just run: pass（cargo test・pytest 594 passed / 20 skipped・pyright 0 errors・mkdocs build --strict）。作業ツリーに変更は無い
- `on_thread_start` の grep（`.git`・`target`・`.venv`・`site`・`memory/agents/` を除く）: 現役の記述は 0 件。経緯の記述は上のとおり
