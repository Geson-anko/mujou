# stack #7〜#24 の最終形の整合性監査（コードと仕様）レビュー

対象: branch `chore/2026-10-17/consistency`（12e2558）の `src/mujou/`、`crates/`、`_native.pyi`、`specs/`、`tests/`。origin/main（3ac1e23）からの差分の最終形。

## verdict: request-changes

コードの動作を壊す食い違いは見つからなかった。指摘は、仕様書の古い記述・誤った例、docstring と実装のずれ、内部に残った古い既定値が中心。合わせて 1 回の docs/chore コミットで直せる量である。

## must-fix

なし。

## should-fix

1. **specs/network.md:92-94 の例が動かない**
   - 食い違い: `stack.connect(head)` の後に `stack.build()  # ルートで構築する` としているが、`head` は `stack` の木の外にある。そのため `build` は `ValueError: the port 'l2.next' is connected to a port outside the tree being built` になる（手元で再現した）。「ルートで build」という古い API（host 操作がネットワークにあった頃）の名残でもある
   - 正: 実装（`network._build` が木の外への Channel を拒む）
   - 直し方: `build()` の行を消すか、`stack` と `head` を包むネットワーク（または `mujou.run(stack, head)`）の例にする
   - 確信度: 高

2. **specs/messages.md:90-92「グローバル時刻の付与」**
   - 食い違い: スロットが「書いた時点のグローバル時刻」を付けると書いている。2026-10-16 以降は実行ごとのタイマーの時刻を付ける
   - 正: memory/project_timer_per_run.md、specs/execution-model.md:9,20
   - 直し方: 見出しと本文を「その実行のタイマーの時刻」にする（execution-model.md:20 と同じ言い方）
   - 確信度: 高

3. **specs/open-decisions.md:37 のトップレベル公開名の一覧**
   - 食い違い: `Node, Network, Port, PrevPort, NextPort, Channel, TensorSpec, DictSpec, NetworkError, PauseTimeoutError, run` に `Compose`・`Graph`・`Run` が無い
   - 正: `src/mujou/__init__.py` の `__all__` と specs/network.md:250
   - 直し方: 一覧に `Compose`・`Graph`・`Run` を足す（`mujou.testing` もあると書く）
   - 確信度: 高

4. **specs/open-decisions.md:45「`mujou.run(..., timer=...)` はホスト側の作業で入れる」**
   - 食い違い: 未実装の作業として書いているが、すでに実装済み（`run.py` の `timer` 引数、`Session` が受け取る）
   - 正: 実装、memory/project_timer_per_run.md「公開（2026-10-16）」
   - 直し方: 文を「`mujou.run(..., timer=...)` で実行ごとに渡す（実装済み）」にする
   - 確信度: 高

5. **specs/torch.md:96 のテストの場所**
   - 食い違い: 「`tests/mujou/_core/test_node.py` と `test_network.py` の `TestCuda`」とあるが、`test_network.py` に `TestCuda` は無い。実際は `test_node.py:687`、`test_run.py:1561`、`test_port.py:365`
   - 正: 実装（tests/）
   - 直し方: 3 ファイルを挙げ直す
   - 確信度: 高

6. **specs/network.md:250 の `mujou.time` の一覧に `ReadOnlyTimer` が無い**
   - 食い違い: `time` / `time_ns` / `sleep` / `get_time_scale` / `set_time_scale` / `Timer` / `Timestamp` だけを挙げている。実装（`src/mujou/time.py:28`）と specs/execution-model.md:79 は `ReadOnlyTimer` も公開している
   - 正: 実装、execution-model.md:79
   - 直し方: `ReadOnlyTimer` を足す
   - 確信度: 高

7. **`pause_timeout` の値の制約が文書に無く、`math.inf` を受け付けない**
   - 対象: `src/mujou/_core/run.py:295-296`（`mujou.run` の Args と Raises）、`src/mujou/testing.py:40`、specs/execution-model.md:232
   - 食い違い: `pause_timeout=math.inf`・負・NaN は `ValueError: duration must be finite and >= 0 seconds, got inf` になる（手元で確認）。docstring の Raises にこの場合が無く、メッセージも `pause_timeout` という名前を出さない。同じ「実時間の秒」を取る `shutdown` / `join` の `timeout` は `math.inf` を「いつまでも」として受け付けるので、利用者には不揃いに見える
   - 正: 実装の振る舞い（有限で 0 以上）を正とし、文書を合わせるのが最小。`inf` を受け付けるかは提案 1
   - 直し方: `mujou.run` の Args に「有限で 0 以上」、Raises の `ValueError` に `pause_timeout` を足す。できればメッセージに引数名を入れる
   - 確信度: 高（振る舞い）、中（どちらに寄せるか）

8. **コアの `pause_timeout` の既定値が 10 秒のまま**
   - 対象: `crates/mujou-core/src/runtime.rs:109`（`RuntimeConfig::default`）、`crates/mujou-python/src/runtime.rs:471`、`src/mujou/_core/_native.pyi:361`
   - 食い違い: 停止のタイムアウトの既定は実時間 30 秒（ユーザーの決定）。Python の層は常に 30.0 を渡すので利用者には影響しないが、コアと bindings の既定は 10 秒で、stub にもそのまま出ている
   - 正: memory/project_runtime_api_revision.md「停止のタイムアウトの既定は実時間 30 秒」、specs/execution-model.md:232
   - 直し方: コアと bindings の既定を 30 秒にして `just stub`。または既定を外して必須にする
   - 確信度: 高

9. **例外による終了でタイマーを保持する、という記述が Python の経路と合わない**
   - 対象: specs/execution-model.md:236、`crates/mujou-python/src/runtime.rs:417-420`（stub `_native.pyi:297-302` の `Runtime.__new__`「freezes it ... and on an exception」）
   - 食い違い: コアが例外で凍結するのは `capture_on_failure`、つまり `on_exception_capture` を渡したときだけ（`crates/mujou-core/src/runtime.rs:438-440`）。Python はもう渡さないので、`mujou.run` の実行は例外で凍結しない（起動中・取得中の保持は除く）。execution-model.md:236 の「保持している間（例外の記録から終了処理の完了まで）は、仮想時計の `advance` もできない」は、今の利用者の経路では起きない。同じ文書の :82 は「例外のときに状態を取得する設定のときだけ」と正しく書いており、文書の中でも食い違う
   - 正: 実装と execution-model.md:82
   - 直し方: :236 を「`on_exception_capture` を使うときだけ」に限るか、`on_exception_capture` を取り除くときに消す（提案 2）。bindings の doc comment も同様に直して `just stub`
   - 確信度: 中〜高

10. **specs/native-core.md:63「crate を 2 つに分ける」**
    - 食い違い: 直後の表は `mujou-core`・`mujou-python`・`mujou-stubgen` の 3 つ
    - 正: 実装（`crates/` に 3 つ）
    - 直し方: 「3 つ」にするか、「本体と bindings の 2 つに分け、stub の生成器を別に持つ」とする
    - 確信度: 高

11. **`src/mujou/__init__.py:1-2` のパッケージの docstring「trained with PC-ALM」**
    - 食い違い: PC-ALM は暫定の既定で、更新手法は差し替えられる（CLAUDE.md、memory/project_pluggable_update_rule.md）。`mujou.pc` もまだ無い。`PC-\nALM` と語の途中で改行してもいる
    - 正: memory/project_purpose.md、project_pluggable_update_rule.md
    - 直し方: CLAUDE.md の 1 行説明（Asynchronous Distributed Deep Learning Library for Continuous Learning）に寄せる
    - 確信度: 中

## nit

1. `src/mujou/_core/network.py:413,422,427-434` — `Compose` はノードも取るのに、エラー文が `"cannot compose a network as {name!r}"`、ループ変数も `network` のまま（`*networks` の名残）。「a component」などにする。確信度: 高
2. 根がノードの実行でも「network」と出るエラー文: `port.py:164,168`（`cannot connect a port of a network that is already built`）、`session.py:325,337`（`the network did not stop within ...`）、`session.py:404`（`the network has stopped; ...`）。仕様（network.md:211）は根のノードをクラス名で呼ぶ方針。「the run」にそろえると根がノードでも正しい。確信度: 中
3. `src/mujou/_core/network.py:111` の `Graph.__init__` は `"a Graph is made by Network.build()"`、`run.py:63` の `Run.__init__` は `"a Run is made by mujou.run()"`。前者は `mujou.run(node)` でも作られ、後者は `ManualRun()` でも同じ文言が出る（`mujou.testing.run` が作る）。確信度: 中
4. `src/mujou/_core/node.py:334` — docstring の 1 行が 127 文字（後から足した「``timer`` is a reserved name ...」が折り返されていない。E501 は無視設定なので検出されない）。確信度: 高
5. `crates/mujou-core/src/runtime.rs:98-100` — `RuntimeConfig::period_ns` の doc comment の 1 行が長く、文の途中で切れている。確信度: 高
6. 「グローバルタイマー／グローバルな時間」で Δt を測るという古い言い方が残っている（実行ごとのタイマーが正、既定がグローバル）: specs/pc-alm-continuous-time.md:3,79、specs/torch.md:47、specs/overview.md:13、specs/update-rules.md:43。確信度: 高（ずれ）、低（重要度）
7. specs/network.md:105「ネットワークの公開インターフェイスで Port を読み書きする」— ホストの操作は今は `Run` が持つ（同じ文書の :114）。確信度: 高
8. specs/open-decisions.md:3「2026-10-14 時点で」— 本文は 2026-10-17 まで更新されている。:21 の hook の一覧に `on_thread_start` が無い（nodes.md:62 にはある）。確信度: 高
9. specs/torch.md:12-13 のモジュールの表 — `_core` の中に `run` / `timer` が無く、公開モジュールの行に `mujou.testing` が無い。確信度: 高
10. `src/mujou/_core/__init__.py:3-4`「`_native` is the only module here that does not use torch」— `timer.py` も torch を import しない。方針（torch 非依存を約束するのは `_native` だけ）の言い方にすると正確。確信度: 中
11. `src/mujou/_core/network.py:515-520` の import 順のコメント「mujou/__init__.py imports it first」— 実際の `__init__` は `mujou.time`（→ `_core.timer`）と `_native` を先に import する。`timer` は `node` を遅延 import するので壊れてはいない。確信度: 中
12. `tests/mujou/_core/test_run.py:258,264` — 引数名 `networks` はノードも取る。確信度: 高

## 提案

1. **`pause_timeout` に `math.inf` を許す。** `shutdown` / `join` の `timeout` と同じく「いつまでも待つ」にする。ただし停止の待ちを無限にすると状態の取得が戻らなくなりうるので、採るかはユーザーの判断（should-fix 7 は文書だけ直す最小案）
2. **`on_exception_capture` / `Runtime.exception_state` を merge 前に取り除く。** 例外のときの状態は取らない、と決まっている（memory/project_shutdown_on_exception.md）。memory と specs/execution-model.md:235 は「後で取り除く」と認めているので食い違いではない。ただし、残っている間は should-fix 9 のような記述のずれの元になる。対象は `crates/mujou-core/src/runtime.rs`（`ExceptionCapture`、`FreezeReason::Failure`、`capture_on_failure`）、bindings、stub、`tests/mujou/_core/test_native_runtime.py:381,860-896`
3. **テストの `Solo` ラッパーを整理する。** `tests/mujou/_core/test_node.py:44` と `test_run.py:136` に同じ `Solo(Network)` が二重にある。2026-10-17 から `mujou.run(node)` はノードを根にして走らせられるので、パス `"node"` を確かめる必要のないテストは `start_manual(node)` で足りる
4. **`mujou.__all__` のテストを足す。** `mujou.testing.__all__` は `tests/mujou/test_testing.py:9` で固定しているが、トップレベルの公開名を固定するテストが無い（`test_core.py` を消した後）。公開名が仕様（network.md:250）とずれたときに検出できるようにする
5. **`session.py` のテストの置き場。** `tests/mujou/` は `src/mujou/` をミラーする規約だが、`session.py` に対応する `test_session.py` は無く、`test_run.py` / `test_port.py` が受け持っている。`Session` は `Run` の実装なので今の形でよいとするなら、規約側に例外として書く

## 範囲外（audit-meta 向けの参考）

memory/ にも古い記述がある。`project_host_io.md:11`（`mujou.run(network)` が network を返す、`state_dict` は実行中だけ、`subscribe` は呼んだ後の値だけ）、`project_open_decisions.md:19,23`（トップレベル名に `Compose`/`Graph`/`Run` が無い、`timer=` はこれから入れる）、`project_message_types.md`（時刻はグローバル時刻）、`project_node_lifecycle.md`（`on_exception` / `on_thread_start` が無い）。

## 検証結果

- just run: 未実行（依頼の範囲は読み取りと軽い実行の確認なので、全テストは回していない）
- `uv run pyright`: pass（0 errors）
- `uv run pytest tests/mujou/test_testing.py tests/mujou/_core/test_native_stub.py tests/mujou/_core/test_node_typing.py`: 22 passed
- 手元での再現: specs/network.md:92-94 の例 → `ValueError`、`Compose(build=...)` のエラー文、`pause_timeout` に `-1` / `nan` / `inf` → `ValueError`、`0.0` → `PauseTimeoutError`
- grep で名残が無いことを確かめたもの: `Network.run`、`exception_state_dict`、`capture_state_on_exception`、`_manual`、手書き stub、`arc-swap`（説明文と smoke test の名前だけ）、`docs/*.md` への古いパス、`max_dt`（「持たない」とだけ書いてある）、TODO/FIXME、未使用の関数とテストヘルパー
