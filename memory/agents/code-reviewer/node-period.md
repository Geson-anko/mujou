# node-period（ノードの実行周期 `period`）レビュー

対象: `feature/2026-10-17/node-period`、`git diff 3583478..HEAD`（2cca89f..8f82ae2）。

## verdict: approve

must-fix はない。下の should-fix のうち 1〜3 は、run-handle ブランチを merge するときに一緒に片付ければよい。

## must-fix

なし。

## should-fix

1. **merge 後に `TestPeriod` の結合テストが壊れる**（確信度: 高）
   - 対象: `tests/mujou/_core/test_node.py` の `TestPeriod.test_the_node_waits_for_its_period_on_the_timer`
   - 問題: `git merge-tree HEAD feature/2026-10-16/run-handle` では、test_node.py は衝突なしで自動 merge される。ところが merge 後もこのテストは `virtual_timer: GlobalTimer` と `run_network: RunNetwork` を使ったまま残る。run-handle 側の test_node.py は `GlobalTimer` と `RunNetwork` の import を消しており、conftest からも `run_network` がなくなっている。テキストの衝突が出ないので見落としやすく、pyright と pytest（fixture not found）で初めて落ちる
   - 根拠: merge-tree の結果（tree b979acd）の test_node.py の 13〜18 行目には `GlobalTimer` も `RunNetwork` もない。run-handle の `tests/mujou/conftest.py` にあるのは `virtual`・`start`・`start_manual` だけ
   - 方向: `virtual: VirtualTimer` と `start: Start` で `start(Solo(node), timer=virtual)` とし、`virtual.advance(0.005)` で進める（周期の確認には自由実行が要る。手動の `start_manual` は周期を使わない）

2. **merge 後、`period` を予約名の一覧に入れる**（確信度: 高）
   - 対象: `src/mujou/_core/node.py` のクラス docstring と `specs/nodes.md`
   - 問題: `period` は setter のない property なので、サブクラスの `__init__` で `self.period = 2.0` と書くと `AttributeError: property 'period' of 'Sub' object has no setter` になる（実測）。周期を持つ発振器のようなノードの利用者は、この名前を使いがち。run-handle は docstring に「``device``, ``generator`` and ``timer`` ... are reserved names: do not assign attributes of those names.」を足しているが、merge するとこの一覧に `period` が入らない
   - 根拠: `git show feature/2026-10-16/run-handle:src/mujou/_core/node.py` の 66〜67 行目

3. **merge で解消する衝突（周期に関係するもの）**（確信度: 高）
   - `src/mujou/_core/session.py`: run-handle では `Session.run` がなくなり、`Session.start` が `runtime.start(manual=..., pause_timeout=..., **(restored or {}))` を呼ぶ形に変わっている。`period_ns=[_period_ns(node) for node in self._nodes]` をこの呼び出しへ移す。`restored` の dict には入れない（入れると復元しない実行で周期が渡らない）。`_period_ns` 関数は自動 merge で残る
   - `src/mujou/_core/node.py` の `on_update(dt)` の行: run-handle は "global time" を "the time in seconds on the run's ``timer``" に変えた。この文言に "every ``period`` (see below)" を足す。node-state と run-handle の間の衝突（`__setattr__`/`state_dict` と `timer` property）も同じファイルにあるが、これは周期の差分とは別
   - `specs/execution-model.md` の手動モードの行: run-handle の `mujou.testing.run` / `ManualRun.step(*nodes)` の表記に「周期は使わない」を足す。周期の実装の段落（`RuntimeConfig::period_ns` ...）はそのまま残す
   - `specs/nodes.md`: 衝突しているのは「状態の自動収集」の節（node-state 由来）。「実行の周期」の節はそのまま残し、最後の段落は run-handle の「`mujou.run` でネットワークを走らせると…」を採る
   - Rust（`crates/mujou-core/src/runtime.rs`, `crates/mujou-python/src/runtime.rs`）と `_native.pyi` は衝突しない。run-handle は `Runtime(..., timer=...)` を足すだけで、`start` の引数には触れていない

4. **サブクラスが `period` をクラス属性で上書きすると、検証を通らずに周期が決まる**（確信度: 中）
   - 対象: `src/mujou/_core/session.py:570`（`_period_ns` が public の `node.period` を読む）
   - 問題: `class Sub(Node): period = 0.5` と書くと property が隠れ、session はその値をそのまま使う。`period = 0` や `period = -1` は `max(1, ...)` で黙って 1 ns（= 周期なし）になる。`math.inf` は `round` で `OverflowError`、NaN は `ValueError: cannot convert float NaN to integer` になり、どちらも周期と結びつかないメッセージで起動時に落ちる。判断ログには「この API は入れなかった」とあるが、書けば効いてしまう
   - 根拠: `node.py` の `_period = None` は class 属性で、`period` は通常の property（上書きを防ぐものがない）。`_period_ns` は値を検証しない
   - 方向: どちらかに決めて docstring に書く。(a) 予約名にして、session は `node._period` を読む（同じパッケージの private を読む前例は run-handle の `set_timer` にある）。(b) クラス属性での既定を正式に認め、`_period_ns` で同じ検証をする。未決定事項の「クラス属性の既定値」と合わせてユーザーに確認する

5. **復元したときの起点（保存した最後の更新時刻 + P）を確かめるテストがない**（確信度: 中）
   - 対象: `crates/mujou-core/tests/runtime.rs` の `mod period`
   - 問題: specs/nodes.md の表には「復元したときは保存した最後の更新時刻から P」と明記しているが、`last_update_ns` と `period_ns` を同時に与えるテストがない。実装は正しく（`node_loop` の `last_ns` の初期値が `config.last_update_ns`、runtime.rs:1036）、仕様と実装は一致している。ただ、周期の起点を「実行の開始」に変えるような退行があっても検出できない
   - 方向: 仮想時計で `last_update_ns: Some(vec![t0 - P/2])` を与え、P/2 − 1 ns の前進では更新せず、P/2 で Δt = P になることを見るテストを 1 本足す

## nit

1. `src/mujou/_core/node.py`（`Node.__init__`）: 非常に大きい `period`（約 584 年 = u64 の ns を超える値、例 `1e12`）を `__init__` は受け付けるが、`run` のときに `OverflowError: int too big to convert` で落ちる（実測）。判断ログでは対処しないと決めているので、残すならこの挙動でよい。直すなら `__init__` で上限を検査するか、`_period_ns` で u64 の最大値に丸める（確信度: 高、深刻度: 低）
2. `src/mujou/_core/node.py`: `Node(period=True)` を受け付け、`period` が `True` を返す（`run` では 1 秒として動く）。`device` や `seed` の検証の粒度と揃っているなら放置してよい（確信度: 高、深刻度: 低）
3. `src/mujou/_core/node.py` のクラス docstring「since the start of the last update (or since the start of the run)」: specs にある復元の場合（保存した最後の更新時刻）が抜けている。`on_update(dt)` の「every ``period``」は、正確には「at least ``period`` apart」（確信度: 中）
4. `specs/nodes.md`「仮想時計なら、時計を P 進めたときにだけ更新する」: 正しくは「前回の更新から合計で P 進んだとき」。1 回の `advance` で P 進める必要はない（確信度: 中）
5. `src/mujou/_core/session.py` の `_period_ns`: Python の `round` は偶数丸めで、Rust の `seconds_to_ns`（`f64::round` は 0 から遠い側へ丸める）と ちょうど .5 ns の値で 1 ns ずれる。実害はほぼない（確信度: 高、深刻度: 極小）
6. `crates/mujou-core/tests/runtime.rs` の `the_time_scale_scales_the_updates_per_real_second`: 下限 `unscaled = elapsed / P + 1` の `elapsed` に起動と終了の時間が入るので、負荷で起動が遅れると下限だけが厳しくなる。下限は `run`（`thread::sleep` の長さ）で計算すると、意図どおり緩くなる。上限 `≤ elapsed × scale / P` は、タイマーの経過が実時間の経過を超えないので（起動中は凍結、`ns_at` は切り捨て）健全。実測では余裕が十分にある（確信度: 中、深刻度: 低）
7. `crates/mujou-core/src/runtime.rs:555` 付近: `last_ns` が u64::MAX 近くになると `saturating_add` で `next_ns` が頭打ちになり、`now_ns == u64::MAX` で Δt < P の更新がありうる。約 584 年後なので起こりえない。記録のためだけに挙げる（確信度: 高、深刻度: なし）
8. `_period_ns` の「1 ns 未満は 1 ns」は specs に書いてあるが、テストがない。公開 API からは `Node(period=1e-10)` を仮想時計で 1 ns 進めて更新されることで確かめられる（確信度: 中、深刻度: 低）

## 確認して問題がなかった点（質問された観点のみ）

- 周期なしを 1 ns の周期として扱う変形: 自由実行は `now <= last` → `now < last + 1`、手動は `min_dt_ns = 1` に固定。saturating の端（nit 7）以外は従来と同じ
- `wait_until_ns_or` の起床: 停止・capture・終了の要求は `set_command` → `interrupt_waiters`、スケールの変更・仮想時計の前進・pause/resume は `update` の `notify_all` で起きる。待ちの実時間はスケールから計算し直す
- 周期の検証（0 と長さ違い）は `freeze_for_startup` の前に行うので、エラーのときに凍結が漏れない
- 局所性: 待ちはノードごとで、グローバルなバリアを持ち込んでいない
- 成果物の汚染（`</content>` など）: なし

## 検証結果

- just run: pass（397 passed, 15 skipped、pyright 0 errors、mkdocs build --strict 通過）
- `cargo test --test runtime`: 5 回連続 pass（23 passed）。`period` だけを 12 並列で 2 回まわして全 pass。CPU を飽和させた状態でも 8 回 pass（注: このとき負荷用に `yes` を 28 本立て、片付けに失敗してユーザーのマシンを約 9 分間 100% にした。以後この種の負荷試験はしない）
- pytest（worktree の `.venv`、`mujou` が worktree の src を指すことを確認済み）: free-threaded（`sys._is_gil_enabled() == False`）で 3 回、`-X gil=1` で 3 回、すべて 397 passed / 15 skipped
- run-handle との試し merge（`git merge-tree`、作業ツリーは変更していない）: 衝突は specs/execution-model.md、specs/nodes.md、specs/torch.md、node.py、session.py。node-state と run-handle の間でも同じ 5 ファイルが衝突する。周期に関係する箇所は should-fix 1〜3 に書いた
