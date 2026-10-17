# PR stack（#7〜#24）最終形の整合性監査: リポジトリの運用まわり

対象: `CLAUDE.md`、`.claude/`（skills・agents・hooks・settings.json・scripts）、`memory/`、`justfile`、`.github/workflows/`、`.pre-commit-config.yaml`、`pyproject.toml`、`Cargo.toml`、`.devcontainer/`、`.vscode/`、`mkdocs.yml`（運用部分）。ファイルは修正していない。

## verdict: approve

運用領域には、マージを止めるほどの must-fix は無い。ただし should-fix の 1〜4（CI の paths の抜け、worktree で locality guard が効かない問題、memory の古い記述）は、マージ前かマージ直後に直すことを勧める。

## must-fix

なし

## should-fix

1. `.github/workflows/type-check.yml:5-15, 17-27`
   - 何が食い違うか: paths が `src/**` と Rust・ロックファイルだけ。一方 `just type`（pyright）は `include` 未指定のためリポジトリ全体を検査する。`uv run pyright --verbose` の結果は "Found 40 source files" で、内訳は src・tests・examples・hooks/anchors.py。`tests/**` や `examples/**` だけを変えた PR では type-check が走らず、型エラーがあっても CI を通ってしまう
   - 正とすべき側: `just type` の実際の検査範囲（CLAUDE.md の「コミット前の検証は `just run`」とも合う）
   - 直し方の案: paths に `tests/**/*.py`、`examples/**/*.py`、`hooks/**` を足す（または `**/*.py`）
   - 確信度: 高
2. `.github/workflows/docs.yml:7-17, 19-29`
   - 何が食い違うか: `mkdocs.yml` の `hooks: - hooks/anchors.py`（MkDocs の hook。アンカー検査を緩める）が paths に無い。これだけを変えた PR では docs のビルドが走らない
   - 正とすべき側: `mkdocs.yml` の実際の入力
   - 直し方の案: paths に `hooks/**` を足す
   - 確信度: 高
3. `.claude/hooks/posttooluse-locality-guard.py:29-34`（と `tests/test_claude_hooks.py:95-139`）
   - 何が食い違うか: `CLAUDE_PROJECT_DIR`（main の checkout）からの相対パスの先頭 2 要素が `("src", "mujou")` のときだけ検査している。worktree（`<main>/.claude/worktrees/<name>/src/mujou/...`）での編集は相対パスが `.claude/worktrees/...` になるので、黙って素通りする。再現: `CLAUDE_PROJECT_DIR=<proj>` で `<proj>/.claude/worktrees/wt/src/mujou/_core/x.py`（中身 `loss.backward()`）を渡すと出力が無い。同じ内容の `<proj>/src/mujou/_core/y.py` なら警告が出る。この stack の実装は大部分が worktree の中のサブエージェントによるもので、そこで guard が効いていなかった可能性が高い
   - 正とすべき側: CLAUDE.md:58 と skill `locality-and-async`:106（「`src/mujou/` に入った違反しそうな箇所は hook が警告する」）。運用（`do-on-worktree`、`isolation: worktree`）は worktree を前提にしている
   - 直し方の案: `.claude/worktrees/<name>/` を剥がしてから判定する。または相対パスのどこかに連続する `src`/`mujou` があれば対象にする。テストに worktree のケースと `src/mujou/_core/` のケースを足す（今のテストは `src/mujou/train.py` のトップだけ）
   - 確信度: 中〜高（サブエージェントの hook に渡る `CLAUDE_PROJECT_DIR` が main の checkout であることが前提。このセッションの CLAUDE.md は main から読まれている）
4. memory の古い記述（後の決定で上書きされたのに、注記が無いもの）
   - 4a. `memory/project_host_io.md:3, 10-13, 22` と `memory/MEMORY.md:27`
     - 何が食い違うか: 「`mujou.run(network)` は…同じ network を返す」「インターフェイスは `build()` を呼んだルートのインスタンスが持つ」「操作: `write`、`load_state_dict(state)`（run 前）…」「run 前に `write` した値がそのスロットの初期値」「名前は推奨案、ユーザー確認待ち」と書いてある。いずれも 2026-10-16 の run API の見直しより前の内容
     - 正とすべき側: `memory/project_runtime_api_revision.md:11`（run がハンドル `Run` を返し、初期値と復元は `inputs=` / `state=` で渡す）、`memory/project_run_node_and_write_time.md:11`（シグネチャはユーザーの決定）、`src/mujou/_core/run.py`
     - 直し方の案: 冒頭に「2026-10-16 の [[project-runtime-api-revision]] と 2026-10-17 の [[project-run-node-and-write-time]] で、run の戻り値・初期値・復元の渡し方は上書きされた」と注記する。古い行は打ち消すか書き換える。索引の説明も今の形（`Run` ハンドル、`inputs=`）に直す
     - 確信度: 高
   - 4b. `memory/project_open_decisions.md:10, 15, 16`
     - 何が食い違うか: 10 行目のホストの操作に `load_state_dict` があり、「ユーザー確認待ち（[[project-host-io]]）」のまま。15 行目のトップレベルの公開名の一覧に `Compose`・`Graph`・`Run` が無い（実際の `src/mujou/__init__.py:13-29` にはある）。16 行目に「`mujou.run(..., timer=...)` はホスト側の作業で入れる」とあるが、すでに実装済み（`specs/execution-model.md:79`、`mujou.run(component, *components, timer=None, ...)`）
     - 正とすべき側: コード（`src/mujou/__init__.py`、`src/mujou/_core/run.py`）と `project_run_node_and_write_time.md:11`
     - 直し方の案: 決定済みの項目を消すか「決定済み（日付、リンク）」に直す。公開名の一覧を `__all__` に合わせる。`specs/open-decisions.md` にも同じ 2 点（公開名の一覧、`timer=` は「ホスト側の作業で入れる」）がある。担当領域の外なので specs 側の担当に回す
     - 確信度: 高
   - 4c. `memory/project_testing_api.md:8` と `:12` が矛盾している
     - 何が食い違うか: 8 行目は「実装は後の段階で、ドキュメントには…枠だけがある」、12 行目は「実装済み（2026-10-16）」と言っている
     - 正とすべき側: 12 行目（`src/mujou/testing.py` が実在する）
     - 直し方の案: 8 行目の後半を消すか「（当時）」と注記する
     - 確信度: 高
   - 4d. `on_exception` の hook が memory に一度も記録されていない。該当箇所は `memory/project_node_lifecycle.md:3, 12`、`memory/project_shutdown_on_exception.md:11`、`memory/MEMORY.md:23`、`.claude/skills/locality-and-async/SKILL.md:20`
     - 何が食い違うか: hook の一覧が `on_start / on_update / on_pause / on_resume / on_stop / state_dict / load_state_dict` で止まっている。終了の流れも「全ノードの `on_stop()`」だけになっている。実装（`crates/mujou-python/src/runtime.rs:91,308`、`crates/mujou-core/src/runtime.rs:659`）と利用者向け文書（`docs/en/user-guide/errors.md:10`、`docs/en/concepts/nodes.md:38`）には `on_exception()` があり、2026-10-17 に足された公開の `on_thread_start()`（`project_runtime_api_revision.md:16`）も一覧に無い。`git log -S on_exception` で見ると、仕様に入ったのは commit 0046a2c（「notify all blocks via on_exception()」）だが、memory は追従していない
     - 正とすべき側: コードと `specs/nodes.md`
     - 直し方の案: hook の一覧に `on_thread_start()` と `on_exception()` を足し、終了の流れを「`on_exception()` → `on_stop()`」に直す（索引と skill の 1 行も同じ）
     - 確信度: 高
   - 4e. `memory/project_shutdown_on_exception.md:13`
     - 何が食い違うか: 「`run()` はバックグラウンドですぐ戻るので」とある。今は全ノードの起動が終わってから戻り、起動中の例外は `run` 自身が送出する（`project_host_io.md:10`、`project_runtime_api_revision.md:24`）。例外を受け取る窓口も今は `Run` のメソッドになっている
     - 正とすべき側: `specs/network.md:110`、コード
     - 直し方の案: 「（2026-10-16 に変更: run は起動を待って `Run` を返す）」と注記して書き換える
     - 確信度: 高
   - 4f. `memory/project_timer_per_run.md:11`
     - 何が食い違うか: 「将来の API は `mujou.run(*networks, timer=...)`」「ノードは自分の属性としてタイマーを持つ予定」とある。同じファイルの 20 行目には実装済み（`Node.timer`）とあり、シグネチャも後に `mujou.run(component, *components, timer=...)` に変わった
     - 正とすべき側: `project_run_node_and_write_time.md:11`、コード
     - 直し方の案: 「将来」「予定」を外し、今のシグネチャに直すか、上書きされた旨を注記する
     - 確信度: 高
   - 4g. `memory/project_runtime_api_revision.md:11-12`
     - 何が食い違うか: `mujou.run(*networks, ...)` / `Compose(*networks)` のまま。後の決定（ノードも受け付ける、引数なしはエラー、1 つなら包まない）へのリンクが無い。28 行目「実装は PR #13 の上に stack で積む」も、マージ後は意味を失う
     - 正とすべき側: `project_run_node_and_write_time.md:11-12`
     - 直し方の案: 「2026-10-17 に `mujou.run(component, *components)` に改めた（[[project-run-node-and-write-time]]）」と注記する。`project_run_node_and_write_time.md` の末尾にも「関連:」行を足して相互リンクにする
     - 確信度: 高
   - 4h. `memory/project_async_node_execution.md:27, 29` と `.claude/skills/locality-and-async/SKILL.md:112`
     - 何が食い違うか: 27 行目と skill の 112 行目は「ネットワークのテストはグローバルタイマーを仮想時計に切り替える」と書いている。今は実行ごとの `VirtualTimer` を渡す（`tests/mujou/conftest.py` の `virtual_timer` の docstring に「グローバルタイマーそのもののテスト専用。run は `virtual` timer を使う」とある）。29 行目は「利用者に公開するのは `mujou.time` の `time`…`set_time_scale` だけ…仮想時計は `_native.timer` に留める」と書いているが、後に `mujou.time.Timer` / `ReadOnlyTimer` / `Timestamp` と `mujou.testing.VirtualTimer` を公開した
     - 正とすべき側: `project_timer_per_run.md:20`、`project_testing_api.md:12`、conftest
     - 直し方の案: テストの記述を「ネットワークのテストは実行ごとの `VirtualTimer`、グローバルタイマー自体のテストだけグローバルタイマーを仮想時計に」に直す。29 行目に「2026-10-16 に公開範囲を広げた（[[project-timer-per-run]]）」と注記する
     - 確信度: 高
   - 4i. `memory/project_torch_module.md:10, 23`
     - 何が食い違うか: 10 行目の `_core` のモジュールの一覧が `spec / port / network / node / session` で、実在する `run`・`timer` が無い。23 行目「実装はコアの `Node` ができてから」は、`mujou.Node` が実装済みなので古い
     - 正とすべき側: `src/mujou/_core/`
     - 直し方の案: 一覧を今の構成に合わせる（または一覧をやめて「`_native` 以外」と書く）。23 行目の未決定事項を今の状態に直す
     - 確信度: 高（10 行目）／中（23 行目は未決定事項の残りがまだあるかどうか次第）
5. `just run` の説明が古い: `.claude/skills/refactor-conventions/SKILL.md:70`（`just run   # format → test → type`）と `.claude/agents/plan-implementer.md:52`（`just run`（format → test → type））
   - 何が食い違うか: `justfile:35` の `run: format test type docs-build` には docs-build がある
   - 正とすべき側: `justfile`（CLAUDE.md:97 と README.md:19 は直っている）
   - 直し方の案: 「format → test → type → docs-build」に直す
   - 確信度: 高
6. `tests/conftest.py` への参照が 4 か所: `.claude/agents/spec-test-author.md:41`、`.claude/skills/agent-team-startup/SKILL.md:104`、`.claude/skills/agent-team-startup/reference.md:50`、`.claude/skills/testing-strategy/SKILL.md:68`
   - 何が食い違うか: そのファイルは無く、実在するのは `tests/mujou/conftest.py`。`testing-strategy/SKILL.md:40` も「`tests/` 直下に置けるのは共通資産（`conftest.py`）」と書いている
   - 正とすべき側: 実際の配置
   - 直し方の案: `tests/mujou/conftest.py` に直すか、「`tests/` 配下の `conftest.py`」と書く
   - 確信度: 高（深刻度は低い）
7. `CLAUDE.md:88-100` のコマンド表に `just stub` が無い
   - 何が食い違うか: `justfile:60-78` の `stub` は、bindings を変えたときに必須の手順（`memory/project_core_stub.md:17`）。pre-commit の `mujou-stub` もこれを呼ぶので、`just format` には build 済みの拡張と `just` が要る。表から辿れない（`just clean` も載っていないが、これは好みの範囲）
   - 正とすべき側: `justfile` と `project_core_stub.md`
   - 直し方の案: 表に `just stub`（「`_native.pyi` を Rust の doc comment から生成。bindings を変えたら実行してコミット」）を足す
   - 確信度: 中
8. `.claude/skills/solo-dev-cycle/SKILL.md:70, 84` と `.claude/agents/code-simplifier.md:17, 38`
   - 何が食い違うか: ドキュメント段階の対象が「docstring／README／CLAUDE.md／memory／skill／docs/」で、`specs/`（設計仕様。CLAUDE.md:68 では設計判断とその理由を書く場所）と `examples/`（CLAUDE.md:70 では利用者向けのサンプル）が無い。`.claude/agents/implementation-planner.md:31` の「現状把握」にも `specs/` を読む指示が無い
   - 正とすべき側: CLAUDE.md の「ドキュメント」節
   - 直し方の案: 段階 5 の対象と simplifier の同期対象に `specs/` と `examples/` を足す。planner の現状把握に「関係する `specs/` を読む」を足す
   - 確信度: 中

## nit

1. `justfile:58-59`: `stub` の説明コメントが 2 行あり、`just --list` には 2 行目だけが出る（`stub  # signatures and doc comments. Commit the result; pre-commit regenerates it.`）。1 行にまとめる。確信度: 高
2. `.claude/skills/locality-and-async/SKILL.md:93`: 禁止パターン表の代替が「ノードごとの `step` を定義し」になっている。今の hook 名は `on_update(dt)`。確信度: 中
3. `memory/project_pause_and_state.md:10, 17` と `memory/project_message_types.md:21`: 「グローバルタイマーを凍結」「木にはグローバルタイマー」「書き込み時のグローバル時刻」とある。2026-10-16 以降は「その実行のタイマー」が正しい（[[project-timer-per-run]]）。確信度: 中
4. `CLAUDE.md:86`（と `memory/project_torch_module.md:10`、`specs/open-decisions.md` の同じ文）: 「torch に依存しないのは `_native` だけ」とあるが、`src/mujou/_core/timer.py` と `src/mujou/time.py` も torch を import しない。`import mujou` で torch は必ず読まれるので実害はない。「torch を使わなくてよいのは」くらいに緩めるか、このままでよいか判断する。確信度: 低
5. `.claude/skills/solo-dev-cycle/SKILL.md:78-84`: 段階ゲートの表の列幅がそろっていない（`.claude/` は mdformat の対象外）。確信度: 高

## 提案（好みの範囲・判断はユーザー）

1. `pyproject.toml:5` の description（"…trained with PC-ALM instead of backpropagation"）と `src/mujou/__init__.py:1-2` の docstring（"trained with PC-ALM"）は、README.md:7（PC-ALM は未実装で、更新手法は差し替えられる）や CLAUDE.md:23（PC-ALM は暫定の既定）と温度が違う。README 側に寄せるなら "Asynchronous Distributed Deep Learning Library for Continuous Learning" だけにする。確信度: 中
2. `.claude/hooks/pretooluse-guard-bash.py:54-73`: 次の形がブロックされない（実測で exit 0）。`python3.14t -m pip install`（開発の既定は 3.14t）、`uv run pip install`、`uv run python -m pip install`、`.venv/bin/pip install`、`git -c k=v commit -n`。直すなら `pip3?` を basename で比べる、`python3?(\.\d+)?t?` にする、`uv run` を剥がす、`git` のグローバルオプションを飛ばす、のいずれか。確信度: 高（挙動）／判断は好み
3. `.pre-commit-config.yaml:27`: `uv-pre-commit` が rev 0.9.24 で、ローカルと dev container の uv は 0.13.0（`.devcontainer/Dockerfile:4`）。lockfile の書式が uv の版で変わったときに `uv-lock` フックが書き戻す余地がある。そろえておくと安全。確信度: 低
4. `.vscode/extensions.json`: `rust-lang.rust-analyzer` が無い（`.devcontainer/devcontainer.json` にはある）。確信度: 高／深刻度は極小
5. `tests/test_claude_hooks.py`: `statusline.sh`（60% の警告 marker を書く側）にテストが無い。CLAUDE.md:128 は「テストは `tests/test_claude_hooks.py`」としか言っていないので食い違いではないが、閾値の回帰を見るなら 1 件足す余地がある。確信度: 中
6. `.claude/settings.json:170-173`: `additionalDirectories` の `".claude/worktrees/**"` は project の内側を指す相対の glob で、効いているか怪しい（無害）。確信度: 低

## 確認して問題が無かった主な点（参考）

- `CLAUDE.md` の skill の索引（15 個）は `.claude/skills/` と一致する。agent の model と effort の記述も、frontmatter と一致する
- `docs/` → `specs/` の改名: `CLAUDE.md`・`.claude/`・`memory/`（agents を除く）に古い `docs/<spec>.md` への参照は無い。`mujou._session` / `mujou._node` などの旧モジュール名も無い（`.gitignore:56-59` の旧 `.so` の注記だけで、これは意図したもの）
- `memory/MEMORY.md` の索引: トップの 27 ファイルはすべて載っていて、リンク切れも無い。全メモリの `[[wiki-link]]` は frontmatter の name と一致する。name とファイル名も一致する
- memory が参照する specs の節（`実行の周期`・`スロットの実装`・`ツールチェーン`・`複数のネットワーク`・`実装と公開元` など）、`crates/mujou-core/tests/runtime.rs` の `mod period`、`network._child` は実在する
- CI の test は ubuntu と macOS、3.14t と 3.14 の両方で回り、CLAUDE.md:81 と `project_python_free_threaded.md` に一致する。pre-commit は Python 3.14 で、`project_runtime_api_revision.md:22` に一致する。stub の古さの検出（`just stub && git diff --exit-code`）も `project_core_stub.md:13` に一致する
- `Cargo.toml` の `rust-version = "1.85"` は edition 2024 と依存の MSRV（pyo3 0.30.1 などは 1.85）に一致する
- `.devcontainer/Dockerfile` の just 1.58.0 は、CLAUDE.md:82 と README.md:15 の「1.58 以上」に一致する
- 成果物の汚染: git 管理下の全ファイルについて、末尾 200 バイトの `</content>` を確かめた。ヒットはすべて reviewer のメモにある「汚染なし」の文言だけだった

## 検証結果

- just run: 実行していない。pre-commit がファイルを書き換えうること（このタスクはファイルを修正しない）と、slow を含む全テストの負荷を避けるため
- 代わりに `uv run pyright`: 0 errors（40 files。tests・examples が検査対象に入ることを確認）、`uv run pytest tests/test_claude_hooks.py`: 21 passed
- `just --list`: 1.58.0 で `[windows] set shell` を含めてパースできる
