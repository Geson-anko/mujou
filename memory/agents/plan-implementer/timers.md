# timers（実行ごとのタイマー・Timestamp・スタブの自動生成）の判断ログ

ブランチ `feature/2026-10-16/timers`（PR #13 の上）。skill `solo-dev-cycle` で 1 人で回した。

## 1. 計画

### 要求

- 将来の `mujou.run(*networks, timer=...)` のために、タイマーを実行ごとに持てるようにする（`None` なら既定のグローバルタイマー）
- 時刻の型 `Timestamp`。`float(ts)` / `int(ts)` は秒（int は整数部）、ナノ秒は property
- 手書きの `src/mujou/_core.pyi` を自動生成にし、古くなったら気づけるようにする
- 拡張モジュールの名前を後で `mujou._core.<名前>` に移せるようにする
- Python 側（`src/mujou/*.py`）は変更なしで動く（後方互換）

### Rust コアの IF

- `time::GlobalTimer` を `time::Timer` に改名。`Timer::new()` は時刻 0・実時計・スケール 1。共有は `Arc<Timer>`
- `time::global() -> &'static Arc<Timer>`（既定のグローバルタイマー）
- `Timer::hold(self: &Arc<Self>) -> PauseHold`。`PauseHold` は `Arc<Timer>` を持ち、lifetime を持たない（ランタイムが `'static` の借用なしに保持できるように）
- `time::Timestamp`（u64 ns の newtype）: `from_ns` / `as_ns` / `as_secs_f64` / `whole_secs`、`Copy + Ord + Hash`。`slot::Message::timestamp()`（`time_ns()` は残す）
- `slot::Outlet::new(schema, channels, name, timer: Arc<Timer>)` / `slot::InputGroup::new(timer: Arc<Timer>)`
- `runtime::Runtime::start(nodes, names, timer: Arc<Timer>, config, capture)`
- 走っているランタイムの数（LIVE 表）はタイマーごとのまま。キーは `Arc` のアドレスで、数えている間は `Live` が `Arc<Timer>` を持つので、アドレスが別のタイマーに再利用されることはない

### Python（`mujou._core`）の IF

- `Timer()`: 新しいタイマー（時刻 0、実時計、スケール 1）。メソッドは従来の `GlobalTimer` と同じ（`time` / `time_ns` / `sleep` / `time_scale` / `set_time_scale` / `pause` / `resume` / `is_paused` / `state_dict` / `load_state_dict` / `use_virtual_clock` / `advance`）
- `GlobalTimer(Timer)`: グローバルタイマーの型。コンストラクタなし。`timer: Final[GlobalTimer]` は従来どおり
- `Timestamp(*, ns: int)`: `ns` property、`__float__`（秒）、`__int__`（秒の整数部）、比較・hash、`repr` は `Timestamp(ns=...)`
- `Message.timestamp -> Timestamp`（`time_ns` は残す）
- `Outlet(schema, channels, name, *, timer=None)` / `InputGroup(*, timer=None)` / `Runtime(nodes, names, *, timer=None)`。省略でグローバルタイマー
- （計画時）`NetworkError` / `PauseTimeoutError` を `create_exception!` から pyclass に変える。`create_exception!` の型は pyo3 の introspection に出ず、モジュールが incomplete（`__getattr__` 付き）のスタブになるため。→ 実装では変えなかった（3 を参照）

### スタブの生成

- pyo3-stub-gen は pyo3 `<0.30` にしか対応しない（0.23.1 時点）。pyo3 0.30 の `experimental-inspect` と `pyo3-introspection`（同じ 0.30.1）を使う
- そのため `#[pymodule]` を関数形式から inline module（declarative）に変える。関数形式は introspection できない
- 生成器は workspace の小さな bin crate `crates/mujou-stubgen`（`pyo3-introspection = "=0.30.1"` を固定）。`just stub` が `uv run` でビルド済みの拡張を探し、`src/mujou/_core.pyi` に書き、ruff で整形する
- 古さの検出: `.pre-commit-config.yaml` の末尾に local hook を追記し、`just stub` を走らせる。生成物が変われば pre-commit が「ファイルを変更した」として失敗する。`just format`（= `just run` の最初）でも走る

### テスト観点

- Rust: 2 つのタイマーが独立に進む（仮想時計で片方だけ advance、片方の hold が他方を止めない）。`global()` は毎回同じタイマー。`Timestamp` の変換・順序。スロットが渡されたタイマーで時刻を付ける。別タイマーのランタイムの capture / 停止がもう片方を止めない。復元の排他はタイマーごと（同じタイマーなら拒否、別のタイマーなら通る）
- pytest: 上の Python 版と、`Timer()` が 0 から始まる、`GlobalTimer` がコンストラクタを持たない、`Timestamp` の変換・比較・hash・repr、`Message.timestamp`、引数省略でグローバルタイマー（既存テストが担保）

### リスク

- declarative module への移行で、クラスの `__module__` や例外の型が変わらないこと（既存テストと新しいテストで確認）
- スタブの型の精度: 生成器は `Bound<PyAny>` を `Any` にする。手書きにあった `Sequence[SupportsDLPack]`、`capture` の総称型などは失われる（文字列の注釈は import を伴わないので、組み込みとモジュール内の名前だけ使える）

## 3. 実装中の判断（計画からの逸脱を含む）

- **例外は `create_exception!` のまま**: 計画では pyclass にする予定だった。`NetworkError`（`RuntimeError` 継承）はクラスにできたが、`PauseTimeoutError` は import 時に `tp_basicsize ... (96) is too small for base 'TimeoutError' (128)` で失敗した。pyo3 0.30 が `PyTimeoutError` を `BaseException` のレイアウトで宣言しているため（`OSError` の派生は `PyOSErrorObject` のはず）。pyo3 の hidden macro で正しいレイアウトの基底を作る案は、内部の private な型に触れるので不可。2 つとも元の `create_exception!` に戻し、`mujou-stubgen` が同じ docstring で足す。ずれは `tests/mujou/test_core_stub.py` が検出する
- **生成器は pyo3 の足りない所を補う**: `__new__` の戻りを `typing.Self` に（pyo3 はクラス名を書き、`Node` のサブクラスの生成が `Node` と推論されて pyright が 281 件のエラー。pyo3 側にも `TODO: use typing.Self?` がある）、注釈の無い `*args` / `**kwargs` を `typing.Any` に、`object` のメソッドの override に `@typing.override`（`reportImplicitOverride = true` のため）
- **正確な型が要る所は小さな型で表す**（`crates/mujou-python/src/hint.rs`）: `Message.buffers` は `tuple[BufferView, ...]`（getter は signature を取れない）、`Runtime.capture` の `build` は `Callable[[list[object]], Any]`（pyright strict で `_session.py` の lambda の型が不明になるため）。signature の文字列の注釈は import を伴わないので、組み込みとモジュール内の名前（`"float"`、`"object"`、`"dict[str, int | float]"`）だけに使った
- **`GlobalTimer` は `Timer` の派生として残した**: tests の型注釈（`conftest.py`、`test_port.py` など並列作業の範囲）が `GlobalTimer` を使うので、名前を残すと Python 側を変えずに済む。`timer` はモジュールの const（`#[pymodule_export] const timer: GlobalTimer`）。派生 pyclass には `#[pyclass]` が `IntoPyObject` を実装しないので手で実装し、基底にグローバルタイマーの `Arc` を入れる
- **`Timestamp(*, ns)` はキーワード専用**: 数だけだと秒かナノ秒か読めないため。`repr` は `Timestamp(ns=...)` で、`eval` で戻る。差の演算は入れなかった（`a.ns - b.ns` で足りる、必要になったら足す）
- **`Node` の hook の引数名を `_dt` / `_state` から `dt` / `state` に**: スタブに出る名前が override の名前と違うと pyright が `reportIncompatibleMethodOverride` を出す。型も `"float"` / `"object"` に固定（`f64` の入力型 `SupportsFloat | SupportsIndex` は `float` で override できない）
- **モジュール名を 1 か所にできなかった点**: pyo3 はクラスの `module = ...` に文字列リテラルしか取らず、inline に定義したクラスにしか自動で付けない。名前は `lib.rs` の `#[pymodule(module = "mujou", name = "_core")]`、各クラスの `module = "mujou._core"`（`crates/mujou-python/src/*.rs`）、`pyproject.toml` の `module-name` の 3 か所。`just stub` は `module-name` から拡張モジュールとスタブの場所を決める。`tests/mujou/test_core.py::test_every_class_belongs_to_this_module` がクラスの `__module__` のずれを検出する
- **テストの仮想時計は 0 に戻す**: `Timer::new()` は実時計で始まるので、仮想時計に切り替えるまでの数百 ns が残る。新しいテストのヘルパーは `load_state(0)` してから使う

## 4. 自己レビュー

- diff を通しで読み、「global time」と書いていた doc comment を「ランタイムのタイマー」に直した（`runtime.rs` のモジュール doc、`NodeHooks::update`、`last_update_ns`、復元の排他のコメント）
- LIVE 表のキーは `Arc` のアドレス。数えている間は `Live` が `Arc<Timer>` を持つので、別のタイマーが同じアドレスを再利用することはない（`&'static` のころは leak で同じ保証をしていた）
- 生成物の鮮度チェックが働くことを確認した（doc を直した後に `just stub` を忘れた状態で `just run` を回すと、`mujou-stub` hook が「files were modified」で失敗した）

## 5. 未解決（このタスクの範囲外）

- `mujou-core` の lib テスト `slot::tests::readers_see_whole_messages_in_order` が約 2% で落ちる（メッセージのフィールド数が 1 になる `index out of bounds`、まれに SIGSEGV）。ベースのコミット 0e29e6e でも 600 回中 12 回再現したので、この変更の前からある。単独では落ちず、ほかの slot テストと並行に走るときだけ落ちる。別スロットのメッセージを読んでいるように見え、use-after-free かメモリの別名を疑う。安全でないコードは `Buffer::bytes` と arc-swap の内部だけ。別タスクで調べる

## 6. レビュー対応（code-reviewer / algorithm-reviewer、ともに request-changes。裁定は coordinator）

- **code must-fix（CI で `mujou-stub` hook が動かない）**: pre-commit のワークフローでは `SKIP: mujou-stub` で外し、test のワークフロー（拡張をビルドする方）に `just stub && git diff --exit-code src/mujou/_core.pyi` を足した。ローカルの pre-commit は今どおり。test / type-check の paths に `*.pyi` を足した（nit 9）
- **algorithm must-fix（シングルトンの決定を改めた記録）**: `memory/project_async_node_execution.md`・`project_native_core_rust.md`・`project_open_decisions.md`・`docs/open-decisions.md`・skill `locality-and-async`・CLAUDE.md を「時間軸は実行ごとに 1 本。1 つの実行の全ノード・全スロット・ホストの入力はその実行のタイマーを使う。既定はグローバルタイマー」に書き直した。ユーザーの決定として `memory/project_timer_per_run.md` を、スタブの規約として `memory/project_core_stub.md` を足し、`MEMORY.md` から張った
- **should-fix（スロットとランタイムのタイマーの一致）**: `Outlet::new` / `InputGroup::new` / `Runtime::start`（コアと bindings）の doc comment と `docs/execution-model.md` に契約を書いた。一致の強制は run の API の作り直しで入れる（裁定どおり今回は入れない）
- **algorithm should-fix 2**: `docs/execution-model.md` の停止・再開・例外・状態の木・復元の「グローバルタイマー」を「その実行のタイマー」に直した。`mujou.time.sleep` が延びるのはグローバルタイマーで動くネットワークの停止だけ、と書いた
- **code should-fix 3（ruff が 2 か所）**: `just stub` を Python のレシピにし、`.pre-commit-config.yaml` の `ruff-pre-commit` の `rev` を読んで `uv tool run ruff@<版>` で整形する。版の正は pre-commit の設定 1 か所
- **質問への裁定**: `Timer.now() -> Timestamp` と `Timestamp - Timestamp -> Timestamp`（後の方が前なら `ValueError`）を入れた（コアは `Timer::now` / `Timestamp::checked_sub`）
- **nit**:
  - code 1: `test_core_slot.py` の嘘になったコメントを直した
  - code 2: 生成器が、コンストラクタの無いクラスに `__new__(cls) -> NoReturn` を足す（`GlobalTimer()` などが型の上でも通らない）
  - code 4: values を `Sequence[Any]` にした（`hint.rs` の `Values`）。`SupportsDLPack` はスタブで Protocol を定義しないと書けないので、`Any` の列までに留めた。`Schema(fields)` はそのまま
  - code 5: `just stub` を Python にしたので、空白を含むパスでも壊れない
  - code 6: `Timestamp(ns=-1)` / `2**64` は `ValueError`。順序比較と `__sub__` の引数は、生成器が `object` から `Timestamp` に直す（pyo3 は取り出せない引数に `NotImplemented` を返すので、実際に通るのはそのクラスだけ）
  - code 7 / algorithm nit 1: もう片方のタイマーでランタイムを走らせ、capture の `build` の中でそちらが Δt で更新を続けることを確かめるテストにした（Rust と pytest）
  - code 10: `timer=_core.timer` を明示しても省略と同じになるテストを足した
- **記録だけ（対応しない）**:
  - code 3: `Timer` は `GlobalTimer` のために `subclass` にしたので Python から継承できるが、コアは `Arc<Timer>` を直接読むので override は効かない。private なモジュールなので今は問わない。`mujou._core` パッケージへの移動で公開名と合わせて見直す
  - code 8: `mujou-stubgen` は例外を無条件に足して `incomplete` を false にする。pyo3 が将来 `create_exception!` を introspect すると重複する。ほかの見つからないメンバーは `test_lists_every_name_of_the_module` が検出する。pyo3 を上げるときに見直す
  - algorithm nit 2: `Timestamp` はどのタイマーの時刻かを持たない。docstring に「1 つのタイマーの時刻だけを比べる」と書いてあり、実行内でタイマーが 1 本なら起きない
  - algorithm nit 3: pytest の `test_a_new_timer_starts_at_zero_and_runs` の上限は緩いが、Rust 側が厳密に確かめている
- **ホスト側の作業への申し送り（algorithm should-fix 1 より）**: タイマーはビルドの時点で決め、`Session` が 1 つの `Arc<Timer>` を `InputGroup` / `Outlet` / `Runtime` に渡す。グローバルタイマーを直接読む `_session.py` の `_tree` と `_capture_on_exception`、ノードとホストが時刻を読む・待つ手段もそのタイマーに替える。復元した複数のネットワークを同じタイマーで別々のランタイムとして起動すると、2 つ目で `ValueError` になる
