# native-trims: テストだけが使っていた native API の削除

構造監査の指摘を受け、オーナーの「使っていない機能は消す」方針で、フレームワーク（`src/`）が使っていない native の API を削った。各項目は削る前に grep で使われ方を確かめた。

## 削ったもの

| 項目 | 削る前の利用者 | 対応 |
| --- | --- | --- |
| `Runtime.is_running`（binding）と core の `Runtime::is_running` | `tests/mujou/_core/test_native_runtime.py` だけ | 削除。テストは「走っている」を `runtime.join(timeout=0) is False`、「起動していない」を `join` が `RuntimeError("not started")` を出すことで確かめる |
| core の `Runtime::names()` | なし | 削除 |
| `Runtime.capture(timeout=...)` と core `capture` の `Option<Duration>` | `Session` は渡さない。テスト 1 件だけ | 削除。取得は常に `start(pause_timeout=...)` の値で待つ（`mujou.run(pause_timeout=...)` が `Run.state_dict` を決める振る舞いは変わらない）。`Controller::capture` も引数を取らず `self.pause_timeout` を使う |
| `Message.time_ns`（binding） | `session.py` の 2 か所とテスト | `Message.timestamp.ns` に置き換えて削除。core の `Message::time_ns` は core 内で使うので残す |
| `Slot.wait_newer(last=None)` の `None` | `src/` は常に `Message` を渡す | binding も core も `last` を必須にした。`None` のときの即時返却の分岐とテスト 1 件が消えた |
| `Outlet` / `InputGroup` / `Runtime` の `timer=None` → グローバルタイマー | `src/` は常に `timer=` を渡す | 必須の引数にし、`timer_or_global` を削除。テストは `mujou._core._native.timer` を `global_timer` として明示的に渡す。「省略するとグローバル」を確かめていたテストは削除し、「明示したグローバルタイマー」のテストを残した |
| core の `Timer::sleep_ns` | `crates/mujou-core/tests/time.rs` だけ | テストファイルのヘルパー `sleep_ns(&Timer, u64)` に移した |
| `SlotError::Mismatch` の doc の `"field count"` | コードは `"shape"` と `"dtype"` しか作らない | doc を直した |
| Python 側の周期の丸め（`node.period_ns`） | `session.py` | `Runtime.start(periods=...)` に秒のまま渡し、binding がコアの `seconds_to_ns` で丸めて `max(1)` する。`node.period_ns` は `checked_period` に改名し、予約名 `period` の検査だけを残した（検査の場所は変えていない） |
| dev 依存の `pytest-mock` | `mocker` の利用なし | `uv remove --dev pytest-mock` |

コミット: `refactor(native): remove API surface that only tests use` / `refactor(native): pass node periods to the runtime in seconds` / `chore(deps): remove the unused pytest-mock`。

## 残したもの

- `Timer.pause` / `resume` / `is_paused`: オーナーの判断待ちなので触っていない
- dev 依存の `ruff`: `just stub` と pre-commit は pre-commit 側の版を `uv tool run ruff@<rev>` で使い、レシピ・hook・CI から `uv run ruff` は呼ばれていない。エディタ連携で使われている可能性があるので削らず、ここに記録するだけにした
- `.claude/skills/testing-strategy/SKILL.md` は「モックが要るなら `pytest-mock`（`mocker`）」と書いている。依存を消したので、使うときに `uv add --dev pytest-mock` で戻す必要がある。`.claude/` の編集は今回の範囲外なので、記述の更新は別タスクにする
