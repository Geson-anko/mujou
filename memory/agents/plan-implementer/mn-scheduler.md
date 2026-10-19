# M:N スケジューリング PR 1（コアのワーカープール）の判断ログ

計画書: `memory/agents/implementation-planner/mn-scheduler.md`。この PR は計画の PR 1（コアのワーカープール、Python から見た振る舞いは変えない）だけを扱う。

## 実装したもの

- `crates/mujou-core/src/runtime/pool.rs`: プールのノードの状態（`Phase`・`At`・`Task`）、ワーカーのループ、期限のヒープ、timekeeper
- `node.rs`: 周期の規則を純関数 `next_target_ns` に切り出し、単体テストを足した。`end_step` は `&mut Control` を取る形にして、専用スレッドとプールで共有する
- `mod.rs`: `RuntimeConfig::{workers, dedicated}`、`NodeHooks::enter`、スレッドの局所記憶 `LAST_NODE`、`Shared::{dedicated, sched_changed, sched_epoch}`、`Control::sched`。`invoke` が切り替えのときに `enter(hook)` を同じ `catch_unwind` の中で呼ぶ。制御スレッドの `state_dict` は `Shared::state_dict` から `enter` なしで呼ぶ
- `control.rs`: 起動の完了（`go`）・ステップの許可・終了の許可のあとでプールを評価し直す
- bindings: `dedicated: Some(vec![true; n])`、`workers: None` を渡すだけ（Python の振る舞いは変わらない）

## 計画からの逸脱（理由つき）

1. **sched を別の Mutex にせず、`Control` の中に置いた**（計画は `sched: Mutex<Sched>` と lock の順序 control → sched）。プールのノードの遷移は制御の状態（command・epoch・run_epoch・go・grant・failed・stop_allowed）で決まるので、1 つの lock の下に置くと、2 つの lock の間で状態がずれる場合を考えなくて済む。ワーカーが lock を取るのは「仕事を取る」「戻す」の 2 回で、計画と同じ。専用スレッドの自由実行の速い経路（制御語の atomic を読むだけ）は変えていない。制御の操作（停止・ステップ・終了）は頻度が低いので、ワーカーとの取り合いは問題にならない見込み。将来ネイティブの小さいノードで競合が見えたら、ワーカーごとのキュー（計画の「想定リスク」）と一緒に分ける。条件変数はワーカー用の `sched_changed` を別に持つ（専用スレッドを起こさないため）
2. **ライフサイクルのタスクを制御側から積む代わりに、「評価し直す」形にした**。計画は capture・step・finish などから `Task(Pause)` などを積む設計。実装では、ノードの状態（専用スレッドのループのどこにいるか）だけを持ち、制御の状態が変わったら `wake_pool`（プールの全ノード、O(M)）か `wake_node`（ステップの許可）で評価し直す。何をするかはワーカーがノードを取った時点の制御の状態で決める（専用スレッドがループの区切りで制御語を読むのと同じ意味になる）。ヒープの古い項目は世代番号で無効にして、取り出すときに捨てる。制御の操作ごとに専用の経路を書かずに済み、専用スレッドの規則（停止中の `Stop` なら `on_resume` を呼ばない、正常な終了後の例外でも `exception` → `stop`）をそのまま写せる
3. **ワーカーを 1 本でも立てられなければ、今と同じく起動を諦める（`abandon`）**。計画は「一部しか立たなかったら、立ったワーカーで続ける」。専用スレッドを立てられなかったときと扱いを揃えた。立ったワーカーは、プールのノードを `on_start` → `on_stop` まで進めてから終わる
4. **`enter` が失敗したら、`LAST_NODE` を更新しない**。次にそのスレッドがそのノードの hook を呼ぶとき、もう一度 `enter` を呼ぶ。レビュー対応で `on_exception`・`on_stop` の扱いを変えた（下の「レビュー対応」の 1）
5. **既存のテストは、1 つのテストの中で 3 つの実行の形を順に回す**（`each_mode`: 専用スレッド・プールの既定のワーカー数・ワーカー 1 本）。計画は「2 通り」。ワーカーがノードより少ない場合（デッドロックしないこと）も既存の全テストで確かめるために 1 本の形を足した。失敗したときは `eprintln!` の出力で形がわかる
6. **排他の smoke test は重なりを捉えられない**。ノードは `Mutex<N>` の中にあるので、仮に 2 つのワーカーが同じノードを取っても hook は重ならず、debug ビルドの `debug_assert` が落ちてハングする形で現れる。レビュー対応で `AtomicBool` の検査を外し、テスト名と doc をそれに合わせた。決定的な保証は、`At::Running` への遷移が lock の下でキューから取るときだけ、という構造に置いている
7. **`an_exception_after_a_normal_stop_still_comes_before_stop` は 50 ms 眠る**。片方のノードが正常な終了でループを抜けてから、もう片方の例外が来る順にするため。結果（`exception` → `stop` が 1 回ずつ）は眠る長さに依らない
8. Python のテストは足していない（PR 1 では Python から見た振る舞いが変わらないので、既存の pytest が回帰を見る）

## 次の PR への申し送り

- `PyNode::on_thread` はワーカーの本体も包むので、ワーカーは寿命の間 Python に attach したまま待つ間は detach する（今の専用スレッドと同じ）。PR 2 でプールに載せたときにそのまま使える
- `NodeHooks::enter(next_hook)` の `next_hook` は公開の hook の名前（`on_start`・`on_update`・`on_pause`・`on_resume`・`on_exception`・`on_stop`）。エラーの `HookError::hook` もその名前にすれば、失敗は次の hook の名前で報告される
- main の作業ツリー（ブランチ `docs/2026-10-18/specs-sync`）に、`specs/native-core.md`・`specs/execution-model.md`・`crates/mujou-core/src/runtime/mod.rs` の未コミットの変更がある。この PR と同じファイルなので、merge でぶつかる可能性がある

## レビュー対応（2026-10-18）

記録: `memory/agents/code-reviewer/mn-scheduler-pr1.md`、`memory/agents/algorithm-reviewer/mn-scheduler-pr1.md`（どちらも approve）。orchestrator の裁定に従った。

### 対応したもの

1. `enter` の失敗（code sf1・sf3、algo sf3）: `Shared::invoke_as` を `Entry::{Required, BestEffort, Skip}` で分けた（code nit8 の bool 引数もこれで消えた）。`enter` の失敗（Err でも panic でも）は次の hook の失敗として、コアが次の hook の名前を付けて返す（`HookError::hook` は実装に任せない）。`on_exception`・`on_stop` は `Shared::clean_up` から `BestEffort` で呼び、`enter` が失敗しても本体を呼ぶ。ログは今の hook と同じくノードの側（bindings は捕まえた場所でログを出し、panic は `NodeHooks::panicked`）。専用スレッドで `on_start` の前の `enter` が失敗したときは、`on_start` の失敗と同じ経路で `on_exception` → `on_stop` に進む。trait の doc と `specs/native-core.md` を合わせた
2. テスト（code sf2・sf4、algo sf4）: `a_failed_enter_fails_the_next_hook_and_still_lets_the_node_clean_up`（Err と panic、専用スレッドとワーカー 1 本）、`a_failed_enter_on_a_switch_fails_the_update`（切り替えの `enter` の失敗が `on_update` で報告される）、`another_worker_keeps_the_time_while_one_is_held`（`run_work` の `notify_one` を消すと落ちることを確かめた）、`a_capture_freezes_the_time_before_any_later_update_in_the_pool`（凍結の後に `on_update` が来ないこと、順序、`last_update_ns ≤ T`）。最後のテストは、凍結を待つためにタイマーの `is_paused()` を 1 ms おきに見る。結果は待つ長さに依らない
3. `specs/execution-model.md` に、プールで弱まる約束を書いた（algo sf1: 停止の手順 3 を実装に合わせた、再開後の最初の更新、停止のタイムアウトの名前、例外の手順 3）。algo nit2 の 1 行と code nit5（手動モードのステップの更新も先に取る）は `specs/native-core.md` に入れた
4. 文体（code sf5）: `specs/native-core.md` の節を `memory/feedback_japanese_style.md` に合わせて書き直した。code nit6（リンクの文言）も直した
5. 細かいもの: code nit1（smoke test の名前と doc）、nit2（スレッド名でしか見ない 2 本の doc に限界を書き、ワーカーが立たないことを確かめていなかった方は名前を変えた）、nit7（コメントの折り返し）、nit11（1 ノードの実時間テスト 2 本は専用スレッドとワーカー 1 本の 2 通りに）、algo nit1（過負荷のテストで Δt も確かめる）
6. 計画書をこのブランチにコミットした（code nit10）。PR 3 の項目に、PR 3 へ回した 3 点の測定を書き足した

### 対応しなかったもの

- code nit3（`an_exception_after_a_normal_stop_still_comes_before_stop` の 50 ms の sleep）: ノード 1 がループを抜けたことは hook を呼ばずに起きるので、テストから観測する手段が無い。sleep は順序を起こりやすくするだけで、結果はどちらの順でも同じ。コメントにそう書いた
- code nit9（specs-sync の 0520a5a との衝突）: この PR では触らない。merge のときに 0520a5a の訂正（`state_dict` の doc から "(or has stopped after an exception)" を消す）を採る
- algo sf2・code nit4・algo nit3（timekeeper の起こしすぎ、lock の待ち）: 裁定どおり PR 3 に回した
