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
4. **`enter` が失敗したら、`LAST_NODE` を更新しない**。次にそのスレッドがそのノードの hook を呼ぶとき（`on_exception`・`on_stop`）、もう一度 `enter` を呼ぶ。設定が済んでいないまま hook を呼ばないため。PR 2 で `_on_enter` の失敗を扱うときに見直してよい
5. **既存のテストは、1 つのテストの中で 3 つの実行の形を順に回す**（`each_mode`: 専用スレッド・プールの既定のワーカー数・ワーカー 1 本）。計画は「2 通り」。ワーカーがノードより少ない場合（デッドロックしないこと）も既存の全テストで確かめるために 1 本の形を足した。失敗したときは `eprintln!` の出力で形がわかる
6. **排他の smoke test は弱い**。ノードは `Mutex<N>` の中にあるので、仮に 2 つのワーカーが同じノードを取っても hook は重ならない（debug ビルドでは `try_lock` の失敗で `debug_assert` が落ちる）。テストの `AtomicBool` は、その保険が外れたときに捉えるためのもの。決定的な保証は、`At::Running` への遷移が lock の下でキューから取るときだけ、という構造に置いている
7. **`an_exception_after_a_normal_stop_still_comes_before_stop` は 50 ms 眠る**。片方のノードが正常な終了でループを抜けてから、もう片方の例外が来る順にするため。結果（`exception` → `stop` が 1 回ずつ）は眠る長さに依らない
8. Python のテストは足していない（PR 1 では Python から見た振る舞いが変わらないので、既存の pytest が回帰を見る）

## 次の PR への申し送り

- `PyNode::on_thread` はワーカーの本体も包むので、ワーカーは寿命の間 Python に attach したまま待つ間は detach する（今の専用スレッドと同じ）。PR 2 でプールに載せたときにそのまま使える
- `NodeHooks::enter(next_hook)` の `next_hook` は公開の hook の名前（`on_start`・`on_update`・`on_pause`・`on_resume`・`on_exception`・`on_stop`）。エラーの `HookError::hook` もその名前にすれば、失敗は次の hook の名前で報告される
- main の作業ツリー（ブランチ `docs/2026-10-18/specs-sync`）に、`specs/native-core.md`・`specs/execution-model.md`・`crates/mujou-core/src/runtime/mod.rs` の未コミットの変更がある。この PR と同じファイルなので、merge でぶつかる可能性がある
