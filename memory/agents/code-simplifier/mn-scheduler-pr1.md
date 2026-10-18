# code-simplifier: M:N スケジューリング PR 1（ワーカープール）の仕上げ

対象は `git diff origin/main..HEAD`（ブランチ `feature/2026-10-18/mn-scheduler`）。大きな書き換えはしていない。

## 再レビューの nit（orchestrator の裁定どおり）

- nit 1（`248c8cb`）: hook が失敗したプールのノードは、`on_exception` の仕事を ready キューの先頭に積む。`run_work` が `advance` → `place(i, next, first = !ok)` と進み、`place` は `first` なら `push_front` する。これで `specs/execution-model.md` の例外の手順 3「例外を投げたノードも捕まえた直後に呼ぶ」をプールでも保つ。同じ手順の追記は「その時点でワーカーを待っていたほかのノードの分より先に呼ぶ」に直し、`specs/native-core.md` の取る順の項と `pool.rs` のモジュール doc にも書き足した。テスト `the_failed_node_calls_exception_before_the_nodes_its_failure_woke`（ワーカー 1 本・3 ノード・手動モードで、exception の順が `[0, 1, 2]`）を足した。`!ok` を `false` に戻すと `[1, 2, 0]` で落ちることを確かめた
- nit 2（`106b1b6`）: `a_failed_enter_on_a_switch_fails_the_update` でノード 0 のログを `Enter` 込みで丸ごと比べる。`Enter("on_update")` の後に `Enter("on_exception")`・`Enter("on_stop")` が来ることを断定している
- nit 3（`11e1572`）: 再開の追記を「その場合 $\Delta t$ には、…を待つ時間が入る」に直した

## 簡素化（`1141612`）

- `Shared::invoke`（`invoke_as(.., Entry::Required, ..)` を呼ぶだけの pass-through、呼び出し元は `call` だけ）を消し、`invoke_as` を `invoke` に改名した。どの呼び出しも `Entry` を明示するので、入り方が呼び出し側で読める
- `pool.rs` の `record`: `Start`・`Update`・`Pause`・`Resume` の 4 か所に重複していた「失敗したら `leave_loop`」を、先頭の 1 か所にまとめた（`Start` の `started = true` は成否によらず先に立てる）。`Exception`・`Stop` は常に `ok` なので振る舞いは変わらない
- `wake_pool` は `dedicated` の判定を `wake_node` に任せる
- テスト: `failure` と `pool` のモジュールに同じ意味の `assert_exception_then_stop_once` が 2 つあったので、厳しい方（最後の 2 つが `[Exception, Stop]`）をトップレベルに 1 つだけ置いた

## 触らなかったもの

- `call` と `state_dict` の `map_err(fail)` の重複は 2 か所だけなので残した
- 自由実行の更新の取り出し（`run_worker`）と手動モードの更新（`advance`）は、時刻を読む場所の保証が違うので分けたまま
- `memory/agents/` の過去のノートにある `invoke_as` の名前は、当時の記録なので直していない

## 検証

`just run` green（cargo test 34 + 47 + 37、pytest 558 passed / 18 skipped、pyright 0 errors、mkdocs build --strict）。
