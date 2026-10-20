# carry（付随データのゼロコピー中継）レビュー

> 2026-10-20 code-reviewer。対象: `git diff feature/2026-10-20/node-register-state...feature/2026-10-20/carry`（4 commit）。計画 `memory/agents/implementation-planner/carry.md`、実装メモ `memory/agents/plan-implementer/carry.md`。

## verdict: approve

must-fix はなし。決定済みの 3 点（`send_carry=` / `recv_carry=`、木の `"carries"`、ホストは ValueError）どおりに実装されている。Rust の unsafe・Arc・プールの寿命、mutex の区間、bindings の GIL / free-threaded の扱いに問題は見つからなかった。復元のオラクル（止めずに回した実行との一致。変異で効くことも確認済み）とスナップショットの一貫性の並行テストもある。

## must-fix

なし

## should-fix

1. `tests/mujou/_core/test_port.py:106`（`passed.send(..., carry=tagged(0))  # pyright: ignore[reportArgumentType]`）、`:144`（`NextPort(send=h(), send_carry=received)  # pyright: ignore[reportArgumentType]`）、`:935`（`path.next.send(x, carry=tagged(1))  # pyright: ignore[reportArgumentType]`）
   - 問題: どれも pyright はエラーを出さないので、ignore が不要。`carry_types()` にある `# Only for the type checker` の行は、静的に弾ける前提で書かれているように読めるが、実際には弾けない。`Port.send(msg, carry: CS | None = None)` は、中継する Port に `carry` を渡しても、作る Port で `carry` を省いても、型検査を通る。`send_carry=` に `recv_carry` の無い Port を渡しても通る（CR=Never から CS=Never と解かれる）
   - 根拠: test_port.py の先頭に `# pyright: reportUnnecessaryTypeIgnoreComment=true` を足したコピーにかけると、上の 3 行が「不要な ignore」として出る。ほかに `:116` / `:118` / `:127` / `:129` の `reportCallIssue` と `reportUnknownVariableType` も不要と出た
   - 直し方の方向: ignore を外して「実行時だけ TypeError」と明記するか、overload で静的にも弾く（計画は実行時だけの想定）
   - 確信度: 高 / 深刻度: 低（テストが誤った契約を示唆する）
2. `tests/mujou/_core/test_port.py:913-925`（`test_shares_the_carried_buffers_along_the_path`）
   - 問題: `run._session._slots` という private な内部を読んでゼロコピーを確かめている。ゼロコピーは Rust の `relay_shares_the_buffers` / `relays_along_a_path_allocate_no_carry` と `test_native_slot.py::TestCarry::test_relay_shares_the_buffers` が既に確かめているので、二重の検証でもある
   - 根拠: skill `testing-strategy`「公開インターフェースと観測可能な振る舞いをテストする。内部実装の詳細は見ない」「3 層で分担し、同じことを二重に検証しない」。計画のテスト観点（「Session の内部（`_slots`）から」）が指示したものなので、計画の段階で既に方針から外れていた
   - 確信度: 高 / 深刻度: 低

## nit

1. `crates/mujou-core/src/slot.rs`（`check_relay`）: 付随データの無いスキーマへ空の relay を渡したときの文言が `"expected to carry nothing, got nothing"` で、何がいけないのか読み取れない。「the slot carries nothing to pass on」のような文言がよい（確信度: 高）
2. `specs/messages.md`（バッファプールの段落）の「中継する `Outlet` は、付随データのバッファを最初のゼロの 1 つしか確保しない」: 復元（`Slot.restore` は全フィールドを自分のプールから埋める）の後は、中継する Outlet のプールにも付随データのバッファが確保される。「復元するまでは」などの限定が要る（確信度: 中）
3. `examples/carry.py:88-101`: `# --8<-- [start:host]` の節を、どの docs も取り込んでいない。ID で自分のサンプルを待つホストの書き方は docs に出ない（取り込むか、印を外す）（確信度: 高）
4. `src/mujou/_core/state.py`（`_slot_restores`）: `else ("(value, time_ns, sent, carried)")` の括弧が余計（確信度: 高）
5. `crates/mujou-core/src/slot.rs:331`: `Buffer::field` を `pub(crate)` から `pub` に広げたが、crate の外からの呼び出しは無い（`Message::values()` で `Buffer` が外へ出るので、意図して公開したなら問題ない）（確信度: 低）
6. 新しいテストの例外メッセージの照合は、`^...$` の完全一致が多い（`test_rejects_passing_on_the_carry_of_another_node`、`test_the_host_cannot_{write,read}_carried_data`、Rust の `mismatched_relay_publishes_nothing` の `assert_eq!`）。skill は「完全一致は書かない」。ただし既存のテストにも `^` で固定した照合がある（確信度: 中）
7. `Port` の docstring と `docs/*/building-a-network.md` は「最初の `recv` の前はゼロ」と書いているが、復元した実行では木の `"carries"` の値になる。state.md に書いてあるので、ここでは書かなくてもよい（確信度: 低）
8. 中継する Port の `send_to` を、Python の Port の層では試していない（native では `test_relay_to_one_channel` が確かめている）（確信度: 低）

## 検証結果

- just run: pass（643 passed, 21 skipped、pyright 0 errors、mkdocs build --strict 通過）。実行したのは `/data/geson/mujou`（carry ブランチ）。終わった後も作業ツリーは clean
- 追加で確かめたこと: test_port.py のコピーに `reportUnnecessaryTypeIgnoreComment=true` を付けて pyright にかけた（should-fix 1）
