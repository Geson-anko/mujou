# specs と memory を実装に合わせる（2026-10-18）

監査（specs 12 本と実装の突き合わせ）の結果のうち、M:N と generator を除いたものを直した。M:N と generator は別の PR（feature/2026-10-18/mn-scheduler、#72）で実装と一緒に直す。

- 分担: sync-a（native-core・overview・README・python-environment）、sync-b（execution-model・nodes）、sync-c（messages・network）、sync-d（torch・update-rules・pc-alm-continuous-time・open-decisions）。memory 7 件と crates/mujou-core/src/runtime/mod.rs のコメントは orchestrator
- レビュー（code-reviewer、approve。記録は memory/agents/code-reviewer/specs-sync.md）の裁定:
    - should-fix 1・2・3・5・6 は採用（秒と ns の記述の残り、memory の残り 3 件と GIL、PauseTimeout で on_pause の途中だったノード（node.rs の pause_node で確認）、リンク先 `#実装-1`）
    - should-fix 4 は却下: open-decisions.md:9 から名前の「確認待ち」が消えたのは、memory/project_open_decisions.md の「名前とホスト入出力は決定済み」と整合する
    - nit 1・2・3・4・7・8・9・10・12 は採用。nit 5・6・11 は見送り（事実の誤りではない）
