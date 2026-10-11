# memory/agents/

エージェント（およびソロ開発時のメインエージェント）の中間メモ置き場。`<agent-name>/<task-slug>.md` の形で置く。

| ディレクトリ | 書き手 | 内容 |
| --- | --- | --- |
| `orchestrator/` | メイン | 委譲判断・レビュー裁定・計画外の決定。ソロ開発時の記録もここ（skill `solo-dev-cycle`） |
| `implementation-planner/` | planner | 計画書 |
| `spec-test-author/` | test author | 書いたテスト・期待される失敗・実装側への要求 |
| `plan-implementer/` | implementer | 判断ログ・計画逸脱・IF 変更通知（並列時は `<task>-<instance>.md`） |
| `code-reviewer/` | reviewer | verdict と指摘 |
| `algorithm-reviewer/` | reviewer | verdict・指摘・式の照合メモ |
| `code-simplifier/` | simplifier | 簡素化内容とドキュメント修正点 |
| `doc-teacher/` | teacher | 理解度テストとラウンド記録 |

エージェント固有の長期的な知見（次のタスクでも役立つもの）は `<agent-name>/MEMORY.md` を索引にして 1 知見 1 ファイルで残す。
