# simplify-d: 本体以外のコードの簡素化（2026-10-19）

ブランチ `refactor/2026-10-19/simplify-d`。担当は examples/、benchmarks/、tests/test_examples.py、tests/test_claude_hooks.py、.claude/hooks/、.claude/scripts/、hooks/、justfile、.github/workflows/、.pre-commit-config.yaml。

## 結果

12 ファイル、+59 / -130（純減 71 行）。`uv sync` のあと `just run` は green（595 passed, 20 skipped、pyright 0 errors、`mkdocs build --strict` 成功）。

| commit                                                    | 変更                                                                                                                                                                                          | 行数      |
| --------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------- |
| `chore(pre-commit)`                                       | codespell の `--ignore-words-list` から、コメントで「examples で使う」とされていた 3 文字の語を外す。今はリポジトリのどのファイルにも現れない（`git grep -iw` で 0 件）                                                                       | +2 / -3   |
| `refactor(just)`                                          | `test-fast` が `cargo test --workspace` を重複して書いていたのを、依存 `test-rs` にする。実行するコマンドは同じ（`just --dry-run test-fast` で確認）                                          | +1 / -2   |
| `chore(ci)`                                               | docs / test / type-check の 3 workflow で push と pull_request に同じ `paths` を 12 行ずつ書いていたのを YAML anchor（`&paths` / `*paths`）で共有する。pre-commit workflow の `actions/cache` を消す。`pre-commit/action@v3.0.1` が同じ `~/.cache/pre-commit` を自分でキャッシュする | +7 / -48  |
| `refactor(examples)`                                      | branching.py で期待値 `2 * x + (-x) ** 2` を 2 回書いていたのを `want` 1 つにする。この部分はどのページの snippet 区間にも入っていない                                                       | +2 / -2   |
| `refactor(benchmarks)`                                    | `describe()` で続けて 2 回あった `if sys.platform == "darwin"` を 1 つにまとめる。`mixed()` の入れ子関数 `host_sleeps` をやめ、`mujou.time.sleep` を直接渡す（コメントは残す）                | +4 / -6   |
| `refactor(claude)`                                        | shell の hook 3 本が同じ session_id 抽出の python を、2 本が同じ additionalContext 出力の python を持っていたのを `.claude/hooks/_common.sh` の関数 `read_session_id` / `print_prompt_context` にまとめる | +43 / -69 |

`_common.sh` は source されるだけで hook として登録しない（settings.json は無変更）。source に失敗したら `exit 0` で fail-open。lib が無い場所に hook だけを置いて実行しても exit 0 になることを確かめた。依存は python3 と bash のまま。

## 整理しなかったもの

- examples: 各ファイルは利用者が単体で読む教材で、`Clock`（pause / subscribe / timers）や学習ループの分割（torch_node / online_learning）は似ていても、それぞれの説明に合わせて違う。共通化すると 1 ファイルで読めなくなるので残した。quickstart.py などの `wait_for` の直後の `assert` は述語と同じ内容で冗長だが、quickstart.py はページにファイル全体が載るので変えなかった。testing_a_node.py の `u` の重複も、ページに載る 2 つの区間がそれぞれ単独で読めるように残した
- justfile の `set dotenv-load := true`: 使っている箇所は見つからない（`.env` は gitignore 済み）が、各開発者の `.env` を読む振る舞いを変えるので残した
- justfile の `clean`: CONTRIBUTING.md に載っているので残した
- docs.yml の deploy 条件（`env.DEPLOY` と `deploy_docs.if` に同じ式）: job の outputs で繋ぐと行は減らず読みにくくなるので残した
- statusline.sh / get-session-id.sh / pretooluse・posttooluse の python hook / hooks/anchors.py / tests/test_examples.py / tests/test_claude_hooks.py: すでに短く、防御的な分岐は fail-open の規約のためなので残した

## 申し送り

- YAML anchor は GitHub Actions が 2025-09 から受け付ける。ローカルでは PyYAML で push と pull_request の `paths` が一致することだけを確かめた。CI 上の実行は PR で確認すること
- benchmarks/mn_scheduler.py は private な `mujou._core.hooks.enter_node`、`Node._on_enter`、`Node._before_hook` を使う。Python パッケージを簡素化する担当がこれらの名前を変えると、pyright（`just type` は benchmarks/ も basic で見る）で失敗する。合流時に確認すること
