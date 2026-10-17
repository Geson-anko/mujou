# docs-site（第 1 段階: specs への改名と mkdocs の土台）レビュー

対象: `git diff feature/2026-10-16/host-lifecycle..HEAD`（ブランチ `docs/2026-10-16/docs-site`、6 commit）

## verdict: request-changes

must-fix は 1 件で、規約どうしの矛盾をユーザーに確認すれば解消する。改名の漏れ、ビルド、テスト、CI の構成に、マージを止める問題はなかった。

## must-fix

1. `CLAUDE.md:13` / 「docstring は英語」が既存の feedback memory と矛盾する
   - 問題: `memory/feedback_doc_language.md:12` には「コード中の識別子・コメント・docstring は英語でも日本語でもよいが、ファイル内で混ぜない」とある。これはユーザーのフィードバックとして記録された規約で、今回の CLAUDE.md の追記がそれを黙って上書きしている。`.claude/agents/doc-teacher.md:62` にも日本語 docstring を前提にした書き方の規則が残っている
   - 根拠: CLAUDE.md の「メモリ参照」では、新しい規約は memory に 1 ファイルで足して `MEMORY.md` からリンクすることになっている。ユーザーの指示（API リファレンスを docstring から生成し、英語と日本語の 2 言語で出す）から「公開 API の docstring は英語」を導くのは妥当だが、確定させる前にユーザーへの確認が要る。確認が取れたら feedback memory と doc-teacher.md も合わせて直す。取れなければ CLAUDE.md から外す
   - 確信度: 高（矛盾は確実。どちらに寄せるかはユーザーが決める）

## should-fix

1. `mkdocs.yml:113`（`inherited_members: true`）/ Node の hook が API リファレンスに出ない
   - 問題: 判断ログ 25 行目は「Node の hook はコアの `_core.Node`（stub）にあるので inherited_members を足した」としている。しかし生成された `site/api/node/index.html` のメンバーは `device` / `generator` / `connect` だけで、`on_start` / `on_update` / `on_pause` / `on_resume` / `on_exception` / `on_stop` / `state_dict` / `load_state_dict` が出ない
   - 根拠: `src/mujou/_core.pyi:203-213` の hook には docstring がなく、mkdocstrings の既定 `show_if_no_docstring: false` で隠れる。scratch で同じ設定に `show_if_no_docstring: true` を足してビルドすると 8 個の hook がすべて出ることを確認した。stub に docstring を書くか、オプションを足すか、どちらかが要る。今の設定は、判断ログに書いた狙いを果たしていない
   - 確信度: 高（出ないことは確認済み。原因も再現で確認した）
2. `docs/en/api/time.md` / `mujou.time` のページに関数が 1 つも出ない。開発者向けの仕様書のパスが利用者の目に入る
   - 問題: `src/mujou/time.py` は `time = _timer.time` のような代入で関数を公開しているため、ページにはモジュールの docstring しか出ない。その docstring にある `` ``specs/execution-model.md`` `` が、利用者向けのサイトにそのまま表示される（`site/api/time/index.html`。公開 API のページで specs/ を含むのはこの 1 ページだけ）
   - 根拠: CLAUDE.md の新しい「ドキュメント」節は「API リファレンスは docstring から生成する」としているが、このページは time の API をリファレンスとして示せていない。本文は第 2 段階の範囲としても、`:::` の書き方（関数ごとに並べる、など）は土台の側で決めておく事柄
   - 確信度: 中
3. `README.md:19` / `just run     # format -> test -> type` が古い
   - 問題: justfile の `run` は `format test type docs-build` になり、CLAUDE.md の表も直してある。README のこの行だけが直っていない
   - 根拠: `justfile:36`、`CLAUDE.md` のコマンド表
   - 確信度: 高
4. `tests/test_examples.py:22-29` と `examples/README.md` / ノードの中の例外を見逃すことがある
   - 問題: 判定には終了コードしか使っていない。別スレッドで起きた例外は、プログラムが join しない限り終了コードに出ない（scratch で `threading.Thread` の中で raise するだけのプログラムを流すと PASS した）。mujou ではノードの例外は `Network.shutdown()` / `join()` / `read()` などを呼んだときに `NetworkError` として表に出る（`src/mujou/_network.py:430,450`）。そのため、ネットワークを走らせたまま最後に `shutdown()` も `join()` も呼ばずに終わるサンプルでは、ノードが落ちてもテストが通る可能性がある
   - 根拠: 依頼の観点「失敗を確実に検出するか」。`examples/README.md` で「最後に `shutdown()` か `join()` を呼んで、ノードの例外を表に出す」をサンプルの決まりにするか、stderr に `Traceback` が出たら失敗とするか、どちらかを入れておくのが安い。終了コードが 0 でないとき（SystemExit）と timeout（`TimeoutExpired`）で失敗になることは scratch で確認した
   - 確信度: 中（スレッドの例外が終了コードに出ないことは確認済み。mujou のノード例外がプロセスの終了時にどう扱われるか、つまり atexit の有無やスレッドが daemon かどうかは確かめていない）
5. 並列ブランチとの合流で `docs/` を指す参照が戻ってくる
   - 問題: `feature/2026-10-16/timers` は `docs/execution-model.md` と `docs/native-core.md` を編集し、`crates/mujou-core/src/runtime.rs` と `time.rs` に `` `docs/execution-model.md` `` を含む新しいコメントを足している。どちらが先にマージされても、後から入る側で参照が古くなる。ファイルの rename は git が追えるが、コメントの中の文字列は追えない
   - 根拠: `git diff $(git merge-base HEAD feature/2026-10-16/timers)..feature/2026-10-16/timers` に `docs/*.md` を指す追加行が 4 件ある。どちらかを合流したら、判断ログ 10 行目の grep をもう一度流す手順を PR か orchestrator の TODO に残しておく
   - 確信度: 高
6. 新しい決定が `memory/` に記録されていない
   - 問題: specs/ と docs/ の分け方、利用者向けの文書を 2 言語で書くこと、サンプルをテストすることは、プロジェクトの規約に当たる決定。それなのに記録は `memory/agents/plan-implementer/docs-site.md`（作業ログ）と CLAUDE.md にしかない。`memory/feedback_doc_language.md` の description（`MEMORY.md:33` も同じ文）にも docs/ の 2 言語が入っていない
   - 根拠: CLAUDE.md の「メモリ参照」（新しい規約や決定は 1 メモリ 1 ファイルで足し、`MEMORY.md` からリンクする）
   - 確信度: 中
7. `.claude/skills/solo-dev-cycle/SKILL.md:70,83`、`.claude/agents/code-simplifier.md:17,38` / ドキュメント段階の対象に `docs/` が入っていない
   - 問題: CLAUDE.md は「`docs/` は実装済みの公開 API の使い方を書く」としたが、ドキュメントの段階で同期する対象は今も docstring / README / CLAUDE.md / memory のまま。公開 API を変えたときに `docs/en/api/*.md` の `:::` やガイドを直す流れがない。名前が消えた場合は strict ビルドで気づけるが、名前が増えた場合と説明が古くなった場合は気づけない
   - 根拠: 判断ログ 36 行目「API が変わったら `docs/en/api/*.md` の `:::` を差し替える」。この手順がどの skill にも書かれていない
   - 確信度: 中（本文を書く第 2 段階で足す判断もありうる）

## nit

1. `mkdocs.yml` / `just docs` のライブリロードが `src/` と `examples/` を見ていない。serve のログは `Watching paths for changes: 'docs', 'mkdocs.yml'` で、docstring やサンプルを変えても再ビルドされない。`watch: [src, examples]` を足す案がある。確信度: 高
2. `mkdocs.yml` / `site_url` がない（pamiq-core にはある）。デプロイしない今は害がないが、sitemap と言語切替の絶対 URL はこれに依存する。デプロイを足すときに忘れやすい。確信度: 中
3. `specs/open-decisions.md:9` / 「docs の当初の案から変えた」の「docs」は設計仕様のことを指している。改名後は「specs」と読み替えないと、利用者向けの docs/ と取り違えやすい。改名のコミットを純粋に保つ方針は理解できるので、参照を書き換えるコミットに入れる案。確信度: 低
4. `tests/test_examples.py:29` / 失敗時のメッセージが stderr だけで stdout が出ない。サンプルが print で状態を出す場合は原因を追いにくい。確信度: 中
5. `tests/test_examples.py` / サンプルが 0 本の今は `test_program_exits_successfully[NOTSET] SKIPPED` と表示される（空のパラメータ集合）。害はないが、第 2 段階で最初のサンプルを足すまで「実行された」ように見えない点に注意。確信度: 高
6. `.github/workflows/docs.yml:47` / `uv sync` が Rust の拡張までビルドする。mkdocstrings は `paths: [src]` の静的解析で `_core.pyi` を読むので、拡張のビルドは要らない可能性がある（`--no-install-project` などで CI を短くできるかもしれない）。type-check.yml と揃えた判断なら現状のままでよい。確信度: 低（拡張なしでビルドが通るかは試していない）
7. `pyproject.toml:34-37` / mkdocs 系を `dev` グループに入れたので、test / type-check の CI（4 + 1 ジョブ）も mkdocs 一式をインストールする。`docs` グループに分ける案がある。pamiq-core と同じ構成なら今のままでよい。確信度: 低
8. `.pre-commit-config.yaml:80-89` / mdformat 1.0.0 と mdformat-mkdocs 5.3.0 は、admonition・content tabs・`--8<--`（行単独、code fence の中）・`::: mujou.X` とその options を壊さず、冪等だった（scratch で確認）。ただし `\[x\]` は `[x]` に変わる。PC-ALM の式を載せるために後で arithmatex を入れるなら、そのときに mdformat 側の数式対応が要る。確信度: 中
9. 検索のインデックスは en と ja が 1 つにまとまっていて（`search_index.json` の `lang: [en, ja]`、111 件）、検索結果に両方の言語が混ざる。mkdocs-static-i18n の既定の挙動で、今は対処不要。確信度: 中
10. `just docs` を 3.14t で起動すると、`_watchdog_fsevents` の読み込みで GIL が有効になるという RuntimeWarning が出る（macOS）。serve のプロセスに限った話で、害はない。情報として残す。確信度: 高

## 確認して問題がなかった点（観点ごとの結論）

- 改名の漏れ: `git grep -nI "docs/"` で、`memory/agents/**`、`uv.lock`、`docs.astral.sh`、新しい docs/ そのものを指す意図的な参照を除くと 0 件。`.claude/`、hooks、tests、`.devcontainer`、workflow にも残っていない。`execution-model.md` などのファイル名だけで参照している箇所もない。改名のコミット（67cdbc9）は specs/ の中身が 0 行の変更で、rename として検出される
- `.claude/` の変更は `locality-and-async/SKILL.md` のパスの書き換え 4 箇所だけで、既存の規約との矛盾はない。CLAUDE.md の矛盾は must-fix 1
- i18n: `/` と `/ja/` が両方 200。ja のページは `<html lang="ja">` で、言語切替のリンクは対応するページ同士を指す。`docs/ja/` に無い `api/*` 6 ページは英語版にフォールバックし、ナビは `nav_translations` で日本語になる（26 要素）
- pamiq-core との対応: theme、palette、markdown_extensions、mkdocstrings の options、機能単位の API ページ、`docs-serve` / `docs-build` に当たるレシピは、判断ログのとおり対応している。違いは判断ログに理由付きで書かれている
- `just docs *args` は `[positional-arguments]` と `"$@"` で引数を渡せる（`-a 127.0.0.1:8765` で起動を確認）
- CI の `docs.yml` は既存の `type-check.yml` と同じ手順（setup-uv 3.14t、setup-just、`uv sync`）。paths に `src/**/*.pyi` も入っている

## 検証結果

- `just docs-build`: pass（strict、1.9 秒）
- `just run`: pass（pre-commit は `mdformat (docs)` を含む全フックが Passed、cargo test と pytest は 314 passed / 14 skipped（CUDA と test_examples の空集合）、pyright は 0 errors）
- `just docs -a 127.0.0.1:8765`: 起動後に `/`、`/ja/`、`/api/node/`、`/ja/api/node/`、`/ja/user-guide/writing-a-node/`、`/getting-started/quickstart/` が 200、存在しないパスが 404。取得後に停止した
- `tests/test_examples.py` の判定を scratch で模擬した（TIMEOUT_SECONDS=2）: 正常終了は PASS、`SystemExit("boom")` は AssertionError で失敗、`sleep(5)` は TimeoutExpired で失敗、スレッド内の例外は PASS（should-fix 4）
- 成果物の汚染（`</content>` など）: なし
