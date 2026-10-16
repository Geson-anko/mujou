# docs-site（第 1 段階）の判断ログ

2026-10-16。ブランチ `docs/2026-10-16/docs-site`（PR #15 の上）。設計仕様の `docs/` を `specs/` に改名し、利用者向けドキュメントの mkdocs の土台を作った。本文は第 2 段階で書く。

## 改名

- `git mv docs specs` のコミットは純粋な改名にした。ただし mdformat の除外設定（`.pre-commit-config.yaml` の `docs/`）を同じコミットで `specs/` に変えないと、pre-commit が specs/ の LaTeX を書き換えてコミットできない。そのため除外の 2 行だけは改名のコミットに含めた。specs/ の中身は 0 行の変更で、rename として検出される
- 参照の書き換えは次のコミットに分けた。対象は CLAUDE.md、memory/ 直下、`.claude/skills/locality-and-async/SKILL.md`、src/ の docstring、crates/ のコメント。crates/ はパスの文字列だけを変えた（並列の Rust の作業と行が重なると merge で衝突しうる）
- `memory/agents/**` は書き換えない。過去の作業の記録で、当時のパスのまま残すほうが履歴として正しい。rename は git が追えるので読み替えられる。並列の worktree も memory/agents に追記するので、衝突を避ける意味もある
- 確認: `grep -rn "docs/" --exclude-dir=.git --exclude-dir=.venv --exclude-dir=target --exclude-dir=agents .` が `uv.lock` 以外で 0 件。`docs.astral.sh` とブランチ名 `docs/<日付>/...` は対象外

## pamiq-core から倣ったもの

- theme は material。features、palette（teal と indigo、ライトとダーク）、markdown_extensions、mkdocstrings の options（google、show_source、separate_signature、merge_init_into_class、spacy、members_order: source など）をほぼそのまま使った
- API リファレンスは 1 ページに `::: mujou.X` を並べる方式。ページの分け方も pamiq-core と同じく機能単位
- ナビゲーションは Home / User Guide / API References の骨格に倣った
- レシピは pamiq-core の Makefile の `docs-serve` / `docs-build` に相当する

## pamiq-core から変えたもの

- i18n: pamiq-core は英語だけ。mujou は mkdocs-static-i18n の folder 構成（`docs/en/`、`docs/ja/`）にした。suffix 構成（`index.ja.md`）より、2 言語が丸ごと並ぶ場合に見通しがよい。`fallback_to_default: true` なので、API リファレンス（英語の docstring から生成）は `docs/en/api/` にだけ置き、日本語版は英語のページを使う。ナビの日本語名は `nav_translations` で付ける
- 目次に Getting Started、Concepts、Examples を足した。pamiq-core は User Guide だけだが、mujou はノード・Port・グローバル時間などの考え方を先に説明しないと使い方が読めない
- サンプル: pamiq-core は `samples/` を置くだけでテストしない。mujou は `examples/` に置き、`tests/test_examples.py` が `examples/*.py` と `examples/*/main.py` をサブプロセスで実行して終了コード 0 を確かめる（slow マーカー、タイムアウト 60 秒）。ページには `pymdownx.snippets`（`check_paths: true`）で取り込むので、サイトのコードとテストするコードが一致する
- 厳密なビルド: pamiq-core は `mkdocs build` のみ。mujou は `just docs-build` で `--strict` にし、`just run` と CI（`.github/workflows/docs.yml`）に入れた。デプロイはしない
- mkdocstrings: `inherited_members: true` と `filters: ["!^_"]` を足した。Node の hook はコアの `_core.Node`（stub）にあり、例外クラスは `__weakref__` などの dunder が出てしまうため
- `mkdocs<2` に上限を付けた。Material が MkDocs 2.0 でプラグインが使えなくなると警告している
- `show_submodules` は false（pamiq-core は true）。公開の名前はすべて `mujou` 直下から import するため

## mdformat

- 既存の mdformat（0.7.18、gfm）は admonition の字下げを code block に変えてしまう。`docs/` はその対象から外し、別のエントリで mdformat 1.0.0 と `mdformat-mkdocs==5.3.0` を使う。0.7.18 に mdformat-mkdocs を足す案は、more-itertools と mdformat-admon の版の食い違いで動かなかった。mdformat-tables は mdformat 1.0.0 と依存が衝突するので外した（mdformat-mkdocs が gfm の表を扱う）

## 目次とページ

- 目次は decisions-2026-10-16（`mujou._core` パッケージ、`h = mujou.run(*networks, timer=None, inputs=..., state=...)` のハンドル、`Compose`、`Timestamp`、自動の状態収集、`on_thread_start` の公開）を前提にした
- 各ページは見出しと「書く内容」の admonition だけ。API リファレンスは今ある公開の名前だけを載せた（まだ無い `Compose` や `Timestamp` を書くと strict ビルドが落ちる）。API が変わったら `docs/en/api/*.md` の `:::` を差し替える
- テスト用の口（仮想時計など）は公開しない決定なので、pamiq-core の Testing ページに当たるものは作らなかった（レビュー対応で変更。下の節）
- 日本語のメモには yomiyasu をかけた。リンターの「英単語の前後の半角空白」の警告は、specs/ と CLAUDE.md の表記に合わせて従わなかった

## 残る確認事項（第 1 版の時点。レビュー対応の節で解消）

- サイトの公開先とデプロイ: 今は公開しない（ユーザーの回答）
- 利用者向けのテストのページ: 作る（ユーザーの回答）
- インストールの手順: PyPI とソースの両方を書く（ユーザーの回答）

## レビュー対応（`memory/agents/code-reviewer/docs-site.md`）

- must-fix 1: ユーザーの回答は「docstring とコードのコメントは英語に統一。API リファレンスは英語で、日本語のサイトでも英語のまま出す。日本語は specs/ とユーザー向けの本文で使う」。CLAUDE.md の記述を残してこの内容にし、`memory/feedback_doc_language.md`（以前の「どちらでもよい」から変わったことを書いた）と `.claude/agents/doc-teacher.md` の「日本語 docstring は 1 文を 1 行」を直した
- should-fix 1: `show_if_no_docstring: true` を足した。Node の hook 8 個が API リファレンスに出る。stub に docstring を書く案は採らなかった（stub は自動生成に切り替える決定があり、並列のブランチも `_core.pyi` を変えている）
- should-fix 2: `docs/en/api/time.md` で 5 つの関数を `members` に並べた。`src/mujou/time.py` の docstring から `specs/execution-model.md` のパスを消した。ビルドした site/ に `specs/` を含むファイルが無いことを確かめた。利用者向けのページと公開の docstring に specs/ のパスを書かない規則を `memory/project_docs_site.md` に書いた
- should-fix 3: README の `just run` の説明に docs-build を足した
- should-fix 4: `tests/test_examples.py` は stderr に `Traceback` が出ても失敗にする。`examples/README.md` に「ネットワークを走らせるプログラムは最後に shutdown() か join() を呼ぶ」を足した
- should-fix 5: 合流は orchestrator が行う。手順: timers などのブランチと合流したら `grep -rn "docs/" --exclude-dir=.git --exclude-dir=.venv --exclude-dir=target --exclude-dir=agents .` を流し直し、設計仕様を指すものを `specs/` に直す
- should-fix 6: `memory/project_docs_site.md` と `memory/project_testing_api.md` を足して MEMORY.md からリンクし、feedback_doc_language の description を更新した
- should-fix 7: skill solo-dev-cycle の段階 5 と、agent code-simplifier のドキュメント同期の対象に `docs/` と `docs/en/api/*.md` の `:::` を足した（edit-dot-claude の手順）
- ユーザーの回答から、User Guide に Testing ページ、API リファレンスに Testing ページの枠（`mujou.testing` はまだ無いので `:::` は書かない）を足した。Installation のメモを PyPI とソースの両方にした
- nit 1: `watch: [src, examples]` を足した。nit 3: `specs/open-decisions.md` の「docs の当初の案」を「仕様書（specs/）の当初の案」にした。nit 4: 失敗時に stdout と stderr の両方を出す
- nit 7: CI はどのジョブも `uv sync` で既定の dev グループを入れる。docs グループに分けると test と type-check のジョブは mkdocs 一式（純 Python で 20 個ほど）を入れずに済むが、`just setup` と `just run` でも docs グループを入れる設定が要る。得るものが小さいので、pamiq-core と同じく dev に置いたままにした
- 記録だけの nit: 2（`site_url` はデプロイを足すときに入れる）、5（サンプルが 0 本の間は `[NOTSET] SKIPPED` と出る）、6（docs の CI も Rust の拡張をビルドする。type-check.yml とそろえた）、8（`\[x\]` は mdformat で `[x]` になる。数式を入れるときに mdformat 側の数式対応が要る）、9（検索のインデックスは en と ja が 1 つ）、10（3.14t の `just docs` で watchdog が GIL を有効にする警告が出る。害はない）
- `src/mujou/time.py` の docstring の 1 行も、並列のブランチが同じ行を変えていれば合流時に衝突する
