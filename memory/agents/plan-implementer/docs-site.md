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
- テスト用の口（仮想時計など）は公開しない決定なので、pamiq-core の Testing ページに当たるものは作らなかった
- 日本語のメモには yomiyasu をかけた。リンターの「英単語の前後の半角空白」の警告は、specs/ と CLAUDE.md の表記に合わせて従わなかった

## 残る確認事項

- サイトの公開先（GitHub Pages など）とデプロイの workflow をいつ足すか
- 利用者向けのテストのページ（仮想時計で決定的にノードを試す方法）を作るか。テスト用の口を見せない決定との兼ね合い
- インストールの手順。PyPI に出すまではソースからのビルド（Rust のツールチェーンが要る）になる
