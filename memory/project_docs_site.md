---
name: project-docs-site
description: specs/ は開発者向けの設計仕様（日本語）、docs/ はユーザー向けの mkdocs のサイト（英語と日本語、tutorial / topics / api）。サンプルは examples/ に置き snippet で取り込む（テストでは実行しない）。GitHub Pages に mike で版ごとに公開。Installation は PyPI とソースの両方を書く
metadata:
  type: project
---

2026-10-16 のユーザーの指示と回答。

- 設計仕様（日本語）は `specs/` に置く。今の設計とその理由を書き、日付・経緯・エラー文言は書かない（履歴は git）。知識の置き場所の表は `specs/README.md`
- ユーザー向けのドキュメントは `docs/` の mkdocs のサイト。構成は pamiq-core に倣う（material、mkdocstrings）。英語と日本語の 2 言語で、mkdocs-static-i18n の folder 構成（`docs/en/`、`docs/ja/`）を同じページ構成で書く。ページは `index`、`install`、`concepts`（注意書きの正本「ノードが守る規則」の表 `#rules` を持つ）、`tutorial/`（6 ページ）、`topics/`（12 ページ。測った結果の記録は `topics/pc-benchmarks.md` に分け、`pc.md` は値の決め方だけ。2026-11-23）、`examples`。API リファレンスは英語の docstring から生成し、`docs/en/api/` にだけ置く（日本語のサイトは英語のページにフォールバックする）
- サンプルは `examples/` に置き、ページには snippets（`--8<-- "examples/<file>.py"`）で取り込む。一部だけ取り込むときは、行番号ではなく名前付きの区間（`# --8<-- [start:<name>]` / `[end:<name>]` と `"examples/<file>.py:<name>"`）を使う（2026-10-17）。1 つの区間はサイト全体で 1 回だけ取り込み、同じページでファイル全体と区間を両方取り込まない。どこからも取り込まれない区間の印は消す（2026-11-23）。テストでは実行しない（[[feedback-examples-are-guides]]）。規則は `examples/README.md`
- ほかのページからリンクされうる見出しには、en と ja で同じ ASCII の明示 id `{#id}` を付け、番号を付けない。ページ間のリンクは相対パス＋明示 id。API へのリンクは mkdocs-autorefs（`[`Node`][mujou.Node]`、`[mujou.Run.wait_for][]`）で書き、`api/*.md#...` への相対リンクは使わない（例外は index のカードの API のページへのリンク）。ja のページからも同じ書き方で ja の API ページに解決する（`resolve_closest`）。事実の置き場所と長さは [[feedback-docs-one-fact-one-place]]（2026-11-23）
- `just docs` でローカルに立ち上げ、`just docs-build`（strict）は `just run` と CI に入っている
- サイトは GitHub Pages（<https://geson-anko.github.io/mujou/>）に公開する（2026-10-18、ユーザーの決定）。ドキュメントはバージョンごとに分ける（ユーザーの指示、MLShukai/ResoniteIO に倣う）。mike で gh-pages ブランチに版を積み、`.github/workflows/docs.yml` がそのブランチを `actions/deploy-pages` で配信する。main への push で `dev`、安定版のタグ `v*` で `X.Y.Z` と別名 `latest`（ルートの既定）、プレリリース（`a`/`b`/`rc`）は `X.Y.Z` だけ。最初の安定版までルートは `dev` を指す。手動のデプロイは workflow_dispatch
- Installation のページには、PyPI（`uv add mujou` / `pip install mujou`）とソースからのビルドの両方を書く
- Changelog のページ（`changelog.md`）はリポジトリ直下の `CHANGELOG.md` を snippet で取り込む（2026-10-23、[[project-release-strategy]]）

**Why:** 開発者向けの設計判断と利用者向けの使い方は読者が違う。サンプルを snippet で取り込めば、ページのコードと手で実行できるプログラムが同じものになり、`just type` が API の変更による壊れを捕まえる。

**How to apply:** 公開 API を変えたら、`docs/en/api/*.md` の `:::` とガイドのページも直す（skill solo-dev-cycle の段階 5）。利用者向けのページに specs/ のパスを出さない（公開の docstring にも書かない）。言語の規則は [[feedback-doc-language]]、手動の実行（シミュレーションとテスト）の API は [[project-manual-run-simulation]]。
