# docs-infra: mkdocs の設定の不具合

## 変えたもの

- `mkdocs.yml`
  - `toc.slugify` を `pymdownx.slugs.slugify`（`case: lower`）にした。日本語の見出しの id が `_1` などから `ノードの大きさ` のような Unicode のままになる
  - mermaid の custom fence に `format: !!python/name:pymdownx.superfences.fence_code_format` を明示した（Material の推奨どおり）
- `.pre-commit-config.yaml`: `check-yaml` に `args: [--unsafe]`（Material の推奨）。`yaml.safe_load` は `!!python/...` タグで失敗するため
- `README.md`: Status を「API はあるが変わりうる、`just docs` で読める、`examples/` はテストが走らせる」に。PC-ALM は未実装なので「will be ... not implemented yet」に

## 確かめたこと

- 英語のページの見出し id は slugify の変更前後で 1 つも変わらない（全 `index.html` の `id=` を diff）
- 日本語のページの TOC は `href="#ノードの大きさ"`、見出しは `id="ノードの大きさ"`
- mermaid は `<pre class="mermaid"><code>` で出力される（Material が描画する形）。mermaid 本体は Material の bundle が unpkg から読む
- mdformat（1.0.0 + mdformat-mkdocs 5.3.0、docs 用 hook と同じ版）はリンクの非 ASCII のアンカーを `#%E3%83%8E...` に percent-encode する
  - 生成 HTML の href も encode されたまま。ブラウザは HTML 仕様（indicated part の探索で fragment を percent-decode して id と照合）により `id="ノード..."` に飛べる
  - Material の TOC 追従（`navigation.tracking`）は `decodeURIComponent(a.hash)` で id を探すので、id は生の Unicode でなければならない。`pymdownx.slugs` の `percent_encode: true` は Material の追従を壊すので採らない

## 採らなかったもの: `validation.links.anchors: warn`

MkDocs 1.6 のアンカー検証はリンクの fragment を decode せずに id と文字列比較する。mdformat が encode した日本語のアンカーはすべて「存在しない」と誤検出され、strict ビルドが落ちる。設定ファイルだけでは直せないので入れなかった（既定の `info` のまま）。

mkdocs の hook（`hooks:`）で、各ページの `present_anchor_ids` に `quote(id)` を足せば、encode したリンクも生のリンクも正しく検証でき、本当に切れたアンカーだけが WARNING になることを docs のコピーで確かめた。担当範囲外のファイル（例: リポジトリ直下の `hooks/anchors.py`）が要るので、orchestrator の判断待ち。

```python
from urllib.parse import quote


def on_page_content(html, page, **kwargs):
    if page.present_anchor_ids is not None:
        page.present_anchor_ids |= {quote(a) for a in page.present_anchor_ids}
    return html
```

```yaml
hooks:
  - hooks/anchors.py
validation:
  links:
    anchors: warn
```

2026-10-17 時点の docs には見出しへのアンカー付きリンクが 1 つもなく、hook + warn の状態でも切れたアンカーは 0 件。

## そのまま（記録だけ）

- MathJax は jsdelivr の CDN（`mathjax@3.2.2`）から読む。オフラインでは数式が描画されない

## 追記: hook を採用（orchestrator の判断）

- `hooks/anchors.py`（リポジトリ直下）を足し、`mkdocs.yml` に `hooks: [hooks/anchors.py]` と `validation.links.anchors: warn` を入れた
- 一時的に `docs/ja/zz-anchor-probe.md` を置いて確かめ、削除済み: 生の正しいリンク・encode した正しいリンク（同じページ内のものも含む）は通り、切れたリンクは生も encode も WARNING になって strict が落ちる
- ツールでの扱い（設定は変えていない）
  - pre-commit: ruff / ruff-format / pyupgrade / check-docstring-first などが `hooks/` にもかかる。すべて Passed
  - pyright: `include` がないのでプロジェクト全体が basic で検査される（strict は `src/` だけ）。`uv run pyright` は 0 errors
  - coverage: `source = ["src/mujou"]` なので対象外。pytest の `testpaths = "tests/"` にも入らない
