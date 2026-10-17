"""MkDocs hook: accept percent-encoded links to headings with non-ASCII ids.

mdformat percent-encodes link destinations ("#%E3%83%8E..."), and MkDocs compares
the anchors of links with heading ids without decoding them.
"""

from typing import Any
from urllib.parse import quote

from mkdocs.structure.pages import Page


def on_page_content(html: str, page: Page, **kwargs: Any) -> str:
    # Browsers decode the fragment before matching it with an id, so the encoded
    # form of every id is a valid anchor too.
    if page.present_anchor_ids is not None:
        page.present_anchor_ids |= {quote(a) for a in page.present_anchor_ids}
    return html
