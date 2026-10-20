---
name: feedback-examples-are-guides
description: examples/ は利用者への手引きであって動作確認のためのものではない。テストで全部を実行しない（2026-10-20 のユーザーの指摘）
metadata:
  type: feedback
---

2026-10-20、MNIST の example の置き場所を相談したとき、ユーザーが「そもそも example をフルテストで回すのが良くない。example は動作確認のためのものではなく、ユーザーにガイドとして参考になる情報を渡すためのもの」と指摘した。

**Why:** テストで回すために「60 秒以内・ネットワーク不要」の制約を課すと、MNIST のような実用的な手引きが書けなくなる。動作の保証は tests/ の役目。

**How to apply:** example は読者にとって参考になることを優先して書く（データのダウンロードや数分の実行も可）。`tests/test_examples.py` の削除と `examples/README.md`・CLAUDE.md・[[project-docs-site]] の修正は `specs/pc-module.md` の実装の順番 0 で行う。
