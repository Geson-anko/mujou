---
name: feedback-doc-language
description: 文書の言語。仕様書・計画書・CLAUDE.md・skill・memory は日本語、コミットと PR タイトルは英語の規約形式
metadata:
  type: feedback
---

仕様書・計画書・CLAUDE.md・skill・agent 定義・memory は日本語で書く。コミットメッセージと PR タイトルは `<種別>(<スコープ>): <内容>` の英語形式。

**Why:** ユーザーとのやり取りが日本語で、参考にした 2 つのリポジトリ（[[user-role]]）も同じ方針。

**How to apply:** コード中の識別子・コメント・docstring は英語でも日本語でもよいが、ファイル内で混ぜない。
