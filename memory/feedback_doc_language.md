---
name: feedback-doc-language
description: 文書の言語。仕様書（specs/）・計画書・CLAUDE.md・skill・memory は日本語、ユーザー向けの docs/ の本文は英語と日本語、docstring とコメントは英語、コミットと PR タイトルは英語の規約形式
metadata:
  type: feedback
---

仕様書（`specs/`）・計画書・CLAUDE.md・skill・agent 定義・memory は日本語で書く。ユーザー向けのドキュメント（`docs/`）の本文は英語（`docs/en/`）と日本語（`docs/ja/`）の両方で書く。docstring とコードのコメントは英語に統一する。API リファレンスは docstring から生成した英語で、日本語のサイトでも英語のまま出す。コミットメッセージと PR タイトルは `<種別>(<スコープ>): <内容>` の英語形式。

**Why:** ユーザーとのやり取りが日本語で、参考にした 2 つのリポジトリ（[[user-role]]）も同じ方針。docstring とコメントは、2026-10-16 のユーザーの決定で英語に統一した。それまでは「英語でも日本語でもよいが、ファイル内で混ぜない」だった。API リファレンスを docstring から生成するので、言語を 1 つにそろえる。

**How to apply:** 新しく書く docstring とコメントは英語にする。日本語の docstring やコメントを見つけたら、そのファイルを触るときに英語に直す。日本語を使うのは specs/ とユーザー向けの本文（`docs/ja/`）。関連: [[project-docs-site]]
