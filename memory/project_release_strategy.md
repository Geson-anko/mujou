---
name: project-release-strategy
description: リリースは release/X.Y ブランチのタグ vX.Y.Z、Trusted Publishing で Test PyPI → PyPI → GitHub Release。hotfix は release から切って release → main。リリースノートは CHANGELOG.md から自動生成。最初の版は 0.1.0（2026-10-23）
metadata:
  type: project
---

2026-10-23 のユーザーの指示（参考: MLShukai/vrcpilot）。最初のリリースは 0.1.0。

- `main` は開発。マイナー系列ごとに `release/X.Y` を main から切り、消さない
- リリースは `release/X.Y` 上のタグ `vX.Y.Z` の push。`.github/workflows/publish.yml` がタグと `Cargo.toml` の版・ブランチ・CHANGELOG の節を検査し、sdist と wheel（3.14 / 3.14t × Linux x86_64・aarch64、macOS arm64）を作って quickstart を走らせ、Test PyPI → PyPI（Trusted Publishing、environment `testpypi` / `pypi`）→ GitHub Release
- hotfix は `release/X.Y` から `fix/...` を切って release へ PR、版を上げてタグ、その後 **release → main を merge**（vrcpilot の cherry-pick ではない）
- リリースノートは `CHANGELOG.md`（Keep a Changelog）の `## [X.Y.Z]` 節を抜き出す。docs のサイトの Changelog のページも同じファイルを snippet で取り込む（`<!-- --8<-- [start:releases] -->` の区間）
- 版の正本は `Cargo.toml` の `[workspace.package].version`（maturin が Python のパッケージに渡す）
- Windows の wheel は CI で試していないので出さない（sdist からのビルドになる）
- 手順の正本は `CONTRIBUTING.md` の Releases

**Why:** 0.x の間も破壊的変更が続くので、main の変更がパッチに混ざらないよう release ブランチで系列を分ける。

**How to apply:** 利用者に見える変更は `CHANGELOG.md` の Unreleased に書く。リリース作業は CONTRIBUTING.md の手順に従う。公開後の互換方針は [[feedback-no-compat-before-release]]。
