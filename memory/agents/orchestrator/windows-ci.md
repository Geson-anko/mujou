# Windows の CI の試行（2026-10-23）

## 計画

- `test.yml` の `rust` と `run_tests` のマトリクスに `windows-latest` を足し、green になるまで直す
- torch 2.14.1 は cp314 / cp314t の win_amd64 wheel がある
- 範囲外: Windows の wheel の公開（`publish.yml`）。CI が通ってから別の PR
- 段階 2 は省略: CI の設定の変更で、CI そのものが検証になる
- 検証はローカルでできないので、draft PR #105 の CI で反復する

## 反復

### 1 回目（4b22cdc）

- rust / windows-latest は通った。pytest は 3.14 / 3.14t とも 885 passed、6 failed
- `tests/test_layers.py`: `read_text()` が cp1252 で UTF-8 を読む → `read_text("utf-8")`
- `tests/test_claude_hooks.py`: `bash` が WSL の `bash.exe` に解決 → Windows では Git Bash（justfile と同じパス）
- Ctrl-C のテスト: Windows の `os.kill(pid, SIGINT)` は TerminateProcess（終了コード 2）→ `signal.raise_signal`
- CI 外の気付き: `examples/visualize.py` と `viz.py` の docstring の `write_text` は `⇄` を cp1252 で書けない → `"utf-8"`

### 2 回目（9610536）

- 残りは `TestCompactionRecovery` の 2 件。Git Bash でも hook の出力が空（fail-open で黙って抜ける）
- hook は `python3` に依存し（settings.json の Python の hook と statusline も）、Windows の Python には `python3` が無い。`.claude/` を native の Windows で動かすのは mujou の Windows 対応の範囲外と判断し、Git Bash への切り替えを戻して Windows では skip
- 却下: hook で `python3 || python` に落とす案（settings.json と statusline にも波及する。要るならユーザーの判断で別タスク）

### 自己レビュー + code-reviewer（approve）

- should-fix 1（hook）: 上の skip で対応
- should-fix 2（`just stub` の shebang）: 3 回目の CI で Windows でも通った。対応不要
- should-fix 3（CONTRIBUTING.md の CI の表）: Windows を追記
- nit: `203d1ff` にスコープが無い。push 済みで force push 禁止のため直さない

### 3 回目（99ce0d2）

- 全ジョブ green。Windows の pytest は 889 passed、stub の照合も通った（CRLF の警告のみ）
