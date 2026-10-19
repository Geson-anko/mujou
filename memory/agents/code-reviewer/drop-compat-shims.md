# drop-compat-shims レビュー

対象: `git diff origin/main..HEAD`（c3fa1c1 まで 4 コミット、base 42da4cc）。

## verdict: request-changes

消した 2 件は互換のためだけのもので、消しても現行の仕様は壊れない。残した中に、`period=None` を廃止したときの互換の層が 1 つ残っている（must-fix 1）。

## must-fix

### 1. `period=0.0` を native へ `None` で渡す変換は、`period=None` の廃止で残った互換の層で、Rust を変えずに消せる

- 対象: `src/mujou/_core/node.py:513-526`（`checked_period` の `-> float | None` と `return period if period > 0 else None`、docstring の「or `None` for `0.0`」）。根は `crates/mujou-python/src/runtime.rs:455-457,475`（`periods: Option<Vec<Option<f64>>>`）、`crates/mujou-core/src/runtime/mod.rs:93-98,561-574`（`period_ns: Option<Vec<Option<u64>>>`、`None => Ok(1)`）
- 問題: 実装者のメモは「内部の表現の対応で利用者向けの互換ではない。native を変えるのはこの作業の範囲外」として残した。しかし、この変換は 2026-10-17 に Python の `period` を `None`（できるかぎり速く）から `float` に変えたとき、native を変えずに済ませるために入れたもの（`memory/project_node_period.md`「native へは 0.0 を None にして渡す（Rust は変えていない）」）。ユーザーの指示「破壊的変更で互換のために実装した機能があれば、複雑さを減少させるためにすべて廃止」に当たる
- 根拠（Python 側だけで消せる）: bindings の `period_to_ns` は `seconds_to_ns(seconds)?.max(1)`（`runtime.rs:385-387`）で、`seconds_to_ns(0.0)` は `Ok(0)`（`crates/mujou-core/src/time.rs:101-106`）なので、`0.0` をそのまま渡すと 1 ns になる。`None` も core で 1 ns になる（`mod.rs:570`）。挙動は同じなので、`checked_period` は `float` を返すだけでよい。native の内側の `Option`（`Vec<Option<f64>>` / `Vec<Option<u64>>`、`Some(0)` の検査）も同じ経緯の名残で、消すなら `crates/mujou-core/tests/runtime.rs:857` の `with_period(None, false)` も直す。どこまで消すかは orchestrator の裁定に任せる（最小は Python 側の 2 行と docstring、`specs/execution-model.md:249`、`memory/project_node_period.md`）
- 確信度: 範囲に入るかの判断は中。`0.0` を直接渡しても挙動が変わらないことは高（コードで確かめた）
- 付記: `specs/execution-model.md:249` の「native は `None` を『時刻が進みしだい』と扱い、0 を受け付けない」は、bindings が `0.0` 秒を 1 ns に丸めて受け付けるので、今の実装でも不正確（main から）

## should-fix

なし。

## nit

### 1. `period=None` や文字列の `TypeError` のメッセージが `period` を名指ししない

- 対象: `src/mujou/_core/node.py:161-164`
- 問題: `Node(period=None)` は `math.isfinite` の `TypeError: must be real number, not NoneType` になる（実行して確かめた。文字列も同じ）。`bool` の `TypeError` は「the period must be a number of seconds」と名指しするので不揃い。型の上では注釈 `float` で pyright が弾くので、方針（公開前は互換の検査を足さない）に照らせば今のままでよい。docstring の `Raises: TypeError: period is a bool.` も、`None` や文字列の `TypeError` を書いていないが、`math.isfinite` が出すものなので書かなくてよい
- 確信度: 高（事実）／直す必要は低

### 2. DLPack の旧形式は `torch>=2.14.1` の経路では使われない

- 対象: `crates/mujou-python/src/dlpack.rs:30-56,104-112,128-134`、`crates/mujou-python/src/slot.rs:191`
- 問題: 残す判断（mujou の破壊的変更の互換ではなく、DLPack の規格への対応）は妥当。ただ、必須の `torch>=2.14.1` の `torch.from_dlpack` は常に `max_version=(1, 0)` を先に渡し、`torch.Tensor.__dlpack__` も `max_version` を受け付けるので、mujou が torch とやり取りする経路で旧形式は通らない。通るのは torch 以外の producer / consumer（`tests/mujou/_core/test_native_slot.py:165,183`）だけ。今回の指示の範囲外で、複雑さを減らす別の作業の候補として記録する
- 確信度: 低（範囲外という判断に同意。記録のみ）

### 3. 実装者のメモの「検証」に結果が無い

- 対象: `memory/agents/drop-compat/drop-compat-shims.md` 末尾
- 問題: 「`uv sync` → `just run`」だけで pass / fail が無い
- 確信度: 高

### 4. `feedback_no_compat_before_release.md` が main にまだ無い変更を前提にしている

- 対象: `memory/feedback_no_compat_before_release.md:8`
- 問題: 「`on_thread_start` を廃止するとき」の経緯を書くが、main と このブランチには `on_thread_start` がまだある（`src/mujou/_core/_native.pyi:178-202`）。別ブランチの作業。経緯の記録なので誤りではない
- 確信度: 中

## 見た点の結論

- `period=None` の専用の `TypeError`: 2026-10-17 の既定変更の移行の案内だけ。消した後は pyright が `reportArgumentType` で弾き、実行時も `math.isfinite` の `TypeError` になる。仕様（`specs/nodes.md` の表）は壊れない。テスト・docstring・specs・memory の更新は一貫している
- `src/mujou/*.so` / `*.pyd` の clean と `.gitignore`: 2026-10-16 の改名前の配置の残骸のためだけのもの。全 worktree（mujou、-generator、-mn、-mn2、-mn3、-nocompat）に残骸が無いことを確かめた。今の配置（`src/mujou/_core/*.so`）の ignore は残っている
- `Port(send=None)` を実行時に受け付けること: `_NO_SPEC` が実行時に `None` そのもので、分岐を持たない。妥当
- 予約名の検査、`bool` の検査、MPS の拒否: 現行の仕様。妥当
- ほかの互換の仕組み: `src/` `crates/` `tests/` `examples/` `pyproject.toml` `justfile` `.github` `.gitignore` を deprecat / compat / legacy / alias / renamed / instead / 互換 / 旧 / 改名 / 廃止 で検索し、`mujou.__init__` と `mujou.typing` の再 export、rename / drop / reject のコミットを見た。上の must-fix 1 と nit 2 のほかに見つからなかった
- 日本語の文体: diff の specs・memory に翻訳調の目立つ箇所は無い

## 検証結果

- just run: pass（cargo test 全 pass、pytest 557 passed / 18 skipped、pyright 0 errors、mkdocs build --strict 成功）。実行後の作業ツリーは clean
