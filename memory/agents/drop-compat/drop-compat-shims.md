# drop-compat-shims（互換のための仕組みの廃止）

2026-10-19。ブランチ `refactor/2026-10-19/drop-compat-shims`（main 42da4cc から分岐）。ユーザーの指示「破壊的変更で互換のために実装した機能があれば、複雑さを減少させるためにすべて廃止して」（[[feedback-no-compat-before-release]]）。

## 洗い出しの方法

- `src/` `crates/` `tests/` `examples/` を deprecat / compat / legacy / alias / instead / renamed / 互換 / 廃止 / 旧 などで検索した
- 利用者を別の書き方へ導くエラーメッセージ（use / pass / instead）を全部読んだ
- git log の rename / remove / drop / refuse / reject のコミットと、memory・specs の「以前は」「廃止」「改名」の記録をたどった
- `mujou.__init__` と `mujou.typing` の再 export、`justfile`・`.gitignore`・設定ファイル

## 消すもの

| 候補 | 場所 | 理由 |
| --- | --- | --- |
| `period=None` を専用の `TypeError` で弾き `period=0.0` を案内する分岐 | `src/mujou/_core/node.py` の `Node.__init__`、テスト `test_rejects_none_and_points_to_zero`、`specs/nodes.md`、`memory/project_node_period.md` | 2026-10-17 に既定を `None` から 0.01 に変えたときの移行の案内。型は `float` なので pyright が `None` を弾く。実行時に渡しても `math.isfinite` が `TypeError` を出す |
| Python が `period=0.0` を native へ `None` で渡す変換（`checked_period`） | `src/mujou/_core/node.py`、`specs/execution-model.md`、`memory/project_node_period.md` | 当初は内部の対応として残したが、レビュー（must-fix 1）を受けて消した。`None` を廃止したときに native を変えずに済ませた名残。bindings の `period_to_ns` が `0.0` 秒を 1 ns に丸めるので、`0.0` をそのまま渡しても振る舞いは同じ（既存の `test_a_period_of_zero_updates_as_soon_as_the_time_moves` が押さえる） |
| 古い配置（拡張モジュール `mujou._core` を `src/mujou/` に直接ビルドしていたころ）の残骸を消す・無視する設定 | `justfile` の `clean` の `rm -f src/mujou/*.so src/mujou/*.pyd`、`.gitignore` の `src/mujou/*.so` `src/mujou/*.pyd` | 2026-10-16 の改名前の checkout を掃除するためだけのもの。手元の checkout に残骸が無いことを確かめた |

## 残すもの

| 候補 | 理由 |
| --- | --- |
| DLPack の旧形式のカプセル（`"dltensor"`、`export_legacy`） | mujou の破壊的変更の互換ではなく、DLPack の規格の 2 つの形式への対応。`max_version` を渡さない呼び出しはこの形式でやり取りする |
| `Port(send=None)` を実行時に受け付けること | `None` は実装の中で「省いた spec」を表す値そのもの（`_NO_SPEC`）。互換の分岐ではない |
| 予約名 `period` / `device` / `generator` / `timer` の置き換えの検査（`checked_period` の `TypeError`） | 現行の仕様（クラス属性で既定の周期を置く書き方は認めない、ユーザーの決定） |
| `period` に `bool` を渡すと `TypeError` | 不正な入力の検査 |
| MPS などのデバイスを `ValueError` にする検査 | 現行の仕様（CPU と CUDA だけをサポート） |
| memory・specs の「以前は〜だった」の記録 | 決定の経緯。現行の仕様として互換を約束していない |

## 後回しにしたもの

- native の内側の `Option`（`crates/mujou-python/src/runtime.rs` の `periods: Option<Vec<Option<f64>>>` と docstring の「(> 0) ... or `None`」、`crates/mujou-core/src/runtime/mod.rs` の `period_ns: Option<Vec<Option<u64>>>`、`crates/mujou-core/tests/runtime.rs` の `with_period(None, false)`）も同じ経緯の名残。M:N の PR（#74）が同じ箇所を大きく直していて衝突するので、このブランチでは消さず、M:N のブランチで後から消す（orchestrator の裁定）

## レビュー（memory/agents/code-reviewer/drop-compat-shims.md）への対応

- must-fix 1: 採用。上の表のとおり Python 側だけ消した
- nit 1（`period=None` や文字列の `TypeError` が `period` を名指ししない）: 見送り。注釈 `float` で型の上で弾かれ、公開前は互換の検査を足さない方針
- nit 2（DLPack の旧形式は `torch>=2.14.1` との経路では使われない）: 見送り。この作業の範囲外で、記録だけ残す
- nit 3（検証の結果が無い）: 下に書いた
- nit 4（feedback のメモが main にない `on_thread_start` の廃止を前提にしている）: 見送り。経緯の記録として誤りではない

## 検証

- 1 回目（c3fa1c1）: `uv sync` → `just run` が pass。cargo test 全 pass、pytest 557 passed / 18 skipped、pyright 0 errors、`mkdocs build --strict` 成功
- 2 回目（must-fix 1 の対応後）: `just run` が pass。cargo test 全 pass、pytest 557 passed / 18 skipped、pyright 0 errors、`mkdocs build --strict` 成功
