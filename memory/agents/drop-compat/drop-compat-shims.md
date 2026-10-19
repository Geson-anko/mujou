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
| 古い配置（拡張モジュール `mujou._core` を `src/mujou/` に直接ビルドしていたころ）の残骸を消す・無視する設定 | `justfile` の `clean` の `rm -f src/mujou/*.so src/mujou/*.pyd`、`.gitignore` の `src/mujou/*.so` `src/mujou/*.pyd` | 2026-10-16 の改名前の checkout を掃除するためだけのもの。手元の checkout に残骸が無いことを確かめた |

## 残すもの

| 候補 | 理由 |
| --- | --- |
| DLPack の旧形式のカプセル（`"dltensor"`、`export_legacy`） | mujou の破壊的変更の互換ではなく、DLPack の規格の 2 つの形式への対応。`max_version` を渡さない呼び出しはこの形式でやり取りする |
| `Port(send=None)` を実行時に受け付けること | `None` は実装の中で「省いた spec」を表す値そのもの（`_NO_SPEC`）。互換の分岐ではない |
| 予約名 `period` / `device` / `generator` / `timer` の置き換えの検査（`checked_period` の `TypeError`） | 現行の仕様（クラス属性で既定の周期を置く書き方は認めない、ユーザーの決定） |
| `period` に `bool` を渡すと `TypeError` | 不正な入力の検査 |
| MPS などのデバイスを `ValueError` にする検査 | 現行の仕様（CPU と CUDA だけをサポート） |
| Python が `period=0.0` を native へ `None` で渡す変換 | 内部の表現の対応で、利用者向けの互換ではない。native を変えるのはこの作業の範囲外 |
| memory・specs の「以前は〜だった」の記録 | 決定の経緯。現行の仕様として互換を約束していない |

## 検証

- `uv sync` → `just run`
