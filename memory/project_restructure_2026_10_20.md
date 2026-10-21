---
name: project-restructure-2026-10-20
description: 2026-10-20〜21 の作り直し（ブランチ refactor/2026-10-20/restructure）の決定。R1〜R8 と最終レビューでの変更は 2026-10-21 にユーザーが承認した。テンソルの容器は register_state で状態にし、pc.Chain は layers を公開する。設計の正本は specs/architecture.md と specs/scenarios.md
metadata:
  type: project
---

2026-10-20、ユーザーの基準（ストレートな木構造、ユーザー目線、切り分けの明快さ、後方互換なし、ガードは提供シナリオの中だけ、good taste）で mujou を作り直した。設計と理由の正本は `specs/architecture.md`（層、境界、内部の名前の規約）と `specs/scenarios.md`（提供シナリオとガードの台帳）。

2026-10-21、ユーザーが次をすべて承認した（「それ以外は OK」）。

| # | 決定 | 覆したもの |
| --- | --- | --- |
| R1 | コアの carry を廃止し、PC は `y` / `mask` / `time` を `Activity` のフィールドで運ぶ | carry の決定（[[project-carry]]） |
| R2 | `PrevPort` / `NextPort` を `Port` / `MultiPort` にし、向きは属性名の prev / next で表す | [[feedback-direction-names]] のクラス名 |
| R3 | `mujou.Node` を純 Python にし、Rust は `Driver` を hook 1 回につき 1 回呼ぶ | 「`mujou.Node` はコアの `Node` を継承」（[[project-torch-module]]） |
| R4 | `mujou.run` の root は 1 つ、`Group` は名前付きの引数だけ | [[project-run-node-and-write-time]]、[[project-group-and-typing]] |
| R5 | 捨てた Run を救う監視スレッドを消し、atexit だけにする | — |
| R6 | `mujou.time` のモジュール関数を廃止し、ホストは `run.timer` / `global_timer` を使う | [[project-timer-pause]] の `mujou.time.pause()` など |
| R7 | 属性の登録は、run の段 1 でのインスタンスの `__dict__` の走査で行う | [[project-port-registry]] の代入時の登録 |
| R8 | 復元した実行は、保存した時刻から始まる専用のタイマーで走り、`timer=` との同時指定を断る | 共有のタイマーへの時刻の読み込み（[[project-timer-per-run]]） |
| — | PC のノードの引数 `act` を `act_spec` に改名、`learn_interval` は必須、旧版 DLPack の経路は削除 | 設計書 D38・D39 の名前と既定 |

同じ日のユーザーの決定:

- **テンソルの容器（リプレイバッファのような list や dict）は、それだけでは状態に入らない。`register_state` で状態にする。** ガード（台帳の K6 の拡張）は足さない。`Node` と `register_state` の docstring、`docs/*/concepts.md` に書いてある
- **`pc.Chain.layers`** を公開する（`tuple[Layer, ...]`、`layers[i]` が子 `layer{i}`）。動的な属性 `layer0` には pyright が型を付けられないため

**Why:** 1 つの事実を 2 か所で持つこと（レジストリ、終了の状態、時刻源）と、特別扱いの状態（build 済み、凍結、巻き戻し）を消すと、上の形に行き着いた。

**How to apply:** 上の決定を前提に作業する。作り直しの前に推奨案で入れた別件の確認待ち（GIL ありのワーカー数、周期のシグネチャ、復元した後の周期の目標、`send_to` だけのポートの読み出し）は `specs/open-decisions.md` に残っている。GPU での確認（CUDA のテスト、R10・R12・R13 の計測）と GIL ありの通常ビルドでの全テストは、2026-10-21 に RTX 4090 / 4060 の環境で済ませた（結果は `specs/research/benchmarks.md`）。
