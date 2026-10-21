---
name: project-restructure-2026-10-20
description: 2026-10-20〜21 の作り直し（ブランチ refactor/2026-10-20/restructure）で、確認事項 R1〜R8 を推奨案で進めた。ユーザーの確認はまだ取っていない。設計の正本は specs/architecture.md と specs/scenarios.md
metadata:
  type: project
---

2026-10-20、ユーザーの基準（ストレートな木構造、ユーザー目線、切り分けの明快さ、後方互換なし、ガードは提供シナリオの中だけ、good taste）で mujou を作り直した。設計と理由の正本は `specs/architecture.md`（層、境界、内部の名前の規約）と `specs/scenarios.md`（提供シナリオとガードの台帳）で、ほかの specs もその形に書き直した。

次の 8 つは、以前のユーザーの決定を覆すか CLAUDE.md を変えるので確認が要る。**推奨案で実装まで進めた。ユーザーの確認待ち**（作業の最後にまとめて聞く）。断られたときの代案も記す。

| # | 推奨案（実装済み） | 覆すもの | 断られた場合 |
| --- | --- | --- | --- |
| R1 | コアの carry を廃止し、PC は `y` / `mask` / `time` を `Activity` のフィールドで運ぶ | 2026-10-20 の carry の決定（[[project-carry]]） | Port の引数を分け、空の Carry で送信を 1 本にする |
| R2 | `PrevPort` / `NextPort` を `Port` / `MultiPort` に置き換え、向きは属性名の prev / next で表す | 向きの名前の指示（[[feedback-direction-names]]） | — |
| R3 | `mujou.Node` を純 Python にし、Rust は `Driver` を hook 1 回につき 1 回呼ぶ。CLAUDE.md のツーリングの一文を書き換えた | 「`mujou.Node` はコアの `Node` を継承」（[[project-torch-module]]） | `_native.Node` を残し、private hook を 1 つの `_dispatch` にまとめる |
| R4 | `mujou.run` の root を 1 つにし、`Group` の位置引数を廃止する | [[project-run-node-and-write-time]]、[[project-group-and-typing]] | — |
| R5 | 捨てた Run を救う監視 daemon スレッドを消し、atexit だけにする | — | スレッドを残し、理由を `specs/execution-model.md` に書く |
| R6 | `mujou.time` のモジュール関数 7 個を廃止し、ホストは `run.timer` / `global_timer` を使う | [[project-timer-pause]] の `mujou.time.pause()` など | — |
| R7 | 属性の登録を代入時の `__setattr__` から、run の段 1 でのインスタンスの `__dict__` の走査に変える | [[project-port-registry]] | `__setattr__` を残し、K6 を Node の Port にも広げる |
| R8 | 復元した実行は、保存した時刻から始まる専用のタイマーで走らせ、`timer=` との同時指定を断る。CLAUDE.md の不変の原則に一文を足した | 共有のタイマーへの時刻の読み込み（[[project-timer-per-run]]） | 向きを問わず読み込みを拒み、検査・読み込み・計上を `_native.start` の中で 1 つの lock の下で行う |

**Why:** 1 つの事実を 2 か所で持つこと（レジストリ、終了の状態、時刻源）と、特別扱いの状態（build 済み、凍結、巻き戻し）を消すと、上の形に行き着いた。

**How to apply:** ユーザーの確認が取れたら、この表の行を消し、各メモリの「確認待ち」の注記を外す。断られた項目は代案で実装し直す。作り直しの前に推奨案で実装した確認待ち（GIL ありのワーカー数、周期のシグネチャ、復元した後の周期の目標、`send_to` だけのポートの読み出し）は `specs/open-decisions.md` の「ユーザーの確認待ち」にあり、R1〜R8 と一緒に聞く。GPU のマシンでの確認（TestCuda、CUDA のノードのコスト、GPU のノードの capture）と GIL ありのビルドの CI も残っている（`specs/open-decisions.md`）。
