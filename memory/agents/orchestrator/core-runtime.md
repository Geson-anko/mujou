# コアのランタイム（`mujou.run(network)` まで）

2026-10-15、agent-team-startup。ユーザーの指示: PR を stack、並列化を最大化、コメントは doc-teacher-student、日本語は yomiyasu。疑問は書き溜めて最後に提示し、推奨で進める。

計画書: `memory/agents/implementation-planner/core-runtime.md`

## PR の stack

1. `chore/2026-10-15/core-scaffold`（main から）: torch の追加（Linux は CPU 版 index）、並列作業の衝突を避ける骨組み、計画書
2. `feature/2026-10-15/core-slots`（WP-A）
3. `feature/2026-10-15/core-runtime`（WP-B）
4. `feature/2026-10-15/network-structure`（WP-C）
5. `feature/2026-10-15/run-network`（WP-D）

A/B/C は scaffold から別 worktree で並列に実装し、stack にするときは前の PR のブランチを merge する（rebase・force push はしない）。

## 裁定

- 計画書の確認事項 1〜12 はすべて推奨案で進める（ユーザーへの質問として最後に提示）

## ユーザーへの質問（書き溜め）

計画書「確認事項」の 1〜12 に加えて:

- Q-a: docs は `net.run()`、指示は `mujou.run(network)`。両方を用意し、`mujou.run` は未 build なら build してネットワークを返す形にした
- Q-b: PC-ALM（`mujou.pc`）は「`mujou.run` まで」の範囲外として今回入れていない
- Q-c: タスク実行クラス（ワーカープール）はネイティブノードが無いので future work にした。今は全ノードが専用スレッド。→ 2026-10-18 に実装（Python のノードも既定でワーカープールに載せる、`memory/project_mn_scheduler.md`）
- Q-d: pre-commit のフック環境が Python 3.12 で作られており、`debug-statements` が 3.13 以降の構文（型パラメータの既定値）を parse できない。WP-C は旧式の `TypeVar(default=...)` で回避した。`.pre-commit-config.yaml` で `python3.14` を指定するかどうか
- Q-e: torch が import 時に「NumPy を初期化できない」UserWarning を出す。numpy を依存に足すか、警告を無視する設定にするか（今は放置）

## 進行

- WP-C 完了（7d132c4）。code-reviewer 起動
- WP-A 完了（aa38fe9）。code-reviewer・algorithm-reviewer 起動。懸念: GIL ありビルド未検証（CI で確認）、SIGNAL_CHECK_INTERVAL の重複は D でまとめる
- WP-A algorithm-reviewer: approve。D への申し送り: docs/network.md 160 行の「一瞬も存在しない」を「どの瞬間に見ても揃う。順に読む間に次の write が終わることはあり、時刻で判別できる」に弱める／Port に wait_newer を出さない（subscribe だけ）／recv が必ず clone することを Port でテスト／InputGroup はネットワークに 1 つ／同時 write の値を同じ時刻で読むテストは手動モードで
- WP-C code-reviewer: request-changes（M1 graph() 関数ほか）→ impl-structure に差し戻し
- WP-C 修正（68ca639）を確認、code-simplifier 起動
- WP-B 完了（36f1ad2, 03e4c26）。code-reviewer・algorithm-reviewer 起動
- Q-f: start 前の shutdown/check は何もしない、正常終了後の step/capture は RuntimeError、という Runtime の振る舞いでよいか（推奨で実装済み）
- Q-g: 正常終了の処理にタイムアウトが無い（止まらないノードがあると shutdown が戻らず Ctrl-C だけ）
- WP-C simplifier 完了（7e98617）。WP-C は完了
- 方針: doc-teacher-student は D の後に、公開 API の docstring（_port/_network/_spec/_node）・runtime.rs のモジュール doc・D で更新した docs を対象に、ファイルごとに並列で回す（D で recv/send の docstring が変わるため）。stack の最後に docs PR を積む
- WP-A 修正（d967e20）完了、simplifier 起動
- WP-B algorithm-reviewer: request-changes。must-fix「凍結の理由をランタイムごとに数えるがタイマーは bool」→ 裁定: 案 (a) タイマー側に停止の保持数（トークン）を持たせ、全部解けたときだけ再開。should-fix 1〜3 も対応
- Q-h: 複数のネットワークを同時に走らせたとき、1 つの停止（状態の取得）で全ネットワークの時間が止まる（グローバル時間は 1 つ）。この意味でよいか
- WP-A 完了、PR #8（base: scaffold）。WP-B に WP-A を merge（0c58c5f）、stub の import を整理（8b906ea）
- WP-B code-reviewer: request-changes。M1 hook ごとに PyThreadState が作り直される → スレッドの寿命の間 attach を保つ。N2 control Mutex の fast path、N7 も対応。impl-runtime に差し戻し
- WP-C に WP-B（現時点）を merge（81d29b3）。WP-D を run-network worktree で先行着手（impl-run）。WP-B の修正後に B → C → D と merge し直す
- reviewer のノート（memory/agents/{code-reviewer,algorithm-reviewer}/）はメインの checkout に未コミット。最後に docs PR か D に含める
- WP-B 修正（60068e8, 694d9c9）+ orchestrator が pause の docstring（7c585fb）。再レビュー 2 本起動。測定: タイマーの time_ns の Mutex が 600〜800 万回/秒で頭打ち（open-decisions の既存項目）
- WP-B 再監査: approve。should-fix「timer_state 付き start が他のネットワークの時刻を書き換える」→ 裁定: 案 (a) 他のランタイムが動いている間の timer_state 付き start をエラーに。docs nit 2 件は impl-run に送付
- Q-i: 状態の木からの復元は、他のネットワークが走っていないときだけにした（グローバル時間を書き換えるため）。複数ネットワークを同時に走らせる使い方を想定するか
- WP-D 完了（76c58f2, 35bf08c, fad3871, 745a40f）。code-reviewer・algorithm-reviewer 起動
- Q-j: subscribe は呼んだ時点より後に書かれた値だけを返す（最初に今の値を返す案もある）
- Q-k: read は秒（float）、recv_with_time は ns（int）で非対称
- Q-l: `import mujou` で mujou.time も import する（計画外の追加）
- Q-m: CUDA のノードのテストは CUDA の無い環境なので skip（未確認）
- WP-B 再レビュー: approve。残り（復元と同時実行の排他、hook を引く位置、nit）を impl-runtime に依頼
- WP-D algorithm-reviewer: request-changes。must-fix 復元待ちの write の時刻 → 木の T で restore。should-fix: recv もスレッド照合（＋ノードの hook からホスト IF を呼ばせない）、state_dict/load_state_dict の CUDA ストリーム同期、復元 E2E を (b,a) 順でも。code-reviewer の結果と合わせて差し戻す
- WP-B r2 対応完了（0bdc2b9, 87b08d6）。simplifier B 起動。87b08d6 を C に、C を D に merge（衝突なし）
- WP-D code-reviewer: request-changes（M1 stub の __new__、M2 WeakSet で atexit が効かない → 強参照、M3 CUDA の同期）。algorithm と合わせて impl-run に差し戻し。裁定: recv も照合、ノードの hook からホスト IF を呼ばせない
- WP-B simplifier（03b2bca）。PR #9（base core-slots）。WP-C に最終 B を merge、PR #10（base core-runtime）。D には 03b2bca を impl-run の修正後に merge する
- WP-D 修正（84a9225, 021100e, 054d485）。最新 C を D に merge、just run 279 passed。再レビュー 2 本起動
- Q-n: on_update / state_dict を _on_thread_start でインスタンス属性として包む（CUDA の同期とホスト IF 呼び出しの検出のため）。threading._register_atexit（private API）を使う
- WP-D 再監査: request-changes（must-fix CUDA の取得の前後の順序付け。state_dict をノードのストリームの下で呼び、_tree の最後で各ノードのストリームを同期する案を採る）。nit 1（run と write の同時呼び出し）は lock で、nit 2 は current_stream の同期に
- WP-D 再レビュー: request-changes（cuda の set_device、取得のコピー後の同期）。裁定: インスタンス属性で hook を包む方式をやめ、コアに内部 hook `_after_update` / `_capture_state` を足す（D のブランチに別コミット）。往復 2 回目（reference.md の上限）。次で収束しなければユーザーに報告
- 仕様の判断: ノードの hook からはどのネットワークのホスト IF も呼べない（橋渡しはホストのスレッドか I/O ノードで）
- docs PR の計画: D から docs/2026-10-16/core-runtime-docs を切り、teacher を並列に 4 体（T1 _port+_spec の docstring、T2 _network+_node の docstring、T3 docs/network.md、T4 docs/execution-model.md+torch.md+nodes.md）。最後に日本語の docs（B/D で変えた docs/*.md、memory の変更）に yomiyasu。PR 説明文も yomiyasu
- WP-D 修正 r2（a37a88d, 229d8d8, 065459c, 032751f）→ 3 回目レビュー request-changes（起動前のストリーム同期、_run_lock の範囲、同期デバイスの絞り込み）。往復上限を超えたが小さく明確なので差し戻し、確認は orchestrator が diff で行う
- Q-o: CUDA の経路（デバイスの添字、ステップごとの同期、取得・復元・起動の前後の同期）は CUDA の無い環境で未実測。CUDA の CI か実機での確認が要る
- WP-D r3 対応（c66ada1, 4d115a1, 163239f）を orchestrator が diff で確認し approve。simplifier D 起動
- simplifier D（d564e7c）。PR #11（base network-structure）
- docs PR: worktree core-docs（docs/2026-10-16/core-runtime-docs、D から）。reviewer のノートを移してコミット（c73d19f）、メインの未コミットのコピーは削除。doc-teacher 4 体（port-spec, network-node, network-md, exec-md）を並列起動
- doc-teacher: port-spec 完了（12/12、保留 3/3、fe86874）。network-node 完了（13/13、保留 3/3、未コミット）
- Q-p: `run` / `mujou.run` のシグネチャにテスト用の `_manual` 引数が見えている。テスト用の別の入口（例 `mujou.testing`）に移すか
- Q-q: `Network.exception_state_dict()` は hook の中から呼んでも RuntimeError にならない（読むだけなのでそのまま）
- doc-teacher: exec-md 完了（R1 12/12 問題不良 4 を除外、保留 3/3、未コミット）
- doc-teacher: network-md 完了（R1 10/11 → R2 11/11、保留 3/3）。network-node（fc98d43）、exec-md（f9c2c65）をコミット
- yomiyasu: exec+nodes、torch+open-decisions+memory、network.md、PR #7〜#11 の説明文の 4 体を並列起動
- yomiyasu 完了（docs 3 体、PR 説明文 1 体）。docs のコミット a1bf990, 4ebc9f8。just run 288 passed / 2 skipped（CUDA）
- docs PR: docs/2026-10-16/core-runtime-docs（base run-network）
- 2026-10-16 ユーザーの指示で std-00（RTX 4090 + 4060、torch 2.14.1+cu130）で CUDA を検証。Q-m・Q-o は解決: 両ビルドで全件通過、同期を外した陰性対照で失敗を確認。CUDA テストを cuda-tests ブランチ（fb02edf）に追加し、docs/torch.md の注記を更新。std-00 には rustup（~/.cargo）と /data/geson/mujou-cuda-check を置いた
