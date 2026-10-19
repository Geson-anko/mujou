# M:N スケジューリング PR 3（ベンチマークと既定値の確認）レビュー

対象: `git log --no-merges 2ee224f..HEAD` の 8 コミット（fc417dc, 84a3102, 08c2d78, 54e2400, b1dca76, d5177c7, b249f7c, 76d6b2b）と、c198478 の衝突の解消（`src/mujou/_core/node.py` の docstring、`specs/execution-model.md` の周期の段落）。

## verdict: approve

must-fix は無い。timekeeper の起こし方を変えても、更新が誰にも拾われずに残る経路は見つからなかった。ただし遅れる経路が 1 つある（S1）。period の `Option` を消した変更で振る舞いは変わっていない。should-fix は 4 件で、マージ前に直すかどうかは orchestrator が決める。

## must-fix

なし

## should-fix

### S1. ワーカーが戻したノードの期限が、ready の hook 1 回分だけ遅れる経路がある
- 対象: `crates/mujou-core/src/runtime/pool.rs:136-141`（ループは ready を先に見る）、`:258-261` と `:415-418`（戻したワーカーは `place` の戻り値を捨てる）、`:390-394`（`run_work` は timekeeper がいると誰も起こさない）
- 問題: `place` のドキュメントと specs（`specs/native-core.md:56`）は「ノードを戻したワーカーは自分ですぐキューを見る」としている。しかしループが最初に見るのは ready で、そこに hook があれば先にそれを回し、その間は lock を手放す。timekeeper の期限より早い更新を戻した直後に ready が空でないと、その更新を timekeeper は知らない。timekeeper がいるので `run_work` はほかのワーカーも起こさない。結果として、更新は ready の hook が戻るまで（それより早く timekeeper の期限が来ればそこまで）遅れる。修正前は `place` が timekeeper を起こしていたので遅れなかった
- 起こる場面: 自由実行中に ready に hook が積まれるのは、実質的に再開（`Resume`）のときだけ。例はワーカー 2 本でノード A（周期 P）・B（周期 1000P）・C（`on_resume` がブロックする）の場合。再開で ready = [A, B, C]。W2 が B の resume を終えて timekeeper になり、B の遠い期限を待つ。その後 W1 が A の resume を終え、A を期限 t_A（< B の期限）でヒープに戻す（戻り値 true は捨てられる）。W1 はループで ready の C を取り、C の resume でブロックする。t_A を過ぎても A は更新されず、仮想時計なら C の resume が戻るまで止まる
- 根拠: コードを読んでの解析で、再現テストは書いていない。期限の早い順に回すという EDF の保証（`specs/native-core.md:55`）が、狭い場面とはいえ修正前より弱くなる。直すなら、ready の hook を回す前（`run_work` の冒頭の notify の判定）にもループ末尾と同じ判定（先頭の期限 < timekeeper の期限なら起こす）を入れる。または ready を回す前にヒープを見る
- 確信度: 中 / 深刻度: 低〜中（遅れは hook 1 回分で有限。ただしブロックする `on_resume` では長くなる）

### S2. GIL の既定のテストは、CPU の quota がある Linux で落ちる
- 対象: `tests/mujou/_core/test_run.py:533-539`（`default_workers` が `os.process_cpu_count()` を使う）、`:620-628`（`len(idents) == default_workers()`）
- 問題: コアと bindings は `std::thread::available_parallelism()` を使う（`crates/mujou-python/src/runtime.rs:464`、`crates/mujou-core/src/runtime/mod.rs:500`）。Linux ではこれが cgroup の quota（`cpu.max`、`docker --cpus`、k8s の limits）も数に入れる。`os.process_cpu_count()` が見るのは affinity だけ。quota があると実際のワーカーは `process_cpu_count` より少なくなり、`==` が落ちる。ヘルパーの docstring にも「quota も数える」とあり、ずれは認識されている。手前のテスト（`:617` の `<=`）は quota があっても通る
- 根拠: Rust の `available_parallelism` の Linux 実装は cgroup v1/v2 の quota を読む。CPython の `process_cpu_count` は `sched_getaffinity` だけを見る。`just container` の dev container や quota 付きの CI runner に当たる
- 確信度: 中（GitHub Actions の標準 runner には quota が無く、今の CI では表に出ない見込み）/ 深刻度: 中

### S3. specs の制御の lock の数値が、記録の表と合わない
- 対象: `specs/native-core.md:86`
- 問題: 「待ちは取得 1 回あたり平均 1.5〜3.7 µs」「`tanh` の負荷では待つ取得は 3〜6%、取得 1 回あたり平均 0.2〜2.8 µs」「約 78 µs の更新に対して多くて 7% 程度」。specs が出典に挙げる `memory/agents/plan-implementer/mn-benchmark.md` の「制御の lock の待ち」の表では、light の w=14 が 3,441 ns、w=28 が 10,301 ns（w=7 は測っていない）。tanh は待った割合が 4.2〜6.2%、取得 1 回あたり 206 / 905 / 5,662 ns。1.5 µs と 2.8 µs はどの行にも無い。tanh の w=28 は 5.66 µs × 2 / 78 µs ≈ 14.5% で、「多くて 7%」は w ≤ 論理コア数に限った値に見える（記録の本文の「多くて 7%」も自分の表と合わない）
- 根拠: 上の 2 ファイルの突き合わせ。別の測定の値なら記録に残す。そうでなければ表の値に直し、範囲の条件（ワーカー数）を書く
- 確信度: 中 / 深刻度: 低

### S4. memory の 2 か所が PR 3 の結論に追従していない
- 対象: `memory/project_torch_module.md:17`（「既定のワーカー数（論理 CPU 数）が適切かは M:N の PR 3 で測る」）、`memory/project_native_core_rust.md:11`（「既定は論理 CPU 数」。GIL ありで半分になることが無い）
- 問題: PR 3 で測り終え、GIL ありの既定も変えた。`memory/project_mn_scheduler.md`、specs、skill は直してあるが、この 2 か所は古いまま
- 根拠: CLAUDE.md「メモリ参照」。古い記述が次の作業の前提になる
- 確信度: 高 / 深刻度: 低

## nit

- N1. `src/mujou/_core/run.py:445` の「on a build of Python with the GIL」と `specs/nodes.md:46` の「GIL ありのビルドでは」は、判定をビルドでするように読める。実装は実行時の `sys._is_gil_enabled()` で判定するので、free-threaded のビルドでも `PYTHON_GIL=1` なら半分になる。`specs/native-core.md:64` と docs の「with the GIL」は正確（確信度: 高）
- N2. `benchmarks/mn_scheduler.py:216` は達成率を `/ 100` で固定し、見出しも「100 Hz」。`--periods 0.005` などを渡すと誤った値が出る（確信度: 高）
- N3. `benchmarks/mn_scheduler.py:12` が勧める `UV_PROJECT_ENVIRONMENT=.venv-gil` は `.gitignore`（`.venv/`・`.venv-devcontainer/` だけ）に入っていない。手順どおりにすると、未追跡のディレクトリがリポジトリのルートにできる（確信度: 高）
- N4. `benchmarks/mn_scheduler.py:135-138` は macOS で、計測の窓の途中で `ps` を起動する。fork と待ちのぶんのスイッチと時間が測定に混じる。量は小さい（確信度: 中）
- N5. GIL ありの既定は「ユーザーへの確認待ち」のままコードに入っている（`specs/native-core.md:70`、`specs/open-decisions.md:15`、`memory/project_mn_scheduler.md:13`）。計画の「既定を変えてユーザーに報告する」の範囲内だが、マージ前にユーザーの返事で注記を確定させるとよい（欠陥ではない）
- N6. 半分という規則の根拠は、1 台（M4 Pro 14 コア、`tanh`）での測定だけ。2〜4 vCPU の runner では GIL ありのワーカーが 1〜2 本になる。GIL ありで全テストが通ったのは 14 コアの機械だけなので、少ないコアで GIL ありの CI が通るかを確かめるとよい（確信度: 低）
- N7. `test_the_default_number_of_workers_all_take_updates` は、実時間の 0.5 s の間に全ワーカーが 1 回以上更新することに頼る。起動時の `wake_pool` が全ワーカーを起こし、`run_work` が次々に起こすので、揺れる余地はほぼ無い。ただし負荷の高い runner では理屈の上で落ちうる（確信度: 低）
- N8. 制御語を変える経路では、`set_command` がもともと無条件に `timer.interrupt_waiters()` を呼び（`crates/mujou-core/src/runtime/mod.rs:245`）、続く `wake_pool`（`:246`）が条件付きでもう 1 度呼ぶ。「評価し直し 1 回につき 1 度だけ」（`specs/native-core.md:56`）は wake_pool の中に限った話で、全体では 2 度。修正前からある冗長さで、影響は無視できる（確信度: 高）

## 観点ごとの確認（要点だけ）

- lost wakeup: 制御の側の呼び出し元（`startup`・`step`・`set_command`・`fail`・`shutdown`・`abandon`）はどれも `wake_pool` か `wake_node` を通る。`wake_pool` の `look |= !dedicated && schedule()` で漏れは無い。ワーカーの側では、戻してから次に待つまで lock を手放さない（`run_work` と `take` は guard を返し、ループは drop しない）。先頭の期限が timekeeper の期限より早く、まだ来ていなければ必ず起こすので、拾われずに残る経路は無い。来ている期限は自分で取る。この場合、不変条件「timekeeper の期限 ≤ ヒープの先頭」から timekeeper もすでに起きている。例外は S1 の ready の経路だけ
- 足したテスト `period::a_target_a_worker_puts_back_before_the_timekeepers_is_kept`: hold はノード 0 だけに付き、ワーカー 2 本でノード 1 の 1000P を timekeeper に待たせる。規則 3 が無いと仮想時計が 1000P に届かず、`recv_timeout(LONG)` が落ちる。検出力は十分。5 回続けて通った。順序が想定と違っても偽の失敗にはならない。S1 の経路と、制御の側で 1 回にまとめた起こし方を直接見るテストは無い
- period の `Option` の削除: 0.0 秒は `period_to_ns` で 1 ns（以前は `None` から 1）。0 ns は `InvalidArgument`（以前の `Some(0)` と同じ）。外側の `None` は全ノード 1 ns（以前と同じ）。Python の呼び出し元は `session.py:156` だけで、常に float を渡す。振る舞いは変わらない
- c198478 の衝突の解消: `node.py` の Raises（#75 による `None` の削除と M:N の `dedicated_thread` を合わせた）も、execution-model の周期の段落（#75 の「0.0 もそのまま」に、プールのヒープの 1 文を残した）も正しい。リンク先は `torch.md#切り替えごとの設定` に追従済み。「周期が無ければ」は b1dca76 で「`0.0` なら」に直っている
- ベンチマーク: `testpaths = "tests/"` で、ファイル名も `test_*` でないので pytest は集めない。使い方は docstring にある。計測（窓の前後の `len(dts)`、`getrusage` の差分、Δt の分位点）は正しい
- 日本語: 足した specs・docs/ja・memory の文は文体の基準を大きく外れていない。docs/ja の「GIL を取り合うだけなので」は少し強い言い切りだが、ベンチマークの読み取りと矛盾はしない

## 検証結果
- just run: pass（cargo test 全通過、pytest 595 passed / 20 skipped、pyright 0 errors、docs-build 成功、format 後の差分なし）
- `cargo test -p mujou-core --test runtime a_target_a_worker_puts_back` を 5 回: 5 回とも pass
