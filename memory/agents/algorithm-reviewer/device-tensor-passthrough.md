# device-tensor-passthrough（CUDA のメッセージを device に置いたまま渡す）の監査

対象: worktree `.claude/worktrees/feature+2026-10-24+device-tensor-passthrough`、ブランチ `feature/2026-10-24/device-tensor-passthrough`（`a44fab3`）、`37b42e8..HEAD` の差分。基準は skill `locality-and-async`、`specs/execution-model.md`・`specs/messages.md`・`specs/nodes.md`、計画書 `memory/agents/implementation-planner/device-tensor-passthrough.md`、実装ログ `memory/agents/plan-implementer/device-tensor-passthrough.md`。

## verdict

**approve（条件付き）**（確信度: 高）。コードの修正は必須ではない。条件は must-fix 1 の 1 つで、手続きの問題である。ノード間の不変条件（計算グラフ、読み手は書き手を待たない、スロットの時刻、不変とコピー、停止の例外、更新手法の焼き付け）は、通常の構成ではすべて満たしている。実機（RTX 4090 + RTX 4060）でも確かめた。例外は stream のプールが尽きたとき（32 本超）の 1 件だけで、再現した（should-fix 1）。

- 計算グラフ: ノードをまたがない。書き手は `value.detach().clone(...)`、読み手は `view.to(device, copy=True)` で受け取る。`requires_grad=True` の積を送っても、受け取った値は `requires_grad=False` になる（`test_a_message_of_a_cuda_node_arrives_on_the_readers_device`）。`inference_mode` の中で送った場合も確かめた。読み手は通常のテンソルを受け取り、その場の `add_` も通る（照合メモ 4）
- 読み手が書き手を待たない（S2）: 満たす。書き手は送信の中で自分の current stream に event を記録し、その event だけを待ってから公開する。読み手は同じ device なら自分の stream に、別の device なら元の device の副 stream にコピーを積むだけである。どちらも書き手の stream を参照しない。実測では、書き手が送った直後に自分の stream へ約 0.5 s の仕事を積んでも、読み手の recv（とその完了の同期）は cuda:0 で 0.10 ms、cuda:1 で 1.5 ms、CPU で 1.8 ms だった。書き手の stream はその後も約 450 ms 動いていた（照合メモ 1）
- グローバル同期: 混入していない。`src/mujou/` に `torch.cuda.synchronize()` は無い。stream 全体の同期は起動前の `_synchronize` だけで、これは既存のものである。それ以外はすべて event の待ちになっている。CPU の読み手の D2H は副 stream の上で同期するが、その stream に積まれているのは、そのノード自身のコピーだけである（プールが尽きた場合を除く。should-fix 1）
- スロットの時刻: 意味は変わらない。`to_core` の event の待ちは `Outlet.send` より前にあり、コアは `messages.fill` の後で `clock.now_ns()` を取る（`slot.rs` の `Outlet::send`）。時刻は「値が device の上で出来上がった後」になり、これまで（D2H が終わった後）と同じ意味である。`record_event().synchronize()` を外す変異で `test_a_reader_gets_what_a_late_stream_wrote` が落ちることは、実装ログに記録がある
- 不変・必ずコピー: 保たれる。送るときは clone、読むときは `.to(copy=True)` でコピーし、保持テンソルそのものを利用者に渡す経路は無い（`Run.read` / `subscribe` / `wait_for` / `recv` はどれも `from_core` を通る）。`state_dict` が木に入れる参照（`peek`）も、`Run.state_dict` が `cpu(tree)` でコピーしてから返す。`record_stream` はアロケータのメタデータを変えるだけで、値は変えない。アロケータが塊を再利用する前に読み手のコピーが終わることは、`record_stream` を外す変異テストで確かめてある
- 停止・状態取得の例外: 範囲は広がっていない。停止中は参照を取るだけ（`codec.peek`）で、D2H は再開した後にホストのスレッドで行う（`state.cpu`）。保持テンソルは不変で、書き手は差し替えるだけなので、再開した後にコピーしても取得した時点の値のままである。復元・`initial`・ホストの `write` はホストのメモリを通る（`to_core` を `device=None` で呼ぶのは 4 か所すべて。照合メモ 3）
- 更新手法の焼き付け: 無い。コアに入ったのは不透明な `Held = Arc<dyn Any + Send + Sync>` だけで、device・stream・event の概念は持ち込んでいない。`Wire.cuda` と `Wiring.read_devices` は配線の静的な情報で、PC-ALM の量は含まない。PC-ALM の式には触れていない（`tests/mujou/pc/` は全て通る）
- 局所性: `read_devices` は、配線を組むときに隣のノードの `device` から決める静的な情報である。実行中にほかのノードの状態を読むことはない。`_sources` と同じ `port.channels` / `channel.other(port)` から作るので、読むスロットの漏れも無い

## must-fix

1. **CLAUDE.md と skill `locality-and-async` に、この変更と矛盾する不変条件の文言が残っている**（確信度: 高）
    - `CLAUDE.md:85` の「torch とコアの間は DLPack で必ずコピーして受け渡し」、`CLAUDE.md:86` の「GPU はコアでサポートせず、ノードの内側でだけ使う」、`.claude/skills/locality-and-async/SKILL.md:47` の「（ゼロコピー共有・GPU 共有はしない）」
    - 実装者は、サブエージェントには変更を承認できないとして編集していない。この判断は正しい。ただ、CLAUDE.md の「破る必要があるときはユーザーに確認し、理由を `memory/` に残す」に当たるので、merge の前にメインがユーザーの確認を取り、文言を直す必要がある（skill `edit-dot-claude`）。`memory/project_gpu_policy.md` には依頼の原文と S2 の理由が残っている。一方、確認事項 1（S2）と 2（device に置く範囲）は「推奨案で確定」と書いてあるだけで、ユーザーが選んだという記録が無い。メインはこの 2 点も実際にユーザーへ確認したかを確かめる
    - 直す案: skill 47 行は「読み書きとも必ずコピーする（書き手は clone、読み手は device の上でも新しいテンソル。ゼロコピーで利用者と共有しない）」。CLAUDE.md 86 行は「コアは CUDA の値を不透明な値として持つだけで、device・stream を知らない」

## should-fix

1. **stream のプール（device ごとに 32 本）が尽きると、副 stream が書き手の stream と重なり、読み手が書き手の GPU の仕事を待つ。CPU のノードでもホストが止まる**（確信度: 高。再現済み）
    - `Driver` の副 stream は `torch.cuda.Stream(d)` で、ノード自身の stream と同じプールから round robin で割り当てられる。書き手 1 つと idle な cuda:0 のノード 31 個のあとに CPU の読み手を並べると、読み手の副 stream が書き手の stream と同じになる。このとき読み手の recv は 468 ms かかり、書き手が送った後に積んだ `_sleep` の終わりまで待った。間のノードが 30 個なら 2.0 ms で済む（照合メモ 2）
    - 原因: CPU の読み手の `.to("cpu")` は ATen の `memcpy_and_sync` で、stream 全体を同期する（`cudaStreamSynchronize`）。そのため、書き手が公開した後にその stream へ積んだ仕事まで待つ。`specs/nodes.md:213` の「event なら、記録より後にほかのノードが積んだ仕事を待たない」は書き手の待ちには当てはまるが、読み手の D2H には当てはまらない。CUDA の読み手も同じで、コピーが共有の stream の後ろに並び、`on_update` の後の同期を通じてホストまで待つ
    - 32 本を超えると stream を共有するという制約は、もともとあった。この変更で変わったのは 2 点である。(a) CPU のノードと別の device のノードもプールを消費するので、32 本に早く達する。(b) 重なったときに「読み手が書き手の公開後の仕事を待つ」形になる。これは S1 を却下した理由そのものである
    - 対応案（どれか 1 つ）: (i) `specs/nodes.md` の CUDA の節と `memory/project_gpu_policy.md` に「副 stream も 32 本に数える。超えると、読み手が書き手の GPU の仕事を待つことがある」と制約として書く（最小の対応）。(ii) 副 stream だけ別の優先度のプール（`torch.cuda.Stream(d, priority=-1)`）から取り、ノードの stream と重ならないようにする。優先度の副作用は要確認。(iii) 副 stream をノードごとではなく（ワーカー × device）ごとに持つ。ただしノードはワーカーの間を移るので、設計の見直しが要る
2. **S2 の要の性質「読み手は書き手の公開後の GPU の仕事を待たない」に回帰テストが無い**（確信度: 中）
    - 既存の 2 つの slow テストは、値の正しさ（書き手が待つこと、アロケータが塊を再利用しないこと）を確かめている。読み手が待たないことは確かめていない。将来、誰かが読み手の側に `wait_event` を足して S1 寄りに戻しても、テストは落ちない
    - 案: 照合メモ 1 のスクリプトと同じ形にする。書き手は send の後で `delay_the_stream()` を積んで知らせ、読み手（cuda:0 / cuda:1 / cpu でパラメタ化）は知らせを受けて recv と自分の stream の同期を行い、書き手の stream がまだ動いていること（`torch.cuda.Event.query()` が False）を確かめる。実時間に依存させず、「読み手が終わった時点で書き手の event が未完了」という順序だけで判定できる
3. **ノードの hook が `with torch.cuda.device(other):`（または別の device の stream の `torch.cuda.stream(...)`）の中で自分の device のテンソルを送ると `BufferError` になる。これまでは通っていた**（確信度: 高。再現済み）
    - `_value` は `value.device == held` を見て clone を返すが、bindings の `__dlpack__` は current device がテンソルの device であることを要求する。出たエラー: `BufferError("Can't export tensors on a different CUDA device index. Expected: 0. Current device: 1.")`。以前は `.to("cpu")` だったので通っていた
    - 不変条件の違反ではない。それでも、利用者に見える退行である。案: `to_core` で `torch.cuda.current_device() == device.index` のときだけ保持の経路に入れ、違えばホストの経路に回す。これは毎回の send の hot path に入るので、性能の予算と相談する。もう 1 つの案は、制約を `Port.send` の docstring と `specs/messages.md` に書くこと

## nit

1. 書き手が保持テンソルを捨てると、アロケータは `record_stream` した読み手の stream に event を記録してから塊を戻す。待つことはない。ただし GPU のメモリが足りず、アロケータが `release_cached_blocks` に入ると、未完了の event をホストで同期する。このとき書き手が読み手の GPU の仕事を待ちうる。メモリが逼迫したときだけの PyTorch の既定の振る舞いなので、仕様に一言あればよい（確信度: 中、優先度: 低）
2. bindings の `copy_into` は、どの呼び手（`Outlet.send`、`InputGroup.write`、`Slot.restore`）でも、CUDA のテンソルなら照合だけしてそのまま持つ。不変であることの保証は、codec が必ず clone を渡すことに頼っている。今の呼び手は 4 か所とも codec を通り、ホストの書き込みは `device=None` で CPU に移すので問題は無い。`_native` は private なので、`Outlet` の doc comment にある契約で十分とも言える。ホストの書き込みの経路で、CUDA のテンソルが `Held` に入らないことを白箱で 1 つ確かめておくと、将来の変更から守れる（確信度: 中）
3. `state.cpu` の docstring の「as copies」は、CPU にすでにあるテンソルについては `.cpu()` が同じオブジェクトを返すので、厳密には正しくない。実害は無い。`peek` も `Driver.capture` も新しいテンソルを作っているので、結果としてはコピーになる（確信度: 高、優先度: 低）

## 検証結果

- `just run`: 通った（format → `cargo test` 全部 ok → pytest 951 passed / 1 skipped（`test_rejects_cuda_without_cuda`。CUDA があるので skip になる）、slow と `cuda(2)` を含む → pyright 0 errors → docs-build ok）。実行後も working tree は clean
- 実機での追加の確かめ（scratch のスクリプト。`/tmp/claude-1001/-data-geson-mujou/d2146d35-f3d8-44cf-90d5-5029742d85ef/scratchpad/algo/` の `probe.py`・`pool.py`・`leak.py`）
    - 読み手は書き手の公開後の仕事を待たない（照合メモ 1）
    - プールが尽きると待つ（should-fix 1、照合メモ 2）
    - `inference_mode` の中で送った値: 読み手は通常のテンソルを受け取り、その場の変更も通る。`state_dict` のポートの値は CPU にある
    - `torch.cuda.device(1)` の中で送ると `BufferError` になる（should-fix 3）
    - `ManualRun.step` の後もホストのスレッドの current stream は既定の stream のまま。hook はワーカーで走るので、Driver の `set_stream` がホストに漏れることはない

## 照合メモ

1. **S2 の順序。** 書き手のスレッドでは、`_value` の clone が current stream（Driver が設定した自分の stream）に積まれる。次に `record_event().synchronize()` がその stream の event をホストで待ち、そのあと `Outlet.send`（GIL を外したまま fill、`now_ns`、publish）に進む。読み手の側の ATen `copy_device_to_device` は、`copy_stream = getCurrentCUDAStream(src)` に積む。その前に `dst_ready`（読み手の stream）を `copy_stream` に待たせ、後に `src_ready` を読み手の stream に待たせる。どちらの依存も読み手自身の stream どうしで、書き手の stream は出てこない。続く `record_stream(current_stream(view.device))` は `copy_stream` と同じ stream を指す。同じ device の読み手では、`copy_stream` が読み手自身の stream になる。実測は書き手の送信後に `_sleep(1.25e9)` を積んだ状態で、recv の完了まで cuda:0 0.10 ms / cuda:1 1.47 ms / cpu 1.79 ms、書き手の stream はその後も約 450 ms 動いていた
2. **プールの重なり。** `Driver.__init__` は members の順にノード自身の stream を作り、続いて副 stream を作る。`{"w": Writer, i0..i30: Idle(cuda:0), "r": Reader(cpu)}` だと、cuda:0 の stream は 33 本目になり、r の副 stream が w の stream と重なる。このとき recv は 467.8 ms。i を 30 個にすると 2.0 ms
3. **ホストのメモリを通る経路。** `to_core` の呼び手は `Wire._values`（`device=self.cuda`）、`wiring._input_value`（ホストの write / inputs）、`wiring._outlet`（initial）、`state._write`（復元）。保持の経路に入るのは最初の 1 つだけである。保持するかどうかを決める条件は `held is not None and value.device == held and value.layout == torch.strided`。sparse はホストの経路に回り、DLPack で `BufferError` になる（テストあり）。ほかの device の CUDA テンソルは `.to("cpu")` の経路を通る（テストあり）
4. **計算グラフ。** `detach()` で autograd のグラフから切ってから `clone` するので、保持テンソルは葉でもなく、グラフも持たない。読み手の `.to(copy=True)` は勾配を記録する状態で呼ばれるが、入力が `requires_grad=False` なのでグラフは作られない。ノード内 autograd の目印（`# mujou: local-autograd`）が要る箇所は、この差分には無い
