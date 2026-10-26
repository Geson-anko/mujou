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

---

# 再監査（2026-10-25）: ホストの CUDA の write を device に置く（A）、CUDA の読み手が居るポートだけ device に置く（B）

対象: `2435496..HEAD`（`5542352` wiring、`08a195b` codec、`6697546` docs、`1954322` memory）。基準は skill `locality-and-async`、`specs/execution-model.md`・`specs/messages.md`・`specs/nodes.md`、`memory/project_gpu_policy.md`・`memory/project_host_io.md`、実装ログ `memory/agents/plan-implementer/device-tensor-passthrough.md`。

## verdict

**request-changes**（確信度: 高）。B は正しく、前回の should-fix 3（hook の中で別の device を current にした send が `BufferError`）も解消した。A も、値の正しさ・版番号の原子性・時刻の意味・コピー・グラフについては問題無い。ただし A では、**ホストが書いた CUDA の値を CPU のノードや別の device のノードが読むと、そのノードがホストの write の後の GPU の仕事を待つ**。実機で再現した（must-fix 1）。S2 で避けたはずの「読み手が書き手を待つ」形が、ホストを書き手として戻ってきている。仕様（`specs/messages.md` の「stream の同期」の最後の項）は、これを「別のノードの stream の後ろに並ぶ。性能だけ」と説明している。ホストが stream を設定しない普通の使い方では、並ぶ先はホスト自身の既定の stream なので、この説明は実態と違う。ユーザーはこの説明を前提に判断したので、選び直してもらう必要がある。

- 読み手が書き手を待たない（ノードの send）: 変わらず満たす。B は配線時に `Wire.cuda` を決めるだけで、書き手の待ち（`record_event().synchronize()`）と読み手の経路は前回と同じ
- 読み手がホストを待たない: **同じ device の読み手は満たす。CPU と別の device の読み手は満たさない**（must-fix 1）
- 一斉 write の版番号の原子性: 満たす。`Wiring.write` は、全ポートの `_item`→`to_core(device=THEIR_DEVICES)`（clone と各 device の event の待ち）を list の中で終えてから、`group.write` を 1 回呼ぶ。途中のポートで例外が出れば何も公開しない。複数の device の値を混ぜた write（ホストの current device が値と違う状態）も、1 つの時刻で一斉に切り替わることを確かめた（照合メモ 6）
- スロットの時刻: 意味は変わらない。待ちは `group.write` より前にあり、コアは公開のときに時刻を付ける。時刻の意味は「値が device の上で出来上がった後」で、ノードの send と同じ
- 不変・必ずコピー: 保たれる。ホストの write は `detach().clone()`。write の後に元のテンソルを `fill_` で書き換えても、読み手には届かない。読み手は `.to(device, copy=True)` で受け取り、状態の取得は `peek` が `_OnItsDevice` を外した後、`state.cpu` が CPU へコピーする（照合メモ 6）
- グラフ: ノードをまたがない。`requires_grad=True` から作った値を write しても、`detach` してから clone する
- グローバル同期: 持ち込んでいない。`_on_their_devices` が待つのは、ホストのスレッドの各 device の current stream に記録した event だけで、ノードの側にはバリアが無い
- B（静的な判定）: 正しい。`connect` は両端の spec が互いに逆であることを要求する（`a.send_spec == b.recv_spec`）。そのため、`channel.other(port)` は必ずそのポートの読み手で、`_held` に読み手でないものが紛れることは無い。`read_devices` も `held` から作るので、device に置かないポートの分の副 stream は作らない。stream のプールの消費が減るので、前回の should-fix 1（32 本を超えると重なる）にも少し効く
- 更新手法の焼き付け: 無い。PC-ALM の式には触れていない

## must-fix

1. **ホストが write した CUDA の値を、CPU のノードや別の CUDA device のノードが読むと、ホストが write の後に積んだ GPU の仕事を待つ**（確信度: 高。再現済み）
    - 仕組み: こうした読み手には副 stream が無い（`read_devices` に入れない）。そのため ATen は、コピーを読み手のワーカーのスレッドの、値の device の current stream に積む。そのワーカーで値の device のノードがまだ一度も走っていなければ、それは **legacy default stream（`cuda_stream=0x0`）** になる。この stream はプロセスの全スレッドで共有されていて、ホストが stream を設定せずに計算していれば、ホストの仕事もここに積まれている。CPU の読み手の `.to("cpu")` は `memcpy_and_sync` で stream 全体を同期するので、ノードのスレッドがホストの仕事の終わりまで止まる。CUDA の読み手は、自分の stream が event でつながり、`Driver.update` の後の待ちを通じて同じように止まる
    - 実測（`hostwrite.py`）: ホストが cuda:0 に write し、続けて自分の stream に約 0.5 s の `_sleep` を積んだ後に、読み手が recv した。recv（と CUDA の読み手は自分の stream の同期）にかかった時間は次のとおり

        | 読み手 | ホストが既定の stream | ホストが自分の副 stream |
        | --- | --- | --- |
        | cpu | **483 ms** | 2.0 ms |
        | cuda:1 | **467 ms** | 1.5 ms |
        | cuda:0（同じ device） | 0.13 ms | — |

        値はすべて正しい。読み手の cuda:0 の current stream は `0x0` だった
    - そのワーカーで値の device のノードが以前に走っていた場合は、そのノードの stream に積まれる。そのノードがいま別のワーカーで走っていれば、読み手はそのノードの GPU の仕事を待つ。逆に、そのノードの send と `update` の後の event の待ちも、読み手のコピーを待つ。関係の無い 2 つのノードが、互いに相手を待つことになる（確信度: 中。順序の推論で、実行順が非決定的なので再現はしていない。前回の should-fix 1 と同じ機構）
    - 仕様の誤り: `specs/messages.md` の「ホストが書いた値の読み手」の項と `memory/project_gpu_policy.md` の「ホストが書いた値には副 stream を作らない」は、これを「別のノードの stream の後ろに並ぶ（性能だけ）」と書いている。実際に待つ相手はホスト（書き手）自身で、skill `locality-and-async` 47 行の「読み手は書き手を待たない」に反する。全 device に副 stream を作る案を却下した理由（使わない GPU にも context を作る）も、遅延して作る案には当てはまらない。値が device Y にあれば、Y の context はすでにある
    - 直す案（ユーザーの決定 A を狭めるものを含むので、メインがユーザーに確かめる）
        - (i) **推奨。** B と同じく、読み手を見て静的に決める。ホストが書くポート（接続の無い受信ポート）の読み手は、そのポートを持つノード 1 つだけで、その device は配線のときに分かる。そこで `_item` が `device=` に持ち主の device を渡す。持ち主が CUDA なら、その device のテンソルだけを device に置く（ホストの current device と違っても置けるよう、`_OnItsDevice` で包むのは今のまま）。持ち主が CPU のノードなら `None` にして、ホストが D2H を払う（変更前と同じ）。持ち主と違う device のテンソルはホストのメモリを通す。hot path は増えず、ホストが書いた値の副 stream という問題そのものが無くなる。失うのは、ホストと読み手の device が違うときの D2D だけで、P2P の無い組では、もともとホストを中継している
        - (ii) A を保つ。`_read` が `_OnItsDevice` を別の device で読むとき、Driver がノードごと・device ごとに副 stream を遅延して作り、current にしてからコピーする。経路が増え、副 stream はプールの 32 本を消費する
        - どちらの案でも回帰テストを足す。既存の `test_a_reader_does_not_wait_for_what_the_writer_queues_after_sending` と同じ形で、書き手をホストにする（ホストが write の後に `_sleep` を積み、読み手（cpu / cuda:1 / cuda:0）が recv を終えた時点で、ホストの event が未完了であることを `Event.query()` で確かめる）。今のコードでは cpu と cuda:1 が落ちるはずである

## should-fix

1. **`CLAUDE.md:85` と skill `locality-and-async` 47 行が、この差分の後の挙動と合っていない**（確信度: 高）。どちらも「CUDA のノードが自分の device で送ったテンソルは device の上でコピー」と書いているが、置くのは CUDA のノードが読むポートだけ（B）で、ホストの write も置く（A）。47 行の「読み手は書き手を待たない」は、must-fix 1 を直すまでは事実と違う。どちらもメインがユーザーに確かめて直す（skill `edit-dot-claude`）
2. **ホストが書いた CUDA の値を持つ InputSlot の状態の取得（`peek` が `_OnItsDevice` を外す経路）に、テストが無い**（確信度: 中）。手元では動いた（照合メモ 6）。`test_state.py` の `TestCuda` に、write の後の `state_dict` で `ports.<p>.in.value` が CPU のテンソルになることと、復元できることのテストを足すとよい

## nit

1. 1 回の write で同じ device の複数のポートに書くと、`_item` がポートごとに `to_core` を呼ぶので、event の待ちがポートの数だけ繰り返される。全ポートの clone を積んでから device ごとに 1 回だけ待てば足りる。正しさには影響しない（確信度: 中、優先度: 低）
2. B では、CUDA と CPU の読み手が混ざった `MultiPort` は device に置く。その CPU の読み手は、小さいメッセージで前回測った「書き手の待ちと D2H の同期の 2 回」をそのまま払う。仕様の条件（CUDA のノードが 1 つ以上）どおりなので、`specs/research/benchmarks.md` に一言あればよい（確信度: 高、優先度: 低）

## 検証結果（再監査）

- `just run`: 通った（exit 0。format → `cargo test` 全部 ok → pytest 963 passed / 1 skipped（slow と `cuda(2)` を含む）→ pyright 0 errors → docs-build ok）。この差分のテストには must-fix 1 を捕まえるものが無いので、緑でも must-fix 1 は残る
- 実機（RTX 4090 + RTX 4060）での確かめ。scratch のスクリプトは `/tmp/claude-1001/-data-geson-mujou/d2146d35-f3d8-44cf-90d5-5029742d85ef/scratchpad/algo2/hostwrite.py` と `/tmp/claude-1001/-data-geson-mujou/d2146d35-f3d8-44cf-90d5-5029742d85ef/scratchpad/algo2/state_probe.py`
    - must-fix 1 の表
    - 複数の device・ホストの current device が違う状態の write、write の後の書き換えが届かないこと、`requires_grad` の値の detach、状態の取得の値が CPU にあること（照合メモ 6）

## 照合メモ（再監査）

5. **ホストが書いた値の読み手の経路。** `_read` → `held = view.tensor` → `held.to(device, copy=True)`。device が違えば、ATen の `copy_device_to_device`（CUDA の読み手）と `memcpy_and_sync`（CPU の読み手）は、`getCurrentCUDAStream(src)` に積む。それは呼んだスレッド（ワーカー）の thread-local な current stream である。Driver の `_use` が設定するのは、ノード自身の device と `read_devices` の device だけなので、ホストが書いた値の device はワーカーに残っている状態のまま使われる。何も設定されていなければ legacy default stream になる。PyTorch のプールの stream は non-blocking なので、legacy default stream との暗黙の同期は無い。一方、ホストのスレッドが stream を設定していなければ、ホストの仕事も legacy default stream に載るので、読み手のコピーはその後ろに並ぶ。続く `record_stream(current_stream(held.device))` は、コピーした stream と同じ stream を指すので、アロケータについては正しい
6. **A の正しさの確かめ（`state_probe.py`）。** CPU のノードの 2 つの受信ポートに、cuda:0 の値（`requires_grad=True` のテンソルから作った値）と cuda:1 の値を 1 回の write で書いた。ホストの current device は `torch.cuda.set_device(1)` で cuda:1 にした。write の直後に cuda:1 の元のテンソルを `fill_(9.0)` で書き換えた。受け取った値は `[1, 2]` と `[3, 4]` で、どちらも CPU にあり、時刻は同じ（`Timestamp(ns=0)`）だった。`state_dict` の `ports.{a,b}.in.value` は CPU のテンソルで、値も一致した

---

# 再々監査（2026-10-26）: ホストの write を持ち主の device に狭めた（決定 (i)）、別の device が current の send も device に置く

対象: `a3c1e08..HEAD`（`f1f7208` wiring、`2b71a47` port、`b500abb` docs、`50569f8` memory）と、未コミットの `CLAUDE.md`・`.claude/skills/locality-and-async/SKILL.md` の文言。基準は前回と同じ。`just run` は orchestrator が回すので実行していない。

## verdict

**approve**（確信度: 高）。前回の must-fix 1 は直った。実機で再現スクリプトを流し直し、CPU と cuda:1 の読み手が、ホストが write の後に積んだ GPU の仕事を待たなくなったことを確かめた。別の device が current のときのノードの send（`_OnItsDevice`）も、S2 が成り立つ。clone は値の device のノード自身の stream に積まれ、待つのもその stream の event で、読み手は書き手の公開後の仕事を待たない（実測）。一斉の write の原子性と時刻、計算グラフ、グローバル同期にも問題は無い。should-fix は 2 つある。1 つは、別の device が current のときの待ちの stream を取り違えても落ちるテストが無いこと（変異を入れても 42 件すべて通った）。もう 1 つは、CLAUDE.md と skill の要約が条件を広く書きすぎていること。

- must-fix 1（前回）: **直った**。`Wiring.input_devices` は配線時に、ホストが書くポート（接続の無い受信 `Port`）ごとに持ち主の CUDA device か `None` を決める。読み手は持ち主のノードだけなので（`_sources` がそのスロットを持ち主にだけ渡す）、device に置いた値はいつも読み手自身の stream で同じ device の中でコピーされる。それ以外の値は、ホストが自分の stream の上で D2H してからホストのメモリに置く
- 一斉の write の原子性と時刻: 保たれる。全ポートの `_item`（clone と D2H）を list の中で終えてから、`wait_for_clones` が device ごとに 1 回待ち、そのあと `group.write` を 1 回だけ呼ぶ。途中で例外が出れば何も公開しない。時刻はコアが公開のときに付けるので、意味は「値が device の上で出来上がった後」のまま。2 つのノード（cuda:0 / cuda:1、cuda:0 / cpu、cuda:1 / cuda:0）への 1 回の write を、ホストの current device をもう一方の device にした状態で試した。値はどちらも遅れて埋めたもので、両方とも正しく、時刻は 1 つだった（照合メモ 8）
- `_OnItsDevice` での send の S2: 成り立つ。照合メモ 7 のとおり、clone と待ちは同じ stream（スレッドの値の device の current stream。hook の中ではノード自身の stream）を使う。書き手が送った直後に約 0.1 s の仕事を積んでも、読み手（cuda:0 / cuda:1 / cpu）の recv は 0.3〜2.2 ms で終わり、その時点で書き手の event は未完了だった。遅れて埋めた値を送ると send は約 93〜99 ms 待ち、読み手には正しい値が届いた
- 読み手は書き手を待たない: ノードの send（別の device が current の場合を含む）でも、ホストの write でも成り立つ。例外は、前回の should-fix 1 で指摘した stream のプールの共有だけで、これは変わっていない（nit 1）
- グローバル同期: 持ち込んでいない。`wait_for_clones` が待つのは、呼んだスレッドの、clone のある device の current stream の event だけ。`torch.cuda.synchronize()` は増えていない
- 計算グラフ・不変・コピー: 変わらない（`detach().clone()`、読み手は `.to(copy=True)`、`peek` は `_OnItsDevice` を外す）。core の値を読む所はすべて codec の `from_core` / `peek` を通るので、Outlet のスロットに `_OnItsDevice` が入るようになったことで外し忘れが起きる箇所は無い
- 更新手法の焼き付け: 無い

## must-fix

なし。

## should-fix

1. **別の device が current のときの S2 の待ちを確かめるテストが無い**（確信度: 高。変異で確かめた）。`wait_for_clones` の `torch.cuda.current_stream(d)` を `torch.cuda.current_stream()`（current device の stream）にする変異を pytest のプラグインで入れ、`test_wiring` / `test_port` / `test_execution` / `test_state` の CUDA のテスト（`-k "Cuda or Devices"`、42 件）を流した。42 件すべて通った。`test_a_reader_gets_what_a_late_stream_wrote` は current device が値の device と同じなので、この変異と区別できない。`torch.cuda.device(1)` の中のテスト（`test_sends_while_the_hook_has_made_another_device_current` と `test_what_the_host_writes_stays_only_on_the_device_of_its_node`）は、遅れて埋める stream を持たない。一方、同じ変異のもとで、遅れて埋めた値を `torch.cuda.device(1)` の中で送るスクリプト（`otherdev.py`）を流すと、読み手は cuda:0 / cuda:1 / cpu のどれでも**未完成の値**を受け取った（`ok=False`）。これは正しさの欠陥なので、回帰テストが要る。案: `Late` と同じ形で `torch.cuda.device(1)` の中で送るノードと、すぐ読む読み手のテスト（`cuda(2)`）。ホストの write にも同じ形のテストを足す（cuda:0 の既定の stream で `_sleep` してから埋め、`torch.cuda.device(1)` の中で write し、持ち主の cuda:0 のノードが読む）
2. **CLAUDE.md と skill の新しい文言は、device に置く条件を広く書きすぎている**（確信度: 中。要約なので程度の問題）
    - `CLAUDE.md:85` と skill 47 行の「CUDA のノードが読むテンソル（値）は device の上でコピーし」は、実際より広い。CUDA のノードが読んでもホストのメモリを通る値がある。CPU のノードが送った値、ノードが自分の device 以外の device のテンソルを送った値、ホストが持ち主と違う device のテンソルを write した値の 3 つ。逆に、CUDA のノードが 1 つでも読むポートでは、同じポートの CPU の読み手も device から読む。正確に書くなら次のとおり（どちらもメインがユーザーに確かめて、skill `edit-dot-claude` で直す）。「送ったノード（ホストの write では読むノード）の CUDA device の上のテンソルは、CUDA のノードが読むなら device に置いたまま渡し、読み手が device の上でコピーする」
    - `CLAUDE.md:86` の「CUDA のノードが CUDA のノードに送ったメッセージ」には、「自分の device の上の」が抜けている。ホストの write の部分（「持ち主のノードの device で書いた値」）は挙動と合っている
    - 「書き手（ノードの send、ホストの write）は自分の stream を待ってから公開する。読み手は書き手を待たない」は、今回の実測と合っている（例外は nit 1）

## nit

1. 「読み手は書き手を待たない」の例外は、前回の should-fix 1（device ごとに 32 本の stream のプールを超えると、stream が共有される）だけで、これは変わっていない。ただ、ホストが `torch.cuda.Stream()` で stream を作った場合も、同じプールから取る。そのため、ホストの stream が持ち主のノードの stream と重なることがあり、そのとき持ち主のコピーはホストの write の後の仕事の後ろに並ぶ。docs の「Sharing a stream」の段落は、ノード同士の共有しか書いていない。ノードとホストを合わせて 32 本を超えたときだけ起きるので、優先度は低い（確信度: 中。推論で、再現はしていない）
2. この差分の前からある挙動: CUDA のノードが自分の device ではない device のテンソルを送ると、`_value` の `.to("cpu")` は、ワーカーに残っているその device の current stream（何も設定されていなければ legacy default stream、そうでなければ前にそのワーカーで走ったノードの stream）に積まれ、その stream 全体を同期する。書き手がホストやほかのノードの GPU の仕事を待ちうる。ノードが自分の device 以外で計算するのは普通の使い方ではないので、優先度は低い（確信度: 中）

## 検証結果（再々監査）

- `just run`: 実行していない（orchestrator が実行する）
- 実機（RTX 4090 + RTX 4060）。スクリプトは `/tmp/claude-1001/-data-geson-mujou--claude-worktrees-feature-2026-10-24-device-tensor-passthrough/d2146d35-f3d8-44cf-90d5-5029742d85ef/scratchpad/algo3/` に置いた
    - `hostwrite.py`（前回と同じスクリプト）で must-fix 1 が直ったことを確かめた。条件は前回と同じで、ホストは write の後に約 0.5 s の `_sleep` を積んだ。表の値は recv にかかった時間（CUDA の読み手は、自分の stream の同期まで含む）

        | 読み手 | ホストが既定の stream（前回 → 今回） | ホストが自分の副 stream |
        | --- | --- | --- |
        | cpu | 483 ms → **1.80 ms** | 1.49 ms |
        | cuda:1 | 467 ms → **1.50 ms** | 1.55 ms |
        | cuda:0 | 0.13 ms → 0.09 ms | 0.09 ms |

        値はすべて正しい。ホストの stream は、読み手が読み終えた後もさらに約 440〜480 ms 動いていた
    - `otherdev.py node`: cuda:0 のノードが `torch.cuda.device(1)` の中で送る場合を試した。遅れて埋めた値は、send が 93〜99 ms 待ったうえで正しく届いた。送った後に仕事を積んだ場合は、send が 0.1〜0.4 ms で終わり、読み手が読み終えた時点で書き手の event は未完了だった。読み手は cuda:0 / cuda:1 / cpu の 3 通り
    - `otherdev.py host`: 2 つのノードへの 1 回の write を、ホストの current device をもう一方の device にして試した。値は正しく、write は約 93〜99 ms 待ち、時刻は 1 つだった
    - `mutplug.py` と `run_mut.sh`: should-fix 1 の変異。テストは 42 件すべて通り、`otherdev.py node` は 3 通りとも `ok=False` になった

## 照合メモ（再々監査）

7. **`_OnItsDevice` の send の stream。** Driver の `_use` は、ノード自身の stream を `set_stream` で current にする。この設定は device ごと・スレッドごとに持たれ、`torch.cuda.device(1)` は current device を変えるだけで、cuda:0 の current stream はそのまま残る。`detach().clone()` は `empty_like` と `copy_` からなる。確保はキャッシュアロケータが値の device（cuda:0）の current stream で行い、同じ device の中の `copy_` も device guard の下で `getCurrentCUDAStream(0)` に積む。どちらもノード自身の stream になる。`wait_for_clones` は `torch.cuda.current_stream(d)`（d は clone の device）に event を記録して待つので、clone を積んだ stream と一致する。export の間だけ `_OnItsDevice.__dlpack__` が `torch.cuda.device(d)` に切り替え、bindings は `stream` を渡さないので、余計な同期は起きない。読み手の `_read` は包みを外す。同じ device の読み手は自分の stream で、別の device の読み手は副 stream（`Wire.cuda` が値の device なので `read_devices` に入る）でコピーし、`record_stream` も同じ stream を指す
8. **ホストの write の経路。** `_item` は `to_core(device=input_devices[port], wait=False)` を呼ぶ。clone は、ホストのスレッドの持ち主の device の current stream（設定していなければ legacy default stream）に積まれる。ホストの current device が違うときは `_OnItsDevice` に包む（ポートごとに判定する）。`wait_for_clones` は、生のテンソルと包んだテンソルの両方から device を集め、device ごとに 1 回待つ。持ち主以外の device のテンソルは、`.to("cpu")` がホストの stream を同期する（待つのは書き手のホスト自身）。clone のメモリはホストの stream に属し、読み手の `record_stream` が読み手の stream の event でメモリの再利用を止めるので、ホストが次の確保で同じメモリを使っても安全である
