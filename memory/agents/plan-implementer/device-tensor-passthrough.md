# device-tensor-passthrough の実装ログ（plan-implementer）

計画書: `memory/agents/implementation-planner/device-tensor-passthrough.md`。確認事項 1（S2）と 2（ノードの hook の send / send_to で、ノード自身の CUDA device の値だけ）は推奨案で確定済みとして実装した。

## 計画からの逸脱と判断

1. **`borrow_tensor` に `stream=-1` を渡さない。** torch 2.14.1（`pyproject.toml` の下限）の `Tensor.__dlpack__` は既定が `stream=-1`（同期しない）。引数を足すと CPU の hot path に kwarg が 1 つ増え、NumPy など CPU で `stream` に `None` 以外を受け付けない producer も壊す。`crates/mujou-python/src/dlpack.rs` にコメントで残した
2. **`to_core` の `device` は「送るノードの CUDA device か None」にした（計画は「書き手のノードの device」）。** `device.type == "cuda"` が 1 回 約 0.1 µs かかり、scalar の send が 4〜6% 悪化した。`Wire` に段 2 で決める `cuda: torch.device | None` を足し、`Wire._values` はそれを渡す（`Wire` は内部の名前なので公開 IF は変わらない）
3. **`codec._read` は `type(view) is _native.BufferView` で分岐する。** `isinstance(view, torch.Tensor)` は 1 回 約 0.1 µs で、recv が 5〜8% 悪化した。区別を `TensorSpec` の case の中に置き、スカラーの経路には分岐を足していない
4. **CUDA の値の照合（`check`）は `borrow_values`（GIL を持つ側）で行う。** capsule は照合の直後に落とし、`Py<PyAny>` をそのまま `Held` にする（`HeldTensor` の newtype は作らず、`Py<PyAny>` を `dyn Any` に入れて `downcast_ref` で戻す）
5. **`Message.values` の stub は `list[Any]`。** PyO3 の introspection で `BufferView | Any` を書いても `Any` に潰れるので、`Vec<Bound<PyAny>>` を返し docstring で説明した。codec 側の型は `type Views = Sequence[BufferView | torch.Tensor]`
6. **`FieldError::UnsupportedDevice` の文言を「host memory or on a CUDA device (DLPack device type 1, 3 or 2)」にした。** 出るのは CPU・pinned・CUDA 以外の device だけ
7. **CLAUDE.md と `.claude/skills/locality-and-async` は編集していない。** サブエージェントへの指示はユーザーの同意にならず、CLAUDE.md の変更を承認できないため。差し替え文案は報告に載せた
8. **`specs/research/benchmarks.md` には新しい R 番号を振っていない。** R 番号は作り直しの設計（`restructure-design.md`）のリスクの番号なので、日付の節「CUDA のメッセージを device に置いた後（2026-10-24）」にした。CUDA の測定は scratch のスクリプト（R13 と同じ扱い）で、`Wire._values` を `device=None` に差し替えた同じビルドとの A/B
9. **テスト用の helper `probe_on(device, update, **ports)` を `tests/mujou/nodes.py` に足した**（`Probe` の `on_device` を変えたサブクラスを作る。test_port / test_execution / test_state / test_driver / test_wiring が使う）

## 実装中に分かったこと

- **CUDA の lazy loading がストリームの競合を隠す。** カーネルを初めて実行するとき、CUDA はモジュールを読み込むためにデバイス全体を同期する。そのため「書き手が send で待たない」変異を入れても、最初の送受信では読み手のコピーが書き手の stream を待ってしまい、テストが通った。`tests/mujou/test_port.py` の `OnCuda0.on_start` で、テストが使う種類のカーネル（fill、clone、コピー、D2H、`_sleep`）を一度ずつ走らせてから測る
- 変異テストの結果: `to_core` の `record_event().synchronize()` を外すと `test_a_reader_gets_what_a_late_stream_wrote` が 3/3 で落ち、`_read` の `record_stream` を外すと `test_a_copy_still_queued_keeps_its_message_from_reuse` が 5/5 で落ちた（決定的に落ちるので繰り返しは入れていない）。元に戻して 8 回連続で通る
- profiler のオラクルの対照: `Wire._values` を `device=None` に戻すと `Memcpy DtoH (Device -> Pageable)` と `Memcpy HtoD (Pageable -> Device)` が出る。P2P の無い 4090 と 4060 の間は `Memcpy DtoH (Device -> Device)` などで、端点はどれも Device
- PyO3 0.30 の reference pool は `SuspendAttach` の drop（`detach` から戻るとき）で同じスレッドが処理する。detach 中に落ちた `Py<PyAny>` の decref は send から戻る前に済む（`native/test_slot.py` の `test_holds_a_tensor_on_a_cuda_device_as_it_is` が refcount で確かめる）
- **CPU の読み手の小さいメッセージは合計が約 17 µs 増える。** D2H が send から recv へ移り、書き手の stream の待ちと読み手の D2H の同期の 2 回になるため（100 万要素では合計 1.7 ms → 1.4 ms と減る）。計画の想定外。別タスクの候補として報告した
- 測定中、`cuda:0` は別のプロセス（このコンテナからは見えない pid）が 65〜69% 使っていた。小さいメッセージの `cuda:0` の値はその待ちに埋もれている

## IF 変更通知

なし（並列の implementer はいない）。内部の口の変更は計画書のとおり（`to_core(device=)` の意味だけ上の 2 のとおり変えた）。

## レビュー差し戻しへの対応（2026-10-24）

- must-fix（hook が `torch.cuda.device(1)` の中で cuda:0 のテンソルを送ると `BufferError`）: `to_core` で `torch.cuda.current_device() != device.index` のときは保持せずホストの経路に回す（変更前と同じ挙動）。書き手の device を current にして export する案は、Wire と bindings の両方に手が入るうえ、利用者が device を変えて送る稀な場合のためだけなので採らなかった。確認は CUDA のノードの send にだけ入る（CPU の hot path は `device is None` で抜ける）。`ports.py` は変更前と 3% 以内。回帰テストは `test_sends_while_the_hook_has_made_another_device_current`（`cuda(2)`）
- 「読み手は書き手の公開後の仕事を待たない」の回帰テスト `test_a_reader_does_not_wait_for_what_the_writer_queues_after_sending`（cuda:0 / cuda:1 / cpu）。`_read` に `torch.cuda.synchronize(view.device)` を足す変異で 3 件とも落ちる
- 副 stream がプールの 32 本に数えられ、重なると読み手が書き手の公開後の仕事を待つことを `specs/nodes.md`・docs・`project_gpu_policy.md` に書いた（コードは変えない）
- nit: `codec._CPU` を上へ、`_value` の引数を `stays_on` に、`UnsupportedDevice` の文言を「host memory (1 or 3)」に戻した（コアの `copy_from` が出すので）、`messages.md` の機械固有の記述を benchmarks へ、保持値が落ちる場所の記述、`state.cpu` の docstring、ホストの write が `Held` にならない白箱テスト

## ユーザーの決定（2026-10-25）への対応

- **B（CUDA → CPU の悪化）**: `wiring` が配線時に `held`（ポートごとに、ノードが CUDA で接続先に CUDA のノードが 1 つ以上あればその device、ほかは None）を決め、`Wire.cuda` に入れる。`read_devices` も `held` から作るので、device に置かないポートの分の副 stream は作らない。hot path は増えない。`cuda:1 → CPU` は変更前（`device=None`）と測定の揺れの中で一致
- **A（ホストの write）**: `to_core(device=THEIR_DEVICES)`（`codec.Their` の enum の唯一の値）で、密な CUDA のテンソルを各自の device の上で clone し、device ごとに `with torch.cuda.device(d)` で current stream の event を待つ。clone は `codec._OnItsDevice` に包み、bindings が呼ぶ `__dlpack__` の間だけ device を切り替える（torch の current device の制約。dict の複数 device・一斉の write でも、値ごとに切り替わる）。bindings は包みを保持し、`from_core` / `peek` が外す。bindings の変更は無し
  - 却下: write の全体を `torch.cuda.device(d)` で囲む（複数 device が混ざると効かない）。bindings で device を切り替える（`_native` は torch を使わない）。Python で shape・dtype を照合して capsule を経ない（K9 の正本が二重になる）
- **ホストが書いた値の読み手の副 stream は作らない**（`read_devices` に入れない）。書かれる device は write まで分からない。別の device の読み手は自分のスレッドの元の device の current stream でコピーし、値は正しいが、別のノードの stream の後ろに並ぶことがある（性能だけ）。全 device に副 stream を作る案は、使わない GPU にも CUDA の context（数百 MB）を作るので採らない。`specs/messages.md` の「stream の同期」に記録
- ホストの write の S2 は変異テスト（`_on_their_devices` の待ちを外す）で `test_a_write_waits_for_the_hosts_stream` が落ちることを確かめた
- `ports.py`: input (64,784) の send が 6.0 と 7.6 µs の 2 峰に揺れた（HEAD の Python に戻しても 6.0、新しいコードでも 6.1 と 7.6 が出る）。ほかは変更前と 3% 以内

## ユーザーの決定（2026-10-25、2 回目）への対応

- **algorithm-reviewer must-fix 1 を (i) で直した。** ホストの write で device に置くのは、そのポートを持つノード（唯一の読み手）の CUDA device の上のテンソルだけ。`Wiring.input_devices`（配線時に決める）を `_item` が `to_core(device=...)` に渡す。`THEIR_DEVICES` と `codec.Their` は消した。回帰テスト `test_a_reader_does_not_wait_for_what_the_host_queues_after_writing`（cuda:0 / cpu / cuda:1）は、`input_devices` を常に cuda:0 にする変異で cpu と cuda:1 が落ちる（ホストの `_sleep` を待つ）。profiler のオラクルに host:cuda:0 → cuda:0（ホストを経由しない）、host:cuda:0 → cpu、host:cuda:1 → cuda:0（ホストを経由する）を足した
- **code-reviewer nit 5**: ノードの send で別の device が current のときも device に置く。`to_core` は device が決まっていれば clone し、`torch.cuda.current_device() != device.index` なら `_OnItsDevice` で包む（ホストの write と同じ仕組み）。S2 の待ちは `wait_for_clones` が clone の device の current stream（hook ではノードの stream）に対して行う。`test_sends_while_the_hook_has_made_another_device_current` に memcpy のオラクルを足し、ホストへ回す旧挙動の変異で落ちることを確かめた
- `to_core` に `wait: bool = True` を足した。ホストの write だけ `wait=False` で、`Wiring.write` が全ポートの clone を積んでから device ごとに 1 回待つ
