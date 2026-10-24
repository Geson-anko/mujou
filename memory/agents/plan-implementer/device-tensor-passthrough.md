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
