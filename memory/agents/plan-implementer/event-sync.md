# event-sync（2026-10-19）

CUDA のノードの同期を `stream.synchronize()` から、ノードのストリームに記録したイベントの `synchronize()` に変えた。

## 変更

- `Node._synchronize` を `self._stream.record_event().synchronize()` にした。`_after_update`・`_restore_state` はこれを呼ぶ
- `_capture_state` の 2 回の `stream.synchronize()` を `self._synchronize()` にした。前の 1 回は全 hook の仕事（どれもノードのストリームに積まれる）、後の 1 回は `state_dict` が積んだコピーを待つ。どちらもイベントの記録時点までの仕事なので、待つ範囲は足りる
- イベントは毎回作る（単純さを優先）。std-00 の測定で、ストリームの同期 0.33 µs、毎回作るイベント 1.43 µs、使い回すイベント 0.64 µs
- docstring・`specs/torch.md`・`docs/{en,ja}/user-guide/torch.md` を合わせた。利用者向けの自前の同期の例も `record_event().synchronize()` にした

## GPU 機（std-00、RTX 4090）

- `-m cuda` と全テスト: 失敗は 2 件（`test_port.py::TestCuda::test_receives_on_the_device_and_sends_to_the_host`、`test_state.py::TestStateDict::test_moves_from_cuda_to_the_cpu`）。変更前の node.py でも同じ 2 件が落ちるので、ベースのブランチ（#78）の問題で今回の変更とは無関係
- 更新の速さ（1 ノード比、`CUDA_MODULE_LOADING=EAGER`、2 秒 × 5 回を交互に、中央値）: 64 ノード / 16 ワーカー stream 15.93 倍 / event 15.40 倍、256 ノード / 16 ワーカー stream 15.82 倍 / event 15.77 倍。差は揺れの範囲で、依頼時の測定（15.28 → 16.01 倍）は再現しなかった
- 単体の確認: 300 ms の仕事を積んだストリームを `synchronize()` している間に、別スレッドが 100 ms 後に同じストリームへ 1.5 s の仕事を積んでも、`synchronize()` は 0.29 s で戻った。この環境では、ストリームの同期も呼んだ後に積まれた仕事を待たない

## 判断

- 速さの改善は確認できなかったが、イベントの待ち方は CUDA が文書で保証している（ストリームの同期は呼んだ後に積まれた仕事を待つかが文書にない）。依頼どおり実装し、spec には測定どおりに「変わらなかった」と書いた
- 区別できる決定的なテストは書けなかった（上の通り、この環境では両者の振る舞いが同じ）。既存の `TestCuda::test_waits_for_its_stream_after_each_update` が自分の仕事を待つことを確かめる
