# CUDA の経路のテスト

orchestrator が std-00 で回した検証スクリプト（repo の外）を `tests/mujou/` のテストに移した。CUDA が無い環境では skip、2 枚要るものは `device_count() < 2` で skip する。

## 足したテスト

| テスト | 確かめること | slow |
| ------ | ------------ | ---- |
| `test_node.py::TestCuda::test_hooks_run_on_the_device_on_a_stream_of_their_own[0/1]` | `cuda:N` のノードの hook の中で `current_device()==N`、既定でないストリーム、grad 無効（`[1]` は 2 枚要る） | |
| `test_node.py::TestCuda::test_waits_for_its_stream_after_each_update` | `on_update` の後にノードのストリームを待つ（次の `on_update` の始めに `current_stream().query()` が True） | o |
| `test_network.py::TestCuda::test_a_message_goes_from_one_gpu_to_another` | `cuda:0` → `cuda:1` の鎖で、受け手のデバイスに値が届く（2 枚要る） | |
| `test_network.py::TestCuda::test_manual_steps_match_the_discrete_algorithm` | 手動モード＋仮想時計で、GPU 上の参照ループ（float64 の Euler）とビット単位で一致 | |
| `test_network.py::TestCuda::test_on_start_sees_the_work_the_host_queued` | `__init__` でホストのストリームに積んだ仕事が `on_start` から見える（5 回） | o |
| `test_network.py::TestCuda::test_state_dict_waits_for_the_work_of_on_start` | `on_start` がストリームに残した仕事を、取得が待つ（5 回） | o |
| `test_network.py::TestCuda::test_state_dict_is_whole_while_a_node_updates_in_place` | 自由実行で毎ステップその場で更新するノードを 40 回取得しても、木の値が一様（更新途中が混ざらない）。再開後もコピーが変わらない | o |
| `test_network.py::TestCuda::test_a_restored_node_starts_from_the_tree` | GPU の状態を木から復元した直後（最初の `on_update` の始め）の値が木の値と一致 | o |

`device="cuda"` が `cuda:<current>` に決まることは既存の `TestDevice::test_fixes_the_cuda_index` が確かめている。

## 選んだ大きさ

スクリプトの値をそのまま使った。テンソルは 2^26 要素（float32 で 256 MB）、1 回の `on_update` で `add_` を 20 回、取得は 40 回。ホストや `on_start` が積む仕事は `add_` 50 回。std-00 では CUDA のテスト全体で約 1.9 秒なので、減らす必要はなかった。

時間で結果が変わらないようにした。自由実行のテストは「一様かどうか」だけで判定する。ほかのテストは手動モードで回すか、そもそも時間に依存しない。

## std-00 での結果（RTX 4090 + RTX 4060、torch 2.14.1+cu130、3.14t）

- `uv run pytest tests/mujou -q -rs`（slow を含む）: free-threaded と `-X gil=1` で 3 回ずつ実行し、毎回 277 passed、1 skipped（`needs no CUDA` のテスト）、約 7.1 秒
- CUDA のテストだけを両ビルドで 10 回ずつ（計 20 回）実行し、すべて 13 passed、1 skipped（約 1.9 秒）

## 陰性対照（repo には入れていない）

repo の外に置いた pytest plugin を `-p` で読み込み、同期を外した。

| 外した同期 | 落ちたテスト（両ビルド） |
| ---------- | ------------------------ |
| 3 つすべて（各ビルド 3 回） | 毎回 5 件: `test_on_start_sees_the_work_the_host_queued`、`test_state_dict_waits_for_the_work_of_on_start`、`test_state_dict_is_whole_while_a_node_updates_in_place`、`test_a_restored_node_starts_from_the_tree`、`test_waits_for_its_stream_after_each_update` |
| `Session._synchronize_current_streams` だけ | `test_on_start_sees_the_work_the_host_queued` |
| `Node._after_update` だけ | `test_waits_for_its_stream_after_each_update`（ほかは通るが、溜まった GPU の仕事を取得が待つので、CUDA のテスト全体が約 26 秒かかる） |
| `Node._capture_state` だけ | `test_state_dict_waits_for_the_work_of_on_start` |

気づいたこと:

- `_after_update` と `_capture_state` は、`on_update` の仕事については互いに補い合う。片方だけを外しても、スクリプトの項目 (3)(5) は落ちない。そこで同期ごとに単独で検出できるテストを 2 本足した。`on_update` の後の待ちは Node の docstring が約束している振る舞いで、`on_start` が残した仕事は `_capture_state` だけが待つ
- 最初の版では `on_start` のテストで 5 つのノードを先にまとめて作っていた。そのため 2 つ目以降のノードでは、ホストの仕事が run の前に終わってしまい、陰性対照で落ちなかった。今は各ノードを作った直後に run する
