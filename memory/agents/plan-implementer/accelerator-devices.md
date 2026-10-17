# accelerator-devices（2026-10-17）

ノードが CUDA 以外のアクセラレータ（MPS など）で計算できるようにする。ノードごとのストリームは CUDA だけ。

## 実装の判断

- 検証は `_check_device`（`src/mujou/_core/node.py`）に切り出した。CUDA も同じ経路（`torch.accelerator.current_accelerator(check_available=True)`）で検証する。CUDA が使えないときのメッセージ「CUDA is not available」は変わらない
- 「使えない」と「アクセラレータでない」の区別には、既知の種類の組 `("cuda", "mps", "xpu", "mtia", "hpu")` を使った。torch に公開の一覧 API が無いため。privateuse1 のバックエンドは現在のアクセラレータなら受け付け、そうでなければ一般のメッセージになる
- `torch.device("mps") != torch.device("mps:0")`、MPS のテンソルは `mps:0` になるので、添字は構築時に `current_device_index()` で固定する
- `torch.Generator("mps:0")` は動く。同じ seed で同じ列になることをテストで確かめた
- 同期は `_synchronize()`（CUDA はストリーム、ほかはデバイス全体の `torch.accelerator.synchronize(device)`）
- `Session._synchronize_current_streams` は CUDA のままにした（計画外の変更をしない）。MPS はノードもホストも同じストリームに積むので順序は保たれる

## 計画からの逸脱

- `test_run.py` の `TestCuda` を分け、ストリームに依存しないテスト（起動前のホストの仕事、`state_dict` の前後の同期、復元）を `TestAccelerator` に移して `accelerator` fixture（`tests/mujou/conftest.py`）で走らせる。CUDA のマシンでは CUDA で走る
- `test_manual_steps_match_the_discrete_algorithm`（float64）と `test_state_dict_is_whole_while_a_node_updates_in_place`（ホストがノードと同時に GPU を読む）は MPS で走らないので `TestCuda` に残した
- `test_port.py` の `TestCuda` は `TestAccelerator` にした（同じ内容をアクセラレータ一般で）
- `test_node.py` の `TestRestore` の CUDA のパラメータを「アクセラレータ」に広げた
- 既存テスト `test_rejects_devices_other_than_cpu_and_cuda` はメッセージが変わったので、`test_rejects_devices_other_than_the_cpu_and_accelerators` に変えた

## 発見した問題（要ユーザー判断）

PyTorch の MPS（torch 2.14.1、Apple Silicon、3.14t）は 2 つのスレッドから同時に使うとプロセスが落ちる。

- 素の torch で再現: 2 スレッドが MPS で `add_`・`mul_` をすると SIGSEGV（rc=139）、1 スレッドが更新中にほかのスレッドが `clone`・`min` すると `failed assertion _status < MTLCommandBufferStatusCommitted`（rc=134）。ロックで直列にすると落ちない
- mujou で再現: MPS のノード 1 つの鎖は動く（`mujou.run` で 50 回書き込み、`state_dict`、`shutdown` まで rc=0）。2 つの鎖は 2 回とも rc=139
- 対処の候補: (a) MPS のノードの hook を 1 本のロックで直列にする（Rust に `_before_update` のような hook が要る。ホストの利用者のコードは守れない）、(b) 構築時に 2 つ目の MPS のノードを拒む、(c) 文書の注意だけ
- orchestrator の決定で (b) を実装した: `Session.start` の最初の `_check_one_node_devices`。動いている実行（`_RUNNING`、先に `_settle`）も数える。(a) は後でユーザーに確認する
- 同期をデバイス全体にすること（Q2）と、既知のアクセラレータの組（Q3）は orchestrator が了承した。理由をコードのコメントに足した
