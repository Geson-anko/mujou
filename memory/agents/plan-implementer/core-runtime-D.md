# core-runtime WP-D（結合とホスト入出力、`mujou.run`）の判断ログ

計画書: `memory/agents/implementation-planner/core-runtime.md` の WP-D。確認事項 1〜12 は orchestrator の裁定どおり推奨案で実装した。

## 公開 API の最終形

`mujou.__all__ = ["Channel", "DictSpec", "Network", "NetworkError", "NextPort", "Node", "PauseTimeoutError", "Port", "PrevPort", "TensorSpec", "__version__", "run"]`。`mujou.time` は `__init__` で import する（`__all__` には入れない）。

`Network` のメソッドと `mujou.run` の引数は計画書どおり（`run(*, capture_state_on_exception=False, pause_timeout=10.0, _manual=False)`、`write`、`read`、`subscribe`、`state_dict`、`load_state_dict`、`exception_state_dict`、`shutdown`、`join`、`_step`）。`mujou.Node.__init__(*, device="cpu", seed=None)`、`device` / `generator` property。

## 構成

- `_session.py`: `Session`（build の最後に 1 つ作る。Outlet・InputGroup（ネットワークに 1 つ）・InputSlot を作り、全 Port に `Binding` を結び付け、run で `_core.Runtime` を作って start）と `Binding`（Port の recv/send の本体）。`_port.py` は `Binding` を TYPE_CHECKING でだけ import し、`bind(port, binding)` で結び付ける（`freeze` と同じ形）
- import の循環を避けるため、`_network.py` は `Session` を `build` の中で import する（`_session` → `_node` → `_network`）
- `mujou.run` は `_network.py` に置いた（`network._graph` を見て未 build なら build）

## 計画外の判断

- **`subscribe` は呼んだ時点の値より新しい値だけを返す。** 呼んだ時点の値をまず返す案もあるが、docs の「新しい値が書かれるたびに」に合わせた。正常終了を見たら、待たずにもう 1 回だけ新しい値を確かめてから終える（終了の直前に書かれた値を落とさない）。終わる直前にも `check()` して、例外による終了を正常終了と取り違えない
- **recv の受け手のスレッドは照合しない**（send だけ照合）。計画どおりだが、ほかのノードのスレッドから他人の Port を recv すると局所性を破れる。必要なら同じ照合を足せる
- recv の spec を持たない Port の `recv`、send の spec を持たない Port の `send` は `TypeError`。`send_to` に他の Port の Channel を渡すと `ValueError`。送る側は「スレッドの照合 → spec の有無」の順に確かめる
- `load_state_dict` はノードの状態を `copy.deepcopy` してから保持する（同じ木から 2 つのネットワークを復元すると、Tensor を共有してその場の更新が混ざるため）。スロットの値は全部を検査してから restore する（途中の誤りで半分だけ戻らないように）。木の top-level のキーは `nodes`・`slots`・`last_update_ns`・`timer` が必須で、`exception` などほかのキーは無視する
- `run` の `start` が `NetworkError` 以外（引数の誤りなど、ノードを起動する前に断られた場合）で失敗したら、ネットワークを「実行済み」にせず、もう一度 `run` できるようにした。orchestrator の申し送り 3（他のランタイムが動いている間の timer_state 付きの起動を B がエラーにする）の例外は、そのまま利用者に届く
- 例外のときの取得の callback は Session を `weakref` で持つ（Runtime は GC の追跡対象ではないので、強参照だと循環が回収されない）。Binding も Session を参照しない
- `mujou.Node` の `__init__` は `_core.Node.__new__`（任意の引数を捨てる）とシグネチャが違うので、pyright の `reportInconsistentConstructor` を ignore した
- device は CPU と CUDA だけを受け付け、ほかは `ValueError`（docs の「MPS は対象にしない」）
- テストの `run_network` fixture（`tests/mujou/conftest.py`）は走らせたネットワークを最後に `shutdown` する。`virtual_timer` より後に要求すれば、タイマーを戻す前に止まる
- テストで `on_start` だけを見るノードは手動モードで走らせる（GIL ありのビルドで、何もしない自由実行のノードが GIL を回し続けて遅くなるため）

## 確かめたこと

- torch を import しても 3.14t の GIL は無効のまま（`sys._is_gil_enabled()` が False）
- `tests/mujou` を free-threaded で 5 回、`PYTHON_GIL=1` で 3 回: すべて通過
- CUDA のテストは CUDA の無い環境なので skip（未確認）

## 残った懸念・ユーザーへの質問

- `subscribe` が呼んだ後の値だけを返す振る舞いでよいか
- 公開する名前（`exception_state_dict`、`_step` など）、`read` が秒（float）を返し `recv_with_time` が ns（int）を返す非対称でよいか
- hook ごとに Python のスレッド状態が作り直される問題（B のレビュー M1）が直るまで、`_on_thread_start` で設定した Python レベルのスレッドローカルは `on_update` に残らない（torch の grad mode と CUDA stream は C++ の TLS なので残る）
