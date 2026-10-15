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

## レビュー差し戻しへの対応（2026-10-16）

対象: `memory/agents/{code-reviewer,algorithm-reviewer}/core-runtime-D.md`（main checkout 側）。orchestrator の裁定に従った。B の修正（スレッド状態の保持、停止の保持、復元と同時実行の排他、hook を引く位置）を merge した後のブランチで対応した。

### IF の変更

- `_core.pyi` の `Node.__new__` を `*args: Any, **kwargs: Any` にした（code M1）。`_node.py` の ignore を外し、`tests/mujou/test_node_typing.py`（`# pyright: strict`）で利用者のサブクラスと、変数に組んだ dict の `write` が通ることを確かめる
- `Network.write[P: Port[Any, Any, Any]](values: Mapping[P, ...])`（S6）
- `recv` / `recv_with_time` も所有ノードのスレッドだけ（S4 / algorithm should-fix 1）。ホストの公開 IF（`write`・`read`・`subscribe`・`state_dict`・`load_state_dict`・`run`・`shutdown`・`join`・`_step`）をノードの hook から呼ぶと `RuntimeError`。判定は `_node` のスレッドローカルの印で、ノードのスレッドは `_on_thread_start` で立て、制御スレッドは `state_dict` の hook を包んだ間だけ立てる（B の M1 の修正でスレッド状態が hook 間で保たれる前提）
- `subscribe` は `run` の前は `RuntimeError`（N4）
- `load_state_dict` の後、`run` の前の `write` は、全件を検査してから各スロットを木の時刻 T で `restore` する（algorithm must-fix）。読み手がいないので版番号の切り替えは要らない
- `load_state_dict` は `timer` のキー（`time_ns`・`scale`）と型、`last_update_ns` が int かを確かめ、違えば `ValueError`（N6）

### 計画からの変更

- **実行中の Session は強参照の集合 `_RUNNING: set[Session]` で持つ（計画書の `weakref.WeakSet` から変更、code M2）。** 参照を手放しても走り続け、遅くともインタプリタの終了時に止めて `on_stop` を待つ。集合から外すのは、終了を見たとき（`check`・`shutdown`・`join` の後、と次の `run` のときの掃除）。例外で終わったネットワークは、ホストが何かを呼ぶか次の `run` まで集合に残る（終了済みなので atexit の shutdown は `NetworkError` を握りつぶすだけ）。終了を見たときに各ノードの `_thread_ident` を消す（N2）
- 終了時の停止は `threading._register_atexit`（non-daemon のスレッドを join する前に呼ばれる。private な API なので、無ければ `atexit.register`）に登録した（S1）。subprocess のテスト 2 本（slow）: 参照を捨てたネットワークの `on_stop` が完了する／`subscribe` を回す non-daemon スレッドが残ってもプロセスが終わる
- **CUDA の同期（code M3 / algorithm should-fix 2）。** `_on_thread_start` で、CUDA のノードは `torch.cuda.set_device`（N3）→ 自分のストリームを作って current に → `on_update` をインスタンス属性で包み、終わるたびに自分のストリームを `synchronize` する（B が hook を `_on_thread_start`・`on_start`・`load_state_dict` の後に引くので効く）。これで docs/torch.md の「各ステップの終わりに自分のストリームを同期する」が実装と一致した。加えて、状態の取得の build（制御スレッド）でコピーの前に各ノードのストリームを同期し、`load_state_dict` の deepcopy の後に、初期化済みなら全 CUDA デバイスを `synchronize` する（ノードが走る前のホストの操作なので、デバイス全体を待ってよい）。**CUDA の無い環境なので未実測**。コードを読んで、(1) `.cpu()` は current stream で D2H して同じストリームを同期する、(2) 包んだ `on_update` の後はノードのストリームに仕事が残らない、(3) 制御スレッドの既定ストリームでのコピーは、ホストが同期を待った後なので完了済みの値を読む、と判断した
- 例外のときの取得の callback は `weakref.WeakMethod` で Session のメソッドを包む形にした（N10。private 属性への ignore が要らなくなった）。`Runtime.last_update_ns()` が「not started」を投げたら（起動中の例外で、bindings がまだ runtime を持っていない窓、S7）、復元の `last_update_ns` か、その時点のタイマーの時刻（起動の凍結のまま）で埋める
- `run` は check と Runtime の作成を lock の下で行う（N5）
- 実時間の E2E は slow にし、判定を理論から緩めた（S3）。2 段の鎖の誤差は入力の段差 d に対して d(1 + t/τ)e^{-t/τ}。ノードが提示時間の 1/3（t = 10τ）しか回れなかった場合まで許す
- 復元の E2E を (a, b) と (b, a) の両方で回し、復元の直前に a だけを 1 回進めて `last_update_ns` を食い違わせた。復元直後の `state_dict()` が木のスロット（値と時刻）・`last_update_ns`・タイマーと一致すること、復元後のノードのその場の更新が木に届かないこと（deepcopy）を確かめる（algorithm should-fix 3 / N7）
- TestLoadStateDict は仮想時計で走らせる（N8）。例外での購読の終わりは、イテレータで待ち始めてからノードに投げさせる（N9）
- docstring: `write`（load の前の write は置き換わる。S2）、`read`（復元した値。algorithm nit 2）、`shutdown`（build 前は `RuntimeError`。N1）、`device`（確保に `self.device` を使う。algorithm nit 4）、`generator`（復元で状態は戻らない。S5）、`Node`（`state_dict` は制御スレッドで呼ばれ、Port を使えない）
- algorithm nit 1: 自由実行で取った木の `b.next/out` が `nodes.b.x` と一致することを足した
- N11: merge 後の B は、他のランタイムが走っている間の `timer_state` 付きの起動を `ValueError`（何も起動しない）で断る。D の `run` はそれをそのまま送出し、もう一度 `run` できる。docs の文面を「`ValueError`、終了させてから run し直せる」に合わせた
- conftest の `virtual_timer` の `timer.resume()` はホストのスイッチだけを切る。ランタイムの保持は `run_network` の fixture が先に shutdown して解くので噛み合う（fixture の順序の注意を docstring に書いた）

### 記録だけ（N12, N13）

- N12: Port を持たないノードは build の凍結の検査にかからないので、同じインスタンスを 2 つのネットワークで `run` できる（2 つのランタイムが同じノードの hook を呼ぶ）。C の範囲。直すなら、build でノード自体にも「所属済み」の印を付ける
- N13: `torch.enable_grad()` の下で作った非葉の Tensor をノードの状態が持つと、`copy.deepcopy` が失敗し、`state_dict()` がその例外を送出する（ネットワークは動き続ける）。docs の「pickle できれば中身を問わない」より条件が狭い

### 残る懸念

- CUDA の経路はすべて未実測（上記）
- `on_update` をインスタンス属性で包むので、`vars(node)` に `on_update` / `state_dict` が現れる。利用者がそれを見ることはまず無いが、ノードの属性を走査するコードには見える
- 例外で終わったネットワークは、ホストが触るまで `_RUNNING` に残る（メモリだけで、スレッドは止まっている）
- `threading._register_atexit` は private な API

## 再レビュー（r2）への対応（2026-10-16）

対象: `memory/agents/{code-reviewer,algorithm-reviewer}/core-runtime-D-r2.md`（main checkout 側）。orchestrator の裁定どおり。B の simplifier（03b2bca）を含む merge の後で対応した。

### IF の変更（`_core`）

- `_core.Node` にフレームワーク用の hook を 2 つ足した（`crates/mujou-python/src/runtime.rs`、`_core.pyi`）。どちらも他の hook と同じく `start` の最後に bound method を引く
    - `_after_update()`: ランタイムが各 `on_update` の直後に同じスレッドで呼ぶ。既定は何もしない。投げると `on_update` と同じく全体が終了する（hook 名は `_after_update`）
    - `_capture_state()`: 状態の取得（通常と例外のとき）で、ランタイムは `state_dict()` の代わりにこれを呼ぶ。既定は `self.state_dict()` を返す。エラーの hook 名は `state_dict` のまま（`attribute(hook)` で呼ぶ属性だけを差し替える）
- `test_core_runtime.py` の `TestFrameworkHooks`（順序とスレッド、置き換え、エラーの名前）

### mujou.Node の変更

- インスタンス属性で `on_update` / `state_dict` を包むのをやめた（code should-fix 1 の根本対応）。実行後の `vars(node)` に hook が残らず、pickle・deepcopy・2 回目の実行の問題が消える。利用者が hook を置き換えても同期が外れない（code nit 1）
- `_after_update`: CUDA なら自分のストリームを同期する
- `_capture_state`: hook の中の印を立て、CUDA なら `with torch.cuda.stream(自分のストリーム):` の下で、同期 → `state_dict()` → 同期。最初の同期で、`on_update` 以外の hook（`on_start`・`load_state_dict`・`on_pause`）が積んだ仕事も終わる（algorithm must-fix の RAW）。後の同期で、`state_dict` が自分のストリームに積んだコピーが終わってから、フレームワークが既定のストリームで木をコピーする
- `Session._tree` は、コピーの後、build から戻る前（= ノードの再開の前）に、制御スレッドの current stream を初期化済みの全デバイスで同期する（code must-fix 2 / algorithm must-fix の WAR）。例外のときの取得も同じ `_tree` を通る
- `load_state_dict` の同期は、デバイス全体ではなくホストのスレッドの current stream に（algorithm nit 2）
- `device="cuda"` などの添字の無いデバイスは、`__init__` で `torch.device("cuda", torch.cuda.current_device())` に決める。CUDA が使えなければ構築時に `ValueError`（code must-fix 1）。`set_device` は添字付きで呼ぶ

### Session の変更

- `run` は `start` が戻るまで `_run_lock` を持つ。`write` の分岐の判定と木の時刻での restore、`load_state_dict` 全体も同じ lock の下（algorithm nit 1 / code nit 2）。ノードの `on_start` からホストのメソッドを呼ぶと lock の前に `RuntimeError` になるので、デッドロックはしない
- `load_state_dict` は、スロットの組（`(value, time_ns)`）と時刻の型の検査、ノードの状態の deepcopy をすべて済ませてから書き換える（code should-fix 3）
- `_shutdown_all` は `NetworkError` 以外の例外（Ctrl-C など）も捕まえて残りのセッションを止め、最後に最初の例外を送出する（code nit 5）
- `recv` の docstring の Raises に、所有ノード以外のスレッドからの `RuntimeError` を足した（code should-fix 2）

### テスト

- `state_dict` の hook（制御スレッド）からホストのメソッドを呼ぶと `RuntimeError`（algorithm nit 5）
- 壊れた木（先頭のスロットの値を変え、最後のスロットの時刻を float に）で、何も変わらない
- 実行と取得の後、ノードの `__dict__` に `on_update` / `state_dict` が無い
- CUDA が無いときの `Node(device="cuda")` は `ValueError`。CUDA があれば添字が決まる（skip）

### docs

- torch.md: 同期の表とフレームワーク用の hook、取得の同期の順序、添字の決め方。nodes.md: フレームワーク用の hook 3 つを一段落で。execution-model.md: 実装の節の CUDA の項目
- network.md: 「ホストが終了を見るまで」（code nit 3, 4 / algorithm nit 3）、hook からはほかのネットワークのメソッドも呼べず、橋渡しはホストのスレッドか I/O のノードで書く（code nit 6、仕様として維持）

### 記録だけ

- algorithm nit 4: `exception_state_dict` は `_session()` を通らないので hook から呼べる。読み取りだけで待たないので、そのままにした
- CUDA の経路は引き続き未実測
