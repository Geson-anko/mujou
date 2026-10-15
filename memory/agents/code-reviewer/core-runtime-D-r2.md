# core-runtime WP-D 再レビュー（差し戻しへの修正）

対象: worktree `.claude/worktrees/run-network`、`feature/2026-10-16/run-network`。修正コミット 84a9225（src とテスト）・021100e（docs）・054d485（判断ログ）と、その後の WP-B/C の merge（e00bcea）。WP-D 全体は `git diff feature/2026-10-16/network-structure..HEAD` で見た。

## verdict: request-changes

前回の must-fix のうち M1・M2 と、algorithm-reviewer の must-fix は正しく直っている。S1〜S8 と N1〜N11 も、M3 のほかは意図どおりに直っていて、テストも足りている。ただし M3（CUDA の同期）の修正で新しいバグが 1 つ入り、同期も 1 か所足りない。どちらも CUDA の経路だけの問題だが、`device="cuda"` のノードが起動できなくなるので、マージは止める。

前回の指摘ごとの判定:

- 直った: M1、M2、S1〜S8、N1、N2（ホストが終了を見たときだけ消える。nit 4 を参照）、N4〜N11、algorithm-reviewer の must-fix、should-fix 1・3、nit 1〜4
- 一部だけ直った: M3 と algorithm-reviewer の should-fix 2。取得のときに、コピーの後の同期が無い（must-fix 2）
- 直したことで壊れた: N3。`set_device` が添字の無いデバイスで落ちる（must-fix 1）
- 記録だけにした: N12、N13（orchestrator の裁定どおり）

## must-fix

### M1. `Node(device="cuda")`（添字の無い CUDA デバイス）だと `_on_thread_start` で必ず `ValueError` になり、ネットワークが起動しない

- 対象: `src/mujou/_node.py:115`（`torch.cuda.set_device(self._device)`）
- 問題: `torch.cuda.set_device` は、添字の無い `torch.device("cuda")` を受け付けない。`Node.__init__` はデバイスを正規化しないので、ドキュメントの書き方（`device="cuda"`）をした CUDA のノードはすべて、起動の最初の hook で落ちる。リポジトリで唯一の CUDA のテスト `tests/mujou/test_port.py:376`（`Node.__init__(self, device="cuda")`）も、CUDA のある CI では落ちるはずである（ここでは skip されている）
- 根拠（再現）: torch 2.14.1 で `torch.cuda.set_device(torch.device("cuda"))` を呼ぶと、CUDA の初期化より前に `ValueError: Expected a torch.device with a specified index or an integer, but got:cuda` が出た（`torch/cuda/_utils._get_device_index(optional=False)`）。`torch.cuda.Stream(device)` と `torch.Generator(device)` は添字が無くても通る。前回の N3 に対応した変更で入ったバグで、`docs/torch.md` の表（「`device="cuda"` と書いた確保や演算が、ノードのデバイスに載る」）とも食い違う
- 直し方の方向（参考）: 添字が無いときは `set_device` を呼ばない、`optional=True` で添字を解決する、`__init__` で正規化する、のどれか
- 確信度: 高 / 深刻度: 高（CUDA のノードがすべて動かない）

### M2. 状態の取得で、制御スレッドのコピーが終わるのを待たないままノードを再開している（M3 の修正が半分）

- 対象: `src/mujou/_session.py:415-435`（`_tree`。`synchronize(node)` はコピーの前だけ）と、`crates/mujou-core/src/runtime.rs` の `capture`（`build` が戻るとすぐ `resume_after_pause`）
- 問題: `copy.deepcopy` は CUDA の Tensor を `UntypedStorage.clone()` → `copy_` で複製する。デバイス内のコピーなので、制御スレッドの current stream（既定のストリーム）に非同期に積まれるだけで、ホストは待たない（`torch/_tensor.py` の `__deepcopy__` と `torch/storage.py:262` を確認した）。`_tree` が戻るとランタイムはすぐノードを再開する。ノードの `on_update` は自分のストリーム（プールから取ったもので、既定のストリームとは暗黙に同期しない）で重みをその場で更新するので、まだ走っていない clone より先に書き換えうる。その結果、木に次のステップの途中の重みが入る。キャッシュアロケータの側にも同じ形の危険がある。clone が読み終わる前に、ノードが元の Tensor を手放す（`self.w = new`）と、そのブロックがノードのストリームで再利用されうる（`record_stream` も同期も無い）
- 根拠: `docs/execution-model.md` と `docs/torch.md` は「木のコピーを取る前に各ノードのストリームを同期する」とだけ書いている。復元の側（`load_state_dict` の deepcopy の後の `_synchronize_cuda()`）は正しく後で待っているが、取得の側には対応する同期が無い。CUDA の無い環境なので実機では再現していない
- 直し方の方向（参考）: `_tree` のコピーの後（`build` が戻る前）に、使っている CUDA デバイスの制御スレッドの current stream を同期する。例外のときの取得も同じ `_tree` を通るので、まとめて直る
- 確信度: 中（PyTorch の実装を読んで判断した。実機では確かめていない）/ 深刻度: 高（チェックポイントが、何の知らせも無く壊れる）

## should-fix

### S1. `_on_thread_start` がインスタンス属性で hook を包むので、実行後もノードの `__dict__` に local のクロージャが残る（pickle できない、deepcopy した複製が元のノードを指す、走らせるたびに二重に包む）

- 対象: `src/mujou/_node.py:129, 143, 157`（`self.state_dict = marked` は CPU のノードにも付く。`self.on_update = on_update` は CUDA のノードだけ）
- 問題と根拠（scratchpad の probe_wrap.py で再現した）
  1. 実行の後、`vars(node)` に `state_dict` が現れる。Port を持たないノードは、`pickle.dumps(node)` が `PicklingError: Can't pickle local object Node._mark_state_dict.<locals>.marked` で失敗する（実行の前は通る）。Port を持つノードは、もともと build の後は `_core.Outlet` のために pickle できないので、新たに影響を受けるのは Port を持たないノードだけである
  2. `copy.deepcopy(node)` は関数をそのまま共有する。そのため複製の `state_dict()` は元のノードの状態を返す（複製で `n=999` にしても `{'n': 1}` が返った）。何の知らせも無く誤った値を返す
  3. 同じインスタンスを 2 回走らせると（N12 の経路。Port を持たないノードは 2 つのネットワークに入れられる）、`marked` の中に `marked` がある形で二重に包まれる。走らせるたびに鎖が伸びる
  4. `state_dict` を `vars(self)` や `self.__dict__` から作る利用者のコードには、クロージャが見える
- 案: 終了を見たときに（`end_thread` と同じ場所で）2 つのインスタンス属性を消す。あるいは包む前に、すでに包んであるかを確かめる。最低限、docs に「実行中と実行後のノードは `__dict__` に hook の包みを持つ」と書く
- 確信度: 高（事実は再現済み）/ 深刻度: 低

### S2. `Port.recv` / `recv_with_time` の docstring の Raises に、所有ノード以外のスレッドから呼んだときの `RuntimeError` が無い

- 対象: `src/mujou/_port.py:217-219`（Raises の欄は「RuntimeError: The network is not built.」だけ）と、`recv_with_time`（「as recv」）
- 問題: 84a9225 で recv も所有ノードのスレッドだけにしたが、docstring の Raises は直っていない。`send` の docstring（同じファイルの 241 行目）は直っている。`run` の前にホストから `recv` しても、build の後なら同じ `RuntimeError` になる
- 確信度: 高 / 深刻度: 低

### S3. `load_state_dict` はスロットの時刻の型を事前に確かめないので、壊れた木を渡すと一部のスロットだけが復元されたまま `TypeError` になる

- 対象: `src/mujou/_session.py:396-402`（コメントは「Check every value before restoring any, so that an error changes nothing」）
- 問題: 値は事前に `split` で確かめているが、`time_ns` は確かめていない。復元のループで `slot.restore(tensors, 1.5)` が `TypeError` を投げた時点で、前のスロットはもう書き換わっている。docstring の「ValueError: The tree does not match」とも違う。N6 で `timer` と `last_update_ns` には確認を足したが、スロットの時刻が抜けている
- 根拠（再現、scratchpad の probe_load.py）: 先頭のスロットの値を 7 に、最後のスロットの時刻を 1.5 にした木を渡した。`TypeError: 'float' object cannot be interpreted as an integer` が出た後、先頭のスロットは `tensor([7., 7.])` のままだった
- 確信度: 高 / 深刻度: 低（壊れた木を渡したときだけ起きる）

## nit

- N1 `src/mujou/_node.py:26` と `docs/torch.md` の「on_update の包み直し」: 利用者が `on_start` や `load_state_dict` の中で `self.on_update = ...` と置き換えると、ランタイムはその後に hook を引くので、CUDA の同期の包みが外れる。docstring の「Overriding `on_start` does not skip this」は、置き換えまでは保証していない。この制約を一言書く（確信度: 中）
- N2 `src/mujou/_session.py:280-286`（write の T の経路）と `load_state_dict`: どちらも `_runtime is None` を `_run_lock` の外で確かめる。別のホストのスレッドが同時に `run()` すると、ノードの `on_start` が読んでいる間に `InputSlot.restore` が走りうる。stub の「Call it only while no node runs」に反し、複数の Port への書き込みが一斉に切り替わらない。条件は 2 本のホストのスレッドだけ（確信度: 低）
- N3 `src/mujou/_session.py:462-463`: `_synchronize_cuda` の docstring「only while no node runs」は、ほかのネットワークが走っているときは成り立たない。その場合はそのネットワークの GPU の仕事も待つ（待つのはホストだけで、正しさには影響しない）。`docs/torch.md:40` の「デバイス全体を待つ synchronize は使わない」はノードへの規則だが、フレームワークの例外として一言あるとよい（確信度: 中）
- N4 `docs/network.md:245` の「終了したネットワークのノードはスレッドを忘れる」と、`:262` の「フレームワークが終了まで参照を持ち」: 実際には、ホストが終了を見るまで（`check`・`shutdown`・`join`、または次の `run`）である。例外で終わり、ホストが触らないネットワークは `_RUNNING` に残り、`_thread_ident` も残る。実害は無いが、文面が実装より強い（確信度: 中）
- N5 `src/mujou/_session.py:488-491`: `_shutdown_all` が握りつぶすのは `NetworkError` だけである。ほかの例外（待ちの間の `KeyboardInterrupt` など）が 1 つ出ると、残りのセッションを止めないまま `threading._shutdown` の外へ抜け、non-daemon のスレッドの join も飛ばされる。scratchpad で、終了時の `on_stop` の待ちの間に SIGINT を送ってみたが、再現しなかった（rc=0、両方の `on_stop` が完了した）。コードを読んでの懸念にとどまる（確信度: 低）
- N6 `src/mujou/_network.py` の `_session()`: hook の中からは、どのネットワークのホスト IF も呼べない（自分のネットワーク以外も含む）。docs とは一致している。ノードがほかのネットワークへの橋渡しをする使い方を禁じることになるので、意図した仕様であることを orchestrator が確認しておくとよい（確信度: 低。仕様の確認）

## 検証結果

- just run: pass（pre-commit の全 hook が通過、cargo test が通過、pytest は 279 passed / 1 skipped（CUDA）、pyright は 0 errors、coverage は `_node.py` 84%（CUDA の経路）・`_session.py` 97%。実行後の `git status` は clean）
- 繰り返し実行（slow を含む `tests/mujou`、`--no-cov`）: free-threaded 3.14t で 25 回、`PYTHON_GIL=1`（`sys._is_gil_enabled()` が True になることを確認した）で 25 回を、2 プロセス並行で回して、すべて 258 passed / 1 skipped だった。さらに負荷をかけて、`test_network.py`・`test_port.py`・`test_node.py` を 6 プロセス並行で各 8 回（GIL ありと free-threaded を半分ずつ）回し、48 回すべて pass だった。flaky は見られない
- `</content>` などの混入: diff の全ファイルを grep して無し
- 追加の確認（scratchpad の probes/。worktree は変更していない）
  - `torch.cuda.set_device(torch.device("cuda"))` が `ValueError` になる（M1）
  - 実行後のノードが pickle できず、deepcopy した複製の `state_dict` が元のノードの値を返し、2 回目の実行で二重に包まれる（S1）
  - 壊れたスロットの時刻で、一部だけ復元される（S3）
  - 例外で終わって参照を手放したネットワークと、走ったまま参照を手放したネットワーク 2 つがある状態で終了しても、rc=0 で、両方の `on_stop` が完了した（強参照と `_register_atexit` は、GIL あり・なしのどちらでも意図どおりに動く）
