# stack #7〜#24 最終形の整合性監査（利用者が読むもの）レビュー

対象: `docs/ja/`・`docs/en/`（API リファレンスと mkdocstrings 設定を含む）、`mkdocs.yml`、`examples/`、`README.md`、`pyproject.toml` のメタデータ。
branch `chore/2026-10-17/consistency`（HEAD 12e2558）。ファイルは修正していない。

## verdict: approve

must-fix は無い。下の should-fix は main に merge する前に直すことを勧める。

## must-fix

なし

## should-fix

1. **`docs/en/concepts/ports.md:45` / `docs/ja/concepts/ports.md:46`**
   - 食い違い: 値に付く時刻を「[global time] の `Timestamp`」と書いている。実際の時刻は実行のタイマーの上の値で、`timer=` を渡せばグローバル時間ではない
   - 正: コード（`Run.read` の docstring「The time is on the run's timer」）と `concepts/time.md:23`
   - 直し方: 「その実行のタイマーの `Timestamp`（既定はグローバルタイマー）」にし、リンク先は `time.md#timers` にする
   - 確信度: 高

2. **`docs/en/api/index.md:3-5`**（ja は en にフォールバックするので両言語に出る）
   - 食い違い: 公開サイトに `!!! note "To be written"` と、作業メモのような「How the reference is organized.」が出る
   - 正: なし（ページを仕上げる）
   - 直し方: placeholder を外し、各ページ（Nodes / Ports and Channels / Networks / Global Time / Errors / Testing）への一覧を書く。あわせて「`mujou.testing` は `import mujou.testing` が要る」と書く（11 を参照）
   - 確信度: 高

3. **API リファレンスのシグネチャ**（`src/mujou/_core/run.py` の `Run.write` / `read` / `subscribe`、`src/mujou/_core/timer.py` の `Timer.now`、`ReadOnlyTimer.now`）
   - 食い違い: 戻り値が `-> _native.Timestamp` と描画される（`api/network`、`api/testing`、`api/time`）。利用者が知る名前は `mujou.time.Timestamp` で、`_native` は private
   - 正: 公開名 `mujou.time.Timestamp`（`docs/*/user-guide/running.md:91`、`docs/*/user-guide/time.md:19`）
   - 直し方: 注釈を `Timestamp`（`from mujou._core._native import Timestamp`）にする。または mkdocstrings の `signature_crossrefs` / `modernize_annotations` で公開名に解決させる
   - 確信度: 高（`site/api/*/index.html` の doc-signature で確認）

4. **API リファレンスの `mujou.Node` のフック**（`site/api/node/`）
   - 食い違い: `on_start`・`on_update`・`on_pause`・`on_resume`・`on_stop`・`on_exception`・`load_state_dict` が「method descriptor」とシグネチャだけで描画され、説明が空。`on_thread_start` だけに説明がある
   - 原因: `_native.pyi` の stub に docstring が無く、`show_if_no_docstring: true` でエントリーだけ出ている
   - 正: クラスの docstring の箇条書き（`src/mujou/_core/node.py:28-56`）に各フックの説明がある
   - 直し方: Rust 側の doc comment に 1 行ずつ足して `just stub` で stub を作り直す（推奨）。または、空のフックをメンバー一覧から外す
   - 確信度: 高

5. **`pyproject.toml:4`（description）と `src/mujou/__init__.py:1-2`（モジュールの docstring）**
   - 食い違い: 「trained with PC-ALM instead of backpropagation」「trained with PC-ALM」と書いている。PC-ALM は未実装で、更新手法は差し替えられ、ノード内の BP も使える
   - 正: `README.md:7`（「will be ... not implemented yet」「pluggable ... including backpropagation confined to a single node」）、`docs/*/user-guide/update-rules.md:9-11`
   - 直し方: description を `README.md:3` の 1 行（`Asynchronous Distributed Deep Learning Library for Continuous Learning`）に揃える。`mujou/__init__.py` の docstring も同じにする
   - 確信度: 高

6. **`pyproject.toml` `[project]` のメタデータ**
   - 食い違い: `classifiers`、`[project.urls]`、`license`、`authors` が無い。`mkdocs.yml` の `repo_url` と、`installation.md:22` のリポジトリ URL（`https://github.com/Geson-anko/mujou`）は、メタデータから辿れない。repo に LICENSE ファイルも無い
   - 正: `mkdocs.yml:3`
   - 直し方: `[project.urls]` に `Repository` と `Documentation` を足す。classifiers に `Programming Language :: Python :: 3.14`、`Programming Language :: Python :: Free Threading`、`Programming Language :: Rust`、`Development Status :: 2 - Pre-Alpha` を足す。license はユーザーの判断なので、要確認事項として上げる
   - 確信度: 高（欠けていること自体は確実。何を足すかは判断が要る）

7. **`docs/en/concepts/nodes.md:75` / `docs/ja/concepts/nodes.md:75`（例の `Leaky.on_start`）**
   - 食い違い: `torch.randn(self.n_in, self.n_out)` が全ノードで共有される既定の生成器を使っている
   - 正: `writing-a-node.md:114`（「Draw random numbers from `self.generator`」）、`torch.md:24-32`、`Node.generator` の docstring
   - 直し方: `torch.randn(..., generator=self.generator)` にする（決定性が要るなら `super().__init__(seed=0)` も足す）。ついでに他の例と同じく `@override` を付ける
   - 確信度: 高

8. **PC-ALM を実装済みのように書く箇所**
   - 箇所:
     - `docs/en/user-guide/torch.md:37` / `docs/ja/user-guide/torch.md:37`: 「The default update rule computes its update equations directly, so it needs no autograd graph」
     - `docs/en/index.md:21` / `docs/ja/index.md:21`: 「The default update rule is PC-ALM ... where each node learns with local updates」
   - 食い違い: どちらも現在形で、実装済みに読める。未実装だと書いているのは `update-rules.md` の note だけ
   - 正: `README.md:7`、`update-rules.md:9-11`
   - 直し方: torch.md は「PC-ALM（予定。[Update rules] を参照）は更新式を直接計算するので autograd は要らない」の形にする。index は表のセルに「（実装予定）」を足す
   - 確信度: 中

9. **`docs/en/user-guide/torch.md:10` / `docs/ja/user-guide/torch.md:10`（`--8<-- "examples/torch_node.py:21:55"`）**
   - 食い違い: 取り込む範囲が `MAX_STEP` の行（21）から始まる。クラスで使う `BATCH, N_IN, N_OUT`（19 行）と `INPUT, OUTPUT`（20 行）が抜け、表示されるコードの中で未定義になる。`MAX_STEP` を含めているので、定数を見せる意図だったと読める
   - 正: `examples/torch_node.py:19-21`
   - 直し方: `19:55` にする
   - 確信度: 中

10. **`docs/en/user-guide/compose.md:26` / `docs/ja/user-guide/compose.md:26`**
    - 食い違い: 「a positional name such as `"0"`」はキーワードに使えず `ValueError` になる、と書いている。実際は `Compose(**{"0": net})` が通り、`"0"` を拒むのは同じ位置の引数があるときだけ
    - 正: コード（`network.py:419` の `name in children`）と `Compose` の docstring（「is the path of a position」）
    - 直し方: 「位置で渡したものの名前（`Compose(a, **{"0": b})` の `"0"`）」と書き直すか、この項目を消す（`0=` は Python の構文上そもそも書けない）
    - 確信度: 高（実行して確認）。深刻度は低い

11. **`mujou.testing` の import**
    - 箇所: `docs/*/concepts/time.md:54`、`docs/*/user-guide/time.md:71`、`docs/*/user-guide/testing.md:9-11`
    - 食い違い: `mujou.testing.VirtualTimer` / `mujou.testing.run` と書くが、`import mujou` だけでは `mujou.testing` が無い（`AttributeError: module 'mujou' has no attribute 'testing'` を確認）。`mujou.time` は `import mujou` で入るので、利用者が同じと思い込みやすい
    - 正: コード（`src/mujou/__init__.py` は `time` だけを import する）
    - 直し方: testing.md の「Tools」の直前に「`import mujou.testing` が要る（`import mujou` だけでは入らない）」と 1 文足す
    - 確信度: 高

## nit

- **`docs/*/concepts/networks.md:49-55`（Run のメソッドの表）**
  - `subscribe` を「a new one is written／新しい値が書かれるたびに」と書いている。ノードが `send` するのが正（`Run.subscribe` の docstring「until a newer message is sent」）
  - `join` が表に無い
  - 確信度: 中
- **`docs/*/concepts/ports.md:67-68`**
  - 「unconnected port に `Run.write` で書ける」とだけあり、相手が 1 つ（single-peer）の Port に限ることが抜けている（`running.md:32` には書いてある。multi の Port に書くと `ValueError` になるのを確認）
  - 確信度: 高
- **API リファレンスのシグネチャに出る内部名**
  - `Port` / `PrevPort` / `NextPort` の `send: MessageSpec[S2] | None = _NO_SPEC`（`MessageSpec` も `_NO_SPEC` も公開されていない）
  - `Node.connect` / `Network.connect` の `other: Component`
  - `show_source: true` で、`RunCompose` などの内部名がソース表示に出る
  - 確信度: 高
- **`docs/*/user-guide/torch.md:63`**
  - 利用者向けの文書で、内部フック `_after_update` の名前を出している。実際のエラーメッセージに出る名前なので、出すこと自体は正しい
  - 「内部のフック」と一言添えるとよい
  - 確信度: 中
- **`examples/save_and_restore.py:17`**
  - `SCALAR = TensorSpec((1,))` は shape が `(1,)`。他の例では `SCALAR` が `()`（`errors.py:17`、`subscribe.py:16`、`timers.py:17`、`leaky_chain/leaky.py:10`）
  - `nn.Linear(1, 1)` に合わせるなら `VECTOR1` か `ONE` などに改名する
  - 確信度: 高
- **ja と en の小さな差**（内容の矛盾は無い）
  - `concepts/ports.md`: en の 41 行目は 1 つの箇条。ja は 41・42 の 2 つに分けている
  - `concepts/nodes.md:34`: ja だけ「（例: torch のスレッドごとの設定）」がある
  - `concepts/time.md:17`: en は「Do not cap `dt`: ... falls behind the dynamics」まで書く。ja は「打ち切ると経過時間が失われる」だけ
  - `examples/index.md:16` と `:29`: subscribe の説明は ja のほうが詳しい（「途中の値は飛ばしうる」と、飛ばす仕組みの説明）

## 提案（好みの範囲）

- **ja が見出しへのリンクを張っていない**
  - en の `writing-a-node.md:42` と `building-a-network.md:30` は `#branching-and-merging` へのリンクを張る。ja は「（…の「分岐と合流」）」と書くだけでリンクが無い
  - slugify は Unicode を残すので、`building-a-network.md#分岐と合流` で張れる
- **ja の用語の揺れ**
  - `Tensor` と `テンソル`: 両方が混在する
  - `seed` と `シード`: `torch.md:13` は「デバイスと seed」、`writing-a-node.md:102` は「シードとデバイスを決める」
  - `仮想のタイマー`（4 回）と `仮想タイマー`（`examples/index.md:31`）
  - `誤差逆伝播法`（`index.md`、`concepts/index.md`）と `誤差逆伝播`（`torch.md:3`、`update-rules.md:6`）
  - pause の訳: `一時停止`（`quickstart.md:65`）と `停止`（ほかの全箇所）
  - `concepts/nodes.md:43` の「ネットワークが停止したとき」（on_pause）は stop と読めるので、「一時停止」に統一するほうが誤読が少ない
- **en のリンク文字列の大文字小文字**
  - `examples/index.md` と `quickstart.md` は Title Case（「Writing a Node」）
  - ほかのページは sentence case（「[Writing a node]」「[Global time]」）
- **書き方の不統一**
  - 例の import の書き方: `from mujou import NextPort` と `mujou.NextPort` が混在する（`concepts/nodes.md:67-70`、`writing-a-node.md:84-87`）
  - `concepts/nodes.md` の `Leaky`（行列の `h`）は、`examples/leaky_chain/leaky.py` の `Leaky`（スカラーのリーク積分器）と同じ名前で別物。名前を変えると紛れない
- **API リファレンスへの導線**
  - どの文書ページからも API リファレンスへのリンクが無い
  - `index.md` の「Where to go next」の表に API Reference の行を足す
- **README の Development 節**
  - Python 3.14 以上、開発は free-threaded 版 3.14t で行う（`.python-version`）、という前提が書かれていない
  - 1 行足すと `installation.md` と揃う

## 確かめた事実（食い違いが無かったもの）

- **公開 API と文書の一致**
  - シグネチャ `mujou.run(component, *components, timer=None, inputs=None, state=None, pause_timeout=30.0)` は文書（`running.md:18-24`）と一致する
  - `Node.__init__(*, device="cpu", seed=None, period=None)` は `writing-a-node.md:104-112` と一致する
- **パスの規則**
  - `mujou.run(node)` → `""` / `"prev"` / `"next/out"` を実行して確認した。`Compose` の名前の付き方も確認した
- **エラーの型**（文書どおり）
  - `connect` の spec 不一致は `TypeError`
  - single-peer の Port への 2 本目、クラス属性の Port、ノードがノードを持つ: いずれも `ValueError`
  - build 後の `connect` は `RuntimeError`
  - 2 回目の run は `RuntimeError("... has already run; create a new instance to run again")`
  - クラス属性 `period` は `TypeError`
  - `__init__` で `self.timer` を読むと `RuntimeError`
  - state の不一致と、使用中のタイマーへの復元は `ValueError`
  - 接続済みの Port、または multi の Port への write は `ValueError`
  - 受け取るだけの Port の read は `ValueError`
  - 終了後の write は `RuntimeError`
  - `Timestamp` の負の差は `ValueError`。`Timestamp` は位置引数を受けない
  - `NetworkError` は `RuntimeError` の派生、`PauseTimeoutError` は `TimeoutError` の派生
- **`--8<--` の取り込み範囲**
  - `quickstart.py:43:51` / `54:62`、`branching.py:19:78` / `81:84`、`compose.py:38:59`、`subscribe.py:31:41`、`timers.py:31:34` / `36:54` / `56:61`、`torch_node.py:58:60`、`errors.py:19:21` は、どれも意図した行を取り込んでいる（`torch_node.py:21:55` は should-fix 9）
- **testing.md の保存と復元のスニペット**: 実行してビット一致を確認した
- **README**
  - `just` のレシピ名（setup / run / docs / docs-build / container）と、`just run` = format → test → type → docs-build は justfile と一致する
  - Rust 1.85 は `Cargo.toml` の `rust-version` と一致する
  - `mujou.__version__` は `0.0.1`
- **docstring**: `src/` に日本語は無く、英語で統一されている
- **mkdocs**: `nav_translations` が nav の全項目を覆っている。ja の API はフォールバックで生成されている

## 検証結果

- `uv run mkdocs build --strict`（出力先は scratchpad）: pass（en / ja とも警告なし）
- `examples/*.py` と `examples/leaky_chain/main.py` を全部実行した: すべて exit 0
- `just run`: 未実行（監査の範囲は読み取りと、サンプルの実行・docs のビルドに限った）
