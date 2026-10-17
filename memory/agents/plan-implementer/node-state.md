# node-state: ノードの状態の自動収集と on_thread_start の公開

依頼: `memory/project_runtime_api_revision.md` の「自動の状態収集」「hook」。skill `solo-dev-cycle` で 1 人で回す。

## 段階 1: 計画

### 公開インターフェース

- `Node.__setattr__(name, value)` / `Node.__delattr__(name)`: 代入のたびに登録を更新する
    - `torch.Tensor`（`nn.Parameter` を含む）か `nn.Module` を代入すると、名前の `_` の有無にかかわらず登録する
    - それ以外の値（`None` を含む）で上書きすると登録を外す。`del` でも外す
    - Port は Tensor でも nn.Module でもないので登録されない
- `Node.state_dict() -> dict[str, object]`: 登録順に `{名前: tensor.detach()}` と `{名前: module.state_dict()}` を返す。登録が無ければ `{}`
- `Node.load_state_dict(state: object) -> None`: 登録した Tensor へ `torch.no_grad()` の下で `copy_`、nn.Module は `load_state_dict`（strict）。どちらもその場で、オブジェクトを差し替えない
    - キーの過不足、Tensor の shape の違い: `ValueError`。nn.Module の中の不一致は torch の `RuntimeError` のまま
    - 先に全部を確かめてから書き換える（途中で失敗して半端に書き換わらない）。nn.Module の中身は torch が確かめる
- `Node.on_thread_start()`: 公開の hook。override するなら `super().on_thread_start()` を呼ぶ
- 内部: `node.state_items(node) -> list[tuple[str, torch.Tensor | nn.Module]]`（救済措置の下地）

### requires_grad / nn.Parameter

- `state_dict` は `detach()` した値（記憶域は共有。framework がコピーする）。`requires_grad` は木に入れない（nn.Module.state_dict の keep_vars=False と同じ）。理由: グラフを持つ非 leaf のテンソルは deepcopy も pickle もできない。`requires_grad` は値ではなく属性の性質で、`__init__` / `on_start` が作り直す
- `load_state_dict` は `torch.no_grad()` で `copy_` する。`requires_grad=True` の leaf にもそのまま書ける。`requires_grad` と leaf であることは保たれる

### テスト観点

- 登録: Tensor / private 名 / nn.Module / Parameter / Port と他の値は除外 / `del` / None で上書き / Tensor を別の Tensor で上書き
- 木: detach 済み・記憶域を共有・pickle できる（非 leaf の Tensor でも）
- 復元: 別ノードに load → 値が一致し、テンソルと Parameter のオブジェクトが元のまま（`is` と `data_ptr`）、`requires_grad` が保たれる
- 例外: キーの不足・余り・shape → ValueError、何も書き換えない
- 実行: 自動の state_dict で取った木から復元して続けた結果が、止めずに続けた結果と一致する（ノード内 BP で nn.Linear を更新するノード）
- hook: `on_thread_start` を override して `super()` を呼ぶノードは設定が効く。Rust のランタイムが `on_thread_start` を呼ぶ

### 触らないもの

`session.py` / `network.py` / `port.py`（並列作業の対象）、`crates/mujou-core/src/slot.rs`。

## 段階 3: 実装での判断

- **登録表は `vars(node)["_state_names"]`（`dict[str, None]` を順序つき集合に使う）**。`__setattr__` を通さずに `vars()` へ直に作る（再帰しない・`super().__init__()` を呼ばないサブクラスでも最初の代入で作られる）。値は `__dict__` に普通に置くので、属性の読み出しは速いまま（nn.Module の `__getattr__` 方式は採らない）
- **上書きの規則**: いまの値が Tensor / nn.Module である名前だけが状態。登録済みの名前に Tensor を入れ直しても順序は変わらない。他の値で上書き・`del` で外れ、後で入れ直すと末尾に付く
- **`state_dict` の戻り値の型は `Any`**。`dict[str, object]` に狭めると、`-> object` で override する既存のノード（テストに 8 箇所）が pyright strict で「互換性のない override」になった。基底の契約は「pickle できる任意の木」なので `Any` にし、`super().state_dict()` を足して使う書き方も通るようにした
- **`load_state_dict` の検証**: キーの過不足と Tensor の shape を、コピーの前にまとめて確かめる（`copy_` はブロードキャストするので shape を見ないと黙って通る）。例外は `Network.load_state_dict` の木の不一致に合わせて `ValueError`。mapping でない・Tensor の位置に Tensor でない値は `TypeError`。nn.Module の中身は torch の strict な `load_state_dict` に任せ、`RuntimeError` のまま（途中まで書き換わりうると docstring に書いた）。dtype とデバイスは `copy_` に任せて合わせる
- **requires_grad / nn.Parameter**: 木は `detach()`（記憶域を共有、`requires_grad` は入れない。nn.Module.state_dict の keep_vars=False と同じ）。理由: グラフにつながった非 leaf のテンソルは deepcopy も pickle もできない（テストで確認）。復元は `torch.no_grad()` の下の `copy_` なので、`requires_grad=True` の leaf にも書けて leaf のまま
- **乱数生成器**は Tensor ではないので集めない。`generator` の docstring を「`super().state_dict()` に `get_state()` を足し、`load_state_dict` で取り出してから `super()` に渡す」に直した
- **`on_thread_start`**: Rust の bindings の名前（`Node` の既定メソッド、`start` での呼び出し）を変え、`just stub` で `_native.pyi` を作り直した。エラーのメッセージは `raised in on_thread_start` になる
- `mujou-core` の `cargo test` が 1 回だけ SIGSEGV で落ちた（slot のテスト、今回は触っていない）。直後に 5 回続けて通った。slot.rs を直している別の作業の対象と思われる

## 救済措置の下地（後の設計の材料）

- `mujou._core.node.state_items(node) -> list[tuple[str, Tensor | nn.Module]]` は、`state_dict` を override していても、mujou が登録した状態を生のオブジェクト（detach も copy もしない）で返す
- 救済で使うときの注意:
    - 例外で止まったノードのスレッドは終わっている。CUDA のノードはストリームの仕事が残っているかもしれないので、`_capture_state` と同じく自分のストリームを同期してから読む
    - 例外の途中の状態は、in-place の更新が途中で止まった値かもしれない（ステップの境界ではない）。「救済した木はステップの境界の状態ではない」と明示する必要がある
    - override した `state_dict` が返すもの（optimizer の状態など）は `state_items` には無い。救済で取れるのは mujou 管理下（自動収集）の分だけになる
    - `requires_grad` のテンソルは grad を持っているかもしれない。救済でも `detach` してから deepcopy する（`state_dict` と同じ）

## 却下した案

- `state_dict` の時点で `vars(self)` を走査して Tensor / nn.Module を拾う: 振る舞いは今回の規則と同じになるが、ユーザーは `__setattr__` のフックを指定した。フックなら代入の時点で登録が決まり、将来「登録時に検証する」余地もある
- nn.Module のように `_parameters` 等へ値を移して `__getattr__` で引く: 属性の読み出しが遅くなり、pyright の型も失う
- nn.Module の中身の不一致も `ValueError` に包む: torch の例外の情報（どのキーか・shape）をそのまま見せる方が分かりやすく、コードも増えない

## 段階 4: 自己レビュー

- diff を通しで読んだ。要求から辿れない行なし。`session.py` / `network.py` / `port.py` / `slot.rs` は無変更
- 性能: `__setattr__` は Python の呼び出し 1 回と isinstance・dict の操作。`on_update` で毎ステップ属性を代入し直すノードでも 1 µs 程度で、状態はその場の更新を推奨している（specs/torch.md）
- 整理点なし

## 段階 5: ドキュメント

- `specs/nodes.md`: hook の表に `on_thread_start`、内部の hook の段落を `_after_update` / `_capture_state` だけに、「状態の自動収集」の節を追加、イメージのコードを既定の state_dict に
- `specs/torch.md`: 実行スレッドの設定の hook 名、「状態」の節（nn.Module / requires_grad を許し、自動で集まる）
- `specs/execution-model.md` / `specs/open-decisions.md` / `memory/project_open_decisions.md`: hook 名だけ
- `docs/en/api/node.md` は `::: mujou.Node` のままで `on_thread_start` が出る（dunder は filter で隠れる）ので変更なし。ユーザー向けの本文は別の工程

## レビュー対応（2026-10-17）

レビュー: `memory/agents/code-reviewer/node-state.md`（request-changes）、`memory/agents/algorithm-reviewer/node-state.md`（approve、should-fix あり）。ユーザーの決定と orchestrator の裁定に沿って直した。

### hook の作り直し（ユーザーの決定、algorithm should-fix 1）

- Rust の `PyNode::start` は `_on_thread_start`（内部）→ `on_thread_start`（公開、既定は何もしない）→ `on_start` →（復元なら）`load_state_dict` の順に呼ぶ。外せない設定（grad の無効化・CUDA のデバイスとストリーム・乱数生成器・`_thread_ident`・`_hooks.inside`）は `mujou.Node._on_thread_start` に戻した。`super()` を呼ぶ規約はなくなった
- 回帰テスト `TestOnThreadStart.test_runs_after_the_settings_without_calling_the_base`: `super()` を呼ばない override の中で、grad が無効・generator が seed どおり・所有スレッドが記録済み・`Network.read` が "host methods" の `RuntimeError` になることを確かめる
- 内部の hook の例外は `raised in _on_thread_start` と出る（利用者のコードではないことが分かるので、名前を公開の hook に読み替えない）

### 復元の後の同期と木の解放（algorithm should-fix 3）

- Rust に内部の hook `_restore_state(state)` を足した（既定は `self.load_state_dict(state)`。`_capture_state` と対称）。ランタイムは復元で `load_state_dict` の代わりにこれを呼び、例外は `load_state_dict` と出る（`attribute()` の対応表）
- `mujou.Node._restore_state` は `load_state_dict` の後に自分のストリームを同期する。利用者が `load_state_dict` を override しても同期は漏れない
- **`session.py` の変更は 1 か所だけ**: `Session.run` の `runtime.start(...)` が成功した直後に `self._states = None`（try の中、`except _native.NetworkError:` の直前）。run-handle と merge するときは、その版の「start の成功の直後」に同じ 1 行を置く。Rust 側は `PyNode.state` を `start` で `take()` して `load_state_dict` の呼び出しの後に手放すので、これで木への参照は残らない
- CUDA は手元に無いので、既定の state_dict / load_state_dict の CUDA のテストは `TestRestore` を `cuda:0` でも回す形にした（GPU の無い環境では skip）

### 自動の状態収集の修正

- code M1 / algorithm should-fix 2（裁定）: 木にあって登録に無い Tensor は、木の値の `clone()` を新しい属性として置く（`setattr` なので登録される）。登録に無いキーの値が Tensor でなければ `ValueError`。登録済みで木に無いものは今どおり `ValueError`。`specs/nodes.md` は裁定の文言に直した。`TestRestore` の `Learner` は `on_update` で初めて `_last` を代入し、運動量としてダイナミクスに効かせたので、復元できなければ一致しない
- S1: `type(self)` の MRO にデータ記述子（property など。`__slots__` の member は除く）がある名前は登録しない。setter が別の名前に Tensor を置けば、そちらが登録される
- S2: `__setattr__` / `__delattr__` を `if not TYPE_CHECKING:` の下で定義した。未宣言の属性への代入が pyright strict で再びエラーになることを、一時ファイルで確かめた
- S3: 登録表は書き換えずに作り直す（copy-on-write）。`copy.copy` したノードとは変更を共有しない。`__copy__` を定義するより短く、登録済みの名前に入れ直す毎ステップの代入では作り直さない
- S4: 不一致のテストは登録順で後ろの `p` やモジュールを崩し、前の `w` が書き換わっていないことを確かめる。`TypeError` の経路も足した
- S5: `state_items` → `_state_items`。専用のテストと specs の記述を外した（下の救済のメモは残す）
- N1: エラーメッセージは空でない分だけ出す
- N2: nn.Module の状態も、キーと shape をコピーの前に `module.state_dict()` と照合する。例外の型はユーザーの決定（torch の例外のまま）に合わせて `RuntimeError` にした（strict な `nn.Module.load_state_dict` と同じ型、メッセージは mujou のもの）。これで、どの不一致でも何も書き換えない
- N5: stride が 0 で大きさが 2 以上の次元を持つ Tensor（`expand` など）は、コピーの前に `ValueError`。それ以外の内部の重なり（`as_strided` で作るもの）は検出しない（torch の公開 API に判定が無い。`copy_` の `RuntimeError` になる）
- N6: 登録表の属性名を `_mujou_state_names` にした
- algorithm nit 1: `.grad` が状態に入らないことを `specs/torch.md` に書いた。nit 2: `on_stop` で `None` にするとキーが消えることを `specs/nodes.md` に書いた。nit 4: 上の CUDA のテスト

### 記録だけ（対応しない）

- code N3: 同じ Tensor を 2 つの名前で持つと状態に 2 回入る。害はない（同じ値を 2 回コピーする）
- code N4: `copy_` は dtype を黙って変換する（`nn.Module.load_state_dict` と同じ）
- algorithm nit 3: `TestRestore` の pickle がコピーの役を兼ねる。deepcopy は `test_network.py` の既存のテストが見ている
- algorithm nit 5: 2 つのノードが同じ Tensor / nn.Module を持つと状態の所有に反する。build で storage の重複を検出する余地がある
- code S8: run-handle を merge した後、`TestOnThreadStart` / `TestRestore` を新しい口（`mujou.testing`、`Run`、`state=`）に書き直す

### 救済措置のメモへの追記（algorithm の指摘）

- 関数名は `_state_items(node)` になった
- CUDA の sticky なエラー（device-side assert など）で止まったときは、コンテキストが使えず、CUDA のテンソルは救済できないことがある
- 救済した木はノードごとに時刻が揃っておらず、`on_exception` / `on_stop` の後の値でもある。通常の復元に渡せる木と型で区別するか、印を付ける
- どのスレッドで読むか、全ノードのスレッドの終了を待つかを決めておく
