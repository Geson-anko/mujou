# api-polish: Run.write が時刻を返す / mujou.run がノードを受け付ける

ブランチ `feature/2026-10-17/api-polish`（`feature/2026-10-17/node-period` の上）。skill `solo-dev-cycle` で 1 人で回す。

## 段階 1: 計画

### 公開インターフェース

- `Run.write(values) -> mujou.time.Timestamp`: 書いた値に付いた時刻（その実行のタイマー上）。1 回の write の全 Port に同じ時刻。空の `values` でもその時刻を返す（何も変えない write の時刻）
- `mujou.run(*components: Node | Network, timer=..., inputs=..., state=..., pause_timeout=...) -> Run`
    - ネットワーク 1 つ: そのまま走らせる（従来通り。パスは `"l1"` など）
    - ノード 1 つ: `Compose(node)` として走らせる。ノードのパスは `"0"`、Port は `"0.prev"` など
    - 複数: `Compose(*components)`。ネットワークとノードを混ぜてよい
- `mujou.testing.run`: 同上
- `Compose(*components: Node | Network, **named: Node | Network)`: ノードも受け付ける

### 内部の write に当たる経路

- `inputs`（`state` なし）: `InputGroup.write` で 1 回の write。時刻はタイマーの今
- `inputs`（`state` あり）: 各スロットを木の時刻 T で restore。全 Port に同じ T（従来通り）
- Rust: `core::InputGroup::write` が `Result<Timestamp, E>` を返す。`PyInputGroup.write` は `Timestamp` を返す。`Session.write` / `Run.write` がそれを返す

### テスト観点

- write の戻り値 == ノードが `recv_with_time` で見る時刻 == state_dict の `/in` の時刻 == VirtualTimer の now
- ノードが受けた時刻を送り、`read` で読む値に write の時刻が入ること
- `inputs` の時刻: state なしは開始時のタイマーの今、後の write はそれ以上。state ありは T
- run(node): 入出力（write/read）、木のパス `"0"`, `"0.x/in"`、1 回だけ走る、断られた run のやり直し、ネットワークとの混在のパス、Compose がノードを受け付ける
- 型以外の値は TypeError

### 判断

- ルートのノードのパスを `""` にしない理由: Port のパスが `".prev"` のように `.` で始まり、木のキーが崩れる。`Compose(node)` と同じ規則（`"0"`）にすれば `run(node)` は `run(Compose(node))` と同じで、`run(a, b)` = `run(Compose(a, b))` の規則と一貫する。ネットワーク 1 つだけ特別にそのまま走るのは、ネットワークがすでにルートになれる木だから
- 混在を許す理由: `Compose` が Component を属性に持てば、ノードもネットワークも同じ規則で歩ける（ネットワークの属性にノードを置くのと同じ）。制限する理由がない
- ノードを 2 回走らせると: 2 回目は新しい Compose の build がノードの「構築済みの印」に当たり `RuntimeError`（already built）。ノード 1 つの run はラップの Compose が自前なので、断られたら unbuild してやり直せる

## 段階 2: テスト

- `tests/mujou/_core/test_run.py`: `TestWrite` に時刻のテスト 4 本（返す時刻 = VirtualTimer の now = `recv_with_time` = `/in`、`Echo` ノードが受けた時刻を送り `read` の値で一致、`inputs` の時刻（開始時）より後、復元後は T から）。`TestRunANode`（パス `"0"`、木のキー、同じクラスのノードで復元、1 回だけ、断られた run のやり直し、ネットワークとの混在）。同時呼び出しのテストをノードでもパラメタ化
- 仕様変更で既存テストを更新: `test_takes_only_networks` → `test_takes_only_nodes_and_networks`（ノードは TypeError でなくなった）、`TestCompose.test_rejects_what_is_not_a_network` → ノードを受け付けるテスト + 非 Component の TypeError
- `conftest.py` の Protocol の型を `Node | Network` に
- Rust / native: `InputGroup.write` の戻り値の検査を追加

## 段階 3: 実装の計画外の判断

- `Compose.__init__` の注釈 `Node | Network` を `inspect.signature` が評価する（3.14 の遅延評価）。`network.py` の `Node` は TYPE_CHECKING の import だけだったので NameError。`network.py` の末尾で実行時に `from mujou._core.node import Node` する形に変えた（node.py は Component を network.py から読むので、末尾なら循環しない。`mujou/__init__.py` が network を node より先に読む）。これで `_build` の関数内 import が不要になったので消した
- `start` の分岐は `match networks: case [Network() as network]` でネットワーク 1 つだけ包まない
- 引数名 `*networks` は変えない（ノードも入るが、改名は今回の要求外。確認事項に回す）

## 段階 4: 自己レビュー

- diff を通読。要求外の変更なし（`network.py` の import の移動は上記の理由で必要）
- 「断られた run のやり直し」: ノード 1 つの run は自前の Compose なので unbuild される → どの形でも走らせ直せる（テストあり）
- 構築済みの印: ノードの 2 回目は `Solo(node)` / `Compose(node)` / `node` のどれでも "already built"（テストあり）
- 時刻: `state` ありの `inputs` は従来から全スロットに同じ T。公開の戻り値はない（`mujou.run` は `Run` を返す）ので、`state_dict` の `/in` とその後の `write` の時刻で一貫を確かめた

## 段階 5: ドキュメント

- docstring: `Run.write`, `mujou.run`, `Compose`
- `specs/network.md`: 操作表、複数のネットワーク（ノードの節）、付く時刻の後に write の戻り値、実装の節、ホストの入出力表
- `docs/en/api/*.md`: `:::` は既存（`mujou.run` / `mujou.Compose` / `mujou.Run`）で変更不要。user-guide はスタブで docs-content の担当なので触らない
- memory: `project_run_node_and_write_time.md` と MEMORY.md の 1 行

## 確認事項（ユーザーへ）

- `mujou.run(*networks)` / `Compose(*networks)` の引数名: ノードも受けるようになったが `networks` のまま
- ノードの 2 回目の run のエラーは "already built"（ネットワークの "already run" と文言が違う）

## レビュー後の修正（code-reviewer: approve、should-fix 2 件と nit 1-5）

- 2 回目の run の文言: 構築済みの印（`_BUILT`）に、木ごとに共有する記録 `_Tree(by_run, ran)` を持たせた。Graph の `_ran` もこれに移した。`mujou.run` が作る Compose は private な `RunCompose` にして、`by_run` を立てる。印に当たった部品の木が走った、または `by_run` なら `"<クラス名>[ at '<path>'] has already run; create a new instance to run again"`。ネットワーク 1 つの 2 回目（`start_once`）も同じ文言にそろえた（旧 "the network has already run; build a new instance to run again"）。`mujou.run` に渡したもの（`RunCompose` の直下）にはパスを付けない
- `by_run` も見るのは、並行の run(node) で、片方の Compose の build から `start_once` までの間に、もう片方が印に当たる窓があるため。`RunCompose` は断られたら unbuild されるので、印が残っていれば走らせるために構築されている
- 記録は部品を参照しない小さなオブジェクトにした（Graph への強参照を `_BUILT` に置くと、部品が解放されなくなる）
- テスト: 並行の run の文言をパラメタで厳密に、`TestRunANode.test_runs_once` を 3 形の文言で、走っていない構築済みのネットワークのノードは "belongs ..."、`write({})`、`Solo(node)` での走らせ直し、混在の euler のオラクル
- nit 6（引数名）は変えない

## 再レビュー後（approve、orchestrator の裁定）

- 並行の run の窓（should-fix 1）は実装で塞がず、仕様を挙動に合わせた: 断られた側の文言は "has already run" か "belongs to a network that is already built" のどちらか、走るのは 1 回だけ。並行のテストに `run(net) | run(net, Chain())` と `run(node) | run(Solo(node))` を追加
- nit 1: 並行の run では、相手が後で断られてもこちらが断られることがある（走らせ直せる）と仕様に追記
- nit 2: 文言のパスは mujou.run が作る木でのパス（`'0.x'`）を正とし、仕様とテストを追加
- nit 4: `mujou.run` の Raises を今の文言に合わせた
