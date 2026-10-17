# api-polish 再レビュー（2 回目の run の文言 / `_Tree` / `RunCompose`）

対象: `git diff d26f2ed..088c971`（1c253b0, 6e4439d, 4ddb9ab, 088c971）

## verdict: approve

must-fix はない。should-fix 1 は並行の run で文言が仕様からずれる件で、1 回だけ走ること、断られた run をやり直せることは保たれている。orchestrator の裁定で仕様の書き方を絞るか、実装で塞ぐかを決める。

## must-fix

なし

## should-fix

1. `src/mujou/_core/network.py:238, 256-268` と `src/mujou/_core/run.py:343-349`（単独のネットワークの run が起動を終えるまでの窓）/ 確信度: 高（再現した）/ 深刻度: 低〜中
   - 問題: `by_run` を立てるのは `RunCompose` だけである。`mujou.run(net)` のようにネットワークを 1 つだけ渡すと、`net.build()` は `by_run=False` の `_Tree` を作る。`tree.ran` が立つのは `start()` が戻った後（`start_once` の末尾）だ。そのため、build から起動完了までの間に同じ木の一部を含む別の run が来ると、文言は "has already run" にならず `"... belongs to a network that is already built"` になる。ユーザーの木（`Solo(node)` など）を走らせている間の `run(node)` も同じである。実装担当の説明「`by_run` で並行の窓を塞ぐ」が効くのは `RunCompose` の木だけ
   - 根拠: `specs/network.md:255` は「走った（**または走らせるために `mujou.run` が構築した**）木の一部をもう一度構築したときの文言は … `has already run` にそろえる」と書く。`mujou.run(net)` が呼ぶ `net.build()` も「走らせるために `mujou.run` が構築した」に当たる。2 スレッドを Barrier でそろえ、30 回ずつ再現した
     - free-threaded: `run(net) | run(net, Chain())` は 30/30 で `Chain belongs to a network that is already built`。`run(node) | run(Solo(node))` は 27/30 で `Solo` が走り、`run(node)` 側が `Leaky belongs to a network that is already built`。`run(net) | run(net.a)` は 11/30 で `Leaky belongs to ...`
     - `-X gil=1`: `run(node) | run(Solo(node))` は 29/30 で同じ。`run(net) | run(net, Chain())` は 3/30
     - 走り終えた後に逐次で呼べば、どれも `has already run` になる。既存のテスト `test_runs_once_under_concurrent_calls` は同じオブジェクトを並べるだけなので、この組み合わせは検査していない
   - 補足: 実装で塞ぐ場合でも、利用者が先に `build()` だけ呼んだ木は build の時点では印を立てられない。`start_once` の入口で「起動中」を立てるなど、build の外で立てる必要がある。仕様を「同時に呼んだときは `belongs ...` になることがある」と緩める道もある

## nit

1. 並行の run で、起動しなかった run を「走った」と言うことがある（`run.py:343-366` と `network.py:258`）/ 確信度: 高（再現した）
   - 同じノードに `run(k, inputs=<形の違う値>) | run(k)` を同時に投げると、free-threaded の 30 回中 1 回で両方が断られた。前者は `ValueError`、後者は `Leaky has already run; ...` で、実際には何も走っていない。ネットワークで同じことをすると（`run(net, bad) | run(net)`）、後者が `graph._lock` で待ってから走る。ノードは待たずに断られるので、挙動がそろわない。断られた後の逐次のやり直しはできる（失敗した `RunCompose` は unbuild され、負けた側の `_build` は印を付ける前に例外で抜ける）。拒否そのものは前回からあり、変わったのは文言が事実と食い違う点だけ

2. `mujou.run` に複数渡したとき、利用者の木の中のパスに `RunCompose` の位置 `'0'` が付く（`network.py:259-261`）/ 確信度: 中
   - `run(Outer(net), Chain())` は `Chain at '0.x' has already run`、`run(Solo(net.a), Chain())` は `Leaky at '0.node' ...` になる。仕様は「`mujou.run` が作った `Compose` のパス（`'0'` など）を付けず」「利用者が作った木の中にあるものはそのパスを付ける」と書き、実装担当は「その木でのパス」と説明している。`'0.x'` は実行の `state_dict` のキーとは一致するが、利用者の木でのパス `'x'` ではない。一方、同じクラスを 2 つ渡した `run(b, a)`（`a` は走り済み）は `Chain has already run` で、どちらの引数かが分からない。どちらに倒すにしても、仕様に一文足すとよい。この組み合わせのテストもない

3. `build()` だけ呼んだネットワークを他とまとめて渡すと `Chain belongs to a network that is already built` になる（`network.py:266`）/ 確信度: 中
   - 主語の `Chain` 自身が構築済みの木の根なので、「構築済みのネットワークに属する」は読みにくい。前回までの `Chain at '0' belongs ...` と同じ構文で、今回の変更による退行ではない

4. `src/mujou/_core/run.py:296-299`（`mujou.run` の Raises）/ 確信度: 低
   - 「a node that has run」を "belongs to another built network" の例に入れているが、今の文言は `has already run` である。誤りではないが、文言をそろえた趣旨に合わせて書き分けてもよい

## 前回の指摘の解消状況

- should-fix 1（2 回目の文言）: 解消。逐次の全組み合わせで仕様どおりになった（下表）。並行の窓は上の should-fix 1
- should-fix 2（空の write のテスト）: 解消（`TestWrite.test_of_nothing_returns_the_time_and_changes_nothing`）
- nit 1（並行テストの緩さ）: 解消（文言をパラメタにして `==` で比べる）
- nit 2（別の形で走らせ直す）: 解消（`Solo(node)` で走らせ直し、木のキーも確認）
- nit 3（混在のオラクル）: 解消（`euler` を合成して `torch.equal`）
- nit 4（末尾の import の順序の前提）: コメントで解消
- nit 5（Graph の docstring）: 解消
- nit 6（引数名）: 意図して据え置き（ユーザーに確認中）

## 確認した観点（問題なし）

逐次の文言（`mujou.testing.run` で再現）:

| 操作 | 文言 |
| --- | --- |
| `run(net)` の後に `run(net)` | `Chain has already run; ...`（`start_once`） |
| `run(net)` の後に `run(net, other)` | `Chain has already run; ...` |
| `run(net)` の後に `run(Outer(net))` | `Chain at 'x' has already run; ...` |
| `run(net, other)` の後に `run(net)` / `run(net.a)` | `Chain has already run` / `Leaky has already run` |
| `run(net, other)` の後に `run(Solo(net.a))` | `Leaky at 'node' has already run` |
| `run(node)` の後に `node` / `Solo(node)` / `Compose(node)` | パスなし / `at 'node'` / `at '0'`（テストあり） |
| `run(Solo(node))` の後に `run(node)` / `run(Outer(solo))` | `Leaky has already run` / `Solo at 'x' has already run` |
| `net.build()` の後に `run(net)` | 走る |
| `net.build()` の後に `run(net.a)` / `run(Solo(net.a))` / `run(Outer(net))` | `Leaky belongs ...` / `Leaky at 'node' belongs ...` / `Chain at 'x' belongs ...` |
| 形の違う入力で断られた `run(k)` の後 | `_tree_of(k)` は `None`、`Compose(k)` で走る |
| 単独で断られた `net` の後 | `run(net, other)` は `belongs ...`（仕様どおりまとめては走らない）、`run(net)` は走る |

- 並行（同じノード 2 本、同じネットワーク 3 本）: free-threaded と `-X gil=1` でそれぞれ 30 回、毎回 1 本だけ走り、残りは仕様どおり `<クラス名> has already run; ...`
- `_BUILD_LOCK` と `_Tree` の更新の順序: `by_run` は build の中（lock の下）で決まり、印を付けるのは検査をすべて通った後。`ran` は `graph._lock` の下で立てる。`_build` 側の読みは lock の外だが、bool 1 つの読みで、ずれても文言が変わるだけ（should-fix 1）
- 例外で抜けたとき: `_build` の検査で落ちたら印は付かない。`RunCompose` の起動が `Exception` で断られたら（`NetworkError` を除く）`unbuild` が印を消す。`KeyboardInterrupt` と `NetworkError` は印を残し、`ran=True`。仕様の「走ったものとして数える」と一致する
- GC: `_Tree` は bool を 2 つ持つだけで部品を参照しない。`_BUILT` の値は `(weakref, _Tree)`。`run(node)` / `run(net, other)` の後に参照を捨てると、ノードもネットワークも回収された（`weakref` が `None`、`forget` で `_BUILT` から消える）
- `start_once` の `graph._members[0]` は build の根。`RunCompose` は呼び出しごとに新しく作るので、`start_once` が `RunCompose has already run` を出す経路はない
- `RunCompose` は private パッケージ `mujou._core` の中にあり、`mujou` から公開されていない（規約どおり）
- 成果物汚染（`</content>` など）: なし

## 検証結果

- just run: pass（cargo test すべて ok、pytest 475 passed / 15 skipped、pyright 0 errors、mkdocs build --strict ok、作業ツリーの変更なし）
- `tests/mujou/_core/test_run.py` / `test_network.py` / `test_node.py` を free-threaded と `-X gil=1` で 3 回ずつ: 毎回 167 passed, 12 skipped
- 並行シナリオのスクリプト（scratchpad、2 スレッド + Barrier、各 30 回）を両方のモードで実行した。結果は should-fix 1 と nit 1 に書いた
