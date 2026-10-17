# api-polish レビュー（Run.write が時刻を返す / mujou.run がノードを受け付ける）

対象: `git diff 2ac7ecd..HEAD`（899b202, 37f2ce2, d26f2ed）

## verdict: approve

must-fix はない。下の should-fix は orchestrator の裁定次第で、マージは止めない。

## must-fix

なし

## should-fix

1. `src/mujou/_core/network.py:236-240`（ノードの 2 回目の run の文言）/ 確信度: 高 / 深刻度: 中
   - 問題: `run(node)` を 2 回呼ぶと `RuntimeError: Leaky at '0' belongs to a network that is already built` になる。ネットワークの 2 回目は `the network has already run; build a new instance to run again` で、文言がそろっていない。さらにノードの文言は、利用者が作っていない「network」と、ラップの `Compose` が付けたパス `'0'` を出す。`run(node)` しか書いていない利用者には原因が読み取れない
   - 根拠: 実際に再現した（`mujou.testing.run(n)` の後に `n` / `Solo(n)` / `Compose(n)` を走らせると、それぞれ `'0'` / `'n'` / `'0'` で "already built"）。orchestrator は文言をそろえる方針。そろえると `tests/mujou/_core/test_run.py:220` の `"already run" or "already built"` と、`TestRunANode.test_runs_once` の `match="already built"`、`specs/network.md` の実装の節（「ノードを 2 回走らせると…`RuntimeError` になる」）、memory の記述も一緒に直す必要がある

2. 空の `write` の契約にテストがない（`src/mujou/_core/run.py:73-76`, `specs/network.md:110`）/ 確信度: 高 / 深刻度: 低
   - 問題: docstring と仕様が「空の `values` は何も変えず、その時刻を返す」と約束しているが、検証するテストがない
   - 根拠: `tests/` を grep しても `write({})` の呼び出しがない。手元では `r.write({})` が `Timestamp(ns=0)`（VirtualTimer の今）を返し、`/in` のスロットの時刻は変わらないことを確かめた。Rust 側は空でも版番号 `V` を 1 進めるが、値も時刻も変えないので「何も変えない」とは矛盾しない

## nit

1. `tests/mujou/_core/test_run.py:220`（並行の run の検査の緩さ）/ 確信度: 高
   - `"already run" in str(error) or "already built" in str(error)` が `Chain` と `Leaky` の両方に効くので、`Chain` の側で検査が弱くなった（ネットワーク 1 つでは必ず "already run" になるはず）。期待する文言をパラメタに持たせれば、それぞれを厳密に検査できる。should-fix 1 の文言をそろえれば、この問題もなくなる

2. `TestRunANode.test_a_refused_run_can_be_retried`（`tests/mujou/_core/test_run.py`）/ 確信度: 中
   - docstring と仕様は「断られたノード 1 つの run は、どの形でも走らせ直せる」と書いているが、テストは同じ形（`node`）でしか走らせ直していない。手元では `Solo(node)` で走らせ直せることを確かめた。`Solo(node)` で走らせ直すテストにすれば、仕様の文言をそのまま検査できる

3. `TestRunANode.test_mixes_with_networks` のオラクルが「ゼロでない」だけ / 確信度: 中
   - 主眼はパスと接続なので致命的ではない。ただし skill `testing-strategy` は理論から導いたオラクルを優先する。`TestRunANode.test_runs_as_a_network_holding_it_at_0` と同じく `euler` を合成した値と比べられる

4. `src/mujou/_core/network.py:457`（実行時の末尾の import が import の順序に依存する）/ 確信度: 中
   - 今は循環しない。`mujou/__init__.py` が `mujou.time`（ノードを実行時に読まない）→ `network` → `node` の順に読むからである。ただし、`network` より先に `mujou._core.node` を読むモジュールが将来 `__init__` の前のほうに入ると、`node` → `network` → 末尾の `from mujou._core.node import Node` の順になり、部分的に初期化された `node` から `Node` を取れず `ImportError` になる。コメントに「`network` は `node` より先に読まれる前提」と書いておくか、`Component` を別モジュールに切り出すと前提が要らなくなる。後者は今回の範囲外。`inspect.signature(Compose)` と `typing.get_type_hints(Compose.__init__ / mujou.run / mujou.testing.run)` が解決することは確かめた

5. `src/mujou/_core/network.py:94`（`Graph` の docstring）/ 確信度: 中
   - "``mujou.run`` takes networks, not graphs" の部分が古い。趣旨（Graph は渡せない）は今も正しいが、今はノードも受け付ける

6. 引数名 `*networks` と、`Compose.__init__` / `start` の中の変数名 `network` / `given` にノードも入る / 確信度: 低
   - 確認事項として上がっているとおり。改名する場合は `mujou.run`、`mujou.testing.run`、`Compose`、`conftest.py` の Protocol、仕様の表をまとめて直す

## 確認した観点（問題なし）

- 時刻の一貫性: Rust の `InputGroup::write` は、書き込みの lock の下で読んだ `time_ns` をメッセージに付け、同じ値を返す。`recv_with_time`、`/in`、`read` の値（Echo）が一致することをテストで確認できる。`inputs` の時刻は、`state` なしなら開始時のタイマーの今（VirtualTimer で 1.5 s 進めた状態で `/in` と `write` がどちらも 1.5e9 になることを手元で確認した）、`state` ありなら T
- 1 回だけ走る: 構築は `_BUILD_LOCK` で直列になるので、別々の `Compose` が同じノードを同時に構築しても、印が付くのは 1 回だけ。`_build` は検査をすべて通ってから印を付けるので、build が失敗した側の `Compose` は印を残さない
- 断られた run のやり直し: `run(node)` が断られた後に `Solo(node)` で走らせ直せる。`run(b, a)` が断られた後に `a` と `b` を別々に走らせられる（手元で確認した）
- `run(Compose(node))` は、ネットワーク 1 つとして包まずに走り、パスは `"0"` で `run(node)` と同じになる
- 成果物汚染（`</content>` など）: なし
- free-threaded（3.14t）と `python -X gil=1` で、`test_run.py` / `test_network.py` / `test_native_slot.py` をそれぞれ 3 回ずつ回した。毎回 176 passed, 7 skipped
- pyright strict: 0 errors

## 検証結果

- just run: pass（cargo test すべて ok、pytest 473 passed / 15 skipped、pyright 0 errors、pre-commit すべて Passed、作業ツリーの変更なし）
