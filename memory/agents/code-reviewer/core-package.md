# core-package（mujou._core パッケージへの集約、_native への改名、request_stop）レビュー

対象: `refactor/2026-10-16/core-package`、`git diff feature/2026-10-16/timers..HEAD`（c04f362..942f5ca）

## verdict: approve

must-fix は無い。移動と改名の漏れは見つからなかった。`request_stop` は、制御スレッドへの送り方・`Drop`・開始前と終了後・`join` との組み合わせのどれにも問題が無い。`mujou.__all__` と公開 API の振る舞いは変わっていない。

## must-fix

なし

## should-fix

1. `.gitignore:46-47` / 既存の checkout に残る古い拡張モジュール
   - 問題: 無視するパターンを `src/mujou/*.so` から `src/mujou/_core/*.so` に置き換えた。そのため、既存の checkout に残っている古いビルド成果物が merge 後は無視されなくなり、untracked として出てくる。main の checkout には今 `src/mujou/_core.cpython-314-darwin.so`、`_core.cpython-314t-darwin.so`、`_core.cpython-314t-aarch64-linux-gnu.so`（dev container）の 3 つがある。`git add -A` で誤ってコミットされうる
   - 根拠: `ls /Users/geson/Projects/mujou/src/mujou/`。import への影響は無い。FileFinder はディレクトリのパッケージ（`_core/__init__.py`）を同名の拡張モジュールより先に見つけるので、`mujou._core` はパッケージに解決される
   - 対処案（採否は orchestrator）: 古いパターンもしばらく残す。または merge の手順に古い `.so` の削除を書く
   - 確信度: 高（ファイルが残ること）/ 深刻度: 低〜中

2. c04f362 は単独では `just run` が通らない
   - 問題: 最初のコミットはファイルの移動だけで、`src/mujou/_core/__init__.py` が無い。import 先も古いまま（`from mujou import _core` など）で、拡張モジュールはまだ `mujou._core` の名前で `src/mujou/` にある。このコミットでは import が壊れている。CLAUDE.md の「`just run` が通る前に commit しない」に反し、bisect もここで止まる
   - 根拠: `git show --stat c04f362`（`__init__.py` が含まれていない）。判断ログには履歴を追いやすくするための意図的な分割と書いてある
   - 対処案: 8fa8178 と 1 つにまとめても、類似度は 90% を超えるので git の rename 検出は効く。分けたままにするなら、ユーザーの了承を記録しておく
   - 確信度: 高 / 深刻度: 低（規約違反。動作には影響しない）

## nit

3. `crates/mujou-core/src/runtime.rs:1098-1099` のコメントにある「a node's own thread may ask too」を確かめるテストが無い
   - Rust にも Python にも、ノードのスレッドから `request_stop` を呼ぶテストが無い。コードを読む限りは正しい。`submit` を通らないので `Reentrant` にならず、unbounded の mpsc に送るだけなので詰まらない。制御スレッドで呼ばれた場合（`capture` の build の中など）も、今のジョブの後に処理されるだけで済む。Python では `in_hook` の検査があるので、公開の経路からは到達しない
   - 確信度: 低（あった方がよい程度）

4. `tests/mujou/_core/test_network.py` の `test_asks_without_a_thread_however_often_it_times_out` の `threading.active_count() == before`
   - プロセス全体のスレッド数を等号で比べている。前のテストが残したスレッド（subscribe の消費スレッドなど）がこの間に終わると、数が減って落ちる。daemon スレッドが増えないことを確かめるだけなら `<= before` で足りる。6 回回して再現はしなかった
   - 確信度: 低

5. `specs/native-core.md:20` の「Rust の `mujou._core._native` のことではない」
   - 名前の上ではもう `mujou._core` パッケージ全体と紛らわしい。`memory/project_pc_module.md:13` は「`mujou._core` パッケージのことではない」と書いていて、2 つの記述が揃っていない。「`mujou._core` パッケージ（とその中の `_native`）」のように揃えるとよい
   - 確信度: 中

6. （既存の挙動で、今回の退行ではない）`Session.shutdown` が `Runtime.start` の途中と重なったとき
   - `Session.run` は `runtime.start` より前に `self._runtime` を設定する。一方、Python の `Runtime` は core の起動の後に `inner.set` する。その間に別のスレッドが `shutdown(timeout)` を呼ぶと、`request_stop` は何もせず、`join` は「the runtime has not started」の `RuntimeError` を送出する。以前の daemon 版も同じ挙動で、しかも以前は `_stopper` が 1 回しか要求しなかった。今回の変更で、呼ぶたびに要求し直すので、むしろ良くなっている。実行の API の作り直しで扱う範囲だと思う
   - 確信度: 中

## 確かめたこと（指摘なし）

- 移動・改名の漏れ: tracked な全ファイルを `mujou._(core|spec|port|network|node|session)`、`_core.pyi`、`test_core*.py`、`from mujou import _core`、`mujou-shutdown`、`_stopper`、`_request_stop` で grep した。古い参照は無い。設定（pyproject の `module-name`、`#[pymodule]`、各 `#[pyclass(module=...)]`、`create_exception!`）、stub（`just stub` は `module-name` から生成先を決めるので `src/mujou/_core/_native.pyi` になる）、pre-commit の `files`、CI の `git diff --exit-code`、`.gitignore`、Dockerfile、README、CLAUDE.md、skill 2 本、specs、memory を確認した
- `request_stop`: `finish` は `done` を見るので何度呼んでもよい。終了後は receiver が drop されていて、送っても黙って捨てられる。開始前は Python 側の `inner.get()` が `None` なので何もしない。`finish` の後にキューに残った `shutdown` のジョブは、`Reply` の `ended`（`stopped`）で正しく結果を返す。`Drop` の挙動は以前と同じ
- `Session.shutdown`: 有限の timeout なら `request_stop` を送ってから `join(wait)` で待つ。`math.inf` なら以前どおり `runtime.shutdown()`。`NetworkError` は `join` から伝わり、以前の daemon が握りつぶしていた分は呼び出し側の `join` が報告する。どちらも以前と同じ
- `mujou.__all__` の中身は変わっていない（`__init__.py` は import 先だけを変えた）
- テストのミラー: `tests/mujou/_core/` が `src/mujou/_core/` に対応する。ファイルの basename は重複していない（`__init__.py` の無い rootdir 方式でも衝突しない）
- 成果物汚染: 変更した全ファイルを `</content>` と `</parameter>` で grep した。混入は無い

## 検証結果

- just run: pass（cargo test、pytest 368 passed / 14 skipped、pyright 0 errors、mkdocs --strict）
- pytest の繰り返し: free-threaded（3.14t）で 3 回、`python -X gil=1` で 3 回、すべて 368 passed / 14 skipped（`sys._is_gil_enabled()` が True になることも確かめた）
- `cargo test -p mujou-core --test runtime stop_request` を 5 回、すべて pass
- 未検証: wheel と sdist の中身（`uv build` は permission で拒否された）。`module-name = "mujou._core._native"` と `python-source = "src"` の組み合わせで、`_native*.so` が `mujou/_core/` に入ることは確かめていない
