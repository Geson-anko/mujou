# simplify-b: bindings とスタブ生成の簡素化（2026-10-19）

担当: `crates/mujou-python`、`crates/mujou-stubgen`、`src/mujou/_core/_native.pyi`（生成物）。ブランチ `refactor/2026-10-19/simplify-b`。

## 結果

合計 +73 / −105 行（正味 −32 行）。

| ファイル                            | 増減      |
| ----------------------------------- | --------- |
| `crates/mujou-python/src/runtime.rs` | +42 / −73 |
| `crates/mujou-python/src/time.rs`    | +11 / −12 |
| `crates/mujou-python/src/slot.rs`    | +2 / −2   |
| `crates/mujou-stubgen/src/main.rs`   | +17 / −18 |
| `src/mujou/_core/_native.pyi`        | +1 / −0   |

`just stub` の差分は `Node` の docstring に空行が 1 行入っただけ（下記 4）。API・例外・振る舞いは変えていない。`just run` は green（pytest 595 passed / 20 skipped、pyright 0 errors、mkdocs strict OK）。

## 変更（コミット順）

1. `refactor(bindings): look the cached hooks up from one list`
   - `Methods` 構造体（9 フィールド、`of` と名前で引く `get` の 2 つの match）を、定数 `CACHED_HOOKS: [&str; 9]` と `PyNode::methods: Option<Vec<Py<PyAny>>>` に置き換えた。フックの名前が 3 か所（フィールド・`of`・`get`）から 1 か所に減る
   - `public(hook)`（`_after_update` を `on_update` と報告するためだけの変換）を消し、`update` で `call_as("_after_update", "on_update", ..)` と明示した。`enter` と同じ書き方になる
   - `start` のメソッド探索の失敗時に `Python::attach` を 2 回していたのを 1 回にまとめた（探索とエラーの記録を同じ attach の中で行う。GIL の扱いは同じ）
   - 復元する state を `states.map(Vec::into_iter)` + `as_mut().and_then(Iterator::next)` から `states.into_iter().flatten()` + `next()` にした（長さは直前で `n` と照合済みなので同じ）
2. `refactor(bindings): share the conversion of seconds to nanoseconds`
   - `seconds_to_ns(s).map_err(timer_error)` が 4 か所（`Timer.sleep`、`Timer.advance`、`runtime::duration`、`runtime::period_to_ns`）にあったので `time::to_ns` にした。`timer_error` は `time.rs` の private に戻した
   - `PyTimer::timer()`（素通しのアクセサ）を消し、フィールドを `pub(crate)` にした（`PyTimestamp` の `pub` フィールドと同じ流儀）
3. `refactor(stubgen): group the fixes that need the class's own type`
   - `fix_methods` で `own_type.as_ref().filter(..)` と `own_type.is_some()` に分かれていた判定を `if let Some(own_type) = &own_type` 1 つにまとめた。`object` の `Expr` はクラスごとに 1 回作る。出力は不変
4. `docs(bindings): split the paragraphs of the Node docstring`
   - `Node` の docstring で段落の間の空行が抜けていた（"They do nothing by default." の直後に次の段落が続いていた）ので補った。stub にも反映

## 整理しなかったもの（理由）

- `dlpack.rs` の deleter 呼び出しの重複（`export_capsule` の失敗時と `destroy_versioned` / `destroy_legacy`）: `delete_*` ヘルパーに寄せる案を見積もったが行数はほぼ減らず、unsafe の境界を動かすだけになるので見送った
- `PySlot::restore` と `PyInputSlot::restore` の同じ 4 行: 型が別で 2 回しか現れない。共通化しても `write_values` の呼び出しを包むだけになる
- `PyOutlet.__new__` と `InputGroup.__new__` の同じ段落の docstring: Python から見える docstring なので変えない
- `Runtime.join` の `Option<Result>` の match: `transpose` で畳めるが、明示的な match のほうが読める
- `hint.rs` の `collections.abc` の `PyStaticExpr` が 2 回現れる点: 定数に抜いても数行しか減らず、const の中の参照の書き方が増える
- stubgen の `OBJECT_METHODS` の `__str__` など今のモジュールで使わない名前: 「`object` のメソッド」という定義そのものなので残した（足したときに `override` が漏れる罠を作らない）
- `EXCEPTIONS` の docstring の二重管理: pyo3 が `create_exception!` を introspect しないため必要。`tests/mujou/_core/test_native_stub.py` が一致を検査している
- `wait_interruptibly`・`on_thread`・`call_as` の attach / detach と、`Runtime::start` の `pending` の lock の順序: 触っていない

## ユーザー確認が要る判断

なし。
