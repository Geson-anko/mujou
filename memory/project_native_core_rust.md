---
name: project-native-core-rust
description: コアは Rust で実装し、Python はインターフェース。実行モデルは M:N（実行ごとのワーカープールと opt-in の専用スレッド）。コアの値の型は内部に閉じ、利用者には torch.Tensor だけ。torch とは DLPack で必ずコピー。例外は Python で定義し、Rust は Python の Driver を呼ぶ。max_dt はコアでなく mujou.pc が持つ
metadata:
  type: project
---

2026-10-11 決定。詳細は `specs/native-core.md` と `specs/architecture.md`。

- **コアは Rust、Python はインターフェース。** PyO3 + maturin で拡張モジュール `mujou._core._native` を作る。crate は純 Rust の `crates/mujou-core` と bindings の `crates/mujou-python`
- **Rust は仕組み、Python は方針。** Rust はスロット、バッファプール、shape と dtype の照合、タイマーと RunClock、ライフサイクルの状態機械、ワーカーを持つ。Python は spec と codec、木の読み方とガード、状態の木の形、hook で何をするか（Driver）を持つ
- **Rust は Python の `Driver` を hook 1 回につき 1 回呼ぶ。** 「いま hook を走らせている Driver」は bindings の thread-local（`_native.current()`）の 1 か所が持つ
- **例外は Python で定義する。** `NetworkError` は `mujou._core.errors` にあり、Rust は `import_exception!` で使う。timeout は組み込みの `TimeoutError`
- **実行モデルは M:N。** M 個のノードを N 本のワーカースレッドに載せる（[[project-mn-scheduler]]）。専用スレッドは `Node(dedicated_thread=True)` のノードだけ
- **ノードは「状態 + hook」。** ノードは他のノードを待たないので、async/await のコルーチンにしない。1 ステップが協調スケジューリングの区切り
- **スロットはスロットごとの mutex で `Arc` の複製・差し替えの短い区間だけ待つ（2026-10-17、ユーザーの承認）。** lock-free の `arc-swap` は、インスタンスをまたぐ ABA で別のスロットの値を読み手に返したのでやめた。upstream への報告はユーザーが行う。コアの `Slot` は 1 つの型で、書き手を `Source` で区別する（bindings は `Slot` と `InputSlot` に分けて見せる）
- **メッセージの値の型はコアが独自に持つ。** `mujou._core._native` の内部に閉じ、利用者には `torch.Tensor` だけを見せる。コアは DLPack の dtype の表現を使う（[[project-message-types]]、[[project-gpu-policy]]）
- **タイマーもコアに置く。** GIL なしでノードから参照できる（[[project-timer-per-run]]）

**Why:** asyncio や Redis が 1 スレッドで大量のタスクを捌けるのは、タスクの多くが I/O 待ちだから。mujou のノードは CPU を使い続けるので 1 スレッドでは 1 コアしか使えず、ノードごとに OS スレッドを立てるとスレッド数とコンテキストスイッチに縛られる。Rust は並行処理の安全性をコンパイラが保証し、PyO3 で Python から使える。コアの型を外部ライブラリに依存させないことで、フレームワークの基礎を自分で制御できる。

**How to apply:**

- コアは Δt の上限を持たない。打ち切るかはノードの選択で、`mujou.pc` のノードは `max_dt` で打ち切る
- ノードの周期はスケジューラが期限の順に扱う
- 複数マシンにまたがる実行は想定しない（2026-10-13 決定）
- `_native` の型スタブは生成物（[[project-core-stub]]）

関連: [[project-core-message-framework]], [[project-async-node-execution]]
