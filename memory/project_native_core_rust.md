---
name: project-native-core-rust
description: コアは Rust で実装し、Python はインターフェース。実行モデルは M:N（タスクと専用スレッドの混在）。コアの値の型は内部に閉じ、利用者には torch.Tensor だけを見せる。torch とは DLPack で必ずコピーして受け渡す
metadata:
  type: project
---

2026-10-11 決定。

- **コアは Rust で実装する。Python はインターフェース。** Python からは PyO3 + maturin で呼ぶ想定。コアの処理中は GIL を解放する
- **実行モデルは M:N スケジューリング。** M 個のノードを N 本のワーカースレッド（目安はコア数）に載せる。ノードは実行クラスを持つ
  - **タスク**: 共有ワーカープールで協調的に実行する軽いノード（ネイティブのニューロンノードなど、小さく大量のもの）
  - **専用スレッド**: 自分の OS スレッドを持つノード（Python/PyTorch のノード、ノード内 BP、I/O、重いスタック）。Python のコードを動かすノードは GIL を要するので、タスクのプールに載せない
- **ノードは「状態 + step 関数」。** ノードは他のノードを待たない（[[project-async-node-execution]]）ので、async/await のコルーチンにする必要はない。1 ステップ（スロットを読む → $\Delta t$ ぶん更新 → 書き込む）が協調スケジューリングの区切りになる
- **スロットの読み書きはワーカーを止めない形にする。** atomic なポインタ差し替え（`arc-swap` など）とバッファプールを使い、OS の mutex で長く待たない（seqlock + copy は 2026-10-13 に不採用、[[project-message-types]]）
- **メッセージの値の型はコアが独自に持つ。** PyTorch のテンソルを直接扱えると便利だが、コア・基礎は独自の型で作る。この型は `mujou._core` の内部に閉じ、利用者には `torch.Tensor` だけを見せる。torch とは DLPack で受け渡し、読み書きとも必ずコピーする。コアは独自の dtype を持たず DLPack の dtype 表現を使う（2026-10-15、[[project-message-types]], [[project-gpu-policy]]）
- **グローバルタイマー（シングルトン）もコア側に置く。** GIL なしでノードから参照できるようにする

**Why:** asyncio や Redis が 1 スレッドで大量のタスクを捌けるのは、タスクの多くが I/O 待ちだから。mujou のノードは CPU を使い続けるので、1 スレッドでは 1 コアしか使えない。一方、ノードごとに OS スレッドを立てると、スレッド数の上限とコンテキストスイッチのコストに縛られる。M:N なら大量の小さなノードと、重い・GIL が要るノードを同じ仕組みで扱える。Rust は並行処理の安全性をコンパイラが保証し、PyO3 で Python から使える。コアの型を外部ライブラリに依存させないことで、フレームワークの基礎を自分で制御できる。

**How to apply:**

- 負荷が高いときは各ノードの $\Delta t$ が伸びるだけで、連続時間の定式化がそれを吸収する。$\Delta t$ の上限は別に持つ
- ノードに目標周期を持たせる場合、スケジューラは期限順に実行する
- 複数マシンにまたがる実行は、現状は想定しない（2026-10-13 決定）
- コアとの変換（DLPack、必ずコピー）は torch に依存する側（`mujou.Node` とネットワーク）が受け持つ。ゼロコピー共有はしない（[[project-message-types]]）
- ツールチェーンは 2026-10-15 に整えた。純 Rust の `crates/mujou-core` と PyO3 の `crates/mujou-python` に分け、maturin でビルドする。版数は Cargo workspace の version だけ（詳細は `specs/native-core.md` のツールチェーン節）

関連: [[project-core-message-framework]], [[project-async-node-execution]], [[project-open-decisions]]
