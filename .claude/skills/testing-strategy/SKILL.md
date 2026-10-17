---
name: testing-strategy
description: mujou のテスト戦略。理論から導いた数値的オラクル（有限差分・平衡点での BP 一致・α=0 の PC 退化・エネルギー単調性）を最優先し、乱数とスケジュールを固定して決定的に保つ。tests/ ミラーレイアウト、書く/書かないリスト、slow マーカー、モック方針。テストコードを書く・壊れたテストを調査する・pytest 周りを設定する前に読む。
---

# mujou テスト方針

## 哲学

mujou の既定の更新手法 PC-ALM は「BP を使わずに BP 並みに学習する」ことを主張する。テストは「動いた」ではなく「**理論どおりに動いている**」ことを保証する。学習則は符号や係数を 1 つ間違えても、それなりに損失が下がって見えることがある。損失が下がることだけを見るテストは、こうした誤りを見逃す。

## オラクルの優先順位

1. **理論から導いた厳密な期待値**（最優先）
    - **有限差分**: 手書きした $\nabla_{h_i}\mathcal{L}$, $\nabla_{W_i}\mathcal{L}$ を、拡張ラグランジアン $\mathcal{L}$ の数値微分と比べる（float64 で 1e-6 程度）
    - **平衡点での BP 一致**: 活動を forward 値、$\lambda$ を BP の漸化式で置いた点で、$\nabla_h \mathcal{L} = 0$、重み更新が BP の負勾配と一致する（skill `pc-alm` 4 節。非線形でも成り立つ）
    - **退化**: $\alpha = 0$ で PC と一致する。$\lambda = 0$ かつ $T = 0$ で forward pass と一致する
    - **手計算できる小さな系**: スカラー 2 層 $\hat y = w_2 h,\ h = w_1 x$ で $\lambda \to w_2(y - \hat y)$
2. **ダイナミクスの性質**: 線形ネットで十分な $T$ を回すと不動点へ収束する、$\alpha$ を大きくしすぎると発散する、同期・非同期の固定スケジュールやノードの分け方（ニューロン・レイヤー・スタック）を変えても同じ不動点に達する（skill `locality-and-async`）
3. **学習の結果**: 小さな合成タスクで損失が下がる・精度が出る。最後の砦であって、これだけで正しさを主張しない。時間がかかるものは `@pytest.mark.slow`

## 基本原則

- 公開インターフェースと観測可能な振る舞いをテストする。内部実装の詳細は見ない
- 必要十分なテストだけを書く。網羅率を稼ぐためのテストは書かない
- **決定性**: 乱数は seed を固定し、非同期スケジュールはテストで明示的に与える。たまに落ちるテストを作らない
- **数値精度**: 理論値との比較は float64 で行い、許容誤差は根拠のある値にする。符号ミスを見逃すほど緩い `atol` にしない
- テストは `class TestXxx:` に集約する。テスト関数に戻り値の型アノテーションは不要
- 複数パラメータは `@pytest.mark.parametrize`

## テストレイアウト

`tests/mujou/` は `src/mujou/` を 1 対 1 でミラーリングする（1 source 1 test ファイル）。

```
src/mujou/<module>.py        ↔ tests/mujou/test_<module>.py
src/mujou/<pkg>/<module>.py  ↔ tests/mujou/<pkg>/test_<module>.py
```

例外:

- 拡張モジュール `mujou._core._native` は bindings の Rust ソースをミラーする。`crates/mujou-python/src/<mod>.rs ↔ tests/mujou/_core/test_native_<mod>.py`、`lib.rs` ↔ `test_native.py`、生成した stub ↔ `test_native_stub.py`。ただし `dlpack.rs` は対になるファイルを持たず、`test_native_slot.py` が検証する
- 公開 API の裏方として単独のテストファイルを持たない `src/mujou/_core/` のモジュールは、使う側のテストで検証する。`session.py`・`state.py`・`lifetime.py`・`hooks.py` は `Run` 経由で `test_run.py` が、`graph.py`・`component.py` は `test_network.py` が、`binding.py` は `test_port.py` が検証する
- `test_node_typing.py` は pyright にかけて型を検証するファイルで、ミラー元を持たない

ネイティブの振る舞いは 3 層で分担し、同じことを二重に検証しない。意味論は Rust のテスト（`crates/mujou-core/tests/`）、`test_native_*` は bindings 固有のこと（dtype・DLPack の変換、例外の型とメッセージ、GIL とスレッド状態、pickle、Python の hook、ログ）、公開 API の振る舞いは `test_run.py` / `test_port.py` などが持つ。

`tests/` 直下に置けるのは共通資産（`conftest.py`）と、`src/` 側にミラー元が無いリポジトリ資産のテストだけ（例: `test_claude_hooks.py` は `.claude/hooks/` を検証する）。

## マーカー

- `--strict-markers` が有効。登録済みは `slow` と `cuda`。新しいマーカーは `pyproject.toml` の `[tool.pytest.ini_options].markers` に登録してから使う
- `slow`: 収束・学習を伴う長いテスト。`just test-fast` で除外、`just test` / `just run` では実行する
- `cuda(n=1)`: CUDA デバイスが `n` 個以上ないと skip する（`tests/mujou/conftest.py` の `pytest_runtest_setup`）。クラスとメソッドの両方に付けたら大きい方の `n` が効く。ノードのデバイスは CPU と CUDA だけなので、GPU のテストはこのマーカーで書く

## 共通の fixture（`tests/mujou/conftest.py`）

- `virtual`: テストごとの新しい `VirtualTimer`。`start_manual` / `start` は実行を始め、テストの終わりに止める（`start_manual` の既定のタイマーは `virtual`）
- `step`: `step(run, *order)` で `run.timer`（`VirtualTimer`）を `H`（1 ms）進め、`run.step(*order)` を呼ぶ
- ノードを入れるだけのネットワークは公開の `Group(node=...)` で作る。専用のクラスを書かない

## 何をテストするか / しないか

### 書く

- 正常系、異常系（shape 不一致など境界での検証）、エッジケース（深さ 1、幅 1、バッチ 1）
- 上記オラクルによる数値検証
- 例外メッセージは **substring** で検証（`assert "expected" in str(exc.value)`）

### 書かない

- 継承・import 可能性・定数 literal の追試
- getter / setter ラウンドトリップ
- framework / stdlib（テンソルライブラリ自身の演算）の動作追試
- 例外メッセージの完全一致
- モック戻り値をそのまま検証するテスト

## モック

- 原則使わない。数値計算は実データ・実テンソルで検証する
- 内部の private 関数はモックしない（リファクタのたびに壊れ、振る舞いは何も保証しない）
- 3rd-party のテンソルライブラリはモックしない
- 外部 I/O（センサーストリーム等）が必要になったら、自前の抽象（ABC / Protocol）を定義し、その実装をテスト用に `tests/mujou/conftest.py` などへ置く。`pytest-mock` は依存に入っていない（2026-10-17 に未使用のため外した）。どうしても要るときは `uv add --dev pytest-mock` で足す

## 関連

- 理論: skill `pc-alm` / 実行モデル: skill `locality-and-async`
- 規約: skill `refactor-conventions`
