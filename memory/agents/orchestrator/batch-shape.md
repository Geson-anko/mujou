# batch の shape を一般化する（2026-10-21）

ブランチ `refactor/2026-10-21/batch-shape`。solo-dev-cycle。

## 要求（ユーザーの決定）

- バッチの無い入力を既定にする。`batch: int | tuple[int, ...] = ()`。int は `(n,)`
- 損失はクラス `pc.MSELoss(batch=())` / `pc.CrossEntropyLoss(batch=())` と、関数 `pc.functional.mse` / `pc.functional.cross_entropy`（`batch=()`）。`pc.mse` / `pc.cross_entropy` は消す
- mse の 1 サンプルの値は特徴（バッチでない次元）の平均、½ なし。1/prod(batch) と合わせて θ の更新が `nn.MSELoss()` の勾配降下と一致
- cross_entropy もバッチが要る（1 サンプルの目標が構造を持つとき畳む次元、クラスの次元 = len(batch)。torch の「クラスはバッチの直後」の一般化）。バッチでない残りの次元は平均

## 公開 IF

- `activity_spec(h, y, batch=())`: h と y が batch で始まることを照合、mask は `TensorSpec(batch, bool)`
- `Input(*, x, y, period, batch=(), device)`、`Chain(..., batch=(), ...)`
- `Layer` / `Output`: 引数を増やさず、`act_spec` の mask の shape からバッチを読む
- `functional.learn_(params, g_theta, elapsed, tau, batch)`: batch は int | tuple、prod で割る（必須のまま）
- `functional.mse(y_hat, y, batch=())`、`functional.cross_entropy(y_hat, y, batch=())`
- `MSELoss(batch=())`、`CrossEntropyLoss(batch=())`: 素のクラス（`__call__`）。nn.Module にしない（パラメータが無い）
- `messages.rows(mask, like)`: 多次元の mask に一般化。`messages.batch_shape(batch) -> tuple[int, ...]` で int を正規化（rows と同じく pc.__all__ には入れない）

## テストのオラクル

- 2 次元のバッチ (E, A) と平坦化した (E*A,) で Chain の状態が一致
- batch=() と batch=(1,) で Chain の状態が一致
- mse: 平均が nn.MSELoss()、cross_entropy: 平均が nn.CrossEntropyLoss()（(B,C,H,W) も）、() は F の unbatched と一致
- N_OUT=2 では新しい mse = 旧 ½ 和なので、既存の数値テストはそのまま（TestScalar は下の記録）

## 記録

- default () のため、バッチを書き忘れると全体が 1 サンプル扱いで黙って走る。ユーザーの決定どおり。docs で明示する
- TestScalar（N_OUT=1）は新しい mse で勾配が 2 倍になり陽的 Euler が発散。λ と credit の関係を見るテストなので、損失を 0.5 * functional.mse にして元の期待値を保った

## code-reviewer（request-changes）への対応

- must-fix: rows が mask・like ともに 0 次元で reshape() 引数なしになり TypeError → タプルで渡す形に修正。公開経路のテスト（Chain、batch=()、スカラーの y、CrossEntropyLoss）を追加。TestRows（内部名を shape だけで見るテスト）は削除
- should-fix: functional.mse / cross_entropy に「batch で始まる」照合（_starting）を追加
- nit: Chain の docstring の引数順、test_input のテスト名、_reference の docstring の行長を修正。ignore_index と 0 の大きさの次元は提供シナリオ外として対応しない。__call__ の docstring はクラスの docstring で足りるので足さない
- algorithm-reviewer（approve）: should-fix 3 件（オラクルの 2/d、mse の docstring に勾配と安定性、TestScalar のコメント）と nit（ignore_index、docstring の B）に対応。batch_shape の numpy 整数は対象外
- ユーザーの提案で型の別名 `type Size = int | tuple[int, ...]` を messages.py に置き、`pc.Size` として公開（公開シグネチャと API リファレンスに出るため）
