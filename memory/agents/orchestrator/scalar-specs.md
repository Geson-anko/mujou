# スカラーの spec（IntSpec / FloatSpec / BoolSpec / TimestampSpec）

ブランチ: `feature/2026-10-22/scalar-specs`。2026-10-22 にユーザーと合意。

## 段階 1: 計画

### 決定（ユーザー確認済み）

- `IntSpec(bits=64, signed=True)`（bits は 8 / 16 / 32 / 64）、`FloatSpec()`（f64 固定）、`BoolSpec()`、`TimestampSpec()`（値は `mujou.time.Timestamp`、コアは u64 の ns、初期値は `Timestamp(ns=0)`）
- 値は Python の `int` / `float` / `bool` / `Timestamp`。shape 属性は持たない。単独のメッセージにも `DictSpec` の値にもなる
- `zeros` / `empty` は持たない。`MessageSpec` の抽象メソッドから外し、`TensorSpec` と `DictSpec` だけに残す。`DictSpec.zeros()` はスカラーを 0 / 0.0 / False / `Timestamp(ns=0)` で埋める。`ZERO` / `ONE` は作らない
- `DictSpec.field(key, kind) -> kind`: 種別を検査して取り出す。違えば `TypeError`
- 範囲外の値はエラー（丸め・折り返しなし）
- 変換は bindings が Python のスカラーを直接読み書きする（torch と GPU を経由しない）
- 原則: 利用者の値は `torch.Tensor` と Python のスカラー。将来の拡張候補は `ComplexSpec`。array は torch で足りるので持たない
- pc: `Activity.jumps` は `IntSpec()`、`Activity.time` / `Prediction.time` は `TimestampSpec()`

### 自分で決めたこと

- 型の別名 `type FieldSpec = TensorSpec | IntSpec | FloatSpec | BoolSpec | TimestampSpec` を `mujou.spec` に置き、トップレベルにも出す。`DictSpec(**specs: FieldSpec)`、`specs: Mapping[str, FieldSpec]`、`field[F: FieldSpec]` の公開シグネチャに現れるため。閉じた Union なので pyright が網羅を検査できる
- 受け付ける値は注釈の型が受け付けるもの（Python の型付けの規則）: `IntSpec` は int（bool も int）、`FloatSpec` は int と float（float として読み出す）、`BoolSpec` は bool、`TimestampSpec` は `Timestamp`。構造（種別）の検査は codec（`TypeError`）、範囲の検査は Rust（`OverflowError`）
- Rust 側は値に `__dlpack__` が無ければスカラーとして、フィールドの DLPack の dtype に従って取り出す（int8..64、uint8..64、float64、bool）。コアの `Schema` は種別を知らない。読み出しは `BufferView.item()` が dtype に従って Python の int / float / bool を返す
- `Timestamp` と ns の変換は codec が行う（Rust は u64 の int だけを見る）
- `LearnClock.tick(dt, time_ns, now_ns)` のシグネチャは変えず、呼ぶ側が `act["time"].ns` を渡す（変更を最小にする）

### テスト観点

- spec: 各 spec の等価・hash・repr、`IntSpec` の bits の検査、`DictSpec.field` の成功と `TypeError`、`DictSpec.zeros` / `empty` のスカラー
- codec / 往復: 各スカラーを単独と DictSpec の中で送って読むと同じ値・同じ型。境界値（int8 の -128 / 127、uint64 の 2**64-1）が往復し、範囲外（128、-1 を unsigned に）は `OverflowError`。種別違い（int の場所に Tensor、Tensor の場所に int）は `TypeError`。`FloatSpec` に int を書くと float が返る。初期値はゼロ
- Rust: `BufferView.item()` と scalar の書き込み（cargo test は core 側で済むなら core に）
- pc: 既存テストが time / jumps を Timestamp / int で通る

## 段階 2〜3: テストと実装

- テストは spec（`test_spec.py`）、往復と境界値（`test_codec.py` の `TestScalars`）、型（`test_pyright.py`）、pc の `time` / `jumps`、`__all__`（`test_init.py`）。spec と codec のテストが import で失敗する red を確認した
- 実装中のユーザーの追加の決定: `TimestampSpec` も作り、pc の `time` はそれで運ぶ。初期値は 0
- 計画外の判断:
  - `spec.py` と `types.py` が `Timestamp` を `mujou._native` から取るので、`tests/test_layers.py` で `_native` を `types` / `errors` の 1 つ下の層に分け、`NATIVE` に `mujou.spec` と `mujou.types` を足した
  - `types.Message` にスカラーを足した（`Run.write` と `inputs=` が単独のスカラーのメッセージを受け付けるように）
  - `port._tensors` を `_values` に改名した（スカラーも返すため）
  - `FieldSpec` の説明は docstring でなくコメントにした（pre-commit の check-docstring-first が別名の後の文字列を 2 つ目の module docstring とみなす）
  - `just type` が examples も検査するので、段階 3 で `pc_merge.py` と `feedback_alignment.py` も直し、両方を実行して動作を確かめた
  - 互換は残さない（公開前）。古い木（`jumps` / `time` が 0 次元の Tensor）は復元できない

## 段階 4: 自己レビュー

- `Value` と `Scalar` の使っていない `Clone, Copy` を外した
- 見送り: `LearnClock.tick` を `Timestamp` を受け取る形に変える案（API の変更が広がるので、呼ぶ側が `.ns` を渡す）。example の `TensorSpec(())` のスカラー（subscribe / timers / online_learning）を `FloatSpec` に置き換える案（要求の外）

## レビュー（code-reviewer: request-changes → 対応、algorithm-reviewer: approve）

- M1 対応: 状態の木に `Timestamp` が入り `torch.load(weights_only=True)` で読めなくなっていた。木の時刻はもともと ns の int なので、メッセージの `TimestampSpec` のフィールドも ns の int で持ち、復元で戻す（`codec.times_to_ns` / `times_from_ns`、`specs/execution-model.md`）。`add_safe_globals` で torch の大域状態を変える案は採らない
- S1 対応: スカラーの読み取りの誤りはすべて `scalar::located` を通し、スロットとフィールドを名指しする（float の溢れも `OverflowError`）
- S2 対応: `Port.send` / `Run.write` / `Wiring.write` の Raises と、「send spec の `zeros()`」の言い回しを直した
- S3 対応: bool を int として受け付ける、dict の中の溢れがフィールドを名指しする、`item()` がスカラーでないフィールドを断るテストを足した
- N1（a/an）と N3（parametrize）を直した。N2・N4・N5 は現状のまま（退行でない、hot path の差は小さい、層の規則による）
- algorithm-reviewer の nit: `docs/assets/viz/chain.html` を `examples/visualize.py` で再生成した
- 手順の逸脱: M1 の修正はテストより先に書いた（テストは修正後に green を確認）
