# mujou.pc のバッチと損失 理解度テスト

## 対象
- ファイル: docs/en/topics/pc.md（「## The batch」と Pitfalls の最初の 2 つの warning）、docs/en/tutorial/5-learn-with-pc.md（「## The chain」の `x`/`y` と `loss` の箇条）。日本語版（docs/ja/ の同じ箇所）は英語版に内容をそろえる
- 読者と用途: PyTorch を使える利用者が、`pc.Chain` / `pc.Input` / `pc.activity_spec` でバッチを宣言し（なし・B・多次元）、出力の損失（`pc.MSELoss` / `pc.CrossEntropyLoss`、自作の損失）を正しく選んで渡す
- 生徒が読むのは英語版の 2 ファイルだけ

## 事実確認（R1 時点、実物）
- `Chain(..., batch=())`、`Input(x=, y=, period=, batch=())`、`activity_spec(h, y, batch=())`。`Layer` / `Output` は `act_spec` から読む
- `x` / `y` の spec は batch で始まる（`activity_spec` が検査し、違えば ValueError）
- `functional.mse`: 特徴で平均、0.5 なし。`(*batch, *out) -> batch`
- `functional.cross_entropy`: クラスはバッチの直後、残りの次元は平均。y は int64 `(*batch, *rest)` か y_hat と同じ shape の確率
- `loss_vjp`: 損失の shape が mask の shape と違えば ValueError
- 片方だけに batch: Chain に 64・`MSELoss()` → ValueError（`()` vs `(64,)`）。Chain に 64・`CrossEntropyLoss()` → torch の RuntimeError。Chain に無し・`MSELoss(64)` → ValueError。両方無し → 黙って走る

## 評価用問題
### Q1 センサーから 1 回に 1 サンプルずつ届く。入力は 3 特徴、目標は 1 つの実数。`pc.Chain` に渡す `x`・`y` の spec、`batch`、`loss` をそれぞれ答えよ。
- 要点:
  - `x=TensorSpec((3,))`、`y=TensorSpec((1,))`（バッチの次元を付けない）
  - `batch` は渡さない（既定の `()`）
  - `loss=pc.MSELoss()`（引数なし、または `()`）

### Q2 MNIST を 1 バッチ 128 サンプルで学ぶ。入力は 784 画素、目標はクラスの番号。`x`・`y` の spec（dtype を含む）、`batch`、`loss` を答えよ。
- 要点:
  - `x=TensorSpec((128, 784))`
  - `y=TensorSpec((128,), torch.int64)`
  - `batch=128`
  - `loss=pc.CrossEntropyLoss(128)`

### Q3 `batch=64` の鎖に `loss=nn.MSELoss()` を渡してよいか。何が起き、代わりに何を渡すか。
- 要点:
  - 渡してはいけない。`nn.MSELoss()` はバッチで平均してスカラーを返し、バッチの shape `(64,)` を返さない
  - `pc.Output` が `ValueError` を送出する
  - 代わりに `pc.MSELoss(64)`（`pc.MSELoss(batch)`）

### Q4 同僚が `x=TensorSpec((64, 784))`、`y=TensorSpec((64, 10))` とし、`batch` をどこにも渡さず `loss=pc.MSELoss()` で鎖を作った。エラーは出ない。何が問題で、どう直すか。
- 要点:
  - バッチは spec の shape から読まれないので、鎖は 64 サンプルを 1 つのサンプルとして扱う（損失は 64 サンプルをまとめて平均する）
  - 各サンプルの credit が本来の 1/64 になる
  - `pc.Chain` に `batch=64`、損失に `pc.MSELoss(64)` と、同じ batch を両方に渡す

### Q5 8 個の環境それぞれに 4 エージェントがいて、各エージェントの観測は 10 特徴、目標は 2 つの実数。`batch`、`x`・`y`・`mask` の shape、`loss` を答えよ。また、この学習は何個のサンプルの 1 次元のバッチと同じになるか。
- 要点:
  - `batch=(8, 4)`
  - `x` は `(8, 4, 10)`、`y` は `(8, 4, 2)`、`mask` は `(8, 4)`
  - `loss=pc.MSELoss((8, 4))`
  - 32 サンプルのバッチ（`32`）と同じように学ぶ

### Q6 長さ T の系列を 16 本ずつ見せる。各ノードの module は系列の時刻をまたいで計算する（時刻を混ぜる）。`batch` は何にし、時刻の次元は `x` のどこに置くか。理由も答えよ。
- 要点:
  - `batch=16`。時刻の次元はバッチに含めない
  - 時刻はサンプルの側、バッチの後ろに置く（`x` は `(16, T, ...)`）
  - 理由: サンプルは独立でなければならず、module はバッチのどの次元でも各サンプルを別々に計算する必要がある。module が混ぜる次元はバッチにできない

### Q7 出力が `(64, 5)` で `batch=64` の鎖に、自作の損失関数を渡したい。関数の引数と、返す値の shape を答えよ。バッチで平均してよいか。
- 要点:
  - 引数は `(y_hat, y)`
  - バッチの shape `(64,)`、サンプルごとの損失を返す
  - バッチで平均してはいけない（平均すると `pc.Output` が ValueError）

### Q8 BP で `nn.MSELoss()` を使っていたモデルを `pc.MSELoss(batch)` で学ぶ。`pc.MSELoss` は各サンプルの何で平均するか、`0.5` は付くか。重みはどの勾配に沿って動くか。
- 要点:
  - 各サンプルの特徴で平均する（`nn.MSELoss` と同じ）。バッチでは平均しない
  - `0.5` は付かない
  - 重みはバッチに対する `nn.MSELoss()` の勾配に沿って動く

### Q9 画素ごとのクラス分類（セグメンテーション）。1 バッチ 8 枚、C クラス、画像は H×W。`y_hat` と `y` の shape、クラスの次元の位置、画素の次元の扱い、損失が返す shape を答えよ。
- 要点:
  - `y_hat` は `(8, C, H, W)`、`y` は `(8, H, W)` int64
  - クラスはバッチの直後の次元
  - 画素の次元は平均される
  - 損失は `(8,)` を返す（`pc.CrossEntropyLoss(8)`）

### Q10 `pc.Chain` を使わずに `pc.Input`・`pc.Layer`・`pc.Output` を 1 つずつ作る。バッチ 32 はどのノードの引数に渡すか。`pc.Layer` と `pc.Output` はバッチをどこから知るか。損失には何を渡すか。
- 要点:
  - `pc.Input(..., batch=32)`（自作のノードなら `pc.activity_spec(h, y, 32)`）
  - `pc.Layer` と `pc.Output` は受け取る `act_spec` から読む（batch の引数は無い）
  - 損失にも同じバッチ: `pc.MSELoss(32)` など

## 保留問題
### H1 1 サンプルずつ 10 クラスに分類する（バッチなし）。`y` の spec、`y_hat` の shape、`loss` を答えよ。
- 要点:
  - `y=TensorSpec((), torch.int64)`
  - `y_hat` は `(10,)`
  - `loss=pc.CrossEntropyLoss()`、`batch` は渡さない

### H2 `pc.Chain(..., batch=32, loss=pc.MSELoss())` と書いた（損失にだけバッチを書き忘れた）。黙って走るか。
- 要点:
  - 黙っては走らない。`pc.Output` がエラーを送出する
  - 直し方: `pc.MSELoss(32)`

### H3 64 サンプルのバッチで、目標がクラスの番号ではなく 10 クラスの確率（ソフトラベル）。`y` の spec と `loss` を答えよ。
- 要点:
  - `y=TensorSpec((64, 10))`（`y_hat` と同じ shape、浮動小数点）
  - `loss=pc.CrossEntropyLoss(64)`、鎖に `batch=64`

## ラウンド記録
### R0（執筆、en topics 129 行・en tutorial 48 行、改稿前と同じ）
- 「The batch」冒頭を「バッチの定義 → 渡す先 → x/y の spec もこれで始める」に分けた（欠落: spec が batch で始まることの明示）
- 表に `y`（クラス 1 つ）の列を足した（欠落: CrossEntropy の y の shape）
- 損失の箇条に CrossEntropyLoss の `y`（int64 のクラスか確率）を足した（欠落）
- warning 1 に「shape から読まない」「片方だけなら `pc.Output` がエラー」を足した（欠落: 黙って走るのは両方忘れたときだけ）
- tutorial の `loss` 箇条に「鎖と同じバッチを渡す」と topics へのリンクを足した
- ja も同じ内容に合わせた

### R1（en topics 129 行・en tutorial 48 行、増減なし）
| 問  | 判定   | 原因 | 直したこと |
| --- | ------ | ---- | ---------- |
| Q1  | 不正解 | 欠落 | tutorial の `x`/`y` の箇条に、バッチなしの `y` の spec `TensorSpec((1,))` を足した（回帰の y の shape がどこにも無かった） |
| Q2  | 正解   | -    | 「int64, as in the table」が表に無い dtype を指していた（矛盾）。下の Q9 の直しで同じ文を書き換えた |
| Q3  | 正解   | -    | - |
| Q4  | 正解   | -    | - |
| Q5  | 正解   | -    | - |
| Q6  | 正解   | -    | - |
| Q7  | 正解   | -    | - |
| Q8  | 正解   | -    | - |
| Q9  | 正解   | -    | 根拠は推論で、確信が中。CrossEntropyLoss の `y` を「int64 で、`y_hat` からクラスの次元を除いた shape、または `y_hat` と同じ shape の確率」と一般形で書き直した（曖昧） |
| Q10 | 正解   | -    | - |

- 生徒の指摘で直さなかったもの: x の dtype（読者の作業に要らない。TensorSpec の既定）、自作損失の y の shape（y は利用者が決める spec そのもの）、「The mean gradient」への参照（warning の文だけで答えられていた）、Chain を使わないコード例（対象の範囲外）

### R2（en topics 129 行・en tutorial 48 行、増減なし）
| 問  | 判定   | 原因 | 直したこと |
| --- | ------ | ---- | ---------- |
| Q1〜Q5 | 正解 | - | - |
| Q6  | 不正解 | 曖昧 | 「時刻は batch の直後に置かずサンプルの中に入れる」と言いつつ x を (16, T, ...) と答えた。言葉が矛盾している。原文の「belongs to the sample, after the batch」は「サンプルの次元」と「バッチの直後」を別のものと読める。そこで「サンプルの次元で、バッチの次元ではない」と言い切り、`batch=64` で `x` が `(64, T, 784)` になる例を添えた（問題の数値 16 は使っていない） |
| Q7〜Q10 | 正解 | - | Q6 の確信が中だった点（時刻の次元を MSE が平均するか）に合わせて、`pc.MSELoss` が平均する次元を「各サンプルのすべての次元（バッチ以外）」に統一した。topics の warning と tutorial の両方で、「features」と「dimensions」が混在していた |

- 直さなかった指摘: 実数の目標の y を表に入れる（tutorial の例で足りる）、x の dtype、多次元の y_hat の明示（規則から正しく組み立てられている）、Output の引数名（範囲外）

### R3（en topics 129 行・en tutorial 48 行、改稿なし）
| 問  | 判定 | 原因 | 直したこと |
| --- | ---- | ---- | ---------- |
| Q1〜Q10 | 正解 | - | - |

- 全問正解なので改稿しない。次は保留問題 H1〜H3 を出して汎化を確かめる
- 直さなかった指摘: x の dtype、`pc.Output` の損失の引数名（範囲外）、自作損失の y の shape（利用者の spec そのもの）、損失のシグネチャ（`pc.MSELoss(batch)` の表記で足りる）

### 汎化確認（保留問題 H1〜H3、改稿なし）
| 問 | 判定 | 原因 | 直したこと |
| -- | ---- | ---- | ---------- |
| H1 | 正解 | - | - |
| H2 | 正解 | - | - |
| H3 | 正解 | - | - |

- 3/3 正解。評価用 10 問と合わせて終了。最終行数は en topics 129 行、en tutorial 48 行（R0 から増えていない）
- 直さなかった指摘: 例外の型（MSE は ValueError、CE は torch の RuntimeError で共通の型が無いため「an error」のまま）、回帰と分類の使い分け（対象範囲外の一般知識）、確率の y を表に入れること（表の下の箇条で足りる）
