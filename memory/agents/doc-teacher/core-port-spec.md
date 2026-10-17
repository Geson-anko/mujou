# `_port.py` / `_spec.py` docstring 理解度テスト

## 対象

- ファイル: `src/mujou/_port.py`、`src/mujou/_spec.py`（docstring とコメントだけ。英語）
- 読者と用途: mujou でノードを書く利用者。docstring だけを読んで、TensorSpec / DictSpec で Port を宣言し、接続し、ノードの中で recv / send / send_to / recv_with_time を正しく使う。

## 事実確認（R1 時点）

- `recv` は `Binding.recv`（`_session.py`）: 所有スレッド検査 → recv spec が無ければ TypeError → 各 source を `torch.from_dlpack(b).to(device, copy=True)`。multi は `{channel: value}`（接続順）、single は値 1 つ
- 未接続の single + recv spec はホストが書くスロット（`Network.write`）。multi の未接続は sources が空で `{}`
- 接続済みでも相手が未送信なら、Channel スロットの初期値（ゼロ、または復元値）が返る
- ノードのスレッドは `torch.set_grad_enabled(False)`。受け取る Tensor に勾配は無い
- `send`: `split` で検査 → `detach().cpu()` → outlet.send。外向きスロット（`Network.read`）も更新。`send_to` は Channel スロットだけ
- 検査のエラー: 型・dtype・非 strided は TypeError、shape・dict のキー過不足は ValueError
- 時刻は `message.time_ns`（u64、グローバル時刻 ns）
- `connect` のエラー順: build 済み RuntimeError → 自分自身 ValueError → spec 不一致 TypeError → 両方向 None TypeError → single に 2 本目 ValueError

## 評価用問題

### Q1 ノード A の `self.next = NextPort(send=h, recv=h)`（multi を指定しない）に B と C の 2 つをつないだ。B はもう送ったが C はまだ一度も送っていない。A の `on_update` で `self.next.recv()` は何を返すか。C の分の値も答えよ。
- 要点: `{Channel: Tensor}` の dict（接続順、相手ごとに 1 要素）／C の Channel の値は初期値（既定でゼロ）

### Q2 NextPort につないだ 3 つの相手に、それぞれ違う Tensor を送りたい。どのメソッドを使い、相手をどう指定するか。
- 要点: `send_to(channel, msg)` を使う／channel は `port.channels` の要素、multi の `recv()` の dict のキー、または `connect` の戻り値

### Q3 ノード A の `on_update` の中から、別のノード B の Port の `recv()` を呼んでよいか。呼んだらどうなるか。
- 要点: 呼べない（所有ノードのスレッドからだけ）／`RuntimeError`

### Q4 `self.out.send(x)` の直後に、同じ `x` を `x.add_(1)` でその場で書き換えた。相手が次に `recv` する値は書き換えの影響を受けるか。理由も。
- 要点: 受けない／send の時点で detach してコピーするから

### Q5 `self.inp = Port(recv=TensorSpec((3,)))` を未接続のまま残した。ノードで `self.inp.recv()` を呼ぶと何が返るか。その値を書き換えられるのは誰か。
- 要点: shape (3,) の Tensor（相手 1 つなので値そのもの）。書かれるまでは初期値、既定でゼロ／ホストが `Network.write` で書く

### Q6 send spec が `TensorSpec((4,))`（dtype 省略、既定は float32）の Port で、`torch.zeros(4, dtype=torch.float64)` を `send` した。どうなるか。
- 要点: `TypeError`／dtype は変換されない

### Q7 `a = Port(send=TensorSpec((3,)))` と `b = Port(send=TensorSpec((3,)))` を `a.connect(b)` した。どうなるか。つなぐには b をどう宣言すべきか。
- 要点: `TypeError`（spec が互いに逆向きでない）／b を `Port(recv=TensorSpec((3,)))` にする（a の send と b の recv を一致させる）

### Q8 `self.prev = PrevPort(recv=x)` に、2 つ目の相手を `connect` した。どうなるか。2 つの相手から受け取りたいなら宣言をどう変えるか。
- 要点: `ValueError`（相手 1 つの Port は 1 本まで）／`multi=True` を渡す

### Q9 `Network.build()` の後で `a.next.connect(b.prev)` を呼んだ。どうなるか。
- 要点: `RuntimeError`／接続は build の前に済ませる

### Q10 ある Port で `send_to` だけを使い、`send` は一度も呼ばない。ホストが `Network.read` でこの Port を読むと、ノードが送った値は見えるか。
- 要点: 見えない／`send_to` は `Network.read` が返す値を変えない（`send` だけが変える）

### Q11 相手 1 つの Port で `recv_with_time()` は何を返すか。時刻の単位と種類も答えよ。
- 要点: `(message, time_ns)` のタプル／書かれた時点のグローバル時刻、整数のナノ秒

### Q12 `DictSpec` に渡す TypedDict のキーの 1 つを `NotRequired[torch.Tensor]` にした。どうなるか。
- 要点: `TypeError`（全キーが required でなければならない）

## 保留問題

### H1 `Node(device="cuda:0")` のノードが `recv()` した Tensor は、どのデバイスにあり、勾配を持つか。
- 要点: 所有ノードのデバイス（cuda:0）にある新しい Tensor／勾配を持たない

### H2 ノードの `state_dict` の中で、自分の Port を `recv()` してよいか。
- 要点: できない（`state_dict` は別スレッドで呼ばれる）／`RuntimeError`

### H3 `Port(recv=x, multi=True)` を未接続のまま残した。`recv()` は何を返すか。
- 要点: 空の dict `{}`

## ラウンド記録

### R1（`_port.py` 413 → 427 行、`_spec.py` 317 → 321 行）

初稿の改稿。主な変更:

- 欠落: multi の `recv` で相手が未送信のときの値（初期値）、未接続の single をホストが `Network.write` で書くこと、受け取る Tensor に勾配が無いこと
- 曖昧: `send` の「readers of this port's output」を `Network.read` と明示。エラーを TypeError / ValueError に分けて条件を明記
- 曖昧: `multi` の既定値と `recv` の戻り値の対応を Port の docstring に明記
- 欠落: `send_to` の channel の入手元、`recv_with_time` の戻り値の形と単位
- 欠落: `MessageSpec.zeros` が Port の既定の初期値であること

### R2（R1 の解答を採点。`_port.py` 427 → 434 行、`_spec.py` 321 行で変化なし）

12/12 正解。Q10 で生徒は「send を呼ばないときにホストが読む値は判断できない」と書き添えた。ただし、その値は要点に含まれないので正解とした。

| 問 | 判定 | 原因 | 直したこと |
| -- | ---- | ---- | ---------- |
| Q1〜Q12 | 正解 | - | - |

読みにくかった箇所への対応（すべて妥当と判断した）:

- 2 ファイルに定義が無い `Component.connect` を、「2 つのノード（またはネットワーク）の間の `a.connect(b)`」に言い換えた
- 「zeros by default」が何の既定か曖昧だった。`recv` で、初期値を「recv spec の `zeros()`。ただし run の前にホストが書いた場合と `load_state_dict` で復元した場合を除く」と明記した
- 「written」の書き手が不明だった。`recv` に、書き手は相手の `send` / `send_to`、未接続で相手 1 つの Port ならホストの `Network.write` だと書いた
- `multi` の既定がクラスごとに違い、比べにくかった。`Port` の docstring に、既定は `Port` と `PrevPort` が 1 つ、`NextPort` が複数だと 1 文でまとめた
- `send_to` だけを使う Port で `Network.read` が返す値（初期値）を書いた
- `Network.write` / `Network.read` の定義は対象外のファイル（`_network.py`）にある。参照する名前として残した

### 保留問題（`_port.py` 434 行、`_spec.py` 321 行。改稿なし）

3/3 正解。汎化を確認できたので終了した。

| 問 | 判定 | 原因 | 直したこと |
| -- | ---- | ---- | ---------- |
| H1〜H3 | 正解 | - | - |

生徒の指摘は次のとおり。改稿は不正解があるときだけ行う指示なので、どれも次回以降の候補として残す。

- 「owning node's thread」と「`state_dict` runs on another thread」が 2 ファイルの中で説明されていない。説明は `_node.py` にある
- Raises の RuntimeError の項目に、「not built」と「not the owning node's thread」が同居している
- `None` を渡さない理由（型チェッカーが `Never` を推論するため）は、利用者から見ると内部事情に寄っている
- 型変数のコメントにある pre-commit と Python 3.13 の理由は、開発者向けで、利用者には分かりにくい
