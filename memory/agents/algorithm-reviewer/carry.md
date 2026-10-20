# algorithm-reviewer: carry（付随データのゼロコピー中継）

対象: `git diff feature/2026-10-20/node-register-state...feature/2026-10-20/carry`。観点: 実行モデルの不変条件（局所性・非同期性・状態の所有・停止と復元）と、PC-ALM のターゲット中継（`specs/pc-module.md`「メッセージと Port」）に足りるか。更新式そのものは diff に無い。

## verdict

**approve**（must なし、should 1 件、検討 5 件）。確信度: 高（不変条件について）。PC の用途に足りるかは中〜高（`mujou.pc` が未実装なので、仕様の手順と突き合わせた判断）。

- 中継元は同じノードの Port に限られる。`graph.build_tree` が所有ノードの添字を比べ、`Network.build` と `mujou.run` の両方で効く。中継で読むのは自分の Binding の `_carry` だけで、ほかのノードのスロットを読まない
- バリア・待ちは増えていない。`check_relay` と `fill` は lock の外で行い、lock の下は従来どおり `Arc` の差し替えだけ。プールへの返却で別ノードの pool の mutex を短く取るが、これは従来の読み手の `Arc<Message>` の解放と同じ種類の競合
- torn read は構造的に起きない。値と付随データは 1 つの `Arc<Message>` に入る。`Binding.recv` は 1 回の `slot.read()` から値と `carry` を取り、`make_tree` も同じ。Rust の `relayed_carry_stays_with_its_value`（4 本の読み手 × 2 万回）と、Python の `test_is_the_carry_of_the_value_received_in_any_order`（固定 seed のランダム順 60 ステップ）がこれを確かめる
- 状態の所有: 中継で共有するバッファは publish 前に埋め終わり、以後は不変（`bytes_mut` は `Arc` 化の前だけ）。書き手は 1 人のまま
- 停止と復元: 木にスロットの付随データと、各 Port が最後に受け取った付随データ（`"carries"`）の両方が入る。`recv` より先に `send` するノードが再開後も止めずに回した場合と一致することを、`test_a_restored_run_goes_on_as_one_never_stopped` が確かめる。変異（`binding.carry = ...` を外す）で落ちることも実装者が確認済み。hook の途中では止まらないので、`recv` と中継の `send` の間で木を取ることは無い
- 更新手法の焼き付け: コア（`slot.rs`）・bindings・`port.py`・`binding.py`・`session.py`・`state.py` に PC の語（target・mask・λ）は無い。付随データはフィールドの列としてだけ見える。`pc.carry_spec` は `mujou.pc` 側の予定
- PC の用途: `Input.next(send_carry=spec)` → `Layer.prev(recv_carry)` / `Layer.next(send_carry=self.prev)` → `Output.prev(recv_carry)` で表せる。`Output` は `h_in` と同じメッセージの `y`・`mask` を組み、ターゲットは経路の遅れの分だけ自動で遅れる。付随データの初期値がゼロ（`mask` が False）なので、最初のサンプルが届く前の `Output` は教師なし扱いで学習しない

## must-fix

なし。

## should-fix

### S1. 付随データの初期値（ゼロ）が、実際に送られたゼロと区別できないことを書く（確信度: 高）

- 中継するノードは、最初の `recv` の前は中継元のスロットの初期値（ゼロ）を送る。下流の Port の `has_received()` は中継したノードが送った時点で True になるので、下流からは「本物の付随データ」か「まだ何も届いていないゼロ」かが分からない
- PC は `mask` が False なので害が無い（仕様どおり）。汎用の用途（サンプル ID・時刻）では、ID 0 や時刻 0 と衝突する。`examples/carry.py` も、VirtualTimer の時刻 0 で書くと初期値の報告（`id == 0`）を自分のサンプルの報告と取り違えうる（グローバルタイマーなら実際には起きない）
- 直し方: `specs/network.md`「付随データ」と user guide の "Carrying data along a path" に 1 文足す。「最初の値が届く前の付随データはゼロで、送られたゼロと区別できない。区別したいときは、付随データに有効かどうかのフィールドを入れる（PC の `mask` がこれに当たる）」

## 検討

### K1. 初期値を区別する手段をコアに持たせるか

S1 を API で解く案。どれも今は要らない（PC は `mask` で足りる）。

- (a) `initial_carry=`（計画で「要るまで足さない」とした口）
- (b) 付随データに「作る側が送ったものか」の印を持たせ、中継で一緒に運ぶ（`Message.sent` と同じ 1 bit を `Carry` に足す）。`Port.has_carried()` で読む。初期値と復元の扱いは `sent` に倣う

### K2. `pc.Layer` の学習の位相の判定（`time` の変化）と時刻 0

`specs/pc-module.md`「on_update の手順」2 は `time` が変わったら位相を 0 に戻す。VirtualTimer の時刻 0 で `inputs=` を書くと、最初のサンプルの `time` が初期値のゼロと同じになり、切り替わりとして見えない。位相は開始時に 0 なので実害は小さいが、`pc` を実装するときに「最後に見た `time` の初期値」を明示するか、`mask` も見るかを決めておく。

### K3. 相手が複数の Port の `recv_carry`

今は禁止（どのメッセージの付随データか決まらない）。`recv` と同じく Channel ごとに持ち、`carried(channel)` で読む形にすれば決まる。必要になる場面:

- 下向きの $g$ にサンプルの印を付けたいとき（`Output.prev` が `send_carry=self.prev` で $g_{\mathrm{in}}$ に付随データを返し、`Layer.next` で照合する）。`specs/pc-module.md` が残すずれ「古い credit が新しい入力のヤコビアンを通る」を、ノードの内側で検出・抑制できるようになる。`Layer.next` は multi なので今の API では書けない（相手 1 つなら `NextPort(multi=False, recv_carry=...)` で書ける）
- ニューロンノード（future work）。ニューロンノードは結合ごとに入力を持つ。付随データはサンプル単位の束で、「結合ごとの値の束」という契約からは外れる。レイヤー・スタックでは契約は変わらない（ノードの分け方を変えても、遅れは値と付随データで同じだけ変わる）

### K4. ホストが付随データを持つ Port を読めない

`pc.Layer.next` は `send_carry` を持つので、ホストは隠れ層の活動を `Run.read` / `subscribe` で観測できない。`connect` も相手に `recv_carry` の宣言を求めるので、観測用のノードも付随データの spec を知っている必要がある。案:

- (a) `Run.read` などは付随データを捨てて値だけ返す（値の意味は付随データの有無で変わらない。計画の確認事項 3 の代案）
- (b) `connect` で「送る側だけが付随データを持つ」を許し、受け取る側は付随データを読まない（中継もしない）

どちらも局所性・非同期性は壊さない。観測の需要が出たときに決める。

### K5. 合流での遅れの違い

合流するノード（ResBlock をノードをまたいで分けた場合など）では、経路ごとに段数が違うと、入力ごとに別のサンプルが届く。carry はこれを解かないが、各入力の `time` を `carried` で比べれば検出できる。PC の仕様（「付随データは入力のうちの 1 つから中継する」）に、どの入力から中継するか（段数の多い方）と、ずれを検出するかを書き足すとよい。

## 検証結果

- `just test-fast`: Rust 全 crate ok、pytest 639 passed / 15 skipped
- `uv run pytest -m slow -k "carry or Carr"`: 1 passed（`test_free_running_nodes_keep_values_with_their_carry`）
- `just type`（pyright strict）: 0 errors
- `just run` は回していない（format がほかの agent と共有している作業ツリーのファイルを書き換えうるため。format と docs-build は code-reviewer に任せる）

## 照合メモ

- 値と付随データの組: `Message { buffers: Vec<Arc<Buffer>>, carry_from }`。`Binding.recv` は `message = slot.read()` から `message_value`（`message.buffers` = 値だけ）と `message.carry` を取る → 同じスナップショット
- 中継の経路: `Binding._message` → `self._relay._carry`（同じノードの Binding。build で同じノードと検査）→ `Outlet.send(values, relay=carry)` → Rust `send_with(relay)`: `check_relay`（FieldSpec の等価）→ 値のプールだけ `fill` → `relay` の `Arc` を複製 → 1 つの `Message` を publish
- 初期値: Session は全 outlet を作り `initial` を書いた後で Binding を作り、`_carry = source.read().carry`（ゼロ）。復元は Binding を作った後に `slot.restore(値 + 付随データ)` と `binding.carry = source.new_carry(...)`
- multi: `recv_carry` は single だけ（`_init` と overload の両方）。送る側の multi は `Target::All` で同じ `Message` を全 Channel へ、`send_to` も同じ付随データ
- 1 つのノードの hook は重ならず、前の hook の書き込みは次の hook から見える（M:N の不変条件）ので、`_carry` の書き（recv）と読み（send）はスレッドをまたいでも競合しない
