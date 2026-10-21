# 提供シナリオとガード

mujou が支える使い方（提供シナリオ）と、その中で起こりうる利用者の誤りを知らせる検査（ガード）の正本。ここに無い検査は書かない。利用者向けには、例外を出す API の docstring の `Raises:` に書く（mkdocstrings で docs に出る）。docs に台帳の写しは置かない。

## 判定規則

ガードを残すか消すかは、次の 6 つで決める。

| 記号 | 規則 |
| --- | --- |
| G1 残す | 黙って誤った結果になる誤り、または原因から遠い場所で壊れる誤り。最も早い 1 か所で、利用者が書いた名前（`Scale.next`、`body.l1`）を使って知らせる |
| G2 消す | Python、torch、Rust が、利用者の書いたその行で分かる例外を自然に出すもの |
| G3 消す | 型注釈がすでに表している誤りを、実行時にもう一度確かめているもの |
| G4 データ構造で消す | 内部どうしの呼び出し契約は、表せない形にする。FFI の健全性に要るものだけを内部エラーとして残す |
| G5 消す | 提供シナリオの外への備え |
| G6 状態ごと消す | 特別扱いの状態（build 済みで未実行、巻き戻し、凍結したポートなど） |

1 つの誤りは、最も早い 1 か所で 1 回だけ知らせる。同じ検査を複数の層に重ねない。

## 提供シナリオ

| ID | シナリオ | 利用者の操作 |
| --- | --- | --- |
| S1 | ノードを書く | `class X(mujou.Node)`。`__init__` で `super().__init__(...)` を呼び、ポートと状態を属性に代入し、hook を override する。自作の学習則もここに書く |
| S2 | ネットワークを組む | `Network` のサブクラスか `Group(**named)` に属性で並べ、`connect` で繋ぐ（循環してよい） |
| S3 | 走らせる | ホストのスレッドで `mujou.run(root)` を呼び、`with` の中で write / read / subscribe / wait_for / state_dict を使う。subscribe を別のスレッドで回し、main から shutdown してよい。段 1 か段 2 で断られた run を直して、同じインスタンスで再実行してよい。終わった後に、ノードの属性を読んでよい |
| S4 | 時間 | グローバルタイマーか、実行ごとの `Timer` を使う。sleep / pause / time scale |
| S5 | 保存と再開 | `state_dict` → `torch.save` → 同じ形に作り直した新しいインスタンスで `mujou.run(..., state=tree)`（保存した時刻から始まる専用のタイマーで続く） |
| S6 | テスト | `mujou.testing.run` と `ManualRun.step(dt, *nodes)` |
| S7 | PC-ALM で学習する | `pc.Chain` / `pc.show`、個別のノード（`act_spec` で繋ぐ）、`pc.LearnClock` と `pc.functional` による自作ノード |
| S8 | ノードの例外 | 全ノードが止まり（`on_exception` → `on_stop`）、ホストに `NetworkError` が届く |
| S9 | CUDA | ノードの `device="cuda"` |

## サポートしないもの

次の使い方には備えない。検査も救済も書かない。

- 同じネットワークを、複数のスレッドから同時に run すること
- 走っている run の構造を変えること（走った後の `connect` は何もしない）
- 手で書き換えた状態の木、mujou 以外が作った状態の木
- 種類の違う device への復元（CUDA のノードの乱数生成器の状態を CPU のノードに戻すなど）。S5 は同じ形に作り直すことを前提にする
- ノードの `copy.copy`
- Node のサブクラスで `__eq__` を定義すること（`eq=True` の dataclass を含む。hash できなくなる）
- 型注釈に反する値（pyright か IDE を使うことを前提にする）
- ノードが内部を辿って、タイマーやランタイムを操作すること
- `with` も `shutdown` も使わず、non-daemon のスレッドで subscribe したままインタプリタを終えること
- CPU と CUDA 以外の device
- CUDA が使えるのに `torch.cuda.set_device` や `set_stream` が失敗すること（後片付けの hook も同じ例外になる）
- 未接続の `MultiPort` にホストが書くこと（K17 が「ホストの入力には Port を使う」と案内する）
- KeyboardInterrupt を受けた位置ごとの特別な扱い（Ctrl-C で `with` を抜ければ shutdown される、という規則だけを持つ）
- 復元した実行を、`global_timer` やほかの実行と同じタイマーで走らせること
- view による、ノードをまたいだ部分的なテンソルの共有（K24 が見つけるのは同じオブジェクトだけ）
- 周期どおりに決定的に回す手動モード
- `workers` に、ネイティブの整数に収まらない値を渡すこと

## ガード台帳

起動は 3 段で進む。段 1 は木の検査と引数の照合で、何も変えない。段 2 はタイマー・スロット・Driver を新しく作り、木の復元と inputs の書き込みを行う。変わるのは新しいオブジェクトだけなので、段 1 と段 2 で断られた run は、原因を直せば同じインスタンスで走る。段 3 で各ノードに走った印を付けて起動する。

| K | シナリオ | 利用者の誤り | 知らせる場所（1 か所） | 例外 | 残す理由 | テスト |
| --- | --- | --- | --- | --- | --- | --- |
| K1 | S1/S2 | Node / Network のクラス属性に `Port`・`MultiPort`・`Node`・`Network` を置く。`__slots__` を持つ（`@dataclass(slots=True)` の Network など） | `Component.__init_subclass__`（クラス定義の時点） | `TypeError` | `__dict__` に入らないので run から見えず、黙って無視されるか、インスタンスの間で共有される | `_core/test_component.py` の `TestClassAttributes` |
| K2 | S1 | `super().__init__()` を呼ばない | `tree.walk`（段 1） | `TypeError` | 素の `AttributeError` では原因が分からない | `_core/test_tree.py` の `test_a_node_that_did_not_call_the_base_init` |
| K3 | S1 | ノードの属性に Node / Network を代入する | `tree.walk`（段 1、パス.属性名） | `TypeError` | 子が走らないまま、黙って進む | `_core/test_tree.py` の `test_a_node_holding_a_node_or_a_network` |
| K4 | S9 | `device` が cpu / cuda 以外、または CUDA が使えないのに cuda | `Node.__init__` | `ValueError` | stream の同期が無いまま走るか、実行中の遠い場所で壊れる | `_core/test_node.py` の `TestDevice` |
| K5 | S1 | `period` が負、または NaN | `Node.__init__`（`not period >= 0` の 1 行） | `ValueError` | 起動後に、ノードの外で壊れる | `_core/test_node.py` の `TestPeriod.test_rejects_what_is_negative_or_nan` |
| K6 | S1/S2 | Node か Network の属性に、Port・Node・Network を含む list・tuple・dict を代入する | `tree.walk`（段 1） | `TypeError` | 黙って抜け落ち、Port なら後で K12 が誤った誘導をする | `_core/test_tree.py` の `test_parts_in_a_list_a_tuple_or_a_dict` |
| K7 | S1 | `TensorSpec` にサポート外の dtype を渡す | `TensorSpec.__init__`（`TensorSpec.dtypes` を示す） | `TypeError` | Schema を作る段 2 という、遠い場所で壊れる | `_core/test_spec.py` の `test_rejects_unsupported_dtypes` |
| K8 | S1 | `DictSpec` のキーワード引数が TypedDict のキーと一致しない | `DictSpec.__init__` | `TypeError` | 黙って欠ける | `_core/test_spec.py` の `test_missing_key`、`test_unknown_key` |
| K9 | S1/S3/S5 | send / send_to / write / inputs / initial / 復元する値の、構造（Tensor か、dict のキーの過不足）または shape・dtype の誤り | 構造は `codec.to_core`、shape・dtype は Rust の `copy_from`。どちらもポートの label とフィールドを名指しする。inputs・initial・復元は段 2 | 構造は `TypeError`、shape・dtype は `ValueError` | 最も多い誤り。2 か所はそれぞれ、自分にしか見えないものを 1 回ずつ見る | `_core/test_codec.py` の `TestStructure`、`_core/native/test_slot.py` の `test_rejects_mismatch_naming_slot_and_field`、`_core/test_port.py` の `TestInitial` |
| K10 | S2 | `connect` で spec が逆向きに一致しない | `Port.connect`（両端の spec を示す） | `TypeError` | 根本の誤り | `_core/test_port.py` の `test_rejects_specs_that_are_not_reversed` |
| K11 | S2 | 受け取る `Port` に、別の相手を繋ぐ（どちらの端から繋いでも） | `Port.connect`（両端を見て「MultiPort を使う」） | `ValueError` | recv が曖昧になり、黙って片方だけを読む | `_core/test_port.py` の `test_a_port_that_receives_takes_one_peer` |
| K12 | S1 | Port の recv / recv_with_time / send / send_to を、所有ノードの hook の外で使う（`__init__`、他のノード、ホスト、run の外、`on_start` で作った Port、そのノードの `state_dict` の中）。`Node.timer` をそのノードの hook の外で読む | Port は Driver の `wires` の lookup（`port.py` の `_wire(port)`）、timer は `Node.timer` が走っている Driver のノードと照合する。どちらも `current.running()` を通し、文言は `current.outside(name, hint)` の 1 つの文に、用途ごとの案内（Port なら `__init__` での宣言と `run.read` / `run.write`、timer なら `run.timer`）を足す | `RuntimeError` | 局所性の不変原則。状態を取るための停止が、他のノードとホストから見えないようにする（`state_dict` の間は Driver の `wires` が空） | `_core/test_port.py` の `TestOutsideTheHooks`、`_core/test_node.py` の `TestTimer`、`_core/test_state.py` の `test_the_ports_refuse_in_the_node_state_dict` |
| K13 | S2 | 同じオブジェクト（Node・Network・Port）が、木の 2 か所にある | `tree.walk`（段 1、両方のパス） | `ValueError` | 1 つのノードが 2 回走るか、1 つのポートに書き手が 2 人できる | `_core/test_tree.py` の `test_the_same_node_at_two_places` ほか |
| K14 | S3 | 走ったことのあるインスタンスを、もう一度走らせる | `run` の段 1（走ったことのある最初のノードを名指しする） | `RuntimeError` | notebook で起きやすい | `_core/test_run.py` の `test_runs_once` |
| K15 | S2 | 繋いだ相手のノードを、木に入れ忘れる | `tree.walk`（段 1） | `ValueError` | recv が zeros のまま、黙って進む | `_core/test_tree.py` の `test_a_port_connected_to_one_outside_the_tree` |
| K16 | S3/S4 | hook の中から `mujou.run`、`Run` の操作、`Timer` の制御（sleep / pause / resume / set_time_scale / advance）を呼ぶ | Python の `run._host()` と、bindings の `Timer` の制御メソッド（述語はどちらも `current()`） | `RuntimeError` | ワーカーを占有して他のノードの dt を伸ばすか、デッドロックする | `_core/test_run.py` の `TestHostMethodsFromANode`、`TestOnlyTheHostControlsTheTime` |
| K17 | S3 | 書けない Port（接続済み、`MultiPort`、送るだけ、この run の外）に `write` / `inputs` する。終了後に `write` する | `Wiring.write`（`Run.write` と、段 2 の inputs）。終了後は `Run.write` | 書けない Port は `ValueError`、終了後は `RuntimeError` | 黙って誰も読まない | `_core/test_run.py` の `test_rejects_ports_it_cannot_write`、`test_raises_after_a_normal_end` |
| K18 | S3 | 送らない Port や、ほかの run の Port を read / read_with_time / subscribe / wait_for する | `Run` の outlet の lookup（1 か所） | `ValueError` | 1 回の lookup で済む | `_core/test_run.py` の `TestRead` |
| K19 | S6 | run に無いノードを `step` に渡す | `ManualRun.step` | `ValueError` | 黙って飛ばされる | `_core/test_run.py` の `test_rejects_a_node_of_another_run` |
| K20 | S3 | 明示した `workers` が 1 未満 | `run` の段 1 | `ValueError` | 印を付けた後で native の変換に失敗すると、インスタンスを使い切ってしまう | `_core/test_run.py` の `test_rejects_less_than_one_before_starting` |
| K21 | S5 | 形の違うネットワークに木を復元する（`timer` の項目、ノードのパス、ポートの属性、スロットのキーの集合） | `state.saved_time`（段 2 の最初、木の上の段）と `state.restore`（段 2、ノード以下）。足りない項目と余分な項目を列挙する | `ValueError` | S5 で最もありうる誤り | `_core/test_state.py` の `test_lists_what_does_not_match`、`test_lists_what_its_top_does_not_match` |
| K22 | S5 | クラスを変えた後で、古い状態を読む（キーの過不足、テンソルの shape の違い） | 既定の `Node.load_state_dict`（クラス名とキー） | `ValueError`（`NetworkError` の `__cause__` として届く） | `copy_` は broadcast で黙って通ってしまう | `_core/test_node.py` の `test_rejects_a_state_that_does_not_match` |
| K23 | S4/S5 | `timer=` と `state=` を同時に渡す | `run` の段 1 | `ValueError` | 渡したタイマーで走ると思い込んだまま、別の時計で走る | `_core/test_run.py` の `TestRefusedRun`（`timer-and-state`） |
| K24 | S1/S2 | 2 つのノードが、同じテンソル、同じ `state_dict` を持つ物、または module の同じ Parameter・buffer（重みの tying）を状態に持つ | `tree.walk`（段 1、両方のパス.属性名。module の中は `.キー` まで） | `ValueError` | free-threaded でデータ競合になり、黙って誤った結果になる | `_core/test_tree.py` の `test_two_nodes_holding_the_same_state`、`test_two_nodes_holding_the_same_parameter` |
| K25 | S7 | `TimeConstants` の h が有限で > 0 でない、lam または theta が > 0 でない（inf は可） | `TimeConstants.__post_init__` | `ValueError` | NaN になるか、学習しない | `pc/test_time_constants.py` |
| K26 | S7 | `learn_interval` が None でも > 0 でもない（inf は可）、`max_dt` が > 0 でない、`period > max_dt` | `LearnClock.__init__` | `ValueError` | 学習しないか、時間を黙って捨てる | `pc/test_clock.py` の `TestChecks` |
| K27 | S7 | 平均を取る損失（`nn.MSELoss()` など）を渡す | `functional.loss_vjp` の (B,) の検査（`Output` の構築時の probe も同じ関数を通る。`pc.mse` を案内する） | `ValueError` | 黙って誤った勾配になる | `pc/test_functional.py` の `test_loss_must_be_per_sample`、`pc/test_output.py` の `test_a_loss_that_averages_is_refused` |
| K28 | S7 | h と y の先頭の次元（B）が違う（module の出力の先頭の次元が B でない場合を含む） | `activity_spec` | `ValueError` | mask の行が、黙って broadcast される | `pc/test_messages.py` の `test_h_and_y_must_have_the_same_batch` |
| K29 | S7 | `vjp` の δ と μ の shape が違う、`mse` の y と ŷ の shape が違う | `functional.vjp` / `mse` | `ValueError` | broadcast が黙って通る | `pc/test_functional.py` の `test_delta_must_have_the_shape_of_mu`、`pc/test_losses.py` の `test_shapes_must_match` |
| K30 | S1 | `register_state` に、まだ代入していない名前を渡す（綴りの誤り、property） | `Node.register_state` | `AttributeError` | `__dict__` から決めるので、綴りを誤った名前は黙って状態から漏れる | `_core/test_node.py` の `test_rejects_a_name_without_an_attribute` |

テストのパスは `tests/mujou/` からの相対パス。

## ガードではない意味論

次の例外は誤りの検査ではなく、API の約束として出す。

- `TimeoutError`: timeout（実時間の秒）を過ぎた。`Run.state_dict` は状態を取らずに実行を続け、`shutdown` は止める要求を残し、`join` / `subscribe` / `wait_for` は何も変えない
- `NetworkError`: ノードの例外による終了と、その後のホストの呼び出し（`write`、`read`、`read_with_time`、`subscribe`、`wait_for`、`state_dict`、`shutdown`、`join`、`step`）。`__cause__` にノードの例外が入る
- `Run.__exit__` が shutdown すること。本体の例外が先に出ていれば、shutdown の `NetworkError` は落とす
- `wait_for` が、条件を満たす前に実行が正常に終わったときの `RuntimeError`
- 正常に終わった後の `ManualRun.step` の `RuntimeError`
- pause 中の `VirtualTimer.advance`（`ManualRun.step` を含む）の `RuntimeError`（タイマーの契約）
- 正常に終わった後の `Run.state_dict` が、ノードの `state_dict` の例外をそのまま出すこと（実行は正常に終わったまま）
- `BufferView.__dlpack__` の `BufferError`（DLPack の約束）
- 抽象メソッドは `abc.abstractmethod` で表す（`raise` を書かない）
- Rust の `Reentrant` の検査（コアの内部のデッドロックを防ぐもの。Python からは K16 が先に止めるので届かない）
- FFI の健全性の検査（内部エラー）。文言は作り込まない。利用者に届いたら mujou のバグ
  - `Outlet` / `Slot` / `InputGroup` に渡す値の個数がフィールドの数と違う（codec はフィールドごとに 1 つ渡す）
  - DLPack で読めない値、ホストのメモリに無い値（`FieldError::InvalidDLPack` / `UnsupportedDevice`。codec は CPU のテンソルだけを渡す）
  - `Outlet` に無いチャネルの番号（`IndexError`。`Wire.targets` は connect の順の番号だけを持つ）
  - `InputGroup.write` に別のグループのスロットか、同じスロットを 2 回渡す（コアが panic する。`Wiring.write` は、その run の Port から引いたスロットを 1 回ずつ渡す）
  - `RuntimeError::Internal`（スレッドを作れない、capture の build が panic した、手動でない runtime に `step` を頼んだ、など。`step` を呼ぶのは `ManualRun` だけ）
  - BufferView の `__dlpack__` に DLPack 1.x の `max_version` が無い（`BufferError`。torch の `from_dlpack` は必ず渡す）

## 自然に出る例外（G2）

次の誤りには検査を書かない。torch・Python・Rust が、その行で分かる例外を出す。

- 密でないテンソル（sparse など）を送る: torch の `__dlpack__` が `BufferError` を出す（`Port.send` の docstring が約束し、`_core/test_port.py` の `test_rejects_a_tensor_that_is_not_dense` が固定する）
- `TensorSpec` の次元が 1 未満: Rust の `Schema` が段 2 で `ValueError` を出す
- `DictSpec` に TypedDict でないクラスを渡す: キーが無いので、キーの照合（K8）が断る。キーの注釈が `torch.Tensor` でない TypedDict は、pyright が見る型の誤り（G3）
- `Timestamp(ns=)` が 0 から 2⁶⁴ − 1 の外: 整数の変換が `OverflowError` を出す
- `state_dict` / `shutdown` / `join` の timeout が負か NaN、`sleep` / `advance` の秒が有限で 0 以上でない、time scale が有限で 0 より大きくない: コアの時間（単調に進む ns）への変換が `ValueError` を出す。`shutdown` は止める要求を出す前に変換するので、断られた呼び出しは何も変えない
- `subscribe` / `wait_for` の timeout が負: 期限がもう過ぎているので、最初の値の次の待ちで `TimeoutError` になる（最初の値は待たずに渡す）。NaN は待ちの秒の変換が `ValueError` を出す
- `learn_` に渡す列の長さの違い: `zip(strict=True)` が `ValueError` を出す
- 引数の型の誤り（`TimeConstants` に文字列を渡す、など）: pyright が見る（G3）。実行時は、その値を使う演算が例外を出す
