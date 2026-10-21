# 作り直しの持ち越し事項（orchestrator がまとめる）

ステップ 1〜5 のレビューと実装者の報告で、後のステップに回された事項。担当ステップの実装者は、自分の範囲に入るものを片付ける。

## orchestrator の裁定（ステップ 5 の後）

- ステップ 5 の「CUDA の経路が未実行」「R13 未計測」は、GPU の無い環境の制約として受け入れる。ユーザーへの報告に「GPU のマシンで TestCuda（slow を含む）と R13 の計測が要る」と書く
- state_dict の中で Port の送受信を断る振る舞いは残す（K12 の一部。capture の間は Driver の wires が空、というデータの形で表す）。ステップ 8 で port_owner のフラグを wires の lookup に置き換える
- Node.timer は self と照合する（別のノードの timer を読めば K12）。ステップ 8 で K12 の lookup を 1 つにするときに直し、K12 の文言も 1 つにする
- component.py の関数内 import はステップ 9 で登録の仕組みごと消す
- CUDA の設定（set_device / set_stream）が失敗したときに後片付けの hook も同じ例外になる件は、提供シナリオ S9 の外（CUDA が使えるのに set_device が失敗する）として受け入れる。台帳には載せない
- 根のノードのポートのスロット名の変更は受け入れる（後方互換は考えない）
- 密でないテンソル（sparse）を送ったときの BufferError のテストは、ステップ 11 で 1 件足す
- `hint.rs` の `Views` の削除はステップ 6（carry の廃止で値だけの列になるとき）で行う
- R12 の密な y の基準値は、ステップ 6 で carry を消す前に測る
- 確認事項 R1〜R8 は推奨案で進めている。ユーザーの確認はまだ取っていない（作業の最後にまとめて聞く）。memory への記録は、確認が取れてからステップ 13 で行う

## ステップ 1

### 実装者の open_issues（最後の報告）

- GIL ありのビルドの環境（.venv-gil）が無く、GIL ありの基準値（ports と mn_scheduler）はまだ無い。R5（終了テストを 3.14t と GIL ありの両方で通す、ステップ 9）でもこの環境が要る。
- GPU が無いので、GPU のノードの capture（R10）と CUDA のノードのコスト（R12・R13）は未計測。GPU のマシンで benchmarks/capture.py --device cuda を測り、specs/research/benchmarks.md に足す必要がある。
- MNIST の time scale 1 での epoch の時間は、ホストが各バッチを見せる時間（938 バッチ × 32 ms ≒ 30 s）でほぼ決まり、計算の悪化に鈍い。R12 の「悪化が 5% 未満」は、time scale 2 の epoch の時間（15.3 s）と l1 の更新回数（30,567）で判定することを勧める（benchmarks.md に書いた）。
- capture の CPU 値は 65,536 要素（0.237 ms）の方が 200,704 要素（0.119 ms）より遅く、測定のノイズかアロケータの影響とみられる。R10 で比べるときは測り直す。
- 名前の重なり: nodes.Probe と test_node.py にある別の Probe、nodes.Pair（Node）と test_component.py の Pair（Network）が共存している。test_component の登録系のテストを消すステップ 9 で解消する見込み。

### 最後のレビューの should_fix / nits

- [should] specs/research/benchmarks.md（「状態の取得（capture）」節の末尾）／ステップ 1 の (b) と (c) の「基準値が記録されている」のうち、GPU のノードの capture の値が未計測のまま。R10 を判定するステップ 9 で比べる相手が無い。plan-implementer にはこの環境で直せない（GPU が無い）ので、orchestrator が「GPU のマシンで測るまでステップ 1 を未完とするか、ステップ 9 の前提に回すか」を決める必要がある。／根拠: 設計書 8 節ステップ 1 の (b)「GPU のノードの capture の値を specs/research/benchmarks.md に記録する」、R10（2128 行）／確信度: 高／深刻度: 中
- [should] specs/research/benchmarks.md（MNIST の節）／R12 は密な y（pc_regression）での悪化を記録することを求めているが、その基準値が無い。carry を消すステップ 6 の後では『前』の値を測れなくなるので、carry があるうちに測っておく必要がある。こちらは CPU でも測れる。／根拠: 設計書 D7（613 行「MNIST と pc_regression の密な y の両方で測り、結果を benchmarks.md に記録します」）、R12（2130 行）、ステップ 6 の (c)「密な y（回帰）の悪化を記録する」。ステップ 1 の (b) は MNIST しか挙げていないので、明示の要件ではなく抜けの可能性として報告する／確信度: 中／深刻度: 中
- [should] tests/test_layers.py:21-40（LAYERS）と設計書 4.1／設計書は node と network を同じ L5 に置くが、テストは同じ行どうしの import を禁じているため、network を node の 1 つ上の行に分けた。テストが層の規則の正本（4.1 節、D1）なので、どちらかに揃えないと正本が 2 つになる。理想の形でも Group の注釈には Node が要るので、テストの方が正しい。orchestrator が設計書 4.1 の層図を直す（L5 を node、L5' を network に分ける）べき。binding を port の 1 つ上に置いたことと、timer を spec と同じ行に置いたことも、層図に無い行として同じく残る／根拠: 設計書 4.1「L5 _core/node.py _core/network.py」、272 行「このテストが、層の規則の正本です」、実装者の deviations／確信度: 高／深刻度: 低
- [should] tests/test_layers.py:84-109／4.1 節は『遅延 import、TYPE_CHECKING で逃がした依存』も 0 にするとしている。しかしテストが数えるのは層の向きと _native の規則だけで、下の層への遅延 import や TYPE_CHECKING の import は新しく足しても通る。今ある非トップレベルの import は component.py:14,15,153 と port.py:26,27 の 5 つで、port.py:26（TYPE_CHECKING の _native）以外は向きの違反として ALLOWED に載っているので、今は抜けが表に出ていない。ラチェットとして『今より増えない』ことを保つには、トップレベル以外の import も許可リストで数える必要がある。ステップ 1 の範囲の外と判断するなら、どのステップで入れるかを設計書に書いておく／根拠: 設計書 4.1「循環、遅延 import、TYPE_CHECKING で逃がした依存…は、どれも 0 にします」／確信度: 中／深刻度: 低
- [nit] tests/mujou/_core/pc/test_layer.py:1240-1241（test_no_theta_moves_while_every_mask_is_false）／設計書は『全ノードの θ』と書いているが、テストはパスを ("l0", "l1", "out") に固定している。after の中で "module" を持つノードをすべて回す形にすると、widths を変えても漏れない。mask を MASK に変えると 2 通りとも失敗することは、こちらでも確かめた（オラクルとして効いている）／確信度: 中
- [nit] tests/test_layers.py:50-67（ALLOWED）／許可を (importer, imported) の組で持つので、すでに許可された組の import を別の場所（別の関数の中など）に足しても通る。ラチェットの粒度の問題で、実害は小さい／確信度: 中
- [nit] pyproject.toml:66（pythonpath = ["."]）／tests/ に __init__.py が無いので、tests は名前空間パッケージとして解決される。依存のどれかが site-packages に通常のパッケージ tests を入れると、そちらが優先され、tests.mujou.nodes が import できなくなる。今の環境では /data/geson/mujou/tests に解決されることを確かめた（問題は出ていない）／確信度: 低
- [nit] tests/mujou/_core/pc/test_layer.py:22 と tests/mujou/_core/pc/test_output.py:18／F64 = torch.float64 の定義が tests/mujou/nodes.py と重なって残っている／確信度: 高
- [nit] tests/mujou/nodes.py の Probe と tests/mujou/_core/test_node.py:23 の Probe、nodes.Pair（Node）と tests/mujou/_core/test_component.py:18 の Pair（Network）は、同じ名前で中身が違う。実装者はステップ 9 で解消すると報告している。それまでの読み違いを防ぐなら、test_node 側の名前を変えておくとよい／確信度: 高
- [nit] benchmarks/ports.py:15／docstring の 1 行が 119 文字ある（ruff は docstring の長さを検査していないので通っている）／確信度: 高
- [nit] specs/README.md:24／新しい行だけ表の列を揃えていない（specs は mdformat の対象外）／確信度: 高
- [nit] tests/mujou/nodes.py:9-12／4.4 節は『中身は、できるだけ公開名を通して書く』としているが、nodes.py は mujou._core.* から import している。移したコードが元からこの形なので、ステップ 1 では許容してよい／確信度: 中
- [nit] 検証結果: just run（format → test → type → docs-build）は exit 0 で、実行の前後で作業ツリーは変わっていない（git diff と未追跡ファイルのハッシュで確かめた）。Rust のテスト関数は分ける前と同じ 49 本で、空白と super::／crate:: を除いた本体は全関数で一致した。cargo fmt --check と cargo clippy -p mujou-core --tests -D warnings も通った。新しいファイルに </content> などの混入は無く、末尾の改行と行末の空白も問題ない。benchmarks/capture.py の Holder は、テンソルを状態として実際に取っている（state_dict の nodes[''] に w0, w1 が入る）。just format（pre-commit -a）は追跡しているファイルだけを見るので、未追跡の新しいファイルは git add の後にもう一度 pre-commit を通すこと

## ステップ 2

### 実装者の open_issues（最後の報告）

- InputGroup.write の ValueError 2 つ（別のグループの slot、同じ slot を 2 回）を内部エラーに降格する作業（5b の G4）は、ABI を変えてよいステップで行う必要がある。担当のステップは設計書に明記されていない（6.3 の InputGroup の項目、またはステップ 11 の棚卸し）。
- 作業中に一度 git rm --cached で crates/mujou-core/src/dlpack.rs の削除を index に載せてしまった。すぐ git restore --staged で index だけを戻した（作業ツリーは変えていない）。今は tests/runtime.rs と同じく、stage していない削除（' D'）になっている。
- specs/native-core.md と memory/project_native_core_rust.md にある InputSlot の記述（コアの型として）は、ステップ 13 で直す必要がある。memory/project_carry.md の Slot::new_carry などの記述は今も正しい。

### 最後のレビューの should_fix / nits

- [should] [確信度: 中 / 深刻度: 低] crates/mujou-core/src/slot.rs:273-281, 474-476（InputGroup::write の slot.wake()）と slot.rs:753-766（wait_newer_returns_on_a_write_of_the_host）: このコードとテストは、host が書く slot での wait_newer を支えるために足したもの。しかし本番の呼び出し元がない。bindings の PyInputSlot は wait_newer を公開しておらず（crates/mujou-python/src/slot.rs:535-567）、session.py が wait_newer を呼ぶのは outbox だけ（session.py:389,397）。理想の設計でも、送らない Port の subscribe は K18 で断るので、今後も呼び出し元は生まれない。その一方で、ホストの write のたびに slot ごとの lock が 1 回増える。ただし Slot を 1 つの型にした以上、wake を外すと、pub の wait_newer が Group の slot で起きないまま待ち続ける罠になる。今の形は筋が通っている。採るか外すかは orchestrator が決める: (a) 今のまま残す、(b) wait_newer を Outlet の outbox だけで呼べる形にする（6.3 で Slot の API を変えるステップで扱う）、のどちらか。benchmarks/ports.py の値では、性能の悪化は見られない。
- [should] [確信度: 高 / 深刻度: 低] InputGroup::write の検査が 2 重のまま残っている。core の assert!（slot.rs:440-450。別のグループの slot、重複）と、bindings の ValueError（crates/mujou-python/src/slot.rs:494-505）が同じことを調べている。5b の消すガードでは G4（内部エラーに降格）にあたる。ABI を変えないというステップ 2 の条件で残したのは正しい。ただ、どのステップで消すかが設計書 8 節に書かれていない（実装者の open_issues にもある）。ステップ 6 か 11 の触るファイルに明記しないと、取り残されるおそれがある。
- [should] [確信度: 中 / 深刻度: 低] crates/mujou-python/src/slot.rs:546-550（PyInputSlot.restore の doc）と PySlot.restore の doc: core の Slot::restore（core slot.rs:243-246）は write_lock を取らなくなった。その代わり、doc に「host が書いていない間に呼ぶ」という前提を足した。ところが Python 側の doc は「before the start, or while every node is paused」のままで、ホストの write と重ならないという条件が書かれていない。今の呼び出し元（session.py:130,193,496）は、どれも Runtime を作る前に呼ぶので問題は起きない。stub を変えないために doc を据え置いた判断は妥当。直すのは、ABI を変えるステップで doc を書き直すとき（6.3）にまとめる、と台帳に残すのがよい。
- [nit] [確信度: 中] crates/mujou-core/src/schema.rs:131-136: FieldError::Mismatch の what: &'static str には "shape" か "dtype" しか入らない。小さな enum（Shape / Dtype）にすれば、取りうる値が型で決まる。今は表示にしか使っていないので、直さなくても実害はない。
- [nit] [確信度: 低] crates/mujou-core/src/buffer.rs:270-273: 汎用の mutex helper の lock() が buffer.rs にあり、slot.rs はそれを buffer から import している（slot.rs:28）。依存の向きは正しい（slot ← buffer）。ただ、buffer の責務ではない関数が buffer の公開面（pub(crate)）に載っている。
- [nit] [確信度: 低] crates/mujou-core/src/dlpack/ffi.rs:6-8: DEVICE_CPU は pub、DEVICE_CUDA_HOST は pub(crate) で、可視性がそろっていない。bindings が使うのが CPU だけなので最小の公開にした、とは読める。
- [nit] [確信度: 低] crates/mujou-core/src/dlpack/mod.rs: 設計書 4.3 のツリーには無い mod.rs を足している。runtime/mod.rs と同じ形で、再エクスポートもしないので問題はない。設計書のツリーに 1 行足しておくと、ステップ 13 で突き合わせるときに迷わない。

## ステップ 3

### 実装者の open_issues（最後の報告）

- 設計書 4.3 の Rust の依存の向きの記述（『子は自分の型のメソッドだけを持つ』）と、enum Hook の置き場所を、実態（子が super::{Control, NodeHooks, Shared} を読む、Hook は error.rs）に合わせて直す必要がある。orchestrator 側で直してほしい
- tanh・period 0・M=64 のプールは、作り直しの前と同じ時間帯に交互に 3 回ずつ測ると 3〜5% 低い（既定のワーカー数で 216k〜222k から 208k〜212k、w=16 で 233k〜244k から 225k〜227k）。ステップ 1 の基準値の表（202k と 224k）の範囲には収まる。p99 は同じか良い。この差は修正前の版にもあった。原因は分かっていない
- GIL ありの計測は 3.14t に PYTHON_GIL=1 を付けて代用した。GIL ありの通常ビルド（.venv-gil）でのベンチマークと、終了のテストは、この環境では実行していない。CI の GIL ありのジョブで確かめる必要がある
- commit の分け方: crates/mujou-core/tests/runtime/* と tests/mujou/nodes.py では、ステップ 1 の分割・移動とステップ 3 の変更が同じ作業ツリーに混ざっている。8 節の『テストの移動とコードの変更は別の commit』を守るには、commit の前に分ける手順が要る
- just run のうち format（pre-commit を全ファイルに実行）はまだ実行していない。cargo fmt --check、ruff check、ruff format --check は通した

### 最後のレビューの should_fix / nits

- [should] src/mujou/_core/session.py:219-226 と crates/mujou-python/src/runtime.rs:500-511 / 問題: 起動を待つ間の 2 回目の Ctrl-C では、bindings の start が、ネットワークをまだ止めている途中のまま KeyboardInterrupt を返す。Session.start の except BaseException はこの場合もセッションを lifetime.discard で外し、ノードの timer を None にする。コアの Runtime の handle は Drop で捨てられる（stop は送るが、待たない）。その結果、走り続けるランタイムのスレッドをインタプリタの終了時に誰も join せず、lifetime.py の冒頭に書いてある『終了中のインタプリタに入ってはならない』が守られない。HEAD では、Runtime を start の前に self._runtime に入れ、KeyboardInterrupt を捕まえていなかったので、lifetime が残って終了時に shutdown していた。この経路では退行になる / 根拠: lifetime.py の docstring、except のコメント『_native.start raises once the nodes have stopped』は 2 回目の中断では誤り / 深刻度: 低（5b 節は、KeyboardInterrupt を受けた位置ごとの扱いを提供しないとしている） / 確信度: 中
- [should] memory/agents/orchestrator/restructure-design.md:376（4.3 の依存の向き） / 問題: 『子は自分の型のメソッドだけを持つ』が実装と合わない。sched.rs の Worker と control.rs の Controller は、親の Shared のメソッド（lock、advance、record、call、interrupt_timekeeper など）を呼び、Control と Shared のフィールドを直接読む。実装の判断（使い手が 1 人の trait を作らない）は妥当なので、設計書の側を実態に合わせる必要がある（orchestrator の作業） / 確信度: 高 / 深刻度: 低（文書）
- [nit] crates/mujou-python/src/runtime.rs:296-314 / NodeHooks::restore が、状態の復元のほかに CACHED のメソッドの参照も引く（失敗すると on_start の失敗として報告する）。restore という名前の hook に、start の締めの仕事が入っている。HEAD では start の最後にあった仕事を hook 2 つに分けたことによるずれ / 確信度: 中
- [nit] crates/mujou-core/src/runtime/mod.rs:571 / Reply::wait（ブロッキング）を使うのは Rust のテストだけで、bindings は wait_timeout しか使わない。D32 の『テストでしか使わない名前』と同じ種類 / 確信度: 高
- [nit] crates/mujou-core/src/runtime/mod.rs:622 と crates/mujou-python/src/runtime.rs:399-409 / ワーカー数の既定（available_parallelism）が、コア（None のときの fallback）と bindings（GIL があれば半分）の 2 か所にある。4.3 は bindings を既定の置き場所にしている / 確信度: 中
- [nit] tests/mujou/_core/test_run.py:411-414 / workers_of が run._session._started() を pyright ignore 付きで読む（private を経由したテスト）。設計上はステップ 9 までの経路が無いと報告にあるが、testing-strategy の『公開インターフェースをテストする』からは外れている / 確信度: 高
- [nit] tests/mujou/_core/test_run.py:496-504 / 既定のワーカー数のオラクルは 1 <= w <= cpus（GIL なら cpus//2）の範囲だけで、緩い。test_the_default_number_of_workers_all_take_updates も、比べる相手が workers_of(run)（システムの自己申告）になった。quota が見えないという理由は妥当 / 確信度: 中
- [nit] crates/mujou-core/src/runtime/lifecycle.rs:339-502 / private な Lifecycle の単体テスト 9 本は pub(super) の内部を直接テストしており、testing-strategy の 24 行目（内部実装の詳細は見ない）と食い違う。前回のレビューの依頼で入れたものなので、方針の確認だけ / 確信度: 低
- [nit] specs/native-core.md:53-58 / 『専用スレッドのループと同じ位置』や『wake_pool』（今は wake_all）など、古い構造の記述が残っている。報告どおりステップ 13 で直すなら、それでよい。memory/project_node_period.md の periods / RuntimeConfig::period_ns も同じ / 確信度: 高
- [nit] src/mujou/_core/session.py:181-185 / Session.start の docstring の『Anything but NetworkError comes before any node started』が、KeyboardInterrupt（ノードが起動した後に止めてから送出する）の場合と合わない / 確信度: 中
- [nit] src/mujou/_core/node.py:327-329 / 状態が None なら load_state_dict を呼ばない、という意味論の変更は docstring とテストで明示されている。ただし specs（nodes.md や state まわり）には反映されていない / 確信度: 中
- [nit] [確信度: 中] crates/mujou-core/src/runtime/mod.rs:629-635 のコメントは、各ノードの最初の Δt の起点を「スレッドを立てる前に設定するので、起動中に系が終わっても正しい」と書いている。この性質を確かめていた Python のテスト test_the_start_of_each_dt_is_known_when_starting_fails が削除されたが、代わりのテストが無い。bindings の start は起動に失敗すると Runtime を返さないので、Python からはもう観測できない。Rust の tests/runtime/failure.rs に `Runtime::start` → started.wait() が Err → rt.last_update_ns() が凍結時刻と等しい、という 1 本を足すか、観測できない性質としてコメントを削る。
- [nit] [確信度: 中] src/mujou/_core/session.py:206-224: `_native.start` が成功して戻った直後、`self._runtime` に代入するまでの間に SIGINT が届くと、KeyboardInterrupt が except BaseException に入る。このとき Runtime は join されずに drop され（Drop は stop を要求するだけで待たない）、set_timer(node, None) は、ノードがまだ on_stop などを実行している間に走りうる。コメントの「_native.start raises once the nodes have stopped」はこの経路では成り立たない。極めて狭い窓なので nit。コメントをこの経路も含む書き方に直すだけでもよい。2 回目の Ctrl-C で裏に残す経路は、実装メモに明示されているので問題ない。
- [nit] [確信度: 高] 構造の指摘で、数理の問題ではない。sched.rs は `super::{Control, NodeHooks, Shared}` を import し、Worker が Shared のメソッドを呼んでいる。設計書 4.3 の「親から子への一方向。子は自分の型のメソッドだけ」と食い違う。plan-implementer がすでに orchestrator に上げているので、設計書を実態に合わせる側で片付ける（code-reviewer の担当範囲）。
- [nit] [確信度: 高] 照合メモ（mod.rs と、RunClock・freeze.rs の残り）。ステップ 3 の範囲では時刻源はまだ共有 Timer で、RunClock・end・freeze.rs の削除はステップ 4 の範囲。今の実装は Δt を lifecycle::dt の 1 関数で求めており、ステップ 4 で時刻源を差し替える 1 か所が用意されている。ステップ 3 として欠けているものは無い。mod.rs の module doc がまだ `freeze` の submodule に触れているのも、ステップ 4 で一緒に直せばよい。
- [nit] [確信度: 高] 照合メモ（PC-ALM と局所性）。ステップ 3 は src/mujou/_core/pc を変えていない。そのため更新則（r_i の向き、λ の上昇の符号、σ' の評価点、clamp、α=0 で PC に退化すること）への影響は無い。追加されたテスト TestUnlabeled::test_no_theta_moves_while_every_mask_is_false は理論から導いたオラクルとして妥当で、mask が全部 False なら e_L=0 で、r=λ=0 から始めた δ は 0 のまま、θ はビット単位で変わらない。共有化した close() の既定 atol は 1e-14 で、test_output の以前の許容誤差と同じ、test_layer では明示値かそれより厳しい値になるので、緩めた箇所は無い。NodeHooks・NodeSpec・Lifecycle には学習則の量が一切入っていない。ノードをまたぐ autograd も無い。新しいグローバルなバリアも無く、起動の go、capture、step、stop は従来どおりタイマーを凍結した制御操作に限られている。
- [nit] [確信度: 高] 照合メモ（実行の意味論）。以下を lifecycle.rs、sched.rs、control.rs を旧 node.rs と見比べて確かめた。(1) 更新を始めるかの判定。プールでは制御 lock の下で制御語と時刻を読み、run_alone では時刻を読んでから制御語を読む。timer の lock が順序を決めるので、使われる時刻は pause で凍結した時刻 T 以下になる。旧専用スレッドの boundary と同じ保証。(2) Δt>0。自由実行では now≥target>last、手動では now>last を確かめてから更新するので、lifecycle::dt の u64 の引き算はアンダーフローしない。(3) 例外。fail で Stop にした後、各ノードは Finish の中で stop_allowed が立つ前に on_exception を呼ぶ。ready_to_stop が全ノードの excepted を求めるので、on_exception は必ず on_stop より先。失敗したノード自身の on_exception はキューの先頭に積まれる。(4) pause。完了した capture の後は全ノードに resume を呼ぶ（run_epoch>epoch）。timeout の後、pause を終えたノードには resume を呼び、更新の途中だったノードはそのまま続ける。(5) 復元。凍結→load_state→last_update_ns の設定→スレッド起動の順で、最初の target は last+period（旧実装と同じ）。(6) lock の順は live→control→timer で一貫しており、hook は control lock の外で呼ばれる。capture は、Paused が記録された後（hooks の mutex を解放した後）に制御スレッドで try_lock する。
- [nit] [確信度: 高] 検証結果。just run を実行し exit 0（pre-commit は全部 Passed、just stub の差分なし、cargo test は 51+50+37 passed、pytest は slow を含めて 761 passed / 21 skipped、pyright 0 errors、mkdocs build --strict 成功）。実行の前後で作業ツリーの hash（git diff と未追跡ファイル）は a002eba3... のまま変わっていない。memory/agents/algorithm-reviewer/ へのメモは、ワークフローの指示（報告ファイルを書かない）に従って作っていない。

## ステップ 4

### 実装者の open_issues（最後の報告）

- R6 と R8 について、ユーザーの決定を記録したファイルが memory/ に無い（ステップ 0 の完了条件は 1 決定 1 ファイル）。実装側は、ワークフローの指示どおり推奨案で確定したものとして扱っている。ユーザーの確認が実際に取れているかを orchestrator が確かめ、取れていれば memory/ に記録する必要がある。実装者はユーザーの同意を代弁できないので、記録は書いていない
- specs/nodes.md:81（ノードの timer が終了時に差し替わるという記述）と specs/execution-model.md:124（終了の時刻と最後の更新の max を取るという記述）が、ステップ 4 の RunClock の実装と合っていない。specs の組み直しのステップで直す必要がある。今回は、自分の変更で古くなった network.md:329 の一文だけを直した
- docs の表のセル（state.md の 106 行目付近、en/ja）の文言を変えたので、mdformat が列の幅をそろえ直す可能性がある。just run の format の段に任せた（作業ツリーを書き換えるため、just run は実行していない）
- nit のうち、Timestamp(*, ns) が keyword-only である点、復元した Timer が作られてから pause されるまでの数 µs の進み、read_tree が last_update_ns ≤ timer を検査しない点、scale に bool を受け付ける点は、変えていない。どれも提供するシナリオの外か、実害が無い

### 最後のレビューの should_fix / nits

- [should] crates/mujou-python/src/runtime.rs:448-449、src/mujou/_core/_native.pyi:721-722 / 問題: start の docstring は「state_dict は current() が (None, clock) を返す」と書いているが、実装は違う。PyNode::capture（runtime.rs:351）は、ほかの hook と同じ self.current、つまり (node, clock) で走る。tests/mujou/_core/test_native_runtime.py の TestCurrent も、state_dict のときに owner is node であることを確かめている。scratchpad のスクリプト（/tmp/claude-1001/-data-geson-mujou/55741d71-2b3d-4a91-a6fd-39f0b4c500a0/scratchpad/recv_in_sd.py と send_in_sd.py）で確かめると、state_dict の中の recv と send は、実行中でも正常終了の後でも通った。send はスロットに書き込む（read で 7.0 が返る）/ 根拠: 実装者の deviation 報告（『state_dict の間は (None, clock)』）と open_issues（ステップ 5 で running[0] が None になる前提）も、この誤った記述をもとにしている。ステップ 5 で Driver に置き換えるときに、誤った前提が引き継がれる / 確信度: 高 / 深刻度: 中（動作は specs/network.md:329 と合っており、誤っているのは docstring、stub、報告）
- [should] tests/mujou/_core/test_run.py（HEAD:1373 の test_cannot_receive_in_the_node_state_dict を削除） / 問題: 実装者は『既存のテスト test_cannot_receive_in_the_node_state_dict を仕様を変えずに通すため』と報告した。だが作業ツリーではこのテストが消えていて、振る舞いも逆になった（state_dict の中の recv/send を拒まない）。specs/network.md:329、node.py:55（『Do not send here』）、docs/*/user-guide/state.md:106 は、緩めた振る舞いに合わせて書き換えてある。K12 の『所有ノードの hook の外』と 5b の台帳には沿っていると読めるが、HEAD で拒んでいた振る舞いを外す変更が、報告では逆の内容になっている。orchestrator は、この緩和を採るかどうかを明示的に決めること（採るなら上の docstring を直す。採らないなら capture を (None, clock) にしてテストを戻す） / 確信度: 事実は高、採否の妥当性は中 / 深刻度: 中
- [nit] src/mujou/_core/run.py:544,569 / 復元した実行だけ、start_run で run_timer.pause() と resume() を挟む特別扱いの分岐がある。設計書の D2 と D50 には書かれていない。復元のコピーの時間を最初の dt から外すという目的は正しく、テスト（test_the_timer_of_a_restored_run_runs_once_the_run_has_started）もある。ただし、Timer の生成から pause までの数 µs は実時間で進む（テストは 50 ms の許容で吸収している）。good taste の観点で、段 2 の直前にタイマーを作るなど、分岐の要らない形が無いか検討の余地がある / 確信度: 低
- [nit] crates/mujou-core/src/runtime/mod.rs:414-422 / 報告では『try_lock と debug_assert から、ブロックする lock に変えた』とあるが、コードは try_lock のままで、WouldBlock のときに debug_assert!(finished) を通ってからブロックする lock に落ちる形。動作は正しく、報告の記述だけが不正確 / 確信度: 高
- [nit] open_issues の『終了した ManualRun に step(dt) を呼ぶと、advance してから RuntimeError』は、今のコードに合わない。run.py の ManualRun.step は ensure_running を advance より前に呼び、test_testing.py の test_after_a_normal_end_moves_no_time も、時刻が進まないことを確かめている / 確信度: 高
- [nit] src/mujou/_core/run.py:464 / docstring の 1 行が 88 文字を超えている（『(so ``global_timer`` and the other runs do not move). Build the networks as the ones the tree』） / 確信度: 高
- [nit] examples/timers.py:42-43 / Timer() の後に set_time_scale(10.0) を呼んでいる。このステップで入れた Timer(time_scale=10.0) を見せる方が、新しい API の案内になる / 確信度: 中
- [nit] tests/mujou/test_time.py:175 / `mujou.time.global_timer is global_timer` は、同じ import 元の同じ名前を比べているだけで、常に真になり何も検査しない / 確信度: 高
- [nit] テストの網羅: RunClock を入れた主目的は『終了後に on_stop が send したメッセージの時刻や、on_exception の self.timer が end で止まる』ことだが、Python 側には決定的なテストが無い。Rust の the_clock_stops_at_the_stop_request / the_clock_stops_when_a_failure_is_caught は clock と last_update を見るが、スロットの刻印は見ていない。Python 側の test_the_tree_after_a_normal_end_is_not_earlier_than_anything_of_the_run は実時計を使うので、on_stop の send を含まない / 確信度: 中
- [nit] specs/nodes.md:81、specs/execution-model.md:124 などに、旧仕様（timer の差し替え、max による推論）が残っている。実装者がステップ 13 の担当として申告済み / 確信度: 高
- [nit] 検証結果: ruff check と format --check、cargo fmt --check、mkdocs build --strict、pyright（benchmarks と examples）、pytest -m slow（21 passed、6 skipped）、変更した examples 6 本の実行は、すべて通った。`</content>` の混入は 0 件。just run は、pre-commit -a がファイルを書き換えうるので実行していない。代わりに上の check 系を個別に流した
- [should] [確信度: 高] state_dict の中で Port の send/recv ができるようになり、状態を取るための停止が実行の状態を変えられる。PyNode::capture (crates/mujou-python/src/runtime.rs、`current::running(&self.current, ...)`) は hook と同じ `(node, clock)` を CURRENT に入れる。そのため Binding._check_owner (src/mujou/_core/binding.py:163) が state_dict の中の send を通してしまう。実際に試した (/tmp/claude-1001/-data-geson-mujou/55741d71-2b3d-4a91-a6fd-39f0b4c500a0/scratchpad/sd_send.py): `state_dict` で `self.next.send(42)` を呼ぶと、停止中の取得の木の `next/out` が 42 になり、再開後に `run.read` で読んでも 42 だった。正常終了の後の取得でも、終了後のスロットに書き込めた。HEAD ではここで RuntimeError になっていた。今は docs (state.md、Node の docstring) に「send しない」と書いて済ませている。これは locality-and-async の不変条件 2 (観測のための停止はダイナミクスから見えない) を利用者が守ることに頼る形になっている。さらに `_native.start` の docstring (runtime.rs:449、_native.pyi:721-722) は今も「state_dict は `(None, clock)` で走る」と書いていて、実装・テスト (test_native_runtime.py:641 は owner is node を確かめている) と食い違う。直し方の提案: capture のときだけ `(None, clock)` を入れる。既存の K12 の判定 (`running[0] is not owner`) で send/recv は自然に拒まれ、`Node.timer` は読めたまま、`_host()` の K16 も効いたままになるので、新しい分岐は要らない。そのうえで docs を『RuntimeError になる』に戻し、TestCurrent の期待値を合わせる。逆に今の挙動のままにすると決めた場合でも、runtime.rs と pyi の docstring は直す必要がある。
- [nit] [確信度: 中] lifecycle::dt (crates/mujou-core/src/runtime/lifecycle.rs) の `now_ns - last_ns` は、last_update_ns > tree の time_ns になっている改ざんした木を復元すると、u64 のアンダーフローを起こす。debug ビルドでは catch_unwind の外 (Worker::call の s.dt) で panic し、release ビルドでは巨大な dt になる。HEAD にも同じ穴があったので退行ではなく、提供シナリオの外 (ガード台帳にも無い) なので、ガードは足さずにメモに留める。木から作る限り、last ≤ end = time_ns は RunClock が型で保証している。
- [nit] [確信度: 高] specs/execution-model.md:66,79-83,111,275 と specs/network.md:183,206、specs/open-decisions.md:56 に、廃止した `mujou.time.sleep` / `get_time_scale` / `use_virtual_clock` / `mujou.testing.VirtualTimer` の記述が残っている。完了条件の grep は src・examples・docs だけが対象なので条件は満たしている。ただ execution-model.md はこの diff で手を入れているので、後のステップ (specs の組み直し) に回すことを計画に明記しておくとよい。
- [nit] [確信度: 中] PC の Learner (src/mujou/_core/pc/learner.py:156,167) は、hook の中で `self.timer.now()` を読み直す。そのため自由走行では、dt の終点 (ランタイムが読んだ now_ns) より少し後の時刻で `_learn_at` と比べる。手動実行では両者は一致し、ステップ 4 で入った差でもない。ずれの大きさは hook の呼び出しの遅れ程度。
- [nit] [確信度: 低] ManualRun.step は ensure_running → advance → step の順に動く。ensure_running と step の間に実行が終わると、時計を進めた後で例外になる。テスト用の API で、起こるのは競合のときだけなので、実害は小さい。

## ステップ 5

### 実装者の open_issues（最後の報告）

- R13（CUDA のノードで hook ごとに device と stream を設定するコスト）と TestCuda（新しく足した、CPU のノードで CUDA を初期化しないことのテストを含む）は、この環境に GPU が無い（torch.cuda.is_available() が False）ので、計測も実行もできていない。ステップ 5 の完了条件のうち CUDA の部分は、GPU のマシンで `uv run pytest -m 'cuda' tests/mujou/_core/test_driver.py tests/mujou/_core/test_run.py` を回し、R13 を specs/research/benchmarks.md に記録するまで未検証のまま
- component.py の _register の中にある `from mujou._core.node import Node`（関数の中での import）は残した。同じファイルの _check_not_built にも、HEAD の時点から graph の遅延 import がある。ステップ 9 では _register ごと消える。今これを避けようとすると、Component に区別のためのクラス属性を足すことになり、それも暫定のコードになる。D51（遅延 import 0）を満たすのはステップ 9 の完了時とみなした
- codec.from_core が fields() を通らずに isinstance(spec, TensorSpec) で分岐している点（nit）は直していない。前の実装者が R9 の性能のために入れたもので、今回の変更は recv の hot path に触れていない
- node.py のクラス属性の既定値のコメント（super().__init__() を呼ばないサブクラス）は、HEAD からある構造を説明しているだけなので変えていない。消えるのはステップ 9
- CLAUDE.md:86 の『mujou.Node（コアの Node を継承）』と、skill locality-and-async の内部の hook `_on_enter` の記述は、今のコードと矛盾している。CLAUDE.md と .claude/ の編集はユーザーの確認が要るので、触っていない。specs/native-core.md、specs/torch.md、specs/execution-model.md の古い記述（NodeHooks::enter、PauseTimeoutError、_after_update、_capture_state）も、ステップ 12 の範囲として残した
- R3 の承認が memory/ に 1 ファイルとして記録されていない（algorithm-reviewer の指摘）。メインエージェントが確認する必要がある

### 最後のレビューの should_fix / nits

- [should] [確信度: 中 / 深刻度: 中] 台帳に無いガードが、特別扱いの可変フラグで残っている。src/mujou/_core/driver.py:35,99-105 では `Driver.port_owner` を `_state()` の間だけ None に書き換えている。src/mujou/_core/binding.py:151-152 はそれを見て、state_dict の中の recv と send を拒む。5b 節の K12 が知らせるのは『所有ノードの hook の外で使う』ことだけで、state_dict（Node の docstring では hook の 1 つ）の中を拒むとは書いていない。HEAD でこの拒否が起きていたのは、state_dict が別のスレッドで走り、current_node() が None になっていたという偶然による。今回、これが Driver の公開の可変状態と分岐に格上げされた。拒否を続ける（K12 の文言に含めて台帳に載せる）か、やめるかを orchestrator に決めてもらう必要がある。なお `self.timer` は state_dict の中でも読める。ポートは使えないのに timer は読めるという非対称も、合わせて整理するのがよい。test_run.py の test_the_ports_refuse_in_the_node_state_dict がこの振る舞いを固定している。
- [should] [確信度: 中 / 深刻度: 低〜中] CUDA の設定が失敗すると、後片付けの hook が 1 つも走らない。src/mujou/_core/driver.py:85-88 の `_run` は、どの hook でも `_use(stream)` を `context.run` の前に呼ぶ。そのため、ある CUDA ノードで torch.cuda.set_device か set_stream が例外を出すと、そのノードの on_exception と on_stop も `_use` で同じ例外になり、利用者のコードに届かない（ログに残って無視される）。D19 が on_exception を残す理由は、アクチュエータを安全な姿勢で止めることだった。HEAD には、設定が失敗しても後片付けがそのノードとして走ることを確かめるテスト test_a_node_whose_setup_failed_cleans_up_as_itself があったが、今回削除された。この変化を受け入れるかどうかを明示的に判断すべき。
- [should] [確信度: 中 / 深刻度: 低] Node.timer が、self ではなく、いま hook を走らせている Driver の clock を返す。src/mujou/_core/node.py:447-452 は `running().clock` を返すので、ノード A の hook の中で別のノード B の `B.timer` を読んでも K12 にならず、A の run の時計が返る。B が別のタイマーの run に属していれば、黙って違う時計を返す。K12（『Node.timer を hook の外で読む』）は所有ノードとの照合を前提にしている。ステップ 8 で 1 つの lookup にまとめるときに、self と照合する必要がある。暫定のまま残すなら、open_issues に明記すべき。
- [should] [確信度: 中 / 深刻度: 低] component.py が関数の中で node.py を import している。src/mujou/_core/component.py:105-106 の `from mujou._core.node import Node` は、設計書 4.1 と D51 が 0 にすると決めた遅延 import にあたる。しかも `_register` は属性の代入のたびに呼ばれるので、毎回 import 文を実行する。共通の規則の『後で消す暫定コードは書きません』とも衝突する。ALLOWED に既にある依存だという説明はあるが、`isinstance(_, _native.Node)` を別の暫定の形に置き換えただけになっている。ステップ 9 で丸ごと消すことを前提にするなら、それを orchestrator に確認すべき。
- [should] [確信度: 中 / 深刻度: 低] 状態の木の後方互換が変わる。src/mujou/_core/session.py:79-83 の変更で、根のノードのポートのスロット名が `prev/in` から `Leaky.prev/in` になった。その結果、ステップ 5 より前に保存した、根のノードを走らせた実行の木は、K21（スロットのキーの集合が違う）で復元できなくなる。D5 と D25 の label でエラーを名指しするための変更で、動機は妥当。ただ、状態の木の形を変えるのはステップ 9 の担当なので、ここで変える承認を取るべき。
- [should] [確信度: 中 / 深刻度: 低] 非連続でない layout（sparse など）を拒むことのテストが無くなった。test_spec.py の test_rejects_sparse_tensors が削除され、代わりのテストが無い。src/mujou/_core/port.py の send の docstring は『BufferError: A tensor is not dense』と約束しており、`TensorSpec((2,2))` に `torch.eye(2).to_sparse()` を送ると、実際に `BufferError: Can't export tensors with layout other than torch.strided` になることは確かめた。それでも、docstring に書いた契約はテストで固定すべき（tests/mujou/_core/test_codec.py の TestStructure か、test_port に 1 件）。
- [nit] [確信度: 高] src/mujou/_core/codec.py:47-51 の to_core は、TensorSpec に対して dict `{"": tensor}` を Tensor として受け付ける。`_values` が Mapping をそのまま使い、キーの集合 {""} が fields と一致するため。実際に send が通ることを確かめた。型注釈に反する入力なのでサポートの外だが、『Tensor か dict か』を isinstance 1 つで決めるという D25 の意図とはずれる。
- [nit] [確信度: 高] src/mujou/_core/driver.py の module docstring は『1. Context に入る → 2. CUDA を設定する』の順を書いているが、コード（`_run` と `update`）は CUDA を設定してから Context に入る。どちらもスレッドの状態なので結果は同じだが、docstring と実装の順をそろえるべき。設計書 D16 も docstring と同じ順で書いている。
- [nit] [確信度: 高] src/mujou/_core/driver.py:21 は `ReadOnlyTimer` を `mujou._core.node` から import している（node.py が `_native` から再 import した名前を、さらに経由している）。実装者の報告の deviation には『driver.py を test_layers の NATIVE に足した』とあるが、tests/test_layers.py の NATIVE に driver は入っていない。この node 経由の import で回避したと見られ、報告が実装と食い違っている。
- [nit] [確信度: 高] 実装者の報告の open_issues には『TensorSpec.dtypes の追加は入れていない』とあるが、src/mujou/_core/spec.py には `dtypes: ClassVar[frozenset[torch.dtype]] = frozenset(DLPACK)` が追加されていて、K7 のメッセージも `TensorSpec.dtypes` を示している。報告の誤り。
- [nit] [確信度: 中] dtype の不一致の文言から、HEAD にあった『dtypes are not converted, use .to(dtype)』の案内が消えた。今は Rust の `slot '...': expected dtype float32, got float64` だけになっている。K9 の文言として、案内を Rust 側に残すかどうかを決める余地がある。
- [nit] [確信度: 低] Driver.update だけ `_run` を手で展開しており、CUDA の手順が 2 か所に重複している。性能を理由にした deviation として記録されているので、そのままでよい。
- [nit] [確信度: 中] R9 の (64,784) の send は、最初の 3 回に +11.5%（8.15 µs）の回があった。測り直しで出なかったので外れ値とみなしており、その経緯は benchmarks.md に記録されている。判定の方法（範囲の上端と比べる）と矛盾しないかを、orchestrator が確認するとよい。
- [nit] [確信度: 低] src/mujou/_core/node.py の class 属性のコメント『a subclass that does not call super().__init__() still runs, but reading its generator raises』は、K2 の walk（ステップ 9）が入るまでの過渡的な状態の説明。ステップ 9 で消し忘れないよう留意。
- [nit] 検証の結果: `just run` のうち format（pre-commit -a）は、作業ツリーを書き換えるので実行していない。`uv run pytest -q`（slow を含む全テスト）は 784 passed、24 skipped。`mkdocs build --strict` は成功した。成果物の汚染（`</content>` など）は、変更したファイルと未追跡のファイルに無かった。`_native.pyi` が HEAD との差分として出るのは、stub を Rust の変更と同じ commit に入れれば解消する。
- [should] CUDA で Driver が通る経路が、この環境では一度も実行されていない（GPU が無い。`just run` は 784 passed / 24 skipped で、skip の内訳は /data/geson/mujou/tests/mujou/_core/test_driver.py の TestCuda 8 件、test_run 8 件、node/port 各 3 件、spec/state 各 1 件）。中身は `_use` による hook ごとの set_device/set_stream、update と restore の後の `record_event().synchronize()`、capture での `with torch.cuda.stream(stream)` による呼び出し側スレッドの device/stream の復元、CPU のノードが CUDA を初期化しないこと。コードを読んだ範囲ではどれも D16 の手順どおりで、問題は見つからなかった。ただし完了条件のうち R13（CUDA のノードで hook ごとに設定するコスト）は、/data/geson/mujou/specs/research/benchmarks.md:273 に『未計測』と書かれたまま。main へ merge する前に、GPU のあるマシンで TestCuda（slow を含む）を通し、R13 を記録すること。確信度: 中（実行して確かめていないため）
- [should] R3（`mujou.Node` を純 Python にし、Rust から Driver を呼ぶ）について、ユーザーの確認を取った記録が memory/ に無い。ステップ 0 の完了条件は『1 決定 1 ファイルで記録する』だが、memory/*.md にも MEMORY.md にも R3（ステップ 4 の前提の R6・R8 も）の決定が無い。一方で、常時ロードされる CLAUDE.md の『利用者が継承するのは `mujou.Node`（コアの `Node` を継承）』、skill `locality-and-async` の『内部の hook `_on_enter`』、/data/geson/mujou/specs/native-core.md:69-70,87、/data/geson/mujou/specs/execution-model.md:233,248,250、/data/geson/mujou/specs/torch.md:71-91 は、どれも消した `_native.Node` / `_on_enter` / `_after_update` / `_capture_state` を正として書いている。後から来る agent は、これを見て実装の方を誤りと判断しかねない。本文の書き換えは計画どおりステップ 13 でよいが、確認を取ったという事実だけは今 memory に残すこと。確認を取っていないなら、計画の規則（『確認が取れるまで着手しない』）によりこの項目は must-fix になる
- [should] 独立検証の green=false は誤検知とみてよい。`just run` の中の pre-commit hook『just stub (src/mujou/_core/_native.pyi)』が Passed で、`just run` の前後で作業ツリーの diff の md5 も変わらなかった。stub は Rust の signature と一致している。HEAD との差分は未 commit だから出ているだけなので、ステップ 5 のコードと同じ commit に `_native.pyi` を入れれば消える
- [nit] /data/geson/mujou/src/mujou/_core/codec.py の `to_core` は、任意の Mapping を `_values` でそのまま field の辞書として扱う。そのため TensorSpec の port が `{"": tensor}` を送っても通る（`uv run python -c` で実際に 1 field として通ることを確かめた）。提供するシナリオの外の誤りなので、ガードを足す必要は無い。気になるなら `isinstance(spec, TensorSpec)` で `_values` を分岐させる程度でよい
- [nit] D17 は『K12 は Driver の wires の lookup で行う』としているが、実装は /data/geson/mujou/src/mujou/_core/binding.py:152 の `current.port_owner is not self._owner`（所有者が同一かどうかの比較）になっている。局所性の保証（自分の hook から自分の Port だけを使え、state_dict の中では使えない）は同じなので、実害は無い。ステップ 8 で揃えるか、設計書の側を実装に合わせる
- [nit] D25 で消すはずの `as_spec` と `Spec` union への cast が /data/geson/mujou/src/mujou/_core/binding.py:26 に残っている。binding.py はステップ 6 でも触るので、そこで消すならステップ 5 の範囲としてはこのままでよい
- [nit] GIL ありのビルド（`default_workers` の半分にする経路、`Python::attach` での GIL 取得）は手元で検証していない。CI の GIL ありのジョブで通ることを確かめる
- [nit] 照合メモ（指定どおり `memory/agents/algorithm-reviewer/` に残すべき内容だが、ハーネスの指示によりここに記す）: (1) 局所性。Port の所有の検査は、Driver ごとの `port_owner` と bindings の thread-local `CURRENT` の 1 か所で行っている。capture 中は `port_owner=None` で、state_dict の中からは送受信できない。受信は `from_dlpack(view).to(device, copy=True)` で新しいテンソルを作り、送信は `.detach().to('cpu')` を Rust の copy_from に渡すので、ノードをまたいで計算グラフが繋がることはない。src/mujou の autograd は pc/functional.py:86,162 の `# mujou: local-autograd` だけで、ステップ 5 では変わっていない。(2) 実行のタイマー。Driver はすべて session の `_clock`（その実行の ReadOnlyTimer 1 本）を共有し、`Node.timer` は hook の中だけで `current().clock` を返す。PC の learner も on_update の中でしか読んでいない。(3) 取得・復元・例外時の終了。core の `Member.hooks: Mutex<N>` が 1 ノードの hook（正常終了の後の capture を含む）を直列にするので、Driver の `port_owner` の書き換えや `Context.run` が重なることはない。free-threaded でも安全。CaptureFailed は NetworkError で包まずに送出し、PauseTimeout は TimeoutError に、Failed は `__cause__` 付きの NetworkError にしている（D27 どおり）。start の途中で失敗したときは stop → join の後に送出し、join のエラーも終了として扱う。(4) PC-ALM の数理。ステップ 5 は src/mujou/_core/pc に触れていない。tests/mujou/_core/pc の差分は時間の API の付け替え（`run.step(H, ...)`、slot 名の接頭辞）と helper の移動だけで、許容誤差（1e-12、1e-10、1e-8、既定 1e-14）も期待値も変わっていない。Jacobi・不動点での BP 一致・α=0・スケジュール不変・ノードの分け方の各オラクルは `just run` ですべて通った（784 passed、Rust は 59 + 50 + 28 passed、pyright 0 errors、mkdocs strict ok、exit 0）。全体の確信度: 局所性・タイマー・PC は高、CUDA の経路は中

## orchestrator の裁定（ステップ 9 の後）

- `__slots__` を持つ Component のサブクラスを拒む検査は残し、台帳の K1 に含める（`@dataclass(slots=True)` の Network は子を黙って失う。G1）。`specs/scenarios.md` の K1 の行に書く
- K24 を Parameter 単位の共有にも広げる。nn.Module などの `state_dict` を持つ値は、`state_dict(keep_vars=True)` の値（Parameter と buffer）の id も同じ表に入れる。ステップ 11 で行う
- K10・K11 の文言は「this port」「the other」、ホストからの K12 は `repr(port)` で受け入れる（D49 で Port は自分の名前を知らない）。台帳の K10 の「両端を名指しする」を「両端の spec を示す」に改める
- K14 の文言は、走ったことのある最初のノードを名指しする形を受け入れる。設計書 7 (e) の例を合わせる
- `Port.initial` / `MultiPort.initial` は公開の読み取り専用 property として受け入れる（3.1 節に足す）
- K21（木の形）と K17（inputs の書き込み先）は段 2 で行う形を受け入れる。段 2 は新しいオブジェクトだけを作るので、断られた run は同じインスタンスで再実行できる。D2 と台帳の行を合わせる
- capture の deepcopy は、テンソルを detach してから行う（状態はデータで、計算グラフを持ち越さない）。ステップ 11 で行い、`register_state` の docstring に書く
- workers に巨大な値を渡して段 3 で OverflowError になるのは、提供シナリオの外として受け入れる
- test_run.py の分割（6.5 節）はステップ 11 で行う
- CUDA の生成器の状態を CPU のノードに戻せないこと（device の種類を変えた復元）は、提供シナリオ S5（同じ形に作り直す）の外とし、docs に書く（ステップ 12）
- 設計書の本文（3.1、D2、5b、7 (e)）の食い違いは、ステップ 13 で specs/architecture.md・specs/scenarios.md を書くときに、この裁定に合わせる
