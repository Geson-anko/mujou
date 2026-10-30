# device-tensor-passthrough レビュー

対象: `37b42e8..HEAD`（b936184, 8f99e32, 8ed6c4c, a44fab3）。計画書 `memory/agents/implementation-planner/device-tensor-passthrough.md`、実装ログ `memory/agents/plan-implementer/device-tensor-passthrough.md`。

## verdict: request-changes

must-fix は 1 件（hook の中で current device を変えたまま send すると、以前は通っていた送信が `BufferError` になる回帰）。ほかは文書・ユーザー判断の待ち・nit。S2 の同期、`record_stream`、副 stream、`peek` → `cpu` の流れ、Rust の `Value::Held` / `FieldOut`、PyO3 の reference pool の扱いは計画どおりで、正しいことを確かめた（下の検証結果）。

## must-fix

1. **`src/mujou/codec.py:130-136`（`_value` の保持の条件）と `crates/mujou-python/src/slot.rs:77-92`（`borrow_values` の CUDA の分岐）**
   - 問題: 保持するかどうかを `value.device == held` だけで決めている。torch の `Tensor.__dlpack__` は「呼んだスレッドの current device がテンソルの device であること」を要求する（torch 2.14.1 の `__dlpack__` は `self.device.index != torch.cuda.current_device()` で `BufferError`）。hook の中で利用者が current device を変えていると（例: 2 枚目の GPU も使うノードの `with torch.cuda.device(1): self.out.send(x)`、`x` はノードの device `cuda:0` の上）、clone した値を bindings が `__dlpack__` に渡した時点で `BufferError("Can't export tensors on a different CUDA device index. Expected: 0. Current device: 1.")` になり、送信されない。変更前は `.to("cpu")` の経路で通っていた。`Port.send` の Raises は `BufferError` を「密でないテンソル」としか書いていないので、利用者には原因が分からない
   - 根拠: `/tmp/claude-1001/-data-geson-mujou/d2146d35-f3d8-44cf-90d5-5029742d85ef/scratchpad/rev/other_device.py` で再現した（`cuda:0` の 2 ノードのうち送り手が `with torch.cuda.device(1)` の中で send する → 上の `BufferError`、読み手はゼロを受け取る）。CLAUDE.md の「ノードの内側での GPU の利用は自由」、計画書の想定リスク「保持するのは hook の中のノード自身の device の値だけで、Driver が最後に自分の device を current にするので満たされる」は、hook の中で device が変わる場合を見落としている
   - 直す方向（参考）: `__dlpack__` を呼ぶ間だけ書き手の device を current にする（codec か bindings）、または current device が違うときはホストの経路に回す。どちらにしても回帰テストを足す（`test_port.py` の `TestCuda`、`cuda(2)`）
   - 確信度: 高（再現した）。深刻度: 中（複数の GPU を使うノードに限られるが、黙って送信が欠ける）

## should-fix

1. **`CLAUDE.md:85-86` と `.claude/skills/locality-and-async/SKILL.md:47`**
   - 問題: 「GPU はコアでサポートせず、ノードの内側でだけ使う」「ゼロコピー共有・GPU 共有はしない」が残り、この変更（コアが CUDA の値を不透明な値として device に置く）と食い違う。常時ロードされる CLAUDE.md が古い方針を言い続ける
   - 根拠: 計画書「更新する文書」に入っている。実装ログの逸脱 7 のとおり、サブエージェントには変更を承認できないので、orchestrator がユーザーに確認して直す必要がある
   - 確信度: 高

2. **`CHANGELOG.md`（このブランチに無い）**
   - 問題: 利用者に見える変更（CUDA のノード間のメッセージがホストを経由しない、CPU の読み手とホストの読み出しに D2H が移る、`send` が自分の stream を待つ）の `## [Unreleased]` の項目が無い
   - 根拠: CLAUDE.md「利用者に見える変更は CHANGELOG.md の ## [Unreleased] に書く」。計画書は `chore/2026-10-23/release-v0.1.0` の merge 後に足すとしている。PR の前に片付ける項目として残す
   - 確信度: 高

3. **CUDA → CPU の小さいメッセージの性能の悪化（`specs/research/benchmarks.md` の新しい節の表）**
   - 問題: CUDA のノードから CPU のノードへの 64 要素のメッセージで、recv が 3.5〜3.9 µs から 19〜28 µs、送受信の合計が約 17 µs 増える（書き手の stream の待ちと、読み手の同期の D2H の 2 回になる）。4,096 要素でも recv は 5.0 → 35〜38 µs。計画は想定していなかった（実装ログ「計画の想定外」）。CPU の読み手の hook の中で同期の D2H が走るようにもなった
   - 根拠: 依頼の目的は device → host → device の往復を無くすことで、CUDA → CPU の経路を遅くすることは依頼から辿れない。読み手が全員 CPU とホストのスロットでも device に置くかどうか（例: CUDA の読み手が居ないポートは今までどおり send でホストへ移す）はユーザーの判断が要る
   - 確信度: 中（計測は実装者のもの。`cuda:0` は別プロセスが使っていた）

4. **副 stream も torch の stream のプール（device ごとに 32 本）を消費する: `docs/en/topics/threads-and-devices.md:73`、`docs/ja/topics/threads-and-devices.md:73`、`specs/nodes.md`（「待つときは stream 全体を…32 を超える CUDA のノードは stream を共有する」）**
   - 問題: `torch.cuda.Stream(d)` はプールから割り当てるので、`Driver` の副 stream もノード自身の stream と同じ 32 本を巡回して共有する。文書は「1 つのデバイスに 32 を超える CUDA のノードを置くと共有する」とだけ書くが、別の device を読むノード（CPU のノードを含む）の副 stream も数に入る。共有したとき、CPU の読み手の D2H が書き手自身の stream に積まれると、書き手が後から積んだ GPU の仕事を待つ（正しさは保たれ、性能だけの問題）
   - 根拠: `src/mujou/driver.py:70-71`、torch の `Stream.__new__` は stream のポインタを渡さなければ `getStreamFromPool` を使う
   - 確信度: 中

5. **テストの不足: `tests/mujou/test_port.py` の `TestCuda`**
   - 問題: must-fix 1 の場合（hook の中で current device を変えたまま、ノードの device のテンソルを send する）を通すテストが無い。計画書の「書き手のスレッドの current device」のリスクを確かめるテストが無かったので、回帰を見逃した
   - 根拠: skill `testing-strategy`（異常系とエッジケース）、計画書の「想定リスク」
   - 確信度: 高

## nit

1. `src/mujou/codec.py:191`: `_CPU: Final` が、それを使う `peek` の後、関数の間に置いてある。モジュールの上（`_EXPECTED` のあたり）に置くほうが読みやすい（確信度: 中）
2. `src/mujou/codec.py:123-124`: `_value` の引数名 `held` は device を表すが、`Held`（コアの不透明な値）と紛らわしい。`to_core` の側の名前 `device` か `holds_on` のほうが読みやすい（確信度: 低、好みの範囲）
3. `crates/mujou-core/src/schema.rs:120-124`: `UnsupportedDevice` の文言が「host memory or on a CUDA device (… 1, 3 or 2)」になったが、出すのは `copy_from` で、`copy_from` 自身は CUDA（2）も断る。コアの層の文言としては、2 を受け付けるように読める。実際に利用者が見るのは bindings 経由なので害は小さい（確信度: 中）
4. `specs/messages.md:56` と計画書: 「この機械の RTX 4090 と RTX 4060 は `can_device_access_peer` が False」は設計仕様でなく計測環境の事実。`research/benchmarks.md` に置き、`messages.md` では「P2P の無い GPU の組では」とだけ書くほうが仕様として長持ちする（確信度: 低）
5. `crates/mujou-python/src/dlpack.rs:42-43`: torch の `__dlpack__` の既定 `stream=-1` に頼っている（コメントに明記済み）。DLPack の仕様の既定は `None` なので、将来の torch が既定を変えると、legacy の既定 stream が current stream を待つ依存が黙って増える（正しさは壊れない）。下限を上げるときに見直す、と `specs/messages.md` に一言あるとよい（確信度: 低）
6. `tests/mujou/test_state.py` の `test_a_paused_run_gives_the_values_on_the_device_uncopied`: 停止中に参照だけを入れても「再開後に書き手が送り直しても木の値は取得した時点のまま」（`specs/execution-model.md` の「コピーする理由」に足した主張）を確かめていない。値が不変なことから構造的に成り立つので、足すかは任意（確信度: 低）
7. `specs/messages.md:130`: 保持する値が落ちる場所を「`publish` の `drop(old)`」とだけ書くが、読み手の `Message` が最後の参照のとき（GIL を持った状態）や `InputGroup.write` でも落ちる。記述として不完全なだけ（確信度: 低）

## 検証結果

- just run: pass（cargo test 全 pass、pytest 951 passed / 1 skipped（`test_rejects_cuda_without_cuda`、CUDA がある機械なので想定どおり）、pyright pass、mkdocs build --strict pass。実行後の `git status` は clean）
- GIL ありのビルド（`.venv-gil`、拡張は 01:02 にビルド済み）: `pytest tests/mujou -m cuda` で 60 passed
- 変異テスト（scratch の pytest plugin で monkeypatch、コードは変えていない）:
  - `to_core` の `record_event().synchronize()` を無効にする → `test_a_reader_gets_what_a_late_stream_wrote` が落ちる
  - `Tensor.record_stream` を無効にする → `test_a_copy_still_queued_keeps_its_message_from_reuse` が落ちる
  - 変異なし → 両方 pass。変異なしで 5 回連続 pass（各 0.5 s 前後）
- 回帰の再現: `/tmp/claude-1001/-data-geson-mujou/d2146d35-f3d8-44cf-90d5-5029742d85ef/scratchpad/rev/other_device.py`（must-fix 1）
- 成果物の汚染（`</content>` など）: 変更した全ファイルで無し。全ファイルが改行で終わる

---

# 再レビュー（2026-10-25、ユーザーの決定 A / B への対応）

対象: `2435496..HEAD`（5542352, 08a195b, 6697546, 1954322）。実装ログ `memory/agents/plan-implementer/device-tensor-passthrough.md` の末尾。

## verdict: approve

must-fix は無し。前回の must-fix（hook の中で current device を変えて send すると `BufferError`）は前回の差し戻し対応で直っている。A（ホストの write の CUDA のテンソルをその device に置く、S2）と B（CUDA の読み手が居るポートだけ device に置く、配線時に静的に決める）は仕様どおりに実装されている。下の should-fix 1〜3 は PR の前に片付けるか、orchestrator が採否を決める。

確かめたこと（指摘ではない根拠として）:

- 包みを外し忘れる経路は無い。`Message.values` の読み手は `port._read`→`from_core`（recv 系）、`Run.read` / `read_with_time` / `subscribe` / `wait_for`→`from_core`（outbox だけで、ホストが書いた値は outbox に入らない）、`state._entry`→`peek` の 3 つだけで、`from_core` は `_read` が、`peek` は分岐で外す。復元は `to_core(device=None)` でホストのメモリ
- `_OnItsDevice.__dlpack__` は bindings の `borrow_values`（GIL あり、同じスレッドで同期）から呼ばれ、torch 2.14.1 の `__dlpack__` の device の検査（`self.device.index != torch.cuda.current_device()`）を `with torch.cuda.device(...)` で満たす。capsule はその場で落ち、包みのオブジェクトが `Held` になる
- dict で device が混ざる場合: `_on_their_devices` が device ごとに current stream の event を待ち、各値が自分で device を切り替えるので正しい（scratch で確認）
- 一斉の write: `Wiring.write` は全ポートの `to_core`（待ちを含む）を済ませてから `group.write` を 1 回呼ぶので、版番号が進む前に全値が出来上がっている
- `Their` enum: 単一メンバーの enum は型付きの sentinel の定石で、pyright の `is` による narrowing が効く。必要性はある（`None` / device と区別する第 3 の値）。名前は nit 4
- hot path: CPU の send は `to_core` に `is THEIR_DEVICES` の比較 1 回だけ増える。`_value` は `stays_on is None` で短絡し、ノードの device の比較を先に置く。`_read` の追加の `type(...) is _OnItsDevice` は CUDA の値の分岐の中だけ
- B: `_held` はポートのノードが CUDA で、接続先のノードのどれかが CUDA のとき真。`read_devices` も `held` から作るので、device に置かないポートの分の副 stream は作らない。`connect` が spec の相互一致を検査するので、recv のポートの相手は必ず send を持ち `held[...]` は KeyError にならない

## must-fix

なし。

## should-fix

1. **ホストが書いた CUDA の値の状態の取得・復元を通すテストが無い（`tests/mujou/test_state.py` の `TestCuda`）**
   - 問題: `"in"` スロットに包んだ値が入った状態で `Run.state_dict` → `peek` の包み外し → `state.cpu` → 復元、の経路を通すテストが無い。`peek` の包み外しは `test_wiring.py` の白箱テストだけが確かめ、`state.cpu` まで通る end-to-end は無い。`peek` が外し忘れると `torch.Tensor.cpu` が `_OnItsDevice` に当たる（`map_tensors` が素通しして木に包みが残り、`torch.save(weights_only)` で落ちる）回帰を見逃す
   - 根拠: skill `testing-strategy`（公開 API の経路）。scratch（`/tmp/claude-1001/-data-geson-mujou/d2146d35-f3d8-44cf-90d5-5029742d85ef/scratchpad/rev2/test_scratch_host_cuda.py`、下の検証結果）では現状は正しく動くことを確かめた（current device が cuda:0 のホストが `inputs=` で cuda:1 のテンソル → 木の値は CPU、復元した実行のノードが同じ値を受け取る）
   - 確信度: 高（テストの欠落）。深刻度: 低〜中

2. **ホストが書いた値の別の device の読み手に副 stream を作らない判断はユーザーの決定ではない（`src/mujou/wiring.py` の `read_devices`、`specs/messages.md`「ホストが書いた値の読み手」）**
   - 問題: 決定 A は置き場所と S2 だけを決めている。読み手側の stream は実装者の判断（実装ログ・specs に「2026-10-25 の判断」）。その結果、CPU のノード（や別の GPU のノード）がホストの書いた CUDA の値を読むと、ワーカーに残った別のノードの stream で D2H を積み、CPU のノードはそのノードが積んだ GPU の仕事をホストで待つ。正しさは保たれ、文書にも書いてあるが、ノード同士が stream で絡む（`specs/nodes.md` の副 stream が避けようとした状態そのもの）
   - 根拠: `specs/nodes.md` の副 stream の理由（「副 stream が無いと、ワーカーで直前に走った別のノードの stream に積まれ、ノード同士が同じ stream で絡む」）
   - より単純な緩和策（参考）: ホストが書くスロットを読むノードの `read_devices` に、その実行のノードが使う CUDA device（自分以外）を足す。どの device も既に context があるので「使わない GPU に context を作る」問題は起きず、実行のノードが使わない GPU に書かれた場合だけ今の扱いになる。採るかはユーザーに確認するのがよい
   - 確信度: 中（性能だけの問題で、頻度は CPU のノードがホストの CUDA の入力を読む構成に限られる）

3. **`CHANGELOG.md` が依然として無い（前回の should-fix 2 の持ち越し）**
   - 問題: 利用者に見える変更（CUDA のノード間のメッセージが device に留まる、ホストの write の CUDA のテンソルが device に留まり write が自分の stream を待つ）の `## [Unreleased]` の項目が無い。このブランチにはファイル自体が無い（release ブランチの merge 待ち）
   - 根拠: CLAUDE.md「利用者に見える変更は CHANGELOG.md の ## [Unreleased] に書く」
   - 確信度: 高

## nit

1. `src/mujou/codec.py:137`（`_on_their_devices`）: `with torch.cuda.device(d):` は不要。`torch.cuda.current_stream(d)` は device を引数で取り、`record_event` / `Event.record` は stream の device で event を作る。外すと 1 段浅くなる（確信度: 中。外すなら `test_a_write_waits_for_the_hosts_stream` と cuda(2) のテストで確認）
2. `src/mujou/wiring.py:56-58`（`Wiring.write`）: `to_core` をポートごとに呼ぶので、同じ device の N ポートへの一斉の write で event の待ちが N 回になる。全ポートの clone を積んでから device ごとに 1 回待てば足りる。ホストの経路だけで小さい（確信度: 中、深刻度: 低）
3. `src/mujou/codec.py:234`: `from_core` の docstring の行が 91 文字（E501 は無効で formatter は docstring を折らないので CI は通る）（確信度: 高）
4. `src/mujou/codec.py:121-128`: `Their` と `THEIR_DEVICES` の二重の名前は読み手に `Their` の意味を考えさせる。`class _Where(Enum): THEIR_DEVICES = auto()` のように 1 つの名前にするか、型注釈の `Their` だけ残して `Their.DEVICES` を直接使う程度で足りる。なお `type Views` が private の `_OnItsDevice` を含む（確信度: 低、好みの範囲に近い）
5. より単純な設計（参考、確信度: 低）: ノードの send でも「current device が値の device と違う間はホストへ回す」（`to_core` の `current_device() != device.index` の分岐）を、clone を `_OnItsDevice` に包む扱いに揃えると、例外の記述（`Port.send` の docstring、`specs/messages.md`、docs）が 1 つ減り、hook の中で device を切り替えても device に留まる。包むのはその稀な場合だけなので hot path は増えない（`current_device()` の呼び出しは今もある）。ただし前回の差し戻しで実装者が「稀な場合だけのため」に採らなかった案で、今回 `_OnItsDevice` ができたので費用が下がった、という位置づけ
6. `src/mujou/execution.py:171-174`（`Run.write`）: 「a dense tensor on a CUDA device on that device」は前置詞が続いて読みにくい（「a dense CUDA tensor is copied on its own device」など）。また `mujou.run` の `inputs` 引数の説明（`execution.py:628` 付近）は置き場所に触れない（`run.write` と同じと一言あるとよい）（確信度: 低）
7. `tests/mujou/test_wiring.py`: 接続先の無い（ホストだけが読む）CUDA のノードのポートで `Wire.cuda is None` を確かめるテストは無い（`_held` の `any(())` が False になるだけなので自明に近い）（確信度: 低）

## 検証結果（再レビュー）

- just run: pass（cargo test pass、pytest 963 passed / 1 skipped、pyright 0 errors、mkdocs build --strict pass。実行後の `git status` は clean）
- scratch（`PYTHONPATH=<worktree> uv run pytest -p tests.mujou.conftest <scratch>/rev2/test_scratch_host_cuda.py`、GPU 2 枚）: 4 passed
  - current device が cuda:0 のホストが `inputs=` で cuda:1 のテンソル → `state_dict` の `"in"` の値は CPU、復元した実行の cuda:0 のノードが同じ値を受け取る
  - `with torch.cuda.device(1)` の中で、cuda:0 の非連続な view と、requires_grad の cuda:1 のテンソルを混ぜた dict を write → cuda:1 のノードと CPU のノードが正しい値を受け取る
  - 別スレッドからの write（例外なし）
  - 包む値の shape / dtype の違いはスロット名つきの `ValueError`（`slot 'node.inp/in': expected shape (2,), got (3,)` など）
- 成果物の汚染（`</content>` など）: 変更した全ファイルで無し

---

# 再々レビュー（2026-10-26、ユーザーの 2026-10-25 の 2 回目の決定への対応）

対象: `be790d2..HEAD`（be790d2, 0a9c1ae, c65e0f4, c745a8f, a3c1e08, f1f7208, 2b71a47, b500abb, 50569f8）と、未コミットの `CLAUDE.md`・`.claude/skills/locality-and-async/SKILL.md`。

## verdict: approve

must-fix はありません。決定 1（ホストの write で device に置くのは、ポートを持つノードの CUDA device の上のテンソルだけ。`Wiring.input_devices`）と決定 2（ノードの send で hook が別の device を current にしていても、`_OnItsDevice` で包んで device のまま送る。S2 は保つ）は、仕様どおりに実装されています。algorithm-reviewer の must-fix 1（CPU や別の device の読み手が、ホストが write の後に積んだ GPU の仕事を待つ）は直っており、回帰テストが変異で落ちることも確かめました。下の should-fix 1〜4 は、PR を出す前に片付けるか、orchestrator が採否を決めてください。

確かめたこと（指摘ではなく、判断の根拠）:

- `input_devices` の値は、正規化済みのノードの device（`node.py` の `_device` が `"cuda"` に index を付ける）なので、`to_core` の `torch.cuda.current_device() != device.index` も `value.device == stays_on` も正しく判定できる。`_item` は `inputs` の検査の後で `input_devices[port]` を引くので、KeyError にはならない
- ホストの write: clone はテンソルの device guard の下で、ホストのスレッドのその device の current stream に積まれ、`wait_for_clones` も同じ stream（`current_stream(d)`）を待つ。ホストの current device が違うときは包まれ、`wait_for_clones` が `_OnItsDevice` からも device を集める
- ノードの send で current device が違うとき: clone はノードの device の thread-local な current stream、つまり Driver が設定したノードの stream に積まれる（`torch.cuda.device(1)` は cuda:0 の stream を変えない）。`wait_for_clones` はその stream を待つ。読み手の `_read`、`Run.read`、`peek` は包みを外す
- CPU の読み手が混ざった MultiPort（ノードの send を device に置く場合）: CPU のノードにも `read_devices` から副 stream ができるので、ホストの legacy default stream を待つ形は残っていない
- 前回の指摘の扱い: should-fix 1（state の e2e）は be790d2、should-fix 2（ホストが書いた値の副 stream）は決定 1 で問題そのものが無くなった。nit 1・4（`Their` と `_on_their_devices`）は削除で解消、nit 2 は c65e0f4、nit 3 は a3c1e08 の範囲、nit 5 は 2b71a47、nit 6 は f1f7208、nit 7 は 0a9c1ae で対応済み。should-fix 3（CHANGELOG）は取り下げる。この worktree の CLAUDE.md にその規約は無く、main にも `CHANGELOG.md` は無い

## must-fix

なし。

## should-fix

1. **包んだ clone の待ち（`wait_for_clones` の `_OnItsDevice` の分岐）を確かめるテストが無い（`src/mujou/codec.py:120-125`）**
   - 問題: `wait_for_clones` が `_OnItsDevice` を無視する変異 D（`[v for v in m if isinstance(v, torch.Tensor)]` だけを待つ）を入れても、CUDA の関連テスト 42 件がすべて通る。この分岐が要るのは、ホストの current device がノードの device と違う write（docs は「whichever device is current」と約束している）と、hook の中で `torch.cuda.device(...)` を使った send の 2 つ。どちらでも、読み手が出来上がる前の clone を読む。scratch のテスト 2 本（ホストが `with torch.cuda.device(1)` の中で、cuda:0 の遅い stream が埋めるテンソルを cuda:0 のノードへ write する。`Late` の send を `with torch.cuda.device(1)` で包む）は、変異なしでは通り、変異 D では 2 本とも値が違って落ちる
   - 以前は `_on_their_devices` の中で、包むことと待つことが一体だった。今は包む処理（`to_core`）と待つ処理（`wait_for_clones`）が別の関数に分かれたので、片方だけが壊れても気付けない
   - 根拠: skill `testing-strategy`（変異で落ちるテスト）。再現: `/tmp/claude-1001/-data-geson-mujou--claude-worktrees-feature-2026-10-24-device-tensor-passthrough/d2146d35-f3d8-44cf-90d5-5029742d85ef/scratchpad/mut/test_scratch_wrapped_wait.py`（下の検証結果）。`test_a_write_waits_for_the_hosts_stream` と `test_a_reader_gets_what_a_late_stream_wrote` の `cuda(2)` 版として足すのが素直です
   - 確信度: 高。深刻度: 中（テストの欠落。今の実装は正しい）

2. **`memory/project_gpu_policy.md` に古い記述が 2 か所残っている（23 行、28 行）**
   - 問題: 23 行の Why は「ホストからの書き込みを device に置かないのは、torch の `__dlpack__` が current device を…保証できないため」と書いており、11 行・14 行（ホストの write を持ち主の device に置き、`_OnItsDevice` で包む）と矛盾する。28 行の How to apply「device に置く値を増やす（ホストの書き込みなど）ときは、`__dlpack__` の current device の制約と、読み手の副 stream の device が静的に決まらない問題を先に解く」は、この PR で両方とも解いた（`_OnItsDevice` と `input_devices`）ので、解決済みとして書き換えるべき
   - 根拠: 同じファイルの 11・14 行と `specs/messages.md`「コアの中の値」
   - 確信度: 高。深刻度: 低〜中（後で読むエージェントの判断を誤らせる）

3. **`to_core` の docstring が、ノードの send の `device` を読み手の device と説明している（`src/mujou/codec.py:73-77`）**
   - 問題: 「It is the device of the node that reads the values when the sender knows it … a node that sends to a node on CUDA (``Wire.cuda``)」とあるが、ノードの send で渡す `Wire.cuda` は**送るノード**の device です。cuda:0 のノードから cuda:1 や CPU のノードが読む場合も cuda:0 になる。ホストの write の場合だけ、読み手（持ち主）の device と一致する。`specs/messages.md` の codec の節（「ノードの send なら送るノードの CUDA device」）は正しい
   - 根拠: `src/mujou/wiring.py` の `held`（`device[port] if _held(...)`、port は送る側）
   - 確信度: 高。深刻度: 低

4. **未コミットの `CLAUDE.md:85` と skill `locality-and-async` 47 行の「CUDA のノードが読むテンソル（値）は device の上でコピー」は範囲が広すぎる**
   - 問題: CUDA のノードが読んでも、次の場合はホストのメモリを通る。CPU のノードが送った値、送り手の device 以外のテンソル、ホストが持ち主と別の device で書いたテンソル。正しくは「CUDA のノードが自分の device で送り CUDA のノードが読むテンソルと、ホストが持ち主の CUDA device で書いたテンソル」です。`CLAUDE.md:86` の書き方（「CUDA のノードが CUDA のノードに送ったメッセージと、ホストが持ち主のノードの device で書いた値」）はほぼ正確なので、85 行もこちらに揃えるとよい。skill の「値」はスカラーを含むので「テンソル」にする
   - どちらも未コミットです。CLAUDE.md の変更なので、ユーザーの確認を取ってから別の commit にすること
   - 根拠: `src/mujou/codec.py` の `_value`（`value.device == stays_on` の場合だけ device に置く）、`src/mujou/wiring.py` の `held` と `input_devices`
   - 確信度: 中（要約としての許容範囲の判断による）。深刻度: 低

## nit

1. `tests/mujou/test_state.py`（`test_a_cuda_value_the_host_wrote_is_saved_on_the_cpu_and_restored`）のコメント「The core holds what the host writes on CUDA wrapped」は古い。ホストの current device は cuda:0 でノードと同じなので、今は包まれずに置かれる。e2e で包んだ値の状態取得を通したいなら、`cuda(2)` で write を `with torch.cuda.device(1)` の中に入れる。ただし `peek` の包みの分岐が無くても `_read` が CPU へコピーするので、壊れても値は正しい（確信度: 高）
2. `tests/mujou/test_port.py` の `test_sends_while_the_hook_has_made_another_device_current` は、slow の印が無いまま profiler のサブプロセス（約 1.8 s）を起こすので、`test-fast` が遅くなる。slow の印が付いた `test_a_message_between_cuda_nodes_skips_host_memory` に `pytest.param("cuda:0@1", "cuda:0", False, marks=pytest.mark.cuda(2))` として足すほうが、オラクルの置き場所が 1 つに揃う（確信度: 中）
3. 同じテストの `("host:cuda:0", "cpu", True)` は、決定 1 を戻す変異 A でも通る（D2H がホストから読み手へ移るだけで、Pageable の端点はどちらでも出る）。対照としての記録にはなるが、決定 1 を守るのは `test_execution.py` の新しいテストです。また、テスト名の「between_cuda_nodes」はホストの write を含む今の範囲と合わない（確信度: 高、優先度: 低）
4. `src/mujou/port.py:113-116`（`Port.send` の docstring）: 「a peer on CUDA copies it from device to device, also while the hook has made another device current」では、current device の条件が読み手のコピーに掛かって読める。正しくは送り手の側の条件なので、「It stays on the device also while …」のように独立した文にする（確信度: 中）
5. `src/mujou/codec.py:9-11` のモジュールの docstring は、device に置く場合をノードの send だけと書いており、CUDA の読み手の条件とホストの write が抜けている（確信度: 高、優先度: 低）
6. `src/mujou/wiring.py`: `input_devices` は `inputs` と同じキーを持つ並行の辞書。`inputs: Mapping[AnyPort, tuple[InputSlot, device | None]]` にまとめる手もある。好みの範囲（確信度: 低）
7. `docs/{en,ja}/topics/threads-and-devices.md` の最後の段落: 「Any other tensor …」の直後の「Any other node on the CPU …」（ja も「それ以外のテンソル」の直後に「それ以外の CPU のノード」）は、「other」が何を指すのか分かりにくい。以前からある文（確信度: 低）

## 検証結果（再々レビュー）

- just run: 実行していない（orchestrator が並行して実行中のため。指示どおり）
- 個別の pytest（RTX 4090 + RTX 4060）: `test_execution.py` / `test_port.py` / `test_wiring.py` / `test_state.py` / `test_codec.py` の CUDA 関連 57 件が pass。新しいテスト（`test_a_reader_does_not_wait_for_what_the_host_queues_after_writing` の 3 つ、state の e2e、`test_a_write_waits_for_the_hosts_stream`）は 5 回連続で pass
- CUDA が無いときの skip: `CUDA_VISIBLE_DEVICES=""` で、対象 14 件がすべて skip（1 枚が要るもの 9 件、2 枚が要るもの 5 件）。`CUDA_VISIBLE_DEVICES=0` では 9 passed / 5 skipped
- 変異テスト（`/tmp/claude-1001/-data-geson-mujou--claude-worktrees-feature-2026-10-24-device-tensor-passthrough/d2146d35-f3d8-44cf-90d5-5029742d85ef/scratchpad/mut/sitecustomize.py` を `PYTHONPATH` に置き、`MUT=<名前>` で切り替える。profiler のサブプロセスにも効く）
  - A（ホストの write は、テンソルの CUDA device にそのまま置く。旧来の挙動）→ `test_a_reader_does_not_wait_…[cpu]`・`[cuda:1]`、`skips_host_memory[host:cuda:1-cuda:0-True]`、`test_what_the_host_writes_stays_only_on_the_device_of_its_node` の 4 件が落ちる
  - B（ノードの send で current device が違えばホストへ回す。旧来の挙動）→ `test_sends_while_the_hook_has_made_another_device_current`（memcpy のオラクル）が落ちる
  - C1（ホストの write の `wait_for_clones` を外す）→ `test_a_write_waits_for_the_hosts_stream` が落ちる
  - C2（ノードの send の `wait_for_clones` を外す）→ `test_a_reader_gets_what_a_late_stream_wrote` が落ちる
  - D（`wait_for_clones` が `_OnItsDevice` を無視する）→ **42 件すべて通る**（should-fix 1）。scratch の `test_scratch_wrapped_wait.py` の 2 本は、変異なしで pass し、変異 D で 2 本とも fail
- 成果物の汚染（`</content>` など）: 変更した全ファイルで無し。全ファイルが改行で終わる
