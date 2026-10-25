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
