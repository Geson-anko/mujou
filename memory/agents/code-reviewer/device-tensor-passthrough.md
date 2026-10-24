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
